"""Reinforcement-learning policy strategy with pluggable heads."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar, TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from strand.core.sequence import Sequence
from strand.engine.interfaces import Strategy
from strand.engine.runtime import ModelRuntime, StrategyCaps, StrategyContext
from strand.engine.strategies.rl.policy_heads import PolicyHead, create_policy_head
from strand.engine.strategies.runtime_adapter import StrategyRuntimeAdapter
from strand.engine.types import Metrics

if TYPE_CHECKING:  # pragma: no cover - optional deps
    from strand.data.sequence_dataset import SequenceDataset
    from strand.logging.mlflow_tracker import MLflowTracker


@dataclass
class RLPolicyStrategy(Strategy):
    """Ctrl-DNA style constrained RL strategy with optional SFT warm start."""

    alphabet: str
    min_len: int
    max_len: int
    seed: int | None = None
    learning_rate: float = 0.1
    temperature: float = 1.0
    constraint_penalty: float = 1.0
    policy_head: PolicyHead | None = None
    policy_head_type: str = "per-position"
    policy_head_kwargs: Mapping[str, object] = field(default_factory=dict)
    tokenizer_fn: Callable[[str], list[int]] | None = None

    _CAPS: ClassVar[StrategyCaps] = StrategyCaps(
        requires_runtime=True,
        supports_fine_tuning=True,
        kl_regularization="token",
    )

    _rng: random.Random = field(default_factory=random.Random, init=False, repr=False)
    _q_values: dict[tuple[int, str], float] = field(
        default_factory=lambda: defaultdict(float), init=False, repr=False
    )
    _value_baseline: dict[int, float] = field(
        default_factory=lambda: defaultdict(float), init=False, repr=False
    )
    _visit_counts: dict[tuple[int, str], int] = field(
        default_factory=lambda: defaultdict(int), init=False, repr=False
    )
    _best_sequence: Sequence | None = field(default=None, init=False, repr=False)
    _best_score: float = field(default=float("-inf"), init=False, repr=False)
    _episode_count: int = field(default=0, init=False, repr=False)
    _token_index: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _runtime: ModelRuntime | None = field(default=None, init=False, repr=False)
    _adapter: StrategyRuntimeAdapter | None = field(default=None, init=False, repr=False)
    _policy_head_instance: PolicyHead | None = field(default=None, init=False, repr=False)
    _optimizer: Optimizer | None = field(default=None, init=False, repr=False)
    _head_prepared: bool = field(default=False, init=False, repr=False)
    _pad_token_id: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.alphabet:
            raise ValueError("alphabet must be non-empty")
        if self.min_len <= 0 or self.max_len < self.min_len:
            raise ValueError("invalid length band")
        if self.seed is not None:
            self._rng.seed(self.seed)
        self._token_index = {token: idx for idx, token in enumerate(self.alphabet)}
        self._pad_token_id = self._token_index.get(self.alphabet[0], 0)

    def strategy_caps(self) -> StrategyCaps:
        return self._CAPS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self, context: StrategyContext) -> None:
        self._runtime = context.runtime
        self._adapter = StrategyRuntimeAdapter(context.runtime) if context.runtime else None
        self._ensure_head_ready()

    def ask(self, n: int) -> list[Sequence]:
        self._ensure_head_ready()
        sequences: list[Sequence] = []
        for _ in range(n):
            seq_len = self._sample_length()
            tokens: list[str] = []
            for position in range(seq_len):
                probs = self._token_distribution(tokens, position, seq_len)
                token = self._rng.choices(self.alphabet, weights=probs, k=1)[0]
                tokens.append(token)
            seq = Sequence(id=f"rl_candidate_{self._episode_count}_{len(sequences)}", tokens="".join(tokens))
            sequences.append(seq)
        return sequences

    def tell(self, items: list[tuple[Sequence, float, Metrics]]) -> None:
        for sequence, score, metrics in items:
            if self._best_sequence is None or score > self._best_score:
                self._best_sequence = sequence
                self._best_score = score
            effective_reward = metrics.objective - self.constraint_penalty * sum(metrics.constraints.values())
            advantage = self._update_baseline(len(sequence.tokens), effective_reward)
            if self._policy_head_instance is None:
                self._update_bandit_values(sequence.tokens, advantage)
        self._episode_count += 1

    def best(self) -> tuple[Sequence, float] | None:
        if self._best_sequence is None:
            return None
        return self._best_sequence, self._best_score

    def state(self) -> Mapping[str, object]:
        return {
            "best_score": self._best_score,
            "episodes": self._episode_count,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
        }

    def train_step(
        self,
        items: list[tuple[Sequence, float, Metrics]],
        context: StrategyContext,
    ) -> None:
        if not items:
            return
        self._runtime = context.runtime or self._runtime
        if context.runtime and not self._adapter:\n            self._adapter = StrategyRuntimeAdapter(context.runtime)
        self._ensure_head_ready()
        if self._policy_head_instance is None or self._optimizer is None:
            return

        train_limit = context.batch.train_size or len(items)
        selected = items[:train_limit]
        device = self._policy_device()
        optimizer = self._optimizer
        optimizer.zero_grad(set_to_none=True)
        temperature = max(self.temperature, 1e-6)
        denom = 0

        with self._autocast():
            loss = torch.zeros((), device=device)
            for seq, score, _ in selected:
                token_ids = self._encode_sequence(seq.tokens)
                if not token_ids:
                    continue
                batch_dict, encoded = self._batch_from_sequence(token_ids, device)
                logits = self._policy_head_instance(batch_dict)[0, : len(encoded)]
                log_probs = torch.log_softmax(logits / temperature, dim=-1)
                positions = torch.arange(len(encoded), device=device)
                idx_tensor = torch.tensor(encoded, device=device, dtype=torch.long)
                seq_log_prob = log_probs[positions, idx_tensor].sum()
                weight = torch.tensor(score, device=device, dtype=log_probs.dtype)
                loss = loss - weight * seq_log_prob
                denom += 1
            if denom == 0:
                optimizer.zero_grad(set_to_none=True)
                return
            loss = loss / denom

        self._backward(loss)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def warm_start(
        self,
        dataset: "SequenceDataset",
        *,
        epochs: int = 1,
        batch_size: int | None = None,
        context: StrategyContext | None = None,
        tracker: "MLflowTracker" | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        if context is not None and context.runtime is not None:
            self._runtime = context.runtime
            self._adapter = StrategyRuntimeAdapter(context.runtime)
        self._ensure_head_ready()
        if self._policy_head_instance is None or self._optimizer is None:
            raise RuntimeError("warm_start requires an initialized policy head")

        device = self._policy_device()
        effective_batch = batch_size or (context.batch.train_size if context else None) or 32
        for epoch in range(epochs):
            loader = dataset.train_loader(batch_size=effective_batch, shuffle=True)
            epoch_tokens = 0
            epoch_correct = 0
            epoch_kl = 0.0
            optimizer = self._optimizer
            optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                loss = torch.zeros((), device=device)
                for batch in loader:
                    target_sequences = getattr(batch, "sequences", [])
                    for seq in target_sequences:
                        token_ids = self._encode_sequence(seq.tokens)
                        if not token_ids:
                            continue
                        batch_dict, encoded = self._batch_from_sequence(token_ids, device)
                        logits = self._policy_head_instance(batch_dict)[0, : len(encoded)]
                        target = torch.tensor(encoded, device=device, dtype=torch.long)
                        loss = loss + F.cross_entropy(logits, target, reduction="sum")
                        preds = torch.argmax(logits, dim=-1)
                        epoch_correct += int((preds == target).sum().item())
                        epoch_kl += self._average_kl(logits) * len(encoded)
                        epoch_tokens += len(encoded)
                if epoch_tokens == 0:
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss = loss / epoch_tokens
            self._backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            avg_loss = float(loss.item())
            avg_acc = epoch_correct / max(epoch_tokens, 1)
            avg_kl = epoch_kl / max(epoch_tokens, 1)
            if tracker is not None:
                tracker.log_sft_metrics(epoch=epoch, loss=avg_loss, accuracy=avg_acc, kl=avg_kl)

        if checkpoint_path is not None:
            checkpoint = Path(checkpoint_path)
            self._save_checkpoint(checkpoint)
            if tracker is not None:
                tracker.log_sft_checkpoint(checkpoint)

    def get_policy_entropy(self) -> float:
        self._ensure_head_ready()
        total_entropy = 0.0
        for pos in range(self.max_len):
            probs = self._token_distribution([], pos, self.max_len)
            entropy = -sum(p * math.log2(p + 1e-10) for p in probs)
            total_entropy += entropy
        return total_entropy / max(self.max_len, 1)

    def set_temperature(self, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_head_ready(self) -> None:
        if self._policy_head_instance is None:
            self._policy_head_instance = self.policy_head or self._build_head()
        if self._optimizer is None and self._policy_head_instance is not None:
            self._optimizer = torch.optim.Adam(self._policy_head_instance.parameters(), lr=self.learning_rate)
        if self._adapter and self._policy_head_instance is not None and not self._head_prepared:
            result = self._adapter.prepare_module(self._policy_head_instance, self._optimizer)
            self._policy_head_instance = cast(PolicyHead, result.module)
            self._optimizer = result.optimizer
            self._head_prepared = True

    def _build_head(self) -> PolicyHead:
        kwargs = {"max_seq_len": self.max_len, "vocab_size": len(self.alphabet)}
        kwargs.update(self.policy_head_kwargs)
        return create_policy_head(self.policy_head_type, **kwargs)

    def _token_distribution(self, prefix_tokens: list[str], position: int, seq_len: int) -> list[float]:
        if self._policy_head_instance is None:
            return self._q_distribution(position)
        batch = self._build_inference_batch(prefix_tokens, seq_len)
        logits = self._policy_head_instance(batch)[0, position]
        probs = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        return probs.detach().cpu().tolist()

    def _q_distribution(self, position: int) -> list[float]:
        q_vals = [self._q_values[(position, token)] for token in self.alphabet]
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        max_q = max(q_vals) if q_vals else 0.0
        exp_vals = [exp((q - max_q) / self.temperature) for q in q_vals]
        total = sum(exp_vals)
        if total == 0:
            return [1.0 / len(self.alphabet)] * len(self.alphabet)
        return [val / total for val in exp_vals]

    def _build_inference_batch(self, prefix_tokens: list[str], seq_len: int) -> dict[str, torch.Tensor]:
        device = self._policy_device()
        encoded = self._encode_sequence(prefix_tokens)
        pad = max(seq_len - len(encoded), 0)
        encoded = encoded + [self._pad_token_id] * pad
        encoded = encoded[:seq_len]
        input_ids = torch.tensor([encoded], device=device, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        if prefix_tokens:
            attention_mask[:, : len(prefix_tokens)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _batch_from_sequence(self, token_ids: list[int], device: torch.device) -> tuple[dict[str, torch.Tensor], list[int]]:
        trimmed = token_ids[: self.max_len]
        input_ids = torch.tensor([trimmed], device=device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}, trimmed

    def _encode_sequence(self, tokens: list[str] | str) -> list[int]:
        if isinstance(tokens, list):
            sequence = "".join(tokens)
        else:
            sequence = tokens
        if not sequence:
            return []
        if self.tokenizer_fn is not None:
            return list(self.tokenizer_fn(sequence))
        return [self._token_index.get(char, self._pad_token_id) for char in sequence]

    def _sample_length(self) -> int:
        return self._rng.randint(self.min_len, self.max_len)

    def _policy_device(self) -> torch.device:
        if self._policy_head_instance is None:
            return torch.device("cpu")
        return next(self._policy_head_instance.parameters()).device

    def _autocast(self):
        if self._adapter is not None:
            return self._adapter.autocast_context()
        if self._runtime is not None:
            return self._runtime.autocast()
        return nullcontext()

    def _backward(self, loss: torch.Tensor) -> None:
        if self._adapter is not None:
            self._adapter.backward(loss)
        elif self._runtime is not None:
            self._runtime.backward(loss)
        else:
            loss.backward()

    def _average_kl(self, logits: torch.Tensor) -> float:
        probs = torch.softmax(logits.detach(), dim=-1)
        uniform_log = math.log(1.0 / probs.shape[-1])
        kl = probs * (torch.log(probs + 1e-10) - uniform_log)
        return float(kl.sum().item() / max(logits.shape[0], 1))

    def _save_checkpoint(self, path: Path) -> None:
        if self._policy_head_instance is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._adapter is not None:
            self._adapter.save_checkpoint(self._policy_head_instance, self._optimizer, path)
            return
        payload = {"model_state": self._policy_head_instance.state_dict()}
        if self._optimizer is not None:
            payload["optimizer_state"] = self._optimizer.state_dict()
        torch.save(payload, path)

    def _update_baseline(self, seq_len: int, reward: float) -> float:
        current = self._value_baseline[seq_len]
        updated = current + 0.01 * (reward - current)
        self._value_baseline[seq_len] = updated
        return reward - self._value_baseline[seq_len]

    def _update_bandit_values(self, tokens: str, advantage: float) -> None:
        for pos, token in enumerate(tokens):
            key = (pos, token)
            current_q = self._q_values[key]
            self._q_values[key] = current_q + self.learning_rate * advantage
            self._visit_counts[key] += 1


def exp(x: float) -> float:
    if x > 100:
        return 1e50
    if x < -100:
        return 0.0
    return math.exp(x)
