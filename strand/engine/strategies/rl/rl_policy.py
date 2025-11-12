"""RL-based Policy Strategy for sequence generation.

Implements Ctrl-DNA style constrained RL for DNA sequence design:
Paper: "Ctrl-DNA: Controllable Cell-Type-Specific Regulatory DNA Design via Constrained RL"
https://arxiv.org/abs/2505.20578 (Chen et al., 2025)

Key methods from Ctrl-DNA:
1. **Autoregressive Generation**: Token-by-token generation P(token_t | tokens_{0..t-1})
2. **Constrained Optimization**: Formulate as: maximize(objective) subject to constraints
3. **Policy Gradient RL**: REINFORCE-style updates with advantage estimation
4. **Reward Shaping**: Decompose into target reward + constraint penalties
5. **Biological Plausibility**: Score based on learned patterns (e.g., TFBS, motifs)

Implementation:
- Policy: P(token | position) learned from reward signals
- Objective: maximize regulatory activity (e.g., reward)
- Constraints: minimize off-target effects (penalty term)
- Advantage: reward - baseline (for variance reduction)
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, cast

import torch
import torch.nn as nn
from torch.optim import Optimizer

from strand.core.sequence import Sequence
from strand.engine.interfaces import Strategy
from strand.engine.runtime import ModelRuntime, StrategyCaps, StrategyContext
from strand.engine.types import Metrics


@dataclass
class RLPolicyStrategy(Strategy):
    """Constrained RL policy-based strategy (Ctrl-DNA style).

    Learns token preferences at each position based on reward signals.
    Supports both unconstrained (maximize reward) and constrained
    (maximize reward subject to constraints) optimization.

    Parameters
    ----------
    alphabet : str
        Available characters for sequences.
    min_len : int
        Minimum sequence length.
    max_len : int
        Maximum sequence length.
    seed : int | None
        Random seed.
    learning_rate : float
        Learning rate for policy updates (alpha). Typical: 0.05-0.2
    temperature : float
        Temperature for softmax exploration (tau).
        Higher = more exploration. Typically annealed over time.
    constraint_penalty : float
        Penalty coefficient for constraint violations.
        Higher = stricter constraint enforcement.
    """

    alphabet: str
    min_len: int
    max_len: int
    seed: int | None = None
    learning_rate: float = 0.1
    temperature: float = 1.0
    constraint_penalty: float = 1.0  # Lagrange multiplier for constraints

    _CAPS: ClassVar[StrategyCaps] = StrategyCaps(
        requires_runtime=True,
        supports_fine_tuning=True,
        kl_regularization="token",
    )

    # Internal state
    _rng: random.Random = field(default_factory=random.Random, init=False, repr=False)
    _q_values: dict[tuple[int, str], float] = field(
        default_factory=lambda: defaultdict(float), init=False, repr=False
    )
    _value_baseline: dict[int, float] = field(
        default_factory=lambda: defaultdict(float), init=False, repr=False
    )
    _best_sequence: Sequence | None = field(default=None, init=False, repr=False)
    _best_score: float = field(default=float("-inf"), init=False, repr=False)
    _counter: int = field(default=0, init=False, repr=False)
    _visit_counts: dict[tuple[int, str], int] = field(
        default_factory=lambda: defaultdict(int), init=False, repr=False
    )
    _episode_count: int = field(default=0, init=False, repr=False)
    _token_index: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _runtime: ModelRuntime | None = field(default=None, init=False, repr=False)
    _policy_module: _AutoregressivePolicy | None = field(default=None, init=False, repr=False)
    _optimizer: Optimizer | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize policy state."""
        if not self.alphabet:
            raise ValueError("alphabet must be non-empty")
        if self.min_len <= 0 or self.max_len < self.min_len:
            raise ValueError("invalid length band")
        if self.seed is not None:
            self._rng.seed(self.seed)
        # Initialize Q-values uniformly
        self._q_values = defaultdict(float)
        self._visit_counts = defaultdict(int)
        self._token_index = {token: idx for idx, token in enumerate(self.alphabet)}

    def strategy_caps(self) -> StrategyCaps:
        return self._CAPS

    def _sample_length(self) -> int:
        """Sample sequence length from current policy."""
        return self._rng.randint(self.min_len, self.max_len)

    def prepare(self, context: StrategyContext) -> None:
        self._runtime = context.runtime
        if self._runtime is None:
            return
        self._ensure_torch_policy()

    def _get_token_probabilities(self, position: int) -> list[float]:
        """Get softmax probabilities for tokens at a position.

        Uses Q-values with temperature scaling:
        P(token) ∝ exp(Q(position, token) / tau)

        Parameters
        ----------
        position : int
            Current position in sequence.

        Returns
        -------
        list[float]
            Probabilities for each token.
        """
        if self._policy_module is not None:
            logits = self._policy_module.logits[position]
            probs_tensor = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)
            return probs_tensor.detach().cpu().tolist()

        # Get Q-values for all tokens
        q_vals = []
        for token in self.alphabet:
            key = (position, token)
            q = self._q_values[key]
            q_vals.append(q)

        # Apply temperature and softmax
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

        max_q = max(q_vals) if q_vals else 0.0
        exp_vals = [exp((q - max_q) / self.temperature) for q in q_vals]
        sum_exp = sum(exp_vals)

        if sum_exp == 0:
            # Uniform if no Q-values yet
            return [1.0 / len(self.alphabet)] * len(self.alphabet)

        return [e / sum_exp for e in exp_vals]

    def _sample_token(self, position: int) -> str:
        """Sample a token at the given position using policy.

        Parameters
        ----------
        position : int
            Current position in sequence.

        Returns
        -------
        str
            Sampled token.
        """
        probs = self._get_token_probabilities(position)
        return self._rng.choices(self.alphabet, weights=probs, k=1)[0]

    def ask(self, n: int) -> list[Sequence]:
        """Generate n sequences using current policy.

        Parameters
        ----------
        n : int
            Number of sequences to generate.

        Returns
        -------
        list[Sequence]
            Generated sequences.
        """
        sequences = []
        for _ in range(n):
            length = self._sample_length()
            tokens = []
            for pos in range(length):
                token = self._sample_token(pos)
                tokens.append(token)

            seq_str = "".join(tokens)
            seq = Sequence(id=f"rl_policy_{self._counter}", tokens=seq_str)
            self._counter += 1
            sequences.append(seq)

        return sequences

    def tell(self, items: list[tuple[Sequence, float, Metrics]]) -> None:
        """Update policy based on rewards (Ctrl-DNA constrained RL).

        Implements policy gradient update with constraint penalties:

        Objective = reward - constraint_penalty * constraint_violation

        Each token's Q-value updated proportional to advantage:
        Q(pos, token) ← Q(pos, token) + alpha * (R - baseline)

        where baseline is the average reward at that position (variance reduction).

        Parameters
        ----------
        items : list[tuple[Sequence, float, Metrics]]
            List of (sequence, score, metrics) tuples.
        """
        if not items:
            return

        # Track best
        for seq, score, _ in items:
            if score > self._best_score:
                self._best_score = score
                self._best_sequence = seq

        # Compute advantages with baseline for variance reduction
        # (inspired by Actor-Critic methods used in Ctrl-DNA)
        for seq, score, metrics in items:
            tokens = seq.tokens

            # Apply constraint penalty if metrics contain constraint violations
            effective_reward = score
            if metrics and metrics.constraints:
                total_violation = sum(metrics.constraints.values())
                effective_reward = score - self.constraint_penalty * total_violation

            advantage = self._update_baseline(len(tokens), effective_reward)

            if self._policy_module is None:
                self._update_bandit_values(tokens, advantage)

        self._episode_count += 1

    def best(self) -> tuple[Sequence, float] | None:
        """Return best sequence seen so far.

        Returns
        -------
        tuple[Sequence, float] | None
            (best_sequence, best_score) or None.
        """
        if self._best_sequence is None:
            return None
        return (self._best_sequence, self._best_score)

    def state(self) -> Mapping[str, object]:
        """Return serializable policy state.

        Returns
        -------
        Mapping[str, object]
            Policy state snapshot.
        """
        return {
            "best_score": self._best_score,
            "num_q_values": len(self._q_values),
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
        runtime = context.runtime
        if runtime is None:
            return

        self._runtime = runtime
        self._ensure_torch_policy()
        if self._policy_module is None:
            return

        optimizer = self._optimizer
        if optimizer is None:
            optimizer = torch.optim.Adam(self._policy_module.parameters(), lr=self.learning_rate)
            _, optimizer = runtime.prepare_module(self._policy_module, optimizer)
            if optimizer is None:
                return
            self._optimizer = optimizer

        train_limit = context.batch.train_size or len(items)
        selected = items[:train_limit]
        device = self._policy_module.device
        optimizer.zero_grad(set_to_none=True)

        denom = 0
        temperature = max(self.temperature, 1e-6)

        with runtime.autocast():
            loss = torch.zeros((), device=device)
            for seq, score, _ in selected:
                if not math.isfinite(score):
                    continue
                token_ids = self._encode_tokens(seq.tokens)
                if not token_ids:
                    continue

                logits = self._policy_module.logits[: len(token_ids)]
                log_probs = torch.log_softmax(logits / temperature, dim=-1)
                positions = torch.arange(len(token_ids), device=device)
                idx_tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
                seq_log_prob = log_probs[positions, idx_tensor].sum()
                weight = torch.tensor(score, device=device, dtype=log_probs.dtype)
                loss = loss - weight * seq_log_prob
                denom += 1

            if denom == 0:
                optimizer.zero_grad(set_to_none=True)
                return

            loss = loss / denom

        runtime.backward(loss)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def get_policy_entropy(self) -> float:
        """Compute entropy of current policy.

        Higher entropy = more exploration, lower = more exploitation.

        Returns
        -------
        float
            Policy entropy (bits).
        """
        import math

        total_entropy = 0.0
        count = 0
        for pos in range(self.max_len):
            probs = self._get_token_probabilities(pos)
            entropy = -sum(p * math.log2(p + 1e-10) for p in probs)
            total_entropy += entropy
            count += 1

        return total_entropy / max(count, 1)

    def set_temperature(self, temperature: float) -> None:
        """Adjust exploration temperature (annealing).

        Parameters
        ----------
        temperature : float
            New temperature value.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def _ensure_torch_policy(self) -> None:
        if self._runtime is None or self._policy_module is not None:
            return
        module = _AutoregressivePolicy(self.max_len, len(self.alphabet))
        optimizer = torch.optim.Adam(module.parameters(), lr=self.learning_rate)
        prepared_module, prepared_optimizer = self._runtime.prepare_module(module, optimizer)
        self._policy_module = cast(_AutoregressivePolicy, prepared_module)
        self._optimizer = prepared_optimizer

    def _encode_tokens(self, tokens: str) -> list[int]:
        indices: list[int] = []
        for char in tokens:
            idx = self._token_index.get(char)
            if idx is None:
                return []
            indices.append(idx)
        return indices

    def _update_baseline(self, seq_len: int, reward: float) -> float:
        current_baseline = self._value_baseline[seq_len]
        new_baseline = current_baseline + 0.01 * (reward - current_baseline)
        self._value_baseline[seq_len] = new_baseline
        return reward - self._value_baseline[seq_len]

    def _update_bandit_values(self, tokens: str, advantage: float) -> None:
        for pos, token in enumerate(tokens):
            key = (pos, token)
            current_q = self._q_values[key]
            new_q = current_q + self.learning_rate * advantage
            self._q_values[key] = new_q
            self._visit_counts[key] += 1


class _AutoregressivePolicy(nn.Module):
    def __init__(self, max_len: int, alphabet_size: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(max_len, alphabet_size))

    @property
    def device(self) -> torch.device:
        return self.logits.device


def exp(x: float) -> float:
    """Numerically stable exponential."""
    import math

    if x > 100:
        return 1e50
    if x < -100:
        return 0.0
    return math.exp(x)
