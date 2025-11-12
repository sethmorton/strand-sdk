"""Accelerator-aware executor with token-budgeted batching."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from collections.abc import Sequence as SeqCollection
from dataclasses import dataclass

import torch

from strand.core.sequence import Sequence
from strand.engine.interfaces import Evaluator
from strand.engine.runtime import ModelRuntime
from strand.engine.types import Metrics

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TorchExecutor:
    """Executor that cooperates with :class:`ModelRuntime` for batching."""

    evaluator: Evaluator
    runtime: ModelRuntime
    batch_size: int = 64
    max_tokens_per_batch: int | None = None

    def prepare(self) -> None:
        """Hook for heavy evaluator setup (noop by default)."""

    def run(
        self,
        seqs: list[Sequence],
        *,
        timeout_s: float | None = None,
        batch_size: int | None = None,
    ) -> list[Metrics]:
        if not seqs:
            return []

        effective_batch = batch_size or self.batch_size
        deadline = None if timeout_s is None else time.perf_counter() + timeout_s
        results: list[Metrics] = []

        for batch in self._yield_batches(seqs, effective_batch):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("TorchExecutor run exceeded timeout")

            try:
                with torch.no_grad(), self.runtime.autocast():
                    batch_results = self.evaluator.evaluate_batch(batch)
            except TimeoutError:
                raise
            except Exception as exc:  # pragma: no cover - defensive path
                _LOGGER.exception("Evaluator raised during evaluate_batch", exc_info=exc)
                batch_results = [Metrics(objective=0.0, constraints={}, aux={}) for _ in batch]

            if len(batch_results) != len(batch):  # pragma: no cover - defensive path
                raise RuntimeError(
                    "Evaluator returned mismatched batch size: "
                    f"expected {len(batch)} got {len(batch_results)}"
                )

            results.extend(batch_results)

        return results

    def close(self) -> None:
        """Hook for cleanup work (noop)."""

    def _yield_batches(self, seqs: SeqCollection[Sequence], max_items: int) -> Iterator[list[Sequence]]:
        batch: list[Sequence] = []
        token_count = 0
        token_budget = self.max_tokens_per_batch

        for seq in seqs:
            batch.append(seq)
            token_count += len(seq.tokens)
            if len(batch) >= max_items or (token_budget and token_count >= token_budget):
                yield batch
                batch = []
                token_count = 0

        if batch:
            yield batch
