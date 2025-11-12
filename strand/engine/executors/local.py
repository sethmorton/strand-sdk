"""Local executor that evaluates sequences one after another."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import ClassVar

from strand.core.sequence import Sequence
from strand.engine.interfaces import Evaluator
from strand.engine.types import Metrics


@dataclass(slots=True)
class LocalExecutor:
    """Sequential executor built around an evaluator."""

    evaluator: Evaluator
    batch_size: int = 64

    _LOGGER: ClassVar[logging.Logger] = logging.getLogger(__name__)

    def prepare(self) -> None:
        """Hook for heavy setup (unused for the local executor)."""

    def run(
        self,
        seqs: list[Sequence],
        *,
        timeout_s: float | None = None,
        batch_size: int | None = None,
    ) -> list[Metrics]:
        """Evaluate sequences while preserving order."""

        if not seqs:
            return []

        effective_batch = batch_size or self.batch_size
        deadline = None if timeout_s is None else time.perf_counter() + timeout_s

        results: list[Metrics] = []
        for start in range(0, len(seqs), effective_batch):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("LocalExecutor run exceeded timeout")

            batch = seqs[start : start + effective_batch]
            try:
                batch_results = self.evaluator.evaluate_batch(batch)
            except TimeoutError:
                raise
            except Exception as exc:  # pragma: no cover - defensive path
                self._LOGGER.exception("Evaluator raised during evaluate_batch", exc_info=exc)
                batch_results = [
                    Metrics(objective=0.0, constraints={}, aux={}) for _ in batch
                ]

            if len(batch_results) != len(batch):  # pragma: no cover - defensive path
                raise RuntimeError(
                    "Evaluator returned mismatched batch size: "
                    f"expected {len(batch)} got {len(batch_results)}"
                )

            results.extend(batch_results)

        return results

    def close(self) -> None:
        """Hook for cleanup work (unused)."""
