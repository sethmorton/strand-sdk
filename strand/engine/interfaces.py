"""Engine protocol surfaces (no implementations).

These Protocols define the modular boundaries for the optimization engine.
They are intentionally small and easy to reason about.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from strand.core.sequence import Sequence
from strand.engine.types import Metrics


class Strategy(Protocol):
    """Proposes candidates and ingests feedback (ask/tell).

    A Strategy owns its internal state and may implement any algorithm (e.g.,
    CEM, CMA-ES, GA, constrained RL). The engine treats it as a black box.
    """

    def ask(self, n: int) -> list[Sequence]:
        """Return ``n`` candidate sequences to evaluate."""

    def tell(self, items: list[tuple[Sequence, float, Metrics]]) -> None:
        """Ingest evaluated candidates as ``(sequence, score, metrics)``."""

    def best(self) -> tuple[Sequence, float] | None:
        """Return the best known candidate and its score, if any."""

    def state(self) -> Mapping[str, object]:
        """Return a serializable snapshot of strategy state (for manifests)."""


class Evaluator(Protocol):
    """Pure evaluator that maps sequences to structured metrics.

    Evaluators should be side-effect free and thread/process safe when used with
    executors that parallelize calls to ``evaluate_batch``.
    """

    def evaluate_batch(self, seqs: list[Sequence]) -> list[Metrics]:
        """Return metrics for a batch of sequences.

        Implementations may compute objectives with reward blocks or call
        external predictors. They should not perform parallelism internally; use
        an Executor for that.
        """


class Executor(Protocol):
    """Parallel wrapper around an Evaluator.

    Executors handle concurrency (threads/processes/remote) and timeouts while
    delegating actual scoring to an Evaluator.
    """

    def prepare(self) -> None:  # pragma: no cover - surface only
        """Optional heavy initialization (e.g., model warmup)."""

    def run(self, seqs: list[Sequence], *, timeout_s: float | None = None, batch_size: int = 64) -> list[Metrics]:
        """Evaluate ``seqs`` and return metrics in the same order."""

    def close(self) -> None:  # pragma: no cover - surface only
        """Optional cleanup hook for releasing resources."""
