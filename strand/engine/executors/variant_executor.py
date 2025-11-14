"""Variant-aware executor that evaluates SequenceContext objects."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import ClassVar, TYPE_CHECKING

from strand.core.sequence import Sequence
from strand.engine.interfaces import Evaluator, Executor
from strand.engine.types import Metrics, SequenceContext

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class VariantExecutor(Executor):
    """Executor wrapper that handles context-aware evaluation.

    This executor checks if sequences have associated contexts (stored in metadata)
    and routes them to evaluate_batch_with_context if the evaluator supports it.
    Otherwise falls back to standard evaluate_batch.
    """

    evaluator: Evaluator
    batch_size: int = 64

    _LOGGER: ClassVar[logging.Logger] = logging.getLogger(__name__)

    def prepare(self) -> None:
        """Hook for heavy setup."""
        if hasattr(self.evaluator, "prepare"):
            self.evaluator.prepare()

    def run(
        self,
        seqs: list[Sequence],
        *,
        timeout_s: float | None = None,
        batch_size: int | None = None,
    ) -> list[Metrics]:
        """Evaluate sequences, using context-aware evaluation if available.

        Sequences can carry SequenceContext in their metadata under the key
        '_variant_context'. If present and the evaluator supports
        evaluate_batch_with_context, that method will be used.
        """
        if not seqs:
            return []

        effective_batch = batch_size or self.batch_size
        deadline = None if timeout_s is None else time.perf_counter() + timeout_s

        if not hasattr(self.evaluator, "evaluate_batch_with_context"):
            raise RuntimeError("VariantExecutor requires evaluators that implement evaluate_batch_with_context")

        results: list[Metrics] = []
        for start in range(0, len(seqs), effective_batch):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("VariantExecutor run exceeded timeout")

            batch_seqs = seqs[start : start + effective_batch]
            batch_contexts: list[SequenceContext] = []

            for seq in batch_seqs:
                ctx = self._extract_context(seq)
                batch_contexts.append(ctx)

            try:
                batch_results = self.evaluator.evaluate_batch_with_context(batch_contexts)  # type: ignore[attr-defined]
            except TimeoutError:
                raise
            except Exception as exc:  # pragma: no cover - defensive path
                self._LOGGER.exception("Evaluator raised during evaluation", exc_info=exc)
                batch_results = [
                    Metrics(objective=0.0, constraints={}, aux={}) for _ in batch_seqs
                ]

            if len(batch_results) != len(batch_seqs):  # pragma: no cover - defensive path
                raise RuntimeError(
                    "Evaluator returned mismatched batch size: "
                    f"expected {len(batch_seqs)} got {len(batch_results)}"
                )

            results.extend(batch_results)

        return results

    def close(self) -> None:
        """Hook for cleanup work."""
        if hasattr(self.evaluator, "close"):
            self.evaluator.close()

    @staticmethod
    def _extract_context(seq: Sequence) -> SequenceContext:
        if not hasattr(seq, "metadata"):
            raise ValueError("VariantExecutor expected Sequence.metadata to contain '_variant_context'")

        metadata = dict(seq.metadata)
        if "_variant_context" not in metadata:
            raise ValueError(
                "VariantExecutor received a sequence without '_variant_context'. "
                "Ensure VariantStrategy is used for variant triage runs."
            )

        ctx = metadata["_variant_context"]
        if not isinstance(ctx, SequenceContext):
            raise TypeError("'_variant_context' must be a SequenceContext instance")
        return ctx
