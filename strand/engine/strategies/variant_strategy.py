"""Variant-aware strategy that yields sequences from variant contexts."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from strand.core.sequence import Sequence
from strand.engine.interfaces import Strategy
from strand.engine.types import Metrics

if TYPE_CHECKING:
    from collections.abc import Mapping
    from strand.engine.types import SequenceContext

logger = logging.getLogger(__name__)


@dataclass
class VariantStrategy(Strategy):
    """Strategy that yields sequences from a variant dataset.

    This strategy wraps a VariantDataset and yields sequences from its contexts.
    It stores the context mapping so executors can extract contexts for evaluation.
    """

    contexts: list["SequenceContext"]
    current_idx: int = 0
    _best_sequence: Sequence | None = None
    _best_score: float = float("-inf")

    def ask(self, n: int) -> list[Sequence]:
        """Return next n sequences from variant contexts.

        Sequences are wrapped with their context in metadata for context-aware
        evaluation by VariantExecutor.
        """
        if self.current_idx >= len(self.contexts):
            return []

        end_idx = min(self.current_idx + n, len(self.contexts))
        batch_contexts = self.contexts[self.current_idx:end_idx]
        self.current_idx = end_idx

        # Wrap contexts as sequences, storing context in metadata
        sequences = []
        for ctx in batch_contexts:
            # Use alt_seq as the primary sequence for optimization
            seq = Sequence(
                id=ctx.alt_seq.id,
                tokens=ctx.alt_seq.tokens,
                metadata={
                    **dict(ctx.alt_seq.metadata),
                    "_variant_context": ctx,  # Store context for executor
                },
            )
            sequences.append(seq)

        return sequences

    def tell(self, items: list[tuple[Sequence, float, Metrics]]) -> None:
        """Track the best-scoring variant sequence."""
        for seq, score, _ in items:
            if score > self._best_score:
                self._best_score = score
                self._best_sequence = seq

    def best(self) -> tuple[Sequence, float] | None:
        """Return best variant observed so far."""
        if self._best_sequence is None:
            return None
        return self._best_sequence, self._best_score

    def state(self) -> "Mapping[str, object]":
        """Return strategy state."""
        return {
            "type": "variant_strategy",
            "total_contexts": len(self.contexts),
            "current_idx": self.current_idx,
            "best_score": self._best_score,
        }
