"""Random strategy - baseline proposer.

Samples sequences uniformly from an alphabet and length band. No learning—each
call to ask() generates independent random candidates.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field

from strand.core.sequence import Sequence
from strand.engine.interfaces import Strategy
from strand.engine.types import Metrics


@dataclass
class RandomStrategy(Strategy):
    """Random proposer with configurable alphabet and length band."""

    alphabet: str
    min_len: int
    max_len: int
    seed: int | None = None
    _best_sequence: Sequence | None = field(default=None, init=False, repr=False)
    _best_score: float = field(default=float("-inf"), init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize random state if seed is provided."""
        if self.seed is not None:
            random.seed(self.seed)

    def ask(self, n: int) -> list[Sequence]:
        """Return ``n`` random sequences.

        Each sequence is sampled independently with:
        - Length uniformly in [min_len, max_len]
        - Tokens uniformly from alphabet
        """
        sequences = []
        for i in range(n):
            length = random.randint(self.min_len, self.max_len)
            tokens = "".join(random.choice(self.alphabet) for _ in range(length))
            seq = Sequence(id=f"random_{i}", tokens=tokens)
            sequences.append(seq)
        return sequences

    def tell(self, items: list[tuple[Sequence, float, Metrics]]) -> None:
        """Ingest feedback and track the best observed score."""
        for seq, score, _ in items:
            if score > self._best_score:
                self._best_score = score
                self._best_sequence = seq

    def best(self) -> tuple[Sequence, float] | None:
        """Return the best sequence observed so far, or None if no evaluations yet."""
        if self._best_sequence is None:
            return None
        return (self._best_sequence, self._best_score)

    def state(self) -> Mapping[str, object]:
        """Return serializable state (empty for random)."""
        return {}

