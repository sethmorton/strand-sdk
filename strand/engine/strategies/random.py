"""Random strategy that proposes sequences uniformly at random."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field

from strand.core.sequence import Sequence
from strand.engine.interfaces import Strategy
from strand.engine.types import Metrics


@dataclass
class RandomStrategy(Strategy):
    """Sample sequences with a fixed alphabet and length range."""

    alphabet: str
    min_len: int
    max_len: int
    seed: int | None = None
    _best_sequence: Sequence | None = field(default=None, init=False, repr=False)
    _best_score: float = field(default=float("-inf"), init=False, repr=False)
    _counter: int = field(default=0, init=False, repr=False)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the settings and initialize the random generator."""
        if not self.alphabet:
            raise ValueError("alphabet must be a non-empty string")
        if self.min_len <= 0:
            raise ValueError("min_len must be positive")
        if self.max_len < self.min_len:
            raise ValueError("max_len must be >= min_len")

        self._rng = random.Random(self.seed)

    def ask(self, n: int) -> list[Sequence]:
        """Return ``n`` random sequences."""

        sequences: list[Sequence] = []
        for _ in range(n):
            length = self._rng.randint(self.min_len, self.max_len)
            tokens = "".join(self._rng.choice(self.alphabet) for _ in range(length))
            seq = Sequence(id=f"random_{self._counter}", tokens=tokens)
            self._counter += 1
            sequences.append(seq)
        return sequences

    def tell(self, items: list[tuple[Sequence, float, Metrics]]) -> None:
        """Record feedback and track the best score seen so far."""

        for seq, score, _ in items:
            if score > self._best_score:
                self._best_score = score
                self._best_sequence = seq

    def best(self) -> tuple[Sequence, float] | None:
        """Return the best sequence seen so far, if any."""

        if self._best_sequence is None:
            return None
        return (self._best_sequence, self._best_score)

    def state(self) -> Mapping[str, object]:
        """Return serializable state (empty for RandomStrategy)."""

        return {"seed": self.seed, "generated": self._counter}

