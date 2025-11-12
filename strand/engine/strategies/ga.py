"""Genetic Algorithm strategy (fixed-length, discrete)."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field

from strand.core.sequence import Sequence
from strand.engine.interfaces import Strategy
from strand.engine.types import Metrics


@dataclass
class GAStrategy(Strategy):
    """Genetic Algorithm (fixed-length) with elitism, crossover, and mutation.

    This implementation keeps sequence length fixed to a midpoint between
    ``min_len`` and ``max_len`` for simplicity and stable operators.
    """

    alphabet: str
    min_len: int
    max_len: int
    seed: int | None = None
    elitism: int = 1
    cx_prob: float = 0.6
    mut_prob: float = 0.1

    _rng: random.Random = field(init=False, repr=False)
    _fixed_len: int = field(init=False, repr=False)
    _population: list[Sequence] = field(default_factory=list, init=False, repr=False)
    _best_sequence: Sequence | None = field(default=None, init=False, repr=False)
    _best_score: float = field(default=float("-inf"), init=False, repr=False)
    _counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.alphabet:
            raise ValueError("alphabet must be non-empty")
        if self.min_len <= 0 or self.max_len < self.min_len:
            raise ValueError("invalid length band")
        self._rng = random.Random(self.seed)
        self._fixed_len = (self.min_len + self.max_len) // 2

    def _random_seq(self) -> Sequence:
        tokens = "".join(self._rng.choice(self.alphabet) for _ in range(self._fixed_len))
        seq = Sequence(id=f"ga_{self._counter}", tokens=tokens)
        self._counter += 1
        return seq

    def ask(self, n: int) -> list[Sequence]:
        """Return the current or initial population of size ``n``."""
        if not self._population:
            self._population = [self._random_seq() for _ in range(n)]
        elif len(self._population) != n:
            # Adjust size conservatively
            if len(self._population) > n:
                self._population = self._population[:n]
            else:
                self._population.extend(self._random_seq() for _ in range(n - len(self._population)))
        return list(self._population)

    def tell(self, items: list[tuple[Sequence, float, Metrics]]) -> None:
        """Evolve the population from scored items."""
        if not items:
            return

        # Track best
        for seq, score, _ in items:
            if score > self._best_score:
                self._best_score = score
                self._best_sequence = seq

        # Sort by score (desc)
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        n = len(items_sorted)
        next_pop: list[Sequence] = [seq for (seq, _, _) in items_sorted[: max(0, self.elitism)]]

        def crossover(a: Sequence, b: Sequence) -> Sequence:
            if self._rng.random() > self.cx_prob:
                return Sequence(id=f"ga_{self._counter}", tokens=a.tokens)
            cut = self._rng.randint(1, self._fixed_len - 1)
            child_tokens = a.tokens[:cut] + b.tokens[cut:]
            seq = Sequence(id=f"ga_{self._counter}", tokens=child_tokens)
            self._counter += 1
            return seq

        def mutate(s: Sequence) -> Sequence:
            tokens = list(s.tokens)
            for i in range(self._fixed_len):
                if self._rng.random() < self._mut_prob_per_pos():
                    tokens[i] = self._rng.choice(self.alphabet)
            return Sequence(id=f"ga_{self._counter}", tokens="".join(tokens))

        # Tournament selection helper
        def select() -> Sequence:
            k = 3 if n >= 3 else n
            contenders = self._rng.sample(items_sorted, k=k)
            return max(contenders, key=lambda x: x[1])[0]

        # Fill remaining slots
        while len(next_pop) < n:
            parent_a = select()
            parent_b = select()
            child = crossover(parent_a, parent_b)
            if self._rng.random() < self.mut_prob:
                child = mutate(child)
            next_pop.append(child)

        self._population = next_pop

    def _mut_prob_per_pos(self) -> float:
        # Distribute overall mutation probability per position
        return max(0.0, min(1.0, self.mut_prob / max(1, self._fixed_len)))

    def best(self) -> tuple[Sequence, float] | None:
        if self._best_sequence is None:
            return None
        return (self._best_sequence, self._best_score)

    def state(self) -> Mapping[str, object]:
        return {
            "best_score": self._best_score,
            "population_size": len(self._population),
        }
