"""Abstract optimizer definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from random import Random
from typing import Callable, Iterable, Sequence

from strand.core.sequence import Sequence

ScoreFn = Callable[[Sequence], float]


class BaseOptimizer(ABC):
    """Shared scaffolding for optimization strategies."""

    def __init__(
        self,
        sequences: Sequence[Sequence],
        score_fn: ScoreFn,
        iterations: int,
        population_size: int,
        seed: int | None = None,
    ) -> None:
        self._sequences = list(sequences)
        self._score_fn = score_fn
        self._iterations = max(iterations, 1)
        self._population_size = max(population_size, 1)
        self._random = Random(seed)

    def optimize(self) -> list[tuple[Sequence, float]]:
        population = self.prepare_population()
        return self._score_population(population)

    @abstractmethod
    def prepare_population(self) -> Iterable[Sequence]:
        """Return the population to evaluate."""

    def _score_population(self, population: Iterable[Sequence]) -> list[tuple[Sequence, float]]:
        scored = [(sequence, self._score_fn(sequence)) for sequence in population]
        scored.sort(key=lambda entry: entry[1], reverse=True)
        return scored[: self._population_size]

    def _sample_sequences(self) -> list[Sequence]:
        if len(self._sequences) <= self._population_size:
            return list(self._sequences)
        return self._random.sample(self._sequences, k=self._population_size)
