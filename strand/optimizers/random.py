"""Random search baseline."""

from __future__ import annotations

from random import shuffle

from strand.core.sequence import Sequence
from strand.optimizers.base import BaseOptimizer


class RandomSearchOptimizer(BaseOptimizer):
    def prepare_population(self) -> list[Sequence]:
        population = list(self._sequences)
        shuffle(population)
        return population[: self._population_size * self._iterations]
