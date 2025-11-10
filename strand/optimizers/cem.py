"""Cross-Entropy Method placeholder implementation."""

from __future__ import annotations

from strand.optimizers.base import BaseOptimizer
from strand.core.sequence import Sequence


class CEMOptimizer(BaseOptimizer):
    def prepare_population(self) -> list[Sequence]:
        population = list(self._sample_sequences())
        for _ in range(1, self._iterations):
            population.extend(self._sample_sequences())
        return population
