"""Simplified CMA-ES style optimizer."""

from __future__ import annotations

from statistics import mean

from strand.core.sequence import Sequence
from strand.optimizers.base import BaseOptimizer


class CMAESOptimizer(BaseOptimizer):
    def prepare_population(self) -> list[Sequence]:
        population = self._sample_sequences()
        duplicated: list[Sequence] = []
        for _ in range(max(1, self._iterations // 2)):
            duplicated.extend(population)
        return list(population + duplicated)

    def _score_population(self, population: list[Sequence]):
        scored = super()._score_population(population)
        if not scored:
            return scored
        avg_score = mean(score for _, score in scored)
        return [entry for entry in scored if entry[1] >= avg_score]
