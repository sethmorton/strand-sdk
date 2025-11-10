"""Toy genetic algorithm implementation."""

from __future__ import annotations

from strand.core.sequence import Sequence
from strand.optimizers.base import BaseOptimizer


class GeneticAlgorithmOptimizer(BaseOptimizer):
    def prepare_population(self) -> list[Sequence]:
        parents = self._sample_sequences()
        offspring: list[Sequence] = []
        for idx in range(0, len(parents), 2):
            parent_a = parents[idx]
            parent_b = parents[(idx + 1) % len(parents)]
            midpoint = max(1, min(len(parent_a), len(parent_b)) // 2)
            child_tokens = parent_a.tokens[:midpoint] + parent_b.tokens[midpoint:]
            child = Sequence(id=f"child-{idx}", tokens=child_tokens)
            offspring.append(child)
        return parents + offspring
