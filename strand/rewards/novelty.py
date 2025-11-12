"""Novelty reward block."""

from __future__ import annotations

from strand.core.sequence import Sequence
from strand.rewards.base import BaseRewardBlock, RewardContext
from strand.utils import distances


class NoveltyReward(BaseRewardBlock):
    def __init__(
        self,
        baseline: list[str],
        metric: str = "hamming",
        weight: float = 1.0,
    ):
        super().__init__(name=f"novelty:{metric}", weight=weight)
        if not baseline:
            raise ValueError("baseline sequences are required for novelty scoring")
        self._baseline = baseline
        self._metric = metric

    def _score(self, sequence: Sequence, context: RewardContext) -> float:
        sequences = self._baseline + [sequence.tokens]
        return distances.normalized_score(self._metric, sequences)
