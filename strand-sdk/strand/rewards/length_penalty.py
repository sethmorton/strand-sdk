"""Length penalty reward block."""

from __future__ import annotations

from strand.core.sequence import Sequence
from strand.rewards.base import BaseRewardBlock, RewardContext


class LengthPenaltyReward(BaseRewardBlock):
    def __init__(self, target_length: int, tolerance: int = 5, weight: float = 1.0):
        super().__init__(name="length_penalty", weight=weight)
        self._target_length = target_length
        self._tolerance = max(tolerance, 1)

    def _score(self, sequence: Sequence, context: RewardContext) -> float:
        delta = abs(len(sequence) - self._target_length)
        return max(0.0, 1 - (delta / self._tolerance))
