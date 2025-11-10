"""Custom reward block wrapper."""

from __future__ import annotations

from collections.abc import Callable

from strand.core.sequence import Sequence
from strand.rewards.base import BaseRewardBlock, RewardContext


class CustomReward(BaseRewardBlock):
    def __init__(
        self,
        fn: Callable[[Sequence, RewardContext], float],
        name: str = "custom",
        weight: float = 1.0,
    ):
        super().__init__(name=name, weight=weight)
        self._fn = fn

    def _score(self, sequence: Sequence, context: RewardContext) -> float:
        return self._fn(sequence, context)
