"""Reward block factory exports."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from strand.rewards.base import RewardContext
from strand.rewards.custom import CustomReward
from strand.rewards.length_penalty import LengthPenaltyReward
from strand.rewards.novelty import NoveltyReward
from strand.rewards.registry import RewardRegistry
from strand.rewards.solubility import SolubilityReward
from strand.rewards.stability import StabilityReward

if TYPE_CHECKING:  # pragma: no cover
    from strand.core.sequence import Sequence


class RewardBlock:
    @staticmethod
    def stability(model: str = "esmfold", threshold: float = 0.8, weight: float = 1.0) -> StabilityReward:
        return StabilityReward(model=model, threshold=threshold, weight=weight)

    @staticmethod
    def solubility(model: str = "protbert", weight: float = 1.0) -> SolubilityReward:
        return SolubilityReward(model=model, weight=weight)

    @staticmethod
    def novelty(baseline: list[str], metric: str = "hamming", weight: float = 1.0) -> NoveltyReward:
        return NoveltyReward(baseline=baseline, metric=metric, weight=weight)

    @staticmethod
    def length_penalty(target_length: int, tolerance: int = 5, weight: float = 1.0) -> LengthPenaltyReward:
        return LengthPenaltyReward(target_length=target_length, tolerance=tolerance, weight=weight)

    @staticmethod
    def custom(
        name: str,
        fn: Callable[["Sequence", RewardContext], float],
        weight: float = 1.0,
    ) -> CustomReward:
        return CustomReward(fn=fn, name=name, weight=weight)

    @staticmethod
    def from_registry(name: str, **kwargs):
        return RewardRegistry.create(name, **kwargs)


__all__ = ["RewardBlock", "RewardRegistry"]
