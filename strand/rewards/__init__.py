"""Reward blocks for sequence optimization.

Progressive Disclosure:
- **Basic**: GC content, length, novelty (start here)
- **Advanced**: Enformer, TFBS (require foundation models)

Import by category:
    from strand.rewards.basic import GCContentBlock
    from strand.rewards.advanced import EnformerRewardBlock
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from strand.rewards.base import RewardContext
from strand.rewards.custom import CustomReward
from strand.rewards.gc_content import GCContentReward
from strand.rewards.length_penalty import LengthPenaltyReward
from strand.rewards.novelty import NoveltyReward
from strand.rewards.registry import RewardRegistry
from strand.rewards.solubility import SolubilityReward
from strand.rewards.stability import StabilityReward

if TYPE_CHECKING:  # pragma: no cover
    from strand.core.sequence import Sequence


class RewardBlock:
    @staticmethod
    def stability(model: str = "heuristic", threshold: float = 0.8, weight: float = 1.0) -> StabilityReward:
        return StabilityReward(model=model, threshold=threshold, weight=weight)

    @staticmethod
    def solubility(model: str = "heuristic", weight: float = 1.0) -> SolubilityReward:
        return SolubilityReward(model=model, weight=weight)

    @staticmethod
    def novelty(baseline: list[str], metric: str = "hamming", weight: float = 1.0) -> NoveltyReward:
        return NoveltyReward(baseline=baseline, metric=metric, weight=weight)

    @staticmethod
    def length_penalty(target_length: int, tolerance: int = 5, weight: float = 1.0) -> LengthPenaltyReward:
        return LengthPenaltyReward(target_length=target_length, tolerance=tolerance, weight=weight)

    @staticmethod
    def gc_content(target: float = 0.5, tolerance: float = 0.1, weight: float = 1.0) -> GCContentReward:
        """Create a GC content reward block.

        Parameters
        ----------
        target : float
            Target GC content ratio (0.0 to 1.0), default 0.5 (50%).
        tolerance : float
            Tolerance band around target (0.0 to 1.0), default 0.1 (±10%).
        weight : float
            Weight multiplier for the reward score, default 1.0.

        Returns
        -------
        GCContentReward
            Configured reward block for GC content optimization.
        """
        return GCContentReward(target=target, tolerance=tolerance, weight=weight)

    @staticmethod
    def custom(
        fn: Callable[[Sequence, RewardContext], float],
        name: str | None = None,
        weight: float = 1.0,
    ) -> CustomReward:
        """Create a custom reward block from a scoring function.

        Args:
            fn: Scoring function that takes (Sequence, RewardContext) and returns float
            name: Optional name for the reward block. Auto-generated from function name if not provided
            weight: Weight multiplier for the reward score (default: 1.0)

        Returns:
            CustomReward: A configured reward block

        Example:
            ```python
            def my_scorer(seq: Sequence, ctx: RewardContext) -> float:
                return 1.0 if "MK" in seq.tokens else 0.0

            # With auto-generated name
            block = RewardBlock.custom(fn=my_scorer, weight=0.5)

            # With explicit name
            block = RewardBlock.custom(fn=my_scorer, name="starts_with_mk", weight=0.5)
            ```
        """
        auto_name = name or f"custom_{fn.__name__}"
        return CustomReward(fn=fn, name=auto_name, weight=weight)

    @staticmethod
    def from_registry(name: str, **kwargs: object) -> object:
        return RewardRegistry.create(name, **kwargs)


__all__ = ["RewardBlock", "RewardRegistry"]
