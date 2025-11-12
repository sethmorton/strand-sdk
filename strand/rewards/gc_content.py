"""GC content reward block for DNA/RNA sequences."""

from __future__ import annotations

from strand.core.sequence import Sequence
from strand.rewards.base import BaseRewardBlock, RewardContext


class GCContentReward(BaseRewardBlock):
    """Reward based on GC content similarity to a target value.

    GC content is the fraction of G and C nucleotides in a DNA/RNA sequence.
    This reward incentivizes sequences with GC content within a target band,
    useful for optimizing melting temperature and other sequence properties.

    Parameters
    ----------
    target : float
        Target GC content ratio (0.0 to 1.0), default 0.5 (50%).
    tolerance : float
        Tolerance band around target (0.0 to 1.0), default 0.1 (±10%).
        Score is 1.0 when GC content is within [target-tolerance, target+tolerance].
    weight : float
        Weight multiplier for the reward score, default 1.0.

    Example
    -------
    ```python
    from strand.rewards import RewardBlock

    # Target 40% GC with ±5% tolerance
    gc_reward = RewardBlock.gc_content(target=0.4, tolerance=0.05, weight=1.0)

    # In a DNA optimization pipeline
    rewards = [
        RewardBlock.stability(weight=1.0),
        RewardBlock.gc_content(target=0.5, tolerance=0.1, weight=0.5),
    ]
    ```
    """

    def __init__(
        self,
        target: float = 0.5,
        tolerance: float = 0.1,
        weight: float = 1.0,
    ) -> None:
        """Initialize GC content reward."""
        super().__init__(name=f"gc_content:{target:.2f}", weight=weight)
        self._target = float(target)
        self._tolerance = float(tolerance)

        if not (0.0 <= self._target <= 1.0):
            raise ValueError(f"target must be in [0.0, 1.0], got {self._target}")
        if not (0.0 <= self._tolerance <= 1.0):
            raise ValueError(f"tolerance must be in [0.0, 1.0], got {self._tolerance}")

    def _score(self, sequence: Sequence, context: RewardContext) -> float:  # noqa: ARG002
        """Compute GC content reward.

        Returns 1.0 when GC content is within the tolerance band of the target,
        and decreases linearly outside the band, reaching 0.0 at distance >= 1.0.

        Returns
        -------
        float
            Reward score in [0.0, 1.0].
        """
        if len(sequence) == 0:
            return 0.0

        # Count G and C nucleotides (case-insensitive)
        tokens_upper = sequence.tokens.upper()
        gc_count = tokens_upper.count("G") + tokens_upper.count("C")
        gc_content = gc_count / len(sequence)

        # Compute deviation from target
        deviation = abs(gc_content - self._target)

        # If within tolerance, return 1.0
        if deviation <= self._tolerance:
            return 1.0

        # Otherwise, linearly decay from 1.0 to 0.0 as deviation increases
        # At deviation = self._tolerance + 1.0, score = 0.0
        decay_distance = 1.0
        excess_deviation = deviation - self._tolerance
        return max(0.0, 1.0 - excess_deviation / decay_distance)

