"""Stability reward block."""

from __future__ import annotations

from strand.core.sequence import Sequence
from strand.rewards.base import BaseRewardBlock, RewardContext

HYDROPHOBIC = set("AILMFWYV")


class StabilityReward(BaseRewardBlock):
    def __init__(self, model: str = "heuristic", threshold: float = 0.8, weight: float = 1.0):
        # "stability" here is a hydrophobicity-derived heuristic, not a learned model.
        # The model parameter is a label for provenance only.
        super().__init__(name=f"stability:{model}", weight=weight)
        self._threshold = threshold

    def _score(self, sequence: Sequence, context: RewardContext) -> float:
        if len(sequence) == 0:
            return 0.0
        ratio = sum(aa in HYDROPHOBIC for aa in sequence.tokens.upper()) / len(sequence)
        return 1.0 if ratio >= self._threshold else ratio
