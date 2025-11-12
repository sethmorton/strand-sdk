"""Solubility reward block."""

from __future__ import annotations

from strand.core.sequence import Sequence
from strand.rewards.base import BaseRewardBlock, RewardContext

POLAR = set("STNQ")


class SolubilityReward(BaseRewardBlock):
    def __init__(self, model: str = "heuristic", weight: float = 1.0):
        # "solubility" here is a polar-residue ratio heuristic, not a learned model.
        # The model parameter is a label for provenance only.
        super().__init__(name=f"solubility:{model}", weight=weight)

    def _score(self, sequence: Sequence, context: RewardContext) -> float:
        if len(sequence) == 0:
            return 0.0
        return sum(aa in POLAR for aa in sequence.tokens.upper()) / len(sequence)
