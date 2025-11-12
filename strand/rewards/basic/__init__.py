"""Basic reward blocks - start here for simple optimizations."""

from strand.rewards.gc_content import GCContentBlock
from strand.rewards.length_penalty import LengthPenaltyBlock
from strand.rewards.novelty import NoveltyBlock
from strand.rewards.stability import StabilityBlock
from strand.rewards.solubility import SolubilityBlock

__all__ = [
    "GCContentBlock",
    "LengthPenaltyBlock",
    "NoveltyBlock",
    "StabilityBlock",
    "SolubilityBlock",
]

