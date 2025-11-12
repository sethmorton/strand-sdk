"""Advanced reward blocks - foundation models and complex scoring.

These blocks require additional dependencies (transformers, onnxruntime, etc.)
but provide state-of-the-art cell-type-specific scoring.
"""

from strand.rewards.enformer_block import EnformerRewardBlock, EnformerConfig
from strand.rewards.tfbs_block import TFBSFrequencyCorrelationBlock, TFBSConfig

__all__ = [
    "EnformerRewardBlock",
    "EnformerConfig",
    "TFBSFrequencyCorrelationBlock",
    "TFBSConfig",
]

