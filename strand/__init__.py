"""Strand SDK public interface (surfaces only).

Use the unified Engine surface under ``strand.engine``. Rewards remain available
under ``strand.rewards``.
"""

from __future__ import annotations

from .rewards import RewardBlock

__all__ = [
    "RewardBlock",
]

__version__ = "0.2.0a0"
