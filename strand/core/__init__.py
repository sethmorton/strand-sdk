"""Core primitives.

This module now exposes only low-level, stable types used across the SDK. The
legacy Optimizer facade has been removed in favor of the new engine surfaces
under ``strand.engine``.
"""

from .results import OptimizationResults
from .sequence import Sequence

__all__ = ["OptimizationResults", "Sequence"]
