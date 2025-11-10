"""Strand SDK public interface."""

from .core.optimizer import Optimizer
from .core.results import OptimizationResults
from .rewards import RewardBlock

Results = OptimizationResults

__all__ = ["Optimizer", "OptimizationResults", "Results", "RewardBlock"]
__version__ = "0.1.0a0"
