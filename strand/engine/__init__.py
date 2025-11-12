"""New optimization engine (surfaces)."""

from .constraints import BoundedConstraint, Direction
from .engine import Engine, EngineConfig, EngineResults, IterationStats
from .executors.pool import LocalPoolExecutor
from .interfaces import Evaluator, Executor, Strategy
from .rules import Rules
from .score import default_score
from .strategies import strategy_from_name
from .types import Metrics

__all__ = [
    "Strategy",
    "Evaluator",
    "Executor",
    "Engine",
    "EngineConfig",
    "IterationStats",
    "EngineResults",
    "Metrics",
    "BoundedConstraint",
    "Direction",
    "Rules",
    "default_score",
    "strategy_from_name",
    "LocalPoolExecutor",
]
