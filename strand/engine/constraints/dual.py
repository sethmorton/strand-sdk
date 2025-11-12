"""Dual variable management for adaptive constraint handling.

Implements CBROP (Constraint-Based Relaxation and Optimization Procedure) style
dual variable updates for constrained optimization.

Reference: Boyd & Parikh, "Distributed optimization and statistical learning via
the alternating direction method of multipliers", Foundations and Trends in
Machine Learning, 2011.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)


@dataclass
class DualVariableManager:
    """Manages a single dual variable for adaptive constraint enforcement.

    Dual variables (Lagrange multipliers) scale penalty weights based on constraint
    violations. Higher violations increase penalties, encouraging feasibility.

    Attributes
    ----------
    init_weight : float
        Initial penalty weight (λ₀).
    min_weight : float
        Minimum weight bound.
    max_weight : float
        Maximum weight bound.
    adaptive_step : float
        Multiplicative step for updates (β in CBROP).
    target_violation : float
        Target violation level (0 = enforce strictly).
    violatION_HISTORY : list[float]
        Track violations over iterations.
    weight_history : list[float]
        Track weight updates over iterations.
    """

    init_weight: float = 1.0
    min_weight: float = 0.01
    max_weight: float = 100.0
    adaptive_step: float = 0.1
    target_violation: float = 0.0

    # State (mutable)
    _current_weight: float = field(default=None, init=False)
    _violation_history: list[float] = field(default_factory=list, init=False)
    _weight_history: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Initialize state."""
        if self._current_weight is None:
            object.__setattr__(self, "_current_weight", self.init_weight)

    @property
    def current_weight(self) -> float:
        """Current penalty weight."""
        return self._current_weight

    @property
    def violation_history(self) -> list[float]:
        """Historical violations."""
        return list(self._violation_history)

    @property
    def weight_history(self) -> list[float]:
        """Historical weights."""
        return list(self._weight_history)

    def update(self, violation: float) -> float:
        """Update weight based on constraint violation.

        Uses CBROP-style adaptive updates:
        - If violation > target: increase weight (enforce more)
        - If violation < target: decrease weight (relax)
        - Clamp to [min_weight, max_weight]

        Parameters
        ----------
        violation : float
            Measured constraint violation (e.g., distance from feasibility).

        Returns
        -------
        float
            Updated weight.
        """
        self._violation_history.append(violation)

        # Compute weight update
        if violation > self.target_violation:
            # Constraint violated: increase penalty
            self._current_weight *= (1.0 + self.adaptive_step)
        else:
            # Constraint satisfied: decrease penalty
            self._current_weight *= (1.0 - self.adaptive_step * 0.5)

        # Clamp to bounds
        self._current_weight = np.clip(
            self._current_weight, self.min_weight, self.max_weight
        )

        self._weight_history.append(self._current_weight)

        return self._current_weight

    def reset(self) -> None:
        """Reset to initial state."""
        object.__setattr__(self, "_current_weight", self.init_weight)
        object.__setattr__(self, "_violation_history", [])
        object.__setattr__(self, "_weight_history", [])

    def summary(self) -> dict[str, float]:
        """Get summary statistics.

        Returns
        -------
        dict[str, float]
            Summary metrics:
            - "num_updates": Number of updates
            - "mean_violation": Average violation
            - "max_violation": Maximum violation
            - "current_weight": Current weight
            - "weight_range": Max - Min weights applied
        """
        if not self._violation_history:
            return {
                "num_updates": 0,
                "mean_violation": 0.0,
                "max_violation": 0.0,
                "current_weight": self.init_weight,
                "weight_range": 0.0,
            }

        violations = np.array(self._violation_history)
        weights = np.array(self._weight_history)

        return {
            "num_updates": len(violations),
            "mean_violation": float(np.mean(violations)),
            "max_violation": float(np.max(violations)),
            "min_violation": float(np.min(violations)),
            "std_violation": float(np.std(violations)),
            "current_weight": float(self._current_weight),
            "init_weight": float(self.init_weight),
            "weight_range": float(np.max(weights) - np.min(weights)) if len(weights) > 0 else 0.0,
        }


@dataclass
class DualVariableSet:
    """Collection of dual managers for multiple constraints.

    Manages all dual variables in a constrained optimization problem.
    """

    managers: dict[str, DualVariableManager] = field(default_factory=dict)

    def add_constraint(
        self,
        name: str,
        init_weight: float = 1.0,
        adaptive_step: float = 0.1,
        **kwargs: object,
    ) -> DualVariableManager:
        """Add a constraint with its dual manager.

        Parameters
        ----------
        name : str
            Constraint identifier.
        init_weight : float
            Initial penalty weight.
        adaptive_step : float
            Adaptive update step.
        **kwargs : object
            Additional parameters for DualVariableManager.

        Returns
        -------
        DualVariableManager
            Created manager.
        """
        manager = DualVariableManager(
            init_weight=init_weight,
            adaptive_step=adaptive_step,
            **kwargs,  # type: ignore[arg-type]
        )
        self.managers[name] = manager
        return manager

    def update_all(self, violations: dict[str, float]) -> dict[str, float]:
        """Update all constraints with their violations.

        Parameters
        ----------
        violations : dict[str, float]
            Mapping of constraint names to violation values.

        Returns
        -------
        dict[str, float]
            New weights for each constraint.
        """
        weights = {}
        for name, violation in violations.items():
            if name in self.managers:
                weights[name] = self.managers[name].update(violation)
            else:
                _LOGGER.warning(f"Unknown constraint: {name}")

        return weights

    def get_weights(self) -> dict[str, float]:
        """Get current weights for all constraints.

        Returns
        -------
        dict[str, float]
            Current weight for each constraint.
        """
        return {name: manager.current_weight for name, manager in self.managers.items()}

    def reset_all(self) -> None:
        """Reset all managers."""
        for manager in self.managers.values():
            manager.reset()

    def summary(self) -> dict[str, dict[str, float]]:
        """Get summary for all constraints.

        Returns
        -------
        dict[str, dict[str, float]]
            Summary metrics for each constraint.
        """
        return {name: manager.summary() for name, manager in self.managers.items()}

    def log_summary(self, logger=None) -> None:
        """Log summary statistics.

        Parameters
        ----------
        logger : logging.Logger | None
            Logger to use (defaults to module logger).
        """
        if logger is None:
            logger = _LOGGER

        summary = self.summary()
        logger.info(f"=== Dual Variable Summary ({len(summary)} constraints) ===")

        for name, stats in summary.items():
            logger.info(f"  {name}:")
            logger.info(f"    Updates: {stats['num_updates']}")
            logger.info(f"    Mean Violation: {stats['mean_violation']:.4f}")
            logger.info(f"    Current Weight: {stats['current_weight']:.4f}")
            logger.info(f"    Weight Range: {stats['weight_range']:.4f}")


__all__ = [
    "DualVariableManager",
    "DualVariableSet",
]

