"""Scoring helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from strand.engine.constraints import BoundedConstraint
from strand.engine.types import Metrics

_LOGGER = logging.getLogger(__name__)
_MISSING_CONSTRAINT_WARNINGS: set[str] = set()


def default_score(
    metrics: Metrics,
    rules: Mapping[str, float],
    constraints: list[BoundedConstraint],
) -> float:
    """Turn metrics into a single score.

    Formula: ``objective − Σ rules[name] × violation(name)``.

    Each constraint adds a penalty equal to its rule weight times the violation
    amount. Missing constraint readings count as zero (with a single warning).
    Higher scores are better.
    """
    constraint_values = dict(metrics.constraints)
    penalty = 0.0
    for constraint in constraints:
        name = constraint.name
        if name in constraint_values:
            measured_value = constraint_values[name]
        else:
            measured_value = 0.0
            if name not in _MISSING_CONSTRAINT_WARNINGS:
                _LOGGER.warning(
                    "Constraint '%s' missing from metrics; treating violation as 0.0",
                    name,
                )
                _MISSING_CONSTRAINT_WARNINGS.add(name)

        violation = constraint.violation(measured_value)
        weight = rules.get(name, 0.0)
        penalty += weight * violation

    return metrics.objective - penalty


__all__ = ["default_score"]
