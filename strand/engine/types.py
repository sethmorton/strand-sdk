"""Shared engine datatypes (surfaces)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Metrics:
    """Structured evaluation metrics for a single sequence.

    - ``objective``: raw objective (before constraint penalties)
    - ``constraints``: named constraint measurements (e.g., off_target)
    - ``aux``: arbitrary additional metrics for logging/plots
    """

    objective: float
    constraints: Mapping[str, float]
    aux: Mapping[str, float]

