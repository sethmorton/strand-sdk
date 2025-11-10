"""Configuration utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(slots=True)
class OptimizerConfig:
    method: str = "cem"
    iterations: int = 50
    population_size: int = 200
    seed: int | None = None
    extra: Mapping[str, float | int | str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "method": self.method,
            "iterations": self.iterations,
            "population_size": self.population_size,
            "seed": self.seed or -1,
            "extra": dict(self.extra),
        }
