"""Sequence data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(slots=True)
class Sequence:
    """Simple immutable representation of a biological sequence."""

    id: str
    tokens: str
    metadata: Mapping[str, int | float | str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "id": self.id,
            "tokens": self.tokens,
            "metadata": dict(self.metadata),
        }

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self.tokens)
