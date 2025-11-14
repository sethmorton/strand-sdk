"""Sequence data structures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(slots=True)
class Sequence:
    """Simple immutable representation of a biological sequence."""

    id: str
    tokens: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "tokens": self.tokens,
            "metadata": dict(self.metadata),  # type: ignore[dict-item]
        }

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self.tokens)
