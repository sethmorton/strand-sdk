"""Simple in-memory cache for model embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EmbeddingCache:
    capacity: int = 512
    store: dict[str, list[float]] = field(default_factory=dict)

    def get(self, key: str) -> list[float] | None:
        return self.store.get(key)

    def set(self, key: str, value: list[float]) -> None:
        if len(self.store) >= self.capacity:
            first_key = next(iter(self.store))
            self.store.pop(first_key)
        self.store[key] = value
