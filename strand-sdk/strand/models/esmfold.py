"""ESMFold integration placeholder."""

from __future__ import annotations

from random import Random


class ESMFoldModel:
    def __init__(self, checkpoint: str = "esmfold_v1", seed: int | None = None) -> None:
        self.checkpoint = checkpoint
        self._rng = Random(seed)

    def stability(self, sequence: str) -> float:
        return 0.5 + self._rng.random() / 2
