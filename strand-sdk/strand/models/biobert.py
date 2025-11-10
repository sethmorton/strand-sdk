"""BioBERT integration placeholder."""

from __future__ import annotations

from random import Random


class BioBERTModel:
    def __init__(self, model_name: str = "biobert-base", seed: int | None = None) -> None:
        self.model_name = model_name
        self._rng = Random(seed)

    def novelty(self, sequence: str) -> float:
        return self._rng.random()
