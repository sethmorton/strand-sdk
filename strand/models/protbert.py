"""ProtBERT integration placeholder."""

from __future__ import annotations

from random import Random


class ProtBERTModel:
    def __init__(self, model_name: str = "protbert-base", seed: int | None = None) -> None:
        self.model_name = model_name
        self._rng = Random(seed)

    def solubility(self, sequence: str) -> float:
        return 0.4 + self._rng.random() / 2
