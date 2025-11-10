"""Result containers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from strand.core.sequence import Sequence
from strand.manifests import Manifest


@dataclass(slots=True)
class OptimizationResults:
    ranked_sequences: list[Sequence]
    scores: list[float]
    manifest: Manifest | None = None

    def __post_init__(self) -> None:
        if len(self.ranked_sequences) != len(self.scores):
            msg = "Sequences and scores must be aligned"
            raise ValueError(msg)

    def top(self, limit: int = 5) -> list[tuple[Sequence, float]]:
        return list(zip(self.ranked_sequences[:limit], self.scores[:limit]))

    def export_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"sequence": seq.to_dict(), "score": score}
            for seq, score in zip(self.ranked_sequences, self.scores)
        ]
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def export_csv(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["rank", "sequence_id", "sequence", "score"])
            for idx, (seq, score) in enumerate(zip(self.ranked_sequences, self.scores), start=1):
                writer.writerow([idx, seq.id, seq.tokens, score])
        return target

    def attach_manifest(self, manifest: Manifest) -> None:
        self.manifest = manifest

    def to_manifest(self) -> Manifest | None:
        return self.manifest
