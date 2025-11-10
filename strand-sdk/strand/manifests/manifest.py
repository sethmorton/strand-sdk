"""Manifest data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from strand.manifests.serializers import dump_manifest, load_manifest


@dataclass(slots=True)
class Manifest:
    run_id: str
    timestamp: datetime
    experiment: str
    inputs: dict[str, Any]
    optimizer: dict[str, Any]
    reward_blocks: list[dict[str, Any]] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "experiment": self.experiment,
            "inputs": self.inputs,
            "optimizer": self.optimizer,
            "reward_blocks": self.reward_blocks,
            "results": self.results,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        dump_manifest(self.to_dict(), target)
        return target

    @staticmethod
    def load(path: str | Path) -> "Manifest":
        data = load_manifest(Path(path))
        return Manifest(
            run_id=data["run_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            experiment=data["experiment"],
            inputs=data["inputs"],
            optimizer=data["optimizer"],
            reward_blocks=data["reward_blocks"],
            results=data["results"],
        )
