"""Serializer helpers for manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def dump_manifest(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
