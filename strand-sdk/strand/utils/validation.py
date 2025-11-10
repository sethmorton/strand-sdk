"""Validation helpers for Strand."""

from __future__ import annotations

from typing import Iterable

from strand.core.sequence import Sequence


ALLOWED_TOKENS = set("ACDEFGHIKLMNPQRSTVWY") | set("ACGTU")


def ensure_sequences(sequences: Iterable[Sequence | str]) -> list[Sequence]:
    """Normalize raw strings into `Sequence` objects with validation."""
    result: list[Sequence] = []
    for idx, entry in enumerate(sequences):
        if isinstance(entry, Sequence):
            candidate = entry
        else:
            candidate = Sequence(id=f"seq-{idx}", tokens=str(entry))

        if not candidate.tokens:
            msg = "Empty biological sequences are not allowed"
            raise ValueError(msg)
        invalid = {char for char in candidate.tokens if char.upper() not in ALLOWED_TOKENS}
        if invalid:
            msg = f"Sequence {candidate.id} contains invalid tokens: {sorted(invalid)}"
            raise ValueError(msg)
        result.append(candidate)
    return result
