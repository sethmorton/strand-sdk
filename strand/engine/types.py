"""Shared engine datatypes (surfaces)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strand.core.sequence import Sequence


@dataclass(frozen=True, slots=True)
class VariantMetadata:
    """Metadata for genomic variants."""

    chrom: str
    pos: int
    ref: str
    alt: str
    rsid: str | None = None
    annotations: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SequenceContext:
    """Context-aware sequence with reference/alternative information."""

    ref_seq: Sequence
    alt_seq: Sequence
    metadata: VariantMetadata
    ref_window: tuple[int, int]
    alt_window: tuple[int, int]


@dataclass(frozen=True, slots=True)
class Metrics:
    """Structured evaluation metrics for a single sequence.

    - ``objective``: raw objective (before constraint penalties)
    - ``constraints``: named constraint measurements (e.g., off_target)
    - ``aux``: arbitrary additional metrics for logging/plots
    """

    objective: float
    constraints: Mapping[str, float]
    aux: Mapping[str, float]

