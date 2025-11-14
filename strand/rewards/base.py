"""Reward block abstractions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from strand.core.sequence import Sequence
    from strand.engine.types import SequenceContext


class ObjectiveType(Enum):
    """Objective optimization direction."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class BlockType(Enum):
    """Reward block classification by complexity and determinism."""

    HEURISTIC = "heuristic"  # Fast, rule-based scoring
    ADVANCED = "advanced"    # ML-based, context-aware
    DETERMINISTIC = "deterministic"  # Foundation model predictions


@dataclass(slots=True)
class RewardContext:
    iteration: int = 0
    metadata: Mapping[str, int | float | str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RewardBlockMetadata:
    """Metadata for reward block classification and behavior."""

    block_type: BlockType
    description: str
    requires_context: bool = False


@runtime_checkable
class RewardBlockProtocol(Protocol):
    name: str
    weight: float
    metadata: RewardBlockMetadata

    def score(self, sequence: Sequence, *, context: RewardContext | None = None) -> float:  # noqa: D401
        """Return the weighted score for the given sequence."""

    def score_context(self, context: "SequenceContext", *, reward_context: RewardContext | None = None) -> tuple[float, dict[str, float]]:  # noqa: D401, E501
        """Return (objective, aux_metrics) for context-aware evaluation.

        Args:
            context: SequenceContext with ref/alt sequences and metadata
            reward_context: Optional iteration context

        Returns:
            Tuple of (objective_score, auxiliary_metrics_dict)
        """
        ...  # pragma: no cover


@dataclass(slots=True)
class BaseRewardBlock:
    name: str
    weight: float = 1.0
    metadata: RewardBlockMetadata = field(default_factory=lambda: RewardBlockMetadata(
        block_type=BlockType.HEURISTIC,
        description="Base reward block",
        requires_context=False
    ))

    def score(self, sequence: Sequence, *, context: RewardContext | None = None) -> float:
        return self.weight * self._score(sequence, context or RewardContext())

    def _score(self, sequence: Sequence, context: RewardContext) -> float:
        raise NotImplementedError

    def __add__(self, other: BaseRewardBlock | list[BaseRewardBlock]) -> list[BaseRewardBlock]:
        """Compose reward blocks into a list for convenient chaining."""
        if isinstance(other, list):
            return [self] + other
        return [self, other]

    def score_context(self, context: "SequenceContext", *, reward_context: RewardContext | None = None) -> tuple[float, dict[str, float]]:  # noqa: E501
        """Default implementation falls back to score() method."""
        # For backward compatibility, default to scoring the alt sequence
        objective = self.score(context.alt_seq, context=reward_context or RewardContext())
        return objective, {}

    def __radd__(self, other: int | list[BaseRewardBlock]) -> list[BaseRewardBlock]:
        """Support sum() and prepending to lists."""
        if isinstance(other, int):  # sum() starts with 0
            return [self]
        if isinstance(other, list):
            return other + [self]
        return [self, other]
