"""Genetic Algorithm strategy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from strand.core.sequence import Sequence
from strand.engine.interfaces import Strategy
from strand.engine.types import Metrics


@dataclass
class GAStrategy(Strategy):
    """Genetic Algorithm optimization strategy (surface).

    Implementation pending.
    """

    alphabet: str
    min_len: int
    max_len: int
    seed: int | None = None

    def ask(self, n: int) -> list[Sequence]:
        """Return ``n`` candidate sequences (implementation pending)."""
        raise NotImplementedError("GAStrategy.ask() pending implementation.")

    def tell(self, items: list[tuple[Sequence, float, Metrics]]) -> None:
        """Ingest evaluated candidates and evolve population (implementation pending)."""
        raise NotImplementedError("GAStrategy.tell() pending implementation.")

    def best(self) -> tuple[Sequence, float] | None:
        """Return best sequence found so far (implementation pending)."""
        raise NotImplementedError("GAStrategy.best() pending implementation.")

    def state(self) -> Mapping[str, object]:
        """Return serializable strategy state (implementation pending)."""
        raise NotImplementedError("GAStrategy.state() pending implementation.")
