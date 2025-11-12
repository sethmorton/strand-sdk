"""Hybrid Strategy: Ensemble of multiple strategies working in parallel.

Combines multiple strategies (Random, CEM, GA, CMA-ES, etc.) to explore
the search space more effectively. Picks the best candidates from each
strategy each generation.

Inspired by portfolio/ensemble approaches in optimization.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

from strand.core.sequence import Sequence
from strand.engine.interfaces import Strategy
from strand.engine.runtime import StrategyCaps, StrategyContext, resolve_strategy_caps
from strand.engine.types import Metrics


@dataclass
class HybridStrategy(Strategy):
    """Ensemble strategy combining multiple sub-strategies.

    Parameters
    ----------
    strategies : list[Strategy]
        List of strategy instances to run in parallel.
    selection_method : str
        How to select from multiple strategies:
        - "round-robin": Alternate between strategies
        - "best-of-generation": Keep best from each strategy
        - "weighted": Weight strategies by past performance
    """

    strategies: list[Strategy]
    selection_method: str = "best-of-generation"

    # Internal state
    _strategy_idx: int = field(default=0, init=False, repr=False)
    _best_sequence: Sequence | None = field(default=None, init=False, repr=False)
    _best_score: float = field(default=float("-inf"), init=False, repr=False)
    _strategy_scores: list[float] = field(default_factory=list, init=False, repr=False)
    _caps: StrategyCaps = field(init=False, repr=False)
    _last_sequences: list[list[Sequence]] = field(default_factory=list, init=False, repr=False)
    _pending_sequence_map: dict[tuple[str, str], deque[int]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize strategy ensemble."""
        if not self.strategies:
            raise ValueError("Must provide at least one strategy")
        if self.selection_method not in ("round-robin", "best-of-generation", "weighted"):
            raise ValueError(
                f"Unknown selection_method: {self.selection_method}. "
                "Choose from: round-robin, best-of-generation, weighted"
            )
        # Initialize strategy scores for weighted selection
        self._strategy_scores = [0.0] * len(self.strategies)
        self._last_sequences = [[] for _ in self.strategies]
        self._pending_sequence_map = {}
        self._refresh_caps()

    def ask(self, n: int) -> list[Sequence]:
        """Ask all strategies for candidates.

        Parameters
        ----------
        n : int
            Total number of candidates to generate. Distributed across strategies.

        Returns
        -------
        list[Sequence]
            Sequences from all strategies (may exceed n if using ensemble approach).
        """
        if self.selection_method == "round-robin":
            # Rotate through strategies
            strategy_idx = self._strategy_idx % len(self.strategies)
            strategy = self.strategies[strategy_idx]
            self._strategy_idx += 1

            sequences = list(strategy.ask(n))
            sequences_by_strategy = [[] for _ in self.strategies]
            sequences_by_strategy[strategy_idx] = sequences
            self._register_sequences(sequences_by_strategy)
            return sequences

        if self.selection_method == "best-of-generation":
            # Get candidates from all strategies
            candidates: list[Sequence] = []
            candidates_per_strategy = n // len(self.strategies)
            remainder = n % len(self.strategies)

            sequences_by_strategy: list[list[Sequence]] = [[] for _ in self.strategies]
            for i, strategy in enumerate(self.strategies):
                count = candidates_per_strategy + (1 if i < remainder else 0)
                sequences = list(strategy.ask(count))
                sequences_by_strategy[i] = sequences
                candidates.extend(sequences)

            self._register_sequences(sequences_by_strategy)
            return candidates

        if self.selection_method == "weighted":
            # Allocate based on past performance
            total_score = sum(max(0.01, s) for s in self._strategy_scores)
            weights = [max(0.01, s) / total_score for s in self._strategy_scores]

            generated: list[tuple[Sequence, int]] = []
            sequences_by_strategy: list[list[Sequence]] = [[] for _ in self.strategies]
            for i, strategy in enumerate(self.strategies):
                count = max(1, int(n * weights[i]))
                sequences = list(strategy.ask(count))
                generated.extend((sequence, i) for sequence in sequences)

            selected = generated[:n]
            for sequence, idx in selected:
                sequences_by_strategy[idx].append(sequence)

            self._register_sequences(sequences_by_strategy)
            return [sequence for sequence, _ in selected]

        raise ValueError(f"Unknown selection_method: {self.selection_method}")

    def tell(self, items: list[tuple[Sequence, float, Metrics]]) -> None:
        """Ingest evaluated candidates and update all strategies.

        Parameters
        ----------
        items : list[tuple[Sequence, float, Metrics]]
            List of (sequence, score, metrics) tuples.
        """
        if not items:
            return

        # Track best overall
        for seq, score, _ in items:
            if score > self._best_score:
                self._best_score = score
                self._best_sequence = seq

        # Route results back to generating strategies when possible
        if not any(self._last_sequences):
            for strategy in self.strategies:
                strategy.tell(items)
            return

        items_by_strategy: list[list[tuple[Sequence, float, Metrics]]] = [
            [] for _ in self.strategies
        ]
        for seq, score, metrics in items:
            key = (seq.id, seq.tokens)
            indices = self._pending_sequence_map.get(key)
            if not indices:
                for strategy in self.strategies:
                    strategy.tell(items)
                self._pending_sequence_map.clear()
                self._last_sequences = [[] for _ in self.strategies]
                return

            strategy_idx = indices.popleft()
            items_by_strategy[strategy_idx].append((seq, score, metrics))
            if not indices:
                del self._pending_sequence_map[key]

        for strategy, strategy_items in zip(self.strategies, items_by_strategy):
            if strategy_items:
                strategy.tell(strategy_items)

        self._pending_sequence_map.clear()
        self._last_sequences = [[] for _ in self.strategies]

        # Update strategy scores for weighted selection
        if self.selection_method == "weighted":
            for i, strategy in enumerate(self.strategies):
                best = strategy.best()
                if best:
                    self._strategy_scores[i] = best[1]

    def strategy_caps(self) -> StrategyCaps:
        return self._caps

    def prepare(self, context: StrategyContext) -> None:
        for strategy in self.strategies:
            prepare_fn = getattr(strategy, "prepare", None)
            if callable(prepare_fn):
                prepare_fn(context)

    def train_step(
        self,
        items: list[tuple[Sequence, float, Metrics]],
        context: StrategyContext,
    ) -> None:
        if not items:
            return
        for strategy in self.strategies:
            caps = resolve_strategy_caps(strategy)
            if not caps.supports_fine_tuning:
                continue
            train_fn = getattr(strategy, "train_step", None)
            if callable(train_fn):
                train_fn(items, context)

    def best(self) -> tuple[Sequence, float] | None:
        """Return the best sequence observed across all strategies.

        Returns
        -------
        tuple[Sequence, float] | None
            (best_sequence, best_score) or None if no sequences evaluated.
        """
        if self._best_sequence is None:
            return None
        return (self._best_sequence, self._best_score)

    def state(self) -> Mapping[str, object]:
        """Return serializable state of all sub-strategies.

        Returns
        -------
        Mapping[str, object]
            Combined state from all strategies.
        """
        return {
            "best_score": self._best_score,
            "selection_method": self.selection_method,
            "num_strategies": len(self.strategies),
            "strategy_scores": self._strategy_scores,
            "strategy_states": [s.state() for s in self.strategies],
        }

    def add_strategy(self, strategy: Strategy) -> None:
        """Add a new strategy to the ensemble at runtime.

        Parameters
        ----------
        strategy : Strategy
            Strategy instance to add.
        """
        self.strategies.append(strategy)
        self._strategy_scores.append(0.0)
        self._last_sequences.append([])
        self._refresh_caps()

    def remove_strategy(self, index: int) -> None:
        """Remove a strategy from the ensemble.

        Parameters
        ----------
        index : int
            Index of strategy to remove.
        """
        if index < 0 or index >= len(self.strategies):
            raise IndexError(f"Strategy index {index} out of range")
        if len(self.strategies) == 1:
            raise ValueError("Cannot remove the last strategy")
        self.strategies.pop(index)
        self._strategy_scores.pop(index)
        self._last_sequences.pop(index)
        self._refresh_caps()

    def get_strategy_performance(self) -> list[tuple[str, float]]:
        """Get performance of each strategy.

        Returns
        -------
        list[tuple[str, float]]
            List of (strategy_name, best_score) for each strategy.
        """
        results = []
        for i, strategy in enumerate(self.strategies):
            best = strategy.best()
            score = best[1] if best else 0.0
            name = f"Strategy {i}: {type(strategy).__name__}"
            results.append((name, score))
        return results

    def _refresh_caps(self) -> None:
        self._caps = self._aggregate_caps()

    def _aggregate_caps(self) -> StrategyCaps:
        child_caps = [resolve_strategy_caps(strategy) for strategy in self.strategies]
        requires_runtime = any(cap.requires_runtime for cap in child_caps)
        supports_fine_tuning = any(cap.supports_fine_tuning for cap in child_caps)
        max_tokens_values = [cap.max_tokens_per_batch for cap in child_caps if cap.max_tokens_per_batch]
        max_tokens = min(max_tokens_values) if max_tokens_values else None
        disable_autocast = any(not cap.prefers_autocast for cap in child_caps)
        # If any strategy opts out of autocast, honor that; otherwise default True
        return StrategyCaps(
            requires_runtime=requires_runtime,
            supports_fine_tuning=supports_fine_tuning,
            max_tokens_per_batch=max_tokens,
            prefers_autocast=not disable_autocast,
        )

    def _register_sequences(self, sequences_by_strategy: list[list[Sequence]]) -> None:
        self._last_sequences = [list(seqs) for seqs in sequences_by_strategy]
        self._pending_sequence_map = {}
        for idx, seqs in enumerate(self._last_sequences):
            for seq in seqs:
                key = (seq.id, seq.tokens)
                if key not in self._pending_sequence_map:
                    self._pending_sequence_map[key] = deque()
                self._pending_sequence_map[key].append(idx)
