"""Cross-Entropy Method strategy for discrete fixed-length sequences."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from strand.core.sequence import Sequence
from strand.engine.interfaces import Strategy
from strand.engine.runtime import StrategyCaps
from strand.engine.types import Metrics


@dataclass
class CEMStrategy(Strategy):
    """Cross-Entropy Method optimization strategy (discrete, fixed-length).

    Maintains per-position categorical probabilities initialized uniformly.
    On tell(), selects top-K elite sequences and updates probabilities with smoothing.

    Parameters
    ----------
    alphabet : str
        Available characters for sequences.
    min_len : int
        Minimum sequence length.
    max_len : int
        Maximum sequence length.
    seed : int | None
        Random seed for reproducibility.
    elite_frac : float
        Fraction of top candidates to select as elite (default: 0.2).
    beta : float
        Smoothing parameter for probability update (default: 0.5).
        P_new = (1 - beta) * P_old + beta * P_elite
    eps : float
        Minimum probability to avoid numerical issues (default: 1e-3).
    """

    alphabet: str
    min_len: int
    max_len: int
    seed: int | None = None
    elite_frac: float = 0.2
    beta: float = 0.5
    eps: float = 1e-3

    # Internal state
    _fixed_len: int | None = field(default=None, init=False, repr=False)
    _probs: list[np.ndarray] | None = field(default=None, init=False, repr=False)
    _best_sequence: Sequence | None = field(default=None, init=False, repr=False)
    _best_score: float = field(default=float("-inf"), init=False, repr=False)
    _rng: random.Random = field(default_factory=random.Random, init=False, repr=False)
    _CAPS: ClassVar[StrategyCaps] = StrategyCaps()

    def __post_init__(self) -> None:
        """Initialize random generator and internal state."""
        self._rng = random.Random(self.seed)
        # Fixed-length mode: use mid-point
        self._fixed_len = (self.min_len + self.max_len) // 2

    def _init_probs(self) -> None:
        """Initialize position-wise categorical probabilities."""
        if self._probs is not None:
            return
        n_pos = self._fixed_len
        n_chars = len(self.alphabet)
        # Uniform probability: [1/n_chars, 1/n_chars, ...]
        self._probs = [np.full(n_chars, 1.0 / n_chars, dtype=np.float64) for _ in range(n_pos)]

    def ask(self, n: int) -> list[Sequence]:
        """Sample n sequences from the categorical product distribution."""
        self._init_probs()
        sequences = []
        for i in range(n):
            tokens = []
            for pos_probs in self._probs:
                # Sample from categorical distribution
                char_idx = self._rng.choices(range(len(self.alphabet)), weights=pos_probs, k=1)[0]
                tokens.append(self.alphabet[char_idx])
            seq = Sequence(id=f"cem_{i}", tokens="".join(tokens))
            sequences.append(seq)
        return sequences

    def tell(self, items: list[tuple[Sequence, float, Metrics]]) -> None:
        """Update probabilities based on elite sequences."""
        self._init_probs()

        # Track best overall
        for seq, score, _ in items:
            if score > self._best_score:
                self._best_score = score
                self._best_sequence = seq

        # Select elite (top-K by score)
        elite_count = max(1, int(len(items) * self.elite_frac))
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        elite_seqs = [seq for seq, _, _ in sorted_items[:elite_count]]

        # Update probabilities from elite
        for pos_idx, pos_probs in enumerate(self._probs):
            # Count character occurrences in elite at this position
            char_counts = np.zeros(len(self.alphabet), dtype=np.float64)
            for seq in elite_seqs:
                if pos_idx < len(seq.tokens):
                    char = seq.tokens[pos_idx]
                    char_idx = self.alphabet.index(char)
                    char_counts[char_idx] += 1.0

            # Normalize to get empirical distribution
            elite_dist = char_counts / (len(elite_seqs) + 1e-10)

            # Smooth update
            new_probs = (1 - self.beta) * pos_probs + self.beta * elite_dist

            # Clip to ensure minimum probability
            new_probs = np.clip(new_probs, self.eps, 1.0)

            # Renormalize
            new_probs /= new_probs.sum()
            self._probs[pos_idx] = new_probs

    def best(self) -> tuple[Sequence, float] | None:
        """Return the best sequence observed so far."""
        if self._best_sequence is None:
            return None
        return (self._best_sequence, self._best_score)

    def state(self) -> Mapping[str, object]:
        """Return serializable strategy state."""
        if self._probs is None:
            return {}
        return {
            "probs": [p.tolist() for p in self._probs],
            "best_score": self._best_score,
        }

    def strategy_caps(self) -> StrategyCaps:
        return self._CAPS
