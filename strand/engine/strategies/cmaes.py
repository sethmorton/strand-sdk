"""CMA-ES Strategy for continuous and discrete optimization.

Uses the PyCMA library (Covariance Matrix Adaptation Evolution Strategy).
For sequences, we discretize the continuous output by rounding to nearest alphabet token.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import cma

from strand.core.sequence import Sequence
from strand.engine.interfaces import Strategy
from strand.engine.types import Metrics


@dataclass
class CMAESStrategy(Strategy):
    """CMA-ES optimization strategy (continuous, then discretized for sequences).

    Uses the reference CMA-ES implementation from PyCMA. For biological sequences,
    we discretize the continuous output by mapping to the nearest alphabet token.

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
    sigma0 : float
        Initial step-size (standard deviation).
    cma_params : dict[str, object]
        Additional parameters for cma.CMAEvolutionStrategy.
    """

    alphabet: str
    min_len: int
    max_len: int
    seed: int | None = None
    sigma0: float = 0.3
    cma_params: dict[str, object] = field(default_factory=dict)

    # Internal state
    _fixed_len: int = field(init=False, repr=False)
    _es: cma.CMAEvolutionStrategy | None = field(default=None, init=False, repr=False)
    _last_solutions: list[list[float]] | None = field(default=None, init=False, repr=False)
    _best_sequence: Sequence | None = field(default=None, init=False, repr=False)
    _best_score: float = field(default=float("-inf"), init=False, repr=False)
    _counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize strategy state."""
        if not self.alphabet:
            raise ValueError("alphabet must be non-empty")
        if self.min_len <= 0 or self.max_len < self.min_len:
            raise ValueError("invalid length band")
        self._fixed_len = (self.min_len + self.max_len) // 2

    def _discretize(self, continuous: list[float]) -> Sequence:
        """Convert continuous values to discrete sequence.

        Map each continuous value to nearest alphabet token by:
        1. Scaling continuous values to [0, len(alphabet)-1]
        2. Rounding to nearest integer
        3. Looking up token in alphabet
        """
        tokens = []
        for val in continuous:
            # Scale to alphabet range
            scaled = (val + 1) / 2 * (len(self.alphabet) - 1)  # Assume val in [-1, 1]
            # Clamp and round
            idx = max(0, min(len(self.alphabet) - 1, round(scaled)))
            tokens.append(self.alphabet[int(idx)])

        seq_str = "".join(tokens[: self._fixed_len])
        return Sequence(id=f"cmaes_{self._counter}", tokens=seq_str)

    def ask(self, n: int) -> list[Sequence]:
        """Return n candidate sequences sampled from the CMA-ES distribution."""
        # Initialize ES on first call
        if self._es is None:
            # Start from random point
            x0 = [0.0] * self._fixed_len
            params = {
                "seed": self.seed,
                "maxiter": 1000,
                "verbose": 0,
                **self.cma_params,
            }
            self._es = cma.CMAEvolutionStrategy(x0, self.sigma0, params)

        # Ask ES for solutions
        X = self._es.ask(number=n)

        # Discretize to sequences and store for tell()
        self._last_solutions = X
        sequences = []
        for x in X:
            seq = self._discretize(x)
            self._counter += 1
            sequences.append(seq)

        return sequences

    def tell(self, items: list[tuple[Sequence, float, Metrics]]) -> None:
        """Ingest evaluated candidates and update the ES distribution.

        Parameters
        ----------
        items : list[tuple[Sequence, float, Metrics]]
            List of (sequence, score, metrics) tuples. CMA-ES maximizes scores.
        """
        if not items or self._es is None or self._last_solutions is None:
            return

        # Track best
        for seq, score, _ in items:
            if score > self._best_score:
                self._best_score = score
                self._best_sequence = seq

        # Extract scores for ES (negate for minimization if needed)
        scores = [score for _, score, _ in items]

        # Tell ES - CMA-ES minimizes by default, so negate scores
        self._es.tell(self._last_solutions, [-s for s in scores])

    def best(self) -> tuple[Sequence, float] | None:
        """Return the best sequence observed so far."""
        if self._best_sequence is None:
            return None
        return (self._best_sequence, self._best_score)

    def state(self) -> Mapping[str, object]:
        """Return serializable strategy state."""
        if self._es is None:
            return {}
        # CMA-ES state is complex; return basic info
        return {
            "best_score": self._best_score,
            "iterations": self._es.countiter if hasattr(self._es, "countiter") else 0,
            "evaluations": self._es.countevals if hasattr(self._es, "countevals") else 0,
        }
