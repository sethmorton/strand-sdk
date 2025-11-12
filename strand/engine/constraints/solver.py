"""Constraint satisfaction solver for hard constraints.

Uses python-constraint library for discrete constraint satisfaction problems.
Useful for validating and filtering sequences against hard constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

from strand.core.sequence import Sequence


@dataclass
class ConstraintSolver:
    """Solver for hard constraints using Constraint Satisfaction Problem (CSP).

    Parameters
    ----------
    alphabet : str
        Valid characters for sequences.
    min_len : int
        Minimum sequence length.
    max_len : int
        Maximum sequence length.
    """

    alphabet: str
    min_len: int
    max_len: int

    def is_feasible(self, seq: Sequence) -> bool:
        """Check if a sequence satisfies basic validity constraints.

        Parameters
        ----------
        seq : Sequence
            Sequence to validate.

        Returns
        -------
        bool
            True if sequence is valid, False otherwise.
        """
        # Check length
        if not (self.min_len <= len(seq.tokens) <= self.max_len):
            return False

        # Check alphabet
        return all(char in self.alphabet for char in seq.tokens)

    def filter_feasible(self, sequences: list[Sequence]) -> list[Sequence]:
        """Filter sequences to keep only feasible ones.

        Parameters
        ----------
        sequences : list[Sequence]
            Sequences to filter.

        Returns
        -------
        list[Sequence]
            Only feasible sequences.
        """
        return [seq for seq in sequences if self.is_feasible(seq)]

    def add_custom_constraint(
        self, constraint_fn: callable, variables: list[int] | None = None
    ) -> None:
        """Add a custom constraint function (future extension).

        Parameters
        ----------
        constraint_fn : callable
            Function that returns True if constraint is satisfied.
        variables : list[int] | None
            Variable indices to apply constraint to.
        """
        # Placeholder for custom constraint support
        # Implementation would require problem reformulation

    @staticmethod
    def generate_feasible_set(
        alphabet: str, length: int, max_size: int = 1000
    ) -> list[Sequence]:
        """Generate a set of feasible sequences.

        Parameters
        ----------
        alphabet : str
            Valid characters.
        length : int
            Fixed sequence length.
        max_size : int
            Maximum number of sequences to generate.

        Returns
        -------
        list[Sequence]
            Random valid sequences.
        """
        import random

        sequences = []
        for i in range(max_size):
            tokens = "".join(random.choice(alphabet) for _ in range(length))
            seq = Sequence(id=f"feasible_{i}", tokens=tokens)
            sequences.append(seq)

        return sequences

