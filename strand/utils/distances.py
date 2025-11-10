"""Sequence distance utilities."""

from __future__ import annotations

from typing import Sequence as SeqType


def hamming_distance(a: str, b: str) -> int:
    """Return the Hamming distance between two equal-length strings."""
    if len(a) != len(b):
        msg = "Hamming distance requires sequences of equal length"
        raise ValueError(msg)
    return sum(char_a != char_b for char_a, char_b in zip(a, b))


def levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein distance using dynamic programming."""
    if not a:
        return len(b)
    if not b:
        return len(a)

    rows = len(a) + 1
    cols = len(b) + 1
    matrix = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        matrix[i][0] = i
    for j in range(cols):
        matrix[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            deletion = matrix[i - 1][j] + 1
            insertion = matrix[i][j - 1] + 1
            substitution = matrix[i - 1][j - 1] + cost
            matrix[i][j] = min(deletion, insertion, substitution)

    return matrix[-1][-1]


def normalized_score(distance_fn: str, sequences: SeqType[str]) -> float:
    """Return a normalized distance score (0-1) for a set of sequences."""
    if len(sequences) < 2:
        return 0.0

    if distance_fn not in {"hamming", "levenshtein"}:
        msg = f"Unsupported distance function: {distance_fn}"
        raise ValueError(msg)

    total_distance = 0
    comparisons = 0
    *_, last = sequences
    for seq in sequences[:-1]:
        comparisons += 1
        if distance_fn == "hamming":
            total_distance += hamming_distance(seq, last)
        else:
            total_distance += levenshtein_distance(seq, last)

    return total_distance / max(comparisons * len(last), 1)
