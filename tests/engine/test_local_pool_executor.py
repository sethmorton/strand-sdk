"""Tests for the LocalPoolExecutor."""

from __future__ import annotations

import time
from concurrent.futures import TimeoutError as FuturesTimeoutError

import pytest

from strand.core.sequence import Sequence
from strand.engine.executors.pool import LocalPoolExecutor
from strand.engine.types import Metrics


class _EchoEvaluator:
    """Simple evaluator that returns deterministic metrics."""

    def evaluate_batch(self, seqs: list[Sequence]) -> list[Metrics]:
        return [
            Metrics(
                objective=float(len(seq.tokens)),
                constraints={},
                aux={"token": seq.tokens},
            )
            for seq in seqs
        ]


class _SlowEvaluator:
    """Evaluator that sleeps before returning results."""

    def __init__(self, delay_s: float) -> None:
        self._delay_s = delay_s

    def evaluate_batch(self, seqs: list[Sequence]) -> list[Metrics]:
        time.sleep(self._delay_s)
        return [
            Metrics(
                objective=0.0,
                constraints={},
                aux={"tokens": seq.tokens},
            )
            for seq in seqs
        ]


def _make_sequences(n: int) -> list[Sequence]:
    return [Sequence(id=str(i), tokens=f"seq-{i}") for i in range(n)]


def test_process_pool_evaluation_succeeds() -> None:
    executor = LocalPoolExecutor(
        evaluator=_EchoEvaluator(),
        mode="process",
        num_workers=2,
        batch_size=2,
    )

    seqs = _make_sequences(4)
    metrics = executor.run(seqs)

    assert len(metrics) == len(seqs)
    # Ensure order is preserved and per-item metrics match expectations
    for seq, metric in zip(seqs, metrics, strict=True):
        assert metric.aux["token"] == seq.tokens
        assert metric.objective == float(len(seq.tokens))


def test_timeout_does_not_wait_for_all_batches() -> None:
    executor = LocalPoolExecutor(
        evaluator=_SlowEvaluator(delay_s=0.5),
        mode="thread",
        num_workers=2,
        batch_size=1,
    )

    seqs = _make_sequences(2)
    start = time.perf_counter()
    with pytest.raises(FuturesTimeoutError):
        executor.run(seqs, timeout_s=0.1)
    elapsed = time.perf_counter() - start

    # Should return well before the sleep completes for all tasks.
    assert elapsed < 0.5
