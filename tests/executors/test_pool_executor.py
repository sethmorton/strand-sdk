import time

import pytest

from strand.core.sequence import Sequence
from strand.engine import LocalPoolExecutor
from strand.engine.interfaces import Evaluator
from strand.engine.types import Metrics


class _EchoEvaluator(Evaluator):
    def evaluate_batch(self, seqs: list[Sequence]) -> list[Metrics]:
        return [Metrics(objective=float(len(s.tokens)), constraints={}, aux={}) for s in seqs]


class _SlowEvaluator(Evaluator):
    def __init__(self, delay: float) -> None:
        self.delay = delay

    def evaluate_batch(self, seqs: list[Sequence]) -> list[Metrics]:
        time.sleep(self.delay)
        return [Metrics(objective=0.0, constraints={}, aux={}) for _ in seqs]


def test_pool_executor_preserves_order_thread_mode():
    evaluator = _EchoEvaluator()
    executor = LocalPoolExecutor(evaluator=evaluator, mode="thread", num_workers=2, batch_size=2)

    seqs = [Sequence(id=str(i), tokens="A" * (i + 1)) for i in range(6)]
    results = executor.run(seqs, timeout_s=2.0)

    assert [m.objective for m in results] == [float(len(s.tokens)) for s in seqs]


def test_pool_executor_timeout():
    evaluator = _SlowEvaluator(delay=0.2)
    executor = LocalPoolExecutor(evaluator=evaluator, mode="thread", num_workers=2, batch_size=2)

    seqs = [Sequence(id=str(i), tokens="A") for i in range(4)]
    with pytest.raises(TimeoutError):
        executor.run(seqs, timeout_s=0.05)

