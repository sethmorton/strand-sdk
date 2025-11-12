"""Tests for ExecutorFactory."""

import pytest
from strand.engine.executors.factory import ExecutorFactory, ExecutorConfig
from strand.core.sequence import Sequence
from strand.engine.types import Metrics
from collections.abc import Mapping


class MockEvaluator:
    """Mock evaluator for testing."""

    def evaluate_batch(self, seqs: list[Sequence]) -> list[Metrics]:
        """Return mock metrics."""
        return [
            Metrics(objective=0.5, constraints={}, aux={})
            for _ in seqs
        ]


class TestExecutorFactory:
    """Test ExecutorFactory."""

    @pytest.fixture
    def evaluator(self) -> MockEvaluator:
        """Create mock evaluator."""
        return MockEvaluator()

    def test_build_local_executor(self, evaluator: MockEvaluator) -> None:
        """Build LocalExecutor from config."""
        config = {"executor_type": "local", "num_workers": 2}
        executor = ExecutorFactory.build(config, evaluator)
        assert executor is not None

    def test_build_pool_executor(self, evaluator: MockEvaluator) -> None:
        """Build PoolExecutor from config."""
        config = {"executor_type": "pool", "num_workers": 4}
        executor = ExecutorFactory.build(config, evaluator)
        assert executor is not None

    def test_build_torch_executor(self, evaluator: MockEvaluator) -> None:
        """Build TorchExecutor from config."""
        config = {
            "executor_type": "torch",
            "device": "cpu",
            "batch_size": 32,
        }
        executor = ExecutorFactory.build(config, evaluator)
        assert executor is not None

    def test_from_executor_config(self, evaluator: MockEvaluator) -> None:
        """Build from ExecutorConfig dataclass."""
        config = ExecutorConfig(
            executor_type="local",
            num_workers=2,
        )
        executor = ExecutorFactory.build(config, evaluator)
        assert executor is not None

    def test_unknown_executor_type(self, evaluator: MockEvaluator) -> None:
        """Unknown executor type raises error."""
        config = {"executor_type": "unknown"}
        with pytest.raises(ValueError):
            ExecutorFactory.build(config, evaluator)

    def test_invalid_config_type(self, evaluator: MockEvaluator) -> None:
        """Invalid config type raises error."""
        with pytest.raises(ValueError):
            ExecutorFactory.build("invalid", evaluator)  # type: ignore[arg-type]

