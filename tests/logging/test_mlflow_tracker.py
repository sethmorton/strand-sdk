"""Tests for MLflow experiment tracking."""

import tempfile
from pathlib import Path

import mlflow

from strand.core.sequence import Sequence
from strand.engine.engine import EngineConfig, EngineResults, IterationStats
from strand.logging import MLflowTracker


class TestMLflowTracker:
    """MLflow tracker tests."""

    def test_initialization(self):
        """Test MLflowTracker initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLflowTracker(
                experiment_name="test_exp",
                tracking_uri=str(Path(tmpdir) / "mlruns"),
            )
            assert tracker.experiment_name == "test_exp"

    def test_log_config(self):
        """Test logging engine configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLflowTracker(
                experiment_name="test_exp",
                tracking_uri=str(Path(tmpdir) / "mlruns"),
            )
            config = EngineConfig(
                iterations=10,
                population_size=32,
                seed=42,
                method="random",
            )

            tracker.start_run("test_run")
            tracker.log_config(config)
            tracker.end_run()

            # Verify the run was created
            experiment = mlflow.get_experiment_by_name("test_exp")
            assert experiment is not None

    def test_log_iteration_stats(self):
        """Test logging iteration statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLflowTracker(
                experiment_name="test_exp",
                tracking_uri=str(Path(tmpdir) / "mlruns"),
            )
            tracker.start_run("test_run")

            stats = IterationStats(
                iteration=0,
                best=0.95,
                mean=0.75,
                std=0.1,
                evals=32,
                throughput=100.0,
                timeouts=0,
                errors=0,
                rules={},
                violations={},
            )

            tracker.log_iteration_stats(0, stats)
            tracker.end_run()

    def test_log_results(self):
        """Test logging final results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLflowTracker(
                experiment_name="test_exp",
                tracking_uri=str(Path(tmpdir) / "mlruns"),
            )
            tracker.start_run("test_run")

            best_seq = Sequence(id="best", tokens="ACDE")
            results = EngineResults(
                best=(best_seq, 0.95),
                history=[],
                summary={"total_evals": 1000},
            )

            tracker.log_results(results)
            tracker.end_run()

    def test_artifact_logging(self):
        """Test logging artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MLflowTracker(
                experiment_name="test_exp",
                tracking_uri=str(Path(tmpdir) / "mlruns"),
            )
            tracker.start_run("test_run")

            data = {"key": "value", "score": 0.95}
            tracker.log_artifact_json(data, "results.json")

            tracker.end_run()

