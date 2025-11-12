"""MLflow integration for experiment tracking and reproducibility.

Provides utilities to track optimization runs with MLflow, enabling full
reproducibility and experiment comparison.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow

from strand.engine.engine import EngineConfig, EngineResults


class MLflowTracker:
    """Track optimization runs with MLflow.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment.
    tracking_uri : str | None
        MLflow tracking server URI (default: local filesystem).
    artifact_dir : Path | None
        Directory to save artifacts (default: mlruns).
    """

    def __init__(
        self,
        experiment_name: str = "strand-optimization",
        tracking_uri: str | None = None,
        artifact_dir: Path | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "./mlruns"
        self.artifact_dir = artifact_dir or Path("./mlruns")

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id

    def start_run(self, run_name: str | None = None) -> None:
        """Start a new MLflow run.

        Parameters
        ----------
        run_name : str | None
            Optional name for the run.
        """
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=run_name)

    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()

    def log_config(self, config: EngineConfig) -> None:
        """Log engine configuration as parameters.

        Parameters
        ----------
        config : EngineConfig
            Engine configuration to log.
        """
        params = asdict(config)
        # Convert extra dict to individual params
        extra = params.pop("extra", {})
        mlflow.log_params(params)
        if extra:
            mlflow.log_params({f"extra_{k}": v for k, v in extra.items()})

    def log_iteration_stats(self, iteration: int, stats: Any) -> None:
        """Log per-iteration statistics as metrics.

        Parameters
        ----------
        iteration : int
            Iteration number.
        stats : IterationStats
            Statistics for this iteration.
        """
        metrics = {
            "best": stats.best,
            "mean": stats.mean,
            "std": stats.std,
            "evals": stats.evals,
            "throughput": stats.throughput,
            "timeouts": stats.timeouts,
            "errors": stats.errors,
        }
        mlflow.log_metrics(metrics, step=iteration)

    def log_results(self, results: EngineResults) -> None:
        """Log final optimization results.

        Parameters
        ----------
        results : EngineResults
            Final engine results.
        """
        if results.best:
            best_seq, best_score = results.best
            mlflow.log_metric("final_best_score", best_score)
            mlflow.log_param("best_sequence", best_seq.tokens)

        # Log summary
        summary = results.summary or {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"summary_{key}", value)
            else:
                mlflow.log_param(f"summary_{key}", str(value))

    def log_artifact_json(
        self, data: dict[str, Any], filename: str = "results.json"
    ) -> None:
        """Log a dictionary as a JSON artifact.

        Parameters
        ----------
        data : dict
            Dictionary to save.
        filename : str
            Output filename.
        """
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / filename
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            mlflow.log_artifact(str(path))

    def log_artifact_file(self, filepath: Path | str) -> None:
        """Log a file as an artifact.

        Parameters
        ----------
        filepath : Path | str
            Path to file to log.
        """
        mlflow.log_artifact(str(filepath))

    def log_sft_metrics(self, epoch: int, *, loss: float, accuracy: float, kl: float | None = None) -> None:
        """Log supervised fine-tuning metrics."""

        metrics = {
            "sft_loss": loss,
            "sft_accuracy": accuracy,
        }
        if kl is not None:
            metrics["sft_kl"] = kl
        mlflow.log_metrics(metrics, step=epoch)

    def log_sft_checkpoint(self, path: Path | str) -> None:
        """Log a fine-tuning checkpoint artifact."""

        mlflow.log_artifact(str(path))

    @staticmethod
    def get_best_run(experiment_name: str) -> dict[str, Any] | None:
        """Get the best run from an experiment.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment.

        Returns
        -------
        dict | None
            Best run data or None if no runs exist.
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.final_best_score DESC"],
        )

        if runs.empty:
            return None

        return runs.iloc[0].to_dict()

    @staticmethod
    def compare_runs(experiment_name: str, top_n: int = 5) -> list[dict[str, Any]]:
        """Compare top N runs from an experiment.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment.
        top_n : int
            Number of top runs to return.

        Returns
        -------
        list[dict]
            List of top run data.
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return []

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.final_best_score DESC"],
        )

        return [runs.iloc[i].to_dict() for i in range(min(top_n, len(runs)))]
