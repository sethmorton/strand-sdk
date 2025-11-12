"""Local executor for parallel evaluation.

Chooses threads or processes to evaluate batches with an Evaluator while
preserving order.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from strand.core.sequence import Sequence
from strand.engine.interfaces import Evaluator
from strand.engine.types import Metrics


@dataclass
class LocalExecutor:
    """Parallel executor that wraps an Evaluator.

    Parameters
    ----------
    evaluator : Evaluator
        The pure evaluator to run in parallel.
    mode : str
        "auto" (default), "thread", or "process".
    num_workers : int | str
        Integer count or "auto" to infer from hardware.
    batch_size : int
        Mini-batch size per worker.
    """

    evaluator: Evaluator
    mode: str = "auto"
    num_workers: int | str = "auto"
    batch_size: int = 64

    def __post_init__(self) -> None:
        """Resolve num_workers to an integer."""
        if isinstance(self.num_workers, str) and self.num_workers.lower() == "auto":
            self._num_workers = os.cpu_count() or 1
        else:
            self._num_workers = int(self.num_workers)

    def _resolve_mode(self) -> str:
        """Resolve 'auto' mode to 'thread' or 'process'."""
        if self.mode.lower() == "auto":
            # Use threads by default; they're lighter and work well for I/O
            return "thread"
        return self.mode.lower()

    def prepare(self) -> None:
        """Optional heavy initialization (e.g., model warmup)."""

    def run(
        self,
        seqs: list[Sequence],
        *,
        timeout_s: float | None = None,
        batch_size: int | None = None,
    ) -> list[Metrics]:
        """Evaluate sequences and return metrics in the same order.

        Parameters
        ----------
        seqs : list[Sequence]
            Input sequences to evaluate.
        timeout_s : float | None
            Timeout per batch in seconds (not enforced per-sequence yet).
        batch_size : int | None
            Override instance batch_size for this call.

        Returns
        -------
        list[Metrics]
            Metrics in the same order as input sequences.
        """
        if not seqs:
            return []

        batch_size = batch_size or self.batch_size
        mode = self._resolve_mode()

        # Create batches
        batches = [seqs[i : i + batch_size] for i in range(0, len(seqs), batch_size)]
        batch_indices = list(range(0, len(seqs), batch_size))

        # Choose executor
        executor_class = ThreadPoolExecutor if mode == "thread" else ProcessPoolExecutor

        results_by_index = {}
        with executor_class(max_workers=self._num_workers) as executor:
            # Submit all batches
            futures = {
                executor.submit(self.evaluator.evaluate_batch, batch): (idx, batch)
                for idx, batch in zip(batch_indices, batches)
            }

            # Collect results as they complete
            for future in as_completed(futures):
                idx, batch = futures[future]
                try:
                    metrics = future.result(timeout=timeout_s)
                    results_by_index[idx] = metrics
                except Exception:
                    # On error, return empty metrics for this batch
                    results_by_index[idx] = [
                        Metrics(objective=0.0, constraints={}, aux={}) for _ in batch
                    ]

        # Reconstruct in original order
        all_metrics = []
        for batch_start in batch_indices:
            all_metrics.extend(results_by_index.get(batch_start, []))

        return all_metrics

    def close(self) -> None:
        """Optional cleanup hook."""
