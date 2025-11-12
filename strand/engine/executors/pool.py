"""Local pool executor for parallel evaluation.

Uses threads or processes to evaluate batches via an Evaluator while preserving
input order. This is a drop-in alternative to the sequential LocalExecutor.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import ClassVar, Literal

from strand.core.sequence import Sequence
from strand.engine.interfaces import Evaluator
from strand.engine.types import Metrics


def _chunks(seq: list[Sequence], size: int) -> list[list[Sequence]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


@dataclass(slots=True)
class LocalPoolExecutor:
    """Parallel executor that preserves input order.

    Parameters
    ----------
    evaluator:
        Pure evaluator to call in worker threads/processes.
    mode:
        "auto" (default), "thread", or "process".
    num_workers:
        Number of workers. If 0 or "auto", an implementation-defined default is used.
    batch_size:
        Mini-batch size for evaluator.evaluate_batch.
    """

    evaluator: Evaluator
    mode: Literal["auto", "thread", "process"] = "auto"
    num_workers: int | Literal["auto"] = "auto"
    batch_size: int = 64

    _THREADS_DEFAULT: ClassVar[int] = 4

    def prepare(self) -> None:
        """No-op for now; hook for model warmup in workers."""

    def run(
        self,
        seqs: list[Sequence],
        *,
        timeout_s: float | None = None,
        batch_size: int | None = None,
    ) -> list[Metrics]:
        """Evaluate sequences in parallel and preserve input order."""
        if not seqs:
            return []

        effective_batch = batch_size or self.batch_size
        batches = _chunks(seqs, effective_batch)

        # Select executor type and size
        mode = self.mode
        if mode == "auto":
            # Default to threads; switch to processes later if needed
            mode = "thread"

        max_workers = None
        if self.num_workers == "auto" or self.num_workers == 0:
            max_workers = self._THREADS_DEFAULT
        else:
            max_workers = max(1, int(self.num_workers))

        deadline = None if timeout_s is None else time.perf_counter() + timeout_s

        # Submit batches preserving order by tracking futures with their index
        results: list[list[Metrics]] = [None] * len(batches)  # type: ignore[list-item]

        ExecutorCls = ThreadPoolExecutor if mode == "thread" else ProcessPoolExecutor
        pool = ExecutorCls(max_workers=max_workers)
        futures: list[Future] = []
        shutdown_wait = True
        try:
            try:
                for batch in batches:
                    futures.append(pool.submit(self.evaluator.evaluate_batch, batch))
            except BaseException:
                shutdown_wait = False
                for pending in futures:
                    pending.cancel()
                raise

            # Consume futures in submission order to preserve overall ordering
            for idx, fut in enumerate(futures):
                remaining = None
                if deadline is not None:
                    remaining = max(0.0, deadline - time.perf_counter())
                try:
                    batch_results = fut.result(timeout=remaining)
                except TimeoutError:
                    shutdown_wait = False
                    for pending in futures[idx:]:
                        pending.cancel()
                    raise
                except Exception as exc:
                    shutdown_wait = False
                    for pending in futures[idx + 1 :]:
                        pending.cancel()
                    raise RuntimeError("Worker task failed") from exc
                except BaseException:
                    shutdown_wait = False
                    for pending in futures[idx + 1 :]:
                        pending.cancel()
                    raise

                if len(batch_results) != len(batches[idx]):
                    shutdown_wait = False
                    for pending in futures[idx + 1 :]:
                        pending.cancel()
                    raise RuntimeError(
                        "Evaluator returned mismatched batch size: "
                        f"expected {len(batches[idx])} got {len(batch_results)}"
                    )
                results[idx] = batch_results
        finally:
            pool.shutdown(wait=shutdown_wait, cancel_futures=True)

        # Flatten in original order
        flat: list[Metrics] = []
        for r in results:
            # Safety: should not be None; if so, raise
            if r is None:  # type: ignore[unreachable]
                raise RuntimeError("Missing batch result")
            flat.extend(r)
        return flat

    def close(self) -> None:
        """Hook for cleanup work (unused)."""

