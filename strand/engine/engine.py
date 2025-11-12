"""Engine orchestration surfaces.

The engine drives an iterative ask → run → score → tell → update loop. It is
configured with small, composable parts (Strategy, Evaluator, Executor) and a
simple scoring rule that turns Metrics into a single number per candidate.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

from strand.core.sequence import Sequence
from strand.engine.constraints import BoundedConstraint
from strand.engine.interfaces import Evaluator, Executor, Strategy
from strand.engine.rules import Rules
from strand.engine.types import Metrics

ScoreFn = Callable[[Metrics, Mapping[str, float], list[BoundedConstraint]], float]


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Engine configuration (stable surface).

    The `method` field is a label (for manifests and logs), not used to pick a strategy.
    Pass your chosen Strategy instance directly to Engine.
    """

    iterations: int = 100
    population_size: int = 256
    seed: int = 1337
    timeout_s: float = 60.0
    early_stop_patience: int | None = 10
    max_evals: int | None = None
    method: str = "cem"
    extra: Mapping[str, int | float | str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IterationStats:
    """Per-iteration summary statistics.

    This structure is designed for logging, dashboards, and manifest summaries.

    - `rules`: A snapshot of rule weights applied in this iteration.
    - `violations`: Mean constraint violation per named constraint for the iteration
      (missing constraints default to 0.0).
    """

    iteration: int
    best: float
    mean: float
    std: float
    evals: int
    throughput: float
    timeouts: int
    errors: int
    rules: Mapping[str, float]
    violations: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class EngineResults:
    """Final results emitted by the engine.

    ``best`` contains the best observed (sequence, score) pair if any. ``history``
    tracks iteration summaries. A manifest can be attached by higher layers.
    """

    best: tuple[Sequence, float] | None
    history: list[IterationStats]
    summary: Mapping[str, object]


class Engine:
    """Iterative optimization driver.

    Compose a Strategy, Evaluator, Executor, and a scoring rule (``score_fn``)
    that maps Metrics + rules + constraints → scalar.

    The engine runs an ask → run → score → tell → update loop, recording
    iteration statistics for reproducibility and analysis.

    **Implementation note:** Engine passes `rules.values()` into `score_fn` and validates
    inputs before execution. Strategies must emit valid sequences; invalid sequences may be
    dropped and counted in iteration statistics.
    """

    def __init__(
        self,
        *,
        config: EngineConfig,
        strategy: Strategy,
        evaluator: Evaluator,
        executor: Executor,
        score_fn: ScoreFn,
        constraints: list[BoundedConstraint] | None = None,
        rules: Rules | None = None,
    ) -> None:
        self._config = config
        self._strategy = strategy
        self._evaluator = evaluator
        self._executor = executor
        self._score_fn = score_fn
        self._constraints = list(constraints or [])
        self._rules = rules or Rules()

    def run(self) -> EngineResults:
        """Execute the optimization loop and return results.

        Orchestrates ask → run → score → tell → update, accumulating iteration stats
        and tracking the best sequence and score observed.

        Returns
        -------
        EngineResults
            Final results with best (Sequence, float) pair, history of stats, and summary.
        """
        history: list[IterationStats] = []
        best_overall: tuple[Sequence, float] | None = None

        for stats in self.stream():
            history.append(stats)
            # Update best overall
            if stats.best > (best_overall[1] if best_overall else float("-inf")):
                # We need the best sequence, but stats only has the best score
                # For now, we'll set it when we get it from strategy.best()
                pass

        # Get the best sequence from the strategy
        strategy_best = self._strategy.best()
        if strategy_best is not None:
            best_overall = strategy_best

        summary = {
            "config": {
                "iterations": self._config.iterations,
                "population_size": self._config.population_size,
                "seed": self._config.seed,
                "timeout_s": self._config.timeout_s,
                "early_stop_patience": self._config.early_stop_patience,
                "max_evals": self._config.max_evals,
                "method": self._config.method,
            },
            "total_evals": sum(s.evals for s in history),
            "iterations_completed": len(history),
        }

        return EngineResults(best=best_overall, history=history, summary=summary)

    def stream(self) -> Iterator[IterationStats]:
        """Yield iteration statistics as they are computed.

        Implements the core ask → run → score → tell loop with early stopping
        and max evaluation limits.
        """
        import time

        total_evals = 0
        best_score = float("-inf")
        patience_counter = 0

        for iteration in range(self._config.iterations):
            iter_start = time.time()

            # Ask: get candidates from strategy
            candidates = self._strategy.ask(self._config.population_size)

            # Run: evaluate in parallel, preserving order
            metrics_list = self._executor.run(
                candidates,
                timeout_s=self._config.timeout_s,
            )

            # Score: turn Metrics into scalars
            scores = [
                self._score_fn(m, self._rules.values(), self._constraints)
                for m in metrics_list
            ]

            # Tell: ingest feedback
            items = list(zip(candidates, scores, metrics_list))
            self._strategy.tell(items)

            # Update rules (optional)
            if self._rules is not None:
                violations = {
                    c.name: [m.constraints.get(c.name, 0.0) for m in metrics_list]
                    for c in self._constraints
                }
                self._rules.update(violations)

            # Compute iteration stats
            total_evals += len(candidates)
            iter_time = time.time() - iter_start
            throughput = len(candidates) / iter_time if iter_time > 0 else 0

            best_iter = max(scores) if scores else float("-inf")
            mean_iter = sum(scores) / len(scores) if scores else 0.0
            std_iter = (
                (sum((s - mean_iter) ** 2 for s in scores) / len(scores)) ** 0.5
                if len(scores) > 1
                else 0.0
            )

            # Count timeouts and errors (basic for now)
            timeouts = sum(1 for m in metrics_list if m.objective == 0.0)
            errors = 0

            stats = IterationStats(
                iteration=iteration,
                best=best_iter,
                mean=mean_iter,
                std=std_iter,
                evals=len(candidates),
                throughput=throughput,
                timeouts=timeouts,
                errors=errors,
                rules=dict(self._rules.values()),
                violations={
                    c.name: [m.constraints.get(c.name, 0.0) for m in metrics_list]
                    for c in self._constraints
                },
            )

            yield stats

            # Early stopping logic
            if best_iter > best_score:
                best_score = best_iter
                patience_counter = 0
            else:
                patience_counter += 1

            if (
                self._config.early_stop_patience is not None
                and patience_counter >= self._config.early_stop_patience
            ):
                break

            if (
                self._config.max_evals is not None
                and total_evals >= self._config.max_evals
            ):
                break
