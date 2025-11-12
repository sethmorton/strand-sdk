"""Helpers for running the optimization engine.

The engine runs a simple loop: strategies propose sequences, executors evaluate
them, scorers turn metrics into numbers, and we keep the best result.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

from strand.core.sequence import Sequence
from strand.engine.constraints import BoundedConstraint
from strand.engine.interfaces import Evaluator, Executor, Strategy
from strand.engine.rules import Rules
from strand.engine.runtime import (
    BatchConfig,
    DeviceConfig,
    StrategyContext,
    build_strategy_context,
    resolve_strategy_caps,
)
from strand.engine.types import Metrics

ScoreFn = Callable[[Metrics, Mapping[str, float], list[BoundedConstraint]], float]


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """User-facing settings that control an engine run.

    ``method`` is only a descriptive label for logs and manifests. Supply the
    actual :class:`~strand.engine.interfaces.Strategy` when creating the engine.
    """

    iterations: int = 100
    population_size: int = 256
    seed: int = 1337
    timeout_s: float = 60.0
    early_stop_patience: int | None = 10
    max_evals: int | None = None
    method: str = "cem"
    extra: Mapping[str, int | float | str] = field(default_factory=dict)
    batching: BatchConfig | None = None
    device: DeviceConfig | None = None


@dataclass(frozen=True, slots=True)
class IterationStats:
    """Numbers tracked for each iteration.

    ``rules`` holds the rule weights used in the iteration. ``violations`` stores
    the mean violation per constraint name (missing constraints default to ``0``).
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
    """Drives the optimization loop.

    Compose a strategy, evaluator, executor, and scoring function. The engine
    repeats ``ask → run → score → tell`` while recording iteration summaries.

    **Implementation note:** ``score_fn`` receives the rule weights and
    constraint list on every call. Strategies must emit valid sequences; invalid
    sequences may be dropped and counted in the stats.
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
        self._batch_config = config.batching or BatchConfig()
        self._device_config = config.device or DeviceConfig()
        self._strategy_caps = resolve_strategy_caps(strategy)
        self._strategy_context: StrategyContext | None = None

    def run(self) -> EngineResults:
        """Execute the optimization loop and return results."""

        history: list[IterationStats] = []
        best_overall: tuple[Sequence, float] | None = None

        self._ensure_strategy_context()

        try:
            for stats in self.stream():
                history.append(stats)
                strategy_best = self._strategy.best()
                if strategy_best is not None and (
                    best_overall is None or strategy_best[1] > best_overall[1]
                ):
                    best_overall = strategy_best
        finally:
            self._executor.close()

        if best_overall is None:
            strategy_best = self._strategy.best()
            if strategy_best is not None:
                best_overall = strategy_best

        summary: dict[str, object] = {
            "config": {
                "iterations": self._config.iterations,
                "population_size": self._config.population_size,
                "seed": self._config.seed,
                "timeout_s": self._config.timeout_s,
                "early_stop_patience": self._config.early_stop_patience,
                "max_evals": self._config.max_evals,
                "method": self._config.method,
                "batching": {
                    "eval_size": self._batch_config.eval_size,
                    "train_size": self._batch_config.train_size,
                    "max_tokens": self._batch_config.max_tokens,
                },
                "device": {
                    "target": self._device_config.target,
                    "mixed_precision": self._device_config.mixed_precision,
                    "gradient_accumulation_steps": self._device_config.gradient_accumulation_steps,
                },
            },
            "total_evals": sum(s.evals for s in history),
            "iterations_completed": len(history),
            "stopped_early": len(history) < self._config.iterations,
        }

        if best_overall is not None:
            summary["best_score"] = best_overall[1]

        return EngineResults(best=best_overall, history=history, summary=summary)

    def stream(self) -> Iterator[IterationStats]:
        """Yield iteration statistics as they are computed."""

        total_evals = 0
        best_score = float("-inf")
        patience_counter = 0

        self._executor.prepare()
        self._ensure_strategy_context()

        for iteration in range(self._config.iterations):
            iter_start = time.perf_counter()

            candidates = self._strategy.ask(self._config.population_size)
            if not candidates:
                _LOGGER.warning("Strategy.ask returned no candidates; stopping early")
                break

            rules_snapshot = dict(self._rules.values())
            errors = 0
            timeouts = 0

            eval_batch_size = self._batch_config.eval_size or self._config.population_size

            try:
                metrics_list = self._executor.run(
                    candidates,
                    timeout_s=self._config.timeout_s,
                    batch_size=eval_batch_size,
                )
            except TimeoutError:
                timeouts = len(candidates)
                metrics_list = [
                    Metrics(objective=0.0, constraints={}, aux={}) for _ in candidates
                ]
            except Exception as exc:  # pragma: no cover - defensive path
                _LOGGER.exception("Executor.run failed", exc_info=exc)
                errors = len(candidates)
                metrics_list = [
                    Metrics(objective=0.0, constraints={}, aux={}) for _ in candidates
                ]

            if len(metrics_list) != len(candidates):
                raise RuntimeError(
                    "Executor returned mismatched results: "
                    f"expected {len(candidates)} got {len(metrics_list)}"
                )

            scored_items: list[tuple[Sequence, float, Metrics]] = []
            scores: list[float] = []
            for seq, metrics in zip(candidates, metrics_list):
                try:
                    score = self._score_fn(metrics, rules_snapshot, self._constraints)
                except Exception as exc:  # pragma: no cover - defensive path
                    _LOGGER.exception("score_fn raised", exc_info=exc)
                    errors += 1
                    score = float("-inf")
                scores.append(score)
                scored_items.append((seq, score, metrics))

            self._strategy.tell(scored_items)
            self._maybe_train(scored_items)

            violation_signals: dict[str, list[float]] = {}
            for constraint in self._constraints:
                values = [
                    constraint.violation(metrics.constraints.get(constraint.name, 0.0))
                    for metrics in metrics_list
                ]
                violation_signals[constraint.name] = values

            if self._rules is not None:
                self._rules.update(violation_signals)

            total_evals += len(metrics_list)
            iter_time = time.perf_counter() - iter_start
            throughput = len(metrics_list) / iter_time if iter_time > 0 else 0.0

            if scores:
                best_iter = max(scores)
                mean_iter = sum(scores) / len(scores)
                if len(scores) > 1:
                    variance = sum((s - mean_iter) ** 2 for s in scores) / len(scores)
                    std_iter = math.sqrt(variance)
                else:
                    std_iter = 0.0
            else:
                best_iter = float("-inf")
                mean_iter = 0.0
                std_iter = 0.0

            violations_mean = {
                name: (sum(values) / len(values) if values else 0.0)
                for name, values in violation_signals.items()
            }

            stats = IterationStats(
                iteration=iteration,
                best=best_iter,
                mean=mean_iter,
                std=std_iter,
                evals=len(metrics_list),
                throughput=throughput,
                timeouts=timeouts,
                errors=errors,
                rules=rules_snapshot,
                violations=violations_mean,
            )

            yield stats

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

    def _ensure_strategy_context(self) -> StrategyContext:
        if self._strategy_context is not None:
            return self._strategy_context

        require_runtime = (
            self._strategy_caps.requires_runtime or self._strategy_caps.supports_fine_tuning
        )

        context = build_strategy_context(
            device=self._device_config,
            batch=self._batch_config,
            require_runtime=require_runtime,
        )

        prepare_fn = getattr(self._strategy, "prepare", None)
        if callable(prepare_fn):
            prepare_fn(context)

        self._strategy_context = context
        return context

    def _maybe_train(self, items: list[tuple[Sequence, float, Metrics]]) -> None:
        if not items:
            return
        if not self._strategy_caps.supports_fine_tuning:
            return

        train_step = getattr(self._strategy, "train_step", None)
        if not callable(train_step):
            return

        context = self._ensure_strategy_context()
        train_step(items, context)
