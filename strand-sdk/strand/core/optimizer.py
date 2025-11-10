"""Main optimizer facade."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Iterable
from uuid import uuid4

from strand.core.results import OptimizationResults
from strand.core.sequence import Sequence
from strand.manifests import Manifest
from strand.optimizers import (
    BaseOptimizer,
    CMAESOptimizer,
    CEMOptimizer,
    GeneticAlgorithmOptimizer,
    RandomSearchOptimizer,
)
from strand.rewards.base import RewardBlockProtocol, RewardContext
from strand.utils import OptimizerConfig, ensure_sequences, get_logger

_LOGGER = get_logger("optimizer")


class Optimizer:
    def __init__(
        self,
        sequences: Iterable[Sequence | str],
        reward_blocks: Iterable[RewardBlockProtocol],
        method: str = "cem",
        *,
        iterations: int = 50,
        population_size: int = 200,
        seed: int | None = None,
        experiment: str = "default",
        **extra_config: int | float | str,
    ) -> None:
        self._raw_sequences = list(sequences)
        self._reward_blocks = list(reward_blocks)
        if not self._reward_blocks:
            msg = "At least one reward block is required"
            raise ValueError(msg)
        self._config = OptimizerConfig(
            method=method,
            iterations=iterations,
            population_size=population_size,
            seed=seed,
            extra=extra_config,
        )
        self._experiment = experiment

    def run(self) -> OptimizationResults:
        sequences = ensure_sequences(self._raw_sequences)
        score_fn = self._build_score_fn()
        strategy = self._build_strategy(sequences, score_fn)
        _LOGGER.info("Running optimizer method=%s", self._config.method)
        ranked = strategy.optimize()
        ranked_sequences = [seq for seq, _ in ranked]
        scores = [score for _, score in ranked]
        results = OptimizationResults(ranked_sequences=ranked_sequences, scores=scores)
        manifest = self._build_manifest(sequences, results)
        results.attach_manifest(manifest)
        return results

    def _build_score_fn(self) -> Callable[[Sequence], float]:
        def score(sequence: Sequence) -> float:
            total = 0.0
            context = RewardContext()
            for block in self._reward_blocks:
                total += block.score(sequence, context=context)
            return total

        return score

    def _build_strategy(self, sequences: list[Sequence], score_fn) -> BaseOptimizer:
        strategy_map = {
            "cem": CEMOptimizer,
            "cmaes": CMAESOptimizer,
            "ga": GeneticAlgorithmOptimizer,
            "random": RandomSearchOptimizer,
        }
        if self._config.method not in strategy_map:
            msg = f"Unsupported optimization method: {self._config.method}"
            raise ValueError(msg)
        strategy_cls = strategy_map[self._config.method]
        return strategy_cls(
            sequences=sequences,
            score_fn=score_fn,
            iterations=self._config.iterations,
            population_size=self._config.population_size,
            seed=self._config.seed,
        )

    def _build_manifest(self, sequences: list[Sequence], results: OptimizationResults) -> Manifest:
        return Manifest(
            run_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            experiment=self._experiment,
            inputs={"sequences": [seq.to_dict() for seq in sequences]},
            optimizer=self._config.as_dict(),
            reward_blocks=[{"name": block.name, "weight": block.weight} for block in self._reward_blocks],
            results={
                "top_sequence_ids": [seq.id for seq, _ in results.top()],
                "scores": results.scores,
            },
        )
