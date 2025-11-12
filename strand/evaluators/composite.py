"""Composite evaluator that combines rewards and constraint metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence as TypingSequence

from strand.core.sequence import Sequence
from strand.engine.interfaces import Evaluator
from strand.engine.types import Metrics
from strand.evaluators.reward_aggregator import RewardAggregator
from strand.utils import distances


@dataclass
class CompositeEvaluator(Evaluator):
    """Combine reward objective with optional constraint metrics.

    - Objective is computed by the provided RewardAggregator.
    - Constraints may include:
      - length: sequence length
      - gc: GC fraction in [0,1]
      - novelty: normalized distance to a baseline set (levenshtein by default)
    """

    rewards: RewardAggregator
    include_length: bool = False
    include_gc: bool = False
    novelty_baseline: TypingSequence[str] | None = None
    novelty_metric: str = "levenshtein"

    def evaluate_batch(self, seqs: list[Sequence]) -> list[Metrics]:
        base_metrics = self.rewards.evaluate_batch(seqs)
        results: list[Metrics] = []

        # Precompute novelty if requested; compute per sequence vs baseline
        baseline = list(self.novelty_baseline or [])
        for seq, bm in zip(seqs, base_metrics):
            constraints: dict[str, float] = {}
            if self.include_length:
                constraints["length"] = float(len(seq))
            if self.include_gc:
                tokens = seq.tokens.upper()
                gc = (tokens.count("G") + tokens.count("C")) / len(seq) if len(seq) > 0 else 0.0
                constraints["gc"] = gc
            if baseline:
                sequences = baseline + [seq.tokens]
                try:
                    novelty_val = distances.normalized_score(self.novelty_metric, sequences)
                except Exception:
                    novelty_val = 0.0
                constraints["novelty"] = novelty_val

            results.append(
                Metrics(
                    objective=bm.objective,
                    constraints=constraints,
                    aux=bm.aux,
                )
            )

        return results

