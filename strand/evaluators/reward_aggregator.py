"""Evaluator that sums the scores from reward blocks.

It reuses ``strand.rewards`` blocks and exposes a batched ``evaluate_batch``
method for the engine.
"""

from __future__ import annotations

from collections.abc import Sequence as ABCSequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from strand.core.sequence import Sequence
from strand.engine.interfaces import Evaluator
from strand.engine.types import Metrics
from strand.rewards.base import RewardBlockProtocol, RewardContext

if TYPE_CHECKING:
    from strand.engine.types import SequenceContext


@dataclass
class RewardAggregator(Evaluator):
    """Batch evaluator over a list of reward blocks."""

    reward_blocks: ABCSequence[RewardBlockProtocol]

    def evaluate_batch(self, seqs: list[Sequence]) -> list[Metrics]:
        """Return metrics with the weighted sum objective for each sequence."""
        results: list[Metrics] = []
        for sequence in seqs:
            context = RewardContext()
            weighted_sum = 0.0
            for block in self.reward_blocks:
                weighted_sum += block.score(sequence, context=context)

            results.append(
                Metrics(
                    objective=weighted_sum,
                    constraints={},
                    aux={},
                )
            )

        return results

    def evaluate_batch_with_context(self, contexts: list["SequenceContext"]) -> list[Metrics]:
        """Return metrics for context-aware evaluation.

        Args:
            contexts: List of SequenceContext objects with ref/alt sequences

        Returns:
            List of Metrics with objective, constraints, and auxiliary data
        """
        results: list[Metrics] = []

        for context in contexts:
            reward_context = RewardContext()
            total_objective = 0.0
            all_aux: dict[str, float] = {}

            for block in self.reward_blocks:
                if block.metadata.requires_context:
                    objective, aux = block.score_context(context, reward_context=reward_context)
                    weighted_objective = block.weight * objective
                else:
                    weighted_objective = block.score(context.alt_seq, context=reward_context)
                    aux = {}

                total_objective += weighted_objective

                if aux:
                    for key, value in aux.items():
                        scoped_key = f"{block.name}.{key}" if key else block.name
                        all_aux[scoped_key] = value

            results.append(
                Metrics(
                    objective=total_objective,
                    constraints={},
                    aux=all_aux,
                )
            )

        return results
