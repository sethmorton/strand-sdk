"""Evaluator that sums the scores from reward blocks.

It reuses ``strand.rewards`` blocks and exposes a batched ``evaluate_batch``
method for the engine.
"""

from __future__ import annotations

from collections.abc import Sequence as ABCSequence
from dataclasses import dataclass

from strand.core.sequence import Sequence
from strand.engine.interfaces import Evaluator
from strand.engine.types import Metrics
from strand.rewards import RewardBlock
from strand.rewards.base import RewardContext


@dataclass
class RewardAggregator(Evaluator):
    """Batch evaluator over a list of reward blocks."""

    reward_blocks: ABCSequence[RewardBlock]

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

