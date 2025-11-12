"""Evaluator that wraps existing reward blocks.

This adapter allows immediate reuse of ``strand.rewards`` in the new engine by
exposing a batched ``evaluate_batch`` that computes a weighted sum objective and
optionally fills constraint/aux fields.
"""

from __future__ import annotations

from collections.abc import Sequence as ABCSequence
from dataclasses import dataclass

from strand.core.sequence import Sequence
from strand.engine.interfaces import Evaluator
from strand.engine.types import Metrics
from strand.rewards import RewardBlock


@dataclass
class RewardAggregator(Evaluator):
    """Batch evaluator over a list of reward blocks.

    Computes a weighted-sum objective by calling each reward block on each sequence
    and aggregating scores by weight.
    """

    reward_blocks: ABCSequence[RewardBlock]

    def evaluate_batch(self, seqs: list[Sequence]) -> list[Metrics]:
        """Compute a weighted-sum objective per sequence.

        For each sequence:
        1. Call each reward block to get its score
        2. Weight by the block's weight parameter
        3. Sum to form the objective

        Returns
        -------
        list[Metrics]
            Metrics for each sequence in input order, with:
            - objective: weighted sum of reward block scores
            - constraints: empty mapping (reward blocks don't return constraints)
            - aux: empty mapping (reserved for future use)
        """
        results = []
        for seq in seqs:
            weighted_sum = 0.0
            for block in self.reward_blocks:
                # Call the reward block; it should return a float
                score = block.score(seq)
                weighted_sum += score

            metrics = Metrics(
                objective=weighted_sum,
                constraints={},
                aux={},
            )
            results.append(metrics)

        return results

