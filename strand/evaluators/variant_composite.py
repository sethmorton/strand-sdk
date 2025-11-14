"""Variant-aware composite evaluator for genomic optimization."""

from __future__ import annotations

from collections.abc import Sequence as TypingSequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from strand.core.sequence import Sequence
from strand.engine.interfaces import Evaluator
from strand.engine.types import Metrics
from strand.evaluators.reward_aggregator import RewardAggregator
from strand.utils import distances

if TYPE_CHECKING:
    from strand.engine.types import SequenceContext


@dataclass
class VariantCompositeEvaluator(Evaluator):
    """Variant-aware composite evaluator combining rewards and constraint metrics.

    Extends CompositeEvaluator with variant-specific constraints and context-aware
    evaluation. Supports motif disruption detection and conservation window
    analysis in addition to standard sequence constraints.

    - Objective is computed by the provided RewardAggregator (context-aware)
    - Constraints may include:
      - length: sequence length
      - gc: GC fraction in [0,1]
      - novelty: normalized distance to a baseline set
      - motif_disruption: binary flag for motif loss in variants
      - conservation_windows: conservation scores for ref/alt regions
    """

    rewards: RewardAggregator
    include_length: bool = False
    include_gc: bool = False
    novelty_baseline: TypingSequence[str] | None = None
    novelty_metric: str = "levenshtein"
    include_motif_disruption: bool = False
    include_conservation_windows: bool = False

    def evaluate_batch(self, seqs: list[Sequence]) -> list[Metrics]:
        """Evaluate regular sequences (backward compatibility)."""
        base_metrics = self.rewards.evaluate_batch(seqs)
        return self._add_constraints(seqs, base_metrics, is_context=False)

    def evaluate_batch_with_context(self, contexts: list["SequenceContext"]) -> list[Metrics]:
        """Evaluate variant contexts with full constraint analysis."""
        base_metrics = self.rewards.evaluate_batch_with_context(contexts)
        return self._add_constraints(contexts, base_metrics, is_context=True)

    def _add_constraints(
        self,
        items: list[Sequence] | list["SequenceContext"],
        base_metrics: list[Metrics],
        is_context: bool,
    ) -> list[Metrics]:
        """Add constraint metrics to base evaluation results."""
        results: list[Metrics] = []

        # Precompute novelty if requested
        baseline = list(self.novelty_baseline or [])

        for item, bm in zip(items, base_metrics):
            constraints: dict[str, float] = dict(bm.constraints)  # Copy existing

            if is_context:
                # Variant context-specific constraints
                context = item  # type: ignore[assignment]

                # Sequence-level constraints on alt sequence
                alt_seq = context.alt_seq
                self._add_sequence_constraints(alt_seq, constraints)

                # Variant-specific constraints
                if self.include_motif_disruption:
                    self._add_motif_constraints(bm.aux, constraints)

                if self.include_conservation_windows:
                    self._add_conservation_constraints(bm.aux, constraints)

            else:
                # Regular sequence constraints
                seq = item  # type: ignore[assignment]
                self._add_sequence_constraints(seq, constraints)

                # Novelty computation for regular sequences
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

    def _add_sequence_constraints(self, seq: Sequence, constraints: dict[str, float]) -> None:
        """Add basic sequence-level constraints."""
        if self.include_length:
            constraints["length"] = float(len(seq))

        if self.include_gc:
            tokens = seq.tokens.upper()
            gc = (tokens.count("G") + tokens.count("C")) / len(seq) if len(seq) > 0 else 0.0
            constraints["gc"] = gc

    def _add_motif_constraints(self, aux: dict[str, float], constraints: dict[str, float]) -> None:
        """Add motif-related constraints from auxiliary metrics."""
        # Check for motif disruption flags
        motif_disruption = False
        for key, value in aux.items():
            if key.endswith("_disrupted") and value > 0:
                motif_disruption = True
                break

        constraints["motif_disruption"] = 1.0 if motif_disruption else 0.0

    def _add_conservation_constraints(self, aux: dict[str, float], constraints: dict[str, float]) -> None:
        """Add conservation-related constraints from auxiliary metrics."""
        # Add conservation scores for ref and alt windows
        ref_score = aux.get("ref_window_score", 0.0)
        alt_score = aux.get("alt_window_score", 0.0)

        constraints["conservation_ref_window"] = ref_score
        constraints["conservation_alt_window"] = alt_score
        constraints["conservation_delta"] = alt_score - ref_score
