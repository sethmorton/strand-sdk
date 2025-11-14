"""Motif delta reward using JASPAR motifs and MOODS scanning."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from strand.rewards.base import BaseRewardBlock, BlockType, RewardBlockMetadata, RewardContext

if TYPE_CHECKING:
    from strand.engine.types import SequenceContext

logger = logging.getLogger(__name__)


class MotifDeltaReward(BaseRewardBlock):
    """Reward block that scores variants based on transcription factor motif changes.

    Computes the delta in motif matches between reference and alternative sequences
    using JASPAR position weight matrices and MOODS scanning. Positive scores
    indicate variants that create or enhance TF binding sites.
    """

    def __init__(
        self,
        tf_list: list[str] | None = None,
        threshold: float = 6.0,
        background_freq: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
        weight: float = 1.0,
    ):
        """Initialize MotifDeltaReward.

        Args:
            tf_list: List of JASPAR TF accession IDs (e.g., ["MA0001", "MA0002"]).
                    If None, uses common TFs.
            threshold: Log-odds threshold for motif matches (default: 6.0)
            background_freq: Background nucleotide frequencies (A, C, G, T)
            weight: Reward weight multiplier

        Raises:
            ImportError: If pyjaspar or MOODS-python not installed
        """
        try:
            import pyjaspar  # noqa: F401
            import MOODS.scan
            import MOODS.parsers  # noqa: F401
        except ImportError as e:
            msg = (
                "MotifDeltaReward requires pyjaspar and MOODS-python. "
                "Install with: pip install strand-sdk[variant-triage]"
            )
            raise ImportError(msg) from e

        super().__init__(
            name="motif_delta",
            weight=weight,
            metadata=RewardBlockMetadata(
                block_type=BlockType.ADVANCED,
                description="TF motif presence delta using JASPAR PWMs and MOODS scanning",
                requires_context=True,
            ),
        )

        self.tf_list = tf_list or ["MA0001", "MA0002", "MA0003"]  # Default common TFs
        self.threshold = threshold
        self.background_freq = background_freq

        # Lazy loading
        self._motifs = None
        self._matrices = None

    @property
    def motifs(self):
        """Lazy load JASPAR motifs."""
        if self._motifs is None:
            from pyjaspar import jaspardb

            logger.info(f"Loading JASPAR motifs: {self.tf_list}")
            self._motifs = {}
            for tf_acc in self.tf_list:
                try:
                    motif_list = jaspardb.fetch([tf_acc])
                    if motif_list:
                        self._motifs[tf_acc] = motif_list[0]
                    else:
                        logger.warning(f"TF {tf_acc} not found in JASPAR")
                except Exception as e:
                    logger.warning(f"Failed to load TF {tf_acc}: {e}")

        return self._motifs

    @property
    def matrices(self):
        """Convert motifs to MOODS-compatible log-odds matrices."""
        if self._matrices is None:
            import MOODS.parsers
            
            self._matrices = []
            for tf_acc, motif in self.motifs.items():
                # JASPAR matrices are 4xN (A, C, G, T rows) - position frequency matrices
                # MOODS requires log-odds matrices
                pfm = motif.matrix.astype(float)
                # Convert PFM to log-odds using MOODS parser
                log_odds_matrix = MOODS.parsers.pfm_to_log_odds(pfm, self.background_freq)
                self._matrices.append(log_odds_matrix)
        return self._matrices

    def _scan_sequence(self, sequence: str) -> dict[str, int]:
        """Scan sequence for motif matches.

        Args:
            sequence: DNA sequence string

        Returns:
            Dict mapping TF accession to match count
        """
        if not self.matrices:
            return {}

        # MOODS.scan.scan_dna signature: scan_dna(sequence, matrices, bg, thresholds)
        # thresholds can be a single float or list of floats (one per matrix)
        thresholds = [self.threshold] * len(self.matrices)

        # Run MOODS scan (positional arguments, no keywords)
        results = MOODS.scan.scan_dna(sequence, self.matrices, self.background_freq, thresholds)

        # Count matches per TF
        counts = {}
        for i, tf_acc in enumerate(self.motifs.keys()):
            if i < len(results):
                # results[i] contains (pos, score, strand) tuples for this matrix
                match_count = len(results[i])
                counts[f"{tf_acc}_matches"] = match_count
            else:
                counts[f"{tf_acc}_matches"] = 0

        return counts

    def _compute_delta(self, ref_counts: dict[str, int], alt_counts: dict[str, int]) -> dict[str, float]:
        """Compute deltas between ref and alt counts.

        Args:
            ref_counts: Match counts for reference sequence
            alt_counts: Match counts for alternative sequence

        Returns:
            Dict of deltas and disruption flags
        """
        deltas = {}
        total_delta = 0.0

        for tf_acc in self.motifs.keys():
            ref_count = ref_counts.get(f"{tf_acc}_matches", 0)
            alt_count = alt_counts.get(f"{tf_acc}_matches", 0)

            delta = alt_count - ref_count
            deltas[f"{tf_acc}_delta"] = delta
            total_delta += delta

            # Disruption flag: True if reference had matches but alternative doesn't
            if ref_count > 0 and alt_count == 0:
                deltas[f"{tf_acc}_disrupted"] = 1.0
            else:
                deltas[f"{tf_acc}_disrupted"] = 0.0

        deltas["total_motif_delta"] = total_delta
        return deltas

    def score_context(
        self,
        context: SequenceContext,
        *,
        reward_context: RewardContext | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Score motif changes in variant context.

        Args:
            context: SequenceContext with ref/alt sequences
            reward_context: Optional iteration context (unused)

        Returns:
            Tuple of (total_motif_delta, auxiliary_metrics)
        """
        # Scan both sequences
        ref_counts = self._scan_sequence(context.ref_seq.tokens)
        alt_counts = self._scan_sequence(context.alt_seq.tokens)

        # Compute deltas
        deltas = self._compute_delta(ref_counts, alt_counts)

        # Objective: total motif delta
        objective = deltas["total_motif_delta"]

        # Auxiliary: all per-TF metrics
        aux = deltas.copy()

        return objective, aux

    def _score(self, sequence, context: RewardContext) -> float:
        """Fallback scoring for non-context sequences."""
        # For single sequences, just count total motif matches
        counts = self._scan_sequence(sequence.tokens)
        return sum(counts.values())
