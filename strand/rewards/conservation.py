"""Conservation reward using genomic conservation tracks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from strand.rewards.base import BaseRewardBlock, BlockType, RewardBlockMetadata, RewardContext

if TYPE_CHECKING:
    from strand.engine.types import SequenceContext

logger = logging.getLogger(__name__)


class ConservationReward(BaseRewardBlock):
    """Reward block that scores variants based on conservation changes.

    Computes conservation score differences between reference and alternative
    genomic regions using bigWig tracks (e.g., PhyloP, GERP). Variants that
    increase conservation scores in functional regions are rewarded.
    """

    def __init__(
        self,
        bw_path: str,
        agg_method: Literal["mean", "max", "sum", "min"] = "mean",
        weight: float = 1.0,
    ):
        """Initialize ConservationReward.

        Args:
            bw_path: Path to bigWig file with conservation scores
            agg_method: Aggregation method for window scores
            weight: Reward weight multiplier

        Raises:
            ImportError: If pyBigWig is not installed
        """
        try:
            import pyBigWig  # noqa: F401
        except ImportError as e:
            msg = (
                "ConservationReward requires pyBigWig. "
                "Install with: pip install strand-sdk[variant-triage]"
            )
            raise ImportError(msg) from e

        super().__init__(
            name="conservation",
            weight=weight,
            metadata=RewardBlockMetadata(
                block_type=BlockType.ADVANCED,
                description="Genomic conservation scoring using bigWig tracks",
                requires_context=True,
            ),
        )

        self.bw_path = bw_path
        self.agg_method = agg_method

        # Lazy loading
        self._bw = None

    @property
    def bw(self):
        """Lazy load bigWig file."""
        if self._bw is None:
            import pyBigWig

            logger.info(f"Loading conservation track: {self.bw_path}")
            self._bw = pyBigWig.open(self.bw_path)
        return self._bw

    def _query_region(self, chrom: str, start: int, end: int) -> float | None:
        """Query conservation score for a genomic region.

        Args:
            chrom: Chromosome name
            start: Start position (0-based)
            end: End position (exclusive)

        Returns:
            Aggregated conservation score or None if no data
        """
        try:
            stats = self.bw.stats(
                chrom=chrom,
                start=start,
                end=end,
                type=self.agg_method,
                nBins=1,  # Single bin for whole region
            )
            return stats[0] if stats and stats[0] is not None else None
        except Exception as e:
            logger.warning(f"Failed to query {chrom}:{start}-{end}: {e}")
            return None

    def _score_window(self, chrom: str, start: int, end: int, window_name: str) -> dict[str, float]:
        """Score a genomic window and return metrics.

        Args:
            chrom: Chromosome
            start: Window start
            end: Window end
            window_name: Name for auxiliary metrics

        Returns:
            Dict with score and auxiliary metrics
        """
        score = self._query_region(chrom, start, end)

        if score is None:
            # No data available
            return {
                f"{window_name}_score": 0.0,
                f"{window_name}_has_data": 0.0,
            }

        return {
            f"{window_name}_score": score,
            f"{window_name}_has_data": 1.0,
        }

    def score_context(
        self,
        context: SequenceContext,
        *,
        reward_context: RewardContext | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Score conservation changes in variant context.

        Args:
            context: SequenceContext with genomic coordinates
            reward_context: Optional iteration context (unused)

        Returns:
            Tuple of (conservation_delta, auxiliary_metrics)
        """
        chrom = context.metadata.chrom

        # Score reference window
        ref_metrics = self._score_window(
            chrom=chrom,
            start=context.ref_window[0],
            end=context.ref_window[1],
            window_name="ref_window"
        )

        # Score alternative window
        alt_metrics = self._score_window(
            chrom=chrom,
            start=context.alt_window[0],
            end=context.alt_window[1],
            window_name="alt_window"
        )

        # Compute delta (alt - ref)
        ref_score = ref_metrics["ref_window_score"]
        alt_score = alt_metrics["alt_window_score"]
        conservation_delta = alt_score - ref_score

        # Combine auxiliary metrics
        aux = {}
        aux.update(ref_metrics)
        aux.update(alt_metrics)
        aux["conservation_delta"] = conservation_delta

        return conservation_delta, aux

    def _score(self, sequence, context: RewardContext) -> float:
        """Fallback scoring for non-context sequences."""
        # For sequences without context, return 0 (no conservation info)
        return 0.0

    def __del__(self):
        """Clean up bigWig file handle."""
        if hasattr(self, '_bw') and self._bw is not None:
            try:
                self._bw.close()
            except Exception:
                pass  # Ignore cleanup errors
