"""TFBS motif-based reward block for DNA regulatory element design.

This block scores sequences based on transcription factor binding site (TFBS) patterns,
enabling design of promoters/enhancers with specific TF binding profiles.

Uses JASPAR PWM database (via BioPython) for motif matching and scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from strand.core.sequence import Sequence
from strand.rewards.base import BaseRewardBlock, BlockType, RewardBlockMetadata, RewardContext

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TFBSConfig:
    """Configuration for TFBSFrequencyCorrelationBlock.

    Attributes
    ----------
    motif_names : list[str]
        TFBS motif names to load from JASPAR (e.g., ["CEBPB", "STAT1"]).
    target_frequency : dict[str, float] | None
        Target frequency for each motif (as fraction of sequence).
        If None, computed from training data.
    threshold : float
        PWM score threshold (0.0-1.0) for calling a hit.
    aggregation : {"correlation", "frequency", "weighted"}
        Scoring method:
        - "correlation": Pearson correlation with target profile
        - "frequency": Match frequency difference
        - "weighted": Weighted combination
    return_constraint_channel : bool
        If True, also return divergence channel for constraint enforcement.
    """

    motif_names: list[str]
    target_frequency: dict[str, float] | None = None
    threshold: float = 0.7
    aggregation: str = "correlation"
    return_constraint_channel: bool = True

    def __post_init__(self) -> None:
        valid = {"correlation", "frequency", "weighted"}
        if self.aggregation not in valid:
            msg = f"Invalid aggregation: {self.aggregation}"
            raise ValueError(msg)


class TFBSFrequencyCorrelationBlock(BaseRewardBlock):
    """Score sequences based on TFBS frequency matching.

    Enables optimization toward target TFBS profiles, useful for promoter
    and enhancer design with specific regulatory programs.

    Example
    -------
    >>> config = TFBSConfig(
    ...     motif_names=["CEBPB", "STAT1"],
    ...     target_frequency={"CEBPB": 0.3, "STAT1": 0.2},
    ... )
    >>> block = TFBSFrequencyCorrelationBlock(config)
    >>> scores = block([Sequence(id="seq1", tokens="ACGTACGT")])
    """

    def __init__(self, config: TFBSConfig, *, weight: float = 1.0) -> None:
        """Initialize TFBS reward block.

        Parameters
        ----------
        config : TFBSConfig
            Configuration specifying motifs and scoring method.
        """
        metadata = RewardBlockMetadata(
            block_type=BlockType.ADVANCED,
            description="TFBS frequency correlation block",
            requires_context=False,
        )
        super().__init__(name="tfbs", weight=weight, metadata=metadata)
        self.config = config
        self.motifs = self._load_motifs()
        self.pwm_scores_cache = {}

        self.max_scores = {name: np.sum(np.log(pwm / 0.25 + 1e-8) * (pwm == pwm.max(axis=1, keepdims=True)))
                   for name, pwm in self.motifs.items()}

    def _load_motifs(self) -> dict[str, np.ndarray]:
        """Load JASPAR motifs.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of motif names to PWM matrices.
        """
        try:
            from Bio.motifs import jaspar
            from Bio import SeqIO
        except ImportError:
            _LOGGER.warning(
                "BioPython required for JASPAR. Using stub motifs for testing. "
                "Install with: pip install biopython"
            )
            return self._create_stub_motifs()

        motifs = {}
        try:
            # Try to load from JASPAR database
            # This requires jaspar_core database to be available
            for name in self.config.motif_names:
                try:
                    motif = jaspar.fetch(name, "CORE")
                    # Convert to PWM (log-odds)
                    pwm = np.array(
                        [motif.pwm[:, i] for i in range(4)]
                    ).T  # (motif_len, 4)
                    motifs[name] = pwm
                    _LOGGER.info(f"Loaded JASPAR motif: {name}")
                except Exception as e:
                    _LOGGER.warning(f"Failed to load motif {name}: {e}. Using stub.")
                    motifs[name] = self._create_stub_pwm()

        except Exception as e:
            _LOGGER.warning(f"JASPAR loading failed: {e}. Using stub motifs.")
            return self._create_stub_motifs()

        return motifs

    def _create_stub_motifs(self) -> dict[str, np.ndarray]:
        """Create stub PWM matrices for testing.

        Returns
        -------
        dict[str, np.ndarray]
            Stub motif PWMs.
        """
        motifs = {}
        for name in self.config.motif_names:
            motifs[name] = self._create_stub_pwm()
        return motifs

    def _create_stub_pwm(self, motif_len: int = 8) -> np.ndarray:
        """Create a random PWM for testing.

        Parameters
        ----------
        motif_len : int
            Length of motif.

        Returns
        -------
        np.ndarray
            Random PWM of shape (motif_len, 4).
        """
        pwm = np.random.dirichlet(np.ones(4), size=motif_len)
        return pwm

    def __call__(self, sequences: list[Sequence]) -> dict[str, float]:
        """Score sequences based on TFBS frequencies.

        Parameters
        ----------
        sequences : list[Sequence]
            DNA sequences to score.

        Returns
        -------
        dict[str, float]
            Scoring results:
            - "tfbs_correlation": Correlation with target profile
            - "tfbs_frequency": Raw motif frequency difference
            - "tfbs_divergence": Divergence from target (for constraints)
        """
        if not sequences:
            return {"tfbs_correlation": 0.0, "tfbs_frequency": 0.0}

        # Compute TFBS frequencies in all sequences
        frequencies = []
        for seq in sequences:
            freq = self._compute_tfbs_frequency(seq.tokens)
            frequencies.append(freq)

        frequencies = np.array(frequencies)  # (num_seqs, num_motifs)

        # Get target frequencies
        if self.config.target_frequency:
            target = np.array(
                [
                    self.config.target_frequency.get(name, 0.0)
                    for name in self.config.motif_names
                ]
            )
        else:
            target = np.mean(frequencies, axis=0)  # Use empirical as target

        # Compute correlation
        correlations = []
        for freq in frequencies:
            # Pearson correlation
            if np.std(freq) > 0 and np.std(target) > 0:
                corr = np.corrcoef(freq, target)[0, 1]
            else:
                corr = 0.0 if np.allclose(freq, target) else -1.0
            correlations.append(corr if not np.isnan(corr) else 0.0)

        # Compute frequency difference
        freq_diffs = np.mean(np.abs(frequencies - target), axis=1)

        # Compute divergence (KL or L2)
        divergences = np.linalg.norm(frequencies - target, axis=1)

        mean_corr = float(np.mean(correlations))
        mean_freq = float(1.0 - np.mean(freq_diffs))  # Invert so higher is better

        result = {
            "tfbs_correlation": mean_corr,
            "tfbs_frequency": mean_freq,
        }

        if self.config.return_constraint_channel:
            result["tfbs_divergence"] = float(np.mean(divergences))

        return result

    def _score(self, sequence: Sequence, context: RewardContext) -> float:  # noqa: ARG002
        """Scalar objective used by RewardAggregator.

        The aggregation mode controls which channel drives the objective:
        - correlation: use tfbs_correlation
        - frequency:  use tfbs_frequency
        - weighted:   0.7 * correlation + 0.3 * frequency
        """
        result = self([sequence])

        corr = float(result.get("tfbs_correlation", 0.0))
        freq = float(result.get("tfbs_frequency", 0.0))

        if self.config.aggregation == "correlation":
            return corr
        if self.config.aggregation == "frequency":
            return freq
        # weighted
        return 0.7 * corr + 0.3 * freq

    def _compute_tfbs_frequency(self, seq: str) -> np.ndarray:
        """Compute TFBS frequencies in a sequence.

        Parameters
        ----------
        seq : str
            DNA sequence.

        Returns
        -------
        np.ndarray
            Frequency of each motif (shape: num_motifs).
        """
        seq = seq.upper()
        frequencies = []

        for motif_name, pwm in self.motifs.items():
            # Scan sequence with PWM
            scores = self._scan_pwm(seq, pwm)

            # Count hits above threshold
            #hits = np.sum(scores >= self.config.threshold)
            #freq = hits / len(seq) if len(seq) > 0 else 0.0
            total_strength = np.sum(scores[scores >= self.config.threshold])
            freq = total_strength / (len(seq) - len(pwm) + 1)

            frequencies.append(freq)

        return np.array(frequencies)

    def _scan_pwm(self, seq: str, pwm: np.ndarray) -> np.ndarray:
        """Scan sequence with a PWM.

        Parameters
        ----------
        seq : str
            DNA sequence.
        pwm : np.ndarray
            PWM matrix of shape (motif_len, 4).

        Returns
        -------
        np.ndarray
            PWM scores for each position.
        """
        # One-hot encode sequence
        alphabet = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 0}
        motif_len = pwm.shape[0]

        seq_encoded = np.zeros((len(seq), 4))
        for i, base in enumerate(seq):
            idx = alphabet.get(base, 0)
            seq_encoded[i, idx] = 1.0

        # Compute PWM scores
        scores = np.zeros(len(seq) - motif_len + 1)
        for i in range(len(seq) - motif_len + 1):
            window = seq_encoded[i : i + motif_len]
            # Dot product with PWM (sum of products for matching positions)
            score = np.sum(window * pwm) / motif_len
            scores[i] = score

        return scores

    def _scan_pwm_log_odds(self, seq: str, pwm: np.ndarray, background: np.ndarray = None) -> np.ndarray:
        if background is None:
            background = np.array([0.25, 0.25, 0.25, 0.25])  # uniform
        log_pwm = np.log(pwm / background.reshape(1, 4) + 1e-8)
        
        seq_encoded = np.zeros((len(seq), 4))
        alphabet = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 0}
        for i, base in enumerate(seq.upper()):
            seq_encoded[i, alphabet.get(base, 0)] = 1.0

        motif_len = pwm.shape[0]
        scores = np.zeros(len(seq) - motif_len + 1)
        for i in range(len(seq) - motif_len + 1):
            window = seq_encoded[i:i + motif_len]
            scores[i] = np.sum(window * log_pwm)
        return scores

    def _scan_both_strands(self, seq: str, pwm: np.ndarray) -> np.ndarray:
        rev_seq = str(Seq(seq).reverse_complement())
        fwd_scores = self._scan_pwm(seq, pwm)
        rev_scores = self._scan_pwm(rev_seq, pwm)
        # Align reverse scores to forward coordinate
        aligned_rev = np.zeros(len(seq) - len(pwm) + 1)
        for i, score in enumerate(rev_scores):
            aligned_rev[len(seq) - len(pwm) - i] = score
        return np.maximum(fwd_scores, aligned_rev)

__all__ = [
    "TFBSConfig",
    "TFBSFrequencyCorrelationBlock",
]
