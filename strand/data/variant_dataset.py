"""Variant-aware dataset loader for genomic sequences."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from strand.core.sequence import Sequence

if TYPE_CHECKING:
    from strand.engine.types import SequenceContext, VariantMetadata

logger = logging.getLogger(__name__)


class VariantDataset:
    """Dataset loader for genomic variants with sequence context.

    Loads variants from VCF files and extracts reference/alternative sequences
    with configurable genomic windows. Returns SequenceContext objects that
    pair variants with their sequence neighborhoods.
    """

    def __init__(
        self,
        vcf_path: str | Path,
        fasta_path: str | Path,
        window_size: int = 1000,
        batch_size: int | None = None,
        cache_sequences: bool = True,
    ):
        """Initialize VariantDataset.

        Args:
            vcf_path: Path to VCF file (can be .vcf or .vcf.gz)
            fasta_path: Path to reference FASTA file
            window_size: Base pairs to include on each side of variant
            batch_size: Number of variants to load at once (None = all)
            cache_sequences: Whether to cache loaded sequences

        Raises:
            ImportError: If pysam or PyRanges not installed
        """
        try:
            import pysam
            import pyranges as pr  # noqa: F401
        except ImportError as e:
            msg = (
                "VariantDataset requires pysam and PyRanges. "
                "Install with: pip install strand-sdk[variant-triage]"
            )
            raise ImportError(msg) from e

        self.vcf_path = Path(vcf_path)
        self.fasta_path = Path(fasta_path)
        self.window_size = window_size
        self.batch_size = batch_size
        self.cache_sequences = cache_sequences

        # Lazy loading
        self._vcf = None
        self._fasta = None
        self._variants = None

    @property
    def vcf(self):
        """Lazy load VCF file."""
        if self._vcf is None:
            import pysam

            logger.info(f"Loading VCF file: {self.vcf_path}")
            self._vcf = pysam.VariantFile(str(self.vcf_path))
        return self._vcf

    @property
    def fasta(self):
        """Lazy load FASTA file."""
        if self._fasta is None:
            import pysam

            logger.info(f"Loading FASTA file: {self.fasta_path}")
            self._fasta = pysam.FastaFile(str(self.fasta_path))
        return self._fasta

    @property
    def variants(self) -> list:
        """Load all variants from VCF."""
        if self._variants is None:
            logger.info("Loading variants from VCF")
            self._variants = list(self.vcf)
        return self._variants

    def _create_variant_metadata(self, variant) -> VariantMetadata:
        """Create VariantMetadata from pysam variant record."""
        from strand.engine.types import VariantMetadata

        # Extract basic info
        chrom = variant.chrom
        pos = variant.pos
        ref = variant.ref
        alt = variant.alts[0] if variant.alts else ""

        # Extract rsID if available
        rsid = None
        if hasattr(variant, 'id') and variant.id and variant.id != '.':
            rsid = variant.id

        # Extract annotations from INFO field
        annotations = {}
        if hasattr(variant, 'info'):
            for key, value in variant.info.items():
                annotations[key] = str(value)

        return VariantMetadata(
            chrom=chrom,
            pos=pos,
            ref=ref,
            alt=alt,
            rsid=rsid,
            annotations=annotations,
        )

    def _extract_sequence_window(
        self,
        chrom: str,
        center_pos: int,
        ref_allele: str,
        alt_allele: str | None = None,
    ) -> tuple[str, tuple[int, int]]:
        """Extract sequence window around variant position.

        Args:
            chrom: Chromosome name
            center_pos: Center position of variant (1-based)
            ref_allele: Reference allele
            alt_allele: Alternative allele (if provided, will be substituted)

        Returns:
            Tuple of (sequence_string, (start, end))
        """
        # Convert to 0-based coordinates
        pos_0based = center_pos - 1

        # Calculate window bounds (account for potential length change with ALT)
        chrom_length = self.fasta.get_reference_length(chrom)
        ref_len = len(ref_allele)
        alt_len = len(alt_allele) if alt_allele else ref_len
        
        # Window extends window_size on each side, plus max(ref_len, alt_len) for variant
        max_allele_len = max(ref_len, alt_len)
        start = max(0, pos_0based - self.window_size)
        end = min(chrom_length, pos_0based + max_allele_len + self.window_size)

        # Extract reference sequence
        sequence = self.fasta.fetch(chrom, start, end)

        # If alt_allele is provided, substitute it into the sequence
        if alt_allele is not None:
            # Calculate position within the extracted window (0-based relative to start)
            variant_start_in_window = pos_0based - start
            variant_end_in_window = variant_start_in_window + ref_len
            
            # Substitute ALT allele
            sequence = (
                sequence[:variant_start_in_window] +
                alt_allele +
                sequence[variant_end_in_window:]
            )

        return sequence, (start, end)

    def _create_sequence_context(self, variant) -> SequenceContext:
        """Create SequenceContext from variant record."""
        from strand.engine.types import SequenceContext

        metadata = self._create_variant_metadata(variant)

        # Extract reference sequence window (no ALT substitution)
        ref_seq_str, ref_window = self._extract_sequence_window(
            metadata.chrom, metadata.pos, metadata.ref, alt_allele=None
        )

        # Extract alternative sequence window (with ALT substitution)
        alt_seq_str, alt_window = self._extract_sequence_window(
            metadata.chrom, metadata.pos, metadata.ref, alt_allele=metadata.alt
        )

        # Create Sequence objects
        ref_seq = Sequence(
            id=f"{metadata.chrom}:{metadata.pos}:{metadata.ref}",
            tokens=ref_seq_str,
            metadata={"variant_type": "reference", "window": ref_window},
        )

        alt_seq = Sequence(
            id=f"{metadata.chrom}:{metadata.pos}:{metadata.alt}",
            tokens=alt_seq_str,
            metadata={"variant_type": "alternative", "window": alt_window},
        )

        return SequenceContext(
            ref_seq=ref_seq,
            alt_seq=alt_seq,
            metadata=metadata,
            ref_window=ref_window,
            alt_window=alt_window,
        )

    def __iter__(self) -> Iterator[SequenceContext]:
        """Iterate over variant contexts."""
        for variant in self.variants:
            try:
                context = self._create_sequence_context(variant)
                yield context
            except Exception as e:
                logger.warning(f"Failed to process variant {variant.chrom}:{variant.pos}: {e}")
                continue

    def __len__(self) -> int:
        """Return number of variants."""
        return len(self.variants)

    def get_contexts_batch(self, start_idx: int = 0, batch_size: int | None = None) -> list[SequenceContext]:
        """Get batch of sequence contexts.

        Args:
            start_idx: Starting index in variant list
            batch_size: Number of contexts to return (None = all remaining)

        Returns:
            List of SequenceContext objects
        """
        if batch_size is None:
            end_idx = len(self.variants)
        else:
            end_idx = min(start_idx + batch_size, len(self.variants))

        contexts = []
        for i in range(start_idx, end_idx):
            try:
                variant = self.variants[i]
                context = self._create_sequence_context(variant)
                contexts.append(context)
            except Exception as e:
                logger.warning(f"Failed to process variant {i}: {e}")
                continue

        return contexts

    def close(self):
        """Close file handles."""
        if self._vcf is not None:
            self._vcf.close()
            self._vcf = None
        if self._fasta is not None:
            self._fasta.close()
            self._fasta = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
