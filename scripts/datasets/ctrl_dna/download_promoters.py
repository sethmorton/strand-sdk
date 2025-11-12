#!/usr/bin/env python3
"""Download and prepare promoter datasets for Ctrl-DNA experiments."""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_promoter_library(output_dir: Path, source: str = "mock") -> Path:
    """Download promoter library from source.

    Parameters
    ----------
    output_dir : Path
        Output directory for dataset.
    source : str
        Source ("mock", "gse", "tcga").

    Returns
    -------
    Path
        Path to downloaded file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"promoters_{source}.fasta"

    if source == "mock":
        # Create mock promoter dataset
        with open(output_file, "w") as f:
            for i in range(100):
                seq = "ACGTACGT" * 10  # 80bp
                f.write(f">promoter_{i}\n{seq}\n")
        logger.info(f"Created mock dataset: {output_file}")
    else:
        logger.warning(f"Source {source} not yet implemented. Using mock.")

    return output_file


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and prepare Ctrl-DNA datasets"
    )
    parser.add_argument("--output-dir", default="data/promoters")
    parser.add_argument("--source", default="mock")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    download_promoter_library(output_dir, args.source)
    logger.info(f"Dataset prepared in {output_dir}")


if __name__ == "__main__":
    main()

