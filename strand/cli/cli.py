"""Command-line entry point placeholder (surfaces only).

The new Engine surfaces are not implemented yet. This CLI stub
prints a friendly message and exits with a non-zero status.
"""

from __future__ import annotations

import argparse

from strand.rewards import RewardBlock
from strand.utils import get_logger

_logger = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Strand optimization experiment")
    parser.add_argument("sequence", help="Primary input sequence")
    parser.add_argument("--baseline", nargs="*", default=[], help="Baseline sequences for novelty")
    parser.add_argument("--method", default="cem", help="Optimization method")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    reward_blocks = [RewardBlock.stability(), RewardBlock.solubility()]
    if args.baseline:
        reward_blocks.append(RewardBlock.novelty(baseline=list(args.baseline)))

    _logger.warning("CLI surfaces are placeholders. Use strand.engine Engine programmatically.")
    _logger.error("CLI not implemented yet. Use strand.engine.Engine with RewardAggregator and LocalExecutor")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
