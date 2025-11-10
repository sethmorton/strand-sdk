"""Reward scoring micro benchmarks."""

from time import perf_counter

from strand.core.sequence import Sequence
from strand.rewards import RewardBlock

SEQUENCE = Sequence(id="seq", tokens="ACDEFGHIKLMNPQRSTVWY")


def time_block(block) -> float:
    start = perf_counter()
    for _ in range(1000):
        block.score(SEQUENCE)
    return perf_counter() - start


if __name__ == "__main__":
    blocks = {
        "stability": RewardBlock.stability(),
        "solubility": RewardBlock.solubility(),
        "novelty": RewardBlock.novelty(baseline=[SEQUENCE.tokens]),
        "length": RewardBlock.length_penalty(target_length=len(SEQUENCE)),
    }
    for name, block in blocks.items():
        print(f"{name}: {time_block(block):.6f}s")
