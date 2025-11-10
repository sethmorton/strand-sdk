"""Simple benchmark script for optimizers."""

from time import perf_counter

from strand.core.optimizer import Optimizer
from strand.rewards import RewardBlock


METHODS = ["cem", "cmaes", "ga", "random"]
SEQUENCES = ["ACDEFGHIKLMNPQRSTVWY" for _ in range(50)]


def benchmark() -> dict[str, float]:
    durations: dict[str, float] = {}
    for method in METHODS:
        optimizer = Optimizer(
            sequences=SEQUENCES,
            reward_blocks=[RewardBlock.stability(), RewardBlock.solubility()],
            method=method,
            iterations=5,
            population_size=10,
        )
        start = perf_counter()
        optimizer.run()
        durations[method] = perf_counter() - start
    return durations


if __name__ == "__main__":
    for method, duration in benchmark().items():
        print(f"{method}: {duration:.4f}s")
