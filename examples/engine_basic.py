"""Minimal example showing how to run the new engine."""

from __future__ import annotations

from strand.engine import Engine, EngineConfig, default_score
from strand.engine.executors.local import LocalExecutor
from strand.engine.strategies.random import RandomStrategy
from strand.evaluators.reward_aggregator import RewardAggregator
from strand.rewards import RewardBlock


def main() -> None:
    rewards = [
        RewardBlock.stability(weight=1.0),
        RewardBlock.length_penalty(target_length=12, tolerance=2, weight=0.5),
    ]
    evaluator = RewardAggregator(reward_blocks=rewards)
    executor = LocalExecutor(evaluator=evaluator, batch_size=8)
    strategy = RandomStrategy(
        alphabet="ACDEFGHIKLMNPQRSTVWY",
        min_len=10,
        max_len=15,
        seed=1234,
    )

    config = EngineConfig(iterations=3, population_size=8, seed=1234, method="random")
    engine = Engine(
        config=config,
        strategy=strategy,
        evaluator=evaluator,
        executor=executor,
        score_fn=default_score,
    )

    results = engine.run()

    print("Iterations completed:", results.summary["iterations_completed"])
    print("Total evaluations:", results.summary["total_evals"])
    if results.best:
        best_seq, best_score = results.best
        print("Best score:", best_score)
        print("Best sequence:", best_seq.tokens)

    for stats in results.history:
        print(
            f"Iteration {stats.iteration}: best={stats.best:.4f} mean={stats.mean:.4f}"
            f" evals={stats.evals} throughput={stats.throughput:.2f}/s"
        )


if __name__ == "__main__":
    main()
