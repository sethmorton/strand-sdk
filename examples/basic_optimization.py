#!/usr/bin/env python3
"""
Example pipeline demonstrating the Strand SDK for biological sequence optimization.

This example shows:
1. Creating reward blocks (stability, solubility, novelty, gc_content)
2. Running optimization with different strategies (Random, CEM)
3. Accessing results and iteration statistics
4. Exporting results
"""

from pathlib import Path

from strand.core.sequence import Sequence
from strand.engine import Engine, EngineConfig, default_score
from strand.engine.executors.local import LocalExecutor
from strand.engine.strategies.cem import CEMStrategy
from strand.engine.strategies.random import RandomStrategy
from strand.evaluators.reward_aggregator import RewardAggregator
from strand.rewards import RewardBlock


def main() -> None:
    """Run the example optimization pipeline."""
    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    print("🧬 Strand SDK Example Pipeline (Engine-based)\n")  # noqa: T201
    print("=" * 60)  # noqa: T201

    # 1. Define baseline sequences
    print("\n1️⃣  Defining baseline sequences...")  # noqa: T201
    baseline = [
        "MKTAYIAKQRQISFVKSHFSRQDILDLQY",
        "MKPAYIAKQRQISFVKSHFSRQDILDVQY",
    ]
    print(f"   Loaded {len(baseline)} baseline sequences")  # noqa: T201
    for i, seq in enumerate(baseline, 1):
        print(f"   {i}. {seq}")  # noqa: T201

    # 2. Define reward blocks
    print("\n2️⃣  Defining reward blocks...")  # noqa: T201
    rewards = [
        RewardBlock.stability(weight=1.0),
        RewardBlock.solubility(weight=0.5),
        RewardBlock.novelty(baseline=baseline, weight=0.3),
        RewardBlock.gc_content(target=0.5, tolerance=0.1, weight=0.2),
    ]
    print(f"   Loaded {len(rewards)} reward blocks:")  # noqa: T201
    for block in rewards:
        print(f"   - {block.name} (weight: {block.weight})")  # noqa: T201

    # 3. Test different strategies
    strategies_config = [
        ("random", RandomStrategy(
            alphabet="ACDEFGHIKLMNPQRSTVWY",
            min_len=20,
            max_len=35,
            seed=42,
        )),
        ("cem", CEMStrategy(
            alphabet="ACDEFGHIKLMNPQRSTVWY",
            min_len=20,
            max_len=35,
            seed=42,
        )),
    ]

    all_results = {}

    for strategy_name, strategy in strategies_config:
        print(f"\n3️⃣  Running optimization with strategy: {strategy_name.upper()}")  # noqa: T201
        print("-" * 60)  # noqa: T201

        # Create evaluator and executor
        evaluator = RewardAggregator(reward_blocks=rewards)
        executor = LocalExecutor(evaluator=evaluator, mode="auto")

        # Configure engine
        config = EngineConfig(
            iterations=5,
            population_size=20,
            seed=42,
            method=strategy_name,
        )

        # Create and run engine
        engine = Engine(
            config=config,
            strategy=strategy,
            evaluator=evaluator,
            executor=executor,
            score_fn=default_score,
        )

        results = engine.run()
        all_results[strategy_name] = results

        # Display results
        if results.best:
            best_seq, best_score = results.best
            print(f"\n   Best sequence (score: {best_score:.4f}):")  # noqa: T201
            print(f"   {best_seq.tokens}")  # noqa: T201

        # Display iteration statistics
        print(f"\n   Iteration statistics:")  # noqa: T201
        print(f"   {'Iter':<6} {'Best':<10} {'Mean':<10} {'Std':<10}")  # noqa: T201
        print("   " + "-" * 36)  # noqa: T201
        for stat in results.history:
            print(f"   {stat.iteration:<6} {stat.best:<10.4f} {stat.mean:<10.4f} {stat.std:<10.4f}")  # noqa: T201

    # 4. Compare strategies
    print("\n4️⃣  Comparing Strategies")  # noqa: T201
    print("=" * 60)  # noqa: T201
    print(f"{'Strategy':<12} {'Best Score':<15} {'Mean Score':<15}")  # noqa: T201
    print("-" * 60)  # noqa: T201

    for strategy_name in [s[0] for s in strategies_config]:
        results = all_results[strategy_name]
        if results.best:
            best_score = results.best[1]
            mean_scores = [s.mean for s in results.history]
            mean_score = sum(mean_scores) / len(mean_scores) if mean_scores else 0
            print(f"{strategy_name:<12} {best_score:>13.4f}  {mean_score:>13.4f}")  # noqa: T201

    print("\n" + "=" * 60)  # noqa: T201
    print("✅ Pipeline complete!")  # noqa: T201


if __name__ == "__main__":
    main()
