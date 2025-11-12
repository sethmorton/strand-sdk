"""Example: Ctrl-DNA Style Constrained RL + Hybrid Optimization.

Demonstrates:
1. RL-based sequence generation (policy learning from rewards)
2. Hybrid ensemble of multiple strategies
3. Constrained optimization with biological plausibility
4. Temperature annealing for exploration control

Inspired by: "Ctrl-DNA: Controllable Cell-Type-Specific Regulatory DNA Design 
via Constrained RL" (Chen et al., 2025)
https://arxiv.org/abs/2505.20578
"""

from __future__ import annotations

import torch

from strand.engine import Engine, EngineConfig, default_score, strategy_from_name
from strand.engine.executors.local import LocalExecutor
from strand.engine.constraints import ConstraintSolver
from strand.engine.runtime import BatchConfig, DeviceConfig
from strand.evaluators.reward_aggregator import RewardAggregator
from strand.rewards import RewardBlock


def main() -> None:
    """Run Ctrl-DNA style constrained RL with hybrid ensemble."""

    print("🧬 Ctrl-DNA Style Constrained RL + Hybrid Optimization")
    print("=" * 70)

    # Define rewards (simulating cell-type specificity + biological properties)
    rewards = [
        RewardBlock.stability(weight=1.0),  # Target objective
        RewardBlock.gc_content(target=0.5, tolerance=0.15, weight=0.3),  # Constraint
    ]
    evaluator = RewardAggregator(reward_blocks=rewards)

    # Create constraint solver (biological plausibility filter)
    constraint_solver = ConstraintSolver(
        alphabet="ACDEFGHIKLMNPQRSTVWY",
        min_len=10,
        max_len=30,
    )

    # Create executor
    executor = LocalExecutor(evaluator=evaluator)

    # 1. RL-POLICY STRATEGY (learns from rewards, like Ctrl-DNA)
    print("\n📚 Creating RL-Policy Strategy (Ctrl-DNA style)...")
    rl_strategy = strategy_from_name(
        "rl-policy",
        alphabet="ACDEFGHIKLMNPQRSTVWY",
        min_len=10,
        max_len=30,
        seed=42,
        learning_rate=0.15,  # Policy gradient learning rate
        temperature=1.5,  # Start with high exploration
        constraint_penalty=0.5,  # Penalty for constraint violations
    )

    # 2. HYBRID STRATEGY (ensemble: RL + CEM + GA)
    print("🤝 Creating Hybrid Strategy (ensemble of RL + CEM + GA)...")
    cem_strategy = strategy_from_name(
        "cem",
        alphabet="ACDEFGHIKLMNPQRSTVWY",
        min_len=10,
        max_len=30,
        seed=42,
    )
    ga_strategy = strategy_from_name(
        "ga",
        alphabet="ACDEFGHIKLMNPQRSTVWY",
        min_len=10,
        max_len=30,
        seed=42,
    )

    hybrid_strategy = strategy_from_name(
        "hybrid",
        strategies=[rl_strategy, cem_strategy, ga_strategy],
        selection_method="best-of-generation",
    )

    print(f"  Strategies in ensemble: {len(hybrid_strategy.strategies)}")

    # Configure engine
    device_target = "cuda" if torch.cuda.is_available() else "cpu"
    config = EngineConfig(
        iterations=12,
        population_size=48,
        seed=42,
        method="hybrid-rl-cem-ga",
        batching=BatchConfig(eval_size=24, train_size=12, max_tokens=960),
        device=DeviceConfig(target=device_target, mixed_precision="bf16" if device_target == "cuda" else "no"),
    )

    # Create and run engine
    engine = Engine(
        config=config,
        strategy=hybrid_strategy,
        evaluator=evaluator,
        executor=executor,
        score_fn=default_score,
    )

    print(f"\n⚙️  Optimization Configuration:")
    print(f"  Iterations: {config.iterations}")
    print(f"  Population: {config.population_size}")
    print(f"  Length range: [{constraint_solver.min_len}, {constraint_solver.max_len}]")
    print("=" * 70)

    # Run optimization
    results = engine.run()

    # Display results
    print(f"\n✅ Optimization Complete!")
    print(f"  Total iterations: {len(results.history)}")
    print(f"  Total evaluations: {results.summary.get('total_evals')}")

    if results.best:
        best_seq, best_score = results.best
        is_feasible = constraint_solver.is_feasible(best_seq)
        print(f"\n🏆 Best Sequence Found:")
        print(f"  Score: {best_score:.4f}")
        print(f"  Sequence: {best_seq.tokens}")
        print(f"  Length: {len(best_seq.tokens)}")
        print(f"  Feasible: {'✅ Yes' if is_feasible else '❌ No'}")

    # Strategy performance comparison
    print(f"\n📊 Strategy Performance Comparison:")
    for name, score in hybrid_strategy.get_strategy_performance():
        print(f"  {name}: {score:.4f}")

    # Convergence trajectory
    print(f"\n📈 Convergence (best score per iteration):")
    for i, stats in enumerate(results.history):
        bar_width = int(stats.best * 20)
        bar = "█" * bar_width + "░" * (20 - bar_width)
        print(f"  Iter {i:2d}: [{bar}] {stats.best:.4f}")

    # RL policy entropy (exploration indicator)
    if hasattr(rl_strategy, "get_policy_entropy"):
        entropy = rl_strategy.get_policy_entropy()
        print(f"\n🎲 RL Policy Entropy: {entropy:.4f} bits")
        print(f"   (Higher = more exploration, Lower = more exploitation)")

    print("\n" + "=" * 70)
    print("🎯 Key Takeaways:")
    print("  • RL strategy learns token preferences from rewards")
    print("  • Hybrid ensemble combines RL + evolutionary methods")
    print("  • Constrained optimization enforces biological plausibility")
    print("  • Multiple strategies prevent premature convergence")


if __name__ == "__main__":
    main()
