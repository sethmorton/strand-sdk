"""Complete example with CMA-ES, constraints, and MLflow tracking."""

from __future__ import annotations

from strand.core.sequence import Sequence
from strand.engine import Engine, EngineConfig, default_score
from strand.engine.constraints import ConstraintSolver
from strand.engine.executors.pool import LocalPoolExecutor
from strand.engine.strategies.cmaes import CMAESStrategy
from strand.evaluators.composite import CompositeEvaluator
from strand.evaluators.reward_aggregator import RewardAggregator
from strand.logging import MLflowTracker
from strand.rewards import RewardBlock


def main() -> None:
    """Run a complete optimization pipeline with tracking and constraints."""

    # 1. Set up MLflow tracking
    tracker = MLflowTracker(
        experiment_name="strand-cmaes-demo",
        tracking_uri="./mlruns",
    )

    # 2. Define rewards
    rewards = [
        RewardBlock.stability(weight=1.0),
        RewardBlock.gc_content(target=0.5, tolerance=0.1, weight=0.5),
        RewardBlock.length_penalty(target_length=15, tolerance=2, weight=0.2),
    ]

    # 3. Set up constraint solver
    constraint_solver = ConstraintSolver(
        alphabet="ACDEFGHIKLMNPQRSTVWY",
        min_len=12,
        max_len=20,
    )

    # 4. Create evaluator with constraints
    evaluator = CompositeEvaluator(
        rewards=RewardAggregator(reward_blocks=rewards),
        include_length=True,
        include_gc=True,
    )

    # 5. Use parallel executor
    executor = LocalPoolExecutor(evaluator=evaluator, mode="process", num_workers=4)

    # 6. Create CMA-ES strategy
    strategy = CMAESStrategy(
        alphabet="ACDEFGHIKLMNPQRSTVWY",
        min_len=12,
        max_len=20,
        seed=42,
        sigma0=0.3,
    )

    # 7. Configure engine
    config = EngineConfig(
        iterations=10,
        population_size=32,
        seed=42,
        method="cmaes",
    )

    # 8. Start tracking
    tracker.start_run("cmaes_optimization")
    tracker.log_config(config)

    try:
        # 9. Create and run engine
        engine = Engine(
            config=config,
            strategy=strategy,
            evaluator=evaluator,
            executor=executor,
            score_fn=default_score,
        )

        print("🧬 Running optimization pipeline...")
        print("=" * 60)

        results = engine.run()

        # 10. Track iterations
        for stats in results.history:
            tracker.log_iteration_stats(stats.iteration, stats)

        # 11. Track final results
        tracker.log_results(results)

        # 12. Display results
        print(f"\n✅ Optimization complete!")
        print(f"Total iterations: {results.summary.get('iterations_completed')}")
        print(f"Total evaluations: {results.summary.get('total_evals')}")

        if results.best:
            best_seq, best_score = results.best
            print(f"\nBest score: {best_score:.4f}")
            print(f"Best sequence: {best_seq.tokens}")
            print(f"Feasible: {constraint_solver.is_feasible(best_seq)}")

        # 13. Log summary
        summary_data = {
            "best_score": results.best[1] if results.best else None,
            "best_sequence": results.best[0].tokens if results.best else None,
            "iterations": len(results.history),
            "total_evals": results.summary.get("total_evals"),
            "config": {
                "iterations": config.iterations,
                "population_size": config.population_size,
                "method": config.method,
            },
        }
        tracker.log_artifact_json(summary_data, "optimization_summary.json")

        print("\n📊 Results logged to MLflow:")
        print(f"Experiment: strand-cmaes-demo")
        print(f"View UI: mlflow ui --backend-store-uri ./mlruns")

    finally:
        tracker.end_run()


if __name__ == "__main__":
    main()

