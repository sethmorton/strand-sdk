"""CLI command for running variant triage optimization pipelines."""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from strand.core.sequence import Sequence

_LOGGER = logging.getLogger(__name__)


def run_variant_triage_pipeline(config_path: str | Path, device: str = "auto") -> None:
    """Run a complete variant triage pipeline from config.

    Parameters
    ----------
    config_path : str | Path
        Path to variant triage configuration YAML file.
    device : str
        Device for computation ("auto", "cpu", "cuda").
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    _LOGGER.info(f"Loading variant triage config from {config_path}")

    # Load config
    config = _load_config(config_path)

    # Validate config has required sections
    _validate_config(config)

    _LOGGER.info("Config loaded and validated successfully")

    # Build components
    dataset = _build_variant_dataset(config["dataset"])
    reward_blocks = _build_reward_blocks(config["rewards"])
    evaluator = _build_variant_evaluator(config["evaluator"], reward_blocks)

    # Load all variant contexts from dataset
    _LOGGER.info(f"Loading {len(dataset)} variants from dataset")
    contexts = list(dataset)
    _LOGGER.info(f"Loaded {len(contexts)} variant contexts")

    # Build variant-aware strategy and executor
    strategy = _build_variant_strategy(contexts)
    executor = _build_variant_executor(config.get("executor", {}), device, evaluator)
    engine_config = _build_engine_config(config["engine"], total_contexts=len(contexts))

    _LOGGER.info(f"Dataset: {len(contexts)} variants loaded")
    _LOGGER.info(f"Rewards: {len(reward_blocks)} blocks")
    _LOGGER.info("Using variant-aware strategy and executor")

    # Run optimization
    from strand.engine import Engine
    from strand.engine.score import default_score
    from strand.engine.rules import Rules

    engine = Engine(
        config=engine_config,
        strategy=strategy,
        evaluator=evaluator,
        executor=executor,
        score_fn=default_score,
        constraints=[],  # Constraints handled by VariantCompositeEvaluator
        rules=Rules(),
    )

    results = engine.run()

    # Report results
    if results.best:
        seq, score = results.best
        variant_info = _describe_variant(seq)
        print(f"Best variant ({score:.4f}): {variant_info}")
    else:
        print("No valid variants produced.")

    print("Variant triage optimization complete.")


def _load_config(path: Path) -> dict[str, Any]:
    """Load YAML config file."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        raise RuntimeError("pyyaml is required for config files. Install with `pip install pyyaml`.")
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")


def _validate_config(config: dict[str, Any]) -> None:
    """Validate variant triage config has required sections."""
    required_sections = ["dataset", "rewards", "evaluator", "engine"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config missing required section: {section}")

    # Validate dataset section
    dataset = config["dataset"]
    if dataset.get("type") != "variant_vcf":
        raise ValueError("dataset.type must be 'variant_vcf' for variant triage")

    required_dataset_fields = ["vcf_path", "fasta_path"]
    for field in required_dataset_fields:
        if field not in dataset:
            raise ValueError(f"dataset.{field} is required")


def _build_variant_dataset(dataset_config: dict[str, Any]):
    """Build VariantDataset from config."""
    try:
        from strand.data.variant_dataset import VariantDataset
    except ImportError as e:
        raise RuntimeError(
            f"VariantDataset requires additional dependencies. "
            f"Install with: pip install strand-sdk[variant-triage]. Error: {e}"
        )

    return VariantDataset(
        vcf_path=dataset_config["vcf_path"],
        fasta_path=dataset_config["fasta_path"],
        window_size=dataset_config.get("window_size", 1000),
    )


def _build_reward_blocks(rewards_config: list[dict[str, Any]]):
    """Build reward blocks from config using registry factory."""
    try:
        from strand.rewards.registry import RewardRegistry, register_advanced_blocks
        register_advanced_blocks()
    except ImportError as e:
        raise RuntimeError(
            f"Advanced rewards require additional dependencies. "
            f"Install with: pip install strand-sdk[variant-triage]. Error: {e}"
        )

    blocks = []
    for reward_spec in rewards_config:
        block = RewardRegistry.create_from_config(reward_spec)
        blocks.append(block)

    return blocks


def _build_variant_evaluator(evaluator_config: dict[str, Any], reward_blocks):
    """Build VariantCompositeEvaluator from config."""
    try:
        from strand.evaluators.reward_aggregator import RewardAggregator
        from strand.evaluators.variant_composite import VariantCompositeEvaluator
    except ImportError as e:
        raise RuntimeError("Variant evaluator requires variant-triage dependencies.")

    # Build reward aggregator
    rewards = RewardAggregator(reward_blocks)

    # Build variant composite evaluator
    return VariantCompositeEvaluator(
        rewards=rewards,
        include_length=evaluator_config.get("include_length", False),
        include_gc=evaluator_config.get("include_gc", False),
        include_motif_disruption=evaluator_config.get("include_motif_disruption", False),
        include_conservation_windows=evaluator_config.get("include_conservation_windows", False),
    )


def _build_executor(executor_config: dict[str, Any], device: str, evaluator):
    """Build executor for variant triage.

    Args:
        executor_config: Executor configuration dict
        device: Device specification
        evaluator: The actual evaluator to use (not a dummy)
    """
    try:
        from strand.engine.executors.factory import ExecutorFactory
    except ImportError:
        raise RuntimeError("Executor factory not available.")

    # Set device in config if not specified
    if "device" not in executor_config:
        if device == "auto":
            import torch
            executor_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            executor_config["device"] = device

    # Use the actual evaluator passed in
    return ExecutorFactory.build(executor_config, evaluator)


def _build_variant_strategy(contexts: list):
    """Build variant-aware strategy from contexts."""
    from strand.engine.strategies.variant_strategy import VariantStrategy

    return VariantStrategy(contexts=contexts)


def _build_variant_executor(executor_config: dict[str, Any], device: str, evaluator):
    """Build variant-aware executor.

    Args:
        executor_config: Executor configuration dict
        device: Device specification
        evaluator: The evaluator to use
    """
    from strand.engine.executors.variant_executor import VariantExecutor

    # Set device in config if not specified (for logging)
    if "device" not in executor_config:
        if device == "auto":
            try:
                import torch
                executor_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                executor_config["device"] = "cpu"
        else:
            executor_config["device"] = device

    # Use VariantExecutor which handles context-aware evaluation
    return VariantExecutor(
        evaluator=evaluator,
        batch_size=executor_config.get("batch_size", 64),
    )


def _build_engine_config(engine_config: dict[str, Any], *, total_contexts: int):
    """Build EngineConfig for variant triage."""
    from strand.engine import EngineConfig
    from strand.engine.runtime import BatchConfig, DeviceConfig

    # Build device config
    device_cfg = engine_config.get("device", {})
    device = None
    if device_cfg:
        device = DeviceConfig(
            target=device_cfg.get("target", "cpu"),
            mixed_precision=device_cfg.get("mixed_precision", "no"),
            gradient_accumulation_steps=device_cfg.get("gradient_accumulation_steps", 1),
        )

    # Build batch config
    batch_cfg = engine_config.get("batching", {})
    batching = None
    if batch_cfg:
        batching = BatchConfig(
            eval_size=batch_cfg.get("eval_size"),
            train_size=batch_cfg.get("train_size"),
            max_tokens=batch_cfg.get("max_tokens"),
        )

    configured_iterations = engine_config.get("iterations")
    if configured_iterations not in (None, 1):
        _LOGGER.info(
            "Variant triage enforces single-pass evaluation; overriding iterations=%s -> 1",
            configured_iterations,
        )

    configured_population = engine_config.get("population_size")
    if configured_population not in (None, total_contexts):
        _LOGGER.info(
            "Variant triage evaluates every context; overriding population_size=%s -> %s",
            configured_population,
            total_contexts,
        )

    return EngineConfig(
        iterations=1,
        population_size=max(1, total_contexts),
        seed=engine_config.get("seed", 1337),
        timeout_s=engine_config.get("timeout_s", 300.0),
        method=engine_config.get("method", "cmaes"),
        batching=batching,
        device=device,
    )


__all__ = ["run_variant_triage_pipeline"]


def _describe_variant(seq: Sequence) -> str:
    metadata = seq.metadata if isinstance(seq.metadata, Mapping) else {}
    ctx = metadata.get("_variant_context") if isinstance(metadata, Mapping) else None
    if ctx is None:
        return seq.id

    chrom = getattr(ctx.metadata, "chrom", "chr?")
    pos = getattr(ctx.metadata, "pos", "?")
    ref = getattr(ctx.metadata, "ref", "?")
    alt = getattr(ctx.metadata, "alt", "?")
    return f"{chrom}:{pos} {ref}->{alt}"
