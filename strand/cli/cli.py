"""Strand command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

import torch

from strand.engine import Engine, EngineConfig
from strand.engine.constraints import BoundedConstraint, Direction
from strand.engine.constraints.dual import DualVariableSet
from strand.engine.executors.factory import ExecutorFactory
from strand.engine.rules import Rules
from strand.engine.score import default_score
from strand.engine.strategies import strategy_from_name
from strand.engine.engine import SFTConfig
from strand.engine.runtime import BatchConfig, DeviceConfig
from strand.evaluators.reward_aggregator import RewardAggregator
from strand.rewards import RewardBlock

from strand.data.sequence_dataset import SequenceDataset, SequenceDatasetConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strand SDK CLI")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run an experiment from a config file")
    run_parser.add_argument("config", help="Path to YAML/JSON config file")

    # Variant triage subcommand
    vt_parser = subparsers.add_parser(
        "run-variant-triage",
        help="Run variant triage optimization from config"
    )
    vt_parser.add_argument("config", help="Path to variant triage YAML config file")
    vt_parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device for computation (default: auto)"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _run_from_config(Path(args.config))
    elif args.command == "run-variant-triage":
        return _run_variant_triage(Path(args.config), args.device)

    parser.print_help()
    return 0


def _run_from_config(path: Path) -> int:
    data = _load_config(path)

    strategy = _build_strategy(data.get("strategy", {}))
    reward_blocks = _build_reward_blocks(data.get("rewards", []))
    if not reward_blocks:
        raise ValueError("At least one reward block must be specified in config")
    evaluator = RewardAggregator(reward_blocks)
    executor = ExecutorFactory.build(data.get("executor", {}), evaluator)

    constraints = _build_constraints(data.get("constraints", []))
    rules = _build_rules(data.get("rules"))
    device = _build_device(data.get("device"))
    batching = _build_batching(data.get("batching"))
    engine_cfg = _build_engine_config(data.get("engine", {}), device, batching, strategy_name=data.get("strategy", {}).get("name", ""))
    sft_cfg = _build_sft_config(data.get("sft"))
    dual_manager = _build_dual_manager(data.get("dual"))

    engine = Engine(
        config=engine_cfg,
        strategy=strategy,
        evaluator=evaluator,
        executor=executor,
        score_fn=default_score,
        constraints=constraints,
        rules=rules,
        sft=sft_cfg,
        dual_manager=dual_manager,
    )

    results = engine.run()
    if results.best:
        seq, score = results.best
        print(f"Best sequence ({score:.4f}): {seq.tokens}")
    else:
        print("No valid sequences produced.")
    print(json.dumps(results.summary, indent=2))
    return 0


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() in {".json"}:
        return json.loads(path.read_text())
    if yaml is None:
        raise RuntimeError("pyyaml is required for YAML configs. Install with `pip install pyyaml`. ")
    return yaml.safe_load(path.read_text())


def _build_strategy(cfg: dict[str, Any]) -> Any:
    name = cfg.get("name")
    if not name:
        raise ValueError("strategy.name is required")
    params = cfg.get("params", {})
    return strategy_from_name(name, **params)


def _build_reward_blocks(cfg: list[dict[str, Any]]) -> list[Any]:
    blocks = []
    for entry in cfg:
        block_name = entry.get("name")
        params = entry.get("params", {})
        if not block_name:
            raise ValueError("Reward block entries require a name")
        blocks.append(RewardBlock.from_registry(block_name, **params))
    return blocks


def _build_constraints(cfg: list[dict[str, Any]]) -> list[BoundedConstraint]:
    constraints: list[BoundedConstraint] = []
    for entry in cfg:
        direction = Direction[entry.get("direction", "LE").upper()]
        constraints.append(
            BoundedConstraint(
                name=entry["name"],
                direction=direction,
                bound=float(entry["bound"]),
            )
        )
    return constraints


def _build_rules(cfg: dict[str, Any] | None) -> Rules:
    if not cfg:
        return Rules()
    return Rules(
        init=cfg.get("init", {}),
        step_size=cfg.get("step_size", 0.05),
        clip=tuple(cfg.get("clip", (0.0, 10.0))),
    )


def _build_device(cfg: dict[str, Any] | None) -> DeviceConfig | None:
    if not cfg:
        return None
    return DeviceConfig(
        target=cfg.get("target", "cpu"),
        mixed_precision=cfg.get("mixed_precision", "no"),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
    )


def _build_batching(cfg: dict[str, Any] | None) -> BatchConfig | None:
    if not cfg:
        return None
    return BatchConfig(
        eval_size=cfg.get("eval_size"),
        train_size=cfg.get("train_size"),
        max_tokens=cfg.get("max_tokens"),
    )


def _build_engine_config(
    cfg: dict[str, Any],
    device: DeviceConfig | None,
    batching: BatchConfig | None,
    *,
    strategy_name: str,
) -> EngineConfig:
    return EngineConfig(
        iterations=cfg.get("iterations", 100),
        population_size=cfg.get("population_size", 256),
        seed=cfg.get("seed", 1337),
        timeout_s=cfg.get("timeout_s", 60.0),
        early_stop_patience=cfg.get("early_stop_patience"),
        max_evals=cfg.get("max_evals"),
        method=cfg.get("method", strategy_name or "unknown"),
        batching=batching,
        device=device,
    )


def _build_sft_config(cfg: dict[str, Any] | None) -> SFTConfig | None:
    if not cfg:
        return None
    dataset_path = cfg.get("dataset_path")
    if not dataset_path:
        raise ValueError("sft.dataset_path is required")
    tokenizer = _SimpleDNATokenizer()
    dataset_cfg = SequenceDatasetConfig(
        data_path=dataset_path,
        tokenizer=tokenizer,
        max_seq_len=cfg.get("max_seq_len"),
        min_seq_len=cfg.get("min_seq_len"),
        validation_split=cfg.get("validation_split", 0.1),
    )
    dataset = SequenceDataset(dataset_cfg)
    return SFTConfig(
        dataset=dataset,
        epochs=cfg.get("epochs", 1),
        batch_size=cfg.get("batch_size"),
    )


def _build_dual_manager(cfg: dict[str, Any] | None) -> DualVariableSet | None:
    if not cfg:
        return None
    manager = DualVariableSet()
    for entry in cfg.get("constraints", []):
        manager.add_constraint(
            name=entry["name"],
            init_weight=entry.get("init_weight", 1.0),
            adaptive_step=entry.get("adaptive_step", 0.1),
            target_violation=entry.get("target_violation", 0.0),
            min_weight=entry.get("min_weight", 0.01),
            max_weight=entry.get("max_weight", 100.0),
        )
    return manager


def _run_variant_triage(config_path: Path, device: str) -> int:
    """Run variant triage optimization from config."""
    try:
        from strand.cli.commands.run_variant_triage import run_variant_triage_pipeline
    except ImportError as e:
        raise RuntimeError(
            f"Variant triage command requires additional dependencies. "
            f"Install with: pip install strand-sdk[variant-triage]. Error: {e}"
        )

    run_variant_triage_pipeline(config_path, device)
    return 0


class _SimpleDNATokenizer:
    """Minimal tokenizer for CLI-driven SFT datasets."""

    def __init__(self, alphabet: str = "ACGTN") -> None:
        self._alphabet = alphabet
        self._index = {ch: idx for idx, ch in enumerate(alphabet)}
        if "N" not in self._index:
            self._index["N"] = len(self._index)

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str = "pt",
        max_length: int | None = None,
        truncation: bool = True,
        padding: str = "max_length",
    ) -> dict[str, torch.Tensor]:
        text = text.upper()
        tokens = [self._index.get(ch, self._index["N"]) for ch in text]
        if max_length is not None and truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        if max_length is not None and padding == "max_length" and len(tokens) < max_length:
            tokens = tokens + [self._index["N"]] * (max_length - len(tokens))
        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
