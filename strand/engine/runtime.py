"""Device/runtime helpers for accelerator-aware strategies."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Literal as LiteralType

import torch
from accelerate import Accelerator
from torch.optim import Optimizer

Precision = LiteralType["no", "fp16", "bf16"]


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    """Execution device preferences for strategies and executors."""

    target: str = "cpu"
    mixed_precision: Precision = "no"
    gradient_accumulation_steps: int = 1

    def torch_device(self) -> torch.device:
        return torch.device(self.target)


@dataclass(frozen=True, slots=True)
class BatchConfig:
    """Batch-size hints for evaluation and training phases."""

    eval_size: int | None = None
    train_size: int | None = None
    max_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class StrategyCaps:
    """Strategy capabilities - declare what your strategy needs/supports.

    Start simple: by default, strategies need nothing special.
    Add capabilities only as needed for advanced features.

    Examples
    --------
    Simple strategy (no neural networks):
    >>> caps = StrategyCaps()  # All defaults, works everywhere

    Neural network strategy with GPU:
    >>> caps = StrategyCaps(requires_runtime=True)  # Gets device management

    RL strategy with supervision:
    >>> caps = StrategyCaps(
    ...     requires_runtime=True,
    ...     supports_fine_tuning=True,
    ...     kl_regularization="token",
    ... )
    """

    requires_runtime: bool = False
    """If True, strategy needs device management (GPU, mixed-precision).
    Default: False. Set True if using torch modules."""

    supports_fine_tuning: bool = False
    """If True, strategy implements warm_start() for SFT pre-training.
    Default: False. Set True for RL strategies."""

    needs_sft_dataset: bool = False
    """If True, strategy requires supervised data to function well.
    Default: False. Engine warns if SFT data not provided."""

    kl_regularization: LiteralType["none", "token", "sequence"] = "none"
    """KL divergence regularization level.
    Default: "none" (no KL penalty).
    "token": Per-token KL. "sequence": Sequence-level KL."""

    max_tokens_per_batch: int | None = None
    """Token budget per batch (for very large models).
    Default: None (no limit). E.g., 2048 for context-limited models."""

    prefers_autocast: bool = True
    """If True, strategy prefers mixed-precision training.
    Default: True. Set False if strategy has numerical stability issues."""


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Context shared with strategies that opt into advanced workflows."""

    device: DeviceConfig
    batch: BatchConfig
    runtime: ModelRuntime | None


class ModelRuntime:
    """Thin wrapper around :class:`accelerate.Accelerator`."""

    def __init__(self, accelerator: Accelerator, autocast_dtype: torch.dtype | None) -> None:
        self._accelerator = accelerator
        self._autocast_dtype = autocast_dtype

    @classmethod
    def build(cls, device: DeviceConfig) -> ModelRuntime:
        accelerator = Accelerator(
            cpu=device.target.startswith("cpu"),
            mixed_precision=device.mixed_precision,
            gradient_accumulation_steps=device.gradient_accumulation_steps,
        )
        autocast_dtype = _dtype_from_precision(device.mixed_precision)
        return cls(accelerator=accelerator, autocast_dtype=autocast_dtype)

    @property
    def accelerator(self) -> Accelerator:
        return self._accelerator

    @property
    def device(self) -> torch.device:
        return self._accelerator.device

    def autocast(self) -> AbstractContextManager[None]:
        if self._autocast_dtype is None:
            return nullcontext()
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        return torch.autocast(device_type=device_type, dtype=self._autocast_dtype)

    def prepare(self, *objects: object) -> tuple[object, ...]:
        """Forward to :meth:`Accelerator.prepare` and normalize tuple output."""

        prepared = self._accelerator.prepare(*objects)
        if not isinstance(prepared, tuple):
            return (prepared,)
        return prepared

    def backward(self, loss: torch.Tensor) -> None:
        self._accelerator.backward(loss)

    def wait(self) -> None:
        self._accelerator.wait_for_everyone()

    def prepare_module(
        self, module: torch.nn.Module, optimizer: Optimizer | None = None
    ) -> tuple[torch.nn.Module, Optimizer | None]:
        if optimizer is None:
            prepared_module = self._accelerator.prepare(module)
            return prepared_module, None
        prepared_module, prepared_optimizer = self._accelerator.prepare(module, optimizer)
        return prepared_module, prepared_optimizer


def _dtype_from_precision(precision: Precision | None) -> torch.dtype | None:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


DEFAULT_DEVICE_CONFIG = DeviceConfig()
DEFAULT_BATCH_CONFIG = BatchConfig()
DEFAULT_STRATEGY_CAPS = StrategyCaps()


def resolve_strategy_caps(strategy: object) -> StrategyCaps:
    """Return declared strategy capabilities, defaulting to no extras."""

    caps_attr = getattr(strategy, "strategy_caps", None)
    if callable(caps_attr):  # method returning StrategyCaps
        caps = caps_attr()
        if isinstance(caps, StrategyCaps):
            return caps
    elif isinstance(caps_attr, StrategyCaps):
        return caps_attr
    return DEFAULT_STRATEGY_CAPS


def build_strategy_context(
    *,
    device: DeviceConfig | None,
    batch: BatchConfig | None,
    require_runtime: bool,
) -> StrategyContext:
    device_cfg = device or DEFAULT_DEVICE_CONFIG
    batch_cfg = batch or DEFAULT_BATCH_CONFIG
    runtime = ModelRuntime.build(device_cfg) if require_runtime else None
    return StrategyContext(device=device_cfg, batch=batch_cfg, runtime=runtime)


__all__ = [
    "BatchConfig",
    "DeviceConfig",
    "ModelRuntime",
    "StrategyCaps",
    "StrategyContext",
    "build_strategy_context",
    "resolve_strategy_caps",
]
