"""Shared helpers for strategies that use device-aware runtimes.

This module provides utilities to standardize how strategies prepare modules,
handle mixed-precision, manage checkpoints, and interact with accelerators.

Strategies using these helpers can focus on their core algorithm while delegating
device-specific concerns to the adapter.
"""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn as nn
from torch.optim import Optimizer

from strand.engine.runtime import ModelRuntime

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModulePreparationResult:
    """Result of preparing a module and optional optimizer.

    Attributes
    ----------
    module : nn.Module
        The prepared (possibly distributed/wrapped) module.
    optimizer : Optimizer | None
        The prepared optimizer, if provided; None otherwise.
    """

    module: nn.Module
    optimizer: Optimizer | None


class StrategyRuntimeAdapter:
    """Unified adapter for strategies using ModelRuntime.

    Provides methods for:
    - Device placement and mixed-precision preparation
    - Autocast context management
    - Checkpoint save/load with runtime awareness
    - Synchronization and backward pass utilities

    This adapter is designed to be used by strategies that declare
    ``requires_runtime=True`` in their ``StrategyCaps``.

    Example
    -------
    >>> from strand.engine.runtime import ModelRuntime, DeviceConfig
    >>> runtime = ModelRuntime.build(DeviceConfig(target="cuda"))
    >>> adapter = StrategyRuntimeAdapter(runtime)
    >>> policy = MyPolicyNet()
    >>> optimizer = torch.optim.Adam(policy.parameters())
    >>> result = adapter.prepare_module(policy, optimizer)
    >>> # Now use result.module and result.optimizer in training loop
    """

    _DEFAULT_CHECKPOINT_SUFFIX: ClassVar[str] = ".checkpoint.pt"

    def __init__(self, runtime: ModelRuntime) -> None:
        """Initialize adapter with a runtime."""
        self._runtime = runtime

    @property
    def runtime(self) -> ModelRuntime:
        """Access the underlying ModelRuntime."""
        return self._runtime

    def prepare_module(
        self, module: nn.Module, optimizer: Optimizer | None = None
    ) -> ModulePreparationResult:
        """Prepare module and optimizer for distributed/mixed-precision training.

        This method delegates to the underlying runtime, ensuring consistent
        device placement and optimizer state management.

        Parameters
        ----------
        module : nn.Module
            Model to prepare.
        optimizer : Optimizer | None
            Optional optimizer. If provided, will be prepared alongside the module.

        Returns
        -------
        ModulePreparationResult
            Contains prepared module and optimizer.

        Notes
        -----
        The runtime may wrap the module (e.g., via DistributedDataParallel on
        multi-GPU) or apply other transformations. Callers should use the
        returned module for all forward/backward passes.
        """
        prepared_module, prepared_optimizer = self._runtime.prepare_module(module, optimizer)
        return ModulePreparationResult(module=prepared_module, optimizer=prepared_optimizer)

    def autocast_context(self) -> AbstractContextManager[None]:
        """Get autocast context if supported by the runtime.

        Returns
        -------
        AbstractContextManager[None]
            Autocast context (or nullcontext if not using mixed-precision).

        Example
        -------
        >>> with adapter.autocast_context():
        ...     logits = model(x)
        ...     loss = criterion(logits, y)
        """
        return self._runtime.autocast()

    def backward(self, loss: torch.Tensor) -> None:
        """Perform backward pass with runtime synchronization.

        Handles gradient accumulation, scaling, and synchronization as needed
        by the underlying accelerator.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar loss tensor to backpropagate.
        """
        self._runtime.backward(loss)

    def wait(self) -> None:
        """Synchronize across devices.

        In multi-GPU or distributed settings, ensures all devices wait for
        completion. On single-device, typically a no-op.
        """
        self._runtime.wait()

    def save_checkpoint(
        self,
        module: nn.Module,
        optimizer: Optimizer | None,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save module and optimizer state with optional metadata.

        Saves unwrapped state dicts, suitable for loading in different
        distributed configurations or on CPU.

        Parameters
        ----------
        module : nn.Module
            Module to checkpoint (may be wrapped by accelerator).
        optimizer : Optimizer | None
            Optional optimizer state to save.
        path : str | Path
            Path to save checkpoint.
        metadata : dict[str, Any] | None
            Optional dict of metadata (e.g., epoch, step, loss).

        Raises
        ------
        IOError
            If save fails.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Unwrap module if necessary (accelerate wraps with DataParallel, etc.)
        if hasattr(module, "module"):
            unwrapped_module = module.module
        else:
            unwrapped_module = module

        checkpoint = {"model_state": unwrapped_module.state_dict()}
        if optimizer is not None:
            checkpoint["optimizer_state"] = optimizer.state_dict()
        if metadata is not None:
            checkpoint["metadata"] = metadata

        torch.save(checkpoint, path)
        _LOGGER.info(f"Checkpoint saved to {path}")

    def load_checkpoint(
        self,
        module: nn.Module,
        path: str | Path,
        optimizer: Optimizer | None = None,
        strict: bool = True,
    ) -> dict[str, Any] | None:
        """Load module and optimizer state from checkpoint.

        Loads state dicts into the provided module and optimizer, handling
        wrapped modules appropriately.

        Parameters
        ----------
        module : nn.Module
            Module to load state into (may be wrapped).
        path : str | Path
            Path to checkpoint.
        optimizer : Optimizer | None
            Optional optimizer to restore.
        strict : bool
            If False, allow missing or extra keys in state dict.

        Returns
        -------
        dict[str, Any] | None
            Metadata from checkpoint, if present.

        Raises
        ------
        FileNotFoundError
            If checkpoint path doesn't exist.
        RuntimeError
            If state dict loading fails (e.g., shape mismatch).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self._runtime.device)

        # Unwrap module if necessary
        if hasattr(module, "module"):
            target_module = module.module
        else:
            target_module = module

        target_module.load_state_dict(checkpoint["model_state"], strict=strict)
        _LOGGER.info(f"Model state loaded from {path}")

        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            _LOGGER.info("Optimizer state loaded")

        metadata = checkpoint.get("metadata")
        if metadata is not None:
            _LOGGER.info(f"Loaded metadata: {metadata}")
        return metadata

    def get_device(self) -> torch.device:
        """Get the device where computations should happen.

        Returns
        -------
        torch.device
            Device from the runtime (cuda:X or cpu).
        """
        return self._runtime.device

    def get_gradient_accumulation_steps(self) -> int:
        """Get gradient accumulation steps from the runtime.

        Returns
        -------
        int
            Number of steps to accumulate gradients before stepping optimizer.
        """
        # Access via accelerator attribute
        return self._runtime.accelerator.gradient_accumulation_steps


__all__ = [
    "ModulePreparationResult",
    "StrategyRuntimeAdapter",
]

