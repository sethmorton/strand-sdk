"""Factory for building executors from configuration files.

Supports YAML/JSON config files using Hydra/OmegaConf for flexible executor
instantiation with runtime overrides.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from strand.engine.interfaces import Executor, Evaluator
from strand.engine.runtime import DeviceConfig, ModelRuntime

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
    """Base configuration for executors.

    Attributes
    ----------
    executor_type : str
        Type of executor: "local", "pool", "torch".
    evaluator_type : str
        Type of evaluator to use.
    num_workers : int | None
        Number of worker processes/threads.
    timeout_s : float | None
        Timeout per sequence (seconds).
    batch_size : int
        Batch size for evaluation.
    device : str | None
        Device specification (for torch executor).
    """

    executor_type: str = "local"
    evaluator_type: str = "mock"
    num_workers: int | None = None
    timeout_s: float | None = 60.0
    batch_size: int = 64
    device: str | None = None


class ExecutorFactory:
    """Factory for creating executors from configurations.

    Supports instantiation from:
    - YAML files
    - JSON files
    - Python dicts
    - OmegaConf DictConfig objects

    Example
    -------
    >>> config_dict = {
    ...     "executor_type": "local",
    ...     "num_workers": 4,
    ...     "batch_size": 32,
    ... }
    >>> executor = ExecutorFactory.build(config_dict, evaluator)
    """

    # Registry of available executor types
    _EXECUTORS: dict[str, type[Executor]] = {}

    @classmethod
    def register(cls, name: str, executor_class: type[Executor]) -> None:
        """Register an executor type.

        Parameters
        ----------
        name : str
            Executor name.
        executor_class : type[Executor]
            Executor class.
        """
        cls._EXECUTORS[name] = executor_class
        _LOGGER.info(f"Registered executor: {name}")

    @classmethod
    def build(
        cls,
        config: dict[str, Any] | ExecutorConfig,
        evaluator: Evaluator,
    ) -> Executor:
        """Build an executor from configuration.

        Parameters
        ----------
        config : dict | ExecutorConfig
            Executor configuration.
        evaluator : Evaluator
            Evaluator instance to wrap.

        Returns
        -------
        Executor
            Configured executor instance.

        Raises
        ------
        ValueError
            If executor type unknown or config invalid.
        """
        if isinstance(config, ExecutorConfig):
            cfg_dict = _dataclass_to_dict(config)
        elif isinstance(config, dict):
            cfg_dict = config
        else:
            raise ValueError(f"Invalid config type: {type(config)}")

        executor_type = cfg_dict.get("executor_type", "local").lower()

        if executor_type == "local":
            return cls._build_local_executor(cfg_dict, evaluator)
        elif executor_type == "pool":
            return cls._build_pool_executor(cfg_dict, evaluator)
        elif executor_type == "torch":
            return cls._build_torch_executor(cfg_dict, evaluator)
        else:
            raise ValueError(
                f"Unknown executor type: {executor_type}. "
                "Available: local, pool, torch"
            )

    @classmethod
    def _build_local_executor(
        cls, config: dict[str, Any], evaluator: Evaluator
    ) -> Executor:
        """Build LocalExecutor.

        Parameters
        ----------
        config : dict
            Configuration.
        evaluator : Evaluator
            Evaluator instance.

        Returns
        -------
        Executor
            LocalExecutor instance.
        """
        from strand.engine.executors.local import LocalExecutor

        return LocalExecutor(
            evaluator=evaluator,
            batch_size=config.get("batch_size", 64),
        )

    @classmethod
    def _build_pool_executor(
        cls, config: dict[str, Any], evaluator: Evaluator
    ) -> Executor:
        """Build PoolExecutor.

        Parameters
        ----------
        config : dict
            Configuration.
        evaluator : Evaluator
            Evaluator instance.

        Returns
        -------
        Executor
            PoolExecutor instance.
        """
        from strand.engine.executors.pool import LocalPoolExecutor

        return LocalPoolExecutor(
            evaluator=evaluator,
            mode=config.get("mode", "auto"),
            num_workers=config.get("num_workers", "auto"),
            batch_size=config.get("batch_size", 64),
        )

    @classmethod
    def _build_torch_executor(
        cls, config: dict[str, Any], evaluator: Evaluator
    ) -> Executor:
        """Build TorchExecutor.

        Parameters
        ----------
        config : dict
            Configuration.
        evaluator : Evaluator
            Evaluator instance.

        Returns
        -------
        Executor
            TorchExecutor instance.
        """
        from strand.engine.executors.torch import TorchExecutor

        device_cfg = DeviceConfig(
            target=config.get("device", "cpu"),
            mixed_precision=config.get("mixed_precision", "no"),
            gradient_accumulation_steps=config.get("grad_accum", 1),
        )
        runtime = ModelRuntime.build(device_cfg)

        return TorchExecutor(
            evaluator=evaluator,
            runtime=runtime,
            batch_size=config.get("batch_size", 64),
            max_tokens_per_batch=config.get("max_tokens"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path, evaluator: Evaluator) -> Executor:
        """Build executor from YAML file.

        Parameters
        ----------
        path : str | Path
            Path to YAML config file.
        evaluator : Evaluator
            Evaluator instance.

        Returns
        -------
        Executor
            Configured executor.

        Raises
        ------
        FileNotFoundError
            If config file not found.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("pyyaml required. Install with: pip install pyyaml")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            config = yaml.safe_load(f)

        _LOGGER.info(f"Loaded executor config from {path}")
        return cls.build(config, evaluator)

    @classmethod
    def from_json(cls, path: str | Path, evaluator: Evaluator) -> Executor:
        """Build executor from JSON file.

        Parameters
        ----------
        path : str | Path
            Path to JSON config file.
        evaluator : Evaluator
            Evaluator instance.

        Returns
        -------
        Executor
            Configured executor.

        Raises
        ------
        FileNotFoundError
            If config file not found.
        """
        import json

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            config = json.load(f)

        _LOGGER.info(f"Loaded executor config from {path}")
        return cls.build(config, evaluator)


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert dataclass to dictionary.

    Parameters
    ----------
    obj : Any
        Dataclass instance.

    Returns
    -------
    dict[str, Any]
        Dictionary representation.
    """
    from dataclasses import asdict

    return asdict(obj)


__all__ = [
    "ExecutorConfig",
    "ExecutorFactory",
]
