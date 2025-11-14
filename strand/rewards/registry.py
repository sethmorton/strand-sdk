"""Reward registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from strand.rewards.custom import CustomReward
from strand.rewards.length_penalty import LengthPenaltyReward
from strand.rewards.novelty import NoveltyReward
from strand.rewards.solubility import SolubilityReward
from strand.rewards.stability import StabilityReward

Factory = Callable[..., object]


class RewardRegistry:
    _registry: dict[str, Factory] = {
        "stability": StabilityReward,
        "solubility": SolubilityReward,
        "novelty": NoveltyReward,
        "length_penalty": LengthPenaltyReward,
        "custom": CustomReward,
    }

    @classmethod
    def register(cls, name: str, factory: Factory) -> None:
        cls._registry[name] = factory

    @classmethod
    def create(cls, name: str, **kwargs: object) -> object:
        if name not in cls._registry:
            msg = f"Unknown reward block: {name}"
            raise KeyError(msg)
        return cls._registry[name](**kwargs)

    @classmethod
    def create_from_config(cls, config: dict[str, Any]) -> object:
        """Create reward block from config dict.

        Args:
            config: Dict with 'type' key and optional 'config' subdict.
                   Example: {"type": "gc_content", "config": {"target": 0.5}}

        Returns:
            Instantiated reward block.
        """
        if "type" not in config:
            msg = "Config must contain 'type' key"
            raise ValueError(msg)

        reward_type = config["type"]
        reward_config = config.get("config", {})

        return cls.create(reward_type, **reward_config)

    @classmethod
    def create_from_configs(cls, configs: list[dict[str, Any]]) -> list[object]:
        """Create multiple reward blocks from config list."""
        return [cls.create_from_config(config) for config in configs]

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry)


def register_advanced_blocks() -> None:
    """Register advanced reward blocks that require optional dependencies.

    This function should be called after importing advanced blocks to make
    them available in the registry. It handles ImportError gracefully.
    """
    try:
        from strand.rewards.virtual_cell_delta import VirtualCellDeltaReward
        RewardRegistry.register("virtual_cell_delta", VirtualCellDeltaReward)
    except ImportError:
        pass  # Optional dependency not installed

    try:
        from strand.rewards.motif_delta import MotifDeltaReward
        RewardRegistry.register("motif_delta", MotifDeltaReward)
    except ImportError:
        pass  # Optional dependency not installed

    try:
        from strand.rewards.conservation import ConservationReward
        RewardRegistry.register("conservation", ConservationReward)
    except ImportError:
        pass  # Optional dependency not installed
