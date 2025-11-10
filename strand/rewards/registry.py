"""Reward registry."""

from __future__ import annotations

from collections.abc import Callable

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
    def create(cls, name: str, **kwargs):
        if name not in cls._registry:
            msg = f"Unknown reward block: {name}"
            raise KeyError(msg)
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry)
