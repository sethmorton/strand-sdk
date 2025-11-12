"""Strategy registry surface.

Provides a small factory to obtain a Strategy by name.
"""

from __future__ import annotations

from typing import Final

from strand.engine.interfaces import Strategy
from strand.engine.strategies.cem import CEMStrategy
from strand.engine.strategies.cmaes import CMAESStrategy
from strand.engine.strategies.cmaes_varlen import CMAESVarLenStrategy
from strand.engine.strategies.ga import GAStrategy
from strand.engine.strategies.hybrid import HybridStrategy
from strand.engine.strategies.random import RandomStrategy
from strand.engine.strategies.rl_policy import RLPolicyStrategy

_REGISTRY: Final[dict[str, type[Strategy]]] = {
    "random": RandomStrategy,
    "cem": CEMStrategy,
    "ga": GAStrategy,
    "cmaes": CMAESStrategy,
    "cmaes-varlen": CMAESVarLenStrategy,
    "hybrid": HybridStrategy,
    "rl-policy": RLPolicyStrategy,
}


def strategy_from_name(name: str, **params: object) -> Strategy:
    """Return a Strategy instance from the registry.

    Raises KeyError for unknown strategies.

    Parameters
    ----------
    name : str
        Strategy name: "random", "cem", "ga", "cmaes", "cmaes-varlen", "hybrid", "rl-policy"
    **params : object
        Constructor parameters (e.g., alphabet, min_len, max_len, seed)

    Returns
    -------
    Strategy
        Configured strategy instance
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown strategy: {name}. Available: {list(_REGISTRY.keys())}")
    cls = _REGISTRY[key]
    return cls(**params)  # type: ignore[call-arg]


__all__ = [
    "strategy_from_name",
    "RandomStrategy",
    "CEMStrategy",
    "GAStrategy",
    "CMAESStrategy",
    "CMAESVarLenStrategy",
    "HybridStrategy",
    "RLPolicyStrategy",
]

