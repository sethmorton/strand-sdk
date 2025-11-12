"""Tests for strategy capability declaration and resolution.

Validates that strategies properly declare their capabilities and that the engine
can correctly resolve and validate them.
"""

import pytest

from strand.engine.runtime import (
    StrategyCaps,
    resolve_strategy_caps,
    build_strategy_context,
    DeviceConfig,
    BatchConfig,
)
from strand.engine.strategies import (
    CEMStrategy,
    GAStrategy,
    CMAESStrategy,
    CMAESVarLenStrategy,
    RandomStrategy,
    RLPolicyStrategy,
    HybridStrategy,
)


class TestStrategyCapsDefaults:
    """Test that StrategyCaps has sensible defaults."""

    def test_default_caps(self) -> None:
        """Verify default caps are conservative (no runtime needed, no SFT)."""
        caps = StrategyCaps()

        assert caps.requires_runtime is False
        assert caps.supports_fine_tuning is False
        assert caps.needs_sft_dataset is False
        assert caps.kl_regularization == "none"
        assert caps.max_tokens_per_batch is None
        assert caps.prefers_autocast is True

    def test_kl_regularization_values(self) -> None:
        """Test valid kl_regularization values."""
        for level in ["none", "token", "sequence"]:
            caps = StrategyCaps(kl_regularization=level)
            assert caps.kl_regularization == level

    def test_caps_immutable(self) -> None:
        """Verify StrategyCaps is frozen (immutable)."""
        caps = StrategyCaps()
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            caps.requires_runtime = True  # type: ignore[misc]


class TestSimpleStrategyCapabilities:
    """Test capabilities of basic evolutionary strategies."""

    def test_random_strategy_caps(self) -> None:
        """RandomStrategy should not need runtime or SFT."""
        strategy = RandomStrategy(alphabet="ACGT", min_len=10, max_len=50, seed=42)
        caps = resolve_strategy_caps(strategy)

        assert caps.requires_runtime is False
        assert caps.supports_fine_tuning is False
        assert caps.needs_sft_dataset is False
        assert caps.kl_regularization == "none"
        assert caps.max_tokens_per_batch is None
        assert caps.prefers_autocast is True

    def test_cem_strategy_caps(self) -> None:
        """CEMStrategy should not need runtime or SFT."""
        strategy = CEMStrategy(alphabet="ACGT", min_len=10, max_len=50, seed=42)
        caps = resolve_strategy_caps(strategy)

        assert caps.requires_runtime is False
        assert caps.supports_fine_tuning is False
        assert caps.needs_sft_dataset is False
        assert caps.kl_regularization == "none"

    def test_ga_strategy_caps(self) -> None:
        """GAStrategy should not need runtime or SFT."""
        strategy = GAStrategy(alphabet="ACGT", min_len=10, max_len=50, seed=42)
        caps = resolve_strategy_caps(strategy)

        assert caps.requires_runtime is False
        assert caps.supports_fine_tuning is False
        assert caps.needs_sft_dataset is False
        assert caps.kl_regularization == "none"

    def test_cmaes_strategy_caps(self) -> None:
        """CMAESStrategy should not need runtime or SFT."""
        strategy = CMAESStrategy(alphabet="ACGT", min_len=10, max_len=50, seed=42)
        caps = resolve_strategy_caps(strategy)

        assert caps.requires_runtime is False
        assert caps.supports_fine_tuning is False
        assert caps.needs_sft_dataset is False
        assert caps.kl_regularization == "none"

    def test_cmaes_varlen_strategy_caps(self) -> None:
        """CMAESVarLenStrategy should not need runtime or SFT."""
        strategy = CMAESVarLenStrategy(alphabet="ACGT", min_len=10, max_len=50, seed=42)
        caps = resolve_strategy_caps(strategy)

        assert caps.requires_runtime is False
        assert caps.supports_fine_tuning is False
        assert caps.needs_sft_dataset is False
        assert caps.kl_regularization == "none"


class TestRLPolicyStrategyCapabilities:
    """Test capabilities of RL-based strategy."""

    def test_rl_policy_strategy_caps(self) -> None:
        """RLPolicyStrategy needs runtime, supports fine-tuning, and uses KL."""
        strategy = RLPolicyStrategy(alphabet="ACGT", min_len=10, max_len=50, seed=42)
        caps = resolve_strategy_caps(strategy)

        assert caps.requires_runtime is True
        assert caps.supports_fine_tuning is True
        assert caps.needs_sft_dataset is False  # Not required, but optional
        assert caps.kl_regularization == "token"
        assert caps.prefers_autocast is True

    def test_rl_policy_caps_immutable(self) -> None:
        """RLPolicyStrategy caps should be immutable ClassVar."""
        strategy1 = RLPolicyStrategy(alphabet="ACGT", min_len=10, max_len=50)
        strategy2 = RLPolicyStrategy(alphabet="ACGT", min_len=20, max_len=100)

        caps1 = resolve_strategy_caps(strategy1)
        caps2 = resolve_strategy_caps(strategy2)

        # Both should have the same caps (no per-instance modification)
        assert caps1 == caps2
        assert caps1.requires_runtime is True
        assert caps2.requires_runtime is True


class TestHybridStrategyCapabilities:
    """Test capability merging in hybrid strategies."""

    def test_hybrid_with_simple_strategies(self) -> None:
        """Hybrid of simple strategies should not require runtime."""
        strategies = [
            RandomStrategy(alphabet="ACGT", min_len=10, max_len=50, seed=42),
            CEMStrategy(alphabet="ACGT", min_len=10, max_len=50, seed=42),
        ]
        hybrid = HybridStrategy(strategies=strategies)
        caps = resolve_strategy_caps(hybrid)

        assert caps.requires_runtime is False
        assert caps.supports_fine_tuning is False
        assert caps.needs_sft_dataset is False
        assert caps.kl_regularization == "none"

    def test_hybrid_with_rl_strategy(self) -> None:
        """Hybrid including RL should inherit RL caps."""
        strategies = [
            RandomStrategy(alphabet="ACGT", min_len=10, max_len=50, seed=42),
            RLPolicyStrategy(alphabet="ACGT", min_len=10, max_len=50, seed=42),
        ]
        hybrid = HybridStrategy(strategies=strategies)
        caps = resolve_strategy_caps(hybrid)

        # Should inherit RL needs
        assert caps.requires_runtime is True
        assert caps.supports_fine_tuning is True
        assert caps.kl_regularization == "token"

    def test_hybrid_caps_merge_kl_regularization(self) -> None:
        """Test KL regularization level merging."""
        # Create a mock strategy class for testing different KL levels
        from dataclasses import dataclass
        from typing import ClassVar
        from strand.core.sequence import Sequence
        from strand.engine.interfaces import Strategy
        from strand.engine.types import Metrics
        from collections.abc import Mapping

        @dataclass
        class MockStrategy(Strategy):
            _CAPS: ClassVar[StrategyCaps]

            def ask(self, n: int) -> list[Sequence]:
                return []

            def tell(self, items: list) -> None:
                pass

            def best(self) -> None:
                return None

            def state(self) -> Mapping[str, object]:
                return {}

        # Strategy with "sequence" level KL
        MockStrategy._CAPS = StrategyCaps(kl_regularization="sequence")
        strategy_seq = MockStrategy()

        # Strategy with "token" level KL
        MockStrategy._CAPS = StrategyCaps(kl_regularization="token")
        strategy_token = MockStrategy()

        # Strategy with no KL
        MockStrategy._CAPS = StrategyCaps(kl_regularization="none")
        strategy_none = MockStrategy()

        # sequence + token -> sequence (highest priority)
        hybrid1 = HybridStrategy(strategies=[strategy_seq, strategy_token])
        assert resolve_strategy_caps(hybrid1).kl_regularization == "sequence"

        # token + none -> token
        hybrid2 = HybridStrategy(strategies=[strategy_token, strategy_none])
        assert resolve_strategy_caps(hybrid2).kl_regularization == "token"

        # none + none -> none
        hybrid3 = HybridStrategy(strategies=[strategy_none, strategy_none])
        assert resolve_strategy_caps(hybrid3).kl_regularization == "none"

    def test_hybrid_caps_merge_token_budget(self) -> None:
        """Test token budget merging (uses minimum)."""
        from dataclasses import dataclass
        from typing import ClassVar
        from strand.core.sequence import Sequence
        from strand.engine.interfaces import Strategy
        from strand.engine.types import Metrics
        from collections.abc import Mapping

        @dataclass
        class MockStrategy(Strategy):
            token_budget: int | None = None
            _CAPS: ClassVar[StrategyCaps]

            def ask(self, n: int) -> list[Sequence]:
                return []

            def tell(self, items: list) -> None:
                pass

            def best(self) -> None:
                return None

            def state(self) -> Mapping[str, object]:
                return {}

        # Strategy with 2048 tokens
        MockStrategy._CAPS = StrategyCaps(max_tokens_per_batch=2048)
        strategy_2048 = MockStrategy()

        # Strategy with 1024 tokens
        MockStrategy._CAPS = StrategyCaps(max_tokens_per_batch=1024)
        strategy_1024 = MockStrategy()

        # Strategy with no token limit
        MockStrategy._CAPS = StrategyCaps(max_tokens_per_batch=None)
        strategy_unlimited = MockStrategy()

        # 2048 + 1024 -> 1024 (minimum)
        hybrid = HybridStrategy(strategies=[strategy_2048, strategy_1024])
        assert resolve_strategy_caps(hybrid).max_tokens_per_batch == 1024

        # 1024 + unlimited -> 1024
        hybrid2 = HybridStrategy(strategies=[strategy_1024, strategy_unlimited])
        assert resolve_strategy_caps(hybrid2).max_tokens_per_batch == 1024

        # all unlimited -> None
        hybrid3 = HybridStrategy(strategies=[strategy_unlimited, strategy_unlimited])
        assert resolve_strategy_caps(hybrid3).max_tokens_per_batch is None

    def test_hybrid_caps_merge_autocast(self) -> None:
        """Test autocast preference merging."""
        from dataclasses import dataclass
        from typing import ClassVar
        from strand.core.sequence import Sequence
        from strand.engine.interfaces import Strategy
        from strand.engine.types import Metrics
        from collections.abc import Mapping

        @dataclass
        class MockStrategy(Strategy):
            _CAPS: ClassVar[StrategyCaps]

            def ask(self, n: int) -> list[Sequence]:
                return []

            def tell(self, items: list) -> None:
                pass

            def best(self) -> None:
                return None

            def state(self) -> Mapping[str, object]:
                return {}

        # Strategy that prefers autocast
        MockStrategy._CAPS = StrategyCaps(prefers_autocast=True)
        strategy_autocast = MockStrategy()

        # Strategy that does not prefer autocast
        MockStrategy._CAPS = StrategyCaps(prefers_autocast=False)
        strategy_no_autocast = MockStrategy()

        # True + True -> True
        hybrid1 = HybridStrategy(strategies=[strategy_autocast, strategy_autocast])
        assert resolve_strategy_caps(hybrid1).prefers_autocast is True

        # True + False -> False (any opt-out disables it)
        hybrid2 = HybridStrategy(strategies=[strategy_autocast, strategy_no_autocast])
        assert resolve_strategy_caps(hybrid2).prefers_autocast is False


class TestResolveStrategyCapabilities:
    """Test the resolve_strategy_caps function."""

    def test_resolve_from_method(self) -> None:
        """resolve_strategy_caps should work with strategy_caps() method."""
        strategy = RLPolicyStrategy(alphabet="ACGT", min_len=10, max_len=50)
        caps = resolve_strategy_caps(strategy)

        assert isinstance(caps, StrategyCaps)
        assert caps.requires_runtime is True

    def test_resolve_missing_caps_defaults(self) -> None:
        """Strategies without caps should get defaults."""
        # Create a minimal strategy without caps
        from strand.core.sequence import Sequence
        from strand.engine.interfaces import Strategy
        from strand.engine.types import Metrics
        from collections.abc import Mapping

        class MinimalStrategy(Strategy):
            def ask(self, n: int) -> list[Sequence]:
                return []

            def tell(self, items: list) -> None:
                pass

            def best(self) -> None:
                return None

            def state(self) -> Mapping[str, object]:
                return {}

        strategy = MinimalStrategy()
        caps = resolve_strategy_caps(strategy)

        # Should get all defaults
        assert caps == StrategyCaps()


class TestStrategyContextBuilding:
    """Test building StrategyContext with cap-aware configuration."""

    def test_build_context_without_runtime(self) -> None:
        """Build context for strategies that don't need runtime."""
        device = DeviceConfig(target="cpu")
        batch = BatchConfig(eval_size=32)

        context = build_strategy_context(
            device=device,
            batch=batch,
            require_runtime=False,
        )

        assert context.device == device
        assert context.batch == batch
        assert context.runtime is None

    def test_build_context_with_runtime(self) -> None:
        """Build context for strategies that need runtime."""
        device = DeviceConfig(target="cpu", mixed_precision="bf16")
        batch = BatchConfig(eval_size=32)

        context = build_strategy_context(
            device=device,
            batch=batch,
            require_runtime=True,
        )

        assert context.device == device
        assert context.batch == batch
        assert context.runtime is not None

    def test_build_context_defaults(self) -> None:
        """Build context with defaults when config is missing."""
        context = build_strategy_context(
            device=None,
            batch=None,
            require_runtime=False,
        )

        assert context.device is not None
        assert context.batch is not None
        assert context.runtime is None

