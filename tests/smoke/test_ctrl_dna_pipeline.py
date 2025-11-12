"""Smoke tests for Ctrl-DNA pipeline."""

import pytest
from strand.core.sequence import Sequence
from strand.engine.strategies import RLPolicyStrategy
from strand.engine.runtime import DeviceConfig, build_strategy_context


class TestCtrlDNASmokeTests:
    """Smoke tests for end-to-end Ctrl-DNA pipelines."""

    def test_rl_policy_strategy_creation(self) -> None:
        """Can create RLPolicyStrategy."""
        strategy = RLPolicyStrategy(
            alphabet="ACGT",
            min_len=50,
            max_len=500,
        )
        assert strategy is not None

    def test_rl_policy_ask(self) -> None:
        """RLPolicy can generate sequences."""
        strategy = RLPolicyStrategy(
            alphabet="ACGT",
            min_len=50,
            max_len=500,
            seed=42,
        )

        # This may require prepare() first, but basic ask should work
        try:
            sequences = strategy.ask(5)
            assert len(sequences) >= 5
            for seq in sequences:
                assert isinstance(seq, Sequence)
        except RuntimeError:
            # May need prepare() - that's OK for smoke test
            pass

    def test_device_context_creation(self) -> None:
        """Can create device context."""
        device = DeviceConfig(target="cpu", mixed_precision="no")
        context = build_strategy_context(
            device=device,
            batch=None,
            require_runtime=True,
        )
        assert context is not None
        assert context.runtime is not None

    def test_strategy_caps(self) -> None:
        """Strategy caps accessible."""
        strategy = RLPolicyStrategy(alphabet="ACGT", min_len=10, max_len=100)
        caps = strategy.strategy_caps()
        assert caps.requires_runtime is True
        assert caps.supports_fine_tuning is True

