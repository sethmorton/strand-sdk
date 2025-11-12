"""Tests for DualVariableManager and constraint adaptation."""

import pytest
import numpy as np

from strand.engine.constraints.dual import DualVariableManager, DualVariableSet


class TestDualVariableManager:
    """Test DualVariableManager."""

    def test_initialization(self) -> None:
        """Test default initialization."""
        manager = DualVariableManager(init_weight=1.0)
        assert manager.current_weight == 1.0
        assert len(manager.violation_history) == 0
        assert len(manager.weight_history) == 0

    def test_update_increases_weight_on_violation(self) -> None:
        """Weight increases when constraint is violated."""
        manager = DualVariableManager(
            init_weight=1.0,
            adaptive_step=0.1,
            target_violation=0.0,
        )

        # Violation > target (0.0)
        new_weight = manager.update(violation=0.5)

        # Weight should increase
        assert new_weight > 1.0
        assert new_weight == manager.current_weight

    def test_update_decreases_weight_on_satisfaction(self) -> None:
        """Weight decreases when constraint is satisfied."""
        manager = DualVariableManager(
            init_weight=2.0,
            adaptive_step=0.1,
            target_violation=0.0,
        )

        # Violation < target (0.0) - constraint satisfied
        new_weight = manager.update(violation=-0.1)

        # Weight should decrease (but more slowly)
        assert new_weight < 2.0
        assert new_weight == manager.current_weight

    def test_weight_bounds(self) -> None:
        """Weights are clamped to bounds."""
        manager = DualVariableManager(
            init_weight=1.0,
            min_weight=0.1,
            max_weight=10.0,
            adaptive_step=0.5,
        )

        # Update many times to exceed bounds
        for _ in range(100):
            manager.update(violation=1.0)

        # Should never exceed max
        assert manager.current_weight <= 10.0

    def test_history_tracking(self) -> None:
        """Violation and weight histories tracked."""
        manager = DualVariableManager()

        violations = [0.1, 0.2, -0.1]
        for violation in violations:
            manager.update(violation=violation)

        assert manager.violation_history == violations
        assert len(manager.weight_history) == len(violations)

    def test_reset(self) -> None:
        """Reset restores initial state."""
        manager = DualVariableManager(init_weight=1.0)

        # Update several times
        manager.update(0.5)
        manager.update(0.3)
        manager.update(0.1)

        assert len(manager.violation_history) == 3

        # Reset
        manager.reset()

        assert manager.current_weight == 1.0
        assert len(manager.violation_history) == 0
        assert len(manager.weight_history) == 0

    def test_summary_statistics(self) -> None:
        """Summary provides correct statistics."""
        manager = DualVariableManager(init_weight=1.0)

        violations = [0.5, 0.3, 0.1, 0.2]
        for v in violations:
            manager.update(v)

        summary = manager.summary()

        assert summary["num_updates"] == 4
        assert abs(summary["mean_violation"] - np.mean(violations)) < 1e-6
        assert summary["max_violation"] == max(violations)
        assert summary["min_violation"] == min(violations)

    def test_zero_updates_summary(self) -> None:
        """Summary with zero updates."""
        manager = DualVariableManager(init_weight=1.0)

        summary = manager.summary()

        assert summary["num_updates"] == 0
        assert summary["current_weight"] == 1.0


class TestDualVariableSet:
    """Test DualVariableSet."""

    def test_add_constraint(self) -> None:
        """Add constraints to set."""
        dual_set = DualVariableSet()

        mgr1 = dual_set.add_constraint("c1", init_weight=1.0)
        mgr2 = dual_set.add_constraint("c2", init_weight=2.0)

        assert "c1" in dual_set.managers
        assert "c2" in dual_set.managers
        assert mgr1.current_weight == 1.0
        assert mgr2.current_weight == 2.0

    def test_update_all(self) -> None:
        """Update all constraints."""
        dual_set = DualVariableSet()
        dual_set.add_constraint("c1", init_weight=1.0)
        dual_set.add_constraint("c2", init_weight=1.0)

        violations = {"c1": 0.5, "c2": -0.1}
        weights = dual_set.update_all(violations)

        assert "c1" in weights
        assert "c2" in weights
        # c1 violated -> weight increased
        assert weights["c1"] > 1.0
        # c2 satisfied -> weight decreased
        assert weights["c2"] < 1.0

    def test_get_weights(self) -> None:
        """Get current weights."""
        dual_set = DualVariableSet()
        dual_set.add_constraint("c1", init_weight=1.0)
        dual_set.add_constraint("c2", init_weight=2.0)

        weights = dual_set.get_weights()

        assert weights["c1"] == 1.0
        assert weights["c2"] == 2.0

    def test_reset_all(self) -> None:
        """Reset all constraints."""
        dual_set = DualVariableSet()
        dual_set.add_constraint("c1", init_weight=1.0)
        dual_set.add_constraint("c2", init_weight=2.0)

        # Update
        dual_set.update_all({"c1": 0.5, "c2": 0.5})

        # Reset
        dual_set.reset_all()

        weights = dual_set.get_weights()
        assert weights["c1"] == 1.0
        assert weights["c2"] == 2.0

    def test_summary(self) -> None:
        """Get summary for all constraints."""
        dual_set = DualVariableSet()
        dual_set.add_constraint("c1", init_weight=1.0)
        dual_set.add_constraint("c2", init_weight=1.0)

        dual_set.update_all({"c1": 0.5, "c2": -0.1})
        dual_set.update_all({"c1": 0.3, "c2": 0.0})

        summary = dual_set.summary()

        assert "c1" in summary
        assert "c2" in summary
        assert summary["c1"]["num_updates"] == 2
        assert summary["c2"]["num_updates"] == 2


class TestDualVariableAdaptation:
    """Test adaptive constraint handling."""

    def test_increasing_violations_accelerates_weight_growth(self) -> None:
        """Persistent violations increase weight faster."""
        manager = DualVariableManager(
            init_weight=1.0,
            adaptive_step=0.1,
        )

        # 5 consistent violations
        weights = []
        for _ in range(5):
            w = manager.update(1.0)
            weights.append(w)

        # Weights should increase monotonically
        for i in range(len(weights) - 1):
            assert weights[i + 1] > weights[i]

    def test_improving_violations_stabilizes_weight(self) -> None:
        """Improving violations stabilize weight."""
        manager = DualVariableManager(
            init_weight=10.0,
            adaptive_step=0.1,
        )

        # Start violated, then improve
        violations = [1.0, 0.5, 0.1, 0.0, -0.1, -0.1, -0.1]
        weights = []

        for v in violations:
            w = manager.update(v)
            weights.append(w)

        # Weight should decrease after improving
        improvement_idx = 4  # Where violation becomes negative
        assert weights[-1] < weights[improvement_idx]

    def test_target_violation_level(self) -> None:
        """Target violation enables soft constraints."""
        manager = DualVariableManager(
            init_weight=1.0,
            adaptive_step=0.1,
            target_violation=0.1,  # Allow up to 0.1 violation
        )

        # Small violation (within target)
        w1 = manager.update(0.05)
        assert w1 < 1.05  # Weight should decrease slightly

        # Large violation (beyond target)
        manager.reset()
        w2 = manager.update(0.5)
        assert w2 > 1.0  # Weight should increase


class TestDualVariableIntegration:
    """Integration tests for dual variable system."""

    def test_constraint_adaptation_loop(self) -> None:
        """Simulate constraint adaptation over RL iterations."""
        dual_set = DualVariableSet()
        dual_set.add_constraint("gc_content", init_weight=1.0, adaptive_step=0.1)
        dual_set.add_constraint("tfbs_divergence", init_weight=1.0, adaptive_step=0.1)

        # Simulate 10 iterations
        violations_sequence = [
            {"gc_content": 0.3, "tfbs_divergence": 0.2},
            {"gc_content": 0.2, "tfbs_divergence": 0.15},
            {"gc_content": 0.15, "tfbs_divergence": 0.1},
            {"gc_content": 0.1, "tfbs_divergence": 0.05},
            {"gc_content": 0.05, "tfbs_divergence": 0.0},
            {"gc_content": 0.0, "tfbs_divergence": -0.05},
            {"gc_content": -0.05, "tfbs_divergence": -0.1},
            {"gc_content": 0.02, "tfbs_divergence": 0.01},
            {"gc_content": 0.01, "tfbs_divergence": 0.0},
            {"gc_content": 0.0, "tfbs_divergence": 0.0},
        ]

        for violations in violations_sequence:
            dual_set.update_all(violations)

        # Check convergence
        final_weights = dual_set.get_weights()
        assert final_weights["gc_content"] < 1.1  # Close to initial
        assert final_weights["tfbs_divergence"] < 1.1

        # Check summaries
        summary = dual_set.summary()
        assert summary["gc_content"]["num_updates"] == 10
        assert summary["tfbs_divergence"]["mean_violation"] < 0.05  # Improved

    def test_multiple_constraints_independent_adaptation(self) -> None:
        """Each constraint adapts independently."""
        dual_set = DualVariableSet()
        dual_set.add_constraint("c1", init_weight=1.0)
        dual_set.add_constraint("c2", init_weight=1.0)

        # c1 always violated, c2 always satisfied
        for _ in range(5):
            dual_set.update_all({"c1": 0.5, "c2": -0.5})

        weights = dual_set.get_weights()

        # c1 weight should grow
        assert weights["c1"] > 1.0
        # c2 weight should shrink
        assert weights["c2"] < 1.0
        # They should be different
        assert weights["c1"] != weights["c2"]

