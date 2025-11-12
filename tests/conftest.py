"""Shared test fixtures and configuration for Strand SDK tests."""

import pytest

from strand.rewards import RewardBlock


@pytest.fixture
def basic_rewards():
    """Basic reward blocks for testing."""
    return [
        RewardBlock.stability(weight=1.0),
        RewardBlock.solubility(weight=0.5),
    ]


@pytest.fixture
def dna_alphabet():
    """DNA alphabet (ACGT)."""
    return "ACGT"


@pytest.fixture
def protein_alphabet():
    """Standard protein alphabet."""
    return "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture
def baseline_sequences():
    """Standard baseline sequences for testing."""
    return [
        "MKTAYIAKQRQISFVKSHFSRQDILDLQY",
        "MKPAYIAKQRQISFVKSHFSRQDILDVQY",
    ]

