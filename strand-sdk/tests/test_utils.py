import pytest

from strand.core.sequence import Sequence
from strand.utils import ensure_sequences, hamming_distance


def test_ensure_sequences_rejects_invalid_tokens():
    with pytest.raises(ValueError):
        ensure_sequences(["ABZ"])  # Z invalid for default alphabet


def test_hamming_distance_equal_sequences():
    assert hamming_distance("AAAA", "AAAA") == 0


def test_ensure_sequences_accepts_sequence_instance():
    seq = Sequence(id="seq", tokens="ACDE")
    result = ensure_sequences([seq])
    assert result[0] is seq
