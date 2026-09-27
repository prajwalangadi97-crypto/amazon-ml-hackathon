"""Unit tests for the modular retrieval and indexing pipeline."""

import pytest
from src.blocking.retrieval import (
    extract_compound_number_keys,
    extract_record_blocking_keys,
    rank_candidates_channel_consensus_first,
    apply_selective_residual_recovery,
)


def test_extract_compound_number_keys():
    # Valid slash numbers with >= 3 digits
    keys = extract_compound_number_keys("170/1 Lower Circular Road, Kolkata")
    assert "cnum_170_1" in keys

    keys = extract_compound_number_keys("Plot No-371/1519, Cuttack")
    assert "cnum_371_1519" in keys

    # Trivial fraction with < 3 digits should be filtered
    keys = extract_compound_number_keys("1/2 Main Street")
    assert "cnum_1_2" not in keys

    # Empty address
    assert extract_compound_number_keys("") == []
    assert extract_compound_number_keys("No numbers here") == []


def test_extract_record_blocking_keys():
    keys_with_tier = extract_record_blocking_keys(
        name="Rachana Reality Pvt Ltd",
        address="170/1 Lower Circular Road, Kolkata",
        country="India",
        include_compound_numbers=True,
    )
    assert "en_rachana reality pvt ltd" in keys_with_tier
    assert keys_with_tier["en_rachana reality pvt ltd"] == 1
    assert "cnum_170_1" in keys_with_tier
    assert keys_with_tier["cnum_170_1"] == 2


def test_rank_candidates_channel_consensus_first():
    candidates = [
        {"tid": "T1", "ch_count": 1, "best_tier": 1, "score": 0.90, "spec_sum": 0.5},
        {"tid": "T2", "ch_count": 3, "best_tier": 3, "score": 0.70, "spec_sum": 1.2},
        {"tid": "T3", "ch_count": 2, "best_tier": 2, "score": 0.80, "spec_sum": 0.8},
    ]
    ranked = rank_candidates_channel_consensus_first(candidates, cap=10)
    # T2 has ch_count=3, should be first
    assert ranked[0][0] == "T2"
    # T3 has ch_count=2, should be second
    assert ranked[1][0] == "T3"
    # T1 has ch_count=1, should be third
    assert ranked[2][0] == "T1"
    # Check reciprocal rank
    assert ranked[0][1]["reciprocal_retrieval_rank"] == 1.0
    assert ranked[1][1]["reciprocal_retrieval_rank"] == 0.5
