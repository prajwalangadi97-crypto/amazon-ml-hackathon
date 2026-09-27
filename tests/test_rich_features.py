"""Tests for EXP-009 Rich Pairwise Feature Extraction."""

import pytest
from src.matching.candidate_compression import preprocess_record
from src.matching.rich_features import extract_rich_pairwise_features, RICH_FEATURE_NAMES


def test_rich_features_count_and_completeness():
    s1 = preprocess_record("S1-100", "sri sai ram enterprises", "170/1 main road indiranagar bangalore 560038", "India")
    tgt = preprocess_record("S2-200", "sri sai ram enterprise pvt ltd", "plot 170/1 2nd cross indiranagar bengaluru 560038", "India")

    meta = {
        "channel_agreement_count": 3,
        "retrieval_priority_tier": 1,
        "reciprocal_retrieval_rank": 1.0,
        "s1_candidate_pool_size": 50,
    }
    feats = extract_rich_pairwise_features(s1, tgt, meta)
    assert len(feats) == len(RICH_FEATURE_NAMES)
    assert len(feats) == 52

    for name in RICH_FEATURE_NAMES:
        assert name in feats, f"Missing feature: {name}"
        assert not (feats[name] != feats[name]), f"NaN found in feature: {name}"


def test_rich_features_empty_fields():
    s1 = preprocess_record("S1-101", "", "", "US")
    tgt = preprocess_record("S2-201", "Acme Corp", "123 Main St", "US")

    feats = extract_rich_pairwise_features(s1, tgt)
    assert feats["s1_name_empty"] == 1.0
    assert feats["s1_addr_empty"] == 1.0
    assert feats["both_fields_present"] == 0.0
    assert feats["missing_name_indicator"] == 1.0
    assert feats["missing_addr_indicator"] == 1.0
    assert feats["name_fuzz_ratio"] == 0.0
    assert feats["addr_fuzz_ratio"] == 0.0


def test_rich_features_compound_and_cross_field():
    s1 = preprocess_record("S1-102", "Alpha Logistics", "Plot 45/2 Sector 9 Delhi", "India")
    tgt = preprocess_record("S2-202", "Alpha Logistics India", "Shop 45/2 Sector 9 New Delhi", "India")

    feats = extract_rich_pairwise_features(s1, tgt)
    assert feats["compound_number_agreement"] == 1.0
    assert feats["country_eq"] == 1.0
    assert feats["name_exact_eq"] == 0.0
    assert feats["name_fuzz_wratio"] > 0.80
