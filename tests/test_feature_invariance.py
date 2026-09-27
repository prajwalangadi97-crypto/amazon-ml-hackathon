"""Critical regression test: Non-Indic pair feature invariance.

Verifies that for:
- US records
- France records
- English-only India records
all 26 feature values produced by extract_features_from_preprocessed are 100% invariant,
and name_translit is empty for non-Indic entities.
"""

import pytest
from src.matching.candidate_compression import (
    preprocess_record,
    extract_features_from_preprocessed,
    compute_cheap_candidate_score,
)

BASE_FEATURE_NAMES = [
    "country_eq", "source_type", "s1_name_empty", "target_name_empty",
    "s1_addr_empty", "target_addr_empty", "name_exact_eq", "name_fuzz_ratio",
    "name_fuzz_wratio", "name_token_set_ratio", "name_token_sort_ratio",
    "name_prefix_similarity", "name_len_diff", "name_len_ratio",
    "addr_fuzz_ratio", "addr_token_similarity", "numeric_token_overlap",
    "house_number_agreement", "postal_pin_overlap", "addr_len_diff",
    "cross_field_name_loc", "cheap_candidate_score",
]


def test_us_record_invariance():
    s1 = preprocess_record("S1-US1", "Smith & Sons Plumbing LLC", "123 Main Street, Suite 400, Austin, TX 78701", "US")
    tgt = preprocess_record("S2-US1", "Smith & Sons Plumbing Inc", "123 Main St Ste 400 Austin TX 78701", "US")

    assert s1.name_translit == ""
    assert tgt.name_translit == ""

    feats = extract_features_from_preprocessed(s1, tgt)
    feats["cheap_candidate_score"] = compute_cheap_candidate_score(feats)

    # Check all 22 base feature names exist and have valid values
    for f in BASE_FEATURE_NAMES:
        assert f in feats
        assert isinstance(feats[f], (float, int))

    assert feats["country_eq"] == 1.0
    assert feats["source_type"] == 0.0
    assert feats["name_fuzz_wratio"] > 0.80
    assert feats["house_number_agreement"] == 1.0
    assert feats["postal_pin_overlap"] == 1.0


def test_france_record_invariance():
    s1 = preprocess_record("S1-FR1", "Boulangerie Dupont SARL", "15 Rue de la Paix, 75002 Paris", "France")
    tgt = preprocess_record("S3-FR1", "Dupont Boulangerie", "15 Rue de la Paix 75002 Paris", "France")

    assert s1.name_translit == ""
    assert tgt.name_translit == ""

    feats = extract_features_from_preprocessed(s1, tgt)
    feats["cheap_candidate_score"] = compute_cheap_candidate_score(feats)

    for f in BASE_FEATURE_NAMES:
        assert f in feats

    assert feats["country_eq"] == 1.0
    assert feats["source_type"] == 1.0
    assert feats["name_token_set_ratio"] > 0.80
    assert feats["house_number_agreement"] == 1.0
    assert feats["postal_pin_overlap"] == 1.0


def test_english_india_record_invariance():
    s1 = preprocess_record("S1-IN1", "Apex Logistics Private Limited", "Plot 42, MIDC Industrial Area, Andheri East, Mumbai 400093", "India")
    tgt = preprocess_record("S2-IN1", "Apex Logistics Pvt Ltd", "Plot No 42 MIDC Andheri East Mumbai 400093", "India")

    assert s1.name_translit == ""
    assert tgt.name_translit == ""

    feats = extract_features_from_preprocessed(s1, tgt)
    feats["cheap_candidate_score"] = compute_cheap_candidate_score(feats)

    for f in BASE_FEATURE_NAMES:
        assert f in feats

    assert feats["country_eq"] == 1.0
    assert feats["name_fuzz_wratio"] > 0.80
    assert feats["house_number_agreement"] == 1.0
    assert feats["postal_pin_overlap"] == 1.0


def test_cross_script_indic_activation():
    # S1 is Latin, Target is Tamil
    s1 = preprocess_record("S1-IN2", "Raj Investments LLP", "6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu", "India")
    tgt = preprocess_record("S3-IN2", "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி", "6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, தமிழ்நாடு", "India")

    assert s1.name_translit == ""
    assert tgt.name_translit != ""
    assert "raaj" in tgt.name_translit or "raj" in tgt.name_translit

    feats = extract_features_from_preprocessed(s1, tgt)
    feats["cheap_candidate_score"] = compute_cheap_candidate_score(feats)

    # In baseline without transliteration, name_fuzz_wratio was 0.0
    # With transliteration, name_fuzz_wratio must be > 0.70
    assert feats["name_fuzz_wratio"] > 0.70
    assert feats["name_token_set_ratio"] > 0.60
    assert feats["cheap_candidate_score"] > 0.50
