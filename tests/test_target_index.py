"""Tests for target inverted index integrity and EXP-006B retrieval compatibility."""

import array
import json
import os
from pathlib import Path
import pickle
import pytest

from src.blocking.retrieval import (
    extract_compound_number_keys,
    extract_record_blocking_keys,
    rank_candidates_channel_consensus_first,
)
from tools.build_target_indexes import (
    EXPECTED_TARGET_COUNTS,
    extract_target_blocking_keys,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_extract_target_blocking_keys_parity():
    """Verify that extract_target_blocking_keys produces all required EXP-006B channels."""
    name = "Shree Balaji Traders"
    addr = "Shop No 14/2, GIDC Estate, Vatva, Ahmedabad 382445"
    country = "India"

    keys = extract_target_blocking_keys(name, addr, country, include_compound_numbers=True)

    # 1. Exact name normalized
    assert "en_shree balaji traders" in keys

    # 2. Token blocking keys
    assert any(k.startswith("tk_") for k in keys)

    # 3. PIN code keys
    assert any("382445" in k for k in keys)

    # 4. Compound plot/door number keys (14/2 -> cnum_14_2)
    assert "cnum_14_2" in keys


def test_target_index_france_presence_and_structure():
    """Verify that the France target index is preserved, loads cleanly, and has expected record count."""
    france_path = PROJECT_ROOT / "artifacts" / "target_index_france.pkl"
    if not france_path.is_file():
        france_path = PROJECT_ROOT / "scratch" / "target_index_france.pkl"

    assert france_path.is_file(), f"France index not found at {france_path}"

    with open(france_path, "rb") as f:
        data = pickle.load(f)

    assert "target_ids" in data
    assert "inverted_index" in data

    target_ids = data["target_ids"]
    inverted_index = data["inverted_index"]

    assert len(target_ids) == EXPECTED_TARGET_COUNTS["France"]
    assert len(inverted_index) > 1_000_000
    assert isinstance(target_ids, list)
    assert isinstance(inverted_index, dict)

    # Check that sample postings are array.array('I')
    first_key = next(iter(inverted_index))
    assert isinstance(inverted_index[first_key], array.array)
    assert inverted_index[first_key].typecode == "I"


def test_compound_number_index_lookup_mock():
    """Verify mock target index with compound numbers retrieves candidate targets accurately."""
    mock_target_ids = ["T_IND_001", "T_IND_002", "T_IND_003"]
    mock_inverted = {
        "en_balaji enterprises": array.array("I", [0]),
        "cnum_170_1": array.array("I", [0, 1]),
        "pin_700014": array.array("I", [0, 2]),
    }

    # Query with S1 having compound number 170/1
    s1_addr = "170/1 Lower Circular Road"
    cnum_keys = extract_compound_number_keys(s1_addr)
    assert "cnum_170_1" in cnum_keys

    hits = set()
    for k in cnum_keys:
        if k in mock_inverted:
            hits.update(mock_inverted[k])

    matched_tids = [mock_target_ids[idx] for idx in hits]
    assert "T_IND_001" in matched_tids
    assert "T_IND_002" in matched_tids
    assert "T_IND_003" not in matched_tids


def test_target_index_manifest_verification():
    """Verify that the target index manifest exists, is complete, and records all 9.97M targets."""
    manifest_path = PROJECT_ROOT / "artifacts" / "target_index_manifest.json"
    assert manifest_path.is_file(), f"Manifest file missing: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["status"] == "ALL_INDEXES_VERIFIED"
    assert manifest["total_target_records"] == 9_969_589
    assert manifest["expected_total_targets"] == 9_969_589

    indexes = manifest["indexes"]
    for country in ["France", "US", "India"]:
        assert country in indexes
        entry = indexes[country]
        assert entry["record_count"] == EXPECTED_TARGET_COUNTS[country]
        assert entry["validation_status"] == "VERIFIED"
        assert len(entry["sha256"]) == 64
        idx_file = PROJECT_ROOT / entry["index_path"]
        assert idx_file.is_file()

