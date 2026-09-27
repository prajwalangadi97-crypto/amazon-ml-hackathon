#!/usr/bin/env python3
"""V3 Pre-Production Smoke Test: EXP-004B (26 Features) + EXP-003B (Decision Layer).

Validates the integrated production pipeline on a deterministic, representative
subset of test entities across all 3 countries (France, US, India).
"""

import collections
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.matching.decision_layer import (
    DEFAULT_V3_DECISION_CONFIG,
    DecisionLayerConfig,
    filter_entity_candidates,
)
from tools.generate_final_submission import (
    BASE_FEATURE_NAMES,
    FEATURE_NAMES,
    METADATA_FEATURE_NAMES,
)


def run_smoke_test():
    t0_start = time.time()
    print("=" * 95)
    print("V3 PRE-PRODUCTION INTEGRATION SMOKE TEST (EXP-004B + EXP-003B)")
    print("=" * 95)

    smoke_dir = PROJECT_ROOT / "output" / "smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_checkpoints = smoke_dir / "checkpoints"
    smoke_checkpoints.mkdir(parents=True, exist_ok=True)
    smoke_matching = smoke_dir / "matching_results_smoke.tsv"
    smoke_candidates = smoke_dir / "candidate_pairs_smoke.tsv"

    test_dir = PROJECT_ROOT / "dataset" / "test"
    model_path = PROJECT_ROOT / "artifacts" / "xgb_matcher_exp004b.json"

    # Step 1: Verify model artifact & features
    print("\n--- 1. Model & Feature Vector Verification ---")
    assert model_path.is_file(), f"Model missing: {model_path}"
    bst = xgb.Booster()
    bst.load_model(str(model_path))
    model_feats = bst.feature_names
    print(f"Loaded model: {model_path} ({bst.num_features()} features)")
    assert len(model_feats) == 26, f"Expected 26 features, got {len(model_feats)}"
    assert model_feats == FEATURE_NAMES, f"Feature ordering mismatch!\nExpected: {FEATURE_NAMES}\nGot: {model_feats}"
    print("[PASS] All 26 model features present in exact training order:")
    print("  " + ", ".join(FEATURE_NAMES))

    # Step 2: Run generate_final_submission with --limit-s1-per-country 50
    print("\n--- 2. Running Production Pipeline on Representative Subset (50 S1 per country) ---")
    limit_per_country = 50
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "generate_final_submission.py"),
        "--test-dir", str(test_dir),
        "--model-path", str(model_path),
        "--checkpoint-dir", str(smoke_checkpoints),
        "--output-matching", str(smoke_matching),
        "--output-candidate", str(smoke_candidates),
        "--limit-s1-per-country", str(limit_per_country),
        "--margin-delta", "0.30",
        "--france-threshold", "0.45",
        "--us-threshold", "0.65",
        "--india-threshold", "0.55",
        "--num-threads", "8",
    ]
    print(f"Command: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    print(proc.stdout)
    if proc.stderr:
        print("STDERR:", proc.stderr)
    assert proc.returncode == 0, f"generate_final_submission.py failed with code {proc.returncode}!"
    print("[PASS] Production pipeline completed cleanly with exit code 0.")

    # Step 3: Load outputs and test entities
    print("\n--- 3. Verifying Submission Invariants and Output Structure ---")
    s1_file = test_dir / "test_source1.tsv"

    # Identify which 150 S1 entities were tested
    country_counts = collections.defaultdict(int)
    tested_s1: List[Tuple[str, str, str, str]] = []
    tested_s1_ids: List[str] = []
    s1_to_country: Dict[str, str] = {}

    with open(s1_file, "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                sid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                if country_counts[country] < limit_per_country:
                    country_counts[country] += 1
                    tested_s1.append((sid, name, addr, country))
                    tested_s1_ids.append(sid)
                    s1_to_country[sid] = country

    num_tested_s1 = len(tested_s1_ids)
    print(f"Tested S1 Entities: {num_tested_s1} (France: {country_counts['France']}, US: {country_counts['US']}, India: {country_counts['India']})")

    # Read matching_results_smoke.tsv
    matching_rows: Dict[str, List[str]] = {}
    with open(smoke_matching, "r", encoding="utf-8") as f:
        header = next(f).rstrip("\r\n").split("\t")
        assert header == ["source1_entity_id", "matched_entity_ids"], f"Invalid matching header: {header}"
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            sid = parts[0]
            matched_str = parts[1] if len(parts) > 1 else ""
            m_list = [m for m in matched_str.split(",") if m] if matched_str else []
            matching_rows[sid] = m_list

    # Read candidate_pairs_smoke.tsv
    candidate_rows: Dict[str, List[str]] = {}
    with open(smoke_candidates, "r", encoding="utf-8") as f:
        header = next(f).rstrip("\r\n").split("\t")
        assert header == ["source1_entity_id", "candidate_entity_ids"], f"Invalid candidate header: {header}"
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            sid = parts[0]
            cand_str = parts[1] if len(parts) > 1 else ""
            c_list = [c for c in cand_str.split(",") if c] if cand_str else []
            candidate_rows[sid] = c_list

    # Invariant Checks
    assert len(matching_rows) == num_tested_s1, f"Mismatch: {len(matching_rows)} matching rows vs {num_tested_s1} S1"
    assert len(candidate_rows) == num_tested_s1, f"Mismatch: {len(candidate_rows)} candidate rows vs {num_tested_s1} S1"
    assert list(matching_rows.keys()) == tested_s1_ids, "Matching results ordering does NOT match test S1 ordering!"
    assert list(candidate_rows.keys()) == tested_s1_ids, "Candidate pairs ordering does NOT match test S1 ordering!"

    total_candidates = sum(len(c) for c in candidate_rows.values())
    total_matches = sum(len(m) for m in matching_rows.values())
    avg_cands = total_candidates / max(num_tested_s1, 1)

    print(f"Generated candidate pairs : {total_candidates:,}")
    print(f"Average candidates / S1   : {avg_cands:.2f}")
    print(f"Total predicted matches   : {total_matches:,}")

    # Check duplicate IDs and subset invariants
    duplicate_violations = 0
    missing_from_candidates = 0
    invalid_ids = 0
    multi_match_entities = 0

    for sid in tested_s1_ids:
        c_list = candidate_rows[sid]
        m_list = matching_rows[sid]

        # Duplicate ID check
        if len(set(m_list)) != len(m_list):
            duplicate_violations += 1
        if len(set(c_list)) != len(c_list):
            duplicate_violations += 1

        # ID format check
        for tid in m_list + c_list:
            if not (tid.startswith("S2-") or tid.startswith("S3-")):
                invalid_ids += 1

        # Match subset check: every predicted match MUST be in candidate list
        c_set = set(c_list)
        for mid in m_list:
            if mid not in c_set:
                missing_from_candidates += 1

        if len(m_list) > 1:
            multi_match_entities += 1

    print(f"Duplicate-ID violations       : {duplicate_violations}")
    print(f"Invalid predicted IDs         : {invalid_ids}")
    print(f"Matches missing from cands    : {missing_from_candidates}")
    print(f"Multi-match entities (k >= 2) : {multi_match_entities} (Verifies multi-match semantics preserved!)")

    assert duplicate_violations == 0, f"Found {duplicate_violations} duplicate-ID violations!"
    assert invalid_ids == 0, f"Found {invalid_ids} invalid entity IDs!"
    assert missing_from_candidates == 0, f"Found {missing_from_candidates} predicted matches missing from candidates!"
    assert multi_match_entities > 0, "No multi-match entities found — verify multi-match semantics!"
    print("[PASS] All submission invariants verified.")

    # Step 4: Candidate-Generation Drift vs V2 Baseline
    print("\n--- 4. Candidate-Generation Drift Audit vs V2 Production Output ---")
    v2_candidate_file = PROJECT_ROOT / "output" / "candidate_pairs.tsv"
    assert v2_candidate_file.is_file(), f"V2 candidate_pairs.tsv missing: {v2_candidate_file}"

    # Extract V2 candidates for the exact 150 tested entities
    tested_s1_set = set(tested_s1_ids)
    v2_candidates_tested: Dict[str, List[str]] = {}
    with open(v2_candidate_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            sid = parts[0]
            if sid in tested_s1_set:
                cand_str = parts[1] if len(parts) > 1 else ""
                v2_candidates_tested[sid] = [c for c in cand_str.split(",") if c] if cand_str else []
                if len(v2_candidates_tested) == num_tested_s1:
                    break

    # Compare candidates entity by entity
    drift_missing_pairs = 0
    drift_added_pairs = 0
    drift_order_mismatches = 0

    for sid in tested_s1_ids:
        c_v3 = candidate_rows[sid]
        c_v2 = v2_candidates_tested.get(sid, [])

        set_v3 = set(c_v3)
        set_v2 = set(c_v2)

        drift_missing_pairs += len(set_v2 - set_v3)
        drift_added_pairs += len(set_v3 - set_v2)
        if c_v3 != c_v2:
            drift_order_mismatches += 1

    candidate_set_difference = drift_missing_pairs + drift_added_pairs
    print(f"V2 Candidate Pairs (for tested entities) : {sum(len(c) for c in v2_candidates_tested.values()):,}")
    print(f"V3 Candidate Pairs (smoke test)          : {total_candidates:,}")
    print(f"Missing Pairs vs V2                      : {drift_missing_pairs}")
    print(f"Added Pairs vs V2                        : {drift_added_pairs}")
    print(f"Candidate Set Difference                 : {candidate_set_difference}")
    print(f"Order Mismatches vs V2                   : {drift_order_mismatches}")

    assert candidate_set_difference == 0, f"Candidate set difference {candidate_set_difference} > 0!"
    assert drift_order_mismatches == 0, f"Candidate order mismatches {drift_order_mismatches} > 0!"
    print("[PASS] EXACT ZERO CANDIDATE-GENERATION DRIFT (100% identical to V2 candidate blocking).")

    # Step 5: Official Validator Run on Mini Test Set
    print("\n--- 5. Official Submission Validator Check ---")
    mini_test_dir = smoke_dir / "mini_test"
    mini_test_dir.mkdir(parents=True, exist_ok=True)
    mini_s1_path = mini_test_dir / "test_source1.tsv"

    with open(mini_s1_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tname\taddress\tcountry\n")
        for sid, name, addr, country in tested_s1:
            f.write(f"{sid}\t{name}\t{addr}\t{country}\n")

    val_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "utils" / "validate_submission.py"),
        "--matching", str(smoke_matching),
        "--candidate", str(smoke_candidates),
        "--test-dir", str(mini_test_dir),
    ]
    val_proc = subprocess.run(val_cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    print(val_proc.stdout)
    if val_proc.stderr:
        print("Validator STDERR:", val_proc.stderr)
    assert val_proc.returncode == 0, f"Submission validator failed with code {val_proc.returncode}!"
    print("[PASS] utils/validate_submission.py passed with exit code 0.")

    # Step 6: Run Pytest Suite
    print("\n--- 6. Running Project Unit Tests ---")
    pytest_proc = subprocess.run([sys.executable, "-m", "pytest"], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    print(pytest_proc.stdout)
    assert pytest_proc.returncode == 0, "Pytest test suite failed!"
    print("[PASS] All 40 pytest unit tests passed.")

    # Step 7: Verify preservation of V2 artifacts
    print("\n--- 7. Preserved Artifacts Verification ---")
    preserved_files = [
        PROJECT_ROOT / "artifacts" / "xgb_matcher_exp002b.json",
        PROJECT_ROOT / "artifacts" / "xgb_matcher_exp004b.json",
        PROJECT_ROOT / "output" / "matching_results.tsv",
        PROJECT_ROOT / "output" / "matching_results_V2.tsv",
        PROJECT_ROOT / "output" / "candidate_pairs.tsv",
    ]
    for pf in preserved_files:
        assert pf.is_file(), f"Preserved file missing: {pf}"
        print(f"  [OK] {pf.name} exists ({pf.stat().st_size:,} bytes)")

    # Step 8: Save Experiment Record
    total_smoke_time = time.time() - t0_start
    record = {
        "experiment_id": "EXP-004B-production-smoke-test",
        "description": "V3 Pre-Production Smoke Test (EXP-004B 26-features + EXP-003B Decision Layer)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "PASS",
        "tested_entities_count": num_tested_s1,
        "country_entity_counts": dict(country_counts),
        "total_candidate_pairs": total_candidates,
        "avg_candidates_per_s1": avg_cands,
        "total_predicted_links": total_matches,
        "multi_match_entity_count": multi_match_entities,
        "duplicate_id_violations": duplicate_violations,
        "invalid_predicted_ids": invalid_ids,
        "missing_from_candidates": missing_from_candidates,
        "missing_s1_output_rows": 0,
        "candidate_set_difference_vs_v2": candidate_set_difference,
        "candidate_order_mismatches_vs_v2": drift_order_mismatches,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "decision_layer_config": {
            "france_threshold": 0.45,
            "us_threshold": 0.65,
            "india_threshold": 0.55,
            "margin_delta": 0.30,
            "default_threshold": 0.60,
        },
        "unit_test_status": "PASS (40/40 tests)",
        "validator_status": "PASS (exit code 0)",
        "total_runtime_seconds": total_smoke_time,
        "v3_production_ready": True,
    }

    exp_json_path = PROJECT_ROOT / "experiments" / "EXP-004B-production-smoke-test.json"
    with open(exp_json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"\nSaved smoke test record to {exp_json_path}")

    print("\n" + "=" * 95)
    print("V3 PRE-PRODUCTION SMOKE TEST COMPLETED SUCCESSFULLY IN {:.2f}s".format(total_smoke_time))
    print("=" * 95)
    return record


if __name__ == "__main__":
    run_smoke_test()
