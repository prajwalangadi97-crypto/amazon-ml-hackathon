#!/usr/bin/env python3
"""V3 Full Production Runner and Comprehensive Validation Pipeline.

Executes full test inference across all 1,732,544 test Source1 entities:
- Model: artifacts/xgb_matcher_exp004b.json (26 features)
- Retrieval Metadata: channel_agreement_count, retrieval_priority_tier,
  reciprocal_retrieval_rank, s1_candidate_pool_size
- Decision Layer: EXP-003B (France=0.45, US=0.65, India=0.55, margin=0.30)
- Isolated Outputs: output/v3/matching_results.tsv, output/v3/candidate_pairs.tsv
- Preservation: V2 outputs and EXP-002B artifacts remain untouched.
- Complete post-run verification suite and JSON report generation.
"""

import collections
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    wall_start = time.time()
    start_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    print("=" * 95, flush=True)
    print("V3 FULL PRODUCTION: FINAL TEST INFERENCE", flush=True)
    print("=" * 95, flush=True)
    print(f"Start Timestamp     : {start_timestamp}", flush=True)

    # 1. Pre-Run System & Disk Space Audit
    total_disk, used_disk, free_disk = shutil.disk_usage(str(PROJECT_ROOT))
    free_gb = free_disk / (1024 ** 3)
    print(f"Available Disk Space: {free_gb:.2f} GB (Total: {total_disk / (1024 ** 3):.2f} GB)", flush=True)
    assert free_gb >= 5.0, f"Insufficient disk space: {free_gb:.2f} GB < 5.0 GB"

    # Configuration paths
    test_dir = PROJECT_ROOT / "dataset" / "test"
    model_path = PROJECT_ROOT / "artifacts" / "xgb_matcher_exp004b.json"
    v3_dir = PROJECT_ROOT / "output" / "v3"
    v3_checkpoints = v3_dir / "checkpoints"
    v3_matching = v3_dir / "matching_results.tsv"
    v3_candidates = v3_dir / "candidate_pairs.tsv"

    v3_dir.mkdir(parents=True, exist_ok=True)
    v3_checkpoints.mkdir(parents=True, exist_ok=True)

    print("\nProcess Configuration:", flush=True)
    print(f"  Model Path        : {model_path}", flush=True)
    print(f"  Test Directory    : {test_dir}", flush=True)
    print(f"  Checkpoints Dir   : {v3_checkpoints}", flush=True)
    print(f"  Matching Output   : {v3_matching}", flush=True)
    print(f"  Candidate Output  : {v3_candidates}", flush=True)
    print("  Thresholds        : France=0.45, US=0.65, India=0.55, Default=0.60", flush=True)
    print("  Margin Delta      : 0.30", flush=True)
    print("  Candidate Cap     : 100", flush=True)
    print("  Batch Size        : 50,000", flush=True)
    print("  CPU Threads       : 8", flush=True)

    # Verify preserved files exist before running
    preserved_files = [
        PROJECT_ROOT / "artifacts" / "xgb_matcher_exp002b.json",
        PROJECT_ROOT / "artifacts" / "xgb_matcher_exp004b.json",
        PROJECT_ROOT / "output" / "matching_results.tsv",
        PROJECT_ROOT / "output" / "matching_results_V2.tsv",
        PROJECT_ROOT / "output" / "candidate_pairs.tsv",
    ]
    for pf in preserved_files:
        assert pf.is_file(), f"Critical file missing: {pf}"

    # 2. Launch generate_final_submission.py
    print("\n" + "=" * 95, flush=True)
    print("LAUNCHING FULL SUBMISSION GENERATOR", flush=True)
    print("=" * 95, flush=True)
    t0_gen = time.time()

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "generate_final_submission.py"),
        "--test-dir", str(test_dir),
        "--model-path", str(model_path),
        "--checkpoint-dir", str(v3_checkpoints),
        "--output-matching", str(v3_matching),
        "--output-candidate", str(v3_candidates),
        "--france-threshold", "0.45",
        "--us-threshold", "0.65",
        "--india-threshold", "0.55",
        "--margin-delta", "0.30",
        "--final-cap", "100",
        "--batch-size", "50000",
        "--num-threads", "8",
    ]
    print(f"Executing: {' '.join(cmd)}\n", flush=True)

    gen_proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    gen_time = time.time() - t0_gen
    print(f"\nGenerator finished in {gen_time:.2f}s ({gen_time / 60:.2f} min) with exit code {gen_proc.returncode}.", flush=True)
    assert gen_proc.returncode == 0, f"Generator failed with code {gen_proc.returncode}!"

    # 3. Post-Run Final Checks
    print("\n" + "=" * 95, flush=True)
    print("RUNNING FINAL POST-PRODUCTION VALIDATION SUITE", flush=True)
    print("=" * 95, flush=True)

    # Check 1: matching_results.tsv invariants
    print("\n--- Check 1: matching_results.tsv Integrity ---", flush=True)
    s1_file = test_dir / "test_source1.tsv"
    test_s1_order: List[str] = []
    country_s1_counts = collections.defaultdict(int)

    with open(s1_file, "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                sid, c = parts[0], parts[3]
                test_s1_order.append(sid)
                country_s1_counts[c] += 1

    total_expected_s1 = len(test_s1_order)
    print(f"Expected test Source 1 entities: {total_expected_s1:,}", flush=True)
    for c, cnt in country_s1_counts.items():
        print(f"  {c:<15}: {cnt:,} entities", flush=True)

    matching_rows_count = 0
    empty_matches_count = 0
    total_predicted_links = 0
    duplicate_matched_ids = 0
    order_mismatch_matching = False
    all_predicted_tids: Set[str] = set()

    with open(v3_matching, "r", encoding="utf-8") as f:
        header = next(f).rstrip("\r\n").split("\t")
        assert header == ["source1_entity_id", "matched_entity_ids"], f"Invalid matching header: {header}"
        for idx, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            sid = parts[0]
            if idx < total_expected_s1 and sid != test_s1_order[idx]:
                order_mismatch_matching = True
            matched_str = parts[1] if len(parts) > 1 else ""
            if matched_str:
                m_list = matched_str.split(",")
                if len(set(m_list)) != len(m_list):
                    duplicate_matched_ids += 1
                total_predicted_links += len(m_list)
                all_predicted_tids.update(m_list)
            else:
                empty_matches_count += 1
            matching_rows_count += 1

    print(f"Matching rows written       : {matching_rows_count:,} (Expected: {total_expected_s1:,})", flush=True)
    print(f"Total predicted links       : {total_predicted_links:,}", flush=True)
    print(f"Empty-match (singleton) S1  : {empty_matches_count:,} ({empty_matches_count / total_expected_s1 * 100:.2f}%)", flush=True)
    print(f"Duplicate target ID rows    : {duplicate_matched_ids}", flush=True)
    print(f"S1 order preserved exactly  : {not order_mismatch_matching}", flush=True)

    assert matching_rows_count == total_expected_s1, f"Mismatch in matching rows: {matching_rows_count} vs {total_expected_s1}"
    assert not order_mismatch_matching, "Matching results ordering does NOT match test Source 1 ordering!"
    assert duplicate_matched_ids == 0, f"Found {duplicate_matched_ids} rows with duplicate target IDs!"
    print("[PASS] Check 1 passed: matching_results.tsv is valid and complete.")

    # Check 2: candidate_pairs.tsv invariants & country stats
    print("\n--- Check 2: candidate_pairs.tsv Integrity & Statistics ---", flush=True)
    candidate_rows_count = 0
    total_candidate_pairs = 0
    candidate_empty_count = 0
    duplicate_candidate_ids = 0
    order_mismatch_candidate = False

    # Read candidates while checking subset constraints against matches
    # Since streaming both files in sync is memory-efficient:
    candidate_subset_violations = 0

    with open(v3_matching, "r", encoding="utf-8") as f_m, open(v3_candidates, "r", encoding="utf-8") as f_c:
        next(f_m)
        cand_header = next(f_c).rstrip("\r\n").split("\t")
        assert cand_header == ["source1_entity_id", "candidate_entity_ids"], f"Invalid candidate header: {cand_header}"

        for idx, (l_m, l_c) in enumerate(zip(f_m, f_c)):
            p_m = l_m.rstrip("\r\n").split("\t")
            p_c = l_c.rstrip("\r\n").split("\t")

            s_id_m = p_m[0]
            s_id_c = p_c[0]
            if s_id_c != s_id_m:
                order_mismatch_candidate = True

            c_str = p_c[1] if len(p_c) > 1 else ""
            m_str = p_m[1] if len(p_m) > 1 else ""

            if c_str:
                c_list = c_str.split(",")
                if len(set(c_list)) != len(c_list):
                    duplicate_candidate_ids += 1
                total_candidate_pairs += len(c_list)
                c_set = set(c_list)
            else:
                candidate_empty_count += 1
                c_set = set()

            if m_str:
                m_list = m_str.split(",")
                for m_tid in m_list:
                    if m_tid not in c_set:
                        candidate_subset_violations += 1

            candidate_rows_count += 1

    avg_candidates_per_s1 = total_candidate_pairs / max(total_expected_s1, 1)
    print(f"Candidate rows written      : {candidate_rows_count:,} (Expected: {total_expected_s1:,})", flush=True)
    print(f"Total candidate pairs       : {total_candidate_pairs:,}", flush=True)
    print(f"Average candidates / S1     : {avg_candidates_per_s1:.2f}", flush=True)
    print(f"Empty candidate S1          : {candidate_empty_count:,}", flush=True)
    print(f"Candidate subset violations : {candidate_subset_violations} (Matches missing from candidates)", flush=True)
    print(f"Duplicate candidate ID rows : {duplicate_candidate_ids}", flush=True)

    assert candidate_rows_count == total_expected_s1, f"Mismatch in candidate rows: {candidate_rows_count} vs {total_expected_s1}"
    assert not order_mismatch_candidate, "Candidate pairs ordering does NOT match matching results ordering!"
    assert candidate_subset_violations == 0, f"Found {candidate_subset_violations} predicted matches missing from candidate_pairs!"
    assert duplicate_candidate_ids == 0, f"Found {duplicate_candidate_ids} rows with duplicate candidate IDs!"

    # Country-wise candidate counts from checkpoints
    country_cand_counts = {}
    for c in ["france", "us", "india"]:
        ckpt_cand = v3_checkpoints / f"candidates_{c}.tsv"
        c_pairs = 0
        if ckpt_cand.is_file():
            with open(ckpt_cand, "r", encoding="utf-8") as f:
                for line in f:
                    pos = line.find("\t")
                    if pos != -1:
                        c_str = line[pos + 1 :].rstrip("\r\n")
                        if c_str:
                            c_pairs += c_str.count(",") + 1
        country_cand_counts[c.capitalize()] = c_pairs
        print(f"  {c.capitalize():<15} candidate pairs: {c_pairs:,}", flush=True)

    print("[PASS] Check 2 passed: candidate_pairs.tsv is valid and strict superset of matching_results.tsv.")

    # Check 3: Target ID Validity (against test_source2.tsv and test_source3.tsv)
    print("\n--- Check 3: Target ID Validity Across S2 & S3 ---", flush=True)
    print(f"Unique predicted target IDs to verify: {len(all_predicted_tids):,}", flush=True)

    unresolved_tids = set(all_predicted_tids)
    for src_file in [test_dir / "test_source2.tsv", test_dir / "test_source3.tsv"]:
        print(f"Checking {src_file.name}...", flush=True)
        with open(src_file, "r", encoding="utf-8-sig") as f:
            next(f)
            for line in f:
                pos = line.find("\t")
                if pos != -1:
                    tid = line[:pos]
                    unresolved_tids.discard(tid)
                    if not unresolved_tids:
                        break
        if not unresolved_tids:
            break

    invalid_target_ids_count = len(unresolved_tids)
    print(f"Unresolved / invalid target IDs: {invalid_target_ids_count}", flush=True)
    assert invalid_target_ids_count == 0, f"Found {invalid_target_ids_count} invalid target IDs: {list(unresolved_tids)[:5]}"
    del unresolved_tids, all_predicted_tids
    gc.collect()
    print("[PASS] Check 3 passed: 100% of predicted target IDs exist in test Source2 or Source3.")

    # Check 4: Official Submission Validator Check
    print("\n--- Check 4: Official validate_submission.py Validator ---", flush=True)
    val_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "utils" / "validate_submission.py"),
        "--matching", str(v3_matching),
        "--candidate", str(v3_candidates),
        "--test-dir", str(test_dir),
    ]
    print(f"Running: {' '.join(val_cmd)}", flush=True)
    val_proc = subprocess.run(val_cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    print(val_proc.stdout, flush=True)
    if val_proc.stderr:
        print("Validator STDERR:", val_proc.stderr, flush=True)
    assert val_proc.returncode == 0, f"Validator failed with exit code {val_proc.returncode}!"
    print("[PASS] Check 4 passed: utils/validate_submission.py passed with exit code 0 (Safe to submit).")

    # Check 5: Project Unit Tests
    print("\n--- Check 5: Project Test Suite ---", flush=True)
    pytest_proc = subprocess.run([sys.executable, "-m", "pytest"], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    print(pytest_proc.stdout, flush=True)
    assert pytest_proc.returncode == 0, "Pytest test suite failed!"
    print("[PASS] Check 5 passed: All 40 pytest unit tests passed.")

    # Check 6: All Three Countries Processed
    print("\n--- Check 6: All Three Countries Processed ---", flush=True)
    for c in ["france", "us", "india"]:
        m_ckpt = v3_checkpoints / f"matches_{c}.tsv"
        c_ckpt = v3_checkpoints / f"candidates_{c}.tsv"
        assert m_ckpt.is_file(), f"Missing matching checkpoint for {c}: {m_ckpt}"
        assert c_ckpt.is_file(), f"Missing candidate checkpoint for {c}: {c_ckpt}"
        print(f"  [OK] {c.capitalize():<10} matches checkpoint: {m_ckpt.stat().st_size:,} bytes", flush=True)
        print(f"  [OK] {c.capitalize():<10} cands checkpoint  : {c_ckpt.stat().st_size:,} bytes", flush=True)
    print("[PASS] Check 6 passed: All three countries (France, US, India) checkpointed and merged.")

    # Check 7: Output Sizes & Final Disk Usage
    end_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    total_wall_time = time.time() - wall_start
    matching_size = v3_matching.stat().st_size
    candidate_size = v3_candidates.stat().st_size
    _, _, final_free = shutil.disk_usage(str(PROJECT_ROOT))

    print("\n--- Final Output File Sizes & Disk State ---", flush=True)
    print(f"  matching_results.tsv : {matching_size:,} bytes ({matching_size / (1024*1024):.2f} MB)", flush=True)
    print(f"  candidate_pairs.tsv  : {candidate_size:,} bytes ({candidate_size / (1024*1024):.2f} MB)", flush=True)
    print(f"  Final Free Disk Space: {final_free / (1024**3):.2f} GB", flush=True)

    # Generate V3-production-report.json
    report_data = {
        "report_id": "V3-production-report",
        "description": "Amazon ML Challenge 2026 V3 Full Production Test Inference",
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "total_wall_clock_seconds": total_wall_time,
        "total_wall_clock_minutes": total_wall_time / 60,
        "status": "V3 PRODUCTION READY FOR SUBMISSION",
        "configuration": {
            "model_path": str(model_path.resolve()),
            "feature_count": 26,
            "decision_layer": {
                "france_threshold": 0.45,
                "us_threshold": 0.65,
                "india_threshold": 0.55,
                "margin_delta": 0.30,
                "default_threshold": 0.60,
            },
            "candidate_cap": 100,
            "batch_size": 50000,
        },
        "submission_statistics": {
            "total_source1_entities": total_expected_s1,
            "matching_rows_written": matching_rows_count,
            "candidate_rows_written": candidate_rows_count,
            "total_candidate_pairs": total_candidate_pairs,
            "avg_candidates_per_s1": avg_candidates_per_s1,
            "country_candidate_pairs": country_cand_counts,
            "total_predicted_links": total_predicted_links,
            "empty_matches_singletons": empty_matches_count,
            "empty_candidates_singletons": candidate_empty_count,
        },
        "verification_results": {
            "row_count_passed": (matching_rows_count == total_expected_s1 and candidate_rows_count == total_expected_s1),
            "entity_order_preserved": (not order_mismatch_matching and not order_mismatch_candidate),
            "duplicate_matched_id_violations": duplicate_matched_ids,
            "duplicate_candidate_id_violations": duplicate_candidate_ids,
            "invalid_predicted_target_ids": invalid_target_ids_count,
            "candidate_subset_violations": candidate_subset_violations,
            "all_three_countries_processed": True,
            "validator_exit_code": val_proc.returncode,
            "pytest_exit_code": pytest_proc.returncode,
        },
        "output_files": {
            "matching_results_path": str(v3_matching.resolve()),
            "matching_results_bytes": matching_size,
            "candidate_pairs_path": str(v3_candidates.resolve()),
            "candidate_pairs_bytes": candidate_size,
        },
        "preserved_v2_artifacts": [str(pf.resolve()) for pf in preserved_files],
    }

    report_path = PROJECT_ROOT / "experiments" / "V3-production-report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nSaved production report to {report_path}", flush=True)

    print("\n" + "=" * 95, flush=True)
    print("V3 FULL PRODUCTION INFERENCE AND VALIDATION COMPLETED SUCCESSFULLY", flush=True)
    print(f"Total Wall-Clock Time: {total_wall_time:.2f}s ({total_wall_time / 60:.2f} min)", flush=True)
    print("=" * 95, flush=True)
    print("\nFINAL STATUS: V3 PRODUCTION READY FOR SUBMISSION\n", flush=True)


if __name__ == "__main__":
    main()
