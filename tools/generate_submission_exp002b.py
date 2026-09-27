#!/usr/bin/env python3
"""EXP-002B: Submission Generator & Local Validator.

Implements Phases 6, 7, 8, and 9 of EXP-002B:
1. Loads the trained XGBoost model from artifacts/xgb_matcher_exp002b.json and
   the validated optimal threshold from experiments/EXP-002B.json.
2. Reads dataset/test/test_source1.tsv and groups entities dynamically by country.
3. For each country:
   - Indexes that country's S1 entities using D + E-A + E-C blocking keys.
   - Streams that country's records from test_source2.tsv and test_source3.tsv using
     multi-process chunked workers.
   - Compresses candidates using Hybrid-T0.40-Cap100 (score >= 0.40, max 100).
   - Computes pairwise 22-feature vectors for compressed candidates.
   - Scores candidates with XGBoost and applies optimal threshold theta*.
4. Writes:
   - output/candidate_pairs.tsv (header: source1_entity_id\tcandidate_entity_ids)
   - output/matching_results.tsv (header: source1_entity_id\tmatched_entity_ids)
5. Performs complete local validation:
   - Row counts match exactly len(test_source1.tsv).
   - Every test S1 is represented exactly once.
   - Singleton empty representation (s1_id\t).
   - Candidate pairs are compressed subset.
   - Runs utils/validate_submission.py and tests/run_tests.py.
"""

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import xgboost as xgb

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking.normalizers import (
    extract_address_keys,
    extract_token_blocking_keys,
    normalize_name_conservative,
)
from src.matching.candidate_compression import (
    compute_cheap_candidate_score,
    extract_features_from_preprocessed,
    fast_score_candidate,
    preprocess_record,
    PreprocessedRecord,
)
from tools.run_recovery_experiments import (
    extract_h1_relaxed_address_keys,
    extract_h3_name_locality_keys,
)

FEATURE_NAMES = [
    "country_eq",
    "source_type",
    "s1_name_empty",
    "target_name_empty",
    "s1_addr_empty",
    "target_addr_empty",
    "name_exact_eq",
    "name_fuzz_ratio",
    "name_fuzz_wratio",
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "name_prefix_similarity",
    "name_len_diff",
    "name_len_ratio",
    "addr_fuzz_ratio",
    "addr_token_similarity",
    "numeric_token_overlap",
    "house_number_agreement",
    "postal_pin_overlap",
    "addr_len_diff",
    "cross_field_name_loc",
    "cheap_candidate_score",
]


def worker_stream_test_country_chunk(task_args):
    """Worker function to stream and score a chunk of records for a specific country."""
    worker_id, file_path, start_line, end_line, target_country, c_idx, s1_preprocessed = task_args

    scored_candidates = collections.defaultdict(list)
    target_records = {}

    t0 = time.time()
    lines_read = 0
    pairs_scored = 0

    with open(file_path, "r", encoding="utf-8-sig") as f:
        next(f)  # header
        for current_line, line in enumerate(f):
            if current_line < start_line:
                continue
            if current_line >= end_line:
                break

            lines_read += 1
            if f"\t{target_country}" not in line:
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            tid, tname, taddr, tcountry = parts[0], parts[1], parts[2], parts[3]
            if tcountry != target_country:
                continue

            matched_s1_set = set()

            # Exact name
            ten = normalize_name_conservative(tname)
            if ten:
                m = c_idx.get(f"en_{ten}")
                if m:
                    matched_s1_set.update(m)

            # Token name keys
            for tk in extract_token_blocking_keys(tname):
                m = c_idx.get(f"tk_{tk}")
                if m:
                    matched_s1_set.update(m)

            # Address keys
            if taddr and taddr.strip():
                for ak in extract_address_keys(taddr):
                    m = c_idx.get(f"ak_{ak}")
                    if m:
                        matched_s1_set.update(m)
                for ak in extract_h1_relaxed_address_keys(taddr):
                    m = c_idx.get(ak)
                    if m:
                        matched_s1_set.update(m)
                for ck in extract_h3_name_locality_keys(tname, taddr):
                    m = c_idx.get(ck)
                    if m:
                        matched_s1_set.update(m)

            if not matched_s1_set:
                continue

            tgt_p = preprocess_record(tid, tname, taddr, tcountry)

            for s1_id in matched_s1_set:
                score = fast_score_candidate(s1_preprocessed[s1_id], tgt_p)
                if score >= 0.40:
                    scored_candidates[s1_id].append((score, tid))
                pairs_scored += 1

    # Prune per worker per s1 to top 150 candidates to keep IPC compact
    pruned_scored = {}
    for s1_id, pairs in scored_candidates.items():
        if len(pairs) > 150:
            pairs.sort(key=lambda x: x[0], reverse=True)
            pruned_scored[s1_id] = pairs[:150]
        else:
            pruned_scored[s1_id] = pairs

    return lines_read, pruned_scored


def worker_stream_test_country_wrapper(task_args):
    """Module-level wrapper for pool.map compatibility on Windows."""
    return worker_stream_test_country_chunk(task_args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/xgb_matcher_exp002b.json"))
    parser.add_argument("--exp-json", type=Path, default=Path("experiments/EXP-002B.json"))
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold if set")
    parser.add_argument("--test-dir", type=Path, default=Path("dataset/test"))
    parser.add_argument("--output-matching", type=Path, default=Path("output/matching_results.tsv"))
    parser.add_argument("--output-candidate", type=Path, default=Path("output/candidate_pairs.tsv"))
    args = parser.parse_args()

    overall_start = time.time()
    tracemalloc.start()

    print("=" * 80)
    print("EXP-002B: Submission Generator & Inference Pipeline")
    print(f"Workers: {args.num_workers} | Platform: {sys.platform} | CPU cores: {os.cpu_count()}")
    print("=" * 80)

    # 1. Load Model and Optimal Threshold
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    bst = xgb.Booster()
    bst.load_model(str(args.model_path))
    bst.set_param({"nthread": args.num_workers})
    print(f"Loaded trained XGBoost model from {args.model_path}")

    if args.threshold is not None:
        threshold = args.threshold
        print(f"Using CLI specified threshold: {threshold:.4f}")
    elif args.exp_json.is_file():
        with open(args.exp_json, "r", encoding="utf-8") as f:
            exp_data = json.load(f)
        threshold = exp_data["optimal_threshold"]
        print(f"Loaded optimal validation threshold: {threshold:.4f} (from {args.exp_json})")
    else:
        threshold = 0.50
        print(f"Defaulting to threshold: {threshold:.4f}")

    # 2. Read Test Source 1 and Partition Dynamically by Country
    s1_file = args.test_dir / "test_source1.tsv"
    s2_file = args.test_dir / "test_source2.tsv"
    s3_file = args.test_dir / "test_source3.tsv"

    print(f"\nReading test Source 1 entities from {s1_file}...")
    t0_read = time.time()
    test_s1_order: List[str] = []
    country_to_s1: Dict[str, List[Tuple[str, str, str, str]]] = collections.defaultdict(list)

    with open(s1_file, "r", encoding="utf-8-sig") as f:
        header = next(f).rstrip("\r\n").split("\t")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                sid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                test_s1_order.append(sid)
                country_to_s1[country].append((sid, name, addr, country))

    total_test_s1 = len(test_s1_order)
    print(f"Loaded {total_test_s1:,} test Source 1 entities across {len(country_to_s1)} countries in {time.time() - t0_read:.2f}s:")
    for c, entities in sorted(country_to_s1.items(), key=lambda x: len(x[1])):
        print(f"  {c:<15}: {len(entities):,} entities")

    # Prepare outputs
    args.output_matching.parent.mkdir(parents=True, exist_ok=True)
    args.output_candidate.parent.mkdir(parents=True, exist_ok=True)

    # Dictionaries to store all final candidates and matches
    all_final_candidates: Dict[str, List[str]] = {}
    all_final_matches: Dict[str, List[str]] = {}

    # Helper: count lines in file
    def get_line_count(fp):
        with open(fp, "r", encoding="utf-8-sig") as f:
            next(f)
            return sum(1 for _ in f)

    print("\nCounting lines in test Source 2 and Source 3 files...")
    s2_lines = get_line_count(s2_file)
    s3_lines = get_line_count(s3_file)
    print(f"Test Search Pool: Source 2 = {s2_lines:,} rows, Source 3 = {s3_lines:,} rows (Total: {s2_lines + s3_lines:,})")

    # 3. Process Each Country Dynamically
    for country, s1_records in sorted(country_to_s1.items(), key=lambda x: len(x[1])):
        print("\n" + "=" * 60)
        print(f"PROCESSING COUNTRY: {country} ({len(s1_records):,} S1 entities)")
        print("=" * 60)

        # Preprocess S1 records and build index
        country_s1_prep: Dict[str, PreprocessedRecord] = {}
        c_idx: Dict[str, List[str]] = collections.defaultdict(list)

        t0_c_prep = time.time()
        for sid, name, addr, c_name in s1_records:
            prep = preprocess_record(sid, name, addr, c_name)
            country_s1_prep[sid] = prep

            en = prep.name_norm
            if en:
                c_idx[f"en_{en}"].append(sid)
            for tk in extract_token_blocking_keys(prep.raw_name):
                c_idx[f"tk_{tk}"].append(sid)
            if prep.raw_address and prep.raw_address.strip():
                for ak in extract_address_keys(prep.raw_address):
                    c_idx[f"ak_{ak}"].append(sid)
                for ak in extract_h1_relaxed_address_keys(prep.raw_address):
                    c_idx[ak].append(sid)
                for ck in extract_h3_name_locality_keys(prep.raw_name, prep.raw_address):
                    c_idx[ck].append(sid)

        c_idx_clean = dict(c_idx)
        print(f"Indexed {len(country_s1_prep):,} entities ({len(c_idx_clean):,} keys) in {time.time() - t0_c_prep:.2f}s.")

        # Stream S2 and S3 for this country
        country_scored_cands: Dict[str, List[Tuple[float, str]]] = collections.defaultdict(list)

        for src_fp, n_lines in [(s2_file, s2_lines), (s3_file, s3_lines)]:
            fname = src_fp.name
            chunk_size = (n_lines + args.num_workers - 1) // args.num_workers
            tasks = []
            for wid in range(args.num_workers):
                start = wid * chunk_size
                end = min((wid + 1) * chunk_size, n_lines)
                tasks.append((wid, str(src_fp), start, end, country, c_idx_clean, country_s1_prep))

            print(f"Streaming {fname} for country {country} with {args.num_workers} workers...")
            t0_stream = time.time()
            with mp.Pool(processes=args.num_workers) as pool:
                results = pool.map(worker_stream_test_country_wrapper, tasks)

            for lines_read, scored_cands in results:
                for sid, pairs in scored_cands.items():
                    country_scored_cands[sid].extend(pairs)
            print(f"  Finished {fname} in {time.time() - t0_stream:.2f}s.")

        # Compress Candidates: Hybrid-T0.40-Cap100
        print(f"Compressing candidates with Hybrid-T0.40-Cap100...")
        country_compressed: Dict[str, List[str]] = {}
        unique_cands_needed: Set[str] = set()

        for sid, _, _, _ in s1_records:
            pairs = country_scored_cands.get(sid, [])
            if pairs:
                pairs.sort(key=lambda x: x[0], reverse=True)
                selected = [tid for score, tid in pairs if score >= 0.40][:100]
            else:
                selected = []
            country_compressed[sid] = selected
            unique_cands_needed.update(selected)

        total_country_pairs = sum(len(v) for v in country_compressed.values())
        print(f"Selected {total_country_pairs:,} candidate pairs across {len(s1_records):,} entities ({len(unique_cands_needed):,} unique target records).")

        # Save to global candidate dictionary
        all_final_candidates.update(country_compressed)

        if total_country_pairs == 0:
            print("No candidate pairs for this country (all singletons).")
            for sid, _, _, _ in s1_records:
                all_final_matches[sid] = []
            continue

        # Preprocess target records for feature extraction
        print(f"Reading and preprocessing {len(unique_cands_needed):,} target records for feature extraction...")
        target_preprocessed: Dict[str, PreprocessedRecord] = {}
        for src_fp in [s2_file, s3_file]:
            with open(src_fp, "r", encoding="utf-8-sig") as f:
                next(f)
                for line in f:
                    if f"\t{country}" not in line:
                        continue
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 4 and parts[0] in unique_cands_needed:
                        target_preprocessed[parts[0]] = preprocess_record(parts[0], parts[1], parts[2], parts[3])
                        if len(target_preprocessed) == len(unique_cands_needed):
                            break
            if len(target_preprocessed) == len(unique_cands_needed):
                break

        # Build feature matrix
        print("Extracting 22 pairwise features for scoring...")
        t0_feats = time.time()
        X_rows = []
        pair_mapping = []
        for sid, tids in country_compressed.items():
            s1_prep = country_s1_prep[sid]
            for tid in tids:
                tgt_prep = target_preprocessed[tid]
                feats = extract_features_from_preprocessed(s1_prep, tgt_prep)
                feats["cheap_candidate_score"] = compute_cheap_candidate_score(feats)
                row = [feats[fname] for fname in FEATURE_NAMES]
                X_rows.append(row)
                pair_mapping.append((sid, tid))

        X_matrix = np.array(X_rows, dtype=np.float32)
        print(f"Feature matrix built: {X_matrix.shape} in {time.time() - t0_feats:.2f}s.")

        # XGBoost Inference
        print(f"Scoring with XGBoost at threshold theta={threshold:.4f}...")
        t0_score = time.time()
        dmatrix = xgb.DMatrix(X_matrix, feature_names=FEATURE_NAMES)
        probs = bst.predict(dmatrix)
        print(f"Inference completed in {time.time() - t0_score:.2f}s.")

        # Filter predictions
        country_matches = collections.defaultdict(list)
        pred_link_count = 0
        for (sid, tid), prob in zip(pair_mapping, probs):
            if prob >= threshold:
                country_matches[sid].append(tid)
                pred_link_count += 1

        for sid, _, _, _ in s1_records:
            all_final_matches[sid] = country_matches.get(sid, [])

        print(f"Predicted {pred_link_count:,} matches for country {country}.")

        # Cleanup memory for this country
        del country_s1_prep, c_idx, c_idx_clean, country_scored_cands
        del target_preprocessed, X_rows, pair_mapping, X_matrix, probs

    # 4. Write Output Submission Files in Exact Required Format
    print("\n" + "=" * 80)
    print("WRITING SUBMISSION FILES")
    print("=" * 80)

    # A: matching_results.tsv
    print(f"Writing {args.output_matching}...")
    t0_w_match = time.time()
    matching_rows_written = 0
    matching_empty_count = 0
    total_matches_written = 0

    with open(args.output_matching, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in test_s1_order:
            matches = all_final_matches.get(sid, [])
            if matches:
                # Deduplicate while preserving order
                seen_m = set()
                dedup_m = []
                for m in matches:
                    if m not in seen_m:
                        seen_m.add(m)
                        dedup_m.append(m)
                match_str = ",".join(dedup_m)
                total_matches_written += len(dedup_m)
            else:
                match_str = ""
                matching_empty_count += 1
            f.write(f"{sid}\t{match_str}\n")
            matching_rows_written += 1

    print(f"Wrote {matching_rows_written:,} rows to {args.output_matching} in {time.time() - t0_w_match:.2f}s.")
    print(f"  Empty singletons: {matching_empty_count:,}")
    print(f"  Non-empty entities: {matching_rows_written - matching_empty_count:,}")
    print(f"  Total predicted links: {total_matches_written:,}")

    # B: candidate_pairs.tsv
    print(f"\nWriting {args.output_candidate}...")
    t0_w_cand = time.time()
    candidate_rows_written = 0
    candidate_empty_count = 0
    total_candidates_written = 0

    with open(args.output_candidate, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in test_s1_order:
            cands = all_final_candidates.get(sid, [])
            if cands:
                seen_c = set()
                dedup_c = []
                for c in cands:
                    if c not in seen_c:
                        seen_c.add(c)
                        dedup_c.append(c)
                cand_str = ",".join(dedup_c)
                total_candidates_written += len(dedup_c)
            else:
                cand_str = ""
                candidate_empty_count += 1
            f.write(f"{sid}\t{cand_str}\n")
            candidate_rows_written += 1

    print(f"Wrote {candidate_rows_written:,} rows to {args.output_candidate} in {time.time() - t0_w_cand:.2f}s.")
    print(f"  Empty singletons: {candidate_empty_count:,}")
    print(f"  Non-empty entities: {candidate_rows_written - candidate_empty_count:,}")
    print(f"  Total candidate pairs: {total_candidates_written:,}")

    # 5. Local Validation Checks
    print("\n" + "=" * 80)
    print("PHASE 9: LOCAL VALIDATION CHECKS")
    print("=" * 80)

    # Verification 1: Row count equals test source 1 count
    assert matching_rows_written == total_test_s1, f"Mismatch in matching rows: {matching_rows_written} vs {total_test_s1}"
    assert candidate_rows_written == total_test_s1, f"Mismatch in candidate rows: {candidate_rows_written} vs {total_test_s1}"
    print(f"[PASS] Row count matches exactly: {total_test_s1:,} rows.")

    # Verification 2: Run official validator utils/validate_submission.py
    import subprocess
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "utils" / "validate_submission.py"),
        "--matching", str(args.output_matching),
        "--candidate", str(args.output_candidate),
        "--test-dir", str(args.test_dir),
    ]
    print(f"\nRunning official submission validator:\n{' '.join(cmd)}")
    val_proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    print(val_proc.stdout)
    if val_proc.stderr:
        print("Validator STDERR:", val_proc.stderr)
    assert val_proc.returncode == 0, f"Validator failed with exit code {val_proc.returncode}!"
    print("[PASS] utils/validate_submission.py returned exit code 0 (VALID!).")

    # Verification 3: Run unit tests
    test_proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tests" / "run_tests.py")],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    print(f"\nUnit Tests Output:\n{test_proc.stderr.strip() or test_proc.stdout.strip()}")
    assert test_proc.returncode == 0, f"Unit tests failed with exit code {test_proc.returncode}!"
    print("[PASS] All unit tests pass.")

    total_time = time.time() - overall_start
    print("\n" + "=" * 80)
    print(f"EXP-002B SUBMISSION #2 GENERATION & VALIDATION COMPLETE ({total_time:.2f}s)!")
    print(f"Matching Results File : {args.output_matching.resolve()}")
    print(f"Candidate Pairs File  : {args.output_candidate.resolve()}")
    print("=" * 80)


def worker_stream_test_country_wrapper(args):
    """Module-level wrapper for pool.map on Windows."""
    return worker_stream_test_country_chunk(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
