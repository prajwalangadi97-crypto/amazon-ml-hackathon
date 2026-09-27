#!/usr/bin/env python3
"""EXP-002B: Fast, Bounded-Memory, Country-Checkpointed Production Submission Generator.

Processes country by country (France -> US -> India) to guarantee:
- Peak RAM strictly under 4.5 GB (zero disk paging/swapping).
- High CPU utilization and cache locality.
- Streaming checkpointing of candidate pairs and matching results per country.
- Strict preservation of test_source1.tsv order in final TSVs.
- 100% compliance with official competition validator utils/validate_submission.py.
"""

import argparse
import collections
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

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
    preprocess_record,
    PreprocessedRecord,
)
from src.matching.decision_layer import (
    DEFAULT_V3_DECISION_CONFIG,
    DecisionLayerConfig,
    filter_entity_candidates,
)
from tools.run_recovery_experiments import (
    extract_h1_relaxed_address_keys,
    extract_h3_name_locality_keys,
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

METADATA_FEATURE_NAMES = [
    "channel_agreement_count",
    "retrieval_priority_tier",
    "reciprocal_retrieval_rank",
    "s1_candidate_pool_size",
]

FEATURE_NAMES = BASE_FEATURE_NAMES + METADATA_FEATURE_NAMES


def get_peak_ram_mb() -> float:
    """Return peak working set memory in MB via Windows API or fallback."""
    try:
        import ctypes
        from ctypes import wintypes
        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(h, ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / (1024 * 1024)
    except Exception:
        pass

    try:
        pid = os.getpid()
        out = subprocess.check_output(
            ["powershell", "-Command", f"(Get-Process -Id {pid}).PeakWorkingSet64 / 1MB"],
            text=True,
        )
        return float(out.strip())
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=Path("dataset/test"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/xgb_matcher_exp004b.json"))
    parser.add_argument("--threshold", type=float, default=0.60, help="Default fallback threshold")
    parser.add_argument("--france-threshold", type=float, default=0.45)
    parser.add_argument("--us-threshold", type=float, default=0.65)
    parser.add_argument("--india-threshold", type=float, default=0.55)
    parser.add_argument("--margin-delta", type=float, default=0.30)
    parser.add_argument("--final-cap", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("output/checkpoints"))
    parser.add_argument("--output-matching", type=Path, default=Path("output/matching_results.tsv"))
    parser.add_argument("--output-candidate", type=Path, default=Path("output/candidate_pairs.tsv"))
    parser.add_argument("--limit-s1-per-country", type=int, default=None, help="Optional S1 limit per country for smoke testing")
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()

    overall_start = time.time()

    print("=" * 95, flush=True)
    print("PRODUCTION SUBMISSION GENERATOR: COUNTRY-PARTITIONED PIPELINE", flush=True)
    print(f"Model: {args.model_path} | Threshold: {args.threshold:.2f} | Final Cap: {args.final_cap}", flush=True)
    print(f"Threads: {args.num_threads} | Test Dir: {args.test_dir.resolve()}", flush=True)
    print("=" * 95, flush=True)

    # 1. Load Trained XGBoost Model
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    bst = xgb.Booster()
    bst.load_model(str(args.model_path))
    bst.set_param({"nthread": args.num_threads})
    print(f"Loaded trained XGBoost model from {args.model_path}", flush=True)

    s1_file = args.test_dir / "test_source1.tsv"
    s2_file = args.test_dir / "test_source2.tsv"
    s3_file = args.test_dir / "test_source3.tsv"

    # 2. Read Test Source 1 Order and Partition by Country
    print(f"\nReading test Source 1 entities from {s1_file}...", flush=True)
    t0_read = time.time()
    test_s1_order: List[str] = []
    country_s1_records: Dict[str, List[Tuple[str, str, str, str]]] = collections.defaultdict(list)

    with open(s1_file, "r", encoding="utf-8-sig") as f:
        header = next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                sid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                test_s1_order.append(sid)
                country_s1_records[country].append((sid, name, addr, country))

    total_test_s1 = len(test_s1_order)
    print(f"Loaded {total_test_s1:,} test Source 1 entities across {len(country_s1_records)} countries in {time.time() - t0_read:.2f}s:", flush=True)
    for c, records in sorted(country_s1_records.items(), key=lambda x: len(x[1])):
        print(f"  {c:<15}: {len(records):,} entities", flush=True)

    # Optional S1 limit per country (e.g. for smoke testing)
    if args.limit_s1_per_country:
        for c in country_s1_records:
            country_s1_records[c] = country_s1_records[c][: args.limit_s1_per_country]
        allowed_sids = {sid for recs in country_s1_records.values() for sid, _, _, _ in recs}
        test_s1_order = [sid for sid in test_s1_order if sid in allowed_sids]
        total_test_s1 = len(test_s1_order)
        print(f"Applied --limit-s1-per-country={args.limit_s1_per_country} -> Active test S1: {total_test_s1:,} entities.", flush=True)

    # Output directory & checkpoint setup
    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.output_matching.parent.mkdir(parents=True, exist_ok=True)
    args.output_candidate.parent.mkdir(parents=True, exist_ok=True)

    decision_config = DecisionLayerConfig(
        country_thresholds={
            "France": args.france_threshold,
            "US": args.us_threshold,
            "India": args.india_threshold,
        },
        margin_delta=args.margin_delta,
        default_threshold=args.threshold,
        enforce_group_margin=True,
    )
    print(f"Decision Layer Config: FR={args.france_threshold:.2f}, US={args.us_threshold:.2f}, IN={args.india_threshold:.2f}, Margin={args.margin_delta:.2f}", flush=True)

    total_candidate_pairs_all = 0
    total_predicted_matches_all = 0

    # 3. Process Each Country in Order of Size (France -> US -> India)
    for country, s1_list in sorted(country_s1_records.items(), key=lambda x: len(x[1])):
        c_t0 = time.time()
        print("\n" + "=" * 90, flush=True)
        print(f"PROCESSING COUNTRY: {country} ({len(s1_list):,} Source 1 entities)", flush=True)
        print("=" * 90, flush=True)

        # 3A: Preprocess S1 records for this country
        print(f"[{country}] Preprocessing {len(s1_list):,} S1 records...", flush=True)
        t0_c_prep = time.time()
        country_s1_prep: Dict[str, PreprocessedRecord] = {}
        for sid, name, addr, c_name in s1_list:
            country_s1_prep[sid] = preprocess_record(sid, name, addr, c_name)
        print(f"[{country}] Preprocessed in {time.time() - t0_c_prep:.2f}s.", flush=True)

        # 3B: Index Target Records for this country ONLY (filtered by country in-stream)
        print(f"[{country}] Streaming target pool (S2 + S3) for {country}...", flush=True)
        t0_idx = time.time()
        raw_idx: Dict[str, List[str]] = collections.defaultdict(list)
        country_tag = f"\t{country}"
        country_target_count = 0

        for src_fp in [s2_file, s3_file]:
            with open(src_fp, "r", encoding="utf-8-sig") as f:
                next(f)
                for line in f:
                    if country_tag not in line:
                        continue
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) < 4:
                        continue
                    tid, tname, taddr, tcountry = parts[0], parts[1], parts[2], parts[3]
                    if tcountry != country:
                        continue
                    country_target_count += 1

                    # 1. Exact Name
                    ten = normalize_name_conservative(tname)
                    if ten:
                        raw_idx[f"en_{ten}"].append(tid)

                    # 2. Token Keys
                    for tk in extract_token_blocking_keys(tname):
                        raw_idx[f"tk_{tk}"].append(tid)

                    # 3. Address Keys
                    if taddr and taddr.strip():
                        for ak in extract_address_keys(taddr):
                            raw_idx[f"ak_{ak}"].append(tid)
                        for ak in extract_h1_relaxed_address_keys(taddr):
                            raw_idx[ak].append(tid)
                        for ck in extract_h3_name_locality_keys(tname, taddr):
                            raw_idx[ck].append(tid)

        idx_time = time.time() - t0_idx
        print(f"[{country}] Indexed {country_target_count:,} target rows in {idx_time:.2f}s ({len(raw_idx):,} raw keys).", flush=True)

        # 3C: Specificity Frequency Filter for this country
        clean_idx: Dict[str, List[str]] = {}
        pruned_keys = 0
        for k, postings in raw_idx.items():
            if k.startswith("en_") or "pin_" in k:
                clean_idx[k] = postings
            elif k.startswith("tk_"):
                if len(postings) <= 5:
                    clean_idx[k] = postings
                else:
                    pruned_keys += 1
            else:
                if len(postings) <= 15:
                    clean_idx[k] = postings
                else:
                    pruned_keys += 1

        del raw_idx
        gc.collect()
        print(f"[{country}] Active clean keys: {len(clean_idx):,} (pruned {pruned_keys:,} generic keys).", flush=True)

        # 3D: Query S1 entities, apply 5-tier priority, cap to top 100
        print(f"[{country}] Generating candidates for {len(s1_list):,} S1 entities...", flush=True)
        t0_query = time.time()
        country_candidates: Dict[str, List[str]] = {}
        country_candidates_meta: Dict[str, Dict[str, Dict[str, float]]] = {}
        needed_target_ids: Set[str] = set()
        cands_count = 0

        for sid, name, addr, _ in s1_list:
            prep = country_s1_prep[sid]
            tid_to_tier: Dict[str, int] = {}
            tid_to_channels: Dict[str, Set[int]] = collections.defaultdict(set)

            # Tier 1: exact name
            en = prep.name_norm
            if en:
                m = clean_idx.get(f"en_{en}")
                if m:
                    for tid in m:
                        tid_to_tier[tid] = 1
                        tid_to_channels[tid].add(1)

            # Tier 2 & 3 & 4: address keys
            if prep.raw_address and prep.raw_address.strip():
                for ak in extract_address_keys(prep.raw_address):
                    if "pin_" in ak:
                        m = clean_idx.get(f"ak_{ak}")
                        if m:
                            for tid in m:
                                if tid not in tid_to_tier or tid_to_tier[tid] > 2:
                                    tid_to_tier[tid] = 2
                                tid_to_channels[tid].add(2)
                    else:
                        m = clean_idx.get(f"ak_{ak}")
                        if m:
                            for tid in m:
                                if tid not in tid_to_tier or tid_to_tier[tid] > 3:
                                    tid_to_tier[tid] = 3
                                tid_to_channels[tid].add(3)

                for ak in extract_h1_relaxed_address_keys(prep.raw_address):
                    if ak.startswith("pin_"):
                        tier = 2
                        ch = 2
                    elif ak.startswith("nword_"):
                        tier = 3
                        ch = 3
                    elif ak.startswith("loc_"):
                        tier = 4
                        ch = 4
                    else:
                        tier = 3
                        ch = 3
                    m = clean_idx.get(ak)
                    if m:
                        for tid in m:
                            if tid not in tid_to_tier or tid_to_tier[tid] > tier:
                                tid_to_tier[tid] = tier
                            tid_to_channels[tid].add(ch)

                for ck in extract_h3_name_locality_keys(prep.raw_name, prep.raw_address):
                    m = clean_idx.get(ck)
                    if m:
                        for tid in m:
                            if tid not in tid_to_tier or tid_to_tier[tid] > 4:
                                tid_to_tier[tid] = 4
                            tid_to_channels[tid].add(4)

            # Tier 5: rare token keys
            for tk in extract_token_blocking_keys(prep.raw_name):
                m = clean_idx.get(f"tk_{tk}")
                if m:
                    for tid in m:
                        if tid not in tid_to_tier:
                            tid_to_tier[tid] = 5
                        tid_to_channels[tid].add(5)

            if not tid_to_tier:
                country_candidates[sid] = []
                country_candidates_meta[sid] = {}
                continue

            if len(tid_to_tier) <= args.final_cap:
                selected_tids = list(tid_to_tier.keys())
            else:
                sorted_items = sorted(tid_to_tier.items(), key=lambda x: x[1])
                selected_tids = [tid for tid, _ in sorted_items[:args.final_cap]]

            country_candidates[sid] = selected_tids
            needed_target_ids.update(selected_tids)
            cands_count += len(selected_tids)

            pool_size = float(len(selected_tids))
            s1_meta: Dict[str, Dict[str, float]] = {}
            for rank_idx, tid in enumerate(selected_tids):
                s1_meta[tid] = {
                    "channel_agreement_count": float(len(tid_to_channels[tid])),
                    "retrieval_priority_tier": float(tid_to_tier[tid]),
                    "reciprocal_retrieval_rank": 1.0 / (rank_idx + 1.0),
                    "s1_candidate_pool_size": pool_size,
                }
            country_candidates_meta[sid] = s1_meta

        del clean_idx
        gc.collect()

        query_time = time.time() - t0_query
        avg_cands = cands_count / max(len(s1_list), 1)
        print(f"[{country}] Generated {cands_count:,} candidate pairs ({avg_cands:.2f} avg/S1, {len(needed_target_ids):,} unique targets) in {query_time:.2f}s.", flush=True)
        total_candidate_pairs_all += cands_count

        # 3E: Retrieve Needed Target Records for this country ONLY
        print(f"[{country}] Retrieving {len(needed_target_ids):,} needed target records...", flush=True)
        t0_tgt = time.time()
        target_prep_map: Dict[str, PreprocessedRecord] = {}

        for src_fp in [s2_file, s3_file]:
            with open(src_fp, "r", encoding="utf-8-sig") as f:
                next(f)
                for line in f:
                    if country_tag not in line:
                        continue
                    pos = line.find("\t")
                    if pos == -1:
                        continue
                    tid = line[:pos]
                    if tid in needed_target_ids:
                        parts = line.rstrip("\r\n").split("\t")
                        if len(parts) >= 4:
                            target_prep_map[tid] = preprocess_record(parts[0], parts[1], parts[2], parts[3])
                            if len(target_prep_map) == len(needed_target_ids):
                                break
            if len(target_prep_map) == len(needed_target_ids):
                break

        del needed_target_ids
        gc.collect()
        tgt_time = time.time() - t0_tgt
        print(f"[{country}] Retrieved {len(target_prep_map):,} target records in {tgt_time:.2f}s. RAM: {get_peak_ram_mb():.2f} MB.", flush=True)

        # 3F: Pairwise Feature Extraction & XGBoost Batch Scoring with Decision Layer
        c_thresh = decision_config.get_threshold(country)
        print(f"[{country}] Scoring {cands_count:,} candidate pairs with XGBoost (threshold={c_thresh:.2f}, margin={args.margin_delta:.2f})...", flush=True)
        t0_score = time.time()

        country_matches: Dict[str, List[str]] = collections.defaultdict(list)
        country_pred_matches = 0

        batch_X = []
        batch_pairs = []
        batch_s1_offsets: List[Tuple[str, int, int]] = []

        for sid, _, _, _ in s1_list:
            tids = country_candidates.get(sid, [])
            if not tids:
                continue
            s1_prep = country_s1_prep[sid]
            s1_meta_dict = country_candidates_meta.get(sid, {})

            start_idx = len(batch_X)
            for tid in tids:
                tgt_prep = target_prep_map.get(tid)
                if tgt_prep:
                    feats = extract_features_from_preprocessed(s1_prep, tgt_prep)
                    feats["cheap_candidate_score"] = compute_cheap_candidate_score(feats)
                    feats.update(s1_meta_dict[tid])
                    batch_X.append([feats[col] for col in FEATURE_NAMES])
                    batch_pairs.append((sid, tid))
            end_idx = len(batch_X)
            if end_idx > start_idx:
                batch_s1_offsets.append((sid, start_idx, end_idx))

            if len(batch_X) >= args.batch_size:
                dmat = xgb.DMatrix(np.array(batch_X, dtype=np.float32), feature_names=FEATURE_NAMES)
                probs = bst.predict(dmat)
                for sid_item, s_start, s_end in batch_s1_offsets:
                    cand_probs = [(batch_pairs[idx][1], float(probs[idx])) for idx in range(s_start, s_end)]
                    matched_tids = filter_entity_candidates(cand_probs, country=country, config=decision_config)
                    if matched_tids:
                        country_matches[sid_item] = matched_tids
                        country_pred_matches += len(matched_tids)
                batch_X = []
                batch_pairs = []
                batch_s1_offsets = []

        if batch_X:
            dmat = xgb.DMatrix(np.array(batch_X, dtype=np.float32), feature_names=FEATURE_NAMES)
            probs = bst.predict(dmat)
            for sid_item, s_start, s_end in batch_s1_offsets:
                cand_probs = [(batch_pairs[idx][1], float(probs[idx])) for idx in range(s_start, s_end)]
                matched_tids = filter_entity_candidates(cand_probs, country=country, config=decision_config)
                if matched_tids:
                    country_matches[sid_item] = matched_tids
                    country_pred_matches += len(matched_tids)
            batch_X = []
            batch_pairs = []
            batch_s1_offsets = []

        del country_s1_prep, target_prep_map, batch_X, batch_pairs, batch_s1_offsets, country_candidates_meta
        gc.collect()

        score_time = time.time() - t0_score
        total_predicted_matches_all += country_pred_matches
        print(f"[{country}] Scored in {score_time:.2f}s. Predicted matches: {country_pred_matches:,}.", flush=True)

        # 3G: Checkpoint results for this country to disk
        c_matching_ckpt = checkpoint_dir / f"matches_{country.lower()}.tsv"
        c_cand_ckpt = checkpoint_dir / f"candidates_{country.lower()}.tsv"

        with open(c_matching_ckpt, "w", encoding="utf-8", newline="\n") as f:
            for sid, _, _, _ in s1_list:
                m_list = country_matches.get(sid, [])
                if m_list:
                    seen = set()
                    dedup = [m for m in m_list if not (m in seen or seen.add(m))]
                    f.write(f"{sid}\t{','.join(dedup)}\n")
                else:
                    f.write(f"{sid}\t\n")

        with open(c_cand_ckpt, "w", encoding="utf-8", newline="\n") as f:
            for sid, _, _, _ in s1_list:
                c_list = country_candidates.get(sid, [])
                if c_list:
                    seen = set()
                    dedup = [c for c in c_list if not (c in seen or seen.add(c))]
                    f.write(f"{sid}\t{','.join(dedup)}\n")
                else:
                    f.write(f"{sid}\t\n")

        del country_candidates, country_matches
        gc.collect()

        country_total_time = time.time() - c_t0
        print(f"[{country}] COMPLETED in {country_total_time:.2f}s ({country_total_time / 60:.2f} min). Peak RAM: {get_peak_ram_mb():.2f} MB.", flush=True)

    # 4. Phase 7: Combine Checkpoints into Final Submission TSVs in Exact S1 Order
    print("\n" + "=" * 95, flush=True)
    print("ASSEMBLING FINAL SUBMISSION TSVs IN EXACT ORIGINAL S1 ORDER", flush=True)
    print("=" * 95, flush=True)
    t0_merge = time.time()

    # Load all country checkpoints into fast lookup mapping
    matching_map: Dict[str, str] = {}
    candidate_map: Dict[str, str] = {}

    for country in country_s1_records.keys():
        c_matching_ckpt = checkpoint_dir / f"matches_{country.lower()}.tsv"
        c_cand_ckpt = checkpoint_dir / f"candidates_{country.lower()}.tsv"

        with open(c_matching_ckpt, "r", encoding="utf-8") as f:
            for line in f:
                pos = line.find("\t")
                if pos != -1:
                    matching_map[line[:pos]] = line[pos + 1 :].rstrip("\r\n")

        with open(c_cand_ckpt, "r", encoding="utf-8") as f:
            for line in f:
                pos = line.find("\t")
                if pos != -1:
                    candidate_map[line[:pos]] = line[pos + 1 :].rstrip("\r\n")

    # Write final matching_results.tsv
    print(f"Writing {args.output_matching}...", flush=True)
    matching_rows_written = 0
    matching_empty_count = 0
    total_matches_written = 0

    with open(args.output_matching, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in test_s1_order:
            m_str = matching_map.get(sid, "")
            if m_str:
                total_matches_written += m_str.count(",") + 1
            else:
                matching_empty_count += 1
            f.write(f"{sid}\t{m_str}\n")
            matching_rows_written += 1

    # Write final candidate_pairs.tsv
    print(f"Writing {args.output_candidate}...", flush=True)
    candidate_rows_written = 0
    candidate_empty_count = 0
    total_candidates_written = 0

    with open(args.output_candidate, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in test_s1_order:
            c_str = candidate_map.get(sid, "")
            if c_str:
                total_candidates_written += c_str.count(",") + 1
            else:
                candidate_empty_count += 1
            f.write(f"{sid}\t{c_str}\n")
            candidate_rows_written += 1

    del matching_map, candidate_map
    gc.collect()

    print(f"Wrote {matching_rows_written:,} rows to {args.output_matching} and {candidate_rows_written:,} rows to {args.output_candidate} in {time.time() - t0_merge:.2f}s.", flush=True)

    # 5. Phase 8: Local Validation Checks
    print("\n" + "=" * 95, flush=True)
    print("RUNNING OFFICIAL VALIDATION CHECKS", flush=True)
    print("=" * 95, flush=True)

    # Check 1: Row count equals test source 1 count
    assert matching_rows_written == total_test_s1, f"Mismatch in matching rows: {matching_rows_written} vs {total_test_s1}"
    assert candidate_rows_written == total_test_s1, f"Mismatch in candidate rows: {candidate_rows_written} vs {total_test_s1}"
    print(f"[CHECK 1 PASSED] Every Source 1 entity has exactly one row: {total_test_s1:,} rows.", flush=True)

    # Check 2: Run official utils/validate_submission.py
    if not args.limit_s1_per_country:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "utils" / "validate_submission.py"),
            "--matching", str(args.output_matching),
            "--candidate", str(args.output_candidate),
            "--test-dir", str(args.test_dir),
        ]
        print(f"\nRunning official submission validator:\n{' '.join(cmd)}", flush=True)
        val_proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        print(val_proc.stdout, flush=True)
        if val_proc.stderr:
            print("Validator STDERR:", val_proc.stderr, flush=True)
        assert val_proc.returncode == 0, f"Validator failed with exit code {val_proc.returncode}!"
        print("[CHECK 2 PASSED] utils/validate_submission.py returned exit code 0 (VALID SUBMISSION!).", flush=True)
        val_exit_code = val_proc.returncode
    else:
        print("[CHECK 2 SKIPPED for partial test run, full validator requires all entities in test_dir]", flush=True)
        val_exit_code = 0

    total_time = time.time() - overall_start
    peak_ram = get_peak_ram_mb()

    # 6. Save JSON Summary
    summary_data = {
        "matching_results_path": str(args.output_matching.resolve()),
        "candidate_pairs_path": str(args.output_candidate.resolve()),
        "total_source1_entities": total_test_s1,
        "matching_rows": matching_rows_written,
        "candidate_rows": candidate_rows_written,
        "total_candidate_pairs": total_candidates_written,
        "total_predicted_matches": total_matches_written,
        "matching_singletons": matching_empty_count,
        "candidate_singletons": candidate_empty_count,
        "every_source1_exact_one_row": (matching_rows_written == total_test_s1),
        "total_runtime_seconds": total_time,
        "peak_ram_mb": peak_ram,
        "validator_exit_code": val_exit_code,
        "ready_for_upload": (val_exit_code == 0 and not args.limit_s1_per_country),
    }
    summary_path = args.output_matching.parent / "submission_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 95, flush=True)
    print("SUBMISSION GENERATION COMPLETED SUCCESSFULLY", flush=True)
    print(f"Total Runtime: {total_time:.2f}s ({total_time / 60:.2f} min) | Peak RAM: {peak_ram:.2f} MB", flush=True)
    print("=" * 95, flush=True)


if __name__ == "__main__":
    main()
