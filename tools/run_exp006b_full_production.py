#!/usr/bin/env python3
"""EXP-006B: High-Speed Full Production Runner via Persistent Target Inverted Indexes.

Implements the verified Local Champion Pipeline (Macro F0.5 = 0.845653):
1. Target Retrieval: Inverted index lookups using pre-built persistent indexes:
   - France: artifacts/target_index_france.pkl (1,434,993 targets)
   - US: artifacts/target_index_us.pkl (3,817,031 targets)
   - India: artifacts/target_index_india.pkl (4,717,565 targets)
2. Compound Plot/Door Number Retrieval: Recovers unretrieved entities via specific cnum_ signatures.
3. Channel Consensus First Candidate Ranking: Prioritizes multi-channel agreement over single modalities.
4. Model Matcher: Frozen XGBoost matcher (artifacts/xgb_matcher_exp004b.json, 26 features).
5. Decision Layer: EXP-003B country thresholds (FR=0.45, US=0.65, IN=0.55, margin=0.30).
6. Selective Residual Recovery: Phase 2 recovery for unresolved S1 entities with zero initial predictions.
7. Isolated Production Artifacts:
   - Candidates: output/exp006b/candidate_pairs.tsv
   - Predictions: output/exp006b/matching_results.tsv and output/matching_results_EXP006B.tsv
   - V3 baseline files (output/v3/* and output/matching_results_FINAL.tsv) remain strictly untouched.
"""

import argparse
import array
from collections import Counter, defaultdict
import ctypes
from ctypes import wintypes
import gc
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking.normalizers import (
    extract_address_keys,
    extract_token_blocking_keys,
    normalize_name_conservative,
)
from src.blocking.retrieval import (
    apply_selective_residual_recovery,
    extract_compound_number_keys,
    extract_record_blocking_keys,
    rank_candidates_channel_consensus_first,
)
from src.matching.candidate_compression import (
    compute_cheap_candidate_score,
    extract_features_from_preprocessed,
    fast_score_candidate,
    preprocess_record,
    PreprocessedRecord,
)
from src.matching.decision_layer import (
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


def get_current_ram_mb() -> float:
    """Return current working set RAM in MB via Windows API."""
    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=Path("dataset/test"))
    parser.add_argument("--manifest-path", type=Path, default=Path("artifacts/target_index_manifest.json"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/xgb_matcher_exp004b.json"))
    parser.add_argument("--output-matching", type=Path, default=Path("output/exp006b/matching_results.tsv"))
    parser.add_argument("--output-candidate", type=Path, default=Path("output/exp006b/candidate_pairs.tsv"))
    parser.add_argument("--root-matching", type=Path, default=Path("output/matching_results_EXP006B.tsv"))
    parser.add_argument("--france-threshold", type=float, default=0.45)
    parser.add_argument("--us-threshold", type=float, default=0.65)
    parser.add_argument("--india-threshold", type=float, default=0.55)
    parser.add_argument("--margin-delta", type=float, default=0.30)
    parser.add_argument("--phase1-cap", type=int, default=100)
    parser.add_argument("--residual-recovery-cap", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--limit-s1-per-country", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()

    overall_start = time.time()
    print("=" * 95, flush=True)
    print("EXP-006B FULL PRODUCTION PIPELINE (HIGH-SPEED INDEX RETRIEVAL)", flush=True)
    print("=" * 95, flush=True)
    print(f"Model Path        : {args.model_path}")
    print(f"Index Manifest    : {args.manifest_path}")
    print(f"Matching Output   : {args.output_matching}")
    print(f"Candidate Output  : {args.output_candidate}")
    print(f"Root Matching     : {args.root_matching}")
    print(f"Decision Layer    : FR={args.france_threshold:.2f}, US={args.us_threshold:.2f}, IN={args.india_threshold:.2f}, Margin={args.margin_delta:.2f}")
    print(f"Phase 1 Cap       : {args.phase1_cap} | Phase 2 Recovery Cap: {args.residual_recovery_cap}")
    print(f"Threads           : {args.num_threads}")
    print(f"Initial RAM       : {get_current_ram_mb():.1f} MB")
    print("=" * 95, flush=True)

    # 1. Verify files and models
    assert args.model_path.is_file(), f"Missing model: {args.model_path}"
    assert args.manifest_path.is_file(), f"Missing index manifest: {args.manifest_path}"

    with open(args.manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["status"] == "ALL_INDEXES_VERIFIED", "Manifest status not verified!"

    bst = xgb.Booster()
    bst.load_model(str(args.model_path))
    bst.set_param({"nthread": args.num_threads})
    print(f"Loaded trained 26-feature XGBoost model from {args.model_path}.", flush=True)

    decision_config = DecisionLayerConfig(
        country_thresholds={
            "France": args.france_threshold,
            "US": args.us_threshold,
            "India": args.india_threshold,
        },
        margin_delta=args.margin_delta,
        default_threshold=0.60,
        enforce_group_margin=True,
    )

    s1_file = args.test_dir / "test_source1.tsv"
    s2_file = args.test_dir / "test_source2.tsv"
    s3_file = args.test_dir / "test_source3.tsv"

    # 2. Read Test Source 1 in exact order
    print(f"\nReading test Source 1 records from {s1_file}...", flush=True)
    t0_s1 = time.time()
    test_s1_order: List[str] = []
    country_s1: Dict[str, List[Tuple[str, str, str, str]]] = defaultdict(list)

    with open(s1_file, "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                sid, name, addr, c = parts[0], parts[1], parts[2], parts[3]
                test_s1_order.append(sid)
                country_s1[c].append((sid, name, addr, c))

    if args.limit_s1_per_country:
        print(f"Applying --limit-s1-per-country={args.limit_s1_per_country}...", flush=True)
        for c in country_s1:
            country_s1[c] = country_s1[c][:args.limit_s1_per_country]
        allowed = {sid for recs in country_s1.values() for sid, _, _, _ in recs}
        test_s1_order = [s for s in test_s1_order if s in allowed]

    total_s1 = len(test_s1_order)
    print(f"Loaded {total_s1:,} active test Source 1 entities in {time.time() - t0_s1:.2f}s.", flush=True)

    args.output_matching.parent.mkdir(parents=True, exist_ok=True)
    args.output_candidate.parent.mkdir(parents=True, exist_ok=True)

    all_matched_predictions: Dict[str, List[str]] = {}
    all_final_candidates: Dict[str, List[str]] = {}

    stats_by_country: Dict[str, Dict[str, Any]] = {}

    # 3. Process each country sequentially (France -> US -> India)
    for country in ["France", "US", "India"]:
        s1_list = country_s1.get(country, [])
        if not s1_list:
            continue

        c_start = time.time()
        print("\n" + "=" * 90, flush=True)
        print(f"PROCESSING {country.upper()}: {len(s1_list):,} Source 1 Entities", flush=True)
        print("=" * 90, flush=True)

        idx_info = manifest["indexes"][country]
        idx_path = PROJECT_ROOT / idx_info["index_path"]
        print(f"Loading persistent target index from {idx_path}...", flush=True)
        t0_idx = time.time()
        with open(idx_path, "rb") as f:
            idx_data = pickle.load(f)
        target_ids: List[str] = idx_data["target_ids"]
        inverted_index: Dict[str, array.array] = idx_data["inverted_index"]
        del idx_data
        gc.collect()
        print(f"Loaded {len(target_ids):,} targets and {len(inverted_index):,} keys in {time.time() - t0_idx:.2f}s. RAM: {get_current_ram_mb():.1f} MB.", flush=True)

        # Flag generic keys (tk_ with fanout >= 15)
        generic_keys = {k for k, post in inverted_index.items() if k.startswith("tk_") and len(post) >= 15}
        print(f"Generic token keys flagged (fanout >= 15): {len(generic_keys):,}", flush=True)

        # Preprocess S1 records
        print(f"Preprocessing {len(s1_list):,} {country} S1 records...", flush=True)
        t0_prep = time.time()
        s1_prep_map: Dict[str, PreprocessedRecord] = {
            sid: preprocess_record(sid, name, addr, country)
            for sid, name, addr, _ in s1_list
        }
        print(f"Preprocessed S1 in {time.time() - t0_prep:.2f}s.", flush=True)

        # Inverted Index Lookups & Candidate Tracking
        print(f"Querying persistent target index and building candidate pools...", flush=True)
        t0_query = time.time()
        s1_phase1_candidates: Dict[str, List[Tuple[str, Dict[str, float]]]] = {}
        s1_residual_candidates: Dict[str, List[Dict[str, Any]]] = {}

        include_cnum = idx_info.get("compound_numbers_enabled", True)

        for s1_idx, (sid, name, addr, _) in enumerate(s1_list):
            if s1_idx > 0 and s1_idx % 50_000 == 0:
                print(f"  [{country} Query Progress] {s1_idx:,}/{len(s1_list):,} S1 queried in {time.time() - t0_query:.1f}s | RAM: {get_current_ram_mb():.1f} MB", flush=True)

            keys_dict = extract_record_blocking_keys(
                name=name,
                address=addr,
                country=country,
                include_compound_numbers=include_cnum,
            )

            cand_hits: Dict[int, Dict[str, Any]] = {}
            for k, tier in keys_dict.items():
                postings = inverted_index.get(k)
                if not postings:
                    continue

                fanout = len(postings)
                # Frequency filter: prune generic high-fanout blocking keys
                if k.startswith("tk_") and fanout > 5:
                    continue
                if not (k.startswith("en_") or "pin_" in k or k.startswith("cnum_")) and fanout > 15:
                    continue

                ch = 1 if k.startswith("en_") else (2 if "cnum_" in k or "pin_" in k else 3)
                spec = 1.0 / max(fanout, 1)

                for t_idx in postings:
                    if t_idx not in cand_hits:
                        cand_hits[t_idx] = {
                            "t_idx": t_idx,
                            "ch_set": {ch},
                            "best_tier": tier,
                            "spec_sum": spec,
                        }
                    else:
                        h = cand_hits[t_idx]
                        h["ch_set"].add(ch)
                        if tier < h["best_tier"]:
                            h["best_tier"] = tier
                        h["spec_sum"] += spec

            if not cand_hits:
                s1_phase1_candidates[sid] = []
                s1_residual_candidates[sid] = []
                continue

            raw_cands = [
                {
                    "tid": target_ids[t_idx],
                    "t_idx": t_idx,
                    "ch_count": len(h["ch_set"]),
                    "best_tier": h["best_tier"],
                    "score": 0.50,
                    "spec_sum": h["spec_sum"],
                }
                for t_idx, h in cand_hits.items()
            ]

            ranked = rank_candidates_channel_consensus_first(raw_cands, cap=args.phase1_cap)
            s1_phase1_candidates[sid] = ranked

            # Store compact informative candidates beyond cap for Phase 2 residual recovery
            sel_tids = {t for t, _ in ranked}
            remaining = [c for c in raw_cands if c["tid"] not in sel_tids]
            informative = [
                c for c in remaining
                if c["ch_count"] >= 2 or c["best_tier"] <= 3
            ]
            informative.sort(
                key=lambda c: (
                    -c["ch_count"],
                    c["best_tier"],
                    -c["score"],
                    -c["spec_sum"],
                )
            )
            s1_residual_candidates[sid] = informative[:args.residual_recovery_cap]

        # Index no longer needed in RAM for this country
        del inverted_index
        gc.collect()

        cands_phase1_count = sum(len(c) for c in s1_phase1_candidates.values())
        print(f"Phase 1 generated {cands_phase1_count:,} candidates in {time.time() - t0_query:.2f}s. RAM: {get_current_ram_mb():.1f} MB.", flush=True)

        # Collect needed target IDs for Phase 1
        needed_target_ids = {tid for c_list in s1_phase1_candidates.values() for tid, _ in c_list}
        print(f"Retrieving {len(needed_target_ids):,} preprocessed target records for Phase 1...", flush=True)
        t0_tgt = time.time()
        target_prep_map: Dict[str, PreprocessedRecord] = {}

        country_tag = f"\t{country}"
        for src_fp in [s2_file, s3_file]:
            with open(src_fp, "r", encoding="utf-8-sig") as f:
                next(f)
                for line in f:
                    if country_tag not in line:
                        continue
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 4 and parts[0] in needed_target_ids:
                        target_prep_map[parts[0]] = preprocess_record(parts[0], parts[1], parts[2], parts[3])
                        if len(target_prep_map) == len(needed_target_ids):
                            break
            if len(target_prep_map) == len(needed_target_ids):
                break

        print(f"Retrieved {len(target_prep_map):,} target records in {time.time() - t0_tgt:.2f}s.", flush=True)

        # Phase 1 Scoring with XGBoost
        c_thresh = decision_config.get_threshold(country)
        print(f"Scoring Phase 1 candidate pairs with XGBoost (threshold={c_thresh:.2f}, margin={args.margin_delta:.2f})...", flush=True)
        t0_score = time.time()

        phase1_predictions: Dict[str, Set[str]] = defaultdict(set)
        batch_X = []
        batch_pairs: List[Tuple[str, str]] = []
        s1_pair_offsets: List[Tuple[str, int, int]] = []

        for sid, _, _, _ in s1_list:
            cands = s1_phase1_candidates.get(sid, [])
            if not cands:
                continue
            s1_prep = s1_prep_map[sid]
            start_off = len(batch_X)
            for tid, meta in cands:
                tgt_prep = target_prep_map.get(tid)
                if tgt_prep:
                    feats = extract_features_from_preprocessed(s1_prep, tgt_prep)
                    feats["cheap_candidate_score"] = compute_cheap_candidate_score(feats)
                    feats.update(meta)
                    batch_X.append([feats[col] for col in FEATURE_NAMES])
                    batch_pairs.append((sid, tid))

            end_off = len(batch_X)
            if end_off > start_off:
                s1_pair_offsets.append((sid, start_off, end_off))

            if len(batch_X) >= args.batch_size:
                dmat = xgb.DMatrix(np.array(batch_X, dtype=np.float32), feature_names=FEATURE_NAMES)
                probs = bst.predict(dmat)
                for s_id, s_off, e_off in s1_pair_offsets:
                    entity_candidates = [
                        (batch_pairs[k][1], float(probs[k]))
                        for k in range(s_off, e_off)
                    ]
                    accepted = filter_entity_candidates(entity_candidates, c_thresh, decision_config)
                    if accepted:
                        phase1_predictions[s_id].update(accepted)
                batch_X = []
                batch_pairs = []
                s1_pair_offsets = []

        if batch_X:
            dmat = xgb.DMatrix(np.array(batch_X, dtype=np.float32), feature_names=FEATURE_NAMES)
            probs = bst.predict(dmat)
            for s_id, s_off, e_off in s1_pair_offsets:
                entity_candidates = [
                    (batch_pairs[k][1], float(probs[k]))
                    for k in range(s_off, e_off)
                ]
                accepted = filter_entity_candidates(entity_candidates, c_thresh, decision_config)
                if accepted:
                    phase1_predictions[s_id].update(accepted)

        p1_time = time.time() - t0_score
        p1_matched_entities = len(phase1_predictions)
        p1_links = sum(len(v) for v in phase1_predictions.values())
        print(f"Phase 1 complete in {p1_time:.2f}s: {p1_matched_entities:,}/{len(s1_list):,} entities matched ({p1_links:,} links).", flush=True)

        # Phase 2: Selective Residual Recovery
        residual_s1 = [sid for sid, _, _, _ in s1_list if not phase1_predictions.get(sid)]
        print(f"\nPhase 2 Selective Residual Recovery: {len(residual_s1):,} zero-prediction entities...", flush=True)
        t0_rec = time.time()

        recovered_cands_by_sid: Dict[str, List[Tuple[str, Dict[str, float]]]] = {}
        recovery_needed_tids: Set[str] = set()

        for sid in residual_s1:
            top_rec = s1_residual_candidates.get(sid, [])
            if not top_rec:
                continue

            base_items = s1_phase1_candidates.get(sid, [])
            pool_sz = float(len(base_items) + len(top_rec))
            extra = []
            for rank_idx, c in enumerate(top_rec, start=len(base_items)):
                meta = {
                    "channel_agreement_count": float(c["ch_count"]),
                    "retrieval_priority_tier": float(c["best_tier"]),
                    "reciprocal_retrieval_rank": 1.0 / (rank_idx + 1.0),
                    "s1_candidate_pool_size": pool_sz,
                }
                extra.append((c["tid"], meta))
                recovery_needed_tids.add(c["tid"])

            if extra:
                recovered_cands_by_sid[sid] = extra

        # Retrieve additional targets needed for recovery
        missing_recovery_tids = recovery_needed_tids - set(target_prep_map.keys())
        if missing_recovery_tids:
            print(f"Retrieving {len(missing_recovery_tids):,} additional targets for Phase 2 recovery...", flush=True)
            for src_fp in [s2_file, s3_file]:
                with open(src_fp, "r", encoding="utf-8-sig") as f:
                    next(f)
                    for line in f:
                        if country_tag not in line:
                            continue
                        parts = line.rstrip("\r\n").split("\t")
                        if len(parts) >= 4 and parts[0] in missing_recovery_tids:
                            target_prep_map[parts[0]] = preprocess_record(parts[0], parts[1], parts[2], parts[3])
                            if len(target_prep_map) >= len(needed_target_ids) + len(missing_recovery_tids):
                                break

        # Score recovered residual candidates
        rec_batch_X = []
        rec_batch_pairs: List[Tuple[str, str]] = []
        rec_offsets: List[Tuple[str, int, int]] = []
        recovered_links_count = 0

        for sid, extra_cands in recovered_cands_by_sid.items():
            s1_prep = s1_prep_map[sid]
            start_off = len(rec_batch_X)
            for tid, meta in extra_cands:
                tgt_prep = target_prep_map.get(tid)
                if tgt_prep:
                    feats = extract_features_from_preprocessed(s1_prep, tgt_prep)
                    feats["cheap_candidate_score"] = compute_cheap_candidate_score(feats)
                    feats.update(meta)
                    rec_batch_X.append([feats[col] for col in FEATURE_NAMES])
                    rec_batch_pairs.append((sid, tid))

            end_off = len(rec_batch_X)
            if end_off > start_off:
                rec_offsets.append((sid, start_off, end_off))

        if rec_batch_X:
            dmat = xgb.DMatrix(np.array(rec_batch_X, dtype=np.float32), feature_names=FEATURE_NAMES)
            probs = bst.predict(dmat)
            for s_id, s_off, e_off in rec_offsets:
                entity_candidates = [
                    (rec_batch_pairs[k][1], float(probs[k]))
                    for k in range(s_off, e_off)
                ]
                accepted = filter_entity_candidates(entity_candidates, c_thresh, decision_config)
                if accepted:
                    phase1_predictions[s_id].update(accepted)
                    recovered_links_count += len(accepted)

        print(f"Phase 2 recovered {recovered_links_count:,} additional links in {time.time() - t0_rec:.2f}s.", flush=True)

        # Assemble final candidate sets and predictions for this country
        final_matched_entities = len(phase1_predictions)
        final_links = sum(len(v) for v in phase1_predictions.values())
        country_candidate_count = 0

        for sid, _, _, _ in s1_list:
            base_cands = [t for t, _ in s1_phase1_candidates.get(sid, [])]
            extra_cands = [t for t, _ in recovered_cands_by_sid.get(sid, [])]
            all_cands = base_cands + extra_cands
            # Ensure every accepted prediction is guaranteed present in candidates
            preds = sorted(list(phase1_predictions.get(sid, set())))
            for p in preds:
                if p not in all_cands:
                    all_cands.append(p)

            all_final_candidates[sid] = all_cands
            all_matched_predictions[sid] = preds
            country_candidate_count += len(all_cands)

        c_elapsed = time.time() - c_start
        empty_in_country = len(s1_list) - final_matched_entities
        stats_by_country[country] = {
            "s1_count": len(s1_list),
            "candidates_count": country_candidate_count,
            "avg_candidates_per_s1": country_candidate_count / len(s1_list),
            "matched_entities": final_matched_entities,
            "empty_matches": empty_in_country,
            "predicted_links": final_links,
            "runtime_sec": c_elapsed,
        }

        print(f"Finished {country} in {c_elapsed:.2f}s ({c_elapsed / 60:.2f} min).")
        print(f"  Candidates : {country_candidate_count:,} (avg {country_candidate_count / len(s1_list):.2f}/S1)")
        print(f"  Matched S1 : {final_matched_entities:,} ({final_links:,} links, {empty_in_country:,} empty)")

        # Clean memory for next country
        del s1_prep_map, s1_phase1_candidates, s1_residual_candidates, target_prep_map, recovered_cands_by_sid
        gc.collect()

    # 4. Stream Final TSVs in Exact test_source1.tsv Order
    print("\n" + "=" * 95, flush=True)
    print("EMITTING FINAL SUBMISSION AND CANDIDATE PAIRS TSVS", flush=True)
    print("=" * 95, flush=True)

    tmp_matching = args.output_matching.with_suffix(".tmp")
    tmp_candidate = args.output_candidate.with_suffix(".tmp")

    print(f"Writing matching results to {args.output_matching}...", flush=True)
    total_empty_matches = 0
    total_predicted_links = 0

    with open(tmp_matching, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in test_s1_order:
            m_list = all_matched_predictions.get(sid, [])
            if not m_list:
                total_empty_matches += 1
                f_out.write(f"{sid}\t\n")
            else:
                total_predicted_links += len(m_list)
                f_out.write(f"{sid}\t{','.join(m_list)}\n")

    if args.output_matching.exists():
        args.output_matching.unlink()
    os.replace(tmp_matching, args.output_matching)

    # Mirror to root matching output if requested
    if args.root_matching:
        print(f"Mirroring matching results to root file: {args.root_matching}...", flush=True)
        with open(args.output_matching, "rb") as f_src, open(args.root_matching, "wb") as f_dst:
            while chunk := f_src.read(1024 * 1024):
                f_dst.write(chunk)

    print(f"Writing candidate pairs to {args.output_candidate}...", flush=True)
    total_candidate_pairs = 0
    with open(tmp_candidate, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in test_s1_order:
            c_list = all_final_candidates.get(sid, [])
            total_candidate_pairs += len(c_list)
            if c_list:
                f_out.write(f"{sid}\t{','.join(c_list)}\n")
            else:
                f_out.write(f"{sid}\t\n")

    if args.output_candidate.exists():
        args.output_candidate.unlink()
    os.replace(tmp_candidate, args.output_candidate)

    match_size_mb = os.path.getsize(args.output_matching) / (1024 * 1024)
    cand_size_mb = os.path.getsize(args.output_candidate) / (1024 * 1024)
    print(f"Matching Results: {args.output_matching} ({match_size_mb:.2f} MB)")
    print(f"Candidate Pairs : {args.output_candidate} ({cand_size_mb:.2f} MB)")

    # 5. Validation Check
    print("\nRunning official submission validator...", flush=True)
    validator_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "utils" / "validate_submission.py"),
        "--matching", str(args.output_matching),
        "--candidate", str(args.output_candidate),
        "--test-dir", str(args.test_dir),
    ]
    if not args.limit_s1_per_country:
        val_res = subprocess.run(validator_cmd, cwd=str(PROJECT_ROOT))
        assert val_res.returncode == 0, "Submission validator FAILED!"
    else:
        print("Note: Running validator in smoke-test mode (subset of rows).")
        subprocess.run(validator_cmd, cwd=str(PROJECT_ROOT))

    total_time = time.time() - overall_start
    print("\n" + "=" * 95, flush=True)
    print("EXP-006B FULL PRODUCTION INFERENCE COMPLETE!", flush=True)
    print(f"Total Source 1 Entities : {len(test_s1_order):,}")
    print(f"Total Candidate Pairs   : {total_candidate_pairs:,} (avg {total_candidate_pairs / len(test_s1_order):.2f}/S1)")
    print(f"Total Predicted Links   : {total_predicted_links:,}")
    print(f"Empty Matches           : {total_empty_matches:,} ({total_empty_matches / len(test_s1_order) * 100:.2f}%)")
    print(f"Total Wall Runtime      : {total_time:.2f}s ({total_time / 60:.2f} min)")
    print("=" * 95, flush=True)


if __name__ == "__main__":
    main()
