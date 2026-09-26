#!/usr/bin/env python3
"""EXP-004B: Zero-Candidate-Drift Retrieval Metadata Model Experiment.

Tests whether 4 orthogonal retrieval metadata features improve the XGBoost matcher
while preserving the EXACT existing candidate set (candidate_set_difference == 0):
  1. channel_agreement_count: number of distinct blocking channels (1 to 5)
  2. retrieval_priority_tier: best priority tier (1 to 6)
  3. reciprocal_retrieval_rank: 1.0 / (retrieval_rank + 1)
  4. s1_candidate_pool_size: total candidate pool size for S1

Strict Constraints:
- Zero external data, zero internet/entity lookup, zero external embeddings.
- Deterministic candidate-drift test: stops if candidate_set_difference != 0.
- Trains 26-feature XGBoost matcher on disjoint training split (1,000 entities).
- Evaluates on exact 2,001-entity validation benchmark with EXP-003B decision layer.
- Saves model to artifacts/xgb_matcher_exp004b.json (never overwrites EXP-002B).
- Saves experiment to experiments/EXP-004B.json.
"""

import argparse
import collections
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

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

from src.blocking.frequency_filter import (
    BlockingKeyFilterConfig,
    prune_high_frequency_keys,
)
from src.blocking.normalizers import (
    extract_address_keys,
    extract_token_blocking_keys,
    normalize_name_conservative,
)
from src.data.ground_truth import parse_ground_truth
from src.evaluation.metrics import evaluate_predictions, EvaluationReport
from src.matching.candidate_compression import (
    compute_cheap_candidate_score,
    extract_features_from_preprocessed,
    fast_score_candidate,
    preprocess_record,
    PreprocessedRecord,
)
from src.matching.decision_layer import (
    DEFAULT_V3_DECISION_CONFIG,
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

EXP004B_FEATURE_NAMES = BASE_FEATURE_NAMES + METADATA_FEATURE_NAMES


def worker_stream_chunk_with_metadata(task_args):
    """Worker streaming function that tracks channel agreement count and tier."""
    (
        worker_id,
        file_path,
        start_line,
        end_line,
        query_idx_country_val,
        s1_prep_val,
        query_idx_country_train,
        s1_prep_train,
        generic_keys_set_val,
        generic_keys_set_train,
        max_generic_per_s1,
    ) = task_args

    scored_val = collections.defaultdict(list)
    scored_train = collections.defaultdict(list)

    lines_read = 0

    with open(file_path, "r", encoding="utf-8-sig") as f:
        next(f)  # header
        for current_line, line in enumerate(f):
            if current_line < start_line:
                continue
            if current_line >= end_line:
                break

            lines_read += 1
            c_pos = line.rfind("\t")
            if c_pos == -1:
                continue
            tcountry = line[c_pos + 1 :].rstrip("\r\n")

            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            tid, tname, taddr = parts[0], parts[1], parts[2]

            tgt_p = None  # lazy creation

            # Process Validation S1 queries
            c_idx_val = query_idx_country_val.get(tcountry)
            if c_idx_val:
                matched_val_tiers = {}
                matched_val_channels = collections.defaultdict(set)

                # Tier 1: exact name
                ten = normalize_name_conservative(tname)
                if ten:
                    m = c_idx_val.get(f"en_{ten}")
                    if m:
                        for sid in m:
                            matched_val_tiers[sid] = 1
                            matched_val_channels[sid].add(1)

                # Tier 2 & 3 & 4: address keys
                if taddr and taddr.strip():
                    for ak in extract_address_keys(taddr):
                        if "pin_" in ak:
                            m = c_idx_val.get(f"ak_{ak}")
                            if m:
                                for sid in m:
                                    if sid not in matched_val_tiers or matched_val_tiers[sid] > 2:
                                        matched_val_tiers[sid] = 2
                                    matched_val_channels[sid].add(2)
                        else:
                            m = c_idx_val.get(f"ak_{ak}")
                            if m:
                                for sid in m:
                                    if sid not in matched_val_tiers or matched_val_tiers[sid] > 3:
                                        matched_val_tiers[sid] = 3
                                    matched_val_channels[sid].add(3)

                    for ak in extract_h1_relaxed_address_keys(taddr):
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
                        m = c_idx_val.get(ak)
                        if m:
                            for sid in m:
                                if sid not in matched_val_tiers or matched_val_tiers[sid] > tier:
                                    matched_val_tiers[sid] = tier
                                matched_val_channels[sid].add(ch)

                    for ck in extract_h3_name_locality_keys(tname, taddr):
                        m = c_idx_val.get(ck)
                        if m:
                            for sid in m:
                                if sid not in matched_val_tiers or matched_val_tiers[sid] > 4:
                                    matched_val_tiers[sid] = 4
                                matched_val_channels[sid].add(4)

                # Tier 5 & 6: token keys
                gen_keys_val = generic_keys_set_val.get(tcountry, set())
                for tk in extract_token_blocking_keys(tname):
                    k = f"tk_{tk}"
                    m = c_idx_val.get(k)
                    if m:
                        tier = 6 if k in gen_keys_val else 5
                        for sid in m:
                            if sid not in matched_val_tiers or matched_val_tiers[sid] > tier:
                                matched_val_tiers[sid] = tier
                            matched_val_channels[sid].add(5)

                if matched_val_tiers:
                    if tgt_p is None:
                        tgt_p = preprocess_record(tid, tname, taddr, tcountry)
                    for sid, tier in matched_val_tiers.items():
                        score = fast_score_candidate(s1_prep_val[sid], tgt_p)
                        if score >= 0.15:
                            ch_count = len(matched_val_channels[sid])
                            scored_val[sid].append((tier, score, tid, ch_count))

            # Process Training S1 queries
            c_idx_train = query_idx_country_train.get(tcountry)
            if c_idx_train:
                matched_train_tiers = {}
                matched_train_channels = collections.defaultdict(set)

                # Tier 1
                ten = normalize_name_conservative(tname)
                if ten:
                    m = c_idx_train.get(f"en_{ten}")
                    if m:
                        for sid in m:
                            matched_train_tiers[sid] = 1
                            matched_train_channels[sid].add(1)

                # Tier 2, 3, 4
                if taddr and taddr.strip():
                    for ak in extract_address_keys(taddr):
                        if "pin_" in ak:
                            m = c_idx_train.get(f"ak_{ak}")
                            if m:
                                for sid in m:
                                    if sid not in matched_train_tiers or matched_train_tiers[sid] > 2:
                                        matched_train_tiers[sid] = 2
                                    matched_train_channels[sid].add(2)
                        else:
                            m = c_idx_train.get(f"ak_{ak}")
                            if m:
                                for sid in m:
                                    if sid not in matched_train_tiers or matched_train_tiers[sid] > 3:
                                        matched_train_tiers[sid] = 3
                                    matched_train_channels[sid].add(3)

                    for ak in extract_h1_relaxed_address_keys(taddr):
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
                        m = c_idx_train.get(ak)
                        if m:
                            for sid in m:
                                if sid not in matched_train_tiers or matched_train_tiers[sid] > tier:
                                    matched_train_tiers[sid] = tier
                                matched_train_channels[sid].add(ch)

                    for ck in extract_h3_name_locality_keys(tname, taddr):
                        m = c_idx_train.get(ck)
                        if m:
                            for sid in m:
                                if sid not in matched_train_tiers or matched_train_tiers[sid] > 4:
                                    matched_train_tiers[sid] = 4
                                matched_train_channels[sid].add(4)

                # Tier 5, 6
                gen_keys_train = generic_keys_set_train.get(tcountry, set())
                for tk in extract_token_blocking_keys(tname):
                    k = f"tk_{tk}"
                    m = c_idx_train.get(k)
                    if m:
                        tier = 6 if k in gen_keys_train else 5
                        for sid in m:
                            if sid not in matched_train_tiers or matched_train_tiers[sid] > tier:
                                matched_train_tiers[sid] = tier
                            matched_train_channels[sid].add(5)

                if matched_train_tiers:
                    if tgt_p is None:
                        tgt_p = preprocess_record(tid, tname, taddr, tcountry)
                    for sid, tier in matched_train_tiers.items():
                        score = fast_score_candidate(s1_prep_train[sid], tgt_p)
                        if score >= 0.15:
                            ch_count = len(matched_train_channels[sid])
                            scored_train[sid].append((tier, score, tid, ch_count))

    # Worker-level pruning (top 100 with max 10 generic per S1)
    def prune_worker_dict(d):
        pruned = {}
        for sid, pairs in d.items():
            if len(pairs) <= 100:
                pruned[sid] = pairs
            else:
                specific = [p for p in pairs if p[0] <= 5]
                generic = [p for p in pairs if p[0] == 6]
                specific.sort(key=lambda x: (x[0], -x[1]))
                generic.sort(key=lambda x: -x[1])
                n_gen = min(len(generic), max_generic_per_s1)
                n_spec = min(len(specific), 100 - n_gen)
                pruned[sid] = specific[:n_spec] + generic[:n_gen]
        return pruned

    return lines_read, prune_worker_dict(scored_val), prune_worker_dict(scored_train)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--train-entities", type=int, default=1000)
    parser.add_argument("--final-cap", type=int, default=100)
    parser.add_argument("--split-file", type=Path, default=Path("experiments/splits/validation_split_seed42_benchmark_10000.json"))
    parser.add_argument("--ground-truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--source1", type=Path, default=Path("dataset/train/train_source1.tsv"))
    parser.add_argument("--source2", type=Path, default=Path("dataset/train/train_source2.tsv"))
    parser.add_argument("--source3", type=Path, default=Path("dataset/train/train_source3.tsv"))
    parser.add_argument("--baseline-cache", type=Path, default=Path("scratch/val_candidate_scores_exp002b.pkl"))
    parser.add_argument("--output-model", type=Path, default=Path("artifacts/xgb_matcher_exp004b.json"))
    parser.add_argument("--output-json", type=Path, default=Path("experiments/EXP-004B.json"))
    args = parser.parse_args()

    overall_start = time.time()

    print("=" * 95, flush=True)
    print("EXP-004B: ZERO-CANDIDATE-DRIFT RETRIEVAL METADATA MODEL EXPERIMENT", flush=True)
    print("=" * 95, flush=True)

    # 1. Load Split & Disjoint Entities
    with open(args.split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    val_s1_list: List[str] = split_data["val_s1_ids"]
    val_s1_set = set(val_s1_list)
    all_train_s1: List[str] = split_data["train_s1_ids"]
    train_s1_list: List[str] = all_train_s1[: args.train_entities]
    train_s1_set = set(train_s1_list)

    assert train_s1_set.isdisjoint(val_s1_set), "Train and Val overlap!"
    print(f"Entities: Train={len(train_s1_set):,} | Val={len(val_s1_set):,}", flush=True)

    # 2. Parse Ground Truth
    train_gt = parse_ground_truth(args.ground_truth, allowed_s1_ids=train_s1_set)
    val_gt = parse_ground_truth(args.ground_truth, allowed_s1_ids=val_s1_set)

    # 3. Load & Preprocess S1 Records for Train and Val
    val_s1_records: Dict[str, Tuple[str, str, str, str]] = {}
    train_s1_records: Dict[str, Tuple[str, str, str, str]] = {}
    val_s1_prep: Dict[str, PreprocessedRecord] = {}
    train_s1_prep: Dict[str, PreprocessedRecord] = {}

    raw_idx_val: Dict[str, Dict[str, List[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    raw_idx_train: Dict[str, Dict[str, List[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))

    with open(args.source1, "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            sid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
            if sid in val_s1_set:
                val_s1_records[sid] = (sid, name, addr, country)
                prep = preprocess_record(sid, name, addr, country)
                val_s1_prep[sid] = prep
                c_idx = raw_idx_val[country]
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

            elif sid in train_s1_set:
                train_s1_records[sid] = (sid, name, addr, country)
                prep = preprocess_record(sid, name, addr, country)
                train_s1_prep[sid] = prep
                c_idx = raw_idx_train[country]
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

    # 4. Apply Frequency Filters Independently (Zero Leakage)
    filter_cfg = BlockingKeyFilterConfig(max_key_fanout=15, preserve_exact_name=True, preserve_pin_keys=True)

    query_idx_clean_val: Dict[str, Dict[str, List[str]]] = {}
    generic_keys_set_val: Dict[str, Set[str]] = collections.defaultdict(set)
    for c, r_idx in raw_idx_val.items():
        c_count = sum(1 for sid, _, _, cnt in val_s1_records.values() if cnt == c)
        pruned_idx, _ = prune_high_frequency_keys(r_idx, total_s1=c_count, config=filter_cfg)
        query_idx_clean_val[c] = pruned_idx
        for k, postings in pruned_idx.items():
            if k.startswith("tk_") and len(postings) >= 5:
                generic_keys_set_val[c].add(k)

    query_idx_clean_train: Dict[str, Dict[str, List[str]]] = {}
    generic_keys_set_train: Dict[str, Set[str]] = collections.defaultdict(set)
    for c, r_idx in raw_idx_train.items():
        c_count = sum(1 for sid, _, _, cnt in train_s1_records.values() if cnt == c)
        pruned_idx, _ = prune_high_frequency_keys(r_idx, total_s1=c_count, config=filter_cfg)
        query_idx_clean_train[c] = pruned_idx
        for k, postings in pruned_idx.items():
            if k.startswith("tk_") and len(postings) >= 5:
                generic_keys_set_train[c].add(k)

    del raw_idx_val, raw_idx_train
    gc.collect()

    # 5. Parallel Stream S2 & S3
    def count_lines(fp):
        with open(fp, "r", encoding="utf-8-sig") as f:
            return sum(1 for _ in f) - 1

    s2_lines = count_lines(args.source2)
    s3_lines = count_lines(args.source3)

    tasks = []
    w_idx = 0
    for src_fp, n_lines in [(args.source2, s2_lines), (args.source3, s3_lines)]:
        chunk_sz = (n_lines + args.num_workers - 1) // args.num_workers
        for i in range(args.num_workers):
            start = i * chunk_sz
            end = min((i + 1) * chunk_sz, n_lines)
            if start < n_lines:
                tasks.append((
                    w_idx,
                    str(src_fp),
                    start,
                    end,
                    query_idx_clean_val,
                    val_s1_prep,
                    query_idx_clean_train,
                    train_s1_prep,
                    generic_keys_set_val,
                    generic_keys_set_train,
                    10,
                ))
                w_idx += 1

    print(f"\nStreaming search pool with {args.num_workers} parallel workers...", flush=True)
    t0_stream = time.time()
    with mp.Pool(processes=args.num_workers) as pool:
        results = pool.map(worker_stream_chunk_with_metadata, tasks)

    combined_val = collections.defaultdict(list)
    combined_train = collections.defaultdict(list)

    for _, p_val, p_train in results:
        for sid, pairs in p_val.items():
            combined_val[sid].extend(pairs)
        for sid, pairs in p_train.items():
            combined_train[sid].extend(pairs)

    del results
    gc.collect()
    print(f"Streaming completed in {time.time() - t0_stream:.2f}s.", flush=True)

    # 6. Specificity Cap & Metadata Feature Assembly
    def assemble_candidates(s1_list, combined_dict):
        final_cands = {}
        needed_tids = set()
        for sid in s1_list:
            raw_pairs = combined_dict.get(sid, [])
            best_by_tid = {}
            for tier, score, tid, ch_count in raw_pairs:
                if tid not in best_by_tid:
                    best_by_tid[tid] = (tier, score, ch_count)
                else:
                    prev_tier, prev_score, prev_ch = best_by_tid[tid]
                    best_by_tid[tid] = (
                        min(prev_tier, tier),
                        max(prev_score, score),
                        max(prev_ch, ch_count),
                    )

            deduped = [(tier, score, tid, ch) for tid, (tier, score, ch) in best_by_tid.items()]
            if len(deduped) <= args.final_cap:
                # Retain exact sorting by tier and negative score
                specific = [p for p in deduped if p[0] <= 5]
                generic = [p for p in deduped if p[0] == 6]
                specific.sort(key=lambda x: (x[0], -x[1]))
                generic.sort(key=lambda x: -x[1])
                selected = specific + generic
            else:
                specific = [p for p in deduped if p[0] <= 5]
                generic = [p for p in deduped if p[0] == 6]
                specific.sort(key=lambda x: (x[0], -x[1]))
                generic.sort(key=lambda x: -x[1])
                n_specific = min(len(specific), args.final_cap)
                n_generic = min(len(generic), args.final_cap - n_specific)
                selected = specific[:n_specific] + generic[:n_generic]

            pool_size = float(len(selected))
            s1_items = []
            for rank_idx, (tier, score, tid, ch) in enumerate(selected):
                recip_rank = 1.0 / (rank_idx + 1.0)
                meta_dict = {
                    "channel_agreement_count": float(ch),
                    "retrieval_priority_tier": float(tier),
                    "reciprocal_retrieval_rank": recip_rank,
                    "s1_candidate_pool_size": pool_size,
                }
                s1_items.append((tid, meta_dict))
                needed_tids.add(tid)

            final_cands[sid] = s1_items
        return final_cands, needed_tids

    val_cands_meta, val_needed_tids = assemble_candidates(val_s1_list, combined_val)
    train_cands_meta, train_needed_tids = assemble_candidates(train_s1_list, combined_train)

    del combined_val, combined_train
    gc.collect()

    # 7. CANDIDATE-DRIFT TEST: Compare val_cands_meta against baseline
    print("\n" + "=" * 90, flush=True)
    print("CANDIDATE-DRIFT INTEGRITY TEST", flush=True)
    print("=" * 90, flush=True)

    with open(args.baseline_cache, "rb") as f:
        baseline_data = pickle.load(f)
    baseline_s1_cands = baseline_data["s1_candidate_scores"]

    orig_pairs = set()
    for sid, pairs in baseline_s1_cands.items():
        for tid, _ in pairs:
            orig_pairs.add((sid, tid))

    new_pairs = set()
    for sid, items in val_cands_meta.items():
        for tid, _ in items:
            new_pairs.add((sid, tid))

    missing_pairs = orig_pairs - new_pairs
    added_pairs = new_pairs - orig_pairs
    candidate_set_difference = len(missing_pairs) + len(added_pairs)

    print(f"Original Candidate Pairs : {len(orig_pairs):,}")
    print(f"New Metadata Cand Pairs  : {len(new_pairs):,}")
    print(f"Missing Pairs            : {len(missing_pairs):,}")
    print(f"Additional Pairs         : {len(added_pairs):,}")
    print(f"Candidate Set Difference : {candidate_set_difference}")

    if candidate_set_difference != 0:
        print("\nCRITICAL FAILURE: Candidate set difference > 0! Halting experiment as required.", flush=True)
        sys.exit(1)
    else:
        print("VERIFICATION PASS: candidate_set_difference == 0 (100% IDENTICAL CANDIDATE SET).", flush=True)

    # 8. Retrieve Target Records
    all_needed_tids = val_needed_tids | train_needed_tids
    print(f"\nRetrieving {len(all_needed_tids):,} needed target records...", flush=True)
    target_prep_map: Dict[str, PreprocessedRecord] = {}

    for src_fp in [args.source2, args.source3]:
        with open(src_fp, "r", encoding="utf-8-sig") as f:
            next(f)
            for line in f:
                pos = line.find("\t")
                if pos == -1:
                    continue
                tid = line[:pos]
                if tid in all_needed_tids:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 4:
                        target_prep_map[tid] = preprocess_record(parts[0], parts[1], parts[2], parts[3])
                        if len(target_prep_map) == len(all_needed_tids):
                            break
        if len(target_prep_map) == len(all_needed_tids):
            break

    # 9. Build 26-Feature Matrices
    print("\nBuilding 26-feature training and validation matrices...", flush=True)
    t0_matrix = time.time()

    def build_feature_matrix(s1_list, s1_prep_map, cands_meta, gt_dict):
        X_rows = []
        y_rows = []
        pair_keys = []
        for sid in s1_list:
            s1_p = s1_prep_map[sid]
            true_set = gt_dict.get(sid, set())
            for tid, meta in cands_meta.get(sid, []):
                tgt_p = target_prep_map.get(tid)
                if tgt_p is None:
                    continue
                feats = extract_features_from_preprocessed(s1_p, tgt_p)
                feats["cheap_candidate_score"] = compute_cheap_candidate_score(feats)
                # Append 4 metadata features
                feats.update(meta)
                row = [feats[col] for col in EXP004B_FEATURE_NAMES]
                X_rows.append(row)
                y_rows.append(1 if tid in true_set else 0)
                pair_keys.append((sid, tid))
        return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32), pair_keys

    X_train, y_train, train_pairs = build_feature_matrix(train_s1_list, train_s1_prep, train_cands_meta, train_gt)
    X_val, y_val, val_pairs = build_feature_matrix(val_s1_list, val_s1_prep, val_cands_meta, val_gt)

    print(f"Matrix Construction completed in {time.time() - t0_matrix:.2f}s.", flush=True)
    print(f"Train Matrix: {X_train.shape} (Positives: {np.sum(y_train == 1):,}, Negatives: {np.sum(y_train == 0):,})", flush=True)
    print(f"Val Matrix  : {X_val.shape} (Positives: {np.sum(y_val == 1):,}, Negatives: {np.sum(y_val == 0):,})", flush=True)

    # 10. Train New 26-Feature XGBoost Matcher
    print("\nTraining 26-Feature XGBoost Matcher (max_depth=5, lr=0.1, n_round=150)...", flush=True)
    t0_train = time.time()
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=EXP004B_FEATURE_NAMES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=EXP004B_FEATURE_NAMES)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "nthread": args.num_workers,
        "seed": 42,
    }
    bst_26 = xgb.train(params, dtrain, num_boost_round=150)
    print(f"Training completed in {time.time() - t0_train:.2f}s.", flush=True)

    # Save new model artifact
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    bst_26.save_model(str(args.output_model))
    print(f"Saved new 26-feature model to {args.output_model}", flush=True)

    # 11. Score Validation Pairs
    val_probs_26 = bst_26.predict(dval)

    # Group scores by S1 for Decision Layer Evaluation
    s1_cand_scores_26: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
    for (sid, tid), p in zip(val_pairs, val_probs_26):
        s1_cand_scores_26[sid].append((tid, float(p)))

    # 12. Evaluate Strategy A: Baseline EXP-002B + EXP-003B Decision Layer
    # (Retrieved from cached baseline)
    s1_cand_scores_002b = baseline_data["s1_candidate_scores"]
    s1_country_map = {sid: val_s1_records[sid][3] for sid in val_s1_list}

    # Evaluation function
    def evaluate_model_pipeline(cand_scores_dict, use_decision_layer=True, raw_threshold=0.60):
        predictions: Dict[str, Set[str]] = {sid: set() for sid in val_s1_list}
        total_pred = 0
        for sid in val_s1_list:
            cands = cand_scores_dict.get(sid, [])
            country = s1_country_map.get(sid, "")
            if use_decision_layer:
                matched = filter_entity_candidates(cands, country, DEFAULT_V3_DECISION_CONFIG)
            else:
                matched = [tid for tid, p in cands if p >= raw_threshold]
            predictions[sid] = set(matched)
            total_pred += len(matched)
        rep = evaluate_predictions(predictions, val_gt)
        return rep, total_pred

    # A1: EXP-002B Raw (0.60)
    rep_002b_raw, pred_002b_raw = evaluate_model_pipeline(s1_cand_scores_002b, use_decision_layer=False, raw_threshold=0.60)
    # A2: EXP-002B + EXP-003B Decision Layer (Official Baseline)
    rep_002b_v3, pred_002b_v3 = evaluate_model_pipeline(s1_cand_scores_002b, use_decision_layer=True)

    # B1: EXP-004B 26-Feature Model Raw (0.60)
    rep_004b_raw, pred_004b_raw = evaluate_model_pipeline(s1_cand_scores_26, use_decision_layer=False, raw_threshold=0.60)
    # B2: EXP-004B 26-Feature Model + EXP-003B Decision Layer
    rep_004b_v3, pred_004b_v3 = evaluate_model_pipeline(s1_cand_scores_26, use_decision_layer=True)

    # 13. Comparison Report
    print("\n" + "=" * 125, flush=True)
    print("EXPERIMENTAL EVALUATION RESULTS: EXP-004B vs EXP-002B / EXP-003B BASELINE", flush=True)
    print("=" * 125, flush=True)
    print(f"{'Configuration':<45} | {'Macro F0.5':<11} | {'Precision':<10} | {'Recall':<10} | {'Pred':<7} | {'TP':<6} | {'FP':<5} | {'Single Acc'}", flush=True)
    print("-" * 125, flush=True)
    print(f"{'1. EXP-002B Baseline Raw (theta=0.60)':<45} | {rep_002b_raw.macro_f0_5:<11.4f} | {rep_002b_raw.macro_precision:<10.4f} | {rep_002b_raw.macro_recall:<10.4f} | {pred_002b_raw:<7} | {rep_002b_raw.total_tp:<6} | {rep_002b_raw.total_fp:<5} | {rep_002b_raw.singleton_f0_5:<10.4f}", flush=True)
    print(f"{'2. EXP-002B + EXP-003B Decision Layer (Ref)':<45} | {rep_002b_v3.macro_f0_5:<11.4f} | {rep_002b_v3.macro_precision:<10.4f} | {rep_002b_v3.macro_recall:<10.4f} | {pred_002b_v3:<7} | {rep_002b_v3.total_tp:<6} | {rep_002b_v3.total_fp:<5} | {rep_002b_v3.singleton_f0_5:<10.4f}", flush=True)
    print(f"{'3. EXP-004B 26-Feat Metadata Raw (theta=0.60)':<45} | {rep_004b_raw.macro_f0_5:<11.4f} | {rep_004b_raw.macro_precision:<10.4f} | {rep_004b_raw.macro_recall:<10.4f} | {pred_004b_raw:<7} | {rep_004b_raw.total_tp:<6} | {rep_004b_raw.total_fp:<5} | {rep_004b_raw.singleton_f0_5:<10.4f}", flush=True)
    print(f"{'4. EXP-004B 26-Feat + EXP-003B Decision Layer':<45} | {rep_004b_v3.macro_f0_5:<11.4f} | {rep_004b_v3.macro_precision:<10.4f} | {rep_004b_v3.macro_recall:<10.4f} | {pred_004b_v3:<7} | {rep_004b_v3.total_tp:<6} | {rep_004b_v3.total_fp:<5} | {rep_004b_v3.singleton_f0_5:<10.4f}", flush=True)
    print("=" * 125, flush=True)

    delta_f05 = rep_004b_v3.macro_f0_5 - rep_002b_v3.macro_f0_5
    delta_prec = rep_004b_v3.macro_precision - rep_002b_v3.macro_precision
    delta_rec = rep_004b_v3.macro_recall - rep_002b_v3.macro_recall
    delta_fp = rep_004b_v3.total_fp - rep_002b_v3.total_fp
    delta_tp = rep_004b_v3.total_tp - rep_002b_v3.total_tp

    decision = "KEEP" if delta_f05 > 0.0010 else "REJECT"

    print(f"\nDECISION RULE EVALUATION:")
    print(f"  Macro F0.5 Delta vs EXP-003B Baseline : {delta_f05:+.6f}")
    print(f"  Precision Delta                       : {delta_prec:+.6f}")
    print(f"  Recall Delta                          : {delta_rec:+.6f}")
    print(f"  False Positive Delta                  : {delta_fp:+d}")
    print(f"  True Positive Delta                   : {delta_tp:+d}")
    print(f"  Verdict                               : {decision}", flush=True)

    # 14. Save Experiment Record
    exp_record = {
        "experiment_id": "EXP-004B",
        "description": "Zero-Candidate-Drift Retrieval Metadata Model Experiment (26 features)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_entities": len(val_s1_list),
        "total_candidate_pairs": len(new_pairs),
        "avg_candidates_per_s1": len(new_pairs) / len(val_s1_list),
        "candidate_set_difference": candidate_set_difference,
        "features": {
            "base_feature_count": len(BASE_FEATURE_NAMES),
            "metadata_feature_count": len(METADATA_FEATURE_NAMES),
            "total_feature_count": len(EXP004B_FEATURE_NAMES),
            "metadata_feature_names": METADATA_FEATURE_NAMES,
        },
        "model_artifact": str(args.output_model),
        "metrics": {
            "exp002b_raw_0_60": rep_002b_raw.to_dict(),
            "exp002b_with_exp003b_decision_layer": rep_002b_v3.to_dict(),
            "exp004b_raw_0_60": rep_004b_raw.to_dict(),
            "exp004b_with_exp003b_decision_layer": rep_004b_v3.to_dict(),
            "deltas_vs_exp003b_baseline": {
                "macro_f0_5": delta_f05,
                "precision": delta_prec,
                "recall": delta_rec,
                "tp_delta": delta_tp,
                "fp_delta": delta_fp,
            },
        },
        "decision": decision,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(exp_record, f, indent=2)
    print(f"Saved experiment record to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
