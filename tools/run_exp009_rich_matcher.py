#!/usr/bin/env python3
"""EXP-009-RICH-MATCHER: Rich Entity-Resolution Pairwise Matcher Experiment.

Trains a richer 52-feature pairwise XGBoost matcher on a disjoint training split,
incorporating:
- Jaro, Jaro-Winkler, Levenshtein, char 3-grams, token Jaccard, common token metrics
- Compound slash number agreement (cnum)
- Cross-field synergy (name * addr, weak-name/strong-address indicators)
- Hard-positive emphasis and weighting
- Canonical validation benchmark evaluation against EXP-006B (F0.5 = 0.845653).

Strict Rules:
- Zero data leakage (train and val splits are strictly disjoint).
- Preserves all V3 and EXP-006B artifacts.
- Isolated artifacts: artifacts/xgb_matcher_exp009.json, experiments/EXP-009-rich-matcher.json.
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
from src.blocking.retrieval import (
    apply_selective_residual_recovery,
    extract_compound_number_keys,
    rank_candidates_channel_consensus_first,
)
from src.data.ground_truth import parse_ground_truth
from src.evaluation.metrics import evaluate_predictions, EvaluationReport
from src.matching.candidate_compression import (
    fast_score_candidate,
    preprocess_record,
    PreprocessedRecord,
)
from src.matching.decision_layer import (
    DecisionLayerConfig,
    DEFAULT_V3_DECISION_CONFIG,
    filter_entity_candidates,
)
from src.matching.rich_features import (
    extract_rich_pairwise_features,
    RICH_FEATURE_NAMES,
)
from tools.run_exp005_adaptive_ranking import fast_score_candidate_detailed
from tools.run_recovery_experiments import (
    extract_h1_relaxed_address_keys,
    extract_h3_name_locality_keys,
)


def worker_stream_train_candidates(task_args):
    """Worker streaming function to retrieve candidates for training entities."""
    (
        worker_id,
        file_path,
        start_line,
        end_line,
        query_idx_country_train,
        s1_prep_train,
        generic_keys_set_train,
        train_cnum_keys,
        max_cands_per_s1,
    ) = task_args

    scored_train = collections.defaultdict(list)
    cnum_hits = collections.defaultdict(list)
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

            c_idx_train = query_idx_country_train.get(tcountry)
            if not c_idx_train:
                continue

            matched_tiers = {}
            matched_channels = collections.defaultdict(set)

            # 1. Exact name
            ten = normalize_name_conservative(tname)
            if ten:
                m = c_idx_train.get(f"en_{ten}")
                if m:
                    for sid in m:
                        matched_tiers[sid] = 1
                        matched_channels[sid].add(1)

            # 2. Address keys
            if taddr and taddr.strip():
                for ak in extract_address_keys(taddr):
                    tier = 2 if "pin_" in ak else 3
                    ch = 2 if "pin_" in ak else 3
                    m = c_idx_train.get(f"ak_{ak}")
                    if m:
                        for sid in m:
                            if sid not in matched_tiers or matched_tiers[sid] > tier:
                                matched_tiers[sid] = tier
                            matched_channels[sid].add(ch)

                for ak in extract_h1_relaxed_address_keys(taddr):
                    tier = 2 if ak.startswith("pin_") else (4 if ak.startswith("loc_") else 3)
                    ch = 2 if ak.startswith("pin_") else (4 if ak.startswith("loc_") else 3)
                    m = c_idx_train.get(ak)
                    if m:
                        for sid in m:
                            if sid not in matched_tiers or matched_tiers[sid] > tier:
                                matched_tiers[sid] = tier
                            matched_channels[sid].add(ch)

                for ck in extract_h3_name_locality_keys(tname, taddr):
                    m = c_idx_train.get(ck)
                    if m:
                        for sid in m:
                            if sid not in matched_tiers or matched_tiers[sid] > 4:
                                matched_tiers[sid] = 4
                            matched_channels[sid].add(4)

                # Compound numbers
                if "/" in taddr:
                    cn_keys = extract_compound_number_keys(taddr)
                    for cnk in cn_keys:
                        m = train_cnum_keys.get(cnk)
                        if m:
                            for sid in m:
                                if sid not in matched_tiers or matched_tiers[sid] > 2:
                                    matched_tiers[sid] = 2
                                matched_channels[sid].add(2)
                                cnum_hits[sid].append(tid)

            # 3. Token keys
            gen_keys = generic_keys_set_train.get(tcountry, set())
            for tk in extract_token_blocking_keys(tname):
                k = f"tk_{tk}"
                m = c_idx_train.get(k)
                if m:
                    tier = 6 if k in gen_keys else 5
                    for sid in m:
                        if sid not in matched_tiers or matched_tiers[sid] > tier:
                            matched_tiers[sid] = tier
                        matched_channels[sid].add(5)

            if matched_tiers:
                tgt_p = preprocess_record(tid, tname, taddr, tcountry)
                for sid, tier in matched_tiers.items():
                    s1_p = s1_prep_train[sid]
                    score = fast_score_candidate(s1_p, tgt_p)
                    ch_count = len(matched_channels[sid])
                    if score >= 0.12 or ch_count >= 2:
                        scored_train[sid].append((tier, score, tid, ch_count))

    # Worker pruning
    pruned = {}
    for sid, cands in scored_train.items():
        if len(cands) <= max_cands_per_s1:
            pruned[sid] = cands
        else:
            cands.sort(key=lambda x: (x[0], -x[3], -x[1]))
            pruned[sid] = cands[:max_cands_per_s1]

    return lines_read, pruned, cnum_hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-entities", type=int, default=1500)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--final-cap", type=int, default=100)
    parser.add_argument("--split-file", type=Path, default=Path("experiments/splits/validation_split_seed42_benchmark_10000.json"))
    parser.add_argument("--ground-truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--source1", type=Path, default=Path("dataset/train/train_source1.tsv"))
    parser.add_argument("--source2", type=Path, default=Path("dataset/train/train_source2.tsv"))
    parser.add_argument("--source3", type=Path, default=Path("dataset/train/train_source3.tsv"))
    parser.add_argument("--val-cache", type=Path, default=Path("artifacts/exp005/val_raw_candidates.pkl"))
    parser.add_argument("--output-model", type=Path, default=Path("artifacts/xgb_matcher_exp009.json"))
    parser.add_argument("--output-json", type=Path, default=Path("experiments/EXP-009-rich-matcher.json"))
    args = parser.parse_args()

    overall_start = time.time()
    print("=" * 100, flush=True)
    print("EXP-009-RICH-MATCHER: FULL END-TO-END EXPERIMENTAL PIPELINE", flush=True)
    print("=" * 100, flush=True)

    # 1. Load Split & Disjoint Entities
    with open(args.split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    val_s1_list: List[str] = split_data["val_s1_ids"]
    val_s1_set = set(val_s1_list)
    all_train_s1: List[str] = split_data["train_s1_ids"]
    train_s1_list: List[str] = all_train_s1[: args.train_entities]
    train_s1_set = set(train_s1_list)

    assert train_s1_set.isdisjoint(val_s1_set), "CRITICAL: Train and Val overlap!"
    print(f"Entities: Train={len(train_s1_set):,} | Val={len(val_s1_set):,}", flush=True)

    # 2. Parse Ground Truth
    train_gt = parse_ground_truth(args.ground_truth, allowed_s1_ids=train_s1_set)
    val_gt = parse_ground_truth(args.ground_truth, allowed_s1_ids=val_s1_set)
    print(f"Ground Truth: Train={sum(len(t) for t in train_gt.values()):,} | Val={sum(len(t) for t in val_gt.values()):,}", flush=True)

    # 3. Load & Preprocess S1 Records
    val_s1_records: Dict[str, Tuple[str, str, str, str]] = {}
    train_s1_records: Dict[str, Tuple[str, str, str, str]] = {}
    val_s1_prep: Dict[str, PreprocessedRecord] = {}
    train_s1_prep: Dict[str, PreprocessedRecord] = {}

    raw_idx_train = collections.defaultdict(lambda: collections.defaultdict(list))
    train_cnum_keys = collections.defaultdict(list)
    val_cnum_keys = collections.defaultdict(list)

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
                for cnk in extract_compound_number_keys(addr):
                    val_cnum_keys[cnk].append(sid)
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
                    for cnk in extract_compound_number_keys(prep.raw_address):
                        train_cnum_keys[cnk].append(sid)

    # 4. Filter Frequency Keys for Train
    filter_cfg = BlockingKeyFilterConfig(max_key_fanout=15, preserve_exact_name=True, preserve_pin_keys=True)
    query_idx_clean_train: Dict[str, Dict[str, List[str]]] = {}
    generic_keys_set_train: Dict[str, Set[str]] = collections.defaultdict(set)
    for c, r_idx in raw_idx_train.items():
        c_count = sum(1 for sid, _, _, cnt in train_s1_records.values() if cnt == c)
        pruned_idx, _ = prune_high_frequency_keys(r_idx, total_s1=c_count, config=filter_cfg)
        query_idx_clean_train[c] = pruned_idx
        for k, postings in pruned_idx.items():
            if k.startswith("tk_") and len(postings) >= 5:
                generic_keys_set_train[c].add(k)
    del raw_idx_train
    gc.collect()

    # 5. Parallel Stream Search Pool for Training Candidates & Validation CNUM hits
    print("\nStreaming search pool for training candidates and compound number links...", flush=True)
    t0_stream = time.time()

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
                    query_idx_clean_train,
                    train_s1_prep,
                    generic_keys_set_train,
                    train_cnum_keys,
                    150,
                ))
                w_idx += 1

    with mp.Pool(processes=args.num_workers) as pool:
        stream_results = pool.map(worker_stream_train_candidates, tasks)

    combined_train = collections.defaultdict(list)
    train_cnum_hits = collections.defaultdict(list)
    for _, p_train, p_cnum in stream_results:
        for sid, pairs in p_train.items():
            combined_train[sid].extend(pairs)
        for sid, tids in p_cnum.items():
            train_cnum_hits[sid].extend(tids)

    del stream_results
    gc.collect()
    print(f"Streaming completed in {time.time() - t0_stream:.2f}s.", flush=True)

    # 6. Assemble Training Candidates with Metadata
    train_cands_meta = {}
    train_needed_tids = set()

    for sid in train_s1_list:
        raw_pairs = combined_train.get(sid, [])
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

        # Ensure ground truth targets are present for hard-positive learning
        for true_tid in train_gt.get(sid, set()):
            if true_tid not in best_by_tid:
                best_by_tid[true_tid] = (2, 0.50, 1)

        deduped = [(tier, score, tid, ch) for tid, (tier, score, ch) in best_by_tid.items()]
        # Rank by channel consensus first
        deduped.sort(key=lambda x: (-x[3], x[0], -x[1]))
        selected = deduped[: args.final_cap]

        pool_size = float(len(selected))
        s1_items = []
        for rank_idx, (tier, score, tid, ch) in enumerate(selected):
            meta = {
                "channel_agreement_count": float(ch),
                "retrieval_priority_tier": float(tier),
                "reciprocal_retrieval_rank": 1.0 / (rank_idx + 1.0),
                "s1_candidate_pool_size": pool_size,
            }
            s1_items.append((tid, meta))
            train_needed_tids.add(tid)
        train_cands_meta[sid] = s1_items

    del combined_train
    gc.collect()

    # 7. Assemble Validation Candidates (EXP-006B Candidate Pool)
    print("\nLoading cached raw candidate pool from", args.val_cache, flush=True)
    with open(args.val_cache, "rb") as f:
        val_raw_pool = pickle.load(f)

    # Collect CNUM hits for validation entities from train_source2/3
    print("Collecting validation compound number hits...", flush=True)
    val_cnum_target_records = {}
    val_cnum_hits = collections.defaultdict(list)
    for src in [args.source2, args.source3]:
        with open(src, "r", encoding="utf-8-sig") as f:
            next(f)
            for line in f:
                if "/" in line:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 4:
                        cn_keys = extract_compound_number_keys(parts[2])
                        matched_sids = set()
                        for k in cn_keys:
                            if k in val_cnum_keys:
                                matched_sids.update(val_cnum_keys[k])
                        if matched_sids:
                            val_cnum_target_records[parts[0]] = preprocess_record(parts[0], parts[1], parts[2], parts[3])
                            for sid in matched_sids:
                                val_cnum_hits[sid].append(parts[0])

    # Merge cnum hits into validation pool
    for sid, tids in val_cnum_hits.items():
        s1_p = val_s1_prep[sid]
        existing = {c["tid"]: c for c in val_raw_pool.get(sid, [])}
        for tid in set(tids):
            tgt_p = val_cnum_target_records.get(tid)
            if not tgt_p or tgt_p.country != s1_p.country:
                continue
            if tid in existing:
                c = existing[tid]
                c["best_tier"] = min(c["best_tier"], 2)
                c["ch_count"] = max(c["ch_count"], 2)
                c["spec_sum"] += 0.5
            else:
                score, s_name, s_addr, has_pin, has_house = fast_score_candidate_detailed(s1_p, tgt_p)
                entry = {
                    "tid": tid,
                    "best_tier": 2,
                    "ch_count": 1,
                    "key_count": 1,
                    "spec_sum": 0.5,
                    "score": score,
                    "s_name": s_name,
                    "s_addr": s_addr,
                    "has_pin": has_pin,
                    "has_house": has_house,
                }
                val_raw_pool.setdefault(sid, []).append(entry)

    # Build Phase 1 validation candidates
    val_phase1_cands = {}
    val_needed_tids = set()
    for sid in val_s1_list:
        selected = rank_candidates_channel_consensus_first(val_raw_pool.get(sid, []), cap=args.final_cap)
        val_phase1_cands[sid] = selected
        val_needed_tids.update(t for t, _ in selected)

    total_val_p1_pairs = sum(len(c) for c in val_phase1_cands.values())
    print(f"Validation Phase 1 Candidates : {total_val_p1_pairs:,} ({total_val_p1_pairs / len(val_s1_list):.2f} avg/S1)", flush=True)

    # 8. Retrieve All Needed Target Records
    all_needed_tids = train_needed_tids | val_needed_tids
    print(f"\nRetrieving {len(all_needed_tids):,} needed target records from search pool...", flush=True)
    t0_tgt = time.time()
    target_prep_map: Dict[str, PreprocessedRecord] = dict(val_cnum_target_records)

    for src_fp in [args.source2, args.source3]:
        with open(src_fp, "r", encoding="utf-8-sig") as f:
            next(f)
            for line in f:
                pos = line.find("\t")
                if pos == -1:
                    continue
                tid = line[:pos]
                if tid in all_needed_tids and tid not in target_prep_map:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 4:
                        target_prep_map[tid] = preprocess_record(parts[0], parts[1], parts[2], parts[3])
                        if len(target_prep_map) == len(all_needed_tids):
                            break
        if len(target_prep_map) == len(all_needed_tids):
            break

    print(f"Retrieved {len(target_prep_map):,} target records in {time.time() - t0_tgt:.2f}s.", flush=True)

    # 9. Build 52-Feature Matrices with Hard-Positive Weighting
    print("\nBuilding 52-feature training and validation matrices...", flush=True)
    t0_mat = time.time()

    def build_matrix(s1_list, s1_prep_map, cands_meta, gt_dict, is_training=False):
        X_rows = []
        y_rows = []
        weights = []
        pair_keys = []

        for sid in s1_list:
            s1_p = s1_prep_map[sid]
            true_set = gt_dict.get(sid, set())
            for tid, meta in cands_meta.get(sid, []):
                tgt_p = target_prep_map.get(tid)
                if tgt_p is None:
                    continue
                feats = extract_rich_pairwise_features(s1_p, tgt_p, meta)
                row = [feats[col] for col in RICH_FEATURE_NAMES]
                is_pos = 1 if tid in true_set else 0
                X_rows.append(row)
                y_rows.append(is_pos)
                pair_keys.append((sid, tid))

                if is_training:
                    if is_pos:
                        # Hard-positive weighting: boost pairs with low name similarity but true match
                        name_sim = feats["name_fuzz_wratio"]
                        w = 1.0 + 3.0 * (1.0 - name_sim)
                        weights.append(w)
                    else:
                        weights.append(1.0)
                else:
                    weights.append(1.0)

        return (
            np.array(X_rows, dtype=np.float32),
            np.array(y_rows, dtype=np.int32),
            np.array(weights, dtype=np.float32),
            pair_keys,
        )

    X_train, y_train, w_train, train_pairs = build_matrix(
        train_s1_list, train_s1_prep, train_cands_meta, train_gt, is_training=True
    )
    X_val, y_val, _, val_pairs = build_matrix(
        val_s1_list, val_s1_prep, val_phase1_cands, val_gt, is_training=False
    )

    print(f"Matrix construction completed in {time.time() - t0_mat:.2f}s.", flush=True)
    print(f"Train Matrix: {X_train.shape} (Positives: {np.sum(y_train == 1):,}, Negatives: {np.sum(y_train == 0):,})", flush=True)
    print(f"Val Matrix  : {X_val.shape} (Positives: {np.sum(y_val == 1):,}, Negatives: {np.sum(y_val == 0):,})", flush=True)

    # 10. Train 52-Feature XGBoost Matcher
    print("\nTraining 52-Feature XGBoost Matcher with Hard-Positive Weighting...", flush=True)
    t0_train = time.time()
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, feature_names=RICH_FEATURE_NAMES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=RICH_FEATURE_NAMES)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "nthread": args.num_workers,
        "seed": 42,
    }
    bst_rich = xgb.train(params, dtrain, num_boost_round=180)
    print(f"Training completed in {time.time() - t0_train:.2f}s.", flush=True)

    # Save model artifact
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    bst_rich.save_model(str(args.output_model))
    print(f"Saved new 52-feature model to {args.output_model}", flush=True)

    # 11. Score Phase 1 Validation Pairs
    print("\nScoring validation pairs...", flush=True)
    val_probs_p1 = bst_rich.predict(dval)

    s1_scores_p1: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
    for (sid, tid), p in zip(val_pairs, val_probs_p1):
        s1_scores_p1[sid].append((tid, float(p)))

    s1_country_map = {sid: val_s1_records[sid][3] for sid in val_s1_list}

    # Evaluate Phase 1 with Decision Layer
    def run_decision_layer(cand_scores_dict, margin_delta=0.30):
        cfg = DecisionLayerConfig(
            country_thresholds={"France": 0.45, "US": 0.65, "India": 0.55},
            margin_delta=margin_delta,
            default_threshold=0.60,
        )
        preds = {}
        for sid in val_s1_list:
            cands = cand_scores_dict.get(sid, [])
            country = s1_country_map.get(sid, "")
            matched = filter_entity_candidates(cands, country, cfg)
            preds[sid] = set(matched)
        return preds

    # Phase 1 Evaluation (margin = 0.30 and margin = 0.35)
    preds_p1_m30 = run_decision_layer(s1_scores_p1, margin_delta=0.30)
    preds_p1_m35 = run_decision_layer(s1_scores_p1, margin_delta=0.35)
    rep_p1_m30 = evaluate_predictions(preds_p1_m30, val_gt)
    rep_p1_m35 = evaluate_predictions(preds_p1_m35, val_gt)

    print(f"Phase 1 (margin=0.30): Macro F0.5 = {rep_p1_m30.macro_f0_5:.6f} | Prec = {rep_p1_m30.macro_precision:.6f} | Rec = {rep_p1_m30.macro_recall:.6f} | TP = {rep_p1_m30.total_tp} | FP = {rep_p1_m30.total_fp}", flush=True)
    print(f"Phase 1 (margin=0.35): Macro F0.5 = {rep_p1_m35.macro_f0_5:.6f} | Prec = {rep_p1_m35.macro_precision:.6f} | Rec = {rep_p1_m35.macro_recall:.6f} | TP = {rep_p1_m35.total_tp} | FP = {rep_p1_m35.total_fp}", flush=True)

    # 12. Phase 2: Selective Residual Recovery for Unresolved Entities
    print("\nApplying Phase 2 Selective Residual Recovery for unresolved entities...", flush=True)
    best_p1_preds = preds_p1_m35 if rep_p1_m35.macro_f0_5 > rep_p1_m30.macro_f0_5 else preds_p1_m30
    best_margin = 0.35 if rep_p1_m35.macro_f0_5 > rep_p1_m30.macro_f0_5 else 0.30

    residual_cands = apply_selective_residual_recovery(
        s1_list=val_s1_list,
        phase1_candidates=val_phase1_cands,
        phase1_predictions=best_p1_preds,
        raw_candidate_pool=val_raw_pool,
        max_recovery=20,
    )

    # Retrieve any newly added residual targets
    residual_needed_tids = set()
    for sid, items in residual_cands.items():
        for tid, _ in items:
            if tid not in target_prep_map:
                residual_needed_tids.add(tid)

    if residual_needed_tids:
        print(f"Retrieving {len(residual_needed_tids):,} residual targets...", flush=True)
        for src_fp in [args.source2, args.source3]:
            with open(src_fp, "r", encoding="utf-8-sig") as f:
                next(f)
                for line in f:
                    pos = line.find("\t")
                    if pos == -1:
                        continue
                    tid = line[:pos]
                    if tid in residual_needed_tids and tid not in target_prep_map:
                        parts = line.rstrip("\r\n").split("\t")
                        if len(parts) >= 4:
                            target_prep_map[tid] = preprocess_record(parts[0], parts[1], parts[2], parts[3])

    # Build feature matrix for combined candidates
    X_res_rows = []
    res_pair_keys = []
    total_val_cand_pairs = 0
    for sid in val_s1_list:
        s1_p = val_s1_prep[sid]
        cands = residual_cands.get(sid, [])
        total_val_cand_pairs += len(cands)
        for tid, meta in cands:
            tgt_p = target_prep_map.get(tid)
            if tgt_p is None:
                continue
            feats = extract_rich_pairwise_features(s1_p, tgt_p, meta)
            X_res_rows.append([feats[col] for col in RICH_FEATURE_NAMES])
            res_pair_keys.append((sid, tid))

    dres = xgb.DMatrix(np.array(X_res_rows, dtype=np.float32), feature_names=RICH_FEATURE_NAMES)
    res_probs = bst_rich.predict(dres)

    s1_scores_res: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
    for (sid, tid), p in zip(res_pair_keys, res_probs):
        s1_scores_res[sid].append((tid, float(p)))

    # Evaluate Phase 2 combined matching
    preds_final_m30 = run_decision_layer(s1_scores_res, margin_delta=0.30)
    preds_final_m35 = run_decision_layer(s1_scores_res, margin_delta=0.35)
    rep_final_m30 = evaluate_predictions(preds_final_m30, val_gt)
    rep_final_m35 = evaluate_predictions(preds_final_m35, val_gt)

    if rep_final_m35.macro_f0_5 >= rep_final_m30.macro_f0_5:
        best_final_rep = rep_final_m35
        best_final_preds = preds_final_m35
        selected_margin = 0.35
    else:
        best_final_rep = rep_final_m30
        best_final_preds = preds_final_m30
        selected_margin = 0.30

    total_pred_links = sum(len(p) for p in best_final_preds.values())
    empty_pred_count = sum(1 for p in best_final_preds.values() if len(p) == 0)

    # 13. Baseline Comparison
    BASE_F05 = 0.845653
    BASE_PREC = 0.907073
    BASE_REC = 0.738549
    BASE_TP = 5085
    BASE_FP = 298

    delta_f05 = best_final_rep.macro_f0_5 - BASE_F05
    delta_prec = best_final_rep.macro_precision - BASE_PREC
    delta_rec = best_final_rep.macro_recall - BASE_REC
    delta_tp = best_final_rep.total_tp - BASE_TP
    delta_fp = best_final_rep.total_fp - BASE_FP

    decision = "KEEP" if best_final_rep.macro_f0_5 > BASE_F05 else "REJECT"

    # Feature Importance
    f_score = bst_rich.get_score(importance_type="gain")
    sorted_feats = sorted(f_score.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 110, flush=True)
    print("EXP-009-RICH-MATCHER FINAL EVALUATION REPORT", flush=True)
    print("=" * 110, flush=True)
    print(f"Baseline F0.5 (EXP-006B) : {BASE_F05:.6f}")
    print(f"New F0.5                 : {best_final_rep.macro_f0_5:.6f}")
    print(f"Delta                    : {delta_f05:+.6f}")
    print(f"Precision                : {best_final_rep.macro_precision:.6f} (Delta: {delta_prec:+.6f})")
    print(f"Recall                   : {best_final_rep.macro_recall:.6f} (Delta: {delta_rec:+.6f})")
    print(f"TP                       : {best_final_rep.total_tp} (Delta: {delta_tp:+d})")
    print(f"FP                       : {best_final_rep.total_fp} (Delta: {delta_fp:+d})")
    print(f"FN                       : {best_final_rep.total_fn}")
    print(f"Singleton Accuracy       : {best_final_rep.singleton_f0_5:.6f}")
    print(f"Predicted Links          : {total_pred_links:,}")
    print(f"Empty Predictions        : {empty_pred_count:,}")
    print(f"Candidate Pairs          : {total_val_cand_pairs:,}")
    print(f"Average Candidates/S1    : {total_val_cand_pairs / len(val_s1_list):.2f}")
    print(f"Decision                 : {decision}", flush=True)
    print("=" * 110, flush=True)

    print("\nTOP 15 FEATURES BY GAIN:")
    for rank_i, (f_name, g_val) in enumerate(sorted_feats[:15], start=1):
        print(f"  {rank_i:2d}. {f_name:<30} : {g_val:.2f}")

    # 14. Save Experiment Record
    exp_record = {
        "experiment_id": "EXP-009-RICH-MATCHER",
        "description": "Rich 52-Feature Entity Resolution Matcher with Hard-Positive Weighting",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_entities": len(val_s1_list),
        "total_candidate_pairs": total_val_cand_pairs,
        "avg_candidates_per_s1": total_val_cand_pairs / len(val_s1_list),
        "total_runtime_seconds": time.time() - overall_start,
        "decision": decision,
        "metrics": {
            "baseline_exp006b": {
                "macro_f0_5": BASE_F05,
                "precision": BASE_PREC,
                "recall": BASE_REC,
                "tp": BASE_TP,
                "fp": BASE_FP,
            },
            "new_model_final": best_final_rep.to_dict(),
            "new_model_phase1_m30": rep_p1_m30.to_dict(),
            "new_model_phase1_m35": rep_p1_m35.to_dict(),
            "new_model_phase2_m30": rep_final_m30.to_dict(),
            "new_model_phase2_m35": rep_final_m35.to_dict(),
            "deltas": {
                "macro_f0_5": delta_f05,
                "precision": delta_prec,
                "recall": delta_rec,
                "tp_delta": delta_tp,
                "fp_delta": delta_fp,
            },
        },
        "top_features_by_gain": sorted_feats[:20],
        "model_artifact": str(args.output_model),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(exp_record, f, indent=2)
    print(f"\nSaved experiment record to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
