#!/usr/bin/env python3
"""EXP-002B: Train and Validate XGBoost Pairwise Matcher.

Implements Phases 1 to 5 of EXP-002B:
1. Reuses candidate compression policy 'Hybrid-T0.40-Cap100' on validation split.
2. Extracts training pairs from disjoint training split with zero validation leakage.
3. Generates 22 pairwise features (Name, Address, Context, and Cheap Score).
4. Trains an XGBoost binary classifier (Apache-2.0, <8B parameters).
5. Performs rigorous entity-level Macro F0.5 evaluation over threshold grid:
   [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90].
6. Outputs best validation threshold, metrics, and saves model artifact.
"""

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import time
import tracemalloc
from datetime import datetime, timezone
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
from src.data.ground_truth import parse_ground_truth
from src.evaluation.metrics import evaluate_predictions
from src.experiments.logger import ExperimentLogger, ExperimentRecord
from src.matching.candidate_compression import (
    CandidateCompressor,
    CompressionPolicy,
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


def worker_stream_search_pool(task_args):
    """Worker function to stream and score search pool records."""
    worker_id, file_path, start_line, end_line, query_idx_country, s1_preprocessed = task_args

    scored_candidates = collections.defaultdict(list)

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
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            tid, tname, taddr, tcountry = parts[0], parts[1], parts[2], parts[3]

            c_idx = query_idx_country.get(tcountry)
            if c_idx is None:
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

    elapsed = time.time() - t0
    fname = Path(file_path).name
    print(
        f"[Worker {worker_id} | {fname}] Processed {lines_read:,} lines in {elapsed:.1f}s. "
        f"Scored {pairs_scored:,} pairs.",
        flush=True,
    )

    # Prune per worker per s1 to top 120 candidates to keep IPC compact
    pruned_scored = {}
    for s1_id, pairs in scored_candidates.items():
        if len(pairs) > 120:
            pairs.sort(key=lambda x: x[0], reverse=True)
            pruned_scored[s1_id] = pairs[:120]
        else:
            pruned_scored[s1_id] = pairs

    return lines_read, pruned_scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--train-entities", type=int, default=1000, help="Number of training S1 entities")
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("experiments/splits/validation_split_seed42_benchmark_10000.json"),
    )
    parser.add_argument("--ground-truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--source1", type=Path, default=Path("dataset/train/train_source1.tsv"))
    parser.add_argument("--source2", type=Path, default=Path("dataset/train/train_source2.tsv"))
    parser.add_argument("--source3", type=Path, default=Path("dataset/train/train_source3.tsv"))
    parser.add_argument("--output-model", type=Path, default=Path("artifacts/xgb_matcher_exp002b.json"))
    parser.add_argument("--output-exp", type=Path, default=Path("experiments/EXP-002B.json"))
    args = parser.parse_args()

    overall_start = time.time()
    tracemalloc.start()

    print("=" * 80, flush=True)
    print("EXP-002B: Train and Validate XGBoost Pairwise Matcher", flush=True)
    print(f"Workers: {args.num_workers} | Platform: {sys.platform} | CPU cores: {os.cpu_count()}", flush=True)
    print("=" * 80, flush=True)

    # 1. Load Split
    with open(args.split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    val_s1_list = split_data["val_s1_ids"]
    val_s1_set = set(val_s1_list)
    all_train_s1 = split_data["train_s1_ids"]

    # Select disjoint training entities (sample up to args.train_entities)
    train_s1_list = all_train_s1[: args.train_entities]
    train_s1_set = set(train_s1_list)

    assert train_s1_set.isdisjoint(val_s1_set), "CRITICAL ERROR: Train and Val sets overlap!"
    print(f"Disjoint S1 Partition: Train={len(train_s1_set):,} entities, Val={len(val_s1_set):,} entities.", flush=True)

    # 2. Parse Ground Truth
    train_gt = parse_ground_truth(args.ground_truth, allowed_s1_ids=train_s1_set)
    val_gt = parse_ground_truth(args.ground_truth, allowed_s1_ids=val_s1_set)
    train_true_matches = sum(len(v) for v in train_gt.values())
    val_true_matches = sum(len(v) for v in val_gt.values())
    print(f"Ground Truth: Train={train_true_matches:,} true links, Val={val_true_matches:,} true links.", flush=True)

    # 3. Load S1 Records and Preprocess
    combined_s1_ids = train_s1_set | val_s1_set
    s1_preprocessed: Dict[str, PreprocessedRecord] = {}
    query_idx_country: Dict[str, Dict[str, List[str]]] = collections.defaultdict(dict)

    t0_s1 = time.time()
    with open(args.source1, "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            sid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
            if sid not in combined_s1_ids:
                continue

            prep = preprocess_record(sid, name, addr, country)
            s1_preprocessed[sid] = prep

            c_idx = query_idx_country[country]
            en = prep.name_norm
            if en:
                c_idx.setdefault(f"en_{en}", []).append(sid)
            for tk in extract_token_blocking_keys(prep.raw_name):
                c_idx.setdefault(f"tk_{tk}", []).append(sid)
            if prep.raw_address and prep.raw_address.strip():
                for ak in extract_address_keys(prep.raw_address):
                    c_idx.setdefault(f"ak_{ak}", []).append(sid)
                for ak in extract_h1_relaxed_address_keys(prep.raw_address):
                    c_idx.setdefault(ak, []).append(sid)
                for ck in extract_h3_name_locality_keys(prep.raw_name, prep.raw_address):
                    c_idx.setdefault(ck, []).append(sid)

            if len(s1_preprocessed) == len(combined_s1_ids):
                break

    t1_s1 = time.time()
    total_keys = sum(len(v) for v in query_idx_country.values())
    query_idx_clean = {c: dict(sub) for c, sub in query_idx_country.items()}
    print(f"Preprocessed {len(s1_preprocessed):,} S1 records ({total_keys:,} keys) in {t1_s1 - t0_s1:.2f}s.", flush=True)

    # 4. Stream Search Pool (Source 2 and Source 3)
    s2_total_lines = 5034616
    s3_total_lines = 5285603

    t0_stream = time.time()
    combined_scored_candidates: Dict[str, List[Tuple[float, str]]] = collections.defaultdict(list)

    for src_file, n_lines in [(args.source2, s2_total_lines), (args.source3, s3_total_lines)]:
        fname = src_file.name
        chunk_size = (n_lines + args.num_workers - 1) // args.num_workers
        tasks = []
        for wid in range(args.num_workers):
            start = wid * chunk_size
            end = min((wid + 1) * chunk_size, n_lines)
            tasks.append((wid, str(src_file), start, end, query_idx_clean, s1_preprocessed))

        print(f"\nStreaming {fname} ({n_lines:,} rows) with {args.num_workers} parallel workers...", flush=True)
        with mp.Pool(processes=args.num_workers) as pool:
            results = pool.map(worker_stream_chunk_wrapper, tasks)

        for lines_read, scored_cands in results:
            for sid, pairs in scored_cands.items():
                combined_scored_candidates[sid].extend(pairs)

    stream_elapsed = time.time() - t0_stream
    print(f"\nStreaming complete in {stream_elapsed:.2f}s.", flush=True)

    # 5. Apply Hybrid-T0.40-Cap100 Compression Policy
    train_candidates: Dict[str, List[str]] = {}
    val_candidates: Dict[str, List[str]] = {}
    needed_tids: Set[str] = set()

    for sid in train_s1_list:
        pairs = combined_scored_candidates.get(sid, [])
        pairs.sort(key=lambda x: x[0], reverse=True)
        selected = [tid for score, tid in pairs if score >= 0.40][:100]
        train_candidates[sid] = selected
        needed_tids.update(selected)

    for sid in val_s1_list:
        pairs = combined_scored_candidates.get(sid, [])
        pairs.sort(key=lambda x: x[0], reverse=True)
        selected = [tid for score, tid in pairs if score >= 0.40][:100]
        val_candidates[sid] = selected
        needed_tids.update(selected)

    train_pairs_count = sum(len(v) for v in train_candidates.values())
    val_pairs_count = sum(len(v) for v in val_candidates.values())
    print(f"Compressed Candidates: Train={train_pairs_count:,} pairs, Val={val_pairs_count:,} pairs.", flush=True)
    print(f"Unique target records needed for feature extraction: {len(needed_tids):,}", flush=True)

    # 6. Stream and Preprocess ONLY the Needed Target Records
    print("\nReading and preprocessing only needed candidate targets from search pool...", flush=True)
    t0_targets = time.time()
    target_preprocessed: Dict[str, PreprocessedRecord] = {}

    for src_file in [args.source2, args.source3]:
        with open(src_file, "r", encoding="utf-8-sig") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4 and parts[0] in needed_tids:
                    tid, tname, taddr, tcountry = parts[0], parts[1], parts[2], parts[3]
                    target_preprocessed[tid] = preprocess_record(tid, tname, taddr, tcountry)
                    if len(target_preprocessed) == len(needed_tids):
                        break
        if len(target_preprocessed) == len(needed_tids):
            break

    print(f"Preprocessed {len(target_preprocessed):,} target records in {time.time() - t0_targets:.2f}s.", flush=True)

    # 7. Build Feature Matrices (X_train, y_train, X_val, y_val)
    print("\nBuilding training and validation feature matrices...", flush=True)
    t0_feats = time.time()

    def build_matrix(cand_dict, gt_dict):
        X_rows = []
        y_rows = []
        pair_keys = []
        for sid, tids in cand_dict.items():
            s1_prep = s1_preprocessed[sid]
            true_set = gt_dict.get(sid, set())
            for tid in tids:
                tgt_prep = target_preprocessed.get(tid)
                if tgt_prep is None:
                    continue
                feats = extract_features_from_preprocessed(s1_prep, tgt_prep)
                feats["cheap_candidate_score"] = compute_cheap_candidate_score(feats)
                row = [feats[fname] for fname in FEATURE_NAMES]
                X_rows.append(row)
                y_rows.append(1 if tid in true_set else 0)
                pair_keys.append((sid, tid))
        return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32), pair_keys

    X_train, y_train, train_pairs = build_matrix(train_candidates, train_gt)
    X_val, y_val, val_pairs = build_matrix(val_candidates, val_gt)
    t1_feats = time.time()

    train_pos = int(np.sum(y_train == 1))
    train_neg = int(np.sum(y_train == 0))
    val_pos = int(np.sum(y_val == 1))
    val_neg = int(np.sum(y_val == 0))

    print(f"Feature Extraction complete in {t1_feats - t0_feats:.2f}s.", flush=True)
    print(f"Train Matrix: {X_train.shape} | Positives={train_pos:,} | Negatives={train_neg:,} (ratio: {train_neg/max(train_pos, 1):.1f}:1)", flush=True)
    print(f"Val Matrix  : {X_val.shape} | Positives={val_pos:,} | Negatives={val_neg:,} (ratio: {val_neg/max(val_pos, 1):.1f}:1)", flush=True)

    # 8. Train XGBoost Binary Classifier
    print("\nTraining XGBoost Pairwise Matcher (binary:logistic, num_boost_round=150, max_depth=5)...", flush=True)
    t0_xgb = time.time()
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES)
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
    bst = xgb.train(params, dtrain, num_boost_round=150)
    t1_xgb = time.time()
    print(f"XGBoost training complete in {t1_xgb - t0_xgb:.2f}s.", flush=True)

    # Save model artifact
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(args.output_model))
    print(f"Saved model to {args.output_model}", flush=True)

    # 9. Score Validation Candidate Pairs
    t0_score = time.time()
    val_probs = bst.predict(dval)
    t1_score = time.time()
    print(f"Scored {len(val_probs):,} validation pairs in {t1_score - t0_score:.2f}s.", flush=True)

    # 10. Entity-Level Threshold Optimization Grid Search
    print("\nEvaluating Entity-Level Macro F0.5 over Threshold Grid...", flush=True)
    threshold_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]

    grid_results = []
    best_thresh = 0.50
    best_macro_f05 = -1.0
    best_report = None

    print(
        f"{'Threshold':<10} | {'Macro F0.5':<11} | {'Precision':<10} | {'Recall':<10} | "
        f"{'Pred Links':<11} | {'Singletons':<11} | {'Zero Pred':<10} | {'Multi Pred':<10}",
        flush=True,
    )
    print("-" * 105, flush=True)

    for thresh in threshold_grid:
        predictions: Dict[str, Set[str]] = {sid: set() for sid in val_s1_list}
        for (sid, tid), prob in zip(val_pairs, val_probs):
            if prob >= thresh:
                predictions[sid].add(tid)

        report = evaluate_predictions(predictions, val_gt)
        total_pred_links = sum(len(v) for v in predictions.values())
        zero_pred_count = sum(1 for v in predictions.values() if len(v) == 0)
        multi_pred_count = sum(1 for v in predictions.values() if len(v) > 1)

        grid_results.append({
            "threshold": thresh,
            "macro_f0_5": report.macro_f0_5,
            "macro_precision": report.macro_precision,
            "macro_recall": report.macro_recall,
            "total_predicted_links": total_pred_links,
            "singleton_count": report.singleton_count,
            "singleton_f0_5": report.singleton_f0_5,
            "non_singleton_count": report.non_singleton_count,
            "non_singleton_f0_5": report.non_singleton_f0_5,
            "zero_prediction_entities": zero_pred_count,
            "multi_match_entities": multi_pred_count,
            "total_tp": report.total_tp,
            "total_fp": report.total_fp,
            "total_fn": report.total_fn,
        })

        print(
            f"{thresh:<10.2f} | {report.macro_f0_5:<11.4f} | {report.macro_precision:<10.4f} | "
            f"{report.macro_recall:<10.4f} | {total_pred_links:<11,} | {report.singleton_count:<11,} | "
            f"{zero_pred_count:<10,} | {multi_pred_count:<10,}",
            flush=True,
        )

        if report.macro_f0_5 > best_macro_f05:
            best_macro_f05 = report.macro_f0_5
            best_thresh = thresh
            best_report = report

    print("-" * 105, flush=True)
    print(f"Optimal Validation Threshold: {best_thresh:.2f} (Macro F0.5 = {best_macro_f05:.4f})", flush=True)
    print(f"Validation Precision       : {best_report.macro_precision:.4f}", flush=True)
    print(f"Validation Recall          : {best_report.macro_recall:.4f}", flush=True)
    print(f"Total True Links           : {val_true_matches:,}", flush=True)
    print(f"Total Predicted Links      : {best_report.total_tp + best_report.total_fp:,}", flush=True)

    # Feature Importance
    score_dict = bst.get_score(importance_type="gain")
    feat_imp = sorted([(fname, score_dict.get(fname, 0.0)) for fname in FEATURE_NAMES], key=lambda x: x[1], reverse=True)
    print("\nXGBoost Top Feature Importances (Gain):", flush=True)
    for fname, imp in feat_imp[:10]:
        print(f"  {fname:<25}: {imp:.4f}", flush=True)

    # 11. Save Detailed Experiment Results
    curr_mem, peak_mem = tracemalloc.get_traced_memory()
    total_elapsed = time.time() - overall_start

    exp_data = {
        "experiment_id": "EXP-002B",
        "description": "First Practical XGBoost Pairwise Matcher & Threshold Optimization",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_architecture": "XGBoost Gradient Boosted Decision Trees (native Booster)",
        "model_license": "Apache-2.0",
        "model_parameters_est": len(bst.get_dump()),
        "training_entities": len(train_s1_set),
        "validation_entities": len(val_s1_set),
        "training_candidate_pairs": train_pairs_count,
        "validation_candidate_pairs": val_pairs_count,
        "training_positives": train_pos,
        "training_negatives": train_neg,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "xgboost_config": {
            "n_estimators": 150,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        },
        "optimal_threshold": best_thresh,
        "validation_metrics": {
            "macro_f0_5": best_macro_f05,
            "macro_precision": best_report.macro_precision,
            "macro_recall": best_report.macro_recall,
            "total_predicted_links": best_report.total_tp + best_report.total_fp,
            "total_true_links": val_true_matches,
            "total_tp": best_report.total_tp,
            "total_fp": best_report.total_fp,
            "total_fn": best_report.total_fn,
            "singleton_count": best_report.singleton_count,
            "singleton_f0_5": best_report.singleton_f0_5,
            "non_singleton_count": best_report.non_singleton_count,
            "non_singleton_f0_5": best_report.non_singleton_f0_5,
        },
        "threshold_grid_results": grid_results,
        "feature_importances": {k: float(v) for k, v in feat_imp},
        "total_runtime_seconds": total_elapsed,
        "peak_memory_mb": peak_mem / (1024 * 1024),
    }

    args.output_exp.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_exp, "w", encoding="utf-8") as f:
        json.dump(exp_data, f, indent=2)
    print(f"\nSaved experiment record to {args.output_exp}", flush=True)

    # Log to centralized experiment logger
    logger = ExperimentLogger()
    record = ExperimentRecord(
        experiment_id="EXP-002B",
        hypothesis="First practical XGBoost pairwise matcher on Hybrid-T0.40-Cap100 candidates with macro F0.5 threshold tuning",
        data_split_description="1,000 train S1 vs 2,001 val S1 from validation_split_seed42_benchmark_10000.json",
        blocking_method="Hybrid-T0.40-Cap100",
        features=FEATURE_NAMES,
        model="XGBoost (n_estimators=150, max_depth=5, lr=0.1)",
        hyperparameters={
            "n_estimators": 150,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "threshold": best_thresh,
        },
        validation_method="Macro F0.5 entity-level evaluation",
        macro_F0_5=best_macro_f05,
        precision=best_report.macro_precision,
        recall=best_report.macro_recall,
        training_time_seconds=t1_xgb - t0_xgb,
        inference_time_seconds=t1_score - t0_score,
        notes=f"Optimal threshold: {best_thresh:.2f}, Predicted links: {best_report.total_tp + best_report.total_fp:,}, True links: {val_true_matches:,}",
        conclusion=f"XGBoost baseline achieved Macro F0.5={best_macro_f05:.4f}, Precision={best_report.macro_precision:.4f}, Recall={best_report.macro_recall:.4f}",
        next_experiment="Submission #2 generation on test set",
        extra_metrics={
            "train_pairs": train_pairs_count,
            "val_pairs": val_pairs_count,
            "train_pos": train_pos,
            "train_neg": train_neg,
            "optimal_threshold": best_thresh,
        },
    )
    logger.log(record)
    print("Logged to experiments/experiments_log.jsonl.", flush=True)


def worker_stream_chunk_wrapper(args):
    """Module-level wrapper for pool.map compatibility on Windows."""
    return worker_stream_search_pool(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
