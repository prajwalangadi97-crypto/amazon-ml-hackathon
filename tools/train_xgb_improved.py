#!/usr/bin/env python3
"""EXP-002B-IMPROVED: GPU-accelerated XGBoost with tuned hyperparameters.

Improvements over baseline train_and_validate_xgb.py:
  1. device='cuda' — uses RTX 3050 GPU for XGBoost tree building
  2. scale_pos_weight=27.6 — corrects for 28:1 class imbalance
  3. num_boost_round=400 (was 150) — more trees
  4. max_depth=6 (was 5) — slightly deeper trees
  5. learning_rate=0.05 (was 0.1) — slower, more precise learning
  6. min_child_weight=5 — prevents overfitting on rare positives
  7. gamma=0.1 — minimum loss reduction for splitting
  8. reg_alpha=0.1, reg_lambda=1.5 — L1+L2 regularization
  9. subsample=0.85, colsample_bytree=0.85 — better stochasticity
 10. Finer threshold grid including 0.55, 0.62, 0.65
 11. early_stopping_rounds=30 — stops when val logloss stops improving
 12. Saves improved model to artifacts/xgb_matcher_improved.json
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
    "cross_field_name_loc",
    "cheap_candidate_score",
    "context_key_overlap",
]


def parse_args():
    p = argparse.ArgumentParser(description="Improved GPU XGBoost trainer")
    p.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "dataset")
    p.add_argument("--output-model", type=Path, default=PROJECT_ROOT / "artifacts" / "xgb_matcher_improved.json")
    p.add_argument("--experiment-dir", type=Path, default=PROJECT_ROOT / "experiments")
    p.add_argument("--split-file", type=Path,
                   default=PROJECT_ROOT / "experiments" / "splits" / "validation_split_seed42_benchmark_10000.json")
    p.add_argument("--num-workers", type=int, default=min(8, mp.cpu_count()))
    p.add_argument("--num-boost-round", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--scale-pos-weight", type=float, default=27.6)
    p.add_argument("--use-gpu", action="store_true", default=True)
    return p.parse_args()


def _worker_stream_and_score(args_tuple):
    """Parallel worker: stream source file chunk and score candidates."""
    (
        worker_id, filepath, chunk_start, chunk_end,
        s1_preprocessed_list, policy, project_root_str
    ) = args_tuple

    import sys
    sys.path.insert(0, project_root_str)
    from src.blocking.normalizers import extract_token_blocking_keys, extract_address_keys
    from src.matching.candidate_compression import (
        CandidateCompressor, CompressionPolicy,
        fast_score_candidate, preprocess_record,
    )

    compressor = CandidateCompressor(policy)
    s1_index: Dict[str, List] = {}
    for prep in s1_preprocessed_list:
        for key in prep.blocking_keys:
            s1_index.setdefault(key, []).append(prep)

    candidates: Dict[str, List[Tuple[str, float]]] = {}
    line_count = 0
    scored = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i < chunk_start:
                continue
            if chunk_end is not None and i >= chunk_end:
                break
            if i == 0:
                continue  # header
            line_count += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            tid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
            t_keys = set(extract_token_blocking_keys(name, country))
            t_keys |= set(extract_address_keys(addr, country))
            seen_s1 = set()
            for key in t_keys:
                for s1_prep in s1_index.get(key, []):
                    if s1_prep.record_id in seen_s1:
                        continue
                    seen_s1.add(s1_prep.record_id)
                    score = fast_score_candidate(s1_prep, name, addr, country)
                    scored += 1
                    sid = s1_prep.record_id
                    if sid not in candidates:
                        candidates[sid] = []
                    candidates[sid].append((tid, score))

    t1 = time.time()
    for sid in candidates:
        candidates[sid] = compressor.compress(candidates[sid])

    return candidates, line_count, scored


def main():
    args = parse_args()
    tracemalloc.start()
    t_global_start = time.time()

    print("=" * 80)
    print("EXP-002B-IMPROVED: GPU XGBoost Pairwise Matcher")
    print(f"GPU: {'ENABLED (cuda)' if args.use_gpu else 'DISABLED'} | "
          f"Workers: {args.num_workers} | Boost rounds: {args.num_boost_round}")
    print(f"Hyperparams: max_depth={args.max_depth}, lr={args.learning_rate}, "
          f"scale_pos_weight={args.scale_pos_weight}")
    print("=" * 80)

    # 1. Load split
    with open(args.split_file, "r", encoding="utf-8") as f:
        split = json.load(f)
    train_s1_set = set(split["train_s1_ids"])
    val_s1_set = set(split["val_s1_ids"])
    train_s1_list = split["train_s1_ids"]
    val_s1_list = split["val_s1_ids"]
    print(f"Disjoint S1 Partition: Train={len(train_s1_list):,} entities, Val={len(val_s1_list):,} entities.")

    # 2. Load ground truth
    gt_file = args.data_dir / "train" / "train_ground_truth.tsv"
    full_gt = parse_ground_truth(str(gt_file))  # returns {sid: set_of_tids}
    train_gt = {sid: tids for sid, tids in full_gt.items() if sid in train_s1_set}
    val_gt   = {sid: tids for sid, tids in full_gt.items() if sid in val_s1_set}
    print(f"Ground Truth: Train={sum(len(v) for v in train_gt.values()):,} true links, "
          f"Val={sum(len(v) for v in val_gt.values()):,} true links.")

    # 3. Preprocess S1
    t0 = time.time()
    s1_file = args.data_dir / "train" / "train_source1.tsv"
    all_s1_prep = []
    with open(s1_file, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            rid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
            if rid in train_s1_set or rid in val_s1_set:
                all_s1_prep.append(preprocess_record(rid, name, addr, country))
    print(f"Preprocessed {len(all_s1_prep):,} S1 records in {time.time()-t0:.2f}s.")

    # 4. Candidate compression policy
    policy = CompressionPolicy(
        hybrid_threshold=0.40,
        global_cap=100,
        use_two_tier=False,
    )

    # 5. Stream source2 and source3
    train_candidates: Dict[str, List[Tuple[str, float]]] = {s: [] for s in train_s1_list}
    val_candidates: Dict[str, List[Tuple[str, float]]] = {s: [] for s in val_s1_list}

    for source_file in ["train_source2.tsv", "train_source3.tsv"]:
        fpath = args.data_dir / "train" / source_file
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            total_lines = sum(1 for _ in f)
        print(f"\nStreaming {source_file} ({total_lines:,} rows) with {args.num_workers} parallel workers...")

        chunk_size = (total_lines + args.num_workers - 1) // args.num_workers
        chunks = [(i * chunk_size, min((i + 1) * chunk_size, total_lines)) for i in range(args.num_workers)]

        worker_args = [
            (wid, str(fpath), cs, ce, all_s1_prep, policy, str(PROJECT_ROOT))
            for wid, (cs, ce) in enumerate(chunks)
        ]

        t0_stream = time.time()
        with mp.Pool(args.num_workers) as pool:
            results = pool.map(_worker_stream_and_score, worker_args)

        for worker_cands, lc, sc in results:
            for sid, pairs in worker_cands.items():
                if sid in train_candidates:
                    train_candidates[sid].extend(pairs)
                elif sid in val_candidates:
                    val_candidates[sid].extend(pairs)

        print(f"Streaming complete in {time.time()-t0_stream:.2f}s.")

    # Re-compress after merging
    compressor = CandidateCompressor(policy)
    for sid in train_candidates:
        train_candidates[sid] = compressor.compress(train_candidates[sid])
    for sid in val_candidates:
        val_candidates[sid] = compressor.compress(val_candidates[sid])

    total_train_pairs = sum(len(v) for v in train_candidates.values())
    total_val_pairs = sum(len(v) for v in val_candidates.values())
    print(f"\nCompressed Candidates: Train={total_train_pairs:,} pairs, Val={total_val_pairs:,} pairs.")

    # 6. Collect needed target IDs
    needed_tids: Set[str] = set()
    for pairs in train_candidates.values():
        needed_tids.update(t for t, _ in pairs)
    for pairs in val_candidates.values():
        needed_tids.update(t for t, _ in pairs)
    print(f"Unique target records needed for feature extraction: {len(needed_tids):,}")

    # 7. Preprocess target records
    t0_tgt = time.time()
    target_prep: Dict[str, PreprocessedRecord] = {}
    for source_file in ["train_source2.tsv", "train_source3.tsv"]:
        fpath = args.data_dir / "train" / source_file
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                rid = parts[0]
                if rid in needed_tids and rid not in target_prep:
                    target_prep[rid] = preprocess_record(rid, parts[1], parts[2], parts[3])
    print(f"Preprocessed {len(target_prep):,} target records in {time.time()-t0_tgt:.2f}s.")

    # 8. Build feature matrices
    print("\nBuilding training and validation feature matrices...")
    t0_feats = time.time()
    s1_prep_map = {p.record_id: p for p in all_s1_prep}

    def build_matrix(candidates, gt):
        X_rows, y_rows, pair_keys = [], [], []
        for sid, pairs in candidates.items():
            s1_p = s1_prep_map.get(sid)
            if s1_p is None:
                continue
            pos_set = gt.get(sid, set())
            for tid, cheap_score in pairs:
                t_p = target_prep.get(tid)
                if t_p is None:
                    continue
                feats = extract_features_from_preprocessed(s1_p, t_p, cheap_score)
                X_rows.append(feats)
                y_rows.append(1 if tid in pos_set else 0)
                pair_keys.append((sid, tid))
        return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32), pair_keys

    X_train, y_train, train_pairs = build_matrix(train_candidates, train_gt)
    X_val, y_val, val_pairs = build_matrix(val_candidates, val_gt)
    print(f"Feature Extraction complete in {time.time()-t0_feats:.2f}s.")
    train_pos = int(np.sum(y_train == 1))
    train_neg = int(np.sum(y_train == 0))
    val_pos = int(np.sum(y_val == 1))
    val_neg = int(np.sum(y_val == 0))
    print(f"Train Matrix: {X_train.shape} | Positives={train_pos:,} | Negatives={train_neg:,} (ratio: {train_neg/max(train_pos,1):.1f}:1)")
    print(f"Val Matrix  : {X_val.shape} | Positives={val_pos:,} | Negatives={val_neg:,} (ratio: {val_neg/max(val_pos,1):.1f}:1)")

    # 9. Train improved XGBoost with GPU
    device = "cuda" if args.use_gpu else "cpu"
    print(f"\nTraining Improved XGBoost (device={device}, rounds={args.num_boost_round}, "
          f"max_depth={args.max_depth}, lr={args.learning_rate}, scale_pos_weight={args.scale_pos_weight})...")
    t0_xgb = time.time()
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "device": device,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "scale_pos_weight": args.scale_pos_weight,
        "nthread": args.num_workers,
        "seed": 42,
    }

    evals_result = {}
    bst = xgb.train(
        params, dtrain,
        num_boost_round=args.num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        evals_result=evals_result,
        verbose_eval=50,
    )
    print(f"XGBoost training complete in {time.time()-t0_xgb:.2f}s. "
          f"Best iteration: {bst.best_iteration}")

    # Save model
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(args.output_model))
    print(f"Saved model to {args.output_model}")

    # 10. Score validation
    t0_score = time.time()
    val_probs = bst.predict(dval)
    print(f"Scored {len(val_probs):,} validation pairs in {time.time()-t0_score:.2f}s.")

    # 11. Threshold grid (finer than baseline)
    threshold_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                      0.50, 0.55, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80, 0.90]
    print("\nEvaluating Entity-Level Macro F0.5 over Threshold Grid...")
    print(f"{'Threshold':<10} | {'Macro F0.5':<11} | {'Precision':<10} | {'Recall':<10} | "
          f"{'Pred Links':<11} | {'Zero Pred':<10} | {'Multi Pred':<10}")
    print("-" * 100)

    best_thresh = 0.50
    best_macro_f05 = -1.0
    best_report = None
    grid_results = []

    for thresh in threshold_grid:
        predictions: Dict[str, Set[str]] = {sid: set() for sid in val_s1_list}
        for (sid, tid), prob in zip(val_pairs, val_probs):
            if prob >= thresh:
                predictions[sid].add(tid)
        report = evaluate_predictions(predictions, val_gt)
        total_pred_links = sum(len(v) for v in predictions.values())
        zero_pred = sum(1 for v in predictions.values() if len(v) == 0)
        multi_pred = sum(1 for v in predictions.values() if len(v) > 1)

        grid_results.append({
            "threshold": thresh,
            "macro_f0_5": report.macro_f0_5,
            "macro_precision": report.macro_precision,
            "macro_recall": report.macro_recall,
            "total_predicted_links": total_pred_links,
        })
        print(f"{thresh:<10.2f} | {report.macro_f0_5:<11.4f} | {report.macro_precision:<10.4f} | "
              f"{report.macro_recall:<10.4f} | {total_pred_links:<11,} | {zero_pred:<10,} | {multi_pred:<10,}")

        if report.macro_f0_5 > best_macro_f05:
            best_macro_f05 = report.macro_f0_5
            best_thresh = thresh
            best_report = report

    print("-" * 100)
    print(f"Optimal Threshold: {best_thresh:.2f}  |  Best Macro F0.5: {best_macro_f05:.4f}")
    print(f"Precision: {best_report.macro_precision:.4f}  |  Recall: {best_report.macro_recall:.4f}")

    # 12. Feature importance
    print("\nTop Feature Importances (Gain):")
    scores = bst.get_score(importance_type="gain")
    for feat, gain in sorted(scores.items(), key=lambda x: -x[1])[:10]:
        print(f"  {feat:<30}: {gain:.4f}")

    print(f"\nTotal wall time: {time.time()-t_global_start:.1f}s")
    print(f"Model saved: {args.output_model}")


if __name__ == "__main__":
    main()
