#!/usr/bin/env python3
"""Amazon ML Challenge 2026 — EXP-002: Pairwise Matching & Classifier Experiments.

Compares:
1. EXP-002A: Rule-Based Exact + Numeric Verification Baseline
2. EXP-002B: Logistic Regression Baseline Classifier with Threshold Optimization
3. EXP-002C: HistGradientBoosting (GBDT) Classifier with F0.5 Threshold Optimization

Evaluates against the official fixed validation benchmark split (2,001 val entities).
Logs results via src.experiments.logger to experiments/ and experiments_log.jsonl.
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

# Set project root in path
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
from src.matching.features import (
    FEATURE_NAMES,
    extract_numbers,
    extract_pair_features,
    extract_tokens,
)
from src.matching.model import EntityMatchingClassifier


def run_exp002(
    split_path: str = "experiments/splits/validation_split_seed42_benchmark_10000.json",
    train_s1_sample_size: int = 1500,
    max_negatives_per_s1: int = 20,
):
    start_total_time = time.time()
    print("=" * 80)
    print("Amazon ML Challenge 2026 — EXP-002: Pairwise Matching & ML Classifier Experiments")
    print("=" * 80)

    # 1. Load Split
    print(f"Loading split from {split_path}...")
    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    all_train_s1 = split_data["train_s1_ids"]
    val_s1 = split_data["val_s1_ids"]
    
    # Deterministic subset of training S1 entities for feature extraction
    train_s1 = all_train_s1[:train_s1_sample_size]
    train_s1_set = set(train_s1)
    val_s1_set = set(val_s1)
    eval_s1_set = train_s1_set | val_s1_set

    print(f"Selected {len(train_s1):,} Train S1 entities and {len(val_s1):,} Validation S1 entities.")

    # 2. Load Ground Truth
    gt_path = Path("dataset/train/train_ground_truth.tsv")
    print(f"Parsing ground truth from {gt_path}...")
    full_train_gt = parse_ground_truth(gt_path, allowed_s1_ids=eval_s1_set)
    train_gt = {s1: full_train_gt.get(s1, set()) for s1 in train_s1}
    val_gt = {s1: full_train_gt.get(s1, set()) for s1 in val_s1}

    print(f"Train true matches: {sum(len(v) for v in train_gt.values()):,} across {len(train_gt):,} entities.")
    print(f"Val true matches  : {sum(len(v) for v in val_gt.values()):,} across {len(val_gt):,} entities.")

    # 3. Load S1 Records
    print("Loading Source 1 records...")
    s1_path = Path("dataset/train/train_source1.tsv")
    s1_records: Dict[str, Tuple[str, str, str, str]] = {}  # s1_id -> (name, addr, country, norm_name)
    with open(s1_path, "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and parts[0] in eval_s1_set:
                norm_name = normalize_name_conservative(parts[1])
                s1_records[parts[0]] = (parts[1], parts[2], parts[3], norm_name)

    print(f"Loaded {len(s1_records):,} Source 1 records for training & validation.")

    # Build Inverted Indexes for Candidate Retrieval
    print("Building blocking indexes...")
    index_exact = defaultdict(list)
    index_token = defaultdict(list)
    index_addr = defaultdict(list)

    for s1_id, (name, addr, country, norm_name) in s1_records.items():
        if norm_name:
            index_exact[(country, norm_name)].append(s1_id)
        for tk in extract_token_blocking_keys(name):
            index_token[(country, tk)].append(s1_id)
        for ak in extract_address_keys(addr):
            index_addr[(country, ak)].append(s1_id)

    # 4. Stream Source 2 and Source 3 to Generate Candidates & Extract Features
    print("Streaming Source 2 and Source 3 records to collect candidate pairs and features...")
    stream_start = time.time()

    train_pairs: List[Tuple[str, str]] = []
    train_features_list: List[List[float]] = []
    train_labels: List[int] = []

    val_pairs: List[Tuple[str, str]] = []
    val_features_list: List[List[float]] = []

    # Cache for tracking negatives per train S1
    train_neg_counts: Dict[str, int] = defaultdict(int)

    def process_record(tgt_id: str, name: str, addr: str, country: str):
        norm_name = normalize_name_conservative(name)
        matched_s1_ids: Set[str] = set()

        # Exact normalized name
        if norm_name:
            matches = index_exact.get((country, norm_name))
            if matches:
                matched_s1_ids.update(matches)

        # First token name
        for tk in extract_token_blocking_keys(name):
            matches = index_token.get((country, tk))
            if matches:
                matched_s1_ids.update(matches)

        # Address numeric keys
        for ak in extract_address_keys(addr):
            matches = index_addr.get((country, ak))
            if matches:
                matched_s1_ids.update(matches)

        if not matched_s1_ids:
            return

        for s1_id in matched_s1_ids:
            s1_info = s1_records[s1_id]
            is_train = s1_id in train_s1_set

            if is_train:
                is_pos = tgt_id in train_gt[s1_id]
                if is_pos:
                    feats = extract_pair_features(
                        s1_info[0], s1_info[1], s1_info[2],
                        tgt_id, name, addr, country,
                        norm_s1_name=s1_info[3], norm_tgt_name=norm_name,
                    )
                    train_pairs.append((s1_id, tgt_id))
                    train_features_list.append(feats)
                    train_labels.append(1)
                else:
                    if train_neg_counts[s1_id] < max_negatives_per_s1:
                        feats = extract_pair_features(
                            s1_info[0], s1_info[1], s1_info[2],
                            tgt_id, name, addr, country,
                            norm_s1_name=s1_info[3], norm_tgt_name=norm_name,
                        )
                        train_pairs.append((s1_id, tgt_id))
                        train_features_list.append(feats)
                        train_labels.append(0)
                        train_neg_counts[s1_id] += 1
            else:
                # Validation pair
                feats = extract_pair_features(
                    s1_info[0], s1_info[1], s1_info[2],
                    tgt_id, name, addr, country,
                    norm_s1_name=s1_info[3], norm_tgt_name=norm_name,
                )
                val_pairs.append((s1_id, tgt_id))
                val_features_list.append(feats)

    # Stream Source 2
    s2_path = Path("dataset/train/train_source2.tsv")
    print("  Streaming Source 2 records...")
    with open(s2_path, "r", encoding="utf-8-sig") as f:
        next(f)
        for i, line in enumerate(f, start=1):
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4:
                process_record(p[0], p[1], p[2], p[3])
            if i % 1500000 == 0:
                print(f"    Processed {i:,} Source 2 records (Train pairs: {len(train_pairs):,}, Val pairs: {len(val_pairs):,})...")

    # Stream Source 3
    s3_path = Path("dataset/train/train_source3.tsv")
    print("  Streaming Source 3 records...")
    with open(s3_path, "r", encoding="utf-8-sig") as f:
        next(f)
        for i, line in enumerate(f, start=1):
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 4:
                process_record(p[0], p[1], p[2], p[3])
            if i % 1500000 == 0:
                print(f"    Processed {i:,} Source 3 records (Train pairs: {len(train_pairs):,}, Val pairs: {len(val_pairs):,})...")

    print(f"Streaming and feature extraction completed in {time.time() - stream_start:.1f}s.")
    print(f"Train Dataset: {len(train_features_list):,} pairs (Positives: {sum(train_labels):,}, Negatives: {len(train_labels) - sum(train_labels):,})")
    print(f"Val Dataset  : {len(val_features_list):,} candidate pairs.")

    X_train = np.array(train_features_list, dtype=np.float32)
    y_train = np.array(train_labels, dtype=np.int32)
    X_val = np.array(val_features_list, dtype=np.float32)

    logger = ExperimentLogger("experiments")

    # -------------------------------------------------------------------------
    # EXP-002A: Baseline Rule-Based Matcher
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Running EXP-002A: Baseline Rule-Based Matching (Exact Name + Address Numeric Verification)")
    print("=" * 80)
    t0 = time.time()
    rule_predictions: Dict[str, Set[str]] = {s1: set() for s1 in val_s1}

    # In rule-based matching: exact normalized name + numeric address overlap or address token overlap
    for (s1_id, tgt_id), feats in zip(val_pairs, val_features_list):
        exact_name = feats[0] == 1.0
        both_have_nums = feats[8] == 1.0
        num_overlap = feats[6] >= 1.0
        addr_tok_jacc = feats[9]

        is_match = False
        if exact_name:
            if both_have_nums and num_overlap:
                is_match = True
            elif not both_have_nums and addr_tok_jacc >= 0.2:
                is_match = True

        if is_match:
            rule_predictions[s1_id].add(tgt_id)

    report_2a = evaluate_predictions(rule_predictions, val_gt)
    timing_2a = time.time() - t0
    print(report_2a.summary())

    record_2a = ExperimentRecord(
        experiment_id="EXP-002A",
        hypothesis="Evaluate rule-based exact name + address numeric verification as baseline matching method.",
        data_split_description="Validation benchmark split (2,001 val S1) against full S2+S3 search pool.",
        blocking_method="Validated Inverted Index Backbone",
        features=["name_exact_match", "addr_num_overlap", "addr_token_jaccard"],
        model="Heuristic Decision Rules",
        hyperparameters={"exact_name": True, "min_addr_num_overlap": 1, "min_addr_jaccard": 0.2},
        validation_method="Macro-averaged F0.5 (Official Metric)",
        macro_F0_5=report_2a.macro_f0_5,
        precision=report_2a.macro_precision,
        recall=report_2a.macro_recall,
        training_time_seconds=0.0,
        inference_time_seconds=round(timing_2a, 3),
        notes=f"Singletons: {report_2a.singleton_count} (F0.5: {report_2a.singleton_f0_5:.4f}), Non-Singletons: {report_2a.non_singleton_count} (F0.5: {report_2a.non_singleton_f0_5:.4f})",
        conclusion=f"Rule-based baseline achieved Macro F0.5 of {report_2a.macro_f0_5:.4f} with {report_2a.macro_precision:.4f} precision and {report_2a.macro_recall:.4f} recall.",
        next_experiment="EXP-002B",
        extra_metrics=report_2a.to_dict(),
    )
    logger.log(record_2a)

    # -------------------------------------------------------------------------
    # EXP-002B: Logistic Regression Baseline Classifier
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Running EXP-002B: Logistic Regression Classifier with F0.5 Threshold Optimization")
    print("=" * 80)
    t_train_b = time.time()
    clf_lr = EntityMatchingClassifier(model_type="logistic", random_state=42)
    clf_lr.fit(X_train, y_train)
    train_time_b = time.time() - t_train_b

    t_inf_b = time.time()
    val_probs_lr = clf_lr.predict_proba(X_val)
    inf_time_b = time.time() - t_inf_b

    best_t_b, best_f05_b, scores_b = clf_lr.optimize_threshold_f05(
        val_pairs, val_probs_lr, val_gt, threshold_range=(0.30, 0.95, 0.05)
    )
    print(f"Optimal threshold for Logistic Regression: T = {best_t_b:.2f} (Macro F0.5: {best_f05_b:.4f})")

    lr_predictions: Dict[str, Set[str]] = {s1: set() for s1 in val_s1}
    for (s1_id, tgt_id), prob in zip(val_pairs, val_probs_lr):
        if prob >= best_t_b:
            lr_predictions[s1_id].add(tgt_id)

    report_2b = evaluate_predictions(lr_predictions, val_gt)
    print(report_2b.summary())

    record_2b = ExperimentRecord(
        experiment_id="EXP-002B",
        hypothesis="Evaluate standardized linear Logistic Regression classifier with F0.5 threshold calibration.",
        data_split_description="Validation benchmark split (2,001 val S1) against full S2+S3 search pool.",
        blocking_method="Validated Inverted Index Backbone",
        features=FEATURE_NAMES,
        model="LogisticRegression(class_weight='balanced')",
        hyperparameters={"optimal_threshold": best_t_b, "C": 1.0, "scaler": "StandardScaler"},
        validation_method="Macro-averaged F0.5 (Official Metric)",
        macro_F0_5=report_2b.macro_f0_5,
        precision=report_2b.macro_precision,
        recall=report_2b.macro_recall,
        training_time_seconds=round(train_time_b, 3),
        inference_time_seconds=round(inf_time_b, 3),
        notes=f"Singletons: {report_2b.singleton_count} (F0.5: {report_2b.singleton_f0_5:.4f}), Non-Singletons: {report_2b.non_singleton_count} (F0.5: {report_2b.non_singleton_f0_5:.4f})",
        conclusion=f"Logistic Regression achieved Macro F0.5 of {report_2b.macro_f0_5:.4f} at threshold T={best_t_b:.2f}.",
        next_experiment="EXP-002C",
        extra_metrics=report_2b.to_dict(),
    )
    logger.log(record_2b)

    # -------------------------------------------------------------------------
    # EXP-002C: HistGradientBoosting (GBDT) Classifier
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Running EXP-002C: HistGradientBoosting (GBDT) Classifier with F0.5 Threshold Optimization")
    print("=" * 80)
    t_train_c = time.time()
    clf_gbdt = EntityMatchingClassifier(
        model_type="hist_gbdt",
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=42,
    )
    clf_gbdt.fit(X_train, y_train)
    train_time_c = time.time() - t_train_c

    t_inf_c = time.time()
    val_probs_gbdt = clf_gbdt.predict_proba(X_val)
    inf_time_c = time.time() - t_inf_c

    best_t_c, best_f05_c, scores_c = clf_gbdt.optimize_threshold_f05(
        val_pairs, val_probs_gbdt, val_gt, threshold_range=(0.30, 0.95, 0.05)
    )
    print(f"Optimal threshold for HistGradientBoosting: T = {best_t_c:.2f} (Macro F0.5: {best_f05_c:.4f})")

    gbdt_predictions: Dict[str, Set[str]] = {s1: set() for s1 in val_s1}
    for (s1_id, tgt_id), prob in zip(val_pairs, val_probs_gbdt):
        if prob >= best_t_c:
            gbdt_predictions[s1_id].add(tgt_id)

    report_2c = evaluate_predictions(gbdt_predictions, val_gt)
    print(report_2c.summary())

    record_2c = ExperimentRecord(
        experiment_id="EXP-002C",
        hypothesis="Evaluate non-linear HistGradientBoosting (GBDT) with feature interactions and calibrated F0.5 thresholding.",
        data_split_description="Validation benchmark split (2,001 val S1) against full S2+S3 search pool.",
        blocking_method="Validated Inverted Index Backbone",
        features=FEATURE_NAMES,
        model="HistGradientBoostingClassifier(learning_rate=0.08, max_iter=150, class_weight='balanced')",
        hyperparameters={
            "optimal_threshold": best_t_c,
            "learning_rate": 0.08,
            "max_iter": 150,
            "max_leaf_nodes": 31,
        },
        validation_method="Macro-averaged F0.5 (Official Metric)",
        macro_F0_5=report_2c.macro_f0_5,
        precision=report_2c.macro_precision,
        recall=report_2c.macro_recall,
        training_time_seconds=round(train_time_c, 3),
        inference_time_seconds=round(inf_time_c, 3),
        notes=f"Singletons: {report_2c.singleton_count} (F0.5: {report_2c.singleton_f0_5:.4f}), Non-Singletons: {report_2c.non_singleton_count} (F0.5: {report_2c.non_singleton_f0_5:.4f})",
        conclusion=f"HistGradientBoosting achieved Macro F0.5 of {report_2c.macro_f0_5:.4f} at threshold T={best_t_c:.2f}.",
        next_experiment="EXP-003",
        extra_metrics=report_2c.to_dict(),
    )
    logger.log(record_2c)

    # -------------------------------------------------------------------------
    # Comparative Summary Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXP-002 COMPARATIVE EVALUATION SUMMARY (Macro-Averaged F0.5)")
    print("=" * 80)
    print(f"{'Experiment':<12} | {'Model':<28} | {'Macro F0.5':<10} | {'Macro Prec':<10} | {'Macro Rec':<10} | {'Singleton F0.5':<14} | {'Non-Single F0.5'}")
    print("-" * 115)
    print(f"{'EXP-002A':<12} | {'Rule-Based Heuristic':<28} | {report_2a.macro_f0_5:<10.4f} | {report_2a.macro_precision:<10.4f} | {report_2a.macro_recall:<10.4f} | {report_2a.singleton_f0_5:<14.4f} | {report_2a.non_singleton_f0_5:.4f}")
    print(f"{'EXP-002B':<12} | {'Logistic Regression':<28} | {report_2b.macro_f0_5:<10.4f} | {report_2b.macro_precision:<10.4f} | {report_2b.macro_recall:<10.4f} | {report_2b.singleton_f0_5:<14.4f} | {report_2b.non_singleton_f0_5:.4f}")
    print(f"{'EXP-002C':<12} | {'HistGradientBoosting (GBDT)':<28} | {report_2c.macro_f0_5:<10.4f} | {report_2c.macro_precision:<10.4f} | {report_2c.macro_recall:<10.4f} | {report_2c.singleton_f0_5:<14.4f} | {report_2c.non_singleton_f0_5:.4f}")
    print("=" * 80)
    print(f"Total runtime: {time.time() - start_total_time:.1f}s")


if __name__ == "__main__":
    split_file = sys.argv[1] if len(sys.argv) > 1 else "experiments/splits/validation_split_seed42_benchmark_10000.json"
    run_exp002(split_path=split_file)
