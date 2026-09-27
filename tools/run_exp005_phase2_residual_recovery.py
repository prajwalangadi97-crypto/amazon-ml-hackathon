#!/usr/bin/env python3
"""EXP-005 Phase 2: Selective Residual Recovery Benchmark.

Investigates selective residual recovery for Source1 entities that remain
unresolved after the Phase 1 adaptive candidate ranking stage.

Strict Constraints:
- 100% competition compliance (zero external data, zero internet).
- Final candidate set must contain every candidate fed into the model.
- Frozen V3 XGBoost model (artifacts/xgb_matcher_exp004b.json).
- Strict acceptance criteria to protect precision on Macro F0.5.
- Preserves correct empty predictions for true singletons.
- Compares directly against Phase 1 winning baseline (Macro F0.5 = 0.843060).
"""

import argparse
import collections
import gc
import json
import os
from pathlib import Path
import pickle
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import unicodedata

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

from src.data.ground_truth import parse_ground_truth
from src.evaluation.metrics import evaluate_predictions
from src.matching.candidate_compression import (
    compute_cheap_candidate_score,
    extract_features_from_preprocessed,
    preprocess_record,
    PreprocessedRecord,
)
from src.matching.decision_layer import (
    DEFAULT_V3_DECISION_CONFIG,
    filter_entity_candidates,
)
from tools.run_exp005_adaptive_ranking import (
    fast_score_candidate_detailed,
    rank_candidates_channel_consensus_first,
    EXP004B_FEATURE_NAMES,
)


def extract_selective_numeric_address_keys(address: str) -> List[str]:
    """Extract clean, stripped numeric building/plot keys paired with city tokens."""
    if not address or not address.strip():
        return []
    norm = unicodedata.normalize("NFKC", address).lower()
    # Strip special punctuation like #, -, / inside number strings
    clean_addr = re.sub(r"[#*@]", " ", norm)
    nums = re.findall(r"\b\d+\b", clean_addr)
    words = [
        w for w in re.findall(r"[a-z]+", clean_addr)
        if len(w) >= 3 and w not in {"the", "and", "near", "opp", "road", "street", "lane", "floor", "unit"}
    ]
    keys = []
    if nums and words:
        for n in nums[:2]:
            for w in words[:2]:
                keys.append(f"rec_num_{n}_{w}")
    return list(dict.fromkeys(keys))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, default=Path("experiments/splits/validation_split_seed42_benchmark_10000.json"))
    parser.add_argument("--ground-truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--source1", type=Path, default=Path("dataset/train/train_source1.tsv"))
    parser.add_argument("--source2", type=Path, default=Path("dataset/train/train_source2.tsv"))
    parser.add_argument("--source3", type=Path, default=Path("dataset/train/train_source3.tsv"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/xgb_matcher_exp004b.json"))
    parser.add_argument("--cache-pool", type=Path, default=Path("artifacts/exp005/val_raw_candidates.pkl"))
    parser.add_argument("--output-json", type=Path, default=Path("experiments/EXP-005B-residual-recovery.json"))
    args = parser.parse_args()

    overall_start = time.time()

    print("=" * 95, flush=True)
    print("EXP-005 PHASE 2: SELECTIVE RESIDUAL RECOVERY BENCHMARK", flush=True)
    print("=" * 95, flush=True)

    # 1. Load Validation Split & Ground Truth
    with open(args.split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    val_s1_list: List[str] = split_data["val_s1_ids"]
    val_s1_set = set(val_s1_list)
    val_gt = parse_ground_truth(args.ground_truth, allowed_s1_ids=val_s1_set)
    print(f"Validation Entities: {len(val_s1_list):,} | Ground Truth: {sum(len(t) for t in val_gt.values()):,} true links", flush=True)

    # 2. Load Preprocessed S1 Records
    val_s1_records: Dict[str, Tuple[str, str, str, str]] = {}
    val_s1_prep: Dict[str, PreprocessedRecord] = {}

    with open(args.source1, "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and parts[0] in val_s1_set:
                sid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                val_s1_records[sid] = (sid, name, addr, country)
                val_s1_prep[sid] = preprocess_record(sid, name, addr, country)

    s1_country_map = {sid: val_s1_records[sid][3] for sid in val_s1_list}

    # 3. Load Raw Candidate Pool & Phase 1 Selection
    print(f"Loading raw candidate pool from {args.cache_pool}...", flush=True)
    with open(args.cache_pool, "rb") as f:
        raw_pool = pickle.load(f)

    print("Generating Phase 1 candidate set (Channel Consensus First, Cap=100)...", flush=True)
    phase1_cands_meta = {}
    phase1_needed_tids = set()

    for sid in val_s1_list:
        raw_c = raw_pool.get(sid, [])
        selected = rank_candidates_channel_consensus_first(raw_c, cap=100)
        phase1_cands_meta[sid] = selected
        phase1_needed_tids.update(t for t, _ in selected)

    # 4. Load Needed Target Records for Phase 1
    print(f"Retrieving {len(phase1_needed_tids):,} target records...", flush=True)
    target_prep_map: Dict[str, PreprocessedRecord] = {}
    for src_fp in [args.source2, args.source3]:
        with open(src_fp, "r", encoding="utf-8-sig") as f:
            next(f)
            for line in f:
                pos = line.find("\t")
                if pos != -1 and line[:pos] in phase1_needed_tids:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 4:
                        target_prep_map[parts[0]] = preprocess_record(parts[0], parts[1], parts[2], parts[3])
                        if len(target_prep_map) == len(phase1_needed_tids):
                            break
        if len(target_prep_map) == len(phase1_needed_tids):
            break

    # 5. Score Phase 1 with XGBoost
    print("Scoring Phase 1 baseline...", flush=True)
    bst = xgb.Booster()
    bst.load_model(str(args.model_path))
    bst.set_param({"nthread": 8})

    def score_and_predict(cands_meta, target_map):
        X_rows = []
        pair_keys = []
        for sid in val_s1_list:
            s1_p = val_s1_prep[sid]
            for tid, meta in cands_meta.get(sid, []):
                tgt_p = target_map.get(tid)
                if tgt_p is None:
                    continue
                feats = extract_features_from_preprocessed(s1_p, tgt_p)
                feats["cheap_candidate_score"] = compute_cheap_candidate_score(feats)
                feats.update(meta)
                X_rows.append([feats[c] for c in EXP004B_FEATURE_NAMES])
                pair_keys.append((sid, tid))

        dmat = xgb.DMatrix(np.array(X_rows, dtype=np.float32), feature_names=EXP004B_FEATURE_NAMES)
        probs = bst.predict(dmat)

        s1_cand_scores = collections.defaultdict(list)
        for (sid, tid), p in zip(pair_keys, probs):
            s1_cand_scores[sid].append((tid, float(p)))

        preds = {}
        for sid in val_s1_list:
            cands = s1_cand_scores.get(sid, [])
            country = s1_country_map.get(sid, "")
            matched = filter_entity_candidates(cands, country, DEFAULT_V3_DECISION_CONFIG)
            preds[sid] = set(matched)
        return preds, s1_cand_scores, len(pair_keys)

    phase1_preds, phase1_scores, phase1_cand_count = score_and_predict(phase1_cands_meta, target_prep_map)
    phase1_rep = evaluate_predictions(phase1_preds, val_gt)

    print(f"\nPhase 1 Baseline Verified:")
    print(f"  Macro F0.5 = {phase1_rep.macro_f0_5:.6f} | Precision = {phase1_rep.macro_precision:.6f} | Recall = {phase1_rep.macro_recall:.6f} | TP = {phase1_rep.total_tp} | FP = {phase1_rep.total_fp}")

    # 6. Identify Residual Entities (Zero predicted links in Phase 1)
    residual_s1 = [sid for sid in val_s1_list if not phase1_preds[sid]]
    print(f"\nResidual Source 1 Entities: {len(residual_s1)} / {len(val_s1_list)} ({(len(residual_s1)/len(val_s1_list)):.2%})", flush=True)

    # 7. Evaluate Recovery Strategies for Residual Entities
    # Strategy R1: Secondary pool scan for residual entities (evaluate candidates 101-200 from raw pool)
    print("\nEvaluating Strategy R1: Secondary Pool Scan for Residual Entities...", flush=True)
    r1_cands_meta = {}
    r1_needed_tids = set(phase1_needed_tids)

    for sid in val_s1_list:
        base_items = phase1_cands_meta[sid]
        if sid not in residual_s1:
            r1_cands_meta[sid] = base_items
        else:
            # For residual entities only: add top 20 candidates from secondary pool sorted by address/modality evidence
            raw_c = raw_pool.get(sid, [])
            sel_tids = {t for t, _ in base_items}
            remaining = [c for c in raw_c if c["tid"] not in sel_tids]

            # Filter remaining for high-evidence signals (multi-channel, high address score, or PIN/house match)
            informative = [
                c for c in remaining
                if c["ch_count"] >= 2 or c["s_addr"] >= 0.60 or c["has_pin"] or c["has_house"]
            ]
            informative.sort(
                key=lambda c: (
                    -c["ch_count"],
                    c["best_tier"],
                    -max(c["score"], c["s_addr"]),
                    -c["spec_sum"],
                )
            )
            top_rec = informative[:20]
            pool_size = float(len(base_items) + len(top_rec))

            combined_items = list(base_items)
            for rank_idx, c in enumerate(top_rec, start=len(base_items)):
                meta = {
                    "channel_agreement_count": float(c["ch_count"]),
                    "retrieval_priority_tier": float(c["best_tier"]),
                    "reciprocal_retrieval_rank": 1.0 / (rank_idx + 1.0),
                    "s1_candidate_pool_size": pool_size,
                }
                combined_items.append((c["tid"], meta))
                r1_needed_tids.add(c["tid"])

            r1_cands_meta[sid] = combined_items

    # Retrieve new target records for R1
    missing_r1_tids = r1_needed_tids - set(target_prep_map.keys())
    if missing_r1_tids:
        print(f"Retrieving {len(missing_r1_tids):,} additional target records for R1...", flush=True)
        for src_fp in [args.source2, args.source3]:
            with open(src_fp, "r", encoding="utf-8-sig") as f:
                next(f)
                for line in f:
                    pos = line.find("\t")
                    if pos != -1 and line[:pos] in missing_r1_tids:
                        parts = line.rstrip("\r\n").split("\t")
                        if len(parts) >= 4:
                            target_prep_map[parts[0]] = preprocess_record(parts[0], parts[1], parts[2], parts[3])
                            if len(target_prep_map) == len(r1_needed_tids):
                                break
            if len(target_prep_map) == len(r1_needed_tids):
                break

    r1_preds, r1_scores, r1_cand_count = score_and_predict(r1_cands_meta, target_prep_map)
    r1_rep = evaluate_predictions(r1_preds, val_gt)

    # 8. Strategy R2: Strict Threshold Residual Acceptance
    # For residual entities, only accept matches if P >= country_threshold + 0.05 to guard against false merges
    print("\nEvaluating Strategy R2: Strict Threshold Residual Acceptance...", flush=True)
    r2_preds = {}
    for sid in val_s1_list:
        if sid not in residual_s1:
            r2_preds[sid] = set(phase1_preds[sid])
        else:
            cands = r1_scores.get(sid, [])
            country = s1_country_map.get(sid, "")
            base_thresh = DEFAULT_V3_DECISION_CONFIG.get_threshold(country)
            strict_thresh = base_thresh + 0.05
            strict_cfg = DEFAULT_V3_DECISION_CONFIG.__class__(
                country_thresholds={
                    "France": 0.45 + 0.05,
                    "US": 0.65 + 0.05,
                    "India": 0.55 + 0.05,
                },
                margin_delta=0.30,
                default_threshold=0.60 + 0.05,
                enforce_group_margin=True,
            )
            matched = filter_entity_candidates(cands, country, strict_cfg)
            r2_preds[sid] = set(matched)

    r2_rep = evaluate_predictions(r2_preds, val_gt)

    # 9. Comparison Table
    print("\n" + "=" * 125, flush=True)
    print("EXPERIMENTAL EVALUATION RESULTS: EXP-005 PHASE 2 (SELECTIVE RESIDUAL RECOVERY)", flush=True)
    print("=" * 125, flush=True)
    print(f"{'Strategy':<38} | {'Macro F0.5':<11} | {'Precision':<10} | {'Recall':<10} | {'Pred':<7} | {'TP':<6} | {'FP':<5} | {'Single Acc'}", flush=True)
    print("-" * 125, flush=True)
    print(f"{'1. V3 Baseline (Ref)':<38} | {0.831214:<11.6f} | {0.895836:<10.6f} | {0.720218:<10.6f} | {5219:<7} | {4956:<6} | {263:<5} | {0.7679:<10.4f}", flush=True)
    print(f"{'2. Phase 1 (Consensus First Cap100)':<38} | {phase1_rep.macro_f0_5:<11.6f} | {phase1_rep.macro_precision:<10.6f} | {phase1_rep.macro_recall:<10.6f} | {sum(len(p) for p in phase1_preds.values()):<7} | {phase1_rep.total_tp:<6} | {phase1_rep.total_fp:<5} | {phase1_rep.singleton_f0_5:<10.4f}", flush=True)
    print(f"{'3. Phase 2 (R1: Residual Secondary Pool)':<38} | {r1_rep.macro_f0_5:<11.6f} | {r1_rep.macro_precision:<10.6f} | {r1_rep.macro_recall:<10.6f} | {sum(len(p) for p in r1_preds.values()):<7} | {r1_rep.total_tp:<6} | {r1_rep.total_fp:<5} | {r1_rep.singleton_f0_5:<10.4f}", flush=True)
    print(f"{'4. Phase 2 (R2: Strict Residual Guard)':<38} | {r2_rep.macro_f0_5:<11.6f} | {r2_rep.macro_precision:<10.6f} | {r2_rep.macro_recall:<10.6f} | {sum(len(p) for p in r2_preds.values()):<7} | {r2_rep.total_tp:<6} | {r2_rep.total_fp:<5} | {r2_rep.singleton_f0_5:<10.4f}", flush=True)
    print("=" * 125, flush=True)

    # Identify best
    options = [
        ("Phase1_Consensus_First", phase1_rep, phase1_preds, phase1_cand_count),
        ("Phase2_R1_Secondary_Pool", r1_rep, r1_preds, r1_cand_count),
        ("Phase2_R2_Strict_Residual", r2_rep, r2_preds, r1_cand_count),
    ]
    best_opt = max(options, key=lambda x: x[1].macro_f0_5)
    best_name, best_rep, best_preds, best_cand_cnt = best_opt

    delta_f05_v3 = best_rep.macro_f0_5 - 0.831214
    delta_prec_v3 = best_rep.macro_precision - 0.895836
    delta_rec_v3 = best_rep.macro_recall - 0.720218

    print(f"\nOVERALL BEST PIPELINE: {best_name}")
    print(f"  Macro F0.5 Delta vs V3 Baseline : {delta_f05_v3:+.6f} (0.831214 -> {best_rep.macro_f0_5:.6f})")
    print(f"  Precision Delta                 : {delta_prec_v3:+.6f} (0.895836 -> {best_rep.macro_precision:.6f})")
    print(f"  Recall Delta                    : {delta_rec_v3:+.6f} (0.720218 -> {best_rep.macro_recall:.6f})")
    print(f"  True Positive Delta             : {best_rep.total_tp - 4956:+d} (4956 -> {best_rep.total_tp})")
    print(f"  False Positive Delta            : {best_rep.total_fp - 263:+d} (263 -> {best_rep.total_fp})")

    decision = "KEEP" if delta_f05_v3 > 0.0 else "REJECT"
    print(f"  Final Decision                  : {decision}", flush=True)

    # 10. Save JSON Report
    exp_record = {
        "experiment_id": "EXP-005B",
        "description": "Selective Residual Recovery for Unresolved Entities",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_entities": len(val_s1_list),
        "total_runtime_seconds": time.time() - overall_start,
        "metrics": {
            "v3_baseline": {
                "macro_f0_5": 0.831214,
                "precision": 0.895836,
                "recall": 0.720218,
                "tp": 4956,
                "fp": 263,
            },
            "phase1_consensus_first": {
                "macro_f0_5": phase1_rep.macro_f0_5,
                "precision": phase1_rep.macro_precision,
                "recall": phase1_rep.macro_recall,
                "tp": phase1_rep.total_tp,
                "fp": phase1_rep.total_fp,
            },
            "phase2_r1_secondary_pool": {
                "macro_f0_5": r1_rep.macro_f0_5,
                "precision": r1_rep.macro_precision,
                "recall": r1_rep.macro_recall,
                "tp": r1_rep.total_tp,
                "fp": r1_rep.total_fp,
            },
            "phase2_r2_strict_residual": {
                "macro_f0_5": r2_rep.macro_f0_5,
                "precision": r2_rep.macro_precision,
                "recall": r2_rep.macro_recall,
                "tp": r2_rep.total_tp,
                "fp": r2_rep.total_fp,
            },
        },
        "best_configuration": best_name,
        "deltas_vs_v3": {
            "macro_f0_5": delta_f05_v3,
            "precision": delta_prec_v3,
            "recall": delta_rec_v3,
            "tp_delta": best_rep.total_tp - 4956,
            "fp_delta": best_rep.total_fp - 263,
        },
        "verdict": decision,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(exp_record, f, indent=2)
    print(f"Saved experiment report to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
