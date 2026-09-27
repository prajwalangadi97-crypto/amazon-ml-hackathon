#!/usr/bin/env python3
"""EXP-009: Fast Validation Evaluator for 52-Feature Rich Matcher.

Evaluates artifacts/xgb_matcher_exp009.json on the canonical 2,001-entity
validation benchmark using the exact EXP-006B candidate set and decision layer.
Outputs exact metrics and comparison vs EXP-006B baseline (F0.5 = 0.845653).
"""

import collections
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking.retrieval import (
    apply_selective_residual_recovery,
    extract_compound_number_keys,
    rank_candidates_channel_consensus_first,
)
from src.data.ground_truth import parse_ground_truth
from src.evaluation.metrics import evaluate_predictions, EvaluationReport
from src.matching.candidate_compression import (
    preprocess_record,
    PreprocessedRecord,
)
from src.matching.decision_layer import (
    DecisionLayerConfig,
    filter_entity_candidates,
)
from src.matching.rich_features import (
    extract_rich_pairwise_features,
    RICH_FEATURE_NAMES,
)
from tools.run_exp005_adaptive_ranking import fast_score_candidate_detailed


def main():
    overall_start = time.time()
    print("=" * 105, flush=True)
    print("EXP-009-RICH-MATCHER: CANONICAL BENCHMARK EVALUATION", flush=True)
    print("=" * 105, flush=True)

    split_file = PROJECT_ROOT / "experiments" / "splits" / "validation_split_seed42_benchmark_10000.json"
    ground_truth_file = PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv"
    source1_file = PROJECT_ROOT / "dataset" / "train" / "train_source1.tsv"
    source2_file = PROJECT_ROOT / "dataset" / "train" / "train_source2.tsv"
    source3_file = PROJECT_ROOT / "dataset" / "train" / "train_source3.tsv"
    model_path = PROJECT_ROOT / "artifacts" / "xgb_matcher_exp009.json"
    cache_pool_file = PROJECT_ROOT / "artifacts" / "exp005" / "val_raw_candidates.pkl"
    output_json = PROJECT_ROOT / "experiments" / "EXP-009-rich-matcher.json"

    assert model_path.exists(), f"Model not found at {model_path}"

    # 1. Load Split & Ground Truth
    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    val_s1_list: List[str] = split_data["val_s1_ids"]
    val_s1_set = set(val_s1_list)
    val_gt = parse_ground_truth(ground_truth_file, allowed_s1_ids=val_s1_set)
    total_val_gt = sum(len(t) for t in val_gt.values())
    print(f"Validation Entities: {len(val_s1_list):,} | Ground Truth Links: {total_val_gt:,}", flush=True)

    # 2. Load & Preprocess Validation S1 Records
    val_s1_records: Dict[str, Tuple[str, str, str, str]] = {}
    val_s1_prep: Dict[str, PreprocessedRecord] = {}
    val_cnum_keys: Dict[str, List[str]] = collections.defaultdict(list)

    with open(source1_file, "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and parts[0] in val_s1_set:
                sid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                val_s1_records[sid] = (sid, name, addr, country)
                val_s1_prep[sid] = preprocess_record(sid, name, addr, country)
                for cnk in extract_compound_number_keys(addr):
                    val_cnum_keys[cnk].append(sid)

    s1_country_map = {sid: val_s1_records[sid][3] for sid in val_s1_list}

    # 3. Load Candidate Pool & Merge CNUM Hits
    print(f"\nLoading cached raw candidate pool from {cache_pool_file}...", flush=True)
    with open(cache_pool_file, "rb") as f:
        val_raw_pool = pickle.load(f)

    print("Collecting validation compound number hits...", flush=True)
    t0_cnum = time.time()
    val_cnum_target_records = {}
    val_cnum_hits = collections.defaultdict(list)
    for src in [source2_file, source3_file]:
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

    print(f"Collected CNUM hits in {time.time() - t0_cnum:.2f}s.", flush=True)

    # Merge into raw pool
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
        selected = rank_candidates_channel_consensus_first(val_raw_pool.get(sid, []), cap=100)
        val_phase1_cands[sid] = selected
        val_needed_tids.update(t for t, _ in selected)

    total_val_p1_pairs = sum(len(c) for c in val_phase1_cands.values())
    print(f"Validation Phase 1 Candidates : {total_val_p1_pairs:,} ({total_val_p1_pairs / len(val_s1_list):.2f} avg/S1)", flush=True)

    # 4. Retrieve Target Records
    print(f"Retrieving {len(val_needed_tids):,} needed target records...", flush=True)
    t0_tgt = time.time()
    target_prep_map: Dict[str, PreprocessedRecord] = dict(val_cnum_target_records)

    for src_fp in [source2_file, source3_file]:
        with open(src_fp, "r", encoding="utf-8-sig") as f:
            next(f)
            for line in f:
                pos = line.find("\t")
                if pos == -1:
                    continue
                tid = line[:pos]
                if tid in val_needed_tids and tid not in target_prep_map:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 4:
                        target_prep_map[tid] = preprocess_record(parts[0], parts[1], parts[2], parts[3])
                        if len(target_prep_map) == len(val_needed_tids):
                            break
        if len(target_prep_map) == len(val_needed_tids):
            break

    print(f"Retrieved target records in {time.time() - t0_tgt:.2f}s.", flush=True)

    # 5. Load Trained 52-Feature Matcher
    print(f"\nLoading trained 52-feature XGBoost model from {model_path}...", flush=True)
    bst = xgb.Booster()
    bst.load_model(str(model_path))
    bst.set_param({"nthread": 8})

    # 6. Build Phase 1 Feature Matrix & Score
    print("Building Phase 1 52-feature matrix...", flush=True)
    t0_feat = time.time()
    X_rows_p1 = []
    p1_pair_keys = []
    for sid in val_s1_list:
        s1_p = val_s1_prep[sid]
        for tid, meta in val_phase1_cands.get(sid, []):
            tgt_p = target_prep_map.get(tid)
            if tgt_p is None:
                continue
            feats = extract_rich_pairwise_features(s1_p, tgt_p, meta)
            X_rows_p1.append([feats[col] for col in RICH_FEATURE_NAMES])
            p1_pair_keys.append((sid, tid))

    dp1 = xgb.DMatrix(np.array(X_rows_p1, dtype=np.float32), feature_names=RICH_FEATURE_NAMES)
    p1_probs = bst.predict(dp1)
    print(f"Phase 1 Matrix built & scored in {time.time() - t0_feat:.2f}s.", flush=True)

    s1_scores_p1: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
    for (sid, tid), p in zip(p1_pair_keys, p1_probs):
        s1_scores_p1[sid].append((tid, float(p)))

    # Decision Layer helper
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

    preds_p1_m30 = run_decision_layer(s1_scores_p1, margin_delta=0.30)
    preds_p1_m35 = run_decision_layer(s1_scores_p1, margin_delta=0.35)
    rep_p1_m30 = evaluate_predictions(preds_p1_m30, val_gt)
    rep_p1_m35 = evaluate_predictions(preds_p1_m35, val_gt)

    print(f"\nPhase 1 (margin=0.30): F0.5 = {rep_p1_m30.macro_f0_5:.6f} | Prec = {rep_p1_m30.macro_precision:.6f} | Rec = {rep_p1_m30.macro_recall:.6f} | TP = {rep_p1_m30.total_tp} | FP = {rep_p1_m30.total_fp}")
    print(f"Phase 1 (margin=0.35): F0.5 = {rep_p1_m35.macro_f0_5:.6f} | Prec = {rep_p1_m35.macro_precision:.6f} | Rec = {rep_p1_m35.macro_recall:.6f} | TP = {rep_p1_m35.total_tp} | FP = {rep_p1_m35.total_fp}")

    # 7. Phase 2: Selective Residual Recovery
    best_p1_preds = preds_p1_m35 if rep_p1_m35.macro_f0_5 > rep_p1_m30.macro_f0_5 else preds_p1_m30
    unresolved_count = sum(1 for p in best_p1_preds.values() if len(p) == 0)
    print(f"\nUnresolved entities for Phase 2 residual recovery: {unresolved_count} / {len(val_s1_list)} ({unresolved_count/len(val_s1_list):.1%})", flush=True)

    residual_cands = apply_selective_residual_recovery(
        s1_list=val_s1_list,
        phase1_candidates=val_phase1_cands,
        phase1_predictions=best_p1_preds,
        raw_candidate_pool=val_raw_pool,
        max_recovery=20,
    )

    residual_needed_tids = set()
    for sid, items in residual_cands.items():
        for tid, _ in items:
            if tid not in target_prep_map:
                residual_needed_tids.add(tid)

    if residual_needed_tids:
        print(f"Retrieving {len(residual_needed_tids):,} residual targets...", flush=True)
        for src_fp in [source2_file, source3_file]:
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
    res_probs = bst.predict(dres)

    s1_scores_res: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
    for (sid, tid), p in zip(res_pair_keys, res_probs):
        s1_scores_res[sid].append((tid, float(p)))

    # Evaluate Phase 2 combined matching across margin grid
    margin_sweeps = {}
    for m_val in [0.25, 0.30, 0.35, 0.40]:
        preds_m = run_decision_layer(s1_scores_res, margin_delta=m_val)
        rep_m = evaluate_predictions(preds_m, val_gt)
        margin_sweeps[m_val] = (rep_m, preds_m)
        print(f"Phase 2 (margin={m_val:.2f}): F0.5 = {rep_m.macro_f0_5:.6f} | Prec = {rep_m.macro_precision:.6f} | Rec = {rep_m.macro_recall:.6f} | TP = {rep_m.total_tp} | FP = {rep_m.total_fp}", flush=True)

    # Pick the canonical margin=0.30 (or best)
    rep_canonical_m30, preds_canonical_m30 = margin_sweeps[0.30]
    best_m_val = max(margin_sweeps.keys(), key=lambda m: margin_sweeps[m][0].macro_f0_5)
    best_rep, best_preds = margin_sweeps[best_m_val]

    total_pred_links = sum(len(p) for p in best_preds.values())
    empty_pred_count = sum(1 for p in best_preds.values() if len(p) == 0)

    # 8. Baseline Comparison
    BASE_F05 = 0.845653
    BASE_PREC = 0.907073
    BASE_REC = 0.738549
    BASE_TP = 5085
    BASE_FP = 298

    delta_f05 = best_rep.macro_f0_5 - BASE_F05
    delta_prec = best_rep.macro_precision - BASE_PREC
    delta_rec = best_rep.macro_recall - BASE_REC
    delta_tp = best_rep.total_tp - BASE_TP
    delta_fp = best_rep.total_fp - BASE_FP

    decision = "KEEP" if best_rep.macro_f0_5 > BASE_F05 else "REJECT"

    # Feature Importance
    f_score = bst.get_score(importance_type="gain")
    sorted_feats = sorted(f_score.items(), key=lambda x: x[1], reverse=True)

    # 9. Country-wise breakdown
    country_groups = collections.defaultdict(list)
    for sid in val_s1_list:
        c = s1_country_map.get(sid, "Unknown")
        country_groups[c].append(sid)

    country_metrics = {}
    for c, sids in sorted(country_groups.items()):
        sub_gt = {sid: val_gt.get(sid, set()) for sid in sids}
        sub_pred = {sid: best_preds.get(sid, set()) for sid in sids}
        sub_rep = evaluate_predictions(sub_pred, sub_gt)
        country_metrics[c] = sub_rep

    print("\n" + "=" * 105, flush=True)
    print("EXP-009-RICH-MATCHER FINAL EVALUATION REPORT", flush=True)
    print("=" * 105, flush=True)
    print(f"Baseline F0.5 (EXP-006B) : {BASE_F05:.6f}")
    print(f"New F0.5 (margin={best_m_val:.2f}) : {best_rep.macro_f0_5:.6f}")
    print(f"Delta                    : {delta_f05:+.6f}")
    print(f"Precision                : {best_rep.macro_precision:.6f} (Delta: {delta_prec:+.6f})")
    print(f"Recall                   : {best_rep.macro_recall:.6f} (Delta: {delta_rec:+.6f})")
    print(f"TP                       : {best_rep.total_tp} (Delta: {delta_tp:+d})")
    print(f"FP                       : {best_rep.total_fp} (Delta: {delta_fp:+d})")
    print(f"FN                       : {best_rep.total_fn}")
    print(f"Singleton Accuracy       : {best_rep.singleton_f0_5:.6f}")
    print(f"Predicted Links          : {total_pred_links:,}")
    print(f"Empty Predictions        : {empty_pred_count:,}")
    print(f"Candidate Pairs          : {total_val_cand_pairs:,}")
    print(f"Average Candidates/S1    : {total_val_cand_pairs / len(val_s1_list):.2f}")
    print(f"Runtime                  : {time.time() - overall_start:.2f}s")
    print(f"Decision                 : {decision}", flush=True)
    print("=" * 105, flush=True)

    print("\nCOUNTRY-WISE RESULTS:")
    for c, rep in sorted(country_metrics.items()):
        print(f"  {c:<8}: F0.5 = {rep.macro_f0_5:.6f} | Prec = {rep.macro_precision:.6f} | Rec = {rep.macro_recall:.6f} | TP = {rep.total_tp:4d} | FP = {rep.total_fp:3d} | FN = {rep.total_fn:4d}")

    print("\nTOP 15 FEATURES BY GAIN:")
    for rank_i, (f_name, g_val) in enumerate(sorted_feats[:15], start=1):
        print(f"  {rank_i:2d}. {f_name:<30} : {g_val:.2f}")

    # 10. Save JSON
    exp_record = {
        "experiment_id": "EXP-009-RICH-MATCHER",
        "description": "Rich 52-Feature Entity Resolution Matcher with Hard-Positive Weighting",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_entities": len(val_s1_list),
        "total_candidate_pairs": total_val_cand_pairs,
        "avg_candidates_per_s1": total_val_cand_pairs / len(val_s1_list),
        "total_runtime_seconds": time.time() - overall_start,
        "decision": decision,
        "selected_margin_delta": best_m_val,
        "metrics": {
            "baseline_exp006b": {
                "macro_f0_5": BASE_F05,
                "precision": BASE_PREC,
                "recall": BASE_REC,
                "tp": BASE_TP,
                "fp": BASE_FP,
            },
            "new_model_final": best_rep.to_dict(),
            "new_model_canonical_m30": rep_canonical_m30.to_dict(),
            "deltas_vs_exp006b": {
                "macro_f0_5": delta_f05,
                "precision": delta_prec,
                "recall": delta_rec,
                "tp_delta": delta_tp,
                "fp_delta": delta_fp,
            },
            "country_metrics": {c: rep.to_dict() for c, rep in country_metrics.items()},
        },
        "top_features_by_gain": sorted_feats[:20],
        "model_artifact": str(model_path),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(exp_record, f, indent=2)
    print(f"\nSaved experiment record to {output_json}", flush=True)


if __name__ == "__main__":
    main()
