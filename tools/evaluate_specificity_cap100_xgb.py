#!/usr/bin/env python3
"""End-to-End Validation Evaluation of Specificity-Aware FINAL-CAP-100 with EXP-002B XGBoost Matcher.

Feeds the 190,331 candidates produced by the specificity-aware candidate selection (cap=100)
into the trained EXP-002B XGBoost pairwise matcher at optimal threshold theta=0.60.

Evaluates on the exact 2,001-entity validation split against the 10,320,219 search pool.
"""

import argparse
import collections
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
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

from src.blocking.evaluator import evaluate_blocking
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
from src.evaluation.metrics import evaluate_predictions
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
    "country_eq", "source_type", "s1_name_empty", "target_name_empty",
    "s1_addr_empty", "target_addr_empty", "name_exact_eq", "name_fuzz_ratio",
    "name_fuzz_wratio", "name_token_set_ratio", "name_token_sort_ratio",
    "name_prefix_similarity", "name_len_diff", "name_len_ratio",
    "addr_fuzz_ratio", "addr_token_similarity", "numeric_token_overlap",
    "house_number_agreement", "postal_pin_overlap", "addr_len_diff",
    "cross_field_name_loc", "cheap_candidate_score",
]


def get_process_peak_ram_mb() -> float:
    """Return peak working set memory in MB via Windows API with fallback."""
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


def worker_stream_chunk_specificity(task_args):
    """Worker streaming function with 6-tier specificity priority tagging."""
    (
        worker_id,
        file_path,
        start_line,
        end_line,
        query_idx_country,
        s1_preprocessed,
        val_true_target_ids,
        generic_keys_set,
        max_generic_per_s1,
    ) = task_args

    scored_candidates = collections.defaultdict(list)
    target_records = {}

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
            c_pos = line.rfind("\t")
            if c_pos == -1:
                continue
            tcountry = line[c_pos + 1 :].rstrip("\r\n")
            c_idx = query_idx_country.get(tcountry)
            if c_idx is None:
                continue

            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            tid, tname, taddr = parts[0], parts[1], parts[2]

            matched_s1_best_tier = {}

            # Tier 1: Exact normalized name
            ten = normalize_name_conservative(tname)
            if ten:
                m = c_idx.get(f"en_{ten}")
                if m:
                    for sid in m:
                        matched_s1_best_tier[sid] = 1

            # Tier 2: PIN keys
            if taddr and taddr.strip():
                for ak in extract_address_keys(taddr):
                    if "pin_" in ak:
                        m = c_idx.get(f"ak_{ak}")
                        if m:
                            for sid in m:
                                if sid not in matched_s1_best_tier or matched_s1_best_tier[sid] > 2:
                                    matched_s1_best_tier[sid] = 2
                for ak in extract_h1_relaxed_address_keys(taddr):
                    if ak.startswith("pin_"):
                        m = c_idx.get(ak)
                        if m:
                            for sid in m:
                                if sid not in matched_s1_best_tier or matched_s1_best_tier[sid] > 2:
                                    matched_s1_best_tier[sid] = 2

            # Tier 3: Rare address-derived keys (ak_, nword_)
            if taddr and taddr.strip():
                for ak in extract_address_keys(taddr):
                    if "pin_" not in ak:
                        m = c_idx.get(f"ak_{ak}")
                        if m:
                            for sid in m:
                                if sid not in matched_s1_best_tier or matched_s1_best_tier[sid] > 3:
                                    matched_s1_best_tier[sid] = 3
                for ak in extract_h1_relaxed_address_keys(taddr):
                    if ak.startswith("nword_"):
                        m = c_idx.get(ak)
                        if m:
                            for sid in m:
                                if sid not in matched_s1_best_tier or matched_s1_best_tier[sid] > 3:
                                    matched_s1_best_tier[sid] = 3

            # Tier 4: Rare locality/name keys (loc_, nl_)
            if taddr and taddr.strip():
                for ak in extract_h1_relaxed_address_keys(taddr):
                    if ak.startswith("loc_"):
                        m = c_idx.get(ak)
                        if m:
                            for sid in m:
                                if sid not in matched_s1_best_tier or matched_s1_best_tier[sid] > 4:
                                    matched_s1_best_tier[sid] = 4
                for ck in extract_h3_name_locality_keys(tname, taddr):
                    m = c_idx.get(ck)
                    if m:
                        for sid in m:
                            if sid not in matched_s1_best_tier or matched_s1_best_tier[sid] > 4:
                                matched_s1_best_tier[sid] = 4

            # Tier 5 & Tier 6: Token keys (Tier 5 = rare token, Tier 6 = generic token)
            gen_keys = generic_keys_set.get(tcountry, set())
            for tk in extract_token_blocking_keys(tname):
                k = f"tk_{tk}"
                m = c_idx.get(k)
                if m:
                    tier = 6 if k in gen_keys else 5
                    for sid in m:
                        if sid not in matched_s1_best_tier or matched_s1_best_tier[sid] > tier:
                            matched_s1_best_tier[sid] = tier

            if not matched_s1_best_tier:
                continue

            if tid in val_true_target_ids:
                target_records[tid] = (tname, taddr, tcountry)

            tgt_p = preprocess_record(tid, tname, taddr, tcountry)

            for sid, tier in matched_s1_best_tier.items():
                score = fast_score_candidate(s1_preprocessed[sid], tgt_p)
                if score >= 0.15:
                    scored_candidates[sid].append((tier, score, tid))
                pairs_scored += 1

    # Worker pruning:
    pruned_scored = {}
    for sid, pairs in scored_candidates.items():
        if len(pairs) <= 100:
            pruned_scored[sid] = pairs
        else:
            specific = [p for p in pairs if p[0] <= 5]
            generic = [p for p in pairs if p[0] == 6]

            specific.sort(key=lambda x: (x[0], -x[1]))
            generic.sort(key=lambda x: -x[1])

            n_gen = min(len(generic), max_generic_per_s1)
            n_spec = min(len(specific), 100 - n_gen)
            n_gen = min(len(generic), 100 - n_spec)

            selected = specific[:n_spec] + generic[:n_gen]
            selected.sort(key=lambda x: (x[0], -x[1]))
            pruned_scored[sid] = selected

    return lines_read, pairs_scored, target_records, pruned_scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--final-cap", type=int, default=100, help="Candidate cap per S1")
    parser.add_argument("--threshold", type=float, default=0.60, help="XGBoost decision threshold (default: 0.60)")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/xgb_matcher_exp002b.json"),
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("experiments/splits/validation_split_seed42_benchmark_10000.json"),
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("dataset/train/train_ground_truth.tsv"),
    )
    parser.add_argument(
        "--source1",
        type=Path,
        default=Path("dataset/train/train_source1.tsv"),
    )
    parser.add_argument(
        "--source2",
        type=Path,
        default=Path("dataset/train/train_source2.tsv"),
    )
    parser.add_argument(
        "--source3",
        type=Path,
        default=Path("dataset/train/train_source3.tsv"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("experiments/EXP-002B-specificity-cap100-xgb-evaluation.txt"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experiments/EXP-002B-specificity-cap100-xgb-evaluation.json"),
    )
    args = parser.parse_args()

    overall_start = time.time()

    print("=" * 95, flush=True)
    print("END-TO-END VALIDATION EVALUATION: SPECIFICITY-AWARE CAP=100 + EXP-002B XGBOOST", flush=True)
    print(f"Model: {args.model_path} | Threshold: {args.threshold:.2f} | Final Cap: {args.final_cap}", flush=True)
    print("=" * 95, flush=True)

    # 1. Load Validation Split
    with open(args.split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    val_s1_list: List[str] = split_data["val_s1_ids"]
    val_s1_set = set(val_s1_list)

    # 2. Parse Ground Truth
    val_ground_truth = parse_ground_truth(args.ground_truth, allowed_s1_ids=val_s1_set)
    total_true_matches = sum(len(v) for v in val_ground_truth.values())
    val_true_target_ids = {tid for tids in val_ground_truth.values() for tid in tids}

    # 3. Load & Preprocess Validation S1 Records
    val_s1_records: Dict[str, Tuple[str, str, str, str]] = {}
    val_s1_preprocessed: Dict[str, PreprocessedRecord] = {}
    query_idx_country: Dict[str, Dict[str, List[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))

    with open(args.source1, "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            tab_pos = line.find("\t")
            if tab_pos == -1:
                continue
            sid = line[:tab_pos]
            if sid not in val_s1_set:
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                name, addr, country = parts[1], parts[2], parts[3]
                val_s1_records[sid] = (sid, name, addr, country)
                prep = preprocess_record(sid, name, addr, country)
                val_s1_preprocessed[sid] = prep

                c_idx = query_idx_country[country]
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

                if len(val_s1_records) == len(val_s1_set):
                    break

    # 4. Frequency Filter & Generic Keys
    filter_cfg = BlockingKeyFilterConfig(max_key_fanout=15, preserve_exact_name=True, preserve_pin_keys=True)
    query_idx_clean: Dict[str, Dict[str, List[str]]] = {}
    generic_keys_set: Dict[str, Set[str]] = collections.defaultdict(set)

    for country, raw_idx in query_idx_country.items():
        c_count = sum(1 for sid, _, _, c in val_s1_records.values() if c == country)
        pruned_idx, _ = prune_high_frequency_keys(raw_idx, total_s1=c_count, config=filter_cfg)
        query_idx_clean[country] = pruned_idx
        for k, postings in pruned_idx.items():
            if k.startswith("tk_") and len(postings) >= 5:
                generic_keys_set[country].add(k)

    # 5. Parallel Candidate Generation Streaming
    s2_file = args.source2
    s3_file = args.source3

    def count_lines(fp):
        with open(fp, "r", encoding="utf-8-sig") as f:
            return sum(1 for _ in f) - 1

    s2_lines = count_lines(s2_file)
    s3_lines = count_lines(s3_file)

    tasks = []
    w_idx = 0
    for src_fp, n_lines in [(s2_file, s2_lines), (s3_file, s3_lines)]:
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
                    dict(query_idx_clean),
                    val_s1_preprocessed,
                    val_true_target_ids,
                    dict(generic_keys_set),
                    30,
                ))
                w_idx += 1

    with mp.Pool(processes=args.num_workers) as pool:
        results = pool.map(worker_stream_chunk_specificity, tasks)

    combined_scored_candidates = collections.defaultdict(list)
    for _, _, _, pruned_scored in results:
        for sid, pairs in pruned_scored.items():
            combined_scored_candidates[sid].extend(pairs)

    del results
    gc.collect()

    # 6. Final Specificity-Aware Cap of 100
    final_capped_cands: Dict[str, List[str]] = {}
    all_needed_target_ids: Set[str] = set()

    for sid in val_s1_list:
        raw_pairs = combined_scored_candidates.get(sid, [])
        if not raw_pairs:
            final_capped_cands[sid] = []
            continue

        best_by_tid: Dict[str, Tuple[int, float]] = {}
        for tier, score, tid in raw_pairs:
            if tid not in best_by_tid:
                best_by_tid[tid] = (tier, score)
            else:
                prev_tier, prev_score = best_by_tid[tid]
                best_by_tid[tid] = (min(prev_tier, tier), max(prev_score, score))

        deduped = [(tier, score, tid) for tid, (tier, score) in best_by_tid.items()]

        if len(deduped) <= args.final_cap:
            selected_tids = [tid for _, _, tid in deduped]
        else:
            specific = [p for p in deduped if p[0] <= 5]
            generic = [p for p in deduped if p[0] == 6]

            specific.sort(key=lambda x: (x[0], -x[1]))
            generic.sort(key=lambda x: -x[1])

            n_specific = min(len(specific), args.final_cap)
            n_generic = min(len(generic), args.final_cap - n_specific)

            selected = specific[:n_specific] + generic[:n_generic]
            selected_tids = [tid for _, _, tid in selected]

        final_capped_cands[sid] = selected_tids
        all_needed_target_ids.update(selected_tids)

    total_cands = sum(len(v) for v in final_capped_cands.values())
    avg_cands_per_s1 = total_cands / len(val_s1_list)

    # Candidate pair recall:
    retrieved_true = 0
    for sid, tids in final_capped_cands.items():
        true_set = val_ground_truth.get(sid, set())
        retrieved_true += len(set(tids) & true_set)
    candidate_pair_recall = retrieved_true / total_true_matches

    print(f"Generated {total_cands:,} candidate pairs ({avg_cands_per_s1:.2f} avg/S1). Candidate Recall: {candidate_pair_recall * 100.0:.2f}%", flush=True)

    # 7. Retrieve Target Records for Needed Targets
    print(f"Retrieving and preprocessing {len(all_needed_target_ids):,} target records...", flush=True)
    target_prep_map: Dict[str, PreprocessedRecord] = {}

    for src_fp in [s2_file, s3_file]:
        with open(src_fp, "r", encoding="utf-8-sig") as f:
            next(f)
            for line in f:
                pos = line.find("\t")
                if pos == -1:
                    continue
                tid = line[:pos]
                if tid in all_needed_target_ids:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 4:
                        target_prep_map[tid] = preprocess_record(parts[0], parts[1], parts[2], parts[3])
                        if len(target_prep_map) == len(all_needed_target_ids):
                            break
        if len(target_prep_map) == len(all_needed_target_ids):
            break

    print(f"Retrieved {len(target_prep_map):,} target records.", flush=True)

    # 8. Feature Extraction and XGBoost Scoring
    print("Scoring candidates with EXP-002B XGBoost matcher...", flush=True)
    t0_xgb_scoring = time.time()

    bst = xgb.Booster()
    bst.load_model(str(args.model_path))
    bst.set_param({"nthread": args.num_workers})

    predictions: Dict[str, Set[str]] = {sid: set() for sid in val_s1_list}
    total_pairs_scored = 0
    xgb_pure_predict_runtime = 0.0

    batch_X = []
    batch_pairs = []

    for sid in val_s1_list:
        tids = final_capped_cands.get(sid, [])
        s1_prep = val_s1_preprocessed[sid]
        for tid in tids:
            tgt_prep = target_prep_map.get(tid)
            if tgt_prep:
                feats = extract_features_from_preprocessed(s1_prep, tgt_prep)
                feats["cheap_candidate_score"] = compute_cheap_candidate_score(feats)
                batch_X.append([feats[col] for col in FEATURE_NAMES])
                batch_pairs.append((sid, tid))
                total_pairs_scored += 1

                if len(batch_X) >= 50000:
                    dmat = xgb.DMatrix(np.array(batch_X, dtype=np.float32), feature_names=FEATURE_NAMES)
                    t0_pred = time.time()
                    probs = bst.predict(dmat)
                    xgb_pure_predict_runtime += time.time() - t0_pred
                    for (s_id, t_id), prob in zip(batch_pairs, probs):
                        if prob >= args.threshold:
                            predictions[s_id].add(t_id)
                    batch_X = []
                    batch_pairs = []

    if batch_X:
        dmat = xgb.DMatrix(np.array(batch_X, dtype=np.float32), feature_names=FEATURE_NAMES)
        t0_pred = time.time()
        probs = bst.predict(dmat)
        xgb_pure_predict_runtime += time.time() - t0_pred
        for (s_id, t_id), prob in zip(batch_pairs, probs):
            if prob >= args.threshold:
                predictions[s_id].add(t_id)

    xgb_scoring_runtime = time.time() - t0_xgb_scoring
    print(f"XGBoost total scoring (feat extract + predict) completed in {xgb_scoring_runtime:.2f}s (pure predict: {xgb_pure_predict_runtime:.2f}s) for {total_pairs_scored:,} pairs.", flush=True)

    peak_ram_mb = get_process_peak_ram_mb()

    # 9. Evaluate Official Metrics
    report = evaluate_predictions(predictions, val_ground_truth)
    macro_f0_5 = report.macro_f0_5
    precision = report.macro_precision
    recall = report.macro_recall
    total_tp = report.total_tp
    total_fp = report.total_fp
    total_fn = report.total_fn
    total_predicted_links = sum(len(v) for v in predictions.values())

    # 10. Format Report
    report_lines = [
        "=" * 100,
        "END-TO-END VALIDATION EVALUATION: SPECIFICITY-AWARE CAP=100 + EXP-002B XGBOOST",
        "=" * 100,
        f"Validation Entities         : {len(val_s1_list):,} Source 1 entities",
        f"Model                       : {args.model_path.resolve()}",
        f"Decision Threshold (theta)  : {args.threshold:.2f}",
        f"Candidate Cap Policy        : Specificity-Aware Max {args.final_cap} per S1",
        "",
        "REQUIRED VALIDATION METRICS:",
        f"  1.  Entity-Level Macro F0.5 : {macro_f0_5:.4f}",
        f"  2.  Precision               : {precision:.4f}",
        f"  3.  Recall                  : {recall:.4f}",
        f"  4.  Predicted Links         : {total_predicted_links:,}",
        f"  5.  True Positives (TP)     : {total_tp:,}",
        f"  6.  False Positives (FP)    : {total_fp:,}",
        f"  7.  False Negatives (FN)    : {total_fn:,}",
        f"  8.  Candidate Pair Recall   : {candidate_pair_recall * 100.0:.2f}% ({retrieved_true:,} / {total_true_matches:,})",
        f"  9.  Total Candidate Pairs   : {total_cands:,}",
        f"  10. Average Candidates / S1 : {avg_cands_per_s1:.2f}",
        f"  11. XGBoost Scoring Runtime : {xgb_scoring_runtime:.2f}s (inference: {xgb_pure_predict_runtime:.2f}s)",
        f"  12. Peak RAM                : {peak_ram_mb:.2f} MB",
        "",
        "-" * 100,
        "COMPARISON AGAINST EXP-002B VALIDATION BASELINE:",
        "-" * 100,
        f"{'Metric':<32} | {'EXP-002B Baseline':<20} | {'Specificity Cap=100':<20} | {'Delta':<15}",
        "-" * 100,
        f"{'Macro F0.5':<32} | {'0.8364':<20} | {macro_f0_5:<20.4f} | {macro_f0_5 - 0.8364:+.4f}",
        f"{'Precision':<32} | {'0.8943':<20} | {precision:<20.4f} | {precision - 0.8943:+.4f}",
        f"{'Recall':<32} | {'0.7361':<20} | {recall:<20.4f} | {recall - 0.7361:+.4f}",
        f"{'Decision Threshold':<32} | {'0.60':<20} | {args.threshold:<20.2f} | {'0.00':<15}",
        f"{'Total Candidate Pairs':<32} | {'3,345,467':<20} | {f'{total_cands:,}':<20} | {f'{total_cands - 3345467:,} (-94.3%)':<15}",
        f"{'Average Candidates / S1':<32} | {'1,671.90':<20} | {f'{avg_cands_per_s1:.2f}':<20} | {f'{avg_cands_per_s1 - 1671.90:.2f}':<15}",
        f"{'Predicted Links':<32} | {'5,393':<20} | {f'{total_predicted_links:,}':<20} | {f'{total_predicted_links - 5393:+,}':<15}",
        f"{'True Positives (TP)':<32} | {'5,103':<20} | {f'{total_tp:,}':<20} | {f'{total_tp - 5103:+,}':<15}",
        f"{'False Positives (FP)':<32} | {'290':<20} | {f'{total_fp:,}':<20} | {f'{total_fp - 290:+,}':<15}",
        f"{'False Negatives (FN)':<32} | {'1,842':<20} | {f'{total_fn:,}':<20} | {f'{total_fn - 1842:+,}':<15}",
        "=" * 100,
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text, flush=True)

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    json_data = {
        "macro_f0_5": macro_f0_5,
        "precision": precision,
        "recall": recall,
        "predicted_links": total_predicted_links,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "candidate_pair_recall": candidate_pair_recall,
        "total_candidate_pairs": total_cands,
        "avg_candidates_per_s1": avg_cands_per_s1,
        "xgboost_scoring_runtime_seconds": xgb_scoring_runtime,
        "xgboost_inference_runtime_seconds": xgb_pure_predict_runtime,
        "peak_ram_mb": peak_ram_mb,
        "threshold": args.threshold,
        "comparison_baseline": {
            "macro_f0_5": 0.8364,
            "precision": 0.8943,
            "recall": 0.7361,
            "threshold": 0.60,
        },
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"Results saved to {args.output_report} and {args.output_json}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
