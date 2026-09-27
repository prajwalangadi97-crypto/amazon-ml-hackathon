#!/usr/bin/env python3
"""EXP-003A: Candidate-Group-Aware Decision Layer Evaluation.

Evaluates decision-layer strategies on the existing validation candidate probabilities
from the EXP-002B XGBoost matcher (specificity-aware cap=100 candidate set):
  Strategy A: Current baseline (global threshold = 0.60)
  Strategy B: Global threshold + candidate margin
  Strategy C: Country-specific threshold
  Strategy D: Country-specific threshold + candidate margin

Strict constraints:
- Candidate competition operates strictly within each Source1 candidate group.
- Preserves multi-match semantics (does NOT enforce 1-to-1 matching).
- Uses exact same validation split (2,001 entities) and official evaluator.
- Caches scored candidate pairs to scratch/ for instant reproducible re-evaluation.
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

            # Tier 5 & Tier 6: Token keys
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
            pruned_scored[sid] = specific[:n_spec] + generic[:n_gen]

    return lines_read, pairs_scored, pruned_scored, target_records


def generate_and_cache_validation_scores(
    split_file: Path,
    ground_truth_file: Path,
    source1_file: Path,
    source2_file: Path,
    source3_file: Path,
    model_path: Path,
    cache_path: Path,
    num_workers: int = 8,
    final_cap: int = 100,
) -> Dict[str, Any]:
    """Generate candidate pairs, extract features, and score with XGBoost once."""
    print("=" * 90, flush=True)
    print("GENERATING & SCORING VALIDATION CANDIDATES FOR DECISION LAYER EVALUATION", flush=True)
    print("=" * 90, flush=True)

    # 1. Load Split
    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    val_s1_list: List[str] = split_data["val_s1_ids"]
    val_s1_set = set(val_s1_list)

    # 2. Ground Truth
    val_ground_truth = parse_ground_truth(ground_truth_file, allowed_s1_ids=val_s1_set)
    val_true_target_ids = {tid for tids in val_ground_truth.values() for tid in tids}

    # 3. Preprocess S1
    val_s1_records: Dict[str, Tuple[str, str, str, str]] = {}
    val_s1_preprocessed: Dict[str, PreprocessedRecord] = {}
    query_idx_country: Dict[str, Dict[str, List[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))

    with open(source1_file, "r", encoding="utf-8-sig") as f:
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

    # 4. Frequency Filter
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

    # 5. Parallel Stream Search Pool
    def count_lines(fp):
        with open(fp, "r", encoding="utf-8-sig") as f:
            return sum(1 for _ in f) - 1

    s2_lines = count_lines(source2_file)
    s3_lines = count_lines(source3_file)

    tasks = []
    w_idx = 0
    for src_fp, n_lines in [(source2_file, s2_lines), (source3_file, s3_lines)]:
        chunk_sz = (n_lines + num_workers - 1) // num_workers
        for i in range(num_workers):
            start = i * chunk_sz
            end = min((i + 1) * chunk_sz, n_lines)
            if start < n_lines:
                tasks.append((
                    w_idx,
                    str(src_fp),
                    start,
                    end,
                    query_idx_clean,
                    val_s1_preprocessed,
                    val_true_target_ids,
                    generic_keys_set,
                    10,
                ))
                w_idx += 1

    print(f"Streaming search pool with {num_workers} parallel workers...", flush=True)
    t0_stream = time.time()
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(worker_stream_chunk_specificity, tasks)

    combined_candidates = collections.defaultdict(list)
    for _, _, pruned_scored, _ in results:
        for sid, pairs in pruned_scored.items():
            combined_candidates[sid].extend(pairs)
    del results
    gc.collect()
    print(f"Streaming completed in {time.time() - t0_stream:.2f}s.", flush=True)

    # 6. Specificity Cap
    final_capped_cands: Dict[str, List[str]] = {}
    all_needed_target_ids: Set[str] = set()

    for sid in val_s1_list:
        raw_pairs = combined_candidates.get(sid, [])
        best_by_tid: Dict[str, Tuple[int, float]] = {}
        for tier, score, tid in raw_pairs:
            if tid not in best_by_tid:
                best_by_tid[tid] = (tier, score)
            else:
                prev_tier, prev_score = best_by_tid[tid]
                best_by_tid[tid] = (min(prev_tier, tier), max(prev_score, score))

        deduped = [(tier, score, tid) for tid, (tier, score) in best_by_tid.items()]
        if len(deduped) <= final_cap:
            selected_tids = [tid for _, _, tid in deduped]
        else:
            specific = [p for p in deduped if p[0] <= 5]
            generic = [p for p in deduped if p[0] == 6]
            specific.sort(key=lambda x: (x[0], -x[1]))
            generic.sort(key=lambda x: -x[1])
            n_specific = min(len(specific), final_cap)
            n_generic = min(len(generic), final_cap - n_specific)
            selected = specific[:n_specific] + generic[:n_generic]
            selected_tids = [tid for _, _, tid in selected]

        final_capped_cands[sid] = selected_tids
        all_needed_target_ids.update(selected_tids)

    del combined_candidates
    gc.collect()

    # 7. Retrieve Target Records
    print(f"Retrieving {len(all_needed_target_ids):,} target records...", flush=True)
    target_prep_map: Dict[str, PreprocessedRecord] = {}
    for src_fp in [source2_file, source3_file]:
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

    # 8. Score Pairs with XGBoost
    print(f"Scoring candidate pairs with XGBoost ({model_path})...", flush=True)
    bst = xgb.Booster()
    bst.load_model(str(model_path))
    bst.set_param({"nthread": num_workers})

    s1_candidate_scores: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
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

                if len(batch_X) >= 50000:
                    dmat = xgb.DMatrix(np.array(batch_X, dtype=np.float32), feature_names=FEATURE_NAMES)
                    probs = bst.predict(dmat)
                    for (s_id, t_id), prob in zip(batch_pairs, probs):
                        s1_candidate_scores[s_id].append((t_id, float(prob)))
                    batch_X = []
                    batch_pairs = []

    if batch_X:
        dmat = xgb.DMatrix(np.array(batch_X, dtype=np.float32), feature_names=FEATURE_NAMES)
        probs = bst.predict(dmat)
        for (s_id, t_id), prob in zip(batch_pairs, probs):
            s1_candidate_scores[s_id].append((t_id, float(prob)))

    del target_prep_map, val_s1_preprocessed, batch_X, batch_pairs
    gc.collect()

    cache_data = {
        "val_s1_records": val_s1_records,
        "val_ground_truth": val_ground_truth,
        "s1_candidate_scores": dict(s1_candidate_scores),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved validation scores cache to {cache_path} ({cache_path.stat().st_size / (1024*1024):.2f} MB).", flush=True)

    return cache_data


def compute_group_features(
    s1_candidate_scores: Dict[str, List[Tuple[str, float]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Compute candidate group features for each Source1 entity.

    Features per candidate:
    - candidate_id (tid)
    - prob: candidate probability
    - rank: 1-based rank (descending by prob)
    - recip_rank: 1.0 / rank
    - max_prob: highest probability for this S1
    - second_highest_prob: 2nd highest prob for this S1 (or 0.0)
    - margin_from_max: max_prob - prob
    - margin_lead: prob - second_highest_prob (for rank 1)
    - ratio_to_max: prob / max_prob
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for sid, pairs in s1_candidate_scores.items():
        if not pairs:
            grouped[sid] = []
            continue

        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        max_p = sorted_pairs[0][1]
        second_p = sorted_pairs[1][1] if len(sorted_pairs) > 1 else 0.0

        cands = []
        for rank_idx, (tid, p) in enumerate(sorted_pairs, start=1):
            margin_from_max = max_p - p
            margin_lead = p - second_p if rank_idx == 1 else (p - max_p)
            ratio_to_max = p / max_p if max_p > 0 else 0.0
            ratio_to_second = p / second_p if second_p > 0 else (1.0 if p == 0 else 999.0)

            cands.append({
                "tid": tid,
                "prob": p,
                "rank": rank_idx,
                "recip_rank": 1.0 / rank_idx,
                "max_prob": max_p,
                "second_prob": second_p,
                "margin_from_max": margin_from_max,
                "margin_lead": margin_lead,
                "ratio_to_max": ratio_to_max,
                "ratio_to_second": ratio_to_second,
            })
        grouped[sid] = cands

    return grouped


def evaluate_decision_rule(
    val_s1_list: List[str],
    s1_records: Dict[str, Tuple[str, str, str, str]],
    grouped_cands: Dict[str, List[Dict[str, Any]]],
    val_ground_truth: Dict[str, Set[str]],
    rule_fn,
) -> Tuple[EvaluationReport, int]:
    """Apply decision rule and evaluate using official metrics."""
    predictions: Dict[str, Set[str]] = {sid: set() for sid in val_s1_list}
    total_predicted = 0

    for sid in val_s1_list:
        country = s1_records[sid][3]
        cands = grouped_cands.get(sid, [])
        for cand in cands:
            if rule_fn(cand, country):
                predictions[sid].add(cand["tid"])
                total_predicted += 1

    report = evaluate_predictions(predictions, val_ground_truth)
    return report, total_predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, default=Path("scratch/val_candidate_scores_exp002b.pkl"))
    parser.add_argument("--split-file", type=Path, default=Path("experiments/splits/validation_split_seed42_benchmark_10000.json"))
    parser.add_argument("--ground-truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--source1", type=Path, default=Path("dataset/train/train_source1.tsv"))
    parser.add_argument("--source2", type=Path, default=Path("dataset/train/train_source2.tsv"))
    parser.add_argument("--source3", type=Path, default=Path("dataset/train/train_source3.tsv"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/xgb_matcher_exp002b.json"))
    parser.add_argument("--output-json", type=Path, default=Path("experiments/EXP-003A-decision-layer.json"))
    args = parser.parse_args()

    overall_start = time.time()

    # 1. Load or Generate Scored Validation Pairs
    if args.cache_path.is_file():
        print(f"Loading cached validation scores from {args.cache_path}...", flush=True)
        with open(args.cache_path, "rb") as f:
            cache_data = pickle.load(f)
    else:
        cache_data = generate_and_cache_validation_scores(
            split_file=args.split_file,
            ground_truth_file=args.ground_truth,
            source1_file=args.source1,
            source2_file=args.source2,
            source3_file=args.source3,
            model_path=args.model_path,
            cache_path=args.cache_path,
        )

    val_s1_records = cache_data["val_s1_records"]
    val_ground_truth = cache_data["val_ground_truth"]
    s1_candidate_scores = cache_data["s1_candidate_scores"]

    with open(args.split_file, "r", encoding="utf-8") as f:
        val_s1_list: List[str] = json.load(f)["val_s1_ids"]

    # 2. Compute Group-Level Features
    grouped_cands = compute_group_features(s1_candidate_scores)

    # 3. STRATEGY A: Baseline (Global Threshold = 0.60)
    print("\n" + "=" * 95, flush=True)
    print("STRATEGY A: BASELINE (Global Threshold = 0.60)", flush=True)
    print("=" * 95, flush=True)
    rep_a, pred_a = evaluate_decision_rule(
        val_s1_list, val_s1_records, grouped_cands, val_ground_truth,
        lambda c, country: c["prob"] >= 0.60
    )
    print(f"Strategy A Baseline: Macro F0.5={rep_a.macro_f0_5:.4f}, Precision={rep_a.macro_precision:.4f}, Recall={rep_a.macro_recall:.4f}, Pred={pred_a}, TP={rep_a.total_tp}, FP={rep_a.total_fp}, Singleton Acc={rep_a.singleton_f0_5:.4f}")

    # 4. STRATEGY B: Global Threshold + Candidate Margin
    print("\n" + "=" * 95, flush=True)
    print("STRATEGY B: Global Threshold + Candidate Margin (Within-Group Competition)", flush=True)
    print("=" * 95, flush=True)

    # Sweep thresholds and margin deltas
    best_b_cfg = None
    best_b_rep = None
    best_b_pred = 0
    all_b_results = []

    thresh_grid_b = [0.50, 0.55, 0.60, 0.65]
    margin_grid_b = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

    for th in thresh_grid_b:
        for delta in margin_grid_b:
            rep, pred = evaluate_decision_rule(
                val_s1_list, val_s1_records, grouped_cands, val_ground_truth,
                lambda c, country, th=th, delta=delta: (c["prob"] >= th) and (c["margin_from_max"] <= delta)
            )
            all_b_results.append({
                "threshold": th,
                "margin_delta": delta,
                "macro_f0_5": rep.macro_f0_5,
                "precision": rep.macro_precision,
                "recall": rep.macro_recall,
                "predicted_links": pred,
                "tp": rep.total_tp,
                "fp": rep.total_fp,
                "singleton_acc": rep.singleton_f0_5,
            })
            if best_b_rep is None or rep.macro_f0_5 > best_b_rep.macro_f0_5:
                best_b_rep = rep
                best_b_pred = pred
                best_b_cfg = {"threshold": th, "margin_delta": delta}

    print(f"Best Strategy B: theta={best_b_cfg['threshold']:.2f}, margin_delta={best_b_cfg['margin_delta']:.2f} -> Macro F0.5={best_b_rep.macro_f0_5:.4f} (Delta: {best_b_rep.macro_f0_5 - rep_a.macro_f0_5:+.4f})")

    # 5. STRATEGY C: Country-Specific Thresholds
    print("\n" + "=" * 95, flush=True)
    print("STRATEGY C: Country-Specific Thresholds (Without Margin)", flush=True)
    print("=" * 95, flush=True)

    # Countries: France, US, India
    country_threshold_candidates = {
        "France": [0.45, 0.50, 0.55, 0.60, 0.65],
        "US": [0.50, 0.55, 0.60, 0.65, 0.70],
        "India": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    }

    # First optimize single-country thresholds
    best_country_th = {}
    for country in ["France", "US", "India"]:
        c_s1 = [sid for sid in val_s1_list if val_s1_records[sid][3] == country]
        best_c_f05 = -1.0
        best_c_th = 0.60
        for th in country_threshold_candidates[country]:
            rep, _ = evaluate_decision_rule(
                c_s1, val_s1_records, grouped_cands, val_ground_truth,
                lambda c, cnt, th=th: c["prob"] >= th
            )
            if rep.macro_f0_5 > best_c_f05:
                best_c_f05 = rep.macro_f0_5
                best_c_th = th
        best_country_th[country] = best_c_th
        print(f"  Optimized {country:<7}: threshold={best_c_th:.2f} (Macro F0.5={best_c_f05:.4f})")

    rep_c, pred_c = evaluate_decision_rule(
        val_s1_list, val_s1_records, grouped_cands, val_ground_truth,
        lambda c, country: c["prob"] >= best_country_th.get(country, 0.60)
    )
    print(f"Strategy C Overall: Thresholds={best_country_th} -> Macro F0.5={rep_c.macro_f0_5:.4f} (Delta: {rep_c.macro_f0_5 - rep_a.macro_f0_5:+.4f}), Precision={rep_c.macro_precision:.4f}, Recall={rep_c.macro_recall:.4f}, Pred={pred_c}, TP={rep_c.total_tp}, FP={rep_c.total_fp}, Singleton Acc={rep_c.singleton_f0_5:.4f}")

    # 6. STRATEGY D: Country-Specific Threshold + Candidate Margin
    print("\n" + "=" * 95, flush=True)
    print("STRATEGY D: Country-Specific Threshold + Candidate Margin", flush=True)
    print("=" * 95, flush=True)

    best_d_cfg = None
    best_d_rep = None
    best_d_pred = 0
    all_d_results = []

    margin_grid_d = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

    for delta in margin_grid_d:
        rep, pred = evaluate_decision_rule(
            val_s1_list, val_s1_records, grouped_cands, val_ground_truth,
            lambda c, country, delta=delta: (c["prob"] >= best_country_th.get(country, 0.60)) and (c["margin_from_max"] <= delta)
        )
        all_d_results.append({
            "country_thresholds": dict(best_country_th),
            "margin_delta": delta,
            "macro_f0_5": rep.macro_f0_5,
            "precision": rep.macro_precision,
            "recall": rep.macro_recall,
            "predicted_links": pred,
            "tp": rep.total_tp,
            "fp": rep.total_fp,
            "singleton_acc": rep.singleton_f0_5,
        })
        if best_d_rep is None or rep.macro_f0_5 > best_d_rep.macro_f0_5:
            best_d_rep = rep
            best_d_pred = pred
            best_d_cfg = {"country_thresholds": dict(best_country_th), "margin_delta": delta}

    print(f"Best Strategy D: Country Thresh={best_d_cfg['country_thresholds']}, margin_delta={best_d_cfg['margin_delta']:.2f} -> Macro F0.5={best_d_rep.macro_f0_5:.4f} (Delta: {best_d_rep.macro_f0_5 - rep_a.macro_f0_5:+.4f})")

    # 7. Comparison Summary Table
    print("\n" + "=" * 115, flush=True)
    print(f"{'Strategy':<35} | {'Macro F0.5':<11} | {'Precision':<10} | {'Recall':<10} | {'Pred':<7} | {'TP':<6} | {'FP':<5} | {'Single Acc':<10} | {'Configuration'}", flush=True)
    print("-" * 115, flush=True)
    print(f"{'A) Baseline (theta=0.60)':<35} | {rep_a.macro_f0_5:<11.4f} | {rep_a.macro_precision:<10.4f} | {rep_a.macro_recall:<10.4f} | {pred_a:<7} | {rep_a.total_tp:<6} | {rep_a.total_fp:<5} | {rep_a.singleton_f0_5:<10.4f} | theta=0.60", flush=True)
    print(f"{'B) Global Thresh + Margin':<35} | {best_b_rep.macro_f0_5:<11.4f} | {best_b_rep.macro_precision:<10.4f} | {best_b_rep.macro_recall:<10.4f} | {best_b_pred:<7} | {best_b_rep.total_tp:<6} | {best_b_rep.total_fp:<5} | {best_b_rep.singleton_f0_5:<10.4f} | theta={best_b_cfg['threshold']:.2f}, delta={best_b_cfg['margin_delta']:.2f}", flush=True)
    print(f"{'C) Country-Specific Thresh':<35} | {rep_c.macro_f0_5:<11.4f} | {rep_c.macro_precision:<10.4f} | {rep_c.macro_recall:<10.4f} | {pred_c:<7} | {rep_c.total_tp:<6} | {rep_c.total_fp:<5} | {rep_c.singleton_f0_5:<10.4f} | FR={best_country_th['France']:.2f}, US={best_country_th['US']:.2f}, IN={best_country_th['India']:.2f}", flush=True)
    print(f"{'D) Country Thresh + Margin':<35} | {best_d_rep.macro_f0_5:<11.4f} | {best_d_rep.macro_precision:<10.4f} | {best_d_rep.macro_recall:<10.4f} | {best_d_pred:<7} | {best_d_rep.total_tp:<6} | {best_d_rep.total_fp:<5} | {best_d_rep.singleton_f0_5:<10.4f} | FR={best_d_cfg['country_thresholds']['France']:.2f}, US={best_d_cfg['country_thresholds']['US']:.2f}, IN={best_d_cfg['country_thresholds']['India']:.2f}, delta={best_d_cfg['margin_delta']:.2f}", flush=True)
    print("=" * 115, flush=True)

    # 8. Save Artifact
    experiment_record = {
        "experiment_id": "EXP-003A",
        "description": "Candidate-Group-Aware Decision Layer Optimization (Margin & Country Thresholds)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_entities": len(val_s1_list),
        "strategies": {
            "strategy_a_baseline": {
                "config": {"threshold": 0.60},
                "macro_f0_5": rep_a.macro_f0_5,
                "precision": rep_a.macro_precision,
                "recall": rep_a.macro_recall,
                "predicted_links": pred_a,
                "tp": rep_a.total_tp,
                "fp": rep_a.total_fp,
                "singleton_acc": rep_a.singleton_f0_5,
            },
            "strategy_b_global_margin": {
                "config": best_b_cfg,
                "macro_f0_5": best_b_rep.macro_f0_5,
                "precision": best_b_rep.macro_precision,
                "recall": best_b_rep.macro_recall,
                "predicted_links": best_b_pred,
                "tp": best_b_rep.total_tp,
                "fp": best_b_rep.total_fp,
                "singleton_acc": best_b_rep.singleton_f0_5,
                "delta_vs_baseline": best_b_rep.macro_f0_5 - rep_a.macro_f0_5,
            },
            "strategy_c_country_thresholds": {
                "config": best_country_th,
                "macro_f0_5": rep_c.macro_f0_5,
                "precision": rep_c.macro_precision,
                "recall": rep_c.macro_recall,
                "predicted_links": pred_c,
                "tp": rep_c.total_tp,
                "fp": rep_c.total_fp,
                "singleton_acc": rep_c.singleton_f0_5,
                "delta_vs_baseline": rep_c.macro_f0_5 - rep_a.macro_f0_5,
            },
            "strategy_d_country_margin": {
                "config": best_d_cfg,
                "macro_f0_5": best_d_rep.macro_f0_5,
                "precision": best_d_rep.macro_precision,
                "recall": best_d_rep.macro_recall,
                "predicted_links": best_d_pred,
                "tp": best_d_rep.total_tp,
                "fp": best_d_rep.total_fp,
                "singleton_acc": best_d_rep.singleton_f0_5,
                "delta_vs_baseline": best_d_rep.macro_f0_5 - rep_a.macro_f0_5,
            },
        },
        "all_b_sweep": all_b_results,
        "all_d_sweep": all_d_results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(experiment_record, f, indent=2)
    print(f"\nSaved experiment record to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
