#!/usr/bin/env python3
"""Calibrate candidate generation, feature extraction, and XGBoost scoring on test S1 chunk."""
import collections
import json
import multiprocessing as mp
import os
import sys
import time
import tracemalloc
from pathlib import Path
import numpy as np
import xgboost as xgb

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
from src.matching.candidate_compression import (
    fast_score_candidate,
    preprocess_record,
    extract_features_from_preprocessed,
    compute_cheap_candidate_score,
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

def worker_stream_chunk(task_args):
    worker_id, file_path, start_line, end_line, target_country, c_idx, s1_prep, generic_keys_set, max_generic_per_s1 = task_args
    scored_candidates = collections.defaultdict(list)
    lines_read = 0
    pairs_scored = 0

    with open(file_path, "r", encoding="utf-8-sig") as f:
        next(f)
        for current_line, line in enumerate(f):
            if current_line < start_line:
                continue
            if current_line >= end_line:
                break
            lines_read += 1
            if f"\t{target_country}" not in line:
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            tid, tname, taddr, tcountry = parts[0], parts[1], parts[2], parts[3]
            if tcountry != target_country:
                continue

            matched_s1_high_spec = set()
            matched_s1_generic = set()

            ten = normalize_name_conservative(tname)
            if ten:
                m = c_idx.get(f"en_{ten}")
                if m:
                    matched_s1_high_spec.update(m)

            for tk in extract_token_blocking_keys(tname):
                k = f"tk_{tk}"
                m = c_idx.get(k)
                if m:
                    if k in generic_keys_set:
                        matched_s1_generic.update(m)
                    else:
                        matched_s1_high_spec.update(m)

            if taddr and taddr.strip():
                for ak in extract_address_keys(taddr):
                    m = c_idx.get(f"ak_{ak}")
                    if m:
                        matched_s1_high_spec.update(m)
                for ak in extract_h1_relaxed_address_keys(taddr):
                    m = c_idx.get(ak)
                    if m:
                        matched_s1_high_spec.update(m)
                for ck in extract_h3_name_locality_keys(tname, taddr):
                    m = c_idx.get(ck)
                    if m:
                        matched_s1_high_spec.update(m)

            all_matched = matched_s1_high_spec | matched_s1_generic
            if not all_matched:
                continue

            tgt_p = preprocess_record(tid, tname, taddr, tcountry)

            for sid in all_matched:
                score = fast_score_candidate(s1_prep[sid], tgt_p)
                if score >= 0.15:
                    is_high_spec = (sid in matched_s1_high_spec)
                    scored_candidates[sid].append((score, is_high_spec, tid))
                pairs_scored += 1

    # Worker pruning
    pruned_scored = {}
    for s1_id, pairs in scored_candidates.items():
        if len(pairs) <= 300:
            pruned_scored[s1_id] = pairs
        else:
            high_spec = [p for p in pairs if p[1]]
            generic = [p for p in pairs if not p[1]]
            high_spec.sort(key=lambda x: x[0], reverse=True)
            generic.sort(key=lambda x: x[0], reverse=True)
            n_generic = min(len(generic), max_generic_per_s1)
            n_high_spec = min(len(high_spec), 300 - n_generic)
            n_generic = min(len(generic), 300 - n_high_spec)
            selected = high_spec[:n_high_spec] + generic[:n_generic]
            selected.sort(key=lambda x: x[0], reverse=True)
            pruned_scored[s1_id] = selected

    return lines_read, pairs_scored, pruned_scored

def run_test():
    tracemalloc.start()
    t_start = time.time()
    
    # Take first 1,000 S1 records of France
    N_S1 = 1000
    s1_records = []
    with open("dataset/test/test_source1.tsv", "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and parts[3] == "France":
                s1_records.append((parts[0], parts[1], parts[2], parts[3]))
                if len(s1_records) >= N_S1:
                    break
    print(f"Loaded {len(s1_records)} France S1 records.")

    s1_prep = {}
    c_idx = collections.defaultdict(list)
    for sid, name, addr, country in s1_records:
        prep = preprocess_record(sid, name, addr, country)
        s1_prep[sid] = prep
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

    # Filter
    filter_cfg = BlockingKeyFilterConfig(max_key_fanout=15, preserve_exact_name=True, preserve_pin_keys=True)
    pruned_idx, audit = prune_high_frequency_keys(c_idx, total_s1=len(s1_records), config=filter_cfg)
    generic_keys = {k for k, v in pruned_idx.items() if k.startswith("tk_") and len(v) >= 5}
    print(f"Index built: {len(pruned_idx)} keys, {len(generic_keys)} generic.")

    # Stream France test_source2 and test_source3
    num_workers = 8
    # Count lines for France in s2 and s3
    s2_fp = Path("dataset/test/test_source2.tsv")
    s3_fp = Path("dataset/test/test_source3.tsv")
    
    with open(s2_fp, "r", encoding="utf-8-sig") as f:
        next(f)
        s2_lines = sum(1 for _ in f)
    with open(s3_fp, "r", encoding="utf-8-sig") as f:
        next(f)
        s3_lines = sum(1 for _ in f)

    t0_stream = time.time()
    tasks = []
    for fp, n_lines in [(s2_fp, s2_lines), (s3_fp, s3_lines)]:
        chunk_sz = (n_lines + num_workers - 1) // num_workers
        for i in range(num_workers):
            start = i * chunk_sz
            end = min((i + 1) * chunk_sz, n_lines)
            tasks.append((i, str(fp), start, end, "France", pruned_idx, s1_prep, generic_keys, 150))

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(worker_stream_chunk, tasks)

    t_stream = time.time() - t0_stream
    total_lines = sum(r[0] for r in results)
    total_pairs_scored = sum(r[1] for r in results)
    combined = collections.defaultdict(list)
    for r in results:
        for sid, pairs in r[2].items():
            combined[sid].extend(pairs)

    print(f"Streaming took {t_stream:.2f}s. Scored {total_pairs_scored:,} pairs across {total_lines:,} lines.")

    # Global cap 1000
    t0_cap = time.time()
    global_cands = {}
    total_cands = 0
    all_target_ids = set()
    for sid, _, _, _ in s1_records:
        raw_pairs = combined.get(sid, [])
        best = {}
        for score, is_high, tid in raw_pairs:
            if tid not in best:
                best[tid] = (score, is_high)
            else:
                s, h = best[tid]
                best[tid] = (max(s, score), h or is_high)
        deduped = [(s, h, tid) for tid, (s, h) in best.items()]
        if len(deduped) <= 1000:
            c_set = [tid for _, _, tid in deduped]
        else:
            high_spec = [p for p in deduped if p[1]]
            generic = [p for p in deduped if not p[1]]
            high_spec.sort(key=lambda x: x[0], reverse=True)
            generic.sort(key=lambda x: x[0], reverse=True)
            n_high = min(len(high_spec), 1000)
            n_gen = min(len(generic), 1000 - n_high)
            selected = high_spec[:n_high] + generic[:n_gen]
            c_set = [tid for _, _, tid in selected]
        global_cands[sid] = c_set
        total_cands += len(c_set)
        all_target_ids.update(c_set)

    t_cap = time.time() - t0_cap
    print(f"Global cap took {t_cap:.2f}s. Total candidates: {total_cands:,} (avg {total_cands/N_S1:.1f}/S1). Unique targets: {len(all_target_ids):,}")

    # Read target records for these candidate pairs
    t0_read_tgt = time.time()
    target_prep = {}
    for fp in [s2_fp, s3_fp]:
        with open(fp, "r", encoding="utf-8-sig") as f:
            next(f)
            for line in f:
                if "\tFrance" not in line:
                    continue
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4 and parts[0] in all_target_ids:
                    target_prep[parts[0]] = preprocess_record(parts[0], parts[1], parts[2], parts[3])
                    if len(target_prep) == len(all_target_ids):
                        break
        if len(target_prep) == len(all_target_ids):
            break
    print(f"Target read took {time.time() - t0_read_tgt:.2f}s for {len(target_prep):,} targets.")

    # Feature extraction
    t0_feats = time.time()
    X_rows = []
    pairs_list = []
    for sid, tids in global_cands.items():
        s1_p = s1_prep[sid]
        for tid in tids:
            tgt_p = target_prep.get(tid)
            if tgt_p:
                f = extract_features_from_preprocessed(s1_p, tgt_p)
                f["cheap_candidate_score"] = compute_cheap_candidate_score(f)
                X_rows.append([f[c] for c in FEATURE_NAMES])
                pairs_list.append((sid, tid))
    t_feat = time.time() - t0_feats
    print(f"Feature extraction took {t_feat:.2f}s for {len(X_rows):,} pairs ({len(X_rows)/max(t_feat, 0.001):.0f} pairs/s).")

    # XGBoost scoring
    t0_xgb = time.time()
    bst = xgb.Booster()
    bst.load_model(str(PROJECT_ROOT / "artifacts" / "xgb_matcher_exp002b.json"))
    bst.set_param({"nthread": num_workers})
    X_mat = np.array(X_rows, dtype=np.float32)
    dmat = xgb.DMatrix(X_mat, feature_names=FEATURE_NAMES)
    preds = bst.predict(dmat)
    t_xgb = time.time() - t0_xgb
    print(f"XGBoost scoring took {t_xgb:.2f}s for {len(preds):,} pairs.")

    matches = sum(1 for p in preds if p >= 0.60)
    print(f"Predicted matches >= 0.60: {matches:,}")

    total_batch_time = time.time() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Total calibration time for {N_S1} entities: {total_batch_time:.2f}s. Peak RAM: {peak_mem/(1024*1024):.1f} MB.")

if __name__ == "__main__":
    mp.freeze_support()
    run_test()
