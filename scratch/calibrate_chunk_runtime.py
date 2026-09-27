#!/usr/bin/env python3
"""Accurately calibrate the exact throughput of 1 batch of 2,000 S1 entities without tracemalloc."""
import collections
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
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

def worker_stream(task_args):
    file_path, start_line, end_line, target_country, c_idx, s1_prep, generic_keys_set, max_generic_per_s1 = task_args
    scored_candidates = collections.defaultdict(list)
    lines_read = 0

    with open(file_path, "r", encoding="utf-8-sig") as f:
        next(f)
        for cur, line in enumerate(f):
            if cur < start_line:
                continue
            if cur >= end_line:
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
                if m: matched_s1_high_spec.update(m)

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
                    if m: matched_s1_high_spec.update(m)
                for ak in extract_h1_relaxed_address_keys(taddr):
                    m = c_idx.get(ak)
                    if m: matched_s1_high_spec.update(m)
                for ck in extract_h3_name_locality_keys(tname, taddr):
                    m = c_idx.get(ck)
                    if m: matched_s1_high_spec.update(m)

            all_matched = matched_s1_high_spec | matched_s1_generic
            if not all_matched:
                continue

            tgt_p = preprocess_record(tid, tname, taddr, tcountry)

            for sid in all_matched:
                score = fast_score_candidate(s1_prep[sid], tgt_p)
                if score >= 0.15:
                    is_high = (sid in matched_s1_high_spec)
                    scored_candidates[sid].append((score, is_high, tid))

    pruned = {}
    for sid, pairs in scored_candidates.items():
        if len(pairs) <= 300:
            pruned[sid] = pairs
        else:
            hs = [p for p in pairs if p[1]]
            gen = [p for p in pairs if not p[1]]
            hs.sort(key=lambda x: x[0], reverse=True)
            gen.sort(key=lambda x: x[0], reverse=True)
            n_gen = min(len(gen), max_generic_per_s1)
            n_hs = min(len(hs), 300 - n_gen)
            n_gen = min(len(gen), 300 - n_hs)
            sel = hs[:n_hs] + gen[:n_gen]
            sel.sort(key=lambda x: x[0], reverse=True)
            pruned[sid] = sel

    return lines_read, pruned

def main():
    t_start = time.time()
    num_workers = 8
    N_S1 = 2000

    print("=" * 80, flush=True)
    print(f"BENCHMARKING 1 CHUNK OF {N_S1:,} TEST S1 RECORDS (FRANCE)", flush=True)
    print("=" * 80, flush=True)

    # 1. Load S1
    t0 = time.time()
    s1_records = []
    with open("dataset/test/test_source1.tsv", "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            if "\tFrance" in line:
                parts = line.rstrip("\r\n").split("\t")
                s1_records.append((parts[0], parts[1], parts[2], parts[3]))
                if len(s1_records) >= N_S1:
                    break

    s1_prep = {}
    c_idx = collections.defaultdict(list)
    for sid, name, addr, country in s1_records:
        prep = preprocess_record(sid, name, addr, country)
        s1_prep[sid] = prep
        if prep.name_norm:
            c_idx[f"en_{prep.name_norm}"].append(sid)
        for tk in extract_token_blocking_keys(prep.raw_name):
            c_idx[f"tk_{tk}"].append(sid)
        if prep.raw_address and prep.raw_address.strip():
            for ak in extract_address_keys(prep.raw_address):
                c_idx[f"ak_{ak}"].append(sid)
            for ak in extract_h1_relaxed_address_keys(prep.raw_address):
                c_idx[ak].append(sid)
            for ck in extract_h3_name_locality_keys(prep.raw_name, prep.raw_address):
                c_idx[ck].append(sid)

    filter_cfg = BlockingKeyFilterConfig(max_key_fanout=15, preserve_exact_name=True, preserve_pin_keys=True)
    pruned_idx, _ = prune_high_frequency_keys(c_idx, total_s1=len(s1_records), config=filter_cfg)
    generic_keys = {k for k, v in pruned_idx.items() if k.startswith("tk_") and len(v) >= 5}
    t_index = time.time() - t0
    print(f"1. Indexing & key filtering: {t_index:.2f}s ({len(pruned_idx)} keys)", flush=True)

    # 2. Streaming search pool
    s2_fp = Path("dataset/test/test_source2.tsv")
    s3_fp = Path("dataset/test/test_source3.tsv")
    s2_lines = 4887273
    s3_lines = 5082316

    tasks = []
    for fp, n_lines in [(s2_fp, s2_lines), (s3_fp, s3_lines)]:
        chunk_sz = (n_lines + num_workers - 1) // num_workers
        for i in range(num_workers):
            tasks.append((
                str(fp),
                i * chunk_sz,
                min((i + 1) * chunk_sz, n_lines),
                "France",
                pruned_idx,
                s1_prep,
                generic_keys,
                150,
            ))

    t0 = time.time()
    with mp.Pool(num_workers) as pool:
        res = pool.map(worker_stream, tasks)

    combined = collections.defaultdict(list)
    for _, pruned in res:
        for sid, pairs in pruned.items():
            combined[sid].extend(pairs)
    t_stream = time.time() - t0
    print(f"2. Parallel streaming: {t_stream:.2f}s", flush=True)

    # 3. Global Cap 1000
    t0 = time.time()
    global_cands = {}
    unique_target_ids = set()
    total_cand_pairs = 0
    for sid, _, _, _ in s1_records:
        raw_pairs = combined.get(sid, [])
        best = {}
        for s, h, tid in raw_pairs:
            if tid not in best:
                best[tid] = (s, h)
            else:
                bs, bh = best[tid]
                best[tid] = (max(bs, s), bh or h)
        dedup = [(s, h, tid) for tid, (s, h) in best.items()]
        if len(dedup) <= 1000:
            sel = [tid for _, _, tid in dedup]
        else:
            hs = [p for p in dedup if p[1]]
            gen = [p for p in dedup if not p[1]]
            hs.sort(key=lambda x: x[0], reverse=True)
            gen.sort(key=lambda x: x[0], reverse=True)
            n_hs = min(len(hs), 1000)
            n_gen = min(len(gen), 1000 - n_hs)
            sel = [tid for _, _, tid in hs[:n_hs] + gen[:n_gen]]
        global_cands[sid] = sel
        total_cand_pairs += len(sel)
        unique_target_ids.update(sel)
    t_cap = time.time() - t0
    print(f"3. Global Cap: {t_cap:.2f}s. Total candidate pairs: {total_cand_pairs:,} (avg {total_cand_pairs/N_S1:.1f}/S1). Unique targets: {len(unique_target_ids):,}", flush=True)

    # Clean intermediate memory
    del combined, res
    gc.collect()

    # 4. Target record reading
    t0 = time.time()
    target_prep = {}
    for fp in [s2_fp, s3_fp]:
        with open(fp, "r", encoding="utf-8-sig") as f:
            next(f)
            for line in f:
                if "\tFrance" not in line:
                    continue
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4 and parts[0] in unique_target_ids:
                    target_prep[parts[0]] = preprocess_record(parts[0], parts[1], parts[2], parts[3])
                    if len(target_prep) == len(unique_target_ids):
                        break
        if len(target_prep) == len(unique_target_ids):
            break
    t_tgt = time.time() - t0
    print(f"4. Target record retrieval & preprocessing: {t_tgt:.2f}s for {len(target_prep):,} targets", flush=True)

    # 5. Feature extraction & XGBoost scoring in batches of 50,000 pairs to keep memory flat
    t0 = time.time()
    bst = xgb.Booster()
    bst.load_model(str(PROJECT_ROOT / "artifacts" / "xgb_matcher_exp002b.json"))
    bst.set_param({"nthread": num_workers})

    matches = collections.defaultdict(list)
    total_preds = 0
    batch_X = []
    batch_pairs = []

    for sid, tids in global_cands.items():
        s1_p = s1_prep[sid]
        for tid in tids:
            tgt_p = target_prep.get(tid)
            if tgt_p:
                f = extract_features_from_preprocessed(s1_p, tgt_p)
                f["cheap_candidate_score"] = compute_cheap_candidate_score(f)
                batch_X.append([f[col] for col in FEATURE_NAMES])
                batch_pairs.append((sid, tid))

                if len(batch_X) >= 50000:
                    dmat = xgb.DMatrix(np.array(batch_X, dtype=np.float32), feature_names=FEATURE_NAMES)
                    probs = bst.predict(dmat)
                    for (s_id, t_id), prob in zip(batch_pairs, probs):
                        if prob >= 0.60:
                            matches[s_id].append(t_id)
                            total_preds += 1
                    batch_X = []
                    batch_pairs = []

    if batch_X:
        dmat = xgb.DMatrix(np.array(batch_X, dtype=np.float32), feature_names=FEATURE_NAMES)
        probs = bst.predict(dmat)
        for (s_id, t_id), prob in zip(batch_pairs, probs):
            if prob >= 0.60:
                matches[s_id].append(t_id)
                total_preds += 1

    t_score = time.time() - t0
    print(f"5. Feature extraction + XGBoost scoring: {t_score:.2f}s. Predicted links: {total_preds:,}", flush=True)

    total_batch_time = time.time() - t_start
    print("=" * 80, flush=True)
    print(f"TOTAL CHUNK TIME FOR {N_S1:,} ENTITIES: {total_batch_time:.2f}s ({total_batch_time/60:.2f} min)", flush=True)
    rate_per_sec = N_S1 / total_batch_time
    print(f"Throughput: {rate_per_sec:.2f} S1 entities/sec", flush=True)
    
    # Extrapolate for full test set (1,732,544 entities)
    total_test_s1 = 1732544
    est_total_seconds = total_test_s1 / rate_per_sec
    est_total_hours = est_total_seconds / 3600
    print(f"PROJECTED FULL RUNTIME FOR ALL {total_test_s1:,} S1 ENTITIES: {est_total_seconds:,.1f}s ({est_total_hours:.2f} HOURS)", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    mp.freeze_support()
    main()
