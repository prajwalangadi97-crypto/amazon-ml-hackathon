#!/usr/bin/env python3
"""Feasibility Benchmark: Reusable Target-Side Inverted Blocking Index for test_source2 + test_source3.

Measures:
1. Index build runtime
2. Index disk size
3. Peak RAM
4. Number of target records indexed
5. Total postings
6. Lookup speed for a representative 2,000-S1 sample
7. Estimated full-test runtime if this architecture is used
"""
import array
import collections
import ctypes
from ctypes import wintypes
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    "country_eq", "source_type", "s1_name_empty", "target_name_empty",
    "s1_addr_empty", "target_addr_empty", "name_exact_eq", "name_fuzz_ratio",
    "name_fuzz_wratio", "name_token_set_ratio", "name_token_sort_ratio",
    "name_prefix_similarity", "name_len_diff", "name_len_ratio",
    "addr_fuzz_ratio", "addr_token_similarity", "numeric_token_overlap",
    "house_number_agreement", "postal_pin_overlap", "addr_len_diff",
    "cross_field_name_loc", "cheap_candidate_score",
]

class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
        ('PrivateUsage', ctypes.c_size_t),
    ]

def get_peak_ram_mb():
    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / (1024 * 1024)
    except Exception:
        return 0.0

def extract_all_keys(name: str, addr: str):
    keys = []
    ten = normalize_name_conservative(name)
    if ten:
        keys.append(f"en_{ten}")
    for tk in extract_token_blocking_keys(name):
        keys.append(f"tk_{tk}")
    if addr and addr.strip():
        for ak in extract_address_keys(addr):
            keys.append(f"ak_{ak}")
        for ak in extract_h1_relaxed_address_keys(addr):
            keys.append(ak)
        for ck in extract_h3_name_locality_keys(name, addr):
            keys.append(ck)
    return list(dict.fromkeys(keys))

def main():
    t_start = time.time()
    print("=" * 90, flush=True)
    print("FEASIBILITY TEST: TARGET-SIDE INVERTED BLOCKING INDEX", flush=True)
    print("=" * 90, flush=True)

    target_country = "France"
    s2_fp = PROJECT_ROOT / "dataset" / "test" / "test_source2.tsv"
    s3_fp = PROJECT_ROOT / "dataset" / "test" / "test_source3.tsv"
    s1_fp = PROJECT_ROOT / "dataset" / "test" / "test_source1.tsv"

    # =========================================================================
    # 1. BUILD TARGET INDEX FOR FRANCE (1,434,993 target records)
    # =========================================================================
    print(f"\n[PHASE 1] Building reusable target-side inverted index for {target_country}...", flush=True)
    t0_build = time.time()

    target_ids = []
    target_preps = []
    inverted_index = collections.defaultdict(lambda: array.array('I'))

    records_indexed = 0
    total_postings = 0

    for fp in [s2_fp, s3_fp]:
        print(f"  Scanning {fp.name}...", flush=True)
        with open(fp, "r", encoding="utf-8-sig") as f:
            next(f)  # header
            for line in f:
                if f"\t{target_country}" not in line:
                    continue
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4:
                    continue
                tid, tname, taddr, tcountry = parts[0], parts[1], parts[2], parts[3]
                if tcountry != target_country:
                    continue

                t_idx = len(target_ids)
                target_ids.append(tid)
                
                # Preprocess target record
                prep = preprocess_record(tid, tname, taddr, tcountry)
                target_preps.append(prep)

                # Extract keys
                keys = extract_all_keys(tname, taddr)
                for k in keys:
                    inverted_index[k].append(t_idx)
                    total_postings += 1

                records_indexed += 1

    t_build = time.time() - t0_build
    unique_keys = len(inverted_index)
    peak_ram_build = get_peak_ram_mb()

    print(f"\nTarget Index Build Summary for {target_country}:")
    print(f"  Target Records Indexed : {records_indexed:,}")
    print(f"  Unique Blocking Keys   : {unique_keys:,}")
    print(f"  Total Postings         : {total_postings:,}")
    print(f"  Index Build Runtime    : {t_build:.2f}s ({t_build/60:.2f} min)")
    print(f"  Peak RAM during Build  : {peak_ram_build:.2f} MB")

    # Flag generic keys (tk_ with fanout >= 15)
    generic_keys = {k for k, post in inverted_index.items() if k.startswith("tk_") and len(post) >= 15}
    print(f"  Generic Keys Flagged   : {len(generic_keys):,} (fanout >= 15)")

    # =========================================================================
    # 2. MEASURE DISK SIZE (Serialize to Disk)
    # =========================================================================
    print("\n[PHASE 2] Serializing index to disk...", flush=True)
    t0_disk = time.time()
    index_cache_file = PROJECT_ROOT / "scratch" / f"target_index_{target_country.lower()}.pkl"
    
    # Clean dict representation for serialization
    index_dict = dict(inverted_index)
    with open(index_cache_file, "wb") as f:
        pickle.dump({
            "target_ids": target_ids,
            "inverted_index": index_dict,
        }, f, protocol=5)

    disk_size_bytes = os.path.getsize(index_cache_file)
    disk_size_mb = disk_size_bytes / (1024 * 1024)
    t_disk = time.time() - t0_disk
    print(f"  Index serialized in {t_disk:.2f}s")
    print(f"  Index Disk Size        : {disk_size_mb:.2f} MB ({disk_size_bytes:,} bytes)")
    print(f"  Disk File Location     : {index_cache_file.resolve()}")

    # =========================================================================
    # 3. BENCHMARK LOOKUP SPEED FOR 2,000 S1 SAMPLE
    # =========================================================================
    print(f"\n[PHASE 3] Benchmarking lookup speed on representative 2,000-S1 sample...", flush=True)
    N_S1 = 2000
    s1_sample = []
    with open(s1_fp, "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            if f"\t{target_country}" in line:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4 and parts[3] == target_country:
                    s1_sample.append((parts[0], parts[1], parts[2], parts[3]))
                    if len(s1_sample) >= N_S1:
                        break

    s1_preps = {sid: preprocess_record(sid, name, addr, c) for sid, name, addr, c in s1_sample}
    print(f"  Loaded and preprocessed {len(s1_sample):,} {target_country} S1 records.", flush=True)

    # Perform Inverted Index Lookups with Anti-Crowding & Global Cap 1000
    t0_lookup = time.time()
    all_s1_candidates = {}
    total_candidates_generated = 0
    unique_targets_matched = set()

    for sid, name, addr, _ in s1_sample:
        s1_p = s1_preps[sid]
        s1_keys = extract_all_keys(name, addr)

        high_spec_targets = set()
        generic_targets = set()

        for k in s1_keys:
            postings = inverted_index.get(k)
            if not postings:
                continue
            if k.startswith("tk_") and k in generic_keys:
                generic_targets.update(postings)
            else:
                high_spec_targets.update(postings)

        # Anti-crowding slot management:
        # Score candidates with fast_score_candidate >= 0.15
        scored_high = []
        for t_idx in high_spec_targets:
            tgt_p = target_preps[t_idx]
            sc = fast_score_candidate(s1_p, tgt_p)
            if sc >= 0.15:
                scored_high.append((sc, t_idx))

        scored_gen = []
        for t_idx in generic_targets:
            if t_idx in high_spec_targets:
                continue
            tgt_p = target_preps[t_idx]
            sc = fast_score_candidate(s1_p, tgt_p)
            if sc >= 0.15:
                scored_gen.append((sc, t_idx))

        scored_high.sort(key=lambda x: x[0], reverse=True)
        scored_gen.sort(key=lambda x: x[0], reverse=True)

        # Priority protection: max 150 generic, then global cap 1000
        n_gen = min(len(scored_gen), 150)
        n_high = min(len(scored_high), 1000 - n_gen)
        # If high spec has fewer than 850, generic can take up to 150
        # If high spec has more, generic is capped at 150 and high spec takes the rest up to 1000:
        n_high = min(len(scored_high), 1000)
        n_gen = min(len(scored_gen), 1000 - n_high, 150)

        selected = scored_high[:n_high] + scored_gen[:n_gen]
        selected_tids = [target_ids[t_idx] for _, t_idx in selected]
        all_s1_candidates[sid] = selected_tids
        total_candidates_generated += len(selected_tids)
        unique_targets_matched.update(selected_tids)

    t_lookup = time.time() - t0_lookup
    lookup_rate = N_S1 / t_lookup
    avg_cands = total_candidates_generated / N_S1

    print(f"\nLookup Performance for {N_S1:,} S1 sample:")
    print(f"  Lookup Runtime         : {t_lookup:.2f}s")
    print(f"  Lookup Throughput      : {lookup_rate:.1f} S1 queries / sec ({t_lookup/N_S1*1000:.2f} ms/query)")
    print(f"  Total Candidates       : {total_candidates_generated:,} (avg {avg_cands:.1f}/S1)")
    print(f"  Unique Targets Matched : {len(unique_targets_matched):,}")

    # =========================================================================
    # 4. MEASURE XGBOOST SCORING FOR THESE CANDIDATES
    # =========================================================================
    print("\n[PHASE 4] Scoring candidates with EXP-002B XGBoost matcher...", flush=True)
    t0_score = time.time()
    
    # Map tid to prep
    target_prep_by_id = {target_ids[i]: target_preps[i] for i in range(len(target_ids)) if target_ids[i] in unique_targets_matched}

    bst = xgb.Booster()
    bst.load_model(str(PROJECT_ROOT / "artifacts" / "xgb_matcher_exp002b.json"))
    bst.set_param({"nthread": 8})

    pred_matches = 0
    batch_X = []
    for sid, tids in all_s1_candidates.items():
        s1_p = s1_preps[sid]
        for tid in tids:
            tgt_p = target_prep_by_id[tid]
            f = extract_features_from_preprocessed(s1_p, tgt_p)
            f["cheap_candidate_score"] = compute_cheap_candidate_score(f)
            batch_X.append([f[col] for col in FEATURE_NAMES])
            if len(batch_X) >= 50000:
                dmat = xgb.DMatrix(np.array(batch_X, dtype=np.float32), feature_names=FEATURE_NAMES)
                probs = bst.predict(dmat)
                pred_matches += sum(1 for p in probs if p >= 0.60)
                batch_X = []

    if batch_X:
        dmat = xgb.DMatrix(np.array(batch_X, dtype=np.float32), feature_names=FEATURE_NAMES)
        probs = bst.predict(dmat)
        pred_matches += sum(1 for p in probs if p >= 0.60)

    t_score = time.time() - t0_score
    print(f"  Feature Extraction + Scoring : {t_score:.2f}s")
    print(f"  Predicted Links (theta >= 0.60): {pred_matches:,}")

    total_2k_runtime = t_lookup + t_score
    peak_ram_overall = get_peak_ram_mb()
    print(f"\nTotal Processing Time for 2,000 S1 records (Lookup + Scoring): {total_2k_runtime:.2f}s ({total_2k_runtime/60:.2f} min)")
    print(f"Peak RAM Overall: {peak_ram_overall:.2f} MB")

    # =========================================================================
    # 5. FULL TEST INFERENCE PROJECTION ACROSS ALL 1,732,544 S1 RECORDS
    # =========================================================================
    print("\n" + "=" * 90, flush=True)
    print("PROJECTION: FULL-TEST RUNTIME UNDER TARGET-SIDE INDEX ARCHITECTURE", flush=True)
    print("=" * 90, flush=True)

    # Breakdown by country:
    # Target pool counts:
    # France: 1,434,993 targets | 259,452 S1
    # US:     3,817,031 targets | 663,106 S1
    # India:  4,717,565 targets | 809,986 S1
    # Total:  9,969,589 targets | 1,732,544 S1

    # Build time scales linearly with target pool size:
    rate_targets_per_sec = records_indexed / t_build
    total_targets_all = 9969589
    est_total_index_build_sec = total_targets_all / rate_targets_per_sec

    # S1 lookup + scoring time scales linearly with S1 count:
    # 2,000 S1 took total_2k_runtime
    rate_s1_processed_per_sec = N_S1 / total_2k_runtime
    total_s1_all = 1732544
    est_total_s1_inference_sec = total_s1_all / rate_s1_processed_per_sec

    # Overall full test runtime:
    est_grand_total_sec = est_total_index_build_sec + est_total_s1_inference_sec
    est_grand_total_hours = est_grand_total_sec / 3600

    print(f"Target Index Build Rate          : {rate_targets_per_sec:,.0f} target records / sec")
    print(f"Estimated Total Index Build Time : {est_total_index_build_sec:,.1f}s ({est_total_index_build_sec/60:.2f} min)")
    print(f"S1 Inference Throughput          : {rate_s1_processed_per_sec:.2f} S1 entities / sec")
    print(f"Estimated S1 Total Inference Time: {est_total_s1_inference_sec:,.1f}s ({est_total_s1_inference_sec/3600:.2f} hours)")
    print(f"ESTIMATED FULL-TEST RUNTIME      : {est_grand_total_sec:,.1f}s ({est_grand_total_hours:.2f} HOURS)")
    print(f"Runtime vs Previous Architecture : ~{43.54 / max(est_grand_total_hours, 0.01):.1f}x speedup (down from ~43.5 hours)")
    print("=" * 90, flush=True)

if __name__ == "__main__":
    main()
