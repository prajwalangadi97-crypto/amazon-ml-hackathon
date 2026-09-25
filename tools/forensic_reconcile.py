#!/usr/bin/env python3
"""Forensic reconciliation: verify actual row counts, ground truth, and blocking metrics.

This script independently recomputes EXP-001A/B/C/D metrics from scratch
using the exact same normalizers and validation split, printing every
intermediate count for audit.
"""
import json
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
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

print("=" * 80)
print("FORENSIC RECONCILIATION: EXP-001 Baseline Metrics")
print("=" * 80)

# === Step 1: Verify validation split ===
split_path = Path("experiments/splits/validation_split_seed42_benchmark_10000.json")
with open(split_path) as f:
    split_data = json.load(f)
val_s1_list = split_data["val_s1_ids"]
val_s1_set = set(val_s1_list)
print(f"\n[STEP 1] Validation split loaded from: {split_path}")
print(f"  val_s1_ids count: {len(val_s1_list)}")
print(f"  val_s1_ids unique: {len(val_s1_set)}")
print(f"  First 5 IDs: {val_s1_list[:5]}")

# === Step 2: Verify ground truth ===
gt_path = Path("dataset/train/train_ground_truth.tsv")
val_gt = parse_ground_truth(gt_path, allowed_s1_ids=val_s1_set)
total_true = sum(len(v) for v in val_gt.values())
singletons = sum(1 for v in val_gt.values() if len(v) == 0)
gt_s2 = sum(1 for v in val_gt.values() for m in v if m.startswith("S2-"))
gt_s3 = sum(1 for v in val_gt.values() for m in v if m.startswith("S3-"))
print(f"\n[STEP 2] Ground truth parsed for {len(val_gt)} S1 entities")
print(f"  Total true links: {total_true}")
print(f"  S2 true links: {gt_s2}")
print(f"  S3 true links: {gt_s3}")
print(f"  Singletons (0 matches): {singletons}")
all_true_targets = set()
for matches in val_gt.values():
    all_true_targets.update(matches)
print(f"  Distinct true target IDs: {len(all_true_targets)}")

# === Step 3: Load S1 records ===
s1_path = Path("dataset/train/train_source1.tsv")
val_s1_records = {}
with open(s1_path, "r", encoding="utf-8-sig") as f:
    next(f)  # skip header
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if p[0] in val_s1_set:
            val_s1_records[p[0]] = (p[1], p[2], p[3])
print(f"\n[STEP 3] Loaded {len(val_s1_records)} S1 records from {s1_path}")
# Check all val IDs have records
missing_s1 = val_s1_set - set(val_s1_records.keys())
if missing_s1:
    print(f"  WARNING: {len(missing_s1)} val S1 IDs have NO record in source1!")
else:
    print(f"  All {len(val_s1_set)} validation S1 IDs found in source1.")

# === Step 4: Precompute query keys ===
print(f"\n[STEP 4] Precomputing query keys...")
import collections
query_exact = collections.defaultdict(set)   # (country, normalized_name) -> set(s1_ids)
query_token = collections.defaultdict(set)   # (country, token_key) -> set(s1_ids)
query_addr  = collections.defaultdict(set)   # (country, addr_key) -> set(s1_ids)

for s1_id in val_s1_list:
    name, addr, country = val_s1_records[s1_id]
    
    en = normalize_name_conservative(name)
    if en:
        query_exact[(country, en)].add(s1_id)
    
    for tk in extract_token_blocking_keys(name):
        query_token[(country, tk)].add(s1_id)
    
    for ak in extract_address_keys(addr):
        query_addr[(country, ak)].add(s1_id)

print(f"  Exact name keys: {len(query_exact)}")
print(f"  Token name keys: {len(query_token)}")
print(f"  Address keys   : {len(query_addr)}")

# === Step 5: Stream S2 and S3, count rows, collect candidates ===
cands_a = {s1: set() for s1 in val_s1_list}
cands_b = {s1: set() for s1 in val_s1_list}
cands_c = {s1: set() for s1 in val_s1_list}

source_files = [
    Path("dataset/train/train_source2.tsv"),
    Path("dataset/train/train_source3.tsv"),
]

print(f"\n[STEP 5] Streaming search pool...")
total_pool = 0
source_counts = {}

for src_path in source_files:
    t0 = time.time()
    file_count = 0
    with open(src_path, "r", encoding="utf-8-sig") as f:
        header = f.readline().rstrip("\r\n")
        print(f"\n  Source file: {src_path}")
        print(f"  Header: {header}")
        for line in f:
            file_count += 1
            p = line.rstrip("\r\n").split("\t")
            if len(p) < 4:
                continue
            tid, tname, taddr, tcountry = p[0], p[1], p[2], p[3]
            
            # Strategy A: Exact Name
            tn = normalize_name_conservative(tname)
            if tn:
                matches = query_exact.get((tcountry, tn))
                if matches:
                    for s1 in matches:
                        cands_a[s1].add(tid)
            
            # Strategy B: Token Name
            for tk in extract_token_blocking_keys(tname):
                matches = query_token.get((tcountry, tk))
                if matches:
                    for s1 in matches:
                        cands_b[s1].add(tid)
            
            # Strategy C: Address
            for ak in extract_address_keys(taddr):
                matches = query_addr.get((tcountry, ak))
                if matches:
                    for s1 in matches:
                        cands_c[s1].add(tid)
    
    elapsed = time.time() - t0
    source_counts[src_path.name] = file_count
    total_pool += file_count
    print(f"  Rows read: {file_count:,} in {elapsed:.1f}s")

print(f"\n  TOTAL POOL ROWS: {total_pool:,}")
for k, v in source_counts.items():
    print(f"    {k}: {v:,}")

# === Step 6: Compute strategy D (union) ===
cands_d = {}
for s1 in val_s1_list:
    cands_d[s1] = cands_a[s1] | cands_b[s1] | cands_c[s1]

# === Step 7: Evaluate each strategy ===
print(f"\n[STEP 7] Evaluating strategies...")

def eval_strategy(name, cands_map, gt, val_list):
    total_true = 0
    retrieved_true = 0
    s2_true = 0
    s2_ret = 0
    s3_true = 0
    s3_ret = 0
    total_cands = 0
    zero_cand = 0
    perfect = 0
    missed = 0
    
    for s1_id in val_list:
        true_set = gt.get(s1_id, set())
        cands = cands_map.get(s1_id, set())
        n_cands = len(cands)
        total_cands += n_cands
        
        if n_cands == 0:
            zero_cand += 1
        
        n_true = len(true_set)
        total_true += n_true
        
        s2t = {m for m in true_set if m.startswith("S2-")}
        s3t = {m for m in true_set if m.startswith("S3-")}
        s2_true += len(s2t)
        s3_true += len(s3t)
        
        retrieved_set = cands & true_set
        n_ret = len(retrieved_set)
        retrieved_true += n_ret
        
        s2_ret += len(retrieved_set & s2t)
        s3_ret += len(retrieved_set & s3t)
        
        if n_true == 0:
            perfect += 1
        elif n_ret == n_true:
            perfect += 1
        else:
            missed += 1
    
    recall = retrieved_true / total_true if total_true > 0 else 0.0
    s2_recall = s2_ret / s2_true if s2_true > 0 else 0.0
    s3_recall = s3_ret / s3_true if s3_true > 0 else 0.0
    avg_cands = total_cands / len(val_list)
    
    # compute candidate counts list
    counts = sorted([len(cands_map.get(s1, set())) for s1 in val_list])
    median_c = counts[len(counts)//2]
    p95_idx = int(0.95 * len(counts))
    p95_c = counts[min(p95_idx, len(counts)-1)]
    max_c = counts[-1]
    
    full_space = len(val_list) * total_pool
    rr = 1.0 - (total_cands / full_space) if full_space > 0 else 0.0
    
    print(f"\n  {name}:")
    print(f"    Pair Recall: {recall*100:.2f}% ({retrieved_true:,}/{total_true:,})")
    print(f"    S2 Recall  : {s2_recall*100:.2f}% ({s2_ret:,}/{s2_true:,})")
    print(f"    S3 Recall  : {s3_recall*100:.2f}% ({s3_ret:,}/{s3_true:,})")
    print(f"    Total Cand Pairs: {total_cands:,}")
    print(f"    Avg Cands/S1   : {avg_cands:.2f}")
    print(f"    Median Cands   : {median_c}")
    print(f"    P95 Cands      : {p95_c}")
    print(f"    Max Cands      : {max_c}")
    print(f"    Reduction Ratio: {rr*100:.6f}%")
    print(f"    Zero-Cand Ent  : {zero_cand}")
    print(f"    Perfect Recall : {perfect}")
    print(f"    Missed Matches : {missed}")
    
    return {
        "strategy": name,
        "pair_recall": recall,
        "retrieved_true": retrieved_true,
        "total_true": total_true,
        "s2_recall": s2_recall,
        "s3_recall": s3_recall,
        "total_cands": total_cands,
        "avg_cands": avg_cands,
        "median_cands": median_c,
        "p95_cands": p95_c,
        "max_cands": max_c,
        "reduction_ratio": rr,
        "zero_cand": zero_cand,
        "perfect": perfect,
        "missed": missed,
    }

results = {}
results["A"] = eval_strategy("EXP-001A: Exact Name (Country)", cands_a, val_gt, val_s1_list)
results["B"] = eval_strategy("EXP-001B: Token Name (Country)", cands_b, val_gt, val_s1_list)
results["C"] = eval_strategy("EXP-001C: Address (Country)", cands_c, val_gt, val_s1_list)
results["D"] = eval_strategy("EXP-001D: Union A+B+C (Country)", cands_d, val_gt, val_s1_list)

# === Step 8: Save audit results ===
audit_path = Path("experiments/blocking/FORENSIC_RECONCILIATION_AUDIT.json")
audit_data = {
    "validation_s1_count": len(val_s1_set),
    "total_true_links": total_true,
    "gt_s2_true": gt_s2,
    "gt_s3_true": gt_s3,
    "singletons": singletons,
    "source_row_counts": source_counts,
    "total_pool_rows": total_pool,
    "results": results,
}
with open(audit_path, "w", encoding="utf-8") as f:
    json.dump(audit_data, f, indent=2)
print(f"\n[STEP 8] Audit saved to {audit_path}")

# === Step 9: Compare with both historical baselines ===
print("\n" + "=" * 80)
print("COMPARISON WITH HISTORICAL BASELINES")
print("=" * 80)

v1 = {"recall": 0.9755, "cands": 191732, "avg": 95.82}  # Lines 1-4 (hardcoded log_blocking_results.py)
v2 = {"recall": 0.8922, "cands": 20149645, "avg": 10069.79}  # Lines 5+ (run_blocking_experiments.py)

d_recall = results["D"]["pair_recall"]
d_cands = results["D"]["total_cands"]
d_avg = results["D"]["avg_cands"]

print(f"\n  Version 1 (log_blocking_results.py hardcoded): recall={v1['recall']*100:.2f}%, cands={v1['cands']:,}, avg={v1['avg']:.2f}")
print(f"  Version 2 (run_blocking_experiments.py):        recall={v2['recall']*100:.2f}%, cands={v2['cands']:,}, avg={v2['avg']:.2f}")
print(f"  THIS RUN (forensic_reconcile.py):               recall={d_recall*100:.2f}%, cands={d_cands:,}, avg={d_avg:.2f}")
print(f"\n  Match with V1? recall={abs(d_recall - v1['recall']) < 0.001}, cands={d_cands == v1['cands']}")
print(f"  Match with V2? recall={abs(d_recall - v2['recall']) < 0.001}, cands={d_cands == v2['cands']}")

print("\n" + "=" * 80)
print("RECONCILIATION COMPLETE")
print("=" * 80)
