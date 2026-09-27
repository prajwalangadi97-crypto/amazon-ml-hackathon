#!/usr/bin/env python3
"""High-performance multicore blocking-key distribution analysis on the FULL TEST S1 data.

Analyzes the full S1 blocking index distribution for France, US, and India
without scanning Source 2/Source 3, generating candidates, or running inference.

Computes:
- Total S1 entities
- Total unique blocking keys
- Keys with fanout >= [15, 25, 50, 75, 100, 150, 200, 300, 500]
- Maximum fanout and top pathological keys
- Percentage of keys removed at each threshold (both raw fanout and via frequency filter)
- Postings reduction percentage at each threshold

Engineered for strict bounded memory (<400 MB peak RAM) using 8-core parallel chunking.
"""

import collections
import gc
import json
import multiprocessing as mp
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from tools.run_recovery_experiments import (
    extract_h1_relaxed_address_keys,
    extract_h3_name_locality_keys,
)

THRESHOLDS = [15, 25, 50, 75, 100, 150, 200, 300, 500]


def extract_s1_keys_for_record(sid: str, name: str, addr: str) -> List[str]:
    """Extract all blocking keys for a single S1 record (exact match to EXP-002B)."""
    keys = []
    en = normalize_name_conservative(name)
    if en:
        keys.append(f"en_{en}")

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


def worker_extract_chunk(args: Tuple[int, str, int, int]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    """Worker task to process a chunk of lines from test_source1.tsv."""
    worker_id, file_path, start_line, end_line = args

    country_counts: Dict[str, Dict[str, int]] = {
        "France": collections.defaultdict(int),
        "US": collections.defaultdict(int),
        "India": collections.defaultdict(int),
    }
    entity_counts: Dict[str, int] = collections.defaultdict(int)

    with open(file_path, "r", encoding="utf-8-sig") as f:
        next(f)  # skip header
        for current_line, line in enumerate(f):
            if current_line < start_line:
                continue
            if current_line >= end_line:
                break

            c_pos = line.rfind("\t")
            if c_pos == -1:
                continue
            country = line[c_pos + 1 :].rstrip("\r\n")
            if country not in country_counts:
                continue

            entity_counts[country] += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                continue
            sid, name, addr = parts[0], parts[1], parts[2]

            record_keys = extract_s1_keys_for_record(sid, name, addr)
            c_dict = country_counts[country]
            for k in record_keys:
                c_dict[k] += 1

    # Convert defaultdicts to regular dicts for clean pickling
    res_counts = {c: dict(d) for c, d in country_counts.items() if d}
    return res_counts, dict(entity_counts)


def analyze_test_s1_distribution(num_workers: int = 8) -> Dict[str, Any]:
    test_s1_path = PROJECT_ROOT / "dataset" / "test" / "test_source1.tsv"
    if not test_s1_path.exists():
        raise FileNotFoundError(f"File not found: {test_s1_path}")

    overall_start = time.time()
    tracemalloc.start()

    # 1. Count lines
    with open(test_s1_path, "r", encoding="utf-8-sig") as f:
        total_lines = sum(1 for _ in f) - 1  # exclude header

    chunk_size = (total_lines + num_workers - 1) // num_workers
    tasks = []
    for i in range(num_workers):
        start_line = i * chunk_size
        end_line = min((i + 1) * chunk_size, total_lines)
        if start_line < total_lines:
            tasks.append((i, str(test_s1_path), start_line, end_line))

    # 2. Parallel Extraction across workers
    with mp.Pool(processes=num_workers) as pool:
        chunk_results = pool.map(worker_extract_chunk, tasks)

    # 3. Merge counts in master
    master_counts: Dict[str, Dict[str, int]] = {
        "France": collections.defaultdict(int),
        "US": collections.defaultdict(int),
        "India": collections.defaultdict(int),
    }
    master_entities: Dict[str, int] = collections.defaultdict(int)

    for worker_country_counts, worker_entity_counts in chunk_results:
        for c, e_cnt in worker_entity_counts.items():
            master_entities[c] += e_cnt
        for c, k_dict in worker_country_counts.items():
            c_master = master_counts[c]
            for k, cnt in k_dict.items():
                c_master[k] += cnt

    del chunk_results
    gc.collect()

    # 4. Statistical Analysis per Country
    country_reports: Dict[str, Any] = {}
    for country in ["France", "US", "India"]:
        c_dict = master_counts[country]
        total_s1 = master_entities[country]
        total_unique_keys = len(c_dict)
        total_postings = sum(c_dict.values())
        avg_keys_per_entity = total_postings / max(total_s1, 1)

        # Sort keys descending by fanout
        sorted_keys = sorted(c_dict.items(), key=lambda x: x[1], reverse=True)
        max_fanout_key, max_fanout = sorted_keys[0] if sorted_keys else ("", 0)

        # Build virtual index for frequency filter: key -> range(sz)
        # range(sz) has __len__ == sz and takes 48 bytes
        virtual_c_idx = {k: range(sz) for k, sz in c_dict.items()}

        threshold_stats = []
        for th in THRESHOLDS:
            # Raw fanout >= th
            raw_ge_count = sum(1 for _, sz in sorted_keys if sz >= th)
            raw_ge_pct = (raw_ge_count / total_unique_keys * 100.0) if total_unique_keys > 0 else 0.0

            # Using prune_high_frequency_keys with max_key_fanout = th (filters fanout > th)
            cfg_gt = BlockingKeyFilterConfig(
                max_key_fanout=th,
                preserve_exact_name=True,
                preserve_pin_keys=True,
            )
            _, filter_audit_gt = prune_high_frequency_keys(virtual_c_idx, total_s1, cfg_gt)

            # Using prune_high_frequency_keys with max_key_fanout = th - 1 (filters fanout >= th)
            cfg_ge = BlockingKeyFilterConfig(
                max_key_fanout=th - 1,
                preserve_exact_name=True,
                preserve_pin_keys=True,
            )
            _, filter_audit_ge = prune_high_frequency_keys(virtual_c_idx, total_s1, cfg_ge)

            threshold_stats.append({
                "threshold": th,
                "raw_keys_ge_th": raw_ge_count,
                "raw_pct_ge_th": raw_ge_pct,
                "filter_pruned_ge_th": filter_audit_ge["pruned_key_count"],
                "filter_pct_ge_th": (filter_audit_ge["pruned_key_count"] / total_unique_keys * 100.0) if total_unique_keys > 0 else 0.0,
                "postings_reduction_ge_th": filter_audit_ge["postings_reduction_pct"],
                "filter_pruned_gt_th": filter_audit_gt["pruned_key_count"],
                "filter_pct_gt_th": (filter_audit_gt["pruned_key_count"] / total_unique_keys * 100.0) if total_unique_keys > 0 else 0.0,
            })

        # Top 15 pathological keys
        top_15 = [
            {
                "key": k,
                "fanout": sz,
                "ratio_pct": (sz / total_s1 * 100.0) if total_s1 > 0 else 0.0,
            }
            for k, sz in sorted_keys[:15]
        ]

        country_reports[country] = {
            "country": country,
            "total_s1_entities": total_s1,
            "total_unique_blocking_keys": total_unique_keys,
            "total_postings": total_postings,
            "avg_keys_per_entity": avg_keys_per_entity,
            "max_fanout": max_fanout,
            "max_fanout_key": max_fanout_key,
            "top_15_keys": top_15,
            "threshold_stats": threshold_stats,
        }

    total_time = time.time() - overall_start
    curr_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    full_report = {
        "analysis": "FULL TEST S1 BLOCKING-KEY DISTRIBUTION ANALYSIS",
        "dataset": str(test_s1_path),
        "total_test_s1_records": total_lines,
        "runtime_seconds": total_time,
        "peak_ram_mb": peak_mem / 1024**2,
        "countries": country_reports,
    }

    # Save to JSON
    output_json = PROJECT_ROOT / "experiments" / "test_s1_blocking_key_distribution.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    return full_report


if __name__ == "__main__":
    rep = analyze_test_s1_distribution(num_workers=8)
    print("DONE. Analysis completed successfully in {:.2f}s.".format(rep["runtime_seconds"]))
