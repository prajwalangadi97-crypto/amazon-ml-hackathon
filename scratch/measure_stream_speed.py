#!/usr/bin/env python3
"""Measure raw streaming throughput of 8 workers scanning France search pool."""
import collections
import multiprocessing as mp
from pathlib import Path
import sys
import time

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
)
from tools.run_recovery_experiments import (
    extract_h1_relaxed_address_keys,
    extract_h3_name_locality_keys,
)

def worker_bench(args):
    worker_id, file_path, start_line, end_line, c_idx, s1_prep = args
    lines_read = 0
    matches = 0
    t0 = time.time()
    with open(file_path, "r", encoding="utf-8-sig") as f:
        next(f)
        for cur, line in enumerate(f):
            if cur < start_line:
                continue
            if cur >= end_line:
                break
            lines_read += 1
            if "\tFrance" not in line:
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            tid, tname, taddr = parts[0], parts[1], parts[2]
            
            matched = set()
            ten = normalize_name_conservative(tname)
            if ten:
                m = c_idx.get(f"en_{ten}")
                if m: matched.update(m)
            for tk in extract_token_blocking_keys(tname):
                m = c_idx.get(f"tk_{tk}")
                if m: matched.update(m)
            if taddr and taddr.strip():
                for ak in extract_address_keys(taddr):
                    m = c_idx.get(f"ak_{ak}")
                    if m: matched.update(m)
            if matched:
                matches += len(matched)
    dur = time.time() - t0
    return lines_read, matches, dur

def main():
    # Load 500 France S1 entities
    s1_prep = {}
    c_idx = collections.defaultdict(list)
    with open("dataset/test/test_source1.tsv", "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            if "\tFrance" in line:
                parts = line.rstrip("\r\n").split("\t")
                sid, name, addr = parts[0], parts[1], parts[2]
                prep = preprocess_record(sid, name, addr, "France")
                s1_prep[sid] = prep
                if prep.name_norm:
                    c_idx[f"en_{prep.name_norm}"].append(sid)
                for tk in extract_token_blocking_keys(prep.raw_name):
                    c_idx[f"tk_{tk}"].append(sid)
                if len(s1_prep) >= 500:
                    break
    c_idx = dict(c_idx)
    print(f"Loaded 500 France S1 entities, {len(c_idx)} keys.")

    s2_fp = "dataset/test/test_source2.tsv"
    # test_source2 has 4,887,273 lines total
    num_workers = 8
    n_lines = 4887273
    chunk_sz = (n_lines + num_workers - 1) // num_workers
    tasks = [(i, s2_fp, i * chunk_sz, min((i + 1) * chunk_sz, n_lines), c_idx, s1_prep) for i in range(num_workers)]
    
    t0 = time.time()
    with mp.Pool(num_workers) as pool:
        res = pool.map(worker_bench, tasks)
    total_time = time.time() - t0
    print(f"Streaming test_source2 with 8 workers took: {total_time:.2f}s")

if __name__ == "__main__":
    mp.freeze_support()
    main()
