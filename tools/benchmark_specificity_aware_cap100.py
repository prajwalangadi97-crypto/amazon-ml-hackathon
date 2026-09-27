#!/usr/bin/env python3
"""Validation benchmark for specificity-aware candidate selection with cap=100.

Evaluates specificity-aware candidate selection on the 2,001-entity validation split
against the full 10,320,219-record training search pool (train_source2 + train_source3).

Priority order:
1. exact normalized name (en_...)
2. PIN (pin_...)
3. rare address-derived keys (ak_..., nword_...)
4. rare locality/name keys (loc_..., nl_...)
5. rare token keys (tk_... with fanout < 5)
6. generic high-frequency token keys (tk_... with fanout >= 5)

Final candidate budget = 100 candidates per S1 entity.
Anti-crowding protection guarantees generic high-frequency token keys cannot crowd out specific candidates.
"""

import argparse
import collections
import ctypes
from ctypes import wintypes
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking.evaluator import evaluate_blocking, BlockingResult
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
from src.matching.candidate_compression import (
    fast_score_candidate,
    preprocess_record,
    PreprocessedRecord,
)
from tools.run_recovery_experiments import (
    extract_h1_relaxed_address_keys,
    extract_h3_name_locality_keys,
)


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


def get_peak_ram_mb() -> float:
    """Return peak working set memory in MB on Windows, or 0.0 on failure."""
    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / (1024 * 1024)
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

    t0 = time.time()
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

    # Specificity-aware worker pruning:
    # Retain up to 100 candidates per S1 in each worker, sorting by (tier ASC, score DESC)
    pruned_scored = {}
    for sid, pairs in scored_candidates.items():
        if len(pairs) <= 100:
            pruned_scored[sid] = pairs
        else:
            specific = [p for p in pairs if p[0] <= 5]
            generic = [p for p in pairs if p[0] == 6]

            # Specific candidates sorted by tier first, then score descending
            specific.sort(key=lambda x: (x[0], -x[1]))
            # Generic candidates sorted by score descending
            generic.sort(key=lambda x: -x[1])

            # Anti-crowding: generic candidates cannot exceed max_generic_per_s1
            n_gen = min(len(generic), max_generic_per_s1)
            n_spec = min(len(specific), 100 - n_gen)
            n_gen = min(len(generic), 100 - n_spec)

            selected = specific[:n_spec] + generic[:n_gen]
            selected.sort(key=lambda x: (x[0], -x[1]))
            pruned_scored[sid] = selected

    return lines_read, pairs_scored, target_records, pruned_scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel worker processes")
    parser.add_argument("--final-cap", type=int, default=100, help="Final candidate cap per S1 entity (default: 100)")
    parser.add_argument("--max-key-fanout", type=int, default=15, help="Max S1 postings per key in frequency filter")
    parser.add_argument("--generic-fanout-threshold", type=int, default=5, help="Fanout threshold for generic token keys")
    parser.add_argument("--max-generic-worker", type=int, default=30, help="Max generic matches per S1 per worker")
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
        default=Path("experiments/EXP-002B-specificity-aware-cap100.txt"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experiments/EXP-002B-specificity-aware-cap100.json"),
    )
    args = parser.parse_args()

    overall_start = time.time()

    print("=" * 95, flush=True)
    print("BENCHMARK: SPECIFICITY-AWARE CANDIDATE SELECTION (FINAL CAP = 100)", flush=True)
    print(f"Workers: {args.num_workers} | Final Budget: {args.final_cap} candidates/S1", flush=True)
    print("Priority Order: 1. Exact Name | 2. PIN | 3. Rare Address | 4. Locality/Name | 5. Rare Token | 6. Generic Token", flush=True)
    print("=" * 95, flush=True)

    # 1. Load Validation Split
    with open(args.split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    val_s1_list: List[str] = split_data["val_s1_ids"]
    val_s1_set = set(val_s1_list)
    print(f"Loaded validation split: {len(val_s1_list):,} Source 1 entities.", flush=True)

    # 2. Parse Ground Truth
    val_ground_truth = parse_ground_truth(args.ground_truth, allowed_s1_ids=val_s1_set)
    total_true_matches = sum(len(v) for v in val_ground_truth.values())
    val_true_target_ids = {tid for tids in val_ground_truth.values() for tid in tids}
    print(f"Validation Ground Truth: {total_true_matches:,} true links ({len(val_true_target_ids):,} unique targets).", flush=True)

    # Load 542 baseline misses from analysis
    prev_misses_file = PROJECT_ROOT / "experiments" / "validation_missed_links_analysis.json"
    prev_misses_set = set()
    if prev_misses_file.exists():
        with open(prev_misses_file, "r", encoding="utf-8") as f:
            pm_data = json.load(f)
        for cat, examples in pm_data.get("category_examples", {}).items():
            for ex in examples:
                prev_misses_set.add((ex["s1_id"], ex["target_id"]))
        print(f"Tracking recovery across {len(prev_misses_set)} previously missed ground-truth links.", flush=True)

    # 3. Load & Preprocess Validation S1 Records
    val_s1_records: Dict[str, Tuple[str, str, str, str]] = {}
    val_s1_preprocessed: Dict[str, PreprocessedRecord] = {}
    query_idx_country: Dict[str, Dict[str, List[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))

    print(f"\nLoading and indexing validation S1 records from {args.source1}...", flush=True)
    t0_s1 = time.time()
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

    s1_index_time = time.time() - t0_s1
    total_raw_keys = sum(len(sub) for sub in query_idx_country.values())
    print(f"Indexed {len(val_s1_records):,} S1 entities ({total_raw_keys:,} keys) in {s1_index_time:.2f}s.", flush=True)

    # 4. Apply Frequency Filter & Flag Generic Keys
    filter_cfg = BlockingKeyFilterConfig(
        max_key_fanout=args.max_key_fanout,
        preserve_exact_name=True,
        preserve_pin_keys=True,
    )

    query_idx_clean: Dict[str, Dict[str, List[str]]] = {}
    generic_keys_set: Dict[str, Set[str]] = collections.defaultdict(set)

    for country, raw_idx in query_idx_country.items():
        c_count = sum(1 for sid, _, _, c in val_s1_records.values() if c == country)
        pruned_idx, audit = prune_high_frequency_keys(raw_idx, total_s1=c_count, config=filter_cfg)
        query_idx_clean[country] = pruned_idx

        for k, postings in pruned_idx.items():
            if k.startswith("tk_") and len(postings) >= args.generic_fanout_threshold:
                generic_keys_set[country].add(k)

        print(
            f"  [{country}] Pruned {audit['pruned_key_count']} keys. "
            f"Active keys: {len(pruned_idx):,}. Generic keys: {len(generic_keys_set[country]):,}.",
            flush=True,
        )

    # 5. Measure Line Counts in Search Pool
    s2_file = args.source2
    s3_file = args.source3

    def count_lines(fp):
        with open(fp, "r", encoding="utf-8-sig") as f:
            return sum(1 for _ in f) - 1

    s2_lines = count_lines(s2_file)
    s3_lines = count_lines(s3_file)
    total_pool_records = s2_lines + s3_lines

    # 6. Stream and Score Candidates in Parallel across 8 workers
    print(f"\nStreaming search pool ({total_pool_records:,} records) with {args.num_workers} parallel workers...", flush=True)
    stream_t0 = time.time()

    all_target_records = {}
    combined_scored_candidates = collections.defaultdict(list)

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
                    args.max_generic_worker,
                ))
                w_idx += 1

    with mp.Pool(processes=args.num_workers) as pool:
        results = pool.map(worker_stream_chunk_specificity, tasks)

    # Merge worker results
    total_pairs_scored_all = 0
    for lines_read, pairs_scored, target_recs, pruned_scored in results:
        total_pairs_scored_all += pairs_scored
        all_target_records.update(target_recs)
        for sid, pairs in pruned_scored.items():
            combined_scored_candidates[sid].extend(pairs)

    stream_time = time.time() - stream_t0
    print(f"Parallel streaming completed in {stream_time:.2f}s ({total_pairs_scored_all:,} pairs scored).", flush=True)

    # 7. Apply Specificity-Aware Selection & Final Cap of 100 per S1
    print(f"\nApplying Specificity-Aware Selection with Final Cap of {args.final_cap} per S1...", flush=True)
    t0_cap = time.time()
    final_capped_cands: Dict[str, Set[str]] = {}

    for sid in val_s1_list:
        raw_pairs = combined_scored_candidates.get(sid, [])
        if not raw_pairs:
            final_capped_cands[sid] = set()
            continue

        # Deduplicate across worker chunks by target ID:
        # Preserve best priority tier (min) and best score (max)
        best_by_tid: Dict[str, Tuple[int, float]] = {}
        for tier, score, tid in raw_pairs:
            if tid not in best_by_tid:
                best_by_tid[tid] = (tier, score)
            else:
                prev_tier, prev_score = best_by_tid[tid]
                best_by_tid[tid] = (min(prev_tier, tier), max(prev_score, score))

        deduped = [(tier, score, tid) for tid, (tier, score) in best_by_tid.items()]

        if len(deduped) <= args.final_cap:
            final_capped_cands[sid] = {tid for _, _, tid in deduped}
        else:
            # Separate into specific (tiers 1-5) and generic (tier 6)
            specific = [p for p in deduped if p[0] <= 5]
            generic = [p for p in deduped if p[0] == 6]

            # Specific candidates sorted by tier first, then score descending
            specific.sort(key=lambda x: (x[0], -x[1]))
            # Generic candidates sorted by score descending
            generic.sort(key=lambda x: -x[1])

            # Priority rule: Specific candidates take up to args.final_cap slots
            n_specific = min(len(specific), args.final_cap)
            # Remaining slots (if any) go to highest-scoring generic candidates
            n_generic = min(len(generic), args.final_cap - n_specific)

            selected = specific[:n_specific] + generic[:n_generic]
            final_capped_cands[sid] = {tid for _, _, tid in selected}

    cap_time = time.time() - t0_cap
    print(f"Final cap applied in {cap_time:.2f}s.", flush=True)

    peak_ram_mb = get_peak_ram_mb()
    total_runtime = time.time() - overall_start

    # 8. Evaluate Metrics on Validation Split
    s1_eval_records = {sid: (name, addr, country) for sid, name, addr, country in val_s1_records.values()}

    eval_result = evaluate_blocking(
        candidates_map=final_capped_cands,
        ground_truth=val_ground_truth,
        s1_records=s1_eval_records,
        target_records=all_target_records,
        total_pool_size=total_pool_records,
        strategy_id=f"SPECIFICITY-CAP{args.final_cap}",
        strategy_name=f"Specificity-Aware Candidate Selection (Cap {args.final_cap})",
        country_restricted=True,
        runtime_seconds=total_runtime,
        peak_memory_mb=peak_ram_mb,
    )

    pair_recall = eval_result.pair_recall
    s2_recall = eval_result.s2_recall
    s3_recall = eval_result.s3_recall
    total_cands = eval_result.total_candidate_pairs
    avg_cands_per_s1 = eval_result.avg_candidates_per_s1

    # Number of entities with zero candidates
    zero_cands_count = sum(1 for sid in val_s1_list if len(final_capped_cands.get(sid, set())) == 0)

    # Recovered links among 542 baseline misses
    recovered_links = []
    for s1_id, tid in prev_misses_set:
        if tid in final_capped_cands.get(s1_id, set()):
            recovered_links.append((s1_id, tid))
    num_recovered = len(recovered_links)

    # 9. Format Report and Save JSON
    report_lines = [
        "=" * 100,
        f"EXP-002B: SPECIFICITY-AWARE CANDIDATE SELECTION (FINAL CAP = {args.final_cap}) REPORT",
        "=" * 100,
        f"Validation Entities     : {len(val_s1_list):,} Source 1 entities",
        f"Search Pool Size        : {total_pool_records:,} records (Source 2 + Source 3)",
        f"Total Ground Truth      : {total_true_matches:,} true links",
        f"Final Candidate Budget  : Max {args.final_cap} candidates per S1 entity",
        "Priority Ranking Tiers  : 1. Exact normalized name (en_...)",
        "                          2. PIN (pin_...)",
        "                          3. Rare address-derived keys (ak_..., nword_...)",
        "                          4. Rare locality/name keys (loc_..., nl_...)",
        "                          5. Rare token keys (tk_... fanout < 5)",
        "                          6. Generic high-frequency token keys (low priority)",
        "",
        "BENCHMARK METRICS:",
        f"  Pair Recall                     : {pair_recall * 100.0:.2f}% ({eval_result.retrieved_true_matches:,} / {total_true_matches:,})",
        f"    Source 2 Recall               : {s2_recall * 100.0:.2f}% ({eval_result.s2_retrieved_count:,} / {eval_result.s2_true_count:,})",
        f"    Source 3 Recall               : {s3_recall * 100.0:.2f}% ({eval_result.s3_retrieved_count:,} / {eval_result.s3_true_count:,})",
        f"  Total Candidates                : {total_cands:,}",
        f"  Average Candidates / S1         : {avg_cands_per_s1:.2f}",
        f"  Previously Missed Recovered     : {num_recovered} / {len(prev_misses_set)}",
        f"  S1 Entities with 0 Candidates   : {zero_cands_count} ({zero_cands_count / len(val_s1_list) * 100.0:.2f}%)",
        f"  Runtime                         : {total_runtime:.2f}s ({total_runtime / 60:.2f} min)",
        f"  Peak RAM                        : {peak_ram_mb:.2f} MB",
        "=" * 100,
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text, flush=True)

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    json_data = {
        "strategy_id": f"SPECIFICITY-CAP{args.final_cap}",
        "final_cap": args.final_cap,
        "validation_entities": len(val_s1_list),
        "total_true_matches": total_true_matches,
        "pair_recall": pair_recall,
        "s2_recall": s2_recall,
        "s3_recall": s3_recall,
        "retrieved_true_matches": eval_result.retrieved_true_matches,
        "total_candidate_pairs": total_cands,
        "avg_candidates_per_s1": avg_cands_per_s1,
        "previously_missed_recovered": num_recovered,
        "zero_candidates_count": zero_cands_count,
        "runtime_seconds": total_runtime,
        "peak_ram_mb": peak_ram_mb,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"Results saved to {args.output_report} and {args.output_json}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
