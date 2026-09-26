#!/usr/bin/env python3
"""Validation benchmark with anti-crowding priority protection and GLOBAL per-S1 cap of 1,000.

Evaluates candidate volume reduction for full-test inference readiness while
preserving the anti-crowding priority protection for:
1. exact-name matches (en_...)
2. PIN matches (pin_...)
3. address-derived matches (ak_..., nword_..., loc_..., nloc_...)

Then applies a GLOBAL per-S1 candidate cap of 1,000 after priority protection.
Evaluates on the exact 2,001-entity validation split against the 10,320,219 search pool.
"""

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import time
import tracemalloc
from pathlib import Path
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

PREV_PIPELINE_PAIR_RECALL = 0.9219582433405327  # 92.20% (6,403 / 6,945)
PREV_PIPELINE_TOTAL_CANDS = 3345467
PREV_PIPELINE_AVG_CANDS = 1671.90
EXP002B_UNFILTERED_PAIR_RECALL = 0.9196544276457883  # 91.97%


def worker_stream_chunk_priority(task_args):
    """Worker streaming function with priority tracking."""
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
    broad_counts = collections.defaultdict(int)
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

            matched_s1_high_spec = set()
            matched_s1_generic = set()

            # 1. Exact name matches (high specificity / exempt)
            ten = normalize_name_conservative(tname)
            if ten:
                m = c_idx.get(f"en_{ten}")
                if m:
                    matched_s1_high_spec.update(m)

            # 2. Token name keys (high specificity if rare, generic if in generic_keys_set)
            gen_keys = generic_keys_set.get(tcountry, set())
            for tk in extract_token_blocking_keys(tname):
                k = f"tk_{tk}"
                m = c_idx.get(k)
                if m:
                    if k in gen_keys:
                        matched_s1_generic.update(m)
                    else:
                        matched_s1_high_spec.update(m)

            # 3. Address keys (exact PIN and address-derived keys are high specificity)
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

            all_matched_s1 = matched_s1_high_spec | matched_s1_generic
            if not all_matched_s1:
                continue

            if tid in val_true_target_ids:
                target_records[tid] = (tname, taddr, tcountry)

            tgt_p = preprocess_record(tid, tname, taddr, tcountry)

            for s1_id in all_matched_s1:
                broad_counts[s1_id] += 1
                score = fast_score_candidate(s1_preprocessed[s1_id], tgt_p)
                if score >= 0.15:
                    is_high_spec = (s1_id in matched_s1_high_spec)
                    scored_candidates[s1_id].append((score, is_high_spec, tid))
                pairs_scored += 1

    elapsed = time.time() - t0
    fname = Path(file_path).name
    print(
        f"[Worker {worker_id} | {fname}] Processed {lines_read:,} lines in {elapsed:.1f}s. "
        f"Scored {pairs_scored:,} pairs ({lines_read/max(elapsed, 0.001):.0f} rows/s).",
        flush=True,
    )

    # Anti-crowding candidate compression per worker
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

    return lines_read, broad_counts, target_records, pruned_scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel worker processes (default: 8)",
    )
    parser.add_argument(
        "--global-cap",
        type=int,
        default=1000,
        help="Global candidate cap per S1 entity after priority protection (default: 1000)",
    )
    parser.add_argument(
        "--max-key-fanout",
        type=int,
        default=15,
        help="Max S1 postings per blocking key in frequency filter (default: 15)",
    )
    parser.add_argument(
        "--max-generic-per-s1",
        type=int,
        default=150,
        help="Max generic matches per S1 per worker (default: 150)",
    )
    parser.add_argument(
        "--generic-fanout-threshold",
        type=int,
        default=5,
        help="Fanout threshold at or above which token keys are classified as generic (default: 5)",
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
        default=Path("experiments/EXP-002B-global-cap1000-benchmark.txt"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experiments/EXP-002B-global-cap1000-benchmark.json"),
    )
    args = parser.parse_args()

    overall_start = time.time()
    tracemalloc.start()

    print("=" * 95, flush=True)
    print("BENCHMARK: ANTI-CROWDING PRIORITY PROTECTION + GLOBAL CAP OF 1,000 PER S1", flush=True)
    print(f"Workers: {args.num_workers} | Platform: {sys.platform} | CPU cores: {os.cpu_count()}", flush=True)
    print(f"Policy: Global Cap = {args.global_cap} | Priority Protection: Exact Name, PIN, Address Matches", flush=True)
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

    # Load 542 baseline misses from analysis to track recovery
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
    total_raw_postings = sum(sum(len(v) for v in sub.values()) for sub in query_idx_country.values())
    print(f"Indexed {len(val_s1_records):,} S1 entities ({total_raw_keys:,} keys, {total_raw_postings:,} postings) in {s1_index_time:.2f}s.", flush=True)

    # 4. Apply Frequency Filter
    filter_cfg = BlockingKeyFilterConfig(
        max_key_fanout=args.max_key_fanout,
        preserve_exact_name=True,
        preserve_pin_keys=True,
    )

    query_idx_clean: Dict[str, Dict[str, List[str]]] = {}
    generic_keys_set: Dict[str, Set[str]] = collections.defaultdict(set)

    print("\nApplying Frequency Filter and Identifying Generic Keys...", flush=True)
    for country, raw_idx in query_idx_country.items():
        c_count = sum(1 for sid, _, _, c in val_s1_records.values() if c == country)
        pruned_idx, audit = prune_high_frequency_keys(raw_idx, total_s1=c_count, config=filter_cfg)
        query_idx_clean[country] = pruned_idx

        # Flag generic keys (fanout >= generic_fanout_threshold)
        for k, postings in pruned_idx.items():
            if k.startswith("tk_") and len(postings) >= args.generic_fanout_threshold:
                generic_keys_set[country].add(k)

        print(
            f"  [{country}] Pruned {audit['pruned_key_count']} keys ({audit['pruned_postings']} postings). "
            f"Active keys: {len(pruned_idx):,}. Generic keys: {len(generic_keys_set[country]):,}.",
            flush=True,
        )

    # 5. Measure Line Counts
    s2_file = args.source2
    s3_file = args.source3

    def count_lines(fp):
        with open(fp, "r", encoding="utf-8-sig") as f:
            return sum(1 for _ in f) - 1

    s2_lines = count_lines(s2_file)
    s3_lines = count_lines(s3_file)
    total_pool_records = s2_lines + s3_lines

    # 6. Stream and Score Candidates in Parallel
    print(f"\nStreaming search pool with {args.num_workers} parallel workers...", flush=True)
    stream_t0 = time.time()

    all_broad_counts = collections.defaultdict(int)
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
                    args.max_generic_per_s1,
                ))
                w_idx += 1

    with mp.Pool(processes=args.num_workers) as pool:
        results = pool.map(worker_stream_chunk_priority, tasks)

    # Merge worker results
    for _, broad_counts, target_recs, pruned_scored in results:
        for sid, cnt in broad_counts.items():
            all_broad_counts[sid] += cnt
        all_target_records.update(target_recs)
        for sid, pairs in pruned_scored.items():
            combined_scored_candidates[sid].extend(pairs)

    stream_time = time.time() - stream_t0
    print(f"Parallel streaming completed in {stream_time:.2f}s.", flush=True)

    # 7. Apply GLOBAL Per-S1 Candidate Cap of 1,000 with Priority Protection
    print(f"\nApplying GLOBAL per-S1 candidate cap of {args.global_cap:,} with priority protection...", flush=True)
    t0_cap = time.time()
    global_capped_cands: Dict[str, Set[str]] = {}

    for sid in val_s1_list:
        raw_pairs = combined_scored_candidates.get(sid, [])
        if not raw_pairs:
            global_capped_cands[sid] = set()
            continue

        # De-duplicate by target ID across chunks, preserving best score and high-spec flag
        best_by_tid: Dict[str, Tuple[float, bool]] = {}
        for score, is_high_spec, tid in raw_pairs:
            if tid not in best_by_tid:
                best_by_tid[tid] = (score, is_high_spec)
            else:
                prev_score, prev_spec = best_by_tid[tid]
                best_by_tid[tid] = (max(prev_score, score), prev_spec or is_high_spec)

        deduped_pairs = [
            (score, is_high_spec, tid)
            for tid, (score, is_high_spec) in best_by_tid.items()
        ]

        if len(deduped_pairs) <= args.global_cap:
            global_capped_cands[sid] = {tid for _, _, tid in deduped_pairs}
        else:
            high_spec = [p for p in deduped_pairs if p[1]]
            generic = [p for p in deduped_pairs if not p[1]]

            high_spec.sort(key=lambda x: x[0], reverse=True)
            generic.sort(key=lambda x: x[0], reverse=True)

            # Guaranteed priority protection for high-spec matches:
            # High-spec matches take up to args.global_cap slots
            n_high_spec = min(len(high_spec), args.global_cap)
            # Remaining slots go to highest scoring generic candidates
            n_generic = min(len(generic), args.global_cap - n_high_spec)

            selected = high_spec[:n_high_spec] + generic[:n_generic]
            global_capped_cands[sid] = {tid for _, _, tid in selected}

    cap_time = time.time() - t0_cap
    print(f"Global cap applied in {cap_time:.2f}s.", flush=True)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem / (1024 * 1024)

    # 8. Evaluate Metrics on Validation Split
    s1_eval_records = {sid: (name, addr, country) for sid, name, addr, country in val_s1_records.values()}

    eval_result = evaluate_blocking(
        candidates_map=global_capped_cands,
        ground_truth=val_ground_truth,
        s1_records=s1_eval_records,
        target_records=all_target_records,
        total_pool_size=total_pool_records,
        strategy_id=f"OPT-GLOBAL-CAP{args.global_cap}",
        strategy_name=f"Anti-Crowding Priority Protection + Global Cap of {args.global_cap}",
        country_restricted=True,
        runtime_seconds=stream_time + cap_time,
        peak_memory_mb=peak_mem_mb,
    )

    pair_recall = eval_result.pair_recall
    s2_recall = eval_result.s2_recall
    s3_recall = eval_result.s3_recall
    total_cands = eval_result.total_candidate_pairs
    avg_cands_per_s1 = eval_result.avg_candidates_per_s1

    # Cap metrics: number hitting the global cap
    cand_counts = [len(global_capped_cands.get(sid, set())) for sid in val_s1_list]
    num_hitting_cap = sum(1 for cnt in cand_counts if cnt >= args.global_cap)
    pct_hitting_cap = num_hitting_cap / len(val_s1_list) * 100.0

    # Recovered links among 542 baseline misses
    recovered_links = []
    for s1_id, tid in prev_misses_set:
        if tid in global_capped_cands.get(s1_id, set()):
            recovered_links.append((s1_id, tid))
    num_recovered = len(recovered_links)

    # 9. Format Report
    report_lines = [
        "=" * 100,
        f"EXP-002B: ANTI-CROWDING PRIORITY PROTECTION + GLOBAL CAP OF {args.global_cap:,} REPORT",
        "=" * 100,
        f"Validation Entities     : {len(val_s1_list):,} Source 1 entities",
        f"Search Pool Size        : {total_pool_records:,} records (Source 2 + Source 3)",
        f"Total Ground Truth      : {total_true_matches:,} true links",
        f"Global Cap Policy       : Max {args.global_cap:,} candidates per S1 entity",
        "Priority Protected Keys : 1. exact-name matches (en_...)",
        "                          2. PIN matches (pin_...)",
        "                          3. address-derived matches (ak_..., nword_..., loc_..., nloc_...)",
        "",
        "BENCHMARK METRICS:",
        f"  Pair Recall                     : {pair_recall * 100.0:.2f}% ({eval_result.retrieved_true_matches:,} / {total_true_matches:,})",
        f"    Source 2 Recall               : {s2_recall * 100.0:.2f}% ({eval_result.s2_retrieved_count:,} / {eval_result.s2_true_count:,})",
        f"    Source 3 Recall               : {s3_recall * 100.0:.2f}% ({eval_result.s3_retrieved_count:,} / {eval_result.s3_true_count:,})",
        f"  Total Candidates                : {total_cands:,} (vs {PREV_PIPELINE_TOTAL_CANDS:,} in 92.20% pipeline)",
        f"  Avg Candidates / S1             : {avg_cands_per_s1:.2f} (vs {PREV_PIPELINE_AVG_CANDS:.2f} in 92.20% pipeline)",
        f"  Entities Hitting Cap ({args.global_cap:,})    : {num_hitting_cap:,} ({pct_hitting_cap:.2f}%)",
        f"  Previously Missed Links Recovered: {num_recovered:,} links",
        f"  Recall Delta vs 92.20% Pipeline : {(pair_recall - PREV_PIPELINE_PAIR_RECALL) * 100.0:+.2f}%",
        f"  Recall Delta vs 91.97% Baseline : {(pair_recall - EXP002B_UNFILTERED_PAIR_RECALL) * 100.0:+.2f}%",
        f"  Runtime                         : {stream_time + cap_time:.2f}s",
        f"  Peak RAM                        : {peak_mem_mb:.2f} MB",
        "=" * 100,
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text, flush=True)

    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    benchmark_json = {
        "experiment": f"EXP-002B: Priority Protection + Global Cap {args.global_cap}",
        "validation_entities": len(val_s1_list),
        "total_true_matches": total_true_matches,
        "global_cap": args.global_cap,
        "pair_recall": pair_recall,
        "s2_recall": s2_recall,
        "s3_recall": s3_recall,
        "retrieved_true_matches": eval_result.retrieved_true_matches,
        "total_candidates": total_cands,
        "avg_candidates_per_s1": avg_cands_per_s1,
        "num_hitting_cap": num_hitting_cap,
        "pct_hitting_cap": pct_hitting_cap,
        "recovered_previously_missed_links": num_recovered,
        "delta_recall_vs_92_20_pipeline": (pair_recall - PREV_PIPELINE_PAIR_RECALL) * 100.0,
        "delta_recall_vs_91_97_baseline": (pair_recall - EXP002B_UNFILTERED_PAIR_RECALL) * 100.0,
        "runtime_seconds": stream_time + cap_time,
        "peak_ram_mb": peak_mem_mb,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_json, f, indent=2)

    print(f"\nSaved report to {args.output_report}")
    print(f"Saved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
