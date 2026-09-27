#!/usr/bin/env python3
"""Benchmark high-frequency blocking key optimization on existing validation split.

Evaluates the high-frequency blocking-key optimization against the existing
EXP-002A broad blocker baseline (91.97% pair recall).

Measures:
- Pair recall (Overall, Source 2, Source 3)
- Candidate count and average candidates / S1
- Wall-clock runtime and streaming throughput
- Peak memory usage (tracemalloc)
- Recall delta versus the existing 91.97% baseline
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

BASELINE_PAIR_RECALL = 0.9196544276457883  # 91.97% (6,387 / 6,945) from EXP-002A
BASELINE_TOTAL_CANDIDATES = 3478343
BASELINE_AVG_CANDS_PER_S1 = 1738.30
BASELINE_RUNTIME_SECONDS = 395.34
BASELINE_PEAK_RAM_MB = 648.99


def worker_stream_chunk_optimized(task_args):
    """Worker function to stream and score a chunk of search pool records."""
    worker_id, file_path, start_line, end_line, query_idx_country, s1_preprocessed, val_true_target_ids = task_args

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

            matched_s1_set = set()

            # D: Exact name
            ten = normalize_name_conservative(tname)
            if ten:
                m = c_idx.get(f"en_{ten}")
                if m:
                    matched_s1_set.update(m)

            # D: Token name keys
            for tk in extract_token_blocking_keys(tname):
                m = c_idx.get(f"tk_{tk}")
                if m:
                    matched_s1_set.update(m)

            # D, E-A, E-C: Address keys (only if address non-empty)
            if taddr and taddr.strip():
                for ak in extract_address_keys(taddr):
                    m = c_idx.get(f"ak_{ak}")
                    if m:
                        matched_s1_set.update(m)
                for ak in extract_h1_relaxed_address_keys(taddr):
                    m = c_idx.get(ak)
                    if m:
                        matched_s1_set.update(m)
                for ck in extract_h3_name_locality_keys(tname, taddr):
                    m = c_idx.get(ck)
                    if m:
                        matched_s1_set.update(m)

            if not matched_s1_set:
                continue

            if tid in val_true_target_ids:
                target_records[tid] = (tname, taddr, tcountry)

            tgt_p = preprocess_record(tid, tname, taddr, tcountry)

            for s1_id in matched_s1_set:
                broad_counts[s1_id] += 1
                score = fast_score_candidate(s1_preprocessed[s1_id], tgt_p)
                if score >= 0.15:
                    scored_candidates[s1_id].append((score, tid))
                pairs_scored += 1

    elapsed = time.time() - t0
    fname = Path(file_path).name
    print(
        f"[Worker {worker_id} | {fname}] Processed {lines_read:,} lines in {elapsed:.1f}s. "
        f"Scored {pairs_scored:,} pairs ({lines_read/max(elapsed, 0.001):.0f} rows/s).",
        flush=True,
    )

    # Cap per worker per s1 to top 300 to match EXP-002A protocol
    pruned_scored = {}
    for s1_id, pairs in scored_candidates.items():
        if len(pairs) > 300:
            pairs.sort(key=lambda x: x[0], reverse=True)
            pruned_scored[s1_id] = pairs[:300]
        else:
            pruned_scored[s1_id] = pairs

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
        "--max-key-fanout",
        type=int,
        default=15,
        help="Max S1 postings per blocking key (e.g. 15 for ~0.75%% of 2,001 val entities). Set <=0 for baseline.",
    )
    parser.add_argument(
        "--max-key-fanout-ratio",
        type=float,
        default=None,
        help="Max ratio of S1 entities per blocking key (e.g. 0.0075 for 0.75%%)",
    )
    parser.add_argument(
        "--preserve-exact-name",
        action="store_true",
        default=True,
        help="Exempt exact name keys (en_...) from pruning",
    )
    parser.add_argument(
        "--preserve-pin-keys",
        action="store_true",
        default=True,
        help="Exempt postal PIN keys (pin_...) from pruning",
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
        default=Path("experiments/EXP-002B-optimized-blocking-benchmark.txt"),
    )
    args = parser.parse_args()

    overall_start = time.time()
    tracemalloc.start()

    print("=" * 90, flush=True)
    print("BENCHMARK: HIGH-FREQUENCY BLOCKING KEY OPTIMIZATION ON VALIDATION SPLIT", flush=True)
    print(f"Workers: {args.num_workers} | Platform: {sys.platform} | CPU cores: {os.cpu_count()}", flush=True)
    print("=" * 90, flush=True)

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

    # 4. Apply High-Frequency Blocking-Key Filter
    filter_cfg = None
    if args.max_key_fanout is not None and args.max_key_fanout > 0:
        filter_cfg = BlockingKeyFilterConfig(
            max_key_fanout=args.max_key_fanout,
            max_key_fanout_ratio=args.max_key_fanout_ratio,
            preserve_exact_name=args.preserve_exact_name,
            preserve_pin_keys=args.preserve_pin_keys,
        )
    elif args.max_key_fanout_ratio is not None and args.max_key_fanout_ratio > 0.0:
        filter_cfg = BlockingKeyFilterConfig(
            max_key_fanout=None,
            max_key_fanout_ratio=args.max_key_fanout_ratio,
            preserve_exact_name=args.preserve_exact_name,
            preserve_pin_keys=args.preserve_pin_keys,
        )

    query_idx_clean: Dict[str, Dict[str, List[str]]] = {}
    total_pruned_keys = 0
    total_pruned_postings = 0
    filter_audits = {}

    print("\nApplying High-Frequency Blocking Key Optimization...", flush=True)
    for country, raw_idx in query_idx_country.items():
        c_count = sum(1 for sid, _, _, c in val_s1_records.values() if c == country)
        pruned_idx, audit = prune_high_frequency_keys(raw_idx, total_s1=c_count, config=filter_cfg)
        query_idx_clean[country] = pruned_idx
        total_pruned_keys += audit["pruned_key_count"]
        total_pruned_postings += audit["pruned_postings"]
        filter_audits[country] = audit
        print(
            f"  {country:<15}: keys {audit['total_keys_before']:,} -> {audit['total_keys_after']:,} "
            f"(pruned {audit['pruned_key_count']} keys, {audit['pruned_postings']:,} postings; threshold={audit['effective_threshold']})",
            flush=True,
        )
        if audit["top_pruned_keys"]:
            print("    Top pruned keys:")
            for item in audit["top_pruned_keys"][:5]:
                print(f"      {item['key']:<30}: fanout {item['fanout']} ({item['ratio']*100:.1f}%)", flush=True)

    total_clean_keys = sum(len(sub) for sub in query_idx_clean.values())
    total_clean_postings = sum(sum(len(v) for v in sub.values()) for sub in query_idx_clean.values())
    print(f"Total keys after filter: {total_clean_keys:,} ({total_pruned_keys:,} keys pruned, {total_pruned_postings:,} postings removed).", flush=True)

    # 5. Build Chunk Tasks for Multi-Processing Stream
    s2_total_lines = 5034616
    s3_total_lines = 5285603
    total_pool_records_expected = s2_total_lines + s3_total_lines
    print(f"\nSearch pool: Source 2 = {s2_total_lines:,} rows, Source 3 = {s3_total_lines:,} rows (Total: {total_pool_records_expected:,})", flush=True)

    workers_s2 = max(1, args.num_workers // 2)
    workers_s3 = args.num_workers - workers_s2
    s2_chunk = (s2_total_lines + workers_s2 - 1) // workers_s2
    s3_chunk = (s3_total_lines + workers_s3 - 1) // workers_s3

    tasks = []
    worker_idx = 0
    # Source 2
    for i in range(workers_s2):
        start = i * s2_chunk
        end = min((i + 1) * s2_chunk, s2_total_lines)
        tasks.append((
            worker_idx,
            str(args.source2),
            start,
            end,
            query_idx_clean,
            val_s1_preprocessed,
            val_true_target_ids,
        ))
        worker_idx += 1

    # Source 3
    for i in range(workers_s3):
        start = i * s3_chunk
        end = min((i + 1) * s3_chunk, s3_total_lines)
        tasks.append((
            worker_idx,
            str(args.source3),
            start,
            end,
            query_idx_clean,
            val_s1_preprocessed,
            val_true_target_ids,
        ))
        worker_idx += 1

    print(f"Dispatching {len(tasks)} tasks across {args.num_workers} parallel workers...", flush=True)
    t0_stream = time.time()
    with mp.Pool(processes=args.num_workers) as pool:
        chunk_results = pool.map(worker_stream_chunk_optimized, tasks)

    stream_time = time.time() - t0_stream
    print(f"\nAll search pool chunks processed in {stream_time:.2f}s ({total_pool_records_expected/stream_time:.0f} rows/s)!", flush=True)

    # 6. Aggregate Worker Results
    total_pool_records = 0
    all_broad_counts = collections.defaultdict(int)
    all_target_records = {}
    combined_scored_candidates = collections.defaultdict(list)

    for lines_read, broad_counts, target_recs, scored_cands in chunk_results:
        total_pool_records += lines_read
        for sid, cnt in broad_counts.items():
            all_broad_counts[sid] += cnt
        all_target_records.update(target_recs)
        for sid, pairs in scored_cands.items():
            combined_scored_candidates[sid].extend(pairs)

    total_broad_pairs = sum(all_broad_counts.values())

    # 7. Sort candidates descending by score
    for sid in val_s1_list:
        combined_scored_candidates[sid].sort(key=lambda x: x[0], reverse=True)

    # 8. Evaluate Broad Candidates (D + E-A + E-C under optimized filter)
    broad_cands_map = {
        sid: {tid for _, tid in pairs}
        for sid, pairs in combined_scored_candidates.items()
    }

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem / (1024 * 1024)

    # Cache chunk results to scratch in case evaluation needs re-running
    scratch_dir = PROJECT_ROOT / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(scratch_dir / "benchmark_chunk_results.pkl", "wb") as f:
        pickle.dump((total_pool_records, all_broad_counts, all_target_records, broad_cands_map, stream_time, peak_mem_mb), f)

    s1_eval_records = {sid: (name, addr, country) for sid, name, addr, country in val_s1_records.values()}

    eval_result = evaluate_blocking(
        candidates_map=broad_cands_map,
        ground_truth=val_ground_truth,
        s1_records=s1_eval_records,
        target_records=all_target_records,
        total_pool_size=total_pool_records,
        strategy_id="OPT-BLOCK-CAP15",
        strategy_name=f"D + E-A + E-C with High-Frequency Key Cap ({args.max_key_fanout})",
        country_restricted=True,
        runtime_seconds=stream_time,
        peak_memory_mb=peak_mem_mb,
    )

    pair_recall = eval_result.pair_recall
    total_candidates = eval_result.total_candidate_pairs
    avg_cands_per_s1 = eval_result.avg_candidates_per_s1
    recall_change_pct = (pair_recall - BASELINE_PAIR_RECALL) * 100.0

    print("\n" + "=" * 90, flush=True)
    print("BENCHMARK RESULTS SUMMARY", flush=True)
    print("=" * 90, flush=True)
    print(f"Pair Recall                   : {pair_recall*100:.2f}% ({eval_result.retrieved_true_matches:,} / {eval_result.total_true_matches:,})", flush=True)
    print(f"  Source 2 Recall             : {eval_result.s2_recall*100:.2f}% ({eval_result.s2_retrieved_count:,} / {eval_result.s2_true_count:,})", flush=True)
    print(f"  Source 3 Recall             : {eval_result.s3_recall*100:.2f}% ({eval_result.s3_retrieved_count:,} / {eval_result.s3_true_count:,})", flush=True)
    print(f"Candidate Count               : {total_candidates:,} pairs", flush=True)
    print(f"Average Candidates / S1       : {avg_cands_per_s1:.2f} (Median: {eval_result.median_candidates_per_s1:.0f}, P95: {eval_result.p95_candidates_per_s1:.0f}, Max: {eval_result.max_candidates_per_s1:,})", flush=True)
    print(f"Runtime (Stream & Score)      : {stream_time:.2f}s", flush=True)
    print(f"Peak RAM Usage                : {peak_mem_mb:.2f} MB", flush=True)
    print(f"Recall Change vs 91.97% Base  : {recall_change_pct:+.2f}%", flush=True)
    print("=" * 90, flush=True)

    # Comparative Table
    print("\nCOMPARISON WITH EXISTING 91.97% BASELINE (EXP-002A):", flush=True)
    print(f"{'Metric':<32} | {'Existing Baseline (EXP-002A)':<30} | {'Optimized Filter':<20} | {'Delta':<15}", flush=True)
    print("-" * 105, flush=True)
    print(f"{'Pair Recall':<32} | {BASELINE_PAIR_RECALL*100:6.2f}% ({6387:,}/{6945:,}){'':<13} | {pair_recall*100:6.2f}% ({eval_result.retrieved_true_matches:,}/{eval_result.total_true_matches:,}) | {recall_change_pct:+6.2f}%", flush=True)
    print(f"{'Source 2 Recall':<32} | {0.9152*100:6.2f}% ({3053:,}/{3336:,}){'':<13} | {eval_result.s2_recall*100:6.2f}% ({eval_result.s2_retrieved_count:,}/{eval_result.s2_true_count:,}) | {(eval_result.s2_recall-0.9152)*100:+6.2f}%", flush=True)
    print(f"{'Source 3 Recall':<32} | {0.9238*100:6.2f}% ({3334:,}/{3609:,}){'':<13} | {eval_result.s3_recall*100:6.2f}% ({eval_result.s3_retrieved_count:,}/{eval_result.s3_true_count:,}) | {(eval_result.s3_recall-0.9238)*100:+6.2f}%", flush=True)
    print(f"{'Candidate Count':<32} | {BASELINE_TOTAL_CANDIDATES:<30,d} | {total_candidates:<20,d} | {total_candidates - BASELINE_TOTAL_CANDIDATES:+15,d}", flush=True)
    print(f"{'Avg Candidates / S1':<32} | {BASELINE_AVG_CANDS_PER_S1:<30.2f} | {avg_cands_per_s1:<20.2f} | {avg_cands_per_s1 - BASELINE_AVG_CANDS_PER_S1:+15.2f}", flush=True)
    print(f"{'Streaming & Scoring Runtime':<32} | {BASELINE_RUNTIME_SECONDS:<28.2f}s | {stream_time:<18.2f}s | {stream_time - BASELINE_RUNTIME_SECONDS:+13.2f}s", flush=True)
    print(f"{'Peak RAM':<32} | {BASELINE_PEAK_RAM_MB:<27.2f}MB | {peak_mem_mb:<17.2f}MB | {peak_mem_mb - BASELINE_PEAK_RAM_MB:+12.2f}MB", flush=True)
    print("-" * 105, flush=True)

    # Save benchmark report artifact
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "=" * 105,
        "EXP-002B: HIGH-FREQUENCY BLOCKING KEY OPTIMIZATION BENCHMARK REPORT",
        "=" * 105,
        f"Validation Entities     : {len(val_s1_list):,} Source 1 entities",
        f"Search Pool Size        : {total_pool_records:,} records (Source 2 + Source 3)",
        f"Total Ground Truth      : {total_true_matches:,} true links",
        f"Filter Configuration    : max_key_fanout={args.max_key_fanout}, preserve_exact_name={args.preserve_exact_name}, preserve_pin_keys={args.preserve_pin_keys}",
        f"Pruned Keys             : {total_pruned_keys:,} keys, {total_pruned_postings:,} postings removed",
        "",
        "BENCHMARK METRICS:",
        f"  Pair Recall           : {pair_recall*100:.2f}% ({eval_result.retrieved_true_matches:,} / {eval_result.total_true_matches:,})",
        f"    Source 2 Recall     : {eval_result.s2_recall*100:.2f}% ({eval_result.s2_retrieved_count:,} / {eval_result.s2_true_count:,})",
        f"    Source 3 Recall     : {eval_result.s3_recall*100:.2f}% ({eval_result.s3_retrieved_count:,} / {eval_result.s3_true_count:,})",
        f"  Candidate Count       : {total_candidates:,}",
        f"  Avg Candidates / S1   : {avg_cands_per_s1:.2f} (Median: {eval_result.median_candidates_per_s1:.0f}, P95: {eval_result.p95_candidates_per_s1:.0f}, Max: {eval_result.max_candidates_per_s1:,})",
        f"  Runtime               : {stream_time:.2f}s",
        f"  Peak RAM              : {peak_mem_mb:.2f} MB",
        f"  Recall Change vs Base : {recall_change_pct:+.2f}% (versus 91.97% baseline)",
        "=" * 105,
    ]
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nSaved benchmark report to: {args.output_report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
