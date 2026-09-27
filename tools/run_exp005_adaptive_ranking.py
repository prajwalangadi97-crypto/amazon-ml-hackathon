#!/usr/bin/env python3
"""EXP-005 Phase 1: Adaptive Candidate Ranking Benchmark.

Tests whether multi-channel agreement, key specificity, and modality-balanced
retrieval evidence can improve candidate ranking within a strict cap of 100 per S1,
recovering candidate-cap losses without candidate explosion.

Constraints:
- 100% compliant with competition rules (zero external data, zero internet).
- Frozen V3 model (artifacts/xgb_matcher_exp004b.json) and decision layer (EXP-003B).
- Validates on canonical 2,001 validation split against V3 Macro F0.5 = 0.831259.
- Final candidate budget strictly <= 100 candidates per S1 entity.
- Saves experiment records to experiments/EXP-005A-adaptive-ranking.json.
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
from rapidfuzz import fuzz
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
    preprocess_record,
    PreprocessedRecord,
)
from src.matching.decision_layer import (
    DEFAULT_V3_DECISION_CONFIG,
    filter_entity_candidates,
)
from tools.run_recovery_experiments import (
    extract_h1_relaxed_address_keys,
    extract_h3_name_locality_keys,
)

BASE_FEATURE_NAMES = [
    "country_eq", "source_type", "s1_name_empty", "target_name_empty",
    "s1_addr_empty", "target_addr_empty", "name_exact_eq", "name_fuzz_ratio",
    "name_fuzz_wratio", "name_token_set_ratio", "name_token_sort_ratio",
    "name_prefix_similarity", "name_len_diff", "name_len_ratio",
    "addr_fuzz_ratio", "addr_token_similarity", "numeric_token_overlap",
    "house_number_agreement", "postal_pin_overlap", "addr_len_diff",
    "cross_field_name_loc", "cheap_candidate_score",
]

METADATA_FEATURE_NAMES = [
    "channel_agreement_count",
    "retrieval_priority_tier",
    "reciprocal_retrieval_rank",
    "s1_candidate_pool_size",
]

EXP004B_FEATURE_NAMES = BASE_FEATURE_NAMES + METADATA_FEATURE_NAMES


def fast_score_candidate_detailed(
    s1: PreprocessedRecord,
    tgt: PreprocessedRecord,
) -> Tuple[float, float, float, bool, bool]:
    """Inline RapidFuzz scoring returning detailed modality evidence."""
    if s1.country != tgt.country:
        return 0.0, 0.0, 0.0, False, False

    # 1. Name Component
    if not s1.name_norm or not tgt.name_norm:
        s_name = 0.0
    elif s1.name_norm == tgt.name_norm:
        s_name = 1.0
    else:
        wratio = fuzz.WRatio(s1.name_norm, tgt.name_norm) / 100.0
        tset = fuzz.token_set_ratio(s1.name_norm, tgt.name_norm) / 100.0
        fratio = fuzz.ratio(s1.name_norm, tgt.name_norm) / 100.0
        pfx_len = 0
        for c1, c2 in zip(s1.name_norm, tgt.name_norm):
            if c1 == c2:
                pfx_len += 1
            else:
                break
        pfx_sim = pfx_len / max(len(s1.name_norm), len(tgt.name_norm), 1)
        s_name = 0.40 * wratio + 0.30 * tset + 0.20 * fratio + 0.10 * pfx_sim

    # 2. Address Component
    has_pin_match = False
    has_house_match = False
    s_addr = 0.0

    if s1.addr_norm and tgt.addr_norm:
        addr_ratio = fuzz.ratio(s1.addr_norm, tgt.addr_norm) / 100.0
        union_w = s1.addr_words_set | tgt.addr_words_set
        inter_w = s1.addr_words_set & tgt.addr_words_set
        addr_tok_sim = len(inter_w) / max(len(union_w), 1) if union_w else 0.0
        s_addr_sim = 0.50 * addr_tok_sim + 0.50 * addr_ratio

        if s1.pin and tgt.pin:
            has_pin_match = (s1.pin == tgt.pin)
            pin_score = 1.0 if has_pin_match else 0.0
        else:
            pin_score = s_addr_sim

        if s1.first_num and tgt.first_num:
            has_house_match = (s1.first_num == tgt.first_num)
            house_score = 1.0 if has_house_match else 0.0
        else:
            house_score = s_addr_sim

        if s1.nums_set and tgt.nums_set:
            union_n = s1.nums_set | tgt.nums_set
            inter_n = s1.nums_set & tgt.nums_set
            num_score = len(inter_n) / max(len(union_n), 1)
        else:
            num_score = s_addr_sim

        s_addr = (
            0.40 * s_addr_sim
            + 0.25 * pin_score
            + 0.20 * house_score
            + 0.15 * num_score
        )

        if s1.pin and tgt.pin and not has_pin_match:
            s_addr *= 0.60

        score = 0.55 * s_name + 0.45 * s_addr
    else:
        score = 0.90 * s_name

    cross_field = False
    if s1.name_norm and tgt.addr_words_set:
        for t in s1.name_norm.split():
            if len(t) >= 4 and t in tgt.addr_words_set:
                cross_field = True
                break
    if not cross_field and tgt.name_norm and s1.addr_words_set:
        for t in tgt.name_norm.split():
            if len(t) >= 4 and t in s1.addr_words_set:
                cross_field = True
                break

    if cross_field:
        score = min(1.0, score + 0.05)

    composite_score = max(0.0, min(1.0, score))
    return composite_score, s_name, s_addr, has_pin_match, has_house_match


def worker_stream_chunk_exp005(task_args):
    """Worker streaming function with multi-channel and specificity signal tracking."""
    (
        worker_id,
        file_path,
        start_line,
        end_line,
        query_idx_country_val,
        s1_prep_val,
        key_fanouts_val,
        generic_keys_set_val,
        max_worker_candidates_per_s1,
    ) = task_args

    scored_val = collections.defaultdict(list)
    lines_read = 0

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

            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            tid, tname, taddr = parts[0], parts[1], parts[2]

            c_idx_val = query_idx_country_val.get(tcountry)
            if not c_idx_val:
                continue

            c_fanouts = key_fanouts_val.get(tcountry, {})
            gen_keys = generic_keys_set_val.get(tcountry, set())

            # Track matched S1 entities and retrieval signals
            s1_hits: Dict[str, Dict[str, Any]] = {}

            def record_hit(sid, ch, tier, k):
                if sid not in s1_hits:
                    s1_hits[sid] = {
                        "channels": {ch},
                        "best_tier": tier,
                        "matched_keys": [k],
                        "spec_sum": 1.0 / max(c_fanouts.get(k, 15), 1),
                    }
                else:
                    h = s1_hits[sid]
                    h["channels"].add(ch)
                    if tier < h["best_tier"]:
                        h["best_tier"] = tier
                    h["matched_keys"].append(k)
                    h["spec_sum"] += 1.0 / max(c_fanouts.get(k, 15), 1)

            # Channel 1: Exact Name (Tier 1)
            ten = normalize_name_conservative(tname)
            if ten:
                k = f"en_{ten}"
                m = c_idx_val.get(k)
                if m:
                    for sid in m:
                        record_hit(sid, 1, 1, k)

            # Channels 2, 3, 4: Address keys
            if taddr and taddr.strip():
                for ak in extract_address_keys(taddr):
                    k = f"ak_{ak}"
                    m = c_idx_val.get(k)
                    if m:
                        if "pin_" in ak:
                            for sid in m:
                                record_hit(sid, 2, 2, k)
                        else:
                            for sid in m:
                                record_hit(sid, 3, 3, k)

                for ak in extract_h1_relaxed_address_keys(taddr):
                    m = c_idx_val.get(ak)
                    if m:
                        if ak.startswith("pin_"):
                            for sid in m:
                                record_hit(sid, 2, 2, ak)
                        elif ak.startswith("nword_"):
                            for sid in m:
                                record_hit(sid, 3, 3, ak)
                        elif ak.startswith("loc_"):
                            for sid in m:
                                record_hit(sid, 4, 4, ak)
                        else:
                            for sid in m:
                                record_hit(sid, 3, 3, ak)

                for ck in extract_h3_name_locality_keys(tname, taddr):
                    m = c_idx_val.get(ck)
                    if m:
                        for sid in m:
                            record_hit(sid, 4, 4, ck)

            # Channel 5: Token Keys (Tier 5 if rare, Tier 6 if generic)
            for tk in extract_token_blocking_keys(tname):
                k = f"tk_{tk}"
                m = c_idx_val.get(k)
                if m:
                    tier = 6 if k in gen_keys else 5
                    for sid in m:
                        record_hit(sid, 5, tier, k)

            if not s1_hits:
                continue

            tgt_p = preprocess_record(tid, tname, taddr, tcountry)

            for sid, h in s1_hits.items():
                s1_p = s1_prep_val[sid]
                score, s_name, s_addr, has_pin, has_house = fast_score_candidate_detailed(s1_p, tgt_p)

                # Keep candidate if:
                # 1. fast_score >= 0.12
                # 2. OR address match evidence is high (s_addr >= 0.50 and key_count >= 2) to catch cross-lingual Indic names
                # 3. OR multi-channel consensus (channels >= 2)
                ch_count = len(h["channels"])
                key_cnt = len(h["matched_keys"])

                if score >= 0.12 or (s_addr >= 0.50 and key_cnt >= 2) or ch_count >= 2:
                    entry = {
                        "tid": tid,
                        "best_tier": h["best_tier"],
                        "ch_count": ch_count,
                        "key_count": key_cnt,
                        "spec_sum": h["spec_sum"],
                        "score": score,
                        "s_name": s_name,
                        "s_addr": s_addr,
                        "has_pin": has_pin,
                        "has_house": has_house,
                    }
                    scored_val[sid].append(entry)

    # Worker-level pruning to max_worker_candidates_per_s1
    pruned = {}
    for sid, items in scored_val.items():
        if len(items) <= max_worker_candidates_per_s1:
            pruned[sid] = items
        else:
            # Sort by a comprehensive preliminary ranking:
            # Prioritize tier, multi-channel consensus, and max(score, s_addr)
            items.sort(
                key=lambda x: (
                    x["best_tier"],
                    -x["ch_count"],
                    -max(x["score"], x["s_addr"]),
                    -x["spec_sum"],
                )
            )
            pruned[sid] = items[:max_worker_candidates_per_s1]

    return lines_read, pruned


def deduplicate_candidate_pool(combined_dict: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Deduplicate candidates per S1 preserving best signals across chunks."""
    deduped_pool = {}
    for sid, items in combined_dict.items():
        best_by_tid: Dict[str, Dict[str, Any]] = {}
        for it in items:
            tid = it["tid"]
            if tid not in best_by_tid:
                best_by_tid[tid] = it
            else:
                prev = best_by_tid[tid]
                best_by_tid[tid] = {
                    "tid": tid,
                    "best_tier": min(prev["best_tier"], it["best_tier"]),
                    "ch_count": max(prev["ch_count"], it["ch_count"]),
                    "key_count": max(prev["key_count"], it["key_count"]),
                    "spec_sum": max(prev["spec_sum"], it["spec_sum"]),
                    "score": max(prev["score"], it["score"]),
                    "s_name": max(prev["s_name"], it["s_name"]),
                    "s_addr": max(prev["s_addr"], it["s_addr"]),
                    "has_pin": prev["has_pin"] or it["has_pin"],
                    "has_house": prev["has_house"] or it["has_house"],
                }
        deduped_pool[sid] = list(best_by_tid.values())
    return deduped_pool


def rank_candidates_v3_baseline(
    candidates: List[Dict[str, Any]],
    cap: int = 100,
) -> List[Tuple[str, Dict[str, float]]]:
    """Baseline V3 Ranking: Tier ASC, then -fast_score ASC, generic tokens separated."""
    specific = [c for c in candidates if c["best_tier"] <= 5]
    generic = [c for c in candidates if c["best_tier"] == 6]

    specific.sort(key=lambda x: (x["best_tier"], -x["score"]))
    generic.sort(key=lambda x: -x["score"])

    n_specific = min(len(specific), cap)
    n_generic = min(len(generic), cap - n_specific)
    selected = specific[:n_specific] + generic[:n_generic]

    pool_size = float(len(selected))
    result = []
    for rank_idx, c in enumerate(selected):
        meta = {
            "channel_agreement_count": float(c["ch_count"]),
            "retrieval_priority_tier": float(c["best_tier"]),
            "reciprocal_retrieval_rank": 1.0 / (rank_idx + 1.0),
            "s1_candidate_pool_size": pool_size,
        }
        result.append((c["tid"], meta))
    return result


def rank_candidates_adaptive_consensus(
    candidates: List[Dict[str, Any]],
    cap: int = 100,
) -> List[Tuple[str, Dict[str, float]]]:
    """Strategy A: Multi-Channel Consensus + Modality-Aware Specificity Ranking.

    Key Principles:
    1. Multi-channel agreement (ch_count >= 2) provides strong corroborating evidence.
    2. High physical address agreement (s_addr >= 0.60) is not suppressed when s_name is low.
    3. Highly specific keys (spec_sum) outrank generic keys.
    """
    def ranking_key(c: Dict[str, Any]):
        # Effective similarity: protect address matches for cross-lingual records
        effective_sim = max(c["score"], c["s_addr"] if c["key_count"] >= 2 else c["score"])

        # Composite priority score (higher is better)
        # Tier bonus: Tier 1 (+5.0), Tier 2 (+4.0), Tier 3 (+3.0), Tier 4 (+2.0), Tier 5 (+1.0), Tier 6 (+0.0)
        tier_weight = max(0.0, 6.0 - c["best_tier"])

        # Channel agreement bonus: +1.5 per additional channel
        channel_bonus = (c["ch_count"] - 1.0) * 1.5

        # Key specificity bonus: up to +1.0
        spec_bonus = min(1.0, c["spec_sum"] / 2.0)

        # High address corroboration bonus: +1.0 if physical address is verified
        addr_bonus = 1.0 if (c["s_addr"] >= 0.65 and c["key_count"] >= 2) else 0.0

        total_priority = (
            tier_weight * 2.0
            + channel_bonus
            + spec_bonus
            + addr_bonus
            + effective_sim * 3.0
        )
        return -total_priority

    sorted_cands = sorted(candidates, key=ranking_key)
    selected = sorted_cands[:cap]

    pool_size = float(len(selected))
    result = []
    for rank_idx, c in enumerate(selected):
        meta = {
            "channel_agreement_count": float(c["ch_count"]),
            "retrieval_priority_tier": float(c["best_tier"]),
            "reciprocal_retrieval_rank": 1.0 / (rank_idx + 1.0),
            "s1_candidate_pool_size": pool_size,
        }
        result.append((c["tid"], meta))
    return result


def rank_candidates_channel_consensus_first(
    candidates: List[Dict[str, Any]],
    cap: int = 100,
) -> List[Tuple[str, Dict[str, float]]]:
    """Strategy B: Channel Consensus First Ranking.

    Candidates supported by 2 or more independent blocking channels are placed
    first, followed by tier and cheap similarity.
    """
    sorted_cands = sorted(
        candidates,
        key=lambda c: (
            -c["ch_count"],
            c["best_tier"],
            -c["score"],
            -c["spec_sum"],
        ),
    )
    selected = sorted_cands[:cap]

    pool_size = float(len(selected))
    result = []
    for rank_idx, c in enumerate(selected):
        meta = {
            "channel_agreement_count": float(c["ch_count"]),
            "retrieval_priority_tier": float(c["best_tier"]),
            "reciprocal_retrieval_rank": 1.0 / (rank_idx + 1.0),
            "s1_candidate_pool_size": pool_size,
        }
        result.append((c["tid"], meta))
    return result


def rank_candidates_modality_balanced(
    candidates: List[Dict[str, Any]],
    cap: int = 100,
) -> List[Tuple[str, Dict[str, float]]]:
    """Strategy C: Modality-Balanced Specificity Ranking.

    Protects physical address evidence from being suppressed by cross-lingual names,
    while scaling priority by key specificity and channel consensus.
    """
    def ranking_key(c: Dict[str, Any]):
        # Max evidence across modalities
        best_modality = max(c["s_name"], c["s_addr"])
        
        # Priority score
        score = (
            (6.0 - c["best_tier"]) * 2.0
            + (c["ch_count"] - 1.0) * 1.5
            + min(1.0, c["spec_sum"] / 2.0) * 1.0
            + best_modality * 2.5
            + (1.0 if c["has_house"] and c["key_count"] >= 2 else 0.0)
            + (1.0 if c["has_pin"] else 0.0)
        )
        return -score

    sorted_cands = sorted(candidates, key=ranking_key)
    selected = sorted_cands[:cap]

    pool_size = float(len(selected))
    result = []
    for rank_idx, c in enumerate(selected):
        meta = {
            "channel_agreement_count": float(c["ch_count"]),
            "retrieval_priority_tier": float(c["best_tier"]),
            "reciprocal_retrieval_rank": 1.0 / (rank_idx + 1.0),
            "s1_candidate_pool_size": pool_size,
        }
        result.append((c["tid"], meta))
    return result



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--final-cap", type=int, default=100)
    parser.add_argument("--split-file", type=Path, default=Path("experiments/splits/validation_split_seed42_benchmark_10000.json"))
    parser.add_argument("--ground-truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--source1", type=Path, default=Path("dataset/train/train_source1.tsv"))
    parser.add_argument("--source2", type=Path, default=Path("dataset/train/train_source2.tsv"))
    parser.add_argument("--source3", type=Path, default=Path("dataset/train/train_source3.tsv"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/xgb_matcher_exp004b.json"))
    parser.add_argument("--cache-pool", type=Path, default=Path("artifacts/exp005/val_raw_candidates.pkl"))
    parser.add_argument("--output-json", type=Path, default=Path("experiments/EXP-005A-adaptive-ranking.json"))
    args = parser.parse_args()

    overall_start = time.time()

    print("=" * 95, flush=True)
    print("EXP-005 PHASE 1: ADAPTIVE CANDIDATE RANKING BENCHMARK", flush=True)
    print("=" * 95, flush=True)

    # 1. Load Validation Split
    with open(args.split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    val_s1_list: List[str] = split_data["val_s1_ids"]
    val_s1_set = set(val_s1_list)
    print(f"Validation Entities: {len(val_s1_list):,}", flush=True)

    # 2. Parse Ground Truth
    val_gt = parse_ground_truth(args.ground_truth, allowed_s1_ids=val_s1_set)
    total_val_gt = sum(len(t) for t in val_gt.values())
    print(f"Validation Ground Truth: {total_val_gt:,} true links.", flush=True)

    # 3. Load & Preprocess Validation S1 Records
    val_s1_records: Dict[str, Tuple[str, str, str, str]] = {}
    val_s1_prep: Dict[str, PreprocessedRecord] = {}
    raw_idx_val: Dict[str, Dict[str, List[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))

    with open(args.source1, "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            sid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
            if sid in val_s1_set:
                val_s1_records[sid] = (sid, name, addr, country)
                prep = preprocess_record(sid, name, addr, country)
                val_s1_prep[sid] = prep
                c_idx = raw_idx_val[country]
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

    # 4. Apply Frequency Filters
    filter_cfg = BlockingKeyFilterConfig(max_key_fanout=15, preserve_exact_name=True, preserve_pin_keys=True)
    query_idx_clean_val: Dict[str, Dict[str, List[str]]] = {}
    generic_keys_set_val: Dict[str, Set[str]] = collections.defaultdict(set)
    key_fanouts_val: Dict[str, Dict[str, int]] = collections.defaultdict(dict)

    for c, r_idx in raw_idx_val.items():
        c_count = sum(1 for sid, _, _, cnt in val_s1_records.values() if cnt == c)
        pruned_idx, _ = prune_high_frequency_keys(r_idx, total_s1=c_count, config=filter_cfg)
        query_idx_clean_val[c] = pruned_idx
        for k, postings in pruned_idx.items():
            key_fanouts_val[c][k] = len(postings)
            if k.startswith("tk_") and len(postings) >= 5:
                generic_keys_set_val[c].add(k)

    del raw_idx_val
    gc.collect()

    # 5. Check if raw candidate pool is cached, otherwise stream
    if args.cache_pool.is_file():
        print(f"\nLoading cached candidate pool from {args.cache_pool}...", flush=True)
        t0_load = time.time()
        with open(args.cache_pool, "rb") as f:
            deduped_raw_pool = pickle.load(f)
        print(f"Loaded {sum(len(v) for v in deduped_raw_pool.values()):,} candidate pairs in {time.time() - t0_load:.2f}s.", flush=True)
    else:
        print(f"\nStreaming search pool (10.32M records) across {args.num_workers} parallel workers...", flush=True)
        t0_stream = time.time()

        def count_lines(fp):
            with open(fp, "r", encoding="utf-8-sig") as f:
                return sum(1 for _ in f) - 1

        s2_lines = count_lines(args.source2)
        s3_lines = count_lines(args.source3)

        tasks = []
        w_idx = 0
        for src_fp, n_lines in [(args.source2, s2_lines), (args.source3, s3_lines)]:
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
                        query_idx_clean_val,
                        val_s1_prep,
                        key_fanouts_val,
                        generic_keys_set_val,
                        300,  # Worker cap
                    ))
                    w_idx += 1

        with mp.Pool(processes=args.num_workers) as pool:
            worker_results = pool.map(worker_stream_chunk_exp005, tasks)

        combined_val = collections.defaultdict(list)
        for _, p_val in worker_results:
            for sid, items in p_val.items():
                combined_val[sid].extend(items)

        del worker_results
        gc.collect()

        print(f"Streaming completed in {time.time() - t0_stream:.2f}s.", flush=True)
        print("Deduplicating candidates across chunks...", flush=True)
        deduped_raw_pool = deduplicate_candidate_pool(combined_val)
        del combined_val
        gc.collect()

        args.cache_pool.parent.mkdir(parents=True, exist_ok=True)
        with open(args.cache_pool, "wb") as f:
            pickle.dump(deduped_raw_pool, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Cached raw candidate pool to {args.cache_pool}.", flush=True)

    # Calculate Raw Pool Candidate Recall before capping
    raw_hit_gt = 0
    raw_total_cands = 0
    for sid, tids in val_gt.items():
        pool_tids = {c["tid"] for c in deduped_raw_pool.get(sid, [])}
        raw_hit_gt += len(tids & pool_tids)
        raw_total_cands += len(pool_tids)

    print(f"\nRAW CANDIDATE POOL (Before Cap=100):", flush=True)
    print(f"  Total Candidate Pairs : {raw_total_cands:,} ({raw_total_cands / len(val_s1_list):.2f} avg/S1)", flush=True)
    print(f"  Candidate Recall      : {raw_hit_gt:,} / {total_val_gt:,} ({raw_hit_gt / total_val_gt:.4%})", flush=True)
    print(f"  Missed Ground Truth   : {total_val_gt - raw_hit_gt:,}", flush=True)

    # 6. Evaluate Candidate Ranking Strategies
    ranking_strategies = {
        "V3_Baseline_Cap100": rank_candidates_v3_baseline,
        "EXP005_Channel_Consensus_First_Cap100": rank_candidates_channel_consensus_first,
        "EXP005_Adaptive_Consensus_Cap100": rank_candidates_adaptive_consensus,
        "EXP005_Modality_Balanced_Cap100": rank_candidates_modality_balanced,
    }

    strategy_cands_meta = {}
    strategy_needed_tids: Set[str] = set()

    for strat_name, strat_fn in ranking_strategies.items():
        cands_meta = {}
        hit_gt = 0
        total_cands = 0
        for sid in val_s1_list:
            raw_c = deduped_raw_pool.get(sid, [])
            selected = strat_fn(raw_c, cap=args.final_cap)
            cands_meta[sid] = selected
            total_cands += len(selected)
            true_tids = val_gt.get(sid, set())
            sel_tids = {t for t, _ in selected}
            hit_gt += len(true_tids & sel_tids)
            strategy_needed_tids.update(sel_tids)

        recall = hit_gt / total_val_gt
        cap_loss = raw_hit_gt - hit_gt
        print(f"\n--- Strategy: {strat_name} ---", flush=True)
        print(f"  Total Candidates     : {total_cands:,} ({total_cands / len(val_s1_list):.2f} avg/S1)", flush=True)
        print(f"  Candidate Recall     : {hit_gt:,} / {total_val_gt:,} ({recall:.4%})", flush=True)
        print(f"  Candidate Cap Loss   : {cap_loss:,} true links dropped by cap", flush=True)
        strategy_cands_meta[strat_name] = cands_meta

    # 7. Retrieve Target Records for all selected candidates
    print(f"\nRetrieving {len(strategy_needed_tids):,} target records from search pool...", flush=True)
    t0_tgt = time.time()
    target_prep_map: Dict[str, PreprocessedRecord] = {}

    for src_fp in [args.source2, args.source3]:
        with open(src_fp, "r", encoding="utf-8-sig") as f:
            next(f)
            for line in f:
                pos = line.find("\t")
                if pos == -1:
                    continue
                tid = line[:pos]
                if tid in strategy_needed_tids:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 4:
                        target_prep_map[tid] = preprocess_record(parts[0], parts[1], parts[2], parts[3])
                        if len(target_prep_map) == len(strategy_needed_tids):
                            break
        if len(target_prep_map) == len(strategy_needed_tids):
            break

    print(f"Retrieved {len(target_prep_map):,} target records in {time.time() - t0_tgt:.2f}s.", flush=True)

    # 8. Load Frozen V3 Model
    print(f"\nLoading trained V3 XGBoost model from {args.model_path}...", flush=True)
    bst = xgb.Booster()
    bst.load_model(str(args.model_path))
    bst.set_param({"nthread": args.num_workers})

    # 9. Build Feature Matrices and Score with Decision Layer
    s1_country_map = {sid: val_s1_records[sid][3] for sid in val_s1_list}
    eval_results = {}

    for strat_name, cands_meta in strategy_cands_meta.items():
        print(f"\nEvaluating end-to-end matching for {strat_name}...", flush=True)
        t0_score = time.time()

        X_rows = []
        pair_keys = []
        for sid in val_s1_list:
            s1_p = val_s1_prep[sid]
            for tid, meta in cands_meta.get(sid, []):
                tgt_p = target_prep_map.get(tid)
                if tgt_p is None:
                    continue
                feats = extract_features_from_preprocessed(s1_p, tgt_p)
                feats["cheap_candidate_score"] = compute_cheap_candidate_score(feats)
                feats.update(meta)
                row = [feats[col] for col in EXP004B_FEATURE_NAMES]
                X_rows.append(row)
                pair_keys.append((sid, tid))

        dmat = xgb.DMatrix(np.array(X_rows, dtype=np.float32), feature_names=EXP004B_FEATURE_NAMES)
        probs = bst.predict(dmat)

        s1_cand_scores = collections.defaultdict(list)
        for (sid, tid), p in zip(pair_keys, probs):
            s1_cand_scores[sid].append((tid, float(p)))

        # Apply EXP-003B Decision Layer
        predictions = {sid: set() for sid in val_s1_list}
        total_pred = 0
        for sid in val_s1_list:
            cands = s1_cand_scores.get(sid, [])
            country = s1_country_map.get(sid, "")
            matched = filter_entity_candidates(cands, country, DEFAULT_V3_DECISION_CONFIG)
            predictions[sid] = set(matched)
            total_pred += len(matched)

        rep = evaluate_predictions(predictions, val_gt)
        eval_results[strat_name] = {
            "report": rep,
            "total_predicted": total_pred,
            "candidate_count": len(pair_keys),
            "scoring_time": time.time() - t0_score,
        }

    # 10. Comparative Summary Table
    print("\n" + "=" * 125, flush=True)
    print("EXPERIMENTAL EVALUATION RESULTS: EXP-005 PHASE 1 (ADAPTIVE RANKING vs V3 BASELINE)", flush=True)
    print("=" * 125, flush=True)
    print(f"{'Strategy':<35} | {'Macro F0.5':<11} | {'Precision':<10} | {'Recall':<10} | {'Pred':<7} | {'TP':<6} | {'FP':<5} | {'Single Acc'}", flush=True)
    print("-" * 125, flush=True)
    for strat_name, res in eval_results.items():
        rep = res["report"]
        print(f"{strat_name:<35} | {rep.macro_f0_5:<11.6f} | {rep.macro_precision:<10.6f} | {rep.macro_recall:<10.6f} | {res['total_predicted']:<7} | {rep.total_tp:<6} | {rep.total_fp:<5} | {rep.singleton_f0_5:<10.4f}", flush=True)
    print("=" * 125, flush=True)

    base_f05 = eval_results["V3_Baseline_Cap100"]["report"].macro_f0_5
    best_strat = max(eval_results.keys(), key=lambda k: eval_results[k]["report"].macro_f0_5)
    best_rep = eval_results[best_strat]["report"]
    base_rep = eval_results["V3_Baseline_Cap100"]["report"]

    delta_f05 = best_rep.macro_f0_5 - base_rep.macro_f0_5
    delta_prec = best_rep.macro_precision - base_rep.macro_precision
    delta_rec = best_rep.macro_recall - base_rep.macro_recall

    print(f"\nBEST STRATEGY: {best_strat}")
    print(f"  Macro F0.5 Delta vs Baseline : {delta_f05:+.6f} ({base_rep.macro_f0_5:.6f} -> {best_rep.macro_f0_5:.6f})")
    print(f"  Precision Delta              : {delta_prec:+.6f} ({base_rep.macro_precision:.6f} -> {best_rep.macro_precision:.6f})")
    print(f"  Recall Delta                 : {delta_rec:+.6f} ({base_rep.macro_recall:.6f} -> {best_rep.macro_recall:.6f})")
    print(f"  True Positive Delta          : {best_rep.total_tp - base_rep.total_tp:+d} ({base_rep.total_tp} -> {best_rep.total_tp})")
    print(f"  False Positive Delta         : {best_rep.total_fp - base_rep.total_fp:+d} ({base_rep.total_fp} -> {best_rep.total_fp})")

    decision = "KEEP" if delta_f05 > 0.0 else "REJECT"
    print(f"  Phase 1 Verdict              : {decision}", flush=True)


    # 11. Save Report
    output_record = {
        "experiment_id": "EXP-005A",
        "description": "Phase 1: Adaptive Candidate Ranking Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_entities": len(val_s1_list),
        "total_runtime_seconds": time.time() - overall_start,
        "results": {
            k: {
                "macro_f0_5": v["report"].macro_f0_5,
                "precision": v["report"].macro_precision,
                "recall": v["report"].macro_recall,
                "tp": v["report"].total_tp,
                "fp": v["report"].total_fp,
                "predicted": v["total_predicted"],
                "singleton_accuracy": v["report"].singleton_f0_5,
                "candidates": v["candidate_count"],
            }
            for k, v in eval_results.items()
        },
        "deltas": {
            "macro_f0_5": delta_f05,
            "precision": delta_prec,
            "recall": delta_rec,
        },
        "verdict": decision,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_record, f, indent=2)
    print(f"Saved experiment report to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
