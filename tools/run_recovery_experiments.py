#!/usr/bin/env python3
"""Run EXP-001E: Candidate Generation / Recall Recovery Experiments.

Evaluates 3 new recovery strategies motivated by failure analysis of EXP-001D:
  - EXP-001E-A: Relaxed Numeric-Address Composite Keys
  - EXP-001E-B: Domain/Handle Stripping & Distinctive Salient Tokens
  - EXP-001E-C: Name-Locality Cross-Field Keys
  - EXP-001E-Union: Union of EXP-001D + Recovery Keys

Evaluates against the complete training search pool (10.32M records)
using the 2,001 validation S1 entities.
"""

import argparse
import collections
import json
import os
import re
import sys
import time
import tracemalloc
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking.evaluator import BlockingResult, evaluate_blocking
from src.blocking.normalizers import (
    ADDRESS_STOPWORDS,
    LEGAL_AND_BUSINESS_STOPWORDS,
    extract_address_keys,
    extract_token_blocking_keys,
    normalize_name_conservative,
    tokenize_name,
)
from src.data.ground_truth import parse_ground_truth
from src.experiments.logger import ExperimentLogger, ExperimentRecord

# Extended address stopwords including prefixes and states observed in failure cases
EXTENDED_ADDRESS_STOPWORDS: Set[str] = ADDRESS_STOPWORDS | {
    "door", "no", "hno", "house", "flat", "shop", "plot", "khasra",
    "kh", "premises", "surat", "delhi", "mumbai", "kolkata", "chennai",
    "bangalore", "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow",
    "texas", "california", "florida", "illinois", "georgia", "ohio",
    "pennsylvania", "indiana", "michigan", "virginia", "washington",
}

# Stop words specifically to prevent single-token candidate explosion
EXTRA_TOKEN_STOPWORDS: Set[str] = LEGAL_AND_BUSINESS_STOPWORDS | {
    "india", "indian", "america", "american", "national", "international",
    "partners", "health", "estate", "brothers", "enterprises", "solutions",
    "services", "global", "group", "holdings", "trading", "consulting",
    "management", "associates", "center", "centre", "industries",
}


def extract_h1_relaxed_address_keys(address: str) -> List[str]:
    """Hypothesis 1: Relaxed Numeric Address Keys.
    
    Pairs the first 2 numeric tokens with the first 2 non-stop street words.
    Robust to address word-order variations and prefixes like 'Door No'.
    """
    if not address or not address.strip():
        return []

    norm = unicodedata.normalize("NFKC", address).lower()
    nums = re.findall(r"\b\d+\b", norm)
    
    clean = [
        " " if unicodedata.category(c).startswith(("P", "S")) else c
        for c in norm
    ]
    words = [
        w for w in "".join(clean).split()
        if len(w) >= 3 and not w.isdigit() and w not in EXTENDED_ADDRESS_STOPWORDS
    ]

    keys = []
    # Pair top 2 numbers with top 2 street words
    if nums and words:
        for n in nums[:2]:
            for w in words[:2]:
                keys.append(f"nword_{n}_{w}")
            # If PIN code (5 or 6 digits)
            if len(n) in (5, 6):
                for w in words[:2]:
                    keys.append(f"pin_{n}_{w}")

    # First two informative street/locality words
    if len(words) >= 2:
        keys.append(f"loc_{words[0]}_{words[1]}")

    return list(dict.fromkeys(keys))


def clean_domain_or_handle(name: str) -> str:
    """Clean domain names and social handles."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", name).lower().strip()
    s = s.lstrip("@#")
    for suffix in [".co.in", ".com", ".org", ".net", ".in", ".co", ".biz", ".info"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def extract_h2_domain_and_salient_token_keys(name: str) -> List[str]:
    """Hypothesis 2: Cleaned Domain/Handle + Salient Unique Tokens.
    
    Extracts brand tokens from domains/handles and emits distinctive non-generic tokens.
    """
    cleaned = clean_domain_or_handle(name)
    norm = normalize_name_conservative(cleaned)
    if not norm:
        return []
        
    words = norm.split()
    tokens = [
        w for w in words
        if len(w) >= 3 and w not in EXTRA_TOKEN_STOPWORDS
    ]
    
    keys = []
    if len(tokens) == 1:
        if len(tokens[0]) >= 4:
            keys.append(f"tok_{tokens[0]}")
    elif len(tokens) >= 2:
        # Top token pairs
        top = tokens[:3]
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                t1, t2 = sorted([top[i], top[j]])
                keys.append(f"{t1}_{t2}")
        # Add distinctive proper nouns (length >= 6 and not common)
        for t in tokens:
            if len(t) >= 6 and t not in EXTRA_TOKEN_STOPWORDS:
                keys.append(f"tok_{t}")

    # Unspaced representation for concatenated domains (e.g. 'vadodaraindustries')
    clean_unspaced = norm.replace(" ", "")
    if len(clean_unspaced) >= 6:
        keys.append(f"unspaced_{clean_unspaced[:10]}")

    return list(dict.fromkeys(keys))


def extract_h3_name_locality_keys(name: str, address: str) -> List[str]:
    """Hypothesis 3: Name-Locality Cross-Field Keys.
    
    Pairs primary distinctive name token (prefix) with primary locality word.
    Handles empty/damaged numbers and transliterated suffix changes.
    """
    if not name or not address:
        return []
    
    norm_n = normalize_name_conservative(clean_domain_or_handle(name))
    n_tokens = [w for w in norm_n.split() if len(w) >= 4 and w not in EXTRA_TOKEN_STOPWORDS]
    
    norm_a = unicodedata.normalize("NFKC", address).lower()
    clean_a = [
        " " if unicodedata.category(c).startswith(("P", "S")) else c
        for c in norm_a
    ]
    a_words = [
        w for w in "".join(clean_a).split()
        if len(w) >= 4 and not w.isdigit() and w not in EXTENDED_ADDRESS_STOPWORDS
    ]

    keys = []
    if n_tokens and a_words:
        name_prefix = n_tokens[0][:5]
        for loc in a_words[:2]:
            keys.append(f"nl_{name_prefix}_{loc}")
    return list(dict.fromkeys(keys))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--output-dir",
        type=Path,
        default=Path("experiments/blocking"),
    )
    args = parser.parse_args()

    overall_start = time.time()
    tracemalloc.start()

    print("=" * 80)
    print("EXP-001E: Recovery Blocking Experiments")
    print("=" * 80)

    # 1. Load validation split
    with open(args.split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    val_s1_list = split_data["val_s1_ids"]
    val_s1_set = set(val_s1_list)
    print(f"Validation entities: {len(val_s1_set):,}")

    # 2. Load ground truth
    val_ground_truth = parse_ground_truth(args.ground_truth, allowed_s1_ids=val_s1_set)
    val_true_target_ids: Set[str] = set()
    for matches in val_ground_truth.values():
        val_true_target_ids.update(matches)
    print(f"Ground truth targets: {len(val_true_target_ids):,}")

    # 3. Load S1 records
    val_s1_records: Dict[str, Tuple[str, str, str]] = {}
    with open(args.source1, "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in val_s1_set:
                val_s1_records[parts[0]] = (parts[1], parts[2], parts[3])

    # 4. Precompute query keys
    print("Precomputing query keys for recovery strategies...")
    query_exp001d: Dict[Tuple[str, str], Set[str]] = collections.defaultdict(set)
    query_ea: Dict[Tuple[str, str], Set[str]] = collections.defaultdict(set)
    query_eb: Dict[Tuple[str, str], Set[str]] = collections.defaultdict(set)
    query_ec: Dict[Tuple[str, str], Set[str]] = collections.defaultdict(set)

    for s1_id in val_s1_list:
        name, addr, country = val_s1_records[s1_id]
        
        # EXP-001D baseline keys
        en = normalize_name_conservative(name)
        if en:
            query_exp001d[(country, f"en_{en}")].add(s1_id)
        for tk in extract_token_blocking_keys(name):
            query_exp001d[(country, f"tk_{tk}")].add(s1_id)
        for ak in extract_address_keys(addr):
            query_exp001d[(country, f"ak_{ak}")].add(s1_id)

        # EXP-001E-A: Relaxed Address Keys
        for ak in extract_h1_relaxed_address_keys(addr):
            query_ea[(country, ak)].add(s1_id)

        # EXP-001E-B: Domain & Salient Tokens
        for bk in extract_h2_domain_and_salient_token_keys(name):
            query_eb[(country, bk)].add(s1_id)

        # EXP-001E-C: Name-Locality Cross-Field Keys
        for ck in extract_h3_name_locality_keys(name, addr):
            query_ec[(country, ck)].add(s1_id)

    print(f"  EXP-001D keys: {len(query_exp001d):,}")
    print(f"  EXP-001E-A keys: {len(query_ea):,}")
    print(f"  EXP-001E-B keys: {len(query_eb):,}")
    print(f"  EXP-001E-C keys: {len(query_ec):,}")

    # 5. Initialize candidate sets
    cands_exp001d: Dict[str, Set[str]] = {s1: set() for s1 in val_s1_list}
    cands_ea: Dict[str, Set[str]] = {s1: set() for s1 in val_s1_list}
    cands_eb: Dict[str, Set[str]] = {s1: set() for s1 in val_s1_list}
    cands_ec: Dict[str, Set[str]] = {s1: set() for s1 in val_s1_list}

    target_records: Dict[str, Tuple[str, str, str]] = {}

    # 6. Stream full search pool
    print("Streaming complete search pool (Source 2 + Source 3)...")
    stream_t0 = time.time()
    total_pool_records = 0

    for src_path in [args.source2, args.source3]:
        file_t0 = time.time()
        file_records = 0
        with open(src_path, "r", encoding="utf-8-sig") as f:
            next(f)
            for line in f:
                file_records += 1
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4:
                    continue
                tid, tname, taddr, tcountry = parts[0], parts[1], parts[2], parts[3]

                if tid in val_true_target_ids:
                    target_records[tid] = (tname, taddr, tcountry)

                # 1. EXP-001D baseline
                ten = normalize_name_conservative(tname)
                if ten and (tcountry, f"en_{ten}") in query_exp001d:
                    for s1 in query_exp001d[(tcountry, f"en_{ten}")]:
                        cands_exp001d[s1].add(tid)
                for tk in extract_token_blocking_keys(tname):
                    if (tcountry, f"tk_{tk}") in query_exp001d:
                        for s1 in query_exp001d[(tcountry, f"tk_{tk}")]:
                            cands_exp001d[s1].add(tid)
                for ak in extract_address_keys(taddr):
                    if (tcountry, f"ak_{ak}") in query_exp001d:
                        for s1 in query_exp001d[(tcountry, f"ak_{ak}")]:
                            cands_exp001d[s1].add(tid)

                # 2. EXP-001E-A: Relaxed Address
                for ak in extract_h1_relaxed_address_keys(taddr):
                    if (tcountry, ak) in query_ea:
                        for s1 in query_ea[(tcountry, ak)]:
                            cands_ea[s1].add(tid)

                # 3. EXP-001E-B: Domain & Salient Tokens
                for bk in extract_h2_domain_and_salient_token_keys(tname):
                    if (tcountry, bk) in query_eb:
                        for s1 in query_eb[(tcountry, bk)]:
                            cands_eb[s1].add(tid)

                # 4. EXP-001E-C: Name-Locality Cross-Field
                for ck in extract_h3_name_locality_keys(tname, taddr):
                    if (tcountry, ck) in query_ec:
                        for s1 in query_ec[(tcountry, ck)]:
                            cands_ec[s1].add(tid)

        print(f"  Processed {file_records:,} records in {time.time() - file_t0:.2f}s.")
        total_pool_records += file_records

    stream_time = time.time() - stream_t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem / (1024 * 1024)
    print(f"\nStreaming completed in {stream_time:.2f}s. Peak RAM: {peak_mem_mb:.2f} MB.")

    # 7. Form Union Strategy (EXP-001D + Recovery Strategies)
    print("Forming EXP-001E-Union (EXP-001D + H1 + H2 + H3)...")
    cands_e_union: Dict[str, Set[str]] = {}
    for s1 in val_s1_list:
        cands_e_union[s1] = (
            cands_exp001d[s1]
            | cands_ea[s1]
            | cands_eb[s1]
            | cands_ec[s1]
        )

    # 8. Evaluate all strategies
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(experiments_dir="experiments")

    eval_plan = [
        ("EXP-001D", "Baseline Strategy D (Exact + Token + Address)", cands_exp001d),
        ("EXP-001E-A", "Strategy E-A: Relaxed Numeric Address Keys", cands_ea),
        ("EXP-001E-B", "Strategy E-B: Domain/Handle & Salient Tokens", cands_eb),
        ("EXP-001E-C", "Strategy E-C: Name-Locality Cross-Field Keys", cands_ec),
        ("EXP-001E-Union", "Strategy E-Union: EXP-001D + E-A + E-B + E-C", cands_e_union),
    ]

    results: List[BlockingResult] = []

    print("\n" + "=" * 80)
    print("RECOVERY EVALUATION RESULTS")
    print("=" * 80)

    for exp_id, name, cand_map in eval_plan:
        res = evaluate_blocking(
            candidates_map=cand_map,
            ground_truth=val_ground_truth,
            s1_records=val_s1_records,
            target_records=target_records,
            total_pool_size=total_pool_records,
            strategy_id=exp_id,
            strategy_name=name,
            country_restricted=True,
            runtime_seconds=stream_time,
            peak_memory_mb=peak_mem_mb,
            config={"country_restricted": True},
        )
        results.append(res)
        print(res.summary())

        # Save individual metrics JSON
        metric_file = args.output_dir / f"{exp_id}_metrics.json"
        with open(metric_file, "w", encoding="utf-8") as f:
            json.dump(res.to_dict(), f, indent=2)

        # Log to experiments
        exp_record = ExperimentRecord(
            experiment_id=exp_id,
            hypothesis=f"Evaluate recall recovery and candidate volume for {name}.",
            data_split_description=f"Validation benchmark split (2,001 val S1) against full S2+S3 search pool ({total_pool_records:,} records).",
            blocking_method=name,
            features=[name.lower().replace(" ", "_")],
            model="Inverted Index Hash Table",
            hyperparameters={"country_restricted": True},
            validation_method="Candidate Pair Recall & Reduction Ratio",
            macro_F0_5=res.macro_candidate_recall,
            precision=res.retrieved_true_matches / max(1, res.total_candidate_pairs),
            recall=res.pair_recall,
            training_time_seconds=0.0,
            inference_time_seconds=res.runtime_seconds,
            notes=f"Avg candidates/S1: {res.avg_candidates_per_s1:.2f}, Median: {res.median_candidates_per_s1:.0f}, P95: {res.p95_candidates_per_s1:.0f}, Max: {res.max_candidates_per_s1:,}, Total pairs: {res.total_candidate_pairs:,}",
            conclusion=f"Pair recall: {res.pair_recall*100:.2f}%, S2 recall: {res.s2_recall*100:.2f}%, S3 recall: {res.s3_recall*100:.2f}%, Reduction Ratio: {res.reduction_ratio*100:.6f}%.",
            next_experiment="EXP-001E-Union" if "E-" in exp_id and exp_id != "EXP-001E-Union" else "EXP-002",
            extra_metrics={
                "total_candidate_pairs": res.total_candidate_pairs,
                "reduction_ratio": res.reduction_ratio,
                "avg_candidates_per_s1": res.avg_candidates_per_s1,
                "median_candidates_per_s1": res.median_candidates_per_s1,
                "p95_candidates_per_s1": res.p95_candidates_per_s1,
                "max_candidates_per_s1": res.max_candidates_per_s1,
                "s2_recall": res.s2_recall,
                "s3_recall": res.s3_recall,
                "zero_candidate_entities": res.entities_with_zero_candidates,
                "perfect_recall_entities": res.entities_with_all_true_retrieved,
                "missed_entities": res.entities_with_missed_true_matches,
            },
        )
        logger.log(exp_record)

    print(f"\nCompleted EXP-001E recovery experiments in {time.time() - overall_start:.2f}s.")


if __name__ == "__main__":
    main()
