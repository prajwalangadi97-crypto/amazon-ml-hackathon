#!/usr/bin/env python3
"""EXP-001E: Missed-Link Error Analysis and Recall Recovery.

Task 1: Extract every missed true pair from EXP-001D.
Task 2: Save missed pairs to artifacts/blocking/missed_links_EXP001D.tsv.
Task 3: Compute automated failure analysis signals (name, address, other).
Task 4: Cluster failure modes into measurable categories.
Task 5: Check why each existing blocking strategy (A, B, C) failed for each pair.
"""

import argparse
import collections
import json
import math
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking.normalizers import (
    ADDRESS_STOPWORDS,
    LEGAL_AND_BUSINESS_STOPWORDS,
    extract_address_keys,
    extract_token_blocking_keys,
    normalize_name_conservative,
    tokenize_name,
)
from src.data.ground_truth import parse_ground_truth


def char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generate character n-grams."""
    if not text:
        return set()
    cleaned = "".join(c for c in text.lower() if c.isalnum() or c.isspace())
    cleaned = f" {cleaned.strip()} "
    if len(cleaned) < n:
        return {cleaned}
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def jaccard_similarity(s1: Set[Any], s2: Set[Any]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("experiments/splits/validation_split_seed42_benchmark_10000.json"),
        help="Path to validation split JSON",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("dataset/train/train_ground_truth.tsv"),
        help="Path to train ground truth TSV",
    )
    parser.add_argument(
        "--source1",
        type=Path,
        default=Path("dataset/train/train_source1.tsv"),
        help="Path to train source1 TSV",
    )
    parser.add_argument(
        "--source2",
        type=Path,
        default=Path("dataset/train/train_source2.tsv"),
        help="Path to train source2 TSV",
    )
    parser.add_argument(
        "--source3",
        type=Path,
        default=Path("dataset/train/train_source3.tsv"),
        help="Path to train source3 TSV",
    )
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=Path("artifacts/blocking/missed_links_EXP001D.tsv"),
        help="Path to save missed pairs TSV",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("EXP-001E: Missed-Link Error Analysis (Tasks 1 - 5)")
    print("=" * 80)

    # 1. Load validation split
    with open(args.split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    val_s1_list = split_data["val_s1_ids"]
    val_s1_set = set(val_s1_list)
    print(f"Validation entities: {len(val_s1_set):,}")

    # 2. Load ground truth for validation entities
    val_ground_truth = parse_ground_truth(args.ground_truth, allowed_s1_ids=val_s1_set)
    val_true_target_ids: Set[str] = set()
    total_true_links = 0
    for s1_id, matches in val_ground_truth.items():
        val_true_target_ids.update(matches)
        total_true_links += len(matches)
    print(f"Total true links for validation S1: {total_true_links:,}")
    print(f"Distinct true targets: {len(val_true_target_ids):,}")

    # 3. Load validation S1 records
    val_s1_records: Dict[str, Tuple[str, str, str]] = {}
    with open(args.source1, "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in val_s1_set:
                val_s1_records[parts[0]] = (parts[1], parts[2], parts[3])

    # 4. Precompute query keys for EXP-001D (country-restricted)
    query_exact_country: Dict[Tuple[str, str], Set[str]] = collections.defaultdict(set)
    query_token_country: Dict[Tuple[str, str], Set[str]] = collections.defaultdict(set)
    query_addr_country: Dict[Tuple[str, str], Set[str]] = collections.defaultdict(set)

    for s1_id in val_s1_list:
        name, addr, country = val_s1_records[s1_id]
        
        # Exact Name
        norm_name = normalize_name_conservative(name)
        if norm_name:
            query_exact_country[(country, norm_name)].add(s1_id)
            
        # Token Name
        token_keys = extract_token_blocking_keys(name)
        for tk in token_keys:
            query_token_country[(country, tk)].add(s1_id)
            
        # Address
        addr_keys = extract_address_keys(addr)
        for ak in addr_keys:
            query_addr_country[(country, ak)].add(s1_id)

    # 5. Stream S2 and S3, retrieve candidates for EXP-001D and collect target records
    cands_exact: Dict[str, Set[str]] = {s1: set() for s1 in val_s1_list}
    cands_token: Dict[str, Set[str]] = {s1: set() for s1 in val_s1_list}
    cands_addr: Dict[str, Set[str]] = {s1: set() for s1 in val_s1_list}
    cands_union: Dict[str, Set[str]] = {s1: set() for s1 in val_s1_list}

    target_records: Dict[str, Tuple[str, str, str]] = {}

    print("Streaming S2 and S3 to generate candidates and collect target metadata...")
    t0 = time.time()
    for src_path in [args.source2, args.source3]:
        with open(src_path, "r", encoding="utf-8-sig") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4:
                    continue
                tid, tname, taddr, tcountry = parts[0], parts[1], parts[2], parts[3]

                if tid in val_true_target_ids:
                    target_records[tid] = (tname, taddr, tcountry)

                # Exact Name
                tnorm_name = normalize_name_conservative(tname)
                if tnorm_name:
                    m = query_exact_country.get((tcountry, tnorm_name))
                    if m:
                        for s1_id in m:
                            cands_exact[s1_id].add(tid)
                            cands_union[s1_id].add(tid)

                # Token Name
                t_tokens = extract_token_blocking_keys(tname)
                for tk in t_tokens:
                    m = query_token_country.get((tcountry, tk))
                    if m:
                        for s1_id in m:
                            cands_token[s1_id].add(tid)
                            cands_union[s1_id].add(tid)

                # Address
                t_addrs = extract_address_keys(taddr)
                for ak in t_addrs:
                    m = query_addr_country.get((tcountry, ak))
                    if m:
                        for s1_id in m:
                            cands_addr[s1_id].add(tid)
                            cands_union[s1_id].add(tid)

    print(f"Streaming completed in {time.time() - t0:.2f}s.")

    # TASK 1: Identify all missed true links
    retrieved_true_links = 0
    missed_pairs: List[Dict[str, Any]] = []

    for s1_id, true_matches in val_ground_truth.items():
        gen_cands = cands_union[s1_id]
        for tid in true_matches:
            if tid in gen_cands:
                retrieved_true_links += 1
            else:
                s1_name, s1_addr, s1_country = val_s1_records[s1_id]
                t_name, t_addr, t_country = target_records.get(
                    tid, ("<UNKNOWN>", "<UNKNOWN>", "<UNKNOWN>")
                )
                source_type = "Source 2" if tid.startswith("S2-") else "Source 3"
                missed_pairs.append({
                    "s1_entity_id": s1_id,
                    "missed_entity_id": tid,
                    "source": source_type,
                    "s1_business_name": s1_name,
                    "s1_business_address": s1_addr,
                    "s1_country": s1_country,
                    "target_business_name": t_name,
                    "target_business_address": t_addr,
                    "target_country": t_country,
                })

    missed_count = len(missed_pairs)
    recall_pct = (retrieved_true_links / total_true_links) * 100
    missed_pct = (missed_count / total_true_links) * 100

    print("\n--- TASK 1 REPORT: MISSED TRUE LINKS ---")
    print(f"Total true links     : {total_true_links:,}")
    print(f"Retrieved true links : {retrieved_true_links:,} ({recall_pct:.2f}%)")
    print(f"Missed true links    : {missed_count:,} ({missed_pct:.2f}%)")

    # TASK 2: Save missed pairs TSV
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_tsv, "w", encoding="utf-8") as f:
        header = [
            "s1_entity_id",
            "missed_entity_id",
            "source",
            "s1_business_name",
            "s1_business_address",
            "s1_country",
            "target_business_name",
            "target_business_address",
            "target_country",
        ]
        f.write("\t".join(header) + "\n")
        for p in missed_pairs:
            row = [
                p["s1_entity_id"],
                p["missed_entity_id"],
                p["source"],
                p["s1_business_name"],
                p["s1_business_address"],
                p["s1_country"],
                p["target_business_name"],
                p["target_business_address"],
                p["target_country"],
            ]
            f.write("\t".join(row) + "\n")
    print(f"Saved missed links TSV to: {args.output_tsv}")

    # TASK 3: Automatic Failure Analysis Signals
    # For every missed pair, calculate signals
    detailed_missed_analysis = []
    
    # Counters for diagnostics
    country_equal_count = 0
    s1_addr_empty_count = 0
    target_addr_empty_count = 0
    name_norm_equal_count = 0
    addr_norm_equal_count = 0

    token_jaccard_dist = []
    char_jaccard_dist = []
    addr_token_jaccard_dist = []
    addr_num_overlap_count = 0

    for p in missed_pairs:
        s1_n, s1_a, s1_c = p["s1_business_name"], p["s1_business_address"], p["s1_country"]
        t_n, t_a, t_c = p["target_business_name"], p["target_business_address"], p["target_country"]

        # NAME SIGNALS
        s1_norm = normalize_name_conservative(s1_n)
        t_norm = normalize_name_conservative(t_n)
        name_exact_norm = (s1_norm == t_norm) and bool(s1_norm)
        if name_exact_norm:
            name_norm_equal_count += 1

        s1_toks = set(tokenize_name(s1_n))
        t_toks = set(tokenize_name(t_n))
        shared_name_toks = s1_toks & t_toks
        name_tok_jaccard = jaccard_similarity(s1_toks, t_toks)
        token_jaccard_dist.append(name_tok_jaccard)

        s1_char_ng = char_ngrams(s1_norm)
        t_char_ng = char_ngrams(t_norm)
        name_char_jaccard = jaccard_similarity(s1_char_ng, t_char_ng)
        char_jaccard_dist.append(name_char_jaccard)

        len_max_n = max(len(s1_norm), len(t_norm), 1)
        name_len_ratio = min(len(s1_norm), len(t_norm)) / len_max_n

        # ADDRESS SIGNALS
        s1_a_norm = unicodedata.normalize("NFKC", s1_a).lower().strip()
        t_a_norm = unicodedata.normalize("NFKC", t_a).lower().strip()
        addr_exact_norm = (s1_a_norm == t_a_norm) and bool(s1_a_norm)
        if addr_exact_norm:
            addr_norm_equal_count += 1

        s1_a_clean = [
            " " if unicodedata.category(c).startswith(("P", "S")) else c
            for c in s1_a_norm
        ]
        t_a_clean = [
            " " if unicodedata.category(c).startswith(("P", "S")) else c
            for c in t_a_norm
        ]
        s1_a_toks = set(
            w for w in "".join(s1_a_clean).split()
            if len(w) >= 3 and not w.isdigit() and w not in ADDRESS_STOPWORDS
        )
        t_a_toks = set(
            w for w in "".join(t_a_clean).split()
            if len(w) >= 3 and not w.isdigit() and w not in ADDRESS_STOPWORDS
        )
        addr_tok_jaccard = jaccard_similarity(s1_a_toks, t_a_toks)
        addr_token_jaccard_dist.append(addr_tok_jaccard)

        s1_nums = set(re.findall(r"\b\d+\b", s1_a_norm))
        t_nums = set(re.findall(r"\b\d+\b", t_a_norm))
        shared_nums = s1_nums & t_nums
        if shared_nums:
            addr_num_overlap_count += 1

        len_max_a = max(len(s1_a_norm), len(t_a_norm), 1)
        addr_len_ratio = min(len(s1_a_norm), len(t_a_norm)) / len_max_a

        # OTHER SIGNALS
        country_eq = (s1_c == t_c)
        if country_eq:
            country_equal_count += 1

        s1_a_empty = not bool(s1_a.strip())
        t_a_empty = not bool(t_a.strip())
        if s1_a_empty:
            s1_addr_empty_count += 1
        if t_a_empty:
            target_addr_empty_count += 1

        # TASK 5: Why did A, B, C fail?
        s1_exact_key = (s1_c, s1_norm)
        t_exact_key = (t_c, t_norm)
        why_A_failed = []
        if not s1_norm:
            why_A_failed.append("S1 normalized name empty")
        if not t_norm:
            why_A_failed.append("Target normalized name empty")
        if s1_norm != t_norm:
            why_A_failed.append(f"Normalized name mismatch: '{s1_norm}' != '{t_norm}'")
        if s1_c != t_c:
            why_A_failed.append(f"Country mismatch: '{s1_c}' != '{t_c}'")

        s1_token_keys = set(extract_token_blocking_keys(s1_n))
        t_token_keys = set(extract_token_blocking_keys(t_n))
        shared_token_keys = s1_token_keys & t_token_keys
        why_B_failed = []
        if not shared_token_keys:
            why_B_failed.append(
                f"No shared token key. S1 keys: {sorted(s1_token_keys)[:3]} vs Target keys: {sorted(t_token_keys)[:3]}"
            )
        if s1_c != t_c:
            why_B_failed.append("Country mismatch")

        s1_addr_keys = set(extract_address_keys(s1_a))
        t_addr_keys = set(extract_address_keys(t_a))
        shared_addr_keys = s1_addr_keys & t_addr_keys
        why_C_failed = []
        if t_a_empty:
            why_C_failed.append("Target address empty")
        elif not s1_addr_keys:
            why_C_failed.append("No address keys generated for S1")
        elif not t_addr_keys:
            why_C_failed.append("No address keys generated for Target")
        elif not shared_addr_keys:
            why_C_failed.append(
                f"No shared address key. S1 keys: {sorted(s1_addr_keys)[:2]} vs Target: {sorted(t_addr_keys)[:2]}"
            )
        if s1_c != t_c:
            why_C_failed.append("Country mismatch")

        # TASK 4: Cluster failure modes into measurable hypotheses
        # Category classification
        failure_category = "J. Other / unclassified"
        
        # Test specific hypotheses
        if t_a_empty:
            failure_category = "G. Empty target address (and name differed/insufficient token overlap)"
        elif s1_a_empty:
            failure_category = "H. Empty source address"
        elif s1_norm and t_norm and (s1_norm.startswith(t_norm) or t_norm.startswith(s1_norm)) and name_len_ratio < 0.7:
            failure_category = "A. Name truncation (one is prefix of another, but keys didn't match)"
        elif len(s1_toks) > 1 and len(t_toks) == 1 and list(t_toks)[0] == "".join(w[0] for w in sorted(s1_toks)):
            failure_category = "B. Strong name abbreviation / acronym"
        elif shared_nums and not shared_addr_keys:
            # They share numeric address tokens (e.g. house number or PIN), but the street word or ordering differed
            failure_category = "E. Address number formatting / street token variation (shared number, but address key missed)"
        elif len(shared_name_toks) >= 1 and not shared_token_keys:
            # They share at least one informative word, but token pair / longest token requirement missed it
            failure_category = "C. Single informative word match (shared token but <2 pairs or length < 5)"
        elif name_tok_jaccard < 0.15 and addr_tok_jaccard < 0.15 and not shared_nums:
            failure_category = "I. Both name and address weak (divergent alias/transliteration and distinct address formatting)"
        elif name_char_jaccard > 0.4 and not shared_name_toks:
            failure_category = "D. Name spelling/transliteration variation (high char similarity, no exact token match)"
        elif len(s1_a_toks & t_a_toks) >= 2 and not shared_addr_keys:
            failure_category = "F. Address locality overlap (shared locality tokens but no numeric key match)"

        analysis_item = {
            "pair": p,
            "name_exact_norm": name_exact_norm,
            "name_tok_jaccard": name_tok_jaccard,
            "name_char_jaccard": name_char_jaccard,
            "shared_name_toks": list(shared_name_toks),
            "shared_nums": list(shared_nums),
            "addr_tok_jaccard": addr_tok_jaccard,
            "country_eq": country_eq,
            "t_a_empty": t_a_empty,
            "why_A": why_A_failed,
            "why_B": why_B_failed,
            "why_C": why_C_failed,
            "failure_category": failure_category,
        }
        detailed_missed_analysis.append(analysis_item)

    print("\n--- TASK 3: AUTOMATIC FAILURE ANALYSIS SIGNALS ---")
    print(f"Country Equality in Missed Pairs: {country_equal_count}/{missed_count} ({country_equal_count/missed_count*100:.1f}%)")
    print(f"Target Address Empty           : {target_addr_empty_count}/{missed_count} ({target_addr_empty_count/missed_count*100:.1f}%)")
    print(f"Source Address Empty           : {s1_addr_empty_count}/{missed_count} ({s1_addr_empty_count/missed_count*100:.1f}%)")
    print(f"Exact Normalized Name Match    : {name_norm_equal_count}/{missed_count}")
    print(f"Exact Normalized Address Match : {addr_norm_equal_count}/{missed_count}")
    print(f"Shared Address Numeric Token   : {addr_num_overlap_count}/{missed_count} ({addr_num_overlap_count/missed_count*100:.1f}%)")
    print(f"Mean Name Token Jaccard        : {sum(token_jaccard_dist)/len(token_jaccard_dist):.4f}")
    print(f"Mean Name Character 3-gram Jac : {sum(char_jaccard_dist)/len(char_jaccard_dist):.4f}")
    print(f"Mean Address Token Jaccard     : {sum(addr_token_jaccard_dist)/len(addr_token_jaccard_dist):.4f}")

    print("\n--- TASK 4: CLUSTERED FAILURE MODES ---")
    category_counts = collections.Counter(item["failure_category"] for item in detailed_missed_analysis)
    for cat, cnt in category_counts.most_common():
        print(f"  {cat:75s}: {cnt:3d} ({cnt/missed_count*100:.1f}%)")

    print("\n--- TASK 5: WHY EXISTING BLOCKERS FAILED ---")
    # Tally reasons for A, B, C failure
    fail_A_types = collections.Counter()
    for item in detailed_missed_analysis:
        for r in item["why_A"]:
            if r.startswith("Normalized name mismatch"):
                fail_A_types["Normalized name mismatch"] += 1
            else:
                fail_A_types[r] += 1
    print("Why Strategy A (Exact Name) Failed:")
    for k, v in fail_A_types.most_common():
        print(f"  - {k}: {v}/{missed_count} ({v/missed_count*100:.1f}%)")

    fail_B_types = collections.Counter()
    for item in detailed_missed_analysis:
        for r in item["why_B"]:
            if r.startswith("No shared token key"):
                fail_B_types["No shared token key"] += 1
            else:
                fail_B_types[r] += 1
    print("\nWhy Strategy B (Token Name) Failed:")
    for k, v in fail_B_types.most_common():
        print(f"  - {k}: {v}/{missed_count} ({v/missed_count*100:.1f}%)")

    fail_C_types = collections.Counter()
    for item in detailed_missed_analysis:
        for r in item["why_C"]:
            if r.startswith("No shared address key"):
                fail_C_types["No shared address key"] += 1
            elif r == "Target address empty":
                fail_C_types["Target address empty"] += 1
            else:
                fail_C_types[r] += 1
    print("\nWhy Strategy C (Address Blocking) Failed:")
    for k, v in fail_C_types.most_common():
        print(f"  - {k}: {v}/{missed_count} ({v/missed_count*100:.1f}%)")

    # Sample concrete failure examples for each cluster
    print("\n--- SAMPLE CONCRETE EXAMPLES PER FAILURE MODE ---")
    by_cat = collections.defaultdict(list)
    for item in detailed_missed_analysis:
        by_cat[item["failure_category"]].append(item)

    for cat in sorted(by_cat.keys()):
        print(f"\nCategory: {cat} (Total {len(by_cat[cat])})")
        for sample in by_cat[cat][:3]:
            p = sample["pair"]
            print(f"  S1: [{p['s1_entity_id']}] Name: '{p['s1_business_name']}' | Addr: '{p['s1_business_address']}' | Country: {p['s1_country']}")
            print(f"  Target: [{p['missed_entity_id']}] Name: '{p['target_business_name']}' | Addr: '{p['target_business_address']}' | Country: {p['target_country']}")
            print(f"    Shared Name Tokens: {sample['shared_name_toks']} | Shared Nums: {sample['shared_nums']}")
            print(f"    Name Char Jaccard: {sample['name_char_jaccard']:.3f} | Addr Token Jaccard: {sample['addr_tok_jaccard']:.3f}")

    # Save complete JSON analysis for report generation
    out_json = Path("artifacts/blocking/missed_analysis_summary.json")
    summary_data = {
        "total_true_links": total_true_links,
        "retrieved_true_links": retrieved_true_links,
        "missed_true_links": missed_count,
        "recall_pct": recall_pct,
        "missed_pct": missed_pct,
        "country_equal_count": country_equal_count,
        "target_addr_empty_count": target_addr_empty_count,
        "shared_nums_count": addr_num_overlap_count,
        "category_counts": dict(category_counts),
        "detailed": detailed_missed_analysis,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, default=str)
    print(f"\nSaved analysis summary JSON to: {out_json}")


if __name__ == "__main__":
    main()
