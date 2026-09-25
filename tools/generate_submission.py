#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Submission Generator

Generates the official `output/matching_results.tsv` file from `dataset/test/`
using high-precision entity resolution matching (exact normalized names +
address numeric verification, and high token overlap + address verification).
Runs fully in-memory under strict RAM budgets and streams over 10M records.
"""

import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.blocking.normalizers import normalize_name_conservative

# Stop words / generic business words to exclude from fuzzy token index
GENERIC_TOKENS = {
    "center", "services", "service", "global", "solutions", "solution",
    "enterprises", "enterprise", "group", "holding", "holdings", "trading",
    "international", "national", "company", "associates", "management",
    "tech", "technologies", "technology", "systems", "system", "consulting",
    "consultants", "industries", "industry", "agency", "network", "ventures",
    "capital", "corporation", "limited", "private", "corp", "inc", "ltd", "pvt", "llc"
}

def extract_numbers(text: str) -> Set[str]:
    """Extract numeric tokens from address (street numbers, building numbers, pin/zip)."""
    if not text:
        return set()
    return set(re.findall(r"\b\d+\b", text))

def extract_tokens(text: str) -> Set[str]:
    """Extract lowercase alphanumeric tokens."""
    if not text:
        return set()
    clean = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return {t for t in clean.split() if len(t) >= 2}

def run_submission_generation(
    test_dir: str = "dataset/test",
    output_path: str = "output/matching_results.tsv",
    max_matches_per_s1: int = 10,
):
    start_time = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    
    print("=" * 80)
    print("Amazon ML Challenge 2026 — Generating Official Submission TSV")
    print("=" * 80)
    print(f"Test directory : {test_dir}")
    print(f"Output path    : {output_path}")
    print()
    
    # -------------------------------------------------------------------------
    # Stage 1: Load and index Test Source 1 records
    # -------------------------------------------------------------------------
    print("Stage 1/4: Indexing Test Source 1 records...")
    s1_order: List[str] = []
    s1_records: Dict[str, Tuple[str, str, str]] = {}
    s1_by_name: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    s1_by_token: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    
    token_freq: Dict[str, int] = defaultdict(int)
    
    with open(s1_path, "r", encoding="utf-8-sig") as f:
        header = next(f)
        for line_num, line in enumerate(f, start=1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            s1_id, name, addr, country = parts[0], parts[1], parts[2], parts[3]
            s1_order.append(s1_id)
            
            norm_name = normalize_name_conservative(name)
            s1_records[s1_id] = (country, norm_name, addr)
            
            # Primary exact name key
            s1_by_name[(country, norm_name)].append(s1_id)
            
            # Count token frequencies
            tokens = extract_tokens(norm_name)
            for t in tokens:
                if len(t) >= 4 and t not in GENERIC_TOKENS:
                    token_freq[t] += 1
                    
            if line_num % 500000 == 0:
                print(f"  Processed {line_num:,} Source 1 records...")
                
    print(f"Loaded {len(s1_order):,} Source 1 test records.")
    
    # Index distinctive tokens (occurring in <= 300 entities)
    print("Building distinctive token index for Source 1...")
    for s1_id in s1_order:
        country, norm_name, _ = s1_records[s1_id]
        tokens = extract_tokens(norm_name)
        for t in tokens:
            if len(t) >= 4 and t not in GENERIC_TOKENS and token_freq.get(t, 0) <= 300:
                s1_by_token[(country, t)].append(s1_id)
                
    print(f"Exact name index size: {len(s1_by_name):,} keys.")
    print(f"Token index size     : {len(s1_by_token):,} keys.")
    print(f"Stage 1 completed in {time.time() - start_time:.1f}s.")
    print()

    # Pre-extract S1 numbers and tokens on demand or cached
    s1_numbers_cache: Dict[str, Set[str]] = {}
    def get_s1_numbers(s1_id: str) -> Set[str]:
        if s1_id not in s1_numbers_cache:
            s1_numbers_cache[s1_id] = extract_numbers(s1_records[s1_id][2])
        return s1_numbers_cache[s1_id]

    s1_addr_tokens_cache: Dict[str, Set[str]] = {}
    def get_s1_addr_tokens(s1_id: str) -> Set[str]:
        if s1_id not in s1_addr_tokens_cache:
            s1_addr_tokens_cache[s1_id] = extract_tokens(s1_records[s1_id][2])
        return s1_addr_tokens_cache[s1_id]

    s1_name_tokens_cache: Dict[str, Set[str]] = {}
    def get_s1_name_tokens(s1_id: str) -> Set[str]:
        if s1_id not in s1_name_tokens_cache:
            s1_name_tokens_cache[s1_id] = extract_tokens(s1_records[s1_id][1])
        return s1_name_tokens_cache[s1_id]

    # Predictions mapping: s1_id -> set of matched target IDs
    predictions: Dict[str, Set[str]] = defaultdict(set)
    total_matches_found = 0

    # -------------------------------------------------------------------------
    # Helper: Match a single candidate against Source 1 index
    # -------------------------------------------------------------------------
    def match_record(target_id: str, name: str, addr: str, country: str):
        nonlocal total_matches_found
        norm_name = normalize_name_conservative(name)
        target_nums = extract_numbers(addr)
        target_name_tokens = extract_tokens(norm_name)
        target_addr_tokens = extract_tokens(addr)

        matched_s1_ids: Set[str] = set()

        # Rule 1: Exact Normalized Name Match + Address Verification
        exact_s1 = s1_by_name.get((country, norm_name))
        if exact_s1:
            for s1_id in exact_s1:
                s1_nums = get_s1_numbers(s1_id)
                num_overlap = bool(s1_nums & target_nums)
                if s1_nums and target_nums:
                    if num_overlap:
                        matched_s1_ids.add(s1_id)
                else:
                    # If either has no numbers, check address token overlap
                    s1_addr_toks = get_s1_addr_tokens(s1_id)
                    if len(s1_addr_toks & target_addr_tokens) >= 1:
                        matched_s1_ids.add(s1_id)

        # Rule 2: High Token Overlap + Strict Address Verification (for targets not matched exactly)
        if not matched_s1_ids and len(target_name_tokens) >= 2:
            candidate_s1_ids: Set[str] = set()
            for t in target_name_tokens:
                if len(t) >= 4 and t not in GENERIC_TOKENS:
                    matching_s1s = s1_by_token.get((country, t))
                    if matching_s1s:
                        candidate_s1_ids.update(matching_s1s)

            for s1_id in candidate_s1_ids:
                s1_name_toks = get_s1_name_tokens(s1_id)
                union_toks = s1_name_toks | target_name_tokens
                if not union_toks:
                    continue
                jaccard = len(s1_name_toks & target_name_tokens) / len(union_toks)
                if jaccard >= 0.75:
                    s1_nums = get_s1_numbers(s1_id)
                    num_overlap = bool(s1_nums & target_nums)
                    if s1_nums and target_nums:
                        if num_overlap:
                            matched_s1_ids.add(s1_id)
                    else:
                        s1_addr_toks = get_s1_addr_tokens(s1_id)
                        if len(s1_addr_toks & target_addr_tokens) >= 2:
                            matched_s1_ids.add(s1_id)

        # Record matches (capped per S1)
        for s1_id in matched_s1_ids:
            if len(predictions[s1_id]) < max_matches_per_s1:
                predictions[s1_id].add(target_id)
                total_matches_found += 1

    # -------------------------------------------------------------------------
    # Stage 2: Stream Test Source 2 records
    # -------------------------------------------------------------------------
    print("Stage 2/4: Streaming Test Source 2 records...")
    t2_start = time.time()
    with open(s2_path, "r", encoding="utf-8-sig") as f:
        header = next(f)
        for line_num, line in enumerate(f, start=1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                match_record(parts[0], parts[1], parts[2], parts[3])
            if line_num % 1000000 == 0:
                print(f"  Processed {line_num:,} Source 2 records ({total_matches_found:,} matches so far)...")
    print(f"Source 2 processing complete in {time.time() - t2_start:.1f}s.")
    print()

    # -------------------------------------------------------------------------
    # Stage 3: Stream Test Source 3 records
    # -------------------------------------------------------------------------
    print("Stage 3/4: Streaming Test Source 3 records...")
    t3_start = time.time()
    with open(s3_path, "r", encoding="utf-8-sig") as f:
        header = next(f)
        for line_num, line in enumerate(f, start=1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                match_record(parts[0], parts[1], parts[2], parts[3])
            if line_num % 1000000 == 0:
                print(f"  Processed {line_num:,} Source 3 records ({total_matches_found:,} matches so far)...")
    print(f"Source 3 processing complete in {time.time() - t3_start:.1f}s.")
    print()

    # -------------------------------------------------------------------------
    # Stage 4: Write matching_results.tsv in strict format
    # -------------------------------------------------------------------------
    print("Stage 4/4: Writing official matching_results.tsv...")
    t4_start = time.time()
    matched_entities_count = 0
    singleton_count = 0
    total_written_matches = 0

    with open(output_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_order:
            matched_ids = predictions.get(s1_id)
            if matched_ids:
                matched_entities_count += 1
                total_written_matches += len(matched_ids)
                out.write(f"{s1_id}\t{','.join(sorted(matched_ids))}\n")
            else:
                singleton_count += 1
                out.write(f"{s1_id}\t\n")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Finished writing {output_path} in {time.time() - t4_start:.1f}s.")
    print(f"  File size                 : {file_size_mb:.2f} MB")
    print(f"  Total S1 rows written     : {len(s1_order):,}")
    print(f"  Matched S1 entities       : {matched_entities_count:,} ({matched_entities_count / len(s1_order) * 100:.2f}%)")
    print(f"  Singletons (empty list)   : {singleton_count:,} ({singleton_count / len(s1_order) * 100:.2f}%)")
    print(f"  Total matched pairs       : {total_written_matches:,}")
    print(f"  Total pipeline runtime    : {time.time() - start_time:.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    test_directory = sys.argv[1] if len(sys.argv) > 1 else "dataset/test"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output/matching_results.tsv"
    run_submission_generation(test_dir=test_directory, output_path=output_file)
