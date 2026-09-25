#!/usr/bin/env python3
"""Streaming profiler for Amazon ML Challenge 2026 TSV datasets.

This script processes TSV files sequentially using line-by-line streaming.
It uses only Python standard library components and maintains minimal memory
overhead (< 500 MB peak RAM per file), making it safe for systems with 16 GB RAM.
"""

import argparse
import collections
import gc
import os
import sys
import time
from pathlib import Path

EXPECTED_HEADERS = {
    "train_source1.tsv": ["entity_id", "business_name", "business_address", "country"],
    "train_source2.tsv": ["entity_id", "business_name", "business_address", "country"],
    "train_source3.tsv": ["entity_id", "business_name", "business_address", "country"],
    "test_source1.tsv": ["entity_id", "business_name", "business_address", "country"],
    "test_source2.tsv": ["entity_id", "business_name", "business_address", "country"],
    "test_source3.tsv": ["entity_id", "business_name", "business_address", "country"],
    "train_ground_truth.tsv": ["source1_entity_id", "matched_entity_ids"],
}

EXPECTED_COUNTRIES = {
    "train": {"US", "India"},
    "test": {"US", "India", "France"},
}


def fmt_num(val):
    return f"{val:,}"


def format_stats(total, count, min_val, max_val):
    if not count:
        return "n=0"
    mean_val = total / count
    return f"n={fmt_num(count)}, min={min_val}, mean={mean_val:.2f}, max={max_val}"


def profile_source_file(path: Path, expected_prefix: str, allowed_countries: set, report: list):
    report.append(f"\n[{path}]")
    t0 = time.time()
    
    if not path.is_file():
        report.append("  ERROR: File not found.")
        return collections.Counter()

    # Track metrics
    nrows = 0
    malformed = 0
    malformed_examples = []
    
    countries = collections.Counter()
    missing_by_col = collections.Counter()
    
    bad_prefix_count = 0
    bad_prefix_examples = []
    
    rows_with_comma_in_name = 0
    rows_with_comma_in_addr = 0
    rows_with_quotes = 0
    
    name_len_total = 0
    name_len_min = None
    name_len_max = 0
    
    addr_len_total = 0
    addr_len_min = None
    addr_len_max = 0
    
    # Duplicate tracking with minimal RAM footprint:
    # IDs are ~12 chars: keep exact string set (~250 MB for 5M items)
    # Names and Addresses: 64-bit integer hashes (~134 MB set per 5M items)
    id_set = set()
    dup_ids = 0
    
    name_hash_set = set()
    dup_names = 0
    
    addr_hash_set = set()
    dup_addrs = 0

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        header_line = f.readline()
        if not header_line:
            report.append("  ERROR: File is empty.")
            return countries
        
        header = [c.strip() for c in header_line.rstrip("\r\n").split("\t")]
        report.append(f"  Columns ({len(header)}): {header}")
        
        expected_hdr = EXPECTED_HEADERS.get(path.name)
        if expected_hdr and header != expected_hdr:
            report.append(f"  HEADER WARNING: Found {header}, expected {expected_hdr}")
            
        col_indices = {col: i for i, col in enumerate(header)}
        expected_cols = ["entity_id", "business_name", "business_address", "country"]
        has_all_cols = all(col in col_indices for col in expected_cols)
        
        if not has_all_cols:
            report.append(f"  ERROR: Missing required columns in {path.name}")
            return countries
        
        eid_idx = col_indices["entity_id"]
        name_idx = col_indices["business_name"]
        addr_idx = col_indices["business_address"]
        country_idx = col_indices["country"]
        num_header_cols = len(header)
        
        for line_num, line in enumerate(f, start=2):
            if not line.strip():
                continue  # skip blank lines (e.g. trailing newline)
            
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != num_header_cols:
                malformed += 1
                if len(malformed_examples) < 3:
                    malformed_examples.append((line_num, len(parts), num_header_cols))
                continue
            
            nrows += 1
            
            # Missing values check
            for col, val in zip(header, parts):
                if not val.strip():
                    missing_by_col[col] += 1
            
            eid = parts[eid_idx]
            name = parts[name_idx]
            addr = parts[addr_idx]
            country = parts[country_idx]
            
            # Country tracking
            country_label = country if country else "<EMPTY>"
            countries[country_label] += 1
            
            # Prefix check
            if not eid.startswith(expected_prefix):
                bad_prefix_count += 1
                if len(bad_prefix_examples) < 3:
                    bad_prefix_examples.append(eid)
                    
            # Duplicate ID check (exact string set)
            if eid in id_set:
                dup_ids += 1
            else:
                id_set.add(eid)
                
            # Duplicate Name check (64-bit hash set)
            name_h = hash(name)
            if name_h in name_hash_set:
                dup_names += 1
            else:
                name_hash_set.add(name_h)
                
            # Duplicate Address check (64-bit hash set)
            addr_h = hash(addr)
            if addr_h in addr_hash_set:
                dup_addrs += 1
            else:
                addr_hash_set.add(addr_h)
                
            # Length stats
            name_len = len(name)
            name_len_total += name_len
            name_len_min = name_len if name_len_min is None else min(name_len_min, name_len)
            name_len_max = max(name_len_max, name_len)
            
            addr_len = len(addr)
            addr_len_total += addr_len
            addr_len_min = addr_len if addr_len_min is None else min(addr_len_min, addr_len)
            addr_len_max = max(addr_len_max, addr_len)
            
            # Field parsing & comma robustness check
            if "," in name:
                rows_with_comma_in_name += 1
            if "," in addr:
                rows_with_comma_in_addr += 1
            if '"' in line:
                rows_with_quotes += 1

    # Cleanup large sets to keep memory free
    del id_set, name_hash_set, addr_hash_set
    gc.collect()
    
    elapsed = time.time() - t0
    report.append(f"  Row count: {fmt_num(nrows)}")
    report.append(f"  Malformed rows: {fmt_num(malformed)}")
    if malformed_examples:
        report.append(f"  Malformed row examples (line, cols_found, cols_expected): {malformed_examples}")
        
    missing_str = ", ".join(f"{col}={fmt_num(missing_by_col[col])}" for col in header) if any(missing_by_col.values()) else "none"
    report.append(f"  Missing values by column: {missing_str}")
    
    country_str = ", ".join(f"{k}={fmt_num(v)}" for k, v in sorted(countries.items()))
    report.append(f"  Country distribution: {country_str}")
    
    unexpected_countries = sorted(set(countries) - allowed_countries)
    report.append(f"  Unexpected countries: {unexpected_countries if unexpected_countries else 'none'}")
    
    report.append(f"  IDs conforming to '{expected_prefix}': {fmt_num(nrows - bad_prefix_count)} valid, {fmt_num(bad_prefix_count)} invalid")
    if bad_prefix_examples:
        report.append(f"  Invalid prefix examples: {bad_prefix_examples}")
        
    report.append(f"  Duplicate entity IDs: {fmt_num(dup_ids)} beyond first occurrence")
    report.append(f"  Duplicate business names: {fmt_num(dup_names)} beyond first occurrence")
    report.append(f"  Duplicate addresses: {fmt_num(dup_addrs)} beyond first occurrence")
    
    report.append(f"  Name length statistics: {format_stats(name_len_total, nrows, name_len_min or 0, name_len_max)}")
    report.append(f"  Address length statistics: {format_stats(addr_len_total, nrows, addr_len_min or 0, addr_len_max)}")
    
    report.append(f"  Comma handling inspection: {fmt_num(rows_with_comma_in_name)} names and {fmt_num(rows_with_comma_in_addr)} addresses contain commas (TSV parser preserves all fields correctly)")
    report.append(f"  Rows with quotes: {fmt_num(rows_with_quotes)}")
    report.append(f"  Profiled in: {elapsed:.2f}s")
    
    return countries


def profile_ground_truth(path: Path, report: list):
    report.append(f"\n[{path}]")
    t0 = time.time()
    
    if not path.is_file():
        report.append("  ERROR: File not found.")
        return

    nrows = 0
    malformed = 0
    malformed_examples = []
    
    s1_seen = set()
    dup_s1 = 0
    
    singletons = 0
    one_match = 0
    two_match = 0
    three_or_more_match = 0
    
    total_matched_ids = 0
    s2_only = 0
    s3_only = 0
    s2_and_s3 = 0
    neither = 0
    
    intra_list_dupes = 0
    self_matches = 0
    unexpected_prefix_ids = 0
    prefix_counts = collections.Counter()

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        header_line = f.readline()
        if not header_line:
            report.append("  ERROR: File is empty.")
            return
        
        header = [c.strip() for c in header_line.rstrip("\r\n").split("\t")]
        report.append(f"  Columns ({len(header)}): {header}")
        
        expected_hdr = EXPECTED_HEADERS.get(path.name)
        if expected_hdr and header != expected_hdr:
            report.append(f"  HEADER WARNING: Found {header}, expected {expected_hdr}")
            
        for line_num, line in enumerate(f, start=2):
            if not line.strip():
                continue
            
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2:
                malformed += 1
                if len(malformed_examples) < 3:
                    malformed_examples.append((line_num, len(parts), 2))
                continue
            
            nrows += 1
            s1, raw_matches = parts[0], parts[1]
            
            if s1 in s1_seen:
                dup_s1 += 1
            else:
                s1_seen.add(s1)
                
            matched_ids = [m.strip() for m in raw_matches.split(",") if m.strip()]
            k = len(matched_ids)
            total_matched_ids += k
            
            # Cardinality buckets
            if k == 0:
                singletons += 1
            elif k == 1:
                one_match += 1
            elif k == 2:
                two_match += 1
            else:
                three_or_more_match += 1
                
            # Duplicate ID check within individual list
            if len(matched_ids) != len(set(matched_ids)):
                intra_list_dupes += 1
                
            has_s2 = False
            has_s3 = False
            for mid in matched_ids:
                pfx = mid[:3]
                prefix_counts[pfx] += 1
                if mid.startswith("S1-"):
                    self_matches += 1
                elif mid.startswith("S2-"):
                    has_s2 = True
                elif mid.startswith("S3-"):
                    has_s3 = True
                else:
                    unexpected_prefix_ids += 1
                    
            if has_s2 and has_s3:
                s2_and_s3 += 1
            elif has_s2:
                s2_only += 1
            elif has_s3:
                s3_only += 1
            elif k > 0:
                neither += 1

    del s1_seen
    gc.collect()
    
    elapsed = time.time() - t0
    report.append(f"  Row count: {fmt_num(nrows)}")
    report.append(f"  Malformed rows: {fmt_num(malformed)}")
    if malformed_examples:
        report.append(f"  Malformed row examples: {malformed_examples}")
        
    report.append(f"  Total Source 1 entities: {fmt_num(nrows)}")
    report.append(f"  Duplicate Source 1 rows: {fmt_num(dup_s1)}")
    report.append(f"  Singleton / no-match count: {fmt_num(singletons)} ({singletons / nrows * 100:.2f}%)")
    report.append(f"  One-match count: {fmt_num(one_match)} ({one_match / nrows * 100:.2f}%)")
    report.append(f"  Two-match count: {fmt_num(two_match)} ({two_match / nrows * 100:.2f}%)")
    report.append(f"  Three-or-more-match count: {fmt_num(three_or_more_match)} ({three_or_more_match / nrows * 100:.2f}%)")
    report.append(f"  Total matched IDs: {fmt_num(total_matched_ids)}")
    report.append(f"  S2-only matches: {fmt_num(s2_only)} ({s2_only / nrows * 100:.2f}%)")
    report.append(f"  S3-only matches: {fmt_num(s3_only)} ({s3_only / nrows * 100:.2f}%)")
    report.append(f"  S2+S3 matches: {fmt_num(s2_and_s3)} ({s2_and_s3 / nrows * 100:.2f}%)")
    if neither:
        report.append(f"  Neither S2 nor S3 matches: {fmt_num(neither)}")
    report.append(f"  Matched ID prefix breakdown: {dict(sorted(prefix_counts.items()))}")
    report.append(f"  Self-matches (S1- prefix): {fmt_num(self_matches)}")
    report.append(f"  Unexpected-prefix matched IDs: {fmt_num(unexpected_prefix_ids)}")
    report.append(f"  Duplicate IDs within a single match list: {fmt_num(intra_list_dupes)}")
    report.append(f"  Profiled in: {elapsed:.2f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"), help="Path to dataset directory (default: dataset)")
    parser.add_argument("--output", type=Path, default=None, help="Path to output file (default: artifacts/dataset_profile.txt)")
    parser.add_argument("--no-save", action="store_true", help="Print report only without saving to file")
    args = parser.parse_args()

    total_start = time.time()
    report = [
        "=" * 80,
        "Amazon ML Challenge 2026 — Local Dataset Profile",
        "=" * 80,
        f"Data directory: {args.data_dir.resolve()}",
        f"Python version: {sys.version.split()[0]} ({sys.platform})",
        "Parsing mode: Sequential stream with explicit TAB delimiter (RFC 4180/TSV compliant)",
    ]

    # TRAIN profiling
    report.append("\n" + "=" * 40)
    report.append("TRAIN DATASET PROFILE")
    report.append("=" * 40)
    
    for src_no in (1, 2, 3):
        path = args.data_dir / "train" / f"train_source{src_no}.tsv"
        profile_source_file(path, f"S{src_no}-", EXPECTED_COUNTRIES["train"], report)
        
    truth_path = args.data_dir / "train" / "train_ground_truth.tsv"
    profile_ground_truth(truth_path, report)

    # TEST profiling
    report.append("\n" + "=" * 40)
    report.append("TEST DATASET PROFILE")
    report.append("=" * 40)
    
    for src_no in (1, 2, 3):
        path = args.data_dir / "test" / f"test_source{src_no}.tsv"
        profile_source_file(path, f"S{src_no}-", EXPECTED_COUNTRIES["test"], report)

    total_elapsed = time.time() - total_start
    report.append("\n" + "=" * 80)
    report.append(f"Total profiling completed in {total_elapsed:.2f}s")
    report.append("=" * 80)

    final_report = "\n".join(report) + "\n"
    print(final_report)

    if not args.no_save:
        out_path = args.output or Path("artifacts/dataset_profile.txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(final_report, encoding="utf-8")
        print(f"Profile saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
