#!/usr/bin/env python3
"""Persistent Target Index Builder and Integrity Auditor for Amazon ML Challenge 2026.

Builds country-partitioned, persistent target inverted indexes for:
- France (reused or audited from scratch/target_index_france.pkl)
- US (built incrementally from test_source2 + test_source3)
- India (built incrementally from test_source2 + test_source3)

Features:
- Full EXP-006B key extraction parity (exact name, PIN composites, numeric street words,
  relaxed address, name-locality composites, salient token pairs, and compound plot/door numbers).
- Memory-bounded streaming: no raw datasets held in RAM; utilizes compact array.array('I') postings.
- Safe atomic persistence: writes to temporary files first, then atomically renames.
- Comprehensive post-build integrity audit: record count, channels, sample lookups, latency,
  memory profiling, and SHA-256 manifest generation.
"""

import argparse
import array
from collections import Counter, defaultdict
import ctypes
from ctypes import wintypes
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking.normalizers import (
    extract_address_keys,
    extract_token_blocking_keys,
    normalize_name_conservative,
)
from src.blocking.retrieval import (
    extract_compound_number_keys,
    extract_record_blocking_keys,
)
from tools.run_recovery_experiments import (
    extract_h1_relaxed_address_keys,
    extract_h3_name_locality_keys,
)

# Expected Target Counts from test_source2 + test_source3
EXPECTED_TARGET_COUNTS = {
    "France": 1_434_993,
    "US": 3_817_031,
    "India": 4_717_565,
}

EXPECTED_S1_COUNTS = {
    "France": 259_452,
    "US": 663_106,
    "India": 809_986,
}


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


def get_current_ram_mb() -> float:
    """Return current working set RAM in MB via Windows API or fallback."""
    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        return 0.0


def get_peak_ram_mb() -> float:
    """Return peak working set RAM in MB via Windows API or fallback."""
    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / (1024 * 1024)
    except Exception:
        return 0.0


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file using 1MB chunk streaming."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def extract_target_blocking_keys(
    name: str,
    addr: str,
    country: str,
    include_compound_numbers: bool = True,
) -> List[str]:
    """Extract all active blocking keys for a target record under EXP-006B logic.

    Uses extract_record_blocking_keys from src/blocking/retrieval.py for guaranteed parity.
    """
    keys_dict = extract_record_blocking_keys(
        name=name,
        address=addr,
        country=country,
        include_compound_numbers=include_compound_numbers,
    )
    return list(keys_dict.keys())


def build_country_target_index(
    country: str,
    s2_path: Path,
    s3_path: Path,
    output_path: Path,
    include_compound_numbers: bool = True,
) -> Dict[str, Any]:
    """Incrementally build and safely persist a country-partitioned target inverted index."""
    print("=" * 90, flush=True)
    print(f"BUILDING TARGET INVERTED INDEX FOR COUNTRY: {country}", flush=True)
    print("=" * 90, flush=True)

    t0 = time.time()
    target_ids: List[str] = []
    inverted_index: Dict[str, array.array] = defaultdict(lambda: array.array("I"))

    total_postings = 0
    records_indexed = 0
    country_tag = f"\t{country}"

    for src_file in [s2_path, s3_path]:
        print(f"  Streaming {src_file.name} for {country} targets...", flush=True)
        t_src_start = time.time()
        file_records = 0
        with open(src_file, "r", encoding="utf-8-sig") as f:
            next(f)  # Header
            for line in f:
                if country_tag not in line:
                    continue
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4:
                    continue
                tid, tname, taddr, tcountry = parts[0], parts[1], parts[2], parts[3]
                if tcountry != country:
                    continue

                t_idx = len(target_ids)
                target_ids.append(tid)

                keys = extract_target_blocking_keys(
                    name=tname,
                    addr=taddr,
                    country=country,
                    include_compound_numbers=include_compound_numbers,
                )
                for k in keys:
                    inverted_index[k].append(t_idx)
                    total_postings += 1

                records_indexed += 1
                file_records += 1

                if records_indexed % 500_000 == 0:
                    elapsed = time.time() - t0
                    print(
                        f"    [Progress] {records_indexed:,} targets indexed | "
                        f"{len(inverted_index):,} keys | {total_postings:,} postings | "
                        f"RAM: {get_current_ram_mb():.1f} MB (Peak: {get_peak_ram_mb():.1f} MB) | "
                        f"Elapsed: {elapsed:.1f}s ({elapsed / 60:.2f}m)",
                        flush=True,
                    )

        t_src = time.time() - t_src_start
        print(f"  Finished {src_file.name}: {file_records:,} records in {t_src:.1f}s.", flush=True)

    build_time = time.time() - t0
    peak_ram = get_peak_ram_mb()
    unique_keys = len(inverted_index)

    print(f"\nIndex In-Memory Summary for {country}:")
    print(f"  Total Targets Indexed : {records_indexed:,} (Expected: {EXPECTED_TARGET_COUNTS[country]:,})")
    print(f"  Unique Blocking Keys  : {unique_keys:,}")
    print(f"  Total Postings        : {total_postings:,}")
    print(f"  Build Runtime         : {build_time:.2f}s ({build_time / 60:.2f} min)")
    print(f"  Peak RAM              : {peak_ram:.1f} MB")

    assert records_indexed == EXPECTED_TARGET_COUNTS[country], (
        f"Target record count mismatch for {country}: "
        f"got {records_indexed:,}, expected {EXPECTED_TARGET_COUNTS[country]:,}"
    )

    # Atomic persistence
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".pkl.tmp")

    print(f"\nSerializing index to temporary checkpoint: {tmp_path}...", flush=True)
    t0_save = time.time()
    with open(tmp_path, "wb") as f:
        pickle.dump(
            {
                "target_ids": target_ids,
                "inverted_index": dict(inverted_index),
            },
            f,
            protocol=5,
        )
    save_time = time.time() - t0_save
    tmp_size = os.path.getsize(tmp_path)
    print(f"  Saved in {save_time:.2f}s ({tmp_size / (1024 * 1024):.2f} MB).", flush=True)

    # Atomic rename
    print(f"Atomically renaming {tmp_path.name} -> {output_path.name}...", flush=True)
    if output_path.exists():
        output_path.unlink()
    os.replace(tmp_path, output_path)

    final_size = os.path.getsize(output_path)
    print(f"Index successfully persisted: {output_path} ({final_size / (1024 * 1024):.2f} MB)", flush=True)

    return {
        "country": country,
        "records_indexed": records_indexed,
        "unique_keys": unique_keys,
        "total_postings": total_postings,
        "build_time_sec": build_time,
        "save_time_sec": save_time,
        "total_time_sec": build_time + save_time,
        "peak_ram_mb": peak_ram,
        "file_size_bytes": final_size,
    }


def audit_index_integrity(
    country: str,
    index_path: Path,
    s1_path: Path,
    sample_queries: int = 1_000,
) -> Dict[str, Any]:
    """Perform rigorous post-build integrity audit and lookup benchmark."""
    print("\n" + "-" * 90, flush=True)
    print(f"INTEGRITY AUDIT & BENCHMARK: {country} ({index_path.name})", flush=True)
    print("-" * 90, flush=True)

    assert index_path.is_file(), f"Index file missing: {index_path}"
    file_size_bytes = os.path.getsize(index_path)

    # 1. Load test
    t0_load = time.time()
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    load_time = time.time() - t0_load

    target_ids: List[str] = data["target_ids"]
    inverted_index: Dict[str, array.array] = data["inverted_index"]

    print(f"  Load Time              : {load_time:.2f}s")
    print(f"  File Size              : {file_size_bytes / (1024 * 1024):.2f} MB ({file_size_bytes:,} bytes)")
    print(f"  Total Indexed Targets  : {len(target_ids):,}")
    print(f"  Total Blocking Keys    : {len(inverted_index):,}")

    # Check record count
    expected_targets = EXPECTED_TARGET_COUNTS[country]
    assert len(target_ids) == expected_targets, (
        f"Audit Failed: target count {len(target_ids):,} != expected {expected_targets:,}"
    )

    # 2. Key/channel breakdown
    channel_counts = Counter()
    cnum_keys = 0
    total_postings = 0
    max_posting = 0
    max_posting_key = ""

    for k, post in inverted_index.items():
        n = len(post)
        total_postings += n
        if n > max_posting:
            max_posting = n
            max_posting_key = k

        if k.startswith("en_"):
            channel_counts["exact_name"] += 1
        elif k.startswith("tk_"):
            channel_counts["name_token"] += 1
        elif k.startswith("cnum_"):
            channel_counts["compound_number"] += 1
            cnum_keys += 1
        elif "pin_" in k:
            channel_counts["pin_composite"] += 1
        elif "loc_" in k:
            channel_counts["locality"] += 1
        elif "nword_" in k or "num_" in k:
            channel_counts["numeric_street"] += 1
        else:
            channel_counts["other_address"] += 1

    print("\n  Channel Breakdown:")
    for ch, cnt in sorted(channel_counts.items(), key=lambda x: -x[1]):
        print(f"    {ch:<20}: {cnt:,} unique keys")
    print(f"    Compound Number Keys : {cnum_keys:,}")
    print(f"    Max Key Fanout       : {max_posting:,} (key: '{max_posting_key[:40]}...')")

    # 3. Duplicate handling
    unique_tids = set(target_ids)
    assert len(unique_tids) == len(target_ids), "Audit Failed: Duplicate target IDs found in index!"

    # 4. Lookup latency benchmark on real S1 records
    print(f"\n  Running S1 lookup benchmark on {sample_queries:,} representative queries...", flush=True)
    s1_samples: List[Tuple[str, str, str]] = []
    country_tag = f"\t{country}"

    with open(s1_path, "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            if country_tag not in line:
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and parts[3] == country:
                s1_samples.append((parts[0], parts[1], parts[2]))
                if len(s1_samples) >= sample_queries:
                    break

    assert len(s1_samples) == sample_queries, f"Not enough S1 records for {country}: got {len(s1_samples)}"

    t0_lookup = time.time()
    total_candidates_found = 0
    zero_hits = 0

    for sid, name, addr in s1_samples:
        keys = extract_target_blocking_keys(
            name=name,
            addr=addr,
            country=country,
            include_compound_numbers=True,
        )
        hits: Set[int] = set()
        for k in keys:
            post = inverted_index.get(k)
            if post:
                hits.update(post)
        if not hits:
            zero_hits += 1
        total_candidates_found += len(hits)

    lookup_time = time.time() - t0_lookup
    throughput = sample_queries / max(lookup_time, 0.001)
    latency_ms = (lookup_time / sample_queries) * 1000.0
    avg_raw_candidates = total_candidates_found / sample_queries

    print(f"  Lookup Runtime         : {lookup_time:.3f}s for {sample_queries:,} queries")
    print(f"  Lookup Throughput      : {throughput:,.1f} queries / sec")
    print(f"  Average Latency        : {latency_ms:.3f} ms / query")
    print(f"  Avg Raw Candidates/S1  : {avg_raw_candidates:.1f}")
    print(f"  Zero Hits Count        : {zero_hits:,} ({zero_hits / sample_queries * 100:.2f}%)")

    # 5. Check SHA-256
    sha256 = compute_file_sha256(index_path)
    print(f"  SHA-256 Checksum       : {sha256}")
    print("  [PASS] Integrity Audit Passed Successfully!")

    return {
        "country": country,
        "status": "VERIFIED",
        "index_path": str(index_path.relative_to(PROJECT_ROOT)),
        "file_size_bytes": file_size_bytes,
        "sha256": sha256,
        "load_time_sec": load_time,
        "indexed_records": len(target_ids),
        "unique_keys": len(inverted_index),
        "total_postings": total_postings,
        "channel_counts": dict(channel_counts),
        "compound_number_keys": cnum_keys,
        "lookup_benchmark": {
            "queries_benchmarked": sample_queries,
            "runtime_sec": lookup_time,
            "throughput_queries_per_sec": throughput,
            "latency_ms_per_query": latency_ms,
            "avg_raw_candidates": avg_raw_candidates,
            "zero_hits_count": zero_hits,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--countries",
        nargs="+",
        default=["US", "India"],
        help="List of countries to build indexes for (e.g. US India)",
    )
    parser.add_argument(
        "--audit-france",
        action="store_true",
        default=True,
        help="Audit the existing scratch/target_index_france.pkl without overwriting",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("artifacts/target_index_manifest.json"),
    )
    args = parser.parse_args()

    s2_fp = PROJECT_ROOT / "dataset" / "test" / "test_source2.tsv"
    s3_fp = PROJECT_ROOT / "dataset" / "test" / "test_source3.tsv"
    s1_fp = PROJECT_ROOT / "dataset" / "test" / "test_source1.tsv"

    assert s2_fp.is_file(), f"Missing {s2_fp}"
    assert s3_fp.is_file(), f"Missing {s3_fp}"
    assert s1_fp.is_file(), f"Missing {s1_fp}"

    overall_start = time.time()
    manifest_entries: Dict[str, Any] = {}

    # Load existing manifest if present
    if args.manifest_path.is_file():
        try:
            with open(args.manifest_path, "r", encoding="utf-8") as f:
                manifest_entries = json.load(f)
        except Exception:
            manifest_entries = {}

    # 1. Audit / reuse existing France index
    france_scratch = PROJECT_ROOT / "scratch" / "target_index_france.pkl"
    france_artifact = PROJECT_ROOT / "artifacts" / "target_index_france.pkl"

    if args.audit_france and france_scratch.is_file():
        print("\n" + "=" * 90, flush=True)
        print("AUDITING EXISTING FRANCE TARGET INDEX (DO NOT DELETE/OVERWRITE)", flush=True)
        print("=" * 90, flush=True)
        fr_audit = audit_index_integrity("France", france_scratch, s1_fp)

        # Copy safely to artifacts/ if not already present, preserving scratch intact
        if not france_artifact.is_file():
            print(f"Creating immutable reference copy in {france_artifact}...", flush=True)
            with open(france_scratch, "rb") as f_src, open(france_artifact, "wb") as f_dst:
                shutil_copy = f_src.read()
                f_dst.write(shutil_copy)
            print(f"Preserved original {france_scratch} and mirrored to {france_artifact}.", flush=True)

        manifest_entries["France"] = {
            "country": "France",
            "source_files": ["dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"],
            "index_path": str(france_artifact.relative_to(PROJECT_ROOT)),
            "scratch_index_path": str(france_scratch.relative_to(PROJECT_ROOT)),
            "record_count": fr_audit["indexed_records"],
            "byte_size": fr_audit["file_size_bytes"],
            "sha256": fr_audit["sha256"],
            "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(os.path.getmtime(france_scratch))),
            "retrieval_version": "EXP-006B",
            "compound_numbers_enabled": False,  # France has virtually 0 compound numbers
            "validation_status": "VERIFIED",
            "audit_metrics": fr_audit,
        }

    # 2. Build target indexes for missing countries (US, India)
    for country in args.countries:
        idx_path = PROJECT_ROOT / "artifacts" / f"target_index_{country.lower()}.pkl"

        if idx_path.is_file():
            print(f"\nTarget index for {country} already exists at {idx_path}. Verifying integrity...", flush=True)
            audit_res = audit_index_integrity(country, idx_path, s1_fp)
        else:
            build_res = build_country_target_index(
                country=country,
                s2_path=s2_fp,
                s3_path=s3_fp,
                output_path=idx_path,
                include_compound_numbers=True,
            )
            audit_res = audit_index_integrity(country, idx_path, s1_fp)
            audit_res["build_metrics"] = build_res

        manifest_entries[country] = {
            "country": country,
            "source_files": ["dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"],
            "index_path": str(idx_path.relative_to(PROJECT_ROOT)),
            "record_count": audit_res["indexed_records"],
            "byte_size": audit_res["file_size_bytes"],
            "sha256": audit_res["sha256"],
            "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "retrieval_version": "EXP-006B",
            "compound_numbers_enabled": True,
            "validation_status": "VERIFIED",
            "audit_metrics": audit_res,
        }

    # 3. Write manifest
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "manifest_version": "1.0",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_target_records": sum(entry["record_count"] for entry in manifest_entries.values()),
        "expected_total_targets": 9_969_589,
        "status": "ALL_INDEXES_VERIFIED",
        "indexes": manifest_entries,
    }
    with open(args.manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    total_wall = time.time() - overall_start
    print("\n" + "=" * 90, flush=True)
    print("ALL TARGET INDEXES SUCCESSFULLY BUILT, AUDITED, AND MANIFESTED!", flush=True)
    print(f"Manifest written to: {args.manifest_path.resolve()}", flush=True)
    print(f"Total Pipeline Runtime: {total_wall:.2f}s ({total_wall / 60:.2f} min)", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
