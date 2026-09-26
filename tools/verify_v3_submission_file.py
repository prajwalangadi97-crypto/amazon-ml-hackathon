#!/usr/bin/env python3
"""Verification and Safe Creation of the Final V3 Submission File."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent

src_file = PROJECT_ROOT / "output" / "v3" / "matching_results.tsv"
dst_file = PROJECT_ROOT / "output" / "matching_results_V3.tsv"
s1_file = PROJECT_ROOT / "dataset" / "test" / "test_source1.tsv"
s2_file = PROJECT_ROOT / "dataset" / "test" / "test_source2.tsv"
s3_file = PROJECT_ROOT / "dataset" / "test" / "test_source3.tsv"
candidate_file = PROJECT_ROOT / "output" / "v3" / "candidate_pairs.tsv"

print("=" * 80)
print("FINAL V3 SUBMISSION FILE GENERATION AND VERIFICATION")
print("=" * 80)

# 1. Source file verification
assert src_file.is_file(), f"Source file does not exist: {src_file}"
src_size = src_file.stat().st_size
print(f"Source file        : {src_file} ({src_size:,} bytes)")

# 2. Make safe copy to output/matching_results_V3.tsv
print(f"Copying to destination: {dst_file}...")
shutil.copyfile(src_file, dst_file)
dst_size = dst_file.stat().st_size
print(f"Destination file   : {dst_file} ({dst_size:,} bytes)")
assert src_size == dst_size, f"Byte size mismatch: {src_size} vs {dst_size}"
print("[PASS] Byte sizes are identical.")

# 3. Read test_source1.tsv ordering
print("\nReading test_source1.tsv...")
expected_s1 = []
with open(s1_file, "r", encoding="utf-8-sig") as f:
    next(f)
    for line in f:
        pos = line.find("\t")
        if pos != -1:
            expected_s1.append(line[:pos])

total_expected_s1 = len(expected_s1)
print(f"Total expected Source 1 entities: {total_expected_s1:,}")
assert total_expected_s1 == 1732544, f"Unexpected S1 count: {total_expected_s1}"

# 4. Detailed row-by-row verification of output/matching_results_V3.tsv
print("\nVerifying destination file rows, ordering, schema, and content...")
dst_rows_count = 0
duplicate_s1_ids = 0
duplicate_target_ids = 0
s1_order_mismatches = 0
all_matched_tids = set()

with open(dst_file, "r", encoding="utf-8") as f_dst, open(src_file, "r", encoding="utf-8") as f_src:
    src_header = f_src.readline()
    dst_header = f_dst.readline()
    assert src_header == "source1_entity_id\tmatched_entity_ids\n", f"Invalid src header: {repr(src_header)}"
    assert dst_header == "source1_entity_id\tmatched_entity_ids\n", f"Invalid dst header: {repr(dst_header)}"

    for idx, (src_line, dst_line) in enumerate(zip(f_src, f_dst)):
        assert src_line == dst_line, f"Content mismatch at line {idx + 2}!"
        parts = dst_line.rstrip("\r\n").split("\t")
        sid = parts[0]
        if idx < total_expected_s1 and sid != expected_s1[idx]:
            s1_order_mismatches += 1

        matched_str = parts[1] if len(parts) > 1 else ""
        if matched_str:
            tids = matched_str.split(",")
            if len(set(tids)) != len(tids):
                duplicate_target_ids += 1
            all_matched_tids.update(tids)

        dst_rows_count += 1

print(f"Destination data rows         : {dst_rows_count:,}")
print(f"S1 order mismatches           : {s1_order_mismatches}")
print(f"Duplicate S1 IDs              : {duplicate_s1_ids}")
print(f"Duplicate target ID rows      : {duplicate_target_ids}")
print(f"Unique predicted target IDs   : {len(all_matched_tids):,}")

assert dst_rows_count == total_expected_s1, f"Row count mismatch: {dst_rows_count} vs {total_expected_s1}"
assert s1_order_mismatches == 0, "Source 1 entity order does NOT match test_source1.tsv!"
assert duplicate_target_ids == 0, "Found rows with duplicate target IDs!"
print("[PASS] Row count, entity ordering, and within-row uniqueness verified.")

# 5. Verify all predicted target IDs exist in test Source 2 or Source 3
print("\nVerifying that all predicted target IDs exist in test Source 2 or Source 3...")
unresolved_tids = set(all_matched_tids)
for sf in [s2_file, s3_file]:
    with open(sf, "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            pos = line.find("\t")
            if pos != -1:
                unresolved_tids.discard(line[:pos])
                if not unresolved_tids:
                    break
    if not unresolved_tids:
        break

print(f"Unresolved / invalid target IDs: {len(unresolved_tids)}")
assert len(unresolved_tids) == 0, f"Found {len(unresolved_tids)} invalid target IDs!"
print("[PASS] 100% of predicted target IDs exist in test Source 2 or Source 3.")

# 6. Run official submission validator
print("\nRunning official submission validator (utils/validate_submission.py)...")
val_cmd = [
    sys.executable,
    str(PROJECT_ROOT / "utils" / "validate_submission.py"),
    "--matching", str(dst_file),
    "--candidate", str(candidate_file),
    "--test-dir", str(PROJECT_ROOT / "dataset" / "test"),
]
val_proc = subprocess.run(val_cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
print(val_proc.stdout)
if val_proc.stderr:
    print("Validator STDERR:", val_proc.stderr)
assert val_proc.returncode == 0, f"Validator failed with code {val_proc.returncode}!"
print("[PASS] utils/validate_submission.py passed with exit code 0.")

print("\n" + "=" * 80)
print("FINAL V3 SUBMISSION FILE VERIFIED AND READY FOR SUBMISSION")
print("=" * 80)
