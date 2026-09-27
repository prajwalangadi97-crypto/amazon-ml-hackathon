#!/usr/bin/env python3
"""Count records per country in test_source1, test_source2, and test_source3."""
from collections import Counter
from pathlib import Path

DATA_DIR = Path("dataset/test")

for fname in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
    fp = DATA_DIR / fname
    counts = Counter()
    total = 0
    with open(fp, "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            total += 1
            pos = line.rfind("\t")
            if pos != -1:
                c = line[pos+1:].rstrip("\r\n")
                counts[c] += 1
    print(f"=== {fname}: total {total:,} ===")
    for c, cnt in counts.most_common():
        print(f"  {c}: {cnt:,}")
