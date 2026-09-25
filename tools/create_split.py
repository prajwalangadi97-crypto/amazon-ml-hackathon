#!/usr/bin/env python3
"""CLI utility to generate and verify a reproducible validation split.

Usage:
    python tools/create_split.py --val-ratio 0.2 --seed 42
    python tools/create_split.py --fast-benchmark 10000 --seed 42
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ground_truth import stream_ground_truth
from src.data.split import create_validation_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("dataset/train/train_ground_truth.tsv"),
        help="Path to train ground truth TSV (default: dataset/train/train_ground_truth.tsv)",
    )
    parser.add_argument(
        "--source1",
        type=Path,
        default=Path("dataset/train/train_source1.tsv"),
        help="Path to train source1 TSV for country metadata (default: dataset/train/train_source1.tsv)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of S1 entities to allocate to validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--fast-benchmark",
        type=int,
        default=None,
        help="If set, create a fast benchmark split with this number of total S1 entities",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/splits"),
        help="Directory to save split metadata (default: experiments/splits)",
    )
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("Amazon ML Challenge 2026 - Validation Split Generator")
    print("=" * 70)
    print(f"Ground Truth   : {args.ground_truth}")
    print(f"Validation Ratio: {args.val_ratio}")
    print(f"Random Seed    : {args.seed}")
    if args.fast_benchmark:
        print(f"Fast Benchmark : {args.fast_benchmark:,} S1 entities")

    # 1. Read S1 country metadata if available
    s1_metadata = {}
    if args.source1.is_file():
        print(f"Streaming country metadata from {args.source1}...")
        with args.source1.open("r", encoding="utf-8-sig") as f:
            next(f)  # header
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4:
                    s1_metadata[parts[0]] = {"country": parts[3]}

    # 2. Read ground truth
    print(f"Streaming ground truth from {args.ground_truth}...")
    gt_mapping = {}
    for s1_id, matches in stream_ground_truth(args.ground_truth):
        gt_mapping[s1_id] = matches
        if args.fast_benchmark and len(gt_mapping) >= args.fast_benchmark:
            break

    print(f"Loaded {len(gt_mapping):,} Source 1 entities in {time.time() - t0:.2f}s.")

    # 3. Create split
    split = create_validation_split(
        ground_truth_source=gt_mapping,
        s1_metadata=s1_metadata,
        val_ratio=args.val_ratio,
        seed=args.seed,
        stratify=True,
    )

    print()
    print(split.summary())

    # 4. Save split
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_benchmark_{args.fast_benchmark}" if args.fast_benchmark else "_full"
    out_file = args.output_dir / f"validation_split_seed{args.seed}{suffix}.json"
    split.save(out_file)
    print(f"\nSplit saved to: {out_file.resolve()}")
    print(f"Total time elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
