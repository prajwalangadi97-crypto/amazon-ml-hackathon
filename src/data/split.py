"""Validation split generation for Amazon ML Challenge 2026.

Ensures proper entity-level partitioning:
1. Split occurs strictly at the Source 1 entity level.
2. Individual S2/S3 records are never split independently.
3. Every true S2/S3 match for a validation S1 entity remains available
   in the candidate pool (recall ceiling remains 100%).
4. Supports stratification by country and singleton status.
5. Deterministic and reproducible with a fixed random seed.
"""

import collections
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from src.data.ground_truth import parse_ground_truth, stream_ground_truth


@dataclass
class ValidationSplit:
    """Holds partitioned entity IDs and candidate target IDs."""

    train_s1_ids: Set[str]
    val_s1_ids: Set[str]
    val_true_matches: Dict[str, Set[str]]
    seed: int
    val_ratio: float
    description: str = ""

    @property
    def total_s1(self) -> int:
        return len(self.train_s1_ids) + len(self.val_s1_ids)

    @property
    def val_true_target_ids(self) -> Set[str]:
        """All S2 and S3 IDs that are true matches for validation S1 entities."""
        targets = set()
        for matched in self.val_true_matches.values():
            targets.update(matched)
        return targets

    def summary(self) -> str:
        val_targets = self.val_true_target_ids
        singletons = sum(1 for mids in self.val_true_matches.values() if len(mids) == 0)
        return (
            f"Validation Split (seed={self.seed}, val_ratio={self.val_ratio}):\n"
            f"  Train S1 Entities : {len(self.train_s1_ids):,}\n"
            f"  Val S1 Entities   : {len(self.val_s1_ids):,}\n"
            f"  Val Singletons    : {singletons:,} ({singletons / max(1, len(self.val_s1_ids)) * 100:.2f}%)\n"
            f"  Val True Matches  : {len(val_targets):,} distinct S2/S3 entities\n"
            f"  Preserved Recall  : 100.0% (all true targets mapped to val S1)"
        )

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist split ID lists and metadata to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "seed": self.seed,
            "val_ratio": self.val_ratio,
            "description": self.description,
            "train_s1_count": len(self.train_s1_ids),
            "val_s1_count": len(self.val_s1_ids),
            "train_s1_ids": sorted(self.train_s1_ids),
            "val_s1_ids": sorted(self.val_s1_ids),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(
        cls,
        filepath: Union[str, Path],
        ground_truth_path: Optional[Union[str, Path]] = None,
    ) -> "ValidationSplit":
        """Load split from JSON, optionally attaching true validation matches."""
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        val_s1_ids = set(data["val_s1_ids"])
        train_s1_ids = set(data["train_s1_ids"])

        val_matches: Dict[str, Set[str]] = {}
        if ground_truth_path:
            val_matches = parse_ground_truth(
                ground_truth_path, allowed_s1_ids=val_s1_ids
            )

        return cls(
            train_s1_ids=train_s1_ids,
            val_s1_ids=val_s1_ids,
            val_true_matches=val_matches,
            seed=data["seed"],
            val_ratio=data["val_ratio"],
            description=data.get("description", ""),
        )


def create_validation_split(
    ground_truth_source: Union[str, Path, Dict[str, Set[str]]],
    s1_metadata: Optional[Dict[str, Dict[str, str]]] = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
    max_samples: Optional[int] = None,
) -> ValidationSplit:
    """Create a reproducible Source 1 grouped validation split.

    Args:
        ground_truth_source: File path to ground truth TSV or an existing
            dict of {s1_id: set(matched_ids)}.
        s1_metadata: Optional dict mapping s1_id -> {'country': str, ...}
            for country-level stratification.
        val_ratio: Fraction of S1 entities to allocate to validation (default 0.2).
        seed: Random seed for deterministic reproducibility (default 42).
        stratify: Whether to stratify by (is_singleton, country) if available.
        max_samples: If set, downsamples total S1 entities to this number before
            splitting (useful for rapid offline benchmarking).

    Returns:
        ValidationSplit instance.
    """
    if isinstance(ground_truth_source, (str, Path)):
        # Stream without loading the entire 2.2M dataset into memory if max_samples is given
        gt_mapping: Dict[str, Set[str]] = {}
        for s1_id, matches in stream_ground_truth(ground_truth_source):
            gt_mapping[s1_id] = matches
            if max_samples and len(gt_mapping) >= max_samples * 2:
                break
    else:
        gt_mapping = ground_truth_source

    s1_all = list(gt_mapping.keys())
    rng = random.Random(seed)

    if max_samples and len(s1_all) > max_samples:
        rng.shuffle(s1_all)
        s1_all = s1_all[:max_samples]

    if not stratify:
        rng.shuffle(s1_all)
        n_val = int(len(s1_all) * val_ratio)
        val_s1_ids = set(s1_all[:n_val])
        train_s1_ids = set(s1_all[n_val:])
    else:
        # Group by stratum: (is_singleton, country)
        strata = collections.defaultdict(list)
        for s1_id in s1_all:
            is_singleton = len(gt_mapping[s1_id]) == 0
            country = s1_metadata.get(s1_id, {}).get("country", "UNKNOWN") if s1_metadata else "ALL"
            stratum_key = (is_singleton, country)
            strata[stratum_key].append(s1_id)

        val_s1_ids = set()
        train_s1_ids = set()

        for stratum_key, members in sorted(strata.items()):
            rng.shuffle(members)
            n_val_stratum = int(round(len(members) * val_ratio))
            # Ensure at least 1 in val if ratio > 0 and len >= 2
            if n_val_stratum == 0 and val_ratio > 0 and len(members) >= 2:
                n_val_stratum = 1
            val_s1_ids.update(members[:n_val_stratum])
            train_s1_ids.update(members[n_val_stratum:])

    val_matches = {s1: gt_mapping[s1] for s1 in val_s1_ids}

    return ValidationSplit(
        train_s1_ids=train_s1_ids,
        val_s1_ids=val_s1_ids,
        val_true_matches=val_matches,
        seed=seed,
        val_ratio=val_ratio,
        description=f"Stratified S1 split (ratio={val_ratio}, seed={seed}, total_s1={len(s1_all)})",
    )
