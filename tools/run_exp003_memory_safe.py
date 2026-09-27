#!/usr/bin/env python3
"""Amazon ML Challenge 2026 â€” EXP-003: Memory-Safe Two-Stage ML Baseline.

DESIGN:
  Stage 1 â€” Build inverted blocking index (exact-name, token-pair, address) over the
             3,501 relevant S1 entities (1,500 train + 2,001 val). O(S1) memory.
  Stage 2 â€” Stream S2 and S3 records, emit candidate (s1_id, tgt_id) pairs LAZILY.
             Process in BATCH_SIZE chunks: compute features â†’ score â†’ keep only
             compact predictions. Never accumulate 20 M feature rows in RAM.

MEMORY BUDGET (approximate):
  - Blocking index for 3,501 S1 entities: < 200 MB
  - Per-batch feature matrix (BATCH_SIZE Ã— 15 float32): < 10 MB per batch
  - Per-S1 prediction dict:  max ~2,001 sets of ints/strings, negligible

COMMAND LINE:
  python tools/run_exp003_memory_safe.py
  python tools/run_exp003_memory_safe.py --smoke   # 10 000-candidate smoke test only
  python tools/run_exp003_memory_safe.py --model logistic
  python tools/run_exp003_memory_safe.py --model gbdt

REPRODUCIBILITY: random_state=42, fixed split file, fixed training slice.
"""

import argparse
import gc
import json
import os
import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Set, Tuple

import numpy as np

# â”€â”€ Project root on path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking.normalizers import (
    extract_address_keys,
    extract_token_blocking_keys,
    normalize_name_conservative,
)
from src.data.ground_truth import parse_ground_truth
from src.evaluation.metrics import evaluate_predictions
from src.experiments.logger import ExperimentLogger, ExperimentRecord
from src.matching.features import FEATURE_NAMES, extract_pair_features
from src.matching.model import EntityMatchingClassifier

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BATCH_SIZE = 65_536          # ~65 k pairs per batch; 65 k Ã— 15 floats â‰ˆ 4 MB
TRAIN_S1_SAMPLE = 1_500      # first N train S1 used for training pairs
MAX_NEG_PER_S1 = 25          # hard-negative cap per train S1 entity
RANDOM_SEED = 42

SPLIT_FILE = "experiments/splits/validation_split_seed42_benchmark_10000.json"
GT_PATH = Path("dataset/train/train_ground_truth.tsv")
S1_TRAIN_PATH = Path("dataset/train/train_source1.tsv")
S2_TRAIN_PATH = Path("dataset/train/train_source2.tsv")
S3_TRAIN_PATH = Path("dataset/train/train_source3.tsv")

THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# â”€â”€ Smoke-test constants (fast sanity check only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SMOKE_TRAIN_S1   = 75        # deterministic first-N train S1 entities
SMOKE_VAL_S1     = 50        # deterministic first-N val S1 entities
SMOKE_SCAN_LIMIT = 500_000   # max source records to scan (~10% of S2, ~2 min)
SMOKE_NEG_CAP    = 15        # max negatives per train S1 in smoke mode

# â”€â”€ Types â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
S1Record = Tuple[str, str, str, str]   # (name, addr, country, norm_name)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 1. DATA LOADING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def load_split(split_path: str) -> Tuple[List[str], List[str]]:
    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["train_s1_ids"], data["val_s1_ids"]


def load_s1_records(path: Path, allowed_ids: Set[str]) -> Dict[str, S1Record]:
    """Load only the S1 records whose IDs are in allowed_ids."""
    records: Dict[str, S1Record] = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and parts[0] in allowed_ids:
                norm = normalize_name_conservative(parts[1])
                records[parts[0]] = (parts[1], parts[2], parts[3], norm)
    return records


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 2. BLOCKING INDEX
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def build_blocking_index(
    s1_records: Dict[str, S1Record]
) -> Tuple[Dict, Dict, Dict]:
    """Build three inverted indexes over the relevant S1 records.

    Returns:
        index_exact : (country, norm_name) â†’ [s1_id, ...]
        index_token : (country, token_key) â†’ [s1_id, ...]
        index_addr  : (country, addr_key)  â†’ [s1_id, ...]
    """
    index_exact: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    index_token: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    index_addr: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for s1_id, (name, addr, country, norm_name) in s1_records.items():
        if norm_name:
            index_exact[(country, norm_name)].append(s1_id)
        for tk in extract_token_blocking_keys(name):
            index_token[(country, tk)].append(s1_id)
        for ak in extract_address_keys(addr):
            index_addr[(country, ak)].append(s1_id)

    return index_exact, index_token, index_addr


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 3. LAZY CANDIDATE GENERATOR
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _candidate_s1_ids_for_target(
    name: str,
    addr: str,
    country: str,
    norm_name: str,
    index_exact: Dict,
    index_token: Dict,
    index_addr: Dict,
) -> Set[str]:
    """Return the set of matching S1 IDs for a single target record (Union-D)."""
    matched: Set[str] = set()
    if norm_name:
        hits = index_exact.get((country, norm_name))
        if hits:
            matched.update(hits)
    for tk in extract_token_blocking_keys(name):
        hits = index_token.get((country, tk))
        if hits:
            matched.update(hits)
    for ak in extract_address_keys(addr):
        hits = index_addr.get((country, ak))
        if hits:
            matched.update(hits)
    return matched


def stream_candidates(
    source_path: Path,
    index_exact: Dict,
    index_token: Dict,
    index_addr: Dict,
    s1_records: Dict[str, S1Record],
    smoke_limit: Optional[int] = None,
) -> Iterator[Tuple[str, str, List[float]]]:
    """Lazily yield (s1_id, tgt_id, feature_vector) tuples.

    Streams the source file record-by-record. For every target record that
    hits â‰¥ 1 S1 candidate, yields one tuple per (s1_id, tgt_id) pair.
    Features are computed immediately and yielded; no list accumulation.

    Args:
        source_path:  Path to train_source2.tsv or train_source3.tsv.
        smoke_limit:  If set, stop after this many candidate pairs yielded
                      (used for the smoke test).
    """
    yielded = 0
    with open(source_path, "r", encoding="utf-8-sig") as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            tgt_id, tgt_name, tgt_addr, tgt_country = (
                parts[0], parts[1], parts[2], parts[3]
            )
            tgt_norm = normalize_name_conservative(tgt_name)
            matched_s1_ids = _candidate_s1_ids_for_target(
                tgt_name, tgt_addr, tgt_country, tgt_norm,
                index_exact, index_token, index_addr,
            )
            if not matched_s1_ids:
                continue
            for s1_id in matched_s1_ids:
                s1 = s1_records[s1_id]
                feats = extract_pair_features(
                    s1[0], s1[1], s1[2],
                    tgt_id, tgt_name, tgt_addr, tgt_country,
                    norm_s1_name=s1[3], norm_tgt_name=tgt_norm,
                )
                yield s1_id, tgt_id, feats
                yielded += 1
                if smoke_limit is not None and yielded >= smoke_limit:
                    return


def stream_all_candidates(
    index_exact: Dict,
    index_token: Dict,
    index_addr: Dict,
    s1_records: Dict[str, S1Record],
    smoke_limit: Optional[int] = None,
) -> Iterator[Tuple[str, str, List[float]]]:
    """Chain generators over S2 and S3 (lazy composition, zero copy)."""
    n_yielded = 0
    for src_path in (S2_TRAIN_PATH, S3_TRAIN_PATH):
        remaining = None if smoke_limit is None else smoke_limit - n_yielded
        if remaining is not None and remaining <= 0:
            break
        for item in stream_candidates(
            src_path, index_exact, index_token, index_addr,
            s1_records, smoke_limit=remaining,
        ):
            yield item
            n_yielded += 1


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 4. TRAINING DATA COLLECTION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def collect_training_pairs(
    train_s1_set: Set[str],
    train_gt: Dict[str, Set[str]],
    index_exact: Dict,
    index_token: Dict,
    index_addr: Dict,
    s1_records: Dict[str, S1Record],
    max_neg_per_s1: int = MAX_NEG_PER_S1,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect training pairs by streaming S2 + S3 for train S1 entities only.

    Memory-safe: only feature vectors for the ~35 K training pairs are kept.

    Returns:
        X_train: float32 array (N, 15)
        y_train: int32 array  (N,)
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    feats_pos: List[List[float]] = []
    feats_neg: List[List[float]] = []
    neg_counts: Dict[str, int] = defaultdict(int)
    pos_set: Set[Tuple[str, str]] = set()   # dedup positives

    print("  Collecting training pairs from S2 + S3 (train S1 only)...")
    t0 = time.time()
    n_target_records = 0

    for src_path in (S2_TRAIN_PATH, S3_TRAIN_PATH):
        with open(src_path, "r", encoding="utf-8-sig") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4:
                    continue
                n_target_records += 1
                tgt_id, tgt_name, tgt_addr, tgt_country = (
                    parts[0], parts[1], parts[2], parts[3]
                )
                tgt_norm = normalize_name_conservative(tgt_name)
                matched_s1_ids = _candidate_s1_ids_for_target(
                    tgt_name, tgt_addr, tgt_country, tgt_norm,
                    index_exact, index_token, index_addr,
                )
                for s1_id in matched_s1_ids:
                    if s1_id not in train_s1_set:
                        continue
                    s1 = s1_records[s1_id]
                    is_pos = tgt_id in train_gt.get(s1_id, set())
                    feats = extract_pair_features(
                        s1[0], s1[1], s1[2],
                        tgt_id, tgt_name, tgt_addr, tgt_country,
                        norm_s1_name=s1[3], norm_tgt_name=tgt_norm,
                    )
                    if is_pos:
                        if (s1_id, tgt_id) not in pos_set:
                            pos_set.add((s1_id, tgt_id))
                            feats_pos.append(feats)
                    else:
                        if neg_counts[s1_id] < max_neg_per_s1:
                            feats_neg.append(feats)
                            neg_counts[s1_id] += 1

    elapsed = time.time() - t0
    print(f"  Scanned {n_target_records:,} target records in {elapsed:.1f}s.")
    print(f"  Positives : {len(feats_pos):,}")
    print(f"  Negatives : {len(feats_neg):,}")

    X_train = np.array(feats_pos + feats_neg, dtype=np.float32)
    y_train = np.array(
        [1] * len(feats_pos) + [0] * len(feats_neg), dtype=np.int32
    )
    return X_train, y_train


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 5. BATCH SCORING EVALUATOR
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def score_and_evaluate(
    clf: EntityMatchingClassifier,
    val_s1_set: Set[str],
    val_gt: Dict[str, Set[str]],
    index_exact: Dict,
    index_token: Dict,
    index_addr: Dict,
    s1_records: Dict[str, S1Record],
    thresholds: List[float],
    batch_size: int = BATCH_SIZE,
    smoke_limit: Optional[int] = None,
) -> Tuple[int, Dict[float, Dict]]:
    """Score validation candidates in batches; evaluate at every threshold.

    Memory design:
      - A rolling buffer `batch_pairs` and `batch_X` collect up to BATCH_SIZE
        pairs at a time, then predict and immediately discard the feature matrix.
      - Only a compact dict[s1_id â†’ {(tgt_id, prob)}] is kept in memory.
        This is O(val_candidates) entries of (str, float), not 20M Ã— 15 floats.

    Returns:
        n_candidates: total candidate pairs scored
        results: threshold â†’ {"predictions": {s1_id: set(tgt_id)}, ...metrics}
    """
    # Compact per-S1 probability lists (list of (tgt_id, prob) per s1_id)
    s1_to_probs: Dict[str, List[Tuple[str, float]]] = {s: [] for s in val_s1_set}

    # Limit the generator to val S1 only â€” build a restricted index
    val_index_exact: Dict = defaultdict(list)
    val_index_token: Dict = defaultdict(list)
    val_index_addr: Dict = defaultdict(list)
    for s1_id in val_s1_set:
        rec = s1_records[s1_id]
        name, addr, country, norm_name = rec
        if norm_name:
            val_index_exact[(country, norm_name)].append(s1_id)
        for tk in extract_token_blocking_keys(name):
            val_index_token[(country, tk)].append(s1_id)
        for ak in extract_address_keys(addr):
            val_index_addr[(country, ak)].append(s1_id)

    # --- Batch loop ---
    buf_s1: List[str] = []
    buf_tgt: List[str] = []
    buf_X: List[List[float]] = []
    n_candidates = 0

    def _flush_batch():
        nonlocal n_candidates
        if not buf_X:
            return
        X_batch = np.array(buf_X, dtype=np.float32)
        probs = clf.predict_proba(X_batch).tolist()
        for s1_id, tgt_id, prob in zip(buf_s1, buf_tgt, probs):
            if s1_id in s1_to_probs:
                s1_to_probs[s1_id].append((tgt_id, prob))
        n_candidates += len(buf_X)
        buf_s1.clear()
        buf_tgt.clear()
        buf_X.clear()
        # Release the batch numpy array immediately
        del X_batch
        gc.collect()

    gen = stream_all_candidates(
        val_index_exact, val_index_token, val_index_addr,
        {s: s1_records[s] for s in val_s1_set},
        smoke_limit=smoke_limit,
    )

    for s1_id, tgt_id, feats in gen:
        buf_s1.append(s1_id)
        buf_tgt.append(tgt_id)
        buf_X.append(feats)
        if len(buf_X) >= batch_size:
            _flush_batch()
    _flush_batch()   # flush remainder

    # --- Threshold sweep ---
    results: Dict[float, Dict] = {}
    for t in thresholds:
        preds: Dict[str, Set[str]] = {}
        for s1_id in val_s1_set:
            preds[s1_id] = {
                tgt for tgt, prob in s1_to_probs[s1_id] if prob >= t
            }
        report = evaluate_predictions(preds, val_gt)
        results[t] = {
            "macro_f0_5": report.macro_f0_5,
            "precision": report.macro_precision,
            "recall": report.macro_recall,
            "singleton_f0_5": report.singleton_f0_5,
            "non_singleton_f0_5": report.non_singleton_f0_5,
            "total_tp": report.total_tp,
            "total_fp": report.total_fp,
            "total_fn": report.total_fn,
            "predicted_matches": sum(len(v) for v in preds.values()),
        }

    return n_candidates, results


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 6. SMOKE TEST
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def run_smoke_test(
    clf: EntityMatchingClassifier,
    val_s1_set: Set[str],
    val_gt: Dict[str, Set[str]],
    index_exact: Dict,
    index_token: Dict,
    index_addr: Dict,
    s1_records: Dict[str, S1Record],
    n_smoke: int = 10_000,
) -> bool:
    """Sanity-check the pipeline end-to-end on the first 10,000 candidates.

    Checks:
      1. Generator produces valid (s1_id, tgt_id, feature_list) tuples.
      2. Feature vector is exactly len(FEATURE_NAMES) = 15 elements.
      3. Classifier output is a valid probability in [0, 1].
      4. Official F0.5 evaluator runs without error.

    Returns True if all checks pass.
    """
    print(f"\n{'='*70}")
    print(f"SMOKE TEST: first {n_smoke:,} validation candidates")
    print(f"{'='*70}")

    # Restricted val index
    val_idx_exact: Dict = defaultdict(list)
    val_idx_token: Dict = defaultdict(list)
    val_idx_addr: Dict = defaultdict(list)
    for s1_id in val_s1_set:
        rec = s1_records[s1_id]
        name, addr, country, norm_name = rec
        if norm_name:
            val_idx_exact[(country, norm_name)].append(s1_id)
        for tk in extract_token_blocking_keys(name):
            val_idx_token[(country, tk)].append(s1_id)
        for ak in extract_address_keys(addr):
            val_idx_addr[(country, ak)].append(s1_id)

    tracemalloc.start()
    t_start = time.time()

    gen = stream_all_candidates(
        val_idx_exact, val_idx_token, val_idx_addr,
        {s: s1_records[s] for s in val_s1_set},
        smoke_limit=n_smoke,
    )

    s1_ids_seen: Set[str] = set()
    tgt_ids_seen: Set[str] = set()
    feature_lens: Set[int] = set()
    n_pairs = 0
    all_probs: List[float] = []

    batch_s1: List[str] = []
    batch_tgt: List[str] = []
    batch_X: List[List[float]] = []

    def _check_batch():
        X = np.array(batch_X, dtype=np.float32)
        probs = clf.predict_proba(X).tolist()
        all_probs.extend(probs)
        s1_ids_seen.update(batch_s1)
        tgt_ids_seen.update(batch_tgt)
        feature_lens.add(X.shape[1])
        batch_s1.clear(); batch_tgt.clear(); batch_X.clear()
        del X; gc.collect()

    for s1_id, tgt_id, feats in gen:
        batch_s1.append(s1_id)
        batch_tgt.append(tgt_id)
        batch_X.append(feats)
        n_pairs += 1
        if len(batch_X) >= BATCH_SIZE:
            _check_batch()
    if batch_X:
        _check_batch()

    elapsed = time.time() - t_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- Mini evaluation pass ---
    smoke_preds: Dict[str, Set[str]] = {s: set() for s in val_s1_set}
    # Re-stream the first 10 000 candidates with a fixed threshold of 0.5
    gen2 = stream_all_candidates(
        val_idx_exact, val_idx_token, val_idx_addr,
        {s: s1_records[s] for s in val_s1_set},
        smoke_limit=n_smoke,
    )
    buf_s1_: List[str] = []; buf_tgt_: List[str] = []; buf_X_: List[List[float]] = []
    for s1_id, tgt_id, feats in gen2:
        buf_s1_.append(s1_id); buf_tgt_.append(tgt_id); buf_X_.append(feats)
        if len(buf_X_) >= BATCH_SIZE:
            probs = clf.predict_proba(np.array(buf_X_, dtype=np.float32)).tolist()
            for sid, tid, p in zip(buf_s1_, buf_tgt_, probs):
                if p >= 0.5:
                    smoke_preds[sid].add(tid)
            buf_s1_.clear(); buf_tgt_.clear(); buf_X_.clear()
    if buf_X_:
        probs = clf.predict_proba(np.array(buf_X_, dtype=np.float32)).tolist()
        for sid, tid, p in zip(buf_s1_, buf_tgt_, probs):
            if p >= 0.5:
                smoke_preds[sid].add(tid)

    smoke_report = evaluate_predictions(smoke_preds, val_gt)

    # --- Report ---
    all_ok = True
    print(f"  Pairs generated    : {n_pairs:,}")
    print(f"  Distinct S1 IDs    : {len(s1_ids_seen):,}")
    print(f"  Distinct target IDs: {len(tgt_ids_seen):,}")
    print(f"  Feature vector lens: {feature_lens}  (expected: {{{len(FEATURE_NAMES)}}})")
    print(f"  Probability range  : [{min(all_probs):.4f}, {max(all_probs):.4f}]")
    print(f"  Runtime            : {elapsed:.2f}s")
    print(f"  Peak memory        : {peak_bytes / 1e6:.2f} MB")
    print(f"  Smoke F0.5 @T=0.50 : {smoke_report.macro_f0_5:.4f}")

    if feature_lens != {len(FEATURE_NAMES)}:
        print(f"  ERROR: unexpected feature lengths {feature_lens}")
        all_ok = False
    if not (0.0 <= min(all_probs) and max(all_probs) <= 1.0):
        print("  ERROR: probabilities out of [0, 1]")
        all_ok = False
    if n_pairs == 0:
        print("  ERROR: no candidate pairs generated")
        all_ok = False

    status = "PASSED âœ“" if all_ok else "FAILED âœ—"
    print(f"\n  SMOKE TEST {status}")
    return all_ok


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 7. FAST SMOKE TEST (self-contained, caps scanning to SMOKE_SCAN_LIMIT records)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _collect_smoke_pairs(
    train_s1_set: Set[str],
    train_gt: Dict[str, Set[str]],
    index_exact: Dict,
    index_token: Dict,
    index_addr: Dict,
    s1_records: Dict[str, S1Record],
    scan_limit: int,
    neg_cap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect a small training set by scanning at most scan_limit records.

    Scans only train_source2.tsv, stopping after scan_limit lines or when
    the per-source scan is exhausted.  Only features for train S1 entities
    are captured.  Never scans the full 10.32M pool.
    """
    feats_pos: List[List[float]] = []
    feats_neg: List[List[float]] = []
    neg_counts: Dict[str, int] = defaultdict(int)
    pos_seen: Set[Tuple[str, str]] = set()
    n_scanned = 0

    with open(S2_TRAIN_PATH, "r", encoding="utf-8-sig") as f:
        next(f)  # header
        for line in f:
            if n_scanned >= scan_limit:
                break
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            n_scanned += 1
            tgt_id, tgt_name, tgt_addr, tgt_country = (
                parts[0], parts[1], parts[2], parts[3]
            )
            tgt_norm = normalize_name_conservative(tgt_name)
            matched = _candidate_s1_ids_for_target(
                tgt_name, tgt_addr, tgt_country, tgt_norm,
                index_exact, index_token, index_addr,
            )
            for s1_id in matched:
                if s1_id not in train_s1_set:
                    continue
                s1 = s1_records[s1_id]
                is_pos = tgt_id in train_gt.get(s1_id, set())
                feats = extract_pair_features(
                    s1[0], s1[1], s1[2],
                    tgt_id, tgt_name, tgt_addr, tgt_country,
                    norm_s1_name=s1[3], norm_tgt_name=tgt_norm,
                )
                if is_pos:
                    if (s1_id, tgt_id) not in pos_seen:
                        pos_seen.add((s1_id, tgt_id))
                        feats_pos.append(feats)
                else:
                    if neg_counts[s1_id] < neg_cap:
                        feats_neg.append(feats)
                        neg_counts[s1_id] += 1

    print(f"    Records scanned : {n_scanned:,} / {scan_limit:,}")
    print(f"    Positives found : {len(feats_pos):,}")
    print(f"    Negatives found : {len(feats_neg):,}")

    if not feats_pos:
        # Guard: no positives in first scan window â†’ extend slightly into S3
        print("    WARNING: 0 positives in S2 window â€” extending scan into S3...")
        with open(S3_TRAIN_PATH, "r", encoding="utf-8-sig") as f:
            next(f)
            for line in f:
                if n_scanned >= scan_limit * 2:
                    break
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4:
                    continue
                n_scanned += 1
                tgt_id, tgt_name, tgt_addr, tgt_country = (
                    parts[0], parts[1], parts[2], parts[3]
                )
                tgt_norm = normalize_name_conservative(tgt_name)
                matched = _candidate_s1_ids_for_target(
                    tgt_name, tgt_addr, tgt_country, tgt_norm,
                    index_exact, index_token, index_addr,
                )
                for s1_id in matched:
                    if s1_id not in train_s1_set:
                        continue
                    s1 = s1_records[s1_id]
                    is_pos = tgt_id in train_gt.get(s1_id, set())
                    feats = extract_pair_features(
                        s1[0], s1[1], s1[2],
                        tgt_id, tgt_name, tgt_addr, tgt_country,
                        norm_s1_name=s1[3], norm_tgt_name=tgt_norm,
                    )
                    if is_pos and (s1_id, tgt_id) not in pos_seen:
                        pos_seen.add((s1_id, tgt_id))
                        feats_pos.append(feats)
                    elif not is_pos and neg_counts[s1_id] < neg_cap:
                        feats_neg.append(feats)
                        neg_counts[s1_id] += 1
        print(f"    After S3 extension â€” pos: {len(feats_pos):,}  neg: {len(feats_neg):,}")

    all_feats = feats_pos + feats_neg
    all_labels = [1] * len(feats_pos) + [0] * len(feats_neg)
    X = np.array(all_feats, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    return X, y


def run_smoke_only(model_type: str = "logistic") -> None:
    """Fast self-contained smoke test.  Scans at most SMOKE_SCAN_LIMIT source
    records.  Never invokes the full collect_training_pairs() path.

    Stages
    ------
    [1/5] Load a tiny deterministic slice of S1 entities from the split file.
    [2/5] Build a blocking index over just those entities.
    [3/5] Collect smoke training pairs (â‰¤ SMOKE_SCAN_LIMIT records scanned).
    [4/5] Train the requested model on the small pair set.
    [5/5] Score smoke validation candidates + run official F0.5 evaluator.
    """
    import tracemalloc
    tracemalloc.start()
    t0_total = time.time()

    print("=" * 70)
    print("EXP-003 SMOKE TEST (fast, capped scan)")
    print(f"  model={model_type}  train_s1={SMOKE_TRAIN_S1}  val_s1={SMOKE_VAL_S1}")
    print(f"  scan_limit={SMOKE_SCAN_LIMIT:,} records  neg_cap={SMOKE_NEG_CAP}")
    print("=" * 70)

    # â”€â”€ [1/5] Load split and ground truth â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[1/5] Loading benchmark S1 sample...")
    all_train_ids, all_val_ids = load_split(SPLIT_FILE)

    # Deterministic slices â€” same first-N pattern as production
    smoke_train_ids = all_train_ids[:SMOKE_TRAIN_S1]
    smoke_val_ids   = all_val_ids[:SMOKE_VAL_S1]
    smoke_s1_all    = set(smoke_train_ids) | set(smoke_val_ids)
    smoke_train_set = set(smoke_train_ids)
    smoke_val_set   = set(smoke_val_ids)

    full_gt = parse_ground_truth(GT_PATH, allowed_s1_ids=smoke_s1_all)
    smoke_train_gt = {s: full_gt.get(s, set()) for s in smoke_train_ids}
    smoke_val_gt   = {s: full_gt.get(s, set()) for s in smoke_val_ids}

    n_train_pos = sum(len(v) for v in smoke_train_gt.values())
    n_val_pos   = sum(len(v) for v in smoke_val_gt.values())
    print(f"  Train S1 slice : {len(smoke_train_ids)} entities ({n_train_pos} true matches)")
    print(f"  Val   S1 slice : {len(smoke_val_ids)} entities ({n_val_pos} true matches)")

    # â”€â”€ [2/5] Load S1 records and build blocking index â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[2/5] Building blocking index over smoke S1 entities...")
    s1_records = load_s1_records(S1_TRAIN_PATH, smoke_s1_all)
    print(f"  Loaded {len(s1_records)} S1 records.")
    idx_exact, idx_token, idx_addr = build_blocking_index(s1_records)
    print(f"  Index: exact={len(idx_exact):,}  token={len(idx_token):,}  addr={len(idx_addr):,}")

    # â”€â”€ [3/5] Collect smoke training pairs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n[3/5] Collecting smoke training pairs (scan limit={SMOKE_SCAN_LIMIT:,})...")
    t3 = time.time()
    X_train, y_train = _collect_smoke_pairs(
        smoke_train_set, smoke_train_gt,
        idx_exact, idx_token, idx_addr,
        s1_records,
        scan_limit=SMOKE_SCAN_LIMIT,
        neg_cap=SMOKE_NEG_CAP,
    )
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    print(f"  Training matrix: {X_train.shape}  ({n_pos} pos / {n_neg} neg)  [{time.time()-t3:.1f}s]")

    if X_train.shape[0] == 0:
        print("  ERROR: 0 training pairs collected â€” cannot train. SMOKE FAILED.")
        return
    if n_pos == 0:
        print("  ERROR: 0 positive pairs â€” model would predict only negatives. SMOKE FAILED.")
        return

    # Verify feature dimension
    if X_train.shape[1] != len(FEATURE_NAMES):
        print(f"  ERROR: feature dim={X_train.shape[1]} expected {len(FEATURE_NAMES)}. SMOKE FAILED.")
        return
    print(f"  Feature dim    : {X_train.shape[1]} âœ“ (matches FEATURE_NAMES)")

    # â”€â”€ [4/5] Train model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n[4/5] Training {model_type} on {X_train.shape[0]} pairs...")
    t4 = time.time()
    clf = EntityMatchingClassifier(model_type=model_type, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    del X_train, y_train
    gc.collect()
    print(f"  Model trained in {time.time()-t4:.2f}s âœ“")

    # â”€â”€ [5/5] Score smoke val candidates + official evaluator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n[5/5] Scoring smoke val candidates ({len(smoke_val_ids)} val S1 entities)...")
    t5 = time.time()

    # Build val-restricted index
    val_idx_exact: Dict = defaultdict(list)
    val_idx_token: Dict = defaultdict(list)
    val_idx_addr:  Dict = defaultdict(list)
    for s1_id in smoke_val_set:
        rec = s1_records[s1_id]
        name, addr, country, norm_name = rec
        if norm_name:
            val_idx_exact[(country, norm_name)].append(s1_id)
        for tk in extract_token_blocking_keys(name):
            val_idx_token[(country, tk)].append(s1_id)
        for ak in extract_address_keys(addr):
            val_idx_addr[(country, ak)].append(s1_id)

    # Score with a smoke-limited stream (first 20K candidates max)
    SMOKE_VAL_LIMIT = 20_000
    gen = stream_all_candidates(
        val_idx_exact, val_idx_token, val_idx_addr,
        {s: s1_records[s] for s in smoke_val_set},
        smoke_limit=SMOKE_VAL_LIMIT,
    )

    # Keep per-S1 (tgt_id, prob) list â€” compact
    s1_to_probs: Dict[str, List[Tuple[str, float]]] = {s: [] for s in smoke_val_set}
    buf_s1: List[str] = []
    buf_tgt: List[str] = []
    buf_X:  List[List[float]] = []
    n_val_cands = 0

    def _flush():
        nonlocal n_val_cands
        if not buf_X:
            return
        X_b = np.array(buf_X, dtype=np.float32)
        probs = clf.predict_proba(X_b).tolist()
        for sid, tid, p in zip(buf_s1, buf_tgt, probs):
            s1_to_probs[sid].append((tid, p))
        n_val_cands += len(buf_X)
        buf_s1.clear(); buf_tgt.clear(); buf_X.clear()
        del X_b; gc.collect()

    for s1_id, tgt_id, feats in gen:
        buf_s1.append(s1_id)
        buf_tgt.append(tgt_id)
        buf_X.append(feats)
        if len(buf_X) >= BATCH_SIZE:
            _flush()
    _flush()

    print(f"  Val candidates scored: {n_val_cands:,}  [{time.time()-t5:.1f}s]")

    # Run official F0.5 evaluator at T=0.50
    preds_50: Dict[str, Set[str]] = {}
    for s1_id in smoke_val_set:
        preds_50[s1_id] = {tid for tid, p in s1_to_probs[s1_id] if p >= 0.50}
    report = evaluate_predictions(preds_50, smoke_val_gt)

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_elapsed = time.time() - t0_total

    # â”€â”€ Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n{'='*70}")
    print("SMOKE TEST RESULTS")
    print(f"{'='*70}")
    print(f"  Training pairs     : {n_pos} pos + {n_neg} neg = {n_pos+n_neg} total")
    print(f"  Feature dim        : {len(FEATURE_NAMES)} âœ“")
    print(f"  Val candidates     : {n_val_cands:,}")
    print(f"  Smoke Macro F0.5   : {report.macro_f0_5:.4f}  (T=0.50, {len(smoke_val_ids)} val entities)")
    print(f"  Smoke Precision    : {report.macro_precision:.4f}")
    print(f"  Smoke Recall       : {report.macro_recall:.4f}")
    print(f"  Total TP / FP / FN : {report.total_tp} / {report.total_fp} / {report.total_fn}")
    print(f"  Peak RAM           : {peak_bytes/1e6:.1f} MB")
    print(f"  Total runtime      : {total_elapsed:.1f}s")
    print(f"{'='*70}")

    # Final pass/fail gates
    ok = True
    if n_pos == 0:          print("  GATE FAIL: 0 positive training pairs");   ok = False
    if n_val_cands == 0:    print("  GATE FAIL: 0 val candidates scored");      ok = False
    if report.macro_f0_5 < 0.0: print("  GATE FAIL: negative F0.5 (impossible)"); ok = False

    status = "PASSED âœ“" if ok else "FAILED âœ—"
    print(f"\n  SMOKE TEST {status}")
    print("=" * 70)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 8. MAIN PIPELINE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def run_exp003(model_type: str = "logistic", smoke_only: bool = False):
    t_pipeline_start = time.time()

    print("=" * 72)
    print("Amazon ML Challenge 2026 â€” EXP-003: Memory-Safe ML Baseline")
    print(f"  model={model_type}  smoke_only={smoke_only}")
    print("=" * 72)

    # â”€â”€ 1. Load split and ground truth â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[1/6] Loading split and ground truth...")
    all_train_s1_ids, val_s1_ids = load_split(SPLIT_FILE)
    train_s1_ids = all_train_s1_ids[:TRAIN_S1_SAMPLE]   # deterministic slice
    train_s1_set = set(train_s1_ids)
    val_s1_set = set(val_s1_ids)
    eval_s1_set = train_s1_set | val_s1_set

    full_gt = parse_ground_truth(GT_PATH, allowed_s1_ids=eval_s1_set)
    train_gt = {s1: full_gt.get(s1, set()) for s1 in train_s1_ids}
    val_gt = {s1: full_gt.get(s1, set()) for s1 in val_s1_ids}

    n_train_true = sum(len(v) for v in train_gt.values())
    n_val_true = sum(len(v) for v in val_gt.values())
    print(f"  Train S1 entities  : {len(train_s1_ids):,} ({n_train_true:,} true matches)")
    print(f"  Val   S1 entities  : {len(val_s1_ids):,} ({n_val_true:,} true matches)")

    # â”€â”€ 2. Load S1 records and build blocking index â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[2/6] Loading S1 records and building blocking index...")
    s1_records = load_s1_records(S1_TRAIN_PATH, eval_s1_set)
    print(f"  Loaded {len(s1_records):,} S1 records.")
    idx_exact, idx_token, idx_addr = build_blocking_index(s1_records)
    print(f"  Index: exact={len(idx_exact):,} keys, "
          f"token={len(idx_token):,} keys, addr={len(idx_addr):,} keys.")

    # â”€â”€ 3. Collect training pairs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[3/6] Collecting training pairs (train S1 only, memory-safe)...")
    X_train, y_train = collect_training_pairs(
        train_s1_set, train_gt,
        idx_exact, idx_token, idx_addr,
        s1_records, max_neg_per_s1=MAX_NEG_PER_S1,
    )
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    print(f"  Training matrix    : {X_train.shape}  ({n_pos:,} pos / {n_neg:,} neg)")

    # â”€â”€ 4. Train classifier â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n[4/6] Training {model_type} classifier...")
    t_train = time.time()
    clf = EntityMatchingClassifier(model_type=model_type, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    train_elapsed = time.time() - t_train
    print(f"  Training completed in {train_elapsed:.2f}s.")

    # Free training data immediately â€” not needed again
    del X_train, y_train
    gc.collect()

    # â”€â”€ 5. Smoke test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    smoke_passed = run_smoke_test(
        clf, val_s1_set, val_gt,
        idx_exact, idx_token, idx_addr, s1_records,
        n_smoke=10_000,
    )

    if smoke_only:
        print("\nSmoke-only mode: stopping here. Smoke passed:", smoke_passed)
        return

    if not smoke_passed:
        print("\nSmoke test FAILED â€” aborting full run.")
        sys.exit(1)

    # â”€â”€ 6. Full validation sweep â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n[6/6] Full validation sweep over all {len(val_s1_ids):,} S1 entities...")
    print(f"  Batch size: {BATCH_SIZE:,} | Thresholds: {THRESHOLDS}")

    tracemalloc.start()
    t_val = time.time()

    n_candidates, threshold_results = score_and_evaluate(
        clf, val_s1_set, val_gt,
        idx_exact, idx_token, idx_addr, s1_records,
        thresholds=THRESHOLDS,
        batch_size=BATCH_SIZE,
        smoke_limit=None,
    )

    val_elapsed = time.time() - t_val
    _, peak_val_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # â”€â”€ 7. Report best threshold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    best_t = max(threshold_results, key=lambda t: threshold_results[t]["macro_f0_5"])
    best = threshold_results[best_t]

    print(f"\n{'='*72}")
    print(f"EXP-003 RESULTS ({model_type.upper()}) â€” Full Validation Benchmark")
    print(f"{'='*72}")
    print(f"{'Threshold':>10} | {'Macro F0.5':>10} | {'Precision':>10} | {'Recall':>10} | "
          f"{'TP':>8} | {'FP':>8} | {'FN':>8} | {'Predicted':>10}")
    print("-" * 88)
    for t in sorted(threshold_results):
        r = threshold_results[t]
        marker = " â† BEST" if t == best_t else ""
        print(
            f"{t:>10.2f} | {r['macro_f0_5']:>10.4f} | {r['precision']:>10.4f} | "
            f"{r['recall']:>10.4f} | {r['total_tp']:>8,} | {r['total_fp']:>8,} | "
            f"{r['total_fn']:>8,} | {r['predicted_matches']:>10,}{marker}"
        )
    print(f"\n  Validation candidates scored : {n_candidates:,}")
    print(f"  Validation runtime           : {val_elapsed:.1f}s")
    print(f"  Peak memory (val phase)      : {peak_val_bytes / 1e6:.2f} MB")
    print(f"  Total pipeline runtime       : {time.time() - t_pipeline_start:.1f}s")

    # â”€â”€ 8. Persist experiment record â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    exp_id = f"EXP-003-{model_type.upper()}"
    logger = ExperimentLogger("experiments")
    record = ExperimentRecord(
        experiment_id=exp_id,
        hypothesis=(
            f"Memory-safe two-stage ML pipeline with {model_type} classifier. "
            "Stage 1: Union-D blocking. Stage 2: batched streaming inference "
            f"(batch={BATCH_SIZE:,}). No 20M-row feature matrix in RAM."
        ),
        data_split_description=(
            f"Validation benchmark split (2,001 val S1 / {n_candidates:,} candidates) "
            "against full S2+S3 search pool (10,320,219 records)."
        ),
        blocking_method="Union-D (Exact-Name + Token-Pair + Address Composite)",
        features=FEATURE_NAMES,
        model=model_type,
        hyperparameters={
            "batch_size": BATCH_SIZE,
            "train_s1_sample": TRAIN_S1_SAMPLE,
            "max_neg_per_s1": MAX_NEG_PER_S1,
            "random_state": RANDOM_SEED,
            "optimal_threshold": best_t,
            "n_pos_train": n_pos,
            "n_neg_train": n_neg,
        },
        validation_method="Macro-averaged F0.5 (Official Metric); batched streaming",
        macro_F0_5=best["macro_f0_5"],
        precision=best["precision"],
        recall=best["recall"],
        training_time_seconds=round(train_elapsed, 3),
        inference_time_seconds=round(val_elapsed, 3),
        notes=(
            f"Val candidates: {n_candidates:,}  |  Peak RAM: {peak_val_bytes/1e6:.1f} MB  |  "
            f"Best T={best_t:.2f}  |  TP={best['total_tp']:,}  FP={best['total_fp']:,}  "
            f"FN={best['total_fn']:,}  |  Singleton F0.5: {best['singleton_f0_5']:.4f}"
        ),
        conclusion=(
            f"{model_type.title()} achieved Macro F0.5 = {best['macro_f0_5']:.4f} "
            f"(P={best['precision']:.4f}, R={best['recall']:.4f}) at T={best_t:.2f}."
        ),
        next_experiment="EXP-004",
        extra_metrics={
            "threshold_sweep": {str(t): r for t, r in threshold_results.items()},
            "n_train_pos": n_pos,
            "n_train_neg": n_neg,
            "n_val_candidates": n_candidates,
            "peak_mem_mb": round(peak_val_bytes / 1e6, 2),
            "val_runtime_s": round(val_elapsed, 1),
            "batch_size": BATCH_SIZE,
        },
    )
    saved_path = logger.log(record)
    print(f"\n  Experiment record saved â†’ {saved_path}")
    print(f"{'='*72}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# ENTRY POINT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def parse_args():
    parser = argparse.ArgumentParser(
        description="EXP-003: Memory-Safe Two-Stage ML Baseline"
    )
    parser.add_argument(
        "--model",
        choices=["logistic", "hist_gbdt"],
        default="logistic",
        help="Classifier to use (default: logistic)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the 10,000-candidate smoke test, then exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke:
        # Fast self-contained smoke path â€” never runs full collect_training_pairs()
        run_smoke_only(model_type=args.model)
    else:
        run_exp003(model_type=args.model, smoke_only=False)

