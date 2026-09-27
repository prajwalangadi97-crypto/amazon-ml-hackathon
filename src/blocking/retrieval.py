"""Modular, Reusable Production Retrieval and Indexing Pipeline.

Provides:
1. Unified blocking key extraction across all supported channels.
2. Compound numeric signature extraction (address plot/door numbers).
3. Country-partitioned inverted index construction with frequency pruning.
4. Deterministic candidate generation and ranking policies:
   - Channel Consensus First (EXP-005 / EXP-006)
   - V3 Baseline
   - Selective Residual Recovery
5. Persistable target index caching for rapid production and validation iteration.
"""

from collections import Counter, defaultdict
import itertools
import os
from pathlib import Path
import pickle
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import unicodedata

from src.blocking.frequency_filter import (
    BlockingKeyFilterConfig,
    prune_high_frequency_keys,
)
from src.blocking.normalizers import (
    ADDRESS_STOPWORDS,
    extract_address_keys,
    extract_token_blocking_keys,
    normalize_name_conservative,
)
from src.blocking.transliteration import has_indic_script, transliterate_indic_to_latin
from src.matching.candidate_compression import PreprocessedRecord
from tools.run_recovery_experiments import (
    extract_h1_relaxed_address_keys,
    extract_h3_name_locality_keys,
)

# Precompiled regex for compound slash/hyphen numbers
REGEX_COMPOUND_SLASH = re.compile(r"\b\d+[/]\d+\b")


def extract_compound_number_keys(address: str, min_digits: int = 3) -> List[str]:
    """Extract hyper-specific compound plot/door numbers (e.g. 170/1, 371/1519, 116/4).

    Filters out trivial 2-digit fractions (like 1/2 or 1/1) to ensure low fanout.
    """
    if not address or not address.strip():
        return []
    norm = unicodedata.normalize("NFKC", address).lower()
    matches = REGEX_COMPOUND_SLASH.findall(norm)
    keys = []
    for m in matches:
        digits_only = m.replace("/", "")
        if len(digits_only) >= min_digits:
            clean_k = m.replace("/", "_")
            keys.append(f"cnum_{clean_k}")
    return list(dict.fromkeys(keys))


def extract_record_blocking_keys(
    name: str,
    address: str,
    country: str,
    include_compound_numbers: bool = True,
) -> Dict[str, int]:
    """Extract all active blocking keys mapped to their retrieval priority tier.

    Tier 1: Exact normalized name
    Tier 2: PIN code + street word / Compound plot numbers
    Tier 3: Relaxed address keys
    Tier 4: Name-locality composite keys
    Tier 5: Name token pairs / salient tokens
    """
    keys_with_tier: Dict[str, int] = {}

    norm_name = normalize_name_conservative(name)
    if norm_name:
        keys_with_tier[f"en_{norm_name}"] = 1

    if has_indic_script(name):
        trans = transliterate_indic_to_latin(name)
        norm_trans = normalize_name_conservative(trans.replace("aa", "a").replace("ee", "i").replace("oo", "u"))
        if norm_trans and f"en_{norm_trans}" not in keys_with_tier:
            keys_with_tier[f"en_{norm_trans}"] = 1

    if address and address.strip():
        for ak in extract_address_keys(address):
            tier = 2 if "pin_" in ak else 3
            if ak not in keys_with_tier or keys_with_tier[ak] > tier:
                keys_with_tier[f"ak_{ak}"] = tier

        for ak in extract_h1_relaxed_address_keys(address):
            tier = 2 if ak.startswith("pin_") else (4 if ak.startswith("loc_") else 3)
            if ak not in keys_with_tier or keys_with_tier[ak] > tier:
                keys_with_tier[ak] = tier

        for ck in extract_h3_name_locality_keys(name, address):
            if ck not in keys_with_tier or keys_with_tier[ck] > 4:
                keys_with_tier[ck] = 4

        if include_compound_numbers:
            for cnk in extract_compound_number_keys(address):
                if cnk not in keys_with_tier or keys_with_tier[cnk] > 2:
                    keys_with_tier[cnk] = 2

    for tk in extract_token_blocking_keys(name):
        k = f"tk_{tk}"
        if k not in keys_with_tier or keys_with_tier[k] > 5:
            keys_with_tier[k] = 5

    return keys_with_tier


def rank_candidates_channel_consensus_first(
    candidates: List[Dict[str, Any]],
    cap: int = 100,
) -> List[Tuple[str, Dict[str, float]]]:
    """Rank candidates with multi-channel consensus agreement placed first.

    Used by EXP-005 and EXP-006.
    """
    sorted_cands = sorted(
        candidates,
        key=lambda c: (
            -c["ch_count"],
            c["best_tier"],
            -c["score"],
            -c["spec_sum"],
        ),
    )
    selected = sorted_cands[:cap]
    pool_size = float(len(selected))
    result = []
    for rank_idx, c in enumerate(selected):
        meta = {
            "channel_agreement_count": float(c["ch_count"]),
            "retrieval_priority_tier": float(c["best_tier"]),
            "reciprocal_retrieval_rank": 1.0 / (rank_idx + 1.0),
            "s1_candidate_pool_size": pool_size,
        }
        result.append((c["tid"], meta))
    return result


def apply_selective_residual_recovery(
    s1_list: List[str],
    phase1_candidates: Dict[str, List[Tuple[str, Dict[str, float]]]],
    phase1_predictions: Dict[str, Set[str]],
    raw_candidate_pool: Dict[str, List[Dict[str, Any]]],
    max_recovery: int = 20,
) -> Dict[str, List[Tuple[str, Dict[str, float]]]]:
    """Selectively add informative candidates for S1 entities that have zero predictions."""
    recovered_cands = {}
    for sid in s1_list:
        base_items = phase1_candidates.get(sid, [])
        if phase1_predictions.get(sid):
            recovered_cands[sid] = list(base_items)
        else:
            sel_tids = {t for t, _ in base_items}
            raw_c = raw_candidate_pool.get(sid, [])
            remaining = [c for c in raw_c if c["tid"] not in sel_tids]
            informative = [
                c for c in remaining
                if c["ch_count"] >= 2 or c["s_addr"] >= 0.60 or c["has_pin"] or c["has_house"]
            ]
            informative.sort(
                key=lambda c: (
                    -c["ch_count"],
                    c["best_tier"],
                    -max(c["score"], c["s_addr"]),
                    -c["spec_sum"],
                )
            )
            top_rec = informative[:max_recovery]
            pool_sz = float(len(base_items) + len(top_rec))
            combined = list(base_items)
            for rank_idx, c in enumerate(top_rec, start=len(base_items)):
                meta = {
                    "channel_agreement_count": float(c["ch_count"]),
                    "retrieval_priority_tier": float(c["best_tier"]),
                    "reciprocal_retrieval_rank": 1.0 / (rank_idx + 1.0),
                    "s1_candidate_pool_size": pool_sz,
                }
                combined.append((c["tid"], meta))
            recovered_cands[sid] = combined
    return recovered_cands
