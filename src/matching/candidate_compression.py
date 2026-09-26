"""Candidate compression and cheap pairwise similarity matching.

EXP-002A: Determines whether cheap RapidFuzz-based pairwise similarity can
compress the broad D + E-A + E-C candidate pool into a much smaller final
candidate set while preserving true-match recall.

Designed to be lightweight, deterministic, interpretable, and modular,
allowing the same feature extraction and compression logic to feed downstream
ranking models (such as XGBoost).
"""

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import rapidfuzz
from rapidfuzz import fuzz

from src.blocking.normalizers import (
    ADDRESS_STOPWORDS,
    LEGAL_AND_BUSINESS_STOPWORDS,
    normalize_name_conservative,
)


@dataclass(frozen=True)
class PreprocessedRecord:
    """Compact preprocessed representation of an entity record for cheap matching."""

    entity_id: str
    raw_name: str
    raw_address: str
    country: str
    name_norm: str
    addr_norm: str
    nums: Tuple[str, ...]
    nums_set: Set[str]
    first_num: Optional[str]
    pin: Optional[str]
    addr_words_set: Set[str]
    is_source2: bool


def preprocess_record(
    entity_id: str,
    name: str,
    address: str,
    country: str,
) -> PreprocessedRecord:
    """Preprocess record strings into normalized text and numeric/token sets.

    Handles Unicode, missing fields, and case differences gracefully.
    """
    raw_name = name or ""
    raw_addr = address or ""
    c_str = country or ""

    name_norm = normalize_name_conservative(raw_name)
    addr_norm = normalize_name_conservative(raw_addr)

    # Extract numeric tokens (PINs, house numbers, plot numbers)
    nums_list = re.findall(r"\b\d+\b", addr_norm)
    nums_tuple = tuple(nums_list)
    nums_set = set(nums_list)
    first_num = nums_tuple[0] if nums_tuple else None

    # Detect 5 or 6 digit PIN / postal code
    pin = None
    for n in nums_tuple:
        if len(n) in (5, 6):
            pin = n
            break

    # Extract clean address word tokens (excluding stopwords and digits)
    addr_words = [
        w
        for w in addr_norm.split()
        if len(w) >= 3 and not w.isdigit() and w not in ADDRESS_STOPWORDS
    ]
    addr_words_set = set(addr_words)

    is_s2 = entity_id.startswith("S2-")

    return PreprocessedRecord(
        entity_id=entity_id,
        raw_name=raw_name,
        raw_address=raw_addr,
        country=c_str,
        name_norm=name_norm,
        addr_norm=addr_norm,
        nums=nums_tuple,
        nums_set=nums_set,
        first_num=first_num,
        pin=pin,
        addr_words_set=addr_words_set,
        is_source2=is_s2,
    )


def compute_cheap_pairwise_features(
    s1_name: str,
    s1_addr: str,
    s1_country: str,
    target_name: str,
    target_addr: str,
    target_country: str,
    target_id: str = "",
    s1_id: str = "",
) -> Dict[str, float]:
    """Calculate cheap pairwise similarity features between Source 1 and Target.

    Returns an interpretable dictionary of floats suitable for inspection,
    scoring, or tabular ML models.
    """
    s1_p = preprocess_record(s1_id, s1_name, s1_addr, s1_country)
    tgt_p = preprocess_record(target_id, target_name, target_addr, target_country)
    return extract_features_from_preprocessed(s1_p, tgt_p)


def extract_features_from_preprocessed(
    s1: PreprocessedRecord,
    tgt: PreprocessedRecord,
) -> Dict[str, float]:
    """Extract features directly from preprocessed records."""
    feats: Dict[str, float] = {}

    # 1. COUNTRY EQUALITY
    country_eq = 1.0 if (s1.country and tgt.country and s1.country == tgt.country) else 0.0
    feats["country_eq"] = country_eq

    # 2. SOURCE TYPE
    # S2 -> 0.0, S3 -> 1.0
    feats["source_type"] = 0.0 if tgt.is_source2 else 1.0

    # 3. EMPTY / MISSING FIELD INDICATORS
    feats["s1_name_empty"] = 1.0 if not s1.name_norm else 0.0
    feats["target_name_empty"] = 1.0 if not tgt.name_norm else 0.0
    feats["s1_addr_empty"] = 1.0 if not s1.addr_norm else 0.0
    feats["target_addr_empty"] = 1.0 if not tgt.addr_norm else 0.0

    # 4. NAME FEATURES
    if not s1.name_norm or not tgt.name_norm:
        feats["name_exact_eq"] = 0.0
        feats["name_fuzz_ratio"] = 0.0
        feats["name_fuzz_wratio"] = 0.0
        feats["name_token_set_ratio"] = 0.0
        feats["name_token_sort_ratio"] = 0.0
        feats["name_prefix_similarity"] = 0.0
        feats["name_len_diff"] = float(abs(len(s1.name_norm) - len(tgt.name_norm)))
        feats["name_len_ratio"] = 0.0
    else:
        feats["name_exact_eq"] = 1.0 if s1.name_norm == tgt.name_norm else 0.0
        feats["name_fuzz_ratio"] = fuzz.ratio(s1.name_norm, tgt.name_norm) / 100.0
        feats["name_fuzz_wratio"] = fuzz.WRatio(s1.name_norm, tgt.name_norm) / 100.0
        feats["name_token_set_ratio"] = fuzz.token_set_ratio(s1.name_norm, tgt.name_norm) / 100.0
        feats["name_token_sort_ratio"] = fuzz.token_sort_ratio(s1.name_norm, tgt.name_norm) / 100.0

        # Prefix similarity: length of common prefix normalized by max length
        pfx_len = 0
        for c1, c2 in zip(s1.name_norm, tgt.name_norm):
            if c1 == c2:
                pfx_len += 1
            else:
                break
        feats["name_prefix_similarity"] = pfx_len / max(len(s1.name_norm), len(tgt.name_norm), 1)

        len1, len2 = len(s1.name_norm), len(tgt.name_norm)
        feats["name_len_diff"] = float(abs(len1 - len2))
        feats["name_len_ratio"] = min(len1, len2) / max(len1, len2, 1)

    # 5. ADDRESS FEATURES
    has_both_addrs = bool(s1.addr_norm and tgt.addr_norm)
    if not has_both_addrs:
        feats["addr_fuzz_ratio"] = 0.0
        feats["addr_token_similarity"] = 0.0
        feats["numeric_token_overlap"] = 0.0
        feats["house_number_agreement"] = 0.0
        feats["postal_pin_overlap"] = 0.0
        feats["addr_len_diff"] = float(abs(len(s1.addr_norm) - len(tgt.addr_norm)))
    else:
        feats["addr_fuzz_ratio"] = fuzz.ratio(s1.addr_norm, tgt.addr_norm) / 100.0

        # Address word token Jaccard similarity
        union_words = s1.addr_words_set | tgt.addr_words_set
        inter_words = s1.addr_words_set & tgt.addr_words_set
        feats["addr_token_similarity"] = (
            len(inter_words) / max(len(union_words), 1) if union_words else 0.0
        )

        # Numeric token overlap (Jaccard)
        union_nums = s1.nums_set | tgt.nums_set
        inter_nums = s1.nums_set & tgt.nums_set
        feats["numeric_token_overlap"] = (
            len(inter_nums) / max(len(union_nums), 1) if union_nums else 0.0
        )

        # House number agreement (first numeric token match)
        if s1.first_num and tgt.first_num:
            feats["house_number_agreement"] = 1.0 if s1.first_num == tgt.first_num else 0.0
        else:
            feats["house_number_agreement"] = 0.0

        # Postal / PIN code overlap (5-6 digits)
        if s1.pin and tgt.pin:
            feats["postal_pin_overlap"] = 1.0 if s1.pin == tgt.pin else 0.0
        else:
            feats["postal_pin_overlap"] = 0.0

        feats["addr_len_diff"] = float(abs(len(s1.addr_norm) - len(tgt.addr_norm)))

    # 6. CROSS-FIELD EVIDENCE
    # Checks if name token appears in target address or vice versa
    cross_field = 0.0
    if s1.name_norm and tgt.addr_words_set:
        for t in s1.name_norm.split():
            if len(t) >= 4 and t in tgt.addr_words_set:
                cross_field = 1.0
                break
    if cross_field == 0.0 and tgt.name_norm and s1.addr_words_set:
        for t in tgt.name_norm.split():
            if len(t) >= 4 and t in s1.addr_words_set:
                cross_field = 1.0
                break
    feats["cross_field_name_loc"] = cross_field

    return feats


def compute_cheap_candidate_score(feats: Dict[str, float]) -> float:
    """Compute a transparent deterministic candidate score for adaptive compression.

    SCORING FORMULA:
    ----------------
    1. Hard Filter:
       If country_eq == 0.0, Score = 0.0.

    2. Name Evidence Score (S_name in [0.0, 1.0]):
       If name_exact_eq == 1.0:
           S_name = 1.0
       Else:
           S_name = (
               0.40 * name_fuzz_wratio +
               0.30 * name_token_set_ratio +
               0.20 * name_fuzz_ratio +
               0.10 * name_prefix_similarity
           )

    3. Address Evidence Score (S_addr in [0.0, 1.0]):
       If both addresses are present (s1_addr_empty == 0 and target_addr_empty == 0):
           s_addr_sim = 0.50 * addr_token_similarity + 0.50 * addr_fuzz_ratio
           S_addr = (
               0.35 * s_addr_sim +
               0.25 * postal_pin_overlap +
               0.25 * house_number_agreement +
               0.15 * numeric_token_overlap
           )
           # Negative evidence penalty: if both have explicit PINs but they disagree:
           # (handled via postal_pin_overlap being 0 and pin mismatch check if available)
       Else:
           Address score is unavailable.

    4. Composite Score:
       If both addresses are present:
           Score = 0.55 * S_name + 0.45 * S_addr
       Else (address missing in S1 or Target, common in Source 3):
           Score = 0.90 * S_name (10% uncertainty penalty for missing address)

       If cross_field_name_loc == 1.0 and S_addr < 0.3:
           Score = min(1.0, Score + 0.05)

    The resulting score lies in [0.0, 1.0], giving higher priority to
    precision-sensitive evidence (exact names, PIN agreement, house numbers).
    """
    if feats.get("country_eq", 0.0) == 0.0:
        return 0.0

    # Name score
    if feats.get("name_exact_eq", 0.0) == 1.0:
        s_name = 1.0
    else:
        s_name = (
            0.40 * feats.get("name_fuzz_wratio", 0.0)
            + 0.30 * feats.get("name_token_set_ratio", 0.0)
            + 0.20 * feats.get("name_fuzz_ratio", 0.0)
            + 0.10 * feats.get("name_prefix_similarity", 0.0)
        )

    has_both_addrs = (
        feats.get("s1_addr_empty", 1.0) == 0.0
        and feats.get("target_addr_empty", 1.0) == 0.0
    )

    if has_both_addrs:
        s_addr_sim = (
            0.50 * feats.get("addr_token_similarity", 0.0)
            + 0.50 * feats.get("addr_fuzz_ratio", 0.0)
        )
        
        # When PIN / house numbers are present, use them; when absent, fall back to text similarity
        has_pin = feats.get("postal_pin_overlap", 0.0) > 0.0
        # If postal_pin_overlap == 1.0, pin matches
        # If neither has pin, postal_pin_overlap is 0.0. To distinguish no-pin from mismatch:
        pin_score = 1.0 if feats.get("postal_pin_overlap", 0.0) == 1.0 else s_addr_sim
        house_score = 1.0 if feats.get("house_number_agreement", 0.0) == 1.0 else (
            s_addr_sim if feats.get("numeric_token_overlap", 0.0) == 0.0 else feats.get("house_number_agreement", 0.0)
        )
        num_score = feats.get("numeric_token_overlap", s_addr_sim) if feats.get("numeric_token_overlap", 0.0) > 0.0 else s_addr_sim

        s_addr = (
            0.40 * s_addr_sim
            + 0.25 * pin_score
            + 0.20 * house_score
            + 0.15 * num_score
        )
        score = 0.55 * s_name + 0.45 * s_addr
    else:
        # Address absent in at least one record (e.g. Source 3 records often lack address)
        score = 0.90 * s_name

    if feats.get("cross_field_name_loc", 0.0) == 1.0:
        score = min(1.0, score + 0.05)

    return max(0.0, min(1.0, score))


def fast_score_candidate(
    s1: PreprocessedRecord,
    tgt: PreprocessedRecord,
) -> float:
    """Fast inline computation of candidate score directly from PreprocessedRecords.

    Produces mathematically identical results to compute_cheap_candidate_score,
    avoiding the overhead of creating intermediate feature dictionaries for
    millions of candidate pairs during streaming candidate generation.
    """
    if s1.country != tgt.country:
        return 0.0

    # 1. Name Component
    if not s1.name_norm or not tgt.name_norm:
        s_name = 0.0
    elif s1.name_norm == tgt.name_norm:
        s_name = 1.0
    else:
        wratio = fuzz.WRatio(s1.name_norm, tgt.name_norm) / 100.0
        tset = fuzz.token_set_ratio(s1.name_norm, tgt.name_norm) / 100.0
        fratio = fuzz.ratio(s1.name_norm, tgt.name_norm) / 100.0

        pfx_len = 0
        for c1, c2 in zip(s1.name_norm, tgt.name_norm):
            if c1 == c2:
                pfx_len += 1
            else:
                break
        pfx_sim = pfx_len / max(len(s1.name_norm), len(tgt.name_norm), 1)

        s_name = 0.40 * wratio + 0.30 * tset + 0.20 * fratio + 0.10 * pfx_sim

    # 2. Address Component
    if s1.addr_norm and tgt.addr_norm:
        addr_ratio = fuzz.ratio(s1.addr_norm, tgt.addr_norm) / 100.0

        union_w = s1.addr_words_set | tgt.addr_words_set
        inter_w = s1.addr_words_set & tgt.addr_words_set
        addr_tok_sim = len(inter_w) / max(len(union_w), 1) if union_w else 0.0

        s_addr_sim = 0.50 * addr_tok_sim + 0.50 * addr_ratio

        # PIN evaluation
        if s1.pin and tgt.pin:
            pin_score = 1.0 if s1.pin == tgt.pin else 0.0
        else:
            pin_score = s_addr_sim

        # House number evaluation
        if s1.first_num and tgt.first_num:
            house_score = 1.0 if s1.first_num == tgt.first_num else 0.0
        else:
            house_score = s_addr_sim

        # Numeric token overlap
        if s1.nums_set and tgt.nums_set:
            union_n = s1.nums_set | tgt.nums_set
            inter_n = s1.nums_set & tgt.nums_set
            num_score = len(inter_n) / max(len(union_n), 1)
        else:
            num_score = s_addr_sim

        s_addr = (
            0.40 * s_addr_sim
            + 0.25 * pin_score
            + 0.20 * house_score
            + 0.15 * num_score
        )
        
        # Conflict penalty: if both have explicit PINs but they disagree
        if s1.pin and tgt.pin and s1.pin != tgt.pin:
            s_addr *= 0.60

        score = 0.55 * s_name + 0.45 * s_addr
    else:
        score = 0.90 * s_name

    # Cross field check
    cross_field = 0.0
    if s1.name_norm and tgt.addr_words_set:
        for t in s1.name_norm.split():
            if len(t) >= 4 and t in tgt.addr_words_set:
                cross_field = 1.0
                break
    if cross_field == 0.0 and tgt.name_norm and s1.addr_words_set:
        for t in tgt.name_norm.split():
            if len(t) >= 4 and t in s1.addr_words_set:
                cross_field = 1.0
                break

    if cross_field == 1.0:
        score = min(1.0, score + 0.05)

    return max(0.0, min(1.0, score))


@dataclass
class CompressionPolicy:
    """Defines an adaptive candidate compression policy."""

    policy_id: str
    name: str
    method: str  # 'top_k', 'threshold', 'hybrid', 'two_tier'
    top_k: Optional[int] = None
    min_score: float = 0.0
    max_cap: Optional[int] = None
    high_threshold: Optional[float] = None
    low_threshold: Optional[float] = None


class CandidateCompressor:
    """Manages ranking and filtering of candidates according to compression policies."""

    @staticmethod
    def compress_entity_candidates(
        scored_candidates: List[Tuple[float, str]],
        policy: CompressionPolicy,
    ) -> Set[str]:
        """Compress ranked (score, target_id) candidates for a single Source 1 entity.

        Assumes scored_candidates is already sorted descending by score.
        Singleton entities with no matching candidates return set().
        """
        if not scored_candidates:
            return set()

        method = policy.method

        if method == "top_k":
            k = policy.top_k or 25
            min_s = policy.min_score
            selected = [tid for score, tid in scored_candidates if score >= min_s][:k]
            return set(selected)

        elif method == "threshold":
            threshold = policy.min_score
            selected = [tid for score, tid in scored_candidates if score >= threshold]
            if policy.max_cap is not None:
                selected = selected[: policy.max_cap]
            return set(selected)

        elif method == "hybrid":
            threshold = policy.min_score
            max_c = policy.max_cap or 100
            selected = [tid for score, tid in scored_candidates if score >= threshold][:max_c]
            return set(selected)

        elif method == "two_tier":
            # Tier 1: High confidence candidates (always kept up to max_cap)
            # Tier 2: Medium confidence candidates (filled up to max_cap)
            high_t = policy.high_threshold or 0.60
            low_t = policy.low_threshold or 0.35
            max_c = policy.max_cap or 50

            high_tier = [tid for score, tid in scored_candidates if score >= high_t]
            if len(high_tier) >= max_c:
                return set(high_tier[:max_c])

            remaining_budget = max_c - len(high_tier)
            mid_tier = [
                tid
                for score, tid in scored_candidates
                if low_t <= score < high_t
            ][:remaining_budget]
            return set(high_tier + mid_tier)

        else:
            raise ValueError(f"Unknown compression policy method: {method}")

    @classmethod
    def compress_candidates_map(
        cls,
        all_scored_candidates: Dict[str, List[Tuple[float, str]]],
        policy: CompressionPolicy,
    ) -> Dict[str, Set[str]]:
        """Apply a compression policy across all Source 1 entities."""
        compressed: Dict[str, Set[str]] = {}
        for s1_id, scored_list in all_scored_candidates.items():
            compressed[s1_id] = cls.compress_entity_candidates(scored_list, policy)
        return compressed
