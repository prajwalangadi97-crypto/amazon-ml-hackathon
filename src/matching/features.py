"""Pairwise feature engineering for Entity Matching.

Extracts discriminative string similarity, numeric overlap, token Jaccard,
and metadata consistency features for candidate entity pairs.
Designed for high throughput and robustness to typos, missing values, and variations.
"""

import re
from typing import Dict, List, Set, Tuple

# Precompiled regex patterns for speed
RE_NUMBERS = re.compile(r"\b\d+\b")
RE_ALPHANUM = re.compile(r"[^a-z0-9\s]")

# Generic business stop words that carry low matching entropy
GENERIC_WORDS = {
    "center", "services", "service", "global", "solutions", "solution",
    "enterprises", "enterprise", "group", "holding", "holdings", "trading",
    "international", "national", "company", "associates", "management",
    "tech", "technologies", "technology", "systems", "system", "consulting",
    "consultants", "industries", "industry", "agency", "network", "ventures",
    "capital", "corporation", "limited", "private", "corp", "inc", "ltd", "pvt", "llc"
}

FEATURE_NAMES = [
    "name_exact_match",
    "name_token_jaccard",
    "name_char3_jaccard",
    "name_token_overlap",
    "name_len_diff_ratio",
    "name_first_token_match",
    "addr_num_overlap",
    "addr_num_jaccard",
    "addr_both_have_nums",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_len_diff_ratio",
    "country_exact_match",
    "is_source_2",
    "name_addr_cross_overlap",
]


def extract_numbers(text: str) -> Set[str]:
    """Extract numeric tokens (building numbers, street numbers, PIN codes)."""
    if not text:
        return set()
    return set(RE_NUMBERS.findall(text))


def extract_tokens(text: str) -> Set[str]:
    """Extract lowercase alphanumeric tokens of length >= 2."""
    if not text:
        return set()
    clean = RE_ALPHANUM.sub(" ", text.lower())
    return {t for t in clean.split() if len(t) >= 2}


def extract_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from cleaned text."""
    if not text:
        return set()
    clean = re.sub(r"\s+", " ", RE_ALPHANUM.sub(" ", text.lower()).strip())
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection_len = len(set_a & set_b)
    if intersection_len == 0:
        return 0.0
    union_len = len(set_a | set_b)
    return intersection_len / union_len if union_len > 0 else 0.0


def extract_pair_features(
    s1_name: str,
    s1_addr: str,
    s1_country: str,
    tgt_id: str,
    tgt_name: str,
    tgt_addr: str,
    tgt_country: str,
    norm_s1_name: str = "",
    norm_tgt_name: str = "",
) -> List[float]:
    """Extract 15 numerical features for a single entity pair.

    Returns:
        List of 15 float features corresponding to FEATURE_NAMES.
    """
    n1 = norm_s1_name if norm_s1_name else s1_name.lower().strip()
    n2 = norm_tgt_name if norm_tgt_name else tgt_name.lower().strip()

    # 1. Name exact match
    name_exact = 1.0 if (n1 and n1 == n2) else 0.0

    # 2 & 4. Name token Jaccard and overlap
    t1_name = extract_tokens(n1)
    t2_name = extract_tokens(n2)
    name_tok_jaccard = jaccard_similarity(t1_name, t2_name)
    name_tok_overlap = float(len(t1_name & t2_name))

    # 3. Name char 3-gram Jaccard (typo robust)
    c1_name = extract_char_ngrams(n1, 3)
    c2_name = extract_char_ngrams(n2, 3)
    name_char3_jaccard = jaccard_similarity(c1_name, c2_name)

    # 5. Name length difference ratio
    len_n1, len_n2 = len(n1), len(n2)
    max_len_n = max(len_n1, len_n2, 1)
    name_len_diff = abs(len_n1 - len_n2) / max_len_n

    # 6. First token match
    tokens1_ordered = [t for t in RE_ALPHANUM.sub(" ", n1).split() if len(t) >= 2]
    tokens2_ordered = [t for t in RE_ALPHANUM.sub(" ", n2).split() if len(t) >= 2]
    first_tok_match = 0.0
    if tokens1_ordered and tokens2_ordered:
        first_tok_match = 1.0 if tokens1_ordered[0] == tokens2_ordered[0] else 0.0

    # 7 & 8. Address numeric overlap & numeric Jaccard
    nums1 = extract_numbers(s1_addr)
    nums2 = extract_numbers(tgt_addr)
    num_overlap = float(len(nums1 & nums2))
    num_jaccard = jaccard_similarity(nums1, nums2)
    both_have_nums = 1.0 if (nums1 and nums2) else 0.0

    # 10 & 11. Address token Jaccard & overlap
    t1_addr = extract_tokens(s1_addr)
    t2_addr = extract_tokens(tgt_addr)
    addr_tok_jaccard = jaccard_similarity(t1_addr, t2_addr)
    addr_tok_overlap = float(len(t1_addr & t2_addr))

    # 12. Address length difference ratio
    len_a1, len_a2 = len(s1_addr), len(tgt_addr)
    max_len_a = max(len_a1, len_a2, 1)
    addr_len_diff = abs(len_a1 - len_a2) / max_len_a

    # 13. Country exact match
    country_match = 1.0 if (s1_country and s1_country == tgt_country) else 0.0

    # 14. Target source indicator
    is_s2 = 1.0 if tgt_id.startswith("S2-") else 0.0

    # 15. Cross overlap: name tokens present in address
    cross_overlap = float(len((t1_name & t2_addr) | (t2_name & t1_addr)))

    return [
        name_exact,
        name_tok_jaccard,
        name_char3_jaccard,
        name_tok_overlap,
        name_len_diff,
        first_tok_match,
        num_overlap,
        num_jaccard,
        both_have_nums,
        addr_tok_jaccard,
        addr_tok_overlap,
        addr_len_diff,
        country_match,
        is_s2,
        cross_overlap,
    ]
