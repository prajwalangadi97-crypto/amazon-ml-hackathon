"""EXP-009: Rich Pairwise Feature Extraction for Business Entity Resolution.

Extends the baseline 26-feature representation to a 52-feature rich pairwise matrix:
1. Enhanced Name Metrics: Jaro, Jaro-Winkler, Levenshtein, char 3-grams, token Jaccard,
   common token count/ratio, token count diff, transliteration alignment.
2. Enhanced Address Metrics: exact match, token-set ratio, Jaro-Winkler, Levenshtein,
   char 3-grams, common token count/ratio, compound number agreement.
3. Cross-Field Synergy: interaction terms (name * addr, name + addr),
   strong-name/weak-address indicators, weak-name/strong-address indicators,
   strong-address+numeric agreement, field presence indicators.
4. Preserved Retrieval Metadata: channel agreement count, priority tier,
   reciprocal rank, candidate pool size, cheap candidate score.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import re

from rapidfuzz import fuzz
from rapidfuzz.distance import Jaro, JaroWinkler, Levenshtein

from src.matching.candidate_compression import (
    PreprocessedRecord,
    compute_cheap_candidate_score,
)
from src.blocking.retrieval import extract_compound_number_keys

RICH_FEATURE_NAMES: List[str] = [
    # 1. Identification & Context
    "country_eq",
    "source_type",
    "s1_name_empty",
    "target_name_empty",
    "s1_addr_empty",
    "target_addr_empty",
    "both_fields_present",
    "missing_name_indicator",
    "missing_addr_indicator",
    # 2. Name Features (20 features)
    "name_exact_eq",
    "name_fuzz_ratio",
    "name_fuzz_wratio",
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "name_jaro",
    "name_jaro_winkler",
    "name_levenshtein_sim",
    "name_char_3gram_sim",
    "name_token_jaccard",
    "name_common_token_count",
    "name_common_token_ratio",
    "name_token_count_diff",
    "name_prefix_similarity",
    "name_len_diff",
    "name_len_ratio",
    "name_translit_sim",
    # 3. Address Features (16 features)
    "addr_exact_eq",
    "addr_fuzz_ratio",
    "addr_token_similarity",
    "addr_token_set_ratio",
    "addr_jaro_winkler",
    "addr_levenshtein_sim",
    "addr_char_3gram_sim",
    "addr_common_token_count",
    "addr_common_token_ratio",
    "addr_len_diff",
    "numeric_token_overlap",
    "house_number_agreement",
    "postal_pin_overlap",
    "compound_number_agreement",
    # 4. Cross-Field Features (8 features)
    "cross_field_name_loc",
    "name_x_addr",
    "name_plus_addr",
    "strong_name_weak_addr",
    "weak_name_strong_addr",
    "strong_addr_numeric",
    "name_numeric_agreement",
    # 5. Retrieval Metadata & Quality (5 features)
    "cheap_candidate_score",
    "channel_agreement_count",
    "retrieval_priority_tier",
    "reciprocal_retrieval_rank",
    "s1_candidate_pool_size",
]


def _get_char_3grams(s: str) -> Set[str]:
    """Extract character 3-grams from a string."""
    if len(s) < 3:
        return {s} if s else set()
    return {s[i : i + 3] for i in range(len(s) - 2)}


def extract_rich_pairwise_features(
    s1: PreprocessedRecord,
    tgt: PreprocessedRecord,
    metadata: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Extract 52 rich entity-resolution features from preprocessed records."""
    feats: Dict[str, float] = {}

    # 1. IDENTIFICATION & CONTEXT
    country_eq = 1.0 if (s1.country and tgt.country and s1.country == tgt.country) else 0.0
    feats["country_eq"] = country_eq
    feats["source_type"] = 0.0 if tgt.is_source2 else 1.0

    s1_name_empty = 1.0 if not s1.name_norm else 0.0
    tgt_name_empty = 1.0 if not tgt.name_norm else 0.0
    s1_addr_empty = 1.0 if not s1.addr_norm else 0.0
    tgt_addr_empty = 1.0 if not tgt.addr_norm else 0.0

    feats["s1_name_empty"] = s1_name_empty
    feats["target_name_empty"] = tgt_name_empty
    feats["s1_addr_empty"] = s1_addr_empty
    feats["target_addr_empty"] = tgt_addr_empty

    feats["both_fields_present"] = (
        1.0 if (not s1_name_empty and not tgt_name_empty and not s1_addr_empty and not tgt_addr_empty) else 0.0
    )
    feats["missing_name_indicator"] = 1.0 if (s1_name_empty or tgt_name_empty) else 0.0
    feats["missing_addr_indicator"] = 1.0 if (s1_addr_empty or tgt_addr_empty) else 0.0

    # 2. NAME FEATURES
    if s1_name_empty or tgt_name_empty:
        feats["name_exact_eq"] = 0.0
        feats["name_fuzz_ratio"] = 0.0
        feats["name_fuzz_wratio"] = 0.0
        feats["name_token_set_ratio"] = 0.0
        feats["name_token_sort_ratio"] = 0.0
        feats["name_jaro"] = 0.0
        feats["name_jaro_winkler"] = 0.0
        feats["name_levenshtein_sim"] = 0.0
        feats["name_char_3gram_sim"] = 0.0
        feats["name_token_jaccard"] = 0.0
        feats["name_common_token_count"] = 0.0
        feats["name_common_token_ratio"] = 0.0
        feats["name_token_count_diff"] = 0.0
        feats["name_prefix_similarity"] = 0.0
        feats["name_len_diff"] = float(abs(len(s1.name_norm) - len(tgt.name_norm)))
        feats["name_len_ratio"] = 0.0
        feats["name_translit_sim"] = 0.0
    else:
        s1_n = s1.name_norm
        tgt_n = tgt.name_norm
        feats["name_exact_eq"] = 1.0 if s1_n == tgt_n else 0.0
        feats["name_fuzz_ratio"] = fuzz.ratio(s1_n, tgt_n) / 100.0
        feats["name_fuzz_wratio"] = fuzz.WRatio(s1_n, tgt_n) / 100.0
        feats["name_token_set_ratio"] = fuzz.token_set_ratio(s1_n, tgt_n) / 100.0
        feats["name_token_sort_ratio"] = fuzz.token_sort_ratio(s1_n, tgt_n) / 100.0
        feats["name_jaro"] = float(Jaro.similarity(s1_n, tgt_n))
        feats["name_jaro_winkler"] = float(JaroWinkler.similarity(s1_n, tgt_n))
        feats["name_levenshtein_sim"] = float(Levenshtein.normalized_similarity(s1_n, tgt_n))

        # Char 3-gram Jaccard
        ng1 = _get_char_3grams(s1_n)
        ng2 = _get_char_3grams(tgt_n)
        u_ng = ng1 | ng2
        feats["name_char_3gram_sim"] = len(ng1 & ng2) / max(len(u_ng), 1) if u_ng else 0.0

        # Token metrics
        tok1 = s1_n.split()
        tok2 = tgt_n.split()
        set1 = set(tok1)
        set2 = set(tok2)
        u_tok = set1 | set2
        i_tok = set1 & set2
        feats["name_token_jaccard"] = len(i_tok) / max(len(u_tok), 1) if u_tok else 0.0
        feats["name_common_token_count"] = float(len(i_tok))
        feats["name_common_token_ratio"] = len(i_tok) / max(len(u_tok), 1) if u_tok else 0.0
        feats["name_token_count_diff"] = float(abs(len(tok1) - len(tok2)))

        # Prefix similarity
        pfx_len = 0
        for c1, c2 in zip(s1_n, tgt_n):
            if c1 == c2:
                pfx_len += 1
            else:
                break
        feats["name_prefix_similarity"] = pfx_len / max(len(s1_n), len(tgt_n), 1)

        len1, len2 = len(s1_n), len(tgt_n)
        feats["name_len_diff"] = float(abs(len1 - len2))
        feats["name_len_ratio"] = min(len1, len2) / max(len1, len2, 1)

        # Transliteration similarity
        if s1.name_translit or tgt.name_translit:
            s1_c = s1.name_translit if s1.name_translit else s1_n
            tgt_c = tgt.name_translit if tgt.name_translit else tgt_n
            t_ratio = fuzz.ratio(s1_c, tgt_c) / 100.0
            feats["name_translit_sim"] = t_ratio
            if s1_c == tgt_c:
                feats["name_exact_eq"] = 1.0
            feats["name_fuzz_ratio"] = max(feats["name_fuzz_ratio"], t_ratio)
            feats["name_fuzz_wratio"] = max(feats["name_fuzz_wratio"], fuzz.WRatio(s1_c, tgt_c) / 100.0)
            feats["name_token_set_ratio"] = max(feats["name_token_set_ratio"], fuzz.token_set_ratio(s1_c, tgt_c) / 100.0)
            feats["name_token_sort_ratio"] = max(feats["name_token_sort_ratio"], fuzz.token_sort_ratio(s1_c, tgt_c) / 100.0)
        else:
            feats["name_translit_sim"] = feats["name_fuzz_ratio"]

    # 3. ADDRESS FEATURES
    if s1_addr_empty or tgt_addr_empty:
        feats["addr_exact_eq"] = 0.0
        feats["addr_fuzz_ratio"] = 0.0
        feats["addr_token_similarity"] = 0.0
        feats["addr_token_set_ratio"] = 0.0
        feats["addr_jaro_winkler"] = 0.0
        feats["addr_levenshtein_sim"] = 0.0
        feats["addr_char_3gram_sim"] = 0.0
        feats["addr_common_token_count"] = 0.0
        feats["addr_common_token_ratio"] = 0.0
        feats["addr_len_diff"] = float(abs(len(s1.addr_norm) - len(tgt.addr_norm)))
        feats["numeric_token_overlap"] = 0.0
        feats["house_number_agreement"] = 0.0
        feats["postal_pin_overlap"] = 0.0
        feats["compound_number_agreement"] = 0.0
    else:
        s1_a = s1.addr_norm
        tgt_a = tgt.addr_norm
        feats["addr_exact_eq"] = 1.0 if s1_a == tgt_a else 0.0
        feats["addr_fuzz_ratio"] = fuzz.ratio(s1_a, tgt_a) / 100.0
        feats["addr_token_set_ratio"] = fuzz.token_set_ratio(s1_a, tgt_a) / 100.0
        feats["addr_jaro_winkler"] = float(JaroWinkler.similarity(s1_a, tgt_a))
        feats["addr_levenshtein_sim"] = float(Levenshtein.normalized_similarity(s1_a, tgt_a))

        # Char 3-gram Jaccard on address
        ng1_a = _get_char_3grams(s1_a)
        ng2_a = _get_char_3grams(tgt_a)
        u_ng_a = ng1_a | ng2_a
        feats["addr_char_3gram_sim"] = len(ng1_a & ng2_a) / max(len(u_ng_a), 1) if u_ng_a else 0.0

        # Address words token Jaccard similarity
        union_words = s1.addr_words_set | tgt.addr_words_set
        inter_words = s1.addr_words_set & tgt.addr_words_set
        feats["addr_token_similarity"] = len(inter_words) / max(len(union_words), 1) if union_words else 0.0
        feats["addr_common_token_count"] = float(len(inter_words))
        feats["addr_common_token_ratio"] = feats["addr_token_similarity"]
        feats["addr_len_diff"] = float(abs(len(s1_a) - len(tgt_a)))

        # Numeric token overlap (Jaccard)
        union_nums = s1.nums_set | tgt.nums_set
        inter_nums = s1.nums_set & tgt.nums_set
        feats["numeric_token_overlap"] = len(inter_nums) / max(len(union_nums), 1) if union_nums else 0.0

        # House number agreement
        if s1.first_num and tgt.first_num:
            feats["house_number_agreement"] = 1.0 if s1.first_num == tgt.first_num else 0.0
        else:
            feats["house_number_agreement"] = 0.0

        # Postal / PIN code overlap
        if s1.pin and tgt.pin:
            feats["postal_pin_overlap"] = 1.0 if s1.pin == tgt.pin else 0.0
        else:
            feats["postal_pin_overlap"] = 0.0

        # Compound number agreement (e.g. 170/1)
        cnum1 = extract_compound_number_keys(s1.raw_address)
        cnum2 = extract_compound_number_keys(tgt.raw_address)
        feats["compound_number_agreement"] = 1.0 if (set(cnum1) & set(cnum2)) else 0.0

    # 4. CROSS-FIELD FEATURES
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

    name_sim = feats["name_fuzz_wratio"]
    addr_sim = feats["addr_token_similarity"]
    has_num_match = (
        feats["house_number_agreement"] == 1.0
        or feats["postal_pin_overlap"] == 1.0
        or feats["compound_number_agreement"] == 1.0
    )

    feats["name_x_addr"] = name_sim * addr_sim
    feats["name_plus_addr"] = (name_sim + addr_sim) / 2.0
    feats["strong_name_weak_addr"] = 1.0 if (name_sim >= 0.85 and addr_sim <= 0.20) else 0.0
    feats["weak_name_strong_addr"] = 1.0 if (name_sim <= 0.45 and (addr_sim >= 0.55 or has_num_match)) else 0.0
    feats["strong_addr_numeric"] = 1.0 if (addr_sim >= 0.50 and has_num_match) else 0.0
    feats["name_numeric_agreement"] = name_sim * feats["numeric_token_overlap"]

    # 5. RETRIEVAL METADATA
    # Compute cheap candidate score
    feats["cheap_candidate_score"] = compute_cheap_candidate_score(feats)

    # Attach retrieval metadata
    if metadata:
        for m_name in [
            "channel_agreement_count",
            "retrieval_priority_tier",
            "reciprocal_retrieval_rank",
            "s1_candidate_pool_size",
        ]:
            feats[m_name] = float(metadata.get(m_name, 0.0))
    else:
        feats["channel_agreement_count"] = 1.0
        feats["retrieval_priority_tier"] = 3.0
        feats["reciprocal_retrieval_rank"] = 0.5
        feats["s1_candidate_pool_size"] = 100.0

    return feats
