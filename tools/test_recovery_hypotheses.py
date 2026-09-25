#!/usr/bin/env python3
"""Test recovery hypotheses on missed true links.

Evaluates how many of the 749 missed true links are recovered by:
- Hypothesis 1: Relaxed Numeric Address Keys (num_X + word_Y combinations & extended address stop words)
- Hypothesis 2: Cleaned Domain/Handle + Salient Unique Tokens
- Hypothesis 3: Name-Locality Cross-Field Keys
"""

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Set, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking.normalizers import (
    ADDRESS_STOPWORDS,
    LEGAL_AND_BUSINESS_STOPWORDS,
    extract_address_keys,
    extract_token_blocking_keys,
    normalize_name_conservative,
    tokenize_name,
)
from src.data.ground_truth import parse_ground_truth

# Extended address stopwords including prefixes observed in failure cases
EXTENDED_ADDRESS_STOPWORDS = ADDRESS_STOPWORDS | {
    "door", "no", "hno", "house", "flat", "shop", "plot", "khasra",
    "kh", "premises", "surat", "delhi", "mumbai", "kolkata", "chennai",
    "bangalore", "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow",
    "texas", "california", "florida", "illinois", "georgia", "ohio",
    "pennsylvania", "indiana", "michigan", "virginia", "washington",
}

# Web domain suffixes
DOMAIN_SUFFIXES = (
    ".com", ".org", ".net", ".in", ".co.in", ".co", ".biz", ".info", ".edu", ".gov",
    "com", "org", "net", "biz"
)


def extract_relaxed_address_keys(address: str) -> List[str]:
    """Hypothesis 1: Relaxed Numeric Address Keys.
    
    Pairs the first 2 numeric tokens with the first 2 non-stop street words.
    Also handles 5-6 digit PIN codes robustly.
    """
    if not address or not address.strip():
        return []

    norm = unicodedata.normalize("NFKC", address).lower()
    nums = re.findall(r"\b\d+\b", norm)
    
    clean = [
        " " if unicodedata.category(c).startswith(("P", "S")) else c
        for c in norm
    ]
    words = [
        w for w in "".join(clean).split()
        if len(w) >= 3 and not w.isdigit() and w not in EXTENDED_ADDRESS_STOPWORDS
    ]

    keys = []
    # Key Type 1: Pair each of top 2 numbers with top 2 words
    if nums and words:
        for n in nums[:2]:
            for w in words[:2]:
                keys.append(f"nword_{n}_{w}")
            # If PIN code (5 or 6 digits)
            if len(n) in (5, 6):
                for w in words[:2]:
                    keys.append(f"pin_{n}_{w}")

    # Key Type 2: First two informative street/locality words
    if len(words) >= 2:
        keys.append(f"loc_{words[0]}_{words[1]}")

    return list(dict.fromkeys(keys))


def clean_domain_or_handle(name: str) -> str:
    """Clean domain names and handles."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", name).lower().strip()
    s = s.lstrip("@#")
    for suffix in [".co.in", ".com", ".org", ".net", ".in", ".co", ".biz", ".info"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def extract_domain_and_salient_name_keys(name: str) -> List[str]:
    """Hypothesis 2: Cleaned Domain/Handle + Salient Unique Tokens."""
    cleaned = clean_domain_or_handle(name)
    tokens = tokenize_name(cleaned)
    if not tokens:
        tokens = tokenize_name(name)
    
    if not tokens:
        norm = normalize_name_conservative(cleaned)
        return [norm] if norm else []

    keys = []
    # If single token
    if len(tokens) == 1:
        keys.append(tokens[0])
    else:
        # Token pairs
        top_tokens = tokens[:3]
        for i in range(len(top_tokens)):
            for j in range(i + 1, len(top_tokens)):
                t1, t2 = sorted([top_tokens[i], top_tokens[j]])
                keys.append(f"{t1}_{t2}")
        # Add every distinctive token >= 5 chars (not just the single longest)
        for t in tokens:
            if len(t) >= 5 and t not in LEGAL_AND_BUSINESS_STOPWORDS:
                keys.append(f"tok_{t}")

    # Also if cleaned name is a single unspaced word >= 5 chars
    clean_norm = normalize_name_conservative(cleaned).replace(" ", "")
    if len(clean_norm) >= 5:
        keys.append(f"unspaced_{clean_norm[:12]}")

    return list(dict.fromkeys(keys))


def extract_cross_field_keys(name: str, address: str) -> List[str]:
    """Hypothesis 3: Name-Locality Cross-Field Keys."""
    if not name or not address:
        return []
    
    n_tokens = tokenize_name(name)
    a_keys = extract_address_keys(address)
    
    norm_a = unicodedata.normalize("NFKC", address).lower()
    clean_a = [
        " " if unicodedata.category(c).startswith(("P", "S")) else c
        for c in norm_a
    ]
    a_words = [
        w for w in "".join(clean_a).split()
        if len(w) >= 4 and not w.isdigit() and w not in EXTENDED_ADDRESS_STOPWORDS
    ]

    keys = []
    if n_tokens and a_words:
        # Primary distinctive name token + primary locality word
        name_tok = n_tokens[0]
        if len(name_tok) >= 4:
            for loc in a_words[:2]:
                keys.append(f"nl_{name_tok[:5]}_{loc}")
    return list(dict.fromkeys(keys))


def main():
    # Load missed links from artifacts/blocking/missed_links_EXP001D.tsv
    tsv_path = Path("artifacts/blocking/missed_links_EXP001D.tsv")
    missed_pairs = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            missed_pairs.append({
                "s1_id": p[0],
                "target_id": p[1],
                "s1_name": p[3],
                "s1_addr": p[4],
                "s1_country": p[5],
                "target_name": p[6],
                "target_addr": p[7],
                "target_country": p[8],
            })

    total_missed = len(missed_pairs)
    print(f"Total missed true pairs from EXP-001D: {total_missed:,}")

    # Test Hypothesis 1: Relaxed Address Keys
    h1_recovered = 0
    h1_examples = []
    for p in missed_pairs:
        s1_keys = set(extract_relaxed_address_keys(p["s1_addr"]))
        t_keys = set(extract_relaxed_address_keys(p["target_addr"]))
        shared = s1_keys & t_keys
        if shared:
            h1_recovered += 1
            if len(h1_examples) < 3:
                h1_examples.append((p, shared))

    print(f"\nHypothesis 1 (Relaxed Address Keys):")
    print(f"  Recovered: {h1_recovered}/{total_missed} ({h1_recovered/total_missed*100:.2f}%)")
    for ex, sh in h1_examples:
        print(f"    S1: '{ex['s1_name']}' | Addr: '{ex['s1_addr']}'")
        print(f"    Target: '{ex['target_name']}' | Addr: '{ex['target_addr']}'")
        print(f"    Shared keys: {sh}\n")

    # Test Hypothesis 2: Domain/Handle + Salient Name Tokens
    h2_recovered = 0
    h2_examples = []
    for p in missed_pairs:
        s1_keys = set(extract_domain_and_salient_name_keys(p["s1_name"]))
        t_keys = set(extract_domain_and_salient_name_keys(p["target_name"]))
        shared = s1_keys & t_keys
        if shared:
            h2_recovered += 1
            if len(h2_examples) < 3:
                h2_examples.append((p, shared))

    print(f"\nHypothesis 2 (Domain/Handle + Salient Name Tokens):")
    print(f"  Recovered: {h2_recovered}/{total_missed} ({h2_recovered/total_missed*100:.2f}%)")
    for ex, sh in h2_examples:
        print(f"    S1: '{ex['s1_name']}' | Addr: '{ex['s1_addr']}'")
        print(f"    Target: '{ex['target_name']}' | Addr: '{ex['target_addr']}'")
        print(f"    Shared keys: {sh}\n")

    # Test Hypothesis 3: Name-Locality Cross-Field Keys
    h3_recovered = 0
    h3_examples = []
    for p in missed_pairs:
        s1_keys = set(extract_cross_field_keys(p["s1_name"], p["s1_addr"]))
        t_keys = set(extract_cross_field_keys(p["target_name"], p["target_addr"]))
        shared = s1_keys & t_keys
        if shared:
            h3_recovered += 1
            if len(h3_examples) < 3:
                h3_examples.append((p, shared))

    print(f"\nHypothesis 3 (Name-Locality Cross-Field Keys):")
    print(f"  Recovered: {h3_recovered}/{total_missed} ({h3_recovered/total_missed*100:.2f}%)")
    for ex, sh in h3_examples:
        print(f"    S1: '{ex['s1_name']}' | Addr: '{ex['s1_addr']}'")
        print(f"    Target: '{ex['target_name']}' | Addr: '{ex['target_addr']}'")
        print(f"    Shared keys: {sh}\n")

    # Combined Recovery: H1 OR H2 OR H3
    combined_recovered = 0
    for p in missed_pairs:
        h1_m = bool(set(extract_relaxed_address_keys(p["s1_addr"])) & set(extract_relaxed_address_keys(p["target_addr"])))
        h2_m = bool(set(extract_domain_and_salient_name_keys(p["s1_name"])) & set(extract_domain_and_salient_name_keys(p["target_name"])))
        h3_m = bool(set(extract_cross_field_keys(p["s1_name"], p["s1_addr"])) & set(extract_cross_field_keys(p["target_name"], p["target_addr"])))
        if h1_m or h2_m or h3_m:
            combined_recovered += 1

    print("=" * 60)
    print(f"COMBINED RECOVERY (H1 + H2 + H3):")
    print(f"  Recovered: {combined_recovered}/{total_missed} ({combined_recovered/total_missed*100:.2f}%)")
    print(f"  Remaining missed: {total_missed - combined_recovered} ({(total_missed - combined_recovered)/total_missed*100:.2f}%)")
    total_true_links = 6945
    baseline_retrieved = 6196
    new_total_retrieved = baseline_retrieved + combined_recovered
    print(f"  New Total Retrieved True Matches: {new_total_retrieved}/{total_true_links} ({new_total_retrieved/total_true_links*100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
