"""Deterministic normalization and key-extraction functions for blocking.

All operations preserve international characters (Unicode letters, marks, and numbers)
while cleanly stripping punctuation and formatting symbols.
"""

import itertools
import re
import unicodedata
from typing import List, Set


# Common generic legal suffixes and business stop words
LEGAL_AND_BUSINESS_STOPWORDS: Set[str] = {
    "inc", "incorporated", "corp", "corporation",
    "llc", "llp", "ltd", "limited", "pvt", "private",
    "co", "company", "cie", "sa", "sas", "sarl", "gmbh",
    "the", "and", "of", "for", "in", "to", "at",
    "services", "enterprises", "solutions", "group", "holdings",
    "trading", "consulting", "management", "international", "associates",
    "center", "centre", "industries",
}

# Common address stop words across US, India, and France
ADDRESS_STOPWORDS: Set[str] = {
    "road", "rd", "street", "st", "avenue", "ave", "drive", "dr",
    "lane", "ln", "way", "court", "ct", "boulevard", "blvd", "rue", "allée",
    "nagar", "colony", "marg", "chowk", "rasta",
    "near", "opp", "opposite", "behind", "beside",
    "floor", "flr", "plot", "phase", "block", "sector", "sec",
    "main", "cross", "bldg", "building", "tower", "complex",
    "unit", "apt", "apartment", "suite", "ste", "room", "no", "number",
    "po", "box", "post", "dist", "district", "state", "pincode", "pin",
}


def normalize_name_conservative(name: str) -> str:
    """Normalize business name conservatively.

    Steps:
    1. Unicode NFKC normalization.
    2. Lowercase.
    3. Replace punctuation (P*) and symbols (S*) with spaces (preserves letters L*,
       marks M* such as Indic viramas/vowels, and numbers N*).
    4. Collapse whitespace and strip.
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", name).lower()
    clean = [
        " " if unicodedata.category(c).startswith(("P", "S")) else c
        for c in s
    ]
    return " ".join("".join(clean).split())


def tokenize_name(name: str, min_len: int = 3) -> List[str]:
    """Extract informative unique tokens from a business name.

    Filters out short tokens (< min_len) and generic legal/business stop words.
    """
    norm = normalize_name_conservative(name)
    if not norm:
        return []
    words = norm.split()
    informative = []
    seen = set()
    for w in words:
        if len(w) >= min_len and w not in LEGAL_AND_BUSINESS_STOPWORDS:
            if w not in seen:
                seen.add(w)
                informative.append(w)
    return informative


def extract_token_blocking_keys(name: str, max_tokens: int = 3) -> List[str]:
    """Generate conservative token-based blocking keys from a business name.

    Rules:
    - If 0 informative tokens: return [normalize_name_conservative(name)].
    - If 1 informative token: return [token].
    - If >= 2 informative tokens:
      Generate sorted token pairs for the top informative tokens.
      Using token pairs prevents explosive candidate volume from single popular tokens.
      Also includes the individual rarest/longest token.
    """
    tokens = tokenize_name(name)
    if not tokens:
        norm = normalize_name_conservative(name)
        return [norm] if norm else []

    if len(tokens) == 1:
        return [tokens[0]]

    # For names with >= 2 tokens, generate token pairs among top tokens
    keys = []
    top_tokens = tokens[:max_tokens]
    for t1, t2 in itertools.combinations(sorted(top_tokens), 2):
        keys.append(f"{t1}_{t2}")

    # Also add the single longest token (often the most distinctive proper noun)
    longest_token = max(tokens, key=len)
    if len(longest_token) >= 5:
        keys.append(longest_token)

    return list(dict.fromkeys(keys))  # preserve order, remove duplicates


def extract_address_keys(address: str) -> List[str]:
    """Extract conservative address-derived blocking keys from provided address text only.

    No external geocoding or external postal databases are used.
    Handles empty addresses safely by returning an empty list [].
    """
    if not address or not address.strip():
        return []

    norm = unicodedata.normalize("NFKC", address).lower()

    # Extract all numeric sequences (PIN codes, house numbers, plot numbers)
    nums = re.findall(r"\b\d+\b", norm)

    # Extract clean words (letters and marks)
    clean = [
        " " if unicodedata.category(c).startswith(("P", "S")) else c
        for c in norm
    ]
    words = [
        w for w in "".join(clean).split()
        if len(w) >= 3 and not w.isdigit() and w not in ADDRESS_STOPWORDS
    ]

    keys = []
    # Key Type 1: Numeric street/PIN code + first informative locality/street word
    if nums and words:
        keys.append(f"num_{nums[0]}_{words[0]}")
        # If there is a 5-6 digit PIN/ZIP code, create a PIN + street token key
        for n in nums:
            if len(n) in (5, 6):
                keys.append(f"pin_{n}_{words[0]}")
                break

    # Key Type 2: First two informative locality/city words
    if len(words) >= 2:
        keys.append(f"loc_{words[0]}_{words[1]}")

    return list(dict.fromkeys(keys))
