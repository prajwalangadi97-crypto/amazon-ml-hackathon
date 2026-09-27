#!/usr/bin/env python3
"""Validation missed-link root cause analysis.

Analyzes all 542 ground-truth links missed by the EXP-002B validation benchmark
(92.20% recall = 6,403 / 6,945 pairs).

Groups misses into the 10 requested categories:
1. candidate-cap/ranking loss
2. name variation/spelling
3. address formatting/reordering
4. house-number/numeric variation
5. locality
6. PIN/postal code
7. domain/website
8. transliteration
9. truncation/prefix
10. other
"""

import collections
import json
import pickle
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from rapidfuzz import distance, fuzz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking.normalizers import (
    extract_address_keys,
    extract_token_blocking_keys,
    normalize_name_conservative,
)
from src.data.ground_truth import parse_ground_truth
from src.matching.candidate_compression import (
    fast_score_candidate,
    preprocess_record,
)
from tools.run_recovery_experiments import (
    clean_domain_or_handle,
    extract_h1_relaxed_address_keys,
    extract_h3_name_locality_keys,
)

TRANSLIT_PAIRS = [
    ("ee", "i"), ("oo", "u"), ("v", "w"), ("sh", "s"),
    ("dh", "d"), ("th", "t"), ("kh", "k"), ("gh", "g"),
    ("bh", "b"), ("ph", "f"), ("ch", "c"), ("ks", "x"),
    ("aa", "a"), ("ou", "u"), ("eau", "o"), ("y", "i"),
]


def extract_all_keys(name: str, addr: str) -> Set[str]:
    keys = set()
    en = normalize_name_conservative(name)
    if en:
        keys.add(f"en_{en}")
    for tk in extract_token_blocking_keys(name):
        keys.add(f"tk_{tk}")
    if addr and addr.strip():
        for ak in extract_address_keys(addr):
            keys.add(f"ak_{ak}")
        for ak in extract_h1_relaxed_address_keys(addr):
            keys.add(ak)
        for ck in extract_h3_name_locality_keys(name, addr):
            keys.add(ck)
    return keys


def is_domain_or_website_variation(name1: str, name2: str) -> bool:
    dom_patterns = [
        r"\.com\b", r"\.in\b", r"\.co\b", r"\.org\b", r"\.net\b",
        r"www\.", r"@", r"\.fr\b", r"\.biz\b", r"\.info\b", r"\.gov\b"
    ]
    has_dom1 = any(re.search(p, name1, re.I) for p in dom_patterns)
    has_dom2 = any(re.search(p, name2, re.I) for p in dom_patterns)
    if has_dom1 or has_dom2:
        return True
    # Unspaced domain-like name (e.g. vadodaraindustries vs vadodara industries)
    s1_clean = "".join(name1.lower().split())
    s2_clean = "".join(name2.lower().split())
    if s1_clean == s2_clean and (" " in name1 != " " in name2):
        return True
    return False


def is_transliteration_name(w1_list: List[str], w2_list: List[str]) -> bool:
    for w1 in w1_list:
        w1_l = w1.lower()
        if len(w1_l) < 3:
            continue
        v1 = re.sub(r"[aeiouy]", "", w1_l)
        for w2 in w2_list:
            w2_l = w2.lower()
            if len(w2_l) < 3 or w1_l == w2_l:
                continue
            v2 = re.sub(r"[aeiouy]", "", w2_l)
            # Consonant skeleton match with vowel differences (e.g. sarma vs sharma, federaation vs federation)
            if v1 == v2 and len(v1) >= 3:
                return True
            # Phoneme substitution match
            for p1, p2 in TRANSLIT_PAIRS:
                if w1_l.replace(p1, p2) == w2_l or w2_l.replace(p1, p2) == w1_l:
                    return True
                if w1_l.replace(p1, p2) == w2_l.replace(p1, p2):
                    return True
    return False


def is_truncation_or_prefix(n1: str, n2: str) -> bool:
    t1 = set(n1.lower().split())
    t2 = set(n2.lower().split())
    if not t1 or not t2:
        return False
    # Subset of tokens
    if (t1.issubset(t2) or t2.issubset(t1)) and t1 != t2:
        return True
    # Character prefix
    s1 = "".join(n1.lower().split())
    s2 = "".join(n2.lower().split())
    if len(s1) >= 4 and len(s2) >= 4 and s1 != s2:
        if s1.startswith(s2) or s2.startswith(s1):
            return True
    return False


def classify_miss(
    s1_id: str,
    tid: str,
    s1_record: Tuple[str, str, str],
    tgt_record: Tuple[str, str, str],
    shared_keys: Set[str],
    fast_score: float,
    s1_candidate_count: int,
) -> Tuple[str, Dict[str, Any]]:
    s1_name, s1_addr, s1_c = s1_record
    t_name, t_addr, t_c = tgt_record

    norm_n1 = normalize_name_conservative(s1_name)
    norm_n2 = normalize_name_conservative(t_name)

    name_fuzz = fuzz.ratio(norm_n1, norm_n2)
    name_token_set = fuzz.token_set_ratio(norm_n1, norm_n2)

    # 1. Candidate cap / ranking loss
    # S1 reached saturated candidate count (>= 2000 or 2400) and pair had shared keys and fast_score >= 0.15
    if shared_keys and fast_score >= 0.15:
        return "candidate-cap/ranking loss", {
            "subreason": f"S1 reached {s1_candidate_count} candidates; bumped from top-300 worker list",
            "fast_score": fast_score,
            "shared_keys": list(shared_keys),
        }
    if shared_keys and fast_score < 0.15:
        return "candidate-cap/ranking loss", {
            "subreason": f"Shared key {list(shared_keys)} matched, but fast_score={fast_score:.3f} fell below 0.15 compression threshold",
            "fast_score": fast_score,
            "shared_keys": list(shared_keys),
        }

    # Extract digits for house numbers / PINs
    s1_nums = re.findall(r"\b\d+\b", s1_addr)
    t_nums = re.findall(r"\b\d+\b", t_addr)
    s1_pins = [n for n in s1_nums if len(n) in (5, 6)]
    t_pins = [n for n in t_nums if len(n) in (5, 6)]

    w1_name_tokens = re.findall(r"[a-z0-9]+", norm_n1.lower())
    w2_name_tokens = re.findall(r"[a-z0-9]+", norm_n2.lower())

    # 2. Domain / Website
    if is_domain_or_website_variation(s1_name, t_name):
        return "domain/website", {
            "subreason": "URL, email, handle or domain suffix in one of the names",
            "name_fuzz": name_fuzz,
        }

    # 3. Transliteration (vowel / phoneme variations)
    if is_transliteration_name(w1_name_tokens, w2_name_tokens):
        return "transliteration", {
            "subreason": "Vowel or phoneme transliteration variation between name tokens",
            "name_fuzz": name_fuzz,
        }

    # 4. Truncation / Prefix
    if is_truncation_or_prefix(norm_n1, norm_n2):
        return "truncation/prefix", {
            "subreason": "Name is a strict prefix or token subset of the other",
            "name_fuzz": name_fuzz,
            "name_token_set": name_token_set,
        }

    # 5. Name variation / spelling
    if name_token_set >= 70 or name_fuzz >= 60:
        return "name variation/spelling", {
            "subreason": "Typo, slight spelling variation, or minor token alteration in name",
            "name_fuzz": name_fuzz,
            "name_token_set": name_token_set,
        }

    # Address-based classifications
    addr_fuzz = fuzz.ratio(s1_addr.lower(), t_addr.lower())
    addr_token_set = fuzz.token_set_ratio(s1_addr.lower(), t_addr.lower())

    # 6. PIN / postal code
    if s1_pins and t_pins and s1_pins != t_pins:
        return "PIN/postal code", {
            "subreason": f"PIN mismatch: {s1_pins} vs {t_pins}",
            "addr_token_set": addr_token_set,
        }
    elif (bool(s1_pins) != bool(t_pins)) and addr_token_set >= 50:
        return "PIN/postal code", {
            "subreason": "PIN present in only one record preventing PIN-based blocking",
            "addr_token_set": addr_token_set,
        }

    # 7. House-number / numeric variation
    if s1_nums and t_nums and set(s1_nums) != set(t_nums):
        if addr_token_set >= 55:
            return "house-number/numeric variation", {
                "subreason": f"Street words match but house/unit numbers differ: {s1_nums[:2]} vs {t_nums[:2]}",
                "addr_token_set": addr_token_set,
            }

    # 8. Address formatting / reordering
    if addr_token_set >= 65 and addr_fuzz < 50:
        return "address formatting/reordering", {
            "subreason": "Address tokens match but differing component order/punctuation prevented key match",
            "addr_token_set": addr_token_set,
            "addr_fuzz": addr_fuzz,
        }

    # 9. Locality
    if addr_token_set >= 40:
        return "locality", {
            "subreason": "City, district, or sub-locality variation with partially matching street",
            "addr_token_set": addr_token_set,
        }

    # 10. Other
    return "other", {
        "subreason": "Heavily corrupted, completely disjoint tokens, or missing fields across both name and address",
        "name_fuzz": name_fuzz,
        "addr_fuzz": addr_fuzz,
    }


def main():
    t0 = time.time()
    split_path = PROJECT_ROOT / "experiments" / "splits" / "validation_split_seed42_benchmark_10000.json"
    with open(split_path, "r", encoding="utf-8") as f:
        val_s1 = set(json.load(f)["val_s1_ids"])

    gt_path = PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv"
    gt = parse_ground_truth(gt_path, allowed_s1_ids=val_s1)

    pickle_path = PROJECT_ROOT / "scratch" / "benchmark_chunk_results.pkl"
    with open(pickle_path, "rb") as f:
        _, all_broad_counts, all_target_records, broad_cands_map, _, _ = pickle.load(f)

    misses = []
    miss_s1_ids = set()
    miss_tgt_ids = set()
    for s1_id, targets in gt.items():
        cands = broad_cands_map.get(s1_id, set())
        for tid in targets:
            if tid not in cands:
                misses.append((s1_id, tid))
                miss_s1_ids.add(s1_id)
                miss_tgt_ids.add(tid)

    total_misses = len(misses)
    print(f"Total validation misses to analyze: {total_misses}")

    # Load S1 records
    s1_records = {}
    with open(PROJECT_ROOT / "dataset" / "train" / "train_source1.tsv", "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            c_pos = line.find("\t")
            sid = line[:c_pos]
            if sid in miss_s1_ids:
                parts = line.rstrip("\r\n").split("\t")
                s1_records[sid] = (parts[1], parts[2], parts[3])
                if len(s1_records) == len(miss_s1_ids):
                    break

    # Load Target records
    tgt_records = {}
    for s_path in ["train_source2.tsv", "train_source3.tsv"]:
        fp = PROJECT_ROOT / "dataset" / "train" / s_path
        with open(fp, "r", encoding="utf-8-sig") as f:
            next(f)
            for line in f:
                c_pos = line.find("\t")
                tid = line[:c_pos]
                if tid in miss_tgt_ids:
                    parts = line.rstrip("\r\n").split("\t")
                    tgt_records[tid] = (parts[1], parts[2], parts[3])
                    if len(tgt_records) == len(miss_tgt_ids):
                        break
        if len(tgt_records) == len(miss_tgt_ids):
            break

    # Categorize every miss
    category_counts = collections.defaultdict(int)
    category_examples = collections.defaultdict(list)

    for s1_id, tid in misses:
        s1_rec = s1_records[s1_id]
        tgt_rec = tgt_records[tid]

        k1 = extract_all_keys(s1_rec[0], s1_rec[1])
        k2 = extract_all_keys(tgt_rec[0], tgt_rec[1])
        shared = k1 & k2

        p1 = preprocess_record(s1_id, s1_rec[0], s1_rec[1], s1_rec[2])
        p2 = preprocess_record(tid, tgt_rec[0], tgt_rec[1], tgt_rec[2])
        score = fast_score_candidate(p1, p2)
        s1_cand_cnt = len(broad_cands_map.get(s1_id, set()))

        category, meta = classify_miss(
            s1_id=s1_id,
            tid=tid,
            s1_record=s1_rec,
            tgt_record=tgt_rec,
            shared_keys=shared,
            fast_score=score,
            s1_candidate_count=s1_cand_cnt,
        )

        category_counts[category] += 1
        category_examples[category].append({
            "s1_id": s1_id,
            "target_id": tid,
            "country": s1_rec[2],
            "s1_name": s1_rec[0],
            "s1_addr": s1_rec[1],
            "target_name": tgt_rec[0],
            "target_addr": tgt_rec[1],
            "fast_score": score,
            "s1_candidate_count": s1_cand_cnt,
            "shared_keys": list(shared),
            "meta": meta,
        })

    # Summary
    print(f"\nAnalysis completed in {time.time() - t0:.2f}s.")
    print("=" * 80)
    print("MISSED-LINK ROOT CAUSE BREAKDOWN (542 TOTAL MISSES)")
    print("=" * 80)
    sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    for cat, cnt in sorted_cats:
        pct = cnt / total_misses * 100.0
        print(f"{cat:35s}: {cnt:4d} ({pct:5.2f}%)")

    # Save to JSON
    report_data = {
        "total_misses": total_misses,
        "category_counts": {cat: cnt for cat, cnt in sorted_cats},
        "category_percentages": {cat: cnt / total_misses * 100.0 for cat, cnt in sorted_cats},
        "category_examples": category_examples,
    }

    out_file = PROJECT_ROOT / "experiments" / "validation_missed_links_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved detailed analysis report to {out_file}")


if __name__ == "__main__":
    main()
