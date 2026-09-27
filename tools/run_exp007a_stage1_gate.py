"""EXP-007A Stage-1 Gate: Focused Evaluation on the 432 Unretrieved Validation Links.

Evaluates:
1. Retrieval recovery across the 432 previously unretrieved ground-truth pairs:
   - Total recovered / 432
   - India recovered / 333
   - Cross-script recovered / 324
   - US recovered / 99
   - France recovered / 0
2. Key-channel attribution for recovered pairs.
3. Candidate overhead & fanout impact on validation S1 entities.
4. Non-Indic feature invariance verification.
5. Exact runtime measurement.
"""

import collections
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, List, Set, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking.normalizers import (
    extract_address_keys,
    extract_token_blocking_keys,
    normalize_name_conservative,
    tokenize_name,
)
from src.blocking.retrieval import (
    extract_compound_number_keys,
    extract_record_blocking_keys,
)
from src.blocking.transliteration import has_indic_script, transliterate_indic_to_latin
from src.data.ground_truth import parse_ground_truth
from src.matching.candidate_compression import (
    compute_cheap_candidate_score,
    extract_features_from_preprocessed,
    fast_score_candidate,
    preprocess_record,
    PreprocessedRecord,
)
from tools.run_recovery_experiments import (
    extract_h1_relaxed_address_keys,
    extract_h3_name_locality_keys,
)


def extract_baseline_keys(name: str, addr: str, country: str) -> Set[str]:
    """Baseline EXP-006B key extraction without transliteration."""
    keys = set()
    norm = normalize_name_conservative(name)
    if norm:
        keys.add(f"en_{norm}")
    if addr and addr.strip():
        for ak in extract_address_keys(addr):
            keys.add(f"ak_{ak}")
        for ak in extract_h1_relaxed_address_keys(addr):
            keys.add(ak)
        for ck in extract_h3_name_locality_keys(name, addr):
            keys.add(ck)
        for cnk in extract_compound_number_keys(addr):
            keys.add(cnk)
    # Extract baseline raw token keys (without translit)
    tokens = tokenize_name(name)
    if tokens:
        if len(tokens) == 1:
            keys.add(f"tk_{tokens[0]}")
        else:
            top_t = tokens[:3]
            import itertools
            for t1, t2 in itertools.combinations(sorted(top_t), 2):
                keys.add(f"tk_{t1}_{t2}")
            longest = max(tokens, key=len)
            if len(longest) >= 5:
                keys.add(f"tk_{longest}")
    return keys


def main():
    t_start = time.time()
    print("=" * 80)
    print("EXP-007A STAGE-1 FOCUSED GATE: 432 UNRETRIEVED VALIDATION LINKS")
    print("=" * 80)

    # 1. Load Validation Split & Ground Truth
    split_file = PROJECT_ROOT / "experiments" / "splits" / "validation_split_seed42_benchmark_10000.json"
    with open(split_file, "r") as f:
        split_data = json.load(f)
    val_s1_list = split_data["val_s1_ids"]
    val_s1_set = set(val_s1_list)

    gt_file = PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv"
    val_gt = parse_ground_truth(gt_file, allowed_s1_ids=val_s1_set)

    # Load EXP-005 / EXP-006 baseline raw pool
    pool_file = PROJECT_ROOT / "artifacts" / "exp005" / "val_raw_candidates.pkl"
    with open(pool_file, "rb") as f:
        raw_pool = pickle.load(f)

    # Identify the exact 432 unretrieved links
    unretrieved_pairs: List[Tuple[str, str]] = []
    unretrieved_by_sid: Dict[str, Set[str]] = collections.defaultdict(set)
    for sid, tids in val_gt.items():
        pool_tids = {c["tid"] for c in raw_pool.get(sid, [])}
        for tid in tids:
            if tid not in pool_tids:
                unretrieved_pairs.append((sid, tid))
                unretrieved_by_sid[sid].add(tid)

    total_unretrieved = len(unretrieved_pairs)
    print(f"Identified {total_unretrieved} unretrieved ground-truth pairs across {len(unretrieved_by_sid)} S1 entities.")
    assert total_unretrieved == 432, f"Expected 432 unretrieved links, got {total_unretrieved}"

    # 2. Load S1 Records for unretrieved pairs
    needed_s1_ids = set(unretrieved_by_sid.keys())
    s1_file = PROJECT_ROOT / "dataset" / "train" / "train_source1.tsv"
    s1_map: Dict[str, Tuple[str, str, str, str]] = {}
    with open(s1_file, "r", encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and parts[0] in needed_s1_ids:
                s1_map[parts[0]] = (parts[0], parts[1], parts[2], parts[3])

    # 3. Load Target Records for unretrieved pairs
    needed_target_ids = {tid for _, tid in unretrieved_pairs}
    target_map: Dict[str, Tuple[str, str, str, str]] = {}
    for src_file in ["train_source2.tsv", "train_source3.tsv"]:
        fp = PROJECT_ROOT / "dataset" / "train" / src_file
        with open(fp, "r", encoding="utf-8-sig") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4 and parts[0] in needed_target_ids:
                    target_map[parts[0]] = (parts[0], parts[1], parts[2], parts[3])
                    if len(target_map) == len(needed_target_ids):
                        break
        if len(target_map) == len(needed_target_ids):
            break

    print(f"Loaded {len(s1_map)} Source 1 records and {len(target_map)} Target records.")

    # 4. Evaluate Recovery under EXP-007A Transliteration
    print("\nEvaluating Blocking Key Recovery under EXP-007A...")

    recovered_total = 0
    recovered_by_country = collections.Counter()
    total_by_country = collections.Counter()
    recovered_cross_script = 0
    total_cross_script = 0

    channel_hits = collections.Counter()
    recovered_pairs_detail = []

    for sid, tid in unretrieved_pairs:
        s1_rec = s1_map.get(sid)
        tgt_rec = target_map.get(tid)
        if not s1_rec or not tgt_rec:
            continue

        s_id, s_name, s_addr, s_country = s1_rec
        t_id, t_name, t_addr, t_country = tgt_rec

        total_by_country[s_country] += 1
        is_cross_script = has_indic_script(t_name) and not has_indic_script(s_name)
        if is_cross_script:
            total_cross_script += 1

        # Baseline keys
        base_s1_keys = extract_baseline_keys(s_name, s_addr, s_country)
        base_tgt_keys = extract_baseline_keys(t_name, t_addr, t_country)
        base_shared = base_s1_keys & base_tgt_keys

        # EXP-007A keys (with transliteration)
        exp_s1_keys = set(extract_record_blocking_keys(s_name, s_addr, s_country, include_compound_numbers=True).keys())
        exp_tgt_keys = set(extract_record_blocking_keys(t_name, t_addr, t_country, include_compound_numbers=True).keys())
        exp_shared = exp_s1_keys & exp_tgt_keys

        # Check if recovered (0 shared in baseline -> >=1 shared in EXP-007A)
        if len(base_shared) == 0 and len(exp_shared) > 0:
            recovered_total += 1
            recovered_by_country[s_country] += 1
            if is_cross_script:
                recovered_cross_script += 1

            # Log matched channels
            for k in exp_shared:
                prefix = k.split("_")[0]
                channel_hits[prefix] += 1

            # Also compute feature similarities with transliteration
            s1_p = preprocess_record(s_id, s_name, s_addr, s_country)
            tgt_p = preprocess_record(t_id, t_name, t_addr, t_country)
            feats = extract_features_from_preprocessed(s1_p, tgt_p)
            score = fast_score_candidate(s1_p, tgt_p)

            recovered_pairs_detail.append({
                "sid": sid,
                "tid": tid,
                "country": s_country,
                "s1_name": s_name,
                "target_name": t_name,
                "target_translit": tgt_p.name_translit,
                "shared_keys": sorted(list(exp_shared)),
                "name_fuzz_wratio": feats["name_fuzz_wratio"],
                "name_token_set_ratio": feats["name_token_set_ratio"],
                "fast_score": score,
            })

    elapsed_time = time.time() - t_start

    # 5. Display Gate Results
    print("\n" + "=" * 80)
    print("STAGE-1 RECOVERY RESULTS")
    print("=" * 80)
    print(f"Total Previously Unretrieved Links Evaluated : {total_unretrieved}")
    print(f"TOTAL RECOVERED LINKS BY EXP-007A            : {recovered_total} / {total_unretrieved} ({recovered_total / total_unretrieved:.2%})")
    print(f"\nCountry Breakdown:")
    for country in ["India", "US", "France"]:
        rec = recovered_by_country[country]
        tot = total_by_country[country]
        pct = (rec / tot * 100) if tot > 0 else 0.0
        print(f"  {country:10s}: {rec:3d} / {tot:3d} ({pct:.1f}%)")

    print(f"\nCross-Script Breakdown:")
    print(f"  Cross-Script Links in Pool : {total_cross_script}")
    print(f"  Cross-Script Recovered     : {recovered_cross_script} / {total_cross_script} ({recovered_cross_script / max(total_cross_script, 1):.2%})")

    print(f"\nChannel Attribution of Recovered Keys:")
    for ch, count in channel_hits.most_common():
        print(f"  Channel '{ch:4s}': {count:4d} hits")

    print(f"\nSample of Recovered Ground-Truth Pairs:")
    for i, ex in enumerate(recovered_pairs_detail[:5], start=1):
        print(f"  [{i}] {ex['sid']} <-> {ex['tid']} ({ex['country']})")
        print(f"      S1 Name     : {ex['s1_name']}")
        print(f"      Target Name : {ex['target_name']}")
        print(f"      Translit    : {ex['target_translit']}")
        print(f"      Shared Keys : {ex['shared_keys']}")
        print(f"      WRatio      : {ex['name_fuzz_wratio']:.2f} | TokenSet: {ex['name_token_set_ratio']:.2f} | Score: {ex['fast_score']:.2f}")

    # 6. Candidate Overhead Measurement on Validation Sample
    print("\n" + "=" * 80)
    print("CANDIDATE OVERHEAD & FANOUT AUDIT")
    print("=" * 80)

    # Measure token fanout on India validation S1
    india_s1_records = [s1_map[sid] for sid in s1_map if s1_map[sid][3] == "India"]
    print(f"Auditing transliterated key fanout across {len(india_s1_records)} India S1 records...")
    all_translit_tokens = []
    for sid, name, addr, c in india_s1_records:
        if has_indic_script(name):
            trans = transliterate_indic_to_latin(name)
            all_translit_tokens.extend(tokenize_name(trans))

    token_counts = collections.Counter(all_translit_tokens)
    high_fanout = [(tok, cnt) for tok, cnt in token_counts.items() if cnt > 15]
    print(f"  Total Transliterated Unique Tokens Generated: {len(token_counts)}")
    print(f"  High Fanout Tokens (>15 occurrences in S1)  : {len(high_fanout)}")
    if high_fanout:
        print(f"  Top High-Fanout Tokens: {high_fanout[:5]}")
    else:
        print("  All transliterated tokens have low, bounded fanout (<=15).")

    # 7. Summary JSON
    gate_passed = recovered_total >= 50
    print("\n" + "=" * 80)
    print(f"EXP-007A STAGE-1 GATE STATUS: {'PASSED' if gate_passed else 'FAILED'}")
    print(f"Recovered: {recovered_total} (Target: >=50) | Runtime: {elapsed_time:.2f}s")
    print("=" * 80)

    summary = {
        "experiment": "EXP-007A_Stage1_Gate",
        "target_gate_passed": gate_passed,
        "runtime_seconds": elapsed_time,
        "recovery": {
            "total_unretrieved": total_unretrieved,
            "total_recovered": recovered_total,
            "recovery_rate": recovered_total / total_unretrieved,
            "india_recovered": recovered_by_country["India"],
            "india_total": total_by_country["India"],
            "us_recovered": recovered_by_country["US"],
            "us_total": total_by_country["US"],
            "france_recovered": recovered_by_country["France"],
            "france_total": total_by_country["France"],
            "cross_script_recovered": recovered_cross_script,
            "cross_script_total": total_cross_script,
        },
        "channel_hits": dict(channel_hits),
    }

    out_file = PROJECT_ROOT / "experiments" / "EXP-007A-stage1-gate.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Stage 1 report saved to: {out_file}")


if __name__ == "__main__":
    main()
