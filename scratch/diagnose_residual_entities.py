import collections
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ground_truth import parse_ground_truth
from src.matching.candidate_compression import (
    compute_cheap_candidate_score,
    extract_features_from_preprocessed,
    preprocess_record,
)
from src.matching.decision_layer import (
    DEFAULT_V3_DECISION_CONFIG,
    filter_entity_candidates,
)
from tools.run_exp005_adaptive_ranking import (
    rank_candidates_channel_consensus_first,
    EXP004B_FEATURE_NAMES,
)

sys.stdout.reconfigure(encoding='utf-8')

# 1. Load split & ground truth
with open('experiments/splits/validation_split_seed42_benchmark_10000.json', 'r') as f:
    val_s1 = json.load(f)['val_s1_ids']
val_s1_set = set(val_s1)
gt = parse_ground_truth('dataset/train/train_ground_truth.tsv', allowed_s1_ids=val_s1_set)

# 2. Load S1 preprocessed
val_s1_records = {}
val_s1_prep = {}
with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8-sig') as f:
    next(f)
    for line in f:
        parts = line.rstrip('\r\n').split('\t')
        if len(parts) >= 4 and parts[0] in val_s1_set:
            val_s1_records[parts[0]] = (parts[1], parts[2], parts[3])
            val_s1_prep[parts[0]] = preprocess_record(parts[0], parts[1], parts[2], parts[3])

# 3. Load raw candidates pool
with open('artifacts/exp005/val_raw_candidates.pkl', 'rb') as f:
    raw_pool = pickle.load(f)

# 4. Generate candidates using Channel Consensus First
selected_cands_meta = {}
needed_tids = set()
for sid in val_s1:
    raw_c = raw_pool.get(sid, [])
    selected = rank_candidates_channel_consensus_first(raw_c, cap=100)
    selected_cands_meta[sid] = selected
    needed_tids.update(t for t, _ in selected)

print(f"Candidates generated for {len(val_s1)} entities ({len(needed_tids):,} unique targets).")

# 5. Load targets
target_prep_map = {}
for src_fp in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
    with open(src_fp, 'r', encoding='utf-8-sig') as f:
        next(f)
        for line in f:
            pos = line.find('\t')
            if pos == -1:
                continue
            tid = line[:pos]
            if tid in needed_tids:
                parts = line.rstrip('\r\n').split('\t')
                if len(parts) >= 4:
                    target_prep_map[tid] = preprocess_record(parts[0], parts[1], parts[2], parts[3])
                    if len(target_prep_map) == len(needed_tids):
                        break
    if len(target_prep_map) == len(needed_tids):
        break

print(f"Loaded {len(target_prep_map):,} target records.")

# 6. Score with XGBoost
bst = xgb.Booster()
bst.load_model('artifacts/xgb_matcher_exp004b.json')
bst.set_param({'nthread': 8})

X_rows = []
pair_keys = []
for sid in val_s1:
    s1_p = val_s1_prep[sid]
    for tid, meta in selected_cands_meta.get(sid, []):
        tgt_p = target_prep_map.get(tid)
        if tgt_p is None:
            continue
        feats = extract_features_from_preprocessed(s1_p, tgt_p)
        feats['cheap_candidate_score'] = compute_cheap_candidate_score(feats)
        feats.update(meta)
        X_rows.append([feats[c] for c in EXP004B_FEATURE_NAMES])
        pair_keys.append((sid, tid))

dmat = xgb.DMatrix(np.array(X_rows, dtype=np.float32), feature_names=EXP004B_FEATURE_NAMES)
probs = bst.predict(dmat)

s1_cand_scores = collections.defaultdict(list)
for (sid, tid), p in zip(pair_keys, probs):
    s1_cand_scores[sid].append((tid, float(p)))

# 7. Apply Decision Layer
predictions = {}
empty_pred_s1 = []
non_empty_pred_s1 = []

for sid in val_s1:
    cands = s1_cand_scores.get(sid, [])
    country = val_s1_records[sid][2]
    matched = filter_entity_candidates(cands, country, DEFAULT_V3_DECISION_CONFIG)
    predictions[sid] = set(matched)
    if matched:
        non_empty_pred_s1.append(sid)
    else:
        empty_pred_s1.append(sid)

print(f"\nEntities with >= 1 predicted link : {len(non_empty_pred_s1):,}")
print(f"Entities with 0 predicted links    : {len(empty_pred_s1):,}")

# Analyze empty prediction entities
empty_gt_empty = 0
empty_gt_nonempty = 0
missed_true_links = []

for sid in empty_pred_s1:
    true_tids = gt.get(sid, set())
    if not true_tids:
        empty_gt_empty += 1
    else:
        empty_gt_nonempty += 1
        missed_true_links.append((sid, true_tids, len(selected_cands_meta.get(sid, []))))

print(f"  - True singletons correctly left empty : {empty_gt_empty} (Singleton accuracy: {empty_gt_empty / max(1, sum(1 for s in val_s1 if not gt.get(s))):.2%})")
print(f"  - False negatives (unresolved entities): {empty_gt_nonempty}")

print(f"\nTotal missed true links among unresolved entities: {sum(len(t) for _, t, _ in missed_true_links)}")
print("\nFirst 10 unresolved entities:")
for sid, true_tids, cand_cnt in missed_true_links[:10]:
    s1_name, s1_addr, s1_c = val_s1_records[sid]
    print(f"  S1: {sid} ({s1_c}) | Name: '{s1_name}' | Addr: '{s1_addr}' | Cands: {cand_cnt} | GT: {true_tids}")
