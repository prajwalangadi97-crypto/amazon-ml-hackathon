import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ground_truth import parse_ground_truth
from tools.run_exp005_adaptive_ranking import rank_candidates_channel_consensus_first

sys.stdout.reconfigure(encoding='utf-8')

# 1. Load split & ground truth
with open('experiments/splits/validation_split_seed42_benchmark_10000.json') as f:
    val_s1 = json.load(f)['val_s1_ids']
val_s1_set = set(val_s1)
gt = parse_ground_truth('dataset/train/train_ground_truth.tsv', allowed_s1_ids=val_s1_set)

# 2. Load S1 records
val_s1_records = {}
with open('dataset/train/train_source1.tsv', 'r', encoding='utf-8-sig') as f:
    next(f)
    for line in f:
        parts = line.rstrip('\r\n').split('\t')
        if len(parts) >= 4 and parts[0] in val_s1_set:
            val_s1_records[parts[0]] = (parts[1], parts[2], parts[3])

# 3. Load raw candidates pool
with open('artifacts/exp005/val_raw_candidates.pkl', 'rb') as f:
    raw_pool = pickle.load(f)

# 4. Check all 80 unresolved entities
# An entity is unresolved if it has true links in GT but had 0 predicted links in Phase 1
# Let's inspect the coverage of true links
total_gt_all = 0
in_selected_all = 0
in_raw_only_all = 0
not_in_raw_all = 0

unresolved_s1_list = []

for sid in val_s1:
    true_tids = gt.get(sid, set())
    if not true_tids:
        continue
    raw_c = raw_pool.get(sid, [])
    selected = rank_candidates_channel_consensus_first(raw_c, cap=100)
    sel_tids = {t for t, _ in selected}
    raw_tids = {c['tid'] for c in raw_c}

    for tid in true_tids:
        total_gt_all += 1
        if tid in sel_tids:
            in_selected_all += 1
        elif tid in raw_tids:
            in_raw_only_all += 1
        else:
            not_in_raw_all += 1
            unresolved_s1_list.append((sid, tid))

print(f"Overall Ground Truth Links ({total_gt_all:,}):")
print(f"  In selected candidates (cap 100) : {in_selected_all:,} ({in_selected_all / total_gt_all:.2%})")
print(f"  In raw pool but dropped by cap   : {in_raw_only_all:,} ({in_raw_only_all / total_gt_all:.2%})")
print(f"  Not retrieved in raw pool at all : {not_in_raw_all:,} ({not_in_raw_all / total_gt_all:.2%})")

print(f"\nBreakdown of the {not_in_raw_all} true links not retrieved at all:")
# Sample 10 misses that were never retrieved:
print("Sample 10 unretrieved true links:")
for sid, tid in unresolved_s1_list[:10]:
    s1_name, s1_addr, s1_c = val_s1_records[sid]
    print(f"  S1: {sid} ({s1_c}) | '{s1_name}' | '{s1_addr}' -> Target: {tid}")
