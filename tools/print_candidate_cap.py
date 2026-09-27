#!/usr/bin/env python3
import json

with open("experiments/validation_missed_links_analysis.json", "r", encoding="utf-8") as f:
    d = json.load(f)

cat = "candidate-cap/ranking loss"
cnt = d["category_counts"][cat]
pct = d["category_percentages"][cat]
print(f"### {cat.title()} ({cnt} misses, {pct:.2f}%)\n")
for i, ex in enumerate(d["category_examples"][cat][:6]):
    s1 = ex["s1_name"].encode("ascii", "replace").decode("ascii")
    t = ex["target_name"].encode("ascii", "replace").decode("ascii")
    a1 = ex["s1_addr"].encode("ascii", "replace").decode("ascii")
    a2 = ex["target_addr"].encode("ascii", "replace").decode("ascii")
    sc = ex["fast_score"]
    sh = ex.get("shared_keys", [])
    cnt_s1 = ex.get("s1_candidate_count", 0)
    print(f"{i+1}. **`{ex['s1_id']}` $\\leftrightarrow$ `{ex['target_id']}`** ({ex['country']})")
    print(f"   * **S1:** `{s1}` | `{a1}`")
    print(f"   * **Target:** `{t}` | `{a2}`")
    print(f"   * **Metrics:** Fast score = `{sc:.3f}` | S1 Candidate Count = `{cnt_s1}` | Shared Keys = `{sh[:2]}`")
