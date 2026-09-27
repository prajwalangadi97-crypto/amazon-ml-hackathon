import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
data = json.load(open('experiments/validation_missed_links_analysis.json', 'r', encoding='utf-8'))
cap_misses = data['category_examples']['candidate-cap/ranking loss']

print(f"Total cap misses: {len(cap_misses)}")

countries = Counter(x['country'] for x in cap_misses)
print("Countries:", countries)

shared_key_counts = [len(x.get('shared_keys', [])) for x in cap_misses]
print(f"Shared keys per pair: min={min(shared_key_counts)}, max={max(shared_key_counts)}, avg={sum(shared_key_counts)/len(shared_key_counts):.2f}")

scores = [x.get('fast_score', 0.0) for x in cap_misses]
print(f"Fast scores: min={min(scores):.3f}, max={max(scores):.3f}, avg={sum(scores)/len(scores):.3f}")

print("\nFirst 10 cap misses:")
for i, x in enumerate(cap_misses[:10]):
    print(f"[{i+1}] S1: {x['s1_id']} ({x['s1_name']} | {x['s1_addr']})")
    print(f"     Target: {x['target_id']} ({x['target_name']} | {x['target_addr']})")
    print(f"     Country: {x['country']} | Keys ({len(x['shared_keys'])}): {x['shared_keys'][:4]} | Fast Score: {x.get('fast_score', 0):.3f}")
