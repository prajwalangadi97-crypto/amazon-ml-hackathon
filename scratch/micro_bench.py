#!/usr/bin/env python3
"""Measure speed of feature extraction, XGBoost prediction, and streaming."""
import sys
import time
from pathlib import Path
import numpy as np
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.matching.candidate_compression import (
    preprocess_record,
    extract_features_from_preprocessed,
    compute_cheap_candidate_score,
    fast_score_candidate,
)

FEATURE_NAMES = [
    "country_eq",
    "source_type",
    "s1_name_empty",
    "target_name_empty",
    "s1_addr_empty",
    "target_addr_empty",
    "name_exact_eq",
    "name_fuzz_ratio",
    "name_fuzz_wratio",
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "name_prefix_similarity",
    "name_len_diff",
    "name_len_ratio",
    "addr_fuzz_ratio",
    "addr_token_similarity",
    "numeric_token_overlap",
    "house_number_agreement",
    "postal_pin_overlap",
    "addr_len_diff",
    "cross_field_name_loc",
    "cheap_candidate_score",
]

def benchmark():
    s1 = preprocess_record("s1_1", "Acme Industrial Supplies LLC", "123 Main St, Springfield, IL 62701", "US")
    tgt = preprocess_record("s2_1", "Acme Industrial Supply Inc", "123 Main Street Suite 4, Springfield IL", "US")

    # 1. Measure fast_score_candidate
    N_FAST = 50000
    t0 = time.time()
    for _ in range(N_FAST):
        fast_score_candidate(s1, tgt)
    t_fast = time.time() - t0
    fast_rate = N_FAST / t_fast
    print(f"fast_score_candidate rate: {fast_rate:,.0f} pairs/sec ({t_fast/N_FAST*1e6:.1f} us/pair)")

    # 2. Measure 22-feature extraction
    N_FEAT = 10000
    t0 = time.time()
    for _ in range(N_FEAT):
        f = extract_features_from_preprocessed(s1, tgt)
        f["cheap_candidate_score"] = compute_cheap_candidate_score(f)
    t_feat = time.time() - t0
    feat_rate = N_FEAT / t_feat
    print(f"22-feature extraction rate: {feat_rate:,.0f} pairs/sec ({t_feat/N_FEAT*1e6:.1f} us/pair)")

    # 3. Measure XGBoost scoring
    model_path = PROJECT_ROOT / "artifacts" / "xgb_matcher_exp002b.json"
    bst = xgb.Booster()
    bst.load_model(str(model_path))
    
    # Create sample batch of 10,000 feature rows
    f = extract_features_from_preprocessed(s1, tgt)
    f["cheap_candidate_score"] = compute_cheap_candidate_score(f)
    sample_row = [f[col] for col in FEATURE_NAMES]
    X = np.tile(sample_row, (10000, 1)).astype(np.float32)
    dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
    
    t0 = time.time()
    for _ in range(10):
        preds = bst.predict(dmat)
    t_xgb = time.time() - t0
    xgb_rate = 100000 / t_xgb
    print(f"XGBoost scoring rate: {xgb_rate:,.0f} pairs/sec")

if __name__ == "__main__":
    benchmark()
