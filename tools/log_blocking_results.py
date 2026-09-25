"""Script to persist blocking metrics JSON and log EXP-001A-D experiments."""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.logger import ExperimentLogger, ExperimentRecord

data = [
    {
        "id": "EXP-001A",
        "name": "Strategy A: Exact Normalized Name",
        "pair_recall": 0.2436,
        "s2_recall": 0.2336,
        "s3_recall": 0.2530,
        "macro_recall": 0.2669,
        "total_candidates": 16843,
        "reduction_ratio": 0.99999918,
        "avg_candidates": 8.42,
        "median_candidates": 0.0,
        "p95_candidates": 6.0,
        "max_candidates": 2752,
        "zero_candidates": 1326,
        "perfect_recall": 326,
        "missed_matches": 1675,
        "runtime": 77.28,
        "features": ["exact_normalized_name"],
    },
    {
        "id": "EXP-001B",
        "name": "Strategy B: Token Normalized Name",
        "pair_recall": 0.6147,
        "s2_recall": 0.6010,
        "s3_recall": 0.6275,
        "macro_recall": 0.6245,
        "total_candidates": 57750,
        "reduction_ratio": 0.99999720,
        "avg_candidates": 28.86,
        "median_candidates": 3.0,
        "p95_candidates": 111.0,
        "max_candidates": 1944,
        "zero_candidates": 651,
        "perfect_recall": 928,
        "missed_matches": 1073,
        "runtime": 77.28,
        "features": ["token_pairs_salient_tokens"],
    },
    {
        "id": "EXP-001C",
        "name": "Strategy C: Address Blocking",
        "pair_recall": 0.8808,
        "s2_recall": 0.8889,
        "s3_recall": 0.8732,
        "macro_recall": 0.8742,
        "total_candidates": 137844,
        "reduction_ratio": 0.99999332,
        "avg_candidates": 68.89,
        "median_candidates": 7.0,
        "p95_candidates": 289.0,
        "max_candidates": 3470,
        "zero_candidates": 204,
        "perfect_recall": 1583,
        "missed_matches": 418,
        "runtime": 77.28,
        "features": ["address_number_locality_keys"],
    },
    {
        "id": "EXP-001D",
        "name": "Strategy D: Union Blocking (A + B + C)",
        "pair_recall": 0.9755,
        "s2_recall": 0.9747,
        "s3_recall": 0.9763,
        "macro_recall": 0.9735,
        "total_candidates": 191732,
        "reduction_ratio": 0.99999072,
        "avg_candidates": 95.82,
        "median_candidates": 16.0,
        "p95_candidates": 395.0,
        "max_candidates": 3674,
        "zero_candidates": 44,
        "perfect_recall": 1863,
        "missed_matches": 138,
        "runtime": 77.28,
        "features": ["exact_name", "token_name", "address_composite"],
    },
]

out_dir = Path("experiments/blocking")
out_dir.mkdir(parents=True, exist_ok=True)
logger = ExperimentLogger("experiments")

for d in data:
    with open(out_dir / f"{d['id']}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

    rec = ExperimentRecord(
        experiment_id=d["id"],
        hypothesis=f"Evaluate candidate retrieval recall and volume for {d['name']}.",
        data_split_description="Validation benchmark split (10,000 S1 sample -> 2,001 val S1) against full S2+S3 search pool (10,320,219 records).",
        blocking_method=d["name"],
        features=d["features"],
        model="Inverted Index Hash Table",
        hyperparameters={"country_restricted": True},
        validation_method="Candidate Pair Recall & Reduction Ratio",
        macro_F0_5=d["macro_recall"],
        precision=round(6945 * d["pair_recall"] / max(1, d["total_candidates"]), 6),
        recall=d["pair_recall"],
        training_time_seconds=0.0,
        inference_time_seconds=d["runtime"],
        notes=f"Avg cands/S1: {d['avg_candidates']:.2f}, Max: {d['max_candidates']:,}, Total pairs: {d['total_candidates']:,}",
        conclusion=f"Pair recall: {d['pair_recall']*100:.2f}%, S2 recall: {d['s2_recall']*100:.2f}%, S3 recall: {d['s3_recall']*100:.2f}%, Reduction Ratio: {d['reduction_ratio']*100:.6f}%.",
        next_experiment="EXP-001B" if d["id"] == "EXP-001A" else ("EXP-001C" if d["id"] == "EXP-001B" else ("EXP-001D" if d["id"] == "EXP-001C" else "EXP-002")),
        extra_metrics={
            "total_candidate_pairs": d["total_candidates"],
            "reduction_ratio": d["reduction_ratio"],
            "avg_candidates_per_s1": d["avg_candidates"],
            "median_candidates_per_s1": d["median_candidates"],
            "p95_candidates_per_s1": d["p95_candidates"],
            "max_candidates_per_s1": d["max_candidates"],
            "s2_recall": d["s2_recall"],
            "s3_recall": d["s3_recall"],
            "zero_candidate_entities": d["zero_candidates"],
            "perfect_recall_entities": d["perfect_recall"],
        },
    )
    logger.log(rec)

print("Experiments logged successfully.")
print(logger.summary_table())
