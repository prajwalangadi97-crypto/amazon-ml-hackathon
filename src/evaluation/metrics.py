"""Official F0.5 evaluation metric for Amazon ML Challenge 2026.

In this challenge, entity resolution is scored using macro-averaged F0.5:
    F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Computed per Source 1 entity and then macro-averaged across ALL Source 1 entities
in the evaluation set.

Singleton (no-match) handling follows the official specification:
    - true matches empty AND predicted matches empty => score = 1.0 (Precision=1.0, Recall=1.0)
    - true matches empty AND predicted matches non-empty => score = 0.0 (Precision=0.0, Recall=0.0)
    - true matches non-empty AND predicted matches empty => score = 0.0 (Precision=0.0, Recall=0.0)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Set


BETA = 0.5
BETA_SQ = BETA ** 2  # 0.25
NUMERATOR_WEIGHT = 1.0 + BETA_SQ  # 1.25


@dataclass(frozen=True)
class EntityMetricResult:
    """Evaluation result for a single Source 1 entity."""

    source1_id: str
    true_count: int
    predicted_count: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f0_5: float
    is_singleton: bool


@dataclass
class EvaluationReport:
    """Macro-averaged evaluation report across all evaluated Source 1 entities."""

    total_entities: int
    macro_f0_5: float
    macro_precision: float
    macro_recall: float
    singleton_count: int
    singleton_f0_5: float
    non_singleton_count: int
    non_singleton_f0_5: float
    total_tp: int
    total_fp: int
    total_fn: int
    missing_predictions: int = 0
    cardinality_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to standard dictionary format."""
        return {
            "total_entities": self.total_entities,
            "macro_f0_5": round(self.macro_f0_5, 6),
            "macro_precision": round(self.macro_precision, 6),
            "macro_recall": round(self.macro_recall, 6),
            "singleton_count": self.singleton_count,
            "singleton_f0_5": round(self.singleton_f0_5, 6),
            "non_singleton_count": self.non_singleton_count,
            "non_singleton_f0_5": round(self.non_singleton_f0_5, 6),
            "total_tp": self.total_tp,
            "total_fp": self.total_fp,
            "total_fn": self.total_fn,
            "missing_predictions": self.missing_predictions,
            "cardinality_breakdown": self.cardinality_breakdown,
        }

    def summary(self) -> str:
        """Return formatted human-readable summary."""
        lines = [
            "=" * 60,
            "Amazon ML Challenge 2026 - Evaluation Report",
            "=" * 60,
            f"Total Source 1 Entities : {self.total_entities:,}",
            f"Macro F0.5 Score        : {self.macro_f0_5:.6f}",
            f"Macro Precision         : {self.macro_precision:.6f}",
            f"Macro Recall            : {self.macro_recall:.6f}",
            "-" * 60,
            f"Singletons (no-match)   : {self.singleton_count:,} (Score: {self.singleton_f0_5:.6f})",
            f"Non-Singletons (matches): {self.non_singleton_count:,} (Score: {self.non_singleton_f0_5:.6f})",
            f"Total True Positives    : {self.total_tp:,}",
            f"Total False Positives   : {self.total_fp:,}",
            f"Total False Negatives   : {self.total_fn:,}",
        ]
        if self.missing_predictions > 0:
            lines.append(f"Missing Predictions     : {self.missing_predictions:,} (treated as empty)")
        if self.cardinality_breakdown:
            lines.append("-" * 60)
            lines.append("Breakdown by Match Cardinality:")
            for bucket, stats in sorted(self.cardinality_breakdown.items()):
                lines.append(
                    f"  {bucket:<15}: count={int(stats['count']):,}, "
                    f"macro_f0.5={stats['macro_f0_5']:.4f}, "
                    f"p={stats['precision']:.4f}, r={stats['recall']:.4f}"
                )
        lines.append("=" * 60)
        return "\n".join(lines)


def compute_entity_metrics(
    predicted_ids: Set[str],
    true_ids: Set[str],
    source1_id: str = "",
) -> EntityMetricResult:
    """Compute precision, recall, and F0.5 for a single Source 1 entity.

    Rules:
    - true is empty and predicted is empty => precision=1.0, recall=1.0, f0.5=1.0
    - true is empty and predicted is not empty => precision=0.0, recall=0.0, f0.5=0.0
    - true is not empty and predicted is empty => precision=0.0, recall=0.0, f0.5=0.0
    - true and predicted are both non-empty:
        TP = |predicted & true|
        FP = |predicted - true|
        FN = |true - predicted|
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    """
    is_singleton = len(true_ids) == 0

    if is_singleton:
        if len(predicted_ids) == 0:
            return EntityMetricResult(
                source1_id=source1_id,
                true_count=0,
                predicted_count=0,
                tp=0,
                fp=0,
                fn=0,
                precision=1.0,
                recall=1.0,
                f0_5=1.0,
                is_singleton=True,
            )
        else:
            return EntityMetricResult(
                source1_id=source1_id,
                true_count=0,
                predicted_count=len(predicted_ids),
                tp=0,
                fp=len(predicted_ids),
                fn=0,
                precision=0.0,
                recall=0.0,
                f0_5=0.0,
                is_singleton=True,
            )

    # Non-singleton entity
    if len(predicted_ids) == 0:
        return EntityMetricResult(
            source1_id=source1_id,
            true_count=len(true_ids),
            predicted_count=0,
            tp=0,
            fp=0,
            fn=len(true_ids),
            precision=0.0,
            recall=0.0,
            f0_5=0.0,
            is_singleton=False,
        )

    tp_set = predicted_ids & true_ids
    tp = len(tp_set)
    fp = len(predicted_ids) - tp
    fn = len(true_ids) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    denom = (BETA_SQ * precision) + recall
    if denom > 0.0:
        f0_5 = (NUMERATOR_WEIGHT * precision * recall) / denom
    else:
        f0_5 = 0.0

    return EntityMetricResult(
        source1_id=source1_id,
        true_count=len(true_ids),
        predicted_count=len(predicted_ids),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f0_5=f0_5,
        is_singleton=False,
    )


def evaluate_predictions(
    predictions: Mapping[str, Set[str]],
    ground_truth: Mapping[str, Set[str]],
    allow_missing: bool = True,
) -> EvaluationReport:
    """Compute official macro-averaged F0.5 over all entities in ground_truth.

    Args:
        predictions: Mapping of source1_entity_id -> set of predicted matched entity IDs.
        ground_truth: Mapping of source1_entity_id -> set of true matched entity IDs.
        allow_missing: If True, entities missing in predictions are treated as
            predicting the empty set (correct for singletons, 0.0 for non-singletons).
            If False, missing keys raise KeyError.

    Returns:
        EvaluationReport containing macro F0.5, precision, recall, and subgroup breakdowns.
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return EvaluationReport(
            total_entities=0,
            macro_f0_5=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            singleton_count=0,
            singleton_f0_5=0.0,
            non_singleton_count=0,
            non_singleton_f0_5=0.0,
            total_tp=0,
            total_fp=0,
            total_fn=0,
        )

    sum_f0_5 = 0.0
    sum_precision = 0.0
    sum_recall = 0.0

    singleton_count = 0
    singleton_f0_5_sum = 0.0

    non_singleton_count = 0
    non_singleton_f0_5_sum = 0.0

    total_tp = 0
    total_fp = 0
    total_fn = 0
    missing_count = 0

    # Cardinality tracking: 0 (singleton), 1, 2, 3+
    bucket_counts: Dict[str, int] = {"0_singleton": 0, "1_match": 0, "2_match": 0, "3+_match": 0}
    bucket_f0_5: Dict[str, float] = {"0_singleton": 0.0, "1_match": 0.0, "2_match": 0.0, "3+_match": 0.0}
    bucket_prec: Dict[str, float] = {"0_singleton": 0.0, "1_match": 0.0, "2_match": 0.0, "3+_match": 0.0}
    bucket_rec: Dict[str, float] = {"0_singleton": 0.0, "1_match": 0.0, "2_match": 0.0, "3+_match": 0.0}

    for s1_id, true_set in ground_truth.items():
        if s1_id in predictions:
            pred_set = predictions[s1_id]
        elif allow_missing:
            pred_set = set()
            missing_count += 1
        else:
            raise KeyError(
                f"Source 1 entity '{s1_id}' is missing from predictions dictionary."
            )

        res = compute_entity_metrics(pred_set, true_set, source1_id=s1_id)

        sum_f0_5 += res.f0_5
        sum_precision += res.precision
        sum_recall += res.recall

        total_tp += res.tp
        total_fp += res.fp
        total_fn += res.fn

        if res.is_singleton:
            singleton_count += 1
            singleton_f0_5_sum += res.f0_5
            bucket = "0_singleton"
        else:
            non_singleton_count += 1
            non_singleton_f0_5_sum += res.f0_5
            if res.true_count == 1:
                bucket = "1_match"
            elif res.true_count == 2:
                bucket = "2_match"
            else:
                bucket = "3+_match"

        bucket_counts[bucket] += 1
        bucket_f0_5[bucket] += res.f0_5
        bucket_prec[bucket] += res.precision
        bucket_rec[bucket] += res.recall

    macro_f0_5 = sum_f0_5 / total_entities
    macro_precision = sum_precision / total_entities
    macro_recall = sum_recall / total_entities

    singleton_f0_5 = (singleton_f0_5_sum / singleton_count) if singleton_count > 0 else 0.0
    non_singleton_f0_5 = (non_singleton_f0_5_sum / non_singleton_count) if non_singleton_count > 0 else 0.0

    cardinality_breakdown: Dict[str, Dict[str, float]] = {}
    for bucket, count in bucket_counts.items():
        if count > 0:
            cardinality_breakdown[bucket] = {
                "count": float(count),
                "macro_f0_5": round(bucket_f0_5[bucket] / count, 6),
                "precision": round(bucket_prec[bucket] / count, 6),
                "recall": round(bucket_rec[bucket] / count, 6),
            }

    return EvaluationReport(
        total_entities=total_entities,
        macro_f0_5=macro_f0_5,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        singleton_count=singleton_count,
        singleton_f0_5=singleton_f0_5,
        non_singleton_count=non_singleton_count,
        non_singleton_f0_5=non_singleton_f0_5,
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        missing_predictions=missing_count,
        cardinality_breakdown=cardinality_breakdown,
    )
