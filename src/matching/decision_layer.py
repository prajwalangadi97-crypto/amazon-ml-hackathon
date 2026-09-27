"""Candidate-Group-Aware Decision Layer for Record Linkage.

Implements candidate-group competition and country-stratified decision rules (EXP-003).

Key concepts:
1. Multi-match preservation: Does NOT enforce 1-to-1 matching.
2. Group competition: A candidate must be within `margin_delta` of the entity's
   highest-probability candidate to be accepted, pruning runner-up false positives.
3. Country-stratified thresholding: Accounts for data quality heterogeneity across
   countries (e.g., France clean addresses vs. India noise).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class DecisionLayerConfig:
    """Configuration parameters for candidate-group decision layer."""

    country_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "France": 0.45,
            "US": 0.65,
            "India": 0.55,
        }
    )
    margin_delta: float = 0.30
    default_threshold: float = 0.60
    enforce_group_margin: bool = True

    def get_threshold(self, country: str) -> float:
        """Return the decision threshold for a given country."""
        return self.country_thresholds.get(country, self.default_threshold)


# Canonical V3 decision layer configuration from validated EXP-003A
DEFAULT_V3_DECISION_CONFIG = DecisionLayerConfig(
    country_thresholds={
        "France": 0.45,
        "US": 0.65,
        "India": 0.55,
    },
    margin_delta=0.30,
    default_threshold=0.60,
    enforce_group_margin=True,
)


def filter_entity_candidates(
    candidates: List[Tuple[str, float]],
    country: str,
    config: Optional[DecisionLayerConfig] = None,
) -> List[str]:
    """Filter candidates for a single Source 1 entity using group-aware decision rules.

    Args:
        candidates: List of (target_id, predicted_probability) tuples.
        country: Country string for the Source 1 entity.
        config: DecisionLayerConfig instance (defaults to DEFAULT_V3_DECISION_CONFIG).

    Returns:
        List of selected target_ids that satisfy both the country threshold
        and within-group margin constraint. Preserves candidate order.
    """
    if not candidates:
        return []

    cfg = config or DEFAULT_V3_DECISION_CONFIG
    threshold = cfg.get_threshold(country)

    if not cfg.enforce_group_margin or cfg.margin_delta is None:
        return [tid for tid, p in candidates if p >= threshold]

    max_prob = max(p for _, p in candidates)
    min_prob_allowed = max(threshold, max_prob - cfg.margin_delta)

    return [tid for tid, p in candidates if p >= min_prob_allowed]


def batch_filter_entity_candidates(
    all_s1_candidates: Dict[str, List[Tuple[str, float]]],
    s1_country_map: Dict[str, str],
    config: Optional[DecisionLayerConfig] = None,
) -> Dict[str, List[str]]:
    """Apply decision layer filtering across all Source 1 entities in batch.

    Args:
        all_s1_candidates: Mapping from s1_id to list of (target_id, prob) tuples.
        s1_country_map: Mapping from s1_id to country string.
        config: DecisionLayerConfig instance.

    Returns:
        Mapping from s1_id to list of linked target_ids.
    """
    cfg = config or DEFAULT_V3_DECISION_CONFIG
    results: Dict[str, List[str]] = {}

    for s1_id, candidates in all_s1_candidates.items():
        country = s1_country_map.get(s1_id, "")
        results[s1_id] = filter_entity_candidates(candidates, country, cfg)

    return results
