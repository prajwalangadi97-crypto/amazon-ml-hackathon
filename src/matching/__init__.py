"""Matching, candidate compression, and decision layer package."""

from src.matching.candidate_compression import (
    CandidateCompressor,
    CompressionPolicy,
    compute_cheap_candidate_score,
    compute_cheap_pairwise_features,
)
from src.matching.decision_layer import (
    DEFAULT_V3_DECISION_CONFIG,
    DecisionLayerConfig,
    batch_filter_entity_candidates,
    filter_entity_candidates,
)

__all__ = [
    "CandidateCompressor",
    "CompressionPolicy",
    "compute_cheap_candidate_score",
    "compute_cheap_pairwise_features",
    "DecisionLayerConfig",
    "DEFAULT_V3_DECISION_CONFIG",
    "filter_entity_candidates",
    "batch_filter_entity_candidates",
]
