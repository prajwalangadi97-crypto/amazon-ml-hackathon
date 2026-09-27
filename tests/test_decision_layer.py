"""Unit tests for candidate-group decision layer."""

import unittest
from src.matching.decision_layer import (
    DecisionLayerConfig,
    DEFAULT_V3_DECISION_CONFIG,
    filter_entity_candidates,
    batch_filter_entity_candidates,
)


class TestDecisionLayer(unittest.TestCase):
    """Test suite for DecisionLayerConfig and candidate filtering functions."""

    def setUp(self):
        self.config = DEFAULT_V3_DECISION_CONFIG

    def test_empty_candidates(self):
        """Empty candidate list should return empty matches."""
        self.assertEqual(filter_entity_candidates([], "France", self.config), [])
        self.assertEqual(filter_entity_candidates([], "India", self.config), [])

    def test_single_candidate_above_threshold(self):
        """Single candidate meeting country threshold should match."""
        # France threshold is 0.45
        cands = [("T1", 0.50)]
        matched = filter_entity_candidates(cands, "France", self.config)
        self.assertEqual(matched, ["T1"])

        # US threshold is 0.65; 0.50 should fail
        matched_us = filter_entity_candidates(cands, "US", self.config)
        self.assertEqual(matched_us, [])

    def test_multi_match_preservation(self):
        """Multiple candidates within margin should BOTH match (no 1-to-1 constraint)."""
        # India threshold is 0.55; margin delta is 0.30
        cands = [
            ("T1", 0.92),
            ("T2", 0.88),  # 0.92 - 0.88 = 0.04 <= 0.30 -> MATCH
            ("T3", 0.65),  # 0.92 - 0.65 = 0.27 <= 0.30 -> MATCH
            ("T4", 0.58),  # 0.92 - 0.58 = 0.34 > 0.30 -> REJECTED (exceeds margin)
            ("T5", 0.40),  # 0.40 < 0.55 -> REJECTED (below threshold)
        ]
        matched = filter_entity_candidates(cands, "India", self.config)
        self.assertEqual(matched, ["T1", "T2", "T3"])

    def test_country_thresholds(self):
        """Verify country-specific thresholding behavior."""
        cands = [("T1", 0.50)]
        # France: 0.50 >= 0.45 -> MATCH
        self.assertEqual(filter_entity_candidates(cands, "France", self.config), ["T1"])
        # India: 0.50 < 0.55 -> REJECT
        self.assertEqual(filter_entity_candidates(cands, "India", self.config), [])
        # US: 0.50 < 0.65 -> REJECT
        self.assertEqual(filter_entity_candidates(cands, "US", self.config), [])

    def test_fallback_threshold(self):
        """Unknown country falls back to default_threshold (0.60)."""
        cands = [("T1", 0.62)]
        self.assertEqual(filter_entity_candidates(cands, "UnknownCountry", self.config), ["T1"])
        cands_low = [("T1", 0.58)]
        self.assertEqual(filter_entity_candidates(cands_low, "UnknownCountry", self.config), [])

    def test_batch_filtering(self):
        """Batch filtering correctly maps across entities and countries."""
        all_cands = {
            "S1": [("T1", 0.90), ("T2", 0.50)],  # France: 0.50 is 0.40 below max -> reject T2
            "S2": [("T3", 0.48)],                # France: >= 0.45 -> match T3
            "S3": [("T4", 0.60)],                # US: < 0.65 -> reject T4
        }
        country_map = {"S1": "France", "S2": "France", "S3": "US"}
        results = batch_filter_entity_candidates(all_cands, country_map, self.config)
        self.assertEqual(results["S1"], ["T1"])
        self.assertEqual(results["S2"], ["T3"])
        self.assertEqual(results["S3"], [])


if __name__ == "__main__":
    unittest.main()
