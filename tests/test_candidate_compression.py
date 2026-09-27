"""Unit tests for candidate compression and cheap pairwise similarity matching."""

import unittest

from src.matching.candidate_compression import (
    CandidateCompressor,
    CompressionPolicy,
    compute_cheap_candidate_score,
    compute_cheap_pairwise_features,
    extract_features_from_preprocessed,
    fast_score_candidate,
    preprocess_record,
)


class TestCandidateCompressionFeatures(unittest.TestCase):
    """Test feature extraction and normalization behavior."""

    def test_exact_name_match(self):
        s1 = ("Acme Supplies LLC", "100 Main St, Austin, TX 78701", "US")
        t = ("Acme Supplies Inc", "100 Main St, Austin, TX 78701", "US")
        feats = compute_cheap_pairwise_features(
            s1_name=s1[0],
            s1_addr=s1[1],
            s1_country=s1[2],
            target_name=t[0],
            target_addr=t[1],
            target_country=t[2],
            target_id="S2-123",
            s1_id="S1-001",
        )
        self.assertEqual(feats["country_eq"], 1.0)
        self.assertEqual(feats["source_type"], 0.0)  # S2
        self.assertEqual(feats["house_number_agreement"], 1.0)
        self.assertEqual(feats["postal_pin_overlap"], 1.0)
        self.assertGreater(feats["name_fuzz_wratio"], 0.7)

    def test_country_mismatch_yields_zero_score(self):
        feats = compute_cheap_pairwise_features(
            s1_name="Bharat Trading",
            s1_addr="Sector 18, Noida",
            s1_country="India",
            target_name="Bharat Trading",
            target_addr="Sector 18, Noida",
            target_country="US",
            target_id="S2-100",
        )
        self.assertEqual(feats["country_eq"], 0.0)
        score = compute_cheap_candidate_score(feats)
        self.assertEqual(score, 0.0)

    def test_missing_address_handling(self):
        feats = compute_cheap_pairwise_features(
            s1_name="Apex Logistics",
            s1_addr="",
            s1_country="US",
            target_name="Apex Logistics",
            target_addr="",
            target_country="US",
            target_id="S3-999",
        )
        self.assertEqual(feats["s1_addr_empty"], 1.0)
        self.assertEqual(feats["target_addr_empty"], 1.0)
        self.assertEqual(feats["source_type"], 1.0)  # S3
        self.assertEqual(feats["name_exact_eq"], 1.0)
        score = compute_cheap_candidate_score(feats)
        # S_name=1.0, 10% uncertainty penalty for missing address -> 0.90
        self.assertAlmostEqual(score, 0.90, places=3)

    def test_unicode_and_indic_scripts(self):
        s1_name = "हरियाणा ट्रेडिंग"
        t_name = "हरियाणा ट्रेडिंग"
        s1_p = preprocess_record("S1-1", s1_name, "Plot 12, Gurugram", "India")
        tgt_p = preprocess_record("S2-1", t_name, "Plot 12, Gurugram", "India")
        score = fast_score_candidate(s1_p, tgt_p)
        self.assertGreaterEqual(score, 0.95)

    def test_fast_score_matches_compute_cheap_score(self):
        s1_p = preprocess_record("S1-1", "Raj Investments LLP", "6(29), CIT Colony, Chennai, 600004", "India")
        tgt_p = preprocess_record("S2-1", "Raj Investments Pvt Ltd", "6(29), CIT Colony, Chennai, 600004", "India")
        feats = extract_features_from_preprocessed(s1_p, tgt_p)
        dict_score = compute_cheap_candidate_score(feats)
        fast_score = fast_score_candidate(s1_p, tgt_p)
        self.assertAlmostEqual(dict_score, fast_score, places=4)


class TestCompressionPolicies(unittest.TestCase):
    """Test policy filtering and compression logic."""

    def setUp(self):
        # 5 candidates with decreasing scores
        self.scored_cands = [
            (0.95, "S2-001"),
            (0.75, "S2-002"),
            (0.55, "S3-001"),
            (0.35, "S2-003"),
            (0.15, "S3-002"),
        ]

    def test_singleton_entity_preserves_zero_candidates(self):
        policy = CompressionPolicy("top_25", "Top 25", method="top_k", top_k=25)
        res = CandidateCompressor.compress_entity_candidates([], policy)
        self.assertEqual(res, set())

    def test_top_k_policy(self):
        policy = CompressionPolicy("top_2", "Top 2", method="top_k", top_k=2)
        res = CandidateCompressor.compress_entity_candidates(self.scored_cands, policy)
        self.assertEqual(res, {"S2-001", "S2-002"})

    def test_threshold_policy(self):
        policy = CompressionPolicy("t_0.50", "Threshold 0.50", method="threshold", min_score=0.50)
        res = CandidateCompressor.compress_entity_candidates(self.scored_cands, policy)
        self.assertEqual(res, {"S2-001", "S2-002", "S3-001"})

    def test_hybrid_policy(self):
        # Score >= 0.30, cap at 2 candidates
        policy = CompressionPolicy(
            "hybrid_t0.30_cap2",
            "Hybrid T0.30 Cap 2",
            method="hybrid",
            min_score=0.30,
            max_cap=2,
        )
        res = CandidateCompressor.compress_entity_candidates(self.scored_cands, policy)
        self.assertEqual(len(res), 2)
        self.assertEqual(res, {"S2-001", "S2-002"})

    def test_two_tier_policy(self):
        policy = CompressionPolicy(
            "two_tier",
            "Two Tier",
            method="two_tier",
            high_threshold=0.70,
            low_threshold=0.30,
            max_cap=3,
        )
        res = CandidateCompressor.compress_entity_candidates(self.scored_cands, policy)
        # High tier: S2-001 (0.95), S2-002 (0.75) -> 2 items
        # Mid tier: S3-001 (0.55), S2-003 (0.35) -> fills 1 more item (budget = 3)
        self.assertEqual(len(res), 3)
        self.assertIn("S2-001", res)
        self.assertIn("S2-002", res)
        self.assertIn("S3-001", res)
        self.assertNotIn("S3-002", res)


if __name__ == "__main__":
    unittest.main()
