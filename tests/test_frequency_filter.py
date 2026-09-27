"""Unit tests for high-frequency blocking key filtering."""

import unittest
from src.blocking.frequency_filter import (
    BlockingKeyFilterConfig,
    compute_effective_fanout_threshold,
    prune_high_frequency_keys,
)


class TestFrequencyFilter(unittest.TestCase):
    def setUp(self):
        self.raw_c_idx = {
            "tk_rare": ["s1", "s2"],
            "tk_medium": ["s1", "s2", "s3", "s4", "s5"],
            "tk_pathological": [f"s{i}" for i in range(50)],
            "en_common_company": [f"s{i}" for i in range(40)],
            "pin_75001": [f"s{i}" for i in range(35)],
        }
        self.total_s1 = 100

    def test_disabled_by_default(self):
        """When config is None or empty, index is unchanged."""
        pruned, stats = prune_high_frequency_keys(self.raw_c_idx, self.total_s1, config=None)
        self.assertFalse(stats["enabled"])
        self.assertEqual(len(pruned), len(self.raw_c_idx))

        cfg_empty = BlockingKeyFilterConfig()
        pruned_empty, stats_empty = prune_high_frequency_keys(self.raw_c_idx, self.total_s1, config=cfg_empty)
        self.assertFalse(stats_empty["enabled"])
        self.assertEqual(len(pruned_empty), len(self.raw_c_idx))

    def test_absolute_fanout_pruning(self):
        """Keys exceeding max_key_fanout are pruned, rare keys are preserved."""
        cfg = BlockingKeyFilterConfig(
            max_key_fanout=10,
            preserve_exact_name=True,
            preserve_pin_keys=True,
        )
        pruned, stats = prune_high_frequency_keys(self.raw_c_idx, self.total_s1, config=cfg)
        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["effective_threshold"], 10)
        self.assertIn("tk_rare", pruned)
        self.assertIn("tk_medium", pruned)
        self.assertNotIn("tk_pathological", pruned)
        # Exemptions
        self.assertIn("en_common_company", pruned)
        self.assertIn("pin_75001", pruned)

    def test_ratio_fanout_pruning(self):
        """Ratio threshold calculates integer cap based on total_s1."""
        # 10% of 100 is 10
        cfg = BlockingKeyFilterConfig(
            max_key_fanout_ratio=0.10,
            preserve_exact_name=False,
            preserve_pin_keys=False,
        )
        pruned, stats = prune_high_frequency_keys(self.raw_c_idx, self.total_s1, config=cfg)
        self.assertEqual(stats["effective_threshold"], 10)
        self.assertIn("tk_rare", pruned)
        self.assertIn("tk_medium", pruned)
        self.assertNotIn("tk_pathological", pruned)
        self.assertNotIn("en_common_company", pruned)
        self.assertNotIn("pin_75001", pruned)

    def test_no_hardcoding(self):
        """Verify dynamic behavior across arbitrary keys."""
        idx = {
            "any_arbitrary_key_1": ["a", "b"],
            "any_arbitrary_key_2": ["a", "b", "c", "d", "e", "f"],
        }
        cfg = BlockingKeyFilterConfig(max_key_fanout=3, preserve_exact_name=False, preserve_pin_keys=False)
        pruned, stats = prune_high_frequency_keys(idx, total_s1=20, config=cfg)
        self.assertIn("any_arbitrary_key_1", pruned)
        self.assertNotIn("any_arbitrary_key_2", pruned)
        self.assertEqual(stats["pruned_key_count"], 1)


if __name__ == "__main__":
    unittest.main()
