import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    compute_entity_metrics,
    evaluate_predictions,
)


class TestEntityMetrics(unittest.TestCase):
    """Test entity-level precision, recall, and F0.5 against required cases."""

    def test_case_a_perfect_singleton(self):
        """Case A: Perfect singleton (true empty, predicted empty)."""
        res = compute_entity_metrics(predicted_ids=set(), true_ids=set(), source1_id="S1-1")
        self.assertTrue(res.is_singleton)
        self.assertEqual(res.precision, 1.0)
        self.assertEqual(res.recall, 1.0)
        self.assertEqual(res.f0_5, 1.0)
        self.assertEqual(res.tp, 0)
        self.assertEqual(res.fp, 0)
        self.assertEqual(res.fn, 0)

    def test_case_b_false_positive_on_singleton(self):
        """Case B: False positive on singleton (true empty, predicted has items)."""
        res = compute_entity_metrics(predicted_ids={"S2-100"}, true_ids=set(), source1_id="S1-2")
        self.assertTrue(res.is_singleton)
        self.assertEqual(res.precision, 0.0)
        self.assertEqual(res.recall, 0.0)
        self.assertEqual(res.f0_5, 0.0)
        self.assertEqual(res.tp, 0)
        self.assertEqual(res.fp, 1)
        self.assertEqual(res.fn, 0)

    def test_case_c_perfect_one_match(self):
        """Case C: Perfect one-match (true {S2-1}, predicted {S2-1})."""
        res = compute_entity_metrics(predicted_ids={"S2-1"}, true_ids={"S2-1"}, source1_id="S1-3")
        self.assertFalse(res.is_singleton)
        self.assertEqual(res.precision, 1.0)
        self.assertEqual(res.recall, 1.0)
        self.assertEqual(res.f0_5, 1.0)
        self.assertEqual(res.tp, 1)
        self.assertEqual(res.fp, 0)
        self.assertEqual(res.fn, 0)

    def test_case_d_missed_match(self):
        """Case D: Missed match (true {S2-1}, predicted empty)."""
        res = compute_entity_metrics(predicted_ids=set(), true_ids={"S2-1"}, source1_id="S1-4")
        self.assertFalse(res.is_singleton)
        self.assertEqual(res.precision, 0.0)
        self.assertEqual(res.recall, 0.0)
        self.assertEqual(res.f0_5, 0.0)
        self.assertEqual(res.tp, 0)
        self.assertEqual(res.fp, 0)
        self.assertEqual(res.fn, 1)

    def test_case_e_false_extra_match(self):
        """Case E: False extra match (true {S2-1}, predicted {S2-1, S3-2})."""
        res = compute_entity_metrics(
            predicted_ids={"S2-1", "S3-2"}, true_ids={"S2-1"}, source1_id="S1-5"
        )
        self.assertFalse(res.is_singleton)
        self.assertEqual(res.tp, 1)
        self.assertEqual(res.fp, 1)
        self.assertEqual(res.fn, 0)
        self.assertEqual(res.precision, 0.5)
        self.assertEqual(res.recall, 1.0)
        # F0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 5/9 ≈ 0.555556
        expected_f0_5 = 5.0 / 9.0
        self.assertAlmostEqual(res.f0_5, expected_f0_5, places=6)

    def test_case_f_multiple_correct_matches(self):
        """Case F: Multiple correct matches (true {S2-1, S3-1}, predicted {S2-1, S3-1})."""
        res = compute_entity_metrics(
            predicted_ids={"S2-1", "S3-1"}, true_ids={"S2-1", "S3-1"}, source1_id="S1-6"
        )
        self.assertFalse(res.is_singleton)
        self.assertEqual(res.precision, 1.0)
        self.assertEqual(res.recall, 1.0)
        self.assertEqual(res.f0_5, 1.0)

    def test_case_g_mixed_s2_and_s3_matches(self):
        """Case G: Mixed S2 and S3 matches with one false positive."""
        true_set = {"S2-10", "S3-20"}
        pred_set = {"S2-10", "S2-99", "S3-20"}
        res = compute_entity_metrics(predicted_ids=pred_set, true_ids=true_set, source1_id="S1-7")
        self.assertEqual(res.tp, 2)
        self.assertEqual(res.fp, 1)
        self.assertEqual(res.fn, 0)
        self.assertAlmostEqual(res.precision, 2.0 / 3.0, places=6)
        self.assertEqual(res.recall, 1.0)
        # F0.5 = (1.25 * (2/3) * 1.0) / (0.25 * (2/3) + 1.0) = 5/7 ≈ 0.714286
        expected_f0_5 = 5.0 / 7.0
        self.assertAlmostEqual(res.f0_5, expected_f0_5, places=6)

    def test_case_h_empty_prediction_for_non_singleton(self):
        """Case H: Empty prediction for a non-singleton entity."""
        res = compute_entity_metrics(
            predicted_ids=set(), true_ids={"S2-1", "S3-1"}, source1_id="S1-8"
        )
        self.assertFalse(res.is_singleton)
        self.assertEqual(res.precision, 0.0)
        self.assertEqual(res.recall, 0.0)
        self.assertEqual(res.f0_5, 0.0)

    def test_case_i_multiple_false_positives(self):
        """Case I: Multiple false positives with zero true positives."""
        res = compute_entity_metrics(
            predicted_ids={"S2-99", "S3-99"}, true_ids={"S2-1"}, source1_id="S1-9"
        )
        self.assertEqual(res.tp, 0)
        self.assertEqual(res.fp, 2)
        self.assertEqual(res.fn, 1)
        self.assertEqual(res.precision, 0.0)
        self.assertEqual(res.recall, 0.0)
        self.assertEqual(res.f0_5, 0.0)

    def test_case_j_readme_numerical_example(self):
        """Case J: Exact numerical example from README.md line 204-210.
        Model predicts: S1-00001 matches [S2-00047, S2-00193, S3-00812]
        Ground truth  : S1-00001 matches [S2-00047, S3-00812]
        Precision = 2/3, Recall = 2/2 = 1.0
        F_0.5 = (1.25 * 0.667 * 1.0) / (0.25 * 0.667 + 1.0) = 0.714 (approx 5/7)
        """
        pred = {"S2-00047", "S2-00193", "S3-00812"}
        truth = {"S2-00047", "S3-00812"}
        res = compute_entity_metrics(predicted_ids=pred, true_ids=truth, source1_id="S1-00001")
        self.assertEqual(res.tp, 2)
        self.assertEqual(res.fp, 1)
        self.assertEqual(res.fn, 0)
        self.assertAlmostEqual(res.precision, 2.0 / 3.0, places=5)
        self.assertEqual(res.recall, 1.0)
        self.assertAlmostEqual(res.f0_5, 5.0 / 7.0, places=5)
        self.assertEqual(f"{res.f0_5:.3f}", "0.714")


class TestMacroEvaluator(unittest.TestCase):
    """Test macro-averaged evaluation over collections of entities."""

    def test_macro_averaging_mixed_entities(self):
        """Test macro-average over 4 entities:
        E1 (singleton): true empty, pred empty => F0.5 = 1.0
        E2 (one-match): true {S2-1}, pred {S2-1} => F0.5 = 1.0
        E3 (README ex): true {S2-A, S3-B}, pred {S2-A, S2-X, S3-B} => F0.5 = 5/7 ≈ 0.714286
        E4 (missed)   : true {S2-2}, pred empty => F0.5 = 0.0

        Macro F0.5 = (1.0 + 1.0 + 5/7 + 0.0) / 4 = (19/7) / 4 = 19/28 ≈ 0.678571
        """
        ground_truth = {
            "S1-E1": set(),
            "S1-E2": {"S2-1"},
            "S1-E3": {"S2-A", "S3-B"},
            "S1-E4": {"S2-2"},
        }
        predictions = {
            "S1-E1": set(),
            "S1-E2": {"S2-1"},
            "S1-E3": {"S2-A", "S2-X", "S3-B"},
            "S1-E4": set(),
        }
        report = evaluate_predictions(predictions, ground_truth)
        self.assertEqual(report.total_entities, 4)
        expected_macro_f0_5 = (1.0 + 1.0 + (5.0 / 7.0) + 0.0) / 4.0
        self.assertAlmostEqual(report.macro_f0_5, expected_macro_f0_5, places=5)
        self.assertEqual(report.singleton_count, 1)
        self.assertEqual(report.singleton_f0_5, 1.0)
        self.assertEqual(report.non_singleton_count, 3)

    def test_missing_prediction_handled_as_empty(self):
        """Entities omitted from predictions dict should be treated as empty prediction."""
        ground_truth = {
            "S1-SINGLETON": set(),
            "S1-HAS_MATCH": {"S2-1"},
        }
        predictions = {}  # completely empty prediction dict
        report = evaluate_predictions(predictions, ground_truth, allow_missing=True)
        self.assertEqual(report.missing_predictions, 2)
        # S1-SINGLETON gets 1.0 (empty predicted matches empty true)
        # S1-HAS_MATCH gets 0.0 (empty predicted missed true match)
        # Macro F0.5 = (1.0 + 0.0) / 2 = 0.5
        self.assertEqual(report.macro_f0_5, 0.5)


if __name__ == "__main__":
    unittest.main()
