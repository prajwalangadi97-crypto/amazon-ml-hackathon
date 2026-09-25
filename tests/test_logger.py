import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.logger import ExperimentLogger, ExperimentRecord


class TestExperimentLogger(unittest.TestCase):
    """Test experiment logging and retrieval."""

    def test_log_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(experiments_dir=tmpdir)
            record = ExperimentRecord(
                experiment_id="EXP-000-unit-test",
                hypothesis="Unit test experiment logging",
                data_split_description="Synthetic 10 entities",
                blocking_method="exact_match",
                features=["name_jaccard"],
                model="Rule-based",
                hyperparameters={"threshold": 0.8},
                validation_method="macro_f0.5",
                macro_F0_5=0.875,
                precision=0.900,
                recall=0.850,
                training_time_seconds=1.2,
                inference_time_seconds=0.4,
                notes="Initial unit test run",
                conclusion="Logging works as expected",
                next_experiment="EXP-001",
            )
            saved_path = logger.log(record)
            self.assertTrue(saved_path.is_file())

            retrieved = logger.get("EXP-000-unit-test")
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.experiment_id, "EXP-000-unit-test")
            self.assertEqual(retrieved.macro_F0_5, 0.875)
            self.assertEqual(retrieved.precision, 0.900)
            self.assertEqual(retrieved.recall, 0.850)

            experiments = logger.list_experiments()
            self.assertEqual(len(experiments), 1)
            self.assertEqual(experiments[0]["experiment_id"], "EXP-000-unit-test")

            summary = logger.summary_table()
            self.assertIn("EXP-000-unit-test", summary)
            self.assertIn("0.8750", summary)


if __name__ == "__main__":
    unittest.main()
