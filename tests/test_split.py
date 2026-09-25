import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import create_validation_split, ValidationSplit


class TestValidationSplit(unittest.TestCase):
    """Test validation splitting strategy on synthetic entities."""

    def setUp(self):
        # 10 synthetic entities: 2 singletons, 8 non-singletons, mix of US and India
        self.synthetic_gt = {
            f"S1-{i}": ({f"S2-{i}", f"S3-{i}"} if i > 2 else set()) for i in range(1, 11)
        }
        self.synthetic_metadata = {
            f"S1-{i}": {"country": "India" if i % 2 == 0 else "US"} for i in range(1, 11)
        }

    def test_reproducibility_with_seed(self):
        split1 = create_validation_split(
            self.synthetic_gt,
            s1_metadata=self.synthetic_metadata,
            val_ratio=0.3,
            seed=42,
        )
        split2 = create_validation_split(
            self.synthetic_gt,
            s1_metadata=self.synthetic_metadata,
            val_ratio=0.3,
            seed=42,
        )
        self.assertEqual(split1.val_s1_ids, split2.val_s1_ids)
        self.assertEqual(split1.train_s1_ids, split2.train_s1_ids)

    def test_disjointness_and_completeness(self):
        split = create_validation_split(
            self.synthetic_gt,
            s1_metadata=self.synthetic_metadata,
            val_ratio=0.2,
            seed=123,
        )
        # Disjoint
        self.assertEqual(split.train_s1_ids & split.val_s1_ids, set())
        # Complete
        all_s1 = set(self.synthetic_gt.keys())
        self.assertEqual(split.train_s1_ids | split.val_s1_ids, all_s1)

    def test_preserves_candidate_targets(self):
        split = create_validation_split(
            self.synthetic_gt,
            val_ratio=0.3,
            seed=999,
        )
        # Check that val_true_matches contains the exact true targets for every val S1
        for val_s1 in split.val_s1_ids:
            self.assertIn(val_s1, split.val_true_matches)
            self.assertEqual(split.val_true_matches[val_s1], self.synthetic_gt[val_s1])

    def test_save_and_load_round_trip(self):
        split = create_validation_split(
            self.synthetic_gt,
            val_ratio=0.3,
            seed=42,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            split_file = Path(tmpdir) / "split.json"
            split.save(split_file)
            self.assertTrue(split_file.is_file())

            loaded = ValidationSplit.load(split_file)
            self.assertEqual(loaded.seed, 42)
            self.assertEqual(loaded.val_ratio, 0.3)
            self.assertEqual(loaded.val_s1_ids, split.val_s1_ids)
            self.assertEqual(loaded.train_s1_ids, split.train_s1_ids)


if __name__ == "__main__":
    unittest.main()
