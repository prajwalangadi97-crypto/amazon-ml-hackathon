import io
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ground_truth import parse_ground_truth, stream_ground_truth


class TestGroundTruthParser(unittest.TestCase):
    """Test ground-truth parser on synthetic TSV buffers."""

    def test_parse_valid_content(self):
        tsv_content = (
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-101\tS2-201,S3-301\n"
            "S1-102\t\n"  # singleton with trailing tab
            "S1-103\tS2-202\n"
            "\n"  # blank line
            "S1-104\n"  # singleton without trailing tab
        )
        buffer = io.StringIO(tsv_content)
        result = parse_ground_truth(buffer)

        self.assertEqual(len(result), 4)
        self.assertEqual(result["S1-101"], {"S2-201", "S3-301"})
        self.assertEqual(result["S1-102"], set())  # singleton
        self.assertEqual(result["S1-103"], {"S2-202"})
        self.assertEqual(result["S1-104"], set())  # singleton without tab

    def test_allowed_s1_ids_filter(self):
        tsv_content = (
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-1\tS2-1\n"
            "S1-2\tS2-2\n"
            "S1-3\tS2-3\n"
        )
        buffer = io.StringIO(tsv_content)
        result = parse_ground_truth(buffer, allowed_s1_ids={"S1-2"})
        self.assertEqual(list(result.keys()), ["S1-2"])
        self.assertEqual(result["S1-2"], {"S2-2"})

    def test_invalid_header_raises_value_error(self):
        tsv_content = "wrong_col1\twrong_col2\nS1-1\tS2-1\n"
        buffer = io.StringIO(tsv_content)
        with self.assertRaises(ValueError):
            list(stream_ground_truth(buffer))


if __name__ == "__main__":
    unittest.main()
