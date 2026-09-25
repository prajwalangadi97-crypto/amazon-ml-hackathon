#!/usr/bin/env python3
"""Run all unit tests for the validation and evaluation framework without loading the 2.4 GB dataset."""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(str(PROJECT_ROOT / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
