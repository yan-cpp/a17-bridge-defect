from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a17.calibrate import mm_per_pixel_from_distance  # noqa: E402


class CalibrationTests(unittest.TestCase):
    def test_reference_distance(self) -> None:
        self.assertAlmostEqual(mm_per_pixel_from_distance(1000, 420), 0.42)

    def test_invalid_distance(self) -> None:
        with self.assertRaises(ValueError):
            mm_per_pixel_from_distance(0, 10)


if __name__ == "__main__":
    unittest.main()
