from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a17.quantify import quantify_mask  # noqa: E402


class QuantifyTests(unittest.TestCase):
    def test_horizontal_line_length_is_geodesic(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[10, 2:11] = 1
        result = quantify_mask(mask, mm_per_pixel=1.0)
        self.assertEqual(result.skeleton_px, 9)
        self.assertAlmostEqual(result.length_px, 8.0)
        self.assertAlmostEqual(result.length_mm, 8.0)

    def test_diagonal_step_uses_sqrt_two(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        for index in range(5):
            mask[index + 2, index + 2] = 1
        result = quantify_mask(mask, mm_per_pixel=1.0)
        self.assertEqual(result.skeleton_px, 5)
        self.assertAlmostEqual(result.length_px, 4.0 * np.sqrt(2.0))


if __name__ == "__main__":
    unittest.main()
