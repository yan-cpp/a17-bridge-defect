from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a17.quantification_v1 import (  # noqa: E402
    QuantificationConfigV1,
    clean_mask,
    quantify_instance,
)


class QuantificationV1Tests(unittest.TestCase):
    def test_small_components_are_removed(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:10, 5:10] = 1
        mask[15, 15] = 1
        cleaned = clean_mask(mask, QuantificationConfigV1(mm_per_pixel=1.0, min_area_px=5))
        self.assertEqual(int(cleaned.sum()), 25)

    def test_line_quantification_and_topology(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[10, 2:11] = 1
        _, skeleton, result = quantify_instance(
            mask,
            "crack",
            1,
            QuantificationConfigV1(mm_per_pixel=0.5, min_area_px=3),
        )
        self.assertEqual(int(skeleton.sum()), 9)
        self.assertEqual(result.endpoints, 2)
        self.assertAlmostEqual(result.length_px, 8.0)
        self.assertAlmostEqual(result.length_mm, 4.0)


if __name__ == "__main__":
    unittest.main()
