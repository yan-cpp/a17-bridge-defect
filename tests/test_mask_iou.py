from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a17.eval.mask_iou import mask_iou  # noqa: E402


class MaskIouTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = ROOT / (".test-mask-" + os.urandom(4).hex())
        self.temp.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp, ignore_errors=True)

    def test_identical_masks_have_iou_one(self) -> None:
        from PIL import Image
        import numpy as np
        pred = self.temp / "pred.png"
        target = self.temp / "target.png"
        Image.fromarray((np.ones((8, 8), dtype=np.uint8) * 255)).save(pred)
        Image.fromarray((np.ones((8, 8), dtype=np.uint8) * 255)).save(target)
        self.assertAlmostEqual(mask_iou(pred, target), 1.0)

    def test_disjoint_masks_have_iou_zero(self) -> None:
        from PIL import Image
        import numpy as np
        pred = self.temp / "pred.png"
        target = self.temp / "target.png"
        first = np.zeros((8, 8), dtype=np.uint8)
        first[:, :4] = 255
        second = np.zeros((8, 8), dtype=np.uint8)
        second[:, 4:] = 255
        Image.fromarray(first).save(pred)
        Image.fromarray(second).save(target)
        self.assertAlmostEqual(mask_iou(pred, target), 0.0)


if __name__ == "__main__":
    unittest.main()
