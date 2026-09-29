from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a17.schemas import DefectInstance, ImageResult, SchemaError, Timing  # noqa: E402


def sample_defect(instance_id: int = 1) -> DefectInstance:
    return DefectInstance(
        instance_id=instance_id,
        class_key="crack",
        class_name_zh="混凝土裂缝",
        confidence=0.93,
        bbox_xyxy=(120.0, 340.0, 580.0, 410.0),
        mask_path="masks/sample_001.png",
        length_mm=182.4,
        width_mm=1.7,
        width_statistic="p95",
        scale_mm_per_pixel=0.42,
        quantification_valid=True,
    )


class SchemaTests(unittest.TestCase):
    def test_valid_result(self) -> None:
        result = ImageResult(
            image_id="sample.jpg",
            image_width=3840,
            image_height=2160,
            model_version="baseline-v0",
            config_version="0.1.0",
            timing_ms=Timing(total_ms=180.0),
            defects=[sample_defect()],
        )
        result.validate({"crack", "spalling", "exposed_rebar"})
        self.assertEqual(result.to_dict()["defects"][0]["class_key"], "crack")

    def test_duplicate_instance_ids_fail(self) -> None:
        result = ImageResult(
            image_id="sample.jpg",
            image_width=3840,
            image_height=2160,
            model_version="baseline-v0",
            config_version="0.1.0",
            timing_ms=Timing(total_ms=180.0),
            defects=[sample_defect(1), sample_defect(1)],
        )
        with self.assertRaisesRegex(SchemaError, "unique"):
            result.validate({"crack", "spalling", "exposed_rebar"})


if __name__ == "__main__":
    unittest.main()
