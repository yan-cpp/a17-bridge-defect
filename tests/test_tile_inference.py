from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a17.eval.tile_inference import predict_tiled, stitch_mask_records, tile_crops  # noqa: E402


class _Array:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Boxes:
    def __init__(self):
        self.xyxy = _Array([[30, 30, 50, 50]])
        self.cls = _Array([0])
        self.conf = _Array([0.9])


class _Masks:
    def __init__(self):
        mask = np.zeros((80, 80), dtype=np.float32)
        mask[30:50, 30:50] = 1.0
        self.data = _Array([mask])


class _Result:
    boxes = _Boxes()
    masks = _Masks()


class _DummyModel:
    names = {0: "crack"}

    def predict(self, **kwargs):
        return [_Result()]


def record(mask: np.ndarray, confidence: float = 0.8) -> dict:
    ys, xs = np.nonzero(mask)
    bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    return {
        "class_key": "crack",
        "confidence": confidence,
        "bbox_xyxy": bbox,
        "mask": mask.astype(np.uint8),
    }


class TileInferenceTests(unittest.TestCase):
    def test_axis_starts_are_unique_at_right_edge(self) -> None:
        image = np.zeros((2048, 2048, 3), dtype=np.uint8)
        crops = list(tile_crops(image, tile=1024, overlap=128))
        self.assertEqual(len(crops), 9)
        starts = [(x, y) for x, y, _ in crops]
        self.assertEqual(starts[0], (0, 0))
        self.assertEqual(starts[-1], (1024, 1024))
        self.assertEqual(len(starts), len(set(starts)))

    def test_overlapping_fragments_are_stitched(self) -> None:
        first = np.zeros((40, 40), dtype=np.uint8)
        second = np.zeros((40, 40), dtype=np.uint8)
        first[10:14, 8:24] = 1
        second[10:14, 20:36] = 1
        merged = stitch_mask_records([record(first, 0.9), record(second, 0.7)])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["source_count"], 2)
        self.assertEqual(merged[0]["mask_area_px"], 28 * 4)
        self.assertAlmostEqual(merged[0]["confidence"], 0.9)

    def test_separated_fragments_remain_separate(self) -> None:
        first = np.zeros((40, 40), dtype=np.uint8)
        second = np.zeros((40, 40), dtype=np.uint8)
        first[4:8, 4:12] = 1
        second[30:34, 28:36] = 1
        merged = stitch_mask_records([record(first), record(second)])
        self.assertEqual(len(merged), 2)

    def test_nearby_collinear_crack_fragments_are_bridged(self) -> None:
        first = np.zeros((40, 40), dtype=np.uint8)
        second = np.zeros((40, 40), dtype=np.uint8)
        first[18:21, 4:18] = 1
        second[18:21, 22:36] = 1
        merged = stitch_mask_records(
            [record(first, 0.7), record(second, 0.4)],
            bridge_px=8,
            bridge_min_confidence=0.2,
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["source_count"], 2)
        self.assertEqual(merged[0]["bridge_count"], 1)
        self.assertGreater(merged[0]["mask"][19, 19], 0)

    def test_nearby_low_confidence_fragment_is_not_bridged(self) -> None:
        first = np.zeros((40, 40), dtype=np.uint8)
        second = np.zeros((40, 40), dtype=np.uint8)
        first[18:21, 4:18] = 1
        second[18:21, 22:36] = 1
        merged = stitch_mask_records(
            [record(first, 0.7), record(second, 0.1)],
            bridge_px=8,
            bridge_min_confidence=0.2,
        )
        self.assertEqual(len(merged), 2)

    def test_nearby_perpendicular_cracks_are_not_bridged(self) -> None:
        first = np.zeros((40, 40), dtype=np.uint8)
        second = np.zeros((40, 40), dtype=np.uint8)
        first[18:21, 4:18] = 1
        second[22:36, 19:22] = 1
        merged = stitch_mask_records(
            [record(first, 0.7), record(second, 0.6)],
            bridge_px=8,
            bridge_min_confidence=0.2,
        )
        self.assertEqual(len(merged), 2)

    def test_touching_crack_fragments_with_small_overlap_are_merged(self) -> None:
        first = np.zeros((40, 40), dtype=np.uint8)
        second = np.zeros((40, 40), dtype=np.uint8)
        first[18:21, 4:20] = 1
        second[18:21, 20:36] = 1
        merged = stitch_mask_records(
            [record(first, 0.7), record(second, 0.4)],
            bridge_px=8,
            bridge_min_confidence=0.2,
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["source_count"], 2)

    def test_aligned_components_inside_one_prediction_are_bridged(self) -> None:
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[18:23, 4:24] = 1
        mask[18:23, 26:46] = 1
        merged = stitch_mask_records(
            [record(mask, 0.7)],
            bridge_px=8,
            bridge_min_confidence=0.2,
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["bridge_count"], 1)
        self.assertGreater(int(merged[0]["mask"][18:23, 24:26].sum()), 0)

    def test_low_overlap_perpendicular_cracks_are_not_merged(self) -> None:
        first = np.zeros((40, 40), dtype=np.uint8)
        second = np.zeros((40, 40), dtype=np.uint8)
        first[18:21, 4:22] = 1
        second[18:36, 19:22] = 1
        merged = stitch_mask_records(
            [record(first, 0.7), record(second, 0.6)],
            bridge_px=8,
            bridge_min_confidence=0.2,
        )
        self.assertEqual(len(merged), 2)

    def test_spalling_fragments_are_not_merged_by_proximity(self) -> None:
        first = np.zeros((40, 40), dtype=np.uint8)
        second = np.zeros((40, 40), dtype=np.uint8)
        first[18:21, 4:18] = 1
        second[18:21, 22:36] = 1
        first_record = record(first, 0.7)
        second_record = record(second, 0.6)
        first_record["class_key"] = "spalling"
        second_record["class_key"] = "spalling"
        merged = stitch_mask_records(
            [first_record, second_record],
            bridge_px=8,
            bridge_min_confidence=0.2,
        )
        self.assertEqual(len(merged), 2)

    def test_tiled_predictions_are_mapped_and_merged(self) -> None:
        image = np.zeros((96, 96, 3), dtype=np.uint8)
        records = predict_tiled(
            _DummyModel(),
            image,
            tile=80,
            overlap=16,
            conf=0.1,
            scale=1.0,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_count"], 4)
        self.assertEqual(records[0]["bbox_xyxy"], [30.0, 30.0, 66.0, 66.0])
        self.assertGreater(records[0]["length_mm"], 0)


if __name__ == "__main__":
    unittest.main()
