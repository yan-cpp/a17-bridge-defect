import unittest

from a17.eval.benchmark_matrix import compare_records


def _record(class_key: str, confidence: float, bbox: list[float]) -> dict:
    return {
        "class_key": class_key,
        "confidence": confidence,
        "bbox_xyxy": bbox,
    }


class BenchmarkMatrixTests(unittest.TestCase):
    def test_compare_records_matches_identical_outputs(self) -> None:
        reference = [
            _record("crack", 0.8, [0, 0, 10, 10]),
            _record("spalling", 0.7, [20, 20, 30, 30]),
        ]

        result = compare_records(reference, reference)

        self.assertEqual(result["f1_at_iou"], 1.0)
        self.assertEqual(result["mean_matched_iou"], 1.0)
        self.assertEqual(result["max_abs_confidence_delta"], 0.0)

    def test_compare_records_requires_matching_class(self) -> None:
        reference = [_record("crack", 0.8, [0, 0, 10, 10])]
        candidate = [_record("spalling", 0.8, [0, 0, 10, 10])]

        result = compare_records(reference, candidate)

        self.assertEqual(result["matched"], 0)
        self.assertEqual(result["f1_at_iou"], 0.0)
        self.assertIsNone(result["mean_matched_iou"])

    def test_compare_records_reports_confidence_delta(self) -> None:
        reference = [_record("crack", 0.8, [0, 0, 10, 10])]
        candidate = [_record("crack", 0.75, [1, 1, 10, 10])]

        result = compare_records(reference, candidate)

        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["max_abs_confidence_delta"], 0.05)
        self.assertLess(result["mean_matched_iou"], 1.0)
