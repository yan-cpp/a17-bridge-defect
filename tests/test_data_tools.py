from __future__ import annotations

import os
import shutil
import sys
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a17.data.inspect_gyu import build_data_card, count_labels, read_classes  # noqa: E402


def make_train_zip(folder: Path) -> Path:
    path = folder / "train.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("classes.txt", "Crack\nSpalling\n")
        archive.writestr("labels/img_001.txt", "0 0.5 0.5 0.1 0.2\n1 0.2 0.3 0.05 0.05\n")
        archive.writestr("images/img_001.jpg", b"fake-jpeg-bytes")
    return path


class DataToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = ROOT / (".test-data-" + os.urandom(4).hex())
        self.temp.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp, ignore_errors=True)

    def test_inspect_gyu_det_finds_archive_and_counts_labels(self) -> None:
        zip_path = make_train_zip(self.temp)
        classes = read_classes(zip_path)
        self.assertEqual(classes, ["Crack", "Spalling"])
        stats = count_labels(zip_path, classes)
        self.assertEqual(stats["label_files"], 1)
        self.assertEqual(stats["instances"], 2)
        card = build_data_card([str(self.temp)])
        self.assertEqual(card["classes"], ["Crack", "Spalling"])
        self.assertIn("train", card["splits"])


if __name__ == "__main__":
    unittest.main()
