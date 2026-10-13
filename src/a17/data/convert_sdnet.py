"""Export SDNET2018 non-cracked images as segmentation background negatives."""
from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW = PROJECT_ROOT / "data" / "raw" / "senet2018.zip"
OUT = PROJECT_ROOT / "data" / "processed" / "sdnet2018_background"
SEED = 42
TRAIN_N = 800
VAL_N = 100


def main():
    rng = random.Random(SEED)
    counts = {"train": 0, "val": 0}
    with zipfile.ZipFile(RAW) as archive:
        names = [n for n in archive.namelist() if "/Non-cracked/" in n and n.endswith(".jpg")]
        rng.shuffle(names)
        for idx, entry in enumerate(names[: TRAIN_N + VAL_N]):
            split = "val" if idx < VAL_N else "train"
            name = entry.split("/")[-1]
            imgs = OUT / "images" / split
            labs = OUT / "labels" / split
            imgs.mkdir(parents=True, exist_ok=True)
            labs.mkdir(parents=True, exist_ok=True)
            (imgs / name).write_bytes(archive.read(entry))
            (labs / f"{Path(name).stem}.txt").write_text("", encoding="utf-8")
            counts[split] += 1

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dataset.yaml").write_text(
        "\n".join([
            f"path: {OUT.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/val",
            "names:",
            "  0: crack",
            "",
        ]),
        encoding="utf-8",
    )
    (OUT / "conversion_summary.json").write_text(
        json.dumps({"splits": counts, "note": "Non-cracked images only, empty labels as background.", "seed": SEED}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
