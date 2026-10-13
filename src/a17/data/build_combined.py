"""Merge full dacl10k with crack supplement datasets into one YOLO-seg dataset."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT = PROCESSED / "a17_combined"

SOURCES = [
    PROCESSED / "dacl10k",
    PROCESSED / "concrete_crack_seg",
    PROCESSED / "crackseg9k",
    PROCESSED / "sdnet2018_background",
]


def copy_split(split: str) -> dict:
    images_out = OUT / "images" / split
    labels_out = OUT / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    total_images = 0
    total_instances = 0
    per_source: dict[str, int] = {}
    for source in SOURCES:
        src_images = source / "images" / split
        src_labels = source / "labels" / split
        if not src_images.exists():
            continue
        count = 0
        for image in src_images.iterdir():
            if not image.is_file():
                continue
            label = src_labels / f"{image.stem}.txt"
            shutil.copy2(image, images_out / image.name)
            if label.exists():
                shutil.copy2(label, labels_out / label.name)
                total_instances += sum(1 for _ in label.read_text(encoding="utf-8").splitlines() if _.strip())
            else:
                (labels_out / label.name).write_text("", encoding="utf-8")
            count += 1
        per_source[source.name] = count
        total_images += count
    return {"images": total_images, "instances": total_instances, "per_source": per_source}


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    summary: dict[str, dict] = {}
    for split in ("train", "val"):
        summary[split] = copy_split(split)

    (OUT / "dataset.yaml").write_text(
        "\n".join([
            f"path: {OUT.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/val",
            "names:",
            "  0: crack",
            "  1: spalling",
            "  2: exposed_rebar",
            "",
        ]),
        encoding="utf-8",
    )
    (OUT / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
