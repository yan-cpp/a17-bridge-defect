"""Optimized CrackSeg9k conversion: extract nested zips to disk once, then convert."""
from __future__ import annotations

import io
import json
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW = PROJECT_ROOT / "data" / "raw" / "CrackSeg9k.zip"
OUT = PROJECT_ROOT / "data" / "processed" / "crackseg9k"
TMP = PROJECT_ROOT / "data" / "raw" / ".crackseg9k_tmp"
SEED = 42
TRAIN_N = 2000
VAL_N = 250


def contours_to_yolo(mask_binary, width, height):
    contours, _ = cv2.findContours(mask_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        if len(contour) < 3 or cv2.contourArea(contour) < 4:
            continue
        pts = ["0"]
        for x, y in contour[:, 0, :]:
            pts.append(f"{x / width:.6f}")
            pts.append(f"{y / height:.6f}")
        lines.append(" ".join(pts))
    return lines


def extract_inner_zips():
    TMP.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAW) as outer:
        for inner_name in ("Final-Dataset-Vol1.zip", "Final-Dataset-Vol2.zip"):
            target = TMP / inner_name
            if target.exists():
                continue
            with outer.open(inner_name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
    return TMP / "Final-Dataset-Vol1.zip", TMP / "Final-Dataset-Vol2.zip"


def main():
    vol1_path, vol2_path = extract_inner_zips()
    rng = random.Random(SEED)
    counts = {"train": 0, "val": 0, "instances": 0}

    with zipfile.ZipFile(vol1_path) as vol1, zipfile.ZipFile(vol2_path) as vol2:
        mask_entries = [
            n for n in vol1.namelist()
            if n.startswith("Final-Dataset-Vol1/Final_Masks/Masks/") and n.endswith(".png")
        ]
        image_entries = {}
        for archive, prefix in ((vol1, "Final-Dataset-Vol1/Images/"), (vol2, "Final-Dataset-Vol2/Images-2/")):
            for n in archive.namelist():
                if n.startswith(prefix) and n.endswith(".png"):
                    image_entries[n.rsplit("/", 1)[1]] = (archive, n)

        common = sorted({n.rsplit("/", 1)[1] for n in mask_entries} & set(image_entries))
        rng.shuffle(common)
        val_names = set(common[:VAL_N])
        train_names = set(common[VAL_N:VAL_N + TRAIN_N])

        for name in common:
            if name not in train_names and name not in val_names:
                continue
            split = "val" if name in val_names else "train"
            archive, image_entry = image_entries[name]
            image_bytes = archive.read(image_entry)
            mask_entry = f"Final-Dataset-Vol1/Final_Masks/Masks/{name}"
            mask_img = Image.open(io.BytesIO(vol1.read(mask_entry))).convert("L")
            width, height = mask_img.size
            binary = (np.array(mask_img) > 127).astype(np.uint8)
            lines = contours_to_yolo(binary, width, height)
            if not lines:
                continue

            imgs = OUT / "images" / split
            labs = OUT / "labels" / split
            imgs.mkdir(parents=True, exist_ok=True)
            labs.mkdir(parents=True, exist_ok=True)
            (imgs / name).write_bytes(image_bytes)
            (labs / f"{Path(name).stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            counts[split] += 1
            counts["instances"] += len(lines)

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
        json.dumps({"splits": counts, "seed": SEED}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
