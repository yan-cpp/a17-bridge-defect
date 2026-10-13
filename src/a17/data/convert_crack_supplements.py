"""Convert supplementary crack datasets into A17 YOLO-seg crack (class 0) labels.

- Concrete Crack Segmentation: Supervisely bitmap masks -> YOLO-seg polygons.
- CrackSeg9k: nested zip image + binary mask -> YOLO-seg polygons.
- SDNET2018: non-cracked images are exported as background samples (empty labels).
"""
from __future__ import annotations

import base64
import zlib
import io
import json
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
SEED = 42

# Output dataset roots
CONCRETE_DIR = PROCESSED / "concrete_crack_seg"
CRACKSEG_DIR = PROCESSED / "crackseg9k"
SDNET_DIR = PROCESSED / "sdnet2018_background"

CRACKSEG_TRAIN = 2000
CRACKSEG_VAL = 250
SDNET_TRAIN = 800
SDNET_VAL = 100


def contours_to_yolo(mask_binary: np.ndarray, width: int, height: int) -> list[str]:
    """Return YOLO-seg lines for each connected component of a binary mask."""
    contours, _ = cv2.findContours(
        mask_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    lines: list[str] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        area = cv2.contourArea(contour)
        if area < 4:
            continue
        flat: list[str] = ["0"]
        for point in contour[:, 0, :]:
            flat.append(f"{point[0] / width:.6f}")
            flat.append(f"{point[1] / height:.6f}")
        lines.append(" ".join(flat))
    return lines


def save_instance(root: Path, split: str, name: str, image_bytes: bytes, lines: list[str]):
    images = root / "images" / split
    labels = root / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    (images / name).write_bytes(image_bytes)
    stem = Path(name).stem
    (labels / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_yaml(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset.yaml").write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/val",
                "names:",
                "  0: crack",
                "",
            ]
        ),
        encoding="utf-8",
    )


def convert_concrete() -> dict[str, Any]:
    rng = random.Random(SEED)
    source = RAW / "Concrete Crack Segmentation.zip"
    counts = Counter()
    with zipfile.ZipFile(source) as archive:
        archive_names = set(archive.namelist())
        image_map = {
            n[4:].split(".", 1)[0].lower(): n
            for n in archive_names
            if n.startswith("img/")
        }
        anns = [n for n in archive_names if n.startswith("ann/") and n.endswith(".json")]
        rng.shuffle(anns)
        split_of = {}
        val_n = 58
        for idx, ann in enumerate(anns):
            split_of[ann] = "val" if idx < val_n else "train"

        for ann in anns:
            data = json.loads(archive.read(ann).decode("utf-8", "replace"))
            size = data.get("size", {})
            width = int(size.get("width", 0))
            height = int(size.get("height", 0))
            stem = Path(ann).stem.split(".", 1)[0]
            image_entry = image_map.get(stem.lower())
            if width <= 0 or height <= 0 or image_entry is None:
                continue

            lines: list[str] = []
            for obj in data.get("objects", []):
                if obj.get("classTitle") != "crack":
                    continue
                bitmap = obj.get("bitmap", {})
                encoded = bitmap.get("data")
                if not encoded:
                    continue
                try:
                    raw_mask = base64.b64decode(encoded)
                    decompressed = zlib.decompress(raw_mask)
                    mask_img = Image.open(io.BytesIO(decompressed)).convert("L")
                except Exception:
                    continue
                mask = np.array(mask_img.resize((width, height), Image.NEAREST))
                binary = (mask > 127).astype(np.uint8)
                lines.extend(contours_to_yolo(binary, width, height))

            if not lines:
                continue

            split = split_of[ann]
            save_instance(CONCRETE_DIR, split, f"{stem}.jpg", archive.read(image_entry), lines)
            counts[split] += 1
            counts["instances"] += len(lines)

    write_yaml(CONCRETE_DIR)
    return {"root": str(CONCRETE_DIR), "splits": dict(counts)}


def convert_crackseg9k() -> dict[str, Any]:
    rng = random.Random(SEED)
    outer_path = RAW / "CrackSeg9k.zip"
    counts = Counter()
    with zipfile.ZipFile(outer_path) as outer:
        with zipfile.ZipFile(outer.open("Final-Dataset-Vol1.zip")) as vol1, \
             zipfile.ZipFile(outer.open("Final-Dataset-Vol2.zip")) as vol2:

            mask_names = {
                n.rsplit("/", 1)[1]
                for n in vol1.namelist()
                if n.startswith("Final-Dataset-Vol1/Final_Masks/Masks/") and n.endswith(".png")
            }

            image_entries: dict[str, str] = {}
            for archive, prefix in ((vol1, "Final-Dataset-Vol1/Images/"), (vol2, "Final-Dataset-Vol2/Images-2/")):
                for n in archive.namelist():
                    if n.startswith(prefix) and n.endswith(".png"):
                        image_entries[n.rsplit("/", 1)[1]] = n

            common = sorted(mask_names & set(image_entries))
            rng.shuffle(common)
            val_names = set(common[:CRACKSEG_VAL])
            train_names = set(common[CRACKSEG_VAL:CRACKSEG_VAL + CRACKSEG_TRAIN])

            for name in common:
                if name not in train_names and name not in val_names:
                    continue
                split = "val" if name in val_names else "train"
                image_entry = image_entries[name]
                if image_entry.startswith("Final-Dataset-Vol1"):
                    image_bytes = vol1.read(image_entry)
                else:
                    image_bytes = vol2.read(image_entry)
                mask_entry = f"Final-Dataset-Vol1/Final_Masks/Masks/{name}"
                mask_img = Image.open(io.BytesIO(vol1.read(mask_entry))).convert("L")
                width, height = mask_img.size
                mask = np.array(mask_img)
                binary = (mask > 127).astype(np.uint8)
                lines = contours_to_yolo(binary, width, height)
                if not lines:
                    continue
                save_instance(CRACKSEG_DIR, split, name, image_bytes, lines)
                counts[split] += 1
                counts["instances"] += len(lines)

    write_yaml(CRACKSEG_DIR)
    return {"root": str(CRACKSEG_DIR), "splits": dict(counts)}


def convert_sdnet_background() -> dict[str, Any]:
    rng = random.Random(SEED)
    source = RAW / "senet2018.zip"
    counts = Counter()
    with zipfile.ZipFile(source) as archive:
        names = [n for n in archive.namelist() if "/Non-cracked/" in n and n.endswith(".jpg")]
        rng.shuffle(names)
        selected = names[: SDNET_TRAIN + SDNET_VAL]
        for idx, entry in enumerate(selected):
            split = "val" if idx < SDNET_VAL else "train"
            name = entry.split("/")[-1]
            images = SDNET_DIR / "images" / split
            labels = SDNET_DIR / "labels" / split
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            (images / name).write_bytes(archive.read(entry))
            # empty label = background image
            (labels / f"{Path(name).stem}.txt").write_text("", encoding="utf-8")
            counts[split] += 1

    write_yaml(SDNET_DIR)
    return {"root": str(SDNET_DIR), "splits": dict(counts)}


def main() -> int:
    summary: dict[str, Any] = {}
    summary["concrete_crack_seg"] = convert_concrete()
    summary["crackseg9k"] = convert_crackseg9k()
    summary["sdnet2018_background"] = convert_sdnet_background()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
