"""Convert dacl10k labelme-style polygons to A17 YOLO-seg labels.

Only phase-1 classes (crack, spalling, exposed_rebar) are kept. Images without
any phase-1 instance are skipped. Images and labels are written side by side
under data/processed/dacl10k.
"""
from __future__ import annotations

import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ZIP = PROJECT_ROOT / "data" / "raw" / "dacl10k_v2_devphase.zip"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "dacl10k"
MAPPING_FILE = PROJECT_ROOT / "configs" / "data_mapping.json"

SPLIT_MAP = {"train": "train", "validation": "val", "testdev": "testdev"}

# dacl10k -> A17 phase-1 mapping; ignored phase-2 classes are dropped for now.
DACL_PHASE1 = {
    "Crack": "crack",
    "ACrack": "crack",
    "Spalling": "spalling",
    "ExposedRebars": "exposed_rebar",
}


def load_mapping() -> dict[str, Any]:
    with MAPPING_FILE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_polygon(points: list[list[float]], width: float, height: float) -> list[float]:
    """Return a flat YOLO-seg polygon string token list in xy order."""
    normalized: list[float] = []
    for x, y in points:
        normalized.append(round(x / width, 6))
        normalized.append(round(y / height, 6))
    return normalized


def main() -> int:
    mapping = load_mapping()
    dacl_mapping = mapping["source_mappings"]["dacl10k"]
    id_map = {key: index for index, key in enumerate(mapping["phase1_classes"])}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images_out = OUTPUT_DIR / "images"
    labels_out = OUTPUT_DIR / "labels"
    images_out.mkdir(exist_ok=True)
    labels_out.mkdir(exist_ok=True)

    stats: dict[str, dict[str, Any]] = {}
    class_counter = Counter()

    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        names = archive.namelist()
        annotation_names = [
            n
            for n in names
            if "/annotations/" in n and n.lower().endswith(".json")
        ]
        image_names = {
            n.split("/")[-1]: n
            for n in names
            if "/images/" in n and n.lower().endswith((".jpg", ".jpeg", ".png"))
        }

        for annotation_name in annotation_names:
            raw = archive.read(annotation_name).decode("utf-8", "replace")
            data = json.loads(raw)
            split = SPLIT_MAP.get(data.get("split"), "val")
            if split not in ("train", "val"):
                continue

            width = float(data.get("imageWidth", 1))
            height = float(data.get("imageHeight", 1))
            image_name = data.get("imageName")
            image_path_in_zip = image_names.get(image_name)
            if not image_name or not image_path_in_zip:
                continue

            kept_lines: list[str] = []
            for shape in data.get("shapes", []):
                label = shape.get("label")
                mapped = dacl_mapping.get(label)
                if mapped not in id_map:
                    continue
                points = shape.get("points", [])
                if len(points) < 3:
                    continue
                polygon = normalize_polygon(points, width, height)
                if not polygon:
                    continue
                class_id = id_map[mapped]
                kept_lines.append(" ".join([str(class_id), *map(str, polygon)]))
                class_counter[mapped] += 1

            if not kept_lines:
                continue

            split_images = images_out / split
            split_labels = labels_out / split
            split_images.mkdir(parents=True, exist_ok=True)
            split_labels.mkdir(parents=True, exist_ok=True)

            archive.extract(image_path_in_zip, OUTPUT_DIR)
            target_image = OUTPUT_DIR / image_path_in_zip
            desired_image = split_images / image_name
            if target_image.resolve() != desired_image.resolve():
                target_image.replace(desired_image)

            label_name = Path(image_name).stem + ".txt"
            (split_labels / label_name).write_text(
                "\n".join(kept_lines) + "\n", encoding="utf-8"
            )

            bucket = stats.setdefault(
                split,
                {"images": 0, "instances": 0, "labels": 0},
            )
            bucket["images"] += 1
            bucket["labels"] += 1
            bucket["instances"] += len(kept_lines)

    dataset_yaml = OUTPUT_DIR / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {OUTPUT_DIR.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/val",
                "names:",
                "  0: crack",
                "  1: spalling",
                "  2: exposed_rebar",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary = {
        "source": str(SOURCE_ZIP),
        "output": str(OUTPUT_DIR),
        "classes": mapping["phase1_classes"],
        "splits": stats,
        "class_distribution": dict(class_counter),
        "dropped_classes_note": "dacl10k phase-2 and object classes were dropped; only phase-1 classes were kept.",
    }
    (OUTPUT_DIR / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
