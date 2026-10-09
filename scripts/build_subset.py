from __future__ import annotations

import random
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(r"D:\zm\newA\a17-bridge-defect")
INTERIM = PROJECT_ROOT / "data" / "interim" / "gyu_det"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "gyu_det_subset"

SOURCE_CLASSES = ["Crack", "Breakage", "Comb", "Hole", "Reinforcement", "Seepage"]
INTERNAL_IDS = {"crack": 0, "spalling": 1, "exposed_rebar": 2}
ACTIVE_SOURCE = {"Crack": "crack", "Breakage": "spalling", "Reinforcement": "exposed_rebar"}

SPLITS = [
    ("train", "train.zip", 240),
    ("val", "valid.zip", 60),
    ("test", "test.zip", 60),
]


def remap_label(raw: str):
    lines = []
    for line in raw.splitlines():
        tokens = line.strip().split()
        if not tokens:
            continue
        source_id = int(float(tokens[0]))
        source_class = SOURCE_CLASSES[source_id] if 0 <= source_id < len(SOURCE_CLASSES) else None
        internal_class = ACTIVE_SOURCE.get(source_class)
        if internal_class is None:
            continue
        lines.append(f"{INTERNAL_IDS[internal_class]} " + " ".join(tokens[1:5]))
    return lines


def build_split(zip_path: Path, target_count: int, split: str):
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        image_names = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        image_by_stem = {Path(n).stem.lower(): n for n in image_names}
        label_names = [
            n for n in names
            if n.lower().endswith(".txt") and "classes.txt" not in n.lower()
        ]

        candidates = []
        for label_name in label_names:
            raw = archive.read(label_name).decode("utf-8", errors="replace")
            lines = remap_label(raw)
            if not lines:
                continue
            stem = Path(label_name).stem.lower()
            image_name = image_by_stem.get(stem)
            if image_name is None:
                continue
            classes = sorted({int(line.split()[0]) for line in lines})
            candidates.append((label_name, image_name, lines, classes))

        rng = random.Random(42 + SPLITS.index((split, zip_path.name, target_count)))
        selected = []
        if target_count <= 0:
            return selected
        # Simple balanced selection across the three internal classes.
        for class_id in sorted(INTERNAL_IDS.values()):
            pool = [item for item in candidates if class_id in item[3]]
            rng.shuffle(pool)
            selected.extend(pool[: max(1, target_count // 3)])
        selected = selected[:target_count]

        images_dir = OUTPUT / "images" / split
        labels_dir = OUTPUT / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        stats = {"split": split, "images": 0, "instances": 0}
        seen_stems = set()
        for label_name, image_name, lines, _classes in selected:
            stem = Path(image_name).stem
            if stem in seen_stems:
                continue
            seen_stems.add(stem)
            with archive.open(image_name) as source, open(images_dir / Path(image_name).name, "wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            (labels_dir / f"{Path(label_name).stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            stats["images"] += 1
            stats["instances"] += len(lines)
        return stats


def write_dataset_yaml():
    content = (
        "path: " + str(OUTPUT.resolve()).replace("\\", "/") + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: crack\n"
        "  1: spalling\n"
        "  2: exposed_rebar\n"
    )
    (OUTPUT / "dataset.yaml").write_text(content, encoding="utf-8")


def main() -> int:
    if not INTERIM.exists():
        print(f"Missing interim folder: {INTERIM}")
        return 2
    stats = []
    for split, archive_name, target_count in SPLITS:
        archive = INTERIM / archive_name
        if not archive.exists():
            print(f"Missing {archive}")
            continue
        result = build_split(archive, target_count, split)
        stats.append(result)
        print(result)
    write_dataset_yaml()
    (OUTPUT / "conversion_summary.json").write_text(
        __import__("json").dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
