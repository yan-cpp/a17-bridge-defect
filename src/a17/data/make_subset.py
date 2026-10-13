"""Create a small stratified subset of the processed dacl10k dataset for fast baselines."""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = PROJECT_ROOT / "data" / "processed" / "dacl10k"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "dacl10k_subset"

SPLITS = {"train": "train", "val": "val"}
CLASS_NAMES = ["crack", "spalling", "exposed_rebar"]


def label_classes(label_path: Path) -> set[int]:
    classes: set[int] = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        classes.add(int(line.split()[0]))
    return classes


def main() -> int:
    rng = random.Random(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(exist_ok=True)
    (OUTPUT_DIR / "labels").mkdir(exist_ok=True)

    summary: dict[str, dict] = {}
    for source_split, target_split in SPLITS.items():
        src_images = SOURCE_DIR / "images" / source_split
        src_labels = SOURCE_DIR / "labels" / source_split
        images = sorted(p.name for p in src_images.glob("*"))
        # bucket image names by every class they contain
        by_class: dict[int, list[str]] = {i: [] for i in range(len(CLASS_NAMES))}
        for name in images:
            lab = src_labels / (Path(name).stem + ".txt")
            if not lab.exists():
                continue
            for cls in label_classes(lab):
                by_class[cls].append(name)

        target_n = 240 if source_split == "train" else 40
        per_class = max(1, target_n // (3 * 2))
        chosen: list[str] = []
        for cls in range(len(CLASS_NAMES)):
            rng.shuffle(by_class[cls])
            chosen.extend(by_class[cls][:per_class])
        # fill the remainder randomly
        rng.shuffle(images)
        for name in images:
            if len(chosen) >= target_n:
                break
            if name not in chosen:
                chosen.append(name)
        chosen = chosen[:target_n]

        tgt_images = OUTPUT_DIR / "images" / target_split
        tgt_labels = OUTPUT_DIR / "labels" / target_split
        tgt_images.mkdir(parents=True, exist_ok=True)
        tgt_labels.mkdir(parents=True, exist_ok=True)
        instances = 0
        for name in chosen:
            shutil.copy2(src_images / name, tgt_images / name)
            label_file = src_labels / (Path(name).stem + ".txt")
            shutil.copy2(label_file, tgt_labels / label_file.name)
            instances += sum(1 for _ in label_file.read_text(encoding="utf-8").splitlines() if _.strip())
        summary[target_split] = {"images": len(chosen), "instances": instances}

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
    (OUTPUT_DIR / "conversion_summary.json").write_text(
        json.dumps({"source": str(SOURCE_DIR), "splits": summary, "seed": 42}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
