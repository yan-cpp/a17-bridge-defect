from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from .inspect_gyu import _iter_labels, _parse_label_line, find_archives, read_classes

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_mapping() -> dict:
    mapping_path = PROJECT_ROOT / "configs" / "data_mapping.json"
    return json.loads(mapping_path.read_text(encoding="utf-8-sig"))


def load_active_classes() -> list[str]:
    class_map_path = PROJECT_ROOT / "configs" / "class_map.json"
    class_map = json.loads(class_map_path.read_text(encoding="utf-8-sig"))
    return [str(key) for key in class_map["phase1_classes"]]


def build_source_to_internal(source_name: str, active_classes: list[str]) -> dict[str, int]:
    mapping = load_mapping()
    source_mapping = mapping["source_mappings"].get(source_name, {})
    internal_index = {key: index for index, key in enumerate(active_classes)}
    normalized = {
        key.strip().lower(): value
        for key, value in source_mapping.items()
    }
    result: dict[str, int] = {}
    for raw_name, internal_key in normalized.items():
        if internal_key not in internal_index:
            continue
        result[raw_name.lower()] = internal_index[internal_key]
    return result


def convert_labels(archive: zipfile.ZipFile, label_name: str, class_to_internal: dict[str, int], class_lookup: dict[int, str]) -> str:
    raw = archive.read(label_name).decode("utf-8", errors="replace")
    converted: list[str] = []
    dropped = 0
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        class_id = _parse_label_line(stripped)
        if class_id is None or len(tokens) < 5:
            dropped += 1
            continue
        source_class = class_lookup.get(class_id)
        if source_class is None:
            dropped += 1
            continue
        internal_id = class_to_internal.get(source_class.strip().lower())
        if internal_id is None:
            dropped += 1
            continue
        converted.append(f"{internal_id} " + " ".join(tokens[1:5]))
    return "\n".join(converted) + ("\n" if converted else "")


def convert_split(zip_path: Path, split: str, output_root: Path, class_lookup: dict[int, str], class_to_internal: dict[str, int], overwrite: bool = False) -> dict:
    images_dir = output_root / "images" / split
    labels_dir = output_root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    written_labels = 0
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            lower = name.lower()
            if not lower.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            target_image = images_dir / Path(name).name
            if target_image.exists() and not overwrite:
                continue
            with archive.open(name) as source, open(target_image, "wb") as target:
                target.write(source.read())
            copied_images += 1

        for label_name in _iter_labels(archive):
            target_label = labels_dir / Path(label_name).name
            converted = convert_labels(archive, label_name, class_to_internal, class_lookup)
            if not converted:
                continue
            target_label.write_text(converted, encoding="utf-8")
            written_labels += 1

    return {
        "split": split,
        "images": copied_images,
        "labels": written_labels,
        "source": str(zip_path),
    }


def write_dataset_yaml(output_root: Path, classes: list[str]) -> Path:
    path = output_root / "dataset.yaml"
    names = "\n".join(f"  {index}: {key}" for index, key in enumerate(classes))
    content = (
        "path: " + str(output_root.resolve()).replace("\\", "/") + "\n"
        "train: images/train\n"
        "val: images/validation\n"
        "test: images/test\n"
        "names:\n" + names + "\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert GYU-DET detection labels to A17 YOLO detection format.")
    parser.add_argument("--archives-root", help="Folder containing train.zip, valid.zip and test.zip. Defaults to auto-search.")
    parser.add_argument("--output-root", default="data/processed/gyu_det", help="Output root.")
    parser.add_argument("--source", default="gyu_det", help="Source key in data_mapping.json.")
    parser.add_argument("--classes", nargs="*", help="Internal class keys. Defaults to phase1_classes.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing image files.")
    args = parser.parse_args(argv)

    archives = find_archives([args.archives_root] if args.archives_root else None)
    if not archives:
        print("No GYU-DET split archives found.")
        return 2

    classes = args.classes or load_active_classes()
    class_to_internal = build_source_to_internal(args.source, classes)

    for split in ("train", "validation", "test"):
        if split not in archives or not archives[split]:
            print(f"Missing {split} archive. Continue with available splits only.")

    results = []
    class_lookup: dict[int, str] = {}
    for split in ("train", "validation", "test"):
        paths = archives.get(split, [])
        if not paths:
            continue
        classes_txt = read_classes(paths[0])
        if classes_txt and not class_lookup:
            class_lookup = {index: name for index, name in enumerate(classes_txt)}
        output_root = Path(args.output_root)
        result = convert_split(paths[0], split, output_root, class_lookup, class_to_internal, args.overwrite)
        results.append(result)

    output_root = Path(args.output_root)
    dataset_yaml = write_dataset_yaml(output_root, classes)
    summary = {
        "active_classes": classes,
        "results": results,
        "dataset_yaml": str(dataset_yaml),
        "source_classes": class_lookup,
    }
    summary_path = output_root / "conversion_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
