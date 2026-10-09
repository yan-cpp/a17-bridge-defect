from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None


DEFAULT_SEARCH_ROOTS = [
    r"C:\Users\Lenovo\Downloads",
    r"C:\Users\Lenovo\Desktop",
    r"D:\Downloads",
    r"D:\zm",
]


def _split_from_name(name: str) -> str | None:
    stem = Path(name).stem.lower()
    mapping = {
        "train": "train",
        "training": "train",
        "valid": "validation",
        "val": "validation",
        "validation": "validation",
        "test": "test",
        "testing": "test",
    }
    return mapping.get(stem)


def find_archives(search_roots: list[str] | None = None) -> dict[str, list[Path]]:
    roots = [Path(item) for item in (search_roots or DEFAULT_SEARCH_ROOTS)]
    found: dict[str, list[Path]] = {}
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() != ".zip":
                continue
            split = _split_from_name(path.stem)
            if split is None:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.setdefault(split, []).append(path)
    return found


def read_classes(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith("classes.txt")]
        if not names:
            return []
        raw = archive.read(names[0]).decode("utf-8", errors="replace")
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _iter_labels(archive: zipfile.ZipFile):
    for name in archive.namelist():
        lower = name.lower()
        if not lower.endswith(".txt"):
            continue
        if "classes.txt" in lower:
            continue
        parts = Path(name).parts
        if "labels" in (part.lower() for part in parts) or parts[-1].lower().endswith(".txt"):
            yield name


def _parse_label_line(line: str):
    tokens = line.strip().split()
    if not tokens:
        return None
    try:
        class_id = int(float(tokens[0]))
    except ValueError:
        return None
    return class_id


def count_labels(zip_path: Path, classes: list[str]) -> dict:
    label_files = 0
    instances = 0
    class_counts: Counter = Counter()
    malformed = 0
    with zipfile.ZipFile(zip_path) as archive:
        for name in _iter_labels(archive):
            label_files += 1
            raw = archive.read(name).decode("utf-8", errors="replace")
            for line in raw.splitlines():
                parsed = _parse_label_line(line)
                if parsed is None:
                    continue
                if line.strip() and parsed is None:
                    malformed += 1
                    continue
                instances += 1
                if 0 <= parsed < len(classes):
                    class_counts[classes[parsed]] += 1
                else:
                    class_counts[f"__unknown_{parsed}__"] += 1
    return {
        "label_files": label_files,
        "instances": instances,
        "class_counts": dict(sorted(class_counts.items())),
        "malformed_lines": malformed,
    }


def sample_image_sizes(zip_path: Path, limit: int = 5) -> list[dict]:
    if Image is None:
        return []
    samples = []
    with zipfile.ZipFile(zip_path) as archive:
        image_names = [
            name
            for name in archive.namelist()
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        for name in image_names[:limit]:
            try:
                with Image.open(archive.open(name)) as image:
                    samples.append({"file": name, "width": image.width, "height": image.height})
            except Exception:
                samples.append({"file": name, "width": None, "height": None})
    return samples


def build_data_card(search_roots: list[str] | None = None) -> dict:
    archives = find_archives(search_roots)
    splits: dict[str, dict] = {}
    classes: list[str] = []
    for split, paths in archives.items():
        if not paths:
            continue
        classes = classes or read_classes(paths[0])
        stats = count_labels(paths[0], classes)
        sizes = sample_image_sizes(paths[0], limit=5)
        splits[split] = {
            "archives": [str(path) for path in paths],
            "statistics": stats,
            "image_samples": sizes,
        }

    return {
        "name": "GYU-DET",
        "classes": classes,
        "splits": splits,
        "source_archives": {split: [str(path) for path in paths] for split, paths in archives.items()},
    }


def write_markdown(card: dict, output: Path) -> None:
    lines = [
        "# GYU-DET 数据卡",
        "",
        "## 类别",
        "",
    ]
    if card["classes"]:
        for index, name in enumerate(card["classes"]):
            lines.append(f"{index}. {name}")
    else:
        lines.append("- 未找到 classes.txt")
    lines.append("")
    for split, payload in card["splits"].items():
        lines.append(f"## {split}")
        lines.append("")
        stats = payload["statistics"]
        lines.append(f"- 标签文件：{stats['label_files']}")
        lines.append(f"- 实例数量：{stats['instances']}")
        lines.append(f"- 类别分布：{json.dumps(stats['class_counts'], ensure_ascii=False)}")
        lines.append(f"- 格式异常行：{stats['malformed_lines']}")
        lines.append(f"- 图片抽样：{json.dumps(payload['image_samples'], ensure_ascii=False)}")
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect GYU-DET archives and generate a data card.")
    parser.add_argument("--roots", nargs="*", default=DEFAULT_SEARCH_ROOTS, help="Folders to scan for split zips.")
    parser.add_argument("--output-json", default="data/data_card.json", help="JSON output path.")
    parser.add_argument("--output-md", default="data/data_card.md", help="Markdown output path.")
    args = parser.parse_args(argv)

    card = build_data_card(args.roots)
    found = [str(path) for paths in card["source_archives"].values() for path in paths]
    if not found:
        print("No GYU-DET split archives found. Provide --roots or place train.zip/valid.zip/test.zip in a searched folder.")
        return 2

    json_path = Path(args.output_json)
    md_path = Path(args.output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(card, md_path)
    print(f"data card written: {json_path}")
    print(f"data card written: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
