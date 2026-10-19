"""Build the first-round V1 ablation table from evaluation JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _row(name: str, metrics: dict) -> dict:
    return {
        "model": name,
        "box_map50": metrics.get("box_map50"),
        "box_map50_95": metrics.get("box_map50_95"),
        "seg_map50": metrics.get("seg_map50"),
        "seg_map50_95": metrics.get("seg_map50_95"),
        "inference_ms": metrics.get("inference_ms"),
        "fps": metrics.get("fps"),
    }


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload[0]
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Create V1 ablation table from metrics JSON files.")
    parser.add_argument("--input", default="outputs/ablation_v1")
    parser.add_argument("--output", default="outputs/ablation_v1")
    args = parser.parse_args(argv)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = [
        "ablation_v1_baseline",
        "ablation_v1_p2",
        "ablation_v1_p2_ms",
        "ablation_v1_p2_ms_boundary",
    ]
    labels = {
        "ablation_v1_baseline": "Baseline",
        "ablation_v1_p2": "+P2",
        "ablation_v1_p2_ms": "+P2+多尺度",
        "ablation_v1_p2_ms_boundary": "+P2+多尺度+边界损失",
    }
    rows = []
    for name in names:
        path = input_dir / f"{name}_metrics.json"
        if path.exists():
            rows.append(_row(labels.get(name, name), _load(path)))

    json_path = output_dir / "ablation_v1_table.json"
    md_path = output_dir / "ablation_v1_table.md"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    header = (
        "| 模型 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 | "
        "推理 ms | FPS |\n|---|---:|---:|---:|---:|---:|---:|\n"
    )
    body = "\n".join(
        "| {model} | {box_map50} | {box_map50_95} | {seg_map50} | {seg_map50_95} | "
        "{inference_ms} | {fps} |".format(**row)
        for row in rows
    )
    md_path.write_text("# A17 V1 Ablation\n\n" + header + body + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
