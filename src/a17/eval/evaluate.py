from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency
    YOLO = None


def _number(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _list_of_numbers(value):
    if value is None:
        return None
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return None


def _safe_getattr(obj, attr, default=None):
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def collect_metrics(metrics) -> dict[str, Any]:
    row: dict[str, Any] = {}
    box = _safe_getattr(metrics, "box")
    seg = _safe_getattr(metrics, "seg")
    row["box_map50"] = _number(_safe_getattr(box, "map50"))
    row["box_map50_95"] = _number(_safe_getattr(box, "map"))
    row["box_precision"] = _number(_safe_getattr(box, "mp"))
    row["box_recall"] = _number(_safe_getattr(box, "mr"))
    row["seg_map50"] = _number(_safe_getattr(seg, "map50"))
    row["seg_map50_95"] = _number(_safe_getattr(seg, "map"))
    row["mask_iou_mean"] = _number(_safe_getattr(seg, "miou"))
    row["box_map50_per_class"] = _list_of_numbers(_safe_getattr(box, "maps50"))
    row["seg_map50_per_class"] = _list_of_numbers(_safe_getattr(seg, "maps50"))
    row["seg_map50_95_per_class"] = _list_of_numbers(_safe_getattr(seg, "maps"))

    speed = _safe_getattr(metrics, "speed", {}) or {}
    inference_ms = _number(speed.get("inference"))
    preprocess_ms = _number(speed.get("preprocess"))
    postprocess_ms = _number(speed.get("postprocess"))
    row["inference_ms"] = inference_ms
    row["preprocess_ms"] = preprocess_ms
    row["postprocess_ms"] = postprocess_ms
    row["fps"] = 1000.0 / inference_ms if inference_ms and inference_ms > 0 else None
    return row


def write_reports(rows: list[dict], output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{name}_metrics.json"
    csv_path = output_dir / f"{name}_metrics.csv"
    md_path = output_dir / f"{name}_metrics.md"

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    header = "| 指标 | 数值 |\n|---|---|\n"
    body = "\n".join(f"| {key} | {value} |" for key, value in rows[-1].items())
    md_path.write_text("# Baseline Metrics\n\n" + header + body + "\n", encoding="utf-8")
    return md_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO baseline and export mAP, Mask IoU, speed and FPS.")
    parser.add_argument("--model", required=True, help="Path to trained model weights.")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml.")
    parser.add_argument("--task", choices=["detect", "segment"], default="detect")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/eval")
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--save-json", action="store_true", help="Save COCO-style JSON predictions.")
    parser.add_argument("--plots", action="store_true", help="Save validation plots.")
    parser.add_argument("--rect", action="store_true", help="Use rectangular inference for faster validation.")
    return parser


def main(argv: list[str] | None = None) -> int:
    if YOLO is None:
        raise SystemExit("ultralytics is not installed. Install ML dependencies before evaluation.")

    args = build_parser().parse_args(argv)
    model = YOLO(args.model)
    metrics = model.val(
        data=args.data,
        task=args.task,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        plots=args.plots,
        save_json=args.save_json,
        rect=args.rect,
    )
    row = collect_metrics(metrics)
    row["model"] = str(Path(args.model).resolve())
    row["data"] = str(Path(args.data).resolve())
    row["task"] = args.task
    row["split"] = args.split
    output_dir = Path(args.output)
    report = write_reports([row], output_dir, args.name)
    print(json.dumps(row, ensure_ascii=False, indent=2))
    print(f"report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
