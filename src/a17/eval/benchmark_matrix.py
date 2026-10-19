"""Run and validate the 4K inference performance matrix."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import cv2
import numpy as np

from .parallel_tile import benchmark, predict_tiled_parallel


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def compare_records(
    reference: list[dict],
    candidate: list[dict],
    match_iou: float = 0.5,
) -> dict:
    """Greedily match same-class boxes and summarize output parity."""
    unmatched = set(range(len(reference)))
    ious: list[float] = []
    confidence_deltas: list[float] = []
    for item in sorted(candidate, key=lambda record: record["confidence"], reverse=True):
        best_index = None
        best_iou = 0.0
        for index in unmatched:
            target = reference[index]
            if target["class_key"] != item["class_key"]:
                continue
            value = _iou(target["bbox_xyxy"], item["bbox_xyxy"])
            if value > best_iou:
                best_index = index
                best_iou = value
        if best_index is None or best_iou < match_iou:
            continue
        unmatched.remove(best_index)
        ious.append(best_iou)
        confidence_deltas.append(
            abs(float(reference[best_index]["confidence"]) - float(item["confidence"]))
        )

    matched = len(ious)
    precision = matched / len(candidate) if candidate else float(not reference)
    recall = matched / len(reference) if reference else float(not candidate)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "reference_detections": len(reference),
        "candidate_detections": len(candidate),
        "matched": matched,
        "precision_at_iou": round(precision, 4),
        "recall_at_iou": round(recall, 4),
        "f1_at_iou": round(f1, 4),
        "mean_matched_iou": round(float(np.mean(ious)), 4) if ious else None,
        "min_matched_iou": round(float(np.min(ious)), 4) if ious else None,
        "max_abs_confidence_delta": (
            round(float(np.max(confidence_deltas)), 4) if confidence_deltas else None
        ),
        "match_iou_threshold": match_iou,
    }


def _markdown(payload: dict) -> str:
    header = (
        "| Backend | Precision | Batch | Mean ms | P50 ms | P95 ms | FPS | "
        "Detections | F1 | Mean IoU |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for row in payload["rows"]:
        validation = row.get("validation") or {}
        f1 = validation.get("f1_at_iou", "-")
        iou = validation.get("mean_matched_iou", "-")
        lines.append(
            f"| {row['backend']} | {row['precision']} | {row['batch_size']} | "
            f"{row['mean_ms']:.2f} | {row['p50_ms']:.2f} | {row['p95_ms']:.2f} | "
            f"{row['fps']:.3f} | {row['detections']} | {f1} | {iou} |"
        )
    engine = payload["tensorrt_validation"]
    verdict = "PASS" if engine["passed"] else "FAIL"
    details = (
        f"\nTensorRT parity against `{payload['reference']}`: **{verdict}** "
        f"(minimum F1 {engine['min_f1']:.3f}, minimum mean IoU "
        f"{engine['min_mean_iou'] or 0:.3f}, maximum confidence delta "
        f"{engine['max_abs_confidence_delta'] or 0:.3f}).\n"
    )
    return (
        "# A17 4K Performance Matrix\n\n"
        f"- Input: `{payload['config']['image_size']}`\n"
        f"- Tiles: `{payload['config']['tiles']}`; tile=`{payload['config']['tile']}`; "
        f"overlap=`{payload['config']['overlap']}`; imgsz=`{payload['config']['imgsz']}`\n"
        f"- GPU: `{payload['environment']['gpu']}`\n"
        f"- TensorRT: `{payload['environment'].get('tensorrt') or 'unavailable'}`\n"
        f"{details}\n{header}{chr(10).join(lines)}\n"
    )


def _row(config: dict, image: np.ndarray, model, args) -> tuple[dict, list[dict]]:
    result = benchmark(
        model,
        image,
        args.tile,
        args.overlap,
        args.imgsz,
        batch_size=config["batch_size"],
        workers=1,
        repeat=args.repeat,
        device=args.device,
        half=config["half"],
    )
    records = predict_tiled_parallel(
        model,
        image,
        tile=args.tile,
        overlap=args.overlap,
        imgsz=args.imgsz,
        batch_size=config["batch_size"],
        workers=1,
        device=args.device,
        half=config["half"],
    )
    row = {
        **config,
        **result,
        "detections": len(records),
    }
    return row, records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="4K PyTorch/TensorRT performance matrix.")
    parser.add_argument("--pytorch-model", required=True)
    parser.add_argument("--engine-model", default=None)
    parser.add_argument("--image", required=True)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-parity-f1", type=float, default=0.95)
    parser.add_argument("--min-parity-iou", type=float, default=0.95)
    parser.add_argument("--max-confidence-delta", type=float, default=0.05)
    parser.add_argument("--output", default="outputs/benchmark_4k_matrix")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    pt_path = Path(args.pytorch_model)
    engine_path = Path(args.engine_model) if args.engine_model else None
    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Cannot read image: {image_path}")
    if not pt_path.is_file():
        raise SystemExit(f"PyTorch model not found: {pt_path}")
    if engine_path is not None and not engine_path.is_file():
        raise SystemExit(f"TensorRT engine not found: {engine_path}")

    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value]
    configurations = []
    for precision in ("FP32", "FP16"):
        for batch_size in batch_sizes:
            configurations.append(
                {
                    "backend": "PyTorch",
                    "precision": precision,
                    "batch_size": batch_size,
                    "model": str(pt_path),
                    "half": precision == "FP16",
                }
            )
    if engine_path is not None:
        for batch_size in batch_sizes:
            configurations.append(
                {
                    "backend": "TensorRT",
                    "precision": "FP16",
                    "batch_size": batch_size,
                    "model": str(engine_path),
                    "half": False,
                }
            )

    models = {"PyTorch": YOLO(str(pt_path))}
    if engine_path is not None:
        models["TensorRT"] = YOLO(str(engine_path))

    rows = []
    records_by_config = {}
    for config in configurations:
        print(
            f"Benchmarking {config['backend']} {config['precision']} "
            f"batch={config['batch_size']}",
            flush=True,
        )
        row, records = _row(config, image, models[config["backend"]], args)
        key = (config["backend"], config["precision"], config["batch_size"])
        records_by_config[key] = records
        rows.append(row)

    reference_key = ("PyTorch", "FP16", 4)
    if reference_key not in records_by_config:
        reference_key = ("PyTorch", "FP16", batch_sizes[-1])
    reference = records_by_config[reference_key]
    for row in rows:
        key = (row["backend"], row["precision"], row["batch_size"])
        row["validation"] = compare_records(reference, records_by_config[key])

    engine_rows = [row for row in rows if row["backend"] == "TensorRT"]
    engine_validation = {
        "passed": bool(engine_rows),
        "rows": len(engine_rows),
        "min_f1": 0.0,
        "min_mean_iou": None,
        "max_abs_confidence_delta": None,
        "thresholds": {
            "min_f1": args.min_parity_f1,
            "min_mean_iou": args.min_parity_iou,
            "max_confidence_delta": args.max_confidence_delta,
        },
    }
    if engine_rows:
        validations = [row["validation"] for row in engine_rows]
        engine_validation["min_f1"] = min(item["f1_at_iou"] for item in validations)
        ious = [
            item["mean_matched_iou"]
            for item in validations
            if item["mean_matched_iou"] is not None
        ]
        deltas = [
            item["max_abs_confidence_delta"]
            for item in validations
            if item["max_abs_confidence_delta"] is not None
        ]
        engine_validation["min_mean_iou"] = min(ious) if ious else None
        engine_validation["max_abs_confidence_delta"] = max(deltas) if deltas else None
        engine_validation["passed"] = (
            engine_validation["min_f1"] >= args.min_parity_f1
            and engine_validation["min_mean_iou"] is not None
            and engine_validation["min_mean_iou"] >= args.min_parity_iou
            and engine_validation["max_abs_confidence_delta"] is not None
            and engine_validation["max_abs_confidence_delta"] <= args.max_confidence_delta
        )

    try:
        import torch
        import tensorrt as trt

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        cuda = torch.version.cuda
        tensorrt_version = trt.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError as error:
        gpu = "unavailable"
        cuda = None
        cuda_available = False
        tensorrt_version = f"unavailable: {error}"

    height, width = image.shape[:2]
    payload = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu": gpu,
            "cuda_available": cuda_available,
            "cuda": cuda,
            "tensorrt": tensorrt_version,
        },
        "models": {
            "pytorch": {"path": str(pt_path), "sha256": _sha256(pt_path)},
            "tensorrt": (
                {"path": str(engine_path), "sha256": _sha256(engine_path)}
                if engine_path is not None
                else None
            ),
        },
        "config": {
            "image": str(image_path),
            "image_size": f"{width}x{height}",
            "tile": args.tile,
            "overlap": args.overlap,
            "imgsz": args.imgsz,
            "tiles": len(list(_tiles(image, args.tile, args.overlap))),
            "repeat": args.repeat,
        },
        "reference": f"{reference_key[0]} {reference_key[1]} batch={reference_key[2]}",
        "rows": rows,
        "tensorrt_validation": engine_validation,
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "performance_matrix.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "performance_matrix.md").write_text(_markdown(payload), encoding="utf-8")
    with (output / "performance_matrix.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = [
            "backend",
            "precision",
            "batch_size",
            "tiles",
            "mean_ms",
            "p50_ms",
            "p95_ms",
            "fps",
            "detections",
            "f1_at_iou",
            "mean_matched_iou",
            "max_abs_confidence_delta",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            validation = row["validation"]
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "f1_at_iou": validation["f1_at_iou"],
                    "mean_matched_iou": validation["mean_matched_iou"],
                    "max_abs_confidence_delta": validation["max_abs_confidence_delta"],
                }
            )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _tiles(image: np.ndarray, tile: int, overlap: int):
    from .tile_inference import tile_crops

    return tile_crops(image, tile, overlap)


if __name__ == "__main__":
    raise SystemExit(main())
