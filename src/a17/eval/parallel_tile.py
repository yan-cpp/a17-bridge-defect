"""Batched / parallel tiled inference for 4K bridge inspection images."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .tile_inference import public_records, stitch_mask_records, tile_crops


def _class_name(model, class_id: int) -> str:
    names = getattr(model, "names", {})
    return str(names.get(class_id, class_id)) if isinstance(names, dict) else str(names[class_id])


def _bbox_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _result_records(result, model, image: np.ndarray, x: int, y: int) -> list[dict]:
    if result.boxes is None:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None
    height, width = image.shape[:2]
    output: list[dict] = []
    for index, box in enumerate(boxes):
        mask = None
        if masks is not None and index < len(masks):
            tile_mask = np.asarray(masks[index])
            if tile_mask.ndim == 3:
                tile_mask = tile_mask[0]
            if tile_mask.shape[:2] != (result.orig_shape[0], result.orig_shape[1]):
                tile_mask = cv2.resize(
                    tile_mask.astype(np.float32),
                    (int(result.orig_shape[1]), int(result.orig_shape[0])),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = np.zeros((height, width), dtype=np.uint8)
            crop_h, crop_w = tile_mask.shape[:2]
            mask[y : y + crop_h, x : x + crop_w] = (tile_mask > 0.5).astype(np.uint8)

        mask_bbox = _bbox_from_mask(mask) if mask is not None else None
        bbox = mask_bbox or [
            float(box[0] + x),
            float(box[1] + y),
            float(box[2] + x),
            float(box[3] + y),
        ]
        output.append(
            {
                "class_key": _class_name(model, int(classes[index])),
                "confidence": float(confidences[index]),
                "bbox_xyxy": [round(float(value), 2) for value in bbox],
                "mask": mask,
                "mask_area_px": int(np.count_nonzero(mask)) if mask is not None else None,
                "tile_xy": [int(x), int(y)],
            }
        )
    return output


def predict_tiled_parallel(
    model,
    image: np.ndarray,
    tile: int = 1024,
    overlap: int = 128,
    imgsz: int = 1024,
    conf: float = 0.25,
    iou: float = 0.5,
    batch_size: int = 4,
    workers: int = 1,
    scale: float | None = None,
    masks_dir: Path | None = None,
    image_name: str = "image",
    device: str | None = None,
    half: bool = False,
) -> list[dict]:
    """Run tiles in batches, optionally overlapping batch execution with threads."""
    crops = list(tile_crops(image, tile, overlap))
    batches = [crops[index : index + batch_size] for index in range(0, len(crops), batch_size)]

    def run_batch(batch):
        predict_args = {
            "source": [crop for _, _, crop in batch],
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "retina_masks": True,
            "verbose": False,
        }
        if device is not None:
            predict_args["device"] = device
        if half:
            predict_args["half"] = True
        return batch, model.predict(**predict_args)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            completed = list(executor.map(run_batch, batches))
    else:
        completed = [run_batch(batch) for batch in batches]

    raw_records: list[dict] = []
    for batch, results in completed:
        for (x, y, _), result in zip(batch, results):
            raw_records.extend(_result_records(result, model, image, x, y))

    records = stitch_mask_records(raw_records)
    if masks_dir is not None:
        masks_dir.mkdir(parents=True, exist_ok=True)
    for instance_id, record in enumerate(records, start=1):
        record["instance_id"] = instance_id
        mask = record.get("mask")
        if mask is not None and scale is not None:
            from ..quantification_v1 import QuantificationConfigV1, quantify_instance

            mask, skeleton, quant = quantify_instance(
                mask,
                str(record.get("class_key", "defect")),
                instance_id,
                QuantificationConfigV1(mm_per_pixel=scale),
            )
            record["mask"] = mask
            record.update(
                {
                    "length_mm": round(quant.length_mm, 3),
                    "length_px": round(quant.length_px, 2),
                    "width_mm_p95": round(quant.width_mm_p95, 3),
                    "width_mm_max": round(quant.width_mm_max, 3),
                    "area_mm2": round(quant.area_mm2, 3),
                    "skeleton_px": quant.skeleton_px,
                    "connected_components": quant.connected_components,
                    "endpoints": quant.endpoints,
                    "branch_points": quant.branch_points,
                    "quantification_version": "V1",
                    "scale_mm_per_pixel": scale,
                }
            )
        if mask is not None and masks_dir is not None:
            mask_path = masks_dir / f"{Path(image_name).stem}_{instance_id:03d}.png"
            cv2.imwrite(str(mask_path), (mask > 0).astype(np.uint8) * 255)
            record["mask_path"] = str(mask_path)
            if scale is not None:
                skeleton_path = masks_dir / f"{Path(image_name).stem}_{instance_id:03d}_skeleton.png"
                cv2.imwrite(str(skeleton_path), skeleton * 255)
                record["skeleton_path"] = str(skeleton_path)
        record.pop("mask", None)
    return records


def benchmark(
    model,
    image: np.ndarray,
    tile: int,
    overlap: int,
    imgsz: int,
    batch_size: int = 4,
    workers: int = 1,
    repeat: int = 3,
    device: str | None = None,
    half: bool = False,
    warmup: bool = True,
) -> dict:
    if warmup:
        predict_tiled_parallel(
            model,
            image,
            tile=tile,
            overlap=overlap,
            imgsz=imgsz,
            batch_size=batch_size,
            workers=workers,
            device=device,
            half=half,
        )
    latencies = []
    for _ in range(repeat):
        start = time.perf_counter()
        predict_tiled_parallel(
            model,
            image,
            tile=tile,
            overlap=overlap,
            imgsz=imgsz,
            batch_size=batch_size,
            workers=workers,
            device=device,
            half=half,
        )
        latencies.append((time.perf_counter() - start) * 1000.0)
    return {
        "tiles": len(list(tile_crops(image, tile, overlap))),
        "batch_size": batch_size,
        "workers": workers,
        "repeat": repeat,
        "mean_ms": round(float(np.mean(latencies)), 2),
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "fps": round(1000.0 / float(np.mean(latencies)), 3),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Parallel/batched tiled inference benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--output", default="outputs/parallel_tile")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    model = YOLO(args.model)
    output = Path(args.output)
    result = benchmark(
        model,
        image,
        args.tile,
        args.overlap,
        args.imgsz,
        args.batch_size,
        args.workers,
        args.repeat,
        args.device,
        args.half,
    )
    records = predict_tiled_parallel(
        model,
        image,
        tile=args.tile,
        overlap=args.overlap,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        workers=args.workers,
        scale=args.scale,
        masks_dir=output / "masks",
        image_name=Path(args.image).name,
        device=args.device,
        half=args.half,
    )
    payload = {"benchmark": result, "detections": public_records(records)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
