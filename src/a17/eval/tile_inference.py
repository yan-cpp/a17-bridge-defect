"""4K-friendly sliding-window inference and latency benchmark for YOLO models."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def tile_crops(image: np.ndarray, tile: int, overlap: int):
    height, width = image.shape[:2]
    step = tile - overlap
    for y in range(0, height, step):
        y = min(y, height - tile) if height > tile else 0
        for x in range(0, width, step):
            x = min(x, width - tile) if width > tile else 0
            yield (x, y, image[y : y + tile, x : x + tile])


def predict_tiled(model, image: np.ndarray, tile: int = 1024, overlap: int = 128, imgsz: int = 1024, conf: float = 0.25, include_masks: bool = False):
    """Run a YOLO model on overlapping tiles and map boxes/masks to full-image coordinates."""
    records = []
    for x, y, crop in tile_crops(image, tile, overlap):
        results = model.predict(source=crop, imgsz=imgsz, conf=conf, retina_masks=True, verbose=False)
        result = results[0]
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        names = model.names
        for i, box in enumerate(boxes):
            records.append({
                "tile_xy": [int(x), int(y)],
                "bbox_xyxy": [round(float(box[0] + x), 1), round(float(box[1] + y), 1),
                              round(float(box[2] + x), 1), round(float(box[3] + y), 1)],
                "class_key": names.get(int(classes[i]), str(classes[i])),
                "confidence": round(float(confs[i]), 4),
                "mask": masks[i].tolist() if include_masks and masks is not None and i < len(masks) else None,
            })
    return records


def benchmark(model, image: np.ndarray, tile: int = 1024, overlap: int = 128, imgsz: int = 1024):
    crops = list(tile_crops(image, tile, overlap))
    start = time.perf_counter()
    for x, y, crop in crops:
        model.predict(source=crop, imgsz=imgsz, conf=0.25, retina_masks=False, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {"tiles": len(crops), "total_ms": round(elapsed_ms, 2), "per_tile_ms": round(elapsed_ms / len(crops), 2)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sliding-window inference benchmark for large images.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)

    from ultralytics import YOLO
    model = YOLO(args.model)
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    latency = benchmark(model, image, args.tile, args.overlap, args.imgsz)
    records = predict_tiled(model, image, args.tile, args.overlap, args.imgsz)
    print(json.dumps({"latency": latency, "detections": records[:20], "total_detections": len(records)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
