"""Run a YOLO-seg model on images and export masks + per-instance JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..quantify import quantify_mask


def run(
    model_path: Path,
    source: Path,
    output: Path,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    scale: float | None,
) -> int:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed.") from exc

    model = YOLO(str(model_path))
    output.mkdir(parents=True, exist_ok=True)
    masks_dir = output / "masks"
    masks_dir.mkdir(exist_ok=True)

    results = model.predict(
        source=str(source),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        retina_masks=True,
        save=False,
    )

    records: list[dict] = []
    for result in results:
        image_name = Path(result.path).name
        if result.masks is None:
            continue
        boxes = result.boxes
        if boxes is None:
            continue
        mask_data = result.masks.data.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        original = result.orig_shape[:2]

        for index, mask in enumerate(mask_data):
            class_id = int(class_ids[index])
            name = model.names.get(class_id, str(class_id))
            binary = (mask > 0.5).astype(np.uint8)
            x1, y1, x2, y2 = boxes_xyxy[index]
            mask_name = f"{Path(image_name).stem}_{index:03d}.png"
            mask_path = masks_dir / mask_name
            cv2.imwrite(str(mask_path), binary * 255)

            item: dict = {
                "image": image_name,
                "instance_id": index,
                "class_key": name,
                "confidence": round(float(confs[index]), 4),
                "bbox_xyxy": [round(float(v), 2) for v in (x1, y1, x2, y2)],
                "mask_path": str(mask_path),
                "orig_height": int(original[0]),
                "orig_width": int(original[1]),
            }
            if scale is not None:
                quant = quantify_mask(binary, scale)
                item["length_mm"] = round(quant.length_mm, 3)
                item["width_mm_p95"] = round(quant.width_mm_p95, 3)
                item["width_mm_max"] = round(quant.width_mm_max, 3)
                item["area_mm2"] = round(quant.area_mm2, 3)
            records.append(item)

    (output / "predictions.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"Saved to {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Segment images and export masks plus JSON.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="outputs/predict_seg")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--scale", type=float, default=None, help="Millimeters per pixel for quantification.")
    args = parser.parse_args(argv)
    return run(
        Path(args.model),
        Path(args.source),
        Path(args.output),
        args.imgsz,
        args.conf,
        args.iou,
        args.device,
        args.scale,
    )


if __name__ == "__main__":
    raise SystemExit(main())
