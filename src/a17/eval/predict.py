from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit("ultralytics 未安装。请先运行环境安装步骤。") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the trained YOLO model on images, folders, or video.")
    parser.add_argument("--model", default="runs/detect/outputs/train/gyu_det_full/weights/best.pt")
    parser.add_argument("--source", required=True, help="Image path, folder, or video path.")
    parser.add_argument("--output", default="outputs/predict")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)

    model = YOLO(args.model)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        project=args.output,
        name="predict",
        exist_ok=True,
    )

    summary = []
    for result in results:
        counts = {}
        for cls in result.boxes.cls.tolist():
            name = result.names[int(cls)]
            counts[name] = counts.get(name, 0) + 1
        summary.append({"source": Path(result.path).name, "detections": counts})
    print(summary)
    print(f"结果已保存到: {Path(args.output) / 'predict'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
