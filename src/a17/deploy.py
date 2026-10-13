"""Export a YOLO model to ONNX and TensorRT engines for deployment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX or TensorRT.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--format", default="engine", choices=["engine", "onnx"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)

    from ultralytics import YOLO
    model = YOLO(args.model)
    exported = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        device=args.device,
    )
    payload = {"exported": str(exported), "format": args.format, "imgsz": args.imgsz, "half": args.half}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
