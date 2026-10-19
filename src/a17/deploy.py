"""Export a YOLO model to ONNX and TensorRT engines for deployment."""
from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX or TensorRT.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--format", default="engine", choices=["engine", "onnx"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workspace", type=float, default=None, help="TensorRT workspace in GiB.")
    parser.add_argument("--output", default=None, help="Engine destination path.")
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    source = Path(args.model)
    export_model = source
    temporary_model = None
    if args.format == "engine":
        # Ultralytics writes an intermediate ONNX next to the source model. Use a
        # temporary stem so engine builds cannot overwrite a frozen ONNX deliverable.
        temporary_model = source.with_name(f".{source.stem}_trt_{uuid.uuid4().hex}.pt")
        shutil.copy2(source, temporary_model)
        export_model = temporary_model

    try:
        model = YOLO(str(export_model))
        exported = model.export(
            format=args.format,
            imgsz=args.imgsz,
            half=args.half,
            dynamic=args.dynamic,
            batch=args.batch,
            workspace=args.workspace,
            device=args.device,
        )
        if args.format == "engine":
            destination = Path(args.output) if args.output else source.with_suffix(".engine")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(exported, destination)
            exported = destination
    finally:
        if temporary_model is not None:
            temporary_model.unlink(missing_ok=True)
            temporary_model.with_suffix(".onnx").unlink(missing_ok=True)
            temporary_model.with_suffix(".engine").unlink(missing_ok=True)

    payload = {
        "exported": str(exported),
        "format": args.format,
        "imgsz": args.imgsz,
        "half": args.half,
        "dynamic": args.dynamic,
        "batch": args.batch,
        "workspace": args.workspace,
        "output": str(exported),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
