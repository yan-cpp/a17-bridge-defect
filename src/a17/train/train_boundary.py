"""Train YOLO11-seg with boundary-aware loss."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .boundary_loss import enable_boundary_loss


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train YOLO11-seg with boundary-aware mask loss.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="outputs/train")
    parser.add_argument("--name", default="dacl_seg_boundary")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--boundary-weight", type=float, default=0.3)
    args = parser.parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed.") from exc

    model = YOLO(args.model)
    enable_boundary_loss(model, boundary_weight=args.boundary_weight)
    results = model.train(
        data=args.data,
        task="segment",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save_period=args.save_period,
    )

    summary = {
        "task": "segment",
        "boundary_weight": args.boundary_weight,
        "data": args.data,
        "model": args.model,
        "project": args.project,
        "name": args.name,
        "best_weight": getattr(model.trainer, "best", None),
    }
    output = Path(args.project) / args.name / "training_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
