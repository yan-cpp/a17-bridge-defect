from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLO detection or segmentation baseline.")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml.")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained model or YAML config.")
    parser.add_argument("--task", choices=["detect", "segment"], default="detect", help="Ultralytics task.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0", help="GPU index or cpu.")
    parser.add_argument("--project", default="outputs/train", help="Ultralytics project directory.")
    parser.add_argument("--name", default="gyu_det_baseline", help="Run name.")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--save-period", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Install ML dependencies first with:\n"
            "  pip install -r requirements-ml.txt\n"
            "and install PyTorch matching the local CUDA version."
        ) from exc

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        task=args.task,
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
        "task": args.task,
        "data": args.data,
        "model": args.model,
        "project": args.project,
        "name": args.name,
        "best_weight": getattr(model.trainer, "best", None),
        "metrics": getattr(results, "metrics", None),
    }
    output = Path(args.project) / args.name / "training_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
