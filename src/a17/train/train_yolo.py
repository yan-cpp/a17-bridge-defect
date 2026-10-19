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
    parser.add_argument("--optimizer", default="auto", help="Ultralytics optimizer name.")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine learning-rate scheduling.")
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--multi-scale", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-momentum", type=float, default=0.8)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", action="store_true", help="Cache images in memory/disk.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow writing into an existing run directory.")
    parser.add_argument("--boundary-weight", type=float, default=0.0, help="Enable boundary loss with this weight.")
    parser.add_argument("--pretrained", default=None, help="Optional checkpoint used to initialize a custom YAML model.")
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
    if args.boundary_weight > 0:
        from .boundary_loss import enable_boundary_loss

        enable_boundary_loss(model, boundary_weight=args.boundary_weight)
    train_args = {}
    if args.pretrained:
        train_args["pretrained"] = args.pretrained
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
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        multi_scale=args.multi_scale,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        workers=args.workers,
        cache=args.cache,
        exist_ok=args.exist_ok,
        **train_args,
    )

    summary = {
        "task": args.task,
        "data": args.data,
        "model": args.model,
        "boundary_weight": args.boundary_weight,
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
