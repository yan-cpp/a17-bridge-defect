"""Benchmark sequential and batched tiled inference on high-resolution images."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import cv2
import numpy as np

from .parallel_tile import benchmark


def _markdown(rows: list[dict]) -> str:
    header = (
        "| 模型 | 分辨率 | batch | workers | tiles | 平均 ms | P50 ms | P95 ms | FPS |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body = "\n".join(
        "| {model} | {image_size} | {batch_size} | {workers} | {tiles} | "
        "{mean_ms} | {p50_ms} | {p95_ms} | {fps} |".format(**row)
        for row in rows
    )
    return "# A17 4K Speed Benchmark\n\n" + header + body + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="4K tiled inference speed benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--workers", default="1,2")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--output", default="outputs/benchmark_4k")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    model = YOLO(args.model)
    height, width = image.shape[:2]

    rows: list[dict] = []
    for batch_size in [int(value) for value in args.batch_sizes.split(",") if value]:
        for workers in [int(value) for value in args.workers.split(",") if value]:
            result = benchmark(
                model,
                image,
                args.tile,
                args.overlap,
                args.imgsz,
                batch_size=batch_size,
                workers=workers,
                repeat=args.repeat,
                device=args.device,
                half=args.half,
            )
            rows.append(
                {
                    "model": str(args.model),
                    "image": str(args.image),
                    "image_size": f"{width}x{height}",
                    **result,
                }
            )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu": (
                __import__("torch").cuda.get_device_name(0)
                if __import__("torch").cuda.is_available()
                else "cpu"
            ),
        },
        "config": {
            "tile": args.tile,
            "overlap": args.overlap,
            "imgsz": args.imgsz,
            "repeat": args.repeat,
        },
        "rows": rows,
    }
    (output / "speed_benchmark.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "speed_benchmark.md").write_text(_markdown(rows), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
