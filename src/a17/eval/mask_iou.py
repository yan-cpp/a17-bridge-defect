from __future__ import annotations

import argparse
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None


def load_binary_mask(path: Path):
    if np is None:
        raise RuntimeError("numpy is required to compute mask IoU.")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to read masks.") from exc
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    return (array > 127).astype(np.uint8)


def mask_iou(pred_mask: Path, target_mask: Path) -> float:
    pred = load_binary_mask(pred_mask)
    target = load_binary_mask(target_mask)
    if pred.shape != target.shape:
        raise ValueError(f"Mask size mismatch: {pred.shape} != {target.shape}")
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute Mask IoU for one prediction-target pair.")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args(argv)
    value = mask_iou(Path(args.pred), Path(args.target))
    print(f"Mask IoU: {value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
