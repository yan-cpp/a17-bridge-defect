"""A17 病害物理量化：从实例 Mask 计算裂缝长度、最大宽度与面积。

约定：输入为二值 Mask（0/255 或 0/1）。毫米每像素必须由标定提供。
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage import measure, morphology

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


@dataclass
class QuantResult:
    length_mm: float
    width_mm_max: float
    width_mm_p95: float
    area_mm2: float
    scale_mm_per_pixel: float
    skeleton_px: int
    connected_components: int
    length_px: float
    skeleton_length_method: str = "8-connected-geodesic"


def load_binary_mask(path: Path) -> np.ndarray:
    if Image is None:
        raise RuntimeError("Pillow is required to read masks.")
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    return (array > 127).astype(np.uint8)


def _skeleton_length_px(mask: np.ndarray) -> tuple[float, int, int]:
    """Return weighted length, skeleton pixel count and component count.

    Counting skeleton pixels underestimates diagonal paths. Count each
    undirected 4/8-neighbour edge once, with diagonal edges weighted by
    sqrt(2), to obtain a more accurate centre-line length.
    """
    labels, num = measure.label(mask, connectivity=2, return_num=True)
    total = 0.0
    skeleton_px = 0
    kept_components = 0
    directions = ((0, 1), (1, -1), (1, 0), (1, 1))
    for index in range(1, num + 1):
        component = (labels == index)
        if component.sum() < 4:
            continue
        skeleton = morphology.skeletonize(component)
        pixels = {tuple(point) for point in np.argwhere(skeleton)}
        if not pixels:
            continue
        kept_components += 1
        skeleton_px += len(pixels)
        for y, x in pixels:
            for dy, dx in directions:
                if (y + dy, x + dx) in pixels:
                    total += sqrt(dy * dy + dx * dx)
    return total, skeleton_px, kept_components


def _robust_width_px(mask: np.ndarray) -> tuple[float, float]:
    """Return (p95 half-width*2 in px, max width in px)."""
    distance = ndimage.distance_transform_edt(mask)
    nonzero = distance[mask > 0]
    if nonzero.size == 0:
        return 0.0, 0.0
    max_width = float(nonzero.max() * 2)
    p95 = float(np.percentile(nonzero, 95) * 2)
    return p95, max_width


def quantify_mask(mask: np.ndarray, mm_per_pixel: float) -> QuantResult:
    if mm_per_pixel <= 0:
        raise ValueError("mm_per_pixel must be greater than zero.")
    binary = (mask > 127).astype(np.uint8) if mask.dtype == np.uint8 and mask.max() > 1 else (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return QuantResult(0.0, 0.0, 0.0, 0.0, mm_per_pixel, 0, 0, 0.0)
    length_px, skeleton_px, components = _skeleton_length_px(binary)
    width_p95, width_max = _robust_width_px(binary)
    area_px = int(binary.sum())
    return QuantResult(
        length_mm=length_px * mm_per_pixel,
        width_mm_max=width_max * mm_per_pixel,
        width_mm_p95=width_p95 * mm_per_pixel,
        area_mm2=area_px * mm_per_pixel * mm_per_pixel,
        scale_mm_per_pixel=mm_per_pixel,
        skeleton_px=skeleton_px,
        connected_components=components,
        length_px=length_px,
    )


def quantify_mask_file(path: Path, mm_per_pixel: float) -> QuantResult:
    return quantify_mask(load_binary_mask(path), mm_per_pixel)

def main(argv=None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Quantify crack length and width from a binary mask.")
    parser.add_argument("--mask", required=True, help="Path to a binary defect mask.")
    parser.add_argument("--scale", type=float, required=True, help="Millimeters per pixel.")
    args = parser.parse_args(argv)
    result = quantify_mask_file(Path(args.mask), args.scale)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
