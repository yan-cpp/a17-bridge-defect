"""Instance post-processing, skeletonization and physical quantification V1.

This module is the deployment-facing wrapper around :mod:`a17.quantify`.
It performs conservative mask cleanup, optional overlap de-duplication,
skeleton topology analysis, and converts centre-line length / distance-
transform width to physical units with a calibration scale.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .quantify import quantify_mask


@dataclass
class QuantificationConfigV1:
    mm_per_pixel: float
    min_area_px: int = 12
    close_radius: int = 0
    fill_holes: bool = False
    method_version: str = "V1"


@dataclass
class InstanceQuantificationV1:
    instance_id: int
    class_key: str
    area_px: int
    area_mm2: float
    length_px: float
    length_mm: float
    width_mm_p95: float
    width_mm_max: float
    skeleton_px: int
    connected_components: int
    endpoints: int
    branch_points: int
    mask_path: str | None = None
    skeleton_path: str | None = None


def clean_mask(mask: np.ndarray, config: QuantificationConfigV1) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if config.close_radius > 0:
        size = config.close_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if config.fill_holes:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(binary)
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
        binary = filled
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for component_id in range(1, count):
        if int(stats[component_id, cv2.CC_STAT_AREA]) >= config.min_area_px:
            cleaned[labels == component_id] = 1
    return cleaned


def skeleton_topology(skeleton: np.ndarray) -> tuple[int, int]:
    """Return endpoint and branch-point counts for a binary skeleton."""
    pixels = {tuple(point) for point in np.argwhere(skeleton > 0)}
    endpoints = 0
    branches = 0
    for y, x in pixels:
        neighbours = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy or dx) and (y + dy, x + dx) in pixels:
                    neighbours += 1
        if neighbours == 1:
            endpoints += 1
        elif neighbours >= 3:
            branches += 1
    return endpoints, branches


def _skeleton(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize

    return skeletonize(mask > 0).astype(np.uint8)


def quantify_instance(
    mask: np.ndarray,
    class_key: str,
    instance_id: int,
    config: QuantificationConfigV1,
    mask_path: Path | None = None,
    skeleton_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, InstanceQuantificationV1]:
    cleaned = clean_mask(mask, config)
    quant = quantify_mask(cleaned, config.mm_per_pixel)
    skeleton = _skeleton(cleaned)
    endpoints, branches = skeleton_topology(skeleton)
    result = InstanceQuantificationV1(
        instance_id=instance_id,
        class_key=class_key,
        area_px=int(cleaned.sum()),
        area_mm2=quant.area_mm2,
        length_px=quant.length_px,
        length_mm=quant.length_mm,
        width_mm_p95=quant.width_mm_p95,
        width_mm_max=quant.width_mm_max,
        skeleton_px=quant.skeleton_px,
        connected_components=quant.connected_components,
        endpoints=endpoints,
        branch_points=branches,
        mask_path=str(mask_path) if mask_path else None,
        skeleton_path=str(skeleton_path) if skeleton_path else None,
    )
    return cleaned, skeleton, result


def quantify_records_v1(
    records: list[dict],
    config: QuantificationConfigV1,
    output_dir: Path | None = None,
) -> list[dict]:
    """Quantify records carrying either an in-memory ``mask`` or ``mask_path``."""
    if output_dir is not None:
        (output_dir / "masks_v1").mkdir(parents=True, exist_ok=True)
        (output_dir / "skeletons_v1").mkdir(parents=True, exist_ok=True)

    output: list[dict] = []
    for index, record in enumerate(records, start=1):
        mask = record.get("mask")
        if mask is None and record.get("mask_path"):
            mask = cv2.imread(str(record["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        class_key = str(record.get("class_key", "defect"))
        mask_path = None
        skeleton_path = None
        if output_dir is not None:
            mask_path = output_dir / "masks_v1" / f"{index:03d}_{class_key}.png"
            skeleton_path = output_dir / "skeletons_v1" / f"{index:03d}_{class_key}.png"
        cleaned, skeleton, result = quantify_instance(
            np.asarray(mask),
            class_key,
            int(record.get("instance_id", index)),
            config,
            mask_path=mask_path,
            skeleton_path=skeleton_path,
        )
        if mask_path is not None:
            cv2.imwrite(str(mask_path), cleaned * 255)
        if skeleton_path is not None:
            cv2.imwrite(str(skeleton_path), skeleton * 255)
        item = dict(record)
        item.update(asdict(result))
        output.append(item)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Quantify one instance mask with post-processing V1.")
    parser.add_argument("--mask", required=True)
    parser.add_argument("--scale", type=float, required=True, help="Millimeters per pixel.")
    parser.add_argument("--class-key", default="crack")
    parser.add_argument("--min-area-px", type=int, default=12)
    parser.add_argument("--close-radius", type=int, default=0)
    parser.add_argument("--fill-holes", action="store_true")
    parser.add_argument("--output", default="outputs/quantification_v1")
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"Cannot read mask: {args.mask}")

    config = QuantificationConfigV1(
        mm_per_pixel=args.scale,
        min_area_px=args.min_area_px,
        close_radius=args.close_radius,
        fill_holes=args.fill_holes,
    )
    mask_path = output_dir / "mask_v1.png"
    skeleton_path = output_dir / "skeleton_v1.png"
    cleaned, skeleton, result = quantify_instance(
        mask,
        args.class_key,
        1,
        config,
        mask_path=mask_path,
        skeleton_path=skeleton_path,
    )
    cv2.imwrite(str(mask_path), cleaned * 255)
    cv2.imwrite(str(skeleton_path), skeleton * 255)
    payload = asdict(result)
    (output_dir / "quantification_v1.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
