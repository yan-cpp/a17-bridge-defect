"""Sliding-window inference, de-duplication and mask stitching for YOLO models."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def _axis_starts(size: int, tile: int, step: int) -> list[int]:
    if size <= tile:
        return [0]
    last = size - tile
    starts = list(range(0, last + 1, max(1, step)))
    if starts[-1] != last:
        starts.append(last)
    return starts


def tile_crops(image: np.ndarray, tile: int, overlap: int):
    """Yield non-repeating sliding-window crops as (x, y, crop)."""
    if tile <= 0:
        raise ValueError("tile must be greater than zero.")
    if overlap < 0 or overlap >= tile:
        raise ValueError("overlap must satisfy 0 <= overlap < tile.")
    height, width = image.shape[:2]
    step = tile - overlap
    for y in _axis_starts(height, tile, step):
        for x in _axis_starts(width, tile, step):
            yield x, y, image[y : y + tile, x : x + tile]


def _class_name(model, class_id: int) -> str:
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    try:
        return str(names[class_id])
    except (TypeError, IndexError):
        return str(class_id)


def _bbox_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _box_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - inter
    return inter / union if union > 0 else 0.0


def _mask_overlap_ratio(first: dict, second: dict) -> float:
    """Return intersection divided by the smaller mask area."""
    first_mask = first.get("mask")
    second_mask = second.get("mask")
    first_box = first.get("bbox_xyxy")
    second_box = second.get("bbox_xyxy")
    if first_mask is None or second_mask is None or not first_box or not second_box:
        return 0.0

    height, width = first_mask.shape[:2]
    x1 = max(0, int(np.floor(max(first_box[0], second_box[0]))))
    y1 = max(0, int(np.floor(max(first_box[1], second_box[1]))))
    x2 = min(width, int(np.ceil(min(first_box[2], second_box[2]))))
    y2 = min(height, int(np.ceil(min(first_box[3], second_box[3]))))
    if x1 >= x2 or y1 >= y2:
        return 0.0

    first_roi = first_mask[y1:y2, x1:x2] > 0
    second_roi = second_mask[y1:y2, x1:x2] > 0
    intersection = int(np.count_nonzero(first_roi & second_roi))
    if intersection == 0:
        return 0.0
    smaller_area = min(int(np.count_nonzero(first_mask)), int(np.count_nonzero(second_mask)))
    return intersection / smaller_area if smaller_area else 0.0


def _bbox_gap(first: list[float], second: list[float]) -> float:
    dx = max(0.0, max(first[0], second[0]) - min(first[2], second[2]))
    dy = max(0.0, max(first[1], second[1]) - min(first[3], second[3]))
    return float(np.hypot(dx, dy))


def _same_dominant_axis(first: list[float], second: list[float]) -> bool:
    """Conservatively identify collinear thin fragments from their boxes."""
    first_width = max(1.0, first[2] - first[0])
    first_height = max(1.0, first[3] - first[1])
    second_width = max(1.0, second[2] - second[0])
    second_height = max(1.0, second[3] - second[1])
    both_horizontal = first_width >= 1.5 * first_height and second_width >= 1.5 * second_height
    both_vertical = first_height >= 1.5 * first_width and second_height >= 1.5 * second_width
    return both_horizontal or both_vertical


def _weak_overlap_compatible(first: dict, second: dict, overlap: float) -> bool:
    """Reject low-overlap crossings while retaining confident duplicate masks."""
    if overlap >= 0.5 or first["class_key"] != "crack":
        return True
    if _same_dominant_axis(first["bbox_xyxy"], second["bbox_xyxy"]):
        return True

    first_box = first["bbox_xyxy"]
    second_box = second["bbox_xyxy"]
    first_ratio = max(1.0, first_box[2] - first_box[0]) / max(
        1.0,
        first_box[3] - first_box[1],
    )
    second_ratio = max(1.0, second_box[2] - second_box[0]) / max(
        1.0,
        second_box[3] - second_box[1],
    )
    return abs(float(np.log(first_ratio / second_ratio))) < 0.35


def _nearest_mask_points(
    first: dict,
    second: dict,
    max_gap: float,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Find the closest pair of mask pixels when their gap is at most ``max_gap``."""
    first_mask = first.get("mask")
    second_mask = second.get("mask")
    first_box = first.get("bbox_xyxy")
    second_box = second.get("bbox_xyxy")
    if first_mask is None or second_mask is None or not first_box or not second_box:
        return None
    if _bbox_gap(first_box, second_box) > max_gap:
        return None

    height, width = first_mask.shape[:2]
    x1 = max(0, int(np.floor(min(first_box[0], second_box[0]))))
    y1 = max(0, int(np.floor(min(first_box[1], second_box[1]))))
    x2 = min(width, int(np.ceil(max(first_box[2], second_box[2]))))
    y2 = min(height, int(np.ceil(max(first_box[3], second_box[3]))))
    if x1 >= x2 or y1 >= y2:
        return None

    first_roi = first_mask[y1:y2, x1:x2] > 0
    second_roi = second_mask[y1:y2, x1:x2] > 0
    if not first_roi.any() or not second_roi.any():
        return None

    distance = cv2.distanceTransform(
        (~first_roi).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    masked_distance = np.where(second_roi, distance, np.inf)
    sy, sx = np.unravel_index(int(np.argmin(masked_distance)), masked_distance.shape)
    minimum = float(masked_distance[sy, sx])
    if minimum > max_gap:
        return None

    radius = max(2, int(np.ceil(max_gap)) + 2)
    local_x1 = max(0, sx - radius)
    local_y1 = max(0, sy - radius)
    local_x2 = min(first_roi.shape[1], sx + radius + 1)
    local_y2 = min(first_roi.shape[0], sy + radius + 1)
    ys, xs = np.nonzero(first_roi[local_y1:local_y2, local_x1:local_x2])
    if xs.size == 0:
        return None
    ys = ys + local_y1
    xs = xs + local_x1
    nearest = int(np.argmin((xs - sx) ** 2 + (ys - sy) ** 2))
    return (int(xs[nearest] + x1), int(ys[nearest] + y1)), (int(sx + x1), int(sy + y1))


def _keep_large_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if min_area_px <= 1 or not binary.any():
        return binary
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    result = np.zeros_like(binary)
    for component_id in range(1, count):
        if int(stats[component_id, cv2.CC_STAT_AREA]) >= min_area_px:
            result[labels == component_id] = 1
    return result


def _bridge_mask_components(
    mask: np.ndarray,
    bridge_px: float,
    min_component_area_px: int = 50,
) -> tuple[np.ndarray, int]:
    """Connect aligned components that entered the same grouped prediction."""
    if bridge_px <= 0 or not mask.any():
        return mask, 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )
    components: list[dict] = []
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_component_area_px:
            continue
        x, y, width, height = [int(value) for value in stats[component_id, :4]]
        components.append(
            {
                "mask": (labels == component_id).astype(np.uint8),
                "bbox_xyxy": [float(x), float(y), float(x + width), float(y + height)],
            }
        )

    result = mask.copy()
    bridge_count = 0
    for first_index, first in enumerate(components):
        for second_index in range(first_index + 1, len(components)):
            second = components[second_index]
            if not _same_dominant_axis(first["bbox_xyxy"], second["bbox_xyxy"]):
                continue
            points = _nearest_mask_points(first, second, bridge_px)
            if points is None:
                continue
            first_point, second_point = points
            gap = float(np.hypot(
                first_point[0] - second_point[0],
                first_point[1] - second_point[1],
            ))
            if gap > 1.5:
                cv2.line(result, first_point, second_point, 1, thickness=2)
                bridge_count += 1
    return result, bridge_count


def _nms_records(records: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Fallback de-duplication for detection models that do not return masks."""
    kept: list[dict] = []
    for record in sorted(records, key=lambda item: float(item.get("confidence", 0)), reverse=True):
        if any(
            record.get("class_key") == other.get("class_key")
            and _box_iou(record["bbox_xyxy"], other["bbox_xyxy"]) >= iou_threshold
            for other in kept
        ):
            continue
        cleaned = dict(record)
        cleaned.pop("mask", None)
        kept.append(cleaned)
    return kept


def stitch_mask_records(
    records: list[dict],
    min_overlap: float = 0.03,
    min_area_px: int = 12,
    bridge_px: float = 24,
    bridge_min_confidence: float = 0.10,
) -> list[dict]:
    """Merge duplicate/split tile masks into one mask per physical defect.

    Masks are merged when they overlap or, for moderately confident cracks,
    when two collinear thin fragments have a small pixel gap. The latter
    handles cracks split by tile boundaries where the model ends each local
    prediction just before the seam.
    """
    if not records:
        return []
    if any(record.get("mask") is None for record in records):
        return _nms_records(records)

    ordered = sorted(
        records,
        key=lambda item: float(item.get("confidence", 0)),
        reverse=True,
    )
    parents = list(range(len(ordered)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parents[second_root] = first_root

    bridges: list[tuple[int, int, tuple[tuple[int, int], tuple[int, int]]]] = []
    for first_index, first in enumerate(ordered):
        for second_index in range(first_index + 1, len(ordered)):
            second = ordered[second_index]
            if first["class_key"] != second["class_key"]:
                continue

            overlap = _mask_overlap_ratio(first, second)
            if overlap >= min_overlap and _weak_overlap_compatible(first, second, overlap):
                union(first_index, second_index)
                continue

            if (
                bridge_px <= 0
                or first["class_key"] != "crack"
                or min(
                    float(first.get("confidence", 0.0)),
                    float(second.get("confidence", 0.0)),
                )
                < bridge_min_confidence
                or not _same_dominant_axis(first["bbox_xyxy"], second["bbox_xyxy"])
            ):
                continue

            points = _nearest_mask_points(first, second, bridge_px)
            if points is not None:
                union(first_index, second_index)
                bridges.append((first_index, second_index, points))

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(ordered)):
        members_by_root[find(index)].append(index)

    stitched: list[dict] = []
    for member_indices in members_by_root.values():
        members = [ordered[index] for index in member_indices]
        mask = np.zeros_like(members[0]["mask"], dtype=np.uint8)
        for member in members:
            mask[member["mask"] > 0] = 1

        member_set = set(member_indices)
        bridge_count = 0
        for first_index, second_index, (first_point, second_point) in bridges:
            if first_index in member_set and second_index in member_set:
                gap = float(np.hypot(
                    first_point[0] - second_point[0],
                    first_point[1] - second_point[1],
                ))
                if gap > 1.5:
                    cv2.line(mask, first_point, second_point, 1, thickness=2)
                    bridge_count += 1

        mask = _keep_large_components(mask, min_area_px)
        if members[0]["class_key"] == "crack":
            mask, component_bridge_count = _bridge_mask_components(mask, bridge_px)
            bridge_count += component_bridge_count
        bbox = _bbox_from_mask(mask)
        if bbox is None:
            continue
        confidence = max(float(member.get("confidence", 0.0)) for member in members)
        stitched.append(
            {
                "class_key": members[0]["class_key"],
                "confidence": round(confidence, 4),
                "bbox_xyxy": [round(value, 2) for value in bbox],
                "mask": mask,
                "mask_area_px": int(np.count_nonzero(mask)),
                "source_count": len(members),
                "bridge_count": bridge_count,
            }
        )
    stitched.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return stitched


def public_records(records: list[dict]) -> list[dict]:
    """Remove in-memory masks before returning JSON/HTTP responses."""
    output = []
    for record in records:
        item = dict(record)
        item.pop("mask", None)
        output.append(item)
    return output


def predict_tiled(
    model,
    image: np.ndarray,
    tile: int = 1024,
    overlap: int = 128,
    imgsz: int = 1024,
    conf: float = 0.25,
    iou: float = 0.5,
    include_masks: bool = False,
    stitch: bool = True,
    merge_overlap: float = 0.03,
    merge_bridge_px: float = 24,
    merge_bridge_min_confidence: float = 0.10,
    min_area_px: int = 12,
    masks_dir: Path | None = None,
    image_name: str = "image",
    scale: float | None = None,
    device: str | None = None,
    half: bool = False,
) -> list[dict]:
    """Run overlapping tiles, merge duplicate masks and map them to image coordinates.

    ``scale`` is millimeters per original-image pixel. When provided, each
    stitched mask is quantified and receives ``length_mm`` and width/area
    fields. If ``masks_dir`` is supplied, the stitched masks are also written
    as ``*_NNN.png`` files and ``mask_path`` is added to each record.
    """
    raw_records: list[dict] = []
    for x, y, crop in tile_crops(image, tile, overlap):
        predict_args = {
            "source": crop,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "retina_masks": True,
            "verbose": False,
        }
        if device is not None:
            predict_args["device"] = device
        if half:
            predict_args["half"] = True
        results = model.predict(**predict_args)
        result = results[0]
        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        crop_height, crop_width = crop.shape[:2]

        for index, box in enumerate(boxes):
            mask = None
            if masks is not None and index < len(masks):
                tile_mask = np.asarray(masks[index])
                if tile_mask.ndim == 3:
                    tile_mask = tile_mask[0]
                if tile_mask.shape[:2] != (crop_height, crop_width):
                    tile_mask = cv2.resize(
                        tile_mask.astype(np.float32),
                        (crop_width, crop_height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask[y : y + crop_height, x : x + crop_width] = (
                    tile_mask > 0.5
                ).astype(np.uint8)

            mask_bbox = _bbox_from_mask(mask) if mask is not None else None
            bbox = mask_bbox or [
                float(box[0] + x),
                float(box[1] + y),
                float(box[2] + x),
                float(box[3] + y),
            ]
            raw_records.append(
                {
                    "class_key": _class_name(model, int(classes[index])),
                    "confidence": float(confidences[index]),
                    "bbox_xyxy": [round(float(value), 2) for value in bbox],
                    "mask": mask,
                    "mask_area_px": int(np.count_nonzero(mask)) if mask is not None else None,
                    "tile_xy": [int(x), int(y)],
                }
            )

    records = (
        stitch_mask_records(
            raw_records,
            min_overlap=merge_overlap,
            min_area_px=min_area_px,
            bridge_px=merge_bridge_px,
            bridge_min_confidence=merge_bridge_min_confidence,
        )
        if stitch
        else raw_records
    )
    if not stitch and records and records[0].get("mask") is None:
        records = _nms_records(records, iou_threshold=0.7)

    if masks_dir is not None:
        masks_dir.mkdir(parents=True, exist_ok=True)

    for instance_id, record in enumerate(records, start=1):
        record["instance_id"] = instance_id
        mask = record.get("mask")
        if mask is not None:
            if scale is not None:
                from ..quantification_v1 import QuantificationConfigV1, quantify_instance

                quant_mask, skeleton, quant = quantify_instance(
                    mask,
                    str(record.get("class_key", "defect")),
                    instance_id,
                    QuantificationConfigV1(mm_per_pixel=scale, min_area_px=min_area_px),
                )
                mask = quant_mask
                record["mask"] = mask
                record.update(
                    {
                        "length_mm": round(quant.length_mm, 3),
                        "length_px": round(quant.length_px, 2),
                        "width_mm_p95": round(quant.width_mm_p95, 3),
                        "width_mm_max": round(quant.width_mm_max, 3),
                        "area_mm2": round(quant.area_mm2, 3),
                        "skeleton_px": quant.skeleton_px,
                        "connected_components": quant.connected_components,
                        "endpoints": quant.endpoints,
                        "branch_points": quant.branch_points,
                        "quantification_version": "V1",
                        "scale_mm_per_pixel": scale,
                    }
                )
            if masks_dir is not None:
                mask_path = masks_dir / f"{Path(image_name).stem}_{instance_id:03d}.png"
                cv2.imwrite(str(mask_path), (mask > 0).astype(np.uint8) * 255)
                record["mask_path"] = str(mask_path)
                if scale is not None:
                    skeleton_path = masks_dir / f"{Path(image_name).stem}_{instance_id:03d}_skeleton.png"
                    cv2.imwrite(str(skeleton_path), skeleton * 255)
                    record["skeleton_path"] = str(skeleton_path)
        if not include_masks:
            record.pop("mask", None)
    return records


def benchmark(
    model,
    image: np.ndarray,
    tile: int = 1024,
    overlap: int = 128,
    imgsz: int = 1024,
    device: str | None = None,
) -> dict:
    crops = list(tile_crops(image, tile, overlap))
    start = time.perf_counter()
    for _, _, crop in crops:
        predict_args = {
            "source": crop,
            "imgsz": imgsz,
            "conf": 0.25,
            "retina_masks": False,
            "verbose": False,
        }
        if device is not None:
            predict_args["device"] = device
        model.predict(**predict_args)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "tiles": len(crops),
        "total_ms": round(elapsed_ms, 2),
        "per_tile_ms": round(elapsed_ms / len(crops), 2) if crops else 0.0,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sliding-window inference and mask stitching.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=None, help="Millimeters per original pixel.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--output", default="outputs/tile_inference")
    parser.add_argument("--no-stitch", action="store_true")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(args.model)
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    output = Path(args.output)
    masks_dir = output / "masks"
    latency = benchmark(model, image, args.tile, args.overlap, args.imgsz, args.device)
    records = predict_tiled(
        model,
        image,
        tile=args.tile,
        overlap=args.overlap,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        stitch=not args.no_stitch,
        masks_dir=masks_dir,
        image_name=Path(args.image).name,
        scale=args.scale,
        device=args.device,
        half=args.half,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.json").write_text(
        json.dumps(public_records(records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "latency": latency,
                "detections": public_records(records),
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
