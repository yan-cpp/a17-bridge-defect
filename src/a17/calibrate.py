"""Pixel-to-physical scale calibration utilities.

The calibration is intentionally explicit: a user provides either a measured
reference distance in pixels and millimeters, or an image width in pixels and
the physical width in millimeters.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class CalibrationV2:
    version: str
    calibration_id: str
    mm_per_pixel: float
    pixels: float
    known_mm: float
    method: str
    source_image: str | None
    created_utc: str
    provisional: bool = False
    notes: str = ""


def mm_per_pixel_from_distance(pixels: float, known_mm: float) -> float:
    if pixels <= 0 or known_mm <= 0:
        raise ValueError("pixels and known_mm must be greater than zero.")
    return known_mm / pixels


def build_calibration(
    pixels: float,
    known_mm: float,
    method: str,
    calibration_id: str,
    source_image: str | None = None,
    provisional: bool = False,
    notes: str = "",
) -> CalibrationV2:
    return CalibrationV2(
        version="V2",
        calibration_id=calibration_id,
        mm_per_pixel=mm_per_pixel_from_distance(pixels, known_mm),
        pixels=pixels,
        known_mm=known_mm,
        method=method,
        source_image=source_image,
        created_utc=datetime.now(timezone.utc).isoformat(),
        provisional=provisional,
        notes=notes,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Create an A17 calibration V2 JSON file.")
    parser.add_argument("--pixels", type=float, help="Measured reference length in pixels.")
    parser.add_argument("--known-mm", type=float, help="Measured reference length in millimeters.")
    parser.add_argument("--image-width-px", type=float)
    parser.add_argument("--image-width-mm", type=float)
    parser.add_argument("--calibration-id", default="a17_calibration_v2")
    parser.add_argument("--source-image", default=None)
    parser.add_argument("--provisional", action="store_true")
    parser.add_argument("--notes", default="")
    parser.add_argument("--output", default="configs/calibration_v2.json")
    args = parser.parse_args(argv)

    if args.pixels is not None and args.known_mm is not None:
        calibration = build_calibration(
            args.pixels,
            args.known_mm,
            "reference_distance",
            args.calibration_id,
            args.source_image,
            args.provisional,
            args.notes,
        )
    elif args.image_width_px is not None and args.image_width_mm is not None:
        calibration = build_calibration(
            args.image_width_px,
            args.image_width_mm,
            "image_width",
            args.calibration_id,
            args.source_image,
            args.provisional,
            args.notes,
        )
    else:
        parser.error("provide --pixels/--known-mm or --image-width-px/--image-width-mm")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(calibration)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
