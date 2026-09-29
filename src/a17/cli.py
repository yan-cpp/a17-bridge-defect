from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ConfigError, load_config, project_paths, validate_config
from .environment import environment_report, missing_required_ml_packages, report_as_json
from .schemas import DefectInstance, ImageResult, SchemaError, Timing


def command_validate_config(args: argparse.Namespace) -> int:
    bundle = load_config(Path(args.root) if args.root else None)
    warnings = validate_config(bundle)
    payload = {
        "ok": True,
        "root": str(bundle.root),
        "phase1_classes": bundle.class_map["phase1_classes"],
        "warnings": warnings,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    bundle = load_config(Path(args.root) if args.root else None)
    warnings = validate_config(bundle)
    report = environment_report({"ok": True, "warnings": warnings})
    print(report_as_json(report))
    missing = missing_required_ml_packages()
    if missing:
        print(f"\nMissing ML packages: {', '.join(missing)}")
    if args.strict and missing:
        return 2
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    bundle = load_config(Path(args.root) if args.root else None)
    validate_config(bundle)
    paths = project_paths(bundle.root)
    paths.outputs.mkdir(parents=True, exist_ok=True)

    result = ImageResult(
        image_id="smoke-test.jpg",
        image_width=3840,
        image_height=2160,
        model_version="baseline-v0",
        config_version=str(bundle.runtime["version"]),
        timing_ms=Timing(
            preprocess_ms=12.0,
            inference_ms=145.0,
            postprocess_ms=18.0,
            quantification_ms=9.0,
            total_ms=184.0,
        ),
        defects=[
            DefectInstance(
                instance_id=1,
                class_key="crack",
                class_name_zh="混凝土裂缝",
                confidence=0.93,
                bbox_xyxy=(120.0, 340.0, 580.0, 410.0),
                mask_path="masks/smoke-test_001.png",
                length_mm=182.4,
                width_mm=1.7,
                width_statistic="p95",
                scale_mm_per_pixel=0.42,
                quantification_valid=True,
            )
        ],
    )
    result.validate(set(bundle.class_map["phase1_classes"]))
    output = paths.outputs / "smoke_result.json"
    output.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Smoke test passed: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="a17", description="A17 bridge defect project CLI")
    parser.add_argument("--root", help="Project root. Defaults to auto-discovery.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-config", help="Validate project JSON configuration")
    validate_parser.set_defaults(func=command_validate_config)

    doctor_parser = subparsers.add_parser("doctor", help="Report Python, dependency, GPU and config status")
    doctor_parser.add_argument("--strict", action="store_true", help="Fail when model dependencies are missing")
    doctor_parser.set_defaults(func=command_doctor)

    smoke_parser = subparsers.add_parser("smoke", help="Create a schema-valid smoke result")
    smoke_parser.set_defaults(func=command_smoke)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ConfigError, SchemaError) as exc:
        parser.exit(status=2, message=f"error: {exc}\n")
    return 0
