"""Freeze the production segmentation model and its quantification parameters."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Freeze A17 model V2 and quantification manifest.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration", default="configs/calibration_v2.json")
    parser.add_argument("--quantification", default="configs/quantification_v2.json")
    parser.add_argument("--output-model", default="weights/a17_seg_v2_frozen.pt")
    parser.add_argument("--manifest", default="configs/model_v2_freeze.json")
    parser.add_argument("--onnx", default=None, help="Optional exported ONNX artifact to hash.")
    parser.add_argument("--engine", default=None, help="Optional TensorRT engine artifact to hash.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    source = Path(args.model)
    output_model = Path(args.output_model)
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    quantification = json.loads(Path(args.quantification).read_text(encoding="utf-8"))

    if output_model.exists() and not args.overwrite:
        raise SystemExit(f"Output model already exists: {output_model}")
    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output_model)

    manifest = {
        "version": "V2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(output_model),
            "sha256": sha256(output_model),
            "source": str(source),
        },
        "calibration": calibration,
        "quantification": quantification,
        "freeze": {
            "pt": True,
            "fp16_onnx": (
                {"path": args.onnx, "sha256": sha256(Path(args.onnx))}
                if args.onnx and Path(args.onnx).exists()
                else None
            ),
            "tensorrt": (
                {"path": args.engine, "sha256": sha256(Path(args.engine))}
                if args.engine and Path(args.engine).exists()
                else None
            ),
        },
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Saved to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
