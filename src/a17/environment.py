from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any

PACKAGE_MODULES = {
    "numpy": "numpy",
    "cv2": "cv2",
    "PIL": "PIL",
    "pydantic": "pydantic",
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
    "skimage": "skimage",
    "albumentations": "albumentations",
    "onnxruntime": "onnxruntime",
}


def package_status() -> dict[str, bool]:
    return {name: find_spec(module) is not None for name, module in PACKAGE_MODULES.items()}


def gpu_status() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "reason": "nvidia-smi not found on PATH", "devices": []}
    command = [
        executable,
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": str(exc), "devices": []}
    if completed.returncode != 0:
        return {
            "available": False,
            "reason": completed.stderr.strip() or "nvidia-smi returned a non-zero exit code",
            "devices": [],
        }
    devices = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            devices.append({"name": parts[0], "memory_total": parts[1], "driver_version": parts[2]})
    return {"available": bool(devices), "reason": None, "devices": devices}


def environment_report(config_validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "cwd": str(Path.cwd()),
        "environment_variables": {
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "packages": package_status(),
        "gpu": gpu_status(),
        "config_validation": config_validation,
    }


def missing_required_ml_packages() -> list[str]:
    required = ("numpy", "cv2", "torch", "torchvision", "ultralytics")
    status = package_status()
    return [name for name in required if not status.get(name, False)]


def report_as_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2)
