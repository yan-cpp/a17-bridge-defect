from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when project configuration is missing or inconsistent."""


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    configs: Path
    docs: Path
    src: Path
    tests: Path
    data: Path
    outputs: Path
    weights: Path


@dataclass(frozen=True)
class ConfigBundle:
    root: Path
    class_map: dict[str, Any]
    metrics: dict[str, Any]
    dataset: dict[str, Any]
    runtime: dict[str, Any]


CONFIG_FILES = (
    "class_map.json",
    "metrics.json",
    "dataset.template.json",
    "runtime.json",
)


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "configs").is_dir():
            return candidate
    raise ConfigError("Could not find project root. Run the command inside a17-bridge-defect.")


def project_paths(root: Path | None = None) -> ProjectPaths:
    project_root = (root or find_project_root()).resolve()
    return ProjectPaths(
        root=project_root,
        configs=project_root / "configs",
        docs=project_root / "docs",
        src=project_root / "src",
        tests=project_root / "tests",
        data=project_root / "data",
        outputs=project_root / "outputs",
        weights=project_root / "weights",
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Missing configuration file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"Top-level configuration must be an object: {path}")
    return value


def load_config(root: Path | None = None) -> ConfigBundle:
    project_root = (root or find_project_root()).resolve()
    configs: dict[str, dict[str, Any]] = {}
    for filename in CONFIG_FILES:
        configs[filename] = load_json(project_root / "configs" / filename)
    return ConfigBundle(
        root=project_root,
        class_map=configs["class_map.json"],
        metrics=configs["metrics.json"],
        dataset=configs["dataset.template.json"],
        runtime=configs["runtime.json"],
    )


def _require_keys(mapping: dict[str, Any], keys: list[str], path: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"{path} missing keys: {', '.join(missing)}")


def validate_config(bundle: ConfigBundle) -> list[str]:
    warnings: list[str] = []

    _require_keys(bundle.class_map, ["version", "phase1_classes", "classes"], "class_map.json")
    _require_keys(
        bundle.metrics,
        ["phase1_classes", "detection", "segmentation", "quantification", "latency"],
        "metrics.json",
    )
    _require_keys(bundle.dataset, ["roots", "sources", "split_policy"], "dataset.template.json")
    _require_keys(bundle.runtime, ["python", "gpu", "preprocessing", "inference"], "runtime.json")

    classes = bundle.class_map["classes"]
    if not isinstance(classes, list) or not classes:
        raise ConfigError("class_map.json classes must be a non-empty list")

    keys: list[str] = []
    ids: list[int] = []
    for index, item in enumerate(classes):
        if not isinstance(item, dict):
            raise ConfigError(f"class_map.json classes[{index}] must be an object")
        _require_keys(
            item,
            ["id", "key", "display_name_zh", "phase", "source_aliases"],
            f"classes[{index}]",
        )
        keys.append(str(item["key"]))
        ids.append(int(item["id"]))

    if len(keys) != len(set(keys)):
        raise ConfigError("class_map.json contains duplicate class keys")
    if len(ids) != len(set(ids)):
        raise ConfigError("class_map.json contains duplicate class ids")
    if sorted(ids) != list(range(len(ids))):
        warnings.append("Class ids are not contiguous from 0; some model tooling assumes contiguous ids.")

    phase1 = list(bundle.class_map["phase1_classes"])
    if len(phase1) < 3:
        raise ConfigError("A17 phase 1 must contain at least three classes")
    unknown = sorted(set(phase1) - set(keys))
    if unknown:
        raise ConfigError(f"phase1_classes contains unknown classes: {unknown}")
    if list(bundle.metrics["phase1_classes"]) != phase1:
        raise ConfigError("metrics.json phase1_classes must match class_map.json")

    source_names = [str(source.get("name")) for source in bundle.dataset["sources"]]
    if len(source_names) != len(set(source_names)):
        raise ConfigError("dataset.template.json contains duplicate source names")
    if any(source.get("download_status") != "pending" for source in bundle.dataset["sources"]):
        warnings.append("Dataset download_status has advanced beyond the D1/D2 baseline.")

    ratios = bundle.dataset["split_policy"].get("ratios", {})
    if not ratios or abs(sum(float(value) for value in ratios.values()) - 1.0) > 1e-6:
        raise ConfigError("dataset split ratios must sum to 1.0")

    preprocessing = bundle.runtime["preprocessing"]
    if int(preprocessing["tile_overlap"]) >= int(preprocessing["tile_size"]):
        raise ConfigError("tile_overlap must be smaller than tile_size")

    return warnings
