from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class SchemaError(ValueError):
    """Raised when an output object violates the A17 result contract."""


@dataclass(frozen=True)
class Timing:
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    quantification_ms: float = 0.0
    total_ms: float = 0.0

    def validate(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise SchemaError("Timing values must be non-negative")


@dataclass(frozen=True)
class DefectInstance:
    instance_id: int
    class_key: str
    class_name_zh: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    mask_path: str
    length_mm: float | None
    width_mm: float | None
    width_statistic: str
    scale_mm_per_pixel: float | None
    quantification_valid: bool

    def validate(self, allowed_classes: set[str]) -> None:
        if self.instance_id <= 0:
            raise SchemaError("instance_id must be positive")
        if self.class_key not in allowed_classes:
            raise SchemaError(f"Unknown class_key: {self.class_key}")
        if not 0.0 <= self.confidence <= 1.0:
            raise SchemaError("confidence must be in [0, 1]")
        if len(self.bbox_xyxy) != 4:
            raise SchemaError("bbox_xyxy must contain four values")
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 < x1 or y2 < y1:
            raise SchemaError("bbox_xyxy must be ordered as x1, y1, x2, y2")
        if self.width_statistic not in {"p95", "median", "max", "unknown"}:
            raise SchemaError("width_statistic must be p95, median, max, or unknown")
        if self.quantification_valid and (self.length_mm is None or self.width_mm is None):
            raise SchemaError("valid quantification requires length_mm and width_mm")
        if self.quantification_valid and self.scale_mm_per_pixel is None:
            raise SchemaError("valid quantification requires scale_mm_per_pixel")


@dataclass
class ImageResult:
    image_id: str
    image_width: int
    image_height: int
    model_version: str
    config_version: str
    timing_ms: Timing
    defects: list[DefectInstance] = field(default_factory=list)
    schema_version: str = "0.1.0"

    def validate(self, allowed_classes: set[str]) -> None:
        if not self.image_id:
            raise SchemaError("image_id is required")
        if self.image_width <= 0 or self.image_height <= 0:
            raise SchemaError("image dimensions must be positive")
        self.timing_ms.validate()
        instance_ids = [item.instance_id for item in self.defects]
        if len(instance_ids) != len(set(instance_ids)):
            raise SchemaError("instance_id values must be unique within an image")
        for item in self.defects:
            item.validate(allowed_classes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
