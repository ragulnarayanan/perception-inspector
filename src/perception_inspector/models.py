from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Priority(str, Enum):
    low = "Low"
    medium = "Medium"
    high = "High"
    critical = "Critical"


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_xyxy(cls, values: list[float] | tuple[float, float, float, float]) -> "BoundingBox":
        if len(values) != 4:
            raise ValueError("Bounding boxes must contain [x1, y1, x2, y2].")
        return cls(x1=values[0], y1=values[1], x2=values[2], y2=values[3])

    @field_validator("x2")
    @classmethod
    def validate_x2(cls, value: float, info: Any) -> float:
        x1 = info.data.get("x1")
        if x1 is not None and value <= x1:
            raise ValueError("x2 must be greater than x1.")
        return value

    @field_validator("y2")
    @classmethod
    def validate_y2(cls, value: float, info: Any) -> float:
        y1 = info.data.get("y1")
        if y1 is not None and value <= y1:
            raise ValueError("y2 must be greater than y1.")
        return value

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1e-6)

    def to_xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


class Detection(BaseModel):
    class_name: str = Field(min_length=1)
    bbox: BoundingBox
    confidence: float | None = Field(default=None, ge=0, le=1)

    @classmethod
    def from_manifest(cls, payload: dict[str, Any], require_confidence: bool = False) -> "Detection":
        bbox = payload.get("bbox")
        if not isinstance(bbox, list):
            raise ValueError("Detection payload requires a bbox list.")
        confidence = payload.get("confidence")
        if require_confidence and confidence is None:
            raise ValueError("Prediction payload requires confidence.")
        return cls(
            class_name=payload["class_name"],
            confidence=confidence,
            bbox=BoundingBox.from_xyxy(bbox),
        )


class SceneMetrics(BaseModel):
    brightness: float
    contrast: float
    blur_score: float
    sharpness: float
    entropy: float


class DetectionStats(BaseModel):
    detection_count: int
    mean_confidence: float | None
    confidence_variance: float | None
    class_frequency: dict[str, int]


class MatchedPair(BaseModel):
    prediction_index: int
    truth_index: int
    iou: float
    class_match: bool


class InspectionResult(BaseModel):
    image_id: str
    filepath: str
    failure_score: float
    priority: Priority
    flags: list[str]
    scene_metrics: SceneMetrics
    detection_stats: DetectionStats
    matches: list[MatchedPair]
    false_positive_indices: list[int]
    false_negative_indices: list[int]
    failure_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def should_send_to_vlm(self) -> bool:
        return self.priority in {Priority.high, Priority.critical}


class VLMResult(BaseModel):
    image_id: str
    primary_failure: str
    secondary_failures: list[str]
    scene_conditions: list[str]
    priority: Priority
    reasoning: str
    recommendation: str


class ImageRecord(BaseModel):
    image_id: str
    filepath: Path
    predictions: list[Detection] = Field(default_factory=list)
    ground_truth: list[Detection] = Field(default_factory=list)

    @classmethod
    def from_manifest(cls, payload: dict[str, Any], base_dir: Path) -> "ImageRecord":
        filepath = Path(payload["filepath"])
        if not filepath.is_absolute():
            filepath = base_dir / filepath
        return cls(
            image_id=payload["image_id"],
            filepath=filepath,
            predictions=[
                Detection.from_manifest(item, require_confidence=True)
                for item in payload.get("predictions", [])
            ],
            ground_truth=[
                Detection.from_manifest(item)
                for item in payload.get("ground_truth", [])
            ],
        )
