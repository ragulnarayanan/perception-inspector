from __future__ import annotations

from abc import ABC, abstractmethod

from perception_inspector.models import ImageRecord, InspectionResult, Priority, VLMResult


class VLMAnalyzer(ABC):
    @abstractmethod
    def analyze(self, record: ImageRecord, inspection: InspectionResult) -> VLMResult:
        """Return structured failure intelligence for a flagged image."""


class HeuristicVLMAnalyzer(VLMAnalyzer):
    """Deterministic stand-in for local development and tests."""

    def analyze(self, record: ImageRecord, inspection: InspectionResult) -> VLMResult:
        primary = self._primary_failure(inspection.flags)
        scene_conditions = [
            flag
            for flag in inspection.flags
            if flag in {"Low Lighting", "Low Contrast", "Motion Blur", "Small Object"}
        ]
        secondary = [flag for flag in inspection.flags if flag != primary][:3]
        reasoning = (
            f"{record.image_id} was flagged for {', '.join(inspection.flags) or 'no rule violations'}. "
            f"The deterministic inspector assigned a failure score of {inspection.failure_score}."
        )
        recommendation = (
            "Engineer Review"
            if inspection.priority in {Priority.high, Priority.critical}
            else "Monitor"
        )
        return VLMResult(
            image_id=record.image_id,
            primary_failure=primary,
            secondary_failures=secondary,
            scene_conditions=scene_conditions,
            priority=inspection.priority,
            reasoning=reasoning,
            recommendation=recommendation,
        )

    def _primary_failure(self, flags: list[str]) -> str:
        priority_order = [
            "False Negative",
            "Missing Detections",
            "Misclassification",
            "Poor Localization",
            "Low Confidence",
            "Motion Blur",
            "Low Lighting",
            "Small Object",
        ]
        for candidate in priority_order:
            if candidate in flags:
                return candidate
        return flags[0] if flags else "Uncategorized"


class QwenVLAnalyzer(VLMAnalyzer):
    """Integration point for Qwen2.5-VL.

    This class intentionally avoids loading model weights at import time. Wire the model
    and processor in application code, then convert its response into VLMResult.
    """

    def __init__(self, model: object, processor: object) -> None:
        self.model = model
        self.processor = processor

    def analyze(self, record: ImageRecord, inspection: InspectionResult) -> VLMResult:
        raise NotImplementedError(
            "Qwen2.5-VL execution is environment-specific. Use HeuristicVLMAnalyzer "
            "locally or implement model inference here after installing vlm extras."
        )
