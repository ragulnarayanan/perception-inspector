from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from perception_inspector.matching import iou, match_detections
from perception_inspector.models import (
    Detection,
    DetectionStats,
    ImageRecord,
    InspectionResult,
    Priority,
)
from perception_inspector.scene import compute_scene_metrics, load_image


@dataclass(frozen=True)
class InspectorConfig:
    low_confidence_threshold: float = 0.25
    iou_threshold: float = 0.5
    poor_localization_iou: float = 0.35
    tiny_object_area_ratio: float = 0.0025
    abnormal_aspect_ratio: float = 4.0
    low_brightness_threshold: float = 45.0
    low_contrast_threshold: float = 25.0
    high_blur_threshold: float = 0.65
    duplicate_iou_threshold: float = 0.8
    vlm_score_threshold: float = 55.0


class PerceptionInspector:
    def __init__(self, config: InspectorConfig | None = None) -> None:
        self.config = config or InspectorConfig()

    def inspect(self, record: ImageRecord) -> InspectionResult:
        image = load_image(record.filepath)
        height, width = image.shape[:2]
        scene_metrics = compute_scene_metrics(image)
        detection_stats = self._detection_stats(record.predictions)
        has_ground_truth = bool(record.ground_truth)
        if has_ground_truth:
            matches, false_positives, false_negatives = match_detections(
                record.predictions,
                record.ground_truth,
                self.config.iou_threshold,
            )
        else:
            matches = []
            false_positives = []
            false_negatives = []

        flags: list[str] = []
        flags.extend(
            self._prediction_quality_flags(
                record,
                matches,
                false_positives,
                false_negatives,
                has_ground_truth=has_ground_truth,
            )
        )
        flags.extend(self._bbox_flags(record.predictions, width, height))
        flags.extend(self._scene_flags(scene_metrics))
        flags.extend(self._duplicate_flags(record.predictions))

        score = self._score(
            flags=flags,
            predictions=record.predictions,
            scene_metrics=scene_metrics,
            false_positive_count=len(false_positives),
            false_negative_count=len(false_negatives),
        )

        return InspectionResult(
            image_id=record.image_id,
            filepath=str(record.filepath),
            failure_score=score,
            priority=self._priority(score),
            flags=sorted(set(flags)),
            scene_metrics=scene_metrics,
            detection_stats=detection_stats,
            matches=matches,
            false_positive_indices=false_positives,
            false_negative_indices=false_negatives,
            metadata={
                "image_width": width,
                "image_height": height,
                "mode": "offline_validation" if has_ground_truth else "online_inspection",
                "has_ground_truth": has_ground_truth,
            },
        )

    def _prediction_quality_flags(
        self,
        record: ImageRecord,
        matches: list,
        false_positives: list[int],
        false_negatives: list[int],
        has_ground_truth: bool,
    ) -> list[str]:
        flags: list[str] = []
        if any((prediction.confidence or 0) < self.config.low_confidence_threshold for prediction in record.predictions):
            flags.append("Low Confidence")
        if not record.predictions:
            flags.append("No Detections")
        if has_ground_truth:
            if false_positives:
                flags.append("False Positive")
            if false_negatives:
                flags.append("False Negative")
            if record.ground_truth and not record.predictions:
                flags.append("Missing Detections")
            if any(not match.class_match for match in matches):
                flags.append("Misclassification")
            for match in matches:
                if match.iou < self.config.poor_localization_iou:
                    flags.append("Poor Localization")
        return flags

    def _bbox_flags(self, predictions: list[Detection], width: int, height: int) -> list[str]:
        flags: list[str] = []
        image_area = max(width * height, 1)
        for prediction in predictions:
            box = prediction.bbox
            if box.x1 < 0 or box.y1 < 0 or box.x2 > width or box.y2 > height:
                flags.append("Box Outside Image")
            if box.area / image_area < self.config.tiny_object_area_ratio:
                flags.append("Small Object")
            ratio = box.aspect_ratio
            if ratio > self.config.abnormal_aspect_ratio or ratio < 1 / self.config.abnormal_aspect_ratio:
                flags.append("Abnormal Aspect Ratio")
        return flags

    def _scene_flags(self, metrics) -> list[str]:
        flags: list[str] = []
        if metrics.brightness < self.config.low_brightness_threshold:
            flags.append("Low Lighting")
        if metrics.contrast < self.config.low_contrast_threshold:
            flags.append("Low Contrast")
        if metrics.blur_score > self.config.high_blur_threshold:
            flags.append("Motion Blur")
        return flags

    def _duplicate_flags(self, predictions: list[Detection]) -> list[str]:
        for left_index, left in enumerate(predictions):
            for right in predictions[left_index + 1 :]:
                if left.class_name == right.class_name and iou(left.bbox, right.bbox) >= self.config.duplicate_iou_threshold:
                    return ["Duplicate Detection"]
        return []

    def _detection_stats(self, predictions: list[Detection]) -> DetectionStats:
        confidences = [prediction.confidence for prediction in predictions if prediction.confidence is not None]
        return DetectionStats(
            detection_count=len(predictions),
            mean_confidence=float(np.mean(confidences)) if confidences else None,
            confidence_variance=float(np.var(confidences)) if confidences else None,
            class_frequency=dict(Counter(prediction.class_name for prediction in predictions)),
        )

    def _score(
        self,
        flags: list[str],
        predictions: list[Detection],
        scene_metrics,
        false_positive_count: int,
        false_negative_count: int,
    ) -> float:
        weights = {
            "False Negative": 22,
            "Missing Detections": 28,
            "No Detections": 12,
            "False Positive": 12,
            "Misclassification": 20,
            "Poor Localization": 16,
            "Low Confidence": 12,
            "Duplicate Detection": 8,
            "Small Object": 6,
            "Abnormal Aspect Ratio": 7,
            "Box Outside Image": 15,
            "Low Lighting": 8,
            "Low Contrast": 6,
            "Motion Blur": 8,
        }
        score = sum(weights.get(flag, 5) for flag in set(flags))
        score += min(false_positive_count * 4, 12)
        score += min(false_negative_count * 8, 24)
        if predictions:
            lowest_confidence = min(prediction.confidence or 0 for prediction in predictions)
            score += max(0.0, (self.config.low_confidence_threshold - lowest_confidence) * 35)
        score += max(0.0, (self.config.low_brightness_threshold - scene_metrics.brightness) / 5)
        return round(min(score, 100.0), 2)

    def _priority(self, score: float) -> Priority:
        if score >= 75:
            return Priority.critical
        if score >= self.config.vlm_score_threshold:
            return Priority.high
        if score >= 30:
            return Priority.medium
        return Priority.low
