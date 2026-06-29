from __future__ import annotations

from perception_inspector.models import BoundingBox, Detection, MatchedPair


def iou(a: BoundingBox, b: BoundingBox) -> float:
    x_left = max(a.x1, b.x1)
    y_top = max(a.y1, b.y1)
    x_right = min(a.x2, b.x2)
    y_bottom = min(a.y2, b.y2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(
    predictions: list[Detection],
    ground_truth: list[Detection],
    iou_threshold: float,
) -> tuple[list[MatchedPair], list[int], list[int]]:
    candidates: list[tuple[float, int, int]] = []
    for pred_idx, prediction in enumerate(predictions):
        for truth_idx, truth in enumerate(ground_truth):
            score = iou(prediction.bbox, truth.bbox)
            if score >= iou_threshold:
                candidates.append((score, pred_idx, truth_idx))

    candidates.sort(reverse=True)
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    matches: list[MatchedPair] = []

    for score, pred_idx, truth_idx in candidates:
        if pred_idx in used_predictions or truth_idx in used_truths:
            continue
        used_predictions.add(pred_idx)
        used_truths.add(truth_idx)
        matches.append(
            MatchedPair(
                prediction_index=pred_idx,
                truth_index=truth_idx,
                iou=score,
                class_match=predictions[pred_idx].class_name == ground_truth[truth_idx].class_name,
            )
        )

    false_positive_indices = [
        index for index in range(len(predictions)) if index not in used_predictions
    ]
    false_negative_indices = [
        index for index in range(len(ground_truth)) if index not in used_truths
    ]
    return matches, false_positive_indices, false_negative_indices
