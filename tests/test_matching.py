from perception_inspector.matching import iou, match_detections
from perception_inspector.models import BoundingBox, Detection


def test_iou_for_overlapping_boxes() -> None:
    left = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    right = BoundingBox(x1=5, y1=5, x2=15, y2=15)

    assert round(iou(left, right), 3) == 0.143


def test_match_detections_tracks_false_positives_and_negatives() -> None:
    predictions = [
        Detection(class_name="car", confidence=0.9, bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10)),
        Detection(class_name="person", confidence=0.6, bbox=BoundingBox(x1=50, y1=50, x2=60, y2=60)),
    ]
    truths = [
        Detection(class_name="car", bbox=BoundingBox(x1=1, y1=1, x2=11, y2=11)),
        Detection(class_name="traffic light", bbox=BoundingBox(x1=80, y1=80, x2=90, y2=90)),
    ]

    matches, false_positives, false_negatives = match_detections(predictions, truths, 0.5)

    assert len(matches) == 1
    assert false_positives == [1]
    assert false_negatives == [1]
