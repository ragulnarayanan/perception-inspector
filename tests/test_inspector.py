from pathlib import Path

import cv2
import numpy as np

from perception_inspector.inspector import PerceptionInspector
from perception_inspector.models import BoundingBox, Detection, ImageRecord, Priority


def make_dark_blurry_image(path: Path) -> None:
    image = np.full((100, 160, 3), 24, dtype=np.uint8)
    cv2.rectangle(image, (42, 28), (58, 68), (35, 35, 35), -1)
    cv2.GaussianBlur(image, (11, 11), 0, dst=image)
    cv2.imwrite(str(path), image)


def test_inspector_flags_low_confidence_duplicate_and_false_negative(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    make_dark_blurry_image(image_path)
    record = ImageRecord(
        image_id="scene-1",
        filepath=image_path,
        predictions=[
            Detection(class_name="person", confidence=0.18, bbox=BoundingBox(x1=44, y1=30, x2=54, y2=62)),
            Detection(class_name="person", confidence=0.21, bbox=BoundingBox(x1=44, y1=30, x2=54, y2=62)),
        ],
        ground_truth=[
            Detection(class_name="person", bbox=BoundingBox(x1=42, y1=28, x2=58, y2=68)),
            Detection(class_name="car", bbox=BoundingBox(x1=100, y1=40, x2=140, y2=70)),
        ],
    )

    result = PerceptionInspector().inspect(record)

    assert "Low Confidence" in result.flags
    assert "Duplicate Detection" in result.flags
    assert "False Negative" in result.flags
    assert "Low Lighting" in result.flags
    assert result.priority in {Priority.high, Priority.critical}


def test_online_mode_does_not_create_label_dependent_failures(tmp_path: Path) -> None:
    image_path = tmp_path / "online.png"
    make_dark_blurry_image(image_path)
    record = ImageRecord(
        image_id="online-scene",
        filepath=image_path,
        predictions=[
            Detection(class_name="car", confidence=0.9, bbox=BoundingBox(x1=20, y1=20, x2=60, y2=60))
        ],
        ground_truth=[],
    )

    result = PerceptionInspector().inspect(record)

    assert result.metadata["mode"] == "online_inspection"
    assert "False Positive" not in result.flags
    assert "False Negative" not in result.flags
    assert "Misclassification" not in result.flags


def test_online_mode_can_flag_no_detections_without_labels(tmp_path: Path) -> None:
    image_path = tmp_path / "empty.png"
    make_dark_blurry_image(image_path)
    record = ImageRecord(
        image_id="empty-online-scene",
        filepath=image_path,
        predictions=[],
        ground_truth=[],
    )

    result = PerceptionInspector().inspect(record)

    assert result.metadata["mode"] == "online_inspection"
    assert "No Detections" in result.flags
    assert "Missing Detections" not in result.flags
