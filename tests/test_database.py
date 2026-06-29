from pathlib import Path

import cv2
import numpy as np

from perception_inspector.database import FailureDatabase
from perception_inspector.inspector import PerceptionInspector
from perception_inspector.models import BoundingBox, Detection, ImageRecord
from perception_inspector.vlm import HeuristicVLMAnalyzer


def test_database_persists_inspection_and_vlm_result(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    cv2.imwrite(str(image_path), np.zeros((80, 80, 3), dtype=np.uint8))
    record = ImageRecord(
        image_id="db-scene",
        filepath=image_path,
        predictions=[
            Detection(class_name="car", confidence=0.1, bbox=BoundingBox(x1=4, y1=4, x2=12, y2=12))
        ],
        ground_truth=[
            Detection(class_name="person", bbox=BoundingBox(x1=50, y1=50, x2=70, y2=70))
        ],
    )
    result = PerceptionInspector().inspect(record)
    vlm_result = HeuristicVLMAnalyzer().analyze(record, result)

    database = FailureDatabase(tmp_path / "failures.db")
    database.initialize()
    database.upsert_record(record, result)
    database.upsert_vlm_result(vlm_result)

    rows = database.list_failures()
    assert len(rows) == 1
    assert rows[0]["image_id"] == "db-scene"
    assert rows[0]["primary_failure"]

    detections = database.list_detections("db-scene", source="prediction")
    assert len(detections) == 1
    assert detections[0]["class_name"] == "car"
