from __future__ import annotations

from pathlib import Path

from perception_inspector.models import BoundingBox, Detection


class YoloDetector:
    def __init__(self, model_name: str = "yolo11n.pt") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install detector extras with: pip install -e '.[detector]'") from exc
        self.model = YOLO(model_name)

    def predict(self, image_path: Path | str) -> list[Detection]:
        results = self.model(str(image_path), verbose=False)
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                class_id = int(box.cls[0].item())
                detections.append(
                    Detection(
                        class_name=str(names[class_id]),
                        confidence=float(box.conf[0].item()),
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )
        return detections
