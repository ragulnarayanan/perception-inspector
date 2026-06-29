from __future__ import annotations

import json
from pathlib import Path

from perception_inspector.models import ImageRecord


def load_manifest(path: Path) -> list[ImageRecord]:
    payload = json.loads(path.read_text())
    base_dir = path.parent
    images = payload.get("images")
    if not isinstance(images, list):
        raise ValueError("Manifest must contain an images list.")
    return [ImageRecord.from_manifest(item, base_dir=base_dir) for item in images]
