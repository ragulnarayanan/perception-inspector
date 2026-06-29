from __future__ import annotations

import json
from pathlib import Path

from perception_inspector.models import BoundingBox, Detection, ImageRecord


CATEGORY_ALIASES = {
    "pedestrian": "person",
    "bike": "bicycle",
    "motor": "motorcycle",
}


def normalize_category(category: str) -> str:
    normalized = category.strip().lower()
    return CATEGORY_ALIASES.get(normalized, normalized)


def load_labeled_image_records(
    labels_path: Path,
    images_dir: Path,
    predictions_by_image: dict[str, list[Detection]] | None = None,
    limit: int | None = None,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    predictions_by_image = predictions_by_image or {}
    label_items = _load_label_items(labels_path)

    for item in label_items:
        image_name = item.get("name") or item.get("image") or item.get("file_name")
        if not image_name:
            continue
        image_path = resolve_image_path(images_dir, image_name)
        if image_path is None:
            continue

        image_id = Path(image_name).stem
        ground_truth = []
        labels_for_image = item.get("labels", []) or item.get("annotations", [])
        for label in labels_for_image:
            box = label.get("box2d") or label.get("bbox")
            category = label.get("category") or label.get("class_name") or label.get("label")
            if not box or not category:
                continue
            ground_truth.append(_detection_from_label(category, box))

        records.append(
            ImageRecord(
                image_id=image_id,
                filepath=image_path,
                predictions=predictions_by_image.get(image_id, []),
                ground_truth=ground_truth,
            )
        )
        if limit is not None and len(records) >= limit:
            break

    return records


def load_labeled_image_records_from_payload(
    labels_payload: str,
    image_files: list[tuple[str, bytes]],
    predictions_by_image: dict[str, list[Detection]] | None = None,
    limit: int | None = None,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    predictions_by_image = predictions_by_image or {}
    label_items = _parse_label_items(labels_payload)
    images_dir = Path("/tmp/perception-inspector-upload")
    images_dir.mkdir(parents=True, exist_ok=True)
    for file_name, file_bytes in image_files:
        (images_dir / file_name).write_bytes(file_bytes)

    for item in label_items:
        image_name = item.get("name") or item.get("image") or item.get("file_name")
        if not image_name:
            continue
        image_path = resolve_image_path(images_dir, image_name)
        if image_path is None:
            continue

        image_id = Path(image_name).stem
        ground_truth = []
        labels_for_image = item.get("labels", []) or item.get("annotations", [])
        for label in labels_for_image:
            box = label.get("box2d") or label.get("bbox")
            category = label.get("category") or label.get("class_name") or label.get("label")
            if not box or not category:
                continue
            ground_truth.append(_detection_from_label(category, box))

        records.append(
            ImageRecord(
                image_id=image_id,
                filepath=image_path,
                predictions=predictions_by_image.get(image_id, []),
                ground_truth=ground_truth,
            )
        )
        if limit is not None and len(records) >= limit:
            break

    return records


def _parse_label_items(labels_payload: str) -> list[dict]:
    raw = labels_payload.strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return _parse_text_label_items(raw)
    return _normalize_label_payload(payload)


def _parse_text_label_items(raw: str) -> list[dict]:
    items: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        image_name = parts[0]
        category = parts[1]
        x1 = float(parts[2])
        y1 = float(parts[3])
        x2 = float(parts[4]) if len(parts) > 4 else x1 + 1.0
        y2 = float(parts[5]) if len(parts) > 5 else y1 + 1.0
        items.append(
            {
                "name": image_name,
                "labels": [
                    {
                        "category": category,
                        "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    }
                ],
            }
        )
    return items


def _load_label_items(labels_path: Path) -> list[dict]:
    if labels_path.is_dir():
        items = []
        for path in sorted(labels_path.glob("*.json")):
            payload = json.loads(path.read_text())
            items.extend(_normalize_label_payload(payload, fallback_image_name=path.with_suffix(".jpg").name))
        return items

    payload = json.loads(labels_path.read_text())
    return _normalize_label_payload(payload)


def _normalize_label_payload(payload, fallback_image_name: str | None = None) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("Labels JSON should be a record, a list of records, or a directory of JSON records.")

    if "images" in payload and isinstance(payload["images"], list):
        return payload["images"]

    item = dict(payload)
    if not (item.get("name") or item.get("image") or item.get("file_name")) and fallback_image_name:
        item["name"] = fallback_image_name
    return [item]


def resolve_image_path(images_dir: Path, image_name: str) -> Path | None:
    direct = images_dir / image_name
    if direct.exists():
        return direct
    stem = Path(image_name).stem
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _detection_from_label(category: str, box: dict | list) -> Detection:
    if isinstance(box, dict):
        if {"x1", "y1", "x2", "y2"}.issubset(box):
            bbox = BoundingBox(
                x1=float(box["x1"]),
                y1=float(box["y1"]),
                x2=float(box["x2"]),
                y2=float(box["y2"]),
            )
        elif {"x", "y", "width", "height"}.issubset(box):
            x = float(box["x"])
            y = float(box["y"])
            bbox = BoundingBox(
                x1=x,
                y1=y,
                x2=x + float(box["width"]),
                y2=y + float(box["height"]),
            )
        else:
            raise ValueError(f"Unsupported bbox dictionary format: {box}")
    else:
        bbox = BoundingBox.from_xyxy([float(value) for value in box])

    return Detection(class_name=normalize_category(category), bbox=bbox)
