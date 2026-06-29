import json
from pathlib import Path

from perception_inspector.labels import (
    load_labeled_image_records,
    load_labeled_image_records_from_payload,
    normalize_category,
)


def test_label_loader_reads_labels_and_normalizes_categories(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    image = images / "abc123.jpg"
    image.write_bytes(b"placeholder")
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            [
                {
                    "name": "abc123.jpg",
                    "labels": [
                        {
                            "category": "bike",
                            "box2d": {"x1": 1, "y1": 2, "x2": 10, "y2": 12},
                        }
                    ],
                }
            ]
        )
    )

    records = load_labeled_image_records(labels, images)

    assert len(records) == 1
    assert records[0].image_id == "abc123"
    assert records[0].ground_truth[0].class_name == "bicycle"


def test_normalize_category_keeps_unknown_labels() -> None:
    assert normalize_category("traffic sign") == "traffic sign"


def test_label_loader_reads_one_json_file_per_image(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "frame_001.jpg").write_bytes(b"placeholder")
    (labels / "frame_001.json").write_text(
        json.dumps(
            {
                "labels": [
                    {
                        "category": "car",
                        "box2d": {"x1": 5, "y1": 6, "x2": 20, "y2": 22},
                    }
                ]
            }
        )
    )

    records = load_labeled_image_records(labels, images)

    assert len(records) == 1
    assert records[0].image_id == "frame_001"
    assert records[0].ground_truth[0].class_name == "car"


def test_label_loader_supports_uploaded_payload_and_images() -> None:
    labels_payload = json.dumps(
        {
            "name": "frame_002.jpg",
            "labels": [{"category": "car", "box2d": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}}],
        }
    )

    records = load_labeled_image_records_from_payload(labels_payload, [("frame_002.jpg", b"placeholder")])

    assert len(records) == 1
    assert records[0].image_id == "frame_002"
    assert records[0].ground_truth[0].class_name == "car"
