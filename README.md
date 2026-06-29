# Perception Inspector

Perception Inspector is a model-agnostic validation layer for object detection systems. It inspects detector predictions, optionally compares them with ground truth, flags suspicious scenes, stores structured failure intelligence, and exposes a Streamlit dashboard for engineering review.

This scaffold follows the project handoff: the detector is treated as a black box, deterministic inspection happens before AI reasoning, and only flagged images are candidates for Vision Language Model analysis.

## What Is Included

- Deterministic validation rules for confidence, missing detections, no detections, duplicates, false positives, false negatives, misclassifications, localization, abnormal boxes, and scene quality.
- Failure scoring from rule violations, scene metrics, object size, and confidence statistics.
- Structured failure IDs, scene metadata, and virtual bins for review organization.
- SQLite persistence for images, detections, inspector outputs, and VLM results.
- CLI commands for inspecting JSON inputs and initializing the database.
- Streamlit dashboard for browsing high-priority failures and clearing the current database contents.
- A VLM adapter interface with a deterministic local fallback and a placeholder for Qwen2.5-VL integration.
- Sample fixture data and tests.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,dashboard]"
pytest
perception-inspector init-db data/failures.db
perception-inspector inspect data/samples/manifest.json --db data/failures.db
streamlit run dashboard/app.py -- --db data/failures.db
```

## Input Format

The CLI expects a manifest JSON file:

```json
{
  "images": [
    {
      "image_id": "sample-night-001",
      "filepath": "data/samples/sample-night-001.png",
      "predictions": [
        {"class_name": "person", "confidence": 0.18, "bbox": [44, 30, 54, 62]}
      ],
      "ground_truth": [
        {"class_name": "person", "bbox": [43, 28, 56, 66]}
      ]
    }
  ]
}
```

Bounding boxes use `[x1, y1, x2, y2]` pixel coordinates.

## Optional Detector Integration

The package includes a small YOLO adapter. Install detector extras and use it from your own pipeline:

```bash
pip install -e ".[detector]"
```

The core inspector does not require YOLO and can evaluate predictions from any detector that can be converted to the manifest format.

## Inspection Modes

Perception Inspector supports two workflows with the same rule engine.

### Online Inspection

Online mode does not require labels. It uses:

- Image
- Detector predictions
- Scene metrics
- Inspector flags
- Optional VLM reasoning

This mode can flag prediction and scene-quality issues such as low confidence, duplicate detections, no detections, tiny objects, abnormal boxes, low lighting, low contrast, and motion blur. It does not claim false positives, false negatives, misclassifications, or localization errors because those require ground truth.

### Offline Validation

Offline mode uses labels when available. It uses:

- Image
- Ground truth boxes/classes
- Detector predictions
- IoU matching
- Inspector flags
- Optional VLM reasoning

This mode can compute label-dependent failures such as false positives, false negatives, misclassifications, and poor localization. It is the recommended MVP path for validation datasets because the failures are objectively measurable.

## Data Loader

For offline validation, point the app or CLI at:

- Labels path, for example `/path/to/labels.json` or `/path/to/labels/`.
- Image directory, for example `/path/to/images`.

The connection between images and labels comes from the label records. Each record should include an image filename in a `name`, `image`, or `file_name` field. The loader looks for that filename inside the image directory. If the exact filename is not found, it tries the same stem with `.jpg`, `.jpeg`, `.png`, and `.webp`.

If you use one JSON file per image, put all label files in one folder. A label file named `000123.json` will match `000123.jpg`, `000123.jpeg`, `000123.png`, or `000123.webp` in the image directory.

Example label record:

```json
{
  "name": "000123.jpg",
  "labels": [
    {
      "category": "car",
      "box2d": {"x1": 100, "y1": 80, "x2": 180, "y2": 140}
    }
  ]
}
```

Streamlit:

```bash
streamlit run dashboard/app.py -- --db data/failures.db
```

Open the `Data Loader` panel in the sidebar. Choose `Online Inspection` for single-image upload without labels, or `Offline Validation` for dataset evaluation with labels. Use the `Database` section in the sidebar to clear all stored rows from the active SQLite file when you want to start a fresh review session.

CLI:

```bash
perception-inspector inspect-labels \
  /path/to/labels \
  /path/to/images \
  --db data/failures.db \
  --yolo-model yolo11n.pt \
  --limit 100
```

If `--yolo-model` is omitted in offline validation, the loader still verifies that labels/images are readable, but predictions will be empty and the inspector will flag missing detections. For real failure analysis, install detector extras and run YOLO.

## Failure Intelligence Output

Each inspected image receives:

- a deterministic failure ID in the form `FAIL-XXXXXX`
- scene metadata such as lighting, weather, traffic density, road type, occlusion, and scene complexity
- virtual review bins such as `Night`, `Motion Blur`, `Small Objects`, and `High Priority`

These fields are persisted in the SQLite database and shown in the dashboard for faster review.

## Project Layout

```text
src/perception_inspector/
  cli.py              CLI entrypoint
  inspector.py        deterministic validation and scoring
  matching.py         IoU matching and detection comparison
  scene.py            OpenCV scene quality metrics
  database.py         SQLite persistence
  vlm.py              VLM interface and fallback analyzer
  adapters/yolo.py    optional YOLO prediction adapter
dashboard/app.py      Streamlit dashboard
data/samples/         runnable sample manifest
tests/                focused unit tests
```
