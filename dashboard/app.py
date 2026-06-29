from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from perception_inspector.adapters.yolo import YoloDetector
from perception_inspector.database import FailureDatabase
from perception_inspector.inspector import PerceptionInspector
from perception_inspector.labels import load_labeled_image_records
from perception_inspector.models import ImageRecord
from perception_inspector.vlm import HeuristicVLMAnalyzer


UPLOAD_DIR = Path("data/uploads")

APP_DESCRIPTION = (
    "Perception Inspector runs YOLO, checks prediction and scene-quality issues, "
    "scores suspicious scenes, and stores results for review."
)

MODE_DESCRIPTIONS = {
    "Online Inspection": (
        "Use one image without labels. The app flags detector-output and scene issues "
        "such as low confidence, duplicate boxes, no detections, small objects, blur, "
        "low lighting, and low contrast."
    ),
    "Offline Validation": (
        "Use images with labels. The app compares YOLO predictions against ground truth "
        "to find false positives, false negatives, misclassifications, poor localization, "
        "and the same scene-quality issues."
    ),
}

BOX_COLORS = {
    "prediction": "#00C853",
    "ground_truth": "#2979FF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db", default="data/failures.db")
    return parser.parse_known_args()[0]


def inspect_records(database: FailureDatabase, records: list[ImageRecord], run_vlm: bool) -> None:
    inspector = PerceptionInspector()
    analyzer = HeuristicVLMAnalyzer()
    database.initialize()
    progress = st.progress(0)
    status = st.empty()
    total = len(records)

    for index, record in enumerate(records, start=1):
        status.write(f"Inspecting {record.image_id} ({index}/{total})")
        result = inspector.inspect(record)
        database.upsert_record(record, result)
        if run_vlm and result.should_send_to_vlm:
            database.upsert_vlm_result(analyzer.analyze(record, result))
        progress.progress(index / total)

    status.success(f"Inspected {total} images")


def render_annotated_image(image: Image.Image, detections: list[dict]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        source = detection["source"]
        color = BOX_COLORS.get(source, "#FFD600")
        label = detection["class_name"]
        if detection["confidence"] is not None:
            label = f"{label} {detection['confidence']:.2f}"
        if source == "ground_truth":
            label = f"GT {label}"

        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_height = text_box[3] - text_box[1]
        text_width = text_box[2] - text_box[0]
        label_y = max(0, y1 - text_height - 4)
        draw.rectangle((x1, label_y, x1 + text_width + 6, label_y + text_height + 4), fill=color)
        draw.text((x1 + 3, label_y + 2), label, fill="black", font=font)

    return annotated


def rows_to_detections(rows) -> list[dict]:
    detections = []
    for row in rows:
        detections.append(
            {
                "source": row["source"],
                "class_name": row["class_name"],
                "confidence": row["confidence"],
                "bbox": json.loads(row["bbox_json"]),
            }
        )
    return detections


def render_data_loader(database: FailureDatabase) -> None:
    with st.sidebar.expander("Data Loader", expanded=True):
        st.caption(APP_DESCRIPTION)
        mode = st.radio(
            "Mode",
            ["Online Inspection", "Offline Validation"],
            help="Online uses image + YOLO predictions only. Offline also compares against labels.",
        )
        st.info(MODE_DESCRIPTIONS[mode])
        if mode == "Online Inspection":
            render_online_loader(database)
        else:
            render_offline_loader(database)


def render_online_loader(database: FailureDatabase) -> None:
    uploaded_image = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"])
    yolo_model = st.text_input("YOLO Model", value="yolo11n.pt", key="online_yolo_model")
    run_vlm = st.checkbox("Generate VLM fallback analysis", value=True, key="online_run_vlm")

    if st.button("Inspect Image", type="primary"):
        if uploaded_image is None:
            st.error("Upload an image first.")
            return
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        image_path = UPLOAD_DIR / f"{uuid4().hex}_{uploaded_image.name}"
        image_path.write_bytes(uploaded_image.getbuffer())

        try:
            detector = YoloDetector(yolo_model)
        except RuntimeError as exc:
            st.error(str(exc))
            return

        record = ImageRecord(
            image_id=image_path.stem,
            filepath=image_path,
            predictions=detector.predict(image_path),
            ground_truth=[],
        )
        inspect_records(database, [record], run_vlm=run_vlm)
        st.rerun()


def render_offline_loader(database: FailureDatabase) -> None:
    labels_path = st.text_input(
        "Labels Path",
        placeholder="labels.json or labels/",
        help="Use one combined JSON file or a folder with one JSON file per image.",
    )
    images_dir = st.text_input(
        "Image Directory",
        placeholder="/path/to/images",
        help="Labels are matched to files in this folder by image filename.",
    )
    limit = st.number_input("Image Limit", min_value=1, max_value=10000, value=25, step=25)
    run_yolo = st.checkbox("Run YOLO predictions", value=False)
    yolo_model = st.text_input("YOLO Model", value="yolo11n.pt", disabled=not run_yolo)
    run_vlm = st.checkbox("Generate VLM fallback analysis", value=True)

    if st.button("Load Dataset", type="primary"):
        labels = Path(labels_path).expanduser()
        images = Path(images_dir).expanduser()
        if not labels.exists():
            st.error(f"Labels path not found: {labels}")
            return
        if not images.exists():
            st.error(f"Image directory not found: {images}")
            return

        records = load_labeled_image_records(labels, images, limit=int(limit))
        if not records:
            st.error("No records found. Check that image names in the labels JSON exist in the image directory.")
            return

        if run_yolo:
            try:
                detector = YoloDetector(yolo_model)
            except RuntimeError as exc:
                st.error(str(exc))
                return
            prediction_status = st.empty()
            predicted_records = []
            for index, record in enumerate(records, start=1):
                prediction_status.write(f"Running YOLO on {record.image_id} ({index}/{len(records)})")
                predicted_records.append(
                    record.model_copy(update={"predictions": detector.predict(record.filepath)})
                )
            prediction_status.success("YOLO predictions complete")
            records = predicted_records
        else:
            st.warning("Loaded labels without detector predictions. The inspector will flag missing detections.")

        inspect_records(database, records, run_vlm=run_vlm)
        st.rerun()


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Perception Inspector", layout="wide")
    st.title("Perception Inspector")

    database = FailureDatabase(args.db)
    render_data_loader(database)

    rows = database.list_failures() if Path(args.db).exists() else []
    if not rows:
        st.info("No inspected failures found. Use the Data Loader in the sidebar or run `perception-inspector inspect ...`.")
        return

    records = [dict(row) for row in rows]
    for record in records:
        record["flags"] = ", ".join(json.loads(record["flags_json"]))
        scene = json.loads(record["scene_metrics_json"])
        stats = json.loads(record["detection_stats_json"])
        metadata = json.loads(record.get("metadata_json") or "{}") if "metadata_json" in record else {}
        record["brightness"] = round(scene["brightness"], 2)
        record["blur_score"] = round(scene["blur_score"], 3)
        record["detections"] = stats["detection_count"]
        record["mode"] = metadata.get("mode", "unknown").replace("_", " ").title()

    frame = pd.DataFrame(records)
    priorities = ["Critical", "High", "Medium", "Low"]
    selected_priorities = st.sidebar.multiselect(
        "Priority",
        priorities,
        default=["Critical", "High", "Medium"],
    )
    query = st.sidebar.text_input("Flag or image search")

    filtered = frame[frame["priority"].isin(selected_priorities)]
    if query:
        query_lower = query.lower()
        filtered = filtered[
            filtered["image_id"].str.lower().str.contains(query_lower)
            | filtered["flags"].str.lower().str.contains(query_lower)
        ]

    st.dataframe(
        filtered[
            [
                "image_id",
                "failure_score",
                "priority",
                "mode",
                "flags",
                "brightness",
                "blur_score",
                "detections",
                "primary_failure",
                "recommendation",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    if filtered.empty:
        return

    selected_id = st.selectbox("Scene", filtered["image_id"].tolist())
    selected = filtered[filtered["image_id"] == selected_id].iloc[0]
    detections = rows_to_detections(database.list_detections(selected_id))
    prediction_rows = [detection for detection in detections if detection["source"] == "prediction"]

    left, right = st.columns([1, 1])
    with left:
        image_path = Path(selected["filepath"])
        if image_path.exists():
            image = Image.open(image_path)
            st.image(render_annotated_image(image, detections), width="stretch")
            st.caption("Green boxes are YOLO predictions. Blue boxes are labels when available.")
        else:
            st.warning(f"Image not found: {image_path}")

    with right:
        st.metric("Failure Score", selected["failure_score"])
        st.write("Priority:", selected["priority"])
        st.write("Flags:", selected["flags"])
        st.subheader("Predictions")
        if prediction_rows:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "class": detection["class_name"],
                            "confidence": round(detection["confidence"], 3)
                            if detection["confidence"] is not None
                            else None,
                            "bbox": [round(value, 1) for value in detection["bbox"]],
                        }
                        for detection in prediction_rows
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No predictions were stored for this image.")
        if selected.get("reasoning"):
            st.subheader("Failure Intelligence")
            st.write(selected["reasoning"])
            st.write("Recommendation:", selected.get("recommendation") or "Not available")


if __name__ == "__main__":
    main()
