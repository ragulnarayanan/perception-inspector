from __future__ import annotations

import json
from pathlib import Path

import typer

from perception_inspector.database import FailureDatabase
from perception_inspector.inspector import PerceptionInspector
from perception_inspector.labels import load_labeled_image_records
from perception_inspector.manifest import load_manifest
from perception_inspector.vlm import HeuristicVLMAnalyzer

app = typer.Typer(help="Inspect object detection failures and store failure intelligence.")


@app.command("init-db")
def init_db(db: Path = typer.Argument(..., help="SQLite database path.")) -> None:
    database = FailureDatabase(db)
    database.initialize()
    typer.echo(f"Initialized database at {db}")


@app.command("inspect")
def inspect_manifest(
    manifest: Path = typer.Argument(..., help="Manifest JSON containing images and detections."),
    db: Path = typer.Option(Path("data/failures.db"), help="SQLite database path."),
    run_vlm: bool = typer.Option(True, help="Run the deterministic VLM fallback for high-priority images."),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable summary."),
) -> None:
    records = load_manifest(manifest)
    inspector = PerceptionInspector()
    analyzer = HeuristicVLMAnalyzer()
    database = FailureDatabase(db)
    database.initialize()

    summaries = []
    for record in records:
        result = inspector.inspect(record)
        database.upsert_record(record, result)
        vlm_result = None
        if run_vlm and result.should_send_to_vlm:
            vlm_result = analyzer.analyze(record, result)
            database.upsert_vlm_result(vlm_result)
        summaries.append(
            {
                "image_id": result.image_id,
                "failure_score": result.failure_score,
                "priority": result.priority.value,
                "flags": result.flags,
                "sent_to_vlm": vlm_result is not None,
            }
        )

    if json_output:
        typer.echo(json.dumps({"results": summaries}, indent=2))
        return

    typer.echo(f"Inspected {len(records)} images")
    for summary in summaries:
        typer.echo(
            f"- {summary['image_id']}: {summary['priority']} "
            f"({summary['failure_score']}) {', '.join(summary['flags'])}"
        )


@app.command("inspect-labels")
def inspect_labels(
    labels: Path = typer.Argument(..., help="Labels JSON file or directory of per-image JSON files."),
    images: Path = typer.Argument(..., help="Image directory."),
    db: Path = typer.Option(Path("data/failures.db"), help="SQLite database path."),
    yolo_model: str | None = typer.Option(None, help="Optional Ultralytics YOLO model, e.g. yolo11n.pt."),
    limit: int = typer.Option(50, min=1, help="Maximum number of images to inspect."),
    run_vlm: bool = typer.Option(True, help="Run deterministic VLM fallback for high-priority images."),
) -> None:
    records = load_labeled_image_records(labels, images, limit=limit)
    if yolo_model:
        from perception_inspector.adapters.yolo import YoloDetector

        detector = YoloDetector(yolo_model)
        records = [
            record.model_copy(update={"predictions": detector.predict(record.filepath)})
            for record in records
        ]

    inspector = PerceptionInspector()
    analyzer = HeuristicVLMAnalyzer()
    database = FailureDatabase(db)
    database.initialize()

    for record in records:
        result = inspector.inspect(record)
        database.upsert_record(record, result)
        if run_vlm and result.should_send_to_vlm:
            database.upsert_vlm_result(analyzer.analyze(record, result))

    typer.echo(f"Loaded and inspected {len(records)} images into {db}")


if __name__ == "__main__":
    app()
