from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from perception_inspector.models import Detection, ImageRecord, InspectionResult, VLMResult


SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    image_id TEXT PRIMARY KEY,
    filepath TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT NOT NULL,
    source TEXT NOT NULL,
    class_name TEXT NOT NULL,
    confidence REAL,
    bbox_json TEXT NOT NULL,
    FOREIGN KEY(image_id) REFERENCES images(image_id)
);

CREATE TABLE IF NOT EXISTS inspector (
    image_id TEXT PRIMARY KEY,
    failure_id TEXT NOT NULL,
    failure_score REAL NOT NULL,
    priority TEXT NOT NULL,
    flags_json TEXT NOT NULL,
    scene_metrics_json TEXT NOT NULL,
    detection_stats_json TEXT NOT NULL,
    matches_json TEXT NOT NULL,
    false_positive_indices_json TEXT NOT NULL,
    false_negative_indices_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    FOREIGN KEY(image_id) REFERENCES images(image_id)
);

CREATE TABLE IF NOT EXISTS vlm_results (
    image_id TEXT PRIMARY KEY,
    primary_failure TEXT NOT NULL,
    secondary_failures_json TEXT NOT NULL,
    scene_conditions_json TEXT NOT NULL,
    priority TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    recommendation TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    FOREIGN KEY(image_id) REFERENCES images(image_id)
);
"""


class FailureDatabase:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def upsert_record(self, record: ImageRecord, result: InspectionResult) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO images (image_id, filepath) VALUES (?, ?)",
                (record.image_id, str(record.filepath)),
            )
            connection.execute("DELETE FROM detections WHERE image_id = ?", (record.image_id,))
            self._insert_detections(connection, record.image_id, "prediction", record.predictions)
            self._insert_detections(connection, record.image_id, "ground_truth", record.ground_truth)
            connection.execute(
                """
                INSERT OR REPLACE INTO inspector (
                    image_id, failure_id, failure_score, priority, flags_json, scene_metrics_json,
                    detection_stats_json, matches_json, false_positive_indices_json,
                    false_negative_indices_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.image_id,
                    result.failure_id,
                    result.failure_score,
                    result.priority.value,
                    json.dumps(result.flags),
                    result.scene_metrics.model_dump_json(),
                    result.detection_stats.model_dump_json(),
                    json.dumps([match.model_dump() for match in result.matches]),
                    json.dumps(result.false_positive_indices),
                    json.dumps(result.false_negative_indices),
                    json.dumps(result.metadata),
                ),
            )

    def upsert_vlm_result(self, result: VLMResult) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO vlm_results (
                    image_id, primary_failure, secondary_failures_json,
                    scene_conditions_json, priority, reasoning, recommendation, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.image_id,
                    result.primary_failure,
                    json.dumps(result.secondary_failures),
                    json.dumps(result.scene_conditions),
                    result.priority.value,
                    result.reasoning,
                    result.recommendation,
                    result.model_dump_json(),
                ),
            )

    def clear(self) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM vlm_results")
            connection.execute("DELETE FROM inspector")
            connection.execute("DELETE FROM detections")
            connection.execute("DELETE FROM images")

    def list_failures(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(
                connection.execute(
                    """
                    SELECT
                        images.image_id,
                        images.filepath,
                        inspector.failure_id,
                        inspector.failure_score,
                        inspector.priority,
                        inspector.flags_json,
                        inspector.scene_metrics_json,
                        inspector.detection_stats_json,
                        inspector.metadata_json,
                        vlm_results.primary_failure,
                        vlm_results.reasoning,
                        vlm_results.recommendation
                    FROM inspector
                    JOIN images ON images.image_id = inspector.image_id
                    LEFT JOIN vlm_results ON vlm_results.image_id = inspector.image_id
                    ORDER BY inspector.failure_score DESC
                    """
                )
            )

    def list_detections(self, image_id: str, source: str | None = None) -> list[sqlite3.Row]:
        query = """
            SELECT image_id, source, class_name, confidence, bbox_json
            FROM detections
            WHERE image_id = ?
        """
        params: list[str] = [image_id]
        if source:
            query += " AND source = ?"
            params.append(source)
        query += " ORDER BY source, confidence DESC"
        with self.connect() as connection:
            return list(connection.execute(query, params))

    def _insert_detections(
        self,
        connection: sqlite3.Connection,
        image_id: str,
        source: str,
        detections: Iterable[Detection],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO detections (image_id, source, class_name, confidence, bbox_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    image_id,
                    source,
                    detection.class_name,
                    detection.confidence,
                    json.dumps(detection.bbox.to_xyxy()),
                )
                for detection in detections
            ],
        )
