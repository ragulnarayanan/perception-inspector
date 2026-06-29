from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from perception_inspector.models import SceneMetrics


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def compute_scene_metrics(image: np.ndarray) -> SceneMetrics:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = float(1.0 / (1.0 + sharpness / 100.0))

    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = histogram / max(histogram.sum(), 1.0)
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * np.log2(probabilities)).sum())

    return SceneMetrics(
        brightness=brightness,
        contrast=contrast,
        blur_score=blur_score,
        sharpness=sharpness,
        entropy=entropy,
    )
