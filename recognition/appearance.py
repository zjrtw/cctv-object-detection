"""사람 Crop의 외형 특징 분석."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AttributePrediction:
    value: str
    confidence: float


@dataclass(frozen=True)
class AppearanceResult:
    gender: AttributePrediction | None = None
    age_group: AttributePrediction | None = None
    hair_length: AttributePrediction | None = None
    hair_color: AttributePrediction | None = None
    upper_clothing: AttributePrediction | None = None
    upper_color: AttributePrediction | None = None
    lower_clothing: AttributePrediction | None = None
    lower_color: AttributePrediction | None = None
    estimated_height_cm: float | None = None


ATTRIBUTE_LABELS = {
    "gender": ["female", "male"],
    "age_group": ["child", "teenager", "adult", "senior"],
    "hair_length": ["short", "medium", "long"],
    "hair_color": ["black", "brown", "blonde", "gray", "red"],
    "upper_clothing": ["jacket", "coat", "hoodie", "shirt", "sweater", "dress"],
    "upper_color": ["black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "brown"],
    "lower_clothing": ["pants", "shorts", "skirt", "dress"],
    "lower_color": ["black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "brown"],
}


def crop_person(frame: NDArray[np.uint8], bbox: tuple[float, float, float, float]) -> NDArray[np.uint8] | None:
    """프레임 범위를 벗어난 좌표를 보정해 사람 영역을 자른다."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x1 >= x2 or y1 >= y2:
        return None
    return frame[y1:y2, x1:x2].copy()


def estimate_height_cm(
    bbox: tuple[float, float, float, float], pixels_per_cm: float | None
) -> float | None:
    """카메라 보정값이 있을 때 Bounding Box 높이를 실제 키로 환산한다."""
    if pixels_per_cm is None or pixels_per_cm <= 0:
        return None
    pixel_height = max(0.0, bbox[3] - bbox[1])
    return pixel_height / pixels_per_cm if pixel_height else None


class AppearanceAnalyzer:
    """CLIP zero-shot 분류로 사람의 외형 속성을 분석한다."""

    def __init__(self, model_name: str, classifier: Callable[..., Any] | None = None) -> None:
        if classifier is None:
            from transformers import pipeline

            classifier = pipeline(
                "zero-shot-image-classification", model=model_name
            )
        self._classifier = classifier

    def _predict(self, image: NDArray[np.uint8], labels: list[str]) -> AttributePrediction | None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._classifier(rgb, candidate_labels=labels)
        if not results:
            return None
        best = results[0]
        return AttributePrediction(str(best["label"]), float(best["score"]))

    def analyze(
        self,
        frame: NDArray[np.uint8],
        bbox: tuple[float, float, float, float],
        pixels_per_cm: float | None = None,
    ) -> AppearanceResult | None:
        crop = crop_person(frame, bbox)
        if crop is None:
            return None
        predictions = {
            name: self._predict(crop, labels)
            for name, labels in ATTRIBUTE_LABELS.items()
        }
        return AppearanceResult(
            **predictions,
            estimated_height_cm=estimate_height_cm(bbox, pixels_per_cm),
        )
