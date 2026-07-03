"""Ultralytics YOLO 기반 사람 탐지기."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class PersonDetector:
    """프레임에서 confidence 기준을 넘는 사람만 반환한다."""

    PERSON_CLASS_ID = 0

    def __init__(self, model_path: str, confidence: float = 0.5) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence는 0과 1 사이여야 합니다.")
        self.confidence = confidence
        self.model = YOLO(model_path)

    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        if frame.size == 0:
            return []
        results: list[Any] = self.model.predict(frame, conf=self.confidence, verbose=False)
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                score = float(box.conf.item())
                if class_id != self.PERSON_CLASS_ID or score < self.confidence:
                    continue
                coordinates = tuple(float(value) for value in box.xyxy[0].tolist())
                detections.append(Detection(coordinates, score, class_id, "person"))
        return detections
