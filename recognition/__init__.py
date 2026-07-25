"""인식 모듈."""
"""사람 외형 및 얼굴 인식 모듈."""

from recognition.appearance import (
    AppearanceAnalyzer,
    AppearanceResult,
    AttributePrediction,
    crop_person,
    estimate_height_cm,
)

__all__ = [
    "AppearanceAnalyzer",
    "AppearanceResult",
    "AttributePrediction",
    "crop_person",
    "estimate_height_cm",
]
