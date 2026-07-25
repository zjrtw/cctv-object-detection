import numpy as np
import pytest

from config import TargetAppearance
from matcher import calculate_appearance_score, decide_appearance_match
from recognition import (
    AppearanceAnalyzer,
    AppearanceResult,
    AttributePrediction,
    crop_person,
    estimate_height_cm,
)


def prediction(value: str, confidence: float = 0.9) -> AttributePrediction:
    return AttributePrediction(value, confidence)


def test_crop_person_clamps_bbox_to_frame() -> None:
    frame = np.zeros((100, 80, 3), dtype=np.uint8)
    crop = crop_person(frame, (-10, 10, 100, 90))
    assert crop is not None
    assert crop.shape == (80, 80, 3)


def test_invalid_crop_returns_none() -> None:
    frame = np.zeros((100, 80, 3), dtype=np.uint8)
    assert crop_person(frame, (20, 20, 10, 10)) is None


def test_height_requires_calibration() -> None:
    bbox = (0, 10, 50, 270)
    assert estimate_height_cm(bbox, None) is None
    assert estimate_height_cm(bbox, 2.0) == 130


def test_analyzer_returns_structured_predictions() -> None:
    def classifier(_image: np.ndarray, candidate_labels: list[str]):
        return [{"label": candidate_labels[0], "score": 0.8}]

    analyzer = AppearanceAnalyzer("unused", classifier=classifier)
    frame = np.zeros((100, 50, 3), dtype=np.uint8)
    result = analyzer.analyze(frame, (0, 0, 50, 100), pixels_per_cm=1.0)
    assert result is not None
    assert result.gender == prediction("female", 0.8)
    assert result.estimated_height_cm == 100


def test_similarity_ignores_unavailable_height() -> None:
    target = TargetAppearance(
        gender="female", age=7, upper_color="yellow", lower_color="black",
        height_cm=130,
    )
    result = AppearanceResult(
        gender=prediction("female"), age_group=prediction("child"),
        upper_color=prediction("yellow"), lower_color=prediction("black"),
    )
    score = calculate_appearance_score(target, result)
    assert "height" not in score.feature_scores
    assert score.total == pytest.approx(0.9)


def test_final_decision_contains_tracking_context() -> None:
    target = TargetAppearance(gender="female", upper_color="yellow")
    result = AppearanceResult(
        gender=prediction("female", 0.95),
        upper_color=prediction("yellow", 0.9),
    )
    match = decide_appearance_match(target, result, 7, (1, 2, 3, 4), 0.8)
    assert match.is_candidate is True
    assert match.track_id == 7
    assert match.method == "appearance"
    assert set(match.matched_features) == {"gender", "upper_color"}
