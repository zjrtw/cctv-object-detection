"""외형 특징 유사도 계산과 후보 판단."""

from dataclasses import dataclass

from config import TargetAppearance
from recognition import AppearanceResult, AttributePrediction


@dataclass(frozen=True)
class AppearanceScore:
    total: float
    feature_scores: dict[str, float]
    matched_features: tuple[str, ...]


@dataclass(frozen=True)
class AppearanceMatch:
    is_candidate: bool
    track_id: int
    bbox: tuple[float, float, float, float]
    score: float
    matched_features: tuple[str, ...]
    method: str = "appearance"


def _age_group(age: int | None) -> str:
    if age is None:
        return ""
    if age <= 12:
        return "child"
    if age <= 19:
        return "teenager"
    if age <= 64:
        return "adult"
    return "senior"


def calculate_appearance_score(
    target: TargetAppearance, result: AppearanceResult
) -> AppearanceScore:
    comparisons: dict[str, tuple[str, AttributePrediction | None, float]] = {
        "gender": (target.gender, result.gender, 1.0),
        "age_group": (_age_group(target.age), result.age_group, 1.0),
        "hair_length": (target.hair_length, result.hair_length, 0.8),
        "hair_color": (target.hair_color, result.hair_color, 0.8),
        "upper_clothing": (target.upper_clothing, result.upper_clothing, 1.0),
        "upper_color": (target.upper_color, result.upper_color, 1.5),
        "lower_clothing": (target.lower_clothing, result.lower_clothing, 1.0),
        "lower_color": (target.lower_color, result.lower_color, 1.5),
    }
    scores: dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for name, (expected, prediction, weight) in comparisons.items():
        if not expected or prediction is None:
            continue
        score = prediction.confidence if prediction.value == expected else 0.0
        scores[name] = score
        weighted_sum += score * weight
        total_weight += weight

    if target.height_cm is not None and result.estimated_height_cm is not None:
        tolerance = max(target.height_tolerance_cm, 1.0)
        difference = abs(target.height_cm - result.estimated_height_cm)
        height_score = max(0.0, 1.0 - difference / tolerance)
        scores["height"] = height_score
        weighted_sum += height_score
        total_weight += 1.0

    total = weighted_sum / total_weight if total_weight else 0.0
    matched = tuple(name for name, score in scores.items() if score >= 0.5)
    return AppearanceScore(total, scores, matched)


def decide_appearance_match(
    target: TargetAppearance,
    result: AppearanceResult,
    track_id: int,
    bbox: tuple[float, float, float, float],
    threshold: float,
) -> AppearanceMatch:
    score = calculate_appearance_score(target, result)
    return AppearanceMatch(
        is_candidate=score.total >= threshold,
        track_id=track_id,
        bbox=bbox,
        score=score.total,
        matched_features=score.matched_features,
    )
