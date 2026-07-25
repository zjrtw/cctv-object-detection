"""유사도 비교 및 판정 모듈."""
"""실종자 후보 판단 모듈."""

from matcher.appearance_matcher import (
    AppearanceMatch,
    AppearanceScore,
    calculate_appearance_score,
    decide_appearance_match,
)

__all__ = [
    "AppearanceMatch",
    "AppearanceScore",
    "calculate_appearance_score",
    "decide_appearance_match",
]
