"""실종자 후보 이미지 저장 기능."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import cv2
import numpy as np
from numpy.typing import NDArray

from recognition import crop_person


@dataclass(frozen=True)
class CapturePaths:
    full_frame: Path
    bounding_box: Path
    person_crop: Path


def create_capture_directory(root: Path, captured_at: datetime) -> Path:
    directory = root / captured_at.strftime("%Y-%m-%d")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def build_capture_filename(
    captured_at: datetime,
    track_id: int,
    method: str,
    score: float,
    image_type: str,
) -> str:
    safe_method = re.sub(r"[^a-zA-Z0-9_-]+", "-", method).strip("-") or "unknown"
    safe_type = re.sub(r"[^a-zA-Z0-9_-]+", "-", image_type).strip("-")
    return (
        f"{captured_at:%Y%m%d_%H%M%S_%f}_track-{track_id}_"
        f"{safe_method}_score-{score:.3f}_{safe_type}.jpg"
    )


def _write_image(path: Path, image: NDArray[np.uint8]) -> Path:
    if image.size == 0 or not cv2.imwrite(str(path), image):
        raise OSError(f"이미지를 저장하지 못했습니다: {path}")
    return path


def save_detection_captures(
    frame: NDArray[np.uint8],
    bbox: tuple[float, float, float, float],
    track_id: int,
    method: str,
    score: float,
    root: Path,
    captured_at: datetime | None = None,
) -> CapturePaths:
    captured_at = captured_at or datetime.now()
    directory = create_capture_directory(root, captured_at)
    crop = crop_person(frame, bbox)
    if crop is None:
        raise ValueError("유효하지 않은 사람 Bounding Box입니다.")

    full_path = directory / build_capture_filename(
        captured_at, track_id, method, score, "full"
    )
    box_path = directory / build_capture_filename(
        captured_at, track_id, method, score, "bbox"
    )
    crop_path = directory / build_capture_filename(
        captured_at, track_id, method, score, "crop"
    )

    annotated = frame.copy()
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        annotated,
        f"ID {track_id} {method} {score:.2f}",
        (max(0, x1), max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return CapturePaths(
        _write_image(full_path, frame),
        _write_image(box_path, annotated),
        _write_image(crop_path, crop),
    )
