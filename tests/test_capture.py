from datetime import datetime

import cv2
import numpy as np

from utils import build_capture_filename, create_capture_directory, save_detection_captures


def test_create_date_directory(tmp_path) -> None:
    directory = create_capture_directory(tmp_path, datetime(2026, 7, 4))
    assert directory == tmp_path / "2026-07-04"
    assert directory.is_dir()


def test_filename_contains_detection_metadata() -> None:
    name = build_capture_filename(datetime(2026, 7, 4, 12, 30), 7, "appearance", 0.9123, "crop")
    assert "20260704_123000" in name
    assert "track-7" in name
    assert "appearance" in name
    assert "score-0.912" in name
    assert name.endswith("_crop.jpg")


def test_save_detection_captures_returns_existing_paths(tmp_path) -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    paths = save_detection_captures(
        frame, (10, 10, 60, 90), 7, "appearance", 0.9, tmp_path,
        datetime(2026, 7, 4, 12, 0),
    )
    assert paths.full_frame.is_file()
    assert paths.bounding_box.is_file()
    assert paths.person_crop.is_file()
    box_image = cv2.imread(str(paths.bounding_box))
    assert np.count_nonzero(box_image) > 0
