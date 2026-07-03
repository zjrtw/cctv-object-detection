from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from camera import VideoFileSource, VideoSourceError, WebcamSource, create_video_source
from config import AppConfig


def make_config(input_type: str, video_path: str = "") -> AppConfig:
    return AppConfig(input_type, 0, video_path, "rtsp://example", "yolo11n.pt", 0.5, 0.5, 0.7, Path("captures"), 300)


def test_factory_selects_webcam() -> None:
    assert isinstance(create_video_source(make_config("webcam")), WebcamSource)


def test_video_file_requires_existing_file() -> None:
    with pytest.raises(VideoSourceError):
        VideoFileSource("missing.mp4")


@patch("camera.source.cv2.VideoCapture")
def test_webcam_reads_frame(video_capture: MagicMock) -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    capture = video_capture.return_value
    capture.isOpened.return_value = True
    capture.read.return_value = (True, frame)
    source = WebcamSource(0)
    source.open()
    success, actual = source.read()
    source.release()
    assert success is True
    assert actual is frame
    capture.release.assert_called_once()
