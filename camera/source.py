"""OpenCV 기반 영상 입력 소스."""

from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]


class VideoSourceError(RuntimeError):
    """영상 입력을 열거나 읽을 수 없을 때 발생한다."""


class VideoSource(ABC):
    """모든 영상 입력이 따르는 공통 인터페이스."""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> tuple[bool, Frame | None]: ...

    @abstractmethod
    def release(self) -> None: ...

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class OpenCVVideoSource(VideoSource):
    """OpenCV VideoCapture를 사용하는 기본 입력 구현."""

    def __init__(self, source: int | str) -> None:
        self.source = source
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.release()
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(f"영상 입력을 열 수 없습니다: {self.source}")
        self._capture = capture

    def read(self) -> tuple[bool, Frame | None]:
        if self._capture is None:
            raise VideoSourceError("영상 입력이 열려 있지 않습니다.")
        success, frame = self._capture.read()
        return bool(success), frame if success else None

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class WebcamSource(OpenCVVideoSource):
    def __init__(self, device_index: int) -> None:
        super().__init__(device_index)


class VideoFileSource(OpenCVVideoSource):
    def __init__(self, path: str | Path) -> None:
        video_path = Path(path)
        if not video_path.is_file():
            raise VideoSourceError(f"영상 파일을 찾을 수 없습니다: {video_path}")
        super().__init__(str(video_path))


class RTSPSource(OpenCVVideoSource):
    def __init__(self, url: str) -> None:
        if not url.strip():
            raise VideoSourceError("RTSP URL이 비어 있습니다.")
        super().__init__(url)
