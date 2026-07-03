"""설정에 맞는 영상 입력 소스를 생성한다."""

from camera.source import RTSPSource, VideoFileSource, VideoSource, WebcamSource
from config import AppConfig


def create_video_source(settings: AppConfig) -> VideoSource:
    if settings.input_type == "webcam":
        return WebcamSource(settings.webcam_device_index)
    if settings.input_type == "video_file":
        return VideoFileSource(settings.video_file_path)
    if settings.input_type == "rtsp":
        return RTSPSource(settings.rtsp_url)
    raise ValueError(f"지원하지 않는 입력 타입입니다: {settings.input_type}")
