"""YAML 설정 파일을 검증하여 읽는다."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("settings.yaml")
SUPPORTED_INPUT_TYPES = {"webcam", "video_file", "rtsp"}


@dataclass(frozen=True)
class AppConfig:
    input_type: str
    webcam_device_index: int
    video_file_path: str
    rtsp_url: str
    yolo_model_path: str
    detection_confidence: float
    face_similarity: float
    reid_similarity: float
    capture_directory: Path
    duplicate_suppression_seconds: int


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"설정 섹션이 없거나 올바르지 않습니다: {name}")
    return value


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """설정 파일을 읽고 검증된 애플리케이션 설정을 반환한다."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("설정 파일의 최상위 값은 객체여야 합니다.")

    input_settings = _section(data, "input")
    models = _section(data, "models")
    thresholds = _section(data, "thresholds")
    capture = _section(data, "capture")
    notification = _section(data, "notification")

    input_type = str(input_settings["type"])
    if input_type not in SUPPORTED_INPUT_TYPES:
        raise ValueError(f"지원하지 않는 입력 타입입니다: {input_type}")

    return AppConfig(
        input_type=input_type,
        webcam_device_index=int(input_settings["webcam_device_index"]),
        video_file_path=str(input_settings["video_file_path"]),
        rtsp_url=str(input_settings["rtsp_url"]),
        yolo_model_path=str(models["yolo_path"]),
        detection_confidence=float(thresholds["detection_confidence"]),
        face_similarity=float(thresholds["face_similarity"]),
        reid_similarity=float(thresholds["reid_similarity"]),
        capture_directory=Path(capture["directory"]),
        duplicate_suppression_seconds=int(
            notification["duplicate_suppression_seconds"]
        ),
    )
