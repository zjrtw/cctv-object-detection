"""YAML 설정 파일을 검증하여 읽는다."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("settings.yaml")
SUPPORTED_INPUT_TYPES = {"webcam", "video_file", "rtsp"}


@dataclass(frozen=True)
class TargetAppearance:
    gender: str = ""
    age: int | None = None
    hair_length: str = ""
    hair_color: str = ""
    upper_clothing: str = ""
    upper_color: str = ""
    lower_clothing: str = ""
    lower_color: str = ""
    height_cm: float | None = None
    height_tolerance_cm: float = 10.0


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
    target_person: TargetAppearance = field(default_factory=TargetAppearance)
    appearance_similarity_threshold: float = 0.7
    height_pixels_per_cm: float | None = None
    appearance_model_name: str = "openai/clip-vit-base-patch32"
    appearance_analysis_interval_frames: int = 30
    detection_mode: str = "mock"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_data_path: Path = Path("data/missing_persons.json")
    api_media_directory: Path = Path("data/missing_person_media")
    api_max_upload_bytes: int = 10 * 1024 * 1024
    api_camera_catalog_path: Path = Path("../backend/src/main/resources/missing-person/cameras.yml")
    api_live_detection_ttl_seconds: int = 300
    api_key_env: str = "MISSING_PERSON_API_KEY"
    api_key: str = ""
    detection_internal_token_env: str = "MISSING_PERSON_DETECTION_INTERNAL_TOKEN"
    detection_internal_token: str = ""


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
    target = _section(data, "target_person")
    appearance = _section(data, "appearance")
    detection = data.get("detection", {})
    api_settings = data.get("api", {})
    security_settings = data.get("security", {})
    if not isinstance(detection, dict) or not isinstance(api_settings, dict) or not isinstance(security_settings, dict):
        raise ValueError("detection, api와 security 설정은 객체여야 합니다.")

    input_type = str(input_settings["type"])
    if input_type not in SUPPORTED_INPUT_TYPES:
        raise ValueError(f"지원하지 않는 입력 타입입니다: {input_type}")

    appearance_threshold = float(appearance["similarity_threshold"])
    if not 0.0 <= appearance_threshold <= 1.0:
        raise ValueError("외형 유사도 threshold는 0과 1 사이여야 합니다.")

    pixels_per_cm_value = appearance.get("height_pixels_per_cm")
    pixels_per_cm = (
        float(pixels_per_cm_value) if pixels_per_cm_value is not None else None
    )
    if pixels_per_cm is not None and pixels_per_cm <= 0:
        raise ValueError("키 보정값은 0보다 커야 합니다.")
    analysis_interval = int(appearance.get("analysis_interval_frames", 30))
    if analysis_interval < 1:
        raise ValueError("외형 분석 프레임 간격은 1 이상이어야 합니다.")

    height_tolerance = float(target["height_tolerance_cm"])
    if height_tolerance < 0:
        raise ValueError("키 허용 오차는 0 이상이어야 합니다.")

    target_person = TargetAppearance(
        gender=str(target["gender"]),
        age=int(target["age"]) if target.get("age") is not None else None,
        hair_length=str(target["hair_length"]),
        hair_color=str(target["hair_color"]),
        upper_clothing=str(target["upper_clothing"]),
        upper_color=str(target["upper_color"]),
        lower_clothing=str(target["lower_clothing"]),
        lower_color=str(target["lower_color"]),
        height_cm=float(target["height_cm"]) if target.get("height_cm") is not None else None,
        height_tolerance_cm=height_tolerance,
    )

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
        target_person=target_person,
        appearance_similarity_threshold=appearance_threshold,
        height_pixels_per_cm=pixels_per_cm,
        appearance_model_name=str(appearance["model_name"]),
        appearance_analysis_interval_frames=analysis_interval,
        detection_mode=str(detection.get("mode", "mock")),
        api_host=str(api_settings.get("host", "127.0.0.1")),
        api_port=int(api_settings.get("port", 8000)),
        api_data_path=Path(api_settings.get("data_path", "data/missing_persons.json")),
        api_media_directory=Path(
            api_settings.get("media_directory", "data/missing_person_media")
        ),
        api_max_upload_bytes=int(api_settings.get("max_upload_bytes", 10 * 1024 * 1024)),
        api_camera_catalog_path=Path(
            api_settings.get(
                "camera_catalog_path",
                "../backend/src/main/resources/missing-person/cameras.yml",
            )
        ),
        api_live_detection_ttl_seconds=int(api_settings.get("live_detection_ttl_seconds", 300)),
        api_key_env=str(security_settings.get("api_key_env", "MISSING_PERSON_API_KEY")),
        api_key=os.getenv(
            str(security_settings.get("api_key_env", "MISSING_PERSON_API_KEY")), ""
        ),
        detection_internal_token_env=str(
            security_settings.get(
                "detection_internal_token_env",
                "MISSING_PERSON_DETECTION_INTERNAL_TOKEN",
            )
        ),
        detection_internal_token=os.getenv(
            str(
                security_settings.get(
                    "detection_internal_token_env",
                    "MISSING_PERSON_DETECTION_INTERNAL_TOKEN",
                )
            ),
            "",
        ),
    )
