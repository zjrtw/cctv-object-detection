from pathlib import Path

from config import load_config


def test_load_default_config() -> None:
    settings = load_config()

    assert settings.input_type == "webcam"
    assert isinstance(settings.webcam_device_index, int)
    assert settings.webcam_device_index >= 0
    assert settings.yolo_model_path == "yolo11n.pt"
    assert settings.detection_confidence == 0.5
    assert settings.face_similarity == 0.5
    assert settings.reid_similarity == 0.7
    assert settings.capture_directory == Path("captures")
    assert settings.target_person.gender == "female"
    assert settings.target_person.age == 7
    assert settings.target_person.upper_color == "yellow"
    assert settings.target_person.lower_color == "black"
    assert settings.target_person.height_cm == 130
    assert settings.target_person.height_tolerance_cm == 10
    assert settings.appearance_similarity_threshold == 0.7
