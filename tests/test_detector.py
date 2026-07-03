from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from detector import PersonDetector


@patch("detector.person_detector.YOLO")
def test_detect_returns_only_people_above_threshold(yolo: MagicMock) -> None:
    person = MagicMock()
    person.cls.item.return_value = 0
    person.conf.item.return_value = 0.9
    person.xyxy.__getitem__.return_value.tolist.return_value = [1, 2, 3, 4]
    car = MagicMock()
    car.cls.item.return_value = 2
    car.conf.item.return_value = 0.95
    result = MagicMock(boxes=[person, car])
    yolo.return_value.predict.return_value = [result]
    detector = PersonDetector("model.pt", 0.5)
    detections = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert len(detections) == 1
    assert detections[0].class_name == "person"
    assert detections[0].bbox == (1.0, 2.0, 3.0, 4.0)


def test_invalid_threshold_is_rejected() -> None:
    with pytest.raises(ValueError):
        PersonDetector("model.pt", 1.1)
