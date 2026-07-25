from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from detector import Detection
from tracker import PersonTracker, detections_to_boxes


def test_detection_conversion_preserves_values() -> None:
    boxes = detections_to_boxes(
        [Detection((1, 2, 30, 40), 0.9, 0, "person")], (100, 200, 3)
    )
    assert boxes.xyxy.tolist() == [[1.0, 2.0, 30.0, 40.0]]
    assert boxes.conf.tolist() == [pytest.approx(0.9)]
    assert boxes.cls.tolist() == [0.0]


@patch("tracker.byte_tracker.BYTETracker")
def test_tracker_returns_track_id(byte_tracker: MagicMock) -> None:
    byte_tracker.return_value.update.return_value = np.asarray(
        [[1, 2, 30, 40, 7, 0.9, 0, 0]], dtype=np.float32
    )
    tracker = PersonTracker()
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    tracks = tracker.update([Detection((1, 2, 30, 40), 0.9, 0, "person")], frame)
    assert tracks[0].track_id == 7
    assert tracks[0].bbox == (1.0, 2.0, 30.0, 40.0)
