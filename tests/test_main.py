import numpy as np

from detector import Detection
from main import draw_detections


def test_draw_detections_does_not_modify_original_frame() -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    detection = Detection((5, 5, 30, 30), 0.9, 0, "person")

    output = draw_detections(frame, [detection])

    assert np.count_nonzero(frame) == 0
    assert np.count_nonzero(output) > 0
