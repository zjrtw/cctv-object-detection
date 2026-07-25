"""탐지 결과와 분리된 ByteTrack 어댑터."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from detector import Detection


@dataclass(frozen=True)
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int


def detections_to_boxes(detections: list[Detection], frame_shape: tuple[int, ...]) -> Boxes:
    rows = [[*item.bbox, item.confidence, item.class_id] for item in detections]
    data = np.asarray(rows, dtype=np.float32).reshape((-1, 6))
    return Boxes(data, frame_shape[:2])


class PersonTracker:
    def __init__(self) -> None:
        args = SimpleNamespace(
            track_high_thresh=0.25, track_low_thresh=0.1,
            new_track_thresh=0.25, track_buffer=30,
            match_thresh=0.8, fuse_score=True,
        )
        self._tracker = BYTETracker(args)

    def update(self, detections: list[Detection], frame: NDArray[np.uint8]) -> list[Track]:
        tracked = self._tracker.update(detections_to_boxes(detections, frame.shape), img=frame)
        return [
            Track(int(row[4]), tuple(float(v) for v in row[:4]), float(row[5]), int(row[6]))
            for row in tracked
        ]
