"""CCTV AI 애플리케이션 실행 진입점."""

from datetime import datetime
import cv2
import numpy as np
from numpy.typing import NDArray

from camera import VideoSourceError, create_video_source
from config import load_config
from detector import Detection, PersonDetector
from matcher import AppearanceMatch, decide_appearance_match
from missing_persons import MissingPersonRepository
from notification import DuplicateAlertGuard, NotificationEvent, send_notification
from recognition import AppearanceAnalyzer
from tracker import PersonTracker, Track
from utils import save_detection_captures


WINDOW_NAME = "CCTV AI - press q to quit"


def draw_detections(
    frame: NDArray[np.uint8], detections: list[Detection]
) -> NDArray[np.uint8]:
    """프레임에 사람 탐지 결과를 그린다."""
    output = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(value) for value in detection.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person {detection.confidence:.2f}"
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return output


def draw_tracks(
    frame: NDArray[np.uint8],
    tracks: list[Track],
    matches: dict[int, AppearanceMatch] | None = None,
) -> NDArray[np.uint8]:
    """프레임에 추적 상자와 track_id를 그린다."""
    output = frame.copy()
    matches = matches or {}
    for track in tracks:
        x1, y1, x2, y2 = (int(value) for value in track.bbox)
        match = matches.get(track.track_id)
        color = (0, 0, 255) if match and match.is_candidate else (0, 255, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track.track_id} {track.confidence:.2f}"
        if match is not None:
            label += f" {match.method} {match.score:.2f}"
        cv2.putText(
            output, label,
            (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2, cv2.LINE_AA,
        )
    return output


def main() -> None:
    """애플리케이션을 실행한다."""
    settings = load_config()
    source = create_video_source(settings)
    detector = PersonDetector(
        settings.yolo_model_path,
        settings.detection_confidence,
    )
    tracker = PersonTracker()
    appearance_analyzer = AppearanceAnalyzer(settings.appearance_model_name)
    alert_guard = DuplicateAlertGuard(settings.duplicate_suppression_seconds)
    person_repository = MissingPersonRepository(settings.api_data_path)
    latest_matches: dict[int, AppearanceMatch] = {}
    frame_number = 0

    try:
        with source:
            while True:
                success, frame = source.read()
                if not success or frame is None:
                    break

                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame)
                frame_number += 1
                if frame_number % settings.appearance_analysis_interval_frames == 0:
                    if person_repository.exists():
                        registered_targets = person_repository.list_active()
                        targets = [(person.person_id, person.to_target()) for person in registered_targets]
                    else:
                        targets = [(None, settings.target_person)]
                    for track in tracks:
                        appearance = appearance_analyzer.analyze(
                            frame, track.bbox, settings.height_pixels_per_cm
                        )
                        if appearance is None:
                            continue
                        candidate_matches = [
                            decide_appearance_match(
                                target,
                                appearance,
                                track.track_id,
                                track.bbox,
                                settings.appearance_similarity_threshold,
                            )
                            for _, target in targets
                        ]
                        if not candidate_matches:
                            continue
                        match = max(candidate_matches, key=lambda item: item.score)
                        latest_matches[track.track_id] = match
                        if match.is_candidate and alert_guard.allow(track.track_id):
                            detected_at = datetime.now()
                            captures = save_detection_captures(
                                frame,
                                track.bbox,
                                track.track_id,
                                match.method,
                                match.score,
                                settings.capture_directory,
                                detected_at,
                            )
                            send_notification(
                                NotificationEvent(
                                    detected_at,
                                    track.track_id,
                                    match.method,
                                    match.score,
                                    captures,
                                )
                            )
                active_ids = {track.track_id for track in tracks}
                latest_matches = {
                    track_id: match
                    for track_id, match in latest_matches.items()
                    if track_id in active_ids
                }
                cv2.imshow(WINDOW_NAME, draw_tracks(frame, tracks, latest_matches))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except VideoSourceError as error:
        raise SystemExit(str(error)) from error
    finally:
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
