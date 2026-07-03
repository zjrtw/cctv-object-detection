"""CCTV AI 애플리케이션 실행 진입점."""

import cv2
import numpy as np
from numpy.typing import NDArray

from camera import VideoSourceError, create_video_source
from config import load_config
from detector import Detection, PersonDetector


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


def main() -> None:
    """애플리케이션을 실행한다."""
    settings = load_config()
    source = create_video_source(settings)
    detector = PersonDetector(
        settings.yolo_model_path,
        settings.detection_confidence,
    )

    try:
        with source:
            while True:
                success, frame = source.read()
                if not success or frame is None:
                    break

                detections = detector.detect(frame)
                cv2.imshow(WINDOW_NAME, draw_detections(frame, detections))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except VideoSourceError as error:
        raise SystemExit(str(error)) from error
    finally:
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
