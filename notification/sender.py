"""외부 알림 서비스로 교체 가능한 MVP 알림 인터페이스."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Callable

from utils import CapturePaths


@dataclass(frozen=True)
class NotificationEvent:
    detected_at: datetime
    track_id: int
    method: str
    score: float
    captures: CapturePaths


def send_notification(
    event: NotificationEvent,
    output: Callable[[str], None] = print,
) -> None:
    """MVP에서는 발견 정보를 콘솔로 출력한다."""
    output(
        "[실종자 후보 발견] "
        f"시간={event.detected_at.isoformat(timespec='seconds')} "
        f"track_id={event.track_id} 방식={event.method} 점수={event.score:.3f} "
        f"전체={event.captures.full_frame} "
        f"박스={event.captures.bounding_box} "
        f"crop={event.captures.person_crop}"
    )
