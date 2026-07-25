"""track_id 기반 메모리 알림 중복 방지."""

from collections.abc import Callable
from datetime import datetime, timedelta


class DuplicateAlertGuard:
    def __init__(
        self,
        suppression_seconds: int,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        if suppression_seconds < 0:
            raise ValueError("중복 알림 방지 시간은 0 이상이어야 합니다.")
        self._suppression_period = timedelta(seconds=suppression_seconds)
        self._clock = clock
        self._last_alerts: dict[int, datetime] = {}

    def can_send(self, track_id: int) -> bool:
        """현재 track_id에 알림을 보낼 수 있는지 확인한다."""
        last_sent = self._last_alerts.get(track_id)
        return last_sent is None or self._clock() - last_sent >= self._suppression_period

    def mark_sent(self, track_id: int) -> None:
        self._last_alerts[track_id] = self._clock()

    def allow(self, track_id: int) -> bool:
        """전송 가능하면 즉시 이력을 기록하고 True를 반환한다."""
        if not self.can_send(track_id):
            return False
        self.mark_sent(track_id)
        return True
