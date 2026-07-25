from datetime import datetime, timedelta

from notification import DuplicateAlertGuard, NotificationEvent, send_notification
from utils import CapturePaths


def test_duplicate_alert_is_suppressed_until_period_expires(tmp_path) -> None:
    current = datetime(2026, 7, 4, 12, 0, 0)
    guard = DuplicateAlertGuard(300, clock=lambda: current)
    assert guard.allow(7) is True
    assert guard.allow(7) is False
    current += timedelta(seconds=300)
    assert guard.allow(7) is True
    assert guard.allow(8) is True


def test_notification_outputs_all_event_fields(tmp_path) -> None:
    paths = CapturePaths(tmp_path / "full.jpg", tmp_path / "box.jpg", tmp_path / "crop.jpg")
    event = NotificationEvent(datetime(2026, 7, 4, 12, 0), 7, "appearance", 0.91, paths)
    messages: list[str] = []
    send_notification(event, output=messages.append)
    assert "track_id=7" in messages[0]
    assert "appearance" in messages[0]
    assert "full.jpg" in messages[0]
    assert "box.jpg" in messages[0]
    assert "crop.jpg" in messages[0]
