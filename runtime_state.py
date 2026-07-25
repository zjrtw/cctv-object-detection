"""실행 중 감지된 최근 후보를 API와 공유하는 메모리 저장소."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock

from recognition import AppearanceResult


@dataclass(frozen=True)
class LiveDetectionCandidate:
    track_id: int
    captured_at: datetime
    score: float
    matched_features: tuple[str, ...]
    appearance: AppearanceResult
    image_path: Path
    camera_id: str
    camera_name: str
    camera_address: str
    camera_latitude: float | None
    camera_longitude: float | None


class LiveDetectionStore:
    def __init__(self, ttl_seconds: int = 300) -> None:
        self._ttl = timedelta(seconds=max(ttl_seconds, 1))
        self._items: list[LiveDetectionCandidate] = []
        self._lock = RLock()

    def record(self, candidate: LiveDetectionCandidate) -> None:
        with self._lock:
            self._prune_locked()
            self._items.append(candidate)

    def recent(self) -> list[LiveDetectionCandidate]:
        with self._lock:
            self._prune_locked()
            return list(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def set_ttl_seconds(self, ttl_seconds: int) -> None:
        with self._lock:
            self._ttl = timedelta(seconds=max(ttl_seconds, 1))
            self._prune_locked()

    def _prune_locked(self) -> None:
        cutoff = datetime.now(timezone.utc) - self._ttl
        self._items = [item for item in self._items if item.captured_at >= cutoff]


live_detection_store = LiveDetectionStore()
