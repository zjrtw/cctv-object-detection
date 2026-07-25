from datetime import datetime, timedelta, timezone
from pathlib import Path

from recognition import AppearanceResult, AttributePrediction
from runtime_state import LiveDetectionCandidate, LiveDetectionStore


def test_live_detection_store_returns_only_recent_items() -> None:
    store = LiveDetectionStore(ttl_seconds=60)
    recent = LiveDetectionCandidate(
        track_id=1,
        captured_at=datetime.now(timezone.utc),
        score=0.9,
        matched_features=("gender", "upper_color"),
        appearance=AppearanceResult(
            gender=AttributePrediction("female", 0.9),
            upper_color=AttributePrediction("yellow", 0.9),
        ),
        image_path=Path("captures/recent.jpg"),
        camera_id="camera-gate-01",
        camera_name="정문 앞 CCTV",
        camera_address="아파트 정문",
        camera_latitude=37.0,
        camera_longitude=127.0,
    )
    expired = LiveDetectionCandidate(
        track_id=2,
        captured_at=datetime.now(timezone.utc) - timedelta(seconds=120),
        score=0.7,
        matched_features=("gender",),
        appearance=AppearanceResult(gender=AttributePrediction("male", 0.7)),
        image_path=Path("captures/expired.jpg"),
        camera_id="camera-old-01",
        camera_name="오래된 CCTV",
        camera_address="오래된 위치",
        camera_latitude=None,
        camera_longitude=None,
    )

    store.record(expired)
    store.record(recent)

    items = store.recent()
    assert [item.track_id for item in items] == [1]


def test_live_detection_store_updates_ttl() -> None:
    store = LiveDetectionStore(ttl_seconds=300)
    store.set_ttl_seconds(1)
    assert store.recent() == []
