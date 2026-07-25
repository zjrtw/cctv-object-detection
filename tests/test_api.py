import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from api import create_app
from config import load_config
from missing_persons import MissingPersonRepository


def _settings(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSING_PERSON_API_KEY", "test-key")
    monkeypatch.setenv("MISSING_PERSON_DETECTION_INTERNAL_TOKEN", "internal-test-key")
    settings = load_config()
    return settings.__class__(
        **{
            **settings.__dict__,
            "api_data_path": tmp_path / "missing_persons.json",
            "api_media_directory": tmp_path / "media",
            "api_key": "test-key",
            "detection_internal_token": "internal-test-key",
        }
    )


def test_health_does_not_require_authentication(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    client = TestClient(create_app(settings))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_register_list_update_and_delete_missing_person(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    client = TestClient(create_app(settings, MissingPersonRepository(settings.api_data_path)))
    headers = {"X-API-Key": "test-key"}
    payload = {
        "name": "홍길동",
        "age": 7,
        "gender": "female",
        "upper_color": "yellow",
        "height_cm": 130,
    }

    created = client.post(
        "/api/missing-persons",
        headers=headers,
        data={"payload": json.dumps(payload)},
    )
    assert created.status_code == 201
    person_id = created.json()["person_id"]

    listed = client.get("/api/missing-persons", headers=headers)
    assert listed.status_code == 200
    assert listed.json()[0]["name"] == "홍길동"

    updated = client.put(
        f"/api/missing-persons/{person_id}",
        headers=headers,
        json={"active": False},
    )
    assert updated.status_code == 200
    assert updated.json()["active"] is False

    deleted = client.delete(f"/api/missing-persons/{person_id}", headers=headers)
    assert deleted.status_code == 200
    assert client.get(f"/api/missing-persons/{person_id}", headers=headers).status_code == 404


def test_api_rejects_missing_or_invalid_key(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    client = TestClient(create_app(settings))

    assert client.get("/api/missing-persons").status_code == 401
    assert client.get("/api/missing-persons", headers={"X-API-Key": "wrong"}).status_code == 401


def test_detection_contract_returns_matched_response_and_image(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    repository = MissingPersonRepository(settings.api_data_path)
    image_path = settings.api_media_directory / "target-1" / "body.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fake-image-bytes")
    repository.create(
        {
            "name": "테스트 대상",
            "age": 7,
            "gender": "female",
            "hair_length": "short",
            "upper_color": "yellow",
            "body_image_path": str(image_path),
        }
    )
    client = TestClient(create_app(settings, repository))
    headers = {"X-Internal-Token": "internal-test-key"}

    response = client.post(
        "/api/v1/detections/missing-person",
        headers=headers,
        json={
            "detectionRequestId": "det-20260725-0001",
            "missingPersonId": 101,
            "recipientUserId": 2001,
            "requestedAt": "2026-07-25T14:30:00+09:00",
            "appearance": {
                "ageGroup": "CHILD",
                "topColor": "YELLOW",
                "hairType": "SHORT_HAIR",
                "gender": "FEMALE",
            },
            "matchingPolicy": {
                "minimumMatchedAttributes": 2,
                "allowPartialMatch": True,
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "MATCHED"
    assert set(body["matchAttributes"]) >= {"age_group", "gender", "hair_length", "upper_color"}
    assert body["camera"]["cameraId"].startswith("camera-")
    assert body["imageUrl"].endswith("/image")

    image_response = client.get(body["imageUrl"], headers=headers)
    assert image_response.status_code == 200
    assert image_response.content == b"fake-image-bytes"


def test_detection_contract_returns_not_found_when_attributes_do_not_match(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    client = TestClient(create_app(settings, MissingPersonRepository(settings.api_data_path)))
    headers = {"X-Internal-Token": "internal-test-key"}

    response = client.post(
        "/api/v1/detections/missing-person",
        headers=headers,
        json={
            "detectionRequestId": "det-20260725-9999",
            "missingPersonId": 101,
            "recipientUserId": 2001,
            "requestedAt": "2026-07-25T14:30:00+09:00",
            "appearance": {
                "ageGroup": "SENIOR",
                "topColor": "BLUE",
                "hairType": "LONG_HAIR",
                "gender": "MALE",
            },
            "matchingPolicy": {
                "minimumMatchedAttributes": 2,
                "allowPartialMatch": True,
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "NOT_FOUND"
    assert body["camera"] is None
    assert body["detailId"] is None


def test_repository_data_survives_new_instance(tmp_path) -> None:
    path = tmp_path / "missing_persons.json"
    first = MissingPersonRepository(path)
    person = first.create({"name": "저장 테스트"})

    second = MissingPersonRepository(path)

    assert second.get(person.person_id) is not None
