import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from api import create_app
from config import load_config
from missing_persons import MissingPersonRepository


def _settings(tmp_path, monkeypatch):
    monkeypatch.setenv("MISSING_PERSON_API_KEY", "test-key")
    settings = load_config()
    return settings.__class__(
        **{
            **settings.__dict__,
            "api_data_path": tmp_path / "missing_persons.json",
            "api_media_directory": tmp_path / "media",
            "api_key": "test-key",
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


def test_repository_data_survives_new_instance(tmp_path) -> None:
    path = tmp_path / "missing_persons.json"
    first = MissingPersonRepository(path)
    person = first.create({"name": "저장 테스트"})

    second = MissingPersonRepository(path)

    assert second.get(person.person_id) is not None
