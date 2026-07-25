"""실종자 정보 등록·조회·수정·삭제 HTTP API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from pydantic import BaseModel, Field, ValidationError

from config import AppConfig, load_config
from missing_persons import MissingPerson, MissingPersonRepository


class MissingPersonPayload(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    age: int | None = Field(default=None, ge=0, le=150)
    gender: str = Field(default="", max_length=30)
    hair_length: str = Field(default="", max_length=30)
    hair_color: str = Field(default="", max_length=30)
    upper_clothing: str = Field(default="", max_length=50)
    upper_color: str = Field(default="", max_length=30)
    lower_clothing: str = Field(default="", max_length=50)
    lower_color: str = Field(default="", max_length=30)
    height_cm: float | None = Field(default=None, gt=0, le=300)
    height_tolerance_cm: float = Field(default=10.0, ge=0, le=100)
    active: bool = True

    class Config:
        extra = "forbid"


class MissingPersonUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    age: int | None = Field(default=None, ge=0, le=150)
    gender: str | None = Field(default=None, max_length=30)
    hair_length: str | None = Field(default=None, max_length=30)
    hair_color: str | None = Field(default=None, max_length=30)
    upper_clothing: str | None = Field(default=None, max_length=50)
    upper_color: str | None = Field(default=None, max_length=30)
    lower_clothing: str | None = Field(default=None, max_length=50)
    lower_color: str | None = Field(default=None, max_length=30)
    height_cm: float | None = Field(default=None, gt=0, le=300)
    height_tolerance_cm: float | None = Field(default=None, ge=0, le=100)
    active: bool | None = None

    class Config:
        extra = "forbid"


def _response(person: MissingPerson) -> dict[str, Any]:
    return {
        "person_id": person.person_id,
        "name": person.name,
        "age": person.age,
        "gender": person.gender,
        "hair_length": person.hair_length,
        "hair_color": person.hair_color,
        "upper_clothing": person.upper_clothing,
        "upper_color": person.upper_color,
        "lower_clothing": person.lower_clothing,
        "lower_color": person.lower_color,
        "height_cm": person.height_cm,
        "height_tolerance_cm": person.height_tolerance_cm,
        "face_image_path": person.face_image_path,
        "body_image_path": person.body_image_path,
        "active": person.active,
        "created_at": person.created_at,
        "updated_at": person.updated_at,
    }


def _safe_media_path(path: str, media_directory: Path) -> Path:
    candidate = Path(path).resolve()
    root = media_directory.resolve()
    if candidate != root and root not in candidate.parents:
        raise HTTPException(status_code=500, detail="저장된 이미지 경로가 안전하지 않습니다.")
    return candidate


def _parse_payload(payload: str) -> MissingPersonPayload:
    try:
        return MissingPersonPayload.model_validate_json(payload)
    except AttributeError:
        try:
            return MissingPersonPayload.parse_raw(payload)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=error.errors()) from error
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=error.errors()) from error


async def _save_image(
    upload: UploadFile | None,
    directory: Path,
    filename: str,
    max_bytes: int,
) -> str | None:
    if upload is None:
        return None
    if upload.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="JPEG, PNG, WEBP 이미지만 허용됩니다.")
    content = await upload.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="이미지 파일이 허용 크기를 초과했습니다.")
    if not content:
        raise HTTPException(status_code=422, detail="빈 이미지 파일은 저장할 수 없습니다.")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    target.write_bytes(content)
    return str(target)


def create_app(
    settings: AppConfig | None = None,
    repository: MissingPersonRepository | None = None,
) -> FastAPI:
    settings = settings or load_config()
    repository = repository or MissingPersonRepository(settings.api_data_path)
    app = FastAPI(title="CCTV AI Missing Person API", version="1.0.0")
    app.state.repository = repository
    app.state.settings = settings

    def require_api_key(
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> None:
        configured_key = settings.api_key
        if not configured_key:
            raise HTTPException(status_code=503, detail="API 키가 서버 환경 변수에 설정되지 않았습니다.")
        bearer = authorization.removeprefix("Bearer ").strip() if authorization else None
        if x_api_key != configured_key and bearer != configured_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="유효한 API 키가 필요합니다.")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/missing-persons", status_code=201, dependencies=[Depends(require_api_key)])
    async def create_missing_person(
        payload: str = Form(..., description="실종자 정보 JSON 문자열"),
        face_image: UploadFile | None = File(default=None),
        body_image: UploadFile | None = File(default=None),
    ) -> dict[str, Any]:
        data = _parse_payload(payload)
        person = repository.create(data.model_dump() if hasattr(data, "model_dump") else data.dict())
        person_dir = settings.api_media_directory / person.person_id
        face_path = await _save_image(face_image, person_dir, "face", settings.api_max_upload_bytes)
        body_path = await _save_image(body_image, person_dir, "body", settings.api_max_upload_bytes)
        if face_path or body_path:
            person = repository.update(person.person_id, {
                "face_image_path": face_path or person.face_image_path,
                "body_image_path": body_path or person.body_image_path,
            }) or person
        return _response(person)

    @app.get("/api/missing-persons", dependencies=[Depends(require_api_key)])
    def list_missing_persons() -> list[dict[str, Any]]:
        return [_response(person) for person in repository.list_all()]

    @app.get("/api/missing-persons/{person_id}", dependencies=[Depends(require_api_key)])
    def get_missing_person(person_id: str) -> dict[str, Any]:
        person = repository.get(person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="실종자 정보를 찾을 수 없습니다.")
        return _response(person)

    @app.put("/api/missing-persons/{person_id}", dependencies=[Depends(require_api_key)])
    def update_missing_person(person_id: str, update: MissingPersonUpdate) -> dict[str, Any]:
        values = update.model_dump(exclude_unset=True) if hasattr(update, "model_dump") else update.dict(exclude_unset=True)
        person = repository.update(person_id, values)
        if person is None:
            raise HTTPException(status_code=404, detail="실종자 정보를 찾을 수 없습니다.")
        return _response(person)

    @app.delete("/api/missing-persons/{person_id}", dependencies=[Depends(require_api_key)])
    def delete_missing_person(person_id: str) -> dict[str, str]:
        person = repository.delete(person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="실종자 정보를 찾을 수 없습니다.")
        for path in (person.face_image_path, person.body_image_path):
            if path:
                _safe_media_path(path, settings.api_media_directory).unlink(missing_ok=True)
        return {"person_id": person_id, "status": "deleted"}

    return app
