"""실종자 정보 등록·조회·수정·삭제 HTTP API."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError
import yaml

from config import AppConfig, TargetAppearance, load_config
from matcher import calculate_appearance_score
from missing_persons import MissingPerson, MissingPersonRepository
from recognition import AppearanceResult, AttributePrediction
from runtime_state import LiveDetectionCandidate, live_detection_store


AGE_GROUP_TO_AGE = {
    "CHILD": 7,
    "TEEN": 16,
    "ADULT": 30,
    "MIDDLE_AGED": 50,
    "SENIOR": 70,
}

TOP_COLOR_TO_VALUE = {
    "BLACK": "black",
    "WHITE": "white",
    "RED": "red",
    "BLUE": "blue",
    "GREEN": "green",
    "YELLOW": "yellow",
    "GRAY": "gray",
}

HAIR_TYPE_TO_VALUE = {
    "SHORT_HAIR": "short",
    "LONG_HAIR": "long",
}

GENDER_TO_VALUE = {
    "MALE": "male",
    "FEMALE": "female",
}


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


class DetectionAppearancePayload(BaseModel):
    ageGroup: str | None = None
    topColor: str | None = None
    hairType: str | None = None
    gender: str | None = None

    class Config:
        extra = "forbid"


class DetectionPolicyPayload(BaseModel):
    minimumMatchedAttributes: int = Field(ge=1, default=1)
    allowPartialMatch: bool = True

    class Config:
        extra = "forbid"


class DetectionRequestPayload(BaseModel):
    detectionRequestId: str = Field(min_length=1)
    missingPersonId: int
    recipientUserId: int
    requestedAt: datetime
    appearance: DetectionAppearancePayload
    matchingPolicy: DetectionPolicyPayload

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


def _prediction(value: str | None) -> AttributePrediction | None:
    if not value:
        return None
    return AttributePrediction(value=value, confidence=1.0)


def _to_detection_result(appearance: DetectionAppearancePayload) -> AppearanceResult:
    return AppearanceResult(
        gender=_prediction(GENDER_TO_VALUE.get((appearance.gender or "").upper())),
        age_group=_prediction("child" if (appearance.ageGroup or "").upper() == "CHILD"
                              else "teenager" if (appearance.ageGroup or "").upper() == "TEEN"
                              else "adult" if (appearance.ageGroup or "").upper() in {"ADULT", "MIDDLE_AGED"}
                              else "senior" if (appearance.ageGroup or "").upper() == "SENIOR"
                              else None),
        hair_length=_prediction(HAIR_TYPE_TO_VALUE.get((appearance.hairType or "").upper())),
        upper_color=_prediction(TOP_COLOR_TO_VALUE.get((appearance.topColor or "").upper())),
    )


def _provided_attribute_count(appearance: DetectionAppearancePayload) -> int:
    return sum(
        value is not None and str(value).strip() != ""
        for value in (
            appearance.ageGroup,
            appearance.topColor,
            appearance.hairType,
            appearance.gender,
        )
    )


def _camera_info(settings: AppConfig) -> dict[str, Any]:
    catalog_path = settings.api_camera_catalog_path
    if not catalog_path.is_absolute():
        catalog_path = Path(__file__).resolve().parents[1] / catalog_path
    try:
        if catalog_path.exists():
            raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
            cameras = raw.get("cameras", [])
            if cameras:
                first = cameras[0]
                return {
                    "cameraId": first.get("cameraId"),
                    "name": first.get("name"),
                    "address": first.get("address"),
                    "latitude": first.get("latitude"),
                    "longitude": first.get("longitude"),
                }
    except (OSError, yaml.YAMLError, AttributeError):
        pass

    if settings.input_type == "video_file":
        return {
            "cameraId": "camera-video-file-01",
            "name": "샘플 영상 입력",
            "address": "로컬 테스트 비디오",
            "latitude": None,
            "longitude": None,
        }
    if settings.input_type == "rtsp":
        return {
            "cameraId": "camera-rtsp-01",
            "name": "RTSP CCTV 입력",
            "address": "RTSP 스트림 입력",
            "latitude": None,
            "longitude": None,
        }
    return {
        "cameraId": f"camera-webcam-{settings.webcam_device_index}",
        "name": f"웹캠 입력 {settings.webcam_device_index}",
        "address": "Camo Studio 웹캠 입력",
        "latitude": None,
        "longitude": None,
    }


def _camera_info_from_live(candidate: LiveDetectionCandidate) -> dict[str, Any]:
    return {
        "cameraId": candidate.camera_id,
        "name": candidate.camera_name,
        "address": candidate.camera_address,
        "latitude": candidate.camera_latitude,
        "longitude": candidate.camera_longitude,
    }


def _detail_id(request: DetectionRequestPayload) -> int:
    digest = sha256(
        f"{request.detectionRequestId}:{request.missingPersonId}:{request.recipientUserId}".encode("utf-8")
    ).hexdigest()
    return int(digest[:12], 16)


def _request_target(appearance: DetectionAppearancePayload) -> TargetAppearance:
    return TargetAppearance(
        gender=GENDER_TO_VALUE.get((appearance.gender or "").upper(), ""),
        age=AGE_GROUP_TO_AGE.get((appearance.ageGroup or "").upper()),
        hair_length=HAIR_TYPE_TO_VALUE.get((appearance.hairType or "").upper(), ""),
        upper_color=TOP_COLOR_TO_VALUE.get((appearance.topColor or "").upper(), ""),
    )


def _evaluate_match(
    matched_features: tuple[str, ...],
    request: DetectionRequestPayload,
) -> bool:
    matched_count = len(matched_features)
    provided_count = _provided_attribute_count(request.appearance)
    minimum = request.matchingPolicy.minimumMatchedAttributes
    allow_partial = request.matchingPolicy.allowPartialMatch
    return matched_count >= minimum and (allow_partial or matched_count == provided_count)


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
    live_detection_store.set_ttl_seconds(settings.api_live_detection_ttl_seconds)
    app = FastAPI(title="CCTV AI Missing Person API", version="1.0.0")
    app.state.repository = repository
    app.state.settings = settings
    app.state.detection_results = {}

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

    def require_internal_token(
        x_internal_token: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> None:
        configured_token = settings.detection_internal_token
        if not configured_token:
            raise HTTPException(status_code=503, detail="내부 탐지 토큰이 서버 환경 변수에 설정되지 않았습니다.")
        bearer = authorization.removeprefix("Bearer ").strip() if authorization else None
        if x_internal_token != configured_token and bearer != configured_token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="유효한 내부 탐지 토큰이 필요합니다.")

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

    @app.post("/api/v1/detections/missing-person", dependencies=[Depends(require_internal_token)])
    def detect_missing_person(
        request: DetectionRequestPayload,
        x_detection_mode: str | None = Header(default=None),
    ) -> dict[str, Any]:
        if _provided_attribute_count(request.appearance) == 0:
            raise HTTPException(status_code=400, detail="appearance에는 하나 이상의 조건이 필요합니다.")
        detection_mode = settings.detection_mode
        if x_detection_mode in {"mock", "yolo"}:
            detection_mode = x_detection_mode

        detail_id = _detail_id(request)
        if detection_mode == "yolo":
            live_target = _request_target(request.appearance)
            best_live: tuple[LiveDetectionCandidate, float, tuple[str, ...]] | None = None
            for candidate in live_detection_store.recent():
                score = calculate_appearance_score(live_target, candidate.appearance)
                if best_live is None or score.total > best_live[1]:
                    best_live = (candidate, score.total, score.matched_features)

            if best_live is None or not _evaluate_match(best_live[2], request):
                return {
                    "detectionRequestId": request.detectionRequestId,
                    "status": "NOT_FOUND",
                    "detectedAt": None,
                    "matchScore": None if best_live is None else round(best_live[1], 4),
                    "matchAttributes": [] if best_live is None else list(best_live[2]),
                    "camera": None,
                    "imageUrl": None,
                    "detailId": None,
                }

            candidate, match_score, matched_features = best_live
            app.state.detection_results[detail_id] = {"image_path": str(candidate.image_path)}
            return {
                "detectionRequestId": request.detectionRequestId,
                "status": "MATCHED",
                "detectedAt": candidate.captured_at.isoformat(),
                "matchScore": round(match_score, 4),
                "matchAttributes": list(matched_features),
                "camera": _camera_info_from_live(candidate),
                "imageUrl": f"/api/v1/detections/missing-person/results/{detail_id}/image",
                "detailId": detail_id,
            }

        targets = repository.list_active()
        if not targets:
            targets = [MissingPerson(person_id="config-target", name="설정 대상", **settings.target_person.__dict__)]

        requested_result = _to_detection_result(request.appearance)
        best_match: tuple[MissingPerson, float, tuple[str, ...]] | None = None
        for target in targets:
            score = calculate_appearance_score(target.to_target(), requested_result)
            if best_match is None or score.total > best_match[1]:
                best_match = (target, score.total, score.matched_features)

        assert best_match is not None
        target, match_score, matched_features = best_match
        if not _evaluate_match(matched_features, request):
            return {
                "detectionRequestId": request.detectionRequestId,
                "status": "NOT_FOUND",
                "detectedAt": None,
                "matchScore": round(match_score, 4),
                "matchAttributes": list(matched_features),
                "camera": None,
                "imageUrl": None,
                "detailId": None,
            }

        image_path = target.body_image_path or target.face_image_path
        app.state.detection_results[detail_id] = {"image_path": image_path}
        image_url = f"/api/v1/detections/missing-person/results/{detail_id}/image" if image_path else None
        return {
            "detectionRequestId": request.detectionRequestId,
            "status": "MATCHED",
            "detectedAt": datetime.now(timezone.utc).isoformat(),
            "matchScore": round(match_score, 4),
            "matchAttributes": list(matched_features),
            "camera": _camera_info(settings),
            "imageUrl": image_url,
            "detailId": detail_id,
        }

    @app.get(
        "/api/v1/detections/missing-person/results/{detail_id}/image",
        dependencies=[Depends(require_internal_token)],
    )
    def get_detection_result_image(detail_id: int) -> FileResponse:
        result = app.state.detection_results.get(detail_id)
        if result is None or not result.get("image_path"):
            raise HTTPException(status_code=404, detail="탐지 이미지를 찾을 수 없습니다.")
        image_path = _safe_media_path(str(result["image_path"]), settings.api_media_directory)
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="탐지 이미지를 찾을 수 없습니다.")
        return FileResponse(image_path)

    return app
