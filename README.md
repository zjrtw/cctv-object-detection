# AI 기반 CCTV 실종자 감지 시스템

실시간 CCTV 영상에서 사람을 탐지하고 추적한 뒤, 실종자 후보와 비교해 알림까지 이어지는 Python 기반 AI 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 다음 순서로 동작합니다.

1. 카메라, mp4, RTSP 중 하나에서 영상을 입력받습니다.
2. YOLO11로 사람을 탐지합니다.
3. ByteTrack으로 각 사람을 추적하고 `track_id`를 부여합니다.
4. InsightFace와 Person Re-ID로 실종자 후보와 비교합니다.
5. 조건을 만족하면 캡처 이미지를 저장하고 알림을 전송합니다.

현재 MVP는 로컬 실행과 기본 알림 흐름 확인에 초점을 맞추고 있습니다.

## 개발 환경

- Python 3.13
- Windows 기준 `.venv` 가상환경 사용 권장

## 설치 방법

1. 저장소 루트에서 가상환경을 생성합니다.

```powershell
python -m venv .venv
```

2. 가상환경을 활성화합니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

3. 필요한 패키지를 설치합니다.

```powershell
pip install -r requirements.txt
```

## 실행 방법

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

실행하면 영상 창이 열리고, `q` 키를 누르면 종료됩니다.

## 입력 모드 변경 방법

입력 방식은 `config/settings.yaml`의 `input.type` 값으로 바꿉니다.

```yaml
input:
  type: webcam
  webcam_device_index: 1
  video_file_path: ""
  rtsp_url: ""
```

지원하는 값은 다음과 같습니다.

- `webcam`: 웹캠 입력
- `video_file`: mp4 파일 입력
- `rtsp`: RTSP 스트림 입력

입력 방식별 설정은 다음과 같습니다.

- `webcam`: `webcam_device_index` 사용
- `video_file`: `video_file_path` 사용
- `rtsp`: `rtsp_url` 사용

## 대략적인 폴더 구조

- `main.py`: 전체 실행 흐름을 묶는 진입점
- `camera/`: 영상 입력 처리
- `detector/`: 사람 탐지
- `tracker/`: 객체 추적
- `recognition/`: 외형 및 얼굴 특징 분석
- `matcher/`: 최종 매칭 판단
- `notification/`: 알림 전송 및 중복 방지
- `utils/`: 캡처 저장 같은 공통 기능
- `config/`: 설정 파일과 로더
- `faces/`: 실종자 기준 이미지
- `captures/`: 탐지 시 저장되는 결과 이미지
- `tests/`: 자동화 테스트

## MVP 기능

- 웹캠, mp4, RTSP 입력 처리
- YOLO11 사람 탐지
- ByteTrack 추적
- `track_id` 표시
- InsightFace 기반 얼굴 특징 추출
- Person Re-ID 기반 외형 비교
- 최종 매칭 결과에 따른 캡처 저장
- 알림 전송
- 중복 알림 방지

## 추가 기능

- 실종자 기준 이미지 관리
- 얼굴 특징 기반 비교 고도화
- Re-ID 기준 비교 고도화
- 캡처 파일명에 시간, `track_id`, 판정 정보 포함
- 추후 DB, API 서버, 모바일 알림 연동으로 확장 가능

## 주의사항

- RTSP 계정 정보, 개인 사진, 생성된 캡처 파일, 모델 비밀값은 저장소에 커밋하지 않습니다.
- `config/settings.yaml`의 값은 로컬 환경에 맞게 조정해야 합니다.
- 모델 파일 경로와 입력 소스가 실제 환경과 다르면 실행에 실패할 수 있습니다.

## 참고 문서

- `docs/PRD.md`
- `docs/TASKS.md`

## 외부 실종자 정보 API

API 서버는 CCTV 분석 프로그램과 별도로 실행합니다. 서버를 실행하기 전에 API 키 환경 변수를 설정합니다.

```powershell
$env:MISSING_PERSON_API_KEY = "로컬에서 사용할 안전한 키"
python -m api.server
```

기본 서버 주소는 `http://127.0.0.1:8000`이며, 다른 컴퓨터에서 접속하려면 `config/settings.yaml`의 `api.host`를 `0.0.0.0`으로 변경하고 방화벽에서 8000 포트를 허용해야 합니다.

실종자 등록 요청은 `multipart/form-data` 형식입니다. `payload`에는 JSON 문자열을 넣고, 사진은 선택적으로 `face_image` 또는 `body_image` 필드에 첨부합니다.

```powershell
$payload = '{"name":"홍길동","age":7,"gender":"female","upper_color":"yellow","height_cm":130}'
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/missing-persons" `
  -Method Post `
  -Headers @{ "X-API-Key" = $env:MISSING_PERSON_API_KEY } `
  -Form @{ payload = $payload }
```

지원 API는 다음과 같습니다.

* `GET /health`
* `POST /api/missing-persons`
* `GET /api/missing-persons`
* `GET /api/missing-persons/{person_id}`
* `PUT /api/missing-persons/{person_id}`
* `DELETE /api/missing-persons/{person_id}`
* `POST /api/v1/detections/missing-person`
* `GET /api/v1/detections/missing-person/results/{detail_id}/image`

등록 정보는 `data/missing_persons.json`에 저장되며 CCTV 프로그램은 이 파일을 주기적으로 읽어 활성 실종자 정보를 분석에 사용합니다. 저장된 사진과 정보는 개인정보이므로 외부 공개 경로에 두지 않습니다.

탐지 요청 API는 Spring 백엔드가 내부 토큰으로 호출하는 용도입니다. `X-Internal-Token` 헤더에 `MISSING_PERSON_DETECTION_INTERNAL_TOKEN` 환경 변수 값을 넣어야 합니다.
