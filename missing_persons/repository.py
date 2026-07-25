"""실종자 정보를 JSON 파일에 안전하게 저장하는 저장소."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from config import TargetAppearance


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MissingPerson:
    person_id: str
    name: str
    age: int | None = None
    gender: str = ""
    hair_length: str = ""
    hair_color: str = ""
    upper_clothing: str = ""
    upper_color: str = ""
    lower_clothing: str = ""
    lower_color: str = ""
    height_cm: float | None = None
    height_tolerance_cm: float = 10.0
    face_image_path: str | None = None
    body_image_path: str | None = None
    active: bool = True
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MissingPerson":
        return cls(**{field: data.get(field) for field in cls.__dataclass_fields__})

    def to_target(self) -> TargetAppearance:
        return TargetAppearance(
            gender=self.gender,
            age=self.age,
            hair_length=self.hair_length,
            hair_color=self.hair_color,
            upper_clothing=self.upper_clothing,
            upper_color=self.upper_color,
            lower_clothing=self.lower_clothing,
            lower_color=self.lower_color,
            height_cm=self.height_cm,
            height_tolerance_cm=self.height_tolerance_cm,
        )


class MissingPersonRepository:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()

    def _read(self) -> list[MissingPerson]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"실종자 저장소를 읽을 수 없습니다: {self.path}") from error
        if not isinstance(data, list):
            raise RuntimeError("실종자 저장소 형식이 올바르지 않습니다.")
        return [MissingPerson.from_dict(item) for item in data]

    def _write(self, people: list[MissingPerson]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps([asdict(person) for person in people], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def list_all(self) -> list[MissingPerson]:
        with self._lock:
            return self._read()

    def exists(self) -> bool:
        return self.path.exists()

    def list_active(self) -> list[MissingPerson]:
        return [person for person in self.list_all() if person.active]

    def get(self, person_id: str) -> MissingPerson | None:
        return next((person for person in self.list_all() if person.person_id == person_id), None)

    def create(self, values: dict[str, Any]) -> MissingPerson:
        with self._lock:
            timestamp = _now()
            person = MissingPerson(
                person_id=uuid4().hex,
                created_at=timestamp,
                updated_at=timestamp,
                **values,
            )
            people = self._read()
            people.append(person)
            self._write(people)
            return person

    def update(self, person_id: str, values: dict[str, Any]) -> MissingPerson | None:
        with self._lock:
            people = self._read()
            for index, person in enumerate(people):
                if person.person_id == person_id:
                    updated_values = asdict(person)
                    updated_values.update(values)
                    updated_values["updated_at"] = _now()
                    updated = MissingPerson(**updated_values)
                    people[index] = updated
                    self._write(people)
                    return updated
            return None

    def delete(self, person_id: str) -> MissingPerson | None:
        with self._lock:
            people = self._read()
            for person in people:
                if person.person_id == person_id:
                    self._write([item for item in people if item.person_id != person_id])
                    return person
            return None
