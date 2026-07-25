from missing_persons import MissingPersonRepository


def test_repository_persists_and_updates_person(tmp_path) -> None:
    path = tmp_path / "missing_persons.json"
    repository = MissingPersonRepository(path)

    created = repository.create({"name": "저장 테스트", "age": 7})
    updated = repository.update(created.person_id, {"active": False})

    reloaded = MissingPersonRepository(path).get(created.person_id)
    assert updated is not None
    assert updated.active is False
    assert reloaded is not None
    assert reloaded.name == "저장 테스트"


def test_repository_active_filter(tmp_path) -> None:
    repository = MissingPersonRepository(tmp_path / "missing_persons.json")
    active = repository.create({"name": "활성"})
    inactive = repository.create({"name": "비활성", "active": False})

    assert [person.person_id for person in repository.list_active()] == [active.person_id]
    assert repository.get(inactive.person_id) is not None
