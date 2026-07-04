"""Tests for the Flight Deck VFS route layer — creating projects/folders and
uploading files through the HTTP API.

The routes read the on-disk VFS tree under ``<DATA_DIR>/vfs/<user>/``; we point
``DATA_DIR`` at a tmp dir and bypass auth with a fixed user, then drive the
endpoints with a :class:`fastapi.testclient.TestClient`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from captain_claw.flight_deck import vfs_routes


def _make_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FastAPI:
    import captain_claw.flight_deck.server as server

    monkeypatch.setattr(server, "DATA_DIR", tmp_path, raising=False)
    app = FastAPI()
    app.include_router(vfs_routes.router)
    from captain_claw.flight_deck.auth import get_current_user

    app.dependency_overrides[get_current_user] = lambda: {"id": "u1"}
    return app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    return TestClient(_make_app(monkeypatch, tmp_path))


def test_create_project_then_list(client: TestClient) -> None:
    # No projects yet.
    assert client.get("/fd/vfs/projects").json()["projects"] == []

    # A project is a top-level dir: mkdir with an empty inner path creates it.
    r = client.post("/fd/vfs/mkdir", json={"project": "docs", "path": ""})
    assert r.status_code == 200 and r.json()["ok"] is True

    names = [p["name"] for p in client.get("/fd/vfs/projects").json()["projects"]]
    assert names == ["docs"]


def test_upload_files_into_project(client: TestClient) -> None:
    client.post("/fd/vfs/mkdir", json={"project": "docs", "path": ""})

    r = client.post(
        "/fd/vfs/upload",
        data={"project": "docs", "path": ""},
        files=[
            ("files", ("a.txt", b"hello", "text/plain")),
            ("files", ("b.md", b"# hi", "text/markdown")),
        ],
    )
    assert r.status_code == 200
    saved = {f["name"]: f["size"] for f in r.json()["files"]}
    assert saved == {"a.txt": 5, "b.md": 4}

    entries = client.get("/fd/vfs/list", params={"project": "docs", "path": ""}).json()["entries"]
    assert {e["name"] for e in entries} == {"a.txt", "b.md"}
    assert client.get("/fd/vfs/read", params={"project": "docs", "path": "a.txt"}).json()["text"] == "hello"


def test_upload_creates_target_subdir(client: TestClient) -> None:
    # Uploading into a not-yet-existing sub-path creates it (auto-creates project too).
    r = client.post(
        "/fd/vfs/upload",
        data={"project": "docs", "path": "nested/deep"},
        files=[("files", ("note.txt", b"x", "text/plain"))],
    )
    assert r.status_code == 200
    entries = client.get("/fd/vfs/list", params={"project": "docs", "path": "nested/deep"}).json()["entries"]
    assert [e["name"] for e in entries] == ["note.txt"]


def test_upload_strips_directory_from_filename(client: TestClient, tmp_path: Path) -> None:
    """A malicious filename with path components can't escape the target dir."""
    r = client.post(
        "/fd/vfs/upload",
        data={"project": "docs", "path": ""},
        files=[("files", ("../../evil.txt", b"nope", "text/plain"))],
    )
    assert r.status_code == 200
    assert r.json()["files"][0]["name"] == "evil.txt"
    # Landed inside the project, not above the user root.
    assert (tmp_path / "vfs" / "u1" / "docs" / "evil.txt").is_file()
    entries = client.get("/fd/vfs/list", params={"project": "docs", "path": ""}).json()["entries"]
    assert [e["name"] for e in entries] == ["evil.txt"]
