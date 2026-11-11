"""Agent pictures (``/fd/agent-avatars``), through the real routes:

* a PNG, JPEG or WebP goes in and comes back as itself, under an id that is a
  hash of its bytes (the same picture twice is kept once);
* anything else is refused: SVG, a type its bytes don't match, too large, junk;
* each person reads and deletes only their own;
* ``agent-avatar:*`` settings are server-owned: ``GET /fd/settings`` never lists
  them and ``PUT /fd/settings`` can't write one.
"""

from __future__ import annotations

import base64

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from captain_claw.flight_deck import agent_avatar_routes as ar
from captain_claw.flight_deck import auth, settings_routes
from captain_claw.flight_deck.db import FlightDeckDB

ALICE = "user-alice"
BOB = "user-bob"

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
JPEG = b"\xff\xd8\xff\xe0" + b"\x01" * 40
WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x02" * 40


def _data_url(mime: str, raw: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


@pytest.fixture()
async def db(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    d = FlightDeckDB(str(tmp_path / "fd.db"))
    await d.init()
    prev_db = auth._db
    auth.set_auth_db(d)
    now = "2026-01-01T00:00:00Z"
    for uid, email in [(ALICE, "alice@x.co"), (BOB, "bob@x.co")]:
        await d._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (uid, email, "h", uid.title(), "user", now, now),
        )
    await d._db.commit()
    try:
        yield d
    finally:
        auth.set_auth_db(prev_db)
        await d.close()


def _client():
    app = FastAPI()
    app.include_router(ar.router)
    app.include_router(settings_routes.router)
    return TestClient(app, client=("127.0.0.1", 50123))


def _bearer(uid):
    return {"Authorization": f"Bearer {auth.create_access_token(uid, 'user')}"}


@pytest.mark.parametrize("mime,raw", [("image/png", PNG), ("image/jpeg", JPEG), ("image/webp", WEBP)])
async def test_a_picture_goes_in_and_comes_back(db, mime, raw):
    c = _client()
    r = c.post("/fd/agent-avatars", json={"data_url": _data_url(mime, raw)}, headers=_bearer(ALICE))
    assert r.status_code == 200, r.text
    avatar_id = r.json()["id"]
    assert len(avatar_id) == 32

    got = c.get(f"/fd/agent-avatars/{avatar_id}", headers=_bearer(ALICE))
    assert got.status_code == 200
    assert got.content == raw
    assert got.headers["content-type"] == mime
    assert got.headers["x-content-type-options"] == "nosniff"
    assert "immutable" in got.headers["cache-control"]

    # The same bytes again: the same id, kept once.
    again = c.post("/fd/agent-avatars", json={"data_url": _data_url(mime, raw)}, headers=_bearer(ALICE))
    assert again.json()["id"] == avatar_id
    stored = [k for k in (await db.get_all_settings(ALICE)) if k.startswith(ar.SETTING_PREFIX)]
    assert stored == [ar.SETTING_PREFIX + avatar_id]


@pytest.mark.parametrize("data_url,status", [
    (_data_url("image/svg+xml", b"<svg onload='alert(1)'/>"), 400),
    (_data_url("image/png", JPEG), 400),  # its bytes say otherwise
    (_data_url("image/gif", b"GIF89a" + b"\x00" * 20), 400),
    ("data:image/png;base64,@@@not-base64@@@", 400),
    ("data:image/png;base64,", 400),
    ("https://example.com/picture.png", 400),
    (_data_url("image/png", PNG + b"\x00" * (ar.MAX_BYTES + 1)), 413),
])
async def test_anything_else_is_refused(db, data_url, status):
    r = _client().post("/fd/agent-avatars", json={"data_url": data_url}, headers=_bearer(ALICE))
    assert r.status_code == status, r.text
    assert not [k for k in (await db.get_all_settings(ALICE)) if k.startswith(ar.SETTING_PREFIX)]


async def test_each_person_sees_only_their_own(db):
    c = _client()
    avatar_id = c.post("/fd/agent-avatars", json={"data_url": _data_url("image/png", PNG)}, headers=_bearer(ALICE)).json()["id"]
    assert c.get(f"/fd/agent-avatars/{avatar_id}", headers=_bearer(BOB)).status_code == 404
    assert c.get(f"/fd/agent-avatars/{avatar_id}").status_code == 401
    # Bob's delete leaves Alice's picture where it is.
    assert c.delete(f"/fd/agent-avatars/{avatar_id}", headers=_bearer(BOB)).json()["deleted"] is False
    assert c.get(f"/fd/agent-avatars/{avatar_id}", headers=_bearer(ALICE)).status_code == 200
    assert c.delete(f"/fd/agent-avatars/{avatar_id}", headers=_bearer(ALICE)).json()["deleted"] is True
    assert c.get(f"/fd/agent-avatars/{avatar_id}", headers=_bearer(ALICE)).status_code == 404


async def test_ids_are_checked(db):
    c = _client()
    for bad in ("x" * 32, "../../etc", "ABCDEF0123456789ABCDEF0123456789"):
        assert c.get(f"/fd/agent-avatars/{bad}", headers=_bearer(ALICE)).status_code == 404


async def test_pictures_stay_out_of_settings(db):
    c = _client()
    avatar_id = c.post("/fd/agent-avatars", json={"data_url": _data_url("image/png", PNG)}, headers=_bearer(ALICE)).json()["id"]
    c.put("/fd/settings", json={"settings": {"fd:agent-avatars": f'{{"proc-a": {{"image": "{avatar_id}"}}}}'}}, headers=_bearer(ALICE))
    listed = c.get("/fd/settings", headers=_bearer(ALICE)).json()
    assert "fd:agent-avatars" in listed
    assert not [k for k in listed if k.startswith(ar.SETTING_PREFIX)]
    # …and can't be written around the checks.
    r = c.put("/fd/settings", json={"settings": {ar.SETTING_PREFIX + "0" * 32: "image/svg+xml;PHN2Zz4="}}, headers=_bearer(ALICE))
    assert r.status_code == 400
