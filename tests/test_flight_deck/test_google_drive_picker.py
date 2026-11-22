"""Pick files from Google Drive: what the page gets, for whom, and when.

With the per-file Drive scope (the default) an agent opens only files the app
created or the person picked through the Google Picker; a pasted link to the
person's own Sheet otherwise 404s (cohort001, 2026-11-19: "Google račun je
povezan bez punog pristupa Drive-u"). /fd/google/picker gives the page what
the Picker needs — never a token: the page asks Google for a file-only one.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import captain_claw.flight_deck.google_oauth_routes as gr
from captain_claw.flight_deck import agent_secret, auth
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.google_oauth import GoogleOAuthTokens

ALICE = "user-alice"   # admin
BOB = "user-bob"
FILE = "https://www.googleapis.com/auth/drive.file"
FULL = "https://www.googleapis.com/auth/drive"
CLIENT_ID = "123456789012-abcdef.apps.googleusercontent.com"


def _tokens(scope: str) -> GoogleOAuthTokens:
    return GoogleOAuthTokens(access_token="stored-access", refresh_token="stored-refresh",
                             token_type="Bearer", expires_at=time.time() + 3600, scope=scope)


@pytest.fixture()
async def db(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    for var in ("FD_LOCKDOWN", "FD_PUBLIC_URL", "FD_AGENT_SHARED_SECRET", "FD_COOKIE_SECURE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    agent_secret.reset_cache_for_tests()
    d = FlightDeckDB(str(tmp_path / "fd.db"))
    await d.init()
    prev = auth._db
    auth.set_auth_db(d)
    now = "2026-01-01T00:00:00Z"
    for uid, email, role in [(ALICE, "alice@x.co", "admin"), (BOB, "bob@x.co", "user")]:
        await d._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)", (uid, email, "h", uid.title(), role, now, now))
    await d._db.commit()
    await d.set_system_setting(gr._K_CLIENT_ID, CLIENT_ID)
    await d.set_system_setting(gr._K_CLIENT_SECRET, "csecret")
    await d.set_system_setting(gr._K_SCOPES, json.dumps(["openid", "email", FILE]))
    gr._primary_owner_cache.update(id=None, at=0.0)
    try:
        yield d
    finally:
        gr._primary_owner_cache.update(id=None, at=0.0)
        agent_secret.reset_cache_for_tests()
        auth.set_auth_db(prev)
        await d.close()


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(gr.router)
    return TestClient(app, client=("127.0.0.1", 50123))


def _as(uid: str, role: str = "user") -> dict:
    return {"Authorization": f"Bearer {auth.create_access_token(uid, role)}"}


async def _connect(db, uid: str, scope: str) -> None:
    await gr._store_tokens(db, uid, _tokens(scope))
    await gr._store_user(db, uid, {"email": f"{uid}@gmail.com", "name": uid})


async def test_ready_for_a_per_file_connection_once_the_key_is_set(db):
    await _connect(db, BOB, f"openid email {FILE}")
    c = _client()
    assert c.get("/fd/google/picker", headers=_as(BOB)).json() == {"ready": False, "reason": "no_api_key"}

    r = c.post("/fd/google/config", headers=_as(ALICE, "admin"), json={"picker_api_key": " AIzaKey "})
    assert r.status_code == 200
    got = c.get("/fd/google/picker", headers=_as(BOB)).json()
    assert got == {
        "ready": True, "reason": "", "client_id": CLIENT_ID, "app_id": "123456789012",
        "api_key": "AIzaKey", "scope": FILE, "login_hint": f"{BOB}@gmail.com",
    }
    # Never a token: the page asks Google for a file-only one itself.
    assert "stored-access" not in json.dumps(got) and "stored-refresh" not in json.dumps(got)
    # Saving the key disconnected no one.
    assert await gr._load_tokens(db, BOB) is not None
    assert c.get("/fd/google/config", headers=_as(ALICE, "admin")).json()["picker_api_key_set"] is True


async def test_not_offered_where_it_adds_nothing_or_cant_work(db):
    c = _client()
    await db.set_system_setting(gr._K_PICKER_KEY, "AIzaKey")
    assert c.get("/fd/google/picker", headers=_as(BOB)).json()["reason"] == "not_connected"
    await _connect(db, BOB, f"openid email {FULL}")
    assert c.get("/fd/google/picker", headers=_as(BOB)).json()["reason"] == "full_drive"
    await _connect(db, BOB, "openid email https://www.googleapis.com/auth/gmail.readonly")
    assert c.get("/fd/google/picker", headers=_as(BOB)).json()["reason"] == "no_drive_scope"
    await _connect(db, BOB, f"openid email {FILE}")
    await db.set_system_setting(gr._K_CLIENT_ID, "not-a-numbered-client")
    assert c.get("/fd/google/picker", headers=_as(BOB)).json()["reason"] == "no_app_id"


async def test_only_an_admin_sets_the_key_and_empty_removes_it(db):
    c = _client()
    assert c.post("/fd/google/config", headers=_as(BOB), json={"picker_api_key": "x"}).status_code == 403
    c.post("/fd/google/config", headers=_as(ALICE, "admin"), json={"picker_api_key": "AIzaKey"})
    c.post("/fd/google/config", headers=_as(ALICE, "admin"), json={"picker_api_key": ""})
    assert (await db.get_system_setting(gr._K_PICKER_KEY)) == ""
    assert c.get("/fd/google/config", headers=_as(ALICE, "admin")).json()["picker_api_key_set"] is False


async def test_each_user_gets_their_own_account_as_the_hint(db):
    await db.set_system_setting(gr._K_PICKER_KEY, "AIzaKey")
    await _connect(db, ALICE, f"openid email {FILE}")
    await _connect(db, BOB, f"openid email {FILE}")
    c = _client()
    assert c.get("/fd/google/picker", headers=_as(ALICE, "admin")).json()["login_hint"] == f"{ALICE}@gmail.com"
    assert c.get("/fd/google/picker", headers=_as(BOB)).json()["login_hint"] == f"{BOB}@gmail.com"
    assert c.get("/fd/google/picker").status_code == 401
