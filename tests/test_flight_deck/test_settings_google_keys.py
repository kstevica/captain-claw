"""/fd/settings never exposes or accepts the server-owned ``google_oauth:*`` keys.

They share the per-user settings store with the SPA's ``fd:*`` preferences but
belong to the Google OAuth routes alone: readable here, any script running as
the user would get their Google refresh token; writable, a user could forge a
Google identity (e.g. to make another user's Disconnect skip the revoke) or
drop their connection without the revoke.
"""

import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import captain_claw.flight_deck.google_oauth_routes as gr
from captain_claw.flight_deck import auth
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.flight_deck.settings_routes import router
from captain_claw.google_oauth import GoogleOAuthTokens

ALICE = "user-alice"
BOB = "user-bob"


@pytest.fixture()
async def db(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    d = FlightDeckDB(str(tmp_path / "fd.db"))
    await d.init()
    prev_db = auth._db
    auth.set_auth_db(d)
    now = "2026-01-01T00:00:00Z"
    for uid, email, role in [(ALICE, "alice@x.co", "admin"), (BOB, "bob@x.co", "user")]:
        await d._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (uid, email, "h", uid.title(), role, now, now),
        )
    await d._db.commit()
    gr._primary_owner_cache.update(id=ALICE, at=time.time())
    try:
        yield d
    finally:
        gr._primary_owner_cache.update(id=None, at=0.0)
        auth.set_auth_db(prev_db)
        await d.close()


def _client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _as(uid):
    return {"Authorization": f"Bearer {auth.create_access_token(uid, 'user')}"}


async def _connect(db, uid, access, sub):
    await gr._store_tokens(db, uid, GoogleOAuthTokens(
        access_token=access, refresh_token=f"{access}-REFRESH", token_type="Bearer",
        expires_at=time.time() + 3600, scope="openid email"))
    await gr._store_user(db, uid, {"sub": sub, "email": f"{uid}@gmail.com"})


class TestGoogleKeysAreServerOwned:
    async def test_get_hides_them_but_returns_everything_else(self, db):
        await _connect(db, BOB, "BOB-ACCESS", "g-bob")
        await db.set_settings(BOB, {"fd:theme": "dark"})
        r = _client().get("/fd/settings", headers=_as(BOB))
        assert r.status_code == 200
        assert r.json() == {"fd:theme": "dark"}
        assert "REFRESH" not in r.text and "BOB-ACCESS" not in r.text

    @pytest.mark.parametrize("key", [gr._K_TOKENS, gr._K_USER, "Google_OAuth:tokens"])
    async def test_put_refuses_them_and_writes_nothing(self, db, key):
        await _connect(db, BOB, "BOB-ACCESS", "g-bob")
        forged = {key: json.dumps({"email": "alice@corp.com"}), "fd:theme": "light"}
        r = _client().put("/fd/settings", headers=_as(BOB), json={"settings": forged})
        assert r.status_code == 400
        # Nothing from the request was applied — not even the harmless key.
        assert await db.get_setting(BOB, "fd:theme") is None
        assert (await gr._load_user(db, BOB))["sub"] == "g-bob"
        assert (await gr._load_tokens(db, BOB)).access_token == "BOB-ACCESS"

    @pytest.mark.parametrize("key", [gr._K_TOKENS, gr._K_USER])
    async def test_delete_refuses_them(self, db, key):
        await _connect(db, BOB, "BOB-ACCESS", "g-bob")
        r = _client().delete(f"/fd/settings/{key}", headers=_as(BOB))
        assert r.status_code == 400
        assert (await gr._load_tokens(db, BOB)).access_token == "BOB-ACCESS"
        assert (await gr._load_user(db, BOB))["sub"] == "g-bob"

    async def test_ordinary_keys_still_round_trip(self, db):
        c = _client()
        r = c.put("/fd/settings", headers=_as(BOB), json={"settings": {"fd:theme": "dark"}})
        assert r.status_code == 200 and r.json() == {"ok": True, "count": 1}
        assert c.get("/fd/settings", headers=_as(BOB)).json() == {"fd:theme": "dark"}
        assert c.delete("/fd/settings/fd:theme", headers=_as(BOB)).json() == {"ok": True}
        assert c.get("/fd/settings", headers=_as(BOB)).json() == {}

    async def test_cannot_forge_a_shared_account_to_block_anothers_revoke(
        self, db, monkeypatch
    ):
        # The review's scenario end to end: B tries to plant A's Google identity
        # in B's own settings so A's Disconnect thinks the grant is still in use.
        revoked = []

        async def revoke(token):
            revoked.append(token)
            return True

        monkeypatch.setattr(gr, "revoke_token", revoke)
        await _connect(db, ALICE, "ALICE-ACCESS", "g-alice")
        forged = {
            gr._K_USER: json.dumps({"sub": "g-alice", "email": "alice@gmail.com"}),
            gr._K_TOKENS: json.dumps({"access_token": "x", "refresh_token": "x"}),
        }
        c = _client()
        assert c.put("/fd/settings", headers=_as(BOB),
                     json={"settings": forged}).status_code == 400

        app = FastAPI()
        app.include_router(gr.router)
        r = TestClient(app).post("/fd/google/logout", headers=_as(ALICE))
        assert r.json() == {"disconnected": True}
        assert revoked == ["ALICE-ACCESS-REFRESH"]
