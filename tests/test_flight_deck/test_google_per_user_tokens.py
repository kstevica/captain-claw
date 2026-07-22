"""Per-user Google OAuth tokens.

Google used to be one deployment-wide connection: system_settings had no
user_id, so every FD user shared one account and any user's connect overwrote
everyone's. These tests pin the fix — tokens are per-user, an agent gets its
OWNER's token (never a claimed one), and an existing single-user deployment
keeps working via a legacy-global fallback for the primary owner.

Route handlers are called directly (as the other FD route tests do), so no ASGI
event-loop entanglement; the db is closed in teardown (aiosqlite's non-daemon
thread otherwise hangs the process).
"""

import tempfile
import time
from pathlib import Path

import pytest

import captain_claw.flight_deck.google_oauth_routes as gr
from captain_claw.flight_deck import auth
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.google_oauth import GoogleOAuthTokens

ALICE = "user-alice"
BOB = "user-bob"


def _tokens(access="A-tok", refresh="A-refresh", scope="openid https://www.googleapis.com/auth/drive.readonly"):
    # Far-future expiry so _refresh_if_needed never makes a network call.
    return GoogleOAuthTokens(access_token=access, refresh_token=refresh,
                             token_type="Bearer", expires_at=time.time() + 3600, scope=scope)


@pytest.fixture()
async def db(monkeypatch):
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    d = FlightDeckDB(str(Path(tmp) / "fd.db"))
    await d.init()
    auth.set_auth_db(d)
    # Two users with fixed ids (create_user mints its own UUID; the per-user
    # token rows just need valid FKs, and the tests key off known ids).
    now = "2026-01-01T00:00:00Z"
    for uid, email, role in [(ALICE, "alice@x.co", "admin"), (BOB, "bob@x.co", "user")]:
        await d._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (uid, email, "h", uid.title(), role, now, now),
        )
    await d._db.commit()
    # A configured OAuth *client* (deployment-wide) so _token_client resolves.
    await d.set_system_setting(gr._K_CLIENT_ID, "cid")
    await d.set_system_setting(gr._K_CLIENT_SECRET, "csecret")
    # Deterministic primary owner + reset the module cache between tests.
    gr._primary_owner_cache["id"] = ALICE
    try:
        yield d
    finally:
        gr._primary_owner_cache["id"] = None
        await d.close()


class TestPerUserStorage:
    async def test_tokens_are_isolated_between_users(self, db):
        await gr._store_tokens(db, ALICE, _tokens(access="alice"))
        await gr._store_tokens(db, BOB, _tokens(access="bob"))
        a = await gr._load_tokens(db, ALICE)
        b = await gr._load_tokens(db, BOB)
        assert a.access_token == "alice"
        assert b.access_token == "bob"

    async def test_one_user_unconnected_while_another_is(self, db):
        await gr._store_tokens(db, ALICE, _tokens())
        assert await gr._load_tokens(db, ALICE) is not None
        assert await gr._load_tokens(db, BOB) is None  # bob never connected

    async def test_clear_is_per_user(self, db):
        await gr._store_tokens(db, ALICE, _tokens())
        await gr._store_tokens(db, BOB, _tokens())
        await gr._clear_oauth_state(db, BOB)
        assert await gr._load_tokens(db, BOB) is None
        assert await gr._load_tokens(db, ALICE) is not None  # untouched


class TestLegacyGlobalFallback:
    """A deployment that connected before per-user storage keeps working: the
    primary owner transparently reads the old global token, nobody else does."""

    async def test_primary_owner_inherits_global_token(self, db):
        import json
        await db.set_system_setting(gr._K_TOKENS, json.dumps(_tokens(access="legacy").to_dict()))
        got = await gr._load_tokens(db, ALICE)  # ALICE is primary owner
        assert got is not None and got.access_token == "legacy"

    async def test_second_user_does_not_see_the_global_token(self, db):
        import json
        await db.set_system_setting(gr._K_TOKENS, json.dumps(_tokens(access="legacy").to_dict()))
        assert await gr._load_tokens(db, BOB) is None

    async def test_per_user_write_shadows_the_global_fallback(self, db):
        import json
        await db.set_system_setting(gr._K_TOKENS, json.dumps(_tokens(access="legacy").to_dict()))
        await gr._store_tokens(db, ALICE, _tokens(access="fresh"))
        got = await gr._load_tokens(db, ALICE)
        assert got.access_token == "fresh"  # per-user row wins

    async def test_clearing_primary_owner_also_clears_global(self, db):
        import json
        await db.set_system_setting(gr._K_TOKENS, json.dumps(_tokens(access="legacy").to_dict()))
        await gr._clear_oauth_state(db, ALICE)
        assert await gr._load_tokens(db, ALICE) is None  # global fallback also gone


class TestClientRotationClearsEveryone:
    """Rotating the deployment's OAuth client / scopes invalidates ALL tokens —
    a refresh token is bound to the client that minted it."""

    async def test_clear_all_disconnects_every_user(self, db):
        await gr._store_tokens(db, ALICE, _tokens())
        await gr._store_tokens(db, BOB, _tokens())
        await gr._clear_all_oauth_state(db)
        assert await gr._load_tokens(db, ALICE) is None
        assert await gr._load_tokens(db, BOB) is None


class TestAgentOwnerResolution:
    """An agent's Google token follows its OWNER, resolved from FD's records —
    never from what the agent claims."""

    def _req(self, headers=None, port=0):
        import types
        return types.SimpleNamespace(
            headers=headers or {},
            client=types.SimpleNamespace(host="127.0.0.1", port=port),
        )

    async def test_web_auth_token_maps_to_owner(self, db, monkeypatch):
        import captain_claw.flight_deck.server as srv
        monkeypatch.setattr(srv, "_resolve_agent_owner_by_auth",
                            lambda t: {"tok-alice": ALICE, "tok-bob": BOB}.get(t, ""))
        owner = await gr._agent_owner(self._req({"X-Agent-Auth": "tok-bob"}))
        assert owner == BOB

    async def test_falls_back_to_primary_owner(self, db, monkeypatch):
        import captain_claw.flight_deck.server as srv
        monkeypatch.setattr(srv, "_resolve_agent_owner_by_auth", lambda t: "")
        monkeypatch.setattr(srv, "_resolve_agent_owner", lambda p: "")
        owner = await gr._agent_owner(self._req({}))  # nothing resolvable
        assert owner == ALICE  # primary owner


class TestAgentAccessTokenRoute:
    """The /access_token endpoint returns the calling agent's owner's token."""

    def _req(self, token):
        import types
        return types.SimpleNamespace(
            headers={"X-Agent-Auth": token},
            client=types.SimpleNamespace(host="127.0.0.1", port=0),
        )

    async def test_agent_gets_its_owners_token(self, db, monkeypatch):
        import captain_claw.flight_deck.server as srv
        monkeypatch.setattr(srv, "_resolve_agent_owner_by_auth",
                            lambda t: {"tok-alice": ALICE, "tok-bob": BOB}.get(t, ""))
        monkeypatch.setenv("FD_AGENT_SHARED_SECRET", "")  # loopback gate
        await gr._store_tokens(db, ALICE, _tokens(access="alice-only"))

        out = await gr.google_access_token(self._req("tok-alice"))
        assert out["access_token"] == "alice-only"

    async def test_agent_for_unconnected_owner_gets_404(self, db, monkeypatch):
        import captain_claw.flight_deck.server as srv
        from fastapi import HTTPException
        monkeypatch.setattr(srv, "_resolve_agent_owner_by_auth",
                            lambda t: {"tok-alice": ALICE, "tok-bob": BOB}.get(t, ""))
        monkeypatch.setenv("FD_AGENT_SHARED_SECRET", "")
        await gr._store_tokens(db, ALICE, _tokens())  # only alice connected
        with pytest.raises(HTTPException) as exc:
            await gr.google_access_token(self._req("tok-bob"))
        assert exc.value.status_code == 404  # bob has no tokens


class TestStatusRoute:
    async def test_status_is_per_user(self, db):
        await gr._store_tokens(db, ALICE, _tokens())
        import types
        req = types.SimpleNamespace(url=types.SimpleNamespace(scheme="http", netloc="x"),
                                    headers={}, base_url="http://x/")

        # Patch redirect-uri building (needs a real request); assert on connected.
        alice_status = await gr.google_status(req, {"id": ALICE})
        bob_status = await gr.google_status(req, {"id": BOB})
        assert alice_status["connected"] is True
        assert bob_status["connected"] is False
