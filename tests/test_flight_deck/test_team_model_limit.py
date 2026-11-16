"""The team's tiers pick a non-admin's model: an agent on them keeps the model an
admin gave it.

A kiosk user switched their agent from the team's cheap model to the pricey one
with the chat box's model picker (cohort001, 2026-11). The team pays for its
tiers — its key, or a server of its own that needs none — so for a non-admin's
agent on them:

* the agent is told the only model it may run (``limit`` in
  ``/fd/model-switch/agent/models``) — the one its config names;
* Agent settings (``PUT /fd/agent-model``) and the raw config editor
  (``PUT /fd/agent-config``) refuse any other model, a team one too, unless
  the owner gives a key of their own; the config read says it is locked;
* nor can an agent move onto the team's key with a model the tiers don't run;
* an admin's agent, an admin editing one, and an agent on its owner's own key
  are free as before.

Real FlightDeckDB + process registry in tmp dirs; ``DATA_DIR`` is never the
real ``./fd-data``.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import yaml

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import google_oauth_routes, rate_limiter, server
from captain_claw.flight_deck.auth import ACT_AS_HEADER, create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

ADMIN = "user-admin"
BOB = "user-bob"
REMOTE = ("203.0.113.9", 40001)
BASE = "https://api.deepseek.com"
TEAM_KEY = "sk-team-deepseek-0123456789abcd"
OWN_KEY = "sk-own-deepseek-9876543210wxyz"
# The team's own model server: no key at all — and one of the owner's.
LOCAL = "http://10.0.0.5:8080/v1"
OWN_SERVER = "http://192.168.1.9:8080/v1"


def _config(model: str, base: str = BASE) -> str:
    return f"model:\n  provider: openai\n  model: {model}\n  api_key: ''\n  base_url: {base}\n"


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Auth-enabled deck whose team tiers all run deepseek-flash on the team's key."""
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    for uid, role in ((ADMIN, "admin"), (BOB, "user")):
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, ?, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", uid, role))
    await db._db.commit()
    await db.set_system_setting("fd:endpoint-keys", json.dumps({f"openai|{BASE}": TEAM_KEY}))
    tier = {"provider": "openai", "model": "deepseek-flash", "base_url": BASE, "api_key": "@system"}
    local = {"provider": "openai", "model": "qwen-flash", "base_url": LOCAL, "api_key": ""}
    await db.set_system_setting("fd:shared-tier-sets", json.dumps({
        "sets": [{"id": "s1", "name": "Team", "tiers": {
            "fast": tier, "reason": {**tier, "model": "deepseek-chat"}, "local": local}}],
        "defaultSetId": "s1"}))

    data = tmp_path / "fd-data"
    data.mkdir()
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", data / ".processes.json")
    monkeypatch.setattr(server, "_processes", {})
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                 "XAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())
    # The users table lives in the test db, not DATA_DIR/flight-deck.db.
    monkeypatch.setattr(server, "_owner_is_admin", lambda uid: uid == ADMIN)

    server._save_process_registry({
        "bob-agent": {"slug": "bob-agent", "name": "Bob's agent", "web_port": 24101,
                      "web_auth": "bob-tok", "owner": BOB, "pid": None},
        "admin-agent": {"slug": "admin-agent", "name": "Admin's agent", "web_port": 24102,
                        "web_auth": "admin-tok", "owner": ADMIN, "pid": None},
    })
    for slug in ("bob-agent", "admin-agent"):
        (data / slug).mkdir()
        (data / slug / "config.yaml").write_text(_config("deepseek-flash"))
        (data / slug / ".env").write_text(f"OPENAI_API_KEY={TEAM_KEY}\n")
    try:
        yield db
    finally:
        await db.close()
        fd_auth._db = prev


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app, client=REMOTE), base_url="http://fd.test")


def _hdr(caller: str, act_as: str | None = None) -> dict:
    h = {"Authorization": f"Bearer {create_access_token(caller)}"}
    if act_as is not None:
        h[ACT_AS_HEADER] = act_as
    return h


def _model(slug: str) -> str:
    return yaml.safe_load((server.DATA_DIR / slug / "config.yaml").read_text())["model"]["model"]


async def test_a_non_admin_s_agent_on_the_team_s_key_keeps_its_model(deck):
    # …not every team model at the endpoint (deepseek-chat is one too).
    assert await server.team_model_limit("bob-agent", BOB) == ["deepseek-flash"]
    assert await server.team_model_limit("admin-agent", ADMIN) is None
    assert await server.team_model_limit("bob-agent", "") is None  # nobody recorded


async def test_an_agent_on_its_owner_s_own_key_is_not(deck):
    (server.DATA_DIR / "bob-agent" / ".env").write_text(f"OPENAI_API_KEY={OWN_KEY}\n")
    assert await server.team_model_limit("bob-agent", BOB) is None


async def test_a_blank_key_rides_the_team_s(deck):
    (server.DATA_DIR / "bob-agent" / ".env").write_text("")
    assert await server.team_model_limit("bob-agent", BOB) == ["deepseek-flash"]


async def test_a_team_server_with_no_key_is_the_team_s_tiers_too(deck):
    (server.DATA_DIR / "bob-agent" / "config.yaml").write_text(_config("qwen-flash", LOCAL))
    (server.DATA_DIR / "bob-agent" / ".env").write_text("")
    assert await server.team_model_limit("bob-agent", BOB) == ["qwen-flash"]
    async with _client() as c:
        r = await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(BOB),
                        json={"provider": "openai", "model": "qwen-max"})
    assert r.status_code == 403, r.text
    assert "team's tiers" in r.json()["detail"] and _model("bob-agent") == "qwen-flash"


async def test_any_model_on_the_team_s_server_is_held_and_a_server_of_their_own_is_free(deck):
    (server.DATA_DIR / "bob-agent" / "config.yaml").write_text(_config("qwen-max", LOCAL))
    (server.DATA_DIR / "bob-agent" / ".env").write_text("")
    assert await server.team_model_limit("bob-agent", BOB) == ["qwen-max"]
    async with _client() as c:
        # Off to a keyless server the team's tiers don't use: theirs to run…
        r = await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(BOB),
                        json={"provider": "openai", "model": "llama-big", "base_url": OWN_SERVER})
        assert r.status_code == 200, r.text
        assert await server.team_model_limit("bob-agent", BOB) is None
        # …but not back onto the team's with a model its tiers don't run there.
        r = await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(BOB),
                        json={"provider": "openai", "model": "llama-big", "base_url": LOCAL})
        assert r.status_code == 403 and "isn't one of your team's models" in r.json()["detail"]
        r = await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(BOB),
                        json={"provider": "openai", "model": "qwen-flash", "base_url": LOCAL})
        assert r.status_code == 200, r.text


async def test_the_config_read_says_the_model_is_locked(deck):
    async with _client() as c:
        bob = await c.get("/fd/agent-config/process/bob-agent", headers=_hdr(BOB))
        admin = await c.get("/fd/agent-config/process/bob-agent", headers=_hdr(ADMIN, BOB))
        (server.DATA_DIR / "bob-agent" / ".env").write_text(f"OPENAI_API_KEY={OWN_KEY}\n")
        own = await c.get("/fd/agent-config/process/bob-agent", headers=_hdr(BOB))
    assert bob.status_code == 200 and "team's tiers" in bob.json()["model_locked"]
    assert admin.status_code == 200 and "model_locked" not in admin.json()
    assert own.status_code == 200 and "model_locked" not in own.json()


async def test_the_agent_is_told_its_limit(deck, monkeypatch):
    monkeypatch.setattr(google_oauth_routes, "_authorize_agent_call", lambda _r: None)
    params = {"provider": "openai", "base_url": BASE}
    async with _client() as c:
        bob = await c.get("/fd/model-switch/agent/models", params=params, headers={"X-Agent-Auth": "bob-tok"})
        admin = await c.get("/fd/model-switch/agent/models", params=params, headers={"X-Agent-Auth": "admin-tok"})
    assert bob.status_code == 200 and bob.json() == {"models": [], "limit": ["deepseek-flash"]}
    assert admin.status_code == 200 and admin.json() == {"models": []}


async def test_agent_settings_keep_the_model_on_the_team_s_tiers(deck):
    async with _client() as c:
        for other in ("deepseek-v4-pro", "deepseek-chat"):  # a team model is another model too
            r = await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(BOB),
                            json={"provider": "openai", "model": other})
            assert r.status_code == 403, r.text
            assert "team's tiers" in r.json()["detail"] and _model("bob-agent") == "deepseek-flash"
        # Saving the model it has is fine (a vision change rides along)…
        r = await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(BOB),
                        json={"provider": "openai", "model": "deepseek-flash", "vision": "off"})
        assert r.status_code == 200, r.text
        # …so is any model on a key of their own.
        r = await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(BOB),
                        json={"provider": "openai", "model": "deepseek-v4-pro", "api_key": OWN_KEY})
        assert r.status_code == 200, r.text
        assert _model("bob-agent") == "deepseek-v4-pro"


async def test_an_own_key_agent_can_t_bring_another_model_onto_the_team_s_key(deck):
    (server.DATA_DIR / "bob-agent" / ".env").write_text(f"OPENAI_API_KEY={OWN_KEY}\n")
    async with _client() as c:
        r = await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(BOB),
                        json={"provider": "openai", "model": "deepseek-v4-pro", "api_key": TEAM_KEY})
        assert r.status_code == 403, r.text
        assert "isn't one of your team's models" in r.json()["detail"]
        # …a team model is the team's choice, so that move is fine.
        r = await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(BOB),
                        json={"provider": "openai", "model": "deepseek-chat", "api_key": TEAM_KEY})
        assert r.status_code == 200, r.text


async def test_an_admin_picks_any_model(deck):
    async with _client() as c:
        own = await c.put("/fd/agent-model/process/admin-agent", headers=_hdr(ADMIN),
                          json={"provider": "openai", "model": "deepseek-v4-pro"})
        for_bob = await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(ADMIN, BOB),
                              json={"provider": "openai", "model": "deepseek-v4-pro"})
    assert own.status_code == 200, own.text
    assert for_bob.status_code == 200, for_bob.text
    assert _model("bob-agent") == "deepseek-v4-pro"


async def test_the_config_editor_holds_a_changed_model_only(deck):
    async with _client() as c:
        r = await c.put("/fd/agent-config/process/bob-agent", headers=_hdr(BOB),
                        json={"config_yaml": _config("deepseek-v4-pro")})
        assert r.status_code == 403, r.text
        assert _model("bob-agent") == "deepseek-flash"
        # With a key of their own in the same save, it's theirs.
        r = await c.put("/fd/agent-config/process/bob-agent", headers=_hdr(BOB),
                        json={"config_yaml": _config("deepseek-chat"), "env": f"OPENAI_API_KEY={OWN_KEY}\n"})
        assert r.status_code == 200, r.text
        (server.DATA_DIR / "bob-agent" / ".env").write_text(f"OPENAI_API_KEY={TEAM_KEY}\n")
        # An agent an admin already put on another model keeps the rest editable.
        (server.DATA_DIR / "bob-agent" / "config.yaml").write_text(_config("deepseek-v4-pro"))
        r = await c.put("/fd/agent-config/process/bob-agent", headers=_hdr(BOB),
                        json={"config_yaml": _config("deepseek-v4-pro") + "context:\n  max_tokens: 120000\n"})
        assert r.status_code == 200, r.text
