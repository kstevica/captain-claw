"""Retired tools (``gws``) never reach a Flight Deck agent.

The ``gws`` Google Workspace CLI wrapper is retired: agents use the native
``google_drive`` / ``google_calendar`` / ``google_mail`` tools. Its ``raw``
passthrough could also send Gmail with the owner's token, past the FD send gate
(``POST /fd/google/gmail/send``). Pinned here, on the Flight Deck side:

* the Old Man preset lists the native tools, not ``gws``;
* both config writers (Docker and process) drop ``gws`` from ``tools.enabled``,
  whatever the spawn request or a stored archetype still lists;
* ``_resolve_archetype`` drops it when it copies an archetype's tools;
* archetype create / update store no ``gws``, and generated / forged drafts
  come back without it;
* a row stored before the retirement is listed without it (``GET
  /fd/archetypes`` and ``/mine``), and Agent Forge neither shows it in the
  archetype catalog it prompts with nor returns it in a forged team;
* the spawn env still scrubs an operator's ambient gws credentials, so a stray
  binary on PATH can't act as the operator's Google account.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi import FastAPI

import captain_claw.flight_deck.server as server
from captain_claw.config import RETIRED_TOOLS
from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck.db import FlightDeckDB

_NATIVE = ("google_drive", "google_calendar", "google_mail")
_WITH_GWS = ["read", "gws", "google_drive", "web_search"]


def test_gws_is_retired():
    assert "gws" in RETIRED_TOOLS


# ── Old Man preset ──────────────────────────────────────────────────────────


def test_old_man_preset_has_native_google_tools_and_no_gws():
    assert "gws" not in server.OLD_MAN_TOOLS
    for name in _NATIVE:
        assert server.OLD_MAN_TOOLS.count(name) == 1, name
    cfg = server._build_old_man_config()
    assert "gws" not in cfg.tools


# ── the two config writers ──────────────────────────────────────────────────


def _enabled(config_yaml: str) -> list[str]:
    return yaml.safe_load(config_yaml)["tools"]["enabled"]


def test_docker_config_yaml_drops_gws():
    c = server.AgentConfig(name="x", tools=list(_WITH_GWS))
    assert _enabled(server._build_config_yaml(c)) == ["read", "google_drive", "web_search"]
    assert c.tools == _WITH_GWS  # the caller's config is left alone


def test_process_config_yaml_drops_gws(tmp_path: Path):
    c = server.AgentConfig(name="x", tools=list(_WITH_GWS))
    out = server._build_process_config_yaml(c, tmp_path)
    assert _enabled(out) == ["read", "google_drive", "web_search"]


def test_config_writers_keep_a_list_without_retired_tools(tmp_path: Path):
    c = server.AgentConfig(name="x", tools=["read", "google_mail"])
    assert _enabled(server._build_config_yaml(c)) == ["read", "google_mail"]
    assert _enabled(server._build_process_config_yaml(c, tmp_path)) == ["read", "google_mail"]


# ── _resolve_archetype ──────────────────────────────────────────────────────


@pytest.fixture
def archetype_registry(monkeypatch: pytest.MonkeyPatch):
    """Stub the resolver's registry / DB / owner-tier seams (as in
    test_spawn_archetype); set ``.archetypes`` to what it should see."""
    import captain_claw.flight_deck.archetypes as arch_mod
    import captain_claw.flight_deck.basna_routes as basna_mod

    state = types.SimpleNamespace(archetypes=[])

    async def fake_merged(db, uid):
        return list(state.archetypes)

    async def fake_owner_tiers(db, uid):
        return {}, []

    monkeypatch.setattr(arch_mod, "merged_archetypes", fake_merged)
    monkeypatch.setattr(fd_auth, "get_db", lambda: object())
    monkeypatch.setattr(basna_mod, "_load_owner_tiers", fake_owner_tiers)
    return state


def _req():
    return types.SimpleNamespace(state=types.SimpleNamespace(user_id=""))


async def test_resolve_archetype_drops_gws_from_stored_tools(archetype_registry):
    archetype_registry.archetypes = [{
        "id": "drive-clerk", "role": "Drive Clerk", "tools": list(_WITH_GWS),
    }]
    cfg = server.AgentConfig(name="x", archetype="drive-clerk",
                             provider="anthropic", model="claude-opus-4-8")
    await server._resolve_archetype(cfg, _req(), None)
    assert cfg.tools == ["read", "google_drive", "web_search"]
    # The stored archetype itself is not mutated.
    assert archetype_registry.archetypes[0]["tools"] == _WITH_GWS


# ── archetype routes ────────────────────────────────────────────────────────

USER = {"id": "user-arch", "email": "a@x.co", "role": "user"}


@pytest.fixture
async def archetype_app(tmp_path: Path):
    """A bare app with the archetype router, a real FlightDeckDB, a fixed user."""
    import captain_claw.flight_deck.archetype_routes as ar

    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    await db._db.execute(
        "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
        " VALUES (?, ?, 'h', 'A', 'user', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
        (USER["id"], USER["email"]))
    await db._db.commit()
    fd_auth.set_auth_db(db)
    app = FastAPI()
    app.include_router(ar.router)
    app.dependency_overrides[fd_auth.get_current_user] = lambda: dict(USER)
    app.dependency_overrides[fd_auth.get_optional_user] = lambda: dict(USER)
    try:
        yield types.SimpleNamespace(app=app, db=db)
    finally:
        fd_auth._db = prev
        await db.close()


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def _stored_tools(db: FlightDeckDB, archetype_id: str) -> list[str]:
    row = await db.get_user_archetype(USER["id"], archetype_id)
    assert row is not None
    return json.loads(row["data"])["tools"]


async def test_create_and_update_store_no_gws(archetype_app):
    body = {"archetype_id": "drive-clerk", "role": "Drive Clerk", "tools": list(_WITH_GWS)}
    async with _client(archetype_app.app) as c:
        r = await c.post("/fd/archetypes", json=body)
        assert r.status_code == 200, r.text
        assert await _stored_tools(archetype_app.db, "drive-clerk") == [
            "read", "google_drive", "web_search"]

        r = await c.put("/fd/archetypes/drive-clerk",
                        json={**body, "tools": ["gws", "google_calendar"]})
        assert r.status_code == 200, r.text
        assert await _stored_tools(archetype_app.db, "drive-clerk") == ["google_calendar"]

        # PUT upserts a new id (an override of a base archetype) the same way.
        r = await c.put("/fd/archetypes/new-clerk", json={**body, "tools": ["gws"]})
        assert r.status_code == 200, r.text
        assert await _stored_tools(archetype_app.db, "new-clerk") == []


def _fake_llm(monkeypatch: pytest.MonkeyPatch, payload: object) -> None:
    import captain_claw.llm as llm

    class _Provider:
        async def complete(self, **_kw):
            return types.SimpleNamespace(content=json.dumps(payload), finish_reason="stop")

    monkeypatch.setattr(llm, "create_provider", lambda **_kw: _Provider())


async def test_generated_draft_has_no_gws(archetype_app, monkeypatch):
    _fake_llm(monkeypatch, {"id": "drive-clerk", "role": "Drive Clerk",
                            "tools": ["gws", "google_drive"]})
    async with _client(archetype_app.app) as c:
        r = await c.post("/fd/archetypes/generate",
                         json={"prompt": "a drive clerk", "provider": "anthropic"})
    assert r.status_code == 200, r.text
    assert r.json()["tools"] == ["google_drive"]


async def test_forged_drafts_have_no_gws(archetype_app, monkeypatch):
    _fake_llm(monkeypatch, {"archetypes": [
        {"id": "drive-clerk", "role": "Drive Clerk", "tools": ["gws", "google_drive"]},
        {"id": "scheduler", "role": "Scheduler", "tools": ["google_calendar", "gws"]},
    ]})
    async with _client(archetype_app.app) as c:
        r = await c.post("/fd/archetypes/forge",
                         data={"instructions": "a small office team", "provider": "anthropic"})
    assert r.status_code == 200, r.text
    drafts = r.json()["archetypes"]
    assert [d["tools"] for d in drafts] == [["google_drive"], ["google_calendar"]]


async def _store_pre_retirement_row(db: FlightDeckDB) -> None:
    """A row saved before gws was retired: written straight to the DB, past
    the create route's stripping."""
    await db.create_user_archetype(USER["id"], "drive-clerk", json.dumps({
        "role": "Drive Clerk", "description": "Files briefs to Drive",
        "cognitive_mode": "neutra", "tier": "mid", "tools": list(_WITH_GWS),
    }))


async def test_listing_drops_gws_from_a_pre_retirement_row(archetype_app):
    await _store_pre_retirement_row(archetype_app.db)
    async with _client(archetype_app.app) as c:
        listed = (await c.get("/fd/archetypes")).json()["archetypes"]
        mine = (await c.get("/fd/archetypes/mine")).json()
    [row] = [a for a in listed if a["id"] == "drive-clerk"]
    assert row["tools"] == ["read", "google_drive", "web_search"]
    assert [a["tools"] for a in mine] == [["read", "google_drive", "web_search"]]
    # Read-only: the stored row is left as it was.
    assert await _stored_tools(archetype_app.db, "drive-clerk") == _WITH_GWS


# ── Agent Forge (POST /fd/forge) ────────────────────────────────────────────


async def test_forge_catalog_and_team_have_no_gws(monkeypatch):
    import captain_claw.flight_deck.archetypes as arch_mod
    import captain_claw.llm as llm

    async def fake_registry(db, uid):
        return {"tiers": {"mid": {}}, "archetypes": [{
            "id": "drive-clerk", "role": "Drive Clerk", "family": "ops",
            "description": "Files briefs to Drive", "cognitive_mode": "neutra",
            "tier": "mid", "tools": list(_WITH_GWS),
        }]}

    seen: dict[str, str] = {}

    class _Provider:
        async def complete(self, *, messages, **_kw):
            seen["system"] = messages[0].content
            return types.SimpleNamespace(finish_reason="stop", content=json.dumps({
                "agents": [
                    {"name": "a", "archetype": "drive-clerk", "tools": ["gws", "read"]},
                    {"name": "b", "archetype": None, "new_archetype": {
                        "id": "scheduler", "role": "Scheduler",
                        "tools": ["google_calendar", "gws"]}},
                ],
            }))

    monkeypatch.setattr(arch_mod, "merged_registry", fake_registry)
    monkeypatch.setattr(fd_auth, "get_db", lambda: object())
    monkeypatch.setattr(llm, "create_provider", lambda **_kw: _Provider())

    result = await server.forge_decompose(
        server.ForgeRequest(prompt="a team that summarises my Drive folder"),
        request=types.SimpleNamespace(), user=dict(USER))

    [line] = [ln for ln in seen["system"].splitlines() if ln.startswith("- id: `drive-clerk`")]
    assert "tools: read, google_drive, web_search)" in line
    assert "gws" not in line
    a, b = result["agents"]
    assert a["tools"] == ["read"]
    assert b["new_archetype"]["tools"] == ["google_calendar"]


# ── spawn env ───────────────────────────────────────────────────────────────


def test_agent_env_still_scrubs_ambient_gws_credentials(monkeypatch):
    monkeypatch.setenv("GOOGLE_WORKSPACE_CLI_TOKEN", "operator-token")
    monkeypatch.setenv("GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE", "/operator/creds.json")
    env = server._agent_base_env()
    assert "GOOGLE_WORKSPACE_CLI_TOKEN" not in env
    assert "GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE" not in env
