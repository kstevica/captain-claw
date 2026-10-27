"""Own AI keys (BYOK, ``byok.py``): a user's provider keys, the switch that
runs their agents on them, and what it changes — their tiers, the agents'
models and keys, and the daily limits."""

from __future__ import annotations

import json
import types
import uuid
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi import FastAPI

from captain_claw.flight_deck import auth as auth_mod
from captain_claw.flight_deck import basna_routes, byok, fd_crypto, server, usage_quota
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.flight_deck.db import FlightDeckDB

LOOPBACK = ("127.0.0.1", 40001)
TEAM_SET = {
    "id": "team", "name": "Team",
    "tiers": {
        "reason": {"provider": "anthropic", "model": "claude-opus-4-8", "api_key": "@system"},
        "balanced": {"provider": "anthropic", "model": "claude-sonnet-4-6", "api_key": "@system",
                     "input_ctx": 200000, "output_ctx": 32768, "vision": True},
        "fast": {"provider": "openai", "model": "gpt-4.1-mini", "api_key": "@system"},
        "micro": {"provider": "ollama", "model": "qwen3.5:4b", "api_key": ""},
    },
    "envVars": [{"key": "BRAVE_API_KEY", "value": "team-brave"}],
}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setattr(byok, "STANDALONE_MODE", False)
    monkeypatch.setattr(usage_quota, "STANDALONE_MODE", False)
    monkeypatch.setenv("FD_DATA_KEY_FILE", str(tmp_path / "data_key"))
    fd_crypto.reset_for_tests()
    # Agents live on disk under DATA_DIR — never the developer's real ones.
    monkeypatch.setattr(server, "DATA_DIR", tmp_path / "fd-data")
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", tmp_path / "fd-data" / ".processes.json")
    monkeypatch.setattr(server, "_processes", {})
    (tmp_path / "fd-data").mkdir()
    for name in server._PROVIDER_KEY_ENV.values():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(byok, "agent_switcher", server.byok_switch_agents)
    monkeypatch.setattr(byok, "owner_agents", server.byok_owner_agents)
    monkeypatch.setattr(server, "_restart_processes", lambda slugs: None)


@pytest.fixture
async def db(monkeypatch, tmp_path: Path):
    fd_db = FlightDeckDB(tmp_path / "fd.db")
    await fd_db.init()
    monkeypatch.setattr(auth_mod, "_db", fd_db)
    monkeypatch.setattr(basna_routes, "_SYSTEM_PROVIDER_KEYS", {})
    monkeypatch.setattr(basna_routes, "_SYSTEM_ENDPOINT_KEYS", {})
    monkeypatch.setattr(basna_routes, "_SYSTEM_KEYS_TS", 0.0)
    await fd_db.set_system_setting("fd:shared-tier-sets", json.dumps({"sets": [TEAM_SET], "defaultSetId": "team"}))
    await fd_db.set_system_setting("fd:provider-keys", json.dumps({"anthropic": "team-anthropic", "openai": "team-openai"}))
    yield fd_db
    await fd_db.close()


async def _user(db, role="user") -> dict:
    return await db.create_user(f"{role}-{uuid.uuid4().hex[:8]}@example.com", "x", role=role,
                                metadata={"plan": "free"})


def _app(user: dict) -> FastAPI:
    app = FastAPI()
    app.include_router(byok.router)
    app.dependency_overrides[get_current_user] = lambda: user
    return app


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=LOOPBACK), base_url="http://fd.test")


@pytest.fixture
def provider_ok(monkeypatch):
    """Every key the providers are asked about works, with these models."""
    models = {"anthropic": ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
              "openai": ["gpt-4.1", "gpt-4.1-mini", "gpt-4o"],
              "gemini": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]}

    async def check(provider, key):
        if key.startswith("bad"):
            raise byok.KeyRefused("HTTP 401")
        return models.get(provider, [])

    monkeypatch.setattr(byok, "check_key", check)


def _agent(slug: str, owner: str, provider="anthropic", model="claude-sonnet-4-6", env="",
           **extra) -> None:
    agent_dir = server.DATA_DIR / slug
    agent_dir.mkdir(parents=True)
    (agent_dir / "config.yaml").write_text(yaml.safe_dump(
        {"model": {"provider": provider, "model": model, "api_key": "", "base_url": "", "max_tokens": 32768}}))
    (agent_dir / ".env").write_text(env)
    registry = server._load_process_registry()
    registry[slug] = {"slug": slug, "name": slug, "provider": provider, "model": model,
                      "owner": owner, "pid": None, "web_port": 0, **extra}
    server._save_process_registry(registry)


def _config(slug: str) -> dict:
    return yaml.safe_load((server.DATA_DIR / slug / "config.yaml").read_text())["model"]


def _dotenv(slug: str) -> str:
    return (server.DATA_DIR / slug / ".env").read_text()


# ── Which model each tier runs on ────────────────────────────────────


class TestModels:
    def test_newest_of_the_family_alias_before_snapshot(self):
        assert byok.default_model("anthropic", "reason", [
            "claude-opus-4-1-20250805", "claude-opus-4-8", "claude-3-opus-20240229"]) == "claude-opus-4-8"
        assert byok.default_model("anthropic", "fast", ["claude-haiku-4-5-20251001", "claude-haiku-4-5"]) == "claude-haiku-4-5"
        assert byok.default_model("gemini", "reason", ["gemini-2.5-pro-preview-05-06", "gemini-2.5-pro"]) == "gemini-2.5-pro"
        assert byok.default_model("openrouter", "balanced", [
            "anthropic/claude-sonnet-4", "anthropic/claude-sonnet-4.6"]) == "anthropic/claude-sonnet-4.6"

    def test_a_family_is_not_its_sibling(self):
        assert byok.default_model("openai", "balanced", ["gpt-4.1-mini", "gpt-4.1"]) == "gpt-4.1"
        assert byok.default_model("openai", "fast", ["gpt-4.1", "gpt-4.1-mini"]) == "gpt-4.1-mini"

    def test_never_a_model_that_signs_in_through_chatgpt(self):
        assert byok.default_model("openai", "balanced", ["gpt-5", "gpt-5-mini"]) == "gpt-4.1"

    def test_a_tier_keeps_its_model_on_the_users_key_for_that_provider(self):
        planned, missing = byok.plan(TEAM_SET["tiers"], {"anthropic": "sk-a"}, {"tiers": {}, "providers": {}})
        assert planned["balanced"] == {"provider": "anthropic", "model": "claude-sonnet-4-6"}
        assert planned["fast"]["provider"] == "anthropic"  # no OpenAI key: one of theirs
        assert "micro" not in planned and not missing  # local stays as it is

    def test_a_pick_wins_and_no_key_covers_nothing(self):
        meta = {"tiers": {"fast": {"provider": "anthropic", "model": "claude-haiku-4-5"}}, "providers": {}}
        planned, _ = byok.plan(TEAM_SET["tiers"], {"anthropic": "sk-a"}, meta)
        assert planned["fast"]["model"] == "claude-haiku-4-5"
        _planned, missing = byok.plan(TEAM_SET["tiers"], {}, {"tiers": {}})
        assert "balanced" in missing and "micro" not in missing

    def test_overlay_puts_the_users_key_on_every_deck_paid_tier(self):
        tiers = byok.overlay_tiers(TEAM_SET["tiers"], {"anthropic": "sk-a"}, {"tiers": {}, "providers": {}})
        assert tiers["balanced"]["api_key"] == "sk-a" and tiers["balanced"]["byok"]
        assert tiers["balanced"]["vision"] is True and tiers["balanced"]["output_ctx"] == 32768
        assert tiers["fast"]["api_key"] == "sk-a"
        assert tiers["micro"] == TEAM_SET["tiers"]["micro"]
        assert all(t.get("api_key") != "@system" for t in tiers.values())


# ── Keys ─────────────────────────────────────────────────────────────


class TestKeys:
    async def test_a_key_is_checked_sealed_and_never_returned(self, db, provider_ok):
        user = await _user(db)
        async with _client(_app(user)) as c:
            r = await c.put("/fd/byok/key/anthropic", json={"api_key": "sk-ant-secret-1234"})
            assert r.status_code == 200, r.text
            assert "sk-ant-secret-1234" not in r.text
            body = r.json()
        row = next(p for p in body["providers"] if p["id"] == "anthropic")
        assert row["connected"] and row["hint"] == "…1234" and row["verified"]
        assert "claude-sonnet-4-6" in row["models"]
        stored = await db.get_setting(user["id"], "byok:keys")
        assert "sk-ant-secret-1234" not in stored
        assert fd_crypto.unseal(stored) == {"anthropic": "sk-ant-secret-1234"}

    async def test_a_refused_key_is_not_saved(self, db, provider_ok):
        user = await _user(db)
        async with _client(_app(user)) as c:
            r = await c.put("/fd/byok/key/openai", json={"api_key": "bad-key-0000"})
        assert r.status_code == 400 and "bad-key-0000" not in r.text
        assert await byok.read_keys(db, user["id"]) == {}

    async def test_a_malformed_body_never_echoes_the_key(self, db, provider_ok):
        user = await _user(db)
        async with _client(_app(user)) as c:
            r = await c.put("/fd/byok/key/openai", json={"api_key": "sk with space"})
        assert r.status_code == 422 and "sk with space" not in r.text

    async def test_byok_settings_are_server_owned(self):
        from captain_claw.flight_deck.settings_routes import _server_owned

        assert _server_owned("byok:keys") and _server_owned("byok:meta")

    async def test_the_last_key_cant_go_while_on(self, db, provider_ok):
        user = await _user(db)
        async with _client(_app(user)) as c:
            await c.put("/fd/byok/key/anthropic", json={"api_key": "sk-ant-aaaa1111"})
            assert (await c.put("/fd/byok/enabled", json={"enabled": True})).status_code == 200
            assert (await c.delete("/fd/byok/key/anthropic")).status_code == 409
            await c.put("/fd/byok/enabled", json={"enabled": False})
            assert (await c.delete("/fd/byok/key/anthropic")).status_code == 200
        assert await byok.read_keys(db, user["id"]) == {}

    async def test_switch_needs_a_key_and_the_deck_allowing_it(self, db, provider_ok):
        user = await _user(db)
        async with _client(_app(user)) as c:
            assert (await c.put("/fd/byok/enabled", json={"enabled": True})).status_code == 409
            await c.put("/fd/byok/key/anthropic", json={"api_key": "sk-ant-aaaa1111"})
            await db.set_system_setting(byok.ALLOWED_SETTING, "0")
            assert (await c.put("/fd/byok/enabled", json={"enabled": True})).status_code == 403
            await db.set_system_setting(byok.ALLOWED_SETTING, "1")
            assert (await c.put("/fd/byok/enabled", json={"enabled": True})).json()["enabled"]

    async def test_a_chatgpt_model_cant_be_picked(self, db, provider_ok):
        user = await _user(db)
        async with _client(_app(user)) as c:
            await c.put("/fd/byok/key/openai", json={"api_key": "sk-oai-aaaa1111"})
            r = await c.put("/fd/byok/models", json={"tiers": {"fast": {"provider": "openai", "model": "gpt-5-mini"}}})
        assert r.status_code == 400

    async def test_not_offered_without_sign_in(self, db, monkeypatch):
        monkeypatch.setenv("FD_AUTH_ENABLED", "false")
        user = await _user(db)
        async with _client(_app(user)) as c:
            assert (await c.get("/fd/byok")).status_code == 404


# ── Tiers and limits ─────────────────────────────────────────────────


async def _turn_on(db, user, provider="anthropic", key="sk-ant-aaaa1111") -> None:
    async with _client(_app(user)) as c:
        assert (await c.put(f"/fd/byok/key/{provider}", json={"api_key": key})).status_code == 200
        r = await c.put("/fd/byok/enabled", json={"enabled": True})
        assert r.status_code == 200, r.text


class TestTiersAndLimits:
    async def test_the_owners_tiers_run_on_their_keys_while_on(self, db, provider_ok):
        user = await _user(db)
        assert (await basna_routes.owner_tier_set(db, user["id"]))["source"] == "team"
        await _turn_on(db, user)
        resolved = await basna_routes.owner_tier_set(db, user["id"])
        assert resolved["source"] == "byok" and resolved["base_source"] == "team"
        assert resolved["tiers"]["balanced"]["api_key"] == "sk-ant-aaaa1111"
        assert resolved["env_vars"] == TEAM_SET["envVars"]  # tool keys stay the deck's
        assert basna_routes.tier_api_key(resolved["tiers"]["fast"]) == "sk-ant-aaaa1111"

    async def test_no_limits_while_on_and_own_key_tokens_kept_apart(self, db, provider_ok):
        user = await _user(db)
        await db.add_daily_usage(user["id"], usage_quota.today(), messages=30)
        assert (await usage_quota.status(db, user["id"]))["blocked"] == "messages"
        await _turn_on(db, user)
        st = await usage_quota.status(db, user["id"])
        assert st["exempt"] and st["byok"] and not st["blocked"]
        await usage_quota.record_tokens(db, user["id"], 5000, byok=True)
        used = await db.get_daily_usage(user["id"], usage_quota.today())
        assert used["byok_tokens"] == 5000 and used["tokens"] == 0

    async def test_on_a_shared_agent_the_owners_keys_count(self, db, provider_ok):
        owner, member = await _user(db), await _user(db)
        await db.add_daily_usage(member["id"], usage_quota.today(), messages=30)
        allowed, _ = await usage_quota.admit_message(member["id"], "hi", agent_owner=owner["id"])
        assert not allowed
        await _turn_on(db, owner)
        allowed, _ = await usage_quota.admit_message(member["id"], "hi", agent_owner=owner["id"])
        assert allowed
        # A member on their own keys chatting with a team-key agent is still held.
        owner2, member2 = await _user(db), await _user(db)
        await _turn_on(db, member2)
        await db.add_daily_usage(member2["id"], usage_quota.today(), messages=30)
        allowed, _ = await usage_quota.admit_message(member2["id"], "hi", agent_owner=owner2["id"])
        assert not allowed

    async def test_an_own_key_agents_report_is_kept_apart(self, db, monkeypatch):
        user = await _user(db)
        _agent("mine", user["id"], web_auth="tok-mine", key_source="byok")
        _agent("team", user["id"], web_auth="tok-team")
        monkeypatch.setattr(usage_quota, "_reporting_user", lambda request: _async(user["id"]))
        app = FastAPI()
        app.include_router(usage_quota.router)
        async with _client(app) as c:
            await c.post("/fd/quota/agent/usage", json={"tokens": 100}, headers={"X-Agent-Auth": "tok-mine"})
            await c.post("/fd/quota/agent/usage", json={"tokens": 7}, headers={"X-Agent-Auth": "tok-team"})
        used = await db.get_daily_usage(user["id"], usage_quota.today())
        assert used == {"messages": 0, "tokens": 7, "byok_tokens": 100}


async def _async(value):
    return value


# ── Agents move with the switch ──────────────────────────────────────


class TestAgentsMove:
    async def test_on_moves_agents_onto_own_keys_and_off_puts_them_back(self, db, provider_ok):
        user = await _user(db)
        _agent("writer", user["id"], "anthropic", "claude-sonnet-4-6", env="ANTHROPIC_API_KEY=team-anthropic\nBRAVE_API_KEY=b\n")
        _agent("quick", user["id"], "openai", "gpt-4.1-mini", env="OPENAI_API_KEY=team-openai\n", model_tier="fast")
        _agent("local", user["id"], "ollama", "qwen3.5:4b")
        _agent("theirs", "someone-else", "anthropic", "claude-sonnet-4-6", env="ANTHROPIC_API_KEY=team-anthropic\n")
        await _turn_on(db, user)

        reg = server._load_process_registry()
        assert reg["writer"]["key_source"] == "byok" and "ANTHROPIC_API_KEY=sk-ant-aaaa1111" in _dotenv("writer")
        assert "team-anthropic" not in _dotenv("writer") and "BRAVE_API_KEY=b" in _dotenv("writer")
        # The fast tier has no OpenAI key behind it: onto one of theirs.
        assert reg["quick"]["provider"] == "anthropic" and _config("quick")["provider"] == "anthropic"
        assert _config("quick")["api_key"] == "" and "ANTHROPIC_API_KEY=sk-ant-aaaa1111" in _dotenv("quick")
        assert reg["quick"]["byok_prev"]["model"] == "gpt-4.1-mini"
        assert "key_source" not in reg["local"] and "key_source" not in reg["theirs"]
        assert "team-anthropic" in _dotenv("theirs")

        async with _client(_app(user)) as c:
            assert (await c.put("/fd/byok/enabled", json={"enabled": False})).status_code == 200
        reg = server._load_process_registry()
        assert "key_source" not in reg["writer"] and "byok_prev" not in reg["quick"]
        assert (reg["quick"]["provider"], reg["quick"]["model"]) == ("openai", "gpt-4.1-mini")
        assert _config("quick")["provider"] == "openai"
        assert "sk-ant-aaaa1111" not in _dotenv("quick") and "OPENAI_API_KEY=team-openai" in _dotenv("quick")
        # The deck's key comes back where the own key was.
        assert "ANTHROPIC_API_KEY=team-anthropic" in _dotenv("writer")

    async def test_a_new_key_reaches_the_agents_on_it(self, db, provider_ok):
        user = await _user(db)
        _agent("writer", user["id"])
        await _turn_on(db, user)
        async with _client(_app(user)) as c:
            await c.put("/fd/byok/key/anthropic", json={"api_key": "sk-ant-bbbb2222"})
        assert "ANTHROPIC_API_KEY=sk-ant-bbbb2222" in _dotenv("writer")
        assert "aaaa1111" not in _dotenv("writer")

    async def test_the_admin_turning_it_off_moves_everyone_back(self, db, provider_ok):
        user = await _user(db)
        _agent("writer", user["id"])
        await _turn_on(db, user)
        app = FastAPI()
        app.include_router(byok.router)
        admin = await _user(db, role="admin")
        app.dependency_overrides[byok.require_admin] = lambda: admin
        async with _client(app) as c:
            assert (await c.get("/fd/admin/byok")).json()["users_on"] == [user["id"]]
            r = await c.put("/fd/admin/byok", json={"allowed": False})
            assert r.json()["switched_back"] == 1
        assert not await byok.active(db, user["id"])
        assert "key_source" not in server._load_process_registry()["writer"]
        assert not (await usage_quota.status(db, user["id"]))["exempt"]

    async def test_an_agent_on_own_keys_cant_move_to_a_deck_paid_provider(self, db, provider_ok):
        user = await _user(db)
        _agent("writer", user["id"])
        await _turn_on(db, user)
        req = types.SimpleNamespace(state=types.SimpleNamespace(user_id=user["id"]))
        with pytest.raises(server.HTTPException) as refused:
            await server.update_agent_model("process", "writer",
                                            server.AgentModelUpdate(provider="openai", model="gpt-4.1"), req, user)
        assert refused.value.status_code == 409
        await server.update_agent_model("process", "writer",
                                        server.AgentModelUpdate(provider="ollama", model="qwen3.5:4b"), req, user)
        assert "key_source" not in server._load_process_registry()["writer"]


# ── New agents ───────────────────────────────────────────────────────


class TestSpawn:
    async def test_an_archetype_spawn_on_own_keys_is_marked(self, db, provider_ok, monkeypatch):
        from captain_claw.flight_deck import archetypes as arch_mod

        user = await _user(db)
        await _turn_on(db, user)
        arch = {"id": "writer", "role": "Writer", "tier": "balanced", "tools": [], "cognitive_mode": "neutra",
                "fleet_instructions": "Write."}

        async def merged(_db, _uid):
            return [dict(arch)]

        monkeypatch.setattr(arch_mod, "merged_archetypes", merged)
        config = server.AgentConfig(name="w", archetype="writer")
        req = types.SimpleNamespace(state=types.SimpleNamespace(user_id=user["id"]))
        await server._resolve_archetype(config, req, user)
        await server._resolve_spawn_provider_key(config)
        assert (config.provider, config.model) == ("anthropic", "claude-sonnet-4-6")
        assert config.provider_api_key == "sk-ant-aaaa1111"
        assert config._byok and config._model_tier == "balanced"

    def test_the_caller_cant_claim_own_keys(self):
        config = server.AgentConfig.model_validate({"name": "w", "_byok": True, "_model_tier": "x"})
        assert config._byok is False and config._model_tier == ""
