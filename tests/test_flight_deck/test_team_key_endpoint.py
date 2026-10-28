"""The team's key goes only where the team sends it — never on to an endpoint
the caller picks.

A provider's org key belongs to the provider's own endpoint and to the custom
endpoints the published team sets route the provider through (their
"@system" tiers). Before this, a non-admin could have the key sent to a host of
their own without ever reading it:

* resolving it for an endpoint they named — "@system" with any ``base_url``
  (a spawn, a tier set of their own, a request body Flight Deck itself calls);
* moving a team-keyed agent afterwards — a new ``base_url`` (kiosk Settings,
  ``PUT /fd/agent-model``), a config.yaml edit, an allowed model on another
  endpoint, an ``*_API_BASE`` / ``*_BASE_URL`` variable LiteLLM reads, or a
  proxy / certificate authority of their own in the .env.

Pinned here: the key resolves only for the team's endpoints; a non-admin's
change that would send it elsewhere is refused (nothing written) — or, in
kiosk Settings, re-keyed to the new endpoint's own team key; healing never
writes a key the agent would send elsewhere; admins change agents as before.

Real FlightDeckDB + process registry in tmp dirs; ``DATA_DIR`` is never the
real ``./fd-data``.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi import HTTPException

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import basna_routes as basna_mod
from captain_claw.flight_deck import rate_limiter, server
from captain_claw.flight_deck.auth import ACT_AS_HEADER, create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

ADMIN = "user-admin"
BOB = "user-bob"
REMOTE = ("203.0.113.9", 40001)

ORG_OPENAI = "sk-org-openai-0123456789"
ORG_ANTHROPIC = "sk-org-anthropic-9876543210"
GW = "https://gw.example/v1"           # the team's gateway, with a key of its own
GW_KEY = "gw-team-key-5555aaaa"
ROUTED = "https://routed.example/v1"   # a gateway the team runs on the provider's key
EVIL = "https://collector.example/v1"  # a host the owner controls

BOB_CONFIG = {"model": {"provider": "openai", "model": "gpt-4.1", "api_key": "", "base_url": ""}}
BOB_ENV = f"OPENAI_API_KEY={ORG_OPENAI}\nBRAVE_API_KEY=brave-own\n"


def _published(*tiers: dict, env_vars: list | None = None) -> str:
    return json.dumps({"sets": [{"id": "team", "name": "Team", "envVars": env_vars or [],
                                 "tiers": {f"t{i}": t for i, t in enumerate(tiers)}}],
                       "defaultSetId": "team"})


def _tier(base_url: str = "", provider: str = "openai") -> dict:
    return {"provider": provider, "model": "m", "api_key": "@system", "base_url": base_url}


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Auth-enabled deck with org keys, a gateway key and a published set that
    routes openai through ROUTED; Bob (a user) owns an agent on the org key."""
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
    await db.set_system_setting("fd:provider-keys", json.dumps(
        {"openai": ORG_OPENAI, "anthropic": ORG_ANTHROPIC}))
    await db.set_system_setting("fd:endpoint-keys", json.dumps(
        {basna_mod.endpoint_key_id("openai", GW): GW_KEY}))
    await db.set_system_setting("fd:shared-tier-sets", _published(_tier(), _tier(GW), _tier(ROUTED)))

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
                 "XAI_API_KEY", "OPENROUTER_API_KEY", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())
    for name, empty in (("_SYSTEM_PROVIDER_KEYS", {}), ("_SYSTEM_ENDPOINT_KEYS", {}),
                        ("_SYSTEM_NO_FALLBACK", frozenset()), ("_SYSTEM_ROUTED", frozenset()),
                        ("_SYSTEM_KEYS_TS", 0.0)):
        monkeypatch.setattr(basna_mod, name, empty)

    server._save_process_registry({
        "bob-agent": {"slug": "bob-agent", "name": "Bob's agent", "web_port": 24101,
                      "web_auth": "bob-tok", "owner": BOB, "pid": None},
    })
    _write_agent()
    try:
        yield db
    finally:
        await db.close()
        fd_auth._db = prev


def _write_agent(config: dict | None = None, env: str = BOB_ENV) -> None:
    d = server.DATA_DIR / "bob-agent"
    d.mkdir(exist_ok=True)
    (d / "config.yaml").write_text(yaml.dump(config or BOB_CONFIG, sort_keys=False))
    (d / ".env").write_text(env)


def _files() -> tuple[str, str]:
    d = server.DATA_DIR / "bob-agent"
    return (d / "config.yaml").read_text(), (d / ".env").read_text()


def _model() -> dict:
    return yaml.safe_load(_files()[0])["model"]


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app, client=REMOTE), base_url="http://fd.test")


def _hdr(caller: str, act_as: str | None = None) -> dict:
    h = {"Authorization": f"Bearer {create_access_token(caller)}"}
    if act_as is not None:
        h[ACT_AS_HEADER] = act_as
    return h


async def _set_model(body: dict, caller: str = BOB, act_as: str | None = None) -> httpx.Response:
    async with _client() as c:
        return await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(caller, act_as),
                           json={"provider": "openai", "model": "gpt-4.1", **body})


async def _save_config(body: dict, caller: str = BOB) -> httpx.Response:
    async with _client() as c:
        return await c.put("/fd/agent-config/process/bob-agent", headers=_hdr(caller), json=body)


def _spawn_cfg(**kw) -> server.AgentConfig:
    return server.AgentConfig(**{"name": "x", "provider": "openai", "model": "gpt-4.1",
                                 "provider_api_key": "@system", **kw})


# ── where the org key resolves ──────────────────────────────────────────────

class TestTheKeyResolvesOnlyForTheTeamsEndpoints:
    async def test_team_key(self, deck):
        assert await server._team_key("openai", "") == ORG_OPENAI
        assert await server._team_key("openai", GW) == GW_KEY
        assert await server._team_key("openai", ROUTED) == ORG_OPENAI   # the team routes it there
        assert await server._team_key("openai", EVIL) == ""

    async def test_flight_decks_own_calls(self, deck):
        """Basna / Vatra / Code runs and request bodies (/fd/basna/recommend …)."""
        await basna_mod._refresh_system_provider_keys(deck)
        assert basna_mod._effective_key("openai", "@system") == ORG_OPENAI
        assert basna_mod._effective_key("openai", "@system", GW) == GW_KEY
        assert basna_mod._effective_key("openai", "@system", ROUTED) == ORG_OPENAI
        for key in ("@system", ""):
            assert basna_mod._effective_key("openai", key, EVIL) is None
        assert await basna_mod._request_api_key("openai", "@system", EVIL) is None
        assert basna_mod._effective_key("openai", "sk-mine", EVIL) == "sk-mine"

    def test_only_published_system_tiers_route(self):
        blob = json.loads(_published(_tier(GW), {**_tier(EVIL), "api_key": ""}))
        assert basna_mod.routed_endpoint_ids(blob) == {basna_mod.endpoint_key_id("openai", GW)}
        assert basna_mod.routed_endpoint_ids({"sets": "junk"}) == frozenset()

    async def test_a_spawn_on_system_with_its_own_endpoint_gets_no_key(self, deck):
        cfg = _spawn_cfg(base_url=EVIL)
        with pytest.raises(HTTPException) as exc:
            await server._resolve_spawn_provider_key(cfg)
        assert exc.value.status_code == 409
        assert ORG_OPENAI not in cfg.provider_api_key


# ── a spawn that names somewhere else ───────────────────────────────────────

class TestSpawn:
    async def test_on_the_teams_endpoints_it_gets_the_key(self, deck):
        for base_url, want in (("", ORG_OPENAI), (GW, GW_KEY), (ROUTED, ORG_OPENAI)):
            cfg = _spawn_cfg(base_url=base_url)
            await server._resolve_spawn_provider_key(cfg)
            assert cfg.provider_api_key == want

    @pytest.mark.parametrize("extra", [
        {"env_vars": [{"key": "OPENAI_BASE_URL", "value": EVIL}]},
        {"env_vars": [{"key": "HTTPS_PROXY", "value": "http://collector.example:8080"}]},
        {"env_vars": [{"key": "SSL_CERT_FILE", "value": "/tmp/my-ca.pem"}]},
        {"allowed_models": [{"id": "x", "provider": "openai", "model": "m", "base_url": EVIL}]},
    ])
    async def test_somewhere_else_is_refused(self, deck, extra):
        with pytest.raises(HTTPException) as exc:
            await server._resolve_spawn_provider_key(_spawn_cfg(**extra))
        assert exc.value.status_code == 400 and ORG_OPENAI not in exc.value.detail

    async def test_a_team_key_carried_in_the_env_vars_is_held_to_it_too(self, deck):
        """A flow spawn copies its origin agent's .env — the team key with it."""
        carried = [{"key": "OPENAI_API_KEY", "value": ORG_OPENAI}]
        with pytest.raises(HTTPException) as exc:
            await server._resolve_spawn_provider_key(_spawn_cfg(provider_api_key="", base_url=EVIL, env_vars=carried))
        assert exc.value.status_code == 400
        await server._resolve_spawn_provider_key(_spawn_cfg(provider_api_key="", env_vars=carried))

    async def test_a_key_of_its_own_goes_anywhere(self, deck):
        cfg = _spawn_cfg(provider_api_key="sk-bobs-own", base_url=EVIL,
                         env_vars=[{"key": "HTTPS_PROXY", "value": "http://mine:8080"}])
        await server._resolve_spawn_provider_key(cfg)
        assert cfg.provider_api_key == "sk-bobs-own"

    async def test_a_proxy_the_deck_itself_sets_is_fine(self, deck, monkeypatch):
        proxy = "http://corp-proxy:3128"
        monkeypatch.setenv("HTTPS_PROXY", proxy)
        cfg = _spawn_cfg(env_vars=[{"key": "HTTPS_PROXY", "value": proxy}])
        await server._resolve_spawn_provider_key(cfg)
        assert cfg.provider_api_key == ORG_OPENAI

    async def test_a_proxy_the_published_set_carries_is_fine(self, deck):
        proxy = [{"key": "HTTPS_PROXY", "value": "http://team-proxy:3128"}]
        await deck.set_system_setting("fd:shared-tier-sets", _published(_tier(), env_vars=proxy))
        cfg = _spawn_cfg(env_vars=proxy)
        await server._resolve_spawn_provider_key(cfg)
        assert cfg.provider_api_key == ORG_OPENAI

    async def test_an_allowed_model_with_its_own_key_is_fine(self, deck):
        cfg = _spawn_cfg(allowed_models=[{"id": "x", "provider": "anthropic", "model": "m", "base_url": EVIL}])
        await server._resolve_spawn_provider_key(cfg)   # anthropic's key isn't the one being given
        assert cfg.provider_api_key == ORG_OPENAI


# ── kiosk Settings: PUT /fd/agent-model ─────────────────────────────────────

class TestModelChange:
    async def test_a_new_endpoint_without_a_key_is_refused(self, deck):
        r = await _set_model({"base_url": EVIL})
        assert r.status_code == 400
        assert "collector.example" in r.json()["detail"] and ORG_OPENAI not in r.json()["detail"]
        assert _model()["base_url"] == "" and _files()[1] == BOB_ENV   # nothing written

    async def test_with_a_key_of_its_own_it_goes(self, deck):
        r = await _set_model({"base_url": EVIL, "api_key": "sk-bobs-own"})
        assert r.status_code == 200, r.text
        assert (_model()["base_url"], _model()["api_key"]) == (EVIL, "sk-bobs-own")

    async def test_the_teams_gateway_gets_its_own_key(self, deck):
        assert (await _set_model({"base_url": GW})).status_code == 200
        env = _files()[1]
        assert f"OPENAI_API_KEY={GW_KEY}" in env and ORG_OPENAI not in env
        assert (await _set_model({"base_url": ""})).status_code == 200   # …and back
        assert f"OPENAI_API_KEY={ORG_OPENAI}" in _files()[1]

    async def test_a_new_provider_gets_the_teams_key_for_it(self, deck):
        r = await _set_model({"provider": "anthropic", "model": "claude"})
        assert r.status_code == 200, r.text
        assert f"ANTHROPIC_API_KEY={ORG_ANTHROPIC}" in _files()[1]

    async def test_vision_alone_is_no_model_change(self, deck):
        _write_agent({"model": {**BOB_CONFIG["model"], "base_url": EVIL}})   # however it got there
        async with _client() as c:
            r = await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(BOB), json={"vision": "on"})
        assert r.status_code == 200

    async def test_an_admin_changes_it_as_before(self, deck):
        r = await _set_model({"base_url": EVIL}, caller=ADMIN, act_as=BOB)
        assert r.status_code == 200 and _model()["base_url"] == EVIL
        assert _files()[1] == BOB_ENV


# ── the raw editor: PUT /fd/agent-config ────────────────────────────────────

class TestConfigEdit:
    def _config(self, **model) -> str:
        return yaml.dump({"model": {**BOB_CONFIG["model"], **model}}, sort_keys=False)

    @pytest.mark.parametrize("body", [
        {"config_yaml": "model:\n  provider: openai\n  model: gpt-4.1\n  base_url: " + EVIL + "\n"},
        {"env": BOB_ENV + f"OPENAI_BASE_URL={EVIL}\n"},
        {"env": BOB_ENV + "HTTPS_PROXY=http://collector.example:8080\n"},
        {"env": BOB_ENV + "export https_proxy=http://collector.example:8080\n"},
        {"env": BOB_ENV + "REQUESTS_CA_BUNDLE=/tmp/my-ca.pem\n"},
    ])
    async def test_sending_the_key_elsewhere_is_refused(self, deck, body):
        before = _files()
        r = await _save_config(body)
        assert r.status_code == 400, r.text
        assert "Not saved" in r.json()["detail"] and ORG_OPENAI not in r.json()["detail"]
        assert _files() == before

    async def test_an_allowed_model_elsewhere_is_refused(self, deck):
        cfg = {"model": {**BOB_CONFIG["model"], "allowed": [
            {"id": "x", "provider": "openai", "model": "m", "base_url": EVIL}]}}
        assert (await _save_config({"config_yaml": yaml.dump(cfg)})).status_code == 400

    async def test_ordinary_edits_go_through(self, deck):
        r = await _save_config({"config_yaml": self._config(model="gpt-4.1-mini"),
                                "env": BOB_ENV + "TAVILY_API_KEY=tv-1\n"})
        assert r.status_code == 200, r.text
        assert _model()["model"] == "gpt-4.1-mini" and "TAVILY_API_KEY=tv-1" in _files()[1]

    async def test_a_masked_read_saved_back_goes_through(self, deck):
        async with _client() as c:
            got = (await c.get("/fd/agent-config/process/bob-agent", headers=_hdr(BOB))).json()
        before = _files()
        assert (await _save_config(got)).status_code == 200
        assert _files() == before

    async def test_a_proxy_already_there_is_kept_on_save(self, deck):
        env = BOB_ENV + "HTTPS_PROXY=http://admin-set:3128\n"
        _write_agent(env=env)
        assert (await _save_config({"env": env + "X=1\n"})).status_code == 200

    async def test_without_the_teams_key_the_endpoint_is_the_owners(self, deck):
        r = await _save_config({"config_yaml": self._config(base_url=EVIL), "env": "BRAVE_API_KEY=brave-own\n"})
        assert r.status_code == 200, r.text
        assert await server.heal_keyless_agents(restart=False) == []   # …and it isn't healed in
        assert ORG_OPENAI not in _files()[1]

    async def test_an_admin_edits_as_before(self, deck):
        async with _client() as c:
            r = await c.put("/fd/agent-config/process/bob-agent", headers=_hdr(ADMIN, BOB),
                            json={"env": BOB_ENV + f"OPENAI_BASE_URL={EVIL}\n"})
        assert r.status_code == 200


# ── healing a keyless agent ─────────────────────────────────────────────────

class TestHeal:
    async def test_the_teams_endpoint_is_healed(self, deck):
        _write_agent(env="BRAVE_API_KEY=brave-own\n")
        assert await server.heal_keyless_agents(restart=False) == ["bob-agent"]
        assert f"OPENAI_API_KEY={ORG_OPENAI}" in _files()[1]

    @pytest.mark.parametrize("env", [
        f"OPENAI_BASE_URL={EVIL}\n",
        "HTTPS_PROXY=http://collector.example:8080\n",
    ])
    async def test_an_agent_that_would_send_it_elsewhere_is_not(self, deck, env):
        _write_agent(env=env)
        assert await server.heal_keyless_agents(restart=False) == []
        assert ORG_OPENAI not in _files()[1]

    async def test_nor_one_with_an_allowed_model_elsewhere(self, deck):
        _write_agent({"model": {**BOB_CONFIG["model"], "allowed": [
            {"id": "x", "provider": "openai", "model": "m", "base_url": EVIL}]}}, env="")
        assert await server.heal_keyless_agents(restart=False) == []


# ── the routes an agent's key takes ─────────────────────────────────────────

def test_key_routes():
    model = {"provider": "openai", "base_url": "", "api_key": "", "allowed": [
        {"provider": "openai", "model": "a"},                       # its model's endpoint
        {"provider": "openai", "model": "b", "base_url": GW},
        {"provider": "anthropic", "model": "c", "base_url": EVIL},
        {"provider": "ollama", "model": "d"},
    ]}
    assert server._agent_key_routes(model, {"OPENAI_API_BASE": ROUTED}) == [
        ("openai", ROUTED), ("openai", GW), ("anthropic", EVIL), ("ollama", "")]
    # The model's own key covers its provider's entries; another provider's doesn't.
    assert server._agent_key_routes({**model, "api_key": "sk-own"}, {}) == [("anthropic", EVIL), ("ollama", "")]
