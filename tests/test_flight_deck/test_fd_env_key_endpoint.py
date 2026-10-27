"""Flight Deck's own keys go only where the team's do.

#260 kept org keys (``fd:provider-keys`` / ``fd:endpoint-keys``) on the team's
endpoints. A deck can also hold provider keys in its own environment, and
those still followed an endpoint a caller named, by fallback:

* Flight Deck's own model calls — a Basna/Vatra/Code run on a tier set of the
  user's own, a request body like ``/fd/basna/recommend`` — with no key of
  their own on a custom ``base_url``: ``_resolve_api_key`` (and LiteLLM after
  it) took the environment's key and sent it there;
* a non-admin's process agent on such an endpoint inherits Flight Deck's
  environment (``_agent_base_env``) and did the same.

Pinned here: on a deck with accounts, the deck's keys go to the provider's own
endpoint and the ones the published team sets route it through — a call with
no key elsewhere sends ``llm.WITHHELD_KEY``, and a non-admin's agent there has
the inherited key blanked. An admin's agent, a key the agent's own .env holds,
and a deck without accounts are as before; agents' own processes install no
guard at all.

Real FlightDeckDB + process registry in tmp dirs; ``DATA_DIR`` is never the
real ``./fd-data``.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import yaml

from captain_claw import llm
from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import basna_routes as basna_mod
from captain_claw.flight_deck import rate_limiter, server
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

BOB = "user-bob"
REMOTE = ("203.0.113.9", 40001)
FD_KEY = "sk-flight-deck-env-0123456789"
FD_ANTHROPIC = "sk-ant-flight-deck-env-9876"
ROUTED = "https://routed.example/v1"    # a gateway the published set sends openai to
EVIL = "https://collector.example/v1"   # a host the caller controls


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Auth-enabled deck whose environment holds provider keys (no org keys);
    the published set routes openai through ROUTED. Bob owns one agent."""
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    await db._db.execute(
        "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
        " VALUES (?, 'b@x.co', 'h', 'Bob', 'user', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')", (BOB,))
    await db._db.commit()
    await db.set_system_setting("fd:shared-tier-sets", json.dumps({"sets": [{
        "id": "team", "name": "Team", "envVars": [],
        "tiers": {"balanced": {"provider": "openai", "model": "m", "api_key": "@system", "base_url": ROUTED}},
    }], "defaultSetId": "team"}))

    data = tmp_path / "fd-data"
    data.mkdir()
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", data / ".processes.json")
    monkeypatch.setattr(server, "_processes", {})
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY", "OPENROUTER_API_KEY",
                 "OPENAI_BASE_URL", "OPENAI_API_BASE", "ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", FD_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FD_ANTHROPIC)
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())
    for name, empty in (("_SYSTEM_PROVIDER_KEYS", {}), ("_SYSTEM_ENDPOINT_KEYS", {}),
                        ("_SYSTEM_NO_FALLBACK", frozenset()), ("_SYSTEM_ROUTED", frozenset()),
                        ("_SYSTEM_KEYS_TS", 0.0)):
        monkeypatch.setattr(basna_mod, name, empty)
    monkeypatch.setattr(llm, "_ENV_KEY_GUARD", None)
    await server._guard_deck_keys()   # what the lifespan does
    try:
        yield db
    finally:
        llm.set_env_key_guard(None)
        await db.close()
        fd_auth._db = prev


def _agent(model: dict, env: str = "", slug: str = "bob-agent") -> Path:
    d = server.DATA_DIR / slug
    d.mkdir(exist_ok=True)
    (d / "config.yaml").write_text(yaml.dump({"model": {"provider": "openai", "model": "gpt-4.1",
                                                        "api_key": "", "base_url": "", **model}}))
    (d / ".env").write_text(env)
    return d


def _withheld(agent_dir: Path) -> dict[str, str]:
    base = server._agent_base_env()
    environment = dict(base)
    server._withhold_deck_keys(environment, base, agent_dir=agent_dir, owner_id=BOB)
    return environment


# ── Flight Deck's own model calls ───────────────────────────────────────────

class TestFlightDecksOwnCalls:
    async def test_where_the_deck_key_may_go(self, deck):
        assert server._deck_key_may_go("openai", "")
        assert server._deck_key_may_go("openai", "https://api.openai.com/v1/")   # its own, spelled out
        assert server._deck_key_may_go("openai", ROUTED)
        assert not server._deck_key_may_go("openai", EVIL)
        assert not server._deck_key_may_go("anthropic", ROUTED)   # routed for openai only

    async def test_a_call_with_no_key_elsewhere_sends_none_of_the_decks(self, deck):
        assert llm.create_provider("openai", "gpt-4.1", base_url=EVIL).api_key == llm.WITHHELD_KEY
        assert llm.create_provider("anthropic", "claude", api_key="", base_url=EVIL).api_key == llm.WITHHELD_KEY

    async def test_the_teams_endpoints_and_own_keys_are_as_before(self, deck):
        assert llm.create_provider("openai", "gpt-4.1").api_key == FD_KEY
        assert llm.create_provider("openai", "gpt-4.1", base_url=ROUTED).api_key == FD_KEY
        assert llm.create_provider("openai", "gpt-4.1", api_key="sk-mine", base_url=EVIL).api_key == "sk-mine"

    async def test_a_request_body_naming_its_own_endpoint(self, deck, monkeypatch):
        sent: list[tuple[str, str]] = []

        async def fake_complete(self, *a, **k):
            sent.append((self.api_key, self.base_url))
            raise RuntimeError("no network in tests")

        monkeypatch.setattr(llm.LiteLLMProvider, "complete", fake_complete)
        hdr = {"Authorization": f"Bearer {create_access_token(BOB)}"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, client=REMOTE),
                                     base_url="http://fd.test") as c:
            for key in ("", "@system"):
                r = await c.post("/fd/basna/recommend", headers=hdr, json={
                    "intent": "x", "provider": "openai", "model": "gpt-4.1", "api_key": key, "base_url": EVIL})
                assert r.status_code == 502
            await c.post("/fd/basna/recommend", headers=hdr, json={
                "intent": "x", "provider": "openai", "model": "gpt-4.1", "base_url": ROUTED})
        assert sent == [(llm.WITHHELD_KEY, EVIL), (llm.WITHHELD_KEY, EVIL), (FD_KEY, ROUTED)]

    async def test_a_deck_without_accounts_installs_no_guard(self, deck, monkeypatch):
        monkeypatch.setattr(server, "AUTH_ENABLED", False)
        await server._guard_deck_keys()
        assert llm.create_provider("openai", "gpt-4.1", base_url=EVIL).api_key == FD_KEY

    def test_an_agent_process_installs_none(self, monkeypatch):
        monkeypatch.setattr(llm, "_ENV_KEY_GUARD", None)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-agents-own")
        assert llm.create_provider("openai", "gpt-4.1", base_url=EVIL).api_key == "sk-agents-own"


# ── what a process agent inherits ───────────────────────────────────────────

class TestInheritedKeys:
    async def test_withheld_from_an_agent_on_another_endpoint(self, deck):
        env = _withheld(_agent({"base_url": EVIL}))
        assert env["OPENAI_API_KEY"] == "" and env["ANTHROPIC_API_KEY"] == FD_ANTHROPIC

    @pytest.mark.parametrize("model,env", [
        ({}, ""),                                     # the provider's own endpoint
        ({"base_url": ROUTED}, ""),                   # the team's gateway
        ({"base_url": EVIL}, "OPENAI_API_KEY=sk-bobs-own\n"),   # its own key, its own business
        ({"base_url": EVIL, "api_key": "sk-in-config"}, ""),
    ])
    async def test_kept_where_it_may_go_or_isnt_used(self, deck, model, env):
        assert _withheld(_agent(model, env))["OPENAI_API_KEY"] == FD_KEY

    async def test_an_endpoint_the_environment_or_an_allowed_model_names(self, deck):
        env = _withheld(_agent({}, f"OPENAI_BASE_URL={EVIL}\n"))
        assert env["OPENAI_API_KEY"] == ""
        env = _withheld(_agent({"allowed": [{"id": "c", "provider": "anthropic", "model": "c", "base_url": EVIL}]}))
        assert (env["OPENAI_API_KEY"], env["ANTHROPIC_API_KEY"]) == (FD_KEY, "")

    async def test_an_admins_agent_and_a_deck_without_accounts_keep_it(self, deck, monkeypatch):
        d = _agent({"base_url": EVIL})
        monkeypatch.setattr(server, "_owner_is_admin", lambda owner: True)
        assert _withheld(d)["OPENAI_API_KEY"] == FD_KEY
        monkeypatch.setattr(server, "_owner_is_admin", lambda owner: False)
        monkeypatch.setattr(server, "AUTH_ENABLED", False)
        assert _withheld(d)["OPENAI_API_KEY"] == FD_KEY

    async def test_a_start_launches_it_without_the_key(self, deck, monkeypatch):
        launched: list[dict] = []

        class FakePopen:
            pid = 4242

            def __init__(self, args, cwd=None, env=None, **kw):
                launched.append(env)

        monkeypatch.setattr(server.subprocess, "Popen", FakePopen)
        monkeypatch.setattr(server, "_resolve_cc_web_bin", lambda: "captain-claw-web")
        _agent({"base_url": EVIL})
        _agent({}, slug="on-own")
        for slug in ("bob-agent", "on-own"):
            assert server._start_registered_process(slug, {"owner": BOB, "web_port": 24101})
        assert launched[0]["OPENAI_API_KEY"] == "" and launched[1]["OPENAI_API_KEY"] == FD_KEY
