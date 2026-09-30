"""Publishing a tier set to the team must leave teammates with working keys.

The published copy of a set holds only the ``@system`` sentinel, resolved per
provider from the org keys (``fd:provider-keys``). An admin who published a set
without ever entering org keys left every teammate's agent with the team's
models and NO credentials ("Missing credentials … OPENAI_API_KEY"). So:

* publishing makes a key on a provider's OWN endpoint the org key for that
  provider when none is set — never replacing one, never promoting a
  custom-endpoint key (a gateway's, or a local server's placeholder), and never
  publishing ``@system`` where it would send the provider's key to another host;
  it reports what it added / what differs / what isn't shared / what is still
  missing — provider ids and tier names only, never key values;
* a spawn that asks for the team key when there is none is refused (409)
  instead of starting a dead agent — only where the agent really needs the
  provider's key (not ChatGPT-signed-in models, not custom endpoints);
* an unresolvable ``@system`` doesn't shadow the set's own provider env var.

Real FlightDeckDB in a tmp dir; no network, nothing spawned.
"""

from __future__ import annotations

import json
import types

import pytest
from fastapi import HTTPException

import captain_claw.flight_deck.admin_routes as admin_routes
import captain_claw.flight_deck.archetypes as arch_mod
import captain_claw.flight_deck.auth as auth_mod
import captain_claw.flight_deck.basna_routes as basna_mod
import captain_claw.flight_deck.server as server
from captain_claw.flight_deck.db import FlightDeckDB

ADMIN = {"id": "admin-1", "role": "admin"}
_ARCH = {"id": "market-research", "role": "Market Research Specialist", "tier": "balanced"}


def _set(tiers: dict, env: list | None = None, sid: str = "s1") -> dict:
    return {"id": sid, "name": "Team", "tiers": tiers, "forgeTier": "balanced", "envVars": env or []}


def _tier(provider="openai", model="gpt-6", api_key="", base_url="") -> dict:
    return {"provider": provider, "model": model, "api_key": api_key, "base_url": base_url}


@pytest.fixture()
async def db(monkeypatch, tmp_path):
    fd_db = FlightDeckDB(tmp_path / "flight-deck.db")
    await fd_db.init()
    monkeypatch.setattr(auth_mod, "_db", fd_db)
    # The org-key cache is process-global: start every test cold.
    monkeypatch.setattr(basna_mod, "_SYSTEM_PROVIDER_KEYS", {})
    monkeypatch.setattr(basna_mod, "_SYSTEM_KEYS_TS", 0.0)
    for name in server._PROVIDER_KEY_ENV.values():
        monkeypatch.delenv(name, raising=False)

    async def fake_merged(_db, _uid):
        return [dict(_ARCH)]

    monkeypatch.setattr(arch_mod, "merged_archetypes", fake_merged)
    yield fd_db
    await fd_db.close()


async def _publish(*sets: dict) -> dict:
    body = admin_routes.SharedTierSetsRequest(sets=list(sets), defaultSetId=sets[0]["id"])
    return await admin_routes.update_shared_tier_sets(body, admin=ADMIN)


async def _org_keys(db) -> dict:
    raw = await db.get_system_setting("fd:provider-keys")
    return json.loads(raw) if raw else {}


def _req(uid: str):
    return types.SimpleNamespace(state=types.SimpleNamespace(user_id=uid))


async def _spawn_config(uid: str = "new-user") -> server.AgentConfig:
    """What the kiosk picker's spawn resolves to for ``uid`` (no personal set)."""
    cfg = server.AgentConfig(name="x", archetype="market-research")
    await server._resolve_archetype(cfg, _req(uid), None)
    await server._resolve_spawn_provider_key(cfg)
    return cfg


# ── publishing ──────────────────────────────────────────────────────────────


def _published_key(stored: str, tier: str) -> str:
    return json.loads(stored)["sets"][0]["tiers"][tier]["api_key"]


class TestPublishMakesTeamKeys:
    async def test_a_key_in_the_set_becomes_the_team_key(self, db):
        res = await _publish(_set({"balanced": _tier(api_key="sk-admin-openai")}))
        assert res["team_keys_added"] == ["openai"]
        assert res["team_keys_differ"] == res["team_keys_unshared"] == res["team_keys_missing"] == []
        assert await _org_keys(db) == {"openai": "sk-admin-openai"}
        # The published copy — what every user can read — holds only the sentinel.
        stored = await db.get_system_setting("fd:shared-tier-sets")
        assert "sk-admin-openai" not in stored
        assert _published_key(stored, "balanced") == "@system"
        assert "sk-admin-openai" not in json.dumps(res)

    async def test_an_existing_team_key_is_kept_and_the_difference_reported(self, db):
        await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "sk-org"}))
        res = await _publish(_set({"balanced": _tier(api_key="sk-admin-new")}))
        assert res["team_keys_added"] == [] and res["team_keys_differ"] == ["openai"]
        assert await _org_keys(db) == {"openai": "sk-org"}
        assert "sk-" not in json.dumps({k: v for k, v in res.items() if k.startswith("team_")})

    async def test_republishing_the_same_key_reports_nothing(self, db):
        await _publish(_set({"balanced": _tier(api_key="sk-admin-openai")}))
        res = await _publish(_set({"balanced": _tier(api_key="sk-admin-openai")}))
        assert res["team_keys_added"] == res["team_keys_differ"] == []

    async def test_other_providers_keys_are_kept(self, db):
        await db.set_system_setting("fd:provider-keys", json.dumps({"anthropic": "sk-ant"}))
        await _publish(_set({"balanced": _tier(api_key="sk-admin-openai")}))
        assert await _org_keys(db) == {"anthropic": "sk-ant", "openai": "sk-admin-openai"}

    async def test_a_custom_endpoint_key_is_never_promoted(self, db):
        """A local server's placeholder ("lm-studio") or a gateway's key is not
        the provider's key: it must not become — and later block — the org key."""
        res = await _publish(_set({
            "fast": _tier(api_key="lm-studio", base_url="http://localhost:1234/v1")}))
        assert res["team_keys_added"] == [] and res["team_keys_unshared"] == ["fast"]
        assert await _org_keys(db) == {}
        stored = await db.get_system_setting("fd:shared-tier-sets")
        assert "lm-studio" not in stored and _published_key(stored, "fast") == ""
        # …so a real key published later is still adopted.
        res = await _publish(_set({"balanced": _tier(api_key="sk-real-openai")}))
        assert res["team_keys_added"] == ["openai"]

    async def test_mixed_set_never_sends_the_real_key_to_the_other_endpoint(self, db):
        res = await _publish(_set({
            "fast": _tier(api_key="gateway-key", base_url="https://gw.example/v1"),
            "balanced": _tier(api_key="sk-real-openai"),
        }))
        assert res["team_keys_added"] == ["openai"] and res["team_keys_unshared"] == ["fast"]
        assert await _org_keys(db) == {"openai": "sk-real-openai"}
        stored = await db.get_system_setting("fd:shared-tier-sets")
        assert _published_key(stored, "balanced") == "@system"
        assert _published_key(stored, "fast") == ""          # not "@system"
        assert "gateway-key" not in stored

    async def test_a_deck_run_through_one_gateway_keeps_working(self, db):
        """Org key deliberately IS the gateway key: "@system" resolves to the
        very key the admin put on that tier, so it stays shareable."""
        await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "gateway-key"}))
        res = await _publish(_set({
            "fast": _tier(api_key="gateway-key", base_url="https://gw.example/v1")}))
        assert res["team_keys_unshared"] == []
        assert _published_key(await db.get_system_setting("fd:shared-tier-sets"), "fast") == "@system"

    async def test_providers_without_a_key_are_never_promoted(self, db):
        res = await _publish(_set({
            "micro": _tier(provider="ollama", model="qwen", api_key="ollama"),
            "fast": _tier(provider="browser", model="m", api_key="none"),
        }))
        assert res["team_keys_added"] == [] and await _org_keys(db) == {}

    async def test_whitespace_key_is_a_blank_key(self, db):
        await _publish(_set({"fast": _tier(api_key="   ", base_url="http://localhost:1234/v1")}))
        assert _published_key(await db.get_system_setting("fd:shared-tier-sets"), "fast") == ""

    async def test_missing_is_reported_when_nothing_supplies_a_key(self, db):
        res = await _publish(_set({
            "balanced": _tier(api_key="@system"),                  # adopted set: no real key
            "fast": _tier(provider="anthropic", model="claude"),   # blank key, own endpoint
            "micro": _tier(provider="ollama", model="qwen"),       # needs no key
            "local": _tier(provider="xai", model="m", base_url="http://localhost:1234/v1"),
            "coding": _tier(provider="gemini", model="gemini-pro"),
        }))
        assert res["team_keys_added"] == []
        assert res["team_keys_missing"] == ["anthropic", "gemini", "openai"]
        assert await _org_keys(db) == {}

    async def test_models_signed_in_through_chatgpt_need_no_team_key(self, db):
        res = await _publish(_set({"coding": _tier(model="gpt-5.2-codex")}))
        assert res["team_keys_missing"] == []

    async def test_the_sets_own_env_var_or_fd_env_counts_as_supplied(self, db, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-fd-env")
        res = await _publish(_set(
            {"balanced": _tier(), "fast": _tier(provider="anthropic", model="claude"),
             "coding": _tier(provider="gemini", model="gemini-pro")},
            env=[{"key": "OPENAI_API_KEY", "value": "sk-in-set-env"},
                 {"key": "GOOGLE_API_KEY", "value": "g-key"}],   # gemini's other name
        ))
        assert res["team_keys_missing"] == []

    async def test_one_sets_env_var_does_not_cover_another_set(self, db):
        res = await _publish(
            _set({"balanced": _tier()}, env=[{"key": "OPENAI_API_KEY", "value": "k"}], sid="s1"),
            _set({"balanced": _tier(provider="anthropic", model="claude")}, sid="s2"),
        )
        assert res["team_keys_missing"] == ["anthropic"]

    async def test_every_admin_key_write_drops_the_cached_org_keys(self, db, monkeypatch):
        monkeypatch.setattr(basna_mod, "_SYSTEM_PROVIDER_KEYS", {"anthropic": "old"})
        monkeypatch.setattr(basna_mod, "_SYSTEM_KEYS_TS", 10**12)  # "fresh" forever
        await _publish(_set({"balanced": _tier(api_key="sk-admin-openai")}))
        await basna_mod._refresh_system_provider_keys(db)
        assert basna_mod._effective_key("openai", "@system") == "sk-admin-openai"
        monkeypatch.setattr(basna_mod, "_SYSTEM_KEYS_TS", 10**12)
        await admin_routes.update_provider_keys(
            admin_routes.ProviderKeysRequest(keys={"openai": "sk-rotated"}), admin=ADMIN)
        await basna_mod._refresh_system_provider_keys(db)
        assert basna_mod._effective_key("openai", "@system") == "sk-rotated"


# ── a new user's agent, end to end ──────────────────────────────────────────


class TestNewUserAgentGetsTheTeamKey:
    async def test_published_set_reaches_a_user_with_no_set_of_their_own(self, db):
        await _publish(_set({"balanced": _tier(api_key="sk-admin-openai")},
                            env=[{"key": "BRAVE_API_KEY", "value": "brave-1"}]))
        cfg = await _spawn_config()
        assert (cfg.provider, cfg.model, cfg.provider_api_key) == ("openai", "gpt-6", "sk-admin-openai")
        env = server._build_env(cfg)
        assert "OPENAI_API_KEY=sk-admin-openai" in env and "BRAVE_API_KEY=brave-1" in env

    async def test_teammate_on_a_custom_endpoint_tier_never_gets_the_real_key(self, db):
        await _publish(_set({
            "balanced": _tier(api_key="gateway-key", base_url="https://gw.example/v1"),
            "fast": _tier(api_key="sk-real-openai"),
        }))
        cfg = await _spawn_config()   # the archetype's tier is "balanced"
        assert cfg.base_url == "https://gw.example/v1" and cfg.provider_api_key == ""
        assert "sk-real-openai" not in server._build_env(cfg)

    async def test_no_team_key_is_a_clear_refusal_not_a_dead_agent(self, db):
        # The pre-fix state of a deck: published (masked) set, org key never entered.
        await db.set_system_setting("fd:shared-tier-sets", json.dumps(
            {"sets": [_set({"balanced": _tier(api_key="@system")})], "defaultSetId": "s1"}))
        with pytest.raises(HTTPException) as exc:
            await _spawn_config()
        assert exc.value.status_code == 409
        assert "No team API key for openai" in exc.value.detail
        assert "Provider keys" in exc.value.detail

    async def test_unresolvable_system_key_lets_the_sets_own_env_var_through(self, db):
        await db.set_system_setting("fd:shared-tier-sets", json.dumps({"sets": [_set(
            {"balanced": _tier(api_key="@system")},
            env=[{"key": "OPENAI_API_KEY", "value": "sk-in-set-env"}])], "defaultSetId": "s1"}))
        cfg = await _spawn_config()
        assert cfg.provider_api_key == ""
        assert "OPENAI_API_KEY=sk-in-set-env" in server._build_env(cfg)

    async def test_resolvable_system_key_still_wins_over_the_sets_env_var(self, db):
        await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "sk-org"}))
        await db.set_system_setting("fd:shared-tier-sets", json.dumps({"sets": [_set(
            {"balanced": _tier(api_key="@system")},
            env=[{"key": "OPENAI_API_KEY", "value": "stale-extra"}])], "defaultSetId": "s1"}))
        cfg = await _spawn_config()
        env = server._build_env(cfg)
        assert "OPENAI_API_KEY=sk-org" in env and "stale-extra" not in env


# ── the spawn-time resolver on its own ──────────────────────────────────────


class TestSpawnProviderKey:
    async def test_fd_environment_key_is_enough(self, db, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "from-fd-env")  # process agents inherit it
        cfg = server.AgentConfig(name="x", provider="openai", provider_api_key="@system")
        await server._resolve_spawn_provider_key(cfg)
        assert cfg.provider_api_key == ""

    async def test_providers_without_a_key_env_are_never_refused(self, db):
        cfg = server.AgentConfig(name="x", provider="ollama", provider_api_key="@system")
        await server._resolve_spawn_provider_key(cfg)
        assert cfg.provider_api_key == ""

    async def test_a_container_does_not_inherit_the_fd_environment(self, db, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "from-fd-env")
        cfg = server.AgentConfig(name="x", provider="openai", provider_api_key="@system")
        with pytest.raises(HTTPException) as exc:
            await server._resolve_spawn_provider_key(cfg, inherits_fd_env=False)
        assert exc.value.status_code == 409

    async def test_chatgpt_signed_in_models_are_never_refused(self, db):
        cfg = server.AgentConfig(name="x", provider="openai", model="gpt-5.2",
                                 provider_api_key="@system")
        await server._resolve_spawn_provider_key(cfg)
        assert cfg.provider_api_key == ""

    async def test_a_custom_endpoint_is_never_refused(self, db):
        cfg = server.AgentConfig(name="x", provider="openai", model="local-model",
                                 base_url="http://localhost:1234/v1", provider_api_key="@system")
        await server._resolve_spawn_provider_key(cfg)
        assert cfg.provider_api_key == ""

    async def test_gemini_key_under_its_other_name_is_enough(self, db):
        cfg = server.AgentConfig(name="x", provider="gemini", model="gemini-pro",
                                 provider_api_key="@system",
                                 env_vars=[{"key": "GOOGLE_API_KEY", "value": "g-key"}])
        await server._resolve_spawn_provider_key(cfg)
        assert cfg.provider_api_key == ""

    async def test_a_real_or_blank_key_is_left_alone(self, db):
        for key in ("sk-mine", ""):
            cfg = server.AgentConfig(name="x", provider="openai", provider_api_key=key)
            await server._resolve_spawn_provider_key(cfg)
            assert cfg.provider_api_key == key
