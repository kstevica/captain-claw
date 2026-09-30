"""Publishing a tier set to the team must leave teammates with working keys.

The published copy of a set holds only the ``@system`` sentinel, resolved per
provider from the org keys (``fd:provider-keys``). An admin who published a set
without ever entering org keys left every teammate's agent with the team's
models and NO credentials ("Missing credentials … OPENAI_API_KEY"). So:

* publishing stores every key in the set as the org key for ITS endpoint, so a
  published tier works for a teammate the way it works in its owner's own
  space (the tier's key ends up in the agent's .env): the provider's own
  endpoint -> ``fd:provider-keys``; a custom ``base_url`` (a gateway, a local
  server) -> ``fd:endpoint-keys``, kept and only ever resolved per endpoint —
  it never becomes the provider's key and the provider's key is never sent
  there. It reports what it added / replaced / shared / what is still missing —
  provider ids and hostnames only, never key values;
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
    monkeypatch.setattr(basna_mod, "_SYSTEM_ENDPOINT_KEYS", {})
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

GW = "https://gw.example/v1"


def _published_key(stored: str, tier: str) -> str:
    return json.loads(stored)["sets"][0]["tiers"][tier]["api_key"]


async def _endpoint_keys(db) -> dict:
    raw = await db.get_system_setting("fd:endpoint-keys")
    return json.loads(raw) if raw else {}


def _team(res: dict) -> dict:
    return {k: v for k, v in res.items() if k.startswith("team_")}


class TestPublishMakesTeamKeys:
    async def test_a_key_in_the_set_becomes_the_team_key(self, db):
        res = await _publish(_set({"balanced": _tier(api_key="sk-admin-openai")}))
        assert _team(res) == {"team_keys_added": ["openai"], "team_keys_updated": [],
                              "team_endpoints_shared": [], "team_keys_conflict": [],
                              "team_keys_missing": []}
        assert await _org_keys(db) == {"openai": "sk-admin-openai"}
        # The published copy — what every user can read — holds only the sentinel.
        stored = await db.get_system_setting("fd:shared-tier-sets")
        assert "sk-admin-openai" not in stored
        assert _published_key(stored, "balanced") == "@system"
        assert "sk-admin-openai" not in json.dumps(res)

    async def test_a_different_key_replaces_the_team_key_and_says_so(self, db):
        """Re-publishing after rotating the key must not leave the team on the
        old one (which would work for the admin and fail for everyone else)."""
        await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "sk-old"}))
        res = await _publish(_set({"balanced": _tier(api_key="sk-new")}))
        assert res["team_keys_added"] == [] and res["team_keys_updated"] == ["openai"]
        assert await _org_keys(db) == {"openai": "sk-new"}
        assert "sk-" not in json.dumps(_team(res))

    async def test_republishing_the_same_key_reports_nothing(self, db):
        await _publish(_set({"balanced": _tier(api_key="sk-admin-openai")}))
        res = await _publish(_set({"balanced": _tier(api_key="sk-admin-openai")}))
        assert res["team_keys_added"] == res["team_keys_updated"] == []

    async def test_other_providers_keys_are_kept(self, db):
        await db.set_system_setting("fd:provider-keys", json.dumps({"anthropic": "sk-ant"}))
        await _publish(_set({"balanced": _tier(api_key="sk-admin-openai")}))
        assert await _org_keys(db) == {"anthropic": "sk-ant", "openai": "sk-admin-openai"}

    async def test_a_custom_endpoint_key_is_shared_for_that_endpoint_only(self, db):
        """A gateway's key (or a local server's placeholder) is that endpoint's:
        stored per endpoint, never as — and never blocking — the provider's."""
        res = await _publish(_set({"fast": _tier(api_key="lm-studio", base_url="http://localhost:1234/v1/")}))
        assert res["team_keys_added"] == [] and res["team_endpoints_shared"] == ["localhost:1234"]
        assert await _org_keys(db) == {}
        assert await _endpoint_keys(db) == {"openai|http://localhost:1234/v1": "lm-studio"}
        stored = await db.get_system_setting("fd:shared-tier-sets")
        assert "lm-studio" not in stored and _published_key(stored, "fast") == "@system"
        # …and a real provider key published later is adopted as usual.
        res = await _publish(_set({"balanced": _tier(api_key="sk-real-openai")}))
        assert res["team_keys_added"] == ["openai"]
        assert await _org_keys(db) == {"openai": "sk-real-openai"}

    async def test_mixed_set_keeps_each_key_with_its_own_endpoint(self, db):
        res = await _publish(_set({
            "fast": _tier(api_key="gateway-key", base_url=GW),
            "balanced": _tier(api_key="sk-real-openai"),
        }))
        assert res["team_keys_added"] == ["openai"] and res["team_endpoints_shared"] == ["gw.example"]
        assert await _org_keys(db) == {"openai": "sk-real-openai"}
        assert await _endpoint_keys(db) == {f"openai|{GW}": "gateway-key"}
        stored = await db.get_system_setting("fd:shared-tier-sets")
        assert "gateway-key" not in stored and "sk-real-openai" not in stored

    async def test_republishing_an_endpoint_replaces_its_key(self, db):
        await _publish(_set({"fast": _tier(api_key="old", base_url=GW)}))
        await _publish(_set({"fast": _tier(api_key="new", base_url=GW)}))
        assert await _endpoint_keys(db) == {f"openai|{GW}": "new"}

    async def test_a_gateway_key_that_is_the_org_key_keeps_following_it(self, db):
        """A deck that runs everything through one gateway on the org key:
        nothing is stored per endpoint, so rotating the key in Admin →
        Provider keys still reaches the gateway tiers."""
        await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "gw-old"}))
        await _publish(_set({"balanced": _tier(api_key="gw-old", base_url=GW)}))
        assert await _endpoint_keys(db) == {}
        await admin_routes.update_provider_keys(
            admin_routes.ProviderKeysRequest(keys={"openai": "gw-rotated"}), admin=ADMIN)
        assert "OPENAI_API_KEY=gw-rotated" in server._build_env(await _spawn_config())

    async def test_replacing_the_org_key_does_not_move_a_published_gateway_tier_onto_it(self, db):
        """The org key is the gateway's, and a published "@system" gateway tier
        runs on it. Publishing a real own-endpoint key replaces the org key —
        the gateway keeps ITS key instead of starting to receive the provider's."""
        gateway = _tier(api_key="@system", base_url=GW)
        real = _tier(api_key="sk-real-openai")
        cases = (
            ([_set({"balanced": gateway})], {"fast": real}),   # named by the set published now
            ([], {"balanced": gateway, "fast": real}),         # or by the one being published
        )
        for stored, tiers in cases:
            await db.set_system_setting("fd:endpoint-keys", "{}")
            await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "gateway-key"}))
            await db.set_system_setting(
                "fd:shared-tier-sets", json.dumps({"sets": stored, "defaultSetId": "s1"}))
            res = await _publish(_set(tiers))
            assert res["team_keys_updated"] == ["openai"]
            assert await _org_keys(db) == {"openai": "sk-real-openai"}
            assert await _endpoint_keys(db) == {f"openai|{GW}": "gateway-key"}

    async def test_the_org_key_is_kept_when_any_tier_of_the_set_carries_it(self, db):
        """Tier order must not decide: a set holding the working org key and
        another key for the same provider leaves the org key alone — and says
        the set has two."""
        await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "sk-good"}))
        for tiers in ({"fast": _tier(api_key="sk-other"), "balanced": _tier(api_key="sk-good")},
                      {"balanced": _tier(api_key="sk-good"), "fast": _tier(api_key="sk-other")}):
            res = await _publish(_set(tiers))
            assert res["team_keys_updated"] == [] and res["team_keys_conflict"] == ["openai"]
            assert await _org_keys(db) == {"openai": "sk-good"}

    async def test_two_keys_for_one_endpoint_are_reported(self, db):
        res = await _publish(_set({"fast": _tier(api_key="key-a", base_url=GW),
                                   "balanced": _tier(api_key="key-b", base_url=GW + "/")}))
        assert res["team_keys_conflict"] == ["gw.example"]
        assert await _endpoint_keys(db) == {f"openai|{GW}": "key-a"}
        assert "key-a" not in json.dumps(_team(res)) and "key-b" not in json.dumps(_team(res))

    async def test_the_endpoint_label_never_carries_a_credential(self, db):
        res = await _publish(_set({
            "fast": _tier(api_key="k1", base_url="https://user:s3cret@gw.example/v1"),
            "balanced": _tier(api_key="k2", base_url="gw2.example:8443/v1?api_key=tok123"),
        }))
        assert res["team_endpoints_shared"] == ["gw.example", "gw2.example:8443"]
        assert "s3cret" not in json.dumps(_team(res)) and "tok123" not in json.dumps(_team(res))

    async def test_providers_without_a_key_are_never_promoted(self, db):
        res = await _publish(_set({
            "micro": _tier(provider="ollama", model="qwen", api_key="ollama"),
            "fast": _tier(provider="browser", model="m", api_key="none"),
        }))
        assert res["team_keys_added"] == [] and await _org_keys(db) == {}

    async def test_whitespace_key_is_a_blank_key(self, db):
        await _publish(_set({"fast": _tier(api_key="   ", base_url="http://localhost:1234/v1")}))
        assert _published_key(await db.get_system_setting("fd:shared-tier-sets"), "fast") == ""
        assert await _endpoint_keys(db) == {}

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
        await _publish(_set({"balanced": _tier(api_key="sk-admin-openai"),
                             "fast": _tier(api_key="gateway-key", base_url=GW)}))
        await basna_mod._refresh_system_provider_keys(db)
        assert basna_mod._effective_key("openai", "@system") == "sk-admin-openai"
        assert basna_mod._effective_key("openai", "@system", GW) == "gateway-key"
        monkeypatch.setattr(basna_mod, "_SYSTEM_KEYS_TS", 10**12)
        await admin_routes.update_provider_keys(
            admin_routes.ProviderKeysRequest(keys={"openai": "sk-rotated"}), admin=ADMIN)
        await basna_mod._refresh_system_provider_keys(db)
        assert basna_mod._effective_key("openai", "@system") == "sk-rotated"

    async def test_an_endpoint_only_publish_drops_the_cache_too(self, db, monkeypatch):
        monkeypatch.setattr(basna_mod, "_SYSTEM_PROVIDER_KEYS", {"anthropic": "old"})
        monkeypatch.setattr(basna_mod, "_SYSTEM_KEYS_TS", 10**12)
        await _publish(_set({"fast": _tier(api_key="gateway-key", base_url=GW)}))
        await basna_mod._refresh_system_provider_keys(db)
        # …and the endpoint is matched whatever the URL's case or trailing slash.
        assert basna_mod._effective_key("openai", "@system", "HTTPS://GW.example/v1/ ") == "gateway-key"


# ── a new user's agent, end to end ──────────────────────────────────────────


class TestNewUserAgentGetsTheTeamKey:
    async def test_published_set_reaches_a_user_with_no_set_of_their_own(self, db):
        await _publish(_set({"balanced": _tier(api_key="sk-admin-openai")},
                            env=[{"key": "BRAVE_API_KEY", "value": "brave-1"}]))
        cfg = await _spawn_config()
        assert (cfg.provider, cfg.model, cfg.provider_api_key) == ("openai", "gpt-6", "sk-admin-openai")
        env = server._build_env(cfg)
        assert "OPENAI_API_KEY=sk-admin-openai" in env and "BRAVE_API_KEY=brave-1" in env

    async def test_a_published_custom_endpoint_tier_puts_its_key_in_the_agents_env(self, db):
        """The reported case: tiers on an OpenAI-compatible endpoint (provider
        openai + base_url), published — a teammate's agent got the set's
        BRAVE_API_KEY but no model key."""
        await _publish(_set({"balanced": _tier(model="swift", api_key="gateway-key", base_url=GW)},
                            env=[{"key": "BRAVE_API_KEY", "value": "brave-1"}]))
        cfg = await _spawn_config()
        assert (cfg.provider, cfg.model, cfg.base_url) == ("openai", "swift", GW)
        env = server._build_env(cfg)
        assert "OPENAI_API_KEY=gateway-key" in env and "BRAVE_API_KEY=brave-1" in env

    async def test_in_a_mixed_set_each_tier_runs_on_its_own_key(self, db):
        await _publish(_set({
            "balanced": _tier(api_key="gateway-key", base_url=GW),
            "fast": _tier(api_key="sk-real-openai"),
        }))
        cfg = await _spawn_config()   # the archetype's tier is "balanced": the gateway
        env = server._build_env(cfg)
        assert cfg.base_url == GW and "OPENAI_API_KEY=gateway-key" in env
        assert "sk-real-openai" not in env     # the provider's key never goes to the gateway

    async def test_a_teammates_adopted_copy_resolves_the_same_way(self, db):
        """A user whose own set is a copy of the team default holds "@system"
        on a custom-endpoint tier: it resolves to that endpoint's key."""
        await _publish(_set({"balanced": _tier(api_key="gateway-key", base_url=GW)}))
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES ('new-user', 'n@x.co', 'h', 'n', 'user', '2026-01-01', '2026-01-01')")
        await db._db.commit()
        await db.set_settings("new-user", {"fd:forge-tiers": json.dumps({"sets": [_set(
            {"balanced": _tier(api_key="@system", base_url=GW)})], "activeSetId": "s1"})})
        cfg = await _spawn_config()
        assert "OPENAI_API_KEY=gateway-key" in server._build_env(cfg)

    async def test_a_copy_taken_before_endpoints_had_keys_is_healed(self, db):
        """Published before this change, a custom-endpoint tier went out with a
        BLANK key, and teammates' own copies of that set still hold it. The
        endpoint's team key reaches them anyway — its own key only."""
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES ('new-user', 'n@x.co', 'h', 'n', 'user', '2026-01-01', '2026-01-01')")
        await db._db.commit()
        await db.set_settings("new-user", {"fd:forge-tiers": json.dumps({"sets": [_set(
            {"balanced": _tier(api_key="", base_url=GW)},
            env=[{"key": "BRAVE_API_KEY", "value": "brave-1"}])], "activeSetId": "s1"})})
        await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "sk-org"}))
        env = server._build_env(await _spawn_config())
        assert "OPENAI_API_KEY" not in env          # no endpoint key yet: blank stays blank,
        assert "sk-org" not in env                  # never the provider's org key
        await _publish(_set({"balanced": _tier(api_key="gateway-key", base_url=GW)}))
        env = server._build_env(await _spawn_config())
        assert "OPENAI_API_KEY=gateway-key" in env and "BRAVE_API_KEY=brave-1" in env

    async def test_a_blank_key_never_overrides_the_users_own_env_key(self, db):
        await _publish(_set({"balanced": _tier(api_key="gateway-key", base_url=GW)}))
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES ('new-user', 'n@x.co', 'h', 'n', 'user', '2026-01-01', '2026-01-01')")
        await db._db.commit()
        await db.set_settings("new-user", {"fd:forge-tiers": json.dumps({"sets": [_set(
            {"balanced": _tier(api_key="", base_url=GW)},
            env=[{"key": "OPENAI_API_KEY", "value": "my-own-key"}])], "activeSetId": "s1"})})
        env = server._build_env(await _spawn_config())
        assert "OPENAI_API_KEY=my-own-key" in env and "gateway-key" not in env

    async def test_basna_and_vatra_resolve_each_tier_on_its_own_key(self, db):
        """The run paths (lead calls, merges) — not only agent spawns."""
        import types

        from captain_claw.flight_deck import vatra_routes

        await _publish(_set({
            "balanced": _tier(api_key="gateway-key", base_url=GW),
            "fast": _tier(api_key="sk-real-openai"),
        }))
        tiers, _env = await basna_mod._load_owner_tiers(db, "new-user")
        body = types.SimpleNamespace(tiers=tiers, api_key="")
        for tier, want in (("balanced", "gateway-key"), ("fast", "sk-real-openai")):
            assert vatra_routes._resolve_creds({}, tiers, "", tier)["api_key"] == want
            assert basna_mod._resolve_merge_creds(body, {}, tier)["api_key"] == want
            assert basna_mod._tier_creds({"tiers": tiers}, tier, "@system")["api_key"] == want

    async def test_an_endpoint_with_no_key_of_its_own_falls_back_to_the_providers(self, db):
        """As before this change: a deck that runs everything through one
        gateway with the org key."""
        await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "sk-org"}))
        await db.set_system_setting("fd:shared-tier-sets", json.dumps({"sets": [_set(
            {"balanced": _tier(api_key="@system", base_url=GW)})], "defaultSetId": "s1"}))
        cfg = await _spawn_config()
        assert "OPENAI_API_KEY=sk-org" in server._build_env(cfg)

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
        await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "sk-org"}))
        await db.set_system_setting("fd:endpoint-keys", json.dumps({f"openai|{GW}": "gateway-key"}))
        for key, base_url in (("sk-mine", ""), ("sk-mine", GW), ("", ""), ("", "https://other.example/v1")):
            cfg = server.AgentConfig(name="x", provider="openai", provider_api_key=key, base_url=base_url)
            await server._resolve_spawn_provider_key(cfg)
            assert cfg.provider_api_key == key
