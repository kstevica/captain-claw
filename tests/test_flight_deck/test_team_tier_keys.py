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
  instead of starting a dead agent — unless the model needs no key
  (ChatGPT-signed-in models, providers without one) or it arrives another way;
* an unresolvable ``@system`` doesn't shadow the set's own provider env var;
* nobody has to publish again or recreate agents for any of it: a set that is
  already published gets its keys from the publishing admin's own copy, a
  blank key in a user's own set falls back to the team's key, and an agent
  created without a model key is given it.

Real FlightDeckDB in a tmp dir; no network, nothing spawned.
"""

from __future__ import annotations

import asyncio
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
    monkeypatch.setattr(basna_mod, "_SYSTEM_NO_FALLBACK", frozenset())
    monkeypatch.setattr(basna_mod, "_SYSTEM_KEYS_TS", 0.0)
    monkeypatch.setattr(admin_routes, "_TEAM_KEYS_LOCK", None)
    monkeypatch.setattr(admin_routes, "_agent_key_healer", None)
    monkeypatch.setattr(server, "_processes", {})
    # Agents live on disk under DATA_DIR — never the developer's real ones.
    monkeypatch.setattr(server, "DATA_DIR", tmp_path / "fd-data")
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", tmp_path / "fd-data" / ".processes.json")
    (tmp_path / "fd-data").mkdir()
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


async def _add_user(db, uid: str, role: str = "user", own_set: dict | None = None) -> None:
    await db._db.execute(
        "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
        " VALUES (?, ?, 'h', ?, ?, '2026-01-01', '2026-01-01')", (uid, f"{uid}@x.co", uid, role))
    await db._db.commit()
    if own_set is not None:
        await db.set_settings(uid, {"fd:forge-tiers": json.dumps(
            {"sets": [own_set], "activeSetId": own_set["id"]})})


async def _already_published(db, *sets: dict) -> None:
    """A set as an EARLIER Flight Deck left it in the store: no team keys."""
    await db.set_system_setting(
        "fd:shared-tier-sets", json.dumps({"sets": list(sets), "defaultSetId": sets[0]["id"]}))


async def _spawn_config(uid: str = "new-user") -> server.AgentConfig:
    """What the kiosk picker's spawn resolves to for ``uid`` (no personal set)."""
    cfg = server.AgentConfig(name="x", archetype="market-research")
    await server._resolve_archetype(cfg, _req(uid), None)
    server._resolve_tier(cfg)
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
        with pytest.raises(HTTPException) as exc:   # no endpoint key yet — and never the
            await _spawn_config()                   # provider's org key for that host
        assert exc.value.status_code == 409 and "gw.example" in exc.value.detail
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
        assert "No API key for openai:" in exc.value.detail
        assert "Provider keys" in exc.value.detail

    async def test_an_archetype_agent_is_never_created_without_a_model_key(self, db):
        """Flight Deck picked the model, so the caller had no say in the key: a
        BLANK one with nothing behind it is refused as well — also when the
        tier isn't in the set at all and the registry's model is used."""
        for tiers in ({"balanced": _tier(api_key="")},                      # own endpoint
                      {"balanced": _tier(api_key="", base_url=GW)},          # a gateway
                      {"fast": _tier(api_key="sk-x")}):                      # no "balanced" tier
            await _already_published(db, _set(tiers))
            with pytest.raises(HTTPException) as exc:
                await _spawn_config()
            assert exc.value.status_code == 409 and "No API key for" in exc.value.detail

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

    async def test_a_custom_endpoint_with_no_team_key_is_refused_too(self, db):
        """"@system" says the published tier HAD a key; an agent without it
        can't make one call ("Missing credentials") — say so at creation."""
        cfg = server.AgentConfig(name="x", provider="openai", model="local-model",
                                 base_url="http://user:pw@localhost:1234/v1", provider_api_key="@system")
        with pytest.raises(HTTPException) as exc:
            await server._resolve_spawn_provider_key(cfg)
        assert exc.value.status_code == 409
        assert "openai at localhost:1234" in exc.value.detail and "pw" not in exc.value.detail

    async def test_a_keyless_provider_on_a_custom_endpoint_is_never_refused(self, db):
        cfg = server.AgentConfig(name="x", provider="ollama", model="qwen",
                                 base_url="http://localhost:11434", provider_api_key="@system")
        await server._resolve_spawn_provider_key(cfg)
        assert cfg.provider_api_key == ""

    async def test_gemini_key_under_its_other_name_is_enough(self, db):
        cfg = server.AgentConfig(name="x", provider="gemini", model="gemini-pro",
                                 provider_api_key="@system",
                                 env_vars=[{"key": "GOOGLE_API_KEY", "value": "g-key"}])
        await server._resolve_spawn_provider_key(cfg)
        assert cfg.provider_api_key == ""

    async def test_a_real_key_is_left_alone(self, db):
        await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "sk-org"}))
        await db.set_system_setting("fd:endpoint-keys", json.dumps({f"openai|{GW}": "gateway-key"}))
        for base_url in ("", GW):
            cfg = server.AgentConfig(name="x", provider="openai", provider_api_key="sk-mine", base_url=base_url)
            await server._resolve_spawn_provider_key(cfg)
            assert cfg.provider_api_key == "sk-mine"


# ── a blank key in a user's own set ─────────────────────────────────────────

class TestBlankKeyFallsBackToTheTeams:
    """A user's own set (often an automatic copy of a team set) with no key for
    a model the team has one for: the agent gets the team's — never over a key
    the spawn brings itself, and never the provider's key for a foreign host."""

    async def _cfg(self, **kw) -> server.AgentConfig:
        cfg = server.AgentConfig(name="x", provider="openai", model="gpt-4.1", **kw)
        await server._resolve_spawn_provider_key(cfg)
        return cfg

    async def test_own_endpoint_gets_the_providers_team_key(self, db):
        await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "sk-org"}))
        assert (await self._cfg()).provider_api_key == "sk-org"

    async def test_no_team_key_is_no_refusal(self, db):
        assert (await self._cfg()).provider_api_key == ""
        assert (await self._cfg(base_url=GW)).provider_api_key == ""

    async def test_a_key_in_the_spawns_env_vars_wins(self, db):
        await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "sk-org"}))
        cfg = await self._cfg(env_vars=[{"key": "OPENAI_API_KEY", "value": "sk-mine"}])
        assert cfg.provider_api_key == "" and "sk-org" not in server._build_env(cfg)

    async def test_a_key_inherited_from_flight_decks_environment_is_kept(self, db, monkeypatch):
        """A process agent with a blank key already runs on the deck's
        environment key — it keeps it; a container doesn't inherit, so it
        gets the team's."""
        monkeypatch.setenv("OPENAI_API_KEY", "fd-env-key")
        await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "sk-org"}))
        assert (await self._cfg()).provider_api_key == ""
        cfg = server.AgentConfig(name="x", provider="openai", model="gpt-4.1")
        await server._resolve_spawn_provider_key(cfg, inherits_fd_env=False)
        assert cfg.provider_api_key == "sk-org"

    async def test_flight_decks_provider_key_is_not_a_gateways_key(self, db, monkeypatch):
        """On a custom endpoint the inherited key is the wrong one (it would be
        sent to that host): the endpoint's team key still applies."""
        monkeypatch.setenv("OPENAI_API_KEY", "fd-env-openai-key")
        await db.set_system_setting("fd:endpoint-keys", json.dumps({f"openai|{GW}": "gateway-key"}))
        assert (await self._cfg(base_url=GW)).provider_api_key == "gateway-key"
        # …and a gateway that really runs on the environment's key is not refused.
        cfg = server.AgentConfig(name="x", provider="openai", model="m", archetype="market-research",
                                 base_url="https://other.example/v1")
        await server._resolve_spawn_provider_key(cfg)
        assert cfg.provider_api_key == ""

    async def test_a_keyless_local_server_behind_a_keyed_provider_is_not_refused(self, db):
        for provider in ("openrouter", "xai"):
            cfg = server.AgentConfig(name="x", provider=provider, model="m", archetype="market-research",
                                     base_url="http://localhost:1234/v1")
            await server._resolve_spawn_provider_key(cfg)
            assert cfg.provider_api_key == ""
        cfg = server.AgentConfig(name="x", provider="openai", model="m", archetype="market-research",
                                 base_url="http://localhost:1234/v1")
        with pytest.raises(HTTPException):       # the OpenAI client makes no call without a key
            await server._resolve_spawn_provider_key(cfg)

    async def test_a_failing_lookup_leaves_the_spawn_alone(self, db, monkeypatch):
        async def boom(*_a, **_k):
            raise RuntimeError("db gone")

        monkeypatch.setattr(server, "_team_key", boom)
        cfg = server.AgentConfig(name="x", provider="openai", model="gpt-4.1", provider_api_key="@system")
        await server._resolve_spawn_provider_key(cfg)
        assert cfg.provider_api_key == ""

    async def test_chatgpt_signed_in_models_get_no_api_key(self, db):
        await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "sk-org"}))
        cfg = server.AgentConfig(name="x", provider="openai", model="gpt-5.2")
        await server._resolve_spawn_provider_key(cfg)
        assert cfg.provider_api_key == ""

    async def test_a_foreign_endpoint_never_gets_the_providers_key(self, db):
        await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "sk-org"}))
        await db.set_system_setting("fd:endpoint-keys", json.dumps({f"openai|{GW}": "gateway-key"}))
        assert (await self._cfg(base_url="https://other.example/v1")).provider_api_key == ""
        assert (await self._cfg(base_url=GW + "/")).provider_api_key == "gateway-key"

    async def test_an_endpoint_the_team_runs_on_the_providers_key(self, db):
        """One gateway for the whole deck, on the org key: the published tier is
        "@system" with no endpoint key. A user's blank copy resolves like it."""
        await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "gw-key"}))
        await _publish(_set({"balanced": _tier(api_key="gw-key", base_url=GW)}))
        assert await _endpoint_keys(db) == {}
        assert (await self._cfg(base_url=GW)).provider_api_key == "gw-key"
        await _add_user(db, "new-user", own_set=_set({"balanced": _tier(api_key="", base_url=GW)}))
        assert "OPENAI_API_KEY=gw-key" in server._build_env(await _spawn_config())


def _seed_set(**changes) -> dict:
    """What the UI saves on a first visit: every registry tier, its default model, no keys."""
    tiers = {name: {"provider": t["provider"], "model": t["model"], "api_key": "", "base_url": ""}
             for name, t in basna_mod._load_registry()["tiers"].items()}
    tiers["balanced"].update(changes)
    return {"id": "seed", "name": "Default", "envVars": [], "tiers": tiers}


class TestAnUntouchedSeedSetRidesTheTeamDefault:
    """The UI saves a registry-seeded set on a user's first visit when it has
    no team set to copy yet. Nobody chose it — so it must not shadow the team
    default forever. A set somebody DID choose is theirs, however keyless."""

    TEAM = [{"key": "BRAVE_API_KEY", "value": "brave-1"}]

    async def test_the_team_default_applies(self, db):
        await _add_user(db, "new-user", own_set=_seed_set())
        await _publish(_set({"balanced": _tier(model="swift", api_key="gateway-key", base_url=GW)}, env=self.TEAM))
        cfg = await _spawn_config()
        assert (cfg.provider, cfg.model) == ("openai", "swift")
        env = server._build_env(cfg)
        assert "OPENAI_API_KEY=gateway-key" in env and "BRAVE_API_KEY=brave-1" in env

    async def test_a_set_somebody_chose_stays_theirs(self, db):
        await _publish(_set({"balanced": _tier(model="swift", api_key="gateway-key", base_url=GW)}, env=self.TEAM))
        for own in (_seed_set(api_key="sk-mine"),
                    _seed_set(base_url="http://localhost:1234/v1"),
                    _seed_set(provider="ollama", model="qwen3.5:4b"),          # local models: keyless by design
                    _seed_set(model="claude-another"),                        # own model on the deck's key
                    {**_seed_set(), "envVars": [{"key": "BRAVE_API_KEY", "value": "b"}]},
                    {**_seed_set(), "tiers": {"mine": _tier(provider="anthropic", model="claude")}}):
            assert not basna_mod._set_is_unconfigured(own)
        await _add_user(db, "new-user", own_set=_seed_set(provider="ollama", model="qwen3.5:4b"))
        tiers, env = await basna_mod._load_owner_tiers(db, "new-user")
        assert tiers["balanced"]["model"] == "qwen3.5:4b" and env == []

    async def test_a_seed_the_ui_backfilled_with_a_later_tier_is_still_a_seed(self, db):
        """A tier the registry gained later is copied from a sibling by the UI
        (micro ← balanced), not taken from the registry."""
        seed = _seed_set()
        seed["tiers"]["micro"] = dict(seed["tiers"]["balanced"])
        seed["tiers"]["coding"] = dict(seed["tiers"]["reason"])
        assert basna_mod._set_is_unconfigured(seed)
        seed["tiers"]["micro"] = {**seed["tiers"]["balanced"], "model": "some-other-model"}
        assert not basna_mod._set_is_unconfigured(seed)             # somebody picked that one

    async def test_without_a_team_default_the_seed_is_what_there_is(self, db):
        await _add_user(db, "new-user", own_set=_seed_set())
        tiers, _env = await basna_mod._load_owner_tiers(db, "new-user")
        assert tiers["balanced"]["provider"] == "anthropic"

    async def test_an_unreadable_registry_keeps_the_users_set(self, db, monkeypatch):
        def boom():
            raise HTTPException(500, "Archetype registry not found")

        monkeypatch.setattr(basna_mod, "_load_registry", boom)
        assert not basna_mod._set_is_unconfigured(
            {"tiers": {"balanced": _tier(provider="anthropic", model="claude-sonnet-4-6")}})


class TestDirectModelCallsUseTheTeamKey:
    """Code, Dubina and the beings call a tier's model themselves
    (create_provider) instead of spawning an agent: the "@system" sentinel is
    not a key there either."""

    async def test_the_sentinel_becomes_the_team_key_for_its_endpoint(self, db):
        await _publish(_set({"balanced": _tier(api_key="gateway-key", base_url=GW),
                             "fast": _tier(api_key="sk-real-openai")}))
        tiers, _env = await basna_mod._load_owner_tiers(db, "new-user")
        assert basna_mod.tier_api_key(tiers["balanced"]) == "gateway-key"
        assert basna_mod.tier_api_key(tiers["fast"]) == "sk-real-openai"

    async def test_nothing_behind_the_sentinel_is_no_key_not_the_sentinel(self, db):
        assert basna_mod.tier_api_key(_tier(api_key="@system")) is None

    async def test_a_tiers_own_key_or_none_is_passed_through(self, db, monkeypatch):
        monkeypatch.setattr(basna_mod, "_SYSTEM_PROVIDER_KEYS", {"openai": "sk-org"})
        assert basna_mod.tier_api_key(_tier(api_key="sk-mine")) == "sk-mine"
        assert basna_mod.tier_api_key(_tier(api_key="")) is None   # the environment's, as before

    async def test_creds_in_a_request_body_are_resolved_the_same_way(self, db):
        """Basna's router call gets the tier's key as the UI (or agent_start) holds it."""
        await _publish(_set({"balanced": _tier(api_key="gateway-key", base_url=GW)}))
        assert await basna_mod._request_api_key("openai", "@system", GW) == "gateway-key"
        assert await basna_mod._request_api_key("openai", "@system", "") is None
        assert await basna_mod._request_api_key("openai", "sk-mine", GW) == "sk-mine"
        assert await basna_mod._request_api_key("openai", "", GW) is None

    async def test_the_code_map_summary_gets_a_key_not_the_sentinel(self, db, monkeypatch, tmp_path):
        from captain_claw.flight_deck import code_routes

        seen: dict = {}

        async def summarize(_repo, _changed, creds):
            seen.update(creds)

        monkeypatch.setattr(code_routes.code_map, "reindex", lambda repo: {"changed_files": ["a.py"]})
        monkeypatch.setattr(code_routes.code_map, "summarize_changed", summarize)
        await _publish(_set({"fast": _tier(api_key="gateway-key", base_url=GW)}))
        tiers, _env = await basna_mod._load_owner_tiers(db, "new-user")
        await code_routes._update_map(tmp_path, tiers, {})
        assert seen["api_key"] == "gateway-key" and seen["base_url"] == GW

    async def test_dubina_builds_its_provider_with_the_team_key(self, db, monkeypatch):
        from captain_claw.flight_deck import dubina_routes

        seen: dict = {}
        monkeypatch.setattr(dubina_routes, "create_provider", lambda **kw: seen.update(kw) or "provider")
        await _publish(_set({"balanced": _tier(api_key="gateway-key", base_url=GW)}))
        tiers = await dubina_routes._resolve_tiers(db, "new-user")
        assert dubina_routes._library_provider_factory(tiers)("balanced") == "provider"
        assert seen["api_key"] == "gateway-key" and seen["base_url"] == GW


# ── sets that are already published ─────────────────────────────────────────

class TestPublishedSetsGetTheirKeysWithoutRepublishing:
    """The reported case, three times over: the set was published by an earlier
    Flight Deck, the fix was deployed, nobody clicked Publish again — and a new
    agent still had BRAVE_API_KEY and no model key."""

    ENV = [{"key": "BRAVE_API_KEY", "value": "brave-1"}]

    async def test_a_sentinel_with_nothing_behind_it(self, db):
        """Published when the copy was masked but no key was stored anywhere."""
        await _add_user(db, "admin-1", "admin", _set(
            {"balanced": _tier(model="swift", api_key="gateway-key", base_url=GW)}, env=self.ENV))
        await _already_published(db, _set(
            {"balanced": _tier(model="swift", api_key="@system", base_url=GW)}, env=self.ENV))
        env = server._build_env(await _spawn_config())
        assert "OPENAI_API_KEY=gateway-key" in env and "BRAVE_API_KEY=brave-1" in env
        assert await _endpoint_keys(db) == {f"openai|{GW}": "gateway-key"}

    async def test_a_custom_endpoint_tier_that_was_published_blank(self, db):
        await _add_user(db, "admin-1", "admin", _set(
            {"balanced": _tier(api_key="gateway-key", base_url=GW + "/")}))
        await _already_published(db, _set({"balanced": _tier(api_key="", base_url=GW)}))
        assert "OPENAI_API_KEY=gateway-key" in server._build_env(await _spawn_config())
        stored = await db.get_system_setting("fd:shared-tier-sets")
        assert _published_key(stored, "balanced") == "@system" and "gateway-key" not in stored

    async def test_a_tier_on_the_providers_own_endpoint(self, db):
        await _add_user(db, "admin-1", "admin", _set(
            {"balanced": _tier(provider="anthropic", model="claude", api_key="sk-ant")}))
        await _already_published(db, _set(
            {"balanced": _tier(provider="anthropic", model="claude", api_key="@system")}))
        assert "ANTHROPIC_API_KEY=sk-ant" in server._build_env(await _spawn_config())
        assert await _org_keys(db) == {"anthropic": "sk-ant"}

    async def test_it_happens_at_startup_too_and_only_once(self, db):
        await _add_user(db, "admin-1", "admin", _set({
            "balanced": _tier(api_key="gateway-key", base_url=GW),
            "fast": _tier(provider="anthropic", model="claude", api_key="sk-ant")}))
        await _already_published(db, _set({
            "balanced": _tier(api_key="", base_url=GW),
            "fast": _tier(provider="anthropic", model="claude", api_key="@system")}))
        res = await admin_routes.recover_team_keys(db)
        assert res == {"providers": ["anthropic"], "endpoints": ["gw.example"]}
        assert "gateway-key" not in json.dumps(res) and "sk-ant" not in json.dumps(res)
        assert await admin_routes.recover_team_keys(db) == {}

    async def test_a_team_key_that_exists_is_never_replaced(self, db):
        await db.set_system_setting("fd:provider-keys", json.dumps({"anthropic": "sk-team"}))
        await db.set_system_setting("fd:endpoint-keys", json.dumps({f"openai|{GW}": "team-gw"}))
        await _add_user(db, "admin-1", "admin", _set({
            "balanced": _tier(api_key="other-gw", base_url=GW),
            "fast": _tier(provider="anthropic", model="claude", api_key="sk-other")}))
        await _already_published(db, _set({
            "balanced": _tier(api_key="", base_url=GW),
            "fast": _tier(provider="anthropic", model="claude", api_key="@system")}))
        assert await admin_routes.recover_team_keys(db) == {}
        assert await _org_keys(db) == {"anthropic": "sk-team"}
        assert await _endpoint_keys(db) == {f"openai|{GW}": "team-gw"}
        assert "OPENAI_API_KEY=team-gw" in server._build_env(await _spawn_config())

    async def test_only_an_admins_copy_of_that_very_set_counts(self, db):
        published = _set({"balanced": _tier(api_key="@system", base_url=GW)})
        await _already_published(db, published)
        await _add_user(db, "teammate", "user", _set({"balanced": _tier(api_key="not-yours", base_url=GW)}))
        await _add_user(db, "admin-2", "admin", _set(
            {"balanced": _tier(api_key="another-set", base_url=GW)}, sid="other-set"))
        assert await admin_routes.recover_team_keys(db) == {"unresolved": ["openai at gw.example"]}
        assert await _endpoint_keys(db) == {} and await _org_keys(db) == {}
        with pytest.raises(HTTPException) as exc:   # …and the spawn says what is wrong
            await _spawn_config()
        assert exc.value.status_code == 409 and "gw.example" in exc.value.detail

    async def test_a_tier_the_admin_has_since_moved_elsewhere_gives_no_key(self, db):
        await _already_published(db, _set({"balanced": _tier(api_key="@system", base_url=GW)}))
        await _add_user(db, "admin-1", "admin", _set({
            "balanced": _tier(api_key="sk-real-openai"),                          # own endpoint now
            "fast": _tier(provider="anthropic", api_key="sk-ant", base_url=GW)}))  # other provider
        assert await admin_routes.recover_team_keys(db) == {"unresolved": ["openai at gw.example"]}
        assert await _endpoint_keys(db) == {} and await _org_keys(db) == {}

    async def test_another_tier_of_the_set_on_the_same_endpoint_will_do(self, db):
        await _already_published(db, _set({"balanced": _tier(api_key="@system", base_url=GW)}))
        await _add_user(db, "admin-1", "admin", _set({
            "balanced": _tier(api_key="@system", base_url=GW),
            "fast": _tier(api_key="gateway-key", base_url=GW)}))
        assert (await admin_routes.recover_team_keys(db))["endpoints"] == ["gw.example"]
        assert await _endpoint_keys(db) == {f"openai|{GW}": "gateway-key"}

    async def test_a_blank_tier_on_the_providers_own_endpoint(self, db):
        """Published with no key in the tier; the admin's copy has one now."""
        await _add_user(db, "admin-1", "admin", _set({"balanced": _tier(api_key="sk-real-openai")}))
        await _already_published(db, _set({"balanced": _tier(api_key="")}))
        assert "OPENAI_API_KEY=sk-real-openai" in server._build_env(await _spawn_config())
        assert _published_key(await db.get_system_setting("fd:shared-tier-sets"), "balanced") == "@system"

    async def test_the_oldest_admins_copy_wins(self, db):
        await db._db.executemany(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, 'admin', ?, ?)",
            [("late", "late@x.co", "late", "2026-05-01", "2026-05-01"),
             ("first", "first@x.co", "first", "2026-01-01", "2026-01-01")])
        await db._db.commit()
        for uid in ("late", "first"):
            await db.set_settings(uid, {"fd:forge-tiers": json.dumps({"sets": [_set(
                {"balanced": _tier(api_key=f"key-of-{uid}", base_url=GW)})], "activeSetId": "s1"})})
        await _already_published(db, _set({"balanced": _tier(api_key="@system", base_url=GW)}))
        await admin_routes.recover_team_keys(db)
        assert await _endpoint_keys(db) == {f"openai|{GW}": "key-of-first"}

    async def test_a_key_flight_decks_environment_supplies_is_left_to_it(self, db, monkeypatch):
        """The deck runs on a key in its environment; process agents inherit
        it. An admin's personal key must not quietly replace it."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "company-env-key")
        await _add_user(db, "admin-1", "admin", _set(
            {"balanced": _tier(provider="anthropic", model="claude", api_key="sk-admin-personal")}))
        await _already_published(db, _set(
            {"balanced": _tier(provider="anthropic", model="claude", api_key="@system")}))
        assert await admin_routes.recover_team_keys(db) == {}
        env = server._build_env(await _spawn_config())
        assert "ANTHROPIC_API_KEY" not in env and await _org_keys(db) == {}

    async def test_a_team_key_an_admin_removes_stays_removed(self, db):
        """Recovery is a one-time migration of sets published EARLIER — not a
        standing rule that undoes Admin → Provider keys."""
        own = _set({"balanced": _tier(provider="anthropic", model="claude", api_key="sk-ant")})
        await _add_user(db, "admin-1", "admin", own)
        await _publish(own)
        await admin_routes.update_provider_keys(
            admin_routes.ProviderKeysRequest(keys={"anthropic": ""}), admin=ADMIN)
        assert await admin_routes.recover_team_keys(db) == {} and await _org_keys(db) == {}
        with pytest.raises(HTTPException) as exc:
            await _spawn_config()
        assert exc.value.status_code == 409 and await _org_keys(db) == {}
        await _publish(own)                                   # sharing it again is a publish
        assert "ANTHROPIC_API_KEY=sk-ant" in server._build_env(await _spawn_config())

    async def test_a_recovered_set_is_done_too(self, db):
        await _add_user(db, "admin-1", "admin", _set(
            {"balanced": _tier(provider="anthropic", model="claude", api_key="sk-ant")}))
        await _already_published(db, _set(
            {"balanced": _tier(provider="anthropic", model="claude", api_key="@system")}))
        assert (await admin_routes.recover_team_keys(db))["providers"] == ["anthropic"]
        await admin_routes.update_provider_keys(
            admin_routes.ProviderKeysRequest(keys={"anthropic": ""}), admin=ADMIN)
        assert await admin_routes.recover_team_keys(db) == {} and await _org_keys(db) == {}

    async def test_a_set_still_missing_a_key_is_tried_again(self, db):
        await _already_published(db, _set({"balanced": _tier(api_key="@system", base_url=GW)}))
        assert await admin_routes.recover_team_keys(db) == {"unresolved": ["openai at gw.example"]}
        await _add_user(db, "admin-1", "admin", _set({"balanced": _tier(api_key="gateway-key", base_url=GW)}))
        assert (await admin_routes.recover_team_keys(db))["endpoints"] == ["gw.example"]

    async def test_a_gateway_tier_is_not_left_on_the_providers_key(self, db):
        """The deck has an OpenAI key in Admin → Provider keys AND an old
        "@system" tier on a gateway whose key the admin's copy still holds:
        the gateway gets ITS key, as a publish would store it."""
        await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "sk-openai-direct"}))
        await _add_user(db, "admin-1", "admin", _set({"balanced": _tier(api_key="gateway-key", base_url=GW)}))
        await _already_published(db, _set({"balanced": _tier(api_key="@system", base_url=GW)}))
        assert (await admin_routes.recover_team_keys(db))["endpoints"] == ["gw.example"]   # at startup
        assert "OPENAI_API_KEY=gateway-key" in server._build_env(await _spawn_config())
        assert await _endpoint_keys(db) == {f"openai|{GW}": "gateway-key"}
        assert await _org_keys(db) == {"openai": "sk-openai-direct"}

    async def test_a_blank_tier_nobody_has_a_key_for_does_not_keep_the_set_open(self, db):
        """Published without a key and the admin's copy has none either: there
        is nothing to wait for — the set is done, and stays done."""
        own = _set({"balanced": _tier(api_key="", base_url="http://localhost:1234/v1"),
                    "fast": _tier(provider="anthropic", model="claude", api_key="sk-ant")})
        await _add_user(db, "admin-1", "admin", own)
        await _already_published(db, _set({
            "balanced": _tier(api_key="", base_url="http://localhost:1234/v1"),
            "fast": _tier(provider="anthropic", model="claude", api_key="@system")}))
        assert (await admin_routes.recover_team_keys(db)) == {"providers": ["anthropic"], "endpoints": []}
        await admin_routes.update_provider_keys(
            admin_routes.ProviderKeysRequest(keys={"anthropic": ""}), admin=ADMIN)
        assert await admin_routes.recover_team_keys(db) == {} and await _org_keys(db) == {}

    async def test_the_other_published_sets_and_the_default_are_kept(self, db):
        done = _set({"balanced": _tier(provider="anthropic", model="claude", api_key="@system")}, sid="a")
        legacy = _set({"balanced": _tier(api_key="@system", base_url=GW)}, sid="b")
        await db.set_system_setting("fd:shared-tier-sets", json.dumps(
            {"sets": [done, legacy], "defaultSetId": "b", "keys_shared": ["a"]}))
        await _add_user(db, "admin-1", "admin", _set({"balanced": _tier(api_key="gateway-key", base_url=GW)}, sid="b"))
        await admin_routes.recover_team_keys(db)
        blob = json.loads(await db.get_system_setting("fd:shared-tier-sets"))
        assert blob["defaultSetId"] == "b" and blob["keys_shared"] == ["a", "b"]
        assert [s["id"] for s in blob["sets"]] == ["a", "b"]
        assert await _org_keys(db) == {}                     # set "a" was done: not looked at again
        assert "OPENAI_API_KEY=gateway-key" in server._build_env(await _spawn_config())

    async def test_a_set_with_nothing_to_recover_is_marked_done_anyway(self, db):
        await db.set_system_setting("fd:provider-keys", json.dumps({"anthropic": "sk-team"}))
        own = _set({"balanced": _tier(provider="anthropic", model="claude", api_key="sk-mine")})
        await _add_user(db, "admin-1", "admin", own)
        await _already_published(db, _set({"balanced": _tier(provider="anthropic", model="claude", api_key="@system")}))
        assert await admin_routes.recover_team_keys(db) == {}
        assert json.loads(await db.get_system_setting("fd:shared-tier-sets"))["keys_shared"] == ["s1"]
        await admin_routes.update_provider_keys(
            admin_routes.ProviderKeysRequest(keys={"anthropic": ""}), admin=ADMIN)
        assert await admin_routes.recover_team_keys(db) == {} and await _org_keys(db) == {}

    async def test_a_key_the_set_carries_in_its_extra_keys_needs_no_recovery(self, db):
        await _add_user(db, "admin-1", "admin", _set({"balanced": _tier(api_key="sk-real-openai")}))
        await _already_published(db, _set({"balanced": _tier(api_key="@system")},
                                          env=[{"key": "OPENAI_API_KEY", "value": "sk-in-set"}]))
        assert await admin_routes.recover_team_keys(db) == {} and await _org_keys(db) == {}
        assert "OPENAI_API_KEY=sk-in-set" in server._build_env(await _spawn_config())

    async def test_a_key_the_admin_keeps_in_the_sets_extra_keys_is_found(self, db):
        await _add_user(db, "admin-1", "admin", _set(
            {"balanced": _tier(api_key="")}, env=[{"key": "OPENAI_API_KEY", "value": "sk-in-extra"}]))
        await _already_published(db, _set({"balanced": _tier(api_key="@system")}))
        assert "OPENAI_API_KEY=sk-in-extra" in server._build_env(await _spawn_config())

    async def test_a_recovered_provider_key_is_not_sent_to_a_gateway_without_its_own(self, db):
        """Old publish: a tier on api.openai.com and one on a gateway, both
        "@system". The admin's copy still has the OpenAI key but no longer a
        key for THAT gateway (URL edited since). The OpenAI key is recovered —
        and must not become the gateway tier's key by fallback."""
        old_gw = "https://old-gw.example/v1"
        await _already_published(db, _set({"fast": _tier(api_key="@system"),
                                          "balanced": _tier(api_key="@system", base_url=old_gw)}))
        await _add_user(db, "admin-1", "admin", _set({
            "fast": _tier(api_key="sk-openai-direct"),
            "balanced": _tier(api_key="gw-key", base_url="https://new-gw.example/v1")}))
        res = await admin_routes.recover_team_keys(db)
        assert res == {"providers": ["openai"], "endpoints": [], "unresolved": ["openai at old-gw.example"]}
        with pytest.raises(HTTPException) as exc:                 # no key — and it says so
            await _spawn_config()
        assert exc.value.status_code == 409 and "old-gw.example" in exc.value.detail
        assert await server._team_key("openai", old_gw, asked=True) == ""
        assert await server._team_key("openai", old_gw, asked=False) == ""
        await basna_mod._refresh_system_provider_keys(db)          # Basna / Vatra / direct calls too
        assert basna_mod._effective_key("openai", "@system", old_gw) is None
        assert basna_mod._effective_key("openai", "@system") == "sk-openai-direct"
        _agent("on-old-gw", base_url=old_gw)                       # …nor healed into an agent
        assert await server.heal_keyless_agents(restart=False) == []
        # Publishing a key for that gateway settles it; publishing without one keeps the mark.
        await _publish(_set({"fast": _tier(api_key="@system")}))
        assert await server._team_key("openai", old_gw, asked=True) == ""
        await _publish(_set({"balanced": _tier(api_key="old-gw-key", base_url=old_gw)}))
        assert await server._team_key("openai", old_gw, asked=True) == "old-gw-key"
        assert json.loads(await db.get_system_setting("fd:shared-tier-sets"))["no_provider_fallback"] == []

    async def test_a_provider_key_in_the_sets_extra_keys_does_not_hide_a_gateways_own(self, db):
        extra = [{"key": "OPENAI_API_KEY", "value": "sk-openai-direct"}]
        await _add_user(db, "admin-1", "admin", _set(
            {"fast": _tier(api_key=""), "balanced": _tier(api_key="gw-key", base_url=GW)}, env=extra))
        await _already_published(db, _set(
            {"fast": _tier(api_key=""), "balanced": _tier(api_key="@system", base_url=GW)}, env=extra))
        assert (await admin_routes.recover_team_keys(db))["endpoints"] == ["gw.example"]
        env = server._build_env(await _spawn_config())
        assert "OPENAI_API_KEY=gw-key" in env and "sk-openai-direct" not in env

    async def test_in_a_set_that_stays_open_a_removed_key_stays_removed_too(self, db):
        await _already_published(db, _set({
            "fast": _tier(provider="anthropic", model="claude", api_key="@system"),
            "balanced": _tier(api_key="@system", base_url=GW)}))          # no key for it anywhere
        await _add_user(db, "admin-1", "admin", _set(
            {"fast": _tier(provider="anthropic", model="claude", api_key="sk-ant")}))
        res = await admin_routes.recover_team_keys(db)
        assert res["providers"] == ["anthropic"] and res["unresolved"] == ["openai at gw.example"]
        await admin_routes.update_provider_keys(
            admin_routes.ProviderKeysRequest(keys={"anthropic": ""}), admin=ADMIN)
        assert await admin_routes.recover_team_keys(db) == {"unresolved": ["openai at gw.example"]}
        assert await _org_keys(db) == {}

    async def test_a_published_blank_gateway_tier_is_no_team_endpoint(self, db):
        """Until recovery finds its key, a blank tier is no licence to send the
        provider's org key to that host."""
        await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "sk-org"}))
        await _already_published(db, _set({"balanced": _tier(api_key="", base_url=GW)}))
        assert await server._team_key("openai", GW, asked=False) == ""

    async def test_a_gateway_key_that_is_the_org_key_is_not_stored_twice(self, db):
        await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "gw-key"}))
        await _add_user(db, "admin-1", "admin", _set({"balanced": _tier(api_key="gw-key", base_url=GW)}))
        await _already_published(db, _set({"balanced": _tier(api_key="", base_url=GW)}))
        assert "OPENAI_API_KEY=gw-key" in server._build_env(await _spawn_config())
        assert await _endpoint_keys(db) == {}


# ── agents created without a model key ──────────────────────────────────────

def _agent(slug: str, provider="openai", model="swift", base_url=GW, env="BRAVE_API_KEY=brave-1\n",
           api_key="") -> None:
    import yaml

    agent_dir = server.DATA_DIR / slug
    agent_dir.mkdir(parents=True)
    (agent_dir / "config.yaml").write_text(yaml.safe_dump(
        {"model": {"provider": provider, "model": model, "api_key": api_key, "base_url": base_url}}))
    (agent_dir / ".env").write_text(env)
    registry = server._load_process_registry()
    registry[slug] = {"slug": slug, "name": slug, "provider": provider, "model": model,
                      "owner": "new-user", "pid": None, "web_port": 0}
    server._save_process_registry(registry)


def _env(slug: str) -> str:
    return (server.DATA_DIR / slug / ".env").read_text()


class TestAgentsCreatedWithoutAKeyGetIt:
    """An agent's .env is written once, at creation. One created while the team
    had no key for its model stays dead unless somebody recreates it — so give
    it the key when there is one: on publish, on a provider-key change, at
    startup, and when it is started."""

    async def test_publishing_gives_existing_agents_their_key(self, db):
        _agent("marco")
        res = await _publish(_set({"balanced": _tier(api_key="gateway-key", base_url=GW)}))
        assert res["agents_given_key"] == 1
        assert _env("marco") == "BRAVE_API_KEY=brave-1\nOPENAI_API_KEY=gateway-key\n"
        assert (await _publish(_set({"balanced": _tier(api_key="gateway-key", base_url=GW)})))[
            "agents_given_key"] == 0

    async def test_a_provider_key_change_does_too(self, db):
        _agent("claude-agent", provider="anthropic", model="claude", base_url="", env="")
        res = await admin_routes.update_provider_keys(
            admin_routes.ProviderKeysRequest(keys={"anthropic": "sk-ant"}), admin=ADMIN)
        assert res["agents_given_key"] == 1 and _env("claude-agent") == "ANTHROPIC_API_KEY=sk-ant\n"

    async def test_startup_heals_after_recovering_the_keys(self, db):
        """The whole unattended path: old publish, new Flight Deck, restart."""
        _agent("marco")
        await _add_user(db, "admin-1", "admin", _set({"balanced": _tier(api_key="gateway-key", base_url=GW)}))
        await _already_published(db, _set({"balanced": _tier(api_key="@system", base_url=GW)}))
        await admin_routes.recover_team_keys(db)
        assert await server.heal_keyless_agents(restart=False) == ["marco"]
        assert "OPENAI_API_KEY=gateway-key" in _env("marco")

    async def test_agents_that_have_or_need_no_key_are_left_alone(self, db, monkeypatch):
        await db.set_system_setting("fd:provider-keys", json.dumps(
            {"openai": "sk-org", "anthropic": "sk-ant"}))
        await db.set_system_setting("fd:endpoint-keys", json.dumps({f"openai|{GW}": "gateway-key"}))
        _agent("own-key", env="OPENAI_API_KEY=mine\n")
        _agent("key-in-config", api_key="in-config")
        _agent("chatgpt", model="gpt-5.2", base_url="", env="")
        _agent("local", provider="ollama", model="qwen", base_url="http://localhost:11434", env="")
        _agent("elsewhere", base_url="https://other.example/v1", env="")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-fd-env")   # process agents inherit it
        _agent("inherits", provider="anthropic", model="claude", base_url="", env="")
        before = {slug: _env(slug) for slug in server._load_process_registry()}
        assert await server.heal_keyless_agents(restart=False) == []
        assert {slug: _env(slug) for slug in before} == before

    async def test_an_agent_that_is_missing_on_disk_is_skipped(self, db):
        server._save_process_registry({"ghost": {"slug": "ghost", "pid": None}})
        assert await server.heal_keyless_agents(restart=False) == []

    async def test_the_key_is_looked_for_where_the_agent_looks(self, db, monkeypatch):
        import yaml

        await db.set_system_setting("fd:provider-keys", json.dumps({"openai": "sk-org"}))
        await db.set_system_setting("fd:endpoint-keys", json.dumps({f"openai|{GW}": "gateway-key"}))
        _agent("exported", env='export OPENAI_API_KEY="mine"\n')              # a key, however written
        _agent("in-provider-keys")
        (server.DATA_DIR / "in-provider-keys" / "config.yaml").write_text(yaml.safe_dump(
            {"model": {"provider": "openai", "model": "swift", "base_url": GW},
             "provider_keys": {"openai": "from-settings"}}))
        _agent("in-home-config")                                               # the copy the agent loads
        home = server.DATA_DIR / "in-home-config" / "data" / "home-config-parent" / ".captain-claw"
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(yaml.safe_dump({"model": {"api_key": "in-home"}}))
        _agent("unreadable")
        (server.DATA_DIR / "unreadable" / "config.yaml").write_bytes(b"\xff\xfe not yaml: [")
        _agent("blank-value", env='OPENAI_API_KEY=""\n')                      # no key at all
        monkeypatch.setenv("OPENAI_API_KEY", "fd-env-key")                     # …not the gateway's key:
        _agent("gateway")                                                      # it still gets its own
        assert sorted(await server.heal_keyless_agents(restart=False)) == ["blank-value", "gateway"]
        assert "OPENAI_API_KEY=gateway-key" in _env("blank-value")

    async def test_only_the_running_agents_are_restarted(self, db, monkeypatch):
        _agent("running")
        _agent("stopped")
        restarted: list = []
        monkeypatch.setattr(server, "_process_is_alive", lambda slug: slug == "running")
        monkeypatch.setattr(server, "_restart_processes", lambda slugs: restarted.append(list(slugs)))
        res = await _publish(_set({"balanced": _tier(api_key="gateway-key", base_url=GW)}))
        await asyncio.sleep(0.05)                       # the restart runs in a worker thread
        assert res["agents_given_key"] == 2 and restarted == [["running"]]

    async def test_the_serving_modules_healer_is_the_one_called(self, db, monkeypatch):
        """`python -m …server` runs the server as __main__; its lifespan hands
        admin_routes ITS healer, so restarts use the process table the routes use."""
        calls: list = []

        async def serving_healer():
            calls.append("serving")
            return ["a", "b"]

        monkeypatch.setattr(admin_routes, "_agent_key_healer", serving_healer)
        res = await _publish(_set({"balanced": _tier(api_key="gateway-key", base_url=GW)}))
        assert calls == ["serving"] and res["agents_given_key"] == 2

    async def test_startup_recovers_heals_then_reattaches_then_restarts(self, db, monkeypatch):
        _agent("dead")                                   # reattach will start it — with the key
        _agent("live")                                   # running without it: restarted after
        await _add_user(db, "admin-1", "admin", _set({"balanced": _tier(api_key="gateway-key", base_url=GW)}))
        await _already_published(db, _set({"balanced": _tier(api_key="@system", base_url=GW)}))
        order: list = []
        monkeypatch.setattr(server, "_process_is_alive", lambda slug: slug == "live")
        monkeypatch.setattr(server, "_reattach_processes",
                            lambda: order.append(("reattach", "OPENAI_API_KEY=gateway-key" in _env("dead"))))
        monkeypatch.setattr(server, "_do_stop_process", lambda slug: order.append(("stop", slug)))
        monkeypatch.setattr(server, "_do_start_process", lambda slug: order.append(("start", slug)))
        monkeypatch.setattr("time.sleep", lambda _s: None)
        await server._startup_team_keys_then_reattach()
        assert order == [("reattach", True), ("stop", "live"), ("start", "live")]
        assert admin_routes._agent_key_healer is server.heal_keyless_agents

    async def test_startup_says_which_published_tier_has_no_key(self, db, monkeypatch, caplog):
        await _already_published(db, _set({"balanced": _tier(api_key="@system", base_url=GW)}))
        monkeypatch.setattr(server, "_reattach_processes", lambda: None)
        with caplog.at_level("WARNING"):
            await server._startup_team_keys_then_reattach()
        assert "No team API key for openai at gw.example" in caplog.text

    async def test_startup_survives_a_failing_recovery(self, db, monkeypatch):
        async def boom(_db):
            raise RuntimeError("db not ready")

        calls: list = []
        monkeypatch.setattr(admin_routes, "recover_team_keys", boom)
        monkeypatch.setattr(server, "_reattach_processes", lambda: calls.append("reattach"))
        await server._startup_team_keys_then_reattach()
        assert calls == ["reattach"]

    async def test_starting_or_restarting_an_agent_gives_it_the_key_first(self, db, monkeypatch):
        await db.set_system_setting("fd:endpoint-keys", json.dumps({f"openai|{GW}": "gateway-key"}))
        seen: list = []
        monkeypatch.setattr(server, "_do_stop_process", lambda slug: None)
        monkeypatch.setattr(server, "_do_start_process",
                            lambda slug: seen.append("OPENAI_API_KEY=gateway-key" in _env(slug)))
        monkeypatch.setattr("time.sleep", lambda _s: None)
        _agent("a")
        await server.start_process("a", _req("new-user"), None)
        _agent("b")
        await server.restart_process("b", _req("new-user"), None)
        assert seen == [True, True]


class TestStoppingAnAgent:
    def _registered(self, pid):
        server._save_process_registry({"marco": {"slug": "marco", "pid": pid, "web_port": 1}})

    async def test_a_dead_handle_does_not_hide_the_running_process(self, db, monkeypatch):
        """Restarted through another copy of the module, the agent's live pid
        is in the registry while this one still holds the old handle."""
        self._registered(4242)
        killed: list = []
        server._processes["marco"] = types.SimpleNamespace(pid=1111, poll=lambda: 0)   # exited
        monkeypatch.setattr(server, "_process_is_alive", lambda slug: True)
        monkeypatch.setattr(server, "_kill_pid", lambda pid: killed.append(pid))
        assert server._do_stop_process("marco").message == "Stopped"
        assert killed == [4242]

    async def test_a_live_handle_is_the_one_stopped(self, db, monkeypatch):
        self._registered(4242)
        killed: list = []
        server._processes["marco"] = types.SimpleNamespace(pid=1111, poll=lambda: None)
        monkeypatch.setattr(server, "_kill_pid", lambda pid: killed.append(pid))
        server._do_stop_process("marco")
        assert killed == [1111]

    async def test_what_is_written_during_the_kill_survives_it(self, db, monkeypatch):
        self._registered(4242)

        def slow_kill(_pid):   # a spawn lands in the registry while we wait for the process to die
            registry = server._load_process_registry()
            registry["newcomer"] = {"slug": "newcomer", "pid": 7, "web_port": 2}
            server._save_process_registry(registry)

        monkeypatch.setattr(server, "_process_is_alive", lambda slug: True)
        monkeypatch.setattr(server, "_kill_pid", slow_kill)
        server._do_stop_process("marco")
        registry = server._load_process_registry()
        assert "newcomer" in registry and registry["marco"]["stopped"] is True
