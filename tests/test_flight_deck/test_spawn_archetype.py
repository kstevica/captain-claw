"""Tests for `_resolve_archetype` — the archetype selector on the spawn endpoints.

`AgentConfig.archetype` ("id" or "id@tier") is folded into a concrete spawn config
before `_resolve_tier` runs: the archetype supplies cognitive_mode / tools / role /
tier→model, explicit caller fields win, and an unknown id is a non-fatal no-op.
These exercise the resolver directly with stubbed registry + owner-tier seams so no
DB or real archetype file is touched.
"""

from __future__ import annotations

import types

import pytest

import captain_claw.flight_deck.server as server


def _req(uid: str = ""):
    """A stub Request exposing only `.state.user_id`, as the resolver reads."""
    return types.SimpleNamespace(state=types.SimpleNamespace(user_id=uid))


@pytest.fixture
def patch_registry(monkeypatch: pytest.MonkeyPatch):
    """Install stub `merged_archetypes` / `get_db` / `_load_owner_tiers`.

    Call the returned setter with (archetypes_list, tiers_map) to define what the
    resolver sees. `merged_archetypes` is patched on its source module so the
    resolver's in-function import binds the stub.
    """
    import captain_claw.flight_deck.archetypes as arch_mod
    import captain_claw.flight_deck.auth as auth_mod
    import captain_claw.flight_deck.basna_routes as basna_mod

    state = types.SimpleNamespace(archetypes=[], tiers={}, env=[], last_uid=None)

    async def fake_merged(db, uid):
        state.last_uid = uid
        return list(state.archetypes)

    async def fake_owner_tiers(db, uid):
        return dict(state.tiers), list(state.env)

    monkeypatch.setattr(arch_mod, "merged_archetypes", fake_merged)
    monkeypatch.setattr(auth_mod, "get_db", lambda: object())
    monkeypatch.setattr(basna_mod, "_load_owner_tiers", fake_owner_tiers)

    def _set(archetypes=None, tiers=None, env=None):
        state.archetypes = archetypes or []
        state.tiers = tiers or {}
        state.env = env or []

    state.set = _set
    return state


async def test_no_archetype_is_noop(patch_registry):
    cfg = server.AgentConfig(name="x", provider="anthropic", model="claude-opus-4-8")
    before = cfg.model_copy(deep=True)
    await server._resolve_archetype(cfg, _req(), None)
    assert cfg == before


async def test_unknown_id_is_nonfatal_noop(patch_registry):
    patch_registry.set(archetypes=[{"id": "fact-checker", "role": "Checker"}])
    cfg = server.AgentConfig(name="x", archetype="does-not-exist",
                             provider="anthropic", model="claude-opus-4-8")
    await server._resolve_archetype(cfg, _req(), None)
    # caller config preserved; the bad selector didn't raise
    assert cfg.provider == "anthropic" and cfg.model == "claude-opus-4-8"
    assert cfg.description == ""


async def test_fills_cognitive_tools_and_role_from_archetype(patch_registry):
    patch_registry.set(archetypes=[{
        "id": "fact-checker", "role": "Rigorous Fact Checker",
        "cognitive_mode": "kritika", "tools": ["read", "web_search"], "tier": "reason",
    }])
    # No owner tier config → model stays the caller-inherited one; tier recorded.
    cfg = server.AgentConfig(name="x", archetype="fact-checker",
                             provider="anthropic", model="claude-opus-4-8")
    await server._resolve_archetype(cfg, _req(), None)
    assert cfg.cognitive_mode == "kritika"
    assert cfg.tools == ["read", "web_search"]
    assert cfg.description == "Rigorous Fact Checker"
    assert cfg.provider == "anthropic" and cfg.model == "claude-opus-4-8"
    assert cfg.tier == "reason"  # last-resort: let _resolve_tier try the registry


async def test_tier_resolves_model_against_owner_library(patch_registry):
    patch_registry.set(
        archetypes=[{"id": "fact-checker", "role": "Checker",
                     "cognitive_mode": "kritika", "tools": ["read"], "tier": "fast"}],
        tiers={"reason": {"provider": "openai", "model": "gpt-5",
                          "api_key": "sk-owner", "base_url": "https://x"}},
    )
    # `@reason` overrides the archetype's own "fast" tier and resolves to the
    # owner's Library config for that tier.
    cfg = server.AgentConfig(name="x", archetype="fact-checker@reason",
                             provider="anthropic", model="claude-opus-4-8")
    await server._resolve_archetype(cfg, _req("u1"), {"id": "u1"})
    assert cfg.provider == "openai" and cfg.model == "gpt-5"
    assert cfg.provider_api_key == "sk-owner"
    assert cfg.base_url == "https://x"
    assert cfg.tier == ""  # pinned — _resolve_tier must not re-map


async def test_tier_switch_provider_clears_inherited_base_url(patch_registry):
    # Caller runs on a custom OpenAI-compatible endpoint; the archetype tier moves
    # to a different provider WITHOUT naming its own base_url. The stale endpoint
    # must be dropped so the new provider isn't routed at the old URL.
    patch_registry.set(
        archetypes=[{"id": "fact-checker", "role": "Checker", "tier": "reason"}],
        tiers={"reason": {"provider": "anthropic", "model": "claude-opus-4-8"}},  # no base_url
    )
    cfg = server.AgentConfig(name="x", archetype="fact-checker",
                             provider="openai", model="local", base_url="http://localhost:1234/v1")
    await server._resolve_archetype(cfg, _req("u1"), {"id": "u1"})
    assert cfg.provider == "anthropic" and cfg.model == "claude-opus-4-8"
    assert cfg.base_url == ""  # inherited custom endpoint dropped on provider switch


async def test_tier_keeps_base_url_when_provider_unchanged(patch_registry):
    # Same provider, tier names no base_url → keep the caller's inherited endpoint.
    patch_registry.set(
        archetypes=[{"id": "fact-checker", "role": "Checker", "tier": "reason"}],
        tiers={"reason": {"provider": "openai", "model": "other-local"}},  # no base_url
    )
    cfg = server.AgentConfig(name="x", archetype="fact-checker",
                             provider="openai", model="local", base_url="http://localhost:1234/v1")
    await server._resolve_archetype(cfg, _req("u1"), {"id": "u1"})
    assert cfg.provider == "openai" and cfg.model == "other-local"
    assert cfg.base_url == "http://localhost:1234/v1"  # preserved


async def test_explicit_caller_fields_win_over_archetype(patch_registry):
    patch_registry.set(archetypes=[{
        "id": "fact-checker", "role": "Checker",
        "cognitive_mode": "kritika", "tools": ["read", "web_search"], "tier": "reason",
    }])
    # Caller pinned cognitive_mode, tools, and description explicitly.
    cfg = server.AgentConfig(
        name="x", archetype="fact-checker",
        cognitive_mode="neutra_plus" if False else "vizija",  # explicit, non-default
        tools=["shell"], description="Custom desc",
        provider="anthropic", model="claude-opus-4-8",
    )
    await server._resolve_archetype(cfg, _req(), None)
    assert cfg.cognitive_mode == "vizija"      # not overwritten
    assert cfg.tools == ["shell"]              # not overwritten
    assert cfg.description == "Custom desc"    # not overwritten


async def test_owner_hint_used_when_no_authenticated_user(patch_registry):
    # No authenticated user and no request uid → the resolver must fall back to
    # config.owner_hint when looking up the owner's archetypes.
    patch_registry.set(archetypes=[
        {"id": "fact-checker", "role": "Checker", "cognitive_mode": "kritika"}])
    cfg = server.AgentConfig(name="x", archetype="fact-checker", owner_hint="owner-42")
    await server._resolve_archetype(cfg, _req(), None)
    assert patch_registry.last_uid == "owner-42"
    assert cfg.cognitive_mode == "kritika"  # resolution actually happened


# ── keys from the owner's tier set (kiosk "New agent" spawns only send the id) ──

_ANALYST = {"id": "analyst", "role": "Data Analyst", "tier": "balanced"}


async def test_tier_set_env_vars_reach_the_agent(patch_registry):
    """The set's own keys (tool credentials) come along, as the Library spawn
    sends them — the kiosk picker sends only the archetype id."""
    patch_registry.set(
        archetypes=[_ANALYST],
        tiers={"balanced": {"provider": "anthropic", "model": "claude-sonnet-5", "api_key": "sk-tier"}},
        env=[{"key": "BRAVE_API_KEY", "value": "brave-1"}, {"key": "TAVILY_API_KEY", "value": "tav-1"},
             {"key": "EMPTY", "value": ""}],
    )
    cfg = server.AgentConfig(name="x", archetype="analyst")
    await server._resolve_archetype(cfg, _req("u1"), None)
    assert cfg.provider_api_key == "sk-tier"
    env = {e["key"]: e["value"] for e in cfg.env_vars}
    assert env == {"BRAVE_API_KEY": "brave-1", "TAVILY_API_KEY": "tav-1"}
    assert "BRAVE_API_KEY=brave-1" in server._build_env(cfg)


async def test_caller_env_wins_and_set_never_clobbers_the_llm_key(patch_registry):
    patch_registry.set(
        archetypes=[_ANALYST],
        tiers={"balanced": {"provider": "anthropic", "model": "claude-sonnet-5", "api_key": "sk-tier"}},
        env=[{"key": "BRAVE_API_KEY", "value": "from-set"},
             {"key": "ANTHROPIC_API_KEY", "value": "stale-extra"}],
    )
    cfg = server.AgentConfig(name="x", archetype="analyst",
                             env_vars=[{"key": "BRAVE_API_KEY", "value": "from-caller"}])
    await server._resolve_archetype(cfg, _req("u1"), None)
    env = {e["key"]: e["value"] for e in cfg.env_vars}
    assert env == {"BRAVE_API_KEY": "from-caller"}
    assert "ANTHROPIC_API_KEY=sk-tier" in server._build_env(cfg)
    assert "stale-extra" not in server._build_env(cfg)


async def test_system_tier_key_is_left_for_the_fresh_org_key_lookup(patch_registry, monkeypatch):
    """Team-default sets store "@system"; it's carried verbatim (as the Library
    spawn sends it) and _resolve_spawn_provider_key swaps in the org key from a
    fresh read just before .env is written."""
    import captain_claw.flight_deck.auth as auth_mod

    class _DB:
        async def get_system_setting(self, key):
            return '{"openai": "sk-org"}' if key == "fd:provider-keys" else None

    patch_registry.set(archetypes=[_ANALYST],
                       tiers={"balanced": {"provider": "openai", "model": "gpt-6", "api_key": "@system"}})
    cfg = server.AgentConfig(name="x", archetype="analyst")
    await server._resolve_archetype(cfg, _req("u1"), None)
    assert cfg.provider_api_key == "@system"
    monkeypatch.setattr(auth_mod, "get_db", lambda: _DB())
    await server._resolve_spawn_provider_key(cfg)
    assert (cfg.provider, cfg.model, cfg.provider_api_key) == ("openai", "gpt-6", "sk-org")


async def test_blank_tier_key_never_becomes_the_org_key(patch_registry, monkeypatch):
    """A blank tier key stays blank: the set's own OPENAI_API_KEY applies (the
    user's key), never a silent org key — also not to a custom gateway."""
    import captain_claw.flight_deck.basna_routes as basna_mod

    monkeypatch.setattr(basna_mod, "_SYSTEM_PROVIDER_KEYS", {"openai": "sk-org"})
    patch_registry.set(
        archetypes=[_ANALYST],
        tiers={"balanced": {"provider": "openai", "model": "gpt-6", "api_key": "",
                            "base_url": "https://gw.example/v1"}},
        env=[{"key": "OPENAI_API_KEY", "value": "sk-user-own"}])
    cfg = server.AgentConfig(name="x", archetype="analyst")
    await server._resolve_archetype(cfg, _req("u1"), None)
    assert cfg.provider_api_key == ""
    env = server._build_env(cfg)
    assert "OPENAI_API_KEY=sk-user-own" in env and "sk-org" not in env


async def test_blank_tier_key_keeps_the_callers_same_provider_key(patch_registry):
    """The flight_deck tool sends the parent's own key; a blank tier key on the
    same provider doesn't replace it."""
    patch_registry.set(archetypes=[_ANALYST],
                       tiers={"balanced": {"provider": "anthropic", "model": "claude-sonnet-5", "api_key": ""}})
    cfg = server.AgentConfig(name="x", archetype="analyst", provider="anthropic",
                             provider_api_key="sk-ant-caller")
    await server._resolve_archetype(cfg, _req("u1"), None)
    assert cfg.provider_api_key == "sk-ant-caller"


async def test_provider_switch_without_a_key_drops_the_inherited_one(patch_registry):
    patch_registry.set(archetypes=[_ANALYST],
                       tiers={"balanced": {"provider": "openai", "model": "gpt-6", "api_key": ""}})
    cfg = server.AgentConfig(name="x", archetype="analyst", provider="anthropic",
                             provider_api_key="sk-ant-caller")
    await server._resolve_archetype(cfg, _req("u1"), None)
    assert cfg.provider == "openai" and cfg.provider_api_key == ""


async def test_tier_context_sizes_apply_unless_the_caller_set_them(patch_registry):
    patch_registry.set(archetypes=[_ANALYST],
                       tiers={"balanced": {"provider": "openai", "model": "gpt-6", "api_key": "k",
                                           "output_ctx": 8192, "input_ctx": 64000}})
    kiosk = server.AgentConfig(name="x", archetype="analyst")
    await server._resolve_archetype(kiosk, _req("u1"), None)
    assert (kiosk.max_tokens, kiosk.max_context) == (8192, 64000)
    explicit = server.AgentConfig(name="x", archetype="analyst", max_tokens=4096, max_context=32000)
    await server._resolve_archetype(explicit, _req("u1"), None)
    assert (explicit.max_tokens, explicit.max_context) == (4096, 32000)


async def test_malformed_env_entries_are_skipped_not_fatal(patch_registry):
    patch_registry.set(archetypes=[{"id": "plain", "role": "Plain"}],
                       env=["garbage", None, {"key": "BRAVE_API_KEY", "value": "brave-1"}])
    cfg = server.AgentConfig(name="x", archetype="plain")
    await server._resolve_archetype(cfg, _req("u1"), None)
    assert cfg.env_vars == [{"key": "BRAVE_API_KEY", "value": "brave-1"}]


async def test_tier_set_env_applies_without_a_tier(patch_registry):
    patch_registry.set(archetypes=[{"id": "plain", "role": "Plain"}],
                       env=[{"key": "BRAVE_API_KEY", "value": "brave-1"}])
    cfg = server.AgentConfig(name="x", archetype="plain")
    await server._resolve_archetype(cfg, _req("u1"), None)
    assert cfg.env_vars == [{"key": "BRAVE_API_KEY", "value": "brave-1"}]


# ── a child spawned by an agent: the caller's key and endpoint are a baseline ─
#
# The flight_deck tool sends the PARENT's provider / model / key / base_url so a
# child always has a usable model. A key belongs to its endpoint, so the two
# must travel together — never the tier's key to the parent's gateway, never the
# parent's key to the tier's endpoint.

GW = "https://gw.example/v1"


def _from_parent(archetype="analyst", **model) -> "server.AgentConfig":
    unit = {"provider": "openai", "model": "swift", "provider_api_key": "gw-key", "base_url": GW, **model}
    return server.AgentConfig(name="kid", archetype=archetype, **unit)


@pytest.mark.parametrize("tier_key", ["@system", "sk-real-openai"])
async def test_a_keyed_tier_runs_on_its_own_endpoint_not_the_parents_gateway(patch_registry, tier_key):
    """The parent runs on a gateway; the child's tier is the same provider on
    its OWN endpoint, with a key. The child must not be routed at the gateway
    (its key — or the team key behind "@system" — would go there)."""
    patch_registry.set(archetypes=[_ANALYST],
                       tiers={"balanced": {"provider": "openai", "model": "gpt-6", "api_key": tier_key}})
    cfg = _from_parent()
    await server._resolve_archetype(cfg, _req("u1"), None)
    assert (cfg.provider, cfg.model, cfg.base_url, cfg.provider_api_key) == ("openai", "gpt-6", "", tier_key)


@pytest.mark.parametrize("parent_base_url", ["", "https://other-gw.example/v1"])
async def test_the_parents_key_does_not_follow_the_child_to_another_endpoint(patch_registry, parent_base_url):
    patch_registry.set(archetypes=[_ANALYST],
                       tiers={"balanced": {"provider": "openai", "model": "swift", "api_key": "", "base_url": GW}})
    cfg = _from_parent(provider_api_key="parents-key", base_url=parent_base_url)
    await server._resolve_archetype(cfg, _req("u1"), None)
    assert cfg.base_url == GW and cfg.provider_api_key == ""


async def test_the_same_endpoint_written_differently_keeps_the_parents_key(patch_registry):
    patch_registry.set(archetypes=[_ANALYST], tiers={"balanced": {
        "provider": "openai", "model": "other-model", "api_key": "", "base_url": "HTTPS://GW.example/v1/"}})
    cfg = _from_parent()
    await server._resolve_archetype(cfg, _req("u1"), None)
    assert cfg.model == "other-model" and cfg.provider_api_key == "gw-key"


async def test_a_bare_model_tier_takes_the_parents_key_and_endpoint_as_one(patch_registry):
    """No key, no endpoint: the tier is just another model where the parent runs."""
    patch_registry.set(archetypes=[_ANALYST], tiers={"balanced": {"provider": "openai", "model": "other-model"}})
    cfg = _from_parent()
    await server._resolve_archetype(cfg, _req("u1"), None)
    assert (cfg.model, cfg.base_url, cfg.provider_api_key) == ("other-model", GW, "gw-key")


async def test_a_tier_missing_from_the_set_does_not_put_another_providers_model_on_the_parents_gateway(
        patch_registry):
    """The registry's tiers are Anthropic models with no key and no endpoint: on
    the parent's key and gateway URL that child could not run. The parent's
    own working model is kept instead."""
    patch_registry.set(archetypes=[_ANALYST], tiers={})
    cfg = _from_parent()
    await server._resolve_archetype(cfg, _req("u1"), None)
    server._resolve_tier(cfg)
    assert (cfg.provider, cfg.model, cfg.base_url, cfg.provider_api_key) == ("openai", "swift", GW, "gw-key")
    assert cfg.tier == ""


async def test_the_registry_tier_still_applies_on_the_parents_own_provider_or_with_no_model(patch_registry):
    patch_registry.set(archetypes=[_ANALYST], tiers={})
    same = server.AgentConfig(name="kid", archetype="analyst", provider="anthropic",
                              model="claude-haiku", provider_api_key="sk-ant")
    await server._resolve_archetype(same, _req("u1"), None)
    server._resolve_tier(same)
    assert same.provider == "anthropic" and same.model != "claude-haiku"   # the tier's model…
    assert same.provider_api_key == "sk-ant"                                # …on the parent's key
    kiosk = server.AgentConfig(name="kid", archetype="analyst")            # no model of its own
    await server._resolve_archetype(kiosk, _req("u1"), None)
    assert kiosk.tier == "balanced"


async def test_a_chatgpt_sign_in_tier_does_not_land_on_the_parents_gateway(patch_registry):
    """A GPT-5 / Codex tier has no key and no endpoint by nature — it signs in
    through ChatGPT. On the parent's gateway it would post the ChatGPT token
    there."""
    patch_registry.set(archetypes=[_ANALYST], tiers={"balanced": {"provider": "openai", "model": "gpt-5.2"}})
    cfg = _from_parent()
    await server._resolve_archetype(cfg, _req("u1"), None)
    assert (cfg.model, cfg.base_url, cfg.provider_api_key) == ("gpt-5.2", "", "")


async def test_a_keyed_model_tier_does_not_land_on_a_chatgpt_parents_endpoint(patch_registry):
    patch_registry.set(archetypes=[_ANALYST], tiers={"balanced": {"provider": "openai", "model": "gpt-4.1"}})
    cfg = _from_parent(model="gpt-5.2", provider_api_key="",
                       base_url="https://chatgpt.com/backend-api/codex/responses")
    await server._resolve_archetype(cfg, _req("u1"), None)
    assert (cfg.model, cfg.base_url) == ("gpt-4.1", "")


async def test_two_chatgpt_models_stay_together(patch_registry):
    patch_registry.set(archetypes=[_ANALYST], tiers={"balanced": {"provider": "openai", "model": "gpt-5.2-codex"}})
    codex = "https://chatgpt.com/backend-api/codex/responses"
    cfg = _from_parent(model="gpt-5.2", provider_api_key="", base_url=codex)
    await server._resolve_archetype(cfg, _req("u1"), None)
    assert (cfg.model, cfg.base_url) == ("gpt-5.2-codex", codex)


def test_a_providers_own_endpoint_written_out_is_no_other_endpoint():
    same = server._same_endpoint
    assert same("openrouter", "https://openrouter.ai/api/v1", "openrouter", "")      # Freebie agents write it
    assert same("openai", "", "openai", "https://api.openai.com/v1/")
    assert same("openai", "http://localhost:1234/v1", "openai", "http://127.0.0.1:1234/v1")
    assert not same("openai", "https://openrouter.ai/api/v1", "openai", "")          # another provider's URL
    assert not same("openai", GW, "openai", "")
    assert not same("openai", GW, "anthropic", GW)
    assert same("anthropic", "https://api.anthropic.com", "anthropic", "")
    assert same("anthropic", "", "anthropic", "https://api.anthropic.com/v1")
    assert same("xai", "https://api.x.ai/v1", "xai", "")
    assert not same("xai", "https://api.x.ai/v2", "xai", "")
    assert same("chatgpt", GW, "openai", GW) and same(" OpenAI ", "", "openai", "https://api.openai.com/v1")
    assert same("claude", "", "anthropic", "") and same("grok", "", "xai", "") and same("google", GW, "gemini", GW)


async def test_a_tier_that_spells_out_the_providers_endpoint_keeps_the_parents_key(patch_registry):
    patch_registry.set(archetypes=[_ANALYST], tiers={"balanced": {
        "provider": "openrouter", "model": "qwen/qwen3:free", "api_key": "",
        "base_url": "https://openrouter.ai/api/v1"}})
    cfg = _from_parent(provider="openrouter", model="x/y", provider_api_key="sk-or", base_url="")
    await server._resolve_archetype(cfg, _req("u1"), None)
    assert cfg.provider_api_key == "sk-or"

