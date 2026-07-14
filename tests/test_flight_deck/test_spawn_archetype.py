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

    state = types.SimpleNamespace(archetypes=[], tiers={}, last_uid=None)

    async def fake_merged(db, uid):
        state.last_uid = uid
        return list(state.archetypes)

    async def fake_owner_tiers(db, uid):
        return dict(state.tiers), []

    monkeypatch.setattr(arch_mod, "merged_archetypes", fake_merged)
    monkeypatch.setattr(auth_mod, "get_db", lambda: object())
    monkeypatch.setattr(basna_mod, "_load_owner_tiers", fake_owner_tiers)

    def _set(archetypes=None, tiers=None):
        state.archetypes = archetypes or []
        state.tiers = tiers or {}

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
        tiers={"reason": {"provider": "openai", "model": "gpt-5"}},  # no base_url
    )
    cfg = server.AgentConfig(name="x", archetype="fact-checker",
                             provider="openai", model="local", base_url="http://localhost:1234/v1")
    await server._resolve_archetype(cfg, _req("u1"), {"id": "u1"})
    assert cfg.provider == "openai" and cfg.model == "gpt-5"
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
