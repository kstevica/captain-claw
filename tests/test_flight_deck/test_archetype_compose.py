"""Tests for the function×domain archetype grid (PROTOTYPE, FD_ARCHETYPE_GRID).

Covers the pure composition core (`parse_pair`, `_max_tier`, `compose_archetype`),
the flag-gated `resolve_pair` (including against the real shipped registries), and
the `server._resolve_archetype` seam branch that folds a composed pair into a spawn
config when the flag is on and defers to the base lookup when it is off.
"""

from __future__ import annotations

import types

import pytest

import captain_claw.flight_deck.archetype_compose as gc
import captain_claw.flight_deck.server as server


def _req(uid: str = ""):
    return types.SimpleNamespace(state=types.SimpleNamespace(user_id=uid))


# ── parse_pair ───────────────────────────────────────────────────────

def test_parse_pair_splits_function_and_domain():
    assert gc.parse_pair("reviewer.legal") == ("reviewer", "legal")


def test_parse_pair_plain_id_has_no_domain():
    assert gc.parse_pair("code-reviewer") == ("code-reviewer", None)


def test_parse_pair_first_dot_only():
    # A function id may not contain a dot; a domain id may.
    assert gc.parse_pair("writer.us.finance") == ("writer", "us.finance")


def test_parse_pair_empty_half_is_not_a_pair():
    assert gc.parse_pair("reviewer.") == ("reviewer.", None)
    assert gc.parse_pair(".legal") == (".legal", None)


# ── _max_tier ────────────────────────────────────────────────────────

def test_max_tier_floor_raises():
    assert gc._max_tier("balanced", "reason") == "reason"


def test_max_tier_floor_lower_is_noop():
    assert gc._max_tier("reason", "balanced") == "reason"


def test_max_tier_no_floor_keeps_function_tier():
    assert gc._max_tier("fast", None) == "fast"


def test_max_tier_unknown_tier_does_not_raise():
    # A specialist tier outside the ladder → no floor applied, function tier kept.
    assert gc._max_tier("balanced", "coding") == "balanced"
    assert gc._max_tier("vision", "reason") == "vision"


# ── compose_archetype ────────────────────────────────────────────────

def _fn():
    return {
        "id": "reviewer", "role": "Reviewer", "cognitive_mode": "locrian",
        "tier": "balanced", "tools": ["read", "insights"], "recall_mode": "pool",
        "keywords": ["review"], "fleet_instructions": "You are a Reviewer.\nSOP here.",
    }


def _dm():
    return {
        "id": "legal", "label": "Legal", "tier_floor": "reason",
        "tools_add": ["citation_lookup"], "recall_override": "domain",
        "keywords": ["contract"], "overlay": "## Domain: Legal\nGround claims in statute.",
    }


def test_compose_id_role_and_family():
    c = gc.compose_archetype(_fn(), _dm())
    assert c["id"] == "reviewer.legal"
    assert c["role"] == "Legal Reviewer"
    assert c["family"] == "Legal"


def test_compose_instructions_are_function_then_overlay():
    c = gc.compose_archetype(_fn(), _dm())
    assert c["fleet_instructions"].startswith("You are a Reviewer.")
    assert c["fleet_instructions"].endswith("Ground claims in statute.")
    assert "## Domain: Legal" in c["fleet_instructions"]


def test_compose_tools_are_union():
    c = gc.compose_archetype(_fn(), _dm())
    assert c["tools"] == ["citation_lookup", "insights", "read"]  # sorted union


def test_compose_tier_raised_to_domain_floor():
    c = gc.compose_archetype(_fn(), _dm())
    assert c["tier"] == "reason"


def test_compose_cognitive_mode_is_functions():
    assert gc.compose_archetype(_fn(), _dm())["cognitive_mode"] == "locrian"


def test_compose_recall_override_wins():
    assert gc.compose_archetype(_fn(), _dm())["recall_mode"] == "domain"


def test_compose_recall_defaults_to_function_when_no_override():
    dm = _dm()
    dm["recall_override"] = None
    assert gc.compose_archetype(_fn(), dm)["recall_mode"] == "pool"


def test_compose_memory_tags_carry_both_axes():
    assert gc.compose_archetype(_fn(), _dm())["memory_tags"] == ["agent:reviewer", "domain:legal"]


# ── recall_filter ────────────────────────────────────────────────────

def test_recall_filter_pool_is_empty():
    assert gc.recall_filter("pool", ["agent:reviewer", "domain:legal"]) == ""


def test_recall_filter_empty_mode_is_pool():
    assert gc.recall_filter("", ["domain:legal"]) == ""
    assert gc.recall_filter(None, ["domain:legal"]) == ""


def test_recall_filter_domain_narrows_to_domain_tag():
    assert gc.recall_filter("domain", ["agent:reviewer", "domain:legal"]) == "tags:=`domain:legal`"


def test_recall_filter_self_narrows_to_agent_tag():
    assert gc.recall_filter("self", ["agent:reviewer", "domain:legal"]) == "tags:=`agent:reviewer`"


def test_recall_filter_missing_tag_degrades_to_pool():
    assert gc.recall_filter("domain", ["agent:reviewer"]) == ""   # no domain: tag
    assert gc.recall_filter("self", ["domain:legal"]) == ""       # no agent: tag
    assert gc.recall_filter("domain", []) == ""


def test_recall_filter_unknown_mode_is_pool():
    assert gc.recall_filter("everything", ["domain:legal"]) == ""


# ── resolve_pair (flag gating + real registries) ─────────────────────

def test_resolve_pair_off_by_default(monkeypatch):
    monkeypatch.delenv("FD_ARCHETYPE_GRID", raising=False)
    assert gc.resolve_pair("reviewer.legal") is None


def test_resolve_pair_flag_off_returns_none_even_for_valid_pair(monkeypatch):
    monkeypatch.setenv("FD_ARCHETYPE_GRID", "0")
    assert gc.resolve_pair("reviewer.legal") is None


def test_resolve_pair_plain_id_is_none_when_flag_on(monkeypatch):
    monkeypatch.setenv("FD_ARCHETYPE_GRID", "1")
    assert gc.resolve_pair("code-reviewer") is None


def test_resolve_pair_unknown_axis_is_none(monkeypatch):
    monkeypatch.setenv("FD_ARCHETYPE_GRID", "1")
    assert gc.resolve_pair("reviewer.nope") is None
    assert gc.resolve_pair("nope.legal") is None


def test_resolve_pair_shipped_registry_composes(monkeypatch):
    """Against the real functions.json / domains.json this repo ships."""
    monkeypatch.setenv("FD_ARCHETYPE_GRID", "true")
    c = gc.resolve_pair("reviewer.legal")
    assert c is not None
    assert c["id"] == "reviewer.legal"
    assert c["role"] == "Legal Reviewer"
    assert c["tier"] == "reason"                       # legal floor raises balanced
    assert c["recall_mode"] == "domain"                # legal recall_override
    assert c["memory_tags"] == ["agent:reviewer", "domain:legal"]
    assert "## Domain: Legal" in c["fleet_instructions"]


def test_resolve_pair_finance_keeps_function_recall(monkeypatch):
    monkeypatch.setenv("FD_ARCHETYPE_GRID", "1")
    c = gc.resolve_pair("researcher.finance")
    assert c is not None
    assert c["recall_mode"] == "pool"                  # finance has no override
    assert c["tier"] == "balanced"                     # floor == function tier, no raise


# ── server._resolve_archetype seam branch ────────────────────────────

@pytest.fixture
def patch_owner_tiers(monkeypatch: pytest.MonkeyPatch):
    """Stub the owner-tier + db seams so the resolver needs no DB. Empty tiers →
    the composed archetype's tier is left for `_resolve_tier`, model untouched."""
    import captain_claw.flight_deck.auth as auth_mod
    import captain_claw.flight_deck.basna_routes as basna_mod

    async def fake_owner_tiers(db, uid):
        return {}, []

    monkeypatch.setattr(auth_mod, "get_db", lambda: object())
    monkeypatch.setattr(basna_mod, "_load_owner_tiers", fake_owner_tiers)


async def test_seam_folds_composed_pair_when_flag_on(monkeypatch, patch_owner_tiers):
    monkeypatch.setenv("FD_ARCHETYPE_GRID", "1")
    cfg = server.AgentConfig(name="x", archetype="reviewer.legal")
    await server._resolve_archetype(cfg, _req(), None)
    assert cfg.cognitive_mode == "locrian"             # from the reviewer function
    assert cfg.description == "Legal Reviewer"          # composed role
    assert "read" in cfg.tools and "insights" in cfg.tools


async def test_seam_defers_to_base_lookup_when_flag_off(monkeypatch, patch_owner_tiers):
    """Flag off → the dotted id is not composed; the base lookup finds nothing and
    the resolver is a non-fatal no-op that leaves the config untouched."""
    monkeypatch.delenv("FD_ARCHETYPE_GRID", raising=False)

    import captain_claw.flight_deck.archetypes as arch_mod

    async def empty_merged(db, uid):
        return []

    monkeypatch.setattr(arch_mod, "merged_archetypes", empty_merged)
    cfg = server.AgentConfig(name="x", archetype="reviewer.legal")
    before = cfg.model_copy(deep=True)
    await server._resolve_archetype(cfg, _req(), None)
    assert cfg.cognitive_mode == before.cognitive_mode  # unchanged
    assert cfg.description == before.description
