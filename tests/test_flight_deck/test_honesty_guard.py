"""Increment 1 of the quality-tightening plan: honesty guard + output modes.

The honesty guard is the ONE default-on quality flag (locked 2026-07-10, see
docs/vatra-quality-tightening-plan.md): the pipeline's completeness pressure is
always on, so its anti-fabrication counterweight defaults on too. An explicit
``{"honesty_guard": false}`` must restore the pre-guard prompts byte-for-byte,
and the flag must stay OUT of ``any_enabled`` (which keeps meaning "any opt-in
lever on").
"""

from __future__ import annotations

from captain_claw.flight_deck import basna_routes
from captain_claw.flight_deck.quality_profile import (
    COMPLETE_MODE_DIRECTIVE,
    CONSERVATIVE_MODE_DIRECTIVE,
    JUDGMENT_LEDGER_DIRECTIVE,
    REPORTER_HONESTY_DIRECTIVE,
    UNVERIFIED_GUARD_DIRECTIVE,
    QualityProfile,
    output_mode_directive,
)


# ── the default-on exception ─────────────────────────────────────────

def test_honesty_guard_defaults_on_for_every_config_shape():
    for cfg in (None, {}, {"profile": "off"}, {"profile": "balanced"},
                {"profile": "thorough"}, {"profile": "bogus"}):
        p = QualityProfile.from_dict(cfg)
        assert p.honesty_guard is True, cfg


def test_honesty_guard_kill_switch():
    p = QualityProfile.from_dict({"honesty_guard": False})
    assert p.honesty_guard is False
    # The kill-switch composes with presets without disturbing them.
    p = QualityProfile.from_dict({"profile": "balanced", "honesty_guard": False})
    assert p.honesty_guard is False and p.judgment_ledger is True


def test_honesty_guard_stays_out_of_any_enabled():
    # any_enabled means "any opt-in lever on" — a default-on flag must not
    # make every profile read as enabled.
    p = QualityProfile.from_dict(None)
    assert p.honesty_guard is True and not p.any_enabled
    assert "honesty_guard" not in QualityProfile._BOOL_FLAGS


def test_honesty_guard_is_independent_of_judgment_ledger():
    # Pre-split, the guard rode the ledger flag. They are now independent.
    p = QualityProfile.from_dict({"judgment_ledger": True, "honesty_guard": False})
    assert p.judgment_ledger is True and p.honesty_guard is False
    p = QualityProfile.from_dict({"judgment_ledger": False})
    assert p.judgment_ledger is False and p.honesty_guard is True


# ── output_mode knob ─────────────────────────────────────────────────

def test_output_mode_defaults_empty_and_validates():
    assert QualityProfile.from_dict(None).output_mode == ""
    assert QualityProfile.from_dict({"output_mode": "conservative"}).output_mode == "conservative"
    assert QualityProfile.from_dict({"output_mode": "Complete"}).output_mode == "complete"
    # Unknown values degrade to today's behaviour, never raise.
    assert QualityProfile.from_dict({"output_mode": "yolo"}).output_mode == ""
    assert QualityProfile.from_dict({"output_mode": None}).output_mode == ""


def test_output_mode_directive_mapping():
    assert output_mode_directive("") == ""
    assert output_mode_directive("conservative") == CONSERVATIVE_MODE_DIRECTIVE
    assert output_mode_directive("complete") == COMPLETE_MODE_DIRECTIVE
    assert "REVIEW COPY" in CONSERVATIVE_MODE_DIRECTIVE
    assert "FULL DRAFT" in COMPLETE_MODE_DIRECTIVE


# ── directive content (what the prompts actually gain) ───────────────

def test_guard_carries_the_placeholder_and_estimate_policy():
    assert "[TO BE PROVIDED:" in UNVERIFIED_GUARD_DIRECTIVE
    assert "estimate — basis:" in UNVERIFIED_GUARD_DIRECTIVE
    assert "placeholder beats a plausible fabrication" in UNVERIFIED_GUARD_DIRECTIVE


def test_reporter_overlay_names_the_section_and_the_exception():
    assert "Unresolved & assumptions" in REPORTER_HONESTY_DIRECTIVE
    assert "Do not silently absorb disagreements" in REPORTER_HONESTY_DIRECTIVE
    # The ledger directive stays its own thing — no accidental re-bundling.
    assert "Unresolved" not in JUDGMENT_LEDGER_DIRECTIVE


# ── the Basna synthesizer overlay (the kill-switch's byte-for-byte claim) ─

class _FakeResp:
    content = "merged answer"


class _FakeProvider:
    def __init__(self, captured: list):
        self._captured = captured

    async def complete(self, messages, temperature=0.0, max_tokens=0):
        self._captured.extend(messages)
        return _FakeResp()


def _fake_provider_call(captured: list):
    return lambda creds, **kw: (_FakeProvider(captured), 1024)


_GOOD = [
    {"role": "analyst", "weight": 0.8, "output": "Answer A"},
    {"role": "critic", "weight": 0.6, "output": "Answer B"},
]


async def test_synthesize_without_honesty_matches_previous_prompt(monkeypatch):
    captured: list = []
    monkeypatch.setattr(basna_routes, "_provider_call", _fake_provider_call(captured))
    out = await basna_routes._llm_synthesize(_GOOD, "general", {})
    assert out == "merged answer"
    system = captured[0].content
    # The pre-guard system prompt, unchanged: no overlay, no mode block.
    assert "do not narrate the disagreement" in system
    assert "Unresolved & assumptions" not in system
    assert "OUTPUT MODE" not in system


async def test_synthesize_with_honesty_appends_the_overlay(monkeypatch):
    captured: list = []
    monkeypatch.setattr(basna_routes, "_provider_call", _fake_provider_call(captured))
    await basna_routes._llm_synthesize(
        _GOOD, "general", {}, honesty=True, output_mode="conservative")
    system = captured[0].content
    assert "Unresolved & assumptions" in system
    assert "REVIEW COPY" in system
    # The base reconciliation instruction is preserved, not replaced.
    assert "do not narrate the disagreement" in system
