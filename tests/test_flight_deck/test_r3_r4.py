"""Tests for R3 (closer finding-triage) and R4 (delta continuation seed).

Both are pure helpers, so no DB / model / agents are needed.
"""

from __future__ import annotations

from captain_claw.flight_deck.basna_routes import (
    _TRUTH_CHARS,
    _TRUTH_CHARS_DELTA,
    _delta_seed,
)
from captain_claw.flight_deck.horizon_worker import _triage_feedback


# ── R3: critic finding triage ────────────────────────────────────────

def test_triage_empty_is_empty():
    assert _triage_feedback([]) == ""
    assert _triage_feedback(["", "  "]) == ""


def test_triage_single_reason_is_verbatim():
    assert _triage_feedback(["the claim about X is unsupported"]) == \
        "the claim about X is unsupported"


def test_triage_dedupes_and_numbers():
    out = _triage_feedback([
        "The market-size figure is unsupported.",
        "the market-size figure is unsupported.",   # dup (case)
        "No competitor analysis.",
    ])
    assert out.startswith("Fix each of these distinct objections:")
    assert "1. The market-size figure is unsupported." in out
    assert "2. No competitor analysis." in out
    # The duplicate did not produce a third item.
    assert "3." not in out


# ── R4: delta continuation seed ──────────────────────────────────────

def test_delta_off_matches_previous_behaviour():
    # Short prior: full inline, no file, no directive.
    assert _delta_seed("x" * 100, False) == (_TRUTH_CHARS, False, "")
    # Long prior: still full window, but spill to file (the old `big` path).
    length, big, directive = _delta_seed("x" * (_TRUTH_CHARS + 1), False)
    assert length == _TRUTH_CHARS and big is True and directive == ""


def test_delta_on_inlines_short_preview_and_always_spills():
    length, big, directive = _delta_seed("x" * 100, True)
    assert length == _TRUTH_CHARS_DELTA
    assert big is True                     # full text always goes to the workspace file
    assert "DELTA ROUND" in directive
    assert _TRUTH_CHARS_DELTA < _TRUTH_CHARS  # genuinely smaller inline footprint
