"""Tests for A2 — the completion-honesty guard + output modes for Code agents.

Pure prompt strings; the gating (only apply within an active profile) lives in
the route and is exercised by the OFF-path invariant, but the directive assembly
itself is unit-testable here.
"""

from __future__ import annotations

from captain_claw.flight_deck import code_honesty


def test_guard_off_and_no_mode_is_empty():
    # The load-bearing invariant: nothing added → today's prompt byte-for-byte.
    assert code_honesty.guard_directive(False, "") == ""


def test_guard_on_adds_honesty():
    d = code_honesty.guard_directive(True, "")
    assert "CLAIM ONLY WHAT YOU VERIFIED" in d
    assert "not run" in d


def test_output_mode_conservative():
    assert code_honesty.output_mode_directive("conservative") == \
        code_honesty.CODE_CONSERVATIVE_DIRECTIVE
    assert "CAUTIOUS" in code_honesty.CODE_CONSERVATIVE_DIRECTIVE


def test_output_mode_complete():
    assert code_honesty.output_mode_directive("complete") == \
        code_honesty.CODE_COMPLETE_DIRECTIVE
    assert "FULL BUILD" in code_honesty.CODE_COMPLETE_DIRECTIVE


def test_output_mode_unknown_is_empty():
    assert code_honesty.output_mode_directive("") == ""
    assert code_honesty.output_mode_directive("wat") == ""


def test_guard_combines_honesty_and_mode():
    d = code_honesty.guard_directive(True, "conservative")
    assert "CLAIM ONLY WHAT YOU VERIFIED" in d
    assert "CAUTIOUS" in d


def test_guard_mode_only_without_honesty():
    d = code_honesty.guard_directive(False, "complete")
    assert "CLAIM ONLY WHAT YOU VERIFIED" not in d
    assert "FULL BUILD" in d
