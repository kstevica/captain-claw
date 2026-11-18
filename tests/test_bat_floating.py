"""Bat's stubborn worker turns: a floating step budget, a floating time limit,
and notes that outlive the worker.

Pinned here:

* without a floating budget a turn's ceiling is 4x planned, at most 300, and Jev
  decides past 2x; with one (``CLAW_FLOATING_BUDGET``), progress alone carries it
  to that far ceiling;
* a floating time extension has no ceiling while the agent keeps working, and
  ends once it has been quiet for the window; the bounded rule is unchanged;
* a Bat step's worker keeps its context-governor notes in the run's folder, by
  its step, so a retry of the step finds them;
* the deck's ceilings are generous by default and can be set.
"""

from __future__ import annotations

import importlib

from captain_claw import context_spill
from captain_claw.agent_orchestration_mixin import _floating_ceiling, _turn_limits
from captain_claw.flight_deck import basna_routes as b


def test_the_step_budget_floats_only_when_asked(monkeypatch):
    monkeypatch.delenv("CLAW_FLOATING_BUDGET", raising=False)
    assert _turn_limits(40) == (160, 80) and _turn_limits(300) == (300, 600)
    monkeypatch.setenv("CLAW_FLOATING_BUDGET", "5000")
    assert _turn_limits(40) == (5000, 5000)  # progress alone goes all the way
    monkeypatch.setenv("CLAW_FLOATING_BUDGET", "nonsense")
    assert _floating_ceiling() == 0 and _turn_limits(40) == (160, 80)
    monkeypatch.setenv("CLAW_FLOATING_BUDGET", "10000000")
    assert _floating_ceiling() == 100_000


def test_the_time_floats_while_the_agent_works():
    # past the bounded ceiling, still working: another slice
    assert b._extend_deadline(9000.0, 3600.0, 8990.0, floating=900.0) == 9000.0 + b._EXTEND_SLICE_S
    # quiet for longer than the window: it ends
    assert b._extend_deadline(9000.0, 3600.0, 8000.0, floating=900.0) is None
    # the bounded rule is as it was
    assert b._extend_deadline(9000.0, 3600.0, 8990.0) is None
    assert b._extend_deadline(600.0, 1800.0, 590.0) == 600.0 + b._EXTEND_SLICE_S


def test_a_bat_workers_notes_live_in_the_runs_folder(monkeypatch):
    for k in ("CLAW_VATRA_SUBTASK", "CLAW_VATRA_SESSION"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("CLAW_BAT_SUBTASK", raising=False)
    monkeypatch.delenv("CLAW_BAT_SESSION", raising=False)
    monkeypatch.delenv("CLAW_VFS_PROJECT", raising=False)
    assert context_spill.notes_path() == "notes.md"  # an ordinary agent: as before
    monkeypatch.setenv("CLAW_VFS_PROJECT", "bat-bat_68b3")
    monkeypatch.setenv("CLAW_BAT_SESSION", "bat_68b36fcfcff5")
    monkeypatch.setenv("CLAW_BAT_SUBTASK", "step-3")
    path = context_spill.notes_path()
    assert path == "vfs:bat-bat_68b3/notes/bat_68b3-step-3-notes.md"
    assert context_spill.is_governor_notes_path(path)


def test_the_decks_ceilings_are_generous_and_settable(monkeypatch):
    from captain_claw.flight_deck import bat_routes
    assert bat_routes._BAT_STEP_CEILING >= 5000 and bat_routes._BAT_IDLE_S >= 15 * 60
    monkeypatch.setenv("FD_BAT_STEP_CEILING", "20000")
    monkeypatch.setenv("FD_BAT_IDLE_MINUTES", "60")
    try:
        importlib.reload(bat_routes)
        assert bat_routes._BAT_STEP_CEILING == 20000 and bat_routes._BAT_IDLE_S == 3600.0
    finally:
        monkeypatch.delenv("FD_BAT_STEP_CEILING")
        monkeypatch.delenv("FD_BAT_IDLE_MINUTES")
        importlib.reload(bat_routes)


# ── a model call that outgrows the window: llama.cpp's wording ─────────

def test_llama_cpps_overflow_is_read_with_its_numbers():
    from captain_claw.agent_orchestration_mixin import _looks_like_context_overflow
    from captain_claw.llm import _context_overflow, _output_budget_that_fits
    for text in (
        'Error code: 400 - {"error":{"code":400,"message":"the request exceeds the available context size, '
        'try increasing it","type":"exceed_context_size_error","n_prompt_tokens":110312,"n_ctx":98304}}',
        "Error code: 400 - {'error': {'code': 400, 'message': 'the request exceeds the available context size, "
        "try increasing it', 'type': 'exceed_context_size_error', 'n_prompt_tokens': 110312, 'n_ctx': 98304}}",
    ):
        low = text.lower()
        got = _context_overflow(low)
        assert got is not None and (got.capacity, got.prompt_tokens, got.requested) == (98304, 110312, 0)
        assert _looks_like_context_overflow(low)
        assert _output_budget_that_fits(low) is None  # the prompt alone is too big: the governor re-cuts
    # the wording alone, without numbers, is still an overflow
    assert _looks_like_context_overflow("the request exceeds the available context size, try increasing it")
    # numbers the other way round, with room left for some output
    got = _context_overflow('{"n_ctx": 131072, "type": "exceed_context_size_error", "n_prompt_tokens": 120000}')
    assert got is not None and (got.capacity, got.prompt_tokens) == (131072, 120000)
    assert _output_budget_that_fits('{"n_ctx": 131072, "n_prompt_tokens": 120000}') == 131072 - 120000 - 256
