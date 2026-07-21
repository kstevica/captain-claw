"""Flight Deck workers (Basna/Vatra/Council/Code) skip the per-agent task-rephrase
— the orchestrator already framed their task. Standalone agents still rephrase."""

from __future__ import annotations

import pytest

from captain_claw.agent_reasoning_mixin import (
    AgentReasoningMixin,
    _FD_WORKER_MARKERS,
    _is_fd_spawned_worker,
)

# A prompt complex enough to clear the rephrase threshold (2 URLs + list/CSV):
_COMPLEX = (
    "Research each of these organisations and produce a CSV table: "
    "https://example.com/a and https://example.org/b — extract name, "
    "registration number, and processing activities into columns."
)


def _clear(monkeypatch):
    for m in _FD_WORKER_MARKERS:
        monkeypatch.delenv(m, raising=False)


def test_no_marker_is_not_a_worker(monkeypatch):
    _clear(monkeypatch)
    assert _is_fd_spawned_worker() is False


@pytest.mark.parametrize("marker", _FD_WORKER_MARKERS)
def test_each_mode_marker_is_detected(monkeypatch, marker):
    _clear(monkeypatch)
    monkeypatch.setenv(marker, "1")
    assert _is_fd_spawned_worker() is True


def test_marker_must_be_truthy(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("CLAW_BASNA_WORKER", "0")
    assert _is_fd_spawned_worker() is False
    monkeypatch.setenv("CLAW_BASNA_WORKER", "true")
    assert _is_fd_spawned_worker() is True


def test_standalone_agent_rephrases_a_complex_task(monkeypatch):
    _clear(monkeypatch)
    # Default config has task_rephrase_enabled=True; a complex prompt clears the bar.
    assert AgentReasoningMixin._should_rephrase_task(_COMPLEX) is True


@pytest.mark.parametrize("marker", _FD_WORKER_MARKERS)
def test_fd_worker_skips_rephrase_even_for_a_complex_task(monkeypatch, marker):
    _clear(monkeypatch)
    monkeypatch.setenv(marker, "1")
    assert AgentReasoningMixin._should_rephrase_task(_COMPLEX) is False


# ── Queued tasks are already briefs ──
# A queue message carries an exact id range, the standing rules, and explicit
# prohibitions ("never do +1 on the id!"). Rephrasing it is a weaker model
# rewriting instructions the user chose word by word — the same drift the task
# planner avoids by expanding one approved template.

class _Rephraser(AgentReasoningMixin):
    """Just enough object to reach the gate."""

    def _scale_system_disabled(self):
        return False


async def test_a_queued_turn_is_not_rephrased():
    agent = _Rephraser()
    agent._suppress_rephrase = True
    out, changed = await agent._rephrase_task(_COMPLEX, {})
    assert out == _COMPLEX and changed is False


def test_the_gate_reads_the_flag_before_deciding_complexity():
    """Ordering matters: the flag must win over the complexity heuristic,
    which would otherwise send a long enrichment brief off to be rewritten."""
    import inspect

    src = inspect.getsource(AgentReasoningMixin._rephrase_task)
    assert src.index("_suppress_rephrase") < src.index("_should_rephrase_task")


def test_the_flag_defaults_to_off():
    """Standalone chat keeps rephrasing — this suppression is opt-in."""
    agent = _Rephraser()
    assert getattr(agent, "_suppress_rephrase", False) is False
