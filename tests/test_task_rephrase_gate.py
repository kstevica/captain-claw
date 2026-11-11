"""Task rephrase is off unless the agent's owner turns it on (task_rephrase.py).
When on, Flight Deck workers (Basna/Vatra/Council/Code) still skip it — the
orchestrator already framed their task — and standalone agents rephrase."""

from __future__ import annotations

import pytest

from captain_claw import task_rephrase
from captain_claw.agent_reasoning_mixin import (
    AgentReasoningMixin,
    _FD_WORKER_MARKERS,
    _is_fd_spawned_worker,
)
from captain_claw.config import Config, ScaleConfig, get_config, set_config

# A prompt complex enough to clear the rephrase threshold (2 URLs + list/CSV):
_COMPLEX = (
    "Research each of these organisations and produce a CSV table: "
    "https://example.com/a and https://example.org/b — extract name, "
    "registration number, and processing activities into columns."
)


def _clear(monkeypatch):
    for m in _FD_WORKER_MARKERS:
        monkeypatch.delenv(m, raising=False)


@pytest.fixture
def switch(tmp_path, monkeypatch):
    """The agent's ~/.captain-claw/task_rephrase.txt, in a temp folder (absent)."""
    path = tmp_path / task_rephrase.FILENAME
    monkeypatch.setattr(task_rephrase, "SETTING_PATH", path)
    return path


@pytest.fixture
def config():
    """A clean in-memory config (the defaults), put back afterwards."""
    prev = get_config()
    set_config(Config())
    yield
    set_config(prev)


@pytest.fixture
def rephrase_on(switch):
    switch.write_text("on\n")
    return switch


# ── Off by default ──
# Captain began with weaker models; current ones follow the prompt as written.

def test_the_config_default_is_off():
    assert ScaleConfig().task_rephrase_enabled is False


def test_a_complex_task_is_not_rephrased_by_default(monkeypatch, switch, config):
    _clear(monkeypatch)
    assert task_rephrase.enabled() is False
    assert AgentReasoningMixin._should_rephrase_task(_COMPLEX) is False


def test_the_switch_file_turns_it_on_and_off(monkeypatch, switch, config):
    _clear(monkeypatch)
    switch.write_text("on")
    assert AgentReasoningMixin._should_rephrase_task(_COMPLEX) is True
    switch.write_text("off")
    assert AgentReasoningMixin._should_rephrase_task(_COMPLEX) is False


def test_without_the_file_the_config_decides(monkeypatch, switch):
    _clear(monkeypatch)
    prev = get_config()
    try:
        set_config(Config(scale=ScaleConfig(task_rephrase_enabled=True)))
        assert task_rephrase.enabled() is True
        # …and the file wins over it.
        switch.write_text("off")
        assert task_rephrase.enabled() is False
    finally:
        set_config(prev)


def test_an_unreadable_switch_falls_back_to_the_config(switch, config):
    switch.write_text("maybe")
    assert task_rephrase.enabled() is False


def test_clear_drops_the_switch(switch):
    switch.write_text("on")
    task_rephrase.clear()
    assert not switch.exists()
    task_rephrase.clear()  # nothing to drop: no error


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


def test_standalone_agent_rephrases_a_complex_task_when_on(monkeypatch, rephrase_on):
    _clear(monkeypatch)
    # Turned on; a complex prompt clears the bar.
    assert AgentReasoningMixin._should_rephrase_task(_COMPLEX) is True


@pytest.mark.parametrize("marker", _FD_WORKER_MARKERS)
def test_fd_worker_skips_rephrase_even_for_a_complex_task(monkeypatch, marker, rephrase_on):
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


async def test_a_turn_is_not_rephrased_when_off(switch, config):
    out, changed = await _Rephraser()._rephrase_task(_COMPLEX, {})
    assert out == _COMPLEX and changed is False
