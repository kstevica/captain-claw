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
