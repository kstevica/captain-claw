"""The "suggested next steps" follow-up (an extra LLM call that offers the user
interactive options after a turn) must NOT fire for FD-spawned orchestrated
workers (Basna/Vatra/Council/Code) — they're headless and have no interactive
user. Every site that calls extract_next_steps must guard on the worker marker."""

from __future__ import annotations

from pathlib import Path

import pytest

from captain_claw.agent_reasoning_mixin import _FD_WORKER_MARKERS, _is_fd_spawned_worker

_PKG = Path(__file__).resolve().parent.parent / "captain_claw"


def _clear(monkeypatch):
    for m in _FD_WORKER_MARKERS:
        monkeypatch.delenv(m, raising=False)


@pytest.mark.parametrize("marker", _FD_WORKER_MARKERS)
def test_worker_marker_gates_next_steps(monkeypatch, marker):
    _clear(monkeypatch)
    assert _is_fd_spawned_worker() is False   # a normal interactive user
    monkeypatch.setenv(marker, "1")
    assert _is_fd_spawned_worker() is True     # an orchestrated worker


def _firing_sites() -> list[Path]:
    """Every module that actually triggers the follow-up extraction — i.e. calls
    extract_next_steps — excluding the module that *defines* it."""
    sites = []
    for p in _PKG.rglob("*.py"):
        if p.name == "next_steps.py":
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if "extract_next_steps(" in text:
            sites.append(p)
    return sites


def test_there_are_firing_sites_to_guard():
    # If this drops to zero, the scan below is silently vacuous.
    assert _firing_sites(), "expected at least one module calling extract_next_steps"


def test_every_firing_site_guards_on_the_worker_marker():
    ungated = [
        p.relative_to(_PKG).as_posix()
        for p in _firing_sites()
        if "_is_fd_spawned_worker" not in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert not ungated, (
        "these modules fire the next-steps follow-up without skipping FD-spawned "
        f"workers — add a `not _is_fd_spawned_worker()` guard: {ungated}"
    )
