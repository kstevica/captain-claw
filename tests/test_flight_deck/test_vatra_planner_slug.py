"""The Group-0 Long Horizon Planner must spawn under a per-session-unique
process name — never the fixed "long-horizon-planner" slug.

A fixed slug lets only one planner exist FD-wide: two concurrent Vatra runs
(or an orphaned planner left by a crashed run) collide with "process already
running", blocking every future run. Real Vatra workers already name
themselves ``vatra-<sid8>-...``; the planner had been the lone exception.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from captain_claw.flight_deck import vatra_routes as vr


async def test_planner_spawns_with_session_unique_name(monkeypatch):
    captured: dict = {}

    async def _fake_spawn_worker(request, user, *, name, **kwargs):
        captured["name"] = name
        # Report the planner as unavailable → returns the pass-through plan
        # immediately, so we never enter the dispatch path.
        return {"ok": False, "slug": name, "port": 0, "auth": "", "message": "x"}

    monkeypatch.setattr(vr, "_spawn_worker", _fake_spawn_worker)
    # Silence progress + keep the pass-through builder trivial.
    monkeypatch.setattr(vr, "_progress", lambda *a, **k: None)
    monkeypatch.setattr(vr, "_passthrough_group0_plan", lambda subtasks, arch: {"ok": True})

    sid = "abcd1234efgh5678"
    plan = await vr._run_group0_planner(
        request=SimpleNamespace(), user={"id": "u1"}, sid=sid,
        intent="do the thing", shared_context="", file_names=[],
        subtasks=[], arch_by_id={}, tiers=None, api_key="", env_vars=None,
        timeout=1.0)

    assert plan == {"ok": True}  # pass-through path taken
    # The spawn name is per-session-unique — not the fixed "Long Horizon Planner".
    assert captured["name"] == f"vatra-{sid[:8]}-planner"
    assert captured["name"] != "Long Horizon Planner"
