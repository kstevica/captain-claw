"""Bat Phase 8 (backend half) — owner-authenticated read routes for the Bat page.

The page's list/detail/events routes are owner-scoped (get_current_user), distinct
from the agent-guarded /fd/bat/agent/* endpoints. Called here with the user dict
directly (bypassing the Depends) to verify scoping and shape."""

from __future__ import annotations

import uuid

import fastapi
import pytest

from captain_claw.flight_deck import bat_loop, bat_routes
from captain_claw.flight_deck.bat_store import BatStore


@pytest.fixture
async def store(tmp_path):
    s = BatStore(tmp_path / "bat.db")
    await s.init()
    bat_loop.set_store(s)
    yield s
    await s.close()


async def _mk(store, owner="u1"):
    rid = f"bat_{uuid.uuid4().hex[:8]}"
    await store.create_run(run_id=rid, owner_id=owner, title="T", task="do it",
                           config={"steps": ["a"], "real_usd_cap": 50.0}, status="running")
    await store.seed_steps(rid, [{"step_key": "a", "title": "A"}])
    await store.append_event(rid, "phase", "running")
    return rid


async def test_list_runs_is_owner_scoped(store):
    rid = await _mk(store, "u1")
    await _mk(store, "u2")
    mine = await bat_routes.list_runs_ui(user={"id": "u1"})
    assert [r["id"] for r in mine["runs"]] == [rid]
    assert mine["runs"][0]["real_usd_cap"] == 50.0


async def test_get_run_detail_and_404_for_other_owner(store):
    rid = await _mk(store, "u1")
    detail = await bat_routes.get_run_ui(rid, user={"id": "u1"})
    assert detail["run"]["id"] == rid
    assert detail["steps"] and detail["events"]
    assert detail["spend"]["cap"] == 50.0 and "committed_usd" in detail["spend"]
    with pytest.raises(fastapi.HTTPException):
        await bat_routes.get_run_ui(rid, user={"id": "someone-else"})


async def test_events_since_cursor(store):
    rid = await _mk(store, "u1")
    await store.append_event(rid, "step", "A", agent="a")
    ev = await bat_routes.get_events_ui(rid, since=1, user={"id": "u1"})
    assert ev["status"] == "running"
    assert all(e["i"] >= 1 for e in ev["events"])  # cursor honored
