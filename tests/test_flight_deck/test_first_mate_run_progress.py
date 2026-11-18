"""A run's progress on the Runs tab.

Pinned here:

* a run with a Bat run shows that run's steps, its newest events (and then
  only what's new since the last read), the workers it has right now by the
  name First Mate gave them, and its files' folder;
* a run with no runner run yet, or whose Bat run is gone, has nothing to show;
* only its owner sees it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from captain_claw.flight_deck import bat_loop, server
from captain_claw.flight_deck import first_mate_routes as routes
from captain_claw.flight_deck import first_mate_runs as runs
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck.bat_store import BatStore
from captain_claw.flight_deck.captain_swarm_outcome import worker_prefix

BOB, CAROL = "user-bob", "user-carol"


@pytest.fixture(autouse=True)
def _no_register(monkeypatch):
    from captain_claw.flight_deck import tool_audit_routes
    monkeypatch.setattr(tool_audit_routes, "_data_dir", None)
    monkeypatch.setattr(tool_audit_routes, "_store", None)


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", s)
    s.set_settings({"enabled": True, "admins_only": False})
    return s


@pytest.fixture
async def bat(tmp_path, monkeypatch):
    b = BatStore(tmp_path / "bat.db")
    await b.init()

    async def kick():
        return 0

    monkeypatch.setattr(bat_loop, "_STORE", b)
    monkeypatch.setattr(bat_loop, "kick", kick)
    yield b
    await b.close()


@pytest.fixture
def registry(tmp_path: Path, monkeypatch):
    """The deck's agents: a test's own registry, never the deck's."""
    path = tmp_path / ".processes.json"
    alive: set[str] = set()
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", path)
    monkeypatch.setattr(server, "_process_is_alive", lambda slug: slug in alive)

    def put(entries: dict, running: set[str]):
        path.write_text(json.dumps(entries))
        alive.clear()
        alive.update(running)

    return put


def _live(store, owner=BOB):
    f = store.create(owner, title="Weekly report", charter="Every Monday, summarize last week's project mail.")
    return store.approve(f["id"], 1, f["draft"]["sha"], owner)


async def test_progress_follows_its_bat_run(store, bat, registry, monkeypatch):
    f = _live(store)
    run = await runs.start(f["id"], BOB, watch=False)
    bid = run["child_id"]
    await bat.upsert_step(bid, "step-1", title="Find sources", status="done", seq=0)
    await bat.upsert_step(bid, "step-2", title="Collect the list", status="running", seq=1, attempt=2)
    for n in range(5):
        await bat.append_event(bid, "action", "web_fetch", agent="step-2", detail=f"page {n}")
    worker = worker_prefix(bid) + "step-2"
    registry({worker: {"name": "Weekly report · step 2"}, "someone-else": {"name": "Helper"}}, {worker})
    monkeypatch.setattr(runs, "PROGRESS_EVENTS", 3)

    got = await runs.progress(store.get_run(run["id"]))
    assert got["available"] and got["files"] == f"bat-{bid[:8]}"
    assert [(s["key"], s["title"], s["status"], s["attempt"]) for s in got["steps"]] == [
        ("step-1", "Find sources", "done", 0), ("step-2", "Collect the list", "running", 2)]
    # the newest three (its start note came first), with who did them
    assert [(e["agent"], e["detail"]) for e in got["events"]] == [("step-2", f"page {n}") for n in (2, 3, 4)]
    assert got["next"] == 6
    assert got["workers"] == [{"slug": worker, "name": "Weekly report · step 2", "step": "step-2", "running": True}]

    await bat.append_event(bid, "step", "Collect the list", agent="step-2", ok=True)
    more = await runs.progress(store.get_run(run["id"]), since=got["next"])
    assert [e["i"] for e in more["events"]] == [6] and more["next"] == 7
    quiet = await runs.progress(store.get_run(run["id"]), since=7)
    assert quiet["events"] == [] and quiet["next"] == 7


async def test_nothing_to_show(store, bat, registry):
    f = _live(store)
    queued = store.start_run(f["id"])  # no Bat run yet
    assert (await runs.progress(queued))["available"] is False
    gone = {**queued, "child_id": "bat_000000000000"}
    assert (await runs.progress(gone))["available"] is False


async def test_only_its_owner_sees_it(store, bat, registry):
    f = _live(store)
    run = await runs.start(f["id"], BOB, watch=False)
    got = await routes.run_progress(run["id"], since=0, user={"id": BOB, "role": "user"})
    assert got["available"] is True
    with pytest.raises(HTTPException) as no:
        await routes.run_progress(run["id"], since=0, user={"id": CAROL, "role": "user"})
    assert no.value.status_code == 404


async def test_an_unfinished_steps_progress_note_shows(store, bat, registry, tmp_path, monkeypatch):
    from captain_claw.flight_deck import vfs_routes
    monkeypatch.setattr(vfs_routes, "_user_root", lambda owner: tmp_path / "vfs" / owner)
    f = _live(store)
    run = await runs.start(f["id"], BOB, watch=False)
    bid = run["child_id"]
    await bat.upsert_step(bid, "step-1", title="Find sources", status="done", seq=0)
    await bat.upsert_step(bid, "step-2", title="Collect the list", status="running", seq=1)
    registry({}, set())
    notes = tmp_path / "vfs" / BOB / f"bat-{bid[:8]}" / "progress"
    notes.mkdir(parents=True)
    (notes / "step-1.md").write_text("Done: all sources listed.")
    (notes / "step-2.md").write_text("Done: 47 of 200 in leads.csv.\nNext: UK fintech.")
    got = await runs.progress(store.get_run(run["id"]))
    by_key = {s["key"]: s["note"] for s in got["steps"]}
    assert by_key == {"step-1": "", "step-2": "Done: 47 of 200 in leads.csv.\nNext: UK fintech."}
