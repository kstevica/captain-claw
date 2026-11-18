"""Archiving First Mate factories and runs: off the owner's lists, kept whole,
restorable.

Pinned here:

* an archived factory leaves the list and stops counting toward the owner's
  factories; a running one is paused first; it can't be resumed or started
  until restored, and comes back paused; not while a run of it is going;
* archived and retired factories are the Archived view; an archived one can be
  deleted; last week's report still counts it;
* a finished run can be archived (not a going one), or all finished runs at
  once; archived runs leave Runs and Results and have their own view;
* the routes do the same, for the owner only.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import first_mate_routes as routes
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck.auth import get_current_user

BOB = {"id": "user-bob", "role": "user"}
EVE = {"id": "user-eve", "role": "user"}


@pytest.fixture(autouse=True)
def _no_register(monkeypatch):
    from captain_claw.flight_deck import tool_audit_routes
    monkeypatch.setattr(tool_audit_routes, "_data_dir", None)
    monkeypatch.setattr(tool_audit_routes, "_store", None)


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", s)
    s.set_settings({"enabled": True, "admins_only": False, "max_factories": 2})
    return s


def _live(store, title="Leads", owner=BOB["id"]):
    f = store.create(owner, title=title, charter="Collect public leads with a reason they fit.")
    return store.approve(f["id"], 1, f["draft"]["sha"], owner)


def _done_run(store, fid, status="done"):
    run = store.start_run(fid, trigger="manual")
    return store.finish_run(run["id"], status)


def test_an_archived_factory_is_off_the_list_and_comes_back_paused(store):
    a, b = _live(store, "A"), _live(store, "B")
    assert a["state"] == "running"
    with pytest.raises(fm.FirstMateError, match="archive one first"):
        store.create(BOB["id"], title="C", charter="Something to do every day.")  # at most 2
    got = store.archive(a["id"], "owner")
    assert got["archived_at"] and got["state"] == "paused"
    assert [f["id"] for f in store.list_factories(BOB["id"])] == [b["id"]]
    assert [f["id"] for f in store.list_factories(BOB["id"], archived=True)] == [a["id"]]
    assert {f["id"] for f in store.list_factories(BOB["id"], archived=None)} == {a["id"], b["id"]}
    store.create(BOB["id"], title="C", charter="Something to do every day.")  # A no longer counts
    with pytest.raises(fm.FirstMateError, match="restore it first"):
        store.set_state(a["id"], "running", "owner")
    with pytest.raises(fm.FirstMateError, match="restore it first"):
        store.start_run(a["id"], trigger="manual")
    with pytest.raises(fm.FirstMateError, match="archive one first"):
        store.archive(a["id"], "owner", on=False)  # B and C fill the list now
    store.archive(b["id"], "owner")
    back = store.archive(a["id"], "owner", on=False)
    assert not back["archived_at"] and back["state"] == "paused"
    events = [e["event"] for e in store.events(a["id"], limit=10)]
    assert "archived" in events and "restored" in events


def test_not_while_a_run_is_going_and_deleting_an_archived_one(store):
    f = _live(store)
    run = store.start_run(f["id"], trigger="manual")
    with pytest.raises(fm.FirstMateError, match="a run of it is going"):
        store.archive(f["id"], "owner")
    with pytest.raises(fm.FirstMateError, match="archive the factory before deleting it"):
        store.delete(f["id"], BOB["id"])
    store.finish_run(run["id"], "done")
    store.archive(f["id"], "owner")
    assert store.delete(f["id"], BOB["id"]) is True
    assert store.get(f["id"]) is None


def test_finished_runs_can_be_archived(store):
    f = _live(store)
    one, two = _done_run(store, f["id"]), _done_run(store, f["id"], "failed")
    going = store.start_run(f["id"], trigger="manual")
    with pytest.raises(fm.FirstMateError, match="only a finished run"):
        store.archive_run(going["id"])
    assert store.archive_run(one["id"])["archived_at"]
    assert [r["id"] for r in store.list_runs(f["id"], archived=False)] == [going["id"], two["id"]]
    assert [r["id"] for r in store.list_runs(f["id"], archived=True)] == [one["id"]]
    assert len(store.list_runs(f["id"])) == 3  # scores and reports still count them all
    assert store.archive_finished(f["id"]) == 1 and store.archived_runs(f["id"]) == 2
    assert not store.archive_run(one["id"], on=False)["archived_at"]


async def test_the_routes(store):
    f = _live(store)
    run = _done_run(store, f["id"])
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_current_user] = lambda: BOB
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://deck.test") as client:
        got = await client.post(f"/fd/first-mate/runs/{run['id']}/archive", json={"archived": True})
        assert got.status_code == 200 and got.json()["archived_at"]
        detail = (await client.get(f"/fd/first-mate/factories/{f['id']}")).json()
        assert detail["runs"] == [] and detail["archived_runs"] == 1
        shown = (await client.get(f"/fd/first-mate/factories/{f['id']}/runs", params={"archived": True})).json()
        assert [r["id"] for r in shown["runs"]] == [run["id"]]
        assert (await client.post(f"/fd/first-mate/runs/{run['id']}/archive", json={"archived": "yes"})).status_code == 400
        assert (await client.post(f"/fd/first-mate/factories/{f['id']}/archive", json={})).json()["archived_at"]
        listed = (await client.get("/fd/first-mate/factories")).json()
        assert listed["factories"] == [] and listed["archived_count"] == 1
        archived = (await client.get("/fd/first-mate/factories", params={"archived": True})).json()["factories"]
        assert [x["id"] for x in archived] == [f["id"]]
        assert (await client.post(f"/fd/first-mate/factories/{f['id']}/archive", json={"archived": False})).json()["archived_at"] == ""
        again = _done_run(store, f["id"])
        out = (await client.post(f"/fd/first-mate/factories/{f['id']}/runs/archive-finished")).json()
        assert out == {"archived": 1, "archived_runs": 2} and again["id"]
    app.dependency_overrides[get_current_user] = lambda: EVE
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://deck.test") as client:
        assert (await client.post(f"/fd/first-mate/factories/{f['id']}/archive", json={})).status_code == 404
        assert (await client.post(f"/fd/first-mate/runs/{run['id']}/archive", json={})).status_code == 404
