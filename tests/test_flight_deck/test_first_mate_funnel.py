"""First Mate runs a factory with the list funnel.

Pinned here:

* a version's runner may be ``funnel``, with its settings checked (an agent to
  finish what's short or none, a search budget, the agents' context) and nothing
  else;
* a funnel factory's run is a Bat run in the funnel's engine, carrying its
  settings; everything else (caps, results, timing) is read as for any Bat run;
* its progress carries the funnel's counts, per group and stage, for the run panel.
"""

from __future__ import annotations

import json

import pytest

from captain_claw.flight_deck import bat_loop, bat_routes, funnel
from captain_claw.flight_deck import first_mate_results as results
from captain_claw.flight_deck import first_mate_runs as runs
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck.first_mate_store import FirstMateError

from .test_first_mate_runs import BOB, FakeBat


class _Bat(FakeBat):
    """The Bat store calls a run's progress reads too."""

    async def last_event_i(self, run_id):
        return -1

    async def list_events(self, run_id, since=0, limit=300):
        return []

    async def list_steps(self, run_id):
        return []


@pytest.fixture(autouse=True)
def _no_register(monkeypatch):
    from captain_claw.flight_deck import tool_audit_routes
    monkeypatch.setattr(tool_audit_routes, "_data_dir", None)
    monkeypatch.setattr(tool_audit_routes, "_store", None)


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", s)
    return s


@pytest.fixture
def bat(monkeypatch, tmp_path):
    fake = _Bat()

    async def kick():
        return 0

    monkeypatch.setattr(bat_loop, "_STORE", fake)
    monkeypatch.setattr(bat_loop, "kick", kick)
    monkeypatch.setattr(bat_loop, "_STOP_CHILDREN", None)
    from captain_claw.flight_deck import vfs_routes
    monkeypatch.setattr(vfs_routes, "_user_root", lambda owner: tmp_path / "vfs" / owner)
    monkeypatch.setattr(runs, "_workers", lambda prefix: [])
    return fake


def _live(store, **kw):
    kw.setdefault("title", "Small firm leads")
    kw.setdefault("charter", "Find 30 small firms, 10 per segment, with an email on their own site.")
    f = store.create(BOB, **kw)
    return store.approve(f["id"], 1, f["draft"]["sha"], BOB)


def test_a_version_may_run_on_the_funnel(store):
    f = _live(store, runner="funnel", runner_config={"leftovers": "none", "search_budget": 400})
    assert f["live"]["runner"] == "funnel"
    assert f["live"]["runner_config"] == {"leftovers": "none", "loosen": "short", "search_budget": 400}
    assert fm.check_runner_config("funnel", {}) == {"leftovers": "agent", "loosen": "short"}
    assert fm.check_runner_config("funnel", {"loosen": "never"})["loosen"] == "never"
    for bad, why in (({"leftovers": "maybe"}, "leftovers"), ({"loosen": "always"}, "loosen"),
                     ({"search_budget": -1}, "search_budget"),
                     ({"search_budget": True}, "search_budget"), ({"agent_context": 1000}, "agent_context"),
                     ({"steps": ["x"]}, "steps")):
        with pytest.raises(FirstMateError, match=why):
            fm.check_runner_config("funnel", bad)
    assert "funnel" in fm.RUNNERS and "funnel" in fm.BAT_RUNNERS and "funnel" in runs.RUNNERS


async def test_a_funnel_factory_runs_as_a_bat_run_in_the_funnels_engine(store, bat):
    f = _live(store, runner="funnel", runner_config={"leftovers": "none", "search_budget": 400, "agent_context": 48000})
    run = await runs.start(f["id"], BOB, watch=False)
    assert run["runner"] == "funnel" and run["status"] == "running"
    cfg = bat.runs[run["child_id"]]["config"]
    assert cfg["engine"] == "funnel" and cfg["funnel_leftovers"] == "none"
    assert cfg["funnel_search_budget"] == 400 and cfg["funnel_agent_context"] == 48000
    assert cfg["source"] == "first_mate" and cfg["factory_id"] == f["id"]  # a factory run as any other
    assert cfg["funnel_loosen"] == "short"  # rows on a looser reading when a group runs short, unless set to never
    assert runs.funnel_config({"runner_config": {}}) == {"engine": "funnel", "funnel_leftovers": "agent",
                                                         "funnel_loosen": "short"}
    assert runs.funnel_config({"runner_config": {"loosen": "never"}})["funnel_loosen"] == "never"


async def test_its_progress_carries_the_funnels_counts(store, bat):
    f = _live(store, runner="funnel")
    run = await runs.start(f["id"], BOB, watch=False)
    got = await runs.progress(store.get_run(run["id"]))
    assert got["available"] and got["funnel"] is None  # no recipe yet
    folder = bat_routes._run_folder({"id": run["child_id"], "owner_id": BOB})
    recipe = funnel.normalize_recipe({"columns": ["Firm", "Segment"], "group_by": "Segment",
                                      "groups": {"a": 10, "b": 10}, "sources": [{"kind": "file", "path": "x.csv"}]})
    (folder / "shared" / "funnel").mkdir(parents=True)
    (folder / funnel.RECIPE).write_text(json.dumps(recipe))
    for key, group, status, stage in (("x.example", "a", "kept", "kept"), ("y.example", "a", "dropped", "read"),
                                      ("z.example", "b", "maybe", "maybe")):
        funnel._append_csv(folder / funnel.ITEMS, funnel._ITEM_COLS,
                           {"key": key, "group": group, "status": status, "stage": stage})
    funnel._append_csv(folder / funnel.ROWS, ["Firm", "Segment", "_group"], {"Firm": "X", "Segment": "a", "_group": "a"})
    got = await runs.progress(store.get_run(run["id"]))
    a, b = got["funnel"]["groups"]["a"], got["funnel"]["groups"]["b"]
    assert (a["looked"], a["kept"], a["read"], a["on_list"], a["quota"]) == (2, 1, 1, 1, 10)
    assert (b["looked"], b["maybe"], b["on_list"]) == (1, 1, 0)
    # once a note writes them, those are what's read
    funnel.write_counts(folder, recipe)
    assert funnel.live_counts(folder)["groups"]["a"]["kept"] == 1


async def test_results_and_timing_read_a_funnel_run_as_a_bat_run(store, bat):
    f = _live(store, runner="funnel")
    run = await runs.start(f["id"], BOB, watch=False)
    folder = bat_routes._run_folder({"id": run["child_id"], "owner_id": BOB})
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "leads.csv").write_text("Firm\nX\n")
    got = await results.run_results(store.get_run(run["id"]))
    assert got is not None and any(x["path"] == "leads.csv" for x in got["main"])
    assert await results.run_timing(store.get_run(run["id"])) is not None


def test_a_factorys_recipe_is_pinned_per_charter(store, tmp_path, monkeypatch):
    """Every run wrote a new recipe: a rule there or not, a filter that dropped every
    item. One recipe per charter now, the owner's to edit; a new charter, a new one."""
    from captain_claw.flight_deck import first_mate_data
    from captain_claw.flight_deck import first_mate_recipe as fr
    monkeypatch.setattr(first_mate_data, "root", lambda: tmp_path / "fm-data")
    f = _live(store, runner="funnel")
    fid, v1 = f["id"], store.live(f["id"])
    run = {"id": "bat_x", "config": {"factory_id": fid, "fm_version": v1["version"]}}
    recipe = {"item": "firm", "output": "firms.csv", "columns": ["Firm", "Email", "Rating"], "key": "Firm",
              "group_by": "", "groups": {}, "site_column": "", "sources": [{"kind": "search", "queries": ["x"]}],
              "filters": [{"column": "Firm", "op": "regex", "value": "x"}],
              "checks": [{"column": "Firm", "op": "required"}, {"column": "Rating", "op": "min", "value": 4}],
              "rules": [{"rule": "rated 4 or more", "kind": "must", "fact": "measured", "columns": ["Rating"]}]}
    assert fr.for_run(run, tmp_path) == (None, "") and fr.view(fid, v1)["recipe"] is None
    assert fr.pin_for_run(run, recipe) and not fr.pin_for_run(run, {**recipe, "item": "other"})  # the first one stays
    got, how = fr.for_run(run, tmp_path)
    assert got["item"] == "firm" and how.startswith("the factory's pinned recipe, built ")
    # a recipe written for a file this run wasn't given isn't used
    fr.pin(fid, v1, {**recipe, "sources": [{"kind": "file", "path": "extra/firms.csv"}]})
    assert fr.for_run(run, tmp_path) == (None, "")
    fr.pin(fid, v1, recipe)
    # the owner's edit: a rule's alternative, the columns a row must fill, no filters
    fr.edit(fid, v1, {"rules": [{"rule": "rated 4 or more", "kind": "must", "if_not_shown": "judged good",
                                 "columns": ["Rating", "Nope"]}, {"rule": "  "}],
                      "required": ["Firm", "Email", "Nope"], "filters": []})
    seen = fr.view(fid, v1)
    assert seen["edited"] and seen["recipe"]["required"] == ["Firm", "Email"] and seen["recipe"]["filters"] == []
    rule = seen["recipe"]["rules"][0]
    assert rule["if_not_shown"] == "judged good" and rule["columns"] == ["Rating"] and rule["fact"] == "measured"
    assert {"column": "Rating", "op": "min", "value": 4} in fr.pinned(fid, v1)["recipe"]["checks"]  # kept as built
    assert fr.for_run(run, tmp_path)[1].endswith("edited by hand")
    # the charter changes: stale, the next run builds a new one
    d = store.new_draft(fid, BOB, charter="Find 30 small firms with a phone number on their own site.")
    store.approve(fid, d["draft"]["version"], d["draft"]["sha"], BOB)
    v2 = store.live(fid)
    assert fr.pinned(fid, v2) is None and fr.view(fid, v2) == {"recipe": None, "stale": True}
    assert fr.forget(fid) and fr.view(fid, v2) == {"recipe": None, "stale": False}
