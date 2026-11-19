"""What a factory's runs already checked, kept across runs.

Pinned here:

* the record keeps each item once, by key (a site by its domain); a firmer outcome
  (close → dropped → kept) replaces a softer one;
* a run skips kept items always, dropped ones while the charter asks for the same
  work, and never one that only came close;
* a starting run of a funnel or Bat factory gets them as extra/already-checked.csv
  and its brief says so; a test case doesn't;
* a finished run adds what it checked: a funnel run every item, a Bat run its final
  list; a test case adds nothing;
* the funnel doesn't look at them again, and never takes the file as a source;
* the owner can see how much it holds, and forget it.
"""

from __future__ import annotations

import pytest

from captain_claw.flight_deck import bat_routes, funnel
from captain_claw.flight_deck import first_mate_ledger as ledger
from captain_claw.flight_deck import first_mate_runs as runs
from captain_claw.flight_deck import first_mate_store as fm

from .test_first_mate_funnel import _Bat, _live  # noqa: F401 — fixtures below use them
from .test_first_mate_runs import BOB


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
    from captain_claw.flight_deck import bat_loop, vfs_routes
    fake = _Bat()

    async def kick():
        return 0

    monkeypatch.setattr(bat_loop, "_STORE", fake)
    monkeypatch.setattr(bat_loop, "kick", kick)
    monkeypatch.setattr(bat_loop, "_STOP_CHILDREN", None)
    monkeypatch.setattr(vfs_routes, "_user_root", lambda owner: tmp_path / "vfs" / owner)
    monkeypatch.setattr(runs, "_workers", lambda prefix: [])
    return fake


def _row(key, outcome, name="", group="a"):
    return {"key": key, "name": name or key, "group": group, "outcome": outcome, "reason": "r"}


def test_what_the_record_keeps_and_what_a_run_skips(store):
    f = _live(store, runner="funnel")
    fid = f["id"]
    assert ledger.record(fid, [_row("https://www.alpha.example/x", "kept"), _row("beta.example", "dropped"),
                               _row("gamma.example", "maybe"), _row("", "kept"), _row("x.example", "odd")],
                         run_id="r1", work="w1") == 3
    rows = ledger.read(fid)
    assert set(rows) == {"alpha.example", "beta.example", "gamma.example"}  # a site by its domain
    assert {r["key"] for r in ledger.to_skip(rows, "w1")} == {"alpha.example", "beta.example"}
    assert {r["key"] for r in ledger.to_skip(rows, "w2")} == {"alpha.example"}  # new work: dropped ones again
    # firmer replaces softer; softer never replaces firmer
    assert ledger.record(fid, [_row("gamma.example", "kept"), _row("alpha.example", "dropped")], run_id="r2", work="w1") == 1
    rows = ledger.read(fid)
    assert rows["gamma.example"]["outcome"] == "kept" and rows["alpha.example"]["outcome"] == "kept"
    # dropped again under new work: skipped under that work from now on
    ledger.record(fid, [_row("beta.example", "dropped")], run_id="r3", work="w2")
    assert {r["key"] for r in ledger.to_skip(ledger.read(fid), "w2")} == {"alpha.example", "beta.example", "gamma.example"}
    assert ledger.counts(fid) == {"items": 3, "kept": 2, "dropped": 1, "maybe": 0, "runs": 3,
                                  "last": ledger.read(fid)["beta.example"]["checked_at"]}
    snap, n = ledger.snapshot(fid, {"charter": "x"})
    assert n == 2 and "alpha.example" in snap.read_text()  # this "work" isn't w1/w2: kept ones only
    ledger.forget(fid)
    assert ledger.read(fid) == {} and ledger.snapshot(fid, {}) is None
    v1, v2 = f["live"], {**f["live"], "charter": "something else"}
    assert ledger.work_key(v1) == ledger.work_key(dict(v1)) != ledger.work_key(v2)


def test_what_finished_runs_give(tmp_path):
    folder = tmp_path / "run"
    for key, status in (("a.example", "kept"), ("b.example", "dropped"), ("c.example", "maybe")):
        funnel._append_csv(folder / funnel.ITEMS, funnel._ITEM_COLS, {"key": key, "name": key, "status": status})
    assert [(r["key"], r["outcome"]) for r in ledger.rows_from_funnel(folder)] == [
        ("a.example", "kept"), ("b.example", "dropped"), ("c.example", "maybe")]
    (folder / "leads.csv").write_text("Firm,Email,Website,Segment\nAlpha,a@alpha.example,https://www.alpha.example,x\n"
                                      "Beta,b@beta.example,,y\n")
    bat_run = {"config": {"contract": [{"check": {"type": "table", "path": "leads.csv", "unique": ["Email"],
                                                  "min_per": {"Segment": 2}}}]}}
    got = ledger.rows_from_list(folder, bat_run)
    assert [(r["key"], r["name"], r["group"], r["outcome"]) for r in got] == [
        ("https://www.alpha.example", "Alpha", "x", "kept")]  # by its site, as the funnel names it
    assert ledger.rows_from_list(folder, None)[0]["name"] == "Alpha"  # no checks: its main table


async def test_a_starting_run_gets_what_earlier_runs_checked(store, bat):
    f = _live(store, runner="funnel")
    ledger.record(f["id"], [_row("alpha.example", "kept"), _row("beta.example", "dropped")],
                  run_id="old", work=ledger.work_key(f["live"]))
    run = await runs.start(f["id"], BOB, watch=False)
    folder = bat_routes._run_folder({"id": run["child_id"], "owner_id": BOB})
    given = (folder / "extra" / ledger.RUN_FILE).read_text()
    assert "alpha.example" in given and "beta.example" in given
    assert "earlier runs already checked 2 items" in bat.runs[run["child_id"]]["task"]
    assert funnel.known_keys(folder) == {"alpha.example", "beta.example"}
    # a test case starts clean
    store.finish_run(run["id"], "done")
    check = await runs.start(f["id"], BOB, kind="check", watch=False)
    folder2 = bat_routes._run_folder({"id": check["child_id"], "owner_id": BOB})
    assert not (folder2 / "extra" / ledger.RUN_FILE).exists()
    assert "already checked" not in bat.runs[check["child_id"]]["task"]


async def test_a_finished_run_adds_what_it_checked(store, bat):
    f = _live(store, runner="funnel")
    run = await runs.start(f["id"], BOB, watch=False)
    folder = bat_routes._run_folder({"id": run["child_id"], "owner_id": BOB})
    for key, status in (("a.example", "kept"), ("b.example", "dropped")):
        funnel._append_csv(folder / funnel.ITEMS, funnel._ITEM_COLS, {"key": key, "name": key, "status": status})
    bat.runs[run["child_id"]]["status"] = "done"
    await runs.refresh(run["id"])
    assert {k: r["outcome"] for k, r in ledger.read(f["id"]).items()} == {"a.example": "kept", "b.example": "dropped"}
    assert ledger.read(f["id"])["a.example"]["run"] == run["id"]
    # a Bat factory's run adds its final list
    g = _live(store, runner="bat", title="Bat leads")
    brun = await runs.start(g["id"], BOB, watch=False)
    bfolder = bat_routes._run_folder({"id": brun["child_id"], "owner_id": BOB})
    bfolder.mkdir(parents=True, exist_ok=True)
    (bfolder / "leads.csv").write_text("Firm,Website\nGamma,https://gamma.example\n")
    bat.runs[brun["child_id"]]["status"] = "done"
    await runs.refresh(brun["id"])
    assert list(ledger.read(g["id"])) == ["gamma.example"]
    # a test case adds nothing
    check = await runs.start(g["id"], BOB, kind="check", watch=False)
    cfolder = bat_routes._run_folder({"id": check["child_id"], "owner_id": BOB})
    cfolder.mkdir(parents=True, exist_ok=True)
    (cfolder / "leads.csv").write_text("Firm,Website\nDelta,https://delta.example\n")
    bat.runs[check["child_id"]]["status"] = "done"
    await runs.refresh(check["id"])
    assert list(ledger.read(g["id"])) == ["gamma.example"]


def test_the_funnel_never_takes_the_record_as_a_source(tmp_path):
    folder = tmp_path / "run"
    (folder / "extra").mkdir(parents=True)
    (folder / "extra" / funnel.LEDGER_FILE).write_text("key,name\nalpha.example,Alpha\n")
    base = funnel.from_contract([{"check": {"type": "table", "path": "out.csv", "columns": ["Name"]}}])
    assert funnel.default_recipe(folder, base) is None  # the only file is the record
    (folder / "extra" / "people.csv").write_text("Name\nAna\n")
    assert [s["path"] for s in funnel.default_recipe(folder, base)["sources"]] == ["extra/people.csv"]
    assert funnel._source({"kind": "file", "path": f"extra/{funnel.LEDGER_FILE}"}, {}) is None
    assert funnel.LEDGER_FILE not in funnel._folder_listing(folder)
