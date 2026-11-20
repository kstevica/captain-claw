"""A factory's results, the result first; and only what the person picks of a
shared folder goes in the File library.

Pinned here:

* a shared folder is in the File library only once something of it is added:
  a single file (only that file is listed and summarised) or the whole
  folder; taking it out removes it at once, the file stays; linked, mounted,
  shared-with-me and Flight Deck's own folders, hidden files and kinds the
  library can't read are refused; a folder that's gone takes its picks along;
* the Data library still holds every folder's tables;
* a run's files are sorted so the result comes first: what its done-checks
  name, else files called a result, else the newest at the top of the folder;
  the batches beside it are "other"; subfolders, scripts and temporary files
  are "work"; a table says how many rows it has, a file whether it's in the
  library;
* the factory's results route lists its runs that left files, newest first,
  and the factory list carries each one's latest result.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from captain_claw.flight_deck import bat_loop, bat_routes, server
from captain_claw.flight_deck import file_library as fl
from captain_claw.flight_deck import first_mate_results as results
from captain_claw.flight_deck import first_mate_routes as routes
from captain_claw.flight_deck import first_mate_runs as runs
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.flight_deck.bat_store import BatStore

BOB = {"id": "user-bob", "role": "user"}


@pytest.fixture(autouse=True)
def _no_register(monkeypatch):
    from captain_claw.flight_deck import tool_audit_routes
    monkeypatch.setattr(tool_audit_routes, "_data_dir", None)
    monkeypatch.setattr(tool_audit_routes, "_store", None)


@pytest.fixture
def home(tmp_path, monkeypatch):
    data = tmp_path / "fd-data"
    data.mkdir()
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", data / ".processes.json")
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setattr(fl, "_STORE", fl.LibraryStore(tmp_path / "file_library.db"))
    monkeypatch.setattr(runs, "_watch", lambda run_id: None)

    async def no_tier(*_a, **_kw):
        return None

    async def no_media(owner, tier):
        return fl.MediaReach(owner=owner, tier=tier)

    monkeypatch.setattr(fl, "_resolve_tier", no_tier)
    monkeypatch.setattr(fl, "media_reach", no_media)
    server._save_process_registry({})
    return data


def _vfs(home: Path, folder: str) -> Path:
    p = home / "vfs" / BOB["id"] / folder
    p.mkdir(parents=True, exist_ok=True)
    return p


async def _listed(user=BOB) -> list[str]:
    await fl.run_scan(user, fl.ScanJob())
    sources = await fl.caller_sources(user)
    return sorted(f"{fl.folder_of(r['agent_slug'])[1]}/{r['rel_path']}" for r in fl.get_store().search(list(sources), []))


# ── the File library: only what's picked ────────────────────────────────

async def test_only_what_is_picked_is_in_the_library(home):
    run = _vfs(home, "bat-bat_aafd")
    (run / "final_leads.csv").write_text("name,reason\nAna,uses agents\n")
    (run / "leads_batch_01.csv").write_text("name\nAna\n")
    (run / "scripts").mkdir()
    (run / "scripts" / "fix.py").write_text("print(1)\n")
    reports = _vfs(home, "reports")
    (reports / "q3.md").write_text("# Q3\n")
    (reports / "sub").mkdir()
    (reports / "sub" / "q4.md").write_text("# Q4\n")
    assert await _listed() == []  # nothing added: no folder in the library
    assert fl.add_pick(BOB, "bat-bat_aafd", "final_leads.csv") is True
    assert fl.add_pick(BOB, "bat-bat_aafd", "final_leads.csv") is False  # already in
    assert await _listed() == ["bat-bat_aafd/final_leads.csv"]
    fl.add_pick(BOB, "reports")
    assert await _listed() == ["bat-bat_aafd/final_leads.csv", "reports/q3.md", "reports/sub/q4.md"]
    assert fl.drop_pick(BOB, "bat-bat_aafd", "final_leads.csv") is True
    sources = await fl.caller_sources(BOB)
    assert [r["rel_path"] for r in fl.get_store().search(list(sources), [])] in (["q3.md", "sub/q4.md"], ["sub/q4.md", "q3.md"])
    assert (run / "final_leads.csv").is_file()  # out of the library, still in its folder
    assert fl.picks_view(BOB) == [{"folder": "reports", "path": ""}]
    # the Data library still sees every folder
    assert {f["folder"] for f in fl.caller_folders(BOB, picked_only=False).values()} == {"bat-bat_aafd", "reports"}


async def test_what_cant_go_in(home):
    folder = _vfs(home, "reports")
    (folder / ".secret.md").write_text("x")
    (folder / "data.bin").write_bytes(b"\0")
    (folder / "ok.md").write_text("ok")
    outside = home.parent / "repo"
    outside.mkdir()
    root = home / "vfs" / BOB["id"]
    (root / ".vfs-links.json").write_text(json.dumps({"repo": {"path": str(outside), "mode": "ro"}}))
    (root / ".drive").mkdir()
    for folder_name, path in (("repo", ""), (".drive", ""), ("nope", ""), ("reports", "../reports/ok.md")):
        with pytest.raises(HTTPException) as err:
            fl.add_pick(BOB, folder_name, path)
        assert err.value.status_code in (400, 404)
    for path in (".secret.md", "data.bin", "missing.md"):
        with pytest.raises(HTTPException):
            fl.add_pick(BOB, "reports", path)
    assert fl.picks_view(BOB) == []


async def test_a_gone_folder_takes_its_picks(home):
    folder = _vfs(home, "old")
    (folder / "a.md").write_text("a")
    fl.add_pick(BOB, "old", "a.md")
    assert await _listed() == ["old/a.md"]
    (folder / "a.md").unlink()
    os.rmdir(folder)
    assert await _listed() == []
    assert fl.picks_view(BOB) == []


async def test_the_picks_routes(home):
    folder = _vfs(home, "reports")
    (folder / "q3.md").write_text("# Q3\n")
    app = FastAPI()
    app.include_router(fl.router)
    app.dependency_overrides[get_current_user] = lambda: BOB
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://deck.test") as client:
        got = await client.post("/fd/file-library/picks", json={"folder": "reports", "path": "q3.md"})
        assert got.status_code == 200 and got.json()["picks"] == [{"folder": "reports", "path": "q3.md"}]
        assert (await client.get("/fd/file-library/picks")).json()["picks"] == [{"folder": "reports", "path": "q3.md"}]
        refused = await client.post("/fd/file-library/picks", json={"folder": "reports", "path": "nope.md"})
        assert refused.status_code == 404
        gone = await client.delete("/fd/file-library/picks", params={"folder": "reports", "path": "q3.md"})
        assert gone.json() == {"dropped": True, "picks": []}


# ── a run's files, the result first ─────────────────────────────────────

def _found(folder: Path) -> list:
    return bat_routes._run_files(folder)


def test_a_runs_files_are_sorted_result_first(tmp_path):
    f = tmp_path / "run"
    (f / "scripts").mkdir(parents=True)
    (f / "progress").mkdir()
    (f / "extra").mkdir()
    for i, name in enumerate(["leads_batch_01.csv", "leads_batch_02.csv", "tmp_append_08.csv", "final_leads.csv"]):
        (f / name).write_text("name,reason\n" + "".join(f"Org {n},fits\n" for n in range(3 + i)))
        os.utime(f / name, (1000 + i, 1000 + i))
    (f / "scripts" / "validate.py").write_text("print(1)\n")
    (f / "progress" / "step-1.md").write_text("note")
    (f / "extra" / "rules.md").write_text("given")
    got = results.sort_files(_found(f), set(), {"final_leads.csv"})
    assert [x["name"] for x in got["main"]] == ["final_leads.csv"]
    assert got["main"][0]["rows"] == 6 and got["main"][0]["library"] is True
    assert [x["name"] for x in got["other"]] == ["leads_batch_02.csv", "leads_batch_01.csv"]
    assert {x["path"] for x in got["work"]} == {"tmp_append_08.csv", "scripts/validate.py"}
    assert got["total"] == 5  # progress/ and extra/ aren't the run's files
    # what the done-checks name comes first, whatever it's called
    named = results.sort_files(_found(f), {"leads_batch_01.csv"}, set())
    assert [x["name"] for x in named["main"]] == ["leads_batch_01.csv", "final_leads.csv"]
    # nothing called a result: the newest at the top of the folder
    (f / "final_leads.csv").unlink()
    plain = results.sort_files(_found(f), set(), set())
    assert [x["name"] for x in plain["main"]] == ["leads_batch_02.csv"]


def test_rows_of_a_table(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text('name,note\nA,"two\nlines"\nB,x\n\n')
    assert results.rows_of(p) == 2
    q = tmp_path / "t.md"
    q.write_text("x")
    assert results.rows_of(q) is None


async def test_the_factorys_results_and_its_card(home, tmp_path, monkeypatch):
    store = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", store)
    store.set_settings({"enabled": True, "admins_only": False})
    bat = BatStore(tmp_path / "bat.db")
    await bat.init()

    async def kick():
        return 0

    monkeypatch.setattr(bat_loop, "_STORE", bat)
    monkeypatch.setattr(bat_loop, "kick", kick)
    monkeypatch.setattr(bat_loop, "_STOP_CHILDREN", None)
    try:
        f = store.create(BOB["id"], title="Lead list", charter="Collect public leads with a reason they fit.")
        f = store.approve(f["id"], 1, f["draft"]["sha"], BOB["id"])
        run = await runs.start(f["id"], BOB["id"], watch=False)
        folder = bat_routes._run_folder({"id": run["child_id"], "owner_id": BOB["id"]})
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "leads.csv").write_text("name,reason\nAna,fits\nBo,fits\n")
        (folder / "notes.txt").write_text("x")
        cfg = (await bat.get_run(run["child_id"]))["config"]
        await bat.set_config(run["child_id"], {**cfg, "contract": [
            {"id": "a1", "text": "leads.csv exists", "check": {"type": "file_exists", "path": "leads.csv"}}]})
        app = FastAPI()
        app.include_router(routes.router)
        app.dependency_overrides[get_current_user] = lambda: BOB
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://deck.test") as client:
            got = (await client.get(f"/fd/first-mate/factories/{f['id']}/results")).json()["runs"]
            assert len(got) == 1 and got[0]["run_id"] == run["id"] and got[0]["project"] == f"bat-{run['child_id'][:8]}"
            assert [x["name"] for x in got[0]["main"]] == ["leads.csv"] and got[0]["main"][0]["rows"] == 2
            assert [x["name"] for x in got[0]["other"]] == ["notes.txt"]
            cards = (await client.get("/fd/first-mate/factories")).json()["factories"]
            latest = next(c for c in cards if c["id"] == f["id"])["latest_result"]
            assert latest["name"] == "leads.csv" and latest["rows"] == 2 and latest["files"] == 2
    finally:
        await bat.close()


# ── how long a run took, and saying so when it ends ─────────────────────

def _ev(ts, stage, message="", **data):
    return {"ts": ts, "stage": stage, "message": message, "data": data}


def test_timing_from_a_runs_events():
    events = [
        _ev(100, "phase", "running"),
        _ev(110, "step", "A", agent="a", attempt=1), _ev(110, "step", "B", agent="b", attempt=1),
        _ev(410, "step", "A", agent="a", ok=True),
        _ev(510, "step", "B", agent="b", ok=True),        # side by side: 400 s of work, 700 s of agent time
        _ev(520, "step", "C", agent="c", attempt=1),
        _ev(820, "budget", "stopped: llm_token_cap_reached"),   # C cut off by the cap at 820
        _ev(4420, "note", "continued where it stopped, with no token cap"),  # an hour waiting for Continue
        _ev(4430, "step", "C", agent="c", attempt=1),
        _ev(4630, "note", "resumed: re-queued 1 orphaned step(s)"),  # a restart: C's turn ends at its last sign, 4430
        _ev(4640, "step", "C", agent="c", attempt=1),
        _ev(5240, "step", "C", agent="c", ok=True),
        _ev(5300, "done", "done"),
    ]
    got = results.timing_of(events, 100, 5400, now=9999)
    assert got["total_s"] == 5300
    assert got["working_s"] == 400 + 300 + 0 + 600  # A∪B, C before the cap, C after the restart
    assert got["agent_s"] == 300 + 400 + 300 + 0 + 600
    assert got["waiting_s"] == 3600 and got["turns"] == 5
    assert got["other_s"] == 5300 - 1300 - 3600
    # still stopped, nobody pressed Continue: it's waiting until now
    live = results.timing_of(events[:7], 100, None, now=1820)
    assert live["waiting_s"] == 1000 and live["total_s"] == 1720


def test_the_words():
    assert results.duration_text(59) == "59 s" and results.duration_text(3599) == "59 min"
    assert results.duration_text(3600 * 23 + 240) == "23 h 4 min" and results.duration_text(86400 * 3) == "3 d"
    line = results.timing_line({"total_s": 83040, "working_s": 33120, "agent_s": 33500, "waiting_s": 41400})
    assert line == "Took 23 h 4 min · its agents worked 9 h 12 min · waiting for you 11 h 30 min"
    assert "side by side" in results.timing_line({"total_s": 1000, "working_s": 400, "agent_s": 900, "waiting_s": 0})
    got = {"main": [{"name": "final_leads.csv", "rows": 500, "size": 234863}], "total": 28}
    assert results.results_line(got) == "Result: final_leads.csv · 500 rows · 235 KB (and 27 other files in its folder)"


def test_the_final_step_comes_first():
    steps = [{"step_key": "step-1", "seq": 0, "title": "Batch 1", "status": "done", "output": "52 leads in batch 1"},
             {"step_key": "step-2", "seq": 1, "title": "Batch 2", "status": "done", "output": "50 leads in batch 2"},
             {"step_key": "step-3", "seq": 2, "title": "Merge", "status": "done", "output": "500 leads in final_leads.csv"},
             {"step_key": "fix-1", "seq": 3, "title": "Fix", "status": "failed", "output": ""}]
    text = runs.final_first(steps, "## Batch 1\n\n52 leads in batch 1")
    assert text.startswith("## Merge\n\n500 leads in final_leads.csv")
    assert text.index("Earlier steps (2)") < text.index("52 leads in batch 1")
    assert runs.final_first(steps[:1], "only one") == "only one"
    objected = runs.final_first(steps, "> Reviewers still objected when this run ended: thin\n\n## Batch 1")
    assert objected.startswith("> Reviewers still objected")


async def test_a_finished_run_says_what_it_made_and_how_long(home, tmp_path, monkeypatch):
    from captain_claw.flight_deck import first_mate_line, line
    store = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", store)
    store.set_settings({"enabled": True, "admins_only": False})
    bat = BatStore(tmp_path / "bat.db")
    await bat.init()

    async def kick():
        return 0

    monkeypatch.setattr(bat_loop, "_STORE", bat)
    monkeypatch.setattr(bat_loop, "kick", kick)
    monkeypatch.setattr(bat_loop, "_STOP_CHILDREN", None)
    posted = []

    async def post(owner, kind, title, body, **kw):
        posted.append({"title": title, "body": body, **kw})

    monkeypatch.setattr(line, "post", post)
    try:
        f = store.create(BOB["id"], title="Lead list", charter="Collect public leads with a reason they fit.")
        f = store.approve(f["id"], 1, f["draft"]["sha"], BOB["id"])
        run = await runs.start(f["id"], BOB["id"], watch=False, kind="run")
        child = run["child_id"]
        folder = bat_routes._run_folder({"id": child, "owner_id": BOB["id"]})
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "final_leads.csv").write_text("name,reason\n" + "".join(f"Org {i},fits\n" for i in range(500)))
        (folder / "leads_batch_01.csv").write_text("name\nA\n")
        await bat.upsert_step(child, "step-1", title="Batch 1", status="done", seq=0, output="52 leads")
        await bat.upsert_step(child, "step-2", title="Merge", status="done", seq=1, output="500 leads merged")
        await bat.append_event(child, "step", "Batch 1", agent="step-1", attempt=1)
        await bat.append_event(child, "step", "Batch 1", agent="step-1", ok=True)
        await bat.set_status(child, "done")
        finished = await runs.refresh(run["id"])
        assert finished["status"] == "done" and finished["result"].startswith("## Merge")
        assert finished["timing"]["total_s"] >= 0 and "working_s" in finished["timing"]
        assert store.get_run(run["id"])["timing"] == finished["timing"]  # kept
        [msg] = [p for p in posted if p["ref_id"] == run["id"]]
        assert msg["title"] == "Lead list: done"
        assert msg["body"].startswith("Result: final_leads.csv · 500 rows")
        assert "and 1 other file in its folder" in msg["body"]
        assert msg["to_phones"] is True  # started by hand
        progress = await runs.progress(store.get_run(run["id"]))
        assert progress["timing"] == finished["timing"]
    finally:
        await bat.close()


def test_the_list_a_table_check_reads_is_the_result():
    """A list run's summary.md was its result and the list itself one of the "other"
    files: the table its done-checks read is what it was asked to deliver."""
    run = {"config": {"contract": [
        {"check": {"type": "table", "path": "./restorani.csv", "columns": ["Naziv"]}},
        {"check": {"type": "document", "path": "report.docx"}},
        {"check": {"type": "judge"}}]}}
    assert results.named_by_checks(run) == {"restorani.csv", "report.docx"}
