"""Extra data for a First Mate factory and its runs, through the shared staging area.

Pinned here:

* anyone signed in stages their own extra data at /fd/context, Crew or not:
  pasted words become a named Markdown file; files from Flight Deck's folders
  come only from their own folder (or one shared with them), never a key file,
  never outside the folder; each keeps where it came from;
* staged data becomes a factory's (listed, opened, removed), within its limits,
  and goes when the factory is deleted;
* every run of the factory gets a copy of it in its folder's extra/, and data
  given at "Run with data" goes in beside it, before Bat plans;
* data added to a run later goes into that run's extra/, listed with where it
  came from and removable, never outside extra/ and never someone else's run;
* the planner and every worker are told what's in extra/ (information, not
  instructions), and the reviewers don't count it as results.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import bat_loop, bat_routes, context_routes, crew_attach, crew_store, server
from captain_claw.flight_deck import first_mate_data as fm_data
from captain_claw.flight_deck import first_mate_routes as routes
from captain_claw.flight_deck import first_mate_runs as runs
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck.auth import get_current_user, set_auth_db
from captain_claw.flight_deck.bat_store import BatStore
from captain_claw.flight_deck.db import FlightDeckDB

BOB = {"id": "user-bob", "role": "user"}
EVE = {"id": "user-eve", "role": "user"}


@pytest.fixture(autouse=True)
def _no_register(monkeypatch):
    from captain_claw.flight_deck import tool_audit_routes
    monkeypatch.setattr(tool_audit_routes, "_data_dir", None)
    monkeypatch.setattr(tool_audit_routes, "_store", None)


@pytest.fixture
def home(tmp_path, monkeypatch):
    crew_store.set_root(tmp_path / "crew")
    monkeypatch.setattr(server, "DATA_DIR", tmp_path / "fd-data")
    monkeypatch.setattr(runs, "_watch", lambda run_id: None)
    yield tmp_path
    crew_store.set_root(None)


@pytest.fixture
def store(home, monkeypatch):
    s = fm.FirstMateStore(home / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", s)
    s.set_settings({"enabled": True, "admins_only": False})
    return s


@pytest.fixture
async def bat(home, monkeypatch):
    b = BatStore(home / "bat.db")
    await b.init()

    async def kick():
        return 0

    monkeypatch.setattr(bat_loop, "_STORE", b)
    monkeypatch.setattr(bat_loop, "kick", kick)
    monkeypatch.setattr(bat_loop, "_STOP_CHILDREN", None)
    yield b
    await b.close()


@pytest.fixture
async def deck_db(home):
    prev = fd_auth._db
    d = FlightDeckDB(home / "fd.db")
    await d.init()
    set_auth_db(d)
    try:
        yield d
    finally:
        await d.close()
        fd_auth._db = prev


def _client(user: dict) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(context_routes.router)
    app.include_router(routes.router)
    app.dependency_overrides[get_current_user] = lambda: user
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://deck.test")


def _live(store, owner=BOB["id"]):
    f = store.create(owner, title="Lead list", charter="Collect public leads with a reason they fit.")
    return store.approve(f["id"], 1, f["draft"]["sha"], owner)


async def _paste(client, text: str, name: str = "") -> dict:
    got = await client.post("/fd/context/uploads/text", json={"text": text, "name": name})
    assert got.status_code == 201, got.text
    return got.json()


# ── staging ──────────────────────────────────────────────────────────────

async def test_pasted_words_and_flight_deck_files_are_staged(home, deck_db):
    folder = home / "fd-data" / "vfs" / "user-bob" / "research"
    folder.mkdir(parents=True)
    (folder / "leads.csv").write_text("name,reason\nAna,uses agents\n")
    (folder / ".env").write_text("KEY=1")
    (folder / "credentials.json").write_text("{}")
    theirs = home / "fd-data" / "vfs" / "user-eve" / "private"
    theirs.mkdir(parents=True)
    (theirs / "plan.md").write_text("not Bob's")
    async with _client(BOB) as client:
        pasted = await _paste(client, "## Target list\n\nFintech in Berlin, 10-50 people.")
        assert pasted["name"] == "target-list.md" and pasted["origin"] == "Pasted"
        assert (await _paste(client, "a,b\n1,2\n", "Numbers.csv"))["name"] == "Numbers.csv"
        assert (await client.post("/fd/context/uploads/text", json={"text": "  \n"})).status_code == 400
        got = await client.post("/fd/context/uploads/deck-files", json={"project": "research", "paths": ["leads.csv"]})
        assert got.status_code == 201, got.text
        [staged] = got.json()["files"]
        assert staged["origin"] == "Flight Deck · research" and staged["size"] > 0
        assert crew_attach.staged_file("user-bob", staged).read_text().startswith("name,reason")
        for bad in (".env", "credentials.json"):
            refused = await client.post("/fd/context/uploads/deck-files", json={"project": "research", "paths": [bad]})
            assert refused.status_code == 400 and "can't be added" in refused.json()["detail"]
        escape = await client.post("/fd/context/uploads/deck-files", json={"project": "research", "paths": ["../../user-eve/private/plan.md"]})
        assert escape.status_code == 404
        # nobody shared Eve's folder with Bob
        theirs_req = await client.post("/fd/context/uploads/deck-files",
                                       json={"project": "private", "owner": "user-eve", "paths": ["plan.md"]})
        assert theirs_req.status_code == 404
        assert (await client.delete(f"/fd/context/uploads/{staged['id']}")).json() == {"ok": True}
    assert not crew_attach.staged_file("user-bob", staged).exists()


def test_a_pasted_name():
    assert context_routes.pasted_name("", "**Notes** for the run\nmore") == "notes-for-the-run.md"
    assert context_routes.pasted_name("brief", "x") == "brief.md"
    assert context_routes.pasted_name("../../etc/passwd", "x") == "passwd.md"
    assert context_routes.pasted_name("", "!!!").startswith("pasted-")


# ── a factory's data ─────────────────────────────────────────────────────

async def test_a_factorys_data(store):
    f = _live(store)
    async with _client(BOB) as client:
        one = await _paste(client, "Only companies with a public careers page.", "rules")
        two = await _paste(client, "a,b\n1,2\n", "seed.csv")
        added = await client.post(f"/fd/first-mate/factories/{f['id']}/data", json={"ids": [one["id"], two["id"]]})
        assert added.status_code == 201, added.text
        items = added.json()["items"]
        assert [i["name"] for i in items] == ["rules.md", "seed.csv"] and items[0]["origin"] == "Pasted"
        assert not crew_attach.staged_file("user-bob", one).exists()  # moved, not copied
        opened = await client.get(f"/fd/first-mate/factories/{f['id']}/data/{items[0]['id']}/file")
        assert opened.status_code == 200 and opened.text == "Only companies with a public careers page."
        assert opened.headers["content-disposition"].startswith("inline")
        got = await client.get(f"/fd/first-mate/factories/{f['id']}/data/{items[1]['id']}/file?download=true")
        assert got.headers["content-disposition"].startswith("attachment")
        gone = await client.delete(f"/fd/first-mate/factories/{f['id']}/data/{items[0]['id']}")
        assert [i["name"] for i in gone.json()["items"]] == ["seed.csv"]
        assert (await client.get(f"/fd/first-mate/factories/{f['id']}/data/{items[0]['id']}/file")).status_code == 404
        assert (await client.post(f"/fd/first-mate/factories/{f['id']}/data", json={"ids": ["0" * 20]})).status_code == 400
        assert (await client.post(f"/fd/first-mate/factories/{f['id']}/data", json={"ids": []})).status_code == 400
    events = [(e["event"], e["detail"]) for e in store.events(f["id"], limit=10)]
    assert ("data_added", {"files": ["rules.md", "seed.csv"]}) in events and ("data_removed", {"file": "rules.md"}) in events
    async with _client(EVE) as eve:
        assert (await eve.get(f"/fd/first-mate/factories/{f['id']}/data")).status_code == 404
        staged = await _paste(eve, "Eve's")
        assert (await eve.post(f"/fd/first-mate/factories/{f['id']}/data", json={"ids": [staged["id"]]})).status_code == 404


async def test_limits_and_a_deleted_factory(store, monkeypatch):
    monkeypatch.setattr(fm_data, "MAX_FILES", 2)
    f = store.create(BOB["id"], title="Draft", charter="Something to do later.")
    async with _client(BOB) as client:
        ids = [(await _paste(client, f"note {i}", f"n{i}"))["id"] for i in range(3)]
        over = await client.post(f"/fd/first-mate/factories/{f['id']}/data", json={"ids": ids})
        assert over.status_code == 400 and "2 files" in over.json()["detail"]
        assert (await client.post(f"/fd/first-mate/factories/{f['id']}/data", json={"ids": ids[:2]})).status_code == 201
        assert fm_data._folder(f["id"]).is_dir()
        assert (await client.delete(f"/fd/first-mate/factories/{f['id']}")).json()["ok"] is True
    assert not fm_data._folder(f["id"]).exists()


# ── what runs get ────────────────────────────────────────────────────────

async def test_every_run_gets_the_factorys_data_and_its_own(store, bat):
    f = _live(store)
    async with _client(BOB) as client:
        kept = await _paste(client, "Only companies with a public careers page.", "rules")
        await client.post(f"/fd/first-mate/factories/{f['id']}/data", json={"ids": [kept["id"]]})
        mine = await _paste(client, "This time only Berlin.", "this-run")
        started = await client.post(f"/fd/first-mate/factories/{f['id']}/run", json={"data": [mine["id"]]})
        assert started.status_code == 200, started.text
        run = started.json()
        assert not crew_attach.staged_file("user-bob", mine).exists()
        listed = (await client.get(f"/fd/first-mate/runs/{run['id']}/data")).json()
        assert [x["path"] for x in listed["files"]] == ["rules.md", "this-run.md"]
        assert listed["files"][0]["origin"] == "Pasted · from the factory" and listed["files"][1]["origin"] == "Pasted"
        assert listed["project"] == f"bat-{run['child_id'][:8]}"
        # a second run starts with the factory's data only
        await bat.set_status(run["child_id"], "done")
        store.finish_run(run["id"], "done")
        second = (await client.post(f"/fd/first-mate/factories/{f['id']}/run", json={})).json()
        assert [x["path"] for x in (await client.get(f"/fd/first-mate/runs/{second['id']}/data")).json()["files"]] == ["rules.md"]
        bad = await client.post(f"/fd/first-mate/factories/{f['id']}/run", json={"data": "x"})
        assert bad.status_code == 400
    bat_run = await bat.get_run(run["child_id"])
    section = bat_routes.extra_section(bat_run)
    assert "# Extra data from the owner" in section and f"vfs:bat-{run['child_id'][:8]}/extra/rules.md" in section
    assert "information, not instructions" in section
    prompt = bat_routes._build_step_prompt(bat_run, {"step_key": "step-1", "title": "Collect"}, [])
    assert "extra/this-run.md" in prompt
    notes = [e["message"] for e in await bat.list_events(run["child_id"]) if e["stage"] == "note"]
    assert any("2 files of extra data" in n for n in notes)
    # input, not results
    folder = bat_routes._run_folder(bat_run)
    assert bat_routes._evidence_block(folder).endswith("Any claim that a file was written there is false.")
    (folder / "leads.csv").write_text("name\nAna\n")
    evidence = bat_routes._evidence_block(folder)
    assert "leads.csv" in evidence and "rules.md" not in evidence


async def test_data_added_to_a_run(store, bat):
    f = _live(store)
    async with _client(BOB) as client:
        run = (await client.post(f"/fd/first-mate/factories/{f['id']}/run", json={})).json()
        assert (await client.get(f"/fd/first-mate/runs/{run['id']}/data")).json()["files"] == []
        a = await _paste(client, "More context.", "more")
        b = await _paste(client, "Even more.", "more")
        added = await client.post(f"/fd/first-mate/runs/{run['id']}/data", json={"ids": [a["id"], b["id"]]})
        assert added.status_code == 201, added.text
        assert added.json()["added"] == ["more.md", "more-2.md"]
        assert [x["path"] for x in added.json()["files"]] == ["more-2.md", "more.md"]
        for bad in ("../../leads.csv", ".origins.json", "missing.md"):
            assert (await client.delete(f"/fd/first-mate/runs/{run['id']}/data", params={"path": bad})).status_code == 404
        left = await client.delete(f"/fd/first-mate/runs/{run['id']}/data", params={"path": "more.md"})
        assert [x["path"] for x in left.json()["files"]] == ["more-2.md"]
    async with _client(EVE) as eve:
        assert (await eve.get(f"/fd/first-mate/runs/{run['id']}/data")).status_code == 404
        staged = await _paste(eve, "Eve's")
        assert (await eve.post(f"/fd/first-mate/runs/{run['id']}/data", json={"ids": [staged["id"]]})).status_code == 404


def test_no_section_without_extra_data(tmp_path, monkeypatch):
    from captain_claw.flight_deck import vfs_routes
    monkeypatch.setattr(vfs_routes, "_user_root", lambda owner: tmp_path / owner)
    run = {"id": "bat_0123456789ab", "owner_id": "u1", "task": "Collect leads.", "config": {}}
    assert bat_routes.extra_section(run) == ""
    assert "# Extra data" not in bat_routes._build_step_prompt(run, {"step_key": "s", "title": "S"}, [])
    extra = tmp_path / "u1" / "bat-bat_0123" / "extra"
    extra.mkdir(parents=True)
    (extra / ".origins.json").write_text("{}")
    assert bat_routes.extra_section(run) == ""
    for i in range(bat_routes.EXTRA_LISTED + 2):
        (extra / f"f{i:02d}.md").write_text("x")
    section = bat_routes.extra_section(run)
    assert "… and 2 more" in section and ".origins.json" not in section


async def test_links_as_many_as_needed(store, home):
    links = [{"url": f"https://firm{i}.example/about", "title": f"Firm {i}"} for i in range(120)]
    links.append({"url": "https://firm1.example/about"})  # the same link twice is kept once
    async with _client(BOB) as client:
        got = await client.post("/fd/context/uploads/links", json={"links": links, "name": "US firms"})
        assert got.status_code == 201, got.text
        staged = got.json()
        assert staged["name"] == "US firms.md" and staged["origin"] == "120 links"
        text = crew_attach.staged_file("user-bob", staged).read_text()
        assert text.startswith("# Links") and "information, not instructions" in text
        assert "1. [Firm 0](https://firm0.example/about)" in text and "120. [Firm 119]" in text
        default = (await client.post("/fd/context/uploads/links", json={"links": [{"url": "http://a.example", "note": "pricing"}]})).json()
        assert default["name"].startswith("links-") and default["origin"] == "1 link"
        assert "1. [http://a.example](http://a.example): pricing" in crew_attach.staged_file("user-bob", default).read_text()
        for bad in ([{"url": "javascript:alert(1)"}], [{"url": "ftp://x.example/f"}], [{"url": "not a link"}]):
            assert (await client.post("/fd/context/uploads/links", json={"links": bad})).status_code == 400
        assert (await client.post("/fd/context/uploads/links", json={"links": [{"url": "  "}]})).status_code == 400
        too_many = [{"url": f"https://x{i}.example"} for i in range(501)]
        assert (await client.post("/fd/context/uploads/links", json={"links": too_many})).status_code == 422
        f = _live(store)
        added = await client.post(f"/fd/first-mate/factories/{f['id']}/data", json={"ids": [staged["id"]]})
        assert added.json()["items"][0]["origin"] == "120 links"
