"""Renaming an agent (``POST /fd/processes/{slug}/identity`` with ``introduce``).

Pinned here:

* the new name goes in the registry, spaces tidied, and only the owner can
  rename (someone else's agent is a 404, its name untouched);
* ``introduce``: the agent goes by the name too — one that's off or asleep
  gets it in its personality file (the rest of its personality kept), one
  that runs is told through its own PUT /api/personality;
* a running agent that can't be told still has its file written, and the
  answer says what happened; without ``introduce`` its personality is left
  alone.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import httpx
import pytest
from aiohttp import web

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import server
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.personality import parse_personality_markdown

BOB = "user-bob"
CAROL = "user-carol"

_PERSONALITY = """# Name

Captain

# Description

Reads the market for Bob.

# Background

Started as a research helper.

# Expertise

- Market research
- Competitor tables
"""


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    for uid in (BOB, CAROL):
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, 'user', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", uid))
    await db._db.commit()
    data = tmp_path / "fd-data"
    data.mkdir()
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", data / ".processes.json")
    monkeypatch.setattr(server, "_processes", {})
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    server._save_process_registry({
        "deep-researcher-ze90": {"slug": "deep-researcher-ze90", "name": "deep-researcher-ze90", "web_port": 0,
                                 "web_auth": "bob-tok", "owner": BOB, "pid": None},
        "carol-agent": {"slug": "carol-agent", "name": "Carol's agent", "web_port": 0, "web_auth": "", "owner": CAROL,
                        "pid": None},
    })
    home = server._personality_file("deep-researcher-ze90")
    home.parent.mkdir(parents=True)
    home.write_text(_PERSONALITY, "utf-8")
    try:
        yield db
    finally:
        await db.close()
        fd_auth._db = prev


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, client=("203.0.113.9", 40001)),
                             base_url="http://fd.test")


def _hdr(uid: str) -> dict:
    return {"Authorization": f"Bearer {create_access_token(uid)}"}


async def _rename(uid: str, slug: str, **body) -> httpx.Response:
    async with _client() as c:
        return await c.post(f"/fd/processes/{slug}/identity", headers=_hdr(uid), json=body)


def _personality(slug: str):
    return parse_personality_markdown(server._personality_file(slug).read_text("utf-8"))


async def test_an_agent_that_is_off_goes_by_its_new_name_from_its_next_start(deck):
    r = await _rename(BOB, "deep-researcher-ze90", name="  Ana   the  analyst ", introduce=True)
    assert r.status_code == 200, r.text
    assert r.json()["introduced"] == "next_start" and "from its next start" in r.json()["message"]
    assert server._load_process_registry()["deep-researcher-ze90"]["name"] == "Ana the analyst"
    p = _personality("deep-researcher-ze90")
    assert p.name == "Ana the analyst"
    # Nothing else of who it is changed.
    assert p.description == "Reads the market for Bob." and p.expertise == ["Market research", "Competitor tables"]


async def test_without_introduce_only_the_label_changes(deck):
    r = await _rename(BOB, "deep-researcher-ze90", name="Ana", introduce=False)
    assert r.status_code == 200 and r.json()["message"] == "Saved." and r.json()["introduced"] == ""
    assert server._load_process_registry()["deep-researcher-ze90"]["name"] == "Ana"
    assert _personality("deep-researcher-ze90").name == "Captain"


async def test_an_agent_with_no_personality_yet_gets_one_with_its_name(deck):
    server._personality_file("deep-researcher-ze90").unlink()
    r = await _rename(BOB, "deep-researcher-ze90", name="Ana", introduce=True)
    assert r.status_code == 200
    assert _personality("deep-researcher-ze90").name == "Ana"


async def test_only_the_owner_renames(deck):
    r = await _rename(BOB, "carol-agent", name="Mine now", introduce=True)
    assert r.status_code == 404
    assert server._load_process_registry()["carol-agent"]["name"] == "Carol's agent"
    assert not server._personality_file("carol-agent").exists()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _agent(port: int, seen: list, status: int = 200) -> web.AppRunner:
    """The agent's own web server: PUT /api/personality."""
    async def put(request: web.Request) -> web.Response:
        seen.append({"token": request.query.get("token"), "body": await request.json()})
        return web.json_response({"name": "x"}, status=status)
    app = web.Application()
    app.router.add_put("/api/personality", put)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


async def test_a_running_agent_is_told_at_once(deck, monkeypatch):
    port = _free_port()
    reg = server._load_process_registry()
    reg["deep-researcher-ze90"]["web_port"] = port
    server._save_process_registry(reg)
    monkeypatch.setattr(server, "_process_is_alive", lambda slug: slug == "deep-researcher-ze90")
    seen: list = []
    runner = await _agent(port, seen)
    try:
        r = await _rename(BOB, "deep-researcher-ze90", name="Ana", introduce=True)
    finally:
        await runner.cleanup()
    assert r.status_code == 200 and r.json()["introduced"] == "now" and "calls itself Ana now" in r.json()["message"]
    assert seen == [{"token": "bob-tok", "body": {"name": "Ana"}}]
    # The agent writes its own file; the deck didn't.
    assert _personality("deep-researcher-ze90").name == "Captain"


async def test_a_running_agent_that_cant_be_told_gets_its_file(deck, monkeypatch):
    port = _free_port()
    reg = server._load_process_registry()
    reg["deep-researcher-ze90"]["web_port"] = port
    server._save_process_registry(reg)
    monkeypatch.setattr(server, "_process_is_alive", lambda slug: True)
    seen: list = []
    runner = await _agent(port, seen, status=500)
    try:
        r = await _rename(BOB, "deep-researcher-ze90", name="Ana", introduce=True)
    finally:
        await runner.cleanup()
    assert r.status_code == 200 and r.json()["introduced"] == "restart" and "after a restart" in r.json()["message"]
    assert len(seen) == 1 and _personality("deep-researcher-ze90").name == "Ana"


async def test_a_name_too_long_is_refused(deck):
    r = await _rename(BOB, "deep-researcher-ze90", name="x" * 201, introduce=True)
    assert r.status_code == 422
    assert json.loads(server.PROCESS_REGISTRY_FILE.read_text())["deep-researcher-ze90"]["name"] == "deep-researcher-ze90"
