"""Browser hands — an agent's browser tool working in the person's own Chrome
(browser_hands.py).

Pinned here:

* only a browser's own token ("Browser · …") may lend a browser — not an MCP
  or a Mac token, not nothing, and not once it's revoked;
* an agent's act goes to ITS OWNER's browser, named after the agent, and the
  browser's answer comes back (a screenshot too); nobody connected, a stopped
  agent, a browser that goes away or takes too long each get a plain answer;
* files the browser fetched for an act land in the owner's Web clips folder
  and are listed in the act's answer; nobody else can post to that act;
* Connections → Browser shows which browsers agents can use now.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import socket
from pathlib import Path

import httpx
import pytest
import uvicorn
import websockets
from fastapi import FastAPI, HTTPException

from captain_claw.flight_deck import agent_identity, agent_names
from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import browser_clip as bc
from captain_claw.flight_deck import browser_hands as bh
from captain_claw.flight_deck.agent_identity import AgentCaller
from captain_claw.flight_deck.auth import create_access_token, hash_token, new_pat, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch):
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    for uid, name in (("alice", "Alice"), ("bob", "Bob")):
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            f" VALUES ('{uid}', '{uid}@x.co', 'x', '{name}', 'user', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')")
    await db._db.commit()
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setattr(bh, "_CONNS", {})
    monkeypatch.setattr(bh, "_ACTS", {})

    def _root(user_id, project=bc.CLIP_PROJECT):
        root = tmp_path / "vfs" / user_id / project
        root.mkdir(parents=True, exist_ok=True)
        return root

    monkeypatch.setattr(bh, "_clip_root", _root)

    # Which agent is calling: X-Agent-Auth "<owner>:<slug>" (the real check resolves FD's records).
    def _caller(request, *, what):
        if agent_identity.is_browser_request(request):
            raise HTTPException(403, f"{what} is for Flight Deck agents, not browsers")
        raw = request.headers.get("X-Agent-Auth", "")
        owner, _, slug = raw.partition(":")
        if not slug:
            raise HTTPException(403, "Unknown agent")
        return AgentCaller(slug=slug, owner=owner)

    monkeypatch.setattr(agent_identity, "require_agent_caller", _caller)

    async def _name(slug, viewer_id=""):
        return {"scout": "Scout", "ledger": "Ledger"}.get(slug, slug)

    monkeypatch.setattr(agent_names, "display_name", _name)
    app = FastAPI()
    app.include_router(bc.router)
    app.include_router(bc.owner_router)
    app.include_router(bh.router)
    app.include_router(bh.agent_router)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    task = asyncio.create_task(server.serve())
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.02)
    base = f"http://127.0.0.1:{port}"
    client = httpx.AsyncClient(base_url=base, timeout=20)

    async def browser_pat(uid="alice", name="Browser · Chrome on macOS") -> str:
        raw = new_pat()
        await db.create_pat(uid, hash_token(raw), name)
        return raw

    try:
        yield {"client": client, "db": db, "ws": f"ws://127.0.0.1:{port}/fd/clip/hands/ws", "pat": browser_pat,
               "clips": tmp_path / "vfs" / "alice" / "web-clips",
               "jwt": {"Authorization": f"Bearer {create_access_token('alice')}"}}
    finally:
        await client.aclose()
        server.should_exit = True
        await task
        await db.close()
        set_auth_db(prev)


class FakeBrowser:
    """The extension's side of the socket: says hello, answers acts as told."""

    def __init__(self, url: str, token: str, *, stopped=(), answer=None):
        self.url, self.token, self.stopped = url, token, list(stopped)
        self.answer = answer or (lambda act: {"ok": True, "content": f"did {act['action']}"})
        self.acts: list[dict] = []
        self.welcome: dict | None = None
        self.ws = None
        self.task: asyncio.Task | None = None

    async def __aenter__(self):
        self.ws = await websockets.connect(self.url)
        await self.ws.send(json.dumps({"type": "hello", "token": self.token, "browser": "Chrome on macOS",
                                       "version": "0.2.0", "stopped": self.stopped}))
        self.welcome = json.loads(await self.ws.recv())
        self.task = asyncio.create_task(self._serve())
        return self

    async def _serve(self):
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                if msg.get("type") == "act":
                    self.acts.append(msg)
                    got = self.answer(msg)
                    if asyncio.iscoroutine(got):
                        got = await got
                    if got is not None:
                        await self.ws.send(json.dumps({"type": "result", "id": msg["id"], **got}))
        except websockets.ConnectionClosed:
            pass

    async def send(self, msg: dict):
        await self.ws.send(json.dumps(msg))

    async def __aexit__(self, *exc):
        await self.ws.close()
        if self.task:
            await asyncio.wait([self.task], timeout=2)
        await asyncio.sleep(0.05)


def _agent(owner="alice", slug="scout") -> dict:
    return {"X-Agent-Auth": f"{owner}:{slug}"}


async def _act(c, action="screenshot", args=None, *, owner="alice", slug="scout", wait=5):
    r = await c.post("/fd/browser-hands/agent/act", headers=_agent(owner, slug),
                     json={"action": action, "args": args or {}, "wait": wait})
    assert r.status_code == 200, r.text
    return r.json()


# ── who may lend a browser ────────────────────────────────────────────


async def test_only_a_browsers_own_token_may_lend_the_browser(deck):
    raw_mcp = new_pat()
    await deck["db"].create_pat("alice", hash_token(raw_mcp), "Claude Code")
    raw_mac = await deck["pat"](name="Desktop · Captain for Mac")
    for token in ("", "cc_pat_nope", raw_mcp, raw_mac, create_access_token("alice")):
        async with websockets.connect(deck["ws"]) as ws:
            await ws.send(json.dumps({"type": "hello", "token": token}))
            first = json.loads(await ws.recv())
            assert first["type"] == "denied"
            with pytest.raises(websockets.ConnectionClosed) as closed:
                await ws.recv()
            assert closed.value.rcvd.code == 4401
    # Saying nothing (or something else first) gets the socket closed too.
    async with websockets.connect(deck["ws"]) as ws:
        await ws.send(json.dumps({"type": "ping"}))
        with pytest.raises(websockets.ConnectionClosed) as closed:
            await ws.recv()
        assert closed.value.rcvd.code == 4400
    assert bh._CONNS == {}


# ── acts ──────────────────────────────────────────────────────────────


async def test_an_agents_act_runs_in_its_owners_browser_and_comes_back(deck):
    c = deck["client"]
    pic = base64.b64encode(b"\xff\xd8jpeg").decode()
    answers = {"screenshot": {"ok": True, "content": "Screenshot of tab 7", "image": pic, "image_type": "image/jpeg",
                              "data": {"width": 1280, "height": 720}},
               "click": {"ok": False, "error": "ref_9 isn't on the page any more"}}
    status = (await c.get("/fd/browser-hands/agent/status", headers=_agent())).json()
    assert status == {"connected": False, "browsers": [], "stopped": False}
    async with FakeBrowser(deck["ws"], await deck["pat"](), answer=lambda a: answers[a["action"]]) as fake:
        assert fake.welcome["type"] == "welcome" and fake.welcome["user"] == "Alice"
        status = (await c.get("/fd/browser-hands/agent/status", headers=_agent())).json()
        assert status["connected"] and status["browsers"][0]["browser"] == "Chrome on macOS"
        got = await _act(c, "screenshot", {"tab_id": 7})
        assert got["ok"] and got["content"] == "Screenshot of tab 7" and got["image"] == pic
        assert got["data"] == {"width": 1280, "height": 720} and got["browser"] == "Chrome on macOS"
        act = fake.acts[-1]
        assert act["agent"] == {"slug": "scout", "name": "Scout"} and act["args"] == {"tab_id": 7}
        assert act["id"].startswith("ha_") and act["action"] == "screenshot"
        failed = await _act(c, "click", {"ref": "ref_9"})
        assert failed == {"ok": False, "error": "ref_9 isn't on the page any more", "browser": "Chrome on macOS"}
        # Another person's agent never reaches Alice's browser.
        bobs = await _act(c, owner="bob")
        assert bobs["ok"] is False and bobs["no_browser"] and "Agents can use this browser" in bobs["error"]
        assert len(fake.acts) == 2
    assert bh._ACTS == {}


async def test_nobody_connected_a_stopped_agent_and_a_slow_or_vanished_browser(deck):
    c = deck["client"]
    none = await _act(c)
    assert none["no_browser"] and "Captain for Chrome" in none["error"]

    async def never(act):
        return None   # the browser never answers

    async with FakeBrowser(deck["ws"], await deck["pat"](), stopped=["ledger"], answer=never) as fake:
        stopped = await _act(c, slug="ledger")
        assert stopped["stopped"] and "stopped you" in stopped["error"]
        slow = await _act(c, "navigate", {"url": "https://x.com"}, wait=1)
        assert slow["ok"] is False and slow["timeout"] and "1 s" in slow["error"]
        # The person stops Scout from the panel too.
        await fake.send({"type": "stopped", "agents": ["scout", "ledger"]})
        await asyncio.sleep(0.1)
        assert (await _act(c))["stopped"]
        await fake.send({"type": "stopped", "agents": []})
        await asyncio.sleep(0.1)
        pending = asyncio.create_task(_act(c, wait=10))
        while not fake.acts or fake.acts[-1]["action"] != "screenshot":
            await asyncio.sleep(0.02)
    gone = await pending
    assert gone["ok"] is False and "disconnected" in gone["error"]
    assert bh._CONNS == {} and bh._ACTS == {}


async def test_a_revoked_browser_is_cut_off(deck):
    c = deck["client"]
    raw = await deck["pat"]()
    async with FakeBrowser(deck["ws"], raw) as fake:
        listed = (await c.get("/fd/browsers", headers=deck["jwt"])).json()["browsers"]
        assert listed[0]["agents_can_use"] is True
        await c.delete(f"/fd/browsers/{listed[0]['id']}", headers=deck["jwt"])
        await asyncio.sleep(0.1)
        assert (await _act(c))["no_browser"]
        assert fake.ws.close_code == 4401 or fake.ws.state.name in ("CLOSING", "CLOSED")
    listed = (await c.get("/fd/browsers", headers=deck["jwt"])).json()["browsers"]
    assert listed == []


async def test_the_same_browser_reconnecting_replaces_its_old_socket(deck):
    raw = await deck["pat"]()
    async with FakeBrowser(deck["ws"], raw) as first:
        async with FakeBrowser(deck["ws"], raw) as second:
            assert len(bh.connections("alice")) == 1
            got = await _act(deck["client"])
            assert got["ok"] and second.acts and not first.acts


# ── files ─────────────────────────────────────────────────────────────


async def test_files_the_browser_fetched_land_in_web_clips_and_are_listed(deck):
    c = deck["client"]
    raw = await deck["pat"]()
    bob_raw = await deck["pat"]("bob")
    seen: dict = {}

    async def fetch_then_answer(act):
        seen["id"] = act["id"]
        auth = {"Authorization": f"Bearer {bob_raw}"}
        files = [("file", ("../../invoice.pdf", b"%PDF-1.4 one", "application/pdf")),
                 ("file", ("invoice.pdf", b"%PDF-1.4 two", "application/pdf"))]
        meta = {"meta": json.dumps({"page_url": "https://portal.bank.hr/docs", "page_title": "Documents"})}
        seen["bob"] = (await c.post(f"/fd/clip/hands/files/{act['id']}", headers=auth, files=files, data=meta)).status_code
        r = await c.post(f"/fd/clip/hands/files/{act['id']}", headers={"Authorization": f"Bearer {raw}"},
                         files=files, data=meta)
        seen["saved"] = r.json()
        return {"ok": True, "content": "Downloaded 2 files"}

    async with FakeBrowser(deck["ws"], raw, answer=fetch_then_answer):
        got = await _act(c, "download", {"all": True}, wait=10)
    assert seen["bob"] == 404, "another person can't add files to Alice's act"
    assert got["ok"] and got["content"] == "Downloaded 2 files"
    folder = got["folder"]
    assert folder.startswith("vfs:web-clips/") and "portal-bank-hr-scout-browser" in folder
    assert [f["name"] for f in got["files"]] == ["invoice.pdf", "invoice (2).pdf"]
    assert got["files"][0]["path"] == f"{folder}invoice.pdf"
    on_disk = deck["clips"] / folder.removeprefix("vfs:web-clips/")
    assert (on_disk / "invoice.pdf").read_bytes() == b"%PDF-1.4 one"
    assert (on_disk / "invoice (2).pdf").read_bytes() == b"%PDF-1.4 two"
    # After the act, its id takes nothing more.
    assert len(seen["saved"]["saved"]) == 2
    late = await c.post(f"/fd/clip/hands/files/{seen['id']}",
                        headers={"Authorization": f"Bearer {raw}"}, files=[("file", ("x.txt", b"x"))])
    assert late.status_code == 404


async def test_the_agent_route_refuses_browsers(deck):
    r = await deck["client"].post("/fd/browser-hands/agent/act", headers={**_agent(), "Origin": "https://evil.example"},
                                  json={"action": "screenshot"})
    assert r.status_code == 403


def test_answers_are_bounded_and_keep_known_keys_only():
    out = bh._clean_result({"ok": True, "content": "x" * (bh.MAX_CONTENT_CHARS + 50), "image": "a" * 10,
                            "image_type": "image/svg+xml", "evil": "<script>", "data": {"w": 1}})
    assert set(out) == {"ok", "content", "image", "image_type", "data"}
    assert len(out["content"]) <= bh.MAX_CONTENT_CHARS and out["image_type"] == "image/jpeg"
    assert bh._clean_result({"ok": False}) == {"ok": False, "error": "The browser couldn't do that."}


# ── the extension's side ──────────────────────────────────────────────

EXTENSION = Path(__file__).resolve().parents[2] / "browser-extension" / "chrome"


def test_the_extension_asks_for_the_debugger_the_way_chrome_allows():
    m = json.loads((EXTENSION / "manifest.json").read_text())
    assert {"debugger", "tabGroups", "alarms"} <= set(m["permissions"])
    assert "debugger" not in m.get("optional_permissions", []), \
        "Chrome drops debugger as optional, and the worker then failed to register"
    # Nothing the worker runs when it loads may assume a permission's API is there.
    for name in ("background.js", "hands.js"):
        for n, line in enumerate((EXTENSION / name).read_text().splitlines(), 1):
            assert not re.match(r"chrome\.(debugger|tabGroups|permissions)\.", line), f"{name}:{n}: {line}"
