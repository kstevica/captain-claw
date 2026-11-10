"""An agent's own Chrome: Captain for Chrome paired in its headless browser.

Live, end to end (skipped without Playwright's Chromium): the agent's browser
tool asks Flight Deck to connect (the person says yes in its dialog), Captain
for Chrome in the agent's own headless Chromium gets its connection and
connects; then browser(target="chrome") from that agent runs there — navigate,
read_page refs, click, page_text, a screenshot. A browser restart puts the
connection back; Disconnect in Connections → Browser ends it.
"""

from __future__ import annotations

import asyncio
import socket
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, HTTPException

from captain_claw import fd_client
from captain_claw.config import BrowserToolConfig
from captain_claw.flight_deck import agent_identity, agent_names, origin_guard
from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import browser_clip as bc
from captain_claw.flight_deck import browser_hands as bh
from captain_claw.flight_deck import person_asks as asks
from captain_claw.flight_deck.agent_identity import AgentCaller
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.tools import browser as browser_mod
from captain_claw.tools import browser_extensions as bx
from captain_claw.tools.browser import BrowserTool


def _port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            return Path(p.chromium.executable_path).exists()
    except Exception:
        return False


def _ref(text: str, needle: str) -> str:
    for line in text.splitlines():
        if needle in line and "[ref_" in line:
            return line.split("[", 1)[1].split("]", 1)[0]
    raise AssertionError(f"no ref for {needle!r} in:\n{text[:2000]}")


@pytest.mark.skipif(not _chromium_available(), reason="Playwright's Chromium isn't installed")
async def test_live_an_agent_pairs_and_uses_its_own_chrome(tmp_path, monkeypatch):
    # Flight Deck: the clip and hands routes, CORS as the deck has it, a person.
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    await db._db.execute("INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
                         " VALUES ('alice', 'a@x.co', 'x', 'Alice', 'user', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')")
    await db._db.commit()
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setattr(bh, "_CONNS", {})
    monkeypatch.setattr(bh, "_ACTS", {})
    monkeypatch.setattr(bh, "_MINTED", {})
    monkeypatch.setattr(bh, "_agent_store_path", lambda: tmp_path / "agent-browsers.json")
    monkeypatch.setattr(bh, "_clip_root", lambda user_id, project="web-clips": (tmp_path / "vfs" / project))
    asks.reset_for_tests()

    def _caller(request, *, what):
        if agent_identity.is_browser_request(request):
            raise HTTPException(403, "not for browsers")
        owner, _, slug = request.headers.get("X-Agent-Auth", "").partition(":")
        return AgentCaller(slug=slug, owner=owner)

    monkeypatch.setattr(agent_identity, "require_agent_caller", _caller)

    async def _name(slug, viewer_id=""):
        return {"scout": "Scout"}.get(slug, slug)

    monkeypatch.setattr(agent_names, "display_name", _name)
    deck = FastAPI()
    deck.add_middleware(origin_guard.FDCORSMiddleware, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    for r in (bc.router, bc.owner_router, bh.router, bh.agent_router):
        deck.include_router(r)
    deck_port = _port()
    server = uvicorn.Server(uvicorn.Config(deck, host="127.0.0.1", port=deck_port, log_level="warning", lifespan="off"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
    deck_url = f"http://127.0.0.1:{deck_port}"

    # A site for the agent to work on.
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text(
        "<!doctype html><title>Own Portal</title><h1>Hello from the portal</h1>"
        "<button onclick=\"document.getElementById('out').textContent='Clicked it'\">Press me</button><p id=out></p>")
    site_port = _port()
    pages = ThreadingHTTPServer(("127.0.0.1", site_port), partial(SimpleHTTPRequestHandler, directory=str(site)))
    threading.Thread(target=pages.serve_forever, daemon=True).start()

    # The agent: under that deck, as Scout of Alice.
    monkeypatch.setattr(fd_client, "flight_deck_base", lambda: deck_url)
    monkeypatch.setattr(fd_client, "flight_deck_headers", lambda: {"X-Agent-Auth": "alice:scout"})
    cfg = BrowserToolConfig(extension_policy="ask", default_wait_seconds=0.1)
    monkeypatch.setattr(browser_mod, "get_config", lambda: SimpleNamespace(tools=SimpleNamespace(browser=cfg)))
    monkeypatch.setattr(browser_mod, "_TARGETS", {})
    ws = tmp_path / "ws"
    ws.mkdir()
    kw = {"_runtime_base_path": str(ws), "_session_id": f"own-{tmp_path.name}"}
    tool = BrowserTool()

    async def approve():   # the person says yes in Flight Deck's dialog
        while True:
            for a in asks.open_for("alice"):
                asks.answer(a["id"], "alice", answers={"connect": True})
            await asyncio.sleep(0.05)

    approver = asyncio.create_task(approve())
    session = None
    try:
        async def run():
            nonlocal session
            r = await tool.execute("extension_connect", **kw)
            assert r.success, r.error
            assert "Installed Captain for Chrome" in r.content and "Connected" in r.content
            session = browser_mod._BROWSER_SESSIONS[kw["_session_id"]]
            store = bx.ExtensionStore.for_tool(kw)
            assert store.captain_conn()["url"] == deck_url and bh.agent_browser_rows("alice")[0]["slug"] == "scout"
            [conn] = bh.connections("alice")
            assert conn.agent_slug == "scout"

            async def chrome(action, **args):
                res = await tool.execute(action, target="chrome", **args, **kw)
                assert res.success, f"{action}: {res.error}"
                return res

            nav = await chrome("navigate", url=f"http://127.0.0.1:{site_port}/index.html")
            assert "Own Portal" in nav.content and "your own headless browser" in nav.content
            status = await chrome("status")
            assert "your own headless browser" in status.content
            tree = (await chrome("read_page")).content
            await chrome("click", ref=_ref(tree, "Press me"))
            text = (await chrome("page_text")).content
            assert "Hello from the portal" in text and "Clicked it" in text
            shot = await chrome("screenshot")
            assert "Path:" in shot.content

            # A restart (new profile) gets the connection back.
            await session.restart()
            for _ in range(100):
                if bh.connections("alice"):
                    break
                await asyncio.sleep(0.1)
            assert bh.connections("alice"), "it reconnects after a restart"
            assert "Own Portal" in (await chrome("navigate", url=f"http://127.0.0.1:{site_port}/index.html")).content

            # Disconnect in Connections → Browser ends it.
            jwt = {"Authorization": f"Bearer {create_access_token('alice')}"}
            async with httpx.AsyncClient(base_url=deck_url) as c:
                listed = (await c.get("/fd/browsers", headers=jwt)).json()["browsers"]
                [own] = [b for b in listed if b["kind"] == "agent"]
                assert own["agents_can_use"] is True
                await c.delete(f"/fd/browsers/{own['id']}", headers=jwt)
            await asyncio.sleep(0.3)
            gone = await tool.execute("navigate", target="chrome", url=f"http://127.0.0.1:{site_port}/", **kw)
            assert not gone.success and "extension_connect" in gone.error

        await asyncio.wait_for(run(), 180)
    finally:
        approver.cancel()
        if session is not None:
            await session.close()
        browser_mod._BROWSER_SESSIONS.pop(kw["_session_id"], None)
        pages.shutdown()
        server.should_exit = True
        await serving
        await db.close()
        set_auth_db(prev)
