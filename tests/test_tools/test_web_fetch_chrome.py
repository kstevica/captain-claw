"""web_fetch reads a page in the owner's own Chrome (tools/web_chrome.py,
flight_deck/browser_fetch.py).

Pinned here: with *only pages that turn Flight Deck away*, a refusal (LinkedIn's
999), a sign-in wall or a bot check is read again in the person's Chrome and
the agent gets that page, marked as read there; an ordinary page never asks;
with *every page* the person's Chrome reads it first; off, Chrome closed, a
member's turn, ``via_chrome: false`` or no Flight Deck: as before.
"""

from __future__ import annotations

import json

import httpx
import pytest

from captain_claw import fd_client, speaker
from captain_claw.config import get_config, set_config
from captain_claw.tools import web_chrome
from captain_claw.tools.web_fetch import WebFetchTool

DECK = "http://127.0.0.1:9"
SIGNED_IN = ("<html><head><title>Jane Doe | LinkedIn</title></head><body><main><h1>Jane Doe</h1>"
             "<p>Senior engineer at an example company, working on distributed systems.</p></main></body></html>")
WALL = ("<html><head><title>Sign Up | LinkedIn</title></head><body><h1>Join LinkedIn</h1>"
        "<p>Sign in to see who you already know.</p></body></html>")
PLAIN = ("<html><head><title>Docs</title></head><body><main><h1>Asyncio</h1><p>"
         + "Tasks and timeouts explained in plain words. " * 30 + "</p></main></body></html>")


@pytest.fixture
def setup(monkeypatch):
    old = get_config().model_copy(deep=True)
    cfg = old.model_copy(deep=True)
    cfg.tools.web_fetch.save_images = False
    set_config(cfg)
    monkeypatch.setattr(web_chrome, "_MODE", {})
    monkeypatch.setattr(fd_client, "flight_deck_base", lambda: DECK)
    monkeypatch.setattr(fd_client, "flight_deck_headers", lambda: {"X-Agent-Auth": "tok"})
    state = {"deck": [], "mode": "blocked", "site": []}

    def deck(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        state["deck"].append(body)
        assert str(request.url) == f"{DECK}/fd/browser-hands/agent/fetch"
        mode = state["mode"]
        if not mode:
            return httpx.Response(200, json={"ok": False, "why": "off", "mode": ""})
        if body["when"] == "first" and mode != "always":
            return httpx.Response(200, json={"ok": False, "why": "not_first", "mode": mode})
        if state.get("closed"):
            return httpx.Response(200, json={"ok": False, "why": "no_browser", "mode": mode})
        return httpx.Response(200, json={"ok": True, "mode": mode, "browser": "Chrome on macOS",
                                         "url": body["url"], "title": "Jane Doe | LinkedIn", "html": SIGNED_IN})

    real = httpx.AsyncClient
    monkeypatch.setattr(web_chrome.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(deck), **kw))

    def site(status: int, html: str):
        def handler(request: httpx.Request) -> httpx.Response:
            state["site"].append(str(request.url))
            return httpx.Response(status, html=html, headers={"content-type": "text/html"})
        return real(transport=httpx.MockTransport(handler))

    tool = WebFetchTool()
    yield tool, state, site
    set_config(old)


async def _fetch(tool, url="https://www.linkedin.com/in/someone"):
    return await tool.execute(url=url, deep_fetch=False)


async def test_a_refusal_is_read_again_in_the_owners_chrome(setup):
    tool, state, site = setup
    tool.client = site(999, "<html><body>Request denied</body></html>")
    got = await _fetch(tool)
    assert got.success, got.error
    assert "[Read in the person's own Chrome (Chrome on macOS), signed in as them]" in got.content
    assert "[Mode: chrome]" in got.content and "Senior engineer" in got.content
    assert [b["when"] for b in state["deck"]] == ["first", "blocked"]
    assert "refuses automated visitors" in state["deck"][1]["why"]


async def test_a_sign_in_wall_is_read_again_and_an_ordinary_page_never_asks(setup):
    tool, state, site = setup
    tool.client = site(200, WALL)
    got = await _fetch(tool)
    assert got.success and "Jane Doe" in got.content and "[Mode: chrome]" in got.content
    assert "sign-in page" in state["deck"][-1]["why"]
    state["deck"].clear()
    tool.client = site(200, PLAIN)
    got = await _fetch(tool, "https://docs.example.org/asyncio")
    assert got.success and "[Mode: text]" in got.content and "Tasks and timeouts" in got.content
    assert state["deck"] == []   # "blocked" was heard a moment ago: no "first" call either


async def test_every_page_reads_in_chrome_first(setup):
    tool, state, site = setup
    state["mode"] = "always"
    tool.client = site(200, PLAIN)
    got = await _fetch(tool, "https://docs.example.org/asyncio")
    assert got.success and "[Mode: chrome]" in got.content and state["site"] == []


async def test_off_or_closed_chrome_leaves_web_fetch_as_it_was(setup):
    tool, state, site = setup
    state["mode"] = ""
    tool.client = site(200, WALL)
    got = await _fetch(tool)
    assert got.success and "[Mode: text]" in got.content and "Join LinkedIn" in got.content
    await _fetch(tool)
    assert len(state["deck"]) == 1   # off is remembered for a while
    web_chrome._MODE.clear()
    state.update(mode="blocked", closed=True, deck=[])
    tool.client = site(999, "<html><body>Request denied</body></html>")
    got = await _fetch(tool)
    assert not got.success and "999" in got.error


async def test_never_for_a_members_turn_an_opted_out_agent_or_without_flight_deck(setup, monkeypatch):
    tool, state, site = setup
    tool.client = site(200, WALL)
    monkeypatch.setattr(speaker, "member_bound", lambda: True)
    assert await web_chrome.page_from_chrome("https://www.linkedin.com/in/x", "blocked", "wall") is None
    monkeypatch.setattr(speaker, "member_bound", lambda: False)
    cfg = get_config().model_copy(deep=True)
    cfg.tools.web_fetch.via_chrome = False
    set_config(cfg)
    assert (await _fetch(tool)).success
    cfg.tools.web_fetch.via_chrome = True
    set_config(cfg)
    monkeypatch.setattr(fd_client, "flight_deck_base", lambda: "")
    assert (await _fetch(tool)).success
    assert state["deck"] == []


def test_the_big_sites_sign_in_walls_count_at_any_length():
    from captain_claw.tools.web_blocked import login_wall

    banner = "LinkedIn respects your privacy. Select Accept to consent or Reject to decline. " * 200
    assert login_wall("Sign Up | LinkedIn", banner)          # the wall a server gets for a profile
    assert login_wall("Log into Facebook", banner)
    assert not login_wall("Jane Doe - Engineer | LinkedIn", banner)   # the profile itself
    assert not login_wall("Login flows explained", banner)            # a long page about signing in
    assert not login_wall("Signing in with SSO", "short")


def test_web_fetch_waits_long_enough_for_the_owners_chrome():
    # A deep fetch, then the page read again in the person's Chrome (up to ~48 s).
    assert WebFetchTool.timeout_seconds >= 100 > web_chrome.WAIT_S + 25
