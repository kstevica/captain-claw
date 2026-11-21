"""web_search in the owner's own Chrome first (flight_deck/browser_search.py).

Pinned here: under Flight Deck the tool asks the deck first and shows the
results it got from the person's browser; "off" is believed for a while (no
call per search); a robot check falls back to the configured engine with a
note; Chrome being closed falls back silently; a shared agent's member turn
and an agent with ``via_chrome: false`` never ask; without Flight Deck nothing
changes.
"""

from __future__ import annotations

import json

import httpx
import pytest

from captain_claw import fd_client, speaker
from captain_claw.config import get_config, set_config
from captain_claw.tools import web_search as ws
from captain_claw.tools.web_search import WebSearchTool, _Hit

DECK = "http://127.0.0.1:9"


@pytest.fixture
def tool(monkeypatch):
    old = get_config().model_copy(deep=True)
    cfg = old.model_copy(deep=True)
    cfg.tools.web_search.provider = "brave"
    cfg.tools.web_search.api_key = "k"
    set_config(cfg)
    monkeypatch.setattr(ws, "_CHROME_OFF", {})
    monkeypatch.setattr(fd_client, "flight_deck_base", lambda: DECK)
    monkeypatch.setattr(fd_client, "flight_deck_headers", lambda: {"X-Agent-Auth": "tok", "X-Agent-Slug": "scout"})
    t = WebSearchTool()
    t.deck_calls = []
    t.deck_answer = {"ok": False, "why": "off"}
    t.api_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        t.deck_calls.append(json.loads(request.content))
        assert str(request.url) == f"{DECK}/fd/browser-hands/agent/search"
        assert request.headers["X-Agent-Auth"] == "tok"
        return httpx.Response(200, json=t.deck_answer)

    t.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def brave(q, cfg, **kw):
        t.api_calls += 1
        return [_Hit(title="From Brave", url="https://brave.example/r", snippet="api result")], []

    t._brave = brave
    yield t
    set_config(old)


async def test_the_owners_chrome_answers_first(tool):
    tool.deck_answer = {"ok": True, "engine": "Google", "browser": "Chrome on macOS", "notes": [],
                        "results": [{"title": "Timeouts in asyncio", "url": "https://docs.example.org/a",
                                     "snippet": "Use asyncio.timeout().", "site": "Example Docs", "date": "May 5, 2024"},
                                    {"title": "Not a page", "url": "javascript:alert(1)", "snippet": ""}]}
    got = await tool.execute(query="asyncio gather timeout", count=5, search_lang="en")
    assert got.success
    assert "[SEARCH ENGINE: Google (in the person's Chrome)]" in got.content
    assert "Timeouts in asyncio" in got.content and "Source: Example Docs · May 5, 2024" in got.content
    assert "javascript:" not in got.content and "[RESULTS: 1]" in got.content
    assert tool.api_calls == 0
    assert tool.deck_calls == [{"query": "asyncio gather timeout", "count": 5, "page": 0, "search_lang": "en",
                                "country": "", "freshness": "", "safesearch": ""}]


async def test_off_is_believed_for_a_while(tool):
    tool.deck_answer = {"ok": False, "why": "off", "error": "off"}
    for _ in range(3):
        got = await tool.execute(query="x")
        assert got.success and "[SEARCH ENGINE: Brave]" in got.content and "NOTE" not in got.content
    assert len(tool.deck_calls) == 1 and tool.api_calls == 3


async def test_a_robot_check_falls_back_with_a_note_and_a_closed_chrome_silently(tool):
    tool.deck_answer = {"ok": False, "why": "blocked", "engine": "Google",
                        "error": "Google asked to confirm a person is searching (a robot check)."}
    got = await tool.execute(query="x")
    assert got.success and "[SEARCH ENGINE: Brave]" in got.content
    assert "[NOTE: Searched with Brave: Google asked to confirm a person is searching" in got.content
    tool.deck_answer = {"ok": False, "why": "no_browser", "error": "No Chrome of the person's is ready"}
    got = await tool.execute(query="y")
    assert got.success and "NOTE" not in got.content and tool.api_calls == 2 and len(tool.deck_calls) == 2


async def test_member_turns_and_opted_out_agents_never_ask(tool, monkeypatch):
    monkeypatch.setattr(speaker, "member_bound", lambda: True)
    assert (await tool.execute(query="x")).success
    monkeypatch.setattr(speaker, "member_bound", lambda: False)
    cfg = get_config().model_copy(deep=True)
    cfg.tools.web_search.via_chrome = False
    set_config(cfg)
    assert (await tool.execute(query="x")).success
    assert tool.deck_calls == [] and tool.api_calls == 2


async def test_without_flight_deck_nothing_changes(tool, monkeypatch):
    monkeypatch.setattr(fd_client, "flight_deck_base", lambda: "")
    assert (await tool.execute(query="x")).success
    assert tool.deck_calls == [] and tool.api_calls == 1


async def test_an_older_deck_is_left_alone_for_a_while(tool):
    def handler(request: httpx.Request) -> httpx.Response:
        tool.deck_calls.append(1)
        return httpx.Response(404, json={"detail": "Not Found"})

    tool.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    for _ in range(3):
        assert (await tool.execute(query="x")).success
    assert len(tool.deck_calls) == 1 and tool.api_calls == 3
