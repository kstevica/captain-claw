"""Sites that block the agent, and asking the person for their own Chrome (web_blocked.py).

Pinned here:

* a bot wall is told from a page: a challenge's title or marks on a SHORT page,
  a 401/403/429/999 — not a 404, a bare 503, a long article that mentions a
  captcha, or the scripts these vendors put on ordinary pages; browser errors
  that are how walls answer (HTTP/2 dropped) — not a missing site;
* the browser blocked → try web_fetch once; web_fetch blocked → try the browser
  once; BOTH → ask the person with ask_user, worded for what Flight Deck knows:
  their Chrome connected (may I?), Captain for Chrome set up but off (turn on
  "Agents can use this browser"), not set up (recommend installing it, and
  how), or they stopped this agent (don't ask); after that, a short reminder;
* nothing for two timeouts, a member's turn, or without Flight Deck;
* the tools carry it: browser navigate (a self-clearing check gets a moment),
  web_fetch, web_get, and target="chrome" finding no Chrome after a block.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from captain_claw import fd_client
from captain_claw.tools import browser as browser_mod
from captain_claw.tools import browser_chrome as bc
from captain_claw.tools import web_blocked as wb
from captain_claw.tools.browser import BrowserTool
from captain_claw.tools.web_fetch import WebFetchTool, WebGetTool

ORIGINAL_CHROME_STATUS = wb.chrome_status   # before the fixture stands in for it
CF_TITLE = "Just a moment..."
CF_HTML = ("<html><head><title>Just a moment...</title></head><body><div id='cf'>Checking if the site connection "
           "is secure. example.com needs to review the security of your connection before proceeding.</div>"
           "<script>window._cf_chl_opt={cvId: '3'};</script></body></html>")


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(wb, "_SEEN", {})
    monkeypatch.setattr(fd_client, "flight_deck_base", lambda: "http://fd.test")
    state: dict[str, Any] = {"status": {"connected": True, "person_connected": True, "extension_set_up": True,
                                        "own_connected": False, "stopped": False}, "asked": 0}

    async def _status():
        state["asked"] += 1
        return state["status"]

    monkeypatch.setattr(wb, "chrome_status", _status)
    return state


def _kw(sid: str = "s1", **extra) -> dict[str, Any]:
    return {"_session_id": sid, **extra}


# ── telling a wall from a page ───────────────────────────────────────


def test_bot_walls_are_told_from_pages():
    assert wb.block_reason(status=403, title=CF_TITLE).startswith("a bot check")
    assert "HTTP 403" in wb.block_reason(status=403, title=CF_TITLE)
    assert "Cloudflare" in wb.block_reason(status=200, text="Checking if the site connection is secure", html=CF_HTML)
    assert wb.block_reason(html=CF_HTML).startswith("a bot check (Cloudflare)")       # deep fetch: title from markup
    assert "PerimeterX" in wb.block_reason(text="Press & Hold to confirm you are a human (and not a bot).")
    assert "DataDome" in wb.block_reason(status=403, html="<iframe src='https://geo.captcha-delivery.com/captcha/'>")
    assert wb.block_reason(title="Attention Required! | Cloudflare") != ""
    assert wb.block_reason(title="Access to this page has been denied.") != ""
    assert wb.block_reason(status=403) == "it refuses automated visitors (HTTP 403)"
    assert wb.block_reason(status=401) == "it asks for a sign-in (HTTP 401)"
    assert wb.block_reason(status=999).endswith("(HTTP 999)")
    # Not walls: a missing page, a site that's down, a real page.
    assert wb.block_reason(status=404, title="Page not found") == ""
    assert wb.block_reason(status=503, title="Down for maintenance") == ""
    assert wb.block_reason(status=200, title="Forbidden Planet — comics & games") == ""
    article = "How CAPTCHA works: are you a robot? Sites ask you to verify you are human. " * 40
    assert wb.block_reason(status=200, title="CAPTCHA, explained", text=article) == ""
    ordinary = ("<html><head><title>Shop</title><script src='/_Incapsula_Resource?SWJIYLWA=1'></script>"
                "<script src='/cdn-cgi/challenge-platform/scripts/jsd/main.js'></script>"
                "<script src='https://js.datadome.co/tags.js'></script></head><body>Hi</body></html>")
    assert wb.block_reason(status=200, text="Welcome to the shop", html=ordinary) == ""


def test_browser_errors_that_are_walls():
    assert "ERR_HTTP2_PROTOCOL_ERROR" in wb.net_reason("Page.goto: net::ERR_HTTP2_PROTOCOL_ERROR at https://x.com/")
    assert wb.net_reason("net::ERR_CONNECTION_RESET") != ""
    assert wb.net_reason("Timeout 30000ms exceeded.") == "the page never finished loading"
    assert wb.net_reason("Page.goto: net::ERR_NAME_NOT_RESOLVED at https://nope.invalid/") == ""
    assert wb.net_reason("net::ERR_CERT_AUTHORITY_INVALID") == ""
    resp = httpx.Response(403, text="<title>Just a moment...</title>", request=httpx.Request("GET", "https://x.com"))
    why, strong = wb.fetch_error_reason(httpx.HTTPStatusError("403", request=resp.request, response=resp))
    assert strong and why.startswith("a bot check")
    gone = httpx.Response(404, request=httpx.Request("GET", "https://x.com"))
    assert wb.fetch_error_reason(httpx.HTTPStatusError("404", request=gone.request, response=gone)) == ("", False)
    assert wb.fetch_error_reason(httpx.ReadTimeout("slow"))[1] is False


# ── the flow: try the other way, then ask ────────────────────────────


async def test_blocked_both_ways_asks_the_person_for_their_connected_chrome(fresh):
    first = await wb.note_failure(_kw(), "https://www.shop.example/item/1", wb.BROWSER, "a bot check (Cloudflare)")
    assert "--- Blocked ---" in first and "web_fetch(url=\"https://www.shop.example/item/1\", deep_fetch=false)" in first
    assert "ask_user" not in first and fresh["asked"] == 0
    both = await wb.note_failure(_kw(), "https://shop.example/item/1", wb.FETCH, "it refuses automated visitors (HTTP 403)")
    assert "Blocked: ask to use their Chrome" in both and "shop.example blocks your headless browser" in both
    assert "ask_user(question=\"shop.example blocks my browser on the server. May I open it in your Chrome" in both
    assert "confirm=true" in both
    assert "target=\"chrome\"" in both and "don't ask again" in both and "extension_connect" in both
    # Once told, a short reminder — Flight Deck isn't asked again.
    again = await wb.note_failure(_kw(), "https://shop.example/item/2", wb.BROWSER, "a bot check")
    assert "still blocks you" in again and "ask_user" in again and fresh["asked"] == 1


async def test_fetch_first_then_the_browser(fresh):
    first = await wb.note_failure(_kw(), "https://news.example/a", wb.FETCH, "a bot check (DataDome)")
    assert "Try the browser once: browser(action=\"navigate\", url=\"https://news.example/a\")" in first
    both = await wb.note_failure(_kw(), "https://news.example/a", wb.BROWSER, "the site dropped the connection")
    assert "Blocked: ask to use their Chrome" in both


@pytest.mark.parametrize(("status", "expect", "absent"), [
    ({"connected": False, "person_connected": False, "extension_set_up": True},
     ["isn't open to agents right now", "turn on “Agents can use this browser”", "choices=[\"It's on — go ahead\""],
     ["confirm=true"]),
    ({"connected": False, "person_connected": False, "extension_set_up": False},
     ["don't have Captain for Chrome yet", "Want to set it up?", "Connections → Browser → Download",
      "Load unpacked"], ["confirm=true"]),
    ({"connected": True, "person_connected": False, "own_connected": True, "extension_set_up": False},
     ["don't have Captain for Chrome yet"], ["May I open it"]),   # its own server browser is no help
    ({"connected": True, "person_connected": True, "stopped": True},
     ["stopped you using their Chrome", "don't use it or ask"], ["ask_user("]),
    (None, ["May I use your Chrome (Captain for Chrome) for it?", "Connections → Browser"], []),
    ({"connected": True, "own_connected": False}, ["May I open it in your Chrome"], []),   # an older deck
])
async def test_what_is_asked_fits_the_persons_chrome(fresh, status, expect, absent):
    fresh["status"] = status
    await wb.note_failure(_kw(), "https://x.example/", wb.BROWSER, "a bot check")
    note = await wb.note_failure(_kw(), "https://x.example/", wb.FETCH, "it refuses automated visitors (HTTP 403)")
    for bit in expect:
        assert bit in note, bit
    for bit in absent:
        assert bit not in note, bit


async def test_help_lines_fit_the_ask_dialog(fresh):
    import re

    for status in ({"person_connected": True}, {"extension_set_up": True}, {"extension_set_up": False}, None):
        note = wb.ask_chrome_note("site.example", "https://site.example/", "a bot check", "HTTP 403", status)
        for help_text in re.findall(r'help="([^"]*)"', note):
            assert len(help_text) <= 500
        for q in re.findall(r'question="([^"]*)"', note):
            assert len(q) <= 500


async def test_nothing_to_ask_for(fresh, monkeypatch):
    # Two shrugs (the site just didn't answer) — maybe it's down.
    assert await wb.note_failure(_kw(), "https://slow.example/", wb.BROWSER, "the page never finished loading",
                                 strong=False) == ""
    assert await wb.note_failure(_kw(), "https://slow.example/", wb.FETCH, "no answer", strong=False) == ""
    # Another session has its own record.
    await wb.note_failure(_kw("s1"), "https://y.example/", wb.BROWSER, "a bot check")
    assert "Try a plain fetch" in await wb.note_failure(_kw("s2"), "https://y.example/", wb.BROWSER, "a bot check")
    # The page came through after all: that way is forgotten.
    wb.forget(_kw("s1"), "https://y.example/", wb.BROWSER)
    assert "Try the browser once" in await wb.note_failure(_kw("s1"), "https://y.example/", wb.FETCH, "a bot check")
    # A member's turn never hears about the owner's Chrome.
    from captain_claw import speaker

    monkeypatch.setattr(speaker, "principal_for", lambda agent=None: object())
    assert await wb.note_failure(_kw("m"), "https://z.example/", wb.BROWSER, "a bot check") == ""
    monkeypatch.setattr(speaker, "principal_for", lambda agent=None: None)
    # Without Flight Deck there's nobody's Chrome to offer (trying the other way still helps).
    monkeypatch.setattr(fd_client, "flight_deck_base", lambda: "")
    fresh["status"] = None
    assert "Try a plain fetch" in await wb.note_failure(_kw("n"), "https://z.example/", wb.BROWSER, "a bot check")
    assert await wb.note_failure(_kw("n"), "https://z.example/", wb.FETCH, "a bot check") == ""


async def test_chrome_status_reads_the_decks_answer(monkeypatch):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"connected": False, "extension_set_up": True})

    real = httpx.AsyncClient
    monkeypatch.setattr(wb.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    assert await ORIGINAL_CHROME_STATUS() == {"connected": False, "extension_set_up": True}
    assert seen == ["/fd/browser-hands/agent/status"]
    monkeypatch.setattr(fd_client, "flight_deck_base", lambda: "")
    assert await ORIGINAL_CHROME_STATUS() is None and len(seen) == 1


# ── the tools carry it ───────────────────────────────────────────────


class _Page:
    """Each look at the page (it starts with its title) sees the next state; the last one stays."""

    def __init__(self, looks: list[tuple[str, str, str]]):
        self.looks, self.calls, self.url = list(looks), 0, "https://shop.example/"

    @property
    def look(self) -> tuple[str, str, str]:
        return self.looks[min(max(self.calls - 1, 0), len(self.looks) - 1)]

    async def title(self):
        self.calls += 1
        return self.look[0]

    async def evaluate(self, js: str):
        return self.look[2] if "outerHTML" in js else self.look[1]


class _Session:
    def __init__(self, nav: dict | Exception, page: _Page):
        self.nav, self.page = nav, page

    async def navigate(self, url: str):
        if isinstance(self.nav, Exception):
            raise self.nav
        return dict(self.nav)

    async def ensure_page(self):
        return self.page


@pytest.fixture
def no_wait(monkeypatch):
    async def _sleep(_s):
        return None

    monkeypatch.setattr(browser_mod.asyncio, "sleep", _sleep)


async def test_browser_navigate_notes_a_wall_then_web_fetch_completes_the_ask(fresh, no_wait, monkeypatch):
    tool = BrowserTool()
    page = _Page([(CF_TITLE, "Checking if the site connection is secure", CF_HTML)])
    monkeypatch.setattr(tool, "_get_session", lambda sid="default": _Session(
        {"url": "https://shop.example/", "title": CF_TITLE, "status": 403, "ok": False}, page))
    r = await tool._navigate(url="https://shop.example/", **_kw())
    assert "HTTP status: 403" in r.content and "--- Blocked ---" in r.content and "web_fetch(" in r.content

    fetch = WebFetchTool()

    class _Client:
        async def get(self, url):
            resp = httpx.Response(403, text=CF_HTML, request=httpx.Request("GET", url))
            return resp

    fetch.client = _Client()
    out = await fetch.execute(url="https://shop.example/", deep_fetch=False, **_kw())
    assert not out.success and "HTTP error" in out.error
    assert "Blocked: ask to use their Chrome" in out.error and "ask_user(" in out.error


async def test_a_self_clearing_check_is_no_wall(fresh, no_wait, monkeypatch):
    tool = BrowserTool()
    page = _Page([(CF_TITLE, "Checking if the site connection is secure", CF_HTML),
                  ("Shop — Item 1", "A long product page " * 200, "")])
    monkeypatch.setattr(tool, "_get_session", lambda sid="default": _Session(
        {"url": "https://shop.example/", "title": CF_TITLE, "status": 200, "ok": True}, page))
    r = await tool._navigate(url="https://shop.example/", **_kw())
    assert r.success and "Blocked" not in r.content and "Page title: Shop — Item 1" in r.content


async def test_a_dropped_connection_is_a_wall_a_missing_site_is_not(fresh, no_wait, monkeypatch):
    tool = BrowserTool()
    monkeypatch.setattr(tool, "_get_session", lambda sid="default": _Session(
        RuntimeError("Page.goto: net::ERR_HTTP2_PROTOCOL_ERROR at https://akamai.example/"), _Page([("", "", "")])))
    r = await tool._navigate(url="https://akamai.example/", **_kw())
    assert not r.success and "ERR_HTTP2_PROTOCOL_ERROR" in r.error and "--- Blocked ---" in r.error
    monkeypatch.setattr(tool, "_get_session", lambda sid="default": _Session(
        RuntimeError("net::ERR_NAME_NOT_RESOLVED"), _Page([("", "", "")])))
    with pytest.raises(RuntimeError):
        await tool._navigate(url="https://nope.example/", **_kw())


async def test_web_get_and_a_200_challenge_page(fresh):
    class _Client:
        async def get(self, url):
            return httpx.Response(200, text=CF_HTML, request=httpx.Request("GET", url))

    getter = WebGetTool()
    getter.client = _Client()
    r = await getter.execute(url="https://shop.example/", **_kw())
    assert r.success and "Try the browser once" in r.content
    fetch = WebFetchTool()
    fetch.client = _Client()
    r = await fetch.execute(url="https://other.example/", deep_fetch=False, **_kw())
    assert r.success and "--- Blocked ---" in r.content


async def test_chrome_after_a_block_points_at_their_chrome_not_back_at_the_server(fresh, monkeypatch):
    monkeypatch.setattr(browser_mod, "_TARGETS", {})
    answers: dict[str, dict] = {}

    async def _call(action, args, wait):
        return answers.get(action, {"ok": True, "content": f"did {action}"})

    monkeypatch.setattr(bc, "call_deck", _call)
    await wb.note_failure(_kw(), "https://shop.example/", wb.BROWSER, "a bot check")
    tool = BrowserTool()
    answers["navigate"] = {"ok": False, "no_browser": True, "error": "use the headless browser or extension_connect"}
    r = await tool.execute("navigate", url="https://shop.example/", target="chrome", **_kw())
    assert not r.success and "only theirs helps" in r.error and "Agents can use this browser" in r.error
    answers["navigate"] = {"ok": True, "own": True, "content": "Opened"}
    r = await tool.execute("navigate", url="https://shop.example/", target="chrome", **_kw())
    assert "own server browser, which shop.example blocks" in r.content
    # Their Chrome got there: the site is forgotten for this session.
    answers["navigate"] = {"ok": True, "content": "Opened in your tab"}
    r = await tool.execute("navigate", url="https://shop.example/", target="chrome", **_kw())
    assert r.success and wb.recent_block(_kw()) == ""
    # Without a block, the deck's own advice stands.
    answers["navigate"] = {"ok": False, "no_browser": True, "error": "deck advice"}
    r = await tool.execute("navigate", url="https://fine.example/", target="chrome", **_kw("s9"))
    assert r.error == "deck advice"
