"""Web pages read in the person's own Chrome for web_fetch (browser_fetch.py).

Pinned here:

* off until the person picks a mode; "first" (before web_fetch's own fetch)
  answers only for *every page*; a site list limits it to those sites and
  their subdomains;
* never an address on a private network or this machine, nor anything but
  http(s) — asked, or where the page ended up;
* only a browser of theirs that reads pages gets the page; one lent for that
  alone never gets an agent's act; an agent stopped in the panel reads nothing;
* the browser hears the mode in its welcome and whenever it changes;
* a page bigger than an act's usual answer still comes back whole.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import websockets

from captain_claw.flight_deck import browser_fetch as bf
from captain_claw.flight_deck import browser_hands as bh

from . import test_browser_hands as hands_tests

_agent = hands_tests._agent
deck = hands_tests.deck   # the same deck: a browser's socket, an agent's routes

PAGE = "<html><head><title>Jane Doe | LinkedIn</title></head><body><main><h1>Jane Doe</h1><p>Engineer.</p></main></body></html>"


def test_what_the_persons_browser_never_opens():
    for url in ("http://localhost:8080/", "http://127.0.0.1/", "http://10.0.0.5/admin", "http://192.168.1.1/",
                "http://172.20.0.1/", "http://169.254.169.254/latest/meta-data", "http://[::1]/", "http://[fd00::1]/",
                "http://router/", "http://nas.local/", "https://intranet.corp/x", "ftp://example.org/f",
                "file:///etc/passwd", "javascript:alert(1)", "https:///nohost"):
        assert bf.url_refusal(url), url
    for url in ("https://www.linkedin.com/in/someone", "http://example.org/", "https://8.8.8.8/"):
        assert bf.url_refusal(url) == "", url


def test_sites_are_cleaned_and_cover_their_subdomains():
    assert bf.clean_sites("https://www.LinkedIn.com/in/x, facebook.com;  nonsense  m.facebook.com") == \
        ["linkedin.com", "facebook.com", "m.facebook.com"]
    assert bf.clean_sites(["linkedin.com", "linkedin.com", "", "a b"]) == ["linkedin.com"]
    assert bf.site_listed("https://de.linkedin.com/in/x", ["linkedin.com"])
    assert not bf.site_listed("https://notlinkedin.com/", ["linkedin.com"])
    assert bf.site_listed("https://anything.example/", [])


class PageBrowser:
    """Captain for Chrome reading pages for agents (and nothing else)."""

    def __init__(self, url, token, *, answer=None, fetch=True, stopped=()):
        self.url, self.token, self.fetch, self.stopped = url, token, fetch, list(stopped)
        self.answer = answer or (lambda msg: {"ok": True, "content": PAGE, "data": {"url": msg["url"], "title": "Jane Doe | LinkedIn"}})
        self.got: list[dict] = []
        self.welcome: dict = {}

    async def __aenter__(self):
        self.ws = await websockets.connect(self.url, max_size=None)
        await self.ws.send(json.dumps({"type": "hello", "token": self.token, "browser": "Chrome on macOS", "version": "0.7.0",
                                       "stopped": self.stopped, "hands": False, "search": [], "fetch": self.fetch}))
        self.welcome = json.loads(await self.ws.recv())
        self.task = asyncio.create_task(self._serve())
        return self

    async def _serve(self):
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                self.got.append(msg)
                if msg.get("type") in ("fetch", "act"):
                    out = self.answer(msg) if msg["type"] == "fetch" else {"ok": True, "content": "acted"}
                    await self.ws.send(json.dumps({"type": "result", "id": msg["id"], **out}))
        except websockets.ConnectionClosed:
            pass

    async def __aexit__(self, *exc):
        await self.ws.close()
        await asyncio.wait([self.task], timeout=2)
        await asyncio.sleep(0.05)


async def _fetch(c, url="https://www.linkedin.com/in/someone", when="blocked", owner="alice"):
    r = await c.post("/fd/browser-hands/agent/fetch", headers=_agent(owner), json={"url": url, "when": when})
    assert r.status_code == 200, r.text
    return r.json()


async def _mode(mode, sites=(), uid="alice"):
    return await bf.put_chrome_fetch(bf.ChromeFetchBody(mode=mode, sites=list(sites)), {"id": uid})


async def test_off_until_the_person_picks_a_mode(deck):
    c = deck["client"]
    assert (await _fetch(c))["why"] == "off"
    st = await bf.get_chrome_fetch({"id": "alice"})
    assert st == {"mode": "", "sites": [], "browsers": [], "extension_set_up": False}
    with pytest.raises(Exception):
        await _mode("sometimes")
    st = await _mode("blocked", ["https://www.linkedin.com/feed"])
    assert st["mode"] == "blocked" and st["sites"] == ["linkedin.com"]
    assert await bf.fetch_mode("alice") == "blocked" and await bf.fetch_mode("bob") == ""
    assert (await _fetch(c, when="first")) == {"ok": False, "why": "not_first", "mode": "blocked",
                                               "error": "Only pages that turn Flight Deck away are read in the person's Chrome."}
    assert (await _fetch(c, "https://www.facebook.com/someone"))["why"] == "not_listed"
    assert (await _fetch(c, "http://192.168.1.1/"))["why"] == "refused"
    assert (await _fetch(c))["why"] == "no_browser"
    assert (await _mode(""))["mode"] == ""


async def test_a_page_is_read_in_the_owners_chrome(deck):
    c = deck["client"]
    pat = await deck["pat"]()
    await _mode("always")
    async with PageBrowser(deck["ws"], pat) as chrome:
        assert chrome.welcome["fetch_mode"] == "always"
        st = await bf.get_chrome_fetch({"id": "alice"})
        assert st["browsers"] == [{"browser": "Chrome on macOS", "version": "0.7.0", "ready": True}]
        listed = (await c.get("/fd/browsers", headers=deck["jwt"])).json()["browsers"]
        assert listed[0]["reads_pages"] is True and listed[0]["agents_can_use"] is False
        got = await _fetch(c, when="first")
        assert got == {"ok": True, "mode": "always", "browser": "Chrome on macOS", "url": "https://www.linkedin.com/in/someone",
                       "title": "Jane Doe | LinkedIn", "html": PAGE}
        sent = [m for m in chrome.got if m["type"] == "fetch"]
        assert len(sent) == 1 and sent[0]["id"].startswith("hf_") and sent[0]["agent"] == {"slug": "scout", "name": "Scout"}
        # Read for web_fetch only: an agent's browser tool never gets it.
        act = (await c.post("/fd/browser-hands/agent/act", headers=_agent(),
                            json={"action": "screenshot", "args": {}, "wait": 3})).json()
        assert act["no_browser"] and not [m for m in chrome.got if m["type"] == "act"]
        # Bob's agents don't read pages in Alice's Chrome.
        await _mode("always", uid="bob")
        assert (await _fetch(c, owner="bob"))["why"] == "no_browser"
        # The person changes it: the browser hears it.
        await _mode("blocked")
        await asyncio.sleep(0.1)
        assert {"type": "fetch_mode", "mode": "blocked"} in chrome.got
    assert bh._ACTS == {}


async def test_where_a_page_ended_up_counts_too(deck):
    c = deck["client"]
    pat = await deck["pat"]()
    await _mode("blocked")
    answers = iter([
        {"ok": True, "content": "<p>router admin</p>", "data": {"url": "http://192.168.1.1/admin"}},
        {"ok": False, "error": "Not a web page (application/pdf) — fetched the usual way instead."},
        {"ok": True, "content": "<p>" + "x" * 600_000 + "</p>", "data": {"url": "https://www.linkedin.com/in/someone"}},
    ])
    async with PageBrowser(deck["ws"], pat, answer=lambda msg: next(answers)):
        redirected = await _fetch(c)
        assert redirected["ok"] is False and redirected["why"] == "refused" and "html" not in redirected
        pdf = await _fetch(c)
        assert pdf["why"] == "page" and "Not a web page" in pdf["error"]
        big = await _fetch(c)
        assert big["ok"] and len(big["html"]) > 600_000   # more than an act's usual 400k


async def test_a_stopped_agent_reads_nothing_and_a_browser_not_reading_isnt_asked(deck):
    c = deck["client"]
    pat = await deck["pat"]()
    await _mode("blocked")
    async with PageBrowser(deck["ws"], pat, stopped=["scout"]) as chrome:
        assert (await _fetch(c))["why"] == "stopped" and not [m for m in chrome.got if m["type"] == "fetch"]
    async with PageBrowser(deck["ws"], pat, fetch=False) as chrome:
        assert (await _fetch(c))["why"] == "no_browser"
        # It turns reading on later (the person allowed it in the panel).
        await chrome.ws.send(json.dumps({"type": "prefs", "fetch": True}))
        await asyncio.sleep(0.1)
        assert (await _fetch(c))["ok"]


# A robot check where the page should be, handed to the person (browser_waits.py).

class CheckingPageBrowser(PageBrowser):
    def __init__(self, *a, outcome="solved", **kw):
        super().__init__(*a, **kw)
        self.outcome = outcome

    async def _serve(self):
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                self.got.append(msg)
                if msg.get("type") == "fetch":
                    asyncio.create_task(self._check(msg))
        except websockets.ConnectionClosed:
            pass

    async def _check(self, msg):
        site = "www.linkedin.com"
        await self.ws.send(json.dumps({"type": "waiting", "id": msg["id"], "site": site, "vendor": "Cloudflare",
                                       "for_s": msg["hand_over"]}))
        await asyncio.sleep(0.2)
        await self.ws.send(json.dumps({"type": "waiting", "id": msg["id"], "site": site, "over": self.outcome}))
        robot = {"site": site, "vendor": "Cloudflare", "outcome": self.outcome}
        if self.outcome == "solved":
            out = {"ok": True, "content": PAGE, "data": {"url": msg["url"], "title": "Jane Doe | LinkedIn", "robot": robot}}
        else:
            out = {"ok": False, "error": f"{site} showed a robot check (Cloudflare); the person closed that tab.",
                   "data": {"robot": robot}}
        await self.ws.send(json.dumps({"type": "result", "id": msg["id"], **out}))


async def test_a_robot_check_on_a_page_is_handed_to_the_person(deck):
    c = deck["client"]
    pat = await deck["pat"]()
    await _mode("always")
    for outcome in ("solved", "closed"):
        async with CheckingPageBrowser(deck["ws"], pat, outcome=outcome) as chrome:
            r = await c.post("/fd/browser-hands/agent/fetch", headers=_agent(),
                             json={"url": "https://www.linkedin.com/in/someone", "when": "first", "job": f"cw_{outcome}",
                                   "origin": "chat"})
            first = r.json()
            assert first["pending"] == f"cw_{outcome}" and first["waiting"]["vendor"] == "Cloudflare"
            assert [m for m in chrome.got if m["type"] == "fetch"][0]["hand_over"] == 300
            got = (await c.post("/fd/browser-hands/agent/wait/follow", headers=_agent(),
                                json={"id": f"cw_{outcome}", "wait": 5})).json()
            robot = {"site": "www.linkedin.com", "vendor": "Cloudflare", "outcome": outcome}
            if outcome == "solved":
                assert got["ok"] and got["html"] == PAGE and got["robot"] == robot
            else:
                assert got["ok"] is False and got["why"] == "robot" and got["robot"] == robot
                assert "closed that tab" in got["error"]
    assert bh._ACTS == {}
