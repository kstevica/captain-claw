"""Web searches in the person's own Chrome (browser_search.py).

Pinned here:

* each engine's result page is read into title / address / snippet / site /
  date — click-tracking addresses unwrapped, Google's opaque /goto links
  resolved, ads and the engine's own pages left out;
* a robot check, a cookie question and a page that never showed results are
  told apart from "no results", and send the agent to its search API;
* off until the person picks an engine; then only a browser of theirs that
  allowed that engine runs it — a browser lent for searches alone never gets
  an agent's act, and an extension from before this still lends its hands;
* a search past one page loads the next pages in turn.

The HTML here is shaped like each engine's real result page (checked against
live pages in a real browser), with made-up results.
"""

from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest
import websockets

from captain_claw.flight_deck import browser_hands as bh
from captain_claw.flight_deck import browser_search as bs

from . import test_browser_hands as hands_tests

_agent = hands_tests._agent
deck = hands_tests.deck   # the same deck: a browser's socket, an agent's routes

# ── result pages ─────────────────────────────────────────────────────


def _b64(url: str) -> str:
    return base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")


GOOGLE = """<html><body><div id="search"><div id="rso">
<div class="MjjYud"><div class="A6K0A"><div class="yuRUbf"><span><a class="zReHs"
  href="https://www.google.com/goto?url=CAESabc&amp;gsessionid=s1"><h3 class="LC20lb">Timeouts in <em>asyncio</em></h3>
  <div class="CA5RN"><span class="VuuXrf">Example Docs</span><cite>https://docs.example.org<span> › asyncio</span></cite></div></a></span>
  <div aria-label="About this result" role="button"></div></div>
  <div data-sncf="1"><div class="VwiC3b"><span class="YrbPuc"><span>May 5, 2024</span> — </span><span>Use <em>asyncio</em>.<em>timeout</em>() around
  the <em>gather</em>() call to bound the whole group.</span><a class="vzmbzf" href="https://www.google.com/goto?url=x"><span>Read more</span></a></div></div></div></div>
<div class="MjjYud"><div><a href="https://blog.example.com/waiting"><h3>Waiting in asyncio</h3>
  <span class="VuuXrf">A Blog</span></a><div data-sncf="1"><div class="VwiC3b">If the timeout expires, the inner task gets cancelled.</div></div></div></div>
<div class="MjjYud"><div><a href="https://www.google.com/goto?url=CAESvid"><h3>asyncio in 10 minutes</h3>
  <span class="VuuXrf">YouTube · Some Channel</span></a><div>9:46 A short video about tasks, gather and timeouts.</div></div></div>
<div class="MjjYud"><div><a href="https://www.google.com/search?q=asyncio+images&amp;udm=2"><h3>Images for asyncio</h3></a></div></div>
</div></div></body></html>"""

BING = """<html><body><ol id="b_results">
<li class="b_algo"><div class="b_tpcn"><a class="tilk" href="https://www.bing.com/ck/a?!&amp;p=1&amp;u=a1{u1}&amp;ntb=1">
  <div class="tptt">example.org</div><cite>https://example.org</cite></a></div>
  <h2><a href="https://www.bing.com/ck/a?!&amp;p=1&amp;u=a1{u1}&amp;ntb=1">Welcome to <strong>Example</strong></a></h2>
  <div class="b_caption"><p class="b_lineclamp2">22 hours ago · Example explains things clearly for everyone who asks.</p></div></li>
<li class="b_algo"><h2><a href="https://second.example.net/page">Second result</a></h2>
  <div class="b_caption"><p>Another snippet that is long enough to keep.</p></div></li>
<li class="b_pag"></li>
</ol></body></html>""".format(u1=_b64("https://example.org/"))

DDG = """<html><body><div id="links">
<div class="result results_links result--ad"><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">An ad</a></div>
<div class="result results_links web-result"><h2><a class="result__a"
  href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa%2Db&amp;rut=1">First <b>result</b></a></h2>
  <a class="result__url" href="x">example.org/a-b</a><a class="result__snippet" href="x">The first snippet, about <b>things</b>.</a></div>
</div></body></html>"""

YAHOO = """<html><body><div id="web"><ol>
<li><div class="dd algo algo-sr"><div class="compTitle"><h3 class="title"><a
  href="https://r.search.yahoo.com/_ylt=A0/RV=2/RE=1/RO=10/RU=https%3a%2f%2fexample.org%2fy/RK=2/RS=x-">
  <span class="d-ib fz-ms">example.org</span>A Yahoo result</a></h3></div>
  <div class="compText"><p>May 5, 2024 · What the page says, in a line or two.</p></div></div></li>
<li><div class="dd algo"><h3><a href="https://direct.example.com/">Direct one</a></h3><div class="compText"><p>Plain.</p></div></div></li>
</ol></div></body></html>"""

QWANT = """<html><body><div data-testid="sectionWeb">
<div data-testid="webResult"><div data-testid="domain"><a href="https://example.org/q">example.org</a>
  <div data-testid="breadcrumbs"><span>example.org</span><span> › </span><span>q</span></div></div>
  <h2><a href="https://example.org/q"><span>A Qwant result</span></a></h2>
  <div><div>What <mark>Qwant</mark> shows under the title.</div></div>
  <div data-testid="siteLinkEnhancedItem"><a href="https://example.org/other"><span>Other page</span></a><div>A site link.</div></div>
</div></div></body></html>"""


def _read(engine: str, html: str, url: str = ""):
    return bs.read_page(bs.ENGINES[engine], html, url)


def test_each_engines_result_page_is_read():
    hits, why = _read("google", GOOGLE)
    assert why == "" and [h["title"] for h in hits] == ["Timeouts in asyncio", "Waiting in asyncio", "asyncio in 10 minutes"]
    first = hits[0]
    assert first["url"].startswith("https://www.google.com/goto?url=CAESabc")
    assert first["site"] == "Example Docs" and first["date"] == "May 5, 2024"
    assert first["snippet"] == "Use asyncio.timeout() around the gather() call to bound the whole group."
    assert hits[1]["url"] == "https://blog.example.com/waiting" and hits[1]["snippet"].startswith("If the timeout")
    assert hits[2]["site"].startswith("YouTube") and "short video" in hits[2]["snippet"]

    hits, _ = _read("bing", BING)
    assert [h["url"] for h in hits] == ["https://example.org/", "https://second.example.net/page"]
    assert hits[0]["title"] == "Welcome to Example" and hits[0]["site"] == "example.org"
    assert hits[0]["date"] == "22 hours ago" and hits[0]["snippet"].startswith("Example explains")

    hits, _ = _read("duckduckgo", DDG)
    assert hits == [{"title": "First result", "url": "https://example.org/a-b", "snippet": "The first snippet, about things.",
                     "site": "example.org", "date": ""}]

    hits, _ = _read("yahoo", YAHOO)
    assert [(h["title"], h["url"]) for h in hits] == [("A Yahoo result", "https://example.org/y"),
                                                      ("Direct one", "https://direct.example.com/")]
    assert hits[0]["date"] == "May 5, 2024" and hits[0]["snippet"] == "What the page says, in a line or two."

    hits, _ = _read("qwant", QWANT)
    assert hits == [{"title": "A Qwant result", "url": "https://example.org/q", "snippet": "What Qwant shows under the title.",
                     "site": "example.org", "date": ""}]


def test_a_page_of_nothing_known_is_no_results_not_a_crash():
    for engine in bs.ENGINES:
        assert _read(engine, "<html><body><p>Hello</p></body></html>") == ([], "")
        assert _read(engine, "") == ([], "")


def test_click_tracking_addresses_are_unwrapped():
    assert bs.unwrap("https://www.google.com/url?q=https://a.example/x&sa=U") == "https://a.example/x"
    assert bs.unwrap(f"https://www.bing.com/ck/a?!&p=1&u=a1{_b64('https://b.example/y?z=1')}&ntb=1") == "https://b.example/y?z=1"
    assert bs.unwrap("https://duckduckgo.com/l/?uddg=https%3A%2F%2Fc.example%2F&rut=9") == "https://c.example/"
    assert bs.unwrap("https://r.search.yahoo.com/_ylt=1/RU=https%3a%2f%2fd.example%2f/RK=2/RS=x") == "https://d.example/"
    # Opaque or foreign: left as they are.
    for url in ("https://www.google.com/goto?url=CAES", "https://www.bing.com/ck/a?u=zz", "https://example.org/l/?uddg=x",
                "https://duckduckgo.com/l/?uddg=javascript%3Aalert(1)"):
        assert bs.unwrap(url) == url


def test_robot_checks_and_cookie_questions_are_told_apart():
    sorry = '<html><body><form id="captcha-form"></form>Our systems have detected unusual traffic</body></html>'
    hits, why = _read("google", sorry, "https://www.google.com/sorry/index?continue=x")
    assert hits == [] and "robot check" in why and "Google" in why
    _, why = _read("duckduckgo", '<form id="challenge-form"><div class="anomaly-modal__modal"></div></form>')
    assert "robot check" in why
    _, why = _read("google", "<html><body>Before you continue</body></html>", "https://consent.google.com/m?continue=x")
    assert "cookie question" in why


def test_result_page_addresses():
    g = bs.ENGINES["google"].build
    assert g("a b", 0, "", "", "", "") == "https://www.google.com/search?q=a+b"
    assert g("a&b", 2, "hr", "HR", "pw", "strict") == "https://www.google.com/search?q=a%26b&start=20&hl=hr&gl=hr&tbs=qdr%3Aw&safe=active"
    assert bs.ENGINES["bing"].build("x", 1, "en", "US", "pd", "") == \
        "https://www.bing.com/search?q=x&first=11&setlang=en&cc=us&filters=ex1%3A%22ez1%22"
    assert bs.ENGINES["duckduckgo"].build("x", 0, "", "GB", "", "off") == "https://html.duckduckgo.com/html/?q=x&kl=uk-en&kp=-2"
    assert bs.ENGINES["duckduckgo"].build("x", 0, "", "HR", "", "") == "https://html.duckduckgo.com/html/?q=x&kl=hr-hr"
    assert bs.ENGINES["yahoo"].build("x", 2, "de", "", "pm", "") == "https://search.yahoo.com/search?p=x&b=21&vl=lang_de&btf=m"
    assert bs.ENGINES["qwant"].build("x", 0, "en", "GB", "", "") == "https://www.qwant.com/?q=x&t=web&locale=en_GB"
    # The extension opens result pages on the engine's own host only: the two lists agree.
    for e in bs.ENGINES.values():
        assert e.build("x", 0, "", "", "", "").startswith(f"https://{e.host}/")


async def test_google_goto_links_resolve_to_the_page(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if "CAESok" in str(request.url):
            return httpx.Response(302, headers={"location": "https://docs.example.org/asyncio"})
        if "CAESloop" in str(request.url):
            return httpx.Response(302, headers={"location": "https://www.google.com/sorry/index"})
        return httpx.Response(500)

    real = httpx.AsyncClient
    monkeypatch.setattr(bs.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    hits = [{"url": "https://www.google.com/goto?url=CAESok", "site": ""},
            {"url": "https://www.google.com/goto?url=CAESloop", "site": "Kept"},
            {"url": "https://www.google.com/goto?url=CAESbroken", "site": "Kept"},
            {"url": "https://plain.example/", "site": "plain.example"}]
    await bs.resolve_links(hits)
    assert hits[0] == {"url": "https://docs.example.org/asyncio", "site": "docs.example.org"}
    assert hits[1]["url"].endswith("CAESloop") and hits[2]["url"].endswith("CAESbroken")   # still working redirects
    assert len(seen) == 3   # the plain address isn't asked


# ── through the deck ─────────────────────────────────────────────────


class SearchBrowser:
    """Captain for Chrome lending this browser for searches (not its hands)."""

    def __init__(self, url: str, token: str, *, engines=("google",), hands=False, pages=None):
        self.url, self.token, self.engines, self.hands = url, token, list(engines), hands
        self.pages = pages or (lambda msg: {"ok": True, "content": GOOGLE, "data": {"url": msg["url"], "ready": True}})
        self.got: list[dict] = []
        self.ws = None
        self.task = None

    async def __aenter__(self):
        self.ws = await websockets.connect(self.url)
        hello = {"type": "hello", "token": self.token, "browser": "Chrome on macOS", "version": "0.6.0",
                 "stopped": [], "hands": self.hands, "search": self.engines}
        await self.ws.send(json.dumps(hello))
        assert json.loads(await self.ws.recv())["type"] == "welcome"
        self.task = asyncio.create_task(self._serve())
        return self

    async def _serve(self):
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                self.got.append(msg)
                if msg.get("type") == "search":
                    await self.ws.send(json.dumps({"type": "result", "id": msg["id"], **self.pages(msg)}))
                elif msg.get("type") == "act":
                    await self.ws.send(json.dumps({"type": "result", "id": msg["id"], "ok": True, "content": "acted"}))
        except websockets.ConnectionClosed:
            pass

    async def __aexit__(self, *exc):
        await self.ws.close()
        await asyncio.wait([self.task], timeout=2)
        await asyncio.sleep(0.05)


async def _search(c, query="asyncio timeout", *, owner="alice", **extra):
    r = await c.post("/fd/browser-hands/agent/search", headers=_agent(owner), json={"query": query, **extra})
    assert r.status_code == 200, r.text
    return r.json()


async def _pick(engine: str, uid: str = "alice"):
    return await bs.put_chrome_search(bs.ChromeSearchBody(engine=engine), {"id": uid})


@pytest.fixture
def no_resolving(monkeypatch):
    async def _skip(hits):
        return None

    monkeypatch.setattr(bs, "resolve_links", _skip)


async def test_off_until_the_person_picks_an_engine(deck, no_resolving):
    c = deck["client"]
    assert (await _search(c))["why"] == "off"
    st = await bs.get_chrome_search({"id": "alice"})
    assert st["engine"] == "" and {e["id"] for e in st["engines"]} == {"google", "bing", "duckduckgo", "yahoo", "qwant"}
    with pytest.raises(Exception):
        await _pick("brave")   # not one on offer
    st = await _pick("google")
    assert st["engine"] == "google" and st["browsers"] == [] and not st["extension_set_up"]
    assert await bs.clip_note("alice") == {"engine": "google", "name": "Google", "host": "www.google.com"}
    assert await bs.clip_note("bob") is None   # one person's choice, not the deck's
    got = await _search(c)
    assert got["why"] == "no_browser" and "Google" in got["error"]
    assert (await _pick(""))["engine"] == "" and await bs.clip_note("alice") is None


async def test_a_search_runs_in_the_owners_chrome_and_the_page_is_read_here(deck, no_resolving):
    c = deck["client"]
    await _pick("google")
    pat = await deck["pat"]()
    async with SearchBrowser(deck["ws"], pat) as chrome:
        st = await bs.get_chrome_search({"id": "alice"})
        assert st["browsers"] == [{"browser": "Chrome on macOS", "version": "0.6.0", "engines": ["google"], "ready": True}]
        listed = (await c.get("/fd/browsers", headers=deck["jwt"])).json()["browsers"]
        assert listed[0]["searches"] == ["google"] and listed[0]["agents_can_use"] is False
        got = await _search(c, "asyncio gather timeout", count=3, search_lang="en", country="US")
        assert got["ok"] and got["engine"] == "Google" and got["browser"] == "Chrome on macOS"
        assert [h["title"] for h in got["results"]] == ["Timeouts in asyncio", "Waiting in asyncio", "asyncio in 10 minutes"]
        sent = [m for m in chrome.got if m["type"] == "search"]
        assert len(sent) == 1 and sent[0]["engine"] == "google" and sent[0]["id"].startswith("hs_")
        assert sent[0]["url"] == "https://www.google.com/search?q=asyncio+gather+timeout&hl=en&gl=us"
        assert sent[0]["wait_for"] and sent[0]["block_for"]
        # Lent for searches only: an agent's browser tool never gets it.
        act = (await c.post("/fd/browser-hands/agent/act", headers=_agent(),
                            json={"action": "screenshot", "args": {}, "wait": 3})).json()
        assert act["no_browser"] and not [m for m in chrome.got if m["type"] == "act"]
        # Bob's agents don't search in Alice's Chrome.
        await _pick("google", "bob")
        assert (await _search(c, owner="bob"))["why"] == "no_browser"
        # A browser that didn't allow the picked engine isn't asked.
        await _pick("bing")
        assert (await _search(c))["why"] == "no_browser" and len([m for m in chrome.got if m["type"] == "search"]) == 1
    assert bh._ACTS == {}


async def test_a_robot_check_or_an_empty_page_sends_the_agent_to_its_api(deck, no_resolving):
    c = deck["client"]
    await _pick("google")
    pat = await deck["pat"]()
    answers = iter([
        {"ok": True, "content": '<form id="captcha-form"></form>', "data": {"url": "https://www.google.com/sorry/index", "ready": False}},
        {"ok": True, "content": "<html><body>still loading</body></html>", "data": {"url": "https://www.google.com/search", "ready": False}},
        {"ok": True, "content": '<div id="botstuff">No results</div>', "data": {"url": "https://www.google.com/search", "ready": True}},
        {"ok": False, "error": "Searches on Google aren't turned on in this browser"},
    ])
    async with SearchBrowser(deck["ws"], pat, pages=lambda msg: next(answers)):
        blocked = await _search(c)
        assert blocked["ok"] is False and blocked["why"] == "blocked" and "robot check" in blocked["error"]
        empty = await _search(c)
        assert empty["why"] == "blocked" and "didn't show results" in empty["error"]
        none = await _search(c)
        assert none["ok"] is True and none["results"] == []   # the engine found nothing: that's an answer
        refused = await _search(c)
        assert refused["why"] == "page" and "aren't turned on" in refused["error"]


async def test_more_than_a_page_loads_the_next_pages_in_turn(deck, no_resolving):
    c = deck["client"]
    await _pick("bing")
    pat = await deck["pat"]()

    def page(msg):
        n = int(dict(p.split("=") for p in msg["url"].split("?")[1].split("&")).get("first", "1"))
        items = "".join(f'<li class="b_algo"><h2><a href="https://r{n + i}.example/">Result {n + i}</a></h2>'
                        f'<div class="b_caption"><p>Snippet number {n + i} with enough words.</p></div></li>' for i in range(10))
        return {"ok": True, "content": f'<ol id="b_results">{items}</ol>', "data": {"url": msg["url"], "ready": True}}

    async with SearchBrowser(deck["ws"], pat, engines=["bing", "google"], pages=page) as chrome:
        got = await _search(c, count=15)
        assert got["ok"] and len(got["results"]) == 15
        assert got["results"][0]["title"] == "Result 1" and got["results"][-1]["title"] == "Result 15"
        assert [m["url"].endswith("first=11") for m in chrome.got if m["type"] == "search"] == [False, True]
        # The second page of 10: results 11–20, from Bing's second page only.
        got = await _search(c, count=10, page=1)
        assert [h["title"] for h in got["results"]][:1] == ["Result 11"] and len(got["results"]) == 10
        # DuckDuckGo gives one page here: deeper is the API engine's.
        await _pick("duckduckgo")
    chrome2 = SearchBrowser(deck["ws"], pat, engines=["duckduckgo"])
    async with chrome2:
        assert (await _search(c, count=10, page=1))["why"] == "past"


async def test_an_extension_from_before_this_still_lends_its_hands(deck):
    pat = await deck["pat"]()
    async with websockets.connect(deck["ws"]) as ws:
        await ws.send(json.dumps({"type": "hello", "token": pat, "browser": "Chrome on macOS", "version": "0.5.1"}))
        assert json.loads(await ws.recv())["type"] == "welcome"
        await asyncio.sleep(0.05)
        conns = bh.live_connections("alice")
        assert len(conns) == 1 and conns[0].hands and conns[0].search_engines == ()
        assert bh.connections("alice") == conns
        # Later the person turns the hands off and searches on: the browser says so.
        await ws.send(json.dumps({"type": "prefs", "hands": False, "search": ["google", "nope", "google"]}))
        await asyncio.sleep(0.1)
        assert bh.connections("alice") == [] and bh.live_connections("alice")[0].search_engines == ("google",)
