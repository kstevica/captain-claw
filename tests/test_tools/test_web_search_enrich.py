"""web_search: the higher result cap (Brave paging, Tavily's ceiling) and
enriched results (extra excerpts, site/date, passages from the pages)."""

from __future__ import annotations

import re

import httpx
import pytest

from captain_claw.config import get_config, set_config
from captain_claw.tools import web_search as ws
from captain_claw.tools.web_search import WebSearchTool, _focused_passages, _Hit


class _Resp:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)
        self.request = httpx.Request("GET", "https://api.example/search")

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                message=f"status={self.status_code}", request=self.request,
                response=httpx.Response(self.status_code, request=self.request, text=self.text),
            )


class _Client:
    """Answers each call with the next queued response."""

    def __init__(self, *responses: _Resp):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def _next(self) -> _Resp:
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]

    async def get(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self._next()

    async def post(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self._next()


def _brave_page(start: int, n: int, more: bool = True) -> dict:
    return {
        "query": {"more_results_available": more},
        "web": {"results": [
            {"title": f"Result {i}", "url": f"https://site{i}.example/page",
             "description": f"Snippet <strong>{i}</strong> &amp; more",
             "extra_snippets": [f"extra a{i}", f"extra b{i}"],
             "profile": {"name": f"Site {i}"}, "age": "2 days ago"}
            for i in range(start, start + n)
        ]},
    }


@pytest.fixture
def search_cfg(monkeypatch):
    old = get_config().model_copy(deep=True)
    cfg = old.model_copy(deep=True)
    for env in ("BRAVE_API_KEY", "TAVILY_API_KEY", "YDC_API_KEY", "KEENABLE_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    cfg.tools.web_search.provider = "brave"
    cfg.tools.web_search.api_key = "k"
    cfg.tools.web_search.tavily_api_key = "k"
    cfg.tools.web_search.youcom_api_key = "k"
    set_config(cfg)
    monkeypatch.setattr(ws, "_RETRY_DELAY", 0)
    yield cfg.tools.web_search
    set_config(old)


# ── cap ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_default_count_is_ten(search_cfg):
    tool = WebSearchTool()
    tool.client = _Client(_Resp(_brave_page(0, 10)))
    await tool.execute(query="q")
    assert tool.client.calls[0]["params"]["count"] == 10


@pytest.mark.asyncio
async def test_brave_pages_past_twenty_and_trims(search_cfg):
    tool = WebSearchTool()
    tool.client = _Client(_Resp(_brave_page(0, 20)), _Resp(_brave_page(20, 20)),
                          _Resp(_brave_page(40, 20)))
    result = await tool.execute(query="q", count=45)

    assert result.success
    assert [(c["params"]["count"], c["params"]["offset"]) for c in tool.client.calls] \
        == [(20, 0), (20, 1), (20, 2)]
    assert "[RESULTS: 45]" in result.content
    assert "45. Result 44" in result.content
    assert "Result 45" not in result.content


@pytest.mark.asyncio
async def test_brave_offset_window_over_twenty(search_cfg):
    tool = WebSearchTool()
    tool.client = _Client(_Resp(_brave_page(20, 20)), _Resp(_brave_page(40, 20)),
                          _Resp(_brave_page(60, 20)))
    result = await tool.execute(query="q", count=30, offset=1)  # results 30-59
    assert [c["params"]["offset"] for c in tool.client.calls] == [1, 2]
    assert "1. Result 30" in result.content
    assert "30. Result 59" in result.content


@pytest.mark.asyncio
async def test_brave_stops_when_no_more_results(search_cfg):
    tool = WebSearchTool()
    tool.client = _Client(_Resp(_brave_page(0, 20, more=False)), _Resp(_brave_page(20, 20)))
    result = await tool.execute(query="q", count=50)
    assert len(tool.client.calls) == 1
    assert "[RESULTS: 20]" in result.content


@pytest.mark.asyncio
async def test_brave_later_page_error_keeps_earlier_results(search_cfg):
    tool = WebSearchTool()
    tool.client = _Client(_Resp(_brave_page(0, 20)), _Resp({"error": "quota"}, 402))
    result = await tool.execute(query="q", count=40)
    assert result.success
    assert "[RESULTS: 20]" in result.content
    assert "[NOTE: Brave stopped after 20 results (HTTP 402" in result.content


@pytest.mark.asyncio
async def test_count_clamped_to_configured_cap(search_cfg):
    search_cfg.max_results_cap = 30
    tool = WebSearchTool()
    tool.client = _Client(_Resp(_brave_page(0, 20)), _Resp(_brave_page(20, 20)))
    result = await tool.execute(query="q", count=500)
    assert "[RESULTS: 30]" in result.content


@pytest.mark.asyncio
async def test_max_results_argument_is_read_as_count(search_cfg):
    tool = WebSearchTool()
    tool.client = _Client(_Resp(_brave_page(0, 15)))
    await tool.execute(query="q", max_results="15")
    assert tool.client.calls[0]["params"]["count"] == 15


@pytest.mark.asyncio
async def test_brave_retries_once_on_rate_limit(search_cfg):
    tool = WebSearchTool()
    tool.client = _Client(_Resp({}, 429), _Resp(_brave_page(0, 3)))
    result = await tool.execute(query="q", count=3)
    assert result.success
    assert len(tool.client.calls) == 2
    assert "[RESULTS: 3]" in result.content


@pytest.mark.asyncio
async def test_tavily_caps_at_twenty_and_says_so(search_cfg):
    search_cfg.provider = "tavily"
    tool = WebSearchTool()
    tool.client = _Client(_Resp({"results": []}))
    result = await tool.execute(query="q", count=35, freshness="pw")
    payload = tool.client.calls[0]["json"]
    assert payload["max_results"] == 20
    assert payload["search_depth"] == "basic"
    assert payload["time_range"] == "week"
    assert "Tavily returns at most 20 results per search; asked for 35." in result.content


# ── enriched by default ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_brave_result_carries_extras_site_and_date(search_cfg):
    tool = WebSearchTool()
    tool.client = _Client(_Resp(_brave_page(0, 1)))
    result = await tool.execute(query="q", count=1)
    assert tool.client.calls[0]["params"]["extra_snippets"] == "true"
    assert "   Source: Site 0 · 2 days ago" in result.content
    assert "   Snippet: Snippet 0 & more" in result.content  # markup dropped
    assert "   More: extra a0 | extra b0" in result.content


@pytest.mark.asyncio
async def test_brave_extra_snippets_can_be_switched_off(search_cfg):
    search_cfg.extra_snippets = False
    tool = WebSearchTool()
    tool.client = _Client(_Resp(_brave_page(0, 1)))
    await tool.execute(query="q", count=1)
    assert "extra_snippets" not in tool.client.calls[0]["params"]


# ── enrich=true ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tavily_enrich_uses_advanced_chunks(search_cfg):
    search_cfg.provider = "tavily"
    tool = WebSearchTool()
    tool.client = _Client(_Resp({"results": [{
        "title": "T", "url": "https://t.example/a", "published_date": "2026-11-01",
        "content": "first chunk about zagreb [...] second chunk [...] third chunk",
    }]}))
    result = await tool.execute(query="zagreb", enrich=True)
    payload = tool.client.calls[0]["json"]
    assert payload["search_depth"] == "advanced"
    assert payload["chunks_per_source"] == 3
    assert "   Source: t.example · 2026-11-01" in result.content
    assert "   Snippet: first chunk about zagreb" in result.content
    assert "     - second chunk" in result.content
    assert "     - third chunk" in result.content


@pytest.mark.asyncio
async def test_youcom_enrich_asks_for_highlights(search_cfg):
    search_cfg.provider = "youcom"
    tool = WebSearchTool()
    tool.client = _Client(_Resp({"results": {"web": [{
        "title": "Y", "url": "https://y.example/a", "description": "desc",
        "page_age": "2026-10-30T10:00:00",
        "contents": {"highlights": ["a passage that matches", "another one"]},
    }]}}))
    result = await tool.execute(query="q", enrich="true")
    payload = tool.client.calls[0]["json"]
    assert payload["extraction"] == {
        "extraction_mode": "highlights", "highlights": {"total_max_tokens": 8192}}
    assert "   Snippet: desc" in result.content
    assert "   From the page:\n     - a passage that matches\n     - another one" in result.content


@pytest.mark.asyncio
async def test_youcom_shows_all_snippets(search_cfg):
    search_cfg.provider = "youcom"
    tool = WebSearchTool()
    tool.client = _Client(_Resp({"results": {"web": [{
        "title": "Y", "url": "https://y.example/a", "snippets": ["one", "two", "three"],
    }]}}))
    result = await tool.execute(query="q")
    assert "extraction" not in tool.client.calls[0]["json"]
    assert "   Snippet: one" in result.content
    assert "   More: two | three" in result.content


class _PageResp:
    def __init__(self, body: str, ctype: str = "text/html; charset=utf-8", status: int = 200):
        self._body = body.encode()
        self.status_code = status
        self.headers = {"content-type": ctype}
        self.encoding = "utf-8"

    async def aiter_bytes(self):
        yield self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _PageClient:
    def __init__(self, pages: dict):
        self.pages = pages
        self.fetched: list[str] = []

    def stream(self, method: str, url: str):
        self.fetched.append(url)
        page = self.pages.get(url)
        if isinstance(page, Exception):
            raise page
        return page or _PageResp("", status=404)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


_ARTICLE = """<html><head><title>Guide</title><script>var x = 1;</script></head><body>
<nav>Home</nav><nav>About us</nav>
<p>This opening paragraph talks about nothing in particular, filler words only.</p>
<p>The Zagreb funicular is one of the shortest public transport lines in the world.</p>
<p>Another filler paragraph about weather and food in general terms here.</p>
<p>Tickets for the funicular in Zagreb cost a single tram fare.</p>
</body></html>"""


@pytest.mark.asyncio
async def test_brave_enrich_reads_top_pages(search_cfg, monkeypatch):
    search_cfg.enrich_pages = 2
    tool = WebSearchTool()
    tool.client = _Client(_Resp(_brave_page(0, 3)))
    pages = _PageClient({
        "https://site0.example/page": _PageResp(_ARTICLE),
        "https://site1.example/page": _PageResp("%PDF-1.7", ctype="application/pdf"),
    })
    monkeypatch.setattr(tool, "_page_client", lambda: pages)

    result = await tool.execute(query="zagreb funicular tickets", count=3, enrich=True)

    assert pages.fetched == ["https://site0.example/page", "https://site1.example/page"]
    block0 = result.content.split("2. Result 1")[0]
    assert "From the page:" in block0
    assert "Tickets for the funicular in Zagreb cost a single tram fare." in block0
    assert "shortest public transport lines" in block0
    assert "filler" not in block0
    assert "var x" not in block0
    # The PDF wasn't readable as a page: said once, not a failure.
    assert "[NOTE: Enriched 1 of the top 2 results" in result.content
    assert result.content.count("From the page:") == 1


@pytest.mark.asyncio
async def test_enrich_off_by_default_reads_nothing(search_cfg, monkeypatch):
    tool = WebSearchTool()
    tool.client = _Client(_Resp(_brave_page(0, 3)))

    def _boom():
        raise AssertionError("no page should be read")

    monkeypatch.setattr(tool, "_page_client", _boom)
    result = await tool.execute(query="q", count=3)
    assert "From the page:" not in result.content


@pytest.mark.asyncio
async def test_enrich_config_default_turns_it_on(search_cfg, monkeypatch):
    search_cfg.enrich = True
    tool = WebSearchTool()
    tool.client = _Client(_Resp(_brave_page(0, 1)))
    pages = _PageClient({"https://site0.example/page": _PageResp(_ARTICLE)})
    monkeypatch.setattr(tool, "_page_client", lambda: pages)
    result = await tool.execute(query="zagreb funicular", count=1)
    assert "From the page:" in result.content
    # …and the agent can still say no for one call.
    tool.client = _Client(_Resp(_brave_page(0, 1)))
    result = await tool.execute(query="zagreb funicular", count=1, enrich=False)
    assert "From the page:" not in result.content


@pytest.mark.asyncio
async def test_enrich_page_failure_never_fails_search(search_cfg, monkeypatch):
    tool = WebSearchTool()
    tool.client = _Client(_Resp(_brave_page(0, 2)))
    pages = _PageClient({"https://site0.example/page": httpx.ConnectError("refused")})
    monkeypatch.setattr(tool, "_page_client", lambda: pages)
    result = await tool.execute(query="q", count=2, enrich=True)
    assert result.success
    assert "[NOTE: Enriched 0 of the top 2 results" in result.content


def test_page_reads_use_public_only_client():
    from captain_claw import speaker

    client = WebSearchTool()._page_client()
    try:
        pool = client._transport._pool
        assert isinstance(pool._network_backend, speaker._PublicOnlyBackend)
    finally:
        import asyncio

        asyncio.run(client.aclose())


# ── passages and budget ──────────────────────────────────────────────


def test_focused_passages_keep_page_order_and_budget():
    text = "\n".join([
        "An unrelated line that is long enough to count as a real paragraph.",
        "Second line mentions the castle once, which makes it relevant enough.",
        "Third line has castle and tickets, both query words, ranked first.",
        "short castle",  # too short: menu-like
    ])
    out = _focused_passages(text, "castle tickets", budget=140)
    assert out == ["Second line mentions the castle once, which makes it relevant enough.",
                   "Third line has castle and tickets, both query words, ranked first."]
    out = _focused_passages(text, "castle tickets", budget=80)
    assert out == ["Third line has castle and tickets, both query words, ranked first."]


def test_focused_passages_fall_back_to_opening():
    text = "First real paragraph of the page, long enough to count.\nSecond real paragraph of the page, long enough too."
    assert _focused_passages(text, "zeppelin", budget=60) == [
        "First real paragraph of the page, long enough to count."]


def test_render_goes_compact_then_cuts_to_budget():
    tool = WebSearchTool()
    hits = [_Hit(title=f"T{i}", url=f"https://e{i}.example", snippet="s" * 400,
                 extra=["x" * 250] * 3) for i in range(40)]
    out = tool._render("Brave", "q", hits, [], budget=6000)
    assert len(out) <= 6000
    assert "1. T0" in out and "More:" in out
    assert re.search(r"\[… \d+ more results left out to fit the context", out)
    # Compact rows keep title + URL + short snippet.
    compact = [b for b in out.split("\n\n") if b.startswith(tuple(f"{i}. " for i in range(2, 40)))
               and "More:" not in b]
    assert compact, "expected some compact rows before the cut"


def test_output_still_parses_for_url_and_snippet_readers():
    """agent_research_mixin and the Gmail auto-reply read URL:/Snippet: lines."""
    from captain_claw.agent_research_mixin import AgentResearchMixin
    from captain_claw.flight_deck.gmail_autoreply_writer import _web_items

    tool = WebSearchTool()
    hits = [_Hit(title="Zagreb", url="https://zagreb.example/a", snippet="Old town.",
                 extra=["2. looks like a list"], site="zagreb.example", date="1 day ago",
                 passages=["1. A numbered passage line", "URL: https://not-a-result.example"])]
    out = tool._render("Brave", "q", hits, [], budget=60_000)
    assert AgentResearchMixin._parse_search_result_urls(out) == ["https://zagreb.example/a"]
    items = _web_items(out)
    assert len(items) == 1
    assert items[0][0] == "zagreb.example"
    assert "Old town." in items[0][1]


def test_focused_passages_ignore_links_title_and_repeats():
    text = "\n".join([
        "Zagreb funicular guide for visitors and residents alike",  # the title
        "Sign in (https://example.com/login?next=/zagreb/funicular/tickets)",
        "Tickets for the Zagreb funicular cost one tram fare, paid on board.",
        "Tickets for the Zagreb funicular cost one tram fare, paid on board.",
    ])
    out = _focused_passages(text, "zagreb funicular tickets", budget=1500,
                            title="Zagreb funicular guide for visitors and residents alike")
    assert out == ["Tickets for the Zagreb funicular cost one tram fare, paid on board."]


def test_page_blocks_keep_whole_paragraphs_without_menus():
    from captain_claw.tools.web_search import _page_blocks

    text = _page_blocks(
        "<html><body><nav><a href='/'>Home</a></nav><header>Site</header>"
        "<p>Use the <code>asyncio</code> module, <a href='/x'>see docs</a>, to run tasks.</p>"
        "<ul><li><p>Inner paragraph inside a list item.</p></li></ul>"
        "<footer>© 2026</footer></body></html>")
    assert text.splitlines() == [
        "Use the asyncio module, see docs , to run tasks.",
        "Inner paragraph inside a list item.",
    ]


# ── follow-ups: count floor, all excerpts, Brave LLM Context ─────────


@pytest.mark.asyncio
async def test_old_config_max_results_five_is_ignored(search_cfg):
    search_cfg.max_results = 5  # what older agent configs carry
    tool = WebSearchTool()
    tool.client = _Client(_Resp(_brave_page(0, 10)))
    await tool.execute(query="q")
    assert tool.client.calls[0]["params"]["count"] == 10
    # An explicit count still wins, low or high.
    await tool.execute(query="q", count=3)
    assert tool.client.calls[1]["params"]["count"] == 3


@pytest.mark.asyncio
async def test_all_five_brave_excerpts_shown(search_cfg):
    page = _brave_page(0, 1)
    page["web"]["results"][0]["extra_snippets"] = [f"excerpt {i}" for i in range(5)]
    tool = WebSearchTool()
    tool.client = _Client(_Resp(page))
    result = await tool.execute(query="q", count=1)
    assert "   More: excerpt 0 | excerpt 1 | excerpt 2 | excerpt 3 | excerpt 4" in result.content


def _context(*urls_chunks) -> dict:
    return {"grounding": {"generic": [
        {"url": url, "title": "t", "snippets": chunks} for url, chunks in urls_chunks]},
        "sources": {}}


@pytest.mark.asyncio
async def test_brave_enrich_uses_llm_context(search_cfg, monkeypatch):
    tool = WebSearchTool()
    # ~300 characters each, like real page chunks: 6 fit in 2000 chars.
    chunks = [f"chunk {i} about the funicular and its tickets " + "w" * 250 for i in range(12)]
    tool.client = _Client(
        _Resp(_brave_page(0, 3)),
        _Resp(_context(("https://site0.example/page/", chunks),
                       ("https://site2.example/page", ["only one chunk here"]))),
    )

    def _no_reads():
        raise AssertionError("LLM Context answered: no page should be read here")

    monkeypatch.setattr(tool, "_page_client", _no_reads)
    result = await tool.execute(query="funicular tickets", count=3, enrich=True)

    ctx = tool.client.calls[1]
    assert ctx["url"] == "https://api.search.brave.com/res/v1/llm/context"
    assert ctx["headers"]["X-Subscription-Token"] == "k"
    assert ctx["params"]["q"] == "funicular tickets"
    assert ctx["params"]["maximum_number_of_urls"] == 5
    assert ctx["params"]["maximum_number_of_tokens_per_url"] == 512  # 2000 chars
    assert ctx["params"]["count"] == 5
    block0, rest = result.content.split("2. Result 1")
    # Far more than Brave's five extra excerpts — as many as 2000 chars hold.
    assert block0.count("     - chunk ") == 6
    assert "chunk 6 " not in block0  # the budget, not the 12 chunks, decides
    assert "     - only one chunk here" in rest
    assert "[NOTE:" not in result.content


@pytest.mark.asyncio
async def test_brave_llm_context_failure_falls_back_to_reading_pages(search_cfg, monkeypatch):
    tool = WebSearchTool()
    tool.client = _Client(_Resp(_brave_page(0, 1)), _Resp({"error": "plan"}, 403))
    pages = _PageClient({"https://site0.example/page": _PageResp(_ARTICLE)})
    monkeypatch.setattr(tool, "_page_client", lambda: pages)
    result = await tool.execute(query="zagreb funicular", count=1, enrich=True)
    assert result.success
    assert pages.fetched == ["https://site0.example/page"]
    assert "Tickets for the funicular in Zagreb cost a single tram fare." in result.content


@pytest.mark.asyncio
async def test_brave_llm_context_can_be_switched_off(search_cfg, monkeypatch):
    search_cfg.brave_llm_context = False
    tool = WebSearchTool()
    tool.client = _Client(_Resp(_brave_page(0, 1)))
    pages = _PageClient({"https://site0.example/page": _PageResp(_ARTICLE)})
    monkeypatch.setattr(tool, "_page_client", lambda: pages)
    await tool.execute(query="zagreb funicular", count=1, enrich=True)
    assert len(tool.client.calls) == 1
    assert pages.fetched == ["https://site0.example/page"]
