"""Web search tool powered by Brave Search API, Tavily, You.com, or Keenable.

Every engine's answer is turned into one result shape (:class:`_Hit`) and
rendered once, so the cap, the enrichment and the output budget behave the
same whichever engine is configured.

* **Cap** — one call may ask for up to ``tools.web_search.max_results_cap``
  results (50 by default; 10 when the call names no count — never fewer, so the
  ``max_results: 5`` older agent configs carry doesn't hold them back). Brave
  serves 20 per request, so a bigger ask pages through it; Tavily stops at 20
  (the result says so); You.com and Keenable take the count directly.
* **Enriched by default** — at no extra cost every result carries what the
  engine already sent beyond the one snippet: Brave's extra excerpts (all five
  it gives), all of You.com's snippets, the site and the publish date.
* **enrich=true** — passages from the pages themselves that match the query:
  Brave's LLM Context endpoint (same key, many chunks per page), Tavily's
  advanced depth, You.com's highlights. Keenable — and Brave when LLM Context
  fails — get the top pages read here (public addresses only, short timeout).
"""

from __future__ import annotations

import asyncio
import html
import os
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

# Brave Search rejects the whole request (HTTP 422) if `country` isn't one of
# its supported codes — e.g. "HR" (Croatia) is NOT supported, which silently
# breaks every regional query. Validate against this set and drop anything else
# (search globally) instead of forwarding an invalid value.
_BRAVE_COUNTRIES = frozenset({
    "ALL", "AR", "AU", "AT", "BE", "BR", "CA", "CL", "DK", "FI", "FR", "DE",
    "GR", "HK", "IN", "ID", "IT", "JP", "KR", "MY", "MX", "NL", "NZ", "NO",
    "CN", "PL", "PT", "PH", "RU", "SA", "ZA", "ES", "SE", "CH", "TW", "TR",
    "GB", "US",
})

_BRAVE_PAGE = 20          # most results Brave returns for one request
_BRAVE_MAX_OFFSET = 9     # Brave's last page index
_TAVILY_MAX = 20          # Tavily's max_results ceiling
_YOUCOM_MAX = 50          # You.com allows 100, but only 50 with highlights
_RETRY_DELAY = 1.1        # seconds before the one retry after HTTP 429
_MIN_DEFAULT_COUNT = 10   # older agent configs say max_results: 5 — ignored
_BRAVE_CONTEXT_URL = "https://api.search.brave.com/res/v1/llm/context"

_FRESHNESS_WORDS = {"pd": "day", "pw": "week", "pm": "month", "py": "year"}

# Output budget: the whole search is ONE tool result, so it has to fit the
# agent's context next to everything else (same sizing as web_fetch).
_MAX_OUTPUT_CHARS = 60_000
_SNIPPET_CHARS = 500
_EXTRA_CHARS = 300
_EXTRA_SHOWN = 5          # all of Brave's extra excerpts (its own maximum)
_PASSAGE_CHARS = 600

# Page reading for enrich=true (Brave, Keenable).
_PAGE_MAX_BYTES = 1_500_000
_MIN_LINE_CHARS = 40      # shorter lines are menus, buttons, bylines

_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "what", "which", "who",
    "how", "why", "when", "where", "are", "was", "were", "is", "of", "in", "on",
    "to", "a", "an", "or", "by", "at", "as", "be", "it", "its", "about", "into",
    "best", "top", "vs", "does", "do", "can", "you", "your", "all", "any", "not",
})


class _MissingKey(Exception):
    """The configured engine needs a key nobody set."""


@dataclass
class _Hit:
    """One search result, whichever engine sent it."""

    title: str
    url: str
    snippet: str = ""
    extra: list[str] = field(default_factory=list)
    site: str = ""
    date: str = ""
    passages: list[str] = field(default_factory=list)


def _int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _flag(value: Any) -> bool | None:
    """A model's yes/no in whatever shape it arrives; None when unsaid."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return None


def _host(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if isinstance(v, str) and v.strip()]


def _query_terms(query: str) -> set[str]:
    words = re.findall(r"[^\W_][\w'-]*", query.lower())
    terms = {w for w in words if len(w) >= 3 and w not in _STOPWORDS}
    return terms or {w for w in words if w}


def _term_hits(line: str, terms: set[str]) -> int:
    low = line.lower()
    words = set(re.findall(r"[^\W_][\w'-]*", low))
    # Whole words, or — for longer terms — any inflection containing them
    # ("museum" in "museums").
    return sum(1 for t in terms if t in words or (len(t) >= 5 and t in low))


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def _enrich_budget(cfg: Any) -> int:
    """Page text kept per enriched result, in characters."""
    return min(max(_int(getattr(cfg, "enrich_chars", 2000), 2000), 300), 8000)


def _fit(chunks: list[str], budget: int) -> list[str]:
    """As many whole chunks as *budget* holds (each cut to a passage); at
    least the first one, shortened."""
    out: list[str] = []
    used = 0
    for chunk in chunks:
        piece = _cut(chunk, _PASSAGE_CHARS)
        if used + len(piece) > budget:
            if not out:
                out.append(_cut(chunk, budget))
            break
        out.append(piece)
        used += len(piece)
    return out


_LINK_RE = re.compile(r"\(?\bhttps?://[^\s)]+\)?")

_BLOCK_TAGS = ["p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "th", "pre",
               "blockquote", "dd", "dt", "figcaption"]
_NOISE_TAGS = ["script", "style", "noscript", "template", "svg", "canvas", "nav",
               "header", "footer", "aside", "form", "button", "select"]


def _page_blocks(markup: str) -> str:
    """A page's text one paragraph per line — whole paragraphs, not the
    pieces inline tags (<a>, <code>, <strong>) would cut them into — without
    menus, headers and footers."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(markup, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    blocks: list[str] = []
    for el in soup.find_all(_BLOCK_TAGS):
        if el.find(_BLOCK_TAGS):
            continue  # a container: its inner blocks are read on their own
        text = el.get_text(" ", strip=True)
        if text:
            blocks.append(text)
    if blocks:
        return "\n".join(blocks)
    return soup.get_text("\n")  # no paragraphs at all: whatever text there is


def _focused_passages(text: str, query: str, budget: int, title: str = "") -> list[str]:
    """The page's lines that best match *query*, in page order, up to
    *budget* characters; the page's opening when nothing matches.

    Links are dropped before anything is judged: page text carries them as
    "label (url)", and a URL full of query words ("Sign in (…/asyncio/…)")
    is navigation, not an answer. The page's own title is dropped too."""
    head = re.sub(r"\s+", " ", title).strip().lower()[:40]
    lines: list[str] = []
    seen: set[str] = set()
    for raw in (text or "").splitlines():
        ln = re.sub(r"\s+", " ", _LINK_RE.sub("", raw)).strip()
        low = ln.lower()
        if len(ln) < _MIN_LINE_CHARS or (head and low.startswith(head)) or low in seen:
            continue  # too short, the page's title, or said already on the page
        seen.add(low)
        lines.append(ln)
    if not lines:
        return []
    terms = _query_terms(query)
    scored = [(n, i) for i, ln in enumerate(lines) if (n := _term_hits(ln, terms))]
    picked: list[int] = []
    used = 0
    if scored:
        # Most query words first; on a tie, the earlier line.
        for _, i in sorted(scored, key=lambda s: (-s[0], s[1])):
            size = min(len(lines[i]), _PASSAGE_CHARS)
            if used + size > budget:
                continue
            picked.append(i)
            used += size
        picked.sort()
    else:
        for i, ln in enumerate(lines):
            size = min(len(ln), _PASSAGE_CHARS)
            if used + size > budget:
                break
            picked.append(i)
            used += size
    if not picked:
        return [_cut(lines[scored[0][1] if scored else 0], budget)]
    return [_cut(lines[i], _PASSAGE_CHARS) for i in picked]


class WebSearchTool(Tool):
    """Search the web using Brave Search API, Tavily, You.com, or Keenable."""

    name = "web_search"
    description = (
        "Search the web and return ranked results with titles, links, snippets, "
        "site and date. count goes up to 50 — ask for more when you need breadth. "
        "enrich=true also returns passages from the top pages that match the query."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text",
            },
            "count": {
                "type": "number",
                "description": (
                    "How many results to return (default from config, usually 10; max 50). "
                    "Ask for 20-50 when you need breadth: a survey, a list, many sources."
                ),
            },
            "enrich": {
                "type": "boolean",
                "description": (
                    "Enriched search: also return the passages of the top results' pages "
                    "that match the query (a few seconds slower). Use it when snippets "
                    "won't answer the question — it often saves a separate web_fetch."
                ),
            },
            "offset": {
                "type": "number",
                "description": "Page of results to skip to, in units of count (default 0)",
            },
            "country": {
                "type": "string",
                "description": (
                    "Optional 2-letter country for regional ranking. Brave supports a "
                    "limited set (US, GB, DE, FR, IT, ES, etc.); unsupported codes (e.g. HR) "
                    "are ignored. For local results, prefer search_lang (e.g. 'hr') and "
                    "query terms instead."
                ),
            },
            "search_lang": {
                "type": "string",
                "description": "Search language code (example: en, hr)",
            },
            "freshness": {
                "type": "string",
                "description": "Freshness filter (example: pd, pw, pm, py)",
            },
            "safesearch": {
                "type": "string",
                "description": "SafeSearch level: off, moderate, strict",
            },
        },
        "required": ["query"],
    }

    def __init__(self):
        self.client = httpx.AsyncClient(
            follow_redirects=True,
            headers={"User-Agent": "Captain Claw/0.1.0 (Web Search Tool)"},
        )

    @staticmethod
    def _clean_text(value: str, max_chars: int = _SNIPPET_CHARS) -> str:
        """Drop markup, normalize whitespace and bound output size."""
        text = html.unescape(re.sub(r"</?[a-zA-Z][^>]*>", "", value or ""))
        cleaned = re.sub(r"\s+", " ", text).strip()
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "... [truncated]"

    # ── entry point ──────────────────────────────────────────────────

    async def execute(
        self,
        query: str,
        count: int | None = None,
        offset: int = 0,
        country: str = "",
        search_lang: str = "",
        freshness: str = "",
        safesearch: str = "",
        enrich: Any = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Run the search on the configured engine."""
        q = (query or "").strip()
        if not q:
            return ToolResult(success=False, error="Missing required query")

        search_cfg = getattr(get_config().tools, "web_search", None)
        provider = str(getattr(search_cfg, "provider", "brave") or "brave").strip().lower()

        if count is None:
            count = kwargs.get("max_results")  # models often name it this way
        cap = max(1, _int(getattr(search_cfg, "max_results_cap", 50), 50))
        default_count = max(_MIN_DEFAULT_COUNT, _int(getattr(search_cfg, "max_results", 10), 10))
        n = default_count if count is None else _int(count, default_count)
        n = min(max(n, 1), cap)
        page = max(0, _int(offset, 0))
        want_enrich = _flag(enrich)
        if want_enrich is None:
            want_enrich = bool(getattr(search_cfg, "enrich", False))
        country = str(country or "").strip()
        search_lang = str(search_lang or "").strip()
        freshness = str(freshness or "").strip()
        safesearch = str(safesearch or "").strip()

        engines = {
            "brave": ("Brave", self._brave),
            "tavily": ("Tavily", self._tavily),
            "youcom": ("You.com", self._youcom),
            "keenable": ("Keenable", self._keenable),
        }
        if provider not in engines:
            return ToolResult(success=False, error=f"Unsupported web_search provider: {provider}")
        engine, run = engines[provider]

        try:
            hits, notes = await run(
                q, search_cfg, n=n, page=page, country=country, search_lang=search_lang,
                freshness=freshness, safesearch=safesearch, enrich=want_enrich,
            )
        except _MissingKey as e:
            return ToolResult(success=False, error=str(e))
        except httpx.HTTPStatusError as e:
            detail = self._http_detail(e)
            log.error(f"{engine} web search failed", query=q, error=detail)
            return ToolResult(success=False, error=detail)
        except Exception as e:
            log.error("Web search failed", query=q, error=str(e))
            return ToolResult(success=False, error=str(e))

        hits = self._dedupe(hits)
        if want_enrich and provider in {"brave", "keenable"} and hits:
            # Brave's own page extraction first (one request, gets pages that
            # turn plain HTTP clients away); reading the pages here is the net.
            by_brave = provider == "brave" and await self._brave_context(
                hits, q, search_cfg, country=country, search_lang=search_lang)
            read, tried = (0, 0) if by_brave else await self._read_pages(hits, q, search_cfg)
            if tried and read < tried:
                notes.append(
                    f"Enriched {read} of the top {tried} results; the other pages "
                    "didn't answer in time or had no readable text."
                )

        return ToolResult(
            success=True,
            content=self._render(engine, q, hits, notes, self._budget(kwargs.get("_agent"))),
        )

    # ── engines ──────────────────────────────────────────────────────

    async def _brave(self, q: str, cfg: Any, *, n: int, page: int, country: str,
                     search_lang: str, freshness: str, safesearch: str,
                     enrich: bool) -> tuple[list[_Hit], list[str]]:
        api_key = (
            str(getattr(cfg, "api_key", "") or "").strip()
            or str(os.environ.get("BRAVE_API_KEY", "")).strip()
        )
        if not api_key:
            raise _MissingKey(
                "Missing Brave API key. Set tools.web_search.api_key in config "
                "or BRAVE_API_KEY environment variable."
            )

        base_url = str(
            getattr(cfg, "base_url", "https://api.search.brave.com/res/v1/web/search")
            or "https://api.search.brave.com/res/v1/web/search"
        ).strip()
        timeout = float(getattr(cfg, "timeout", 20) or 20)
        safe_value = (safesearch or str(getattr(cfg, "safesearch", "moderate") or "moderate")).lower()
        if safe_value not in {"off", "moderate", "strict"}:
            safe_value = "moderate"

        params: dict[str, Any] = {"q": q, "safesearch": safe_value}
        if bool(getattr(cfg, "extra_snippets", True)):
            params["extra_snippets"] = "true"
        if country:
            _cc = country.upper()
            if _cc in _BRAVE_COUNTRIES:
                params["country"] = _cc
            else:
                # Unsupported by Brave (e.g. HR) — drop it and search globally
                # rather than 422 the whole request.
                log.debug("web_search: dropping unsupported Brave country %r", country)
        if search_lang:
            params["search_lang"] = search_lang
        if freshness:
            params["freshness"] = freshness

        headers = {"Accept": "application/json", "X-Subscription-Token": api_key}

        # Brave's offset counts pages of `count`. Up to 20 results is one
        # request, as before; more is pages of 20 covering the same window
        # (results page*n … page*n+n-1), trimmed to it afterwards.
        if n <= _BRAVE_PAGE:
            requests = [(n, page)]
            skip = 0
        else:
            start = page * n
            first, last = start // _BRAVE_PAGE, (start + n - 1) // _BRAVE_PAGE
            requests = [(_BRAVE_PAGE, p) for p in range(first, min(last, _BRAVE_MAX_OFFSET) + 1)]
            skip = start - first * _BRAVE_PAGE
            if not requests:
                return [], [f"Brave serves at most {(_BRAVE_MAX_OFFSET + 1) * _BRAVE_PAGE} "
                            "results per query; this offset is past them."]

        hits: list[_Hit] = []
        notes: list[str] = []
        for i, (cnt, off) in enumerate(requests):
            try:
                payload = await self._get_json(
                    base_url, params={**params, "count": cnt, "offset": off},
                    headers=headers, timeout=timeout,
                )
            except httpx.HTTPStatusError as e:
                if i == 0:
                    raise
                # Later pages are a bonus: keep what came back.
                notes.append(f"Brave stopped after {len(hits)} results ({self._http_detail(e)}).")
                break
            web_block = payload.get("web", {}) if isinstance(payload, dict) else {}
            results = web_block.get("results", []) if isinstance(web_block, dict) else []
            results = [r for r in results if isinstance(r, dict)] if isinstance(results, list) else []
            hits.extend(self._brave_hit(r) for r in results)
            more = (payload.get("query") or {}).get("more_results_available") \
                if isinstance(payload, dict) else None
            if more is False or len(results) < cnt:
                break
        if n > _BRAVE_PAGE:
            hits = hits[skip:skip + n]
        return hits, notes

    def _brave_hit(self, item: dict) -> _Hit:
        url = str(item.get("url", "") or "").strip()
        meta = item.get("meta_url") if isinstance(item.get("meta_url"), dict) else {}
        profile = item.get("profile") if isinstance(item.get("profile"), dict) else {}
        return _Hit(
            title=self._clean_text(str(item.get("title", "") or "Untitled"), max_chars=180),
            url=url,
            snippet=self._clean_text(str(item.get("description", "") or "")),
            extra=[self._clean_text(s, _EXTRA_CHARS) for s in _strings(item.get("extra_snippets"))],
            site=str(profile.get("name") or meta.get("hostname") or _host(url)).strip(),
            date=str(item.get("age") or item.get("page_age") or "").strip(),
        )

    async def _brave_context(self, hits: list[_Hit], q: str, cfg: Any, *,
                             country: str, search_lang: str) -> bool:
        """Fill ``passages`` from Brave's LLM Context endpoint: query-matched
        chunks of each page, as many as the per-page budget holds (Brave's web
        search gives at most five extra excerpts; this is how to get more).
        True when it answered; never raises."""
        if not bool(getattr(cfg, "brave_llm_context", True)):
            return False
        api_key = (
            str(getattr(cfg, "api_key", "") or "").strip()
            or str(os.environ.get("BRAVE_API_KEY", "")).strip()
        )
        pages = min(max(_int(getattr(cfg, "enrich_pages", 5), 5), 1), 20)
        budget = _enrich_budget(cfg)
        per_url = min(max(budget // 4, 512), 8192)  # ≈4 characters a token
        params: dict[str, Any] = {
            "q": q,
            "count": min(max(len(hits), pages), 50),
            "maximum_number_of_urls": pages,
            "maximum_number_of_tokens_per_url": per_url,
            "maximum_number_of_tokens": min(max(per_url * pages, 1024), 32768),
        }
        if country and country.upper() in _BRAVE_COUNTRIES:
            params["country"] = country.upper()
        if search_lang:
            params["search_lang"] = search_lang
        try:
            data = await self._get_json(
                str(getattr(cfg, "llm_context_url", "") or _BRAVE_CONTEXT_URL).strip(),
                params=params,
                headers={"Accept": "application/json", "X-Subscription-Token": api_key},
                timeout=float(getattr(cfg, "timeout", 20) or 20),
            )
        except Exception as e:  # noqa: BLE001 — plan without it, outage: read pages instead
            detail = self._http_detail(e) if isinstance(e, httpx.HTTPStatusError) else str(e)
            log.warning("Brave LLM Context failed; reading pages instead", error=detail)
            return False
        grounding = data.get("grounding") if isinstance(data, dict) else None
        generic = grounding.get("generic") if isinstance(grounding, dict) else None
        if not isinstance(generic, list):
            return False
        by_url = {h.url.rstrip("/").lower(): h for h in hits}
        filled = 0
        for item in generic:
            if not isinstance(item, dict):
                continue
            hit = by_url.get(str(item.get("url") or "").rstrip("/").lower())
            chunks = [self._clean_text(c, 10_000) for c in _strings(item.get("snippets"))]
            if hit is None or not chunks:
                continue
            hit.passages = _fit(chunks, budget)
            filled += 1
        return filled > 0

    async def _tavily(self, q: str, cfg: Any, *, n: int, page: int, country: str,
                      search_lang: str, freshness: str, safesearch: str,
                      enrich: bool) -> tuple[list[_Hit], list[str]]:
        api_key = (
            str(getattr(cfg, "tavily_api_key", "") or "").strip()
            or str(os.environ.get("TAVILY_API_KEY", "")).strip()
        )
        if not api_key:
            raise _MissingKey(
                "Missing Tavily API key. Set tools.web_search.tavily_api_key in config "
                "or TAVILY_API_KEY environment variable."
            )
        if any([page, country, safesearch]):
            log.debug("Tavily provider ignores offset/country/safesearch parameters")
        notes: list[str] = []
        if n > _TAVILY_MAX:
            notes.append(f"Tavily returns at most {_TAVILY_MAX} results per search; asked for {n}.")

        payload: dict[str, Any] = {
            "query": q,
            "max_results": min(n, _TAVILY_MAX),
            # advanced = query-matched chunks of ~800 chars per page (2 credits).
            "search_depth": "advanced" if enrich else "basic",
            "include_answer": False,
        }
        if enrich:
            payload["chunks_per_source"] = 3
        time_range = _FRESHNESS_WORDS.get(freshness.lower()) if freshness else None
        if time_range:
            payload["time_range"] = time_range
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

        data = await self._post_json(
            "https://api.tavily.com/search", payload=payload, headers=headers,
            timeout=float(getattr(cfg, "timeout", 20) or 20),
        )
        results = data.get("results", []) if isinstance(data, dict) else []
        hits: list[_Hit] = []
        for item in results if isinstance(results, list) else []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "") or "").strip()
            content = str(item.get("content", "") or "")
            hit = _Hit(
                title=self._clean_text(str(item.get("title", "") or "Untitled"), max_chars=180),
                url=url,
                site=_host(url),
                date=str(item.get("published_date") or "").strip(),
            )
            chunks = [c for c in (self._clean_text(p, 10_000) for p in content.split("[...]")) if c]
            if enrich and chunks:
                hit.snippet = _cut(chunks[0], _SNIPPET_CHARS)
                hit.passages = [_cut(c, _PASSAGE_CHARS) for c in chunks[1:]]
            else:
                hit.snippet = self._clean_text(content)
            hits.append(hit)
        return hits, notes

    async def _keenable(self, q: str, cfg: Any, *, n: int, page: int, country: str,
                        search_lang: str, freshness: str, safesearch: str,
                        enrich: bool) -> tuple[list[_Hit], list[str]]:
        """Keenable works without an API key (keyless public endpoint). If
        tools.web_search.keenable_api_key or KEENABLE_API_KEY is set, the
        keyed endpoint is used instead."""
        if any([page, country, freshness, safesearch]):
            log.debug("Keenable provider ignores offset/country/freshness/safesearch parameters")
        api_key = (
            str(getattr(cfg, "keenable_api_key", "") or "").strip()
            or str(os.environ.get("KEENABLE_API_KEY", "")).strip()
        )
        headers = {"Content-Type": "application/json", "X-Keenable-Title": "Captain Claw"}
        url = "https://api.keenable.ai/v1/search/public"
        if api_key:
            headers["X-API-Key"] = api_key
            url = "https://api.keenable.ai/v1/search"

        data = await self._post_json(
            url, payload={"query": q, "max_results": n}, headers=headers,
            timeout=float(getattr(cfg, "timeout", 20) or 20),
        )
        results = data.get("results", []) if isinstance(data, dict) else []
        hits: list[_Hit] = []
        for item in results if isinstance(results, list) else []:
            if not isinstance(item, dict) or not item.get("url"):
                continue
            link = str(item.get("url", "") or "").strip()
            hits.append(_Hit(
                title=self._clean_text(str(item.get("title", "") or "Untitled"), max_chars=180),
                url=link,
                snippet=self._clean_text(
                    str(item.get("description", "") or item.get("snippet", "") or "")
                ),
                site=_host(link),
                date=str(item.get("published_date") or item.get("date") or "").strip(),
            ))
        return hits, []

    async def _youcom(self, q: str, cfg: Any, *, n: int, page: int, country: str,
                      search_lang: str, freshness: str, safesearch: str,
                      enrich: bool) -> tuple[list[_Hit], list[str]]:
        api_key = (
            str(getattr(cfg, "youcom_api_key", "") or "").strip()
            or str(os.environ.get("YDC_API_KEY", "")).strip()
        )
        if not api_key:
            raise _MissingKey(
                "Missing You.com API key. Set tools.web_search.youcom_api_key in config "
                "or YDC_API_KEY environment variable."
            )
        notes: list[str] = []
        payload: dict[str, Any] = {"query": q, "count": min(n, _YOUCOM_MAX)}
        if page:
            if page <= 9:
                payload["offset"] = page  # in multiples of count, like Brave
            else:
                notes.append("You.com pages stop at offset 9; showing the first page.")
        # You.com uses off/moderate/strict (moderate is the API default).
        safe_value = (safesearch or str(getattr(cfg, "safesearch", "moderate") or "moderate")).lower()
        if safe_value not in {"off", "moderate", "strict"}:
            safe_value = "moderate"
        payload["safesearch"] = safe_value
        if country:
            payload["country"] = country.upper()
        if freshness:
            # Tool accepts pd/pw/pm/py; the API wants day/week/month/year words.
            payload["freshness"] = _FRESHNESS_WORDS.get(freshness.lower(), freshness)
        if enrich:
            # Query-matched passages from each page (contents.highlights).
            payload["extraction"] = {
                "extraction_mode": "highlights",
                "highlights": {"total_max_tokens": 8192},
            }
        headers = {"Content-Type": "application/json", "X-API-Key": api_key}

        data = await self._post_json(
            "https://ydc-index.io/v1/search", payload=payload, headers=headers,
            timeout=float(getattr(cfg, "timeout", 20) or 20),
        )
        # The response contains a `results` object with `web` (and, for
        # news-intent queries, `news`) arrays.
        sections = data.get("results", {}) if isinstance(data, dict) else {}
        results: list[Any] = []
        if isinstance(sections, dict):
            results = [*(sections.get("web") or []), *(sections.get("news") or [])]
        elif isinstance(sections, list):
            results = sections

        hits: list[_Hit] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "") or "").strip()
            snippets = [self._clean_text(s) for s in _strings(item.get("snippets"))]
            contents = item.get("contents") if isinstance(item.get("contents"), dict) else {}
            hits.append(_Hit(
                title=self._clean_text(str(item.get("title", "") or "Untitled"), max_chars=180),
                url=url,
                snippet=snippets[0] if snippets else self._clean_text(str(item.get("description", "") or "")),
                extra=[_cut(s, _EXTRA_CHARS) for s in snippets[1:]],
                site=_host(url),
                date=str(item.get("page_age") or "").strip(),
                passages=[
                    _cut(self._clean_text(h, 10_000), _PASSAGE_CHARS)
                    for h in _strings(contents.get("highlights"))
                ],
            ))
        return hits, notes

    # ── HTTP ─────────────────────────────────────────────────────────

    async def _get_json(self, url: str, *, params: dict, headers: dict, timeout: float) -> Any:
        for attempt in (0, 1):
            response = await self.client.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code == 429 and attempt == 0:
                await asyncio.sleep(_RETRY_DELAY)  # per-second plan limits
                continue
            response.raise_for_status()
            return response.json()
        return {}  # pragma: no cover — the loop always returns or raises

    async def _post_json(self, url: str, *, payload: dict, headers: dict, timeout: float) -> Any:
        for attempt in (0, 1):
            response = await self.client.post(url, json=payload, headers=headers, timeout=timeout)
            if response.status_code == 429 and attempt == 0:
                await asyncio.sleep(_RETRY_DELAY)
                continue
            response.raise_for_status()
            return response.json()
        return {}  # pragma: no cover

    def _http_detail(self, e: httpx.HTTPStatusError) -> str:
        body = ""
        try:
            body = (e.response.text or "").strip()
        except Exception:
            body = ""
        detail = f"HTTP {e.response.status_code}" if e.response is not None else str(e)
        if body:
            detail = f"{detail}: {self._clean_text(body, max_chars=300)}"
        return detail

    # ── enrichment: read the top pages (Brave, Keenable) ────────────

    def _page_client(self) -> httpx.AsyncClient:
        """Public addresses only: a search result must never become a way
        into the machine's own network."""
        from captain_claw import speaker as _speaker

        return _speaker.make_public_http_client()

    async def _read_pages(self, hits: list[_Hit], query: str, cfg: Any) -> tuple[int, int]:
        """Fill ``passages`` on the top results from their pages. Returns
        ``(pages read, pages tried)``; never raises."""
        pages = min(max(_int(getattr(cfg, "enrich_pages", 5), 5), 0), 10)
        budget = _enrich_budget(cfg)
        timeout = float(getattr(cfg, "enrich_timeout", 8) or 8)
        targets = [h for h in hits[:pages]
                   if not h.passages and h.url.lower().startswith(("http://", "https://"))]
        if not targets:
            return 0, 0

        async def _one(client: httpx.AsyncClient, hit: _Hit) -> bool:
            try:
                text = await asyncio.wait_for(self._page_text(client, hit.url), timeout)
            except Exception as e:  # noqa: BLE001 — one slow or odd page never fails the search
                log.debug("web_search enrich: page not read", url=hit.url, error=str(e))
                return False
            hit.passages = _focused_passages(text, query, budget, title=hit.title)
            return bool(hit.passages)

        try:
            async with self._page_client() as client:
                done = await asyncio.gather(*(_one(client, h) for h in targets))
        except Exception as e:  # noqa: BLE001
            log.warning("web_search enrich failed", error=str(e))
            return 0, len(targets)
        return sum(1 for d in done if d), len(targets)

    async def _page_text(self, client: httpx.AsyncClient, url: str) -> str:
        async with client.stream("GET", url) as response:
            if response.status_code >= 400:
                return ""
            ctype = (response.headers.get("content-type") or "").lower()
            if ctype and not any(k in ctype for k in ("html", "text/plain", "xml")):
                return ""  # PDFs, images, downloads — web_fetch/pdf tools read those
            chunks: list[bytes] = []
            size = 0
            async for chunk in response.aiter_bytes():
                chunks.append(chunk)
                size += len(chunk)
                if size >= _PAGE_MAX_BYTES:
                    break
            body = b"".join(chunks)[:_PAGE_MAX_BYTES].decode(
                response.encoding or "utf-8", errors="replace")
        if "html" in ctype or body.lstrip()[:1] == "<":
            # Parsing a big page takes a moment: off the event loop.
            return await asyncio.to_thread(_page_blocks, body)
        return body

    # ── output ───────────────────────────────────────────────────────

    @staticmethod
    def _dedupe(hits: list[_Hit]) -> list[_Hit]:
        seen: set[str] = set()
        out: list[_Hit] = []
        for h in hits:
            key = h.url.rstrip("/").lower()
            if key and key in seen:
                continue
            seen.add(key)
            out.append(h)
        return out

    @staticmethod
    def _budget(agent: Any) -> int:
        try:
            from captain_claw.tools.web_fetch import _context_char_ceiling

            ceiling = _context_char_ceiling(agent)
        except Exception:  # noqa: BLE001 — sizing never breaks a search
            ceiling = 0
        return min(_MAX_OUTPUT_CHARS, ceiling) if ceiling else _MAX_OUTPUT_CHARS

    @staticmethod
    def _block(idx: int, hit: _Hit) -> str:
        lines = [f"{idx}. {hit.title or 'Untitled'}", f"   URL: {hit.url or '-'}"]
        source = " · ".join(p for p in (hit.site, hit.date) if p)
        if source:
            lines.append(f"   Source: {source}")
        lines.append(f"   Snippet: {hit.snippet or '-'}")
        extra = [e for e in hit.extra if e and e != hit.snippet][:_EXTRA_SHOWN]
        if extra:
            lines.append(f"   More: {' | '.join(extra)}")
        if hit.passages:
            lines.append("   From the page:")
            lines.extend(f"     - {p}" for p in hit.passages)
        return "\n".join(lines) + "\n"

    def _render(self, engine: str, q: str, hits: list[_Hit], notes: list[str],
                budget: int) -> str:
        head = [f"[SEARCH ENGINE: {engine}]", f"[QUERY: {q}]", f"[RESULTS: {len(hits)}]"]
        head += [f"[NOTE: {n}]" for n in notes]
        head.append("")
        if not hits:
            head.append("No results found.")
            return "\n".join(head).strip()

        out = "\n".join(head) + "\n"
        room = budget - 120  # what the closing "left out" note needs
        compact = False
        for idx, hit in enumerate(hits, start=1):
            block = self._block(idx, hit) + "\n"
            if not compact and len(out) + len(block) > room:
                compact = True  # the rest as title + link + a short snippet
            if compact:
                block = (f"{idx}. {hit.title or 'Untitled'}\n   URL: {hit.url or '-'}\n"
                         f"   Snippet: {_cut(hit.snippet, 160) or '-'}\n\n")
                if len(out) + len(block) > room:
                    out += (f"[… {len(hits) - idx + 1} more results left out to fit "
                            "the context. Narrow the query or lower count to see them.]")
                    break
            out += block
        return out.strip()

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
