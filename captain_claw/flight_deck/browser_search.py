"""Web searches through the person's own Chrome — instead of a paid search API.

Opt-in on both ends, off until the person turns it on:

* **Flight Deck** — Profile → *Web searches*: the person picks a search engine
  (Google, Bing, DuckDuckGo, Yahoo, Qwant) for their agents' ``web_search``.
  Kept per person in their settings (``web-search:chrome``).
* **Captain for Chrome** — its panel asks once to read that engine's result
  pages (Chrome's own permission prompt for that one site). Until then this
  browser serves no searches.

Then an agent's ``web_search`` asks here first (``POST
/fd/browser-hands/agent/search``). Flight Deck builds the engine's result-page
address and hands it to the person's browser over the socket Captain for Chrome
keeps open (``browser_hands``); the extension opens it in a background tab
(never brought to the front, in a collapsed "Captain · search" group), waits
for the results, sends back the page's HTML without scripts and styles, and
closes the tab. The page is read *here* — so a search engine that changes its
markup needs a Flight Deck update, not a new extension.

Anything short of results — Chrome not open, the engine asking to prove the
person isn't a robot, a cookie question, a page that never showed results —
answers ``ok: false`` with the reason, and the agent's tool searches with the
deck's API engine (Brave …) instead. A shared agent's member turns never come
here (the tool doesn't ask; the grant guard refuses member markers anyway): the
owner's browser is theirs alone.

Engines left out on purpose: Brave Search hides its result links ("url-cloak"
— the address is only known on a real click), Startpage and Ecosia answer
automated checks with bot walls their pages couldn't be read through.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlsplit

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from captain_claw.flight_deck.auth import get_current_user, get_db

log = logging.getLogger(__name__)

owner_router = APIRouter(prefix="/fd/web-search", tags=["web-search"])

SETTING_KEY = "web-search:chrome"
PAGE_WAIT_S = 25.0            # one result page in the person's browser
MAX_PAGES = 5                 # pages one search may load (10 results each)
SEARCH_BUDGET_S = 45.0        # one search, all its pages (the agent's tool waits 55 s)
SEARCHES_AT_ONCE = 2          # per browser: a person's engine sees a person's pace
RESOLVE_TIMEOUT_S = 5.0
RESOLVE_AT_ONCE = 8


@dataclass(frozen=True)
class Engine:
    id: str
    name: str
    host: str                         # the result pages' address (the extension checks it)
    per_page: int
    max_pages: int
    wait_for: str                     # CSS: the results are on the page
    block_for: str                    # CSS: the engine is checking for robots instead
    build: Callable[..., str]
    parse: Callable[[Any], list[dict]]

    def view(self) -> dict:
        return {"id": self.id, "name": self.name, "host": self.host}


# ── addresses ──────────────────────────────────────────────────────

_FRESH = {"pd": "d", "pw": "w", "pm": "m", "py": "y"}
_EN_COUNTRIES = {"US", "GB", "UK", "AU", "CA", "NZ", "IE", "ZA", "IN", "SG"}


def _q(**params: Any) -> str:
    return "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items() if v not in (None, ""))


def _google_url(q: str, page: int, lang: str, country: str, fresh: str, safe: str) -> str:
    return "https://www.google.com/search?" + _q(
        q=q, start=page * 10 or None, hl=lang, gl=country.lower(),
        tbs=f"qdr:{_FRESH[fresh]}" if fresh in _FRESH else None,
        safe={"strict": "active", "off": "off"}.get(safe))


def _bing_url(q: str, page: int, lang: str, country: str, fresh: str, safe: str) -> str:
    ez = {"pd": "ez1", "pw": "ez2", "pm": "ez3"}.get(fresh)
    return "https://www.bing.com/search?" + _q(
        q=q, first=page * 10 + 1 if page else None, setlang=lang, cc=country.lower(),
        filters=f'ex1:"{ez}"' if ez else None,
        adlt={"strict": "strict", "off": "off"}.get(safe))


def _ddg_url(q: str, page: int, lang: str, country: str, fresh: str, safe: str) -> str:
    region = ""
    if country:
        cc = country.lower()
        region = f"{'uk' if cc == 'gb' else cc}-{lang.lower() or ('en' if country.upper() in _EN_COUNTRIES else cc)}"
    return "https://html.duckduckgo.com/html/?" + _q(
        q=q, kl=region, df=_FRESH.get(fresh),
        kp={"strict": "1", "off": "-2"}.get(safe))


def _yahoo_url(q: str, page: int, lang: str, country: str, fresh: str, safe: str) -> str:
    return "https://search.yahoo.com/search?" + _q(
        p=q, b=page * 10 + 1 if page else None, vl=f"lang_{lang.lower()}" if lang else None,
        btf={"pd": "d", "pw": "w", "pm": "m"}.get(fresh))


def _qwant_url(q: str, page: int, lang: str, country: str, fresh: str, safe: str) -> str:
    locale = f"{lang.lower()}_{country.upper()}" if lang and country else ""
    return "https://www.qwant.com/?" + _q(
        q=q, t="web", locale=locale,
        freshness={"pd": "day", "pw": "week", "pm": "month"}.get(fresh),
        s={"strict": "2", "off": "0"}.get(safe))


# ── reading result pages ───────────────────────────────────────────

_WS = re.compile(r"\s+")
# "May 5, 2024 — …", "6 days ago · …": a date the engine put before the snippet.
_DATE_HEAD = re.compile(r"^(?P<date>[^—·|]{0,24}?\d[^—·|]{0,24}?)\s+[—·]\s+(?P<rest>.+)$", re.S)


_BLOCK_TAGS = frozenset({"div", "p", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
                         "tr", "td", "th", "section", "article", "header", "footer", "cite", "table"})


def _text(el: Any) -> str:
    """An element's text as read: words inside a line kept together
    (``<em>asyncio</em>.gather`` is "asyncio.gather"), blocks apart."""
    if el is None:
        return ""
    from bs4 import Comment, NavigableString

    parts: list[str] = []
    for node in el.descendants:
        if isinstance(node, NavigableString):
            if not isinstance(node, Comment):
                parts.append(str(node))
        elif node.name in _BLOCK_TAGS:
            parts.append(" ")
    return _WS.sub(" ", "".join(parts)).strip()


def _host(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _b64url(text: str) -> str:
    text = text + "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text.encode("ascii")).decode("utf-8", "replace")


def unwrap(url: str) -> str:
    """The result's own address out of an engine's click-tracking one, where
    the address is in it (Google /url?q=, Bing ck/a u=a1…, DuckDuckGo
    l/?uddg=, Yahoo RU=…). Google's opaque /goto stays as is (resolve_links)."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    host = (parts.hostname or "").lower()
    params = parse_qs(parts.query)
    if host.endswith("google.com") and parts.path == "/url":
        for key in ("q", "url"):
            val = (params.get(key) or [""])[0]
            if val.startswith(("http://", "https://")):
                return val
    if host.endswith("bing.com") and parts.path.startswith("/ck/"):
        val = (params.get("u") or [""])[0]
        if val.startswith("a1"):
            try:
                real = _b64url(val[2:])
            except (ValueError, UnicodeError):
                real = ""
            if real.startswith(("http://", "https://")):
                return real
    if host.endswith("duckduckgo.com") and parts.path.startswith("/l/"):
        val = (params.get("uddg") or [""])[0]
        if val.startswith(("http://", "https://")):
            return val
    if host.endswith("yahoo.com") and "/RU=" in parts.path:
        m = re.search(r"/RU=([^/]+)/", parts.path + "/")
        if m:
            val = unquote(m.group(1))
            if val.startswith(("http://", "https://")):
                return val
    return url


def _split_date(snippet: str) -> tuple[str, str]:
    m = _DATE_HEAD.match(snippet)
    if m and len(m.group("date")) <= 32 and len(m.group("rest")) >= 20:
        return m.group("date").strip(), m.group("rest").strip()
    return "", snippet


def _hit(title: str, url: str, snippet: str, site: str = "", date: str = "") -> dict | None:
    url = unwrap(url.strip())
    if not title or not url.startswith(("http://", "https://")):
        return None
    if not date:
        date, snippet = _split_date(snippet)
    return {"title": title[:300], "url": url, "snippet": snippet[:1200], "site": site[:120] or _host(url),
            "date": date[:40]}


def _google_internal(url: str) -> bool:
    """A link to Google's own pages (images, maps, "People also ask"), not a result."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    return (host == "google.com" or host.endswith(".google.com")) and parts.path not in ("/goto", "/url")


def _parse_google(soup: Any) -> list[dict]:
    out: list[dict] = []
    seen: set[int] = set()
    for h3 in soup.select("a[href] h3"):
        a = h3.find_parent("a")
        box = a   # the widest block that holds this one title
        while box.parent is not None and len(box.parent.select("a[href] h3")) == 1:
            box = box.parent
        if id(box) in seen:
            continue
        seen.add(id(box))
        url = str(a.get("href") or "")
        if _google_internal(url):
            continue
        title, site = _text(h3), _text(box.select_one(".VuuXrf"))
        date = _text(box.select_one(".YrbPuc")).rstrip(" —").strip()
        snips = box.select("[data-sncf] .VwiC3b, [data-sncf]") or box.select(".VwiC3b")
        snippet = ""
        for el in snips:
            for junk in el.select(".YrbPuc, a"):
                junk.decompose()
            snippet = _text(el)
            if snippet:
                break
        if not snippet:   # whatever text the block has besides the title and the address
            for junk in box.select("h3, cite, .VuuXrf, [role=button]"):
                junk.decompose()
            snippet = _text(box)
        hit = _hit(title, url, snippet, site=site, date=date)
        if hit:
            out.append(hit)
    return out


def _parse_bing(soup: Any) -> list[dict]:
    out = []
    for li in soup.select("#b_results > li.b_algo, li.b_algo"):
        a = li.select_one("h2 a[href]")
        if a is None:
            continue
        cap = li.select_one(".b_caption p, p[class*=b_lineclamp], .b_algoSlug, .b_caption")
        for junk in (cap.select(".news_dt, .algoSlug_icon, a") if cap else []):   # "Read more"
            junk.decompose()
        site = _text(li.select_one(".tptt")) or ""
        hit = _hit(_text(a), str(a.get("href") or ""), _text(cap), site=site)
        if hit and not any(h["url"] == hit["url"] for h in out):
            out.append(hit)
    return out


def _parse_ddg(soup: Any) -> list[dict]:
    out = []
    for res in soup.select(".result"):
        classes = res.get("class") or []
        if "result--ad" in classes or "result--ad--small" in classes:
            continue
        a = res.select_one("a.result__a[href]")
        if a is None:
            continue
        hit = _hit(_text(a), str(a.get("href") or ""), _text(res.select_one(".result__snippet")))
        if hit:
            out.append(hit)
    return out


def _parse_yahoo(soup: Any) -> list[dict]:
    out = []
    for box in soup.select("#web .algo, div.algo"):
        a = box.select_one("h3 a[href], .compTitle a[href]")
        if a is None:
            continue
        title_el = box.select_one("h3") or a
        for junk in title_el.select("span.d-ib, .fz-ms, cite"):
            junk.decompose()   # the site's address printed inside the title link
        snippet = _text(box.select_one(".compText p, .compText"))
        hit = _hit(_text(title_el), str(a.get("href") or ""), snippet)
        if hit and not any(h["url"] == hit["url"] for h in out):
            out.append(hit)
    return out


def _parse_qwant(soup: Any) -> list[dict]:
    out = []
    for box in soup.select('[data-testid="webResult"]'):
        a = box.select_one("h2 a[href]") or box.select_one("a[href]")
        if a is None:
            continue
        title, url = _text(box.select_one("h2")) or _text(a), str(a.get("href") or "")
        site = _text(box.select_one('[data-testid="domain"] a'))
        for junk in box.select('[data-testid="domain"], [data-testid="siteLinkEnhancedItem"], h2'):
            junk.decompose()
        hit = _hit(title, url, _text(box), site=site)
        if hit:
            out.append(hit)
    return out


ENGINES: dict[str, Engine] = {e.id: e for e in (
    Engine("google", "Google", "www.google.com", 10, MAX_PAGES,
           wait_for="#search a h3, #rso a h3, #botstuff",
           block_for="#captcha-form, form[action*='sorry'], #recaptcha",
           build=_google_url, parse=_parse_google),
    Engine("bing", "Bing", "www.bing.com", 10, MAX_PAGES,
           wait_for="#b_results li.b_algo, #b_results .b_no",
           block_for="#b_captcha, .captcha, #challenge",
           build=_bing_url, parse=_parse_bing),
    Engine("duckduckgo", "DuckDuckGo", "html.duckduckgo.com", 10, 1,
           wait_for=".result, .no-results",
           block_for="#challenge-form, .anomaly-modal__modal",
           build=_ddg_url, parse=_parse_ddg),
    Engine("yahoo", "Yahoo", "search.yahoo.com", 7, MAX_PAGES,
           wait_for="#web .algo, #web .zrp, .compNoResult",
           block_for="#captcha, .captcha",
           build=_yahoo_url, parse=_parse_yahoo),
    Engine("qwant", "Qwant", "www.qwant.com", 10, 1,
           wait_for='[data-testid="webResult"], [data-testid="noResults"]',
           block_for="#captcha, .captcha",
           build=_qwant_url, parse=_parse_qwant),
)}


def read_page(engine: Engine, html: str, final_url: str = "") -> tuple[list[dict], str]:
    """``(results, why there are none)`` from a result page's HTML."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    reason = blocked_reason(engine, soup, final_url)
    if reason:
        return [], reason
    try:
        hits = engine.parse(soup)
    except Exception:   # a page shaped like nothing we know: no results, not a crash
        log.warning("Reading a %s result page failed", engine.name, exc_info=True)
        hits = []
    return hits, ""


def blocked_reason(engine: Engine, soup: Any, final_url: str = "") -> str:
    """Why the page isn't results: a robot check, a cookie question — else ''."""
    path = (urlsplit(final_url).path if final_url else "").lower()
    host = (urlsplit(final_url).hostname or "").lower() if final_url else ""
    if "/sorry/" in path or (engine.block_for and soup.select_one(engine.block_for)):
        return (f"{engine.name} asked to confirm a person is searching (a robot check). "
                f"Open {engine.name} in that Chrome and answer it; searches go on after that.")
    if host.startswith("consent.") or "consent" in path:
        return (f"{engine.name} wants its cookie question answered first. "
                f"Open {engine.name} in that Chrome once and answer it.")
    return ""


async def resolve_links(hits: list[dict]) -> None:
    """Google's opaque /goto links: the address they send to (asked without
    cookies, not followed). A link that doesn't answer stays a working
    redirect."""
    todo = [h for h in hits if _host(h["url"]) == "google.com" and urlsplit(h["url"]).path == "/goto"]
    if not todo:
        return
    gate = asyncio.Semaphore(RESOLVE_AT_ONCE)

    async def one(client: httpx.AsyncClient, hit: dict) -> None:
        async with gate:
            try:
                resp = await client.get(hit["url"])
            except httpx.HTTPError:
                return
        where = resp.headers.get("location") or ""
        if resp.status_code in (301, 302, 303, 307, 308) and where:
            real = unwrap(urljoin(hit["url"], where))
            if real.startswith(("http://", "https://")) and _host(real) != "google.com":
                hit["url"] = real
                if hit.get("site") in ("", "google.com"):
                    hit["site"] = _host(real)

    async with httpx.AsyncClient(follow_redirects=False, timeout=RESOLVE_TIMEOUT_S,
                                 headers={"User-Agent": "Mozilla/5.0"}) as client:
        await asyncio.gather(*(one(client, h) for h in todo))


# ── the person's choice ────────────────────────────────────────────


async def chosen_engine(user_id: str) -> Engine | None:
    """The engine the person picked for searches through their Chrome; None = off."""
    try:
        raw = await get_db().get_setting(str(user_id), SETTING_KEY)
    except Exception:
        return None
    try:
        data = json.loads(raw) if raw else {}
    except ValueError:
        data = {}
    eid = str((data or {}).get("engine") or "") if isinstance(data, dict) else ""
    return ENGINES.get(eid)


def _browser_rows(user_id: str, engine: Engine | None) -> list[dict]:
    from captain_claw.flight_deck.browser_hands import live_connections

    out = []
    for c in live_connections(user_id):
        if c.agent_slug:
            continue
        out.append({"browser": c.browser, "version": c.version, "engines": list(c.search_engines),
                    "ready": bool(engine and engine.id in c.search_engines)})
    return out


@owner_router.get("/chrome")
async def get_chrome_search(user: dict = Depends(get_current_user)) -> dict:
    """The person's setting, the engines on offer, and which of their browsers
    serve which engines right now."""
    from captain_claw.flight_deck.browser_clip import token_kind

    engine = await chosen_engine(str(user["id"]))
    try:
        rows = await get_db().list_pats(str(user["id"]))
    except Exception:
        rows = []
    return {"engine": engine.id if engine else "",
            "engines": [e.view() for e in ENGINES.values()],
            "browsers": _browser_rows(str(user["id"]), engine),
            "extension_set_up": any(token_kind(r.get("name")) == "browser" for r in rows)}


class ChromeSearchBody(BaseModel):
    engine: str = Field("", max_length=40)


@owner_router.put("/chrome")
async def put_chrome_search(body: ChromeSearchBody, user: dict = Depends(get_current_user)) -> dict:
    engine = body.engine.strip().lower()
    if engine and engine not in ENGINES:
        raise HTTPException(422, "No such search engine.")
    db = get_db()
    if engine:
        await db.set_settings(str(user["id"]), {SETTING_KEY: json.dumps({"engine": engine})})
    else:
        await db.delete_setting(str(user["id"]), SETTING_KEY)
    return await get_chrome_search(user)


async def clip_note(user_id: str) -> dict | None:
    """What the extension's panel needs (``/fd/clip/me``): the engine the person
    picked, so it can ask Chrome for that site once. None when it's off."""
    engine = await chosen_engine(user_id)
    return {"engine": engine.id, "name": engine.name, "host": engine.host} if engine else None


# ── the agent's search ─────────────────────────────────────────────


class SearchBody(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    count: int = Field(10, ge=1, le=50)
    page: int = Field(0, ge=0, le=20)
    search_lang: str = Field("", max_length=10)
    country: str = Field("", max_length=4)
    freshness: str = Field("", max_length=10)
    safesearch: str = Field("", max_length=10)


_GATES: dict[str, asyncio.Semaphore] = {}


def _gate(token_id: str) -> asyncio.Semaphore:
    gate = _GATES.get(token_id)
    if gate is None:
        gate = _GATES[token_id] = asyncio.Semaphore(SEARCHES_AT_ONCE)
    return gate


def _clean_code(value: str, pattern: str) -> str:
    value = value.strip()
    return value if re.fullmatch(pattern, value) else ""


async def _load_page(conn: Any, owner: str, slug: str, name: str, engine: Engine, url: str) -> dict:
    """One result page in the person's browser: ``{ok, html, url} | {ok: False, error}``."""
    from captain_claw.flight_deck import browser_hands as hands

    act = hands._Act(id="hs_" + secrets.token_hex(10), user_id=owner, slug=slug, agent_name=name,
                     action="search", conn=conn, future=asyncio.get_running_loop().create_future())
    hands._ACTS[act.id] = act
    try:
        try:
            await conn.send({"type": "search", "id": act.id, "engine": engine.id, "url": url,
                             "wait_for": engine.wait_for, "block_for": engine.block_for,
                             "wait": PAGE_WAIT_S - 3})
        except Exception:
            return {"ok": False, "error": "Couldn't reach the browser; it may have just closed."}
        try:
            msg = await asyncio.wait_for(asyncio.shield(act.future), PAGE_WAIT_S)
        except TimeoutError:
            try:
                await conn.send({"type": "cancel", "id": act.id})
            except Exception:
                pass
            return {"ok": False, "error": f"{engine.name} didn't show results in {int(PAGE_WAIT_S)} s."}
    finally:
        hands._ACTS.pop(act.id, None)
    return msg


async def search_in_browser(owner: str, slug: str, body: SearchBody) -> dict:
    """Run one search in the person's browser. Never raises for the browser's
    sake: ``{ok: False, why, error}`` sends the agent to its API engine."""
    from captain_claw.flight_deck import browser_hands as hands

    engine = await chosen_engine(owner)
    if engine is None:
        return {"ok": False, "why": "off", "error": "Searching through the person's Chrome is off."}
    conns = [c for c in hands.live_connections(owner) if not c.agent_slug and engine.id in c.search_engines]
    conn = None
    for cand in conns:
        if await hands._still_valid(cand):
            conn = cand
            break
    if conn is None:
        return {"ok": False, "why": "no_browser", "engine": engine.name,
                "error": (f"No Chrome of the person's is ready for {engine.name} searches right now "
                          "(closed, asleep, or not allowed in Captain for Chrome yet).")}
    from captain_claw.flight_deck.agent_names import display_name

    name = await display_name(slug, owner) or slug or "Agent"
    lang = _clean_code(body.search_lang, r"[A-Za-z]{2}(-[A-Za-z]{2})?")
    country = _clean_code(body.country, r"[A-Za-z]{2}")
    fresh = body.freshness.strip().lower()
    safe = body.safesearch.strip().lower()
    want = body.count
    first = body.page * want // engine.per_page   # the window page*count … page*count+count-1
    skip = body.page * want - first * engine.per_page
    if first >= engine.max_pages:
        return {"ok": False, "why": "past", "engine": engine.name,
                "error": f"{engine.name} through the browser doesn't go that many results deep."}
    pages = min(engine.max_pages, -(-(skip + want) // engine.per_page))
    hits: list[dict] = []
    notes: list[str] = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + SEARCH_BUDGET_S
    async with _gate(conn.token_id):
        for i in range(pages):
            if first + i >= engine.max_pages:
                break
            if i and loop.time() > deadline - PAGE_WAIT_S:
                notes.append(f"Stopped at {len(hits)} results to answer in time.")
                break
            url = engine.build(body.query.strip(), first + i, lang, country, fresh, safe)
            msg = await _load_page(conn, owner, slug, name, engine, url)
            if not msg.get("ok"):
                if hits:
                    notes.append(f"{engine.name} stopped after {len(hits)} results ({msg.get('error') or 'no answer'}).")
                    break
                return {"ok": False, "why": "page", "engine": engine.name,
                        "error": str(msg.get("error") or f"{engine.name} didn't answer.")}
            data = msg.get("data") if isinstance(msg.get("data"), dict) else {}
            found, reason = read_page(engine, str(msg.get("content") or ""), str(data.get("url") or ""))
            if not reason and not found and data.get("ready") is False:
                reason = f"{engine.name}'s page didn't show results in that Chrome."
            if reason:
                if hits:
                    notes.append(f"{engine.name} stopped after {len(hits)} results: {reason}")
                    break
                return {"ok": False, "why": "blocked", "engine": engine.name, "error": reason}
            known = {h["url"] for h in hits}
            fresh_hits = [h for h in found if h["url"] not in known]
            hits.extend(fresh_hits)
            if len(found) < max(3, engine.per_page // 2):
                break   # a short page: the last there is
    hits = hits[skip:skip + want]
    await resolve_links(hits)
    if want > engine.per_page * engine.max_pages:
        notes.append(f"{engine.name} through the browser gives at most "
                     f"{engine.per_page * engine.max_pages} results per search.")
    return {"ok": True, "engine": engine.name, "browser": conn.browser, "results": hits, "notes": notes}
