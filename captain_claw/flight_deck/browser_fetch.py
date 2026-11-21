"""Web pages read in the person's own Chrome — signed in as them — for web_fetch.

The same road as the agents' web searches (``browser_search.py``), for any
page: sites that turn Flight Deck away (LinkedIn's 999, Facebook's sign-in
wall, a bot check) open for the person's own browser.

Opt-in on both ends, off until the person turns it on:

* **Flight Deck** — Profile → *Web pages*: off, *only when a site turns Flight
  Deck away* (``blocked``), or *every page* (``always``); optionally only for
  the sites they list (``linkedin.com`` covers its subdomains). Kept per person
  (``web-fetch:chrome``).
* **Captain for Chrome** — reads pages only while Chrome lets it read every
  site ("Work on every site"), and not where it was turned off in its settings.

``web_fetch`` asks here (``POST /fd/browser-hands/agent/fetch``) — first for
*every page*, after Flight Deck's own fetch was turned away for *blocked*. The
extension opens the page in a background tab (never in front, in a collapsed
"Captain · reading" group), waits for it to settle, sends it back without
scripts and styles, and closes the tab; web_fetch reads that HTML like any
other page. Anything short of a page — Chrome closed, not a web page, a site
the person didn't list — answers ``ok: false`` and the agent fetches as before.

Never: a shared agent's member turn (the tool doesn't ask, and the grant guard
refuses member markers), addresses on a private network or this machine
(checked here and again in the extension, which also refuses Flight Deck's
own pages), anything but http(s).
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import re
import secrets
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from captain_claw.flight_deck.auth import get_current_user, get_db

log = logging.getLogger(__name__)

owner_router = APIRouter(prefix="/fd/web-fetch", tags=["web-fetch"])

SETTING_KEY = "web-fetch:chrome"
MODES = ("blocked", "always")
PAGE_WAIT_S = 30.0            # one page in the person's browser, settle and scrolls included
MAX_HTML_CHARS = 3_000_000    # the page as the extension sends it
MAX_SITES = 50
READS_AT_ONCE = 2             # per browser


# ── the person's choice ────────────────────────────────────────────


def _site(text: str) -> str:
    """``https://www.LinkedIn.com/in/x`` → ``linkedin.com``; '' when it isn't a site."""
    raw = str(text or "").strip().lower()
    if "//" not in raw:
        raw = "//" + raw
    try:
        host = urlsplit(raw).hostname or ""
    except ValueError:
        return ""
    host = host[4:] if host.startswith("www.") else host
    return host if re.fullmatch(r"[a-z0-9-]+(\.[a-z0-9-]+)+", host) else ""


def clean_sites(raw: Any) -> list[str]:
    items = raw if isinstance(raw, list) else re.split(r"[\s,;]+", str(raw or ""))
    out: list[str] = []
    for item in items[: MAX_SITES * 2]:
        site = _site(str(item))
        if site and site not in out:
            out.append(site)
    return out[:MAX_SITES]


async def fetch_setting(user_id: str) -> dict:
    """``{"mode": ""|"blocked"|"always", "sites": [...]}`` — mode '' = off."""
    try:
        raw = await get_db().get_setting(str(user_id), SETTING_KEY)
        data = json.loads(raw) if raw else {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    mode = str(data.get("mode") or "")
    return {"mode": mode if mode in MODES else "", "sites": clean_sites(data.get("sites"))}


async def fetch_mode(user_id: str) -> str:
    return (await fetch_setting(user_id))["mode"]


def site_listed(url: str, sites: list[str]) -> bool:
    if not sites:
        return True
    host = (urlsplit(url).hostname or "").lower()
    return any(host == s or host.endswith("." + s) for s in sites)


_LOCAL_SUFFIXES = (".local", ".localhost", ".internal", ".lan", ".home", ".home.arpa", ".intranet", ".corp")


def url_refusal(url: str) -> str:
    """Why the person's browser won't open *url* for an agent — '' when it may."""
    try:
        parts = urlsplit(str(url or "").strip())
    except ValueError:
        return "That address is broken."
    if parts.scheme not in ("http", "https"):
        return "Only web pages (http or https) are read in the person's browser."
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        return "That address has no site in it."
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        if not ip.is_global:
            return "Addresses on a private network or this machine are never opened in the person's browser."
        return ""
    if host == "localhost" or "." not in host or host.endswith(_LOCAL_SUFFIXES):
        return "Addresses on a private network or this machine are never opened in the person's browser."
    return ""


def _browser_rows(user_id: str, mode: str) -> list[dict]:
    from captain_claw.flight_deck.browser_hands import live_connections

    return [{"browser": c.browser, "version": c.version, "ready": bool(mode and c.reads_pages)}
            for c in live_connections(user_id) if not c.agent_slug]


@owner_router.get("/chrome")
async def get_chrome_fetch(user: dict = Depends(get_current_user)) -> dict:
    from captain_claw.flight_deck.browser_clip import token_kind

    setting = await fetch_setting(str(user["id"]))
    try:
        rows = await get_db().list_pats(str(user["id"]))
    except Exception:
        rows = []
    return {**setting, "browsers": _browser_rows(str(user["id"]), setting["mode"]),
            "extension_set_up": any(token_kind(r.get("name")) == "browser" for r in rows)}


class ChromeFetchBody(BaseModel):
    mode: str = Field("", max_length=20)
    sites: list[str] | str = Field(default_factory=list)


async def tell_browsers(user_id: str, mode: str) -> None:
    """The person changed it: their open browsers hear it now."""
    from captain_claw.flight_deck.browser_hands import live_connections

    for conn in live_connections(user_id):
        if conn.agent_slug:
            continue
        try:
            await conn.send({"type": "fetch_mode", "mode": mode})
        except Exception:
            pass


@owner_router.put("/chrome")
async def put_chrome_fetch(body: ChromeFetchBody, user: dict = Depends(get_current_user)) -> dict:
    mode = body.mode.strip().lower()
    if mode and mode not in MODES:
        raise HTTPException(422, "Off, blocked or always.")
    sites = clean_sites(body.sites)
    db = get_db()
    if mode or sites:
        await db.set_settings(str(user["id"]), {SETTING_KEY: json.dumps({"mode": mode, "sites": sites})})
    else:
        await db.delete_setting(str(user["id"]), SETTING_KEY)
    await tell_browsers(str(user["id"]), mode)
    st = await get_chrome_fetch(user)
    for _ in range(10):   # an open browser that may read pages says so in a moment
        if not mode or not st["browsers"] or any(b["ready"] for b in st["browsers"]):
            break
        await asyncio.sleep(0.1)
        st = await get_chrome_fetch(user)
    return st


# ── the agent's page ───────────────────────────────────────────────


class FetchBody(BaseModel):
    url: str = Field(..., min_length=8, max_length=4000)
    # "first": web_fetch asks before its own fetch (only "every page" answers);
    # "blocked": its own fetch was turned away.
    when: str = Field("blocked", pattern=r"^(first|blocked)$")
    why: str = Field("", max_length=300)


_GATES: dict[str, asyncio.Semaphore] = {}


def _gate(token_id: str) -> asyncio.Semaphore:
    gate = _GATES.get(token_id)
    if gate is None:
        gate = _GATES[token_id] = asyncio.Semaphore(READS_AT_ONCE)
    return gate


async def fetch_in_browser(owner: str, slug: str, body: FetchBody) -> dict:
    """Read one page in the person's browser. Never raises for the browser's
    sake: ``{ok: False, why, error}`` sends the agent to its own fetch."""
    from captain_claw.flight_deck import browser_hands as hands

    setting = await fetch_setting(owner)
    mode = setting["mode"]
    if not mode:
        return {"ok": False, "why": "off", "mode": "", "error": "Reading pages in the person's Chrome is off."}
    if body.when == "first" and mode != "always":
        return {"ok": False, "why": "not_first", "mode": mode,
                "error": "Only pages that turn Flight Deck away are read in the person's Chrome."}
    url = body.url.strip()
    refusal = url_refusal(url)
    if refusal:
        return {"ok": False, "why": "refused", "mode": mode, "error": refusal}
    if not site_listed(url, setting["sites"]):
        return {"ok": False, "why": "not_listed", "mode": mode,
                "error": "The person reads only some sites through their Chrome, and not this one."}
    conns = [c for c in hands.live_connections(owner) if not c.agent_slug and c.reads_pages]
    conn = None
    for cand in conns:
        if await hands._still_valid(cand):
            conn = cand
            break
    if conn is None:
        return {"ok": False, "why": "no_browser", "mode": mode,
                "error": ("No Chrome of the person's is ready to read pages right now "
                          "(closed, asleep, or “Work on every site” is off in Captain for Chrome).")}
    if slug in conn.stopped:
        return {"ok": False, "why": "stopped", "mode": mode,
                "error": "The person stopped this agent using their browser (Stop in the Captain for Chrome panel)."}
    from captain_claw.flight_deck.agent_names import display_name

    name = await display_name(slug, owner) or slug or "Agent"
    act = hands._Act(id="hf_" + secrets.token_hex(10), user_id=owner, slug=slug, agent_name=name,
                     action="fetch", conn=conn, future=asyncio.get_running_loop().create_future(),
                     content_limit=MAX_HTML_CHARS)
    hands._ACTS[act.id] = act
    try:
        async with _gate(conn.token_id):
            try:
                await conn.send({"type": "fetch", "id": act.id, "url": url, "wait": PAGE_WAIT_S - 4,
                                 "agent": {"slug": slug, "name": name}})
            except Exception:
                return {"ok": False, "why": "page", "mode": mode,
                        "error": "Couldn't reach the browser; it may have just closed."}
            try:
                msg = await asyncio.wait_for(asyncio.shield(act.future), PAGE_WAIT_S)
            except TimeoutError:
                try:
                    await conn.send({"type": "cancel", "id": act.id})
                except Exception:
                    pass
                return {"ok": False, "why": "page", "mode": mode,
                        "error": f"The page didn't settle in the person's Chrome in {int(PAGE_WAIT_S)} s."}
    finally:
        hands._ACTS.pop(act.id, None)
    if not msg.get("ok"):
        return {"ok": False, "why": "page", "mode": mode, "error": str(msg.get("error") or "The browser couldn't read it.")}
    data = msg.get("data") if isinstance(msg.get("data"), dict) else {}
    final = str(data.get("url") or url)
    if url_refusal(final):   # it went somewhere it shouldn't have: nothing of it comes back
        return {"ok": False, "why": "refused", "mode": mode, "error": url_refusal(final)}
    return {"ok": True, "mode": mode, "browser": conn.browser, "url": final,
            "title": str(data.get("title") or "")[:300], "html": str(msg.get("content") or "")}
