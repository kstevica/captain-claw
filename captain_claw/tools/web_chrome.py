"""A web page read in the owner's own Chrome, for web_fetch (flight_deck/browser_fetch.py).

Under Flight Deck, when the agent's owner turned it on (Profile → Web pages)
and Captain for Chrome is open with "Work on every site", a page can be read
in their browser — signed in as them — instead of by the server: *every page*
(asked before web_fetch's own fetch), or only *pages that turn Flight Deck
away* (asked after it was turned away: a bot check, a refusal, a sign-in
wall). Anything short of a page — off, Chrome closed, a site they didn't list
— is ``None``, and web_fetch goes on as before.

Never for a shared agent's member turn: the owner's browser is the owner's.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from captain_claw.logging import get_logger

log = get_logger(__name__)

WAIT_S = 48.0          # Flight Deck waits up to 40 s for the page to load and be drawn in the browser
KEEP_S = 60.0          # what Flight Deck said about the mode is believed this long
_MODE: dict[str, tuple[float, str]] = {}   # Flight Deck address → (until, "" | "blocked" | "always")


def _deck() -> tuple[str, dict[str, str]] | None:
    """Flight Deck's address and this agent's headers — None when there's no
    deck, or this turn isn't the owner's."""
    try:
        from captain_claw import speaker
        from captain_claw.fd_client import flight_deck_base, flight_deck_headers

        base = flight_deck_base()
        if not base or speaker.member_bound() or speaker.grant_headers():
            return None
        headers = flight_deck_headers()
    except Exception:   # noqa: BLE001 — speaker context lost and the like
        return None
    return (base, headers) if headers.get("X-Agent-Auth") else None


def _wanted() -> bool:
    try:
        from captain_claw.config import get_config

        return bool(getattr(get_config().tools.web_fetch, "via_chrome", True))
    except Exception:
        return True


async def page_from_chrome(url: str, when: str, why: str = "",
                           client: httpx.AsyncClient | None = None) -> dict[str, Any] | None:
    """``{"html", "url", "title", "browser"}`` when the owner's Chrome read *url*
    (*when* = "first" before web_fetch's own fetch, "blocked" after it was
    turned away); else None. Never raises."""
    if not _wanted():
        return None
    deck = _deck()
    if deck is None:
        return None
    base, headers = deck
    now = time.monotonic()
    until, mode = _MODE.get(base, (0.0, "?"))
    if now < until and (mode == "" or (when == "first" and mode != "always")):
        return None   # off, or only for pages that turn Flight Deck away
    body = {"url": url, "when": when, "why": why[:300]}
    own = client is None
    http = client or httpx.AsyncClient()
    try:
        resp = await http.post(f"{base}/fd/browser-hands/agent/fetch", json=body, headers=headers,
                               timeout=WAIT_S + 5)
    except httpx.HTTPError as e:
        log.debug("web_fetch: Flight Deck unreachable for a page in Chrome", error=str(e))
        return None
    finally:
        if own:
            await http.aclose()
    if resp.status_code in (404, 405):   # a Flight Deck from before this
        _MODE[base] = (now + 240, "")
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    if resp.status_code >= 400 or not isinstance(data, dict):
        return None
    if "mode" in data:
        _MODE[base] = (now + KEEP_S, str(data.get("mode") or ""))
    if not data.get("ok") or not isinstance(data.get("html"), str) or not data["html"].strip():
        if data.get("why") not in ("off", "not_first", "no_browser", None):
            log.info("web_fetch: the owner's Chrome didn't read the page", url=url, why=data.get("why"),
                     error=str(data.get("error") or "")[:200])
        return None
    return {"html": data["html"], "url": str(data.get("url") or url), "title": str(data.get("title") or ""),
            "browser": str(data.get("browser") or "Chrome")}
