"""A web page read in the owner's own Chrome, for web_fetch (flight_deck/browser_fetch.py).

Under Flight Deck, when the agent's owner turned it on (Profile → Web pages)
and Captain for Chrome is open with "Work on every site", a page can be read
in their browser — signed in as them — instead of by the server: *every page*
(asked before web_fetch's own fetch), or only *pages that turn Flight Deck
away* (asked after it was turned away: a bot check, a refusal, a sign-in
wall). Anything short of a page — off, Chrome closed, a site they didn't list
— is ``None``, and web_fetch goes on as before.

Never for a shared agent's member turn: the owner's browser is the owner's.

A robot check on the way (Google's "unusual traffic", Cloudflare, a reCAPTCHA
…) is handed to the owner: Captain for Chrome brings that tab to the front,
the chat says the agent waits for them, and once they answered it the search
or read goes on (:func:`call_waiting`, also for web_search; Flight Deck's side
is flight_deck/browser_waits.py).
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from typing import Any

import httpx

from captain_claw.logging import get_logger

log = get_logger(__name__)

WAIT_S = 48.0          # Flight Deck waits up to 40 s for the page to load and be drawn in the browser
KEEP_S = 60.0          # what Flight Deck said about the mode is believed this long
FOLLOW_S = 25.0        # one follow request while the owner answers a robot check
_MODE: dict[str, tuple[float, str]] = {}   # Flight Deck address → (until, "" | "blocked" | "always")


def _json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return None


def waiting_words(waiting: dict) -> str:
    site = str(waiting.get("site") or "The site").strip()[:80]
    minutes = max(1, round(float(waiting.get("for_s") or 300) / 60))
    return (f"{site} wants to know a person is there (a robot check) — the tab is in front in your Chrome. "
            f"Answer it there and I'll go on (I'll wait up to {minutes} min)…\n")


async def call_waiting(http: httpx.AsyncClient, base: str, headers: dict[str, str], path: str, body: dict,
                       *, agent: Any = None, stream: Callable[[str], Any] | None = None,
                       first_wait: float = 50.0) -> tuple[int, Any]:
    """POST *body* to Flight Deck's *path* as a job that may wait for the
    owner (a robot check handed to them in their Chrome), and follow it until
    it's done: ``(status, json)``. While they're at it, the chat says so and
    the tool call gets the time it needs. Stopping the turn stops the wait (the
    check's tab stays open for them). Raises httpx.HTTPError when Flight Deck
    can't be reached."""
    from captain_claw.tools.ask_user import turn_origin
    from captain_claw.tools.registry import more_time

    job = "cw_" + secrets.token_hex(8)
    body = {**body, "job": job, "origin": turn_origin(agent), "wait": first_wait}
    finished = False
    try:
        resp = await http.post(f"{base}{path}", json=body, headers=headers, timeout=first_wait + 15)
        status, data = resp.status_code, _json(resp)
        told = False
        while status == 200 and isinstance(data, dict) and data.get("pending"):
            waiting = data.get("waiting") if isinstance(data.get("waiting"), dict) else None
            if waiting:
                # The person has minutes to answer it: so has this call.
                more_time(float(waiting.get("for_s") or 300) + FOLLOW_S + 60)
                if not told:
                    told = True
                    if callable(stream):
                        try:
                            stream(waiting_words(waiting))
                        except Exception:  # noqa: BLE001
                            pass
            resp = await http.post(f"{base}/fd/browser-hands/agent/wait/follow", json={"id": job, "wait": FOLLOW_S},
                                   headers=headers, timeout=FOLLOW_S + 15)
            status, data = resp.status_code, _json(resp)
        finished = True
        return status, data
    finally:
        if not finished:
            try:
                await http.post(f"{base}/fd/browser-hands/agent/wait/cancel", json={"id": job},
                                headers=headers, timeout=5)
            except Exception:  # noqa: BLE001
                pass


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
                           client: httpx.AsyncClient | None = None, *, agent: Any = None,
                           stream: Callable[[str], Any] | None = None) -> dict[str, Any] | None:
    """``{"html", "url", "title", "browser", "robot"}`` when the owner's Chrome
    read *url* (*when* = "first" before web_fetch's own fetch, "blocked" after
    it was turned away); else None. ``robot``: a check the owner answered on
    the way (``{"site", "vendor", "outcome"}``), or None. Never raises."""
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
        status, data = await call_waiting(http, base, headers, "/fd/browser-hands/agent/fetch", body,
                                          agent=agent, stream=stream, first_wait=WAIT_S)
    except httpx.HTTPError as e:
        log.debug("web_fetch: Flight Deck unreachable for a page in Chrome", error=str(e))
        return None
    finally:
        if own:
            await http.aclose()
    if status in (404, 405):   # a Flight Deck from before this
        _MODE[base] = (now + 240, "")
        return None
    if status >= 400 or not isinstance(data, dict):
        return None
    if "mode" in data:
        _MODE[base] = (now + KEEP_S, str(data.get("mode") or ""))
    if not data.get("ok") or not isinstance(data.get("html"), str) or not data["html"].strip():
        if data.get("why") not in ("off", "not_first", "no_browser", None):
            log.info("web_fetch: the owner's Chrome didn't read the page", url=url, why=data.get("why"),
                     error=str(data.get("error") or "")[:200])
        return None
    robot = data.get("robot") if isinstance(data.get("robot"), dict) else None
    return {"html": data["html"], "url": str(data.get("url") or url), "title": str(data.get("title") or ""),
            "browser": str(data.get("browser") or "Chrome"), "robot": robot}
