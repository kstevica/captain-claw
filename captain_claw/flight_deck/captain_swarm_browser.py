"""Captain Swarm — Captain's own look at a page, in a real browser.

Captain checks a claimed result by opening its page itself
(``captain_swarm_outcome.fetch_text``): a plain fetch, public addresses only.
Many shops, travel and booking sites only show their numbers once the page's
scripts have run, and some refuse a plain fetch outright. For those,
:func:`rendered_text` opens the page in a headless Chromium of Captain's own —
never an agent's browser, never signed in to anything — and reads what it
shows.

The plain fetch's rule holds here too: public addresses only. Every request the
page makes — the page itself, every redirect, its scripts, images and calls —
is checked first, and one to a private, loopback or local address is refused.
Downloads are refused, nothing is typed or clicked, and nothing is kept: each
look is a fresh, empty browser. (The plain fetch's known gap stands: a name
that resolves to a public address when checked and a private one a moment
later.)
"""

from __future__ import annotations

import asyncio
import contextlib
import urllib.parse

from captain_claw.logging import get_logger

log = get_logger(__name__)

_GOTO_TIMEOUT_MS = 20_000
_IDLE_TIMEOUT_MS = 8_000
_MAX_HTML = 3_000_000
_AT_ONCE = 2
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

_SLOT: tuple[asyncio.AbstractEventLoop, asyncio.Semaphore] | None = None


def enabled() -> bool:
    """FD_CAPTAIN_SWARM_BROWSER (on by default), and Playwright installed."""
    import os
    if os.environ.get("FD_CAPTAIN_SWARM_BROWSER", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    from captain_claw.tools.browser_session import has_playwright
    return has_playwright()


def _slot() -> asyncio.Semaphore:
    global _SLOT
    here = asyncio.get_running_loop()
    if _SLOT is None or _SLOT[0] is not here:
        _SLOT = (here, asyncio.Semaphore(_AT_ONCE))
    return _SLOT[1]


async def _launch(pw):
    from captain_claw.tools.browser_session import (
        _is_missing_browser_error,
        ensure_chromium_installed,
    )
    try:
        return await pw.chromium.launch(headless=True)
    except Exception as exc:
        if not _is_missing_browser_error(exc):
            raise
        ok, detail = await ensure_chromium_installed()
        if not ok:
            raise RuntimeError(f"no browser on this deck: {detail}") from exc
        return await pw.chromium.launch(headless=True)


async def rendered_text(url: str) -> tuple[str, str]:
    """``(page text, final url)`` of a public page after its scripts ran, or
    ``("", why not)``."""
    from captain_claw.flight_deck.captain_swarm_outcome import _resolves_public, page_text
    if not enabled():
        return "", "Captain's browser is off on this deck"
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not await _resolves_public(parts.hostname or ""):
        return "", f"not a public web address: {url[:120]}"
    hosts: dict[str, bool] = {}

    async def public(target: str) -> bool:
        p = urllib.parse.urlsplit(target)
        if p.scheme in ("data", "blob", "about"):
            return True
        if p.scheme not in ("http", "https"):
            return False
        host = (p.hostname or "").lower()
        if host not in hosts:
            hosts[host] = await _resolves_public(host)
        return hosts[host]

    async def guard(route) -> None:
        if await public(route.request.url):
            await route.continue_()
        else:
            await route.abort()

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return "", "no browser on this deck"
    async with _slot():
        try:
            async with async_playwright() as pw:
                browser = await _launch(pw)
                try:
                    ctx = await browser.new_context(accept_downloads=False, user_agent=_UA, locale="en-US",
                                                    service_workers="block")
                    await ctx.route("**/*", guard)
                    page = await ctx.new_page()
                    await page.goto(url, wait_until="domcontentloaded", timeout=_GOTO_TIMEOUT_MS)
                    with contextlib.suppress(Exception):
                        await page.wait_for_load_state("networkidle", timeout=_IDLE_TIMEOUT_MS)
                    final = page.url
                    if not await public(final):
                        return "", f"it ended on an address that isn't public: {final[:120]}"
                    html = await page.content()
                finally:
                    await browser.close()
        except Exception as exc:  # noqa: BLE001 — a page that won't open is a failed check, not a crash
            log.info("captain swarm: the browser couldn't open a page", url=url[:200], error=str(exc)[:200])
            return "", f"the browser couldn't open it: {type(exc).__name__}"
    return page_text(html[:_MAX_HTML]), final


async def look(url: str, fetch) -> tuple[str, str, str]:
    """``(text, final url or why not, how)``: a plain *fetch* first; the
    browser when the plain fetch got nothing. ``how`` is "fetch" or "browser"."""
    text, where = await fetch(url)
    if text or not enabled():
        return text, where, "fetch"
    rtext, rwhere = await rendered_text(url)
    if rtext:
        return rtext, rwhere, "browser"
    return "", f"{where}; in a browser too: {rwhere}", "fetch"
