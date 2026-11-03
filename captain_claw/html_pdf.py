"""HTML to PDF: WeasyPrint when it can load, headless Chromium when it can't.

WeasyPrint draws text with Pango, which it loads from the system (GLib, Pango,
HarfBuzz, fontconfig). A Mac without Homebrew and a Windows without GTK have none
of them, so ``import weasyprint`` raises ``OSError`` there, and the desktop app's
own Python is such a machine. Chromium (Playwright's, which the browser tools
install anyway) prints the same HTML instead: scripts off, as in WeasyPrint, and
it never reads files or this machine's own services.
"""

from __future__ import annotations

import asyncio
import ipaddress
from typing import Any
from urllib.parse import urlsplit

from captain_claw.logging import get_logger

log = get_logger(__name__)

# Why WeasyPrint can't load here, once known: the import is tried once per process.
_weasyprint_error: str | None = None


def _weasyprint_html() -> Any:
    global _weasyprint_error
    if _weasyprint_error is not None:
        return None
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as exc:
        _weasyprint_error = str(exc) or type(exc).__name__
        log.info("WeasyPrint unavailable, printing PDFs with Chromium", reason=_weasyprint_error[:300])
        return None
    return HTML


async def html_to_pdf(html: str) -> bytes:
    """The PDF of *html*: WeasyPrint's, or Chromium's where WeasyPrint can't load."""
    weasy = _weasyprint_html()
    if weasy is not None:
        return await asyncio.to_thread(lambda: weasy(string=html).write_pdf())
    return await _chromium_pdf(html)


def _local_host(host: str) -> bool:
    host = host.strip("[]").lower()
    if host in ("localhost", "") or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_unspecified


async def _chromium_pdf(html: str) -> bytes:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            f"Can't make a PDF here: WeasyPrint can't load ({_weasyprint_error}) and Playwright isn't installed."
        ) from exc
    from captain_claw.tools.browser_session import (
        _is_missing_browser_error,
        ensure_chromium_installed,
    )

    async def only_the_web(route: Any) -> None:
        parts = urlsplit(route.request.url)
        if parts.scheme in ("data", "blob", "about"):
            await route.continue_()
        elif parts.scheme in ("http", "https") and not _local_host(parts.hostname or ""):
            await route.continue_()
        else:
            await route.abort()

    async with async_playwright() as pw:
        try:
            browser = await pw.chromium.launch(headless=True)
        except Exception as exc:
            if not _is_missing_browser_error(exc):
                raise
            ok, detail = await ensure_chromium_installed()
            if not ok:
                raise RuntimeError(f"Chromium isn't installed and installing it failed ({detail}).") from exc
            browser = await pw.chromium.launch(headless=True)
        try:
            context = await browser.new_context(java_script_enabled=False)
            await context.route("**/*", only_the_web)
            page = await context.new_page()
            await page.set_content(html, wait_until="load", timeout=60_000)
            return await page.pdf(
                format="A4",
                print_background=True,
                prefer_css_page_size=True,
                margin={"top": "12mm", "right": "12mm", "bottom": "12mm", "left": "12mm"},
            )
        finally:
            await browser.close()
