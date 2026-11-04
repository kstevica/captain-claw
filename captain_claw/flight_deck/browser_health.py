"""Can Chromium run on this deck's host? Checked once after start-up.

A bare Linux server can hold the Chromium binary yet lack the system
libraries it links against (libnspr4, libnss3 …). Agents then fail every
browser call, web page render and HTML→PDF export, and only the host's admin
can fix it (``playwright install-deps`` needs root). This puts that in front
of the admins, once per set of missing libraries, instead of in agents' logs.

It never downloads a browser: with no Chromium installed yet there is
nothing to check, and the agents' first browser call installs it.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from captain_claw.logging import get_logger

log = get_logger(__name__)

_MARKER = ".browser-health.json"
_TITLE = "The browser can't start on this server"


async def probe() -> list[str] | None:
    """The missing libraries when Chromium can't start here; ``[]`` when it
    starts; ``None`` when there's nothing to check (no Playwright or no
    Chromium installed yet, or a failure of another kind)."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return None
    from captain_claw.tools.browser_session import BrowserUnavailable, launch_chromium

    try:
        async with async_playwright() as pw:
            browser = await launch_chromium(pw.chromium, headless=True, auto_install=False)
            await browser.close()
        return []
    except BrowserUnavailable as exc:
        return exc.libraries or ["system libraries"]
    except Exception as exc:
        log.info("Browser health: nothing to check", reason=type(exc).__name__)
        return None


def _already_told(data_dir: Path, libs: list[str]) -> bool:
    try:
        return json.loads((data_dir / _MARKER).read_text()).get("missing") == libs
    except (OSError, ValueError):
        return False


def _remember(data_dir: Path, libs: list[str]) -> None:
    try:
        (data_dir / _MARKER).write_text(json.dumps({"missing": libs}))
    except OSError:
        pass


def notice_body(libs: list[str]) -> str:
    from captain_claw.tools.browser_session import install_deps_command

    named = ", ".join(libs[:8]) + (" and more" if len(libs) > 8 else "")
    return (
        f"Chromium is installed but can't start: the server is missing {named}. "
        "Until this is fixed, agents can't open pages that need scripts, fill web forms "
        "or turn HTML into PDF. Run this on the server, then nothing needs a restart "
        f"(agents check again within 10 minutes):\n\n{install_deps_command()}"
    )


async def check_on_boot(db, data_dir: Path, *, delay: float = 30.0) -> None:
    """Once, a little after start-up: tell the admins when Chromium can't run."""
    if not sys.platform.startswith("linux"):
        return
    await asyncio.sleep(delay)
    try:
        libs = await probe()
        if libs is None:
            return
        if not libs:
            _remember(data_dir, [])
            return
        if _already_told(data_dir, libs):
            return
        body = notice_body(libs)
        for admin_id in await db.list_admin_ids():
            await db.add_notification(admin_id, "system", _TITLE, body=body,
                                      ref_type="browser_health", ref_id="chromium")
        _remember(data_dir, libs)
        log.warning("Chromium can't start on this host; admins told", missing=", ".join(libs[:10]))
    except Exception as exc:  # never let a health check hurt start-up
        log.warning("Browser health check failed", error=type(exc).__name__)
