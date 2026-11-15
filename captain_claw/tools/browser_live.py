"""Watch the agent's own browser, and take control of it.

The person (in Flight Deck: an agent's "Its browser") sees the page the agent's
headless browser is on, about once a second. They may take control — for a
CAPTCHA, a sign-in, a step the agent is stuck on. While they have it:

- the agent's next browser action waits (at most ``HOLD_MAX_S``), so the two
  never act on the page at once;
- their clicks, typing, keys and scrolling go into the page;
- the tool-call register keeps one record, from taking control to handing
  it back (``browser`` · ``take_control``).

On hand back, the agent's next browser result starts with who had control,
for how long, and where the page is now, so it looks again before it goes on.
Only the agent's own (headless) browser: the person's Chrome is theirs to
watch already. Frames are never kept.
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

HOLD_MAX_S = 900.0
FRAME_QUALITY = 60
_KEYS = frozenset({"Enter", "Tab", "Backspace", "Escape", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
                   "PageUp", "PageDown", "Home", "End", "Delete", "Space"})

_STATE: dict[str, Any] = {"session": "", "held": False, "since": 0.0, "by": "", "handle": None, "note": "",
                          "released": None}


def reset_for_tests() -> None:
    _STATE.update(session="", held=False, since=0.0, by="", handle=None, note="", released=None)


def note_session(session_key: str) -> None:
    """The browser session the agent used last: the one the person watches."""
    _STATE["session"] = session_key or "default"


def _session() -> Any:
    from captain_claw.tools.browser import _BROWSER_SESSIONS

    key = _STATE["session"]
    if key and key in _BROWSER_SESSIONS:
        return _BROWSER_SESSIONS[key]
    return next(iter(_BROWSER_SESSIONS.values()), None) if _BROWSER_SESSIONS else None


def _page() -> Any:
    session = _session()
    if session is None or not getattr(session, "is_alive", False):
        return None
    return getattr(session, "_page", None)


def status() -> dict[str, Any]:
    return {"held": bool(_STATE["held"]), "since": _STATE["since"] or None, "by": _STATE["by"]}


async def frame() -> dict[str, Any]:
    """What the page looks like now (a JPEG, base64), or ``alive: False``."""
    page = _page()
    out: dict[str, Any] = {"alive": page is not None, **status()}
    if page is None:
        return out
    try:
        shot = await page.screenshot(type="jpeg", quality=FRAME_QUALITY, timeout=5000)
        size = page.viewport_size or {}
        out.update(url=page.url, title=await page.title(), width=size.get("width"), height=size.get("height"),
                   frame=base64.b64encode(shot).decode("ascii"))
    except Exception as exc:  # noqa: BLE001 — a page mid-navigation: try again next time
        out["error"] = f"No picture right now ({type(exc).__name__})."
    return out


def take(by: str) -> dict[str, Any]:
    """The person takes control; the agent's browser actions wait."""
    if _STATE["held"]:
        return status()
    from captain_claw import tool_audit

    _STATE.update(held=True, since=time.time(), by=(by or "the person")[:80], note="")
    _STATE["released"] = asyncio.Event()
    _STATE["handle"] = tool_audit.start("browser", {"action": "take_control", "by": _STATE["by"]})
    log.info("Person took control of the browser", by=_STATE["by"])
    return status()


async def give_back() -> dict[str, Any]:
    """The person hands control back; the agent hears it on its next result."""
    if not _STATE["held"]:
        return status()
    from captain_claw import tool_audit

    started, by = _STATE["since"], _STATE["by"]
    page = _page()
    where = ""
    if page is not None:
        try:
            where = f" The page is now: {await page.title()} — {page.url}."
        except Exception:  # noqa: BLE001
            where = ""
    minutes = max(0.0, time.time() - started) / 60
    took = f"{minutes:.0f} min" if minutes >= 1 else f"{max(1, int(minutes * 60))} s"
    _STATE["note"] = (f"{by} had control of the browser for {took} and handed it back.{where} "
                      "Look at the page again before you go on.")
    tool_audit.finish(_STATE["handle"], "ok", f"{by} had control for {took}")
    released = _STATE["released"]
    _STATE.update(held=False, since=0.0, by="", handle=None, released=None)
    if released is not None:
        released.set()
    log.info("Browser handed back", by=by, took=took)
    return status()


async def wait_while_held(max_s: float = HOLD_MAX_S) -> str:
    """Called before each of the agent's own browser actions: waits while the
    person has control; "" to go on, or why not (they kept it too long)."""
    released = _STATE["released"]
    if not _STATE["held"] or released is None:
        return ""
    try:
        await asyncio.wait_for(released.wait(), max_s)
    except TimeoutError:
        return (f"{_STATE['by'] or 'The person'} still has control of the browser. Don't touch it; tell them "
                "where you are and wait for them to hand it back.")
    return ""


def pop_note() -> str:
    note, _STATE["note"] = _STATE["note"], ""
    return note


async def person_input(kind: str, *, x: float | None = None, y: float | None = None, text: str = "",
                       key: str = "", dy: float = 0.0) -> dict[str, Any]:
    """The person's click, typing, key or scroll, while they have control."""
    if not _STATE["held"]:
        return {"ok": False, "error": "Take control first."}
    page = _page()
    if page is None:
        return {"ok": False, "error": "The browser isn't open."}
    try:
        if kind == "click" and x is not None and y is not None:
            await page.mouse.click(float(x), float(y))
        elif kind == "type" and text:
            await page.keyboard.insert_text(str(text)[:2000])
        elif kind == "key" and key in _KEYS:
            await page.keyboard.press(" " if key == "Space" else key)
        elif kind == "scroll":
            await page.mouse.wheel(0, max(-5000.0, min(5000.0, float(dy))))
        else:
            return {"ok": False, "error": "Unknown input."}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"That didn't reach the page ({type(exc).__name__})."}
    return {"ok": True}
