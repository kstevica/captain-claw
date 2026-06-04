"""Flow trigger router — classify an inbound message and match it to a Flow.

Lives in Flight Deck. Hooked at the points FD forwards inbound to agents
(WhatsApp bridge, glasses bus). Rules-first and cheap; an opt-in LLM classifier
is a later phase. No match → the caller does its normal forward (no behaviour
change when no flows are enabled).
"""

from __future__ import annotations

import asyncio
import fnmatch
import re
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

# Module-level engine handle, set by the FD server on startup.
_STORE: Any = None
_RUNNER: Any = None

# Flows that pause on an `input` step park a Future here, keyed by the user
# identity. The next inbound message from that user resolves it and the run
# resumes. In-memory (a paused run is a live coroutine), so a FD restart drops
# pending waits — acceptable for v1.
_PENDING_INPUT: dict[str, asyncio.Future] = {}


def input_key(*, waid: str = "", channel: str = "") -> str:
    """Stable key for a paused-input wait. WAID identifies a WhatsApp user
    across messages; otherwise fall back to the channel."""
    waid = str(waid or "")
    if waid:
        return f"waid:{waid}"
    return f"chan:{str(channel or '')}"


async def wait_for_input(key: str, *, timeout: float = 3600.0) -> str:
    """Park until someone delivers input for *key* (or timeout). Last waiter on
    a key wins — a new wait cancels any stale one so a re-triggered flow doesn't
    leak a dangling Future."""
    loop = asyncio.get_event_loop()
    fut: asyncio.Future = loop.create_future()
    old = _PENDING_INPUT.get(key)
    if old is not None and not old.done():
        old.cancel()
    _PENDING_INPUT[key] = fut
    try:
        return await asyncio.wait_for(fut, timeout=timeout)
    finally:
        if _PENDING_INPUT.get(key) is fut:
            _PENDING_INPUT.pop(key, None)


def has_pending_input(*, waid: str = "", channel: str = "") -> bool:
    fut = _PENDING_INPUT.get(input_key(waid=waid, channel=channel))
    return bool(fut is not None and not fut.done())


def deliver_pending_input(*, waid: str = "", channel: str = "", text: str = "") -> bool:
    """Resolve a paused flow's input wait with *text*. Returns True if a wait
    was satisfied (caller should NOT also forward the message to the agent)."""
    fut = _PENDING_INPUT.get(input_key(waid=waid, channel=channel))
    if fut is not None and not fut.done():
        fut.set_result(text)
        return True
    return False


def _flow_has_input(flow: dict[str, Any]) -> bool:
    return any(str(s.get("type")) == "input" for s in (flow.get("steps") or []))


async def _bg_run(flow: dict[str, Any], payload: dict[str, Any]) -> None:
    try:
        await _RUNNER.run(flow, payload)
    except Exception as exc:
        log.warning("flow bg run error", flow=flow.get("name"), error=str(exc))


def set_engine(store: Any, runner: Any) -> None:
    global _STORE, _RUNNER
    _STORE, _RUNNER = store, runner


def engine_ready() -> bool:
    return _STORE is not None and _RUNNER is not None


def classify_payload(
    *, channel: str, text: str = "", mime: str = "",
    image_path: str = "", video_path: str = "", audio_path: str = "",
    waid: str = "", origin_host: str = "localhost", origin_port: int = 0,
    origin_name: str = "", extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize an inbound message into a flat payload the router/runner use."""
    p: dict[str, Any] = {
        "channel": channel,
        "text": text or "",
        "mime": mime or "",
        "has_image": bool(image_path),
        "has_video": bool(video_path),
        "has_audio": bool(audio_path),
        "has_text": bool((text or "").strip()),
        "image_path": image_path or "",
        "video_path": video_path or "",
        "audio_path": audio_path or "",
        "waid": waid or "",
        "whatsapp_waid": waid or "",
        "origin_host": origin_host,
        "origin_port": int(origin_port or 0),
        "origin_name": origin_name or "",
    }
    if extra:
        p.update(extra)
    return p


def _rule_ok(rule: str, payload: dict[str, Any]) -> bool:
    rule = rule.strip()
    if not rule:
        return True
    if ":" in rule:
        key, val = (x.strip() for x in rule.split(":", 1))
        if key == "channel":
            return str(payload.get("channel", "")).lower() == val.lower()
        if key == "from_waid":
            return str(payload.get("waid", "")) == val
        if key == "mime":
            return fnmatch.fnmatch(str(payload.get("mime", "")), val)
        if key == "regex":
            try:
                return bool(re.search(val, str(payload.get("text", "")), re.I))
            except re.error:
                return False
        if key == "contains":
            return val.lower() in str(payload.get("text", "")).lower()
        return False
    # Known boolean flags.
    if rule in ("has_image", "has_video", "has_audio", "has_text", "has_document"):
        return bool(payload.get(rule, False))
    # Any other bare word → case-insensitive substring match on the text
    # (intuitive: a rule of "he-man" fires when the message mentions he-man).
    return rule.lower() in str(payload.get("text", "")).lower()


def _trigger_matches(trigger: dict[str, Any], payload: dict[str, Any]) -> bool:
    if str(trigger.get("on", "message")) != "message":
        return False
    chan = str(trigger.get("channel", "any")).lower()
    if chan not in ("any", "") and chan != str(payload.get("channel", "")).lower():
        return False
    match = trigger.get("match") or {}
    kind = str(match.get("kind", "rule"))
    if kind == "always":
        return True
    if kind == "rule":
        rules = match.get("rules") or []
        return all(_rule_ok(str(r), payload) for r in rules)
    # 'classifier' is a later phase — treat as non-matching for now.
    return False


async def match_flow(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return the highest-priority enabled flow whose trigger matches, else None."""
    if not engine_ready():
        return None
    try:
        flows = await _STORE.enabled_flows()  # already priority-desc
    except Exception as exc:
        log.warning("flow match: store error: %s", exc)
        return None
    for flow in flows:
        try:
            if _trigger_matches(flow.get("trigger") or {}, payload):
                return flow
        except Exception:
            continue
    return None


async def run_flow(flow: dict[str, Any], payload: dict[str, Any]) -> None:
    """Run a already-matched flow (used when the caller needs to enrich the
    payload — e.g. upload an image and set image_path — between match and run)."""
    if _RUNNER is None:
        return
    # A flow that can pause for user input must not block the inbound handler
    # (the user's reply arrives on a *later* message that needs this handler
    # free to deliver it). Run those in the background.
    if _flow_has_input(flow):
        asyncio.create_task(_bg_run(flow, payload))
        return
    try:
        await _RUNNER.run(flow, payload)
    except Exception as exc:
        log.warning("flow run error", flow=flow.get("name"), error=str(exc))


async def try_match_and_run(payload: dict[str, Any]) -> bool:
    """If a flow matches the payload, run it and return True; else False.

    Safe to call from any inbound path — a no-op when no flow matches (so the
    caller falls through to its normal agent-forward).
    """
    flow = await match_flow(payload)
    if not flow:
        return False
    log.info("flow triggered", flow=flow.get("name"), channel=payload.get("channel"))
    if _flow_has_input(flow):
        asyncio.create_task(_bg_run(flow, payload))
        return True
    try:
        await _RUNNER.run(flow, payload)
    except Exception as exc:
        log.warning("flow run error", flow=flow.get("name"), error=str(exc))
    return True
