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

# The announce text of an active input step, keyed the same way — so a
# '/flow resume' can re-show the question the flow is waiting on.
_PENDING_INPUT_PROMPT: dict[str, str] = {}


def set_input_prompt(key: str, text: str) -> None:
    _PENDING_INPUT_PROMPT[key] = text


def clear_input_prompt(key: str) -> None:
    _PENDING_INPUT_PROMPT.pop(key, None)


# A `wait until <condition>` parks here with a predicate; only a message that
# satisfies the condition resolves it (others fall through to the agent).
_PENDING_COND: dict[str, str] = {}


async def wait_for_match(key: str, condition: str, *, timeout: float = 86400.0) -> str:
    """Like wait_for_input, but only an inbound message satisfying *condition*
    (a branch expression evaluated against {{trigger.text}}) resolves it."""
    _PENDING_COND[key] = condition
    return await wait_for_input(key, timeout=timeout)


def _match_condition(condition: str, text: str) -> bool:
    try:
        import captain_claw.flight_deck.flow_runner as fr
        ctx = {"trigger": {"text": text}, "vars": {}, "steps": {}, "system": {}}
        return bool(fr._eval_expr(condition, ctx))
    except Exception:
        return False


def input_key(*, waid: str = "", channel: str = "", origin_port: int = 0) -> str:
    """Stable key for a paused-input wait. WAID identifies a WhatsApp user
    across messages; for agent-handled channels (web/glasses) the same user
    talks to one agent, so key by channel + that agent's port."""
    waid = str(waid or "")
    if waid:
        return f"waid:{waid}"
    return f"chan:{str(channel or '')}:{int(origin_port or 0)}"


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
        _PENDING_COND.pop(key, None)


def has_pending_input(*, waid: str = "", channel: str = "", origin_port: int = 0) -> bool:
    fut = _PENDING_INPUT.get(input_key(waid=waid, channel=channel, origin_port=origin_port))
    return bool(fut is not None and not fut.done())


def deliver_pending_input(*, waid: str = "", channel: str = "", origin_port: int = 0, text: str = "") -> bool:
    """Resolve a paused flow's input wait with *text*. Returns True if a wait
    was satisfied (caller should NOT also forward the message to the agent).

    If the owner's flow is *paused* (via '/flow pause'), the wait is left
    intact and this returns False, so the message goes to the agent as a normal
    turn instead of being swallowed as the flow's answer. The flow stays on the
    input step and consumes the next reply only after it is resumed."""
    key = input_key(waid=waid, channel=channel, origin_port=origin_port)
    fut = _PENDING_INPUT.get(key)
    if fut is None or fut.done():
        return False
    try:
        import captain_claw.flight_deck.flow_runner as fr
        if fr.owner_is_paused(key):
            return False
    except Exception:
        pass
    # A `wait until` parks with a condition — only a matching message resolves it.
    cond = _PENDING_COND.get(key)
    if cond and not _match_condition(cond, text):
        return False
    fut.set_result(text)
    return True


def cancel_pending_input(*, waid: str = "", channel: str = "", origin_port: int = 0) -> bool:
    """Cancel a paused flow's input wait (used when stopping a flow that's
    blocked waiting for the user). Returns True if a wait was cancelled."""
    fut = _PENDING_INPUT.get(input_key(waid=waid, channel=channel, origin_port=origin_port))
    if fut is not None and not fut.done():
        fut.cancel()
        return True
    return False


# Text control commands: '/flow stop|pause|resume' — slash optional, anchored
# so ordinary chat ("the flow stopped") doesn't trigger it. `halt`=stop,
# `continue`=resume. A trailing phrase after `stop` is sent before stopping.
_FLOW_CMD_RE = re.compile(r"^\s*/?flow\s+(stop|halt|pause|resume|continue|status|state)\b\s*(.*)$", re.I)


def _owner_key_for(payload: dict[str, Any]) -> str:
    return input_key(
        waid=str(payload.get("waid") or payload.get("whatsapp_waid") or ""),
        channel=str(payload.get("channel") or ""),
        origin_port=int(payload.get("origin_port") or 0),
    )


async def maybe_handle_flow_command(payload: dict[str, Any]) -> bool:
    """Intercept a '/flow stop|pause|resume' message and control the caller's
    running flow. Returns True if it was a flow command (and was handled), so
    the caller must NOT forward it on as input or a new trigger."""
    text = str(payload.get("text") or "")
    m = _FLOW_CMD_RE.match(text)
    if not m:
        return False
    import captain_claw.flight_deck.flow_runner as fr

    action = m.group(1).lower()
    rest = (m.group(2) or "").strip()
    owner = _owner_key_for(payload)
    waiting = has_pending_input(
        waid=str(payload.get("waid") or payload.get("whatsapp_waid") or ""),
        channel=str(payload.get("channel") or ""),
        origin_port=int(payload.get("origin_port") or 0),
    )
    reply = ""

    def _label(rid: str) -> str:
        c = fr._RUN_CONTROL.get(rid)
        return f"*{c.name or 'Flow'}* `[{c.handle}]`" if c else "the flow"

    if action in ("status", "state"):
        states = fr.owner_run_states(owner)
        if not states:
            reply = "💤 No flow is running."
        else:
            lines = []
            for s in states:
                nm = s.get("name") or "Flow"
                h = s.get("handle") or ""
                if s.get("paused"):
                    detail = "⏸️ paused" + (" — waiting for your input on resume" if waiting else "")
                elif waiting:
                    detail = "⏳ waiting for your input"
                else:
                    detail = "▶️ running"
                crumb = str(s.get("crumb") or "")
                tail = f"  ({crumb})" if crumb and "›" in crumb else ""
                tag = f" `[{h}]`" if h else ""
                lines.append(f"• *{nm}*{tag} — {detail}{tail}")
            reply = "📊 *Flow status*\n" + "\n".join(lines)
            if len(states) > 1:
                reply += "\n\n_Target one with e.g. `/flow stop <handle>`, or `/flow stop all`._"
    elif action in ("stop", "halt"):
        targeted = fr.resolve_runs(owner, rest)
        # A non-empty target that matched nothing is treated as a stop message
        # applied to the most-recent run (e.g. `/flow stop ok, cancelled`).
        msg = ""
        if rest and not targeted and rest.lower() != "all":
            targeted = fr.resolve_runs(owner, "")
            msg = rest
        if not targeted:
            reply = "No running flow to stop." if not rest else f"No flow matching “{rest}”."
        else:
            for rid in targeted:
                fr.request_stop(rid, msg)
            # A flow paused on an `input` step is blocked off the control loop —
            # cancel its wait so the stop takes effect immediately.
            cancel_pending_input(
                waid=str(payload.get("waid") or payload.get("whatsapp_waid") or ""),
                channel=str(payload.get("channel") or ""),
                origin_port=int(payload.get("origin_port") or 0),
            )
            if msg:
                reply = ""  # the stopped run delivers the custom message itself
            elif len(targeted) == 1:
                reply = f"⏹️ Stopped {_label(targeted[0])}."
            else:
                reply = f"⏹️ Stopped {len(targeted)} running flows."
    elif action == "pause":
        targeted = fr.resolve_runs(owner, rest)
        n = sum(1 for rid in targeted if fr.request_pause(rid))
        if not n:
            reply = "No running flow to pause." if not rest else f"No flow matching “{rest}”."
        elif len(targeted) == 1:
            reply = f"⏸️ Paused {_label(targeted[0])} — send */flow resume* to continue."
        else:
            reply = f"⏸️ Paused {n} flows — send */flow resume all* to continue."
    else:  # resume / continue
        targeted = fr.resolve_runs(owner, rest)
        n = sum(1 for rid in targeted if fr.request_resume(rid))
        if not n:
            reply = "No paused flow to resume." if not rest else f"No flow matching “{rest}”."
        elif len(targeted) == 1:
            # If the flow paused on an input step, re-show the question.
            prompt = _PENDING_INPUT_PROMPT.get(owner)
            reply = f"▶️ Resumed {_label(targeted[0])}.\n\n{prompt}" if prompt else f"▶️ Resumed {_label(targeted[0])}."
        else:
            reply = f"▶️ Resumed {n} flows."

    if reply and _RUNNER is not None:
        try:
            await _RUNNER._deliver(payload, reply)
        except Exception as exc:
            log.warning("flow command reply failed: %s", exc)
    return True


def _flow_has_input(flow: dict[str, Any]) -> bool:
    return any(str(s.get("type")) == "input" for s in (flow.get("steps") or []))


def _flow_needs_async(flow: dict[str, Any]) -> bool:
    """A flow must run detached from the caller's turn when it can pause for
    input, or when it consults the ORIGIN agent — on agent-handled channels the
    origin agent is the one waiting on this evaluate call, so a synchronous
    consult would deadlock it. Such flows run in the background and deliver via
    the channel (WhatsApp send / agent chat-push) instead of an inline return."""
    for s in (flow.get("steps") or []):
        if str(s.get("type")) == "input":
            return True
        if str(s.get("on") or "").strip() == "origin" and str(s.get("type")) in ("agent", "tool", "vision"):
            return True
    return False


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
        if not rules:
            return True
        mode = str(match.get("mode") or "all").lower()
        results = [_rule_ok(str(r), payload) for r in rules]
        return any(results) if mode == "any" else all(results)
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
    if _flow_needs_async(flow):
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
    if _flow_needs_async(flow):
        asyncio.create_task(_bg_run(flow, payload))
        return True
    try:
        await _RUNNER.run(flow, payload)
    except Exception as exc:
        log.warning("flow run error", flow=flow.get("name"), error=str(exc))
    return True
