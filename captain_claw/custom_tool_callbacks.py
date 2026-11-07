"""Agent side of custom tools, Phase 7: a custom tool that this agent ran asks
it to use one of its tools.

Flight Deck runs the tool (flight_deck/external_tools.py) and passes each of
its requests here, to ``POST /api/custom-tools/callback``, with the agent's own
token. This agent acts only for a run it is waiting on right now: ``custom_tools
run`` opens a wait under a fresh id, sends the id with the run, and closes it
when the run ends (or the turn is stopped). Any other id is refused.

Who the request runs as comes with the request (Flight Deck's record of the
run: the runner, and on a shared agent the member and their grant). It must be
the identity the run started with; it is then bound for this one call, and the
tool runs through the agent's usual guarded path (its rules, its member
allowlist, folder limits, log), on the same agent instance that ran the tool.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

MAX_CONTENT = 1_000_000
MAX_WAIT_S = 4_000.0
# Try it on the Custom tools page has no turn: its requests run in this session.
TRY_SESSION = "custom-tools-try"


@dataclass
class Waiting:
    agent: Any            # the instance that ran the tool (a member's own, on a shared agent)
    session_id: str
    principal: Any        # speaker.Principal of a member's turn, else None
    grant: str            # that turn's speaker grant, "" for the owner
    member: bool
    tool: str
    until: float          # time.monotonic()


_WAITING: dict[str, Waiting] = {}


def _sweep() -> None:
    now = time.monotonic()
    for cb in [k for k, w in _WAITING.items() if w.until < now]:
        _WAITING.pop(cb, None)


def open_wait(agent: Any, session_id: str, *, tool: str, ttl: float = MAX_WAIT_S) -> str:
    """Wait for requests from the run of *tool* this turn is about to start;
    the id to send with it."""
    from captain_claw import speaker

    _sweep()
    cb = secrets.token_urlsafe(32)
    principal = speaker.current()
    _WAITING[cb] = Waiting(agent=agent, session_id=str(session_id or ""), principal=principal,
                           grant=speaker.current_grant() if principal is not None else "",
                           member=principal is not None, tool=str(tool or ""),
                           until=time.monotonic() + max(1.0, min(float(ttl), MAX_WAIT_S)))
    return cb


def close_wait(cb: str) -> None:
    if cb:
        _WAITING.pop(cb, None)


def waiting(cb: str) -> Waiting | None:
    w = _WAITING.get(str(cb or ""))
    if w is None or w.until < time.monotonic():
        return None
    return w


def reset_for_tests() -> None:
    _WAITING.clear()


def _same_identity(w: Waiting, identity: dict[str, Any]) -> bool:
    from captain_claw import speaker

    if bool(identity.get("member")) != w.member:
        return False
    if not w.member:
        return True
    return (w.principal is not None and str(identity.get("user_id") or "") == str(w.principal.speaker_id)
            and speaker.sanitize_grant(identity.get("grant")) == w.grant != "")


async def handle(body: Any) -> tuple[int, dict[str, Any]]:
    """One request from a running custom tool: ``(status, answer)``."""
    from captain_claw import mail_authority, speaker

    if not isinstance(body, dict):
        return 400, {"ok": False, "error": "The request must be a JSON object."}
    w = waiting(str(body.get("callback") or ""))
    if w is None:
        return 404, {"ok": False, "error": "This agent isn't waiting on that run (it ended, or was stopped)."}
    identity = body.get("identity") if isinstance(body.get("identity"), dict) else {}
    if not _same_identity(w, identity):
        return 403, {"ok": False, "error": "That run was started for someone else."}
    op = body.get("op")
    args = body.get("input")
    if op == "custom_tool":
        # Another custom tool (Phase 7b): this agent runs it with its custom_tools
        # tool, naming the run that asked, and hands back Flight Deck's result.
        other = str(body.get("custom_tool") or "").strip()
        if not other or not isinstance(args, dict):
            return 400, {"ok": False, "error": "A request names a custom tool and gives it an input object."}
        name = "custom_tools"
        args = {"action": "run", "tool": other, "input": args, "_xt_via": str(body.get("via") or ""), "_xt_raw": True}
    elif op == "tool":
        name = str(body.get("tool") or "").strip()
        if not name or not isinstance(args, dict):
            return 400, {"ok": False, "error": "A request names a tool and gives it an input object."}
        if name == "custom_tools":
            return 403, {"ok": False, "error": "A custom tool reaches other custom tools another way."}
    else:
        return 400, {"ok": False, "error": f"Unknown request {op!r}."}
    try:
        timeout = max(1.0, min(float(body.get("timeout_s") or 60), MAX_WAIT_S))
    except (TypeError, ValueError):
        timeout = 60.0
    owner_runs = bool(identity.get("owner_runs"))
    authority = mail_authority.automated("custom_tool", f"custom tool {w.tool}", "allow" if owner_runs else "deny")
    p_tok = speaker.bind(w.principal if w.member else None)
    g_tok = speaker.bind_grant(speaker.sanitize_grant(identity.get("grant")) if w.member else "")
    try:
        with mail_authority.bound(authority):
            result = await asyncio.wait_for(
                w.agent._execute_tool_with_guard(name, dict(args), interaction_label=f"custom_tool:{w.tool}",
                                                 session_id_override=w.session_id or None),
                timeout)
    except TimeoutError:
        return 200, {"ok": False, "error": f"{name} took longer than {int(timeout)} s."}
    except Exception as exc:  # noqa: BLE001 — a refusal or a failure, said to the tool
        log.info("A custom tool's request was refused or failed", tool=w.tool, asked=name, error=str(exc))
        return 200, {"ok": False, "error": str(exc) or type(exc).__name__}
    finally:
        speaker.reset_grant(g_tok)
        speaker.reset(p_tok)
    content = str(getattr(result, "content", "") or "")
    return 200, {"ok": bool(getattr(result, "success", False)), "content": content[:MAX_CONTENT],
                 "error": getattr(result, "error", None) or None}
