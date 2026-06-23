"""Deterministic action execution — the rail that runs a catalog action as a
concrete tool call, bypassing the LLM (see docs/jarvis-actions-events-plan.md, #1).

``run_tool_on_agent`` is the WS primitive: open a socket to the agent, send a
``run_tool`` request, await the matching ``tool_result``. The agent runs the named
tool through its guard and returns the real ``ToolResult``. ``run_action`` resolves
a catalog entry, validates args, and executes it on the user's strongest agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from typing import Any

from captain_claw.flight_deck import action_catalog

_log = logging.getLogger(__name__)


async def run_tool_on_agent(
    agent: dict[str, Any], tool: str, args: dict[str, Any], timeout: float = 60.0,
) -> dict[str, Any]:
    """Run ONE named tool with structured args on ``agent`` over its WebSocket,
    returning ``{ok, content, error}``. No LLM in the loop."""
    import websockets

    port = int(agent.get("port") or 0)
    if not port:
        return {"ok": False, "error": "no agent port"}
    host = agent.get("host") or "localhost"
    auth = str(agent.get("auth") or "")
    params = f"?token={auth}" if auth else ""
    url = f"ws://{host}:{port}/ws{params}"
    req_id = "rt_" + secrets.token_hex(6)
    try:
        async with websockets.connect(
            url, max_size=8 * 1024 * 1024, open_timeout=10, close_timeout=5,
        ) as ws:
            welcome = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if welcome.get("type") != "welcome":
                return {"ok": False, "error": "no welcome from agent"}
            while True:  # drain the session replay before issuing our request
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if m.get("type") == "replay_done":
                    break
            await ws.send(json.dumps({
                "type": "run_tool", "req_id": req_id, "tool": tool, "args": args,
            }))
            deadline = asyncio.get_event_loop().time() + timeout
            while True:
                rem = deadline - asyncio.get_event_loop().time()
                if rem <= 0:
                    return {"ok": False, "error": "timeout waiting for tool_result"}
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=min(rem, 30)))
                if m.get("type") == "tool_result" and m.get("req_id") == req_id:
                    return {
                        "ok": bool(m.get("ok")),
                        "content": m.get("content", "") or "",
                        "error": m.get("error"),
                    }
                # otherwise: a broadcast/replay artifact — keep waiting for ours
    except Exception as exc:
        _log.warning("run_tool_on_agent failed (tool=%s): %s", tool, exc)
        return {"ok": False, "error": str(exc)}


async def list_agent_tools(user_id: str) -> dict[str, Any]:
    """Discover the user's strongest agent's live tools + skills (the menu the
    Tools & Sources UI promotes from). GETs the agent's /api/orchestrator/skills."""
    from captain_claw.flight_deck.fd_dispatch import _strongest_agent
    agent = _strongest_agent(user_id)
    if not agent:
        return {"tools": [], "skills": [], "error": "no running agent"}
    host = agent.get("host") or "localhost"
    port = int(agent.get("port") or 0)
    auth = str(agent.get("auth") or "")
    url = f"http://{host}:{port}/api/orchestrator/skills" + (f"?token={auth}" if auth else "")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return {"tools": [], "skills": [], "error": f"agent returned {r.status_code}"}
            data = r.json()
            return {"tools": data.get("tools") or [], "skills": data.get("skills") or [],
                    "agent": agent.get("slug", "")}
    except Exception as exc:
        return {"tools": [], "skills": [], "error": str(exc)}


async def run_action(user_id: str, action_id: str, args: dict[str, Any]) -> dict[str, Any]:
    """Resolve + validate + execute a catalog action. Returns
    ``{ok, content, error, action_id, risk, reversibility}``."""
    spec = action_catalog.get_action(action_id, user_id)
    if not spec:
        return {"ok": False, "error": f"unknown action: {action_id}"}
    ok, err = action_catalog.validate_args(spec, args or {})
    if not ok:
        return {"ok": False, "error": err, "action_id": action_id}

    tool, tool_args = action_catalog.build_tool_call(spec, args or {})
    meta = {"action_id": action_id, "risk": spec["risk"],
            "reversibility": spec["reversibility"]}

    if spec.get("home") == "agent":
        from captain_claw.flight_deck.fd_dispatch import _strongest_agent
        agent = _strongest_agent(user_id)
        if not agent:
            return {"ok": False, "error": "no running agent to act through", **meta}
        res = await run_tool_on_agent(agent, tool, tool_args)
        # Grounded verification (#5): read the side effect back. 'absent' downgrades
        # to fail; 'unknown' (couldn't read) leaves the tool's success intact.
        verified = "skipped"
        if res.get("ok"):
            verified = await _verify_side_effect(user_id, spec, str(res.get("content") or ""), args or {}, agent)
            if verified == "absent":
                res["ok"] = False
                res["content"] = f"{res.get('content') or ''} [verification: side effect NOT found]".strip()
            elif verified == "unknown":
                res["content"] = f"{res.get('content') or ''} [unverified]".strip()
        _log.info("run_action %s via %s → ok=%s verify=%s", action_id, tool, res.get("ok"), verified)
        return {**res, "verified": verified, **meta}

    return {"ok": False, "error": f"home '{spec.get('home')}' not supported yet", **meta}


async def _verify_side_effect(
    user_id: str, spec: dict[str, Any], result_content: str, in_args: dict[str, Any], agent: dict[str, Any],
) -> str:
    """Read the action's side effect back. Returns 'confirmed' | 'absent' |
    'unknown' | 'skipped'. Gated by verify_enabled and the action's verify spec."""
    try:
        from captain_claw.flight_deck.autonomy import resolve_config
        if not resolve_config(user_id).get("verify_enabled", True):
            return "skipped"
    except Exception:
        pass
    vc = action_catalog.build_verify(spec, result_content, in_args)
    if not vc:
        return "skipped"
    res = await run_tool_on_agent(agent, vc["tool"], vc.get("args") or {})
    if res.get("ok"):
        return "confirmed"
    err = str(res.get("error") or res.get("content") or "").lower()
    if any(s in err for s in ("not found", "404", "no such", "does not exist", "no file")):
        return "absent"
    return "unknown"


async def undo_action(user_id: str, action: dict[str, Any]) -> dict[str, Any]:
    """Run the reverse call captured on a dispatched action's payload, undoing it.
    Returns ``{ok, content, error}``."""
    reverse = (action.get("payload") or {}).get("reverse")
    if not isinstance(reverse, dict) or not reverse.get("tool"):
        return {"ok": False, "error": "no reverse available for this action"}
    from captain_claw.flight_deck.fd_dispatch import _strongest_agent
    agent = _strongest_agent(user_id)
    if not agent:
        return {"ok": False, "error": "no running agent to undo through"}
    res = await run_tool_on_agent(agent, reverse["tool"], reverse.get("args") or {})
    _log.info("undo_action %s via %s → ok=%s", action.get("id"), reverse["tool"], res.get("ok"))
    return res
