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


async def run_action(user_id: str, action_id: str, args: dict[str, Any]) -> dict[str, Any]:
    """Resolve + validate + execute a catalog action. Returns
    ``{ok, content, error, action_id, risk, reversibility}``."""
    spec = action_catalog.get_action(action_id)
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
        _log.info("run_action %s via %s → ok=%s", action_id, tool, res.get("ok"))
        return {**res, **meta}

    return {"ok": False, "error": f"home '{spec.get('home')}' not supported yet", **meta}
