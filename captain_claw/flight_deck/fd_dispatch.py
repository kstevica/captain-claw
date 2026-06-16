"""Efferent dispatch (Topic 2) — turning a chosen action into real work.

The Arbiter decides *what*; this module does it. Every kind is delivered the same
way: as an instruction to the user's strongest running agent over its WebSocket,
with ``trigger_response`` so the agent carries it out with its own tools and
replies on the user's channel. This is the exact mechanism Basna uses to hand a
finished run back to its originating agent (``_notify_source_agent``) — generalised.

Auto-dispatch only happens at autonomy_level ``act_low_risk``+ for low-risk kinds,
or ``act`` for normal-risk; the Arbiter and the approve route gate it. Best-effort:
if no agent is running, dispatch reports failure and the action stays pending.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

_log = logging.getLogger(__name__)


def _strongest_agent(user_id: str) -> dict[str, Any] | None:
    """The user's most capable running agent, or None if none are up."""
    try:
        from captain_claw.flight_deck.consciousness import _model_rank, _user_agents

        agents = _user_agents(user_id)
        if not agents:
            return None
        return sorted(
            agents,
            key=lambda a: _model_rank(a.get("provider", ""), a.get("model", "")),
            reverse=True,
        )[0]
    except Exception:
        return None


def _instruction_for(action: dict[str, Any]) -> str:
    """Render an action into a concrete instruction the agent can act on."""
    kind = str(action.get("kind") or "nudge")
    title = str(action.get("title") or "").strip()
    rationale = str(action.get("rationale") or "").strip()
    if kind == "nudge":
        return (f"[Autonomous nudge] Proactively reach out to the user now: {title}. "
                f"{rationale} Keep it brief and in their language.")
    if kind == "basna":
        return f"Run a Basna on: {title}"
    if kind == "materialize_schedule":
        return (f"[Autonomous task] Set up a scheduled task: {title}. {rationale} "
                f"Use your scheduling tool.")
    # run_prompt and anything else: treat as a task prompt.
    return f"[Autonomous task] {title}\n\n{rationale}".strip()


async def deliver_to_agent(agent: dict[str, Any], content: str) -> bool:
    """Send a notification to one agent's WebSocket so it acts and replies to the
    user. Mirrors Basna's ``_notify_source_agent``. Returns True on delivery."""
    import websockets

    port = int(agent.get("port") or 0)
    if not port:
        return False
    host = agent.get("host") or "localhost"
    auth = str(agent.get("auth") or "")
    params = f"?token={auth}" if auth else ""
    url = f"ws://{host}:{port}/ws{params}"
    payload = {"type": "notification", "content": content, "trigger_response": True}
    try:
        async with websockets.connect(url, open_timeout=10, close_timeout=5) as ws:
            welcome = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if welcome.get("type") != "welcome":
                return False
            while True:  # skip the session replay before our message
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if msg.get("type") == "replay_done":
                    break
            await ws.send(json.dumps(payload))
            try:
                await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                pass
        return True
    except Exception as exc:
        _log.warning("dispatch delivery failed (port=%s): %s", port, exc)
        return False


async def dispatch_action(user_id: str, action: dict[str, Any]) -> dict[str, Any]:
    """Execute one action by handing it to the strongest agent. Returns
    ``{ok, target, note}`` — ok=False (with a note) when no agent is reachable."""
    agent = _strongest_agent(user_id)
    if not agent:
        return {"ok": False, "target": "", "note": "no running agent to dispatch to"}
    content = _instruction_for(action)
    ok = await deliver_to_agent(agent, content)
    return {
        "ok": ok,
        "target": agent.get("slug", "") if ok else "",
        "note": "" if ok else "agent delivery failed",
    }


def should_auto_dispatch(cfg: dict[str, Any], action: dict[str, Any]) -> bool:
    """Whether this action may fire WITHOUT human approval, per the dials.

    - ``act_low_risk``: only kinds in ``low_risk_kinds`` AND risk == 'low'.
    - ``act``: any non-high-risk action (high-risk still gated unless approval is
      not required).
    - otherwise (off / propose): never.
    """
    if not cfg.get("allow_auto_dispatch"):
        return False
    level = str(cfg.get("autonomy_level") or "off")
    risk = str(action.get("risk") or "normal")
    kind = str(action.get("kind") or "")
    if level == "act_low_risk":
        return kind in (cfg.get("low_risk_kinds") or []) and risk == "low"
    if level == "act":
        if risk == "high":
            return not cfg.get("high_risk_requires_approval", True)
        return True
    return False
