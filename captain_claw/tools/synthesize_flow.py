"""synthesize_flow — let an agent crystallize a repeatable goal into a Flow.

The agent describes a goal in plain language; Flight Deck compiles it to a
validated Flow DSL program (via a pooled model + the deterministic parser),
stores it in the agent's **scratch space** (origin=agent, call-only), and can
run it immediately. Re-synthesizing the same behaviour dedups to one entry.

This is a thin client: the real work + safety (validation, dedup, the trust
guard that stops a synthesized flow from calling a vetted world-acting flow)
lives in FD's ``/fd/flows/synthesize`` endpoint.
"""

from __future__ import annotations

import os
from typing import Any

from captain_claw.tools.registry import Tool, ToolResult

_DESCRIPTION = (
    "Turn a REPEATABLE goal into a reusable Flow (a small program Flight Deck runs "
    "deterministically). Use this ONLY when the work is repeatable, spans time, must "
    "be auditable, or should run in the background / be handed off — NOT for a one-off "
    "answer you can just give now. The flow is saved to your private scratch space and "
    "can be run immediately or called later from another flow."
)

_PARAMETERS = {
    "type": "object",
    "properties": {
        "goal": {
            "type": "string",
            "description": "Plain-language description of what the flow should do, end to end.",
        },
        "run": {
            "type": "boolean",
            "description": "Run the flow immediately after creating it (default false).",
        },
    },
    "required": ["goal"],
}


class SynthesizeFlowTool(Tool):
    """Synthesize a Flow from a natural-language goal (stored in scratch)."""

    name = "synthesize_flow"
    description = _DESCRIPTION
    parameters = _PARAMETERS
    timeout_seconds = 200.0

    def _fd_url(self, **kwargs: Any) -> str:
        session = kwargs.get("_session")
        agent = kwargs.get("_agent")
        md = (getattr(session, "metadata", {}) or {}) if session else {}
        url = str(md.get("fd_url") or "")
        if not url and agent is not None:
            url = str(getattr(agent, "_fd_url", "") or "")
        if not url:
            url = os.environ.get("FD_URL", "") or os.environ.get("FD_INTERNAL_URL", "")
        return url.rstrip("/")

    async def execute(self, goal: str = "", run: bool = False, **kwargs: Any) -> ToolResult:
        goal = str(goal or "").strip()
        if not goal:
            return ToolResult(success=False, error="A 'goal' describing the flow is required.")
        fd_url = self._fd_url(**kwargs)
        if not fd_url:
            return ToolResult(success=False, error="Flight Deck URL not available — cannot synthesize.")
        agent = kwargs.get("_agent")
        author = str(getattr(agent, "name", "") or "") if agent is not None else ""

        try:
            import httpx
        except ImportError:
            return ToolResult(success=False, error="httpx is required to reach Flight Deck.")

        try:
            async with httpx.AsyncClient(timeout=200.0) as client:
                resp = await client.post(
                    f"{fd_url}/fd/flows/synthesize",
                    json={"goal": goal, "author": author, "run": bool(run)},
                )
            data = resp.json() or {}
        except Exception as exc:
            return ToolResult(success=False, error=f"Cannot reach Flight Deck: {exc}")

        if not data.get("ok"):
            return ToolResult(success=False, error=str(data.get("error") or "synthesis failed"))

        name = data.get("name") or "flow"
        verb = "Reused existing" if data.get("reused") else "Created"
        msg = f"{verb} flow “{name}”."
        if run:
            out = str(data.get("output") or "")
            msg += f" Ran it → {data.get('status')}." + (f"\n\n{out[:800]}" if out else "")
        else:
            msg += " Saved to your scratch space — call it from a flow with `gosub`, or run it."
        return ToolResult(success=True, content=msg)
