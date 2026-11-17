"""The `first_mate` agent tool — talk to the owner's First Mate factories.

First Mate runs standing jobs ("factories") for the owner, each with a charter
they approved (docs/first-mate-plan.md §11). Through this tool an agent, on its
owner's turn, lists the factories, reads one, runs, pauses or resumes it, and
reads the week's report. It never approves what a factory would do, carries it
out, or raises its trust: those stay the owner's own, on First Mate and the
Line. It refuses inside any worker run and in rehearsals (rehearsal_turn).

Transport mirrors tools/bat.py (pinned-URL gating, X-Agent-Secret,
web_auth/source_port/owner_id identity), so Flight Deck resolves the owner
from its own spawn records, never from what the agent claims.
"""

from __future__ import annotations

from typing import Any

import structlog

from captain_claw.tools.bat import _agent_secret_headers, _fd_url_pinned, _in_worker
from captain_claw.tools.registry import Tool, ToolResult

log = structlog.get_logger(__name__)

ACTIONS = ("list", "status", "run", "pause", "resume", "report")
_NEEDS_FACTORY = ("status", "run", "pause", "resume")


class FirstMateTool(Tool):
    name = "first_mate"
    description = (
        "First Mate — your owner's standing jobs ('factories'), each running on a charter they approved and "
        "checked every run. Actions: 'list' (every factory with its state, trust, health, last and next run, "
        "and what waits for the owner), 'status' (factory — its charter, health, this week, trust, cases, last "
        "run and what it would do), 'run' (factory[, note] — like pressing Run now; a factory that asks first "
        "only rehearses), 'pause' (factory[, note as the reason]), 'resume' (factory), 'report' (week: 'last' "
        "or 'this'). Name a factory by its name; part of it is enough when only one fits. You can't approve "
        "what a factory would do, carry it out or change its trust: tell the owner to do that on First Mate "
        "or the Line."
    )
    timeout_seconds = 60.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ACTIONS)},
            "factory": {"type": "string", "description": "For status, run, pause, resume: the factory's name."},
            "note": {"type": "string", "description": "For run: a note for this run. For pause: why."},
            "week": {"type": "string", "enum": ["last", "this"], "description": "For report: which week."},
        },
        "required": ["action"],
    }

    def _get_fd_url(self, **kwargs: Any) -> str:
        from captain_claw.fd_client import resolve_flight_deck_url
        session = kwargs.get("_session")
        agent = kwargs.get("_agent")
        metadata = (getattr(session, "metadata", {}) or {}) if session else {}
        fd_url = metadata.get("fd_url", "")
        if not fd_url and agent:
            fd_url = getattr(agent, "_fd_url", "") or ""
        return resolve_flight_deck_url(fd_url)

    def _identity(self, fd_url: str) -> dict:
        import os
        try:
            from captain_claw.config import get_config
            web = get_config().web
            port, auth = int(web.port or 0), web.auth_token or ""
        except Exception:
            port, auth = 0, ""
        return {"web_auth": auth if _fd_url_pinned(fd_url) else "", "source_port": port,
                "owner_id": os.environ.get("FD_OWNER_ID", "")}

    async def _post(self, fd_url: str, payload: dict) -> dict:
        import httpx
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(f"{fd_url}/fd/first-mate/agent/tool",
                                     json={**self._identity(fd_url), **payload},
                                     headers=_agent_secret_headers(fd_url))
        if resp.status_code in (400, 403):
            try:
                detail = str(resp.json().get("detail") or "")
            except ValueError:
                detail = ""
            return {"_error": detail or "this agent's owner could not be resolved (not authorized)"}
        if resp.status_code == 404:
            return {"_error": "First Mate isn't on this Flight Deck yet"}
        resp.raise_for_status()
        return resp.json()

    async def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        if _in_worker():
            return ToolResult(success=False, error="First Mate is for your owner's own turns, not inside a run.")
        if action not in ACTIONS:
            return ToolResult(success=False, error=f"Unknown action '{action}'. Use one of: {', '.join(ACTIONS)}.")
        factory = str(kwargs.get("factory") or "").strip()
        if action in _NEEDS_FACTORY and not factory:
            return ToolResult(success=False, error="Say which factory (its name). 'list' shows them.")
        fd_url = self._get_fd_url(**kwargs)
        if not fd_url:
            return ToolResult(success=False, error="Flight Deck URL unavailable; cannot reach First Mate.")
        payload = {"action": action, "factory": factory, "note": str(kwargs.get("note") or "")[:500],
                   "week": str(kwargs.get("week") or "last")}
        try:
            got = await self._post(fd_url, payload)
        except Exception as e:  # noqa: BLE001
            log.warning("first_mate tool error", action=action, error=str(e))
            return ToolResult(success=False, error=f"First Mate request failed: {e}")
        if got.get("_error"):
            return ToolResult(success=False, error=got["_error"])
        if not got.get("ok"):
            return ToolResult(success=False, error=str(got.get("why") or "First Mate said no"))
        return ToolResult(success=True, content=str(got.get("text") or "")[:12000])
