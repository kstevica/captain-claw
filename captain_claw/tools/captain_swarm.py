"""The `captain_swarm` agent tool — propose and follow a Captain Swarm run.

Captain Swarm keeps improving one measured number: it tries ideas one at a
time, measures each itself, keeps what beats the best by more than noise, and
hands the owner the best result (docs/captain-swarm-plan.md). An agent can
propose a run, but a run it proposes never starts on its own: it waits for the
owner's Go on the Captain Swarm page, with the objective, the scorer and the
budget in front of them.

Transport mirrors tools/bat.py (pinned-URL gating, X-Agent-Secret,
web_auth/source_port/owner_id identity), so Flight Deck resolves the owner
from its own spawn records, never from what the agent claims.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from captain_claw.tools.bat import _agent_secret_headers, _fd_url_pinned, _in_worker
from captain_claw.tools.registry import Tool, ToolResult

log = structlog.get_logger(__name__)


class CaptainSwarmTool(Tool):
    name = "captain_swarm"
    description = (
        "Captain Swarm — a run that keeps improving one measured number: it tries ideas one at a time "
        "(each in its own git branch, by a coding worker), measures every attempt itself on open and "
        "held-out cases, keeps only gains bigger than the noise, and hands your owner the best result as "
        "a branch to review. A run you start WAITS for your owner's Go. Templates: 'mrav' (make Captain's "
        "small-model runtime pass more of its eval; options: model, harness, budget_usd, budget_hours, "
        "budget_max_experiments, per_round). Actions: 'start' (template[, options]), 'status' (run_id), "
        "'get' (run_id — the ledger of attempts), 'list', 'cancel' (run_id)."
    )
    timeout_seconds = 60.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["start", "status", "get", "list", "cancel"],
                       "description": "propose a run, or follow / stop one."},
            "template": {"type": "string", "enum": ["mrav"], "description": "For 'start': what to improve."},
            "options": {"type": "object", "description": "For 'start': the template's options, e.g. "
                        '{"model": "gemma4:e2b", "harness": "claude", "budget_usd": 20, "budget_hours": 6}.'},
            "run_id": {"type": "string", "description": "For 'status', 'get' and 'cancel'."},
            "limit": {"type": "integer", "description": "How many runs ('list') or attempts ('get') to show."},
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

    async def _post(self, fd_url: str, path: str, payload: dict) -> dict:
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{fd_url}/fd/captain-swarm/agent/{path}",
                                     json={**self._identity(fd_url), **payload},
                                     headers=_agent_secret_headers(fd_url))
        if resp.status_code == 403:
            return {"_error": "this agent's owner could not be resolved (not authorized)"}
        if resp.status_code == 404:
            return {"_error": "Captain Swarm isn't on this Flight Deck yet"}
        resp.raise_for_status()
        return resp.json()

    async def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        fd_url = self._get_fd_url(**kwargs)
        if not fd_url:
            return ToolResult(success=False, error="Flight Deck URL unavailable; cannot reach Captain Swarm.")
        if action == "start" and _in_worker():
            return ToolResult(success=False,
                              error="A Captain Swarm run can't be started from inside another run.")
        if action not in ("start", "status", "get", "list", "cancel"):
            return ToolResult(success=False, error=f"Unknown action '{action}'.")
        if action in ("status", "get", "cancel") and not str(kwargs.get("run_id") or "").strip():
            return ToolResult(success=False, error="Give the run_id.")
        payload: dict[str, Any] = {"run_id": str(kwargs.get("run_id") or ""),
                                   "limit": int(kwargs.get("limit") or 20)}
        if action == "start":
            agent = kwargs.get("_agent")
            options = kwargs.get("options") if isinstance(kwargs.get("options"), dict) else {}
            payload.update(template=str(kwargs.get("template") or "mrav"), options=options,
                           agent_name=str(getattr(agent, "name", "") or getattr(agent, "_agent_name", "") or ""))
        try:
            got = await self._post(fd_url, action, payload)
        except Exception as e:  # noqa: BLE001
            log.warning("captain_swarm tool error", action=action, error=str(e))
            return ToolResult(success=False, error=f"Captain Swarm request failed: {e}")
        if got.get("_error"):
            return ToolResult(success=False, error=got["_error"])
        if action == "start" and got.get("status") == "rejected":
            return ToolResult(success=False, error=f"Not started: {got.get('reason')}")
        return ToolResult(success=True, content=json.dumps(got, ensure_ascii=False, indent=1)[:12000])
