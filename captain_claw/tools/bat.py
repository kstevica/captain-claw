"""The `bat` agent tool — launch and inspect a stubborn-finisher run.

Bat is the mode that keeps working a task until an independent judge says it is
genuinely done (see docs/bat-stubborn-loop-plan.md). An agent starts a Bat run
the same way it starts a Vatra run: fire-and-forget; Flight Deck drives it in
the background and delivers the result back here when it finishes.

Transport mirrors tools/basna.py exactly (pinned-URL gating, X-Agent-Secret,
web_auth/source_port/owner_id identity) so the owner is resolved from FD's own
spawn records, never from what the agent claims.
"""

from __future__ import annotations

import os
from typing import Any

import structlog

from captain_claw.tools.registry import Tool, ToolResult

log = structlog.get_logger(__name__)

# A Bat run may NOT be started from inside any ensemble worker (no recursion).
# NB: this is the Bat *launcher* guard; it does not stop a Bat worker from using
# the `vatra`/`basna` tools — those have their own guards that intentionally do
# not list CLAW_BAT_WORKER, so a Bat worker can start a Vatra/Basna sub-run.
_WORKER_MARKERS = (
    "CLAW_BAT_WORKER", "CLAW_BASNA_WORKER", "CLAW_VATRA_WORKER",
    "CLAW_COUNCIL_WORKER", "CLAW_CODE_AGENT",
)


def _in_worker() -> bool:
    return any(str(os.environ.get(m, "")).strip().lower() in ("1", "true", "yes")
               for m in _WORKER_MARKERS)


def _fd_url_pinned(fd_url: str) -> bool:
    from captain_claw.tools.flight_deck import _is_pinned_fd_url, _log_unpinned_fd_url_once
    if _is_pinned_fd_url(fd_url):
        return True
    _log_unpinned_fd_url_once(fd_url, "bat")
    return False


def _agent_secret_headers(fd_url: str | None = None) -> dict[str, str]:
    if fd_url is not None and not _fd_url_pinned(fd_url):
        return {}
    try:
        from captain_claw.flight_deck.agent_secret import get_or_create_agent_secret
        return {"X-Agent-Secret": get_or_create_agent_secret()}
    except Exception:  # noqa: BLE001 — the header is an upgrade, never a blocker
        return {}


class BatTool(Tool):
    name = "bat"
    description = (
        "Bat — the stubborn finisher. Start a long-running autonomous run that keeps "
        "working a task across retries and strategy changes until an independent judge "
        "says it is genuinely done; it survives restarts and reports back here when it "
        "finishes. Use it for a goal that must be completed no matter how many attempts "
        "it takes, not a quick answer. Actions: 'start' (task[, title, llm_usd_cap, steps]), "
        "'status' (run_id — quick state + spend), 'get' (run_id — full steps + recent events), "
        "'list' ([query, limit]), 'cancel' (run_id)."
    )
    timeout_seconds = 60.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "status", "get", "list", "cancel"],
                "description": "start a run, or inspect/stop one.",
            },
            "task": {"type": "string", "description": "For 'start': a clear, self-contained statement of the goal to finish."},
            "title": {"type": "string", "description": "Optional short title for the run ('start')."},
            "llm_usd_cap": {"type": "number", "description": "Optional hard ceiling on LLM spend in USD for the whole run ('start'); 0/omitted = unbounded."},
            "worker_mode": {"type": "string", "enum": ["plain", "archetype"], "description": "For 'start': 'plain' (default) spawns a generic full-toolset worker per step; 'archetype' assigns a best-fit specialist agent per step."},
            "steps": {
                "type": "array", "items": {"type": "string"},
                "description": "Optional explicit ordered step list ('start'); if omitted, Bat plans the steps itself.",
            },
            "run_id": {"type": "string", "description": "Target run id for 'status'/'get'/'cancel'."},
            "query": {"type": "string", "description": "Substring filter over title/task for 'list'."},
            "limit": {"type": "integer", "description": "Max runs for 'list' (default 30)."},
        },
        "required": ["action"],
    }

    # ── transport ────────────────────────────────────────────────────

    def _get_fd_url(self, **kwargs: Any) -> str:
        from captain_claw.fd_client import resolve_flight_deck_url
        session = kwargs.get("_session")
        agent = kwargs.get("_agent")
        metadata = (getattr(session, "metadata", {}) or {}) if session else {}
        fd_url = metadata.get("fd_url", "")
        if not fd_url and agent:
            fd_url = getattr(agent, "_fd_url", "") or ""
        return resolve_flight_deck_url(fd_url)

    def _own_port(self) -> int:
        try:
            from captain_claw.config import get_config
            return int(get_config().web.port or 0)
        except Exception:
            return 0

    def _own_auth(self) -> str:
        try:
            from captain_claw.config import get_config
            return get_config().web.auth_token or ""
        except Exception:
            return ""

    def _identity(self, fd_url: str) -> dict:
        return {
            "web_auth": self._own_auth() if _fd_url_pinned(fd_url) else "",
            "source_port": self._own_port(),
            "owner_id": os.environ.get("FD_OWNER_ID", ""),
        }

    async def _post(self, fd_url: str, path: str, payload: dict) -> Any:
        import httpx
        body = {**self._identity(fd_url), **payload}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{fd_url}{path}", json=body,
                                     headers=_agent_secret_headers(fd_url))
        if resp.status_code == 404:
            return {"_error": "not found"}
        if resp.status_code == 403:
            return {"_error": "this agent's owner could not be resolved (not authorized)"}
        resp.raise_for_status()
        return resp

    # ── dispatch ─────────────────────────────────────────────────────

    async def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        fd_url = self._get_fd_url(**kwargs)
        if not fd_url:
            return ToolResult(success=False, error="Flight Deck URL unavailable; cannot reach Bat.")
        try:
            if action == "start":
                return await self._start(fd_url, **kwargs)
            if action == "status":
                return await self._status(fd_url, **kwargs)
            if action == "get":
                return await self._get(fd_url, **kwargs)
            if action == "list":
                return await self._list(fd_url, **kwargs)
            if action == "cancel":
                return await self._cancel(fd_url, **kwargs)
            return ToolResult(success=False, error=f"Unknown action '{action}'.")
        except Exception as e:  # noqa: BLE001
            log.warning("bat tool error", action=action, error=str(e))
            return ToolResult(success=False, error=f"Bat request failed: {e}")

    async def _start(self, fd_url: str, **kwargs: Any) -> ToolResult:
        if _in_worker():
            return ToolResult(
                success=False,
                error="A Bat run cannot be started from inside another run (recursion is not allowed).",
            )
        task = (kwargs.get("task") or "").strip()
        if not task:
            return ToolResult(success=False, error="Provide `task` describing the goal to finish.")
        agent = kwargs.get("_agent")
        origin_platform, origin_user_id, origin_chat_id = "web", "", 0
        origin_kind, origin_address = "", ""
        try:
            from captain_claw.origin import get_session_origin
            _o = get_session_origin(getattr(agent, "session", None)) if agent else None
        except Exception:
            _o = None
        if _o:
            origin_kind, origin_address = _o["kind"], _o["address"]
            if origin_kind == "telegram":
                origin_platform = "telegram"
                origin_user_id = origin_address
                origin_chat_id = int(origin_address) if origin_address.isdigit() else 0
        elif agent and getattr(agent, "_telegram_chat_id", 0):
            origin_platform = "telegram"
            origin_user_id = str(getattr(agent, "_user_id", ""))
            origin_chat_id = int(getattr(agent, "_telegram_chat_id", 0))
            origin_kind, origin_address = "telegram", str(origin_chat_id)
        steps = kwargs.get("steps")
        payload = {
            "task": task,
            "title": kwargs.get("title", "") or "",
            "llm_usd_cap": float(kwargs.get("llm_usd_cap") or 0.0),
            "worker_mode": str(kwargs.get("worker_mode") or "plain"),
            "steps": [str(s) for s in steps] if isinstance(steps, (list, tuple)) else [],
            "origin_platform": origin_platform,
            "origin_user_id": origin_user_id,
            "origin_chat_id": origin_chat_id,
            "origin_kind": origin_kind,
            "origin_address": origin_address,
            "source_host": "localhost",
        }
        r = await self._post(fd_url, "/fd/bat/agent/start", payload)
        if isinstance(r, dict) and r.get("_error"):
            return ToolResult(success=False, error=r["_error"])
        data = r.json()
        if data.get("status") == "rejected":
            return ToolResult(success=True, content=f"Not started — {data.get('reason', 'at limit')}.")
        return ToolResult(success=True, content=(
            f"Started Bat run **{data.get('title') or task[:60]}** (run {data.get('run_id')}). "
            f"It will keep working until an independent judge says it's done; "
            f"I'll report the result back here when it finishes."
        ))

    async def _status(self, fd_url: str, **kwargs: Any) -> ToolResult:
        run_id = (kwargs.get("run_id") or "").strip()
        if not run_id:
            return ToolResult(success=False, error="Provide `run_id`.")
        r = await self._post(fd_url, "/fd/bat/agent/status", {"run_id": run_id})
        if isinstance(r, dict) and r.get("_error"):
            return ToolResult(success=False, error=r["_error"])
        d = r.json()
        if not d.get("found"):
            return ToolResult(success=False, error="Run not found.")
        return ToolResult(success=True, content=(
            f"Bat run {run_id}: **{d.get('status')}** — {d.get('done_steps', 0)}/{d.get('total_steps', 0)} "
            f"step(s) done, ${d.get('cumulative_usd', 0):.2f} spent"
            + (f" (cap ${d['llm_usd_cap']:.2f})" if d.get('llm_usd_cap') else "")
            + (f".\n{d.get('stopped_reason')}" if d.get("stopped_reason") else ".")
        ))

    async def _get(self, fd_url: str, **kwargs: Any) -> ToolResult:
        run_id = (kwargs.get("run_id") or "").strip()
        if not run_id:
            return ToolResult(success=False, error="Provide `run_id`.")
        r = await self._post(fd_url, "/fd/bat/agent/get", {"run_id": run_id})
        if isinstance(r, dict) and r.get("_error"):
            return ToolResult(success=False, error=r["_error"])
        d = r.json()
        if not d.get("found"):
            return ToolResult(success=False, error="Run not found.")
        run = d.get("run", {})
        steps = d.get("steps", [])
        lines = [f"# Bat run {run_id} — {run.get('status')}", run.get("title", "")]
        for s in steps:
            lines.append(f"- [{s.get('status')}] {s.get('title') or s.get('step_key')}"
                         + (f" (attempt {s['attempt']})" if s.get("attempt") else ""))
        truth = (run.get("truth") or "").strip()
        if truth:
            lines += ["", "## Result", truth[:4000]]
        return ToolResult(success=True, content="\n".join(l for l in lines if l is not None))

    async def _list(self, fd_url: str, **kwargs: Any) -> ToolResult:
        r = await self._post(fd_url, "/fd/bat/agent/list",
                             {"query": kwargs.get("query", "") or "",
                              "limit": int(kwargs.get("limit") or 30)})
        if isinstance(r, dict) and r.get("_error"):
            return ToolResult(success=False, error=r["_error"])
        runs = r.json().get("runs", [])
        if not runs:
            return ToolResult(success=True, content="No Bat runs.")
        lines = [f"- {run['id']}: {run.get('title') or run.get('task', '')[:50]} — {run.get('status')}"
                 for run in runs]
        return ToolResult(success=True, content="\n".join(lines))

    async def _cancel(self, fd_url: str, **kwargs: Any) -> ToolResult:
        run_id = (kwargs.get("run_id") or "").strip()
        if not run_id:
            return ToolResult(success=False, error="Provide `run_id`.")
        r = await self._post(fd_url, "/fd/bat/agent/cancel", {"run_id": run_id})
        if isinstance(r, dict) and r.get("_error"):
            return ToolResult(success=False, error=r["_error"])
        d = r.json()
        return ToolResult(success=bool(d.get("ok")),
                          content="Bat run cancel requested." if d.get("ok") else "Could not cancel (already finished?).")
