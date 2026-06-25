"""The `vatra` tool — a specialist's blackboard for a collaborating Vatra run.

Inside a Vatra run, each specialist owns one slice of the task. When it needs
something outside its slice, it does NOT block waiting for a teammate: it POSTS an
ASK to the shared blackboard and keeps working. A coordinator routes the ask to a
helper and writes the answer back; the specialist picks it up later via `inbox`,
and either way the reporter folds every answered ask into the final deliverable.

Context (which run / which subtask / how deep) is injected at spawn time via env
vars (CLAW_VATRA_SESSION / CLAW_VATRA_SUBTASK / CLAW_VATRA_OWNER / CLAW_VATRA_DEPTH).
Identity to Flight Deck is the agent's own web port + auth token (same trust model
as the `basna` and `flight_deck` peer tools). Outside a Vatra run the tool returns
a clear error, so registering it unconditionally is cheap.
"""

from __future__ import annotations

import os
from typing import Any

import structlog

from captain_claw.tools.registry import Tool, ToolResult

log = structlog.get_logger(__name__)


class VatraTool(Tool):
    name = "vatra"
    description = (
        "Collaborate with your teammates during a Vatra run via the shared blackboard. "
        "You own ONE slice of the task; when you need something another specialist should "
        "produce, do NOT wait — post an ask and keep working. "
        "'ask' — post a request for something outside your slice (`text`); returns immediately "
        "with an ask id. A teammate/helper answers it in the background. "
        "'inbox' — collect answers to your asks so far; pass `wait` (seconds, optional) to give "
        "the team a moment to respond before returning. Call it once you've done other work. "
        "Answers you don't receive in time are still folded into the final deliverable by the "
        "reporter, so never block on an ask."
    )
    timeout_seconds = 60.0

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["ask", "inbox"],
                "description": (
                    "'ask' — post a request to the blackboard (requires `text`). "
                    "'inbox' — return answers to your asks (optional `wait` seconds)."
                ),
            },
            "text": {"type": "string", "description": "For 'ask' — what you need, self-contained and specific."},
            "wait": {"type": "integer", "description": "For 'inbox' — seconds to wait for answers (0–30, default 0)."},
        },
        "required": ["action"],
    }

    # ── identity / transport (mirrors the basna tool) ─────────────────

    def _get_fd_url(self, **kwargs: Any) -> str:
        session = kwargs.get("_session")
        agent = kwargs.get("_agent")
        metadata = getattr(session, "metadata", {}) or {} if session else {}
        fd_url = metadata.get("fd_url", "")
        if not fd_url and agent:
            fd_url = getattr(agent, "_fd_url", "") or ""
        if not fd_url:
            fd_url = os.environ.get("FD_URL", "") or os.environ.get("FD_INTERNAL_URL", "")
        return fd_url

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

    def _identity(self) -> dict:
        return {
            "web_auth": self._own_auth(),
            "source_port": self._own_port(),
            "owner_id": os.environ.get("FD_OWNER_ID", ""),
        }

    def _context(self) -> dict:
        return {
            "session_id": os.environ.get("CLAW_VATRA_SESSION", "").strip(),
            "subtask_id": os.environ.get("CLAW_VATRA_SUBTASK", "").strip(),
            "owner": os.environ.get("CLAW_VATRA_OWNER", "").strip(),
            "depth": int(os.environ.get("CLAW_VATRA_DEPTH", "0") or 0),
        }

    async def _post(self, fd_url: str, path: str, payload: dict) -> Any:
        import httpx
        body = {**self._identity(), **payload}
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(f"{fd_url}{path}", json=body)
        if resp.status_code == 403:
            return {"_error": "not authorized for this Vatra session"}
        if resp.status_code == 404:
            return {"_error": "Vatra session not found"}
        resp.raise_for_status()
        return resp

    # ── entry point ──────────────────────────────────────────────────

    async def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        ctx = self._context()
        if not ctx["session_id"]:
            return ToolResult(success=False, error=(
                "The `vatra` tool only works inside a Vatra run (no run context found). "
                "There's nothing to do here."))
        fd_url = self._get_fd_url(**kwargs)
        if not fd_url:
            return ToolResult(success=False, error="Flight Deck URL unavailable; cannot reach the blackboard.")
        try:
            if action == "ask":
                return await self._ask(fd_url, ctx, **kwargs)
            if action == "inbox":
                return await self._inbox(fd_url, ctx, **kwargs)
            return ToolResult(success=False, error=f"Unknown action '{action}' (use 'ask' or 'inbox').")
        except Exception as e:
            log.warning("vatra tool error", action=action, error=str(e))
            return ToolResult(success=False, error=f"Vatra blackboard request failed: {e}")

    async def _ask(self, fd_url: str, ctx: dict, **kwargs: Any) -> ToolResult:
        text = (kwargs.get("text") or "").strip()
        if not text:
            return ToolResult(success=False, error="Provide `text` describing what you need.")
        r = await self._post(fd_url, "/fd/vatra/agent/ask", {
            "session_id": ctx["session_id"], "subtask_id": ctx["subtask_id"],
            "owner": ctx["owner"], "depth": ctx["depth"], "text": text,
        })
        if isinstance(r, dict) and r.get("_error"):
            return ToolResult(success=False, error=r["_error"])
        data = r.json()
        if data.get("status") == "rejected":
            return ToolResult(success=True, content=(
                f"Ask not posted — {data.get('reason', 'team is at its delegation budget')}. "
                f"Continue with your own slice; do the part you can yourself."))
        return ToolResult(success=True, content=(
            f"Posted ask #{data.get('ask_id')} to the team. Keep working on your slice; "
            f"call `vatra` action='inbox' later to collect the answer. If it doesn't arrive "
            f"in time, the reporter will still integrate it — don't wait on it."))

    async def _inbox(self, fd_url: str, ctx: dict, **kwargs: Any) -> ToolResult:
        wait = max(0, min(30, int(kwargs.get("wait") or 0)))
        r = await self._post(fd_url, "/fd/vatra/agent/inbox", {
            "session_id": ctx["session_id"], "owner": ctx["owner"], "wait": wait,
        })
        if isinstance(r, dict) and r.get("_error"):
            return ToolResult(success=False, error=r["_error"])
        data = r.json()
        answered = data.get("answered") or []
        pending = int(data.get("pending") or 0)
        if not answered:
            note = f"{pending} ask(s) still pending. " if pending else ""
            return ToolResult(success=True, content=(
                f"No answers yet. {note}Proceed with your slice — the reporter will fold in "
                f"anything that arrives later."))
        parts = [f"### Answer to your ask: {a.get('text', '')[:120]}\n{(a.get('answer') or '').strip()}"
                 for a in answered]
        tail = f"\n\n({pending} other ask(s) still pending.)" if pending else ""
        return ToolResult(success=True, content="\n\n".join(parts) + tail)
