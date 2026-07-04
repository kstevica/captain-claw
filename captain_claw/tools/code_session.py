"""`code` tool — start and read autonomous coding sessions on Flight Deck.

The chat-agent sibling of the `basna` tool: hands a coding task to the Code
studio (router → plan → auto-approved build → independent review → fix loop,
every phase a git commit). `start` is fire-and-forget — Flight Deck runs the
session server-side and notifies this agent over WebSocket when it finishes,
carrying the origin channel (web / WhatsApp / Telegram / …) so the relayed
result lands wherever the user asked.

Recursion is forbidden: agents spawned inside a coding run (CLAW_CODE_AGENT)
or by a Basna/Vatra ensemble (CLAW_BASNA_WORKER / CLAW_VATRA_WORKER) cannot
start coding sessions. The Code spawner also strips this tool from its
workers; this in-tool check is the authoritative guard.
"""

from __future__ import annotations

import os
from typing import Any

import structlog

from captain_claw.tools.registry import Tool, ToolResult

log = structlog.get_logger(__name__)

_WORKER_MARKERS = ("CLAW_CODE_AGENT", "CLAW_BASNA_WORKER", "CLAW_VATRA_WORKER")


class CodeSessionTool(Tool):
    name = "code"
    description = (
        "Start and read autonomous CODING sessions on Flight Deck's Code studio — a fleet of "
        "specialist agents that plans, builds, reviews, and fixes real software in a git repo. "
        "MANDATORY: when the user asks to build/implement/fix something 'in a coding session', "
        "'with Code', 'in the code studio', or to 'start a coding session', call this tool with "
        "action='start' — do NOT write the code yourself in chat. Fill `task` with a CONCRETE, "
        "self-contained description of what to build or change (derive it from the conversation), "
        "and `context` with any relevant background from this conversation (decisions, constraints, "
        "snippets). If the request is too vague to state a concrete task, ask the user to clarify "
        "instead of calling this tool. 'start' returns immediately (fire-and-forget) and notifies "
        "you when the session finishes — relay that result to the user. "
        "Reuse an existing workspace by passing `project` (and optionally `folder`), or continue a "
        "specific conversation with `project` + `session_id`. Use action='list' to see existing "
        "projects/sessions, 'status' for a live run, 'result' for a finished one."
    )
    timeout_seconds = 60.0

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "list", "status", "result"],
                "description": (
                    "start — launch (or continue) a coding session; list — the owner's projects/"
                    "folders/sessions; status — live state of one session; result — final outcome "
                    "of a finished session."
                ),
            },
            "task": {
                "type": "string",
                "description": (
                    "start: WHAT to build/change, concrete and self-contained (the coding fleet "
                    "sees only this + `context`, not your conversation)."
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "start: relevant background from this conversation — decisions already made, "
                    "constraints, error messages, snippets. Optional but strongly recommended."
                ),
            },
            "project": {
                "type": "string",
                "description": (
                    "Coding project name. start: omit to create one named from the task; give an "
                    "existing name to work in it. Required with session_id."
                ),
            },
            "folder": {
                "type": "string",
                "description": (
                    "start: VFS folder (git repo) inside the project. Omit to default to the "
                    "project name; a missing folder is created as a fresh repo."
                ),
            },
            "session_id": {
                "type": "string",
                "description": (
                    "start: continue this existing session instead of creating a new one (needs "
                    "`project` too). status/result: the session to read."
                ),
            },
            "title": {"type": "string", "description": "start: optional session title."},
        },
        "required": ["action"],
    }

    # ── identity / transport (same pattern as the basna tool) ─────────────

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

    def _identity(self) -> dict:
        from captain_claw.config import get_config
        try:
            cfg = get_config()
            port, auth = int(cfg.web.port or 0), cfg.web.auth_token or ""
        except Exception:  # noqa: BLE001
            port, auth = 0, ""
        return {"web_auth": auth, "source_port": port,
                "owner_id": os.environ.get("FD_OWNER_ID", "")}

    async def _post(self, fd_url: str, path: str, payload: dict) -> Any:
        import httpx
        body = {**self._identity(), **payload}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{fd_url}{path}", json=body)
        if resp.status_code == 404:
            return {"_error": "not found"}
        if resp.status_code == 403:
            return {"_error": "this agent's owner could not be resolved (not authorized)"}
        resp.raise_for_status()
        return resp

    # ── actions ────────────────────────────────────────────────────────────

    async def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        fd_url = self._get_fd_url(**kwargs)
        if not fd_url:
            return ToolResult(success=False, error="Flight Deck URL unavailable; cannot reach Code.")
        try:
            if action == "start":
                return await self._start(fd_url, **kwargs)
            if action == "list":
                return await self._list(fd_url)
            if action in ("status", "result"):
                return await self._read(fd_url, action, **kwargs)
            return ToolResult(success=False, error=f"Unknown action '{action}'.")
        except Exception as e:  # noqa: BLE001
            log.warning("code tool error", action=action, error=str(e))
            return ToolResult(success=False, error=f"Code request failed: {e}")

    async def _start(self, fd_url: str, **kwargs: Any) -> ToolResult:
        # Recursion guard — the authoritative check (spawn endpoints run in the
        # FD process where these markers aren't set, so the gate lives here).
        if any(str(os.environ.get(m, "")).strip().lower() in ("1", "true", "yes")
               for m in _WORKER_MARKERS):
            return ToolResult(
                success=False,
                error=("Coding sessions cannot be started from inside a coding run or a "
                       "Basna/Vatra ensemble (recursion is not allowed)."),
            )
        task = (kwargs.get("task") or "").strip()
        if not task:
            return ToolResult(success=False, error="Provide `task` describing what to build or change.")

        # Origin channel, so the completion report reaches the user where they
        # asked (WhatsApp / Telegram / web / glasses) — same as the basna tool.
        agent = kwargs.get("_agent")
        origin_platform, origin_user_id, origin_chat_id = "web", "", 0
        origin_kind, origin_address = "", ""
        try:
            from captain_claw.origin import get_session_origin
            _o = get_session_origin(getattr(agent, "session", None)) if agent else None
        except Exception:  # noqa: BLE001
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

        payload = {
            "task": task,
            "context": (kwargs.get("context") or "").strip(),
            "title": kwargs.get("title", "") or "",
            "project": kwargs.get("project", "") or "",
            "folder": kwargs.get("folder", "") or "",
            "session_id": kwargs.get("session_id", "") or "",
            "source_host": "localhost",
            "origin_platform": origin_platform,
            "origin_user_id": origin_user_id,
            "origin_chat_id": origin_chat_id,
            "origin_kind": origin_kind,
            "origin_address": origin_address,
        }
        r = await self._post(fd_url, "/fd/code/agent/start", payload)
        if isinstance(r, dict) and r.get("_error"):
            return ToolResult(success=False, error=r["_error"])
        data = r.json()
        if data.get("status") == "rejected":
            return ToolResult(success=True, content=f"Not started — {data.get('reason', 'at limit')}.")
        return ToolResult(success=True, content=(
            f"Started coding session **{data.get('title') or task[:60]}** "
            f"(project `{data.get('project')}`, session {data.get('session_id')}). "
            f"The Code fleet is planning and building autonomously — plans are auto-approved — "
            f"and I'll report the result back here when it finishes."
        ))

    async def _list(self, fd_url: str) -> ToolResult:
        r = await self._post(fd_url, "/fd/code/agent/list", {})
        if isinstance(r, dict) and r.get("_error"):
            return ToolResult(success=False, error=r["_error"])
        projects = r.json().get("projects", [])
        if not projects:
            return ToolResult(success=True, content="No coding projects yet.")
        lines = []
        for p in projects:
            lines.append(f"• {p['project']} — folders: {', '.join(p['folders']) or '(none)'}")
            for s in p.get("sessions", []):
                mark = " [agent]" if s.get("source") == "agent" else ""
                lines.append(f"    session {s['id']} · {s['title']} · folder {s['folder']} · {s['status']}{mark}")
        return ToolResult(success=True, content="\n".join(lines))

    async def _read(self, fd_url: str, action: str, **kwargs: Any) -> ToolResult:
        project = (kwargs.get("project") or "").strip()
        session_id = (kwargs.get("session_id") or "").strip()
        if not project or not session_id:
            return ToolResult(success=False, error="Provide `project` and `session_id` (see action='list').")
        r = await self._post(fd_url, f"/fd/code/agent/{action}",
                             {"project": project, "session_id": session_id})
        if isinstance(r, dict) and r.get("_error"):
            return ToolResult(success=False, error=r["_error"])
        d = r.json()
        if action == "status":
            commits = "\n".join(f"  {c.get('short', '')} {c.get('message', '')}" for c in d.get("last_commits", []))
            return ToolResult(success=True, content=(
                f"Session **{d.get('title')}** — status: {d.get('status')} "
                f"({d.get('messages', 0)} messages)\n"
                f"Recent commits:\n{commits or '  (none)'}\n\n"
                f"Last message:\n{d.get('last_message', '')}"
            ))
        commits = "\n".join(f"  {c.get('short', '')} {c.get('message', '')}" for c in d.get("commits", []))
        return ToolResult(success=True, content=(
            f"Session **{d.get('title')}** — status: {d.get('status')}\n"
            f"Commits:\n{commits or '  (none)'}\n\nResult:\n{d.get('result', '')}"
        ))
