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
        "The shared team board for a Vatra run — a live shared memory where every teammate's "
        "notes, outputs and files appear as they work. You own ONE slice; use the board to build "
        "ON your teammates' work instead of guessing or duplicating it.\n"
        "'search' — find what teammates have produced by keyword (`query`) — use this FIRST "
        "whenever your piece needs a fact, figure, decision, or section another piece owns.\n"
        "'read' — recent entries from teammates (optional `kind` = note|output|narration|file).\n"
        "'post' — share a key finding, decision, or your draft so others can use it (`text`, "
        "optional `title`).\n"
        "'ask' — when you need a teammate to DO new work, post a request (`text`); a helper "
        "answers in the background. 'inbox' — collect answers to your asks (optional `wait`).\n"
        "Prefer search/read/post — the board is always there; only 'ask' when new work is needed."
    )
    timeout_seconds = 60.0

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "read", "post", "ask", "inbox"],
                "description": (
                    "'search' — keyword search the team board (`query`). "
                    "'read' — recent teammate entries (optional `kind`). "
                    "'post' — share a note to the board (`text`, optional `title`). "
                    "'ask' — request new work from a teammate (`text`). "
                    "'inbox' — answers to your asks (optional `wait`)."
                ),
            },
            "query": {"type": "string", "description": "For 'search' — keywords to find teammates' work."},
            "text": {"type": "string", "description": "For 'post'/'ask' — the note to share, or what you need."},
            "title": {"type": "string", "description": "For 'post' — optional short label."},
            "kind": {"type": "string", "description": "For 'read' — filter to note|output|narration|file."},
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
            if action == "search":
                return await self._search(fd_url, ctx, **kwargs)
            if action == "read":
                return await self._read(fd_url, ctx, **kwargs)
            if action == "post":
                return await self._post_note(fd_url, ctx, **kwargs)
            if action == "ask":
                return await self._ask(fd_url, ctx, **kwargs)
            if action == "inbox":
                return await self._inbox(fd_url, ctx, **kwargs)
            return ToolResult(success=False, error=f"Unknown action '{action}' (use search/read/post/ask/inbox).")
        except Exception as e:
            log.warning("vatra tool error", action=action, error=str(e))
            return ToolResult(success=False, error=f"Vatra board request failed: {e}")

    @staticmethod
    def _fmt_entries(entries: list[dict]) -> str:
        lines = []
        for e in entries:
            head = f"[{e.get('kind', '')}] {e.get('from', '?')}"
            if e.get("title"):
                head += f" · {e['title']}"
            body = " ".join(str(e.get("content", "")).split())
            if len(body) > 1200:
                body = body[:1200] + " …"
            lines.append(f"### {head}\n{body}")
        return "\n\n".join(lines)

    async def _search(self, fd_url: str, ctx: dict, **kwargs: Any) -> ToolResult:
        q = (kwargs.get("query") or "").strip()
        if not q:
            return ToolResult(success=False, error="Provide `query` keywords to search the board.")
        r = await self._post(fd_url, "/fd/vatra/agent/board/search", {
            "session_id": ctx["session_id"], "owner": ctx["owner"], "query": q, "limit": 20,
        })
        if isinstance(r, dict) and r.get("_error"):
            return ToolResult(success=False, error=r["_error"])
        entries = r.json().get("entries") or []
        if not entries:
            return ToolResult(success=True, content=(
                f"No teammate board entries match {q!r} yet. They may not have produced it — "
                f"do the part you can, or post a note / ask if you need new work."))
        return ToolResult(success=True, content=f"Board matches for {q!r}:\n\n{self._fmt_entries(entries)}")

    async def _read(self, fd_url: str, ctx: dict, **kwargs: Any) -> ToolResult:
        r = await self._post(fd_url, "/fd/vatra/agent/board/read", {
            "session_id": ctx["session_id"], "owner": ctx["owner"],
            "kind": (kwargs.get("kind") or "").strip(), "limit": 40,
        })
        if isinstance(r, dict) and r.get("_error"):
            return ToolResult(success=False, error=r["_error"])
        entries = r.json().get("entries") or []
        if not entries:
            return ToolResult(success=True, content=(
                "The team board has nothing from your teammates yet — they're still working. "
                "Proceed with your slice; check back with 'search' or 'read' as you go."))
        return ToolResult(success=True, content=f"Recent team board:\n\n{self._fmt_entries(entries)}")

    async def _post_note(self, fd_url: str, ctx: dict, **kwargs: Any) -> ToolResult:
        text = (kwargs.get("text") or "").strip()
        if not text:
            return ToolResult(success=False, error="Provide `text` to share on the board.")
        r = await self._post(fd_url, "/fd/vatra/agent/board/post", {
            "session_id": ctx["session_id"], "owner": ctx["owner"], "subtask_id": ctx["subtask_id"],
            "kind": "note", "title": (kwargs.get("title") or "").strip(), "text": text,
        })
        if isinstance(r, dict) and r.get("_error"):
            return ToolResult(success=False, error=r["_error"])
        return ToolResult(success=True, content=(
            "Shared to the team board — teammates can now search/read it. Keep working."))

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
