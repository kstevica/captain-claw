"""Read-only access to the owner's Basna sessions, like a queryable datastore.

Basna runs an ensemble of specialist agents over a task and produces a compiled
"truth", a confidence, per-agent outputs + tool activity, generated files, and a
cross-agent analysis. This tool lets an agent reach back into Flight Deck and
read its OWNER's Basna sessions — to reuse prior findings, pull a generated
file, or inspect which agents ran and what they concluded.

Identity: the agent is identified to Flight Deck by its own web port (the same
trust model as the `flight_deck` peer tool) and scoped to its owner; it also
sends FD_OWNER_ID as a fallback. No user token is handled here.

(Read side only — generating/starting a Basna run from an agent is a planned v2.)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import structlog

from captain_claw.tools.registry import Tool, ToolResult

log = structlog.get_logger(__name__)

# Above this, a fetched file is saved to the workspace instead of inlined.
_INLINE_FILE_LIMIT = 256 * 1024
_TEXT_EXTS = {".md", ".markdown", ".txt", ".csv", ".tsv", ".json", ".html",
              ".htm", ".xml", ".yaml", ".yml", ".rst", ".log", ".tex", ".py"}


class BasnaTool(Tool):
    name = "basna"
    description = (
        "Run and read Basna — a parallel ensemble of specialist agents that researches/analyses a "
        "task and merges their work into one answer. "
        "MANDATORY: whenever the user asks to run, start, execute, kick off, launch, or 'do a "
        "Basna' (e.g. 'run a basna on X', 'execute new basna', 'have the Basna team research X', "
        "'spin up a basna'), you MUST call this tool with action='start' and pass their request as "
        "`task`. In that case do NOT do the work yourself — do NOT call web_search, web_fetch, "
        "browser, or read to research it; the Basna ensemble's own agents do all of that. Your only "
        "job is to hand off the task and, later, relay the result. 'start' returns immediately "
        "(fire-and-forget) and notifies you when the run finishes. "
        "READ your owner's past/running sessions like a datastore: 'list' (browse/search), 'get' "
        "(full detail), 'agents' (per-agent runs + tool activity), 'output' (one agent's full "
        "output), 'truth' (compiled answer), 'analysis' (agreement/differences/blind-spots), "
        "'files' (list), 'get_file' (fetch a file). Use 'list' to find a session id."
    )
    timeout_seconds = 60.0

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "deepen", "list", "get", "agents", "output", "truth", "analysis", "files", "get_file"],
                "description": (
                    "'start' — launch a NEW autonomous Basna run on `task` (optional `title`, "
                    "`max_agents`); use this when the user asks to run/execute/start a Basna, and "
                    "do NOT research it yourself. Returns immediately, reports back when done. "
                    "'deepen' — launch a follow-up run on `session_id` that resolves that finished "
                    "run's BLIND SPOTS, seeded with its compiled truth; use when the user wants to "
                    "dig further into what the prior run missed. "
                    "'list' — your sessions (optional `query` substring, `status`, `limit`). "
                    "'get' — full session by `session_id`. "
                    "'agents' — per-agent runs for `session_id`. "
                    "'output' — full output of one agent (`session_id` + `archetype_id`). "
                    "'truth' — compiled answer + confidence for `session_id`. "
                    "'analysis' — cross-agent analysis for `session_id`. "
                    "'files' — files attached to `session_id`. "
                    "'get_file' — fetch `name` from `session_id`."
                ),
            },
            "task": {"type": "string", "description": "The task to run for 'start' — a clear, self-contained statement of what to research/produce."},
            "mode": {
                "type": "string",
                "enum": ["basna", "vatra"],
                "description": (
                    "Coordination mode for 'start' (default 'basna'). 'basna' = independent "
                    "ensemble: agents answer the whole task blind, outputs merged — best for "
                    "truth-finding (what's true, options, verification). 'vatra' = a collaborating "
                    "team: a Lead splits the task into complementary parts, specialists each build "
                    "one part, and a reporter assembles them into one deliverable — best for "
                    "building a multi-part artifact whose pieces fit together."
                ),
            },
            "title": {"type": "string", "description": "Optional short title for the run ('start'); auto-generated if blank."},
            "max_agents": {"type": "integer", "description": "Optional cap on agents for 'start' (1-10, default 6)."},
            "session_id": {"type": "string", "description": "Target session id (all read actions except 'list')."},
            "archetype_id": {"type": "string", "description": "Agent/archetype id for the 'output' action."},
            "name": {"type": "string", "description": "File name for the 'get_file' action."},
            "query": {"type": "string", "description": "Substring filter over title/intent/truth for 'list'."},
            "status": {"type": "string", "description": "Filter 'list' by status (routing/routed/running/done)."},
            "limit": {"type": "integer", "description": "Max sessions for 'list' (default 50)."},
        },
        "required": ["action"],
    }

    # ── identity / transport ─────────────────────────────────────────

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
        """Identity FD uses to resolve this agent's owner (auth token is primary)."""
        return {
            "web_auth": self._own_auth(),
            "source_port": self._own_port(),
            "owner_id": os.environ.get("FD_OWNER_ID", ""),
        }

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

    # ── entry point ──────────────────────────────────────────────────

    async def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        fd_url = self._get_fd_url(**kwargs)
        if not fd_url:
            return ToolResult(success=False, error="Flight Deck URL unavailable; cannot reach Basna.")
        try:
            if action == "start":
                return await self._start(fd_url, **kwargs)
            if action == "deepen":
                return await self._deepen(fd_url, **kwargs)
            if action == "list":
                return await self._list(fd_url, **kwargs)
            if action == "get":
                return await self._get(fd_url, **kwargs)
            if action == "agents":
                return await self._agents(fd_url, **kwargs)
            if action == "output":
                return await self._output(fd_url, **kwargs)
            if action == "truth":
                return await self._truth(fd_url, **kwargs)
            if action == "analysis":
                return await self._analysis(fd_url, **kwargs)
            if action == "files":
                return await self._files(fd_url, **kwargs)
            if action == "get_file":
                return await self._get_file(fd_url, **kwargs)
            return ToolResult(success=False, error=f"Unknown action '{action}'.")
        except Exception as e:
            log.warning("basna tool error", action=action, error=str(e))
            return ToolResult(success=False, error=f"Basna request failed: {e}")

    # ── actions ──────────────────────────────────────────────────────

    async def _start(self, fd_url: str, **kwargs: Any) -> ToolResult:
        """Launch a new autonomous Basna run; fire-and-forget with a callback."""
        # No recursion: a worker spawned by ANY run (Basna or Vatra, stamped with
        # its env marker) must never start another run. This worker-side check is
        # the real guard — the spawn endpoints run in the FD process where these
        # markers aren't set, so the gate has to live here, in the tool.
        if any(str(os.environ.get(m, "")).strip().lower() in ("1", "true", "yes")
               for m in ("CLAW_BASNA_WORKER", "CLAW_VATRA_WORKER")):
            return ToolResult(
                success=False,
                error="New runs cannot be started from inside a Basna/Vatra run (recursion is not allowed).",
            )
        task = (kwargs.get("task") or kwargs.get("query") or "").strip()
        if not task:
            return ToolResult(success=False, error="Provide `task` describing what the Basna run should do.")
        # Detect the originating channel so the completion result reaches the user
        # where they asked. Prefer the session's *durable* origin (set per inbound
        # message — covers whatsapp/glasses/channel/web); fall back to the telegram
        # per-user agent attrs. Both feed the same {kind,address} pair FD delivers on.
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
        payload = {
            "task": task,
            "title": kwargs.get("title", "") or "",
            "max_agents": int(kwargs.get("max_agents") or 6),
            "origin_platform": origin_platform,
            "origin_user_id": origin_user_id,
            "origin_chat_id": origin_chat_id,
            "origin_kind": origin_kind,
            "origin_address": origin_address,
            "source_host": "localhost",
        }
        mode = str(kwargs.get("mode") or "basna").strip().lower()
        path = "/fd/vatra/agent/start" if mode == "vatra" else "/fd/basna/agent/start"
        r = await self._post(fd_url, path, payload)
        if isinstance(r, dict) and r.get("_error"):
            return ToolResult(success=False, error=r["_error"])
        data = r.json()
        if data.get("status") == "rejected":
            return ToolResult(success=True, content=f"Not started — {data.get('reason', 'at concurrency limit')}.")
        if mode == "vatra":
            return ToolResult(success=True, content=(
                f"Started Vatra run **{data.get('title') or task[:60]}** "
                f"(collaborative team, session {data.get('session_id')}). "
                f"A Lead is splitting the task into parts; I'll report the assembled "
                f"result back here when it finishes."
            ))
        return ToolResult(success=True, content=(
            f"Started Basna run **{data.get('title') or task[:60]}** "
            f"({data.get('n_agents', '?')} agent(s), session {data.get('session_id')}). "
            f"It's running autonomously — I'll report the result back here when it finishes."
        ))

    async def _deepen(self, fd_url: str, **kwargs: Any) -> ToolResult:
        """Launch a follow-up run that resolves a finished run's blind spots."""
        if str(os.environ.get("CLAW_BASNA_WORKER", "")).strip().lower() in ("1", "true", "yes"):
            return ToolResult(
                success=False,
                error="Cannot start runs (including deepen) from inside a Basna run.",
            )
        sid = (kwargs.get("session_id") or "").strip()
        if not sid:
            return ToolResult(success=False, error="Provide `session_id` of the finished run to deepen.")
        r = await self._post(fd_url, "/fd/basna/agent/deepen", {"session_id": sid})
        if isinstance(r, dict) and r.get("_error"):
            return ToolResult(success=False, error=r["_error"])
        data = r.json()
        return ToolResult(success=True, content=(
            f"Started a deepen run **{data.get('title') or 'follow-up'}** "
            f"({data.get('n_agents', '?')} agent(s), session {data.get('session_id')}) "
            f"focused on the prior run's blind spots. I'll report back when it finishes."
        ))

    async def _list(self, fd_url: str, **kwargs: Any) -> ToolResult:
        r = await self._post(fd_url, "/fd/basna/agent/sessions", {
            "query": kwargs.get("query", "") or "",
            "status": kwargs.get("status", "") or "",
            "limit": int(kwargs.get("limit") or 50),
        })
        if isinstance(r, dict) and r.get("_error"):
            return ToolResult(success=False, error=r["_error"])
        sessions = r.json().get("sessions", [])
        if not sessions:
            return ToolResult(success=True, content="No Basna sessions found.")
        lines = ["| id | title | domain | status | conf | agents | files | updated |",
                 "|----|-------|--------|--------|------|--------|-------|---------|"]
        for s in sessions:
            lines.append(
                f"| {s['id']} | {s.get('title') or s.get('intent','')[:40]} | {s.get('domain','')} "
                f"| {s.get('status','')} | {s.get('confidence',0):.2f} | {s.get('n_agents',0)} "
                f"| {s.get('n_files',0)} | {(s.get('updated_at') or '')[:19]} |"
            )
        return ToolResult(success=True, content="\n".join(lines))

    async def _fetch_session(self, fd_url: str, session_id: str):
        if not session_id:
            return None, ToolResult(success=False, error="session_id is required.")
        r = await self._post(fd_url, "/fd/basna/agent/session", {"session_id": session_id})
        if isinstance(r, dict) and r.get("_error"):
            return None, ToolResult(success=False, error=r["_error"])
        return r.json(), None

    async def _get(self, fd_url: str, **kwargs: Any) -> ToolResult:
        sess, err = await self._fetch_session(fd_url, kwargs.get("session_id", ""))
        if err:
            return err
        route = _safe_json(sess.get("route"), {})
        files = _safe_json(sess.get("files"), [])
        parts = [
            f"# {sess.get('title') or '(untitled)'}",
            f"**id:** {sess['id']}  ·  **status:** {sess.get('status','')}  ·  "
            f"**domain:** {sess.get('domain','')}  ·  **difficulty:** {sess.get('difficulty','')}  ·  "
            f"**merge:** {sess.get('merge_kind','')}  ·  **confidence:** {sess.get('confidence',0):.2f}",
            "",
            f"**Task:** {sess.get('intent','')}",
        ]
        selected = route.get("selected") or []
        if selected:
            parts.append("\n**Agents routed:**")
            for s in selected:
                parts.append(f"- {s.get('role') or s.get('archetype_id')} "
                             f"({s.get('archetype_id')}, tier={s.get('tier','')}, "
                             f"weight={s.get('prior_weight', s.get('weight','?'))}) — {s.get('why','')}")
        if files:
            parts.append("\n**Files:** " + ", ".join(
                f"{f['name']} ({f.get('kind','')}, {f.get('size',0)}B)" for f in files))
        truth = (sess.get("truth") or "").strip()
        if truth:
            parts.append("\n**Compiled truth:**\n" + truth)
        parts.append("\n_Use action='agents' for per-agent detail, 'analysis' for the cross-agent comparison._")
        return ToolResult(success=True, content="\n".join(parts))

    async def _agents(self, fd_url: str, **kwargs: Any) -> ToolResult:
        sid = kwargs.get("session_id", "")
        if not sid:
            return ToolResult(success=False, error="session_id is required.")
        r = await self._post(fd_url, "/fd/basna/agent/runs", {"session_id": sid})
        if isinstance(r, dict) and r.get("_error"):
            return ToolResult(success=False, error=r["_error"])
        runs = r.json().get("runs", [])
        if not runs:
            return ToolResult(success=True, content="No agent runs recorded for this session.")
        parts = []
        for run in runs:
            ok = "✓" if run.get("success") == 1 else ("✗" if run.get("success") == 0 else "·")
            actions = _safe_json(run.get("actions"), [])
            act_str = ", ".join(a.get("tool", "?") for a in actions) if actions else "—"
            out = (run.get("output") or "").strip()
            preview = out[:400] + ("…" if len(out) > 400 else "")
            parts.append(
                f"### {ok} {run.get('role') or run.get('archetype_id')} "
                f"(`{run.get('archetype_id')}`)\n"
                f"tier={run.get('tier','')} · weight={run.get('weight_at_run',0):.2f} · "
                f"latency={run.get('latency_ms',0)}ms · tools: {act_str}\n\n"
                f"{preview or '(no text output)'}"
            )
        parts.append("\n_Use action='output' with `archetype_id` for an agent's full output._")
        return ToolResult(success=True, content="\n\n".join(parts))

    async def _output(self, fd_url: str, **kwargs: Any) -> ToolResult:
        sid = kwargs.get("session_id", "")
        aid = kwargs.get("archetype_id", "")
        if not sid or not aid:
            return ToolResult(success=False, error="session_id and archetype_id are required.")
        r = await self._post(fd_url, "/fd/basna/agent/runs", {"session_id": sid})
        if isinstance(r, dict) and r.get("_error"):
            return ToolResult(success=False, error=r["_error"])
        runs = [x for x in r.json().get("runs", []) if x.get("archetype_id") == aid]
        if not runs:
            return ToolResult(success=False, error=f"No run for archetype '{aid}' in this session.")
        out = (runs[0].get("output") or "").strip()
        return ToolResult(success=True, content=out or "(this agent produced no text output)")

    async def _truth(self, fd_url: str, **kwargs: Any) -> ToolResult:
        sess, err = await self._fetch_session(fd_url, kwargs.get("session_id", ""))
        if err:
            return err
        truth = (sess.get("truth") or "").strip()
        if not truth:
            return ToolResult(success=True, content="This session has no compiled truth yet.")
        return ToolResult(success=True,
                          content=f"**Confidence: {sess.get('confidence',0):.2f}**\n\n{truth}")

    async def _analysis(self, fd_url: str, **kwargs: Any) -> ToolResult:
        sess, err = await self._fetch_session(fd_url, kwargs.get("session_id", ""))
        if err:
            return err
        a = _safe_json(sess.get("analysis"), {})
        if not a:
            return ToolResult(success=True, content="No cross-agent analysis (single-agent or not yet run).")
        parts = []
        if a.get("agreement"):
            parts.append("**Agreement:**\n" + "\n".join(f"- {x}" for x in a["agreement"]))
        if a.get("differences"):
            parts.append("**Differences:**")
            for d in a["differences"]:
                pos = "; ".join(f"{p.get('by')}: {p.get('stance')}" for p in d.get("positions", []))
                parts.append(f"- {d.get('point')} — {pos}")
        if a.get("unique"):
            parts.append("**Unique insights:**\n" + "\n".join(
                f"- {u.get('by')}: {u.get('insight')}" for u in a["unique"]))
        if a.get("blind_spots"):
            parts.append("**Blind spots:**\n" + "\n".join(f"- {x}" for x in a["blind_spots"]))
        return ToolResult(success=True, content="\n\n".join(parts) or "Analysis is empty.")

    async def _files(self, fd_url: str, **kwargs: Any) -> ToolResult:
        sess, err = await self._fetch_session(fd_url, kwargs.get("session_id", ""))
        if err:
            return err
        files = _safe_json(sess.get("files"), [])
        if not files:
            return ToolResult(success=True, content="No files on this session.")
        lines = ["| name | kind | size | mime | agent |", "|------|------|------|------|-------|"]
        for f in files:
            lines.append(f"| {f['name']} | {f.get('kind','')} | {f.get('size',0)} "
                         f"| {f.get('mime','')} | {f.get('agent','')} |")
        lines.append("\n_Use action='get_file' with `name` to retrieve one._")
        return ToolResult(success=True, content="\n".join(lines))

    async def _get_file(self, fd_url: str, **kwargs: Any) -> ToolResult:
        sid = kwargs.get("session_id", "")
        name = kwargs.get("name", "")
        if not sid or not name:
            return ToolResult(success=False, error="session_id and name are required.")
        import httpx
        body = {**self._identity(), "session_id": sid, "name": name}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{fd_url}/fd/basna/agent/file", json=body)
        if resp.status_code == 404:
            return ToolResult(success=False, error="file not found")
        if resp.status_code == 403:
            return ToolResult(success=False, error="not authorized for this session")
        resp.raise_for_status()
        data = resp.content
        ext = Path(name).suffix.lower()
        is_text = ext in _TEXT_EXTS
        if is_text and len(data) <= _INLINE_FILE_LIMIT:
            try:
                return ToolResult(success=True, content=data.decode("utf-8", errors="replace"))
            except Exception:
                pass
        # Binary or large → save into the agent's workspace and return the path.
        base = kwargs.get("_saved_base_path") or kwargs.get("_runtime_base_path")
        if not base:
            return ToolResult(success=False,
                              error="No workspace to save the file into; cannot return binary content.")
        dest_dir = Path(base) / "basna" / sid
        dest_dir.mkdir(parents=True, exist_ok=True)
        safe = Path(name).name
        (dest_dir / safe).write_bytes(data)
        rel = f"basna/{sid}/{safe}"
        return ToolResult(success=True,
                          content=f"Saved `{safe}` ({len(data)} bytes) to your workspace at `{rel}`. "
                                  f"Use the `read` tool to open it.")


def _safe_json(s: Any, default: Any) -> Any:
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s) if s else default
    except (json.JSONDecodeError, TypeError):
        return default
