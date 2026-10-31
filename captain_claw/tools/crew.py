"""`crew` tool — hand coding work to Claude Code, Codex, OpenCode or Pi and supervise it.

The agent is the first mate: the person talks to it, and it runs the crew. Each
task gets one of those CLIs in a git worktree of its own on Flight Deck, which
watches it without spending tokens and calls this agent back only when something
needs saying, on the channel the task was asked for from.

Shown only to an owner an admin approved (``crew_client``); Flight Deck checks the
approval again on every call. An agent can ask for a merge but not do one: the
person's Approve does it. Recursion is refused, as in the `code` tool.
"""

from __future__ import annotations

import os
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

_WORKER_MARKERS = ("CLAW_CODE_AGENT", "CLAW_BASNA_WORKER", "CLAW_VATRA_WORKER")


def _task_line(t: dict) -> str:
    evidence = t.get("evidence") or {}
    git = f" · {evidence.get('commits', 0)} commit(s)" if t.get("kind") == "ship" else ""
    return (f"{t.get('id')} · {t.get('status')} · {t.get('harness_label')} · {t.get('repo_label')}{git} · "
            f"{t.get('title')}" + (f" · {t['note'][:200]}" if t.get("note") else ""))


class CrewTool(Tool):
    name = "crew"
    description = (
        "Run coding CLIs as your crew: Claude Code, Codex, OpenCode or Pi, each in its own git worktree, "
        "several at once. Use this when the user asks you to have one of them do something, or to get coding "
        "work done in parallel: fixes, features, audits, investigations. You supervise; you don't do the work "
        "yourself. 'dispatch' returns at once, and you'll be called back when a task finishes or needs a "
        "decision, so don't poll and don't say you're waiting: tell the user it's on its way. Report outcomes "
        "and decisions, not mechanics. Say what git shows (commits, uncommitted changes) rather than what a "
        "worker claims. "
        "Actions: options (the CLIs you may use, their logins, and the projects); dispatch; list; status; "
        "steer (the next instruction, or the user's answer to a worker's question); stop (keeps the work); "
        "pr (push and open a pull request); request_merge (asks the user to Approve the merge; merging is "
        "their decision, so only ask when they want it merged). You can't merge or discard work yourself."
    )
    timeout_seconds = 200.0

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["options", "dispatch", "list", "status", "steer", "stop", "pr", "request_merge"],
            },
            "harness": {
                "type": "string",
                "enum": ["claude", "codex", "opencode", "pi"],
                "description": "dispatch: which CLI does the work. Use the one the user named; otherwise see 'options'.",
            },
            "brief": {
                "type": "string",
                "description": (
                    "dispatch: the task, complete and self-contained. The worker sees only this, not your "
                    "conversation: include the goal, what done means, and any decisions already made."
                ),
            },
            "title": {"type": "string", "description": "dispatch: a short title, optional."},
            "kind": {
                "type": "string",
                "enum": ["ship", "scout"],
                "description": "dispatch: ship changes code on a branch (default); scout only investigates and reports.",
            },
            "deliver": {
                "type": "string",
                "enum": ["branch", "pr"],
                "description": "dispatch, ship: leave the work on a branch (default), or open a pull request when it's done.",
            },
            "project": {"type": "string", "description": "dispatch: a Code project to work in (created if new)."},
            "folder": {"type": "string", "description": "dispatch: a folder in that project. Omit to use the project's name."},
            "repo_url": {"type": "string", "description": "dispatch: a repository address to clone and work in, instead of a project."},
            "repo_path": {"type": "string", "description": "dispatch: a folder on this computer, instead of a project. Admins only."},
            "model": {"type": "string", "description": "dispatch: a model for the CLI, only if the user asked for one."},
            "effort": {"type": "string", "description": "dispatch: reasoning effort for the CLI, only if the user asked for one."},
            "task_id": {"type": "string", "description": "status, steer, stop, pr, request_merge: the task."},
            "text": {"type": "string", "description": "steer: what to tell the worker."},
        },
        "required": ["action"],
    }

    async def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        from captain_claw import crew_client

        if any(str(os.environ.get(m, "")).strip().lower() in ("1", "true", "yes") for m in _WORKER_MARKERS):
            return ToolResult(success=False, error="Crew can't be used from inside a coding run or an ensemble.")
        if not crew_client.under_flight_deck():
            return ToolResult(success=False, error=crew_client.NO_FLIGHT_DECK)
        if crew_client.member_call():
            return ToolResult(success=False, error=crew_client.MEMBER_REFUSED)
        try:
            handler = getattr(self, f"_do_{action}", None)
            if handler is None:
                return ToolResult(success=False, error=f"Unknown action '{action}'.")
            return await handler(crew_client, **kwargs)
        except Exception as exc:  # noqa: BLE001
            log.warning("crew tool error", action=action, error=str(exc))
            return ToolResult(success=False, error=f"Crew request failed: {exc}")

    @staticmethod
    def _refused(status: int, data: Any) -> ToolResult | None:
        if status < 300:
            return None
        detail = data.get("detail") if isinstance(data, dict) else data
        return ToolResult(success=False, error=str(detail or f"Flight Deck answered {status}"))

    async def _do_options(self, client: Any, **_: Any) -> ToolResult:
        status, data = await client.call("GET", "options")
        if (refused := self._refused(status, data)) is not None:
            return refused
        lines = ["CLIs:"]
        for h in (data.get("me") or {}).get("harnesses", []):
            state = "not installed" if not h.get("installed") else (
                f"ready, on {h['login']['how']}" if h["login"].get("ok") else "no login yet (Crew → Logins)")
            lines.append(f"  {h['id']} ({h['label']}): {state}")
        repos = data.get("repos") or {}
        lines.append("Projects:")
        lines.extend(f"  {p['project']}: {', '.join(p['folders']) or '(no folders)'}"
                     for p in repos.get("projects", []))
        if repos.get("recent"):
            lines.append("Used lately:")
            lines.extend(f"  {r['repo']}" for r in repos["recent"])
        return ToolResult(success=True, content="\n".join(lines))

    async def _do_dispatch(self, client: Any, **kwargs: Any) -> ToolResult:
        harness = str(kwargs.get("harness") or "").strip()
        brief = str(kwargs.get("brief") or "").strip()
        if not harness or not brief:
            return ToolResult(success=False, error="Provide `harness` and `brief` (see action='options').")
        origin_platform, origin_user_id, origin_chat_id = "web", "", 0
        agent = kwargs.get("_agent")
        try:
            from captain_claw.origin import get_session_origin

            found = get_session_origin(getattr(agent, "session", None)) if agent else None
        except Exception:  # noqa: BLE001
            found = None
        # Other channels need nothing here: the agent's session remembers where it was asked.
        if found and found["kind"] == "telegram":
            origin_platform, origin_user_id = "telegram", found["address"]
            origin_chat_id = int(found["address"]) if str(found["address"]).isdigit() else 0
        payload = {
            "harness": harness, "brief": brief, "source_host": "localhost",
            "origin_platform": origin_platform, "origin_user_id": origin_user_id, "origin_chat_id": origin_chat_id,
            **{k: str(kwargs.get(k) or "") for k in ("title", "kind", "deliver", "project", "folder", "repo_url",
                                                     "repo_path", "model", "effort") if kwargs.get(k)},
        }
        status, data = await client.call("POST", "dispatch", payload)
        if (refused := self._refused(status, data)) is not None:
            return refused
        where = "waiting for a free slot" if data.get("status") == "queued" else "working"
        return ToolResult(success=True, content=(
            f"Task {data.get('id')} is {where}: {data.get('harness_label')} in {data.get('repo_label')}, on branch "
            f"{data.get('branch')}. It has only started: nothing is done yet, so tell the user it's under way and "
            f"don't report a result. You'll be called back when it finishes or needs a decision."
        ))

    async def _do_list(self, client: Any, **_: Any) -> ToolResult:
        status, data = await client.call("GET", "tasks")
        if (refused := self._refused(status, data)) is not None:
            return refused
        tasks = data.get("tasks") or []
        return ToolResult(success=True, content="\n".join(map(_task_line, tasks)) or "No crew tasks yet.")

    async def _with_task(self, client: Any, path: str, kwargs: dict, **extra: Any) -> tuple[ToolResult | None, Any]:
        task_id = str(kwargs.get("task_id") or "").strip()
        if not task_id:
            return ToolResult(success=False, error="Provide `task_id` (see action='list')."), None
        status, data = await client.call("POST", path, {"task_id": task_id, **extra})
        return self._refused(status, data), data

    async def _do_status(self, client: Any, **kwargs: Any) -> ToolResult:
        refused, data = await self._with_task(client, "task", kwargs)
        if refused is not None:
            return refused
        lines = [_task_line(data)]
        evidence = data.get("evidence") or {}
        if evidence.get("log"):
            lines.append("Commits:\n" + "\n".join(f"  {c}" for c in evidence["log"]))
        if evidence.get("dirty"):
            lines.append(f"Uncommitted changes: {evidence['dirty']}")
        if evidence.get("off_branch"):
            lines.append(f"Warning: the worker left its branch; it's on {evidence['off_branch']}")
        if data.get("report"):
            lines.append(f"Report:\n{data['report'][:12000]}")
        elif data.get("final"):
            lines.append(f"Its last message:\n{data['final'][:6000]}")
        if data.get("status") == "running":
            lines.append("Latest activity:")
            lines.extend(f"  {e.get('tool') or e.get('kind')}: {str(e.get('text'))[:300]}"
                         for e in data.get("recent", []))
        return ToolResult(success=True, content="\n".join(lines))

    async def _do_steer(self, client: Any, **kwargs: Any) -> ToolResult:
        text = str(kwargs.get("text") or "").strip()
        if not text:
            return ToolResult(success=False, error="Provide `text`: what to tell the worker.")
        refused, data = await self._with_task(client, "steer", kwargs, text=text)
        if refused is not None:
            return refused
        when = "next, once its current turn ends" if data.get("pending") else "now"
        return ToolResult(success=True, content=f"Told task {data.get('id')}; it acts on it {when}.")

    async def _do_stop(self, client: Any, **kwargs: Any) -> ToolResult:
        refused, data = await self._with_task(client, "stop", kwargs)
        if refused is not None:
            return refused
        return ToolResult(success=True, content=f"Task {data.get('id')} is {data.get('status')}. Its work is kept.")

    async def _do_pr(self, client: Any, **kwargs: Any) -> ToolResult:
        refused, data = await self._with_task(client, "pr", kwargs)
        if refused is not None:
            return refused
        return ToolResult(success=True, content=str(data.get("note") or "Pull request opened."))

    async def _do_request_merge(self, client: Any, **kwargs: Any) -> ToolResult:
        refused, data = await self._with_task(client, "request_merge", kwargs)
        if refused is not None:
            return refused
        return ToolResult(success=True, content=(
            "The merge is waiting for the user's Approve in Flight Deck (the bell). It isn't merged until they "
            "approve it; say so."
        ))
