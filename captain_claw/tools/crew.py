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
import re
import time
from datetime import UTC, datetime
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

_WORKER_MARKERS = ("CLAW_CODE_AGENT", "CLAW_BASNA_WORKER", "CLAW_VATRA_WORKER")
_ASKS_TO_DELIVER = re.compile(r"\b(git\s+push|gh\s+(pr|auth|repo)|GH_TOKEN|GITHUB_TOKEN|pull\s+request|push\s+(the|this|it|branch))\b",
                              re.IGNORECASE)
_OPENS_A_PR = re.compile(r"\b(open|create|raise|submit|make)\s+(a|an|the|its)?\s*(new\s+)?(pull\s+request|PR)\b|\bgh\s+pr\s+create\b",
                         re.IGNORECASE)
_NOT_A_WORKERS_JOB = (" A worker doesn't push or open pull requests, and won't see a GitHub login on this computer: use "
                      "crew action=pr once it has committed.")


_WINDOWS = {"five_hour": "the 5-hour limit", "seven_day": "the weekly limit", "seven_day_opus": "the weekly Opus limit",
            "seven_day_sonnet": "the weekly Sonnet limit"}


def _usage_lines(row: dict | None) -> list[str]:
    """What a CLI last said about itself. It says so only while working, so its age is part of the fact."""
    if not row or not (row.get("model") or row.get("limits") or row.get("tasks")):
        return ["    Model and limits: not reported yet. Only some CLIs report them, and only while working."]
    lines = []
    if row.get("model"):
        lines.append(f"    Last ran on {row['model']}, unless a task names another.")
    for name, window in (row.get("limits") or {}).items():
        used = window.get("used")
        said = f"{round(used * 100)}% used" if used is not None else f"status {window.get('status') or 'unknown'}"
        resets = window.get("resets_at")
        when = f", resets {datetime.fromtimestamp(resets, UTC):%a %d %b %H:%M} UTC" if resets else ""
        lines.append(f"    {(_WINDOWS.get(name) or name.replace('_', ' ')).capitalize()}: {said}{when}.")
    if row.get("limits") and row.get("seen_at"):
        lines.append(f"    That's as it reported {round((time.time() - row['seen_at']) / 60)} minutes ago, at its last run: "
                     "there's no way to ask between runs, so say how old it is.")
    if row.get("tasks"):
        lines.append(f"    Its tasks in the last week: {row['tasks']}, {row.get('tokens', 0):,} tokens.")
    return lines


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
        "Reading results: a project's folder is in your VFS (options shows its vfs: path) and so is what a finished "
        "task made (the callback, status and files say where), so read them yourself with read or glob. Never start "
        "a task to look at or fetch files, and never ask a worker to print files into its message: messages get cut. "
        "A link to one file is public_link; a page with its css and scripts needs hosting. "
        "Actions: options (the CLIs you may use, their logins, the model each last ran on and its usage limits, like "
        "Claude's weekly limit, and the projects: use it when asked about any of those); dispatch; list (project narrows "
        "it); status; files (where a task's files are; whole=true copies the whole project, for a big one); "
        "steer (the next instruction, or the user's answer to a worker's question); stop (keeps the work); "
        "pr (Captain pushes the branch and opens the pull request, with the user's GitHub token from Crew → Logins: "
        "that's the only way a task's work gets pushed, so never ask a worker to push or run gh, and never ask the user "
        "to log in on this computer or to give you a token: a token goes in Crew → Logins, not in a chat); request_merge "
        "(asks the user to Approve the merge; merging is "
        "their decision, so only ask when they want it merged); update_deck (for an admin's agent: asks them to Approve "
        "updating this Flight Deck to what's new on its branch. It stops every agent, you included, pulls and restarts; "
        "you're called back when it's over. Only ask when they want the new version running). You can't merge or discard "
        "work yourself, and that includes git in a shell: a worker's branch is the user's to land."
    )
    timeout_seconds = 200.0

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["options", "dispatch", "list", "status", "files", "steer", "stop", "pr", "request_merge", "update_deck"],
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
                    "conversation: include the goal, what done means, and any decisions already made. Don't ask it to "
                    "push or open a pull request: set deliver=pr for that."
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
            "project": {"type": "string", "description": "dispatch: a Code project to work in (created if new). list: only this project's tasks."},
            "folder": {"type": "string", "description": "dispatch: a folder in that project. Omit to use the project's name."},
            "repo_url": {"type": "string", "description": "dispatch: a repository address to clone and work in, instead of a project."},
            "repo_path": {"type": "string", "description": "dispatch: a folder on this computer, instead of a project. Admins only."},
            "model": {"type": "string", "description": "dispatch: a model for the CLI, only if the user asked for one; otherwise their default is used. For OpenCode, exactly a name from 'options': never make one up."},
            "effort": {"type": "string", "description": "dispatch: reasoning effort for the CLI, only if the user asked for one."},
            "task_id": {"type": "string", "description": "status, files, steer, stop, pr, request_merge: the task."},
            "whole": {"type": "boolean", "description": "files: copy the whole project, not only what the task changed."},
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
        used = {u["id"]: u for u in (data.get("usage") or {}).get("harnesses", [])}
        for h in (data.get("me") or {}).get("harnesses", []):
            state = "not installed" if not h.get("installed") else (
                f"ready, on {h['login']['how']}" if h["login"].get("ok") else "no login yet (Crew → Logins)")
            lines.append(f"  {h['id']} ({h['label']}): {state}")
            if h.get("installed"):
                lines.extend(_usage_lines(used.get(h["id"])))
                if h.get("default_model"):
                    lines.append(f"    Tasks use {h['default_model']} unless one names another (the user's choice).")
                elif (h.get("login") or {}).get("free"):
                    lines.append("    No model is chosen, so it needs one named. Its free models differ in what happens to "
                                 "the code, so that's the user's choice: ask them, don't pick for them.")
                if h.get("models"):
                    lines.append(f"    It can use only these, by exactly these names: {', '.join(h['models'][:30])}")
        repos = data.get("repos") or {}
        lines.append("Projects (each folder is in your VFS at the path shown: read it yourself, no task needed):")
        for p in repos.get("projects", []):
            paths = p.get("paths") or {}
            lines.append(f"  {p['project']}: " + (", ".join(f"{f} ({paths.get(f) or 'not in your VFS'})"
                                                            for f in p["folders"]) or "(no folders)"))
        if repos.get("recent"):
            lines.append("Used lately:")
            lines.extend(f"  {r['repo']}" for r in repos["recent"])
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    def _origin_of(agent: Any) -> tuple[str, str, int]:
        """Where a callback goes. Only Telegram needs saying: for other channels the
        agent's session remembers where it was asked."""
        try:
            from captain_claw.origin import get_session_origin

            found = get_session_origin(getattr(agent, "session", None)) if agent else None
        except Exception:  # noqa: BLE001
            found = None
        if found and found["kind"] == "telegram":
            return "telegram", found["address"], int(found["address"]) if str(found["address"]).isdigit() else 0
        return "web", "", 0

    async def _do_dispatch(self, client: Any, **kwargs: Any) -> ToolResult:
        harness = str(kwargs.get("harness") or "").strip()
        brief = str(kwargs.get("brief") or "").strip()
        if not harness or not brief:
            return ToolResult(success=False, error="Provide `harness` and `brief` (see action='options').")
        origin_platform, origin_user_id, origin_chat_id = self._origin_of(kwargs.get("_agent"))
        # Asked for in the words of the brief: it's what they want, and only Captain can do it.
        if not kwargs.get("deliver") and kwargs.get("kind") != "scout" and _OPENS_A_PR.search(brief):
            kwargs["deliver"] = "pr"
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
        behind = ""
        if data.get("unlanded"):
            behind = (f" It starts from {data.get('base_branch') or 'the base branch'}, so it won't see work that hasn't "
                      f"landed: {'; '.join(data['unlanded'][:5])}.")
        return ToolResult(success=True, content=(
            f"Task {data.get('id')} is {where}: {data.get('harness_label')} in {data.get('repo_label')}, on branch "
            f"{data.get('branch')}.{behind} It has only started: nothing is done yet, so tell the user it's under way, "
            f"end your turn and don't report a result. You'll be called back when it finishes or needs a decision."
            + (" Captain opens its pull request when it finishes." if kwargs.get("deliver") == "pr"
               else _NOT_A_WORKERS_JOB if _ASKS_TO_DELIVER.search(brief) else "")
        ))

    async def _do_list(self, client: Any, **kwargs: Any) -> ToolResult:
        status, data = await client.call("GET", "tasks")
        if (refused := self._refused(status, data)) is not None:
            return refused
        tasks = data.get("tasks") or []
        project = str(kwargs.get("project") or "").strip().lower()
        if project:
            tasks = [t for t in tasks if str(t.get("project_name") or "").lower() == project]
        return ToolResult(success=True, content="\n".join(map(_task_line, tasks)) or "No crew tasks here yet.")

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
        if data.get("model_used") or data.get("model"):
            lines.append(f"Model: {data.get('model_used') or data.get('model')}")
        if evidence.get("log"):
            lines.append("Commits:\n" + "\n".join(f"  {c}" for c in evidence["log"]))
        if evidence.get("dirty"):
            lines.append(f"Uncommitted changes: {evidence['dirty']}")
        if evidence.get("off_branch"):
            lines.append(f"Warning: the worker left its branch; it's on {evidence['off_branch']}")
        lines.extend(self._where_lines(data.get("results") or {}))
        lines.extend(data.get("next_steps") or [])
        if data.get("report"):
            lines.append(f"Report:\n{data['report'][:12000]}")
        elif data.get("final"):
            lines.append(f"Its last message:\n{data['final'][:6000]}")
        if data.get("status") == "running":
            lines.append("Latest activity:")
            lines.extend(f"  {e.get('tool') or e.get('kind')}: {str(e.get('text'))[:300]}"
                         for e in data.get("recent", []))
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    def _where_lines(results: dict) -> list[str]:
        lines = []
        if results.get("files_path"):
            what = "the whole project" if results.get("whole") else "only what it changed"
            lines.append(f"Files: {results.get('count')} at {results['files_path']} ({what}). Read them there.")
        if results.get("report"):
            lines.append(f"Its whole message: {results['report']}")
        if results.get("publish"):
            lines.append(f"For a link to this page: hosting action=publish kind=static project={results['publish']['project']} "
                         f"subdir={results['publish']['subdir']} name=<a short name>")
        return lines

    async def _do_files(self, client: Any, **kwargs: Any) -> ToolResult:
        refused, data = await self._with_task(client, "files", kwargs, whole=bool(kwargs.get("whole")))
        if refused is not None:
            return refused
        lines = self._where_lines(data)
        if data.get("landed") and data.get("project_path"):
            lines.insert(0, f"Merged. The project itself: {data['project_path']}")
        names = data.get("files") or []
        if names:
            lines.append("In it:\n" + "\n".join(f"  {n}" for n in names[:100]))
            if len(names) > 100:
                lines.append(f"  and {len(names) - 100} more")
        if data.get("skipped"):
            lines.append(f"{data['skipped']} file(s) were too big to copy.")
        if data.get("uncommitted"):
            lines.append(f"{data['uncommitted']} uncommitted change(s) aren't in this copy: only commits are.")
        return ToolResult(success=True, content="\n".join(lines) or "This task has no files.")

    async def _do_steer(self, client: Any, **kwargs: Any) -> ToolResult:
        text = str(kwargs.get("text") or "").strip()
        if not text:
            return ToolResult(success=False, error="Provide `text`: what to tell the worker.")
        refused, data = await self._with_task(client, "steer", kwargs, text=text)
        if refused is not None:
            return refused
        when = "next, once its current turn ends" if data.get("pending") else "now"
        return ToolResult(success=True, content=f"Told task {data.get('id')}; it acts on it {when}."
                          + (_NOT_A_WORKERS_JOB if _ASKS_TO_DELIVER.search(text) else ""))

    async def _do_stop(self, client: Any, **kwargs: Any) -> ToolResult:
        refused, data = await self._with_task(client, "stop", kwargs)
        if refused is not None:
            return refused
        return ToolResult(success=True, content=f"Task {data.get('id')} is {data.get('status')}. Its work is kept.")

    async def _do_pr(self, client: Any, **kwargs: Any) -> ToolResult:
        refused, data = await self._with_task(client, "pr", kwargs)
        if refused is not None:
            return refused
        return ToolResult(success=True, content=(
            f"{data.get('note') or 'Pull request opened.'} Give the user that link. Merging it is their call: "
            "crew action=request_merge only if they ask."))

    async def _do_request_merge(self, client: Any, **kwargs: Any) -> ToolResult:
        refused, data = await self._with_task(client, "request_merge", kwargs)
        if refused is not None:
            return refused
        return ToolResult(success=True, content=(
            "The merge is waiting for the user's Approve in Flight Deck (the bell). It isn't merged until they "
            "approve it; say so. You'll be called back when it lands."
        ))

    async def _do_update_deck(self, client: Any, **kwargs: Any) -> ToolResult:
        origin_platform, origin_user_id, origin_chat_id = self._origin_of(kwargs.get("_agent"))
        status, data = await client.call("POST", "update_deck", {
            "origin_platform": origin_platform, "origin_user_id": origin_user_id, "origin_chat_id": origin_chat_id,
        })
        if (refused := self._refused(status, data)) is not None:
            return refused
        if data.get("status") == "up_to_date":
            return ToolResult(success=True, content=f"This deck is already on the newest version: {data.get('head_title')}.")
        return ToolResult(success=True, content=(
            f"The update ({data.get('commits')} new commit(s)) is waiting for the user's Approve in Flight Deck (the bell). "
            "Nothing changes until they approve it; say so. When they do, every agent stops, you included, and you'll be "
            "called back when the deck is up again."
        ))
