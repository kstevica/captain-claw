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

import httpx

from captain_claw import crew_client
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


def _env_lines(entry: dict) -> list[str]:
    """A project's environment, in what it means for the work."""
    lines = [f"{entry.get('label')} ({', '.join(entry.get('kinds') or []) or 'kind unknown'}): "
             + ("prepared" if entry.get("ready") else "not prepared")]
    lines.append(f"  Its folder, where the user can set it up by hand: {entry.get('repo')}")
    lines.extend(f"  Python environment, shared by its tasks: {p['path']} (Python {p.get('version') or '?'}, {p.get('packages')} packages)"
                 for p in entry.get("python") or [])
    if entry.get("folders"):
        lines.append(f"  Each task gets its own copy of: {', '.join(f['path'] for f in entry['folders'])}")
    if entry.get("copy"):
        lines.append(f"  And, by its settings: {', '.join(entry['copy'])}")
    if entry.get("missing"):
        lines.append(f"  Not there yet: {', '.join(entry['missing'])}")
    if entry.get("stale"):
        lines.append(f"  Changed since it was prepared, so it may be out of date: {', '.join(entry['stale'])}")
    if entry.get("test"):
        lines.append(f"  Its tests: {entry['test']}")
    if entry.get("run"):
        lines.append(f"  To start it: {entry['run']}")
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
        "Giving a worker files: put them in attachments on dispatch, prepare_env or steer, like a screenshot the user sent "
        "you, a log or a design. It can't see your files any other way, so never paste them into a brief or give it your paths. "
        "Giving a worker folders of the user's VFS: vfs_read and vfs_write on dispatch or prepare_env, or folders for a task under "
        "way. With vfs_read it gets its own copy; with vfs_write it works in the real folder, at once and with no undo, so give "
        "that only when the task has to change their files there. Google Drive and OneDrive folders can't be given. "
        "A task exists once a crew result here gives its id, and not before: never say one started, or that a repository "
        "is cloned, from anything else. If a call fails, tell the user what it said. "
        "Actions: options (the CLIs you may use, their logins, the model each last ran on and its usage limits, like "
        "Claude's weekly limit, and the projects: use it when asked about any of those); dispatch; clone (only when they "
        "want a repository here and no work yet: it runs no worker and spends no tokens, and you're called back when it's "
        "ready. dispatch with repo_url clones by itself, so don't clone first for that); env (what's installed for a project "
        "so workers can build, run and test it, its test command, and the folder where the user can set it up by hand); "
        "prepare_env (a worker on `harness` sets the project up, whatever its language, in that folder, and every later task "
        "gets it: use it when they ask for that, or before work that has to run when env shows nothing prepared); "
        "list (project narrows it); status; files (where a task's files are; whole=true copies the whole project, for a big one); "
        "steer (the next instruction, or the user's answer to a worker's question; it waits for the worker's turn to end, "
        "unless now=true); folders (sets which VFS folders a task has, replacing what it had, from its next turn); stop (keeps the work); "
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
                "enum": ["options", "dispatch", "clone", "env", "prepare_env", "list", "status", "files", "steer", "folders", "stop", "pr", "request_merge", "update_deck"],
            },
            "harness": {
                "type": "string",
                "enum": ["claude", "codex", "opencode", "pi"],
                "description": "dispatch, prepare_env: which CLI does the work. Use the one the user named; otherwise see 'options'.",
            },
            "brief": {
                "type": "string",
                "description": (
                    "dispatch: the task, complete and self-contained. The worker sees only this, not your "
                    "conversation: include the goal, what done means, and any decisions already made. Don't ask it to "
                    "push or open a pull request: set deliver=pr for that. prepare_env: anything particular the user said "
                    "about setting it up, optional."
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
                "description": "dispatch, ship: leave the work on a branch (default), or open a pull request when it's done, which needs a GitHub repository and the user's token in Crew → Logins.",
            },
            "project": {"type": "string", "description": "dispatch, prepare_env: a Code project to work in (created if new). Name exactly one of project, repo_url and repo_path. env: which project. list: only this project's tasks."},
            "folder": {"type": "string", "description": "dispatch, prepare_env, env: a folder in that project, only with project. Omit to use the project's name."},
            "repo_url": {"type": "string", "description": "dispatch, clone, prepare_env, env: a repository's https:// address, with no login in it. Crew clones it itself, in the background, and the worker starts inside it, so never ask a worker to clone. It makes its own project, named after the repository: send no project with it."},
            "repo_path": {"type": "string", "description": "dispatch, prepare_env, env: a git folder on this computer. Admins only."},
            "model": {"type": "string", "description": "dispatch: a model for the CLI, only if the user asked for one; otherwise their default is used. For OpenCode, exactly a name from 'options': never make one up."},
            "effort": {"type": "string", "description": "dispatch: reasoning effort for the CLI, only if the user asked for one."},
            "task_id": {"type": "string", "description": "status, files, steer, stop, pr, request_merge: the task."},
            "whole": {"type": "boolean", "description": "files: copy the whole project, not only what the task changed."},
            "text": {"type": "string", "description": "steer: what to tell the worker, and what any attachments are for."},
            "now": {"type": "boolean", "description": "steer: stop what the worker is doing so it hears this at once, when the user wants it to stop or change course now. What it's running stops; its files stay."},
            "vfs_read": {
                "type": "array",
                "items": {"type": "string"},
                "description": "dispatch, prepare_env, folders: folders of the user's VFS it can read, as vfs:<folder>/<path>. It gets its own copy, made the same as the real folder before each turn.",
            },
            "vfs_write": {
                "type": "array",
                "items": {"type": "string"},
                "description": "dispatch, prepare_env, folders: folders of the user's VFS it can change, as vfs:<folder>/<path>. It works in the real folder: its changes happen at once, outside git. Only when the task has to change their files there.",
            },
            "attachments": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "dispatch, prepare_env, steer: files for the worker, of any kind, as the file tools take them: a saved/ path, "
                    "the path a tool printed (its 'Saved to:' line), or vfs:<project>/<path>. They reach its folder and it's told "
                    f"where, so say in the brief or text what they're for. Up to {crew_client.MAX_FILE_BYTES // 1_000_000} MB each, and "
                    f"{crew_client.MAX_TASK_FILES} files or {crew_client.MAX_TASK_BYTES // 1_000_000} MB for a task."
                ),
            },
        },
        "required": ["action"],
    }

    async def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
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
        except httpx.TimeoutException:
            log.warning("crew tool timed out", action=action)
            return ToolResult(success=False, error=(
                "Flight Deck didn't answer in time, so this may still be going through: it hasn't failed as far as anyone "
                "knows. Check crew action=list before you try again or tell the user what happened."))
        except Exception as exc:  # noqa: BLE001
            why = str(exc) or type(exc).__name__  # a timeout's own text is empty
            log.warning("crew tool error", action=action, error=why)
            return ToolResult(success=False, error=f"Crew request failed: {why}")

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
            lines.append("Used lately (repo_url, or repo_path for a folder here, goes back there):")
            lines.extend(f"  {r.get('repo_url') or (r['repo'] if r.get('repo_kind') == 'path' else r.get('repo_label'))}"
                         for r in repos["recent"])
        if data.get("envs"):
            lines.append("Environments (crew action=env for what's in one):")
            lines.extend(f"  {e.get('label')}: {'prepared' if e.get('ready') else 'not prepared'}" for e in data["envs"])
        return ToolResult(success=True, content="\n".join(lines))

    async def _attach(self, client: Any, kwargs: dict) -> tuple[list[dict], ToolResult | None]:
        """`attachments` sent ahead to Flight Deck: what it now holds, or why not. Every file is checked
        before the first goes, as for an email: one that can't go refuses the call."""
        from captain_claw import mail_attachments

        if not kwargs.get("attachments"):
            return [], None

        def no(why: str) -> tuple[list[dict], ToolResult]:
            return [], ToolResult(success=False, error=why)

        try:
            values = list(dict.fromkeys(mail_attachments.parse_list(kwargs.get("attachments"))))
        except mail_attachments.AttachmentError as exc:
            return no(str(exc))
        if len(values) > client.MAX_TASK_FILES:
            return no(f"That's {len(values)} files, and a task can have {client.MAX_TASK_FILES}.")
        files = []
        for value in values:
            path = mail_attachments.resolve_path(value, kwargs) if "\x00" not in value else None
            if path is None:
                return no(f"File not found: {value}. Name a file, not a folder: the path a tool printed (its 'Saved to:' line), "
                          "a saved/ path, or vfs:<project>/<path>.")
            if why := mail_attachments.credential_reason(path, kwargs):
                return no(f"'{path.name}' can't go to a worker: {why}.")
            size = path.stat().st_size
            if not size:
                return no(f"'{path.name}' is empty, so there's nothing to send.")
            if size > client.MAX_FILE_BYTES:
                return no(f"'{path.name}' is {mail_attachments.human_size(size)}, and a file can be {client.MAX_FILE_BYTES // 1_000_000} MB.")
            files.append((path, size))
        if sum(size for _, size in files) > client.MAX_TASK_BYTES:
            return no(f"Together they're too big: a task's files can come to {client.MAX_TASK_BYTES // 1_000_000} MB.")
        held: list[dict] = []
        for path, _ in files:
            status, data = await client.upload(path, path.name, mail_attachments.guess_mime(path.name))
            if status >= 300:
                await self._let_go(client, held)
                return [], self._refused(status, data)
            held.append(data)
        return held, None

    @staticmethod
    def _folders_of(kwargs: dict) -> tuple[list[dict], str]:
        """`vfs_read` and `vfs_write` as Flight Deck names folders, or why not. What the owner
        may give is its decision; a process walled to some of the VFS stays walled here."""
        from captain_claw import vfs

        scope = vfs.scope_projects()
        out = []
        for key, write in (("vfs_read", False), ("vfs_write", True)):
            raw = kwargs.get(key) or []
            for value in [raw] if isinstance(raw, str) else raw:
                project, _, path = str(value).strip().removeprefix("vfs:").strip("/").partition("/")
                if not project:
                    return [], f"{key} takes folders, as vfs:<folder>/<path>."
                if scope is not None and project not in scope:
                    return [], f"This agent can't reach {project}."
                out.append({"project": project, "path": path, "write": write})
        return out, ""

    @staticmethod
    def _folder_line(folders: list[dict]) -> str:
        return ", ".join(f"vfs:{f['project']}/{f['path'] + '/' if f.get('path') else ''} ({'read and write' if f.get('write') else 'read'})"
                         for f in folders)

    @staticmethod
    async def _let_go(client: Any, held: list[dict]) -> None:
        """What was sent ahead for a call that was refused. Flight Deck clears it in a day anyway."""
        for item in held:
            try:
                await client.call("DELETE", f"uploads/{item['id']}")
            except Exception as exc:  # noqa: BLE001
                log.debug("crew: couldn't let go of an upload", error=str(exc))

    @staticmethod
    def _sent_line(held: list[dict]) -> str:
        """By the names the worker sees: never paths, which a chat bridge would send as files."""
        from captain_claw.mail_attachments import human_size

        names = [f"{item['name']} ({human_size(item['size'])})" for item in held]
        return f" With it: {', '.join(names)}." if names else ""

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
        folders, why = self._folders_of(kwargs)
        if why:
            return ToolResult(success=False, error=why)
        held, refused = await self._attach(client, kwargs)
        if refused is not None:
            return refused
        payload = {
            "harness": harness, "brief": brief, "source_host": "localhost",
            "origin_platform": origin_platform, "origin_user_id": origin_user_id, "origin_chat_id": origin_chat_id,
            **{k: str(kwargs.get(k) or "") for k in ("title", "kind", "deliver", "project", "folder", "repo_url",
                                                     "repo_path", "model", "effort") if kwargs.get(k)},
            **({"attachments": [i["id"] for i in held]} if held else {}),
            **({"vfs_folders": folders} if folders else {}),
        }
        status, data = await client.call("POST", "dispatch", payload)
        if (refused := self._refused(status, data)) is not None:
            await self._let_go(client, held)
            return refused
        where = {"preparing": "getting its repository first, which can take some minutes for a big one, and then it starts",
                 "queued": "waiting for a free slot"}.get(data.get("status"), "working")
        behind = ""
        if data.get("unlanded"):
            behind = (f" It starts from {data.get('base_branch') or 'the base branch'}, so it won't see work that hasn't "
                      f"landed: {'; '.join(data['unlanded'][:5])}.")
        return ToolResult(success=True, content=(
            f"Task {data.get('id')} is {where}: {data.get('harness_label')} in {data.get('repo_label')}, on branch "
            f"{data.get('branch')}.{self._sent_line(held)}{f' Its folders: {self._folder_line(folders)}.' if folders else ''}{behind} It has only started: nothing is done yet, so tell the user it's under way, "
            f"end your turn and don't report a result. You'll be called back when it finishes or needs a decision."
            + (" Captain opens its pull request when it finishes." if kwargs.get("deliver") == "pr"
               else _NOT_A_WORKERS_JOB if _ASKS_TO_DELIVER.search(brief) else "")
        ))

    async def _do_clone(self, client: Any, **kwargs: Any) -> ToolResult:
        url = str(kwargs.get("repo_url") or "").strip()
        if not url:
            return ToolResult(success=False, error="Provide `repo_url`: the repository's https:// address.")
        origin_platform, origin_user_id, origin_chat_id = self._origin_of(kwargs.get("_agent"))
        status, data = await client.call("POST", "clone", {
            "repo_url": url, "source_host": "localhost", "origin_platform": origin_platform,
            "origin_user_id": origin_user_id, "origin_chat_id": origin_chat_id,
        })
        if (refused := self._refused(status, data)) is not None:
            return refused
        return ToolResult(success=True, content=(
            f"Task {data.get('id')} is getting {data.get('repo_label')}, with no worker: it shows under that name in Crew, "
            "and a big one can take some minutes. It has only started, so tell the user it's under way and end your turn. "
            f"You'll be called back when it's ready. To work in it then: crew action=dispatch repo_url={url}."
        ))

    async def _do_env(self, client: Any, **kwargs: Any) -> ToolResult:
        status, data = await client.call("POST", "env", {k: str(kwargs.get(k) or "") for k in ("project", "folder", "repo_url", "repo_path")})
        if (refused := self._refused(status, data)) is not None:
            return refused
        envs = data.get("envs") or []
        if not envs:
            return ToolResult(success=True, content="No project here yet by that name. crew action=clone or dispatch makes one.")
        return ToolResult(success=True, content="\n".join(line for entry in envs for line in _env_lines(entry)))

    async def _do_prepare_env(self, client: Any, **kwargs: Any) -> ToolResult:
        harness = str(kwargs.get("harness") or "").strip()
        if not harness:
            return ToolResult(success=False, error="Provide `harness`: which CLI does it (see action='options').")
        origin_platform, origin_user_id, origin_chat_id = self._origin_of(kwargs.get("_agent"))
        folders, why = self._folders_of(kwargs)
        if why:
            return ToolResult(success=False, error=why)
        held, refused = await self._attach(client, kwargs)
        if refused is not None:
            return refused
        status, data = await client.call("POST", "dispatch", {
            "harness": harness, "kind": "setup", "source_host": "localhost", "origin_platform": origin_platform,
            "origin_user_id": origin_user_id, "origin_chat_id": origin_chat_id,
            **{k: str(kwargs.get(k) or "") for k in ("brief", "title", "project", "folder", "repo_url", "repo_path", "model", "effort") if kwargs.get(k)},
            **({"attachments": [i["id"] for i in held]} if held else {}),
            **({"vfs_folders": folders} if folders else {}),
        })
        if (refused := self._refused(status, data)) is not None:
            await self._let_go(client, held)
            return refused
        return ToolResult(success=True, content=(
            f"Task {data.get('id')}: {data.get('harness_label')} is setting up {data.get('repo_label')}, in the project's main "
            f"folder.{self._sent_line(held)} It has only started: nothing is installed yet, so tell the user it's under way and end your turn. You'll be "
            "called back when it's done. New tasks on this project wait for it."
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
        if data.get("vfs"):
            lines.append(f"Its folders: {self._folder_line(data['vfs'])}")
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
        if not text and not kwargs.get("attachments"):
            return ToolResult(success=False, error="Provide `text`: what to tell the worker.")
        if not str(kwargs.get("task_id") or "").strip():
            return ToolResult(success=False, error="Provide `task_id` (see action='list').")
        held, refused = await self._attach(client, kwargs)
        if refused is not None:
            return refused
        refused, data = await self._with_task(client, "steer", kwargs, text=text, now=bool(kwargs.get("now")),
                                              **({"attachments": [i["id"] for i in held]} if held else {}))
        if refused is not None:
            await self._let_go(client, held)
            return refused
        when = "next, once its current turn ends" if data.get("pending") else "now"
        return ToolResult(success=True, content=f"Told task {data.get('id')}; it acts on it {when}.{self._sent_line(held)}"
                          + (_NOT_A_WORKERS_JOB if _ASKS_TO_DELIVER.search(text) else ""))

    async def _do_folders(self, client: Any, **kwargs: Any) -> ToolResult:
        folders, why = self._folders_of(kwargs)
        if why:
            return ToolResult(success=False, error=why)
        refused, data = await self._with_task(client, "folders", kwargs, vfs_folders=folders)
        if refused is not None:
            return refused
        has = self._folder_line(data.get("vfs") or []) or "none of the user's folders"
        return ToolResult(success=True, content=(
            f"From its next turn, task {data.get('id')} has {has}. A turn already running doesn't get them: "
            "use steer with now=true if it needs them at once, or steer if it's waiting."))

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
