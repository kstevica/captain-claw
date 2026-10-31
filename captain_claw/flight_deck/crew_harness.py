"""Crew — the coding CLIs Flight Deck can run: Claude Code, Codex, OpenCode, Pi.

Each adapter says how to start one turn of the CLI headless, and how to read
the JSON lines it prints into one shape (:class:`TurnState` and the events
below). Nothing here reads a terminal screen: every CLI has a JSON mode and a
session it can resume, which is all the runner needs.

An event is ``{"kind", "text", "tool"?}``: ``text`` (what the worker said),
``tool`` (a call it made), ``result`` (what the call returned), ``error``.

Checked by hand on 2026-10-31 against Claude Code 2.1.316, codex-cli 0.149.1,
OpenCode 1.19.5 and Pi 1.4.0, each run for real in a git worktree. The parsers
ignore what they don't know, so a new event type costs a line in the log view,
not a crash.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

_EVENT_TEXT_MAX = 8000
_INPUT_FIELDS = ("command", "cmd", "file_path", "filePath", "path", "pattern", "query", "url",
                 "description", "prompt")
# Where these CLIs land when a deck's PATH doesn't name them.
_EXTRA_BIN_DIRS = ("~/.local/bin", "~/.npm-global/bin", "~/.bun/bin", "~/.opencode/bin",
                   "/opt/homebrew/bin", "/usr/local/bin")
# Claude's tools that would take a worker out of its worktree, or leave work
# running after the turn the runner is watching.
_CLAUDE_DENIED_TOOLS = "EnterWorktree,ExitWorktree,CronCreate,RemoteTrigger,ScheduleWakeup"
CLAUDE_PERMISSION_MODES = ("auto", "acceptEdits", "bypassPermissions")


@dataclass
class TurnState:
    """What one turn has said so far. Saved after every read, so a Flight Deck
    restart picks a turn up where it was."""

    session: str = ""
    final: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: str = ""
    model: str = ""  # the one it really ran on, when the CLI says
    limits: dict[str, dict] = field(default_factory=dict)  # the account's, when the CLI says

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str | None) -> "TurnState":
        try:
            data = json.loads(raw or "{}")
        except (TypeError, ValueError):
            data = {}
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data} if isinstance(data, dict) else {}
        return cls(**known)


@dataclass(frozen=True)
class RunSpec:
    """One turn of a worker."""

    binary: str
    prompt: str
    cwd: str
    git_common_dir: str = ""
    session: str = ""
    model: str = ""
    effort: str = ""
    task_id: str = ""


@dataclass(frozen=True)
class Harness:
    id: str
    label: str
    binary: str
    install: str
    argv: Callable[[RunSpec], list[str]]
    parse: Callable[[dict, TurnState], list[dict]]
    env: dict[str, str] = field(default_factory=dict)


def _clip(text: Any, limit: int = _EVENT_TEXT_MAX) -> str:
    value = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, default=str)
    return value if len(value) <= limit else value[:limit] + f"\n… ({len(value) - limit} more characters)"


def _ev(kind: str, text: Any, tool: str = "") -> dict:
    event = {"kind": kind, "text": _clip(text)}
    if tool:
        event["tool"] = tool
    return event


def _brief_input(payload: Any) -> str:
    """The part of a tool call's input a person reads first."""
    if not isinstance(payload, dict):
        return _clip(payload, 400) if payload else ""
    for name in _INPUT_FIELDS:
        value = payload.get(name)
        if isinstance(value, str) and value.strip():
            return _clip(value, 400)
    return _clip(payload, 400) if payload else ""


def _safe_prompt(prompt: str) -> str:
    # A prompt starting with "-" would be read as a flag.
    return f" {prompt}" if prompt.startswith("-") else prompt


def _toml_str(value: str) -> str:
    return json.dumps(value)


# ── Claude Code ─────────────────────────────────────────────────────────


def claude_permission_mode() -> str:
    """The deck's choice (``FD_CREW_CLAUDE_PERMISSION``). ``auto`` keeps Claude's
    own safety check on every call a worker makes without stopping to ask."""
    chosen = os.environ.get("FD_CREW_CLAUDE_PERMISSION", "").strip()
    return chosen if chosen in CLAUDE_PERMISSION_MODES else "auto"


def _claude_argv(spec: RunSpec) -> list[str]:
    argv = [spec.binary, "-p", _safe_prompt(spec.prompt), "--output-format", "stream-json", "--verbose",
            "--permission-mode", claude_permission_mode(),
            # Their project's settings, not the deck user's own hooks and plugins.
            "--setting-sources", "project,local",
            "--disallowedTools", _CLAUDE_DENIED_TOOLS]
    if spec.session:
        argv += ["--resume", spec.session]
    if spec.model:
        argv += ["--model", spec.model]
    if spec.effort:
        argv += ["--effort", spec.effort]
    return argv


def _claude_windows(info: dict) -> dict[str, dict]:
    """How much of each of the account's limits is used, as Claude Code prints it on
    every turn: ``{"seven_day": {"used": 0.44, "resets_at": 1793811600, "status": "allowed"}}``.
    It isn't a documented format, so anything else it might say is simply not read."""
    windows = info.get("unifiedWindows")
    if not isinstance(windows, dict) or not windows:
        windows = {info["rateLimitType"]: info} if info.get("rateLimitType") else {}
    out = {}
    for name, window in windows.items():
        if not isinstance(window, dict):
            continue
        used = window.get("utilization")
        resets = window.get("resetsAt")
        out[str(name)] = {"used": float(used) if isinstance(used, (int, float)) else None,
                          "resets_at": int(resets) if isinstance(resets, (int, float)) else 0,
                          "status": str(window.get("status") or "")}
    return out


def _claude_parse(obj: dict, state: TurnState) -> list[dict]:
    kind = obj.get("type")
    if obj.get("session_id"):
        state.session = str(obj["session_id"])
    out: list[dict] = []
    if kind == "system" and obj.get("subtype") == "init" and obj.get("model"):
        state.model = str(obj["model"])
    elif kind == "rate_limit_event" and isinstance(obj.get("rate_limit_info"), dict):
        state.limits.update(_claude_windows(obj["rate_limit_info"]))
    elif kind == "assistant":
        for block in (obj.get("message") or {}).get("content") or []:
            if block.get("type") == "text" and str(block.get("text") or "").strip():
                state.final = str(block["text"])
                out.append(_ev("text", state.final))
            elif block.get("type") == "tool_use":
                out.append(_ev("tool", _brief_input(block.get("input")), str(block.get("name") or "tool")))
    elif kind == "user":
        for block in (obj.get("message") or {}).get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                body = block.get("content")
                if isinstance(body, list):
                    body = "\n".join(str(b.get("text") or "") for b in body if isinstance(b, dict))
                out.append(_ev("error" if block.get("is_error") else "result", body or ""))
    elif kind == "result":
        usage = obj.get("usage") or {}
        state.tokens_in += (int(usage.get("input_tokens") or 0) + int(usage.get("cache_read_input_tokens") or 0)
                            + int(usage.get("cache_creation_input_tokens") or 0))
        state.tokens_out += int(usage.get("output_tokens") or 0)
        state.cost_usd += float(obj.get("total_cost_usd") or 0.0)
        if obj.get("is_error"):
            state.error = _clip(obj.get("result") or obj.get("subtype") or "the turn failed", 2000)
            out.append(_ev("error", state.error))
        elif str(obj.get("result") or "").strip():
            state.final = str(obj["result"])
    return out


# ── Codex ───────────────────────────────────────────────────────────────


def _codex_argv(spec: RunSpec) -> list[str]:
    argv = [spec.binary, "exec"]
    if spec.session:
        argv.append("resume")
    # All through -c: `exec resume` takes neither -s nor --add-dir. A worktree's
    # git data lives in its repo's .git, outside the workspace, so a worker can
    # commit only with that folder writable too.
    roots = [spec.git_common_dir] if spec.git_common_dir else []
    argv += ["--json", "--skip-git-repo-check",
             "-c", 'sandbox_mode="workspace-write"',
             "-c", 'approval_policy="never"',
             "-c", "sandbox_workspace_write.network_access=true",
             "-c", f"sandbox_workspace_write.writable_roots={json.dumps(roots)}",
             # The deck user's hooks would stop the turn at a trust question.
             "-c", "features.hooks=false"]
    if spec.model:
        argv += ["-m", spec.model]
    if spec.effort:
        argv += ["-c", f"model_reasoning_effort={_toml_str(spec.effort)}"]
    if spec.session:
        argv.append(spec.session)
    argv.append(_safe_prompt(spec.prompt))
    return argv


def _codex_parse(obj: dict, state: TurnState) -> list[dict]:
    kind = obj.get("type")
    if kind == "thread.started":
        state.session = str(obj.get("thread_id") or state.session)
        return []
    if kind == "turn.completed":
        state.error = ""  # what it said on the way (it retries on a dropped stream) is over
        usage = obj.get("usage") or {}
        state.tokens_in += int(usage.get("input_tokens") or 0)
        state.tokens_out += int(usage.get("output_tokens") or 0)
        return []
    if kind in ("turn.failed", "error"):
        err = obj.get("error") if kind == "turn.failed" else obj
        message = (err.get("message") if isinstance(err, dict) else err) or "the turn failed"
        state.error = _clip(message, 2000)
        return [_ev("error", state.error)]
    item = obj.get("item") or {}
    item_type = item.get("type")
    if kind == "item.started":
        if item_type == "command_execution":
            return [_ev("tool", item.get("command") or "", "shell")]
        if item_type == "mcp_tool_call":
            return [_ev("tool", _brief_input(item.get("arguments")), str(item.get("tool") or "mcp"))]
        return []
    if kind != "item.completed":
        return []
    if item_type == "agent_message":
        text = str(item.get("text") or "")
        if text.strip():
            state.final = text
            return [_ev("text", text)]
    elif item_type == "command_execution":
        failed = item.get("exit_code") not in (0, None)
        return [_ev("error" if failed else "result", item.get("aggregated_output") or "")]
    elif item_type == "file_change":
        paths = [f"{c.get('kind', 'edit')} {c.get('path', '')}" for c in item.get("changes") or []]
        return [_ev("tool", "\n".join(paths), "edit")]
    elif item_type == "web_search":
        return [_ev("tool", item.get("query") or "", "web_search")]
    elif item_type == "error":
        return [_ev("error", item.get("message") or "error")]
    return []


# ── OpenCode ────────────────────────────────────────────────────────────


def _opencode_argv(spec: RunSpec) -> list[str]:
    argv = [spec.binary, "run", "--format", "json", "--auto", "--dir", spec.cwd]
    if spec.session:
        argv += ["--session", spec.session]
    if spec.model:
        argv += ["--model", spec.model]
    if spec.effort:
        argv += ["--variant", spec.effort]
    argv.append(_safe_prompt(spec.prompt))
    return argv


def _opencode_parse(obj: dict, state: TurnState) -> list[dict]:
    if obj.get("sessionID"):
        state.session = str(obj["sessionID"])
    kind = obj.get("type")
    part = obj.get("part") or {}
    if kind == "text":
        text = str(part.get("text") or "")
        if text.strip():
            state.final = text
            return [_ev("text", text)]
    elif kind == "tool_use":
        run = part.get("state") or {}
        out = [_ev("tool", _brief_input(run.get("input")), str(part.get("tool") or "tool"))]
        if run.get("status") == "error":
            out.append(_ev("error", run.get("error") or "the call failed"))
        elif run.get("output"):
            out.append(_ev("result", run["output"]))
        return out
    elif kind == "step_finish":
        tokens = part.get("tokens") or {}
        cache = tokens.get("cache") or {}
        state.tokens_in += int(tokens.get("input") or 0) + int(cache.get("read") or 0) + int(cache.get("write") or 0)
        state.tokens_out += int(tokens.get("output") or 0)
        state.cost_usd += float(part.get("cost") or 0.0)
    elif kind == "error":
        err = obj.get("error") or {}
        data = err.get("data") if isinstance(err, dict) else None
        message = (data or {}).get("message") if isinstance(data, dict) else None
        state.error = _clip(message or (err.get("message") if isinstance(err, dict) else err) or "error", 2000)
        return [_ev("error", state.error)]
    return []


# ── Pi ──────────────────────────────────────────────────────────────────


def pi_session_id(task_id: str) -> str:
    """Pi lets the caller name a session, so a worker's id is known up front."""
    return f"crew-{task_id}"


def _pi_argv(spec: RunSpec) -> list[str]:
    argv = [spec.binary, "--mode", "json", "--session-id", spec.session or pi_session_id(spec.task_id)]
    if spec.model:
        argv += ["--model", spec.model]
    if spec.effort:
        argv += ["--thinking", spec.effort]
    argv.append(_safe_prompt(spec.prompt))
    return argv


def _pi_parse(obj: dict, state: TurnState) -> list[dict]:
    kind = obj.get("type")
    if kind == "session":
        state.session = str(obj.get("id") or state.session)
    elif kind == "message_end":
        message = obj.get("message") or {}
        if message.get("role") != "assistant":
            return []
        usage = message.get("usage") or {}
        if message.get("model"):
            state.model = str(message["model"])
        state.tokens_in += (int(usage.get("input") or 0) + int(usage.get("cacheRead") or 0)
                            + int(usage.get("cacheWrite") or 0))
        state.tokens_out += int(usage.get("output") or 0)
        state.cost_usd += float((usage.get("cost") or {}).get("total") or 0.0)
        out = []
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text" and str(block.get("text") or "").strip():
                state.final = str(block["text"])
                out.append(_ev("text", state.final))
        if message.get("stopReason") == "error" or message.get("errorMessage"):
            state.error = _clip(message.get("errorMessage") or "the turn failed", 2000)
            out.append(_ev("error", state.error))
        return out
    elif kind == "tool_execution_start":
        return [_ev("tool", _brief_input(obj.get("args")), str(obj.get("toolName") or "tool"))]
    elif kind == "tool_execution_end":
        body = (obj.get("result") or {}).get("content") if isinstance(obj.get("result"), dict) else obj.get("result")
        if isinstance(body, list):
            body = "\n".join(str(b.get("text") or "") for b in body if isinstance(b, dict))
        return [_ev("error" if obj.get("isError") else "result", body or "")]
    return []


HARNESSES: dict[str, Harness] = {
    "claude": Harness(
        "claude", "Claude Code", "claude", "curl -fsSL https://claude.ai/install.sh | bash",
        _claude_argv, _claude_parse,
        {"CLAUDE_CODE_SEND_FEEDBACK": "0", "DISABLE_AUTOUPDATER": "1"},
    ),
    "codex": Harness("codex", "Codex", "codex", "npm install -g @openai/codex", _codex_argv, _codex_parse),
    "opencode": Harness(
        "opencode", "OpenCode", "opencode", "npm install -g opencode-ai",
        _opencode_argv, _opencode_parse, {"OPENCODE_DISABLE_AUTOUPDATE": "1"},
    ),
    "pi": Harness("pi", "Pi", "pi", "npm install -g @earendil-works/pi-coding-agent", _pi_argv, _pi_parse),
}
HARNESS_IDS = tuple(HARNESSES)


def parse_line(harness: str, line: str | bytes, state: TurnState) -> list[dict]:
    """One line a CLI printed → its events. A line that isn't a JSON object is
    shown as it is; the CLI is talking outside its protocol."""
    text = line.decode("utf-8", "replace") if isinstance(line, bytes) else line
    text = text.strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
    except ValueError:
        return [_ev("note", text)]
    if not isinstance(obj, dict):
        return []
    try:
        events = HARNESSES[harness].parse(obj, state)
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        return [_ev("note", f"Couldn't read a line from {harness}: {exc}")]
    if events and events[-1]["kind"] == "text":
        state.error = ""  # it went on after the error: a retry that worked
    return events


def _search_path() -> str:
    extra = [str(Path(d).expanduser()) for d in _EXTRA_BIN_DIRS]
    return os.pathsep.join([os.environ.get("PATH", ""), *extra])


def find_binary(harness: str) -> str:
    """The CLI's path on this host ("" when it isn't installed). The deck's PATH
    first; then the login shell, which knows about nvm and friends."""
    name = HARNESSES[harness].binary
    override = os.environ.get(f"FD_CREW_{harness.upper()}_BIN", "").strip()
    if override:
        return override if os.access(override, os.X_OK) else ""
    found = shutil.which(name, path=_search_path())
    if found:
        return found
    shell = os.environ.get("SHELL") or "/bin/sh"
    try:
        probe = subprocess.run([shell, "-lc", f"command -v {name}"], capture_output=True, text=True,
                               timeout=10, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return ""
    candidate = (probe.stdout or "").strip().splitlines()
    return candidate[-1] if candidate and os.access(candidate[-1], os.X_OK) else ""


def version_of(binary: str) -> str:
    try:
        probe = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=15,
                               stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (probe.stdout or probe.stderr or "").strip().splitlines()[0][:80] if (probe.stdout or probe.stderr) else ""
