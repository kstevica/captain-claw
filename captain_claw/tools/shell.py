"""Shell tool for executing commands."""

import asyncio
import os
import re
import shlex
import tempfile
import time
from collections.abc import Iterator
from typing import Any

from captain_claw.config import get_config
from captain_claw.google_ids import (
    first_drive_url,
    google_drive_can_open,
    google_drive_redirect,
    is_google_drive_url,
)
from captain_claw.logging import get_logger
from captain_claw.tools.registry import (
    Tool,
    ToolResult,
    extract_shell_base_commands,
    is_blocked_shell_command,
)

log = get_logger(__name__)

# ── Google Drive download redirect ────────────────────────────────────
# curl/wget fetching a Drive/Docs file is refused and pointed at the
# google_drive tool (the owner's connection, exports, shared drives) while
# google_drive can open it: Google connected with a scope that reaches
# shared links. Otherwise it runs as before, so public links still work.
# Only a URL the curl/wget fetches counts — not a Drive link in its -d/-H
# payload or elsewhere in the command. (storage.googleapis.com is Cloud
# Storage, not Drive.)

_GDRIVE_HOST_PATTERNS = (
    "docs.google.com", "drive.google.com",
    "sheets.google.com", "slides.google.com",
    "drive.usercontent.google.com",
)

# Cheap pre-filter before the command is tokenized (across lines: a URL
# often sits after a ``\`` line continuation).
_GDRIVE_DOWNLOAD_RE = re.compile(
    r"""\b(?:curl|wget)\s.*(?:"""
    + "|".join(re.escape(h) for h in _GDRIVE_HOST_PATTERNS)
    + r""")""",
    re.IGNORECASE | re.DOTALL,
)

_FETCHERS = frozenset({"curl", "wget"})
# Options whose value is sent or labels the request — never the URL fetched.
_FETCH_VALUE_OPTS = {
    "curl": frozenset({
        "-d", "--data", "--data-raw", "--data-binary", "--data-ascii",
        "--data-urlencode", "--json", "-F", "--form", "--form-string",
        "-H", "--header", "-e", "--referer", "-x", "--proxy",
    }),
    "wget": frozenset({
        "--post-data", "--body-data", "--header", "--referer",
        "-e", "--execute", "-U", "--user-agent",
    }),
}

_GDRIVE_SHELL_BLOCK_INTRO = (
    "Do not use curl/wget to download Google Drive/Docs files — google_drive "
    "uses the Google connection and exports Docs/Sheets/Slides."
)

# ── Retired Google Workspace CLI ──────────────────────────────────────
# Google goes through the native google_* tools (per-owner token, the Flight
# Deck Gmail send gate). The gws binary would run as whatever credentials the
# host has, so a command that RUNS it is refused; gws as a mere argument or
# file name (``cat gws.txt``, ``grep gws f``, ``brew uninstall gws``) is not.

_GWS_RETIRED_MSG = (
    "The Google Workspace CLI (gws) is retired here — use google_drive / "
    "google_calendar / google_mail instead."
)

# Tokens after which the next word is a command again (a backtick toggles:
# opening starts a command, closing returns to the outer command's args).
_CMD_STARTERS = frozenset({";", ";;", "&", "&&", "|", "||", "|&", "("})
_CMD_KEYWORDS = frozenset({
    "if", "then", "else", "elif", "while", "until", "do", "!", "{",
})
# Wrappers that run their (first non-option) argument as the command, and
# the options of theirs that take a value.
_CMD_WRAPPERS = frozenset({
    "sudo", "doas", "env", "time", "nohup", "command", "builtin", "exec",
    "nice", "xargs", "stdbuf", "timeout",
})
_WRAPPER_VALUE_OPTS = frozenset({"-u", "-g", "-n", "-s", "-k", "-C", "-p", "-U", "-S"})
_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh"})
_SHELL_C_OPT_RE = re.compile(r"^-[A-Za-z]*c[A-Za-z]*$")
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_HEREDOC_RE = re.compile(r"(?<!<)<<-?(?!<)\s*(['\"]?)([A-Za-z_][\w.-]*)\1")
_SHELL_LINE_RE = re.compile(r"^\s*(?:\S*/)?(?:ba|z|da|k)?sh\b")
# Fallback when a line won't tokenize (unbalanced quotes).
_GWS_FALLBACK_RE = re.compile(r"(?:^|[;&|(`{]|\$\()\s*(?:\S*/)?gws(?:\s|$|[;&|)`])")


def _open_quote(line: str, quote: str | None) -> str | None:
    """The quote (' or ") still open at the end of *line*, given *quote*
    open at its start."""
    escaped = False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
        elif quote == "'":
            if ch == "'":
                quote = None
        elif ch == "\\":
            escaped = True
        elif quote == '"':
            if ch == '"':
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t;&|()"):
            break  # a comment runs to the end of the line
    return quote


def _command_lines(command: str) -> Iterator[str]:
    """The command lines of *command*, in order.

    Physical lines — except that a quoted string spanning lines (``git
    commit -m "...<newline>..."``, ``python3 -c "..."``) stays in the line it
    started on, so its continuation lines are never read as commands.
    Heredoc bodies are data and skipped, unless a shell reads them (``bash
    <<EOF``), when each body line is a command line too. Of a quote never
    closed (the shell refuses the whole command) only the line that opened
    it is returned.
    """
    text = str(command or "").replace("\\\n", " ")
    heredoc_end: str | None = None
    heredoc_is_shell = False
    pending: list[str] = []
    quote: str | None = None
    for line in text.splitlines():
        if heredoc_end is not None:
            if line.strip() == heredoc_end:
                heredoc_end = None
            elif heredoc_is_shell and line.strip():
                yield line
            continue
        pending.append(line)
        quote = _open_quote(line, quote)
        if quote:
            continue
        logical = "\n".join(pending)
        pending = []
        if logical.strip():
            yield logical
        match = _HEREDOC_RE.search(logical)
        if match:
            heredoc_end = match.group(2)
            heredoc_is_shell = bool(_SHELL_LINE_RE.match(logical))
    if pending and pending[0].strip():
        yield pending[0]


def _line_commands(line: str, depth: int) -> Iterator[tuple[str | None, list[str]]]:
    """``(program, args)`` for each command *line* runs — ``sh -c`` and
    ``eval`` scripts included; ``(None, [line])`` when it won't tokenize."""
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.wordchars += ":@%+,"  # an unquoted URL stays one token
    try:
        tokens = list(lexer)
    except ValueError:
        yield None, [line]
        return

    at_command = True
    in_backtick = False
    wrapper: str | None = None
    skip_value = False
    for i, tok in enumerate(tokens):
        if tok == "`":
            in_backtick = not in_backtick
            at_command, wrapper, skip_value = in_backtick, None, False
            continue
        if tok in _CMD_STARTERS:
            at_command, wrapper, skip_value = True, None, False
            continue
        if tok == ")":
            at_command, wrapper = False, None
            continue
        if not at_command:
            continue
        if skip_value:
            skip_value = False
            continue
        if wrapper is None and (tok in _CMD_KEYWORDS or _ASSIGNMENT_RE.match(tok)):
            continue
        if wrapper is not None:
            if wrapper == "command" and tok in ("-v", "-V"):
                at_command, wrapper = False, None  # a lookup, not a run
                continue
            if tok.startswith("-") or (wrapper == "env" and _ASSIGNMENT_RE.match(tok)):
                skip_value = tok in _WRAPPER_VALUE_OPTS
                continue
            if wrapper == "timeout":
                wrapper = "timeout:duration"  # the duration; the command follows
                continue
        if tok in _CMD_WRAPPERS:
            wrapper = tok
            continue
        program = tok.rsplit("/", 1)[-1]
        rest = tokens[i + 1:]
        args: list[str] = []
        for arg in rest:
            if arg in _CMD_STARTERS or arg in ("`", ")"):
                break
            args.append(arg)
        yield program, args
        # sh -c '<script>' / eval '<script>': the script is a command line too.
        if program in _SHELLS and depth < 3:
            for j, arg in enumerate(rest[:-1]):
                if arg in _CMD_STARTERS:
                    break
                if _SHELL_C_OPT_RE.match(arg):
                    yield from _commands(rest[j + 1], depth + 1)
                    break
        if program == "eval" and depth < 3:
            yield from _commands(" ".join(args), depth + 1)
        at_command, wrapper = False, None


def _commands(command: str, depth: int = 0) -> Iterator[tuple[str | None, list[str]]]:
    """``(program, args)`` for every command *command* runs (see
    :func:`_command_lines` and :func:`_line_commands`)."""
    for line in _command_lines(command):
        yield from _line_commands(line, depth)


def _runs_gws(command: str, depth: int = 0) -> bool:
    """True if *command* runs the gws binary (at a command position)."""
    for program, args in _commands(command, depth):
        if program == "gws":
            return True
        if program is None and _GWS_FALLBACK_RE.search(args[0]):
            return True
    return False


def _drive_download_url(command: str) -> str | None:
    """The first Drive file/folder URL a curl/wget in *command* fetches.

    Only the fetch's own URL operands count: a Drive link in a -d / -F / -H
    / --json / --referer value (a webhook post carrying a Doc link), or in
    another command (``echo <link> >> sources.md``), is no download. Forms,
    published pages and id-less Drive URLs never count — google_drive can't
    open them.
    """
    if not _GDRIVE_DOWNLOAD_RE.search(command):
        return None
    for program, args in _commands(command):
        if program is None:
            # Won't tokenize (the shell refuses it too): the plain match.
            url = first_drive_url(args[0]) if _GDRIVE_DOWNLOAD_RE.search(args[0]) else None
            if url and google_drive_can_open(url):
                return url
            continue
        if program not in _FETCHERS:
            continue
        value_opts = _FETCH_VALUE_OPTS[program]
        skip_value = False
        for arg in args:
            if skip_value:
                skip_value = False
                continue
            if arg.startswith("-"):
                # --data X, or a short bundle ending in one (-sd X).
                skip_value = arg in value_opts or (
                    not arg.startswith("--") and len(arg) > 2 and f"-{arg[-1]}" in value_opts
                )
                continue
            if is_google_drive_url(arg) and google_drive_can_open(arg):
                return arg
    return None


# A private directory holding a ``gws`` stub that refuses, put first on
# PATH for every command — so a script that shells out to gws (which the
# command check above can't see into) gets the same answer.
_GWS_SHIM_DIR: str | None = None


def _gws_shim_dir() -> str | None:
    global _GWS_SHIM_DIR
    if _GWS_SHIM_DIR and os.path.isfile(os.path.join(_GWS_SHIM_DIR, "gws")):
        return _GWS_SHIM_DIR
    if os.name == "nt":
        return None
    try:
        shim_dir = tempfile.mkdtemp(prefix="claw-gws-retired-")  # 0700, unguessable
        stub = os.path.join(shim_dir, "gws")
        with open(stub, "w", encoding="utf-8") as fh:
            fh.write(f"#!/bin/sh\necho '{_GWS_RETIRED_MSG}' >&2\nexit 127\n")
        os.chmod(stub, 0o755)
    except OSError as exc:
        log.warning("gws PATH stub unavailable", error=str(exc))
        return None
    _GWS_SHIM_DIR = shim_dir
    return shim_dir


# Commands that complete nearly instantly and should never hang for the
# full config timeout.  We use a short timeout (5 s) for these — they
# finish in <1 s under normal conditions; the 5 s budget only matters
# when the filesystem is extremely slow or the command hangs.
_QUICK_COMMANDS: frozenset[str] = frozenset({
    "cd", "ls", "pwd", "echo", "printf", "cat", "head", "tail",
    "mkdir", "rmdir", "touch", "cp", "mv", "rm", "ln",
    "chmod", "chown", "chgrp",
    "date", "cal", "whoami", "hostname", "uname", "env", "printenv",
    "which", "type", "file", "stat", "wc", "sort", "uniq",
    "basename", "dirname", "realpath", "readlink",
    "true", "false", "test", "[",
    "export", "unset", "set", "alias",
})
_QUICK_TIMEOUT = 5

# Script interpreters that typically run longer than simple commands.
# When the shell config timeout is low (e.g. 30 s), these get a minimum
# floor so that scripts have a reasonable initial window before the
# activity-based timeout kicks in.
_SCRIPT_COMMANDS: frozenset[str] = frozenset({
    "python3", "python", "python3.11", "python3.12", "python3.13",
    "node", "ruby", "perl", "bash", "sh", "zsh",
})
_SCRIPT_MIN_TIMEOUT = 120

# Hard wall-time cap for activity-based timeout extension (30 minutes).
# Even if a process is producing output, it cannot run longer than this.
_HARD_WALL_TIME = 1800


class ShellTool(Tool):
    """Execute shell commands."""

    name = "shell"
    description = "Execute a shell command and return its output."
    timeout_seconds = 120.0
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (optional, default from config)",
            },
        },
        "required": ["command"],
    }

    def __init__(self):
        self.config = get_config()
        # The shell tool manages its own timeouts internally with an
        # activity-based system + hard wall-time cap (_HARD_WALL_TIME).
        # Set the registry-level timeout to the hard cap so the outer
        # wrapper in tools/registry.py never kills a command before the
        # shell's own timeout handling can act.  Quick commands, normal
        # commands, and scripts all have appropriate internal timeouts
        # that will terminate them long before the hard cap.
        self.timeout_seconds = float(_HARD_WALL_TIME)

    @staticmethod
    def _is_script_command(command: str) -> bool:
        """Return True if *command* runs a script interpreter (python3, node, etc.).

        Checks ALL parts of chained commands (``cd foo && python3 bar.py``)
        so that ``python3`` is detected even if preceded by ``cd``.
        Used to enforce a minimum timeout floor so that scripts have a
        reasonable initial window before the activity-based timeout kicks in.
        """
        stripped = re.sub(r"^\s*(\w+=\S+\s+)*", "", command).strip()
        parts = re.split(r"\s*(?:&&|\|\|?|;)\s*", stripped)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            base = part.split()[0].split("/")[-1] if part.split() else ""
            if base in _SCRIPT_COMMANDS:
                return True
        return False

    @staticmethod
    def _is_quick_command(command: str) -> bool:
        """Return True if *command* consists only of fast, non-blocking builtins.

        Handles pipelines (``ls | head``) and chains (``mkdir a && cd a``).
        If *any* part of the command is not in the quick-list, return False
        so the full timeout is used.
        """
        # Strip leading env vars (VAR=val cmd …)
        stripped = re.sub(r"^\s*(\w+=\S+\s+)*", "", command)
        # Split on shell operators to get individual commands
        parts = re.split(r"\s*(?:&&|\|\|?|;)\s*", stripped)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Get the base command name (handle paths like /usr/bin/ls)
            base = part.split()[0].split("/")[-1] if part.split() else ""
            if base not in _QUICK_COMMANDS:
                return False
        return True

    @staticmethod
    def _is_not_a_command(command: str) -> str | None:
        """Return a reason string if *command* is clearly not a shell command.

        Catches cases where the LLM puts prose, directory-tree diagrams,
        formatted text, or other non-shell content into a ``shell`` tool call.
        Returns ``None`` when the input looks plausibly like a real command.
        """
        stripped = command.strip()
        if not stripped:
            return None  # handled elsewhere as empty

        # Tree-drawing characters (└── ├── │ etc.) — never valid shell.
        if re.search(r"[└├│─┌┐┘┤┬┴┼╔╗╚╝║═]", stripped):
            return "Input contains tree-drawing characters — not a shell command"

        # Multi-line input where the majority of lines don't start with a
        # plausible command token — likely prose or formatted output.
        lines = [l.strip() for l in stripped.splitlines() if l.strip()]
        if len(lines) >= 3:
            non_cmd = 0
            for line in lines:
                # Lines starting with common prose/formatting indicators.
                if re.match(
                    r"^([-*•·▸▹►▻→⇒]|#{1,6}\s|\d+[.)]\s|>|```|$)",
                    line,
                ):
                    non_cmd += 1
            if non_cmd > len(lines) * 0.5:
                return "Input looks like prose or formatted text — not a shell command"

        return None

    def _is_command_safe(
        self, command: str, *, drive_links_readable: bool | None = None,
    ) -> tuple[bool, str]:
        """Check if command is safe to execute.

        Args:
            command: Command to check
            drive_links_readable: Whether google_drive can open a pasted
                Drive link (see ``google_drive_reads_links``); None falls
                back to "Google is connected".

        Returns:
            Tuple of (is_safe, reason)
        """
        # Reject obvious non-commands (tree diagrams, prose, etc.).
        not_cmd_reason = self._is_not_a_command(command)
        if not_cmd_reason:
            return False, not_cmd_reason

        # The retired Google Workspace CLI never runs.
        if _runs_gws(command):
            return False, _GWS_RETIRED_MSG

        # curl/wget fetching a Drive/Docs file → google_drive, while it can
        # open the file (execute() checks the connection's scope; a direct
        # call goes by the connection alone).
        drive_url = _drive_download_url(command)
        if drive_url:
            if drive_links_readable is None:
                from captain_claw.google_oauth_manager import is_google_connected_cached

                drive_links_readable = is_google_connected_cached()
            if drive_links_readable:
                return False, google_drive_redirect(
                    drive_url, _GDRIVE_SHELL_BLOCK_INTRO, connected=True,
                )

        # Check blocked patterns
        blocked, matched = is_blocked_shell_command(command, self.config.tools.shell.blocked)
        if blocked:
            if matched == "empty_command":
                return False, "Command is empty"
            if matched == "unparseable_command":
                return False, "Command is not parseable"
            return False, f"Command matches blocked pattern: {matched}"
        
        # Check allowed list (if non-empty)
        if self.config.tools.shell.allowed_commands:
            allowed = {
                str(item).strip()
                for item in self.config.tools.shell.allowed_commands
                if str(item).strip()
            }
            base_commands = extract_shell_base_commands(command)
            if not base_commands:
                return False, "Command is not parseable"
            for base_cmd in base_commands:
                normalized = base_cmd.split("/")[-1]
                if base_cmd not in allowed and normalized not in allowed:
                    return False, f"Command not in allowed list: {base_cmd}"
        
        return True, ""

    async def execute(self, command: str, timeout: int | None = None, **kwargs: Any) -> ToolResult:
        """Execute a shell command.
        
        Args:
            command: Shell command to execute
            timeout: Optional timeout override
        
        Returns:
            ToolResult with command output
        """
        # Check safety
        drive_links_readable: bool | None = None
        if _drive_download_url(command):
            from captain_claw.tools.google_drive import (
                agent_offers_google_drive,
                google_drive_reads_links,
            )

            drive_links_readable = (
                agent_offers_google_drive(kwargs.get("_agent"))
                and await google_drive_reads_links()
            )
        is_safe, reason = self._is_command_safe(
            command, drive_links_readable=drive_links_readable,
        )
        if not is_safe:
            log.warning("Blocked unsafe command", command=command, reason=reason)
            return ToolResult(
                success=False,
                error=f"Command blocked: {reason}",
            )
        
        # Use config timeout if not provided; auto-shorten for trivial
        # commands so a stalled ``ls`` or ``mkdir`` won't block for 120 s.
        if timeout is None:
            if self._is_quick_command(command):
                timeout = _QUICK_TIMEOUT
            else:
                timeout = self.config.tools.shell.timeout
        timeout = max(1, int(timeout))

        # Script interpreters (python3, node, etc.) need a reasonable
        # initial window — the first API call from within a script may
        # block for many seconds before any output appears.  Enforce a
        # minimum inactivity timeout so the script isn't killed prematurely.
        if self._is_script_command(command):
            timeout = max(timeout, _SCRIPT_MIN_TIMEOUT)

        abort_event = kwargs.get("_abort_event")
        if isinstance(abort_event, asyncio.Event) and abort_event.is_set():
            return ToolResult(success=False, error="Command aborted")
        
        # Set up environment
        env = os.environ.copy()
        env["PATH"] = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
        gws_shim = _gws_shim_dir()
        if gws_shim:
            env["PATH"] = gws_shim + os.pathsep + env["PATH"]

        # Resolve shell CWD to the workspace root so that relative paths
        # in commands (e.g., "ls pdf-test/") behave consistently with other
        # tools (glob, read, write) that also resolve against the workspace.
        runtime_base = kwargs.get("_runtime_base_path")
        shell_cwd: str | None = str(runtime_base) if runtime_base is not None else None

        # Pre-create session-scoped directories under saved/ so that shell
        # commands writing to paths like "saved/tmp/{session_id}/file.png"
        # don't accidentally create a FILE at the session_id path when the
        # intermediate directory is missing.
        session_id = kwargs.get("_session_id")
        if runtime_base and session_id:
            from pathlib import Path

            _saved = Path(runtime_base) / "saved"
            for _cat in ("tmp", "scripts", "showcase", "media", "output", "downloads"):
                _dir = _saved / _cat / str(session_id)
                if not _dir.exists():
                    _dir.mkdir(parents=True, exist_ok=True)

        try:
            log.info("Executing shell command", command=command, timeout=timeout)
            stream_cb = kwargs.get("_stream_callback")

            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=shell_cwd,
            )

            # Read stdout/stderr line-by-line, streaming each line to the UI.
            # Track last activity time for activity-based timeout extension:
            # as long as the process is producing output, the timeout resets.
            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []
            last_activity = [time.monotonic()]

            async def _read_stream(
                stream: asyncio.StreamReader, collected: list[str], prefix: str = "",
            ) -> None:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    last_activity[0] = time.monotonic()
                    text = line.decode("utf-8", errors="replace")
                    collected.append(text)
                    if stream_cb:
                        try:
                            stream_cb(prefix + text)
                        except Exception:
                            pass

            async def _collect() -> None:
                await asyncio.gather(
                    _read_stream(process.stdout, stdout_chunks),
                    _read_stream(process.stderr, stderr_chunks),
                )
                await process.wait()

            collect_task = asyncio.create_task(_collect())
            abort_wait_task: asyncio.Task[bool] | None = None
            if isinstance(abort_event, asyncio.Event):
                abort_wait_task = asyncio.create_task(abort_event.wait())
            try:
                # Activity-based timeout: instead of a single fixed wait, poll
                # periodically.  Whenever new stdout/stderr output arrives, the
                # inactivity deadline resets.  A hard wall-time cap prevents
                # infinite-running processes even if they keep producing output.
                wall_deadline = time.monotonic() + _HARD_WALL_TIME
                inactivity_deadline = time.monotonic() + timeout
                timed_out = False
                aborted = False

                while True:
                    wait_tasks: set[asyncio.Task[Any]] = {collect_task}
                    if abort_wait_task is not None:
                        wait_tasks.add(abort_wait_task)

                    remaining = max(0.1, min(
                        inactivity_deadline - time.monotonic(),
                        wall_deadline - time.monotonic(),
                    ))
                    check_interval = min(5.0, remaining)

                    done, _ = await asyncio.wait(
                        wait_tasks,
                        timeout=check_interval,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if collect_task in done:
                        await collect_task
                        break

                    if abort_wait_task is not None and abort_wait_task in done:
                        aborted = True
                        break

                    # Extend inactivity deadline if output was received recently
                    now = time.monotonic()
                    time_since_activity = now - last_activity[0]
                    if time_since_activity < timeout:
                        inactivity_deadline = max(
                            inactivity_deadline, last_activity[0] + timeout,
                        )

                    if now >= wall_deadline or now >= inactivity_deadline:
                        timed_out = True
                        break

                if aborted:
                    process.kill()
                    await process.wait()
                    collect_task.cancel()
                    try:
                        await collect_task
                    except asyncio.CancelledError:
                        pass
                    return ToolResult(success=False, error="Command aborted")

                if timed_out:
                    process.kill()
                    await process.wait()
                    collect_task.cancel()
                    try:
                        await collect_task
                    except asyncio.CancelledError:
                        pass
                    return ToolResult(
                        success=False,
                        error=f"Command timed out after {timeout}s of inactivity",
                    )
            except asyncio.CancelledError:
                process.kill()
                await process.wait()
                collect_task.cancel()
                raise
            finally:
                if abort_wait_task is not None and not abort_wait_task.done():
                    abort_wait_task.cancel()
                    try:
                        await abort_wait_task
                    except asyncio.CancelledError:
                        pass

            # Combine collected output
            stdout_text = "".join(stdout_chunks).strip()
            stderr_text = "".join(stderr_chunks).strip()

            output = stdout_text
            if stderr_text:
                output += f"\n[stderr] {stderr_text}"

            # Truncate if too long
            max_length = 10000
            if len(output) > max_length:
                output = output[:max_length] + f"\n... [truncated, {len(output)} total chars]"

            return ToolResult(
                success=process.returncode == 0,
                content=output or "[no output]",
            )
            
        except Exception as e:
            log.error("Shell command failed", command=command, error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
            )
