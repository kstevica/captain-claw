"""Shared runtime for the Google Workspace (gws) tool mixins.

Hosts the subprocess runner, binary resolution, constants, and shared
helpers used by :mod:`_gws_drive`, :mod:`_gws_docs`, and
:mod:`_gws_calendar`.  The concrete ``GwsTool`` composes these mixins.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from pathlib import Path
from typing import Any

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.tools.registry import ToolResult

log = get_logger(__name__)

# Maximum output length returned to the agent.
_MAX_OUTPUT_CHARS = 60_000

# Default timeout for gws commands (seconds).
_DEFAULT_TIMEOUT = 120

# Pattern matching inline base64-encoded images (can be hundreds of KB in exported markdown).
_BASE64_IMG_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+")

# ── Google identity under Flight Deck ────────────────────────────────
#
# The gws CLI picks its Google identity, in order (googleworkspace/cli
# ``auth::get_token``): GOOGLE_WORKSPACE_CLI_TOKEN (a raw access token,
# used as-is), GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE, its on-disk
# ``gws auth login`` store, then ADC (GOOGLE_APPLICATION_CREDENTIALS /
# ~/.config/gcloud). Left to itself, an FD-spawned agent's gws acts as
# whichever of those it happens to see — the same one for every user on
# the deck. Under Flight Deck we instead hand it the agent OWNER's access
# token (the one google_drive / google_mail use) and scrub the other
# env-level credential sources, so gws can only ever be that owner.
_GWS_TOKEN_ENV = "GOOGLE_WORKSPACE_CLI_TOKEN"
_GWS_CREDENTIAL_ENV_VARS = frozenset({
    _GWS_TOKEN_ENV,
    "GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE",
    "GOOGLE_APPLICATION_CREDENTIALS",
})

FD_CONNECT_HINT = "Connect your Google account in Flight Deck → Connections → Google."

# Under Flight Deck the token carries the DECK's configured scopes: a user
# reconnecting only re-grants those, and only an admin can change them.
_FD_SCOPE_FIX = (
    "an admin must add that scope in Flight Deck → Connections → Google → "
    "Scopes; each user then reconnects Google there (reconnecting alone "
    "can't add a scope)."
)
FD_SCOPE_HINT = _FD_SCOPE_FIX[0].upper() + _FD_SCOPE_FIX[1:]

_G = "https://www.googleapis.com/auth/"
_DRIVE_READ = frozenset({
    _G + "drive", _G + "drive.readonly",
    _G + "drive.metadata", _G + "drive.metadata.readonly",
})
# Any Drive scope can run a listing — with drive.file alone it returns only the
# files this app created (see _drive_file_only_note).
_DRIVE_LIST = _DRIVE_READ | {_G + "drive.file"}
_DRIVE_ANY = frozenset({_G + "drive", _G + "drive.readonly", _G + "drive.file"})
_DRIVE_WRITE = frozenset({_G + "drive", _G + "drive.file"})
_CALENDAR_READ = frozenset({
    _G + "calendar", _G + "calendar.readonly",
    _G + "calendar.events", _G + "calendar.events.readonly",
})
_CALENDAR_WRITE = frozenset({_G + "calendar", _G + "calendar.events"})
_DOCS_READ = frozenset({
    _G + "documents", _G + "documents.readonly",
    _G + "drive", _G + "drive.readonly", _G + "drive.file",
})
_DOCS_WRITE = frozenset({_G + "documents", _G + "drive", _G + "drive.file"})
_DOCS_WRITE_WORDS = frozenset({"+write", "batchUpdate", "create"})
_CALENDAR_WRITE_WORDS = frozenset({
    "+insert", "insert", "update", "patch", "delete", "move", "quickAdd", "import",
})
_DRIVE_WRITE_WORDS = frozenset({"+upload", "create", "update", "copy", "delete"})


class GwsNotConnected(RuntimeError):
    """Flight Deck mode, but no Google token for the agent's owner."""


class _GwsEnv(dict):
    """A Flight Deck gws env that also remembers the token's granted scopes
    (cached and cleared together with the env — see ``_gws_env``)."""

    granted_scopes: frozenset[str] = frozenset()


def gws_flight_deck_mode() -> bool:
    """True when this agent's Google identity is managed by Flight Deck."""
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    return bool(GoogleOAuthManager._flight_deck_base())


async def gws_subprocess_env() -> dict[str, str] | None:
    """Environment for a gws subprocess.

    Standalone: ``None`` — inherit the process env, so gws keeps using its
    own ``gws auth login`` exactly as before. Flight Deck: a copy of the env
    with the credential vars scrubbed and the owner's access token injected.
    Raises :class:`GwsNotConnected` when Flight Deck has no token for the
    owner (or refuses this agent, with FD's reason) — fail closed rather than
    fall back to an ambient credential that belongs to someone else. The one
    exception is an auth-disabled deck: it has a single tenant and no Google
    via FD, so gws keeps its own credentials there, as before.
    """
    if not gws_flight_deck_mode():
        return None

    from captain_claw.google_oauth_manager import FlightDeckRefused, GoogleOAuthManager
    from captain_claw.session import get_session_manager

    try:
        tokens = await GoogleOAuthManager(get_session_manager()).get_tokens()
    except FlightDeckRefused as exc:
        if exc.auth_disabled:
            return None
        raise GwsNotConnected(f"gws: {exc}") from exc
    if not tokens or not tokens.access_token:
        raise GwsNotConnected(
            "gws: Flight Deck returned no Google token for this agent's owner. "
            + FD_CONNECT_HINT
        )
    env = _GwsEnv((k, v) for k, v in os.environ.items() if k not in _GWS_CREDENTIAL_ENV_VARS)
    env[_GWS_TOKEN_ENV] = tokens.access_token
    env.granted_scopes = frozenset((tokens.scope or "").split())
    return env


def _command_words(args: list[str]) -> list[str]:
    """The leading command words of a gws invocation, up to the first flag."""
    words: list[str] = []
    for a in args:
        if a.startswith("-"):
            break
        words.append(a)
    return words


def _gws_scope_need(args: list[str]) -> tuple[frozenset[str], str] | None:
    """(scopes any one of which the gws command needs, the one to ask an admin
    for) — or None when the command isn't one we know."""
    words = _command_words(args)
    if not words:
        return None
    service, rest = words[0], set(words[1:])
    if service == "calendar":
        if rest & _CALENDAR_WRITE_WORDS:
            return _CALENDAR_WRITE, _G + "calendar"
        return _CALENDAR_READ, _G + "calendar.readonly"
    if service == "docs":
        if rest & _DOCS_WRITE_WORDS:
            return _DOCS_WRITE, _G + "drive"
        return _DOCS_READ, _G + "drive.readonly"
    if service == "drive":
        if rest & _DRIVE_WRITE_WORDS:
            return _DRIVE_WRITE, _G + "drive.file"
        if "list" in rest:
            return _DRIVE_LIST, _G + "drive.readonly"
        return _DRIVE_ANY, _G + "drive.readonly"
    return None


def _missing_scope_error(args: list[str], granted: frozenset[str]) -> str | None:
    """Why the Flight Deck token can't run this gws command, or None.

    Unknown commands and an unknown grant (FD reported no scope) pass — gws
    and Google then decide, as before.
    """
    need = _gws_scope_need(args)
    if not need or not granted:
        return None
    accepted, ask = need
    if granted & accepted:
        return None
    return (
        f"gws: the Flight Deck Google connection lacks the scope this needs "
        f"({ask.removeprefix(_G)} or equivalent). {FD_SCOPE_HINT}"
    )


def _drive_file_only_note(args: list[str], granted: frozenset[str]) -> str | None:
    """For a Drive listing / search run with per-file access (drive.file) and
    no broader Drive read scope: what the result leaves out, and who can fix
    it. None otherwise."""
    words = _command_words(args)
    if not words or words[0] != "drive" or "list" not in words[1:]:
        return None
    if _G + "drive.file" not in granted or granted & _DRIVE_READ:
        return None
    return (
        "Note: with per-file access only (drive.file), Drive listings show just "
        "the files this app created. To list the rest of Drive, an admin must "
        "add the drive.readonly scope in Flight Deck → Connections → Google → "
        "Scopes; each user then reconnects Google there."
    )


def _strip_base64_images(text: str) -> str:
    """Remove inline base64 image data from text to prevent context bloat."""
    cleaned = _BASE64_IMG_RE.sub("[image]", text)
    if len(cleaned) < len(text):
        log.debug(
            "stripped base64 images",
            original_len=len(text),
            cleaned_len=len(cleaned),
        )
    return cleaned


class GwsRuntimeMixin:
    """Base mixin: binary resolution + subprocess runner for gws commands."""

    def __init__(self) -> None:
        self._binary: str | None = None
        self._stream_callback: Any = None
        # Subprocess env for the current tool call (see _gws_env).
        self._gws_env_cache: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # Binary resolution
    # ------------------------------------------------------------------

    def _resolve_binary(self) -> str | None:
        """Find the gws binary (config override → PATH)."""
        if self._binary and shutil.which(self._binary):
            return self._binary

        try:
            cfg = get_config()
            custom = getattr(cfg.tools, "gws", None)
            if custom and hasattr(custom, "binary_path") and custom.binary_path:
                p = Path(custom.binary_path).expanduser()
                if p.exists():
                    self._binary = str(p)
                    return self._binary
        except Exception:
            pass

        found = shutil.which("gws")
        if found:
            self._binary = found
            return self._binary

        return None

    # ------------------------------------------------------------------
    # Subprocess environment
    # ------------------------------------------------------------------

    async def _gws_env(self) -> dict[str, str] | None:
        """Subprocess env, resolved once per tool call.

        ``execute`` clears the cache around each call, so a paginated or
        recursive action costs one Flight Deck round-trip, not one per page.
        Raises :class:`GwsNotConnected` (Flight Deck mode, no owner token).
        """
        env = getattr(self, "_gws_env_cache", None)
        if env is None:
            env = await gws_subprocess_env()
            self._gws_env_cache = env
        return env

    # ------------------------------------------------------------------
    # Core runner
    # ------------------------------------------------------------------

    async def _run_gws(
        self,
        binary: str,
        args: list[str],
        timeout: float = _DEFAULT_TIMEOUT,
        json_output: bool = True,
    ) -> ToolResult:
        """Run a gws command, stream stdout/stderr, return captured output."""
        cmd = [binary] + args
        if json_output and "--format" not in args:
            cmd.extend(["--format", "json"])

        log.debug("Running gws command", cmd=" ".join(cmd))
        stream_cb = getattr(self, "_stream_callback", None)

        try:
            env = await self._gws_env()
        except GwsNotConnected as exc:
            return ToolResult(success=False, error=str(exc))
        scope_note: str | None = None
        if isinstance(env, _GwsEnv):
            missing = _missing_scope_error(args, env.granted_scopes)
            if missing:
                return ToolResult(success=False, error=missing)
            scope_note = _drive_file_only_note(args, env.granted_scopes)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        async def _read_stream(
            stream: asyncio.StreamReader, collected: list[str], prefix: str = "",
        ) -> None:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")
                collected.append(text)
                if stream_cb:
                    try:
                        stream_cb(prefix + text)
                    except Exception:
                        pass

        async def _collect() -> None:
            await asyncio.gather(
                _read_stream(proc.stdout, stdout_chunks),
                _read_stream(proc.stderr, stderr_chunks),
            )
            await proc.wait()

        try:
            await asyncio.wait_for(_collect(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult(success=False, error="gws command timed out.")

        stdout_str = "".join(stdout_chunks).strip()
        stderr_str = "".join(stderr_chunks).strip()

        if proc.returncode != 0:
            error_msg = stderr_str or stdout_str or f"gws exited with code {proc.returncode}"
            lowered = error_msg.lower()
            if env is not None and any(
                w in lowered for w in ("no credentials", "token", "scope", "insufficient")
            ):
                # Flight Deck mode: `gws auth login` would not help — the
                # identity is the owner's Flight Deck connection, with the
                # deck's configured scopes.
                error_msg += (
                    "\n\nHint: gws runs as the Google account connected in "
                    "Flight Deck → Connections → Google. If it was disconnected "
                    "or expired, reconnect it there. If Google reports a missing "
                    "scope or insufficient permission, " + _FD_SCOPE_FIX
                )
            elif "no credentials" in lowered or "token" in lowered:
                error_msg += "\n\nHint: Run 'gws auth login' to authenticate."
            return ToolResult(success=False, error=error_msg)

        output = stdout_str
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + "\n\n... [output truncated]"

        # The note rides in system_hint (appended for the model), not the
        # content: callers json.loads a listing's content.
        return ToolResult(success=True, content=output, system_hint=scope_note)
