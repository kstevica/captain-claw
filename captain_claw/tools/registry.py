"""Tool registry and base tool class."""

import asyncio
import fnmatch
import re
import shlex
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, model_validator

from captain_claw.config import get_config
from captain_claw.exceptions import (
    ToolBlockedError,
    ToolExecutionError,
    ToolNotFoundError,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)

_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_SHELL_SEPARATOR_TOKENS = {";", "&&", "||", "|", "&"}
_SHELL_WRAPPER_TOKENS = {"sudo", "command", "builtin", "nohup", "time"}


_LOGGED_ARG_CHARS = 300

# The two ToolExecutionError messages the tool-call register reads as an outcome.
_ABORTED_MARK = "Execution aborted"
_TIMED_OUT_MARK = "Execution timed out after"


def _loggable_args(arguments: dict[str, Any]) -> dict[str, Any]:
    """The call's own arguments, each cut short, for the "Executing tool" line.

    Leaves out the runtime's injected ``_``-keys: ``_session`` alone is the
    whole conversation, and printing it on every call grew one agent's
    process.log to 3.7 GB.
    """
    out: dict[str, Any] = {}
    for key, value in arguments.items():
        if str(key).startswith("_"):
            continue
        text = value if isinstance(value, str) else repr(value)
        if len(text) > _LOGGED_ARG_CHARS:
            value = f"{text[:_LOGGED_ARG_CHARS]}… ({len(text)} chars)"
        out[key] = value
    return out


def _normalize_tool_name(value: str) -> str:
    """Normalize tool names for policy comparisons."""
    return str(value or "").strip().lower()


# A model writing its calls as Qwen's XML (``<parameter=name>value</parameter>``) that leaves one
# open sends the rest inside it — {"action": "search\n<parameter=query>\nWACC weights"} — or a
# closing tag's debris after a value ("search>\n\n</invoke>").
_LEAKED_PARAMETER = re.compile(r"<parameter=([A-Za-z_][\w-]*)>")
_TAG_DEBRIS = re.compile(r"\s*(?:</?(?:parameter|invoke|function|tool_call)\b[^>]*>|>)\s*$")


def _debris_off(value: str) -> str:
    before = None
    while value != before:
        before, value = value, _TAG_DEBRIS.sub("", value)
    return value.strip()


def _typed(raw: str, spec: dict[str, Any]) -> Any:
    """A leaked parameter's text as its schema's type, when it reads as one."""
    kind = spec.get("type") if isinstance(spec, dict) else None
    if kind == "integer" and re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if kind == "number" and re.fullmatch(r"-?\d+(?:\.\d+)?", raw):
        return float(raw)
    if kind == "boolean" and raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    return raw


def repair_arguments(arguments: dict[str, Any], parameters: dict[str, Any] | None) -> dict[str, Any]:
    """*arguments* with what a model's XML tool call left in them put right:
    parameters left inside another's value split out (one it also gave
    itself is kept), and a closing tag's debris taken off a value that must
    be one of the schema's few words. The same dict when nothing needed it."""
    props = (parameters or {}).get("properties") if isinstance(parameters, dict) else None
    if not isinstance(props, dict) or not props or not isinstance(arguments, dict):
        return arguments
    out = dict(arguments)
    for key, value in arguments.items():
        if not isinstance(value, str) or str(key).startswith("_"):
            continue
        if "<parameter=" in value:
            pieces = _LEAKED_PARAMETER.split(value)
            out[key] = _debris_off(pieces[0])
            for name, raw in zip(pieces[1::2], pieces[2::2], strict=True):
                if name in props and out.get(name) in (None, ""):
                    out[name] = _typed(_debris_off(raw), props[name])
        enum = props.get(key, {}).get("enum") if isinstance(props.get(key), dict) else None
        if enum and out[key] not in enum:
            word = _debris_off(out[key])
            first = re.split(r"[\s<>]+", word, maxsplit=1)[0]
            fixed = word if word in enum else first if first in enum else None
            if fixed is not None:
                out[key] = fixed
    return out if out != arguments else arguments


def _compile_shell_pattern(pattern: str) -> re.Pattern[str]:
    """Compile regex pattern with literal fallback for invalid regex input."""
    try:
        return re.compile(pattern)
    except re.error:
        return re.compile(re.escape(pattern))


def _tokenize_shell_command(command: str) -> list[str]:
    """Tokenize shell command while preserving control operators."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _split_shell_segments(command: str) -> list[list[str]]:
    """Split shell command into tokenized segments separated by control operators."""
    tokens = _tokenize_shell_command(command)
    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in _SHELL_SEPARATOR_TOKENS:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    return segments


def _extract_segment_base_command(tokens: list[str]) -> str:
    """Extract executable command token from a tokenized shell segment."""
    idx = 0
    while idx < len(tokens):
        token = str(tokens[idx]).strip()
        if not token:
            idx += 1
            continue
        if token in _SHELL_WRAPPER_TOKENS:
            idx += 1
            continue
        if _ASSIGNMENT_RE.match(token) and "/" not in token:
            idx += 1
            continue
        return token
    return ""


def extract_shell_base_commands(command: str) -> list[str]:
    """Extract base command token from each shell segment."""
    cleaned = str(command or "").strip()
    if not cleaned:
        return []
    try:
        segments = _split_shell_segments(cleaned)
    except ValueError:
        return []
    base_commands: list[str] = []
    for segment in segments:
        base = _extract_segment_base_command(segment)
        if base:
            base_commands.append(base)
    return base_commands


def is_blocked_shell_command(command: str, blocked_patterns: list[str]) -> tuple[bool, str]:
    """Evaluate command against blocked patterns using parsed command matching."""
    cleaned = str(command or "").strip()
    if not cleaned:
        return True, "empty_command"

    try:
        segments = _split_shell_segments(cleaned)
    except ValueError:
        return True, "unparseable_command"
    if not segments:
        return True, "unparseable_command"

    segment_texts = [" ".join(tokens) for tokens in segments]
    base_commands = [
        base
        for segment in segments
        if (base := _extract_segment_base_command(segment))
    ]
    if not base_commands:
        return True, "unparseable_command"

    for raw_pattern in blocked_patterns or []:
        pattern = str(raw_pattern or "").strip()
        if not pattern:
            continue
        compiled = _compile_shell_pattern(pattern)
        segment_level_pattern = bool(re.search(r"\s", pattern))
        targets = segment_texts if segment_level_pattern else base_commands
        matcher = compiled.search if segment_level_pattern else compiled.match
        for target in targets:
            if matcher(target):
                return True, pattern
    return False, ""


class ToolResult(BaseModel):
    """Result from tool execution."""

    success: bool = True
    content: str = ""
    error: str | None = None
    system_hint: str | None = None
    # Image files for the agent's own model to look at (when it sees
    # images): the tool loop adds a ``[Tool image: <path>]`` marker per path.
    images: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_failure_error(self) -> "ToolResult":
        """Ensure failed results always provide an error message."""
        if not self.success and not (self.error or "").strip():
            fallback = (self.content or "").strip()
            self.error = fallback or "Tool execution failed"
        return self


class Tool(ABC):
    """Base class for all tools."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = 30.0

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult with success status and content
        """
        pass

    def get_definition(self) -> dict[str, Any]:
        """Get the tool definition for LLM.

        Returns:
            OpenAI function-style definition
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        """Validate tool arguments against schema.

        Args:
            arguments: Arguments to validate

        Raises:
            ValidationError if invalid
        """
        # Basic validation - could use Pydantic for more robust validation
        required = self.parameters.get("required", [])
        for field in required:
            if field not in arguments:
                raise ToolExecutionError(
                    self.name,
                    f"Missing required argument: {field}",
                )


class ToolPolicy(BaseModel):
    """Policy rule set for filtering available tools."""

    allow: list[str] | None = None
    deny: list[str] = Field(default_factory=list)
    also_allow: list[str] = Field(default_factory=list)


class ToolPolicyChain:
    """Apply policies in cascade: global -> session -> task."""

    def __init__(self, steps: list[tuple[str, ToolPolicy]] | None = None):
        self.steps: list[tuple[str, ToolPolicy]] = list(steps or [])

    @staticmethod
    def _normalize_name_set(items: list[str] | None) -> set[str] | None:
        """Normalize optional tool-name list into comparable set."""
        if items is None:
            return None
        normalized = {
            _normalize_tool_name(item)
            for item in items
            if _normalize_tool_name(item)
        }
        return normalized

    def _apply(
        self,
        current_names: set[str],
        all_names: set[str],
        policy: ToolPolicy,
    ) -> set[str]:
        """Apply one policy step against current allowed tool names."""
        next_names = set(current_names)
        allow_names = self._normalize_name_set(policy.allow)
        if allow_names is not None:
            next_names = {name for name in next_names if name in allow_names}

        deny_names = self._normalize_name_set(policy.deny) or set()
        next_names -= deny_names

        also_allow_names = self._normalize_name_set(policy.also_allow) or set()
        next_names |= also_allow_names & all_names
        return next_names

    def resolve(self, available_tools: list[Tool]) -> list[Tool]:
        """Resolve final tool list after applying policy chain."""
        if not self.steps:
            return list(available_tools)

        tools_by_name: dict[str, Tool] = {
            _normalize_tool_name(tool.name): tool
            for tool in available_tools
            if _normalize_tool_name(tool.name)
        }
        all_names = set(tools_by_name.keys())
        current_names = set(all_names)
        for _, policy in self.steps:
            current_names = self._apply(current_names, all_names, policy)

        return [
            tool
            for tool in available_tools
            if _normalize_tool_name(tool.name) in current_names
        ]


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self, base_path: Path | str | None = None, saved_dir_name: str = "saved"):
        self._tools: dict[str, Tool] = {}
        self._tool_metadata: dict[str, dict[str, Any]] = {}
        self._saved_dir_name = (saved_dir_name or "saved").strip() or "saved"
        self._runtime_base_path = Path.cwd()
        self._global_policy: ToolPolicy | None = None
        self._session_policies: dict[str, ToolPolicy] = {}
        self._approval_callback: Callable[[str], bool] | None = None
        # Session keys (raw id and slug) owned by shared-agent speakers. A call
        # carrying one is a member call even if the speaker contextvar was lost
        # (executor threads) or never bound (paths that pass only session_id).
        self._speaker_session_keys: set[str] = set()
        self.set_runtime_base_path(base_path or Path.cwd())

    @staticmethod
    def _coerce_policy(policy: ToolPolicy | dict[str, Any] | None) -> ToolPolicy | None:
        """Coerce policy payload into ToolPolicy model."""
        if policy is None:
            return None
        if isinstance(policy, ToolPolicy):
            return policy
        if isinstance(policy, dict):
            return ToolPolicy(**policy)
        raise TypeError(f"Unsupported policy type: {type(policy)!r}")

    def set_runtime_base_path(self, base_path: Path | str) -> None:
        """Set runtime base path used by tools for local file output."""
        self._runtime_base_path = Path(base_path).expanduser().resolve()

    @property
    def runtime_base_path(self) -> Path:
        """Runtime base path from which Captain Claw was launched."""
        return self._runtime_base_path

    def get_saved_base_path(self, create: bool = False) -> Path:
        """Return `<runtime_base>/saved` (or custom save dir name)."""
        saved_root = (self._runtime_base_path / self._saved_dir_name).resolve()
        try:
            saved_root.relative_to(self._runtime_base_path)
        except ValueError:
            saved_root = (self._runtime_base_path / "saved").resolve()
        if create:
            saved_root.mkdir(parents=True, exist_ok=True)
        return saved_root

    def set_approval_callback(self, callback: Callable[[str], bool] | None) -> None:
        """Set approval callback used by ask-mode execution policies."""
        self._approval_callback = callback

    def register(self, tool: Tool, metadata: dict[str, Any] | None = None) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register
        """
        if not tool.name:
            raise ValueError("Tool must have a name")

        log.debug("Registering tool", tool=tool.name)
        self._tools[tool.name] = tool
        if isinstance(metadata, dict):
            self._tool_metadata[tool.name] = dict(metadata)
        elif tool.name not in self._tool_metadata:
            self._tool_metadata[tool.name] = {}

    def unregister(self, name: str) -> None:
        """Unregister a tool.

        Args:
            name: Tool name to unregister
        """
        if name in self._tools:
            del self._tools[name]
        self._tool_metadata.pop(name, None)

    def has_tool(self, name: str) -> bool:
        """Return whether a tool name is currently registered."""
        return name in self._tools

    def get_tool_metadata(self, name: str) -> dict[str, Any]:
        """Return metadata associated with a registered tool."""
        return dict(self._tool_metadata.get(name, {}))

    def set_global_policy(self, policy: ToolPolicy | dict[str, Any] | None) -> None:
        """Set global tool policy step."""
        self._global_policy = self._coerce_policy(policy)

    def set_session_policy(self, session_id: str, policy: ToolPolicy | dict[str, Any] | None) -> None:
        """Set policy for a specific session id."""
        key = str(session_id or "").strip()
        if not key:
            return
        parsed = self._coerce_policy(policy)
        if parsed is None:
            self._session_policies.pop(key, None)
            return
        self._session_policies[key] = parsed

    def clear_session_policy(self, session_id: str) -> None:
        """Clear policy assigned to a session id."""
        key = str(session_id or "").strip()
        if not key:
            return
        self._session_policies.pop(key, None)

    # ── Shared-agent speakers (A1) ──────────────────────────────────

    def register_speaker_session(self, key: str) -> None:
        """Mark a session key as a member's: calls with it get the speaker allowlist."""
        k = str(key or "").strip()
        if k:
            self._speaker_session_keys.add(k)

    def unregister_speaker_session(self, key: str) -> None:
        k = str(key or "").strip()
        if k:
            self._speaker_session_keys.discard(k)

    def _is_speaker_call(
        self, session_id: str | None, arguments: dict[str, Any] | None = None,
    ) -> bool:
        """Whether this call is made for a shared-agent member.

        Three independent signals, any one is enough: the speaker contextvar,
        a registered speaker session key, or a speaker-scoped ``_agent``.
        """
        from captain_claw import speaker as _speaker

        if _speaker.current() is not None:
            return True
        keys = getattr(self, "_speaker_session_keys", None) or ()
        if keys and str(session_id or "").strip() in keys:
            return True
        agent = (arguments or {}).get("_agent") if isinstance(arguments, dict) else None
        return _speaker.is_speaker_agent(agent)

    def _resolve_policy_chain(
        self,
        *,
        session_id: str | None = None,
        session_policy: ToolPolicy | dict[str, Any] | None = None,
        task_policy: ToolPolicy | dict[str, Any] | None = None,
    ) -> ToolPolicyChain:
        """Build tool policy chain in global -> session -> task order."""
        steps: list[tuple[str, ToolPolicy]] = []
        if self._global_policy is not None:
            steps.append(("global", self._global_policy))

        key = str(session_id or "").strip()
        policy_from_session = self._session_policies.get(key)
        policy_from_arg = self._coerce_policy(session_policy)
        effective_session_policy = policy_from_arg or policy_from_session
        if effective_session_policy is not None:
            steps.append(("session", effective_session_policy))

        effective_task_policy = self._coerce_policy(task_policy)
        if effective_task_policy is not None:
            steps.append(("task", effective_task_policy))

        # A member's call: the speaker allowlist is the LAST step and has no
        # also_allow, so no session/task policy (or per-turn set/clear) can
        # widen it. A2: the bound principal's own set (process members get
        # their Google / deep memory / file tools); a session-key-only signal
        # with nothing bound gets A1's set (fail closed) — `execute` evaluates
        # this chain inside the tool context, where the principal is bound.
        if self._is_speaker_call(session_id):
            from captain_claw import speaker as _speaker

            steps.append((
                "principal",
                ToolPolicy(allow=sorted(_speaker.allowed_tools(_speaker.current()))),
            ))
        # A non-admin's agent on a multi-user deck never sees the tools that
        # run code on the host (deck_confine); last, so nothing re-allows them.
        from captain_claw import deck_confine as _deck_confine

        if _deck_confine.tier() == "user":
            steps.append(("deck", ToolPolicy(deny=sorted(_deck_confine.EXEC_TOOLS))))
        return ToolPolicyChain(steps=steps)

    def _resolve_tools(
        self,
        *,
        session_id: str | None = None,
        session_policy: ToolPolicy | dict[str, Any] | None = None,
        task_policy: ToolPolicy | dict[str, Any] | None = None,
    ) -> list[Tool]:
        """Resolve currently available tools after policy filtering."""
        chain = self._resolve_policy_chain(
            session_id=session_id,
            session_policy=session_policy,
            task_policy=task_policy,
        )
        tools = chain.resolve(list(self._tools.values()))

        # A member recognised by its session key alone (nothing bound): the
        # Google flag below is per principal, and with no principal it would
        # read the OWNER's — never show a member Google tools on that (nor the
        # owner-only Microsoft / Jev ones).
        try:
            from captain_claw import speaker as _speaker

            if self._is_speaker_call(session_id) and _speaker.current() is None:
                tools = [
                    t for t in tools
                    if not self._tool_metadata.get(t.name, {}).get("requires_google")
                    and not self._tool_metadata.get(t.name, {}).get("requires_microsoft")
                    and not self._tool_metadata.get(t.name, {}).get("requires_jev")
                    and not self._tool_metadata.get(t.name, {}).get("requires_payments")
                    and not self._tool_metadata.get(t.name, {}).get("requires_telegram_inbox")
                    and not self._tool_metadata.get(t.name, {}).get("requires_whatsapp_web")
                    and not self._tool_metadata.get(t.name, {}).get("requires_knowledge")
                ]
        except Exception:
            pass

        # Auto-hide Google tools when Google OAuth isn't connected. Tools
        # are registered eagerly (so they can reappear mid-session the
        # moment the user connects in Flight Deck) but we don't want the
        # LLM to see them in its tool list if calls would immediately
        # fail. The connection flag is refreshed once per agent turn via
        # ``GoogleOAuthManager.is_connected()`` — see
        # ``agent_orchestration_mixin`` for the refresh hook.
        try:
            from captain_claw.google_oauth_manager import is_google_connected_cached

            if not is_google_connected_cached():
                filtered: list[Tool] = []
                for tool in tools:
                    meta = self._tool_metadata.get(tool.name, {})
                    if meta.get("requires_google"):
                        continue
                    filtered.append(tool)
                tools = filtered
        except Exception:
            # Never let the gating check break tool listing.
            pass

        # Same for Microsoft: shown only while the owner's Microsoft account is
        # connected in Flight Deck (refreshed per turn by
        # ``MicrosoftOAuthManager.is_connected()``).
        try:
            from captain_claw.microsoft_oauth_manager import is_microsoft_connected_cached

            if not is_microsoft_connected_cached():
                tools = [
                    t for t in tools
                    if not self._tool_metadata.get(t.name, {}).get("requires_microsoft")
                ]
        except Exception:
            pass

        # Jev: under Flight Deck shown only while the owner has a Jev key there
        # (refreshed per turn by ``jev_client.refresh_connected()``); standalone
        # only with a configured key. Fails closed — a call couldn't work.
        try:
            if any(self._tool_metadata.get(t.name, {}).get("requires_jev") for t in tools):
                from captain_claw import jev_client

                if not jev_client.jev_usable():
                    tools = [t for t in tools if not self._tool_metadata.get(t.name, {}).get("requires_jev")]
        except Exception:
            tools = [t for t in tools if not self._tool_metadata.get(t.name, {}).get("requires_jev")]

        # Payments: shown only while the owner's Flight Deck has payments on
        # (refreshed per turn by ``payments_client.refresh()``). Fails closed.
        try:
            if any(self._tool_metadata.get(t.name, {}).get("requires_payments") for t in tools):
                from captain_claw import payments_client

                if not payments_client.payments_usable():
                    tools = [t for t in tools if not self._tool_metadata.get(t.name, {}).get("requires_payments")]
        except Exception:
            tools = [t for t in tools if not self._tool_metadata.get(t.name, {}).get("requires_payments")]

        # Soniox (the audio tool): shown only while a key is usable — the
        # owner's (or team's) in Flight Deck, refreshed per turn by
        # ``soniox_client.available()``, or the agent's own SONIOX_API_KEY.
        # Fails closed. (A shared agent's member sees it only if the member
        # allowlist names it — it doesn't today.)
        try:
            if any(self._tool_metadata.get(t.name, {}).get("requires_soniox") for t in tools):
                from captain_claw import soniox_client

                if not soniox_client.soniox_usable():
                    tools = [t for t in tools if not self._tool_metadata.get(t.name, {}).get("requires_soniox")]
        except Exception:
            tools = [t for t in tools if not self._tool_metadata.get(t.name, {}).get("requires_soniox")]

        # Telegram inbox: shown only while the owner keeps one in Flight Deck
        # (refreshed per turn by ``telegram_inbox_client.refresh_connected()``).
        # Owner-only; fails closed.
        try:
            if any(self._tool_metadata.get(t.name, {}).get("requires_telegram_inbox") for t in tools):
                from captain_claw import telegram_inbox_client

                if not telegram_inbox_client.inbox_usable():
                    tools = [t for t in tools
                             if not self._tool_metadata.get(t.name, {}).get("requires_telegram_inbox")]
        except Exception:
            tools = [t for t in tools if not self._tool_metadata.get(t.name, {}).get("requires_telegram_inbox")]

        # WhatsApp Web: shown only while the owner has WhatsApp linked in Flight
        # Deck (refreshed per turn by ``whatsapp_web_client.refresh_connected()``).
        # Owner-only; fails closed.
        try:
            if any(self._tool_metadata.get(t.name, {}).get("requires_whatsapp_web") for t in tools):
                from captain_claw import whatsapp_web_client

                if not whatsapp_web_client.usable():
                    tools = [t for t in tools
                             if not self._tool_metadata.get(t.name, {}).get("requires_whatsapp_web")]
        except Exception:
            tools = [t for t in tools if not self._tool_metadata.get(t.name, {}).get("requires_whatsapp_web")]

        # Knowledge: shown only when the workspace's knowledge/ folder has
        # documents in it (an owner's setup gave some) — the tool looks.
        try:
            tools = [t for t in tools if not self._tool_metadata.get(t.name, {}).get("requires_knowledge")
                     or t.has_knowledge()]
        except Exception:
            tools = [t for t in tools if not self._tool_metadata.get(t.name, {}).get("requires_knowledge")]

        # Same idea for deep memory. Under Flight Deck the tool always shows:
        # FD owns the Typesense connection and can gain one at any moment, and
        # its error names the exact place to fix it. Standalone with no key and
        # no local index, though, the tool cannot do anything but fail — and an
        # advertised tool that always fails is worse than an absent one.
        try:
            from captain_claw.fd_client import is_under_flight_deck

            if not is_under_flight_deck():
                cfg = get_config()
                dm = getattr(cfg, "deep_memory", None)
                usable = bool(dm and getattr(dm, "enabled", False)) or bool(
                    str(getattr(cfg.tools.typesense, "api_key", "") or "").strip()
                )
                if not usable:
                    tools = [t for t in tools if t.name != "typesense"]
        except Exception:
            pass

        # PR D: shared_agent_usage only while Flight Deck says this agent has members.
        try:
            if any(self._tool_metadata.get(t.name, {}).get("requires_shared_members") for t in tools):
                from captain_claw.tenant_context import load_shared_members
                if not load_shared_members():
                    tools = [t for t in tools
                             if not self._tool_metadata.get(t.name, {}).get("requires_shared_members")]
        except Exception:
            tools = [t for t in tools
                     if not self._tool_metadata.get(t.name, {}).get("requires_shared_members")]

        return tools

    def get(self, name: str) -> Tool:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance

        Raises:
            ToolNotFoundError if not found
        """
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]

    def list_tools(
        self,
        *,
        session_id: str | None = None,
        session_policy: ToolPolicy | dict[str, Any] | None = None,
        task_policy: ToolPolicy | dict[str, Any] | None = None,
    ) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return [
            tool.name
            for tool in self._resolve_tools(
                session_id=session_id,
                session_policy=session_policy,
                task_policy=task_policy,
            )
        ]

    def get_definitions(
        self,
        *,
        session_id: str | None = None,
        session_policy: ToolPolicy | dict[str, Any] | None = None,
        task_policy: ToolPolicy | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Get all tool definitions for LLM.

        Returns:
            List of OpenAI function-style definitions
        """
        return [
            tool.get_definition()
            for tool in self._resolve_tools(
                session_id=session_id,
                session_policy=session_policy,
                task_policy=task_policy,
            )
        ]

    @staticmethod
    async def _cancel_task(task: asyncio.Task[Any] | None) -> None:
        """Cancel task and await it to avoid pending task warnings."""
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    @staticmethod
    async def _bridge_abort_event(source: asyncio.Event, target: asyncio.Event) -> None:
        """Mirror external abort event to local tool abort event."""
        await source.wait()
        target.set()

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        confirm: bool = False,
        session_id: str | None = None,
        session_policy: ToolPolicy | dict[str, Any] | None = None,
        task_policy: ToolPolicy | dict[str, Any] | None = None,
        abort_event: asyncio.Event | None = None,
        runtime_base_path: Path | None = None,
        approval_callback: Callable[[str], bool] | None = None,
        file_registry: Any | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> ToolResult:
        """Execute a tool by name, recorded in Flight Deck's tool-call register.

        The call is recorded before any check (tool_audit.py), so a refused
        call is in the register too, then its outcome: ``ok`` / ``error`` (the
        tool said it failed, or raised), ``blocked`` (a policy refused it, or
        no such tool), ``timeout`` or ``aborted``.
        """
        from captain_claw import tool_audit

        if name in self._tools:
            repaired = repair_arguments(arguments, getattr(self._tools[name], "parameters", None))
            if repaired is not arguments:
                log.info("Tool call arguments repaired", tool=name,
                         keys=sorted(k for k in repaired if repaired.get(k) != arguments.get(k)))
                arguments = repaired
        try:
            member = self._is_speaker_call(session_id, arguments)
        except Exception:
            member = False
        audit = tool_audit.start(name, arguments, session_id=session_id, member=member)
        try:
            result = await self._execute_call(
                name, arguments, confirm=confirm, session_id=session_id,
                session_policy=session_policy, task_policy=task_policy,
                abort_event=abort_event, runtime_base_path=runtime_base_path,
                approval_callback=approval_callback, file_registry=file_registry,
                stream_callback=stream_callback,
            )
        except ToolBlockedError as exc:
            tool_audit.finish(audit, "blocked", exc.reason)
            raise
        except ToolNotFoundError as exc:
            tool_audit.finish(audit, "blocked", str(exc))
            raise
        except ToolExecutionError as exc:
            text = str(exc)
            outcome = ("timeout" if _TIMED_OUT_MARK in text
                       else "aborted" if text.endswith(_ABORTED_MARK) else "error")
            tool_audit.finish(audit, outcome, text)
            raise
        except asyncio.CancelledError:
            tool_audit.finish(audit, "aborted", "Cancelled")
            raise
        except Exception as exc:
            tool_audit.finish(audit, "error", f"{type(exc).__name__}: {exc}")
            raise
        if getattr(result, "success", True):
            tool_audit.finish(audit, "ok", result=result)
        else:
            tool_audit.finish(audit, "error", str(getattr(result, "error", "") or ""))
        return result

    async def _execute_call(
        self,
        name: str,
        arguments: dict[str, Any],
        confirm: bool = False,
        session_id: str | None = None,
        session_policy: ToolPolicy | dict[str, Any] | None = None,
        task_policy: ToolPolicy | dict[str, Any] | None = None,
        abort_event: asyncio.Event | None = None,
        runtime_base_path: Path | None = None,
        approval_callback: Callable[[str], bool] | None = None,
        file_registry: Any | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> ToolResult:
        """Execute a tool by name.

        Args:
            name: Tool name
            arguments: Tool arguments
            confirm: Whether to confirm before execution

        Returns:
            ToolResult from execution

        Raises:
            ToolNotFoundError if tool not found
            ToolBlockedError if tool is blocked
            ToolExecutionError if execution fails
        """
        import contextvars

        from captain_claw import deck_confine as _deck_confine
        from captain_claw import pack_access as _pack_access
        from captain_claw import speaker as _speaker

        # Resolve per-call overrides (fall back to instance defaults).
        effective_base_path = Path(runtime_base_path or self._runtime_base_path)
        effective_approval = approval_callback or self._approval_callback
        effective_saved_base = (effective_base_path / self._saved_dir_name).resolve()
        try:
            effective_saved_base.relative_to(effective_base_path)
        except ValueError:
            effective_saved_base = (effective_base_path / "saved").resolve()

        # Every tool task runs in its own context copy, marked as a tool
        # context: code there (and threads that copy it) has an authoritative
        # speaker contextvar (speaker.identity_lost()).
        tool_context = contextvars.copy_context()
        tool_context.run(_speaker.mark_tool_context)

        # Shared-agent member: only the principal's allowlist, narrowed per
        # tool, paths confined. Checked before anything else (classic loop,
        # run_tool-style paths and Mrav all land here).
        speaker_call = self._is_speaker_call(session_id, arguments)
        if speaker_call:
            agent = (arguments or {}).get("_agent")
            bound = _speaker.current()
            principal = bound or _speaker.principal_for(agent) or _speaker.UNKNOWN_PRINCIPAL
            if name not in _speaker.allowed_tools(principal):
                raise ToolBlockedError(
                    name,
                    _speaker.FILES_UNAVAILABLE_MESSAGE
                    if name in _speaker.SPEAKER_FILE_TOOLS else _speaker.NOT_ALLOWED_MESSAGE,
                )
            arguments, rule_error = _speaker.apply_tool_rules(name, arguments, agent)
            if rule_error:
                raise ToolBlockedError(name, rule_error)
            # The turn's grant: the contextvar when the principal is bound,
            # else the one recorded on the member's own instance.
            if bound is not None:
                grant = _speaker.current_grant()
            elif _speaker.is_speaker_agent(agent):
                grant = _speaker.sanitize_grant(getattr(agent, "_turn_grant", ""))
            else:
                grant = ""
            if (name in (_speaker.SPEAKER_GOOGLE_TOOLS | _speaker.SPEAKER_DEEP_MEMORY_TOOLS)
                    and not grant):
                raise ToolBlockedError(name, _speaker.NO_GRANT_MESSAGE)
            if bound is None:
                # Recognised by the session key or `_agent` alone: bind the
                # principal (and its grant) for the tool task, so the tool's
                # own member rules apply on every signal.
                tool_context.run(_speaker.bind, principal)
                tool_context.run(_speaker.bind_grant, grant)

        # Shared folders (context packs, vfs:@alias/…), owner and member calls
        # alike: refuse them outside read-class arguments, resolve the named
        # aliases through Flight Deck and leave the roots in the tool context
        # (vfs.py and the file tools read them there). No vfs:@ value → no
        # HTTP, no table.
        pack_error = await _pack_access.prepare_call(
            name, arguments or {}, (arguments or {}).get("_agent"), tool_context)
        if pack_error:
            raise ToolBlockedError(name, pack_error)

        # PR D (J15): an owner turn that read members' private data can't
        # write the stores every member reads; after reading their
        # conversations it runs only read-only local tools until the owner's
        # next message. Member calls are never checked (their instances never
        # get a level); a call without `_agent` has no level.
        if not speaker_call:
            from captain_claw import member_privacy as _member_privacy

            privacy_error = _member_privacy.tool_block(
                name, arguments or {}, (arguments or {}).get("_agent"))
            if privacy_error:
                raise ToolBlockedError(name, privacy_error)

            # An onboarding exam's question only reads: nothing is sent or changed.
            from captain_claw import exam_turn as _exam_turn

            exam_error = _exam_turn.tool_block(name, arguments or {}, (arguments or {}).get("_agent"))
            if exam_error:
                raise ToolBlockedError(name, exam_error)

        # A multi-user deck (deck_confine): an owner call's paths stay in this
        # agent's own roots, and a non-admin's agent runs no code on the host.
        # Inside the tool context, after prepare_call, so shared folders resolve.
        if not speaker_call and _deck_confine.active():
            if _deck_confine.exec_blocked(name):
                raise ToolBlockedError(name, _deck_confine.EXEC_REFUSED)
            if _deck_confine.hosting_app_blocked(name, arguments):
                raise ToolBlockedError(name, _deck_confine.HOSTING_APP_REFUSED)
            confine_error = tool_context.run(
                _deck_confine.check_tool_paths, name, arguments, effective_base_path)
            if confine_error:
                raise ToolBlockedError(name, confine_error)

        if speaker_call:
            roots = tool_context.run(
                _speaker.speaker_roots, agent, principal, session_id=session_id,
                runtime_base=effective_base_path, saved_base=effective_saved_base,
            )
            arguments, path_error = tool_context.run(
                _speaker.check_tool_paths, name, arguments, roots,
            )
            if path_error:
                raise ToolBlockedError(name, path_error)
            # Disables read/edit/glob's workflow-dir and registry fallbacks.
            file_registry = None

        # Check if tool exists
        tool = self.get(name)

        # The chain inside the tool context: the speaker step and the Google
        # gate see the resolved principal, never None (A1) or the owner.
        allowed_names = tool_context.run(lambda: {
            _normalize_tool_name(item)
            for item in self.list_tools(
                session_id=session_id,
                session_policy=session_policy,
                task_policy=task_policy,
            )
        })
        if _normalize_tool_name(name) not in allowed_names:
            raise ToolBlockedError(name, "Blocked by tool policy chain")

        # Check if tool is blocked
        config = get_config()

        # Bat hard floor (Invariant A): an un-widenable deny of system/drive
        # destruction (catastrophic rm, disk format/erase, raw-device write,
        # drive unmount, host power-off, fork bomb). Active ONLY for Bat workers
        # (CLAW_BAT_WORKER); no config knob and no Bat relax flag can widen it.
        # Placed at this universal chokepoint so it covers EVERY exec-capable
        # tool (shell, terminal, desktop_action, ...), not just `shell`. A no-op
        # for every non-Bat agent (active() is False) and for non-exec tools.
        from captain_claw import bat_floor as _bat_floor

        if _bat_floor.active():
            _bf_blocked, _bf_reason = _bat_floor.screen(name, arguments)
            if _bf_blocked:
                raise ToolBlockedError(name, _bat_floor.refusal(name, _bf_reason))

        if name == "shell":
            command = str(arguments.get("command", "")).strip()
            policy_decision, policy_reason = self._evaluate_shell_exec_policy(command)
            if policy_decision == "deny":
                raise ToolBlockedError(name, policy_reason)
            if policy_decision == "ask":
                question = (
                    "Allow shell command execution?\n"
                    f"Command: {command}\n"
                    f"Reason: {policy_reason}"
                )
                if callable(effective_approval):
                    approved = bool(effective_approval(question))
                    if not approved:
                        raise ToolBlockedError(name, "Blocked by shell execution approval policy")
                else:
                    log.warning(
                        "Shell execution policy requires approval but no callback is configured; allowing",
                        command=command,
                    )

        # Check shell-specific blocked commands
        if name == "shell" and hasattr(config.tools.shell, "blocked"):
            blocked_cmds = config.tools.shell.blocked or []
            cmd = str(arguments.get("command", ""))
            blocked, matched = is_blocked_shell_command(cmd, blocked_cmds)
            if blocked:
                if matched == "empty_command":
                    raise ToolBlockedError(name, "Command is empty")
                if matched == "unparseable_command":
                    raise ToolBlockedError(name, "Command is not parseable")
                raise ToolBlockedError(name, f"Command matches blocked pattern: {matched}")

        # Check if confirmation required
        if confirm and name in config.tools.require_confirmation:
            # For now, just proceed - UI should handle confirmation
            pass

        # Validate arguments
        tool.validate_arguments(arguments)

        # Execute with timeout / abort propagation
        execute_task: asyncio.Task[ToolResult] | None = None
        abort_wait_task: asyncio.Task[bool] | None = None
        bridge_task: asyncio.Task[None] | None = None
        tool_abort_event = asyncio.Event()
        try:
            log.info("Executing tool", tool=name, args=_loggable_args(arguments))
            timeout_seconds = float(getattr(tool, "timeout_seconds", 30.0) or 30.0)
            timeout_override = arguments.get("timeout")
            if timeout_override is not None:
                try:
                    timeout_seconds = float(timeout_override)
                except Exception:
                    pass
            timeout_seconds = max(1.0, timeout_seconds)

            if abort_event is not None:
                bridge_task = asyncio.create_task(
                    self._bridge_abort_event(abort_event, tool_abort_event)
                )

            # Extract workflow_started_at from the shared file registry
            # so tools like glob can filter to files created during this run.
            _workflow_started_at = None
            _workflow_run_dir = None
            if file_registry is not None:
                _workflow_started_at = getattr(file_registry, "workflow_started_at", None)
                _workflow_run_dir = getattr(file_registry, "workflow_run_dir", None)

            _task_kwargs: dict[str, Any] = {"context": tool_context}
            execute_task = asyncio.create_task(
                tool.execute(
                    **arguments,
                    _runtime_base_path=effective_base_path,
                    _saved_base_path=effective_saved_base,
                    _session_id=(session_id or "").strip(),
                    _abort_event=tool_abort_event,
                    _file_registry=file_registry,
                    _workflow_started_at=_workflow_started_at,
                    _workflow_run_dir=_workflow_run_dir,
                    _stream_callback=stream_callback,
                    _approval_callback=effective_approval,
                ),
                **_task_kwargs,
            )
            abort_wait_task = asyncio.create_task(tool_abort_event.wait())
            done, _ = await asyncio.wait(
                {execute_task, abort_wait_task},
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if execute_task in done:
                result = await execute_task
                if not isinstance(result, ToolResult):
                    raise ToolExecutionError(name, "Tool returned invalid result payload")
                log.info("Tool executed", tool=name, success=result.success)
                return result

            if abort_wait_task in done:
                await self._cancel_task(execute_task)
                raise ToolExecutionError(name, _ABORTED_MARK)

            tool_abort_event.set()
            await self._cancel_task(execute_task)
            timeout_label = int(timeout_seconds) if timeout_seconds.is_integer() else timeout_seconds
            raise ToolExecutionError(name, f"{_TIMED_OUT_MARK} {timeout_label}s")
        except asyncio.CancelledError:
            tool_abort_event.set()
            await self._cancel_task(execute_task)
            raise
        except ToolExecutionError:
            raise
        except Exception as e:
            log.error("Tool execution failed", tool=name, error=str(e))
            raise ToolExecutionError(name, str(e))
        finally:
            await self._cancel_task(abort_wait_task)
            await self._cancel_task(bridge_task)

    def _evaluate_shell_exec_policy(self, command: str) -> tuple[str, str]:
        """Evaluate allow/deny/ask policy for a shell command."""
        cleaned = str(command or "").strip()
        if not cleaned:
            return "deny", "Command is empty"

        cfg = get_config()
        shell_cfg = cfg.tools.shell
        deny_patterns = [str(item).strip() for item in getattr(shell_cfg, "deny_patterns", []) if str(item).strip()]
        allow_patterns = [str(item).strip() for item in getattr(shell_cfg, "allow_patterns", []) if str(item).strip()]
        default_policy = str(getattr(shell_cfg, "default_policy", "ask") or "ask").strip().lower()

        for pattern in deny_patterns:
            if self._shell_pattern_matches(cleaned, pattern):
                return "deny", f"Command matches deny pattern: {pattern}"

        for pattern in allow_patterns:
            if self._shell_pattern_matches(cleaned, pattern):
                return "allow", f"Command matches allow pattern: {pattern}"

        if default_policy == "allow":
            return "allow", "Default shell execution policy allows command"
        if default_policy == "deny":
            return "deny", "Default shell execution policy denies command"
        return "ask", "Default shell execution policy requires approval"

    @staticmethod
    def _shell_pattern_matches(command: str, pattern: str) -> bool:
        """Match shell command against glob-like policy pattern."""
        cleaned_command = str(command or "").strip()
        cleaned_pattern = str(pattern or "").strip()
        if not cleaned_command or not cleaned_pattern:
            return False

        lowered_pattern = cleaned_pattern.lower()
        targets: list[str] = [cleaned_command.lower()]
        try:
            segments = _split_shell_segments(cleaned_command)
            targets.extend(" ".join(segment).strip().lower() for segment in segments if segment)
        except Exception:
            pass
        base_commands = [item.lower() for item in extract_shell_base_commands(cleaned_command)]
        targets.extend(base_commands)
        targets.extend(Path(item).name.lower() for item in base_commands if item)

        return any(fnmatch.fnmatchcase(target, lowered_pattern) for target in targets if target)


# Global registry
_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def set_tool_registry(registry: ToolRegistry) -> None:
    """Set the global tool registry."""
    global _registry
    _registry = registry
