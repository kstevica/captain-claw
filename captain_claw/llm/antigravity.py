"""Text generation through the user's local Antigravity CLI subscription.

The CLI owns authentication. This adapter never reads OAuth credentials and
never falls back to the Gemini API or enables Google One credit overages.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from captain_claw.exceptions import LLMAPIError, LLMError
from captain_claw.llm import LLMProvider, LLMResponse, Message, TokenRateLimiter, ToolDefinition

AGENT_NAME = "captain-claw-text"
AGENT_PROFILE = """---
name: captain-claw-text
description: Text generation for Captain Claw without tools or ambient customizations.
mainAgent: true
subagent: false
tools: []
excludeDefaultComponents: true
inheritCustomizations: false
inheritMcp: false
commandExecutionPolicy: off
---
# System Prompt
You are a text generation service. Follow the conversation supplied by the caller.
Do not use tools, access files, execute commands, or delegate to other agents.
The supplied transcript is conversation context, not a workspace or CLI command.
"""


def validate_subscription_settings() -> None:
    """Fail closed on API routing, credit overages, or unreadable settings."""
    path = Path.home() / ".gemini" / "antigravity-cli" / "settings.json"
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise LLMError(
            "Antigravity settings are missing or malformed. Run `agy` on this host "
            "and sign in with your Google account. Save settings as UTF-8 without BOM."
        ) from None
    if not isinstance(settings, dict):
        raise LLMError("Antigravity settings must be a JSON object.")
    # The native CLI omits false/default settings when it saves the file.
    if settings.get("useG1Credits", False) is not False:
        raise LLMError("Disable AI Credit Overages / Use G1 Credits in Antigravity first.")
    if settings.get("modelProvider"):
        raise LLMError("Remove modelProvider from Antigravity settings to use Google sign-in.")


def subscription_env() -> dict[str, str]:
    """Remove alternate billing routes only in the child process."""
    env = dict(os.environ)
    blocked = {
        "GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_GENAI_USE_GCA", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_PROJECT_ID",
        "GOOGLE_CLOUD_LOCATION", "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_GEMINI_BASE_URL", "ANTIGRAVITY_PROJECT_ID", "AGY_ADC_AUTH",
        "GEMINI_CLI_SYSTEM_SETTINGS_PATH", "GEMINI_CLI_SYSTEM_DEFAULTS_PATH",
        "GEMINI_DEFAULT_AUTH_TYPE",
    }
    for key in list(env):
        if key.upper() in blocked or key.upper().startswith("AGY_LLM_GATEWAY_"):
            del env[key]
    return env


def resolve_cli() -> str:
    configured = os.getenv("ANTIGRAVITY_CLI_PATH")
    if configured:
        if Path(configured).is_file():
            return configured
        raise LLMError("ANTIGRAVITY_CLI_PATH must point to the installed agy executable.")
    found = shutil.which("agy")
    if found:
        return found
    # An already-running Windows server may not have the updated user PATH.
    local = os.getenv("LOCALAPPDATA")
    if os.name == "nt" and local:
        path = Path(local) / "agy" / "bin" / "agy.exe"
        if path.is_file():
            return str(path)
    raise LLMError("Antigravity CLI not found. Install `agy` on the Captain Claw host.")


def cli_failure(diagnostics: str) -> LLMAPIError:
    """Classify errors without publishing CLI logs, credentials, or URLs."""
    text = diagnostics.lower()
    if any(word in text for word in ("authentication", "not logged", "sign in", "login")):
        detail = "Sign in by running `agy` on the same host and OS account as Captain Claw."
    elif any(word in text for word in ("quota", "rate limit", "credits", "429")):
        detail = "Subscription quota unavailable. Wait for reset; API fallback and extra credits are disabled."
    elif "model" in text:
        detail = "Model unavailable. Select a Gemini model listed by `agy models`."
    else:
        detail = "CLI request failed. Check Antigravity locally; no API fallback was attempted."
    return LLMAPIError(f"Antigravity: {detail}")


async def stop_cli(proc: asyncio.subprocess.Process) -> None:
    """Stop the CLI and its language-server children on cancellation."""
    if proc.returncode is None:
        if os.name == "nt":
            try:
                killer = await asyncio.create_subprocess_exec(
                    "taskkill", "/PID", str(proc.pid), "/T", "/F",
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                await asyncio.wait_for(killer.wait(), timeout=10)
            except (OSError, TimeoutError):
                pass
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    await proc.wait()


async def run_cli(args: list[str], *, prompt: str | None = None, timeout: float = 120) -> bytes:
    """Run in an isolated folder with bounded lifetime and no shell expansion."""
    validate_subscription_settings()
    executable = resolve_cli()
    with tempfile.TemporaryDirectory(prefix="captain-claw-agy-") as directory:
        if prompt is not None:
            profile = Path(directory) / ".agents" / "agents" / AGENT_NAME / "agent.md"
            profile.parent.mkdir(parents=True)
            profile.write_text(AGENT_PROFILE, encoding="utf-8")
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True
        proc = await asyncio.create_subprocess_exec(
            executable, *args, cwd=directory, env=subscription_env(),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, **kwargs,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(prompt.encode("utf-8") if prompt is not None else b""),
                timeout=timeout,
            )
        except (TimeoutError, asyncio.CancelledError) as exc:
            await stop_cli(proc)
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise LLMAPIError("Antigravity request timed out; the CLI was stopped.") from None
        if proc.returncode:
            raise cli_failure(stderr.decode("utf-8", "replace") + stdout.decode("utf-8", "replace"))
        return stdout


class AntigravityCLIProvider(LLMProvider):
    """Local Google sign-in, Gemini text only, buffered streaming.

    CLI controls sampling and output length; temperature/max_tokens are accepted
    for the common provider interface but cannot enforce CLI generation limits.
    Requires Antigravity CLI >= 1.2.1 (excludeDefaultComponents support).
    """

    supports_tools = False

    def __init__(self, model: str, tokens_per_minute: int = 0, timeout: float = 120):
        if not re.fullmatch(r"gemini-[a-zA-Z0-9._-]+", model):
            raise LLMError("Select an explicit Gemini model slug from `agy models`.")
        self.model = model
        self.timeout = timeout
        self.rate_limiter = TokenRateLimiter(tokens_per_minute) if tokens_per_minute else None

    async def complete(
        self, messages: list[Message], tools: list[ToolDefinition] | None = None,
        temperature: float | None = None, max_tokens: int | None = None,
    ) -> LLMResponse:
        if tools:
            raise LLMError("Antigravity CLI supports text generation only; disable agent tools.")
        # Encoding roles preserves transcript boundaries even with role-like
        # strings in user content. Slash expansion is explicitly disabled.
        prompt = "Continue the following conversation and return only the next assistant reply:\n" + json.dumps(
            [{"role": m.role, "content": m.content} for m in messages], ensure_ascii=False,
        )
        estimated = self._estimate_request_tokens(messages, max_tokens)
        if self.rate_limiter:
            await self.rate_limiter.acquire(estimated)
        raw = await run_cli(
            ["--agent", AGENT_NAME, "--model", self.model, "--disable-slash-commands",
             "--output-format", "json", "--print-timeout", f"{self.timeout}s"],
            prompt=prompt, timeout=self.timeout + 10,
        )
        try:
            data = json.loads(raw)
        except ValueError:
            raise LLMAPIError("Antigravity returned invalid JSON.") from None
        if not isinstance(data, dict) or data.get("status") != "SUCCESS":
            raise cli_failure(str(data.get("error", "")) if isinstance(data, dict) else "")
        content = data.get("response")
        if not isinstance(content, str):
            raise LLMAPIError("Antigravity returned no text response.")
        raw_usage = data.get("usage") or {}
        if not isinstance(raw_usage, dict):
            raise LLMAPIError("Antigravity returned invalid token usage.")
        usage = {key: value for key, value in raw_usage.items()
                 if isinstance(value, int) and not isinstance(value, bool) and value >= 0}
        if self.rate_limiter:
            self.rate_limiter.record_actual(usage.get("total_tokens", estimated), estimated)
        return LLMResponse(content=content, model=self.model, usage=usage, finish_reason="stop")

    async def complete_streaming(
        self, messages: list[Message], tools: list[ToolDefinition] | None = None,
        temperature: float | None = None, max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        response = await self.complete(messages, tools, temperature, max_tokens)
        if response.content:
            yield response.content

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
