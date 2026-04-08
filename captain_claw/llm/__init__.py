"""LLM providers (Ollama + LiteLLM-backed providers)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

import httpx

from captain_claw.exceptions import LLMAPIError, LLMError
from captain_claw.logging import get_logger

log = get_logger(__name__)


OLLAMA_NATIVE_BASE_URL = "http://127.0.0.1:11434"


@dataclass
class Message:
    """A message in the conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class ToolCall:
    """A tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from the LLM."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""


@dataclass
class ToolDefinition:
    """Definition of a tool for the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


class TokenRateLimiter:
    """Sliding-window token rate limiter (tokens per minute).

    Tracks token consumption over a rolling 60-second window and blocks
    callers via ``acquire()`` until capacity is available.  After the API
    call completes, ``record_actual()`` corrects the estimate with the
    real usage reported by the provider.
    """

    def __init__(self, tokens_per_minute: int) -> None:
        self.tpm = max(0, tokens_per_minute)
        self._lock = asyncio.Lock()
        # Each entry: (monotonic_timestamp, token_count)
        self._log: deque[tuple[float, int]] = deque()

    @property
    def enabled(self) -> bool:
        return self.tpm > 0

    def _purge_old(self, now: float) -> int:
        """Remove entries older than 60 s and return current window total."""
        cutoff = now - 60.0
        while self._log and self._log[0][0] < cutoff:
            self._log.popleft()
        return sum(t for _, t in self._log)

    async def acquire(self, estimated_tokens: int) -> None:
        """Wait until the window has room for *estimated_tokens*."""
        if not self.enabled:
            return
        while True:
            async with self._lock:
                now = time.monotonic()
                used = self._purge_old(now)
                if used + estimated_tokens <= self.tpm:
                    self._log.append((now, estimated_tokens))
                    return
                # Calculate how long until enough capacity frees up.
                deficit = (used + estimated_tokens) - self.tpm
                wait = 0.0
                freed = 0
                for ts, tok in self._log:
                    freed += tok
                    if freed >= deficit:
                        wait = (ts + 60.0) - now
                        break
            wait = max(0.1, wait)
            log.info(
                "rate-limiter: waiting %.1fs (used %d/%d TPM)",
                wait,
                used,
                self.tpm,
            )
            await asyncio.sleep(wait)

    def record_actual(self, actual_tokens: int, estimated_tokens: int) -> None:
        """Correct the most-recent log entry with real usage."""
        diff = actual_tokens - estimated_tokens
        if diff == 0 or not self._log:
            return
        ts, tok = self._log[-1]
        self._log[-1] = (ts, max(0, tok + diff))


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    rate_limiter: TokenRateLimiter | None = None

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        pass

    @abstractmethod
    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        pass

    async def complete_with_callback(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Complete with optional streaming callback.

        Streams text chunks to *on_chunk* in real time while still returning
        the full ``LLMResponse`` (including tool_calls and usage).
        Default implementation delegates to ``complete()`` and sends the
        full content as one chunk.  Subclasses may override with true
        token-level streaming.
        """
        response = await self.complete(messages, tools, temperature, max_tokens)
        if on_chunk and response.content:
            try:
                on_chunk(response.content)
            except Exception:
                pass
        return response

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

    def _estimate_request_tokens(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
    ) -> int:
        """Rough estimate of total tokens a request will consume."""
        prompt_tokens = 0
        for msg in messages:
            text = msg.content if isinstance(msg, Message) else str(msg.get("content", ""))
            prompt_tokens += self.count_tokens(text) if text else 0
        # Add a conservative estimate for the completion side.
        completion_budget = max_tokens or 4096
        return prompt_tokens + completion_budget


def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read key from object or dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _safe_json_loads(raw: str) -> dict[str, Any]:
    """Parse JSON dict safely; return empty dict on errors."""
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
        return {"value": value}
    except Exception:
        return {"raw": raw}


# Matches <think>...</think> blocks (and common variants like <thinking>,
# <reasoning>) emitted by some models as raw text in the content field
# instead of in a separate reasoning_content field. We strip these so the
# user-visible chat content never contains the model's private chain of
# thought.
_THINK_BLOCK_RE = re.compile(
    r"<\s*(think|thinking|reasoning|reflection)\s*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
# Matches an unterminated leading think/reasoning block (model started a
# <think> tag but never closed it before the answer began).
_THINK_OPEN_RE = re.compile(
    r"^\s*<\s*(think|thinking|reasoning|reflection)\s*>.*?(?=<\s*/\s*\1\s*>|$)",
    re.IGNORECASE | re.DOTALL,
)


def _strip_reasoning_artifacts(text: str) -> str:
    """Remove <think>...</think> style reasoning blocks from model output.

    Some providers (notably xAI Grok and certain DeepSeek/Qwen variants)
    leak their internal reasoning into the ``content`` field rather than
    keeping it in a separate ``reasoning_content`` field. This helper
    strips those blocks so they don't end up in the chat UI.
    """
    if not text:
        return text
    # Drop fully closed <think>...</think> blocks anywhere in the text.
    cleaned = _THINK_BLOCK_RE.sub("", text)
    # Drop a leading unterminated <think> block (if any).
    cleaned = _THINK_OPEN_RE.sub("", cleaned)
    return cleaned.lstrip()


def _normalize_provider_name(provider: str) -> str:
    """Normalize provider aliases."""
    key = (provider or "").strip().lower()
    aliases = {
        "chatgpt": "openai",
        "openai": "openai",
        "claude": "anthropic",
        "anthropic": "anthropic",
        "gemini": "gemini",
        "google": "gemini",
        "googleai": "gemini",
        "grok": "xai",
        "xai": "xai",
        "ollama": "ollama",
        "openrouter": "openrouter",
        "litert": "litert",
        "litert-lm": "litert",
        "litertlm": "litert",
        "gemma-local": "litert",
    }
    return aliases.get(key, key)


def _provider_model_name(provider: str, model: str) -> str:
    """Ensure model name includes provider prefix for LiteLLM."""
    cleaned = (model or "").strip()
    if not cleaned:
        return cleaned
    # OpenRouter model IDs already contain a slash (e.g. nvidia/model-name),
    # but LiteLLM still needs the openrouter/ prefix to route correctly.
    if provider == "openrouter":
        if cleaned.startswith("openrouter/"):
            return cleaned
        return f"openrouter/{cleaned}"
    if "/" in cleaned:
        return cleaned
    return f"{provider}/{cleaned}"


def _base_model_name(model: str) -> str:
    """Return provider-agnostic model name (without provider prefix)."""
    cleaned = (model or "").strip()
    if "/" in cleaned:
        return cleaned.split("/", 1)[1]
    return cleaned


def _is_openai_gpt5_family(provider: str, model: str) -> bool:
    """Whether model is in OpenAI GPT-5 family."""
    normalized_provider = _normalize_provider_name(provider)
    if normalized_provider != "openai":
        return False
    base = _base_model_name(model).lower()
    return base.startswith("gpt-5")


def _normalize_temperature_for_model(provider: str, model: str, temperature: float | None) -> float | None:
    """Adjust temperature for provider/model constraints."""
    if temperature is None:
        return None
    # OpenAI GPT-5 family only accepts temperature=1.
    if _is_openai_gpt5_family(provider, model):
        return 1.0
    return temperature


def _resolve_api_key(provider: str, explicit_api_key: str | None) -> str | None:
    """Resolve provider API key from explicit value, environment, or provider_keys."""
    if explicit_api_key:
        return explicit_api_key
    if provider == "openai":
        val = os.getenv("OPENAI_API_KEY") or None
        if val:
            return val
    elif provider == "anthropic":
        val = os.getenv("ANTHROPIC_API_KEY") or None
        if val:
            return val
    elif provider == "gemini":
        val = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or None
        if val:
            return val
    elif provider == "xai":
        val = os.getenv("XAI_API_KEY") or None
        if val:
            return val
    elif provider == "openrouter":
        val = os.getenv("OPENROUTER_API_KEY") or None
        if val:
            return val
    # Fallback: provider_keys from config (settings UI).
    try:
        from captain_claw.config import get_config
        pk = get_config().provider_keys
        pk_map = {"openai": pk.openai, "anthropic": pk.anthropic, "gemini": pk.gemini, "xai": pk.xai, "openrouter": pk.openrouter}
        pk_val = str(pk_map.get(provider, "") or "").strip()
        if pk_val:
            return pk_val
    except Exception:
        pass
    return None


_CACHE_SPLIT_MARKER = "<!-- CACHE_SPLIT -->"


def _inject_anthropic_cache_control(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add ``cache_control`` to system and history messages for Anthropic prompt caching.

    Anthropic caches the prompt prefix up to each ``cache_control`` breakpoint.
    We use up to 2 breakpoints (out of 4 allowed):

    1. **System message (static part)** — the large instruction text that is
       identical across turns.  The system prompt template contains a
       ``<!-- CACHE_SPLIT -->`` marker that separates static instructions
       (above) from dynamic context (below, e.g. timestamp, file trees).
       Only the static block gets ``cache_control``; the dynamic block is
       sent as a separate content block without it so that changes to
       timestamps/env info don't bust the cache.

    2. **Last user or assistant message before the current turn** — during a
       tool-use loop the same conversation prefix is sent multiple times.
       Marking the last historical message lets Anthropic cache the entire
       prefix (system + history) across tool-loop iterations within a turn.

    LiteLLM passes ``cache_control`` through to Anthropic when the message
    content is a list of content blocks (not a plain string).
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                if _CACHE_SPLIT_MARKER in content:
                    # Split into static (cached) and dynamic (uncached) blocks.
                    static_part, dynamic_part = content.split(_CACHE_SPLIT_MARKER, 1)
                    static_part = static_part.rstrip()
                    dynamic_part = dynamic_part.strip()
                    blocks: list[dict[str, Any]] = [
                        {
                            "type": "text",
                            "text": static_part,
                            "cache_control": {"type": "ephemeral"},
                        },
                    ]
                    if dynamic_part:
                        blocks.append({"type": "text", "text": dynamic_part})
                    msg = {**msg, "content": blocks}
                else:
                    # No marker — cache the whole thing.
                    msg = {**msg, "content": [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        },
                    ]}
            elif isinstance(content, list):
                blocks = [dict(b) if isinstance(b, dict) else b for b in content]
                if blocks and isinstance(blocks[-1], dict):
                    blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
                msg = {**msg, "content": blocks}
        result.append(msg)

    # Breakpoint 2: mark the last user or assistant message in the
    # conversation history so tool-loop iterations cache the prefix.
    # Walk backwards to find the last user/assistant message.
    for i in range(len(result) - 1, -1, -1):
        role = result[i].get("role", "")
        if role in ("user", "assistant"):
            content = result[i].get("content", "")
            if isinstance(content, str) and content:
                result[i] = {**result[i], "content": [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    },
                ]}
            elif isinstance(content, list):
                blocks = [dict(b) if isinstance(b, dict) else b for b in content]
                if blocks and isinstance(blocks[-1], dict):
                    blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
                result[i] = {**result[i], "content": blocks}
            break

    return result


def _convert_messages_for_openai_style(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert messages to OpenAI-style payload."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = str(msg.get("role", ""))
            content = str(msg.get("content", ""))
            tool_call_id = msg.get("tool_call_id")
            tool_name = msg.get("tool_name")
            tool_calls = msg.get("tool_calls")
        else:
            role = str(getattr(msg, "role", ""))
            content = str(getattr(msg, "content", ""))
            tool_call_id = getattr(msg, "tool_call_id", None)
            tool_name = getattr(msg, "tool_name", None)
            tool_calls = getattr(msg, "tool_calls", None)

        if role not in {"system", "user", "assistant", "tool"}:
            continue
        # Anthropic rejects empty text content blocks.
        if not content and role == "tool":
            content = "[empty tool response]"
        elif not content and role not in {"assistant"}:
            content = " "
        entry: dict[str, Any] = {"role": role, "content": content}
        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            normalized_calls: list[dict[str, Any]] = []
            for idx, raw in enumerate(tool_calls, start=1):
                if not isinstance(raw, dict):
                    continue
                call_id = str(_obj_get(raw, "id", "") or f"call_{idx}")
                call_type = str(_obj_get(raw, "type", "") or "function")
                function_obj = _obj_get(raw, "function", None)
                if isinstance(function_obj, dict):
                    fn_name = str(_obj_get(function_obj, "name", "") or "")
                    fn_args = _obj_get(function_obj, "arguments", "")
                else:
                    fn_name = str(_obj_get(raw, "name", "") or "")
                    fn_args = _obj_get(raw, "arguments", {})
                if not fn_name:
                    continue
                if isinstance(fn_args, str):
                    fn_args_text = fn_args
                elif isinstance(fn_args, dict):
                    fn_args_text = json.dumps(fn_args, ensure_ascii=True)
                else:
                    fn_args_text = "{}"
                normalized_calls.append({
                    "id": call_id,
                    "type": call_type,
                    "function": {
                        "name": fn_name,
                        "arguments": fn_args_text,
                    },
                })
            if normalized_calls:
                entry["tool_calls"] = normalized_calls
        if role == "tool":
            # tool_call_id is required by many providers (OpenRouter, OpenAI, etc.)
            entry["tool_call_id"] = tool_call_id or f"call_{len(result)}"
        if role == "tool" and tool_name:
            entry["name"] = tool_name
        result.append(entry)
    return result


def _convert_messages_for_ollama(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert messages to Ollama API format."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = str(msg.get("role", ""))
            content = str(msg.get("content", ""))
            tool_name = msg.get("tool_name")
        else:
            role = str(getattr(msg, "role", ""))
            content = str(getattr(msg, "content", ""))
            tool_name = getattr(msg, "tool_name", None)

        if role == "tool":
            entry = {"role": "tool", "content": content}
            if tool_name:
                entry["tool_name"] = str(tool_name)
            result.append(entry)
            continue

        if role in {"system", "user", "assistant"}:
            result.append({"role": role, "content": content})
    return result


def _convert_tools_for_openai_style(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert tool schema to OpenAI function-tool format."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})
        else:
            name = getattr(tool, "name", None)
            description = getattr(tool, "description", "") or ""
            parameters = getattr(tool, "parameters", None) or {}

        if not name:
            continue
        result.append({
            "type": "function",
            "function": {
                "name": str(name),
                "description": str(description),
                "parameters": parameters if isinstance(parameters, dict) else {},
            },
        })
    return result


def _extract_usage(usage_obj: Any) -> dict[str, int]:
    """Extract usage token counts from provider response usage object."""
    if usage_obj is None:
        return {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        }

    prompt_tokens = int(_obj_get(usage_obj, "prompt_tokens", 0) or 0)
    completion_tokens = int(_obj_get(usage_obj, "completion_tokens", 0) or 0)
    total_tokens = int(_obj_get(usage_obj, "total_tokens", 0) or 0)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    # Anthropic cache tokens (passed through by LiteLLM)
    cache_creation = int(_obj_get(usage_obj, "cache_creation_input_tokens", 0) or 0)
    cache_read = int(_obj_get(usage_obj, "cache_read_input_tokens", 0) or 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
    }


# ── ChatGPT Responses API helpers ─────────────────────────────────────────


# Models the ChatGPT "Sign in with ChatGPT" / Codex backend actually
# accepts. The endpoint at chatgpt.com/backend-api/codex/responses is
# *not* the same as api.openai.com/v1 — it only serves Codex-family
# models tied to a ChatGPT plan. Anything else either 400s or, worse,
# returns an empty body that looks like a successful but blank response.
_CODEX_BACKEND_SUPPORTED_MODELS: frozenset[str] = frozenset({
    "gpt-5",
    "gpt-5-codex",
    "gpt-5.1",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.1-codex-max",
    "gpt-5.2",
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "codex-mini-latest",
})


def _normalize_chatgpt_model(name: str) -> str:
    """Normalize ChatGPT / Codex model name aliases."""
    base = (name or "").strip()
    if not base:
        return "gpt-5"
    # Strip effort suffixes (e.g. gpt-5-high → gpt-5).
    for sep in ("-", "_"):
        lowered = base.lower()
        for effort in ("minimal", "low", "medium", "high", "xhigh"):
            suffix = f"{sep}{effort}"
            if lowered.endswith(suffix):
                base = base[: -len(suffix)]
                break
    mapping: dict[str, str] = {
        "gpt5": "gpt-5",
        "gpt-5-latest": "gpt-5",
        "gpt-5": "gpt-5",
        "gpt-5.1": "gpt-5.1",
        "gpt5.2": "gpt-5.2",
        "gpt-5.2": "gpt-5.2",
        "gpt-5.2-latest": "gpt-5.2",
        "gpt5.3-codex": "gpt-5.3-codex",
        "gpt-5.3-codex": "gpt-5.3-codex",
        "gpt-5.3-codex-latest": "gpt-5.3-codex",
        "gpt5.2-codex": "gpt-5.2-codex",
        "gpt-5.2-codex": "gpt-5.2-codex",
        "gpt-5.2-codex-latest": "gpt-5.2-codex",
        "gpt5-codex": "gpt-5-codex",
        "gpt-5-codex": "gpt-5-codex",
        "gpt-5-codex-latest": "gpt-5-codex",
        "gpt-5.1-codex": "gpt-5.1-codex",
        "gpt-5.1-codex-max": "gpt-5.1-codex-max",
        "codex": "codex-mini-latest",
        "codex-mini": "codex-mini-latest",
        "codex-mini-latest": "codex-mini-latest",
        "gpt-5.1-codex-mini": "gpt-5.1-codex-mini",
        # Common non-codex aliases users pick from generic OpenAI model
        # lists. The Codex backend doesn't serve these, so remap them to
        # the closest Codex-family equivalent rather than 400ing.
        "gpt-5-mini": "gpt-5.1-codex-mini",
        "gpt-5.1-mini": "gpt-5.1-codex-mini",
        "gpt-5.2-mini": "gpt-5.1-codex-mini",
        "gpt-5.3-mini": "gpt-5.1-codex-mini",
        "gpt-5.4-mini": "gpt-5.1-codex-mini",
    }
    return mapping.get(base.lower(), base)


def _is_codex_family_model(name: str) -> bool:
    """Return True if *name* is a GPT-5 / Codex family model that can
    only be served by the ChatGPT Responses endpoint (never by the
    regular ``api.openai.com/v1`` chat completions API)."""
    normalized = _normalize_chatgpt_model(name)
    if not normalized:
        return False
    lowered = normalized.lower()
    if "codex" in lowered:
        return True
    if lowered.startswith("gpt-5"):
        return True
    return False


def _convert_messages_for_responses_api(
    messages: list[Message],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert internal messages to ChatGPT Responses API format.

    Returns ``(instructions, input_items)`` where *instructions* is the
    concatenated system prompt and *input_items* is a list of Responses
    API input objects.
    """
    system_parts: list[str] = []
    items: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, dict):
            role = str(msg.get("role", ""))
            content = str(msg.get("content", ""))
            tool_call_id = msg.get("tool_call_id")
            tool_calls = msg.get("tool_calls")
        else:
            role = str(getattr(msg, "role", ""))
            content = str(getattr(msg, "content", ""))
            tool_call_id = getattr(msg, "tool_call_id", None)
            tool_calls = getattr(msg, "tool_calls", None)

        if role == "system":
            if content:
                system_parts.append(content)
            continue

        if role == "user":
            items.append({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": content}],
            })
            continue

        if role == "assistant":
            # Emit text content as a message item (if any).
            if content:
                items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                })
            # Emit each tool call as a separate function_call item.
            if isinstance(tool_calls, list):
                for idx, raw in enumerate(tool_calls, start=1):
                    if not isinstance(raw, dict):
                        continue
                    fn = raw.get("function") if isinstance(raw.get("function"), dict) else raw
                    fn_name = str(fn.get("name", "") or "")
                    fn_args = fn.get("arguments", {})
                    if isinstance(fn_args, dict):
                        fn_args = json.dumps(fn_args, ensure_ascii=False)
                    elif not isinstance(fn_args, str):
                        fn_args = "{}"
                    call_id = str(raw.get("id", "") or f"call_{idx}")
                    items.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": fn_name,
                        "arguments": fn_args,
                    })
            continue

        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": str(tool_call_id or f"call_{len(items)}"),
                "output": content,
            })
            continue

    instructions = "\n\n".join(system_parts)
    return instructions, items


def _convert_tools_for_responses_api(
    tools: list[ToolDefinition],
) -> list[dict[str, Any]]:
    """Convert tool definitions to Responses API format.

    Responses API uses a flat structure (``name`` at top level) unlike
    Chat Completions which nests under a ``function`` key.
    """
    result: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})
        else:
            name = getattr(tool, "name", None)
            description = getattr(tool, "description", "") or ""
            parameters = getattr(tool, "parameters", None) or {}
        if not name:
            continue
        result.append({
            "type": "function",
            "name": str(name),
            "description": str(description),
            "parameters": parameters if isinstance(parameters, dict) else {},
        })
    return result


class _CodexAuthExpired(Exception):
    """Internal sentinel — raised when the Responses API returns 401.

    Caught inside :class:`ChatGPTResponsesProvider` to trigger a
    one-shot forced refresh from the Codex auth manager. Never
    surfaces to callers.
    """


class ChatGPTResponsesProvider(LLMProvider):
    """Direct connection to the ChatGPT Responses API.

    Used when the OpenAI provider has ``extra_headers`` configured
    (e.g. ``Authorization``, ``chatgpt-account-id``, ``OpenAI-Beta``).
    Bypasses LiteLLM entirely and speaks the Responses API protocol
    (SSE streaming, ``input`` items instead of ``messages``).
    """

    def __init__(
        self,
        model: str = "gpt-5",
        base_url: str = "https://chatgpt.com/backend-api/codex/responses",
        extra_headers: dict[str, str] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 32000,
        tokens_per_minute: int = 0,
        use_codex_auth_manager: bool = True,
    ):
        import uuid

        self.provider = "openai"
        self.model = _normalize_chatgpt_model(model)
        if self.model != (model or "").strip():
            log.info(
                "ChatGPT Codex: remapped model alias",
                requested=model,
                using=self.model,
            )
        if self.model not in _CODEX_BACKEND_SUPPORTED_MODELS:
            log.warning(
                "ChatGPT Codex: model not in known-supported set — "
                "the chatgpt.com Codex backend will likely reject it. "
                "Pick one of: %s",
                ", ".join(sorted(_CODEX_BACKEND_SUPPORTED_MODELS)),
                extra={"requested_model": model, "normalized": self.model},
            )
        self.base_url = base_url.rstrip("/")
        self.extra_headers = dict(extra_headers or {})
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session_id = uuid.uuid4().hex
        self.client = httpx.AsyncClient(timeout=600.0, follow_redirects=True)
        self.rate_limiter = (
            TokenRateLimiter(tokens_per_minute) if tokens_per_minute > 0 else None
        )

        # Codex tokens (Authorization + chatgpt-account-id) expire roughly
        # every 24 hours and are refreshed in the background by the Codex
        # CLI (or, when running under Flight Deck, by FD re-reading
        # ``~/.codex/auth.json`` on demand). When enabled, we hand the
        # auth problem off to :class:`CodexAuthManager`, which sources
        # tokens from Flight Deck (preferred) or the local auth.json,
        # and refreshes them pre-request when stale or on any 401.
        self._codex_auth: "CodexAuthManager | None" = None
        if use_codex_auth_manager:
            try:
                from captain_claw.codex_auth_manager import CodexAuthManager
                self._codex_auth = CodexAuthManager()
            except Exception as exc:  # pragma: no cover — import safety net
                log.debug("CodexAuthManager unavailable: %s", exc)
                self._codex_auth = None

    # ── helpers ────────────────────────────────────────────────────────

    def _build_headers(self) -> dict[str, str]:
        """Merge provider extra_headers with required defaults."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "session_id": self.session_id,
        }
        headers.update(self.extra_headers)
        return headers

    async def _ensure_fresh_auth(self, *, force: bool = False) -> None:
        """Refresh ``self.extra_headers`` from the Codex auth manager.

        Called before each request (fast path: returns immediately when
        the cached token isn't stale) and again after any 401 response
        with ``force=True``.
        """
        if self._codex_auth is None:
            return
        try:
            fresh = await self._codex_auth.get_headers(force_refresh=force)
        except Exception as exc:
            log.debug("Codex auth refresh failed: %s", exc)
            return
        if not fresh:
            return
        # Replace just the auth-related headers so any caller-supplied
        # extras (e.g. OpenAI-Beta) are preserved.
        for key in ("Authorization", "chatgpt-account-id"):
            if key in fresh:
                self.extra_headers[key] = fresh[key]

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        instructions, input_items = _convert_messages_for_responses_api(messages)
        api_tools = _convert_tools_for_responses_api(tools) if tools else []
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "tools": api_tools,
            "tool_choice": "auto" if api_tools else "none",
            "parallel_tool_calls": False,
            "store": False,
            "stream": True,
            "prompt_cache_key": self.session_id,
        }
        payload["instructions"] = instructions or "Follow the user's instructions."
        return payload

    @staticmethod
    def _parse_sse_events(lines: list[str]) -> list[dict[str, Any]]:
        """Parse raw SSE lines into a list of JSON event dicts."""
        events: list[dict[str, Any]] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("event:"):
                continue
            if stripped.startswith("data: "):
                data_str = stripped[6:]
            elif stripped.startswith("data:"):
                data_str = stripped[5:]
            else:
                continue
            try:
                events.append(json.loads(data_str))
            except json.JSONDecodeError:
                continue
        return events

    def _parse_response_output(
        self,
        completed_event: dict[str, Any],
    ) -> LLMResponse:
        """Extract content and tool calls from a ``response.completed`` event."""
        response_data = completed_event.get("response") or completed_event
        output_items = response_data.get("output", []) or []

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        def _harvest_text(node: Any) -> None:
            """Recursively pull any ``text`` strings out of a Responses API
            output node. The Codex backend's shape varies between
            ``message`` items with ``content[].text``, ``output_text``
            items with a top-level ``text``, and (for reasoning models)
            nested ``content`` arrays — so we just walk everything."""
            if isinstance(node, dict):
                t = node.get("type", "")
                if isinstance(t, str) and ("output_text" in t or t == "text"):
                    text = node.get("text")
                    if isinstance(text, str) and text:
                        content_parts.append(text)
                # Some shapes nest text in {"text": {"value": "..."}}
                txt_field = node.get("text")
                if isinstance(txt_field, dict):
                    val = txt_field.get("value")
                    if isinstance(val, str) and val:
                        content_parts.append(val)
                for key in ("content", "parts", "items"):
                    child = node.get(key)
                    if child is not None:
                        _harvest_text(child)
            elif isinstance(node, list):
                for child in node:
                    _harvest_text(child)

        for item in output_items:
            item_type = item.get("type", "")

            if item_type == "message":
                _harvest_text(item.get("content", []) or [])

            elif item_type in ("output_text", "text"):
                _harvest_text(item)

            elif item_type == "function_call":
                call_id = str(item.get("call_id", "") or item.get("id", ""))
                name = str(item.get("name", ""))
                raw_args = item.get("arguments", "{}")
                args = _safe_json_loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                tool_calls.append(ToolCall(id=call_id, name=name, arguments=args))

            else:
                # Unknown item shape — try to harvest any text fields
                # rather than silently dropping the turn.
                _harvest_text(item)

        usage_raw = response_data.get("usage", {}) or {}
        usage = {
            "prompt_tokens": int(usage_raw.get("input_tokens", 0) or 0),
            "completion_tokens": int(usage_raw.get("output_tokens", 0) or 0),
            "total_tokens": int(usage_raw.get("total_tokens", 0) or 0),
        }
        if usage["total_tokens"] <= 0:
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        return LLMResponse(
            content="".join(content_parts),
            tool_calls=tool_calls,
            model=str(response_data.get("model", self.model) or self.model),
            usage=usage,
            finish_reason="tool_calls" if tool_calls else "stop",
        )

    # ── LLMProvider interface ──────────────────────────────────────────

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        estimated = 0
        if self.rate_limiter:
            estimated = self._estimate_request_tokens(messages, max_tokens or self.max_tokens)
            await self.rate_limiter.acquire(estimated)

        payload = self._build_payload(messages, tools, temperature, max_tokens)

        # Refresh from Codex auth manager pre-request (cheap no-op when
        # cached token isn't stale). On 401 we'll retry once with a
        # forced refresh.
        await self._ensure_fresh_auth()

        async def _do_request() -> list[str]:
            collected: list[str] = []
            headers = self._build_headers()
            async with self.client.stream(
                "POST", self.base_url, json=payload, headers=headers,
            ) as response:
                if response.status_code == 401:
                    await response.aread()
                    raise _CodexAuthExpired()
                if not response.is_success:
                    error_text = await response.aread()
                    raise LLMAPIError(
                        f"ChatGPT Responses API error {response.status_code}: {error_text.decode(errors='replace')}",
                        status_code=response.status_code,
                    )
                log.debug(
                    "ChatGPT Responses API stream opened",
                    status=response.status_code,
                    content_type=response.headers.get("content-type", ""),
                )
                async for line in response.aiter_lines():
                    collected.append(line)
            return collected

        try:
            try:
                collected_lines = await _do_request()
            except _CodexAuthExpired:
                log.info("ChatGPT Responses API 401 — force-refreshing Codex auth and retrying once.")
                await self._ensure_fresh_auth(force=True)
                collected_lines = await _do_request()

            events = self._parse_sse_events(collected_lines)

            log.info(
                "ChatGPT Responses API stream parsed",
                model=self.model,
                raw_lines=len(collected_lines),
                events=len(events),
                event_types=sorted({str(e.get("type", "")) for e in events})[:20],
            )

            # Find the completed event.
            completed: dict[str, Any] | None = None
            for evt in events:
                if evt.get("type") == "response.completed":
                    completed = evt
                    break

            if completed is None:
                # Fallback: try last event or collect text deltas.
                content_parts: list[str] = []
                tc_items: list[dict[str, Any]] = []
                for evt in events:
                    etype = evt.get("type", "")
                    if etype == "response.output_text.delta":
                        content_parts.append(str(evt.get("delta", "")))
                    elif etype == "response.output_item.done":
                        item = evt.get("item", {})
                        if item.get("type") == "function_call":
                            tc_items.append(item)
                tool_calls = []
                for tc in tc_items:
                    call_id = str(tc.get("call_id", "") or tc.get("id", ""))
                    name = str(tc.get("name", ""))
                    raw_args = tc.get("arguments", "{}")
                    args = _safe_json_loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    tool_calls.append(ToolCall(id=call_id, name=name, arguments=args))
                result = LLMResponse(
                    content="".join(content_parts),
                    tool_calls=tool_calls,
                    model=self.model,
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    finish_reason="tool_calls" if tool_calls else "stop",
                )
            else:
                result = self._parse_response_output(completed)

            # If parsing the completed event produced no content/tools,
            # reconstruct directly from the streamed events. The Codex
            # backend reliably emits text via ``response.output_text.delta``
            # and tool calls via ``response.output_item.done`` (with the
            # function_call item) plus per-arg ``response.function_call_arguments.delta``
            # chunks. Either source is enough to recover a valid turn.
            if not result.content and not result.tool_calls:
                delta_parts: list[str] = []
                # call_id → {"id", "name", "arguments_str"}
                fc_by_id: dict[str, dict[str, Any]] = {}
                fc_order: list[str] = []

                def _ingest_fc_item(item: dict[str, Any]) -> None:
                    if not isinstance(item, dict):
                        return
                    if item.get("type") != "function_call":
                        return
                    cid = str(item.get("call_id", "") or item.get("id", ""))
                    if not cid:
                        return
                    slot = fc_by_id.setdefault(
                        cid, {"id": cid, "name": "", "arguments_str": ""}
                    )
                    if cid not in fc_order:
                        fc_order.append(cid)
                    nm = item.get("name")
                    if isinstance(nm, str) and nm:
                        slot["name"] = nm
                    args = item.get("arguments")
                    if isinstance(args, str) and args:
                        slot["arguments_str"] = args
                    elif isinstance(args, dict):
                        slot["arguments_str"] = json.dumps(args)

                for evt in events:
                    et = evt.get("type", "")
                    if et == "response.output_text.delta":
                        d = evt.get("delta", "")
                        if isinstance(d, str) and d:
                            delta_parts.append(d)
                    elif et == "response.output_item.added":
                        _ingest_fc_item(evt.get("item", {}) or {})
                    elif et == "response.output_item.done":
                        _ingest_fc_item(evt.get("item", {}) or {})
                    elif et == "response.function_call_arguments.delta":
                        cid = str(evt.get("call_id", "") or evt.get("item_id", ""))
                        if cid:
                            slot = fc_by_id.setdefault(
                                cid, {"id": cid, "name": "", "arguments_str": ""}
                            )
                            if cid not in fc_order:
                                fc_order.append(cid)
                            d = evt.get("delta", "")
                            if isinstance(d, str):
                                # Only append deltas if we don't already have
                                # the full arguments string from output_item.done
                                # (which would otherwise duplicate).
                                if not slot["arguments_str"]:
                                    slot["arguments_str"] += d
                    elif et == "response.function_call_arguments.done":
                        cid = str(evt.get("call_id", "") or evt.get("item_id", ""))
                        full = evt.get("arguments", "")
                        if cid and isinstance(full, str) and full:
                            slot = fc_by_id.setdefault(
                                cid, {"id": cid, "name": "", "arguments_str": ""}
                            )
                            if cid not in fc_order:
                                fc_order.append(cid)
                            slot["arguments_str"] = full

                recovered_tool_calls: list[ToolCall] = []
                for cid in fc_order:
                    slot = fc_by_id[cid]
                    if not slot.get("name"):
                        continue
                    raw_args = slot.get("arguments_str") or "{}"
                    args = _safe_json_loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    recovered_tool_calls.append(
                        ToolCall(id=slot["id"], name=slot["name"], arguments=args)
                    )

                if delta_parts or recovered_tool_calls:
                    result = LLMResponse(
                        content="".join(delta_parts),
                        tool_calls=recovered_tool_calls or result.tool_calls,
                        model=result.model,
                        usage=result.usage,
                        finish_reason=(
                            "tool_calls" if recovered_tool_calls else result.finish_reason
                        ),
                    )

            if not result.content and not result.tool_calls:
                # The Codex backend sometimes returns 200 with a body
                # that has no message / no tool_calls — usually because
                # the model isn't actually served on this account, or
                # because the request was rejected mid-stream with an
                # error event we didn't recognise. Surface a useful
                # diagnostic instead of letting the orchestrator finish
                # silently with an empty turn.
                preview_lines = [ln for ln in collected_lines if ln.strip()][:20]
                log.warning(
                    "ChatGPT Responses API returned an empty turn",
                    model=self.model,
                    raw_lines=len(collected_lines),
                    events=len(events),
                    completed_seen=completed is not None,
                    preview="\n".join(preview_lines)[:2000],
                )
                # Look for a Codex error event the parser may have missed.
                err_detail = ""
                for evt in events:
                    et = str(evt.get("type", ""))
                    if "error" in et.lower():
                        err_detail = json.dumps(evt)[:500]
                        break
                if err_detail:
                    raise LLMAPIError(
                        f"ChatGPT Responses API returned an error event: {err_detail}"
                    )

            if self.rate_limiter:
                self.rate_limiter.record_actual(result.usage.get("total_tokens", 0), estimated)
            return result

        except LLMAPIError:
            raise
        except httpx.HTTPError as exc:
            raise LLMAPIError(f"ChatGPT Responses API HTTP error: {exc}")
        except Exception as exc:
            raise LLMError(f"ChatGPT Responses API call failed: {exc}")

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        if self.rate_limiter:
            estimated = self._estimate_request_tokens(messages, max_tokens or self.max_tokens)
            await self.rate_limiter.acquire(estimated)

        payload = self._build_payload(messages, tools, temperature, max_tokens)

        await self._ensure_fresh_auth()

        # For streaming we retry on 401 by re-entering the stream once.
        # The retry is only safe before we've yielded any delta to the
        # caller; we check the response status code first and bail out
        # of the pre-stream block via ``continue`` when we see a 401.
        for _attempt in (1, 2):
            headers = self._build_headers()
            try:
                async with self.client.stream(
                    "POST", self.base_url, json=payload, headers=headers,
                ) as response:
                    if response.status_code == 401 and _attempt == 1:
                        await response.aread()
                        log.info("ChatGPT Responses API 401 (streaming) — force-refreshing Codex auth and retrying once.")
                        await self._ensure_fresh_auth(force=True)
                        continue
                    if not response.is_success:
                        error_text = await response.aread()
                        raise LLMAPIError(
                            f"ChatGPT Responses API error {response.status_code}: {error_text.decode(errors='replace')}",
                            status_code=response.status_code,
                        )
                    async for line in response.aiter_lines():
                        stripped = line.strip()
                        if not stripped or stripped.startswith("event:"):
                            continue
                        if stripped.startswith("data: "):
                            data_str = stripped[6:]
                        elif stripped.startswith("data:"):
                            data_str = stripped[5:]
                        else:
                            continue
                        try:
                            evt = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        if evt.get("type") == "response.output_text.delta":
                            delta = evt.get("delta", "")
                            if delta:
                                yield str(delta)
                    return  # successful stream completed
            except LLMAPIError:
                raise
            except httpx.HTTPError as exc:
                raise LLMAPIError(f"ChatGPT Responses API streaming error: {exc}")
            except Exception as exc:
                raise LLMError(f"ChatGPT Responses API streaming failed: {exc}")

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    async def close(self) -> None:
        await self.client.aclose()
        if self._codex_auth is not None:
            await self._codex_auth.close()


class OllamaProvider(LLMProvider):
    """Direct Ollama API provider."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = OLLAMA_NATIVE_BASE_URL,
        temperature: float = 0.7,
        max_tokens: int = 32000,
        num_ctx: int = 160000,
        api_key: str | None = None,
        tokens_per_minute: int = 0,
    ):
        self.provider = "ollama"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = max(1, int(num_ctx))
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=120.0, follow_redirects=True)
        self.rate_limiter = TokenRateLimiter(tokens_per_minute) if tokens_per_minute > 0 else None

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        url = f"{self.base_url}/api/chat"
        ollama_messages = _convert_messages_for_ollama(messages)
        ollama_tools = _convert_tools_for_openai_style(tools) if tools else None

        options: dict[str, Any] = {
            "num_ctx": self.num_ctx,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if max_tokens or self.max_tokens:
            options["num_predict"] = max_tokens or self.max_tokens

        body: dict[str, Any] = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": options,
        }
        if ollama_tools:
            body["tools"] = ollama_tools

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        estimated = 0
        if self.rate_limiter:
            estimated = self._estimate_request_tokens(messages, max_tokens or self.max_tokens)
            await self.rate_limiter.acquire(estimated)

        max_retries = 2
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 2):
            try:
                response = await self.client.post(url, json=body, headers=headers)
                if not response.is_success:
                    raise LLMAPIError(
                        f"Ollama API error {response.status_code}: {response.text}",
                        status_code=response.status_code,
                    )
                data = response.json()
                msg_obj = _obj_get(data, "message", {})
                content = _obj_get(msg_obj, "content", "") or ""
                # Some Ollama models (deepseek-r1, qwq, etc.) emit
                # <think>...</think> reasoning blocks. Strip them.
                _rc = _obj_get(msg_obj, "reasoning_content", None) or _obj_get(msg_obj, "thinking", None)
                if _rc:
                    log.debug(
                        "Discarding Ollama reasoning_content",
                        model=self.model, reasoning_chars=len(str(_rc)),
                    )
                content = _strip_reasoning_artifacts(str(content))

                tool_calls: list[ToolCall] = []
                raw_calls = _obj_get(msg_obj, "tool_calls", []) or []
                for idx, raw_call in enumerate(raw_calls, start=1):
                    function = _obj_get(raw_call, "function", {}) or {}
                    call_name = str(_obj_get(function, "name", "") or "")
                    args = _obj_get(function, "arguments", {})
                    if isinstance(args, str):
                        args = _safe_json_loads(args)
                    if not isinstance(args, dict):
                        args = {}
                    tool_calls.append(ToolCall(
                        id=str(_obj_get(raw_call, "id", f"ollama_call_{idx}") or f"ollama_call_{idx}"),
                        name=call_name,
                        arguments=args,
                    ))

                usage = {
                    "prompt_tokens": int(_obj_get(data, "prompt_eval_count", 0) or 0),
                    "completion_tokens": int(_obj_get(data, "eval_count", 0) or 0),
                    "total_tokens": int(_obj_get(data, "prompt_eval_count", 0) or 0)
                    + int(_obj_get(data, "eval_count", 0) or 0),
                }
                if self.rate_limiter:
                    self.rate_limiter.record_actual(usage.get("total_tokens", 0), estimated)
                finish_reason = str(_obj_get(data, "done_reason", "") or "")
                return LLMResponse(
                    content=str(content),
                    tool_calls=tool_calls,
                    model=str(_obj_get(data, "model", self.model) or self.model),
                    usage=usage,
                    finish_reason=finish_reason,
                )
            except LLMAPIError:
                raise
            except httpx.HTTPError as e:
                last_exc = e
                if attempt <= max_retries:
                    await asyncio.sleep(1.0 * attempt)
                    continue
                err_detail = str(e) or type(e).__name__
                raise LLMAPIError(f"Ollama HTTP error: {err_detail}")
            except json.JSONDecodeError as e:
                raise LLMError(f"Ollama response decode error: {e}")
            except Exception as e:
                raise LLMError(f"Ollama call failed: {e}")
        # Should not reach here, but just in case:
        err_detail = str(last_exc) or type(last_exc).__name__ if last_exc else "unknown"
        raise LLMAPIError(f"Ollama HTTP error after retries: {err_detail}")

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        url = f"{self.base_url}/api/chat"
        ollama_messages = _convert_messages_for_ollama(messages)
        ollama_tools = _convert_tools_for_openai_style(tools) if tools else None

        options: dict[str, Any] = {
            "num_ctx": self.num_ctx,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if max_tokens or self.max_tokens:
            options["num_predict"] = max_tokens or self.max_tokens

        body: dict[str, Any] = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": True,
            "options": options,
        }
        if ollama_tools:
            body["tools"] = ollama_tools

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.rate_limiter:
            estimated = self._estimate_request_tokens(messages, max_tokens or self.max_tokens)
            await self.rate_limiter.acquire(estimated)

        max_retries = 2
        for attempt in range(1, max_retries + 2):
            try:
                async with self.client.stream("POST", url, json=body, headers=headers) as response:
                    if not response.is_success:
                        error_text = await response.atext()
                        raise LLMAPIError(
                            f"Ollama API error {response.status_code}: {error_text}",
                            status_code=response.status_code,
                        )
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        content = _obj_get(_obj_get(chunk, "message", {}), "content", "") or ""
                        if content:
                            yield str(content)
                        if _obj_get(chunk, "done", False):
                            break
                return  # successful stream completed
            except LLMAPIError:
                raise
            except httpx.HTTPError as e:
                if attempt <= max_retries:
                    await asyncio.sleep(1.0 * attempt)
                    continue
                err_detail = str(e) or type(e).__name__
                raise LLMAPIError(f"Ollama streaming error: {err_detail}")
            except Exception as e:
                raise LLMError(f"Ollama stream failed: {e}")

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    async def close(self) -> None:
        await self.client.aclose()


class LiteLLMProvider(LLMProvider):
    """LiteLLM-backed provider for OpenAI, Anthropic (Claude), and Gemini."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 32000,
        tokens_per_minute: int = 0,
        extra_headers: dict[str, str] | None = None,
    ):
        self.provider = _normalize_provider_name(provider)
        self.model = _provider_model_name(self.provider, model)
        self.api_key = _resolve_api_key(self.provider, api_key)
        self.base_url = (base_url or "").strip() or None
        self.temperature = _normalize_temperature_for_model(
            self.provider,
            self.model,
            temperature,
        )
        self.max_tokens = max_tokens
        self.rate_limiter = TokenRateLimiter(tokens_per_minute) if tokens_per_minute > 0 else None
        self.extra_headers = extra_headers or None

        # Google OAuth / Vertex AI credentials (set via set_vertex_credentials).
        self._vertex_credentials: dict[str, Any] | None = None
        self._vertex_project: str = ""
        self._vertex_location: str = "us-central1"

    def set_vertex_credentials(
        self,
        credentials: dict[str, Any],
        project: str,
        location: str = "us-central1",
    ) -> None:
        """Inject Google OAuth ``authorized_user`` credentials for Vertex AI.

        When set and the provider is ``gemini``, requests are routed
        through LiteLLM's ``vertex_ai/`` prefix instead of ``gemini/``,
        using OAuth tokens rather than API keys.
        """
        self._vertex_credentials = credentials
        self._vertex_project = project
        self._vertex_location = location

    def clear_vertex_credentials(self) -> None:
        """Remove injected Vertex AI credentials, reverting to API key auth."""
        self._vertex_credentials = None
        self._vertex_project = ""
        self._vertex_location = "us-central1"

    def _request_kwargs(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        resolved_temperature = _normalize_temperature_for_model(
            self.provider,
            self.model,
            self.temperature if temperature is None else temperature,
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _convert_messages_for_openai_style(messages),
            "temperature": resolved_temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": stream,
            "timeout": 180,
        }
        if tools:
            kwargs["tools"] = _convert_tools_for_openai_style(tools)

        # Anthropic prompt caching: split system message on CACHE_SPLIT marker
        # into static (cached) + dynamic (uncached) blocks, and add a cache
        # breakpoint on the last conversation message for tool-loop caching.
        if self.provider == "anthropic":
            kwargs["messages"] = _inject_anthropic_cache_control(kwargs["messages"])
        else:
            # Strip the cache-split marker for non-Anthropic providers.
            for msg in kwargs["messages"]:
                if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                    msg["content"] = msg["content"].replace(_CACHE_SPLIT_MARKER, "")

        # Force text-only output for Gemini so it uses function tools
        # (e.g. image_gen) instead of native image generation, which
        # returns image bytes that our response parser cannot handle.
        if self.provider == "gemini":
            kwargs["modalities"] = ["text"]

        # Always use the API key for Gemini (Google AI Studio).
        # Vertex AI routing is disabled — use api_key directly.
        if self.api_key:
            kwargs["api_key"] = self.api_key

        if self.base_url:
            kwargs["api_base"] = self.base_url
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        return kwargs

    async def _collect_streaming_response(
        self,
        stream: Any,
        on_chunk: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Collect an async streaming response into a unified response dict.

        Some providers (notably Gemini via LiteLLM) may return a streaming
        object even when ``stream=False``.  This helper iterates the stream
        and reassembles the chunks into the standard non-streaming format
        expected by :meth:`complete`.

        When *on_chunk* is provided, each text delta is forwarded to the
        callback in real time (for UI streaming).
        """
        content_parts: list[str] = []
        collected_tool_calls: dict[int, dict[str, Any]] = {}
        usage_obj: Any = None
        finish_reason = ""
        model = self.model

        try:
            async for chunk in stream:
                choices = _obj_get(chunk, "choices", [])
                if not choices:
                    continue
                first = choices[0]
                delta = _obj_get(first, "delta", {})

                # Skip reasoning_content deltas (Grok, DeepSeek, etc.) —
                # we never want internal chain-of-thought in chat output.
                _rc = _obj_get(delta, "reasoning_content", None)
                if _rc:
                    continue

                # Content text — may be a string or a list of parts
                # (Gemini sometimes returns list of content parts).
                c = _obj_get(delta, "content", "")
                if c:
                    if isinstance(c, str):
                        content_parts.append(c)
                        if on_chunk:
                            try:
                                on_chunk(c)
                            except Exception:
                                pass
                    elif isinstance(c, list):
                        for part in c:
                            text = _obj_get(part, "text", "")
                            if text:
                                content_parts.append(str(text))
                                if on_chunk:
                                    try:
                                        on_chunk(str(text))
                                    except Exception:
                                        pass
                    else:
                        content_parts.append(str(c))
                        if on_chunk:
                            try:
                                on_chunk(str(c))
                            except Exception:
                                pass

                # Finish reason (last chunk wins)
                fr = _obj_get(first, "finish_reason", "")
                if fr:
                    finish_reason = str(fr)

                # Streamed tool calls — accumulated by index
                tc_list = _obj_get(delta, "tool_calls", []) or []
                for tc in tc_list:
                    idx = int(_obj_get(tc, "index", 0) or 0)
                    fn = _obj_get(tc, "function", {}) or {}
                    if idx not in collected_tool_calls:
                        collected_tool_calls[idx] = {
                            "id": str(_obj_get(tc, "id", f"call_{idx}") or f"call_{idx}"),
                            "function": {"name": "", "arguments": ""},
                        }
                    name = str(_obj_get(fn, "name", "") or "")
                    if name:
                        collected_tool_calls[idx]["function"]["name"] = name
                    args = str(_obj_get(fn, "arguments", "") or "")
                    if args:
                        collected_tool_calls[idx]["function"]["arguments"] += args
                        # Stream tool call argument deltas too — this
                        # covers long write/code-generation tool calls
                        # where the model spends most of its tokens on
                        # the tool arguments rather than text content.
                        if on_chunk:
                            try:
                                on_chunk(args)
                            except Exception:
                                pass

                # Usage (typically on the final chunk)
                u = _obj_get(chunk, "usage", None)
                if u is not None:
                    usage_obj = u

                m = str(_obj_get(chunk, "model", "") or "")
                if m:
                    model = m
        except Exception as e:
            # Preserve whatever we collected so far rather than losing
            # the entire response.  Log so we can diagnose.
            log.warning(
                "Stream collection interrupted, returning partial content",
                error=str(e),
                content_parts_collected=len(content_parts),
                provider=self.provider,
                model=self.model,
            )

        tc_out = (
            [collected_tool_calls[i] for i in sorted(collected_tool_calls)]
            if collected_tool_calls
            else None
        )
        final_content = _strip_reasoning_artifacts("".join(content_parts))
        return {
            "choices": [{
                "message": {
                    "content": final_content,
                    "tool_calls": tc_out,
                },
                "finish_reason": finish_reason,
            }],
            "usage": usage_obj,
            "model": model,
        }

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        try:
            from litellm import acompletion
        except Exception as e:
            raise LLMError(f"LiteLLM is required for provider '{self.provider}': {e}")

        estimated = 0
        if self.rate_limiter:
            estimated = self._estimate_request_tokens(messages, max_tokens or self.max_tokens)
            await self.rate_limiter.acquire(estimated)

        try:
            kwargs = self._request_kwargs(
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )

            if self.provider == "gemini":
                # Gemini via LiteLLM's acompletion() has persistent
                # issues: truncated content, unexpected streaming objects.
                # Use the synchronous completion() in a thread to ensure
                # we block until the full response is available.
                from litellm import completion as sync_completion
                import asyncio as _asyncio
                loop = _asyncio.get_event_loop()
                _gemini_timeout = 180  # seconds
                try:
                    response = await _asyncio.wait_for(
                        loop.run_in_executor(
                            None, lambda: sync_completion(**kwargs)
                        ),
                        timeout=_gemini_timeout,
                    )
                except _asyncio.TimeoutError:
                    raise LLMError(
                        f"Gemini completion timed out after {_gemini_timeout}s "
                        f"(model={self.model}). The model may be overloaded or "
                        f"the request may be too large."
                    )
            else:
                response = await acompletion(**kwargs)

            # Safety fallback: if any provider still returns a streaming
            # object despite stream=False, collect it.
            if hasattr(response, "__aiter__"):
                log.warning(
                    "Provider returned streaming response with stream=False, "
                    "collecting full output",
                    provider=self.provider, model=self.model,
                )
                response = await self._collect_streaming_response(response)

            choices = _obj_get(response, "choices", [{}])
            if not choices:
                raise LLMError(
                    f"{self.provider} returned empty choices array "
                    f"(model={self.model})"
                )
            first_choice = choices[0]
            choice = _obj_get(first_choice, "message", {})
            raw_content = _obj_get(choice, "content", "") or ""
            # Some providers return content as a list of parts.
            if isinstance(raw_content, list):
                text_parts = []
                for part in raw_content:
                    t = _obj_get(part, "text", "")
                    if t:
                        text_parts.append(str(t))
                content = "".join(text_parts)
            else:
                content = str(raw_content) if raw_content else ""
            # Discard provider reasoning_content (xAI Grok, DeepSeek, etc.)
            # — never include the model's internal chain-of-thought in the
            # user-visible response. We log its presence for debugging.
            reasoning_content = _obj_get(choice, "reasoning_content", None)
            if reasoning_content:
                log.debug(
                    "Discarding reasoning_content from LLM response",
                    provider=self.provider,
                    model=self.model,
                    reasoning_chars=len(str(reasoning_content)),
                )
            # Strip any <think>...</think> blocks that leaked into content.
            content = _strip_reasoning_artifacts(content)
            finish_reason = str(_obj_get(first_choice, "finish_reason", "") or "")

            tool_calls: list[ToolCall] = []
            raw_calls = _obj_get(choice, "tool_calls", []) or []
            for idx, raw_call in enumerate(raw_calls, start=1):
                function = _obj_get(raw_call, "function", {}) or {}
                name = str(_obj_get(function, "name", "") or "")
                raw_args = _obj_get(function, "arguments", {})
                if isinstance(raw_args, str):
                    arguments = _safe_json_loads(raw_args)
                elif isinstance(raw_args, dict):
                    arguments = raw_args
                else:
                    arguments = {}
                tool_calls.append(ToolCall(
                    id=str(_obj_get(raw_call, "id", f"call_{idx}") or f"call_{idx}"),
                    name=name,
                    arguments=arguments,
                ))

            usage = _extract_usage(_obj_get(response, "usage", None))
            if self.rate_limiter:
                self.rate_limiter.record_actual(usage.get("total_tokens", 0), estimated)
            return LLMResponse(
                content=str(content),
                tool_calls=tool_calls,
                model=str(_obj_get(response, "model", self.model) or self.model),
                usage=usage,
                finish_reason=finish_reason,
            )
        except Exception as e:
            status_code = _obj_get(e, "status_code", None)
            message = f"{self.provider} API call failed: {e}"
            # Log details for debugging vague provider errors.
            msg_count = len(kwargs.get("messages", []))
            total_chars = sum(len(str(m.get("content", ""))) for m in kwargs.get("messages", []))
            tool_count = len(kwargs.get("tools", []) or [])
            log.error(
                "LLM call failed",
                error=str(e),
                model=kwargs.get("model"),
                msg_count=msg_count,
                total_chars=total_chars,
                tool_count=tool_count,
                temperature=kwargs.get("temperature"),
            )
            if status_code is not None:
                raise LLMAPIError(message, status_code=int(status_code))
            raise LLMError(message)

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        try:
            from litellm import acompletion
        except Exception as e:
            raise LLMError(f"LiteLLM is required for provider '{self.provider}': {e}")

        if self.rate_limiter:
            estimated = self._estimate_request_tokens(messages, max_tokens or self.max_tokens)
            await self.rate_limiter.acquire(estimated)

        try:
            stream = await acompletion(
                **self._request_kwargs(
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
            )
            async for chunk in stream:
                delta_obj = _obj_get(_obj_get(chunk, "choices", [{}])[0], "delta", {})
                # Skip reasoning_content deltas (Grok, DeepSeek, etc.).
                if _obj_get(delta_obj, "reasoning_content", None):
                    continue
                delta = _obj_get(delta_obj, "content", "")
                if not delta:
                    continue
                if isinstance(delta, str):
                    yield delta
                    continue
                if isinstance(delta, list):
                    text_parts: list[str] = []
                    for part in delta:
                        text = _obj_get(part, "text", "")
                        if text:
                            text_parts.append(str(text))
                    if text_parts:
                        yield "".join(text_parts)
                    continue
                yield str(delta)
        except Exception as e:
            status_code = _obj_get(e, "status_code", None)
            message = f"{self.provider} streaming failed: {e}"
            if status_code is not None:
                raise LLMAPIError(message, status_code=int(status_code))
            raise LLMError(message)

    async def complete_with_callback(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Stream completion with real-time text callback.

        Uses ``stream=True`` so text chunks are forwarded to *on_chunk*
        as they arrive, while still returning the full ``LLMResponse``
        with tool_calls, usage, and finish_reason.
        """
        if not on_chunk:
            return await self.complete(messages, tools, temperature, max_tokens)

        try:
            from litellm import acompletion
        except Exception as e:
            raise LLMError(f"LiteLLM is required for provider '{self.provider}': {e}")

        estimated = 0
        if self.rate_limiter:
            estimated = self._estimate_request_tokens(messages, max_tokens or self.max_tokens)
            await self.rate_limiter.acquire(estimated)

        try:
            kwargs = self._request_kwargs(
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            # Request usage in the final stream chunk.
            kwargs["stream_options"] = {"include_usage": True}

            stream = await acompletion(**kwargs)
            collected = await self._collect_streaming_response(stream, on_chunk=on_chunk)

            # Parse the collected dict into an LLMResponse (same as complete()).
            choices = _obj_get(collected, "choices", [{}])
            if not choices:
                raise LLMError(f"{self.provider} returned empty choices (streaming)")
            first_choice = choices[0]
            choice = _obj_get(first_choice, "message", {})
            raw_content = _obj_get(choice, "content", "") or ""
            if isinstance(raw_content, list):
                text_parts = []
                for part in raw_content:
                    t = _obj_get(part, "text", "")
                    if t:
                        text_parts.append(str(t))
                content = "".join(text_parts)
            else:
                content = str(raw_content) if raw_content else ""
            finish_reason = str(_obj_get(first_choice, "finish_reason", "") or "")

            tool_calls: list[ToolCall] = []
            raw_calls = _obj_get(choice, "tool_calls", []) or []
            for idx, raw_call in enumerate(raw_calls, start=1):
                function = _obj_get(raw_call, "function", {}) or {}
                name = str(_obj_get(function, "name", "") or "")
                raw_args = _obj_get(function, "arguments", {})
                if isinstance(raw_args, str):
                    arguments = _safe_json_loads(raw_args)
                elif isinstance(raw_args, dict):
                    arguments = raw_args
                else:
                    arguments = {}
                tool_calls.append(ToolCall(
                    id=str(_obj_get(raw_call, "id", f"call_{idx}") or f"call_{idx}"),
                    name=name,
                    arguments=arguments,
                ))

            usage = _extract_usage(_obj_get(collected, "usage", None))
            if self.rate_limiter:
                self.rate_limiter.record_actual(usage.get("total_tokens", 0), estimated)
            return LLMResponse(
                content=str(content),
                tool_calls=tool_calls,
                model=str(_obj_get(collected, "model", self.model) or self.model),
                usage=usage,
                finish_reason=finish_reason,
            )
        except Exception as e:
            status_code = _obj_get(e, "status_code", None)
            message = f"{self.provider} streaming callback failed: {e}"
            if status_code is not None:
                raise LLMAPIError(message, status_code=int(status_code))
            raise LLMError(message)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            from litellm import token_counter

            return int(token_counter(model=self.model, messages=[{"role": "user", "content": text}]))
        except Exception:
            return len(text) // 4


def _litert_tool_fields(tool: Any) -> tuple[str, str, dict[str, Any]]:
    """Extract ``(name, description, parameters)`` from a tool of any shape.

    Captain Claw passes tools to providers in several shapes depending on
    the call site:
      - ``ToolDefinition`` dataclass instances
      - OpenAI-style dicts: ``{"type": "function", "function": {...}}``
      - Flat dicts: ``{"name": ..., "description": ..., "parameters": ...}``
    """
    if isinstance(tool, ToolDefinition):
        return tool.name, tool.description or "", tool.parameters or {}
    if isinstance(tool, dict):
        if isinstance(tool.get("function"), dict):
            fn = tool["function"]
            return (
                str(fn.get("name") or ""),
                str(fn.get("description") or ""),
                fn.get("parameters") or {},
            )
        return (
            str(tool.get("name") or ""),
            str(tool.get("description") or ""),
            tool.get("parameters") or {},
        )
    # Last-resort attribute lookup.
    return (
        str(getattr(tool, "name", "") or ""),
        str(getattr(tool, "description", "") or ""),
        getattr(tool, "parameters", {}) or {},
    )


def _litert_build_tool_manifest(tools: list[Any]) -> str:
    """Render a compact text manifest of tools for the system prompt.

    Local Gemma via litert-lm has no structured function-calling bridge,
    so we teach the model to emit calls as inline text using a fixed
    fence that we can parse out of the reply afterwards.
    """
    lines: list[str] = [
        "You can call tools. To call a tool, write EXACTLY this on its own line:",
        "<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"arg\": \"value\"}}</tool_call>",
        "Use ONE call per turn. After the tool result is returned to you,",
        "continue the conversation. Do not invent tools — only use the ones listed.",
        "",
        "Available tools:",
    ]
    for t in tools:
        name, description, params = _litert_tool_fields(t)
        if not name:
            continue
        props = params.get("properties") if isinstance(params, dict) else None
        required = params.get("required") if isinstance(params, dict) else None
        arg_summary = ""
        if isinstance(props, dict) and props:
            arg_bits = []
            for arg_name, arg_schema in props.items():
                ty = ""
                if isinstance(arg_schema, dict):
                    ty = str(arg_schema.get("type") or "")
                req = (
                    isinstance(required, list) and arg_name in required
                )
                arg_bits.append(
                    f"{arg_name}{':' + ty if ty else ''}{'' if req else '?'}"
                )
            arg_summary = "(" + ", ".join(arg_bits) + ")"
        desc = description.strip().replace("\n", " ")
        if len(desc) > 200:
            desc = desc[:197] + "..."
        lines.append(f"- {name}{arg_summary} — {desc}")
    return "\n".join(lines)


# Patterns we accept for inline tool calls coming back from local Gemma:
#   1. Our preferred fence:  <tool_call>{...json...}</tool_call>
#   2. The model's habit:    <execute_tool_call> name(arg='value') </execute_tool_call>
#                            <execute_tool_call> web_tool_call: name(arg='value') </execute_tool_call>
#   3. Bare JSON-in-fence:   ```tool_call\n{...}\n```
#   4. Gemma-4 native template:
#         <|tool_call>call:name{key:<|"|>value<|"|>,n:42}<tool_call|>
#      where strings are wrapped in the literal delimiter ``<|"|>``.
_LITERT_TOOL_CALL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE),
        "json",
    ),
    (
        re.compile(r"```tool_call\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE),
        "json",
    ),
    (
        re.compile(
            r"<execute_tool_call>\s*(?:web_tool_call:\s*)?([a-zA-Z_][\w\.]*)\s*\((.*?)\)\s*</execute_tool_call>",
            re.DOTALL | re.IGNORECASE,
        ),
        "pyish",
    ),
    (
        re.compile(
            r"<\|tool_call>\s*call:\s*([a-zA-Z_][\w\.]*)\s*\{(.*?)\}\s*<tool_call\|>",
            re.DOTALL,
        ),
        "gemma",
    ),
]


def _litert_parse_gemma_args(arg_str: str) -> dict[str, Any]:
    """Parse Gemma-4 native ``key:<|"|>value<|"|>,n:42`` argument strings.

    Strings are wrapped in the literal three-char delimiter ``<|"|>`` on
    both sides. Other scalars (ints, floats, bools, null) are bare.
    Nested objects/arrays use ``{}`` / ``[]`` and are kept as raw strings
    if encountered (best-effort).
    """
    out: dict[str, Any] = {}
    if not arg_str.strip():
        return out

    QUOTE = "<|\"|>"

    # Tokenise on top-level commas, respecting nested {}/[] and <|"|>...<|"|>.
    parts: list[str] = []
    depth = 0
    in_str = False
    i = 0
    buf: list[str] = []
    n = len(arg_str)
    while i < n:
        # Detect the literal quote delimiter <|"|>
        if arg_str.startswith(QUOTE, i):
            in_str = not in_str
            buf.append(QUOTE)
            i += len(QUOTE)
            continue
        ch = arg_str[i]
        if in_str:
            buf.append(ch)
            i += 1
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf))

    for part in parts:
        if ":" not in part:
            continue
        key, _, raw = part.partition(":")
        key = key.strip()
        raw = raw.strip()
        if not key:
            continue
        # String form: <|"|>...<|"|>
        if raw.startswith(QUOTE) and raw.endswith(QUOTE) and len(raw) >= 2 * len(QUOTE):
            out[key] = raw[len(QUOTE) : -len(QUOTE)]
            continue
        low = raw.lower()
        if low == "true":
            out[key] = True
            continue
        if low == "false":
            out[key] = False
            continue
        if low in ("null", "none"):
            out[key] = None
            continue
        try:
            out[key] = int(raw)
            continue
        except ValueError:
            pass
        try:
            out[key] = float(raw)
            continue
        except ValueError:
            pass
        # Strip any leftover Gemma quote markers we might have missed.
        out[key] = raw.replace(QUOTE, "")
    return out


def _litert_parse_pyish_args(arg_str: str) -> dict[str, Any]:
    """Parse ``key='value', key2=123`` style argument lists into a dict.

    Best-effort — we accept single or double quoted strings, bare ints
    and floats, and ``true``/``false``/``null``. Unparseable values fall
    back to the raw string.
    """
    out: dict[str, Any] = {}
    if not arg_str.strip():
        return out
    # Tokenise on top-level commas (don't split inside quotes/brackets).
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    buf: list[str] = []
    for ch in arg_str:
        if quote:
            buf.append(ch)
            if ch == quote and (len(buf) < 2 or buf[-2] != "\\"):
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf))

    for part in parts:
        if "=" not in part:
            continue
        key, _, raw = part.partition("=")
        key = key.strip()
        raw = raw.strip()
        if not key:
            continue
        # Strip matching quotes.
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
            out[key] = raw[1:-1]
            continue
        low = raw.lower()
        if low == "true":
            out[key] = True
            continue
        if low == "false":
            out[key] = False
            continue
        if low in ("null", "none"):
            out[key] = None
            continue
        try:
            out[key] = int(raw)
            continue
        except ValueError:
            pass
        try:
            out[key] = float(raw)
            continue
        except ValueError:
            pass
        out[key] = raw
    return out


def _litert_extract_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Scan ``text`` for inline tool-call fences and return ``(stripped_text, calls)``.

    The returned text has all matched fences removed so the user only
    sees the natural-language portion of the model's reply.
    """
    if not text:
        return text, []
    calls: list[ToolCall] = []
    cleaned = text
    counter = 0

    for pattern, kind in _LITERT_TOOL_CALL_PATTERNS:
        while True:
            m = pattern.search(cleaned)
            if not m:
                break
            try:
                if kind == "json":
                    payload = json.loads(m.group(1))
                    if isinstance(payload, dict):
                        name = str(payload.get("name") or payload.get("tool") or "").strip()
                        args = payload.get("arguments") or payload.get("args") or {}
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:  # pylint: disable=broad-exception-caught
                                args = {"_raw": args}
                        if name:
                            counter += 1
                            calls.append(
                                ToolCall(
                                    id=f"litert_{counter}",
                                    name=name,
                                    arguments=args if isinstance(args, dict) else {},
                                )
                            )
                elif kind == "pyish":
                    name = m.group(1).strip()
                    args = _litert_parse_pyish_args(m.group(2) or "")
                    if name:
                        counter += 1
                        calls.append(
                            ToolCall(
                                id=f"litert_{counter}",
                                name=name,
                                arguments=args,
                            )
                        )
                elif kind == "gemma":
                    name = m.group(1).strip()
                    args = _litert_parse_gemma_args(m.group(2) or "")
                    if name:
                        counter += 1
                        calls.append(
                            ToolCall(
                                id=f"litert_{counter}",
                                name=name,
                                arguments=args,
                            )
                        )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                log.warning("litert tool-call parse failed: %s", exc)
            # Drop this fence from the visible text either way.
            cleaned = cleaned[: m.start()] + cleaned[m.end():]

    return cleaned.strip(), calls


class LiteRTProvider(LLMProvider):
    """Local provider backed by Google's litert-lm runtime.

    Loads a ``.litertlm`` model file once at construction and serves chat
    requests in-process. Each request creates a fresh ``Conversation`` with
    the prior history as preface and sends only the latest user message —
    Captain Claw is stateless from the model's perspective (it always
    passes the full message list), and litert-lm's ``send_message`` API
    expects exactly one message at a time.

    **Tool calling is not exposed.** litert-lm's ``Conversation`` API does
    accept Python tool functions, but Captain Claw passes JSON-Schema
    ``ToolDefinition`` objects which would need an inverse bridge. For
    now any ``tools=`` argument is logged and ignored. Local Gemma models
    are best used as council members or chat companions where tools
    aren't needed.

    **Concurrency.** The underlying engine is single-threaded, so an
    ``asyncio.Lock`` serializes ``complete()`` and ``complete_streaming()``
    across the process. Streaming runs the sync iterator in a worker
    thread and pipes chunks back through an ``asyncio.Queue``.

    **Model resolution.** ``model_path`` may be either an absolute path to
    a ``.litertlm`` file or a model id like
    ``litert-community/gemma-4-E4B-it-litert-lm`` — in the latter case the
    model must already be present at
    ``~/.litert-lm/models/<repo>--<name>/model.litertlm`` (run
    ``litert-lm import <id>`` once first, or run the upstream
    ``litert-lm run --from-huggingface-repo=<id>`` once to download).
    """

    def __init__(
        self,
        model: str,
        model_path: str | None = None,
        backend: str = "gpu",
        temperature: float = 0.7,  # noqa: ARG002 — kept for API parity
        max_tokens: int = 4096,
        max_num_tokens: int = 16384,
        tokens_per_minute: int = 0,
    ):
        # The provider only validates the model path here. The actual
        # ``litert_lm.Engine`` is owned by a dedicated subprocess that
        # is spawned lazily on the first ``complete()`` call by the
        # shared worker client. Running the engine out-of-process
        # isolates two C++-side failure modes that previously took the
        # whole agent down: KV-cache overflow on long conversations
        # and Metal/GPU context exhaustion. See
        # ``captain_claw/llm/litert_worker.py``.

        self.provider = "litert"
        self.model = model
        # Expose the original reference (HF id or absolute path) as
        # ``base_url`` so the agent's "is current provider already this
        # one?" comparison can short-circuit and skip rebuilding the
        # engine on session-load.
        self.base_url = (model_path or "").strip()
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Resolve the model file. Order of precedence:
        #   1. If `ref` is an existing file path, use it directly.
        #   2. Try the litert-lm CLI layout (~/.litert-lm/models/<id>/model.litertlm).
        #   3. Try the Hugging Face hub cache populated by
        #      `litert-lm run --from-huggingface-repo=<id>` — that path is
        #      ~/.cache/huggingface/hub/models--<owner>--<repo>/snapshots/<rev>/<file>.litertlm.
        ref = model_path or model
        resolved_path = self._resolve_model_file(ref)
        if not resolved_path:
            raise LLMError(
                f"litert model file not found for reference '{ref}'. "
                "Pass an absolute path to a .litertlm file in `base_url`, or run "
                f"`litert-lm run --from-huggingface-repo={ref} <file>.litertlm` "
                "once to download it first."
            )
        self.model_path = resolved_path

        self._backend = (backend or "gpu").strip().lower()
        # Honour the caller's chosen window verbatim. ``create_provider``
        # is responsible for capping this at the .litertlm file's real
        # KV limit (see ``LITERT_MAX_NUM_TOKENS``); the previous
        # ``max(max_num_tokens, max_tokens, 4096)`` floor was leaking
        # the global 160k context budget into the engine and either
        # getting silently clamped or hanging on VRAM pressure.
        self._max_num_tokens = max(int(max_num_tokens or 0), 1024)

        # Acquire (or build) the shared worker client for this
        # (path, backend, max_num_tokens) triple. This does NOT spawn
        # the child yet — that happens on the first send_message call,
        # so importing this provider is cheap and side-effect free.
        from captain_claw.llm.litert_worker import get_or_create_litert_worker

        self._client = get_or_create_litert_worker(
            model_path=self.model_path,
            backend=self._backend,
            max_num_tokens=self._max_num_tokens,
            recycle_after_each=True,
        )

        log.info(
            "LiteRTProvider bound to worker client",
            model=self.model,
            path=self.model_path,
            backend=self._backend,
            max_num_tokens=self._max_num_tokens,
        )

        self.rate_limiter = (
            TokenRateLimiter(tokens_per_minute) if tokens_per_minute > 0 else None
        )

    @staticmethod
    def _resolve_model_file(ref: str) -> str | None:
        """Resolve a model reference to an absolute .litertlm file path."""
        if not ref:
            return None

        # 1. Direct file path.
        if os.path.isfile(ref):
            return os.path.abspath(ref)

        # 2. litert-lm CLI imported-models layout.
        try:
            from litert_lm_cli.model import Model as _CliModel  # type: ignore[import-not-found]

            cli_model = _CliModel.from_model_reference(ref)
            if os.path.isfile(cli_model.model_path):
                return cli_model.model_path
        except ImportError:
            pass

        # 3. Hugging Face hub cache. The reference is expected to look
        # like "owner/repo" (with optional ":filename" suffix). The cache
        # directory follows the form
        # ~/.cache/huggingface/hub/models--<owner>--<repo>/snapshots/<rev>/.
        # IMPORTANT: we return the SYMLINK path inside snapshots/, not
        # the resolved blob — litert-lm relies on the .litertlm suffix
        # for format detection, and the blob filename is a bare sha256
        # without an extension.
        if "/" in ref:
            spec, _, hint_filename = ref.partition(":")
            owner_repo = spec.replace("/", "--")
            try:
                hf_home = os.environ.get("HF_HOME") or os.path.join(
                    os.path.expanduser("~"), ".cache", "huggingface"
                )
                hub_dir = os.path.join(hf_home, "hub", f"models--{owner_repo}", "snapshots")
                if os.path.isdir(hub_dir):
                    candidates: list[str] = []
                    for snap in sorted(os.listdir(hub_dir)):
                        snap_dir = os.path.join(hub_dir, snap)
                        if not os.path.isdir(snap_dir):
                            continue
                        for fname in sorted(os.listdir(snap_dir)):
                            if not fname.endswith(".litertlm"):
                                continue
                            if hint_filename and fname != hint_filename:
                                continue
                            full = os.path.join(snap_dir, fname)
                            # `os.path.isfile` follows the symlink for us;
                            # we keep the symlink path so the .litertlm
                            # suffix is preserved when litert-lm opens it.
                            if os.path.isfile(full):
                                candidates.append(full)
                    if candidates:
                        # Prefer the lexicographically last (newest snapshot
                        # rev sorted, last file alphabetically).
                        return candidates[-1]
            except OSError:
                pass

        return None

    @staticmethod
    def _to_litert_msg(msg: Message) -> dict[str, Any]:
        """Convert a Captain Claw Message to litert-lm's expected dict shape."""
        role = msg.role or "user"
        # litert-lm's chat template expects content as a list of typed parts.
        return {
            "role": role,
            "content": [{"type": "text", "text": msg.content or ""}],
        }

    def _split_history(self, messages: list[Message]) -> tuple[list[dict[str, Any]], str]:
        """Return ``(preface, last_user_text)`` for the conversation.

        The preface is everything up to (but not including) the trailing
        user turn; the trailing user turn's text becomes the
        ``send_message`` argument. Anything after the last user turn (e.g.
        a stale assistant draft) is dropped — the new response replaces it.
        """
        if not messages:
            return [], ""
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                last_user_idx = i
                break
        if last_user_idx < 0:
            # No user turn at all — feed everything as preface and send
            # an empty prompt. Unusual, but the model will produce
            # something deterministic.
            return [self._to_litert_msg(m) for m in messages], ""
        preface = [self._to_litert_msg(m) for m in messages[:last_user_idx]]
        last_text = messages[last_user_idx].content or ""
        return preface, last_text

    def _build_preface(
        self,
        messages: list[Message],
        tool_manifest: str = "",
    ) -> tuple[list[dict[str, Any]], str]:
        """Split history into (preface, last_user) and inject the manifest.

        The actual ``send_message`` call is performed by the worker
        subprocess via ``self._client.send_message(...)``.
        """
        preface, last_user = self._split_history(messages)
        if tool_manifest:
            # Inject the manifest as a synthetic system turn at the very
            # top of the preface so it's always in scope.
            preface = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": tool_manifest}],
                }
            ] + preface
        return preface, last_user

    # ------------------------------------------------------------------
    # CLI fallback (current default)
    # ------------------------------------------------------------------
    #
    # The Python ``litert_lm.Engine`` path still exists above as the
    # subprocess worker client, but right now we deliberately bypass it
    # and shell out to the upstream ``litert-lm run`` CLI for every
    # call. Each invocation pays the full model load cost (~5–15s on
    # warm OS page cache) but starts from a completely fresh process,
    # which sidesteps every state-leak / GPU-context / KV-cache issue
    # we've been chasing. The trade-off is latency, not correctness.
    #
    # The CLI command we run mirrors what the user verified manually:
    #
    #     litert-lm run \
    #         --from-huggingface-repo=<repo> <file> \
    #         --backend=gpu --prompt="<text>"
    #
    # ``LITERT_HF_REPO`` / ``LITERT_HF_FILE`` / ``LITERT_CLI_BIN`` /
    # ``LITERT_CLI_TIMEOUT_SECONDS`` override the defaults.

    def _build_cli_prompt(
        self,
        messages: list[Message],
        manifest: str = "",
    ) -> str:
        """Flatten the conversation into a single string for ``--prompt``.

        Format is intentionally minimal — role label + content per turn,
        blank line between turns. The CLI applies its own chat template
        on top of this so we don't need Gemma-specific markers.
        """
        parts: list[str] = []
        if manifest:
            parts.append(f"System:\n{manifest}")
        for msg in messages:
            role = (msg.role or "user").strip().lower()
            text = (msg.content or "").strip()
            if not text:
                continue
            label = {
                "system": "System",
                "user": "User",
                "assistant": "Assistant",
                "tool": "Tool",
            }.get(role, role.capitalize())
            parts.append(f"{label}:\n{text}")
        return "\n\n".join(parts)

    async def _complete_via_cli(
        self,
        messages: list[Message],
        manifest: str = "",
    ) -> str:
        """Run ``litert-lm run`` as a one-shot subprocess and return stdout.

        Returns the raw stdout text. Caller is responsible for
        post-processing (reasoning-artifact stripping, tool-call
        extraction, etc.). Raises :class:`LLMError` on non-zero exit
        or timeout.
        """
        prompt = self._build_cli_prompt(messages, manifest)

        # ── Pre-flight prune ───────────────────────────────────────
        # The .litertlm file we ship has an 8k KV cache, and the CLI
        # segfaults (SIGSEGV, not a clean error) on prompts much over
        # ~25 KB. Enforce a hard character budget here by dropping
        # middle conversation turns (oldest first, keeping the system
        # preamble + the most recent turns) until we fit. This is a
        # last-line safety net; ideally the agent's context manager
        # wouldn't send us 50 KB in the first place.
        try:
            budget_chars = int(
                os.getenv("LITERT_PROMPT_BUDGET_CHARS", "24000") or 24000
            )
        except ValueError:
            budget_chars = 24000

        if budget_chars > 0 and len(prompt) > budget_chars:
            # Split messages into system (kept) vs. conversation (prunable).
            keep_system: list[Message] = []
            convo: list[Message] = []
            for m in messages:
                if (m.role or "").strip().lower() == "system" and not convo:
                    keep_system.append(m)
                else:
                    convo.append(m)

            before_len = len(prompt)
            before_msgs = len(messages)
            dropped = 0
            # Drop oldest conversation turns one at a time, always
            # preserving the most recent user/assistant turn so the
            # model still has something to respond to.
            while convo and len(prompt) > budget_chars and len(convo) > 1:
                convo.pop(0)
                dropped += 1
                prompt = self._build_cli_prompt(
                    keep_system + convo, manifest
                )

            log.warning(
                "litert prompt pruned",
                before_len=before_len,
                after_len=len(prompt),
                before_msgs=before_msgs,
                after_msgs=len(keep_system) + len(convo),
                dropped=dropped,
                budget_chars=budget_chars,
            )

            # If we're *still* over budget after dropping everything
            # prunable, the system preamble alone is too big. Log
            # loudly and let the CLI try — it may segfault, but at
            # least we'll see it in the logs.
            if len(prompt) > budget_chars:
                log.error(
                    "litert prompt still over budget after pruning",
                    final_len=len(prompt),
                    budget_chars=budget_chars,
                    hint=(
                        "System preamble + manifest exceed the budget. "
                        "Reduce tool manifest or system prompt."
                    ),
                )

        # Dump the exact prompt to disk so we can re-run the failing
        # call manually with ``litert-lm run … --prompt="$(cat …)"``
        # without having to reconstruct it. Overwritten on every call;
        # the most recent prompt always wins.
        try:
            with open("/tmp/big_prompt.txt", "w", encoding="utf-8") as _f:
                _f.write(prompt)
        except Exception as _dump_err:  # pylint: disable=broad-exception-caught
            log.warning(
                "Failed to dump litert prompt to /tmp/big_prompt.txt",
                error=str(_dump_err),
            )

        # Default to the ``litert-lm`` script next to the running
        # Python interpreter. When Captain Claw runs from a venv,
        # ``sys.executable`` is ``<venv>/bin/python`` and the CLI is
        # ``<venv>/bin/litert-lm`` — looking it up via PATH would miss
        # it because nothing puts the venv bin on PATH globally.
        import sys as _sys

        default_cli = os.path.join(
            os.path.dirname(_sys.executable), "litert-lm"
        )
        if not os.path.isfile(default_cli):
            default_cli = "litert-lm"
        cli_bin = os.getenv("LITERT_CLI_BIN", default_cli)
        hf_repo = os.getenv(
            "LITERT_HF_REPO",
            "litert-community/gemma-4-E4B-it-litert-lm",
        )
        hf_file = os.getenv(
            "LITERT_HF_FILE",
            os.path.basename(self.model_path) or "gemma-4-E4B-it.litertlm",
        )
        timeout_s = float(
            os.getenv("LITERT_CLI_TIMEOUT_SECONDS", "300") or 300
        )

        cmd = [
            cli_bin,
            "run",
            f"--from-huggingface-repo={hf_repo}",
            hf_file,
            f"--backend={self._backend}",
            f"--prompt={prompt}",
        ]

        log.info(
            "Running litert-lm CLI",
            bin=cli_bin,
            backend=self._backend,
            hf_repo=hf_repo,
            hf_file=hf_file,
            prompt_len=len(prompt),
            timeout_s=timeout_s,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise LLMError(
                f"litert-lm CLI not found at '{cli_bin}'. Set LITERT_CLI_BIN "
                f"or install litert-lm. ({e})"
            ) from e

        # Stream stdout/stderr line-by-line so the parent's log shows
        # progress in real time instead of waiting for the subprocess
        # to exit. This is the only way to know the CLI is alive when
        # generation is taking 30+ seconds.
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        async def _drain(
            stream: asyncio.StreamReader | None,
            sink: list[str],
            label: str,
        ) -> None:
            if stream is None:
                return
            while True:
                try:
                    line = await stream.readline()
                except Exception as e:  # pylint: disable=broad-exception-caught
                    log.warning(
                        "litert-lm CLI stream read failed",
                        stream=label,
                        error=str(e),
                    )
                    break
                if not line:
                    break
                text = line.decode(errors="replace")
                sink.append(text)
                stripped = text.rstrip("\n")
                if stripped:
                    log.info(
                        "litert-lm CLI",
                        stream=label,
                        line=stripped,
                    )

        drain_out = asyncio.create_task(_drain(proc.stdout, stdout_chunks, "stdout"))
        drain_err = asyncio.create_task(_drain(proc.stderr, stderr_chunks, "stderr"))

        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout_s)
        except asyncio.TimeoutError as e:
            log.error(
                "litert-lm CLI timed out — killing subprocess",
                timeout_s=timeout_s,
                stdout_so_far=len("".join(stdout_chunks)),
                stderr_so_far=len("".join(stderr_chunks)),
            )
            try:
                proc.kill()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            # Cancel the drain tasks so they don't leak.
            for t in (drain_out, drain_err):
                if not t.done():
                    t.cancel()
            raise LLMError(
                f"litert-lm CLI timed out after {timeout_s}s"
            ) from e

        # Make sure the drain tasks finish reading whatever is left in
        # the pipes before we return.
        try:
            await asyncio.wait_for(
                asyncio.gather(drain_out, drain_err, return_exceptions=True),
                timeout=5,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            for t in (drain_out, drain_err):
                if not t.done():
                    t.cancel()

        stdout_text = "".join(stdout_chunks)
        stderr_text = "".join(stderr_chunks)

        log.info(
            "litert-lm CLI finished",
            returncode=proc.returncode,
            stdout_len=len(stdout_text),
            stderr_len=len(stderr_text),
        )

        if proc.returncode != 0:
            tail = stderr_text[-800:] if stderr_text else "(no stderr)"
            raise LLMError(
                f"litert-lm CLI exited with code {proc.returncode}: {tail}"
            )

        return stdout_text

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,  # noqa: ARG002
        max_tokens: int | None = None,
    ) -> LLMResponse:
        manifest = _litert_build_tool_manifest(list(tools)) if tools else ""
        if self.rate_limiter:
            estimated = self._estimate_request_tokens(
                messages, max_tokens or self.max_tokens
            )
            await self.rate_limiter.acquire(estimated)

        preface, last_user = self._build_preface(messages, manifest)
        try:
            content = await self._client.send_message(preface, last_user)
        except Exception as e:  # pylint: disable=broad-exception-caught
            err_text = str(e)
            err_type = type(e).__name__
            log.error(
                "litert-lm complete() failed; returning graceful response",
                error=err_text,
                error_type=err_type,
            )
            friendly = self._friendly_litert_error(err_text, err_type)
            return LLMResponse(
                content=friendly,
                tool_calls=[],
                model=self.model,
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                finish_reason="length",
            )

        content = _strip_reasoning_artifacts(content)
        tool_calls: list[ToolCall] = []
        if tools:
            content, tool_calls = _litert_extract_tool_calls(content)
        completion_tokens = self.count_tokens(content)
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            model=self.model,
            usage={
                "prompt_tokens": 0,
                "completion_tokens": completion_tokens,
                "total_tokens": completion_tokens,
            },
            finish_reason="tool_calls" if tool_calls else "stop",
        )

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,  # noqa: ARG002
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        manifest = _litert_build_tool_manifest(list(tools)) if tools else ""
        if self.rate_limiter:
            estimated = self._estimate_request_tokens(
                messages, max_tokens or self.max_tokens
            )
            await self.rate_limiter.acquire(estimated)

        # The worker client doesn't expose true token streaming yet, so
        # we just run one blocking ``send_message`` and yield the full
        # answer as a single chunk. ``_complete_via_cli`` is still below
        # as an emergency fallback but is no longer used.
        preface, last_user = self._build_preface(messages, manifest)
        try:
            content = await self._client.send_message(preface, last_user)
        except Exception as e:  # pylint: disable=broad-exception-caught
            err_text = str(e)
            err_type = type(e).__name__
            log.error(
                "litert-lm complete_streaming() failed; yielding graceful response",
                error=err_text,
                error_type=err_type,
            )
            yield self._friendly_litert_error(err_text, err_type)
            return

        content = _strip_reasoning_artifacts(content)
        if tools:
            # Strip the inline tool-call fences before streaming the
            # remainder to the user — the agent loop will run the actual
            # call via complete() on its own (it always re-issues with
            # tools when finish_reason == "tool_calls").
            content, _ = _litert_extract_tool_calls(content)
        if content:
            yield content

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    @staticmethod
    def _friendly_litert_error(err_text: str, err_type: str) -> str:
        """Turn a worker-client error into a user-visible message.

        Recognises three buckets:
          1. **Timeout / crash** — the worker hung or died (most common
             cause: KV-cache overflow that aborts the C++ side). The
             worker has already been killed; the next call will respawn
             a fresh child.
          2. **Overflow signalled cleanly** — the worker raised a
             readable error mentioning ``cache``/``length``/etc.
          3. **Anything else** — surfaced verbatim.
        """
        lower = (err_text or "").lower()

        # Bucket 1: timeout / crash from the worker client.
        if (
            "did not respond" in lower
            or "did not boot" in lower
            or "worker boot failed" in lower
            or "unexpected response from worker" in lower
            or "failed to spawn worker" in lower
            or "failed to enqueue request" in lower
        ):
            return (
                "[litert] Local model worker crashed or hung "
                "(usually KV-cache overflow on long conversations — "
                "Gemma-3n previews are capped at ~8192 tokens by the "
                ".litertlm file). A fresh worker will be spawned on "
                "the next message; use /clear or trim the conversation "
                "to stay under the limit."
            )

        # Bucket 2: overflow surfaced cleanly by the worker.
        looks_like_overflow = any(
            tok in lower
            for tok in (
                "kv",
                "cache",
                "max_num_tokens",
                "max_seq",
                "out of range",
                "exceed",
                "too long",
                "length",
                "capacity",
            )
        )
        if looks_like_overflow:
            return (
                "[litert] Local model ran out of context window "
                "(KV cache exhausted — Gemma-3n previews are capped at "
                "~8192 tokens by the .litertlm file). Use /clear or trim "
                "the conversation and try again."
            )

        return (
            f"[litert] Local model call failed ({err_type}): {err_text}. "
            "The conversation was preserved — try again, or /clear if it "
            "keeps failing."
        )

    async def close(self) -> None:
        # The worker client is shared across providers via the registry
        # and the child is daemonized, so it will die with the parent.
        # We deliberately do NOT call ``self._client.shutdown()`` here:
        # another provider in the same process may still be using it.
        return


def create_provider(
    provider: str = "ollama",
    model: str = "llama3.2",
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 32000,
    num_ctx: int = 160000,
    tokens_per_minute: int = 0,
    extra_headers: dict[str, str] | None = None,
) -> LLMProvider:
    """Create an LLM provider.

    Supported providers:
    - `ollama`
    - `openai` / `chatgpt`
    - `anthropic` / `claude`
    - `gemini` / `google`
    - `grok` / `xai`
    - `openrouter`
    - `litert` / `litert-lm` — local Gemma via Google's litert-lm runtime.
      Pass the model id (e.g. ``litert-community/gemma-4-E4B-it-litert-lm``)
      or an absolute path to a ``.litertlm`` file in ``base_url``. The
      model must already be present at
      ``~/.litert-lm/models/<repo>--<name>/model.litertlm``. Set
      ``LITERT_BACKEND=cpu`` env var to force CPU; defaults to GPU.
    """
    normalized = _normalize_provider_name(provider)

    if normalized == "ollama":
        return OllamaProvider(
            model=model,
            base_url=(base_url or os.getenv("OLLAMA_BASE_URL") or OLLAMA_NATIVE_BASE_URL),
            temperature=temperature,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
            api_key=api_key,
            tokens_per_minute=tokens_per_minute,
        )

    # ChatGPT Responses API path — activated when the OpenAI provider
    # has explicit ``extra_headers`` configured OR the selected model
    # is a Codex-family model (which can only be served by the ChatGPT
    # Responses endpoint, never by the regular api.openai.com/v1).
    # In the second case the ``CodexAuthManager`` will resolve the
    # Authorization + chatgpt-account-id headers at call time from
    # Flight Deck (``/fd/codex/access_token``) or ``~/.codex/auth.json``.
    if normalized == "openai" and (bool(extra_headers) or _is_codex_family_model(model)):
        return ChatGPTResponsesProvider(
            model=model,
            base_url=base_url or "https://chatgpt.com/backend-api/codex/responses",
            extra_headers=extra_headers,
            temperature=temperature,
            max_tokens=max_tokens,
            tokens_per_minute=tokens_per_minute,
        )

    if normalized in {"openai", "anthropic", "gemini", "xai", "openrouter"}:
        return LiteLLMProvider(
            provider=normalized,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            tokens_per_minute=tokens_per_minute,
            extra_headers=extra_headers,
        )

    if normalized == "litert":
        # `base_url` doubles as the model file path / model id resolver
        # for the local litert-lm runtime, since this provider has no
        # network endpoint. Falls back to `model` if base_url is empty.
        #
        # IMPORTANT: ``max_num_tokens`` is NOT the same thing as
        # Captain Claw's global ``num_ctx``. It's the *combined*
        # prompt+output KV-cache working window allocated at engine
        # construction time. The upstream Gemma-4 E4B .litertlm file
        # supports up to 32k tokens (per the HF model card:
        # "The model can support up to 32k context length."), but the
        # ``Engine(max_num_tokens=...)`` default is only 4096 — if we
        # don't override it, the engine silently clamps at 4k and then
        # segfaults the moment we feed it a larger prompt.
        #
        # We previously capped this at 8192 while chasing a different
        # bug where Captain Claw's 160k global context budget was
        # leaking into the engine. That cap was the wrong fix: 8k is
        # too small for real multi-round council sessions, and when a
        # ~12k prompt hit the engine with only 8k allocated it blew
        # past the buffer → SIGSEGV. Use the full 32k the model
        # supports and let the agent's own context manager handle
        # higher-level pruning. Override via ``LITERT_MAX_NUM_TOKENS``
        # if you're running a differently-built file.
        litert_max_num_tokens = int(
            os.getenv("LITERT_MAX_NUM_TOKENS", "32768") or 32768
        )
        return LiteRTProvider(
            model=model,
            model_path=(base_url or None),
            backend=os.getenv("LITERT_BACKEND", "gpu"),
            temperature=temperature,
            max_tokens=max_tokens,
            max_num_tokens=litert_max_num_tokens,
            tokens_per_minute=tokens_per_minute,
        )

    raise ValueError(
        f"Provider '{provider}' not supported. "
        "Use one of: ollama, openai/chatgpt, anthropic/claude, gemini/google, "
        "grok/xai, openrouter, litert."
    )


# Global provider instance
_provider: LLMProvider | None = None


def get_provider() -> LLMProvider:
    """Get the global LLM provider instance."""
    global _provider
    if _provider is None:
        from captain_claw.config import get_config

        cfg = get_config()
        normalized = _normalize_provider_name(cfg.model.provider)
        headers = cfg.provider_keys.headers_for(normalized) or None
        api_key = None if headers else (cfg.model.api_key or None)
        _provider = create_provider(
            provider=cfg.model.provider,
            model=cfg.model.model,
            temperature=cfg.model.temperature,
            max_tokens=cfg.model.max_tokens,
            num_ctx=cfg.context.max_tokens,
            api_key=api_key,
            base_url=cfg.model.base_url or None,
            tokens_per_minute=cfg.model.tokens_per_minute,
            extra_headers=headers,
        )
    return _provider


def set_provider(provider: LLMProvider) -> None:
    """Set the global LLM provider instance."""
    global _provider
    _provider = provider
