"""LLM providers (Ollama + LiteLLM-backed providers)."""

from __future__ import annotations

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

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
    }
    return aliases.get(key, key)


def _provider_model_name(provider: str, model: str) -> str:
    """Ensure model name includes provider prefix for LiteLLM."""
    cleaned = (model or "").strip()
    if not cleaned:
        return cleaned
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
    # Fallback: provider_keys from config (settings UI).
    try:
        from captain_claw.config import get_config
        pk = get_config().provider_keys
        pk_map = {"openai": pk.openai, "anthropic": pk.anthropic, "gemini": pk.gemini, "xai": pk.xai}
        pk_val = str(pk_map.get(provider, "") or "").strip()
        if pk_val:
            return pk_val
    except Exception:
        pass
    return None


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
        if role == "tool" and tool_call_id:
            entry["tool_call_id"] = tool_call_id
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
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    prompt_tokens = int(_obj_get(usage_obj, "prompt_tokens", 0) or 0)
    completion_tokens = int(_obj_get(usage_obj, "completion_tokens", 0) or 0)
    total_tokens = int(_obj_get(usage_obj, "total_tokens", 0) or 0)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


# ── ChatGPT Responses API helpers ─────────────────────────────────────────


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
    }
    return mapping.get(base, base)


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
                "call_id": str(tool_call_id or ""),
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
    ):
        import uuid

        self.provider = "openai"
        self.model = _normalize_chatgpt_model(model)
        self.base_url = base_url.rstrip("/")
        self.extra_headers = extra_headers or {}
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session_id = uuid.uuid4().hex
        self.client = httpx.AsyncClient(timeout=600.0, follow_redirects=True)
        self.rate_limiter = (
            TokenRateLimiter(tokens_per_minute) if tokens_per_minute > 0 else None
        )

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
        if instructions:
            payload["instructions"] = instructions
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

        for item in output_items:
            item_type = item.get("type", "")

            if item_type == "message":
                for part in item.get("content", []) or []:
                    text = part.get("text", "")
                    if text:
                        content_parts.append(str(text))

            elif item_type == "function_call":
                call_id = str(item.get("call_id", "") or item.get("id", ""))
                name = str(item.get("name", ""))
                raw_args = item.get("arguments", "{}")
                args = _safe_json_loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                tool_calls.append(ToolCall(id=call_id, name=name, arguments=args))

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
        headers = self._build_headers()

        try:
            collected_lines: list[str] = []
            async with self.client.stream("POST", self.base_url, json=payload, headers=headers) as response:
                if not response.is_success:
                    error_text = await response.aread()
                    raise LLMAPIError(
                        f"ChatGPT Responses API error {response.status_code}: {error_text.decode(errors='replace')}",
                        status_code=response.status_code,
                    )
                async for line in response.aiter_lines():
                    collected_lines.append(line)

            events = self._parse_sse_events(collected_lines)

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
        headers = self._build_headers()

        try:
            async with self.client.stream("POST", self.base_url, json=payload, headers=headers) as response:
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

        try:
            response = await self.client.post(url, json=body, headers=headers)
            if not response.is_success:
                raise LLMAPIError(
                    f"Ollama API error {response.status_code}: {response.text}",
                    status_code=response.status_code,
                )
            data = response.json()
            content = _obj_get(_obj_get(data, "message", {}), "content", "") or ""

            tool_calls: list[ToolCall] = []
            raw_calls = _obj_get(_obj_get(data, "message", {}), "tool_calls", []) or []
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
            raise LLMAPIError(f"Ollama HTTP error: {e}")
        except json.JSONDecodeError as e:
            raise LLMError(f"Ollama response decode error: {e}")
        except Exception as e:
            raise LLMError(f"Ollama call failed: {e}")

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
        except LLMAPIError:
            raise
        except httpx.HTTPError as e:
            raise LLMAPIError(f"Ollama streaming error: {e}")
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

    async def _collect_streaming_response(self, stream: Any) -> dict[str, Any]:
        """Collect an async streaming response into a unified response dict.

        Some providers (notably Gemini via LiteLLM) may return a streaming
        object even when ``stream=False``.  This helper iterates the stream
        and reassembles the chunks into the standard non-streaming format
        expected by :meth:`complete`.
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

                # Content text — may be a string or a list of parts
                # (Gemini sometimes returns list of content parts).
                c = _obj_get(delta, "content", "")
                if c:
                    if isinstance(c, str):
                        content_parts.append(c)
                    elif isinstance(c, list):
                        for part in c:
                            text = _obj_get(part, "text", "")
                            if text:
                                content_parts.append(str(text))
                    else:
                        content_parts.append(str(c))

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
        return {
            "choices": [{
                "message": {
                    "content": "".join(content_parts),
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
                delta = _obj_get(_obj_get(_obj_get(chunk, "choices", [{}])[0], "delta", {}), "content", "")
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

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            from litellm import token_counter

            return int(token_counter(model=self.model, messages=[{"role": "user", "content": text}]))
        except Exception:
            return len(text) // 4


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

    # ChatGPT Responses API path — activated when openai + extra_headers.
    if normalized == "openai" and extra_headers:
        return ChatGPTResponsesProvider(
            model=model,
            base_url=base_url or "https://chatgpt.com/backend-api/codex/responses",
            extra_headers=extra_headers,
            temperature=temperature,
            max_tokens=max_tokens,
            tokens_per_minute=tokens_per_minute,
        )

    if normalized in {"openai", "anthropic", "gemini", "xai"}:
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

    raise ValueError(
        f"Provider '{provider}' not supported. "
        "Use one of: ollama, openai/chatgpt, anthropic/claude, gemini/google, grok/xai."
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
