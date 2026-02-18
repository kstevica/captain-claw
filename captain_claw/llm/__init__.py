"""LLM providers (Ollama + LiteLLM-backed providers)."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
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


@dataclass
class ToolDefinition:
    """Definition of a tool for the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

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
    """Resolve provider API key from explicit value or environment."""
    if explicit_api_key:
        return explicit_api_key
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY") or None
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY") or None
    if provider == "gemini":
        return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or None
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
    ):
        self.provider = "ollama"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = max(1, int(num_ctx))
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=120.0, follow_redirects=True)

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
            return LLMResponse(
                content=str(content),
                tool_calls=tool_calls,
                model=str(_obj_get(data, "model", self.model) or self.model),
                usage=usage,
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
        }
        if tools:
            kwargs["tools"] = _convert_tools_for_openai_style(tools)
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url
        return kwargs

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

        try:
            response = await acompletion(
                **self._request_kwargs(
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
            )
            choice = _obj_get(_obj_get(response, "choices", [{}])[0], "message", {})
            content = _obj_get(choice, "content", "") or ""

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

            return LLMResponse(
                content=str(content),
                tool_calls=tool_calls,
                model=str(_obj_get(response, "model", self.model) or self.model),
                usage=_extract_usage(_obj_get(response, "usage", None)),
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
) -> LLMProvider:
    """Create an LLM provider.

    Supported providers:
    - `ollama`
    - `openai` / `chatgpt`
    - `anthropic` / `claude`
    - `gemini` / `google`
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
        )

    if normalized in {"openai", "anthropic", "gemini"}:
        return LiteLLMProvider(
            provider=normalized,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(
        f"Provider '{provider}' not supported. "
        "Use one of: ollama, openai/chatgpt, anthropic/claude, gemini/google."
    )


# Global provider instance
_provider: LLMProvider | None = None


def get_provider() -> LLMProvider:
    """Get the global LLM provider instance."""
    global _provider
    if _provider is None:
        from captain_claw.config import get_config

        cfg = get_config()
        _provider = create_provider(
            provider=cfg.model.provider,
            model=cfg.model.model,
            temperature=cfg.model.temperature,
            max_tokens=cfg.model.max_tokens,
            num_ctx=cfg.context.max_tokens,
            api_key=cfg.model.api_key or None,
            base_url=cfg.model.base_url or None,
        )
    return _provider


def set_provider(provider: LLMProvider) -> None:
    """Set the global LLM provider instance."""
    global _provider
    _provider = provider
