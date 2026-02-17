"""Ollama provider - direct HTTP calls to Ollama API."""

import asyncio
import json
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


class OllamaProvider(LLMProvider):
    """Direct Ollama API provider."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = OLLAMA_NATIVE_BASE_URL,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        api_key: str | None = None,
    ):
        """Initialize Ollama provider.
        
        Args:
            model: Ollama model name (e.g., 'llama3.2', 'qwen3:32b')
            base_url: Ollama API base URL
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            api_key: Optional API key (Ollama usually doesn't need one locally)
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        
        self.client = httpx.AsyncClient(
            timeout=120.0,
            follow_redirects=True,
        )

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert messages to Ollama format."""
        result = []
        
        for msg in messages:
            # Handle both Message objects and dicts - fix operator precedence!
            if isinstance(msg, dict):
                role = msg.get('role')
                content = msg.get('content')
                tool_name = msg.get('tool_name')
            else:
                role = getattr(msg, 'role', None)
                content = getattr(msg, 'content', None)
                tool_name = getattr(msg, 'tool_name', None)
            
            if role == "system":
                result.append({"role": "system", "content": content or ""})
            elif role == "user":
                result.append({"role": "user", "content": content or ""})
            elif role == "assistant":
                result.append({"role": "assistant", "content": content or ""})
            elif role == "tool":
                entry = {"role": "tool", "content": content or ""}
                if tool_name:
                    entry["tool_name"] = tool_name
                result.append(entry)
        
        return result

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert tools to Ollama format."""
        result = []
        for tool in tools:
            # Handle both ToolDefinition objects and dicts
            if isinstance(tool, dict):
                name = tool.get('name')
                description = tool.get('description', '')
                parameters = tool.get('parameters', {})
            else:
                name = getattr(tool, 'name', None)
                description = getattr(tool, 'description', '') or ""
                parameters = getattr(tool, 'parameters', None) or {}
            
            if name:
                result.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description or "",
                        "parameters": parameters or {},
                    },
                })
        return result

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a completion."""
        url = f"{self.base_url}/api/chat"
        
        ollama_messages = self._convert_messages(messages)
        ollama_tools = self._convert_tools(tools) if tools else None
        
        # Build options
        options: dict[str, Any] = {
            "num_ctx": 65536,  # Large context window
            "temperature": temperature or self.temperature,
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
            log.debug("Calling Ollama", model=self.model, url=url, msg_count=len(ollama_messages))
            
            response = await self.client.post(url, json=body, headers=headers)
            
            log.debug("Ollama response status", status=response.status_code)
            
            if not response.is_success:
                error_text = response.text
                raise LLMAPIError(
                    f"Ollama API error {response.status_code}: {error_text}",
                    status_code=response.status_code,
                )
            
            data = response.json()
            
            # Extract content
            content = data.get("message", {}).get("content", "")
            
            # Extract tool calls
            tool_calls = []
            if data.get("message", {}).get("tool_calls"):
                for tc in data["message"]["tool_calls"]:
                    tool_calls.append(ToolCall(
                        id=f"ollama_call_{tc.get('id', '')}",
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", {}),
                    ))
            
            # Extract usage
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": (data.get("prompt_eval_count", 0) + data.get("eval_count", 0)),
            }
            
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                model=self.model,
                usage=usage,
            )
            
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
        """Stream a completion."""
        url = f"{self.base_url}/api/chat"
        
        ollama_messages = self._convert_messages(messages)
        ollama_tools = self._convert_tools(tools) if tools else None
        
        options: dict[str, Any] = {
            "num_ctx": 65536,
            "temperature": temperature or self.temperature,
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
                
                accumulated_content = ""
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        if chunk.get("message", {}).get("content"):
                            content = chunk["message"]["content"]
                            accumulated_content += content
                            yield content
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
                        
        except httpx.HTTPError as e:
            raise LLMAPIError(f"Ollama streaming error: {e}")
        except Exception as e:
            raise LLMError(f"Ollama stream failed: {e}")

    def count_tokens(self, text: str) -> int:
        """Count tokens (rough estimate for Ollama models)."""
        # Rough estimate: ~1 token per 4 characters for English
        return len(text) // 4

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


def create_provider(
    provider: str = "ollama",
    model: str = "llama3.2",
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> LLMProvider:
    """Create an LLM provider.
    
    Args:
        provider: Provider name (ollama, openai, anthropic)
        model: Model name
        api_key: Optional API key
        base_url: Optional base URL
        temperature: Default temperature
        max_tokens: Default max tokens
    
    Returns:
        Configured LLMProvider instance
    """
    if provider == "ollama":
        # Default base URL for Ollama
        default_base = base_url or OLLAMA_NATIVE_BASE_URL
        return OllamaProvider(
            model=model,
            base_url=default_base,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
    else:
        # Fallback to LiteLLM for other providers
        # For non-Ollama providers, require explicit setup
        raise ValueError(f"Provider '{provider}' not supported. Use 'ollama' or configure manually.")


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
            api_key=cfg.model.api_key or None,
        )
    return _provider


def set_provider(provider: LLMProvider) -> None:
    """Set the global LLM provider instance."""
    global _provider
    _provider = provider
