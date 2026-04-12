"""Remote LLM provider — proxies completion calls to another agent's web server.

Used by agent seats that need to use a different agent's LLM rather than
the local one. The remote agent must expose ``POST /api/llm/complete``.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

import httpx

from captain_claw.llm import LLMProvider, LLMResponse, Message, ToolDefinition
from captain_claw.logging import get_logger

_log = get_logger(__name__)


class RemoteLLMProvider(LLMProvider):
    """Proxy provider that forwards ``complete()`` calls over HTTP."""

    def __init__(self, host: str, port: int, auth: str = "", name: str = "") -> None:
        self.host = host
        self.port = port
        self.auth = auth
        self.name = name
        self._base = f"http://{host}:{port}"

    @property
    def label(self) -> str:
        return f"remote({self.host}:{self.port})"

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        url = f"{self._base}/api/llm/complete"
        if self.auth:
            url += f"?token={self.auth}"

        payload: dict[str, Any] = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                raise RuntimeError(f"remote LLM call failed: HTTP {resp.status_code} — {resp.text}")
            data = resp.json()
            if not data.get("ok"):
                raise RuntimeError(f"remote LLM error: {data.get('error', 'unknown')}")
            return LLMResponse(
                content=data.get("content", ""),
                model=data.get("model", ""),
                usage=data.get("usage", {}),
                finish_reason=data.get("finish_reason", ""),
            )

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        # Not needed for game agents — fall back to non-streaming
        resp = await self.complete(messages, tools, temperature, max_tokens)
        yield resp.content

    def count_tokens(self, text: str) -> int:
        return len(text) // 4  # rough estimate
