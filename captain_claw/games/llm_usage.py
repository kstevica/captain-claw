"""LLM usage tracking for game subsystem.

Wires game LLM calls (agent decisions, NL parsing, world generation)
into the same ``record_llm_usage`` pipeline the agent framework uses,
so all token consumption appears in the usage dashboard.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from captain_claw.llm import LLMResponse, Message
from captain_claw.logging import get_logger

_log = get_logger(__name__)


async def record_game_llm_usage(
    *,
    interaction: str,
    messages: list[Message],
    response: LLMResponse,
    provider: Any,
    max_tokens: int | None = None,
    latency_ms: int = 0,
    error: bool = False,
) -> None:
    """Record a game LLM call to the usage database (fire-and-forget).

    Parameters
    ----------
    interaction : str
        Label for the call type, e.g. "game_agent_decide", "game_nl_parse",
        "game_generate_fast", "game_generate_pipeline".
    messages : list[Message]
        The messages sent to the LLM.
    response : LLMResponse
        The LLM response (contains usage, model, finish_reason).
    provider : Any
        The LLMProvider instance (has .provider attribute for provider name).
    max_tokens : int | None
        Max tokens setting used for the call.
    latency_ms : int
        Wall-clock latency in milliseconds.
    error : bool
        Whether the call resulted in an error.
    """
    try:
        from captain_claw.session import get_session_manager
        sm = get_session_manager()

        usage = response.usage or {}
        model_name = response.model or ""
        provider_name = str(getattr(provider, "provider", "") or "")
        finish_reason = response.finish_reason or ""
        content = response.content or ""

        # Estimate bytes from message content
        input_bytes = 0
        for m in messages:
            c = getattr(m, "content", None) or ""
            if isinstance(c, str):
                input_bytes += len(c.encode("utf-8", errors="replace"))
        output_bytes = len(content.encode("utf-8", errors="replace"))

        await sm.record_llm_usage(
            interaction=interaction,
            provider=provider_name,
            model=model_name,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
            cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
            cache_read_input_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            streaming=False,
            tools_enabled=False,
            max_tokens=max_tokens,
            finish_reason=finish_reason,
            error=error,
            latency_ms=latency_ms,
            task_name="captain_claw_game",
        )
    except Exception:
        pass  # Never fail the main game flow


def fire_and_forget_usage(
    *,
    interaction: str,
    messages: list[Message],
    response: LLMResponse,
    provider: Any,
    max_tokens: int | None = None,
    latency_ms: int = 0,
    error: bool = False,
) -> None:
    """Schedule usage recording as a fire-and-forget async task."""
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(record_game_llm_usage(
            interaction=interaction,
            messages=messages,
            response=response,
            provider=provider,
            max_tokens=max_tokens,
            latency_ms=latency_ms,
            error=error,
        ))
    except Exception:
        pass  # Never fail the main game flow
