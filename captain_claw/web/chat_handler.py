"""Chat message handler for the web UI."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw.logging import get_logger
from captain_claw.next_steps import extract_next_steps, next_steps_to_dicts

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)

# ── Task naming helpers ──────────────────────────────────────────────

# Patterns that indicate the user wants to continue the previous task
# rather than starting a new one.
_CONTINUATION_RE = re.compile(
    r"^("
    r"continue|go\s*on|more|keep\s*going|proceed|next|"
    r"go\s*ahead|do\s*it|yes|ok|okay|sure|yep|yea|yeah|"
    r"sounds?\s*good|that'?s?\s*(fine|good|great|correct|right)|"
    r"perfect|exactly|confirmed?"
    r")[\s!.\-,]*$",
    re.IGNORECASE,
)

# Recent user prompts kept for context when naming continuations.
_MAX_RECENT_PROMPTS = 3


def _is_continuation(text: str) -> bool:
    """Return True if *text* looks like a continuation/affirmation."""
    stripped = text.strip().rstrip("!.,")
    return bool(_CONTINUATION_RE.match(stripped)) and len(stripped) < 60


async def _generate_task_name(
    user_text: str,
    recent_prompts: list[str],
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    extra_headers: dict | None = None,
) -> str:
    """Fire a micro LLM call to name the task in ≤6 words.

    *recent_prompts* provides context for continuation messages.
    Uses the cheapest/fastest model available via litellm.
    """
    try:
        from litellm import acompletion

        # Build the naming prompt.
        if _is_continuation(user_text) and recent_prompts:
            # Combine last few prompts so the namer knows the real task.
            history = "\n".join(f"- {p}" for p in recent_prompts[-_MAX_RECENT_PROMPTS:])
            user_block = (
                f"Recent user messages:\n{history}\n\n"
                f"Latest message (continuation): {user_text}"
            )
        else:
            user_block = user_text

        log.info(
            "Task naming: calling LLM",
            model=model,
            has_api_key=bool(api_key),
            user_text_len=len(user_block),
            is_continuation=_is_continuation(user_text),
        )

        kwargs: dict = dict(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Name the user's task in 5-6 words max. "
                        "Reply ONLY with the short name, no quotes, no punctuation."
                    ),
                },
                {"role": "user", "content": user_block},
            ],
            max_tokens=25,
            temperature=0.0,
            timeout=8,
        )
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["api_base"] = base_url
        if extra_headers:
            kwargs["extra_headers"] = extra_headers

        resp = await acompletion(**kwargs)
        name = (resp.choices[0].message.content or "").strip().strip('"\'.')
        # Safety: truncate if the model got chatty
        if len(name) > 60:
            name = name[:60].rsplit(" ", 1)[0]
        log.info("Task naming: result", task_name=name)
        return name
    except Exception as e:
        log.warning("Task naming failed", error=str(e), error_type=type(e).__name__)
        return ""


async def handle_chat(
    server: WebServer,
    ws: web.WebSocketResponse,
    content: str,
    *,
    image_path: str | None = None,
    file_path: str | None = None,
) -> None:
    """Process a chat message through the agent.

    The actual work is launched as a background asyncio task so that the
    WebSocket read-loop stays free to process incoming messages (most
    importantly ``cancel`` signals) while the agent is running.
    """
    if not server.agent:
        await server._send(ws, {"type": "error", "message": "Agent not initialized"})
        return

    if server._busy:
        await server._send(ws, {
            "type": "error",
            "message": "Agent is busy processing another request. Please wait.",
        })
        return

    # When an image is attached, prepend the path context so the agent
    # knows a file is available for tools like image_ocr.
    effective_content = content
    if image_path:
        prefix = f"[Attached image: {image_path}]\n"
        effective_content = prefix + (content or "Please analyze this image.")

    # When a data file is attached, prepend the path context so the agent
    # knows a file is available. The user's message determines what to do
    # with it (datastore import, deep memory indexing, extraction, etc.).
    if file_path:
        prefix = f"[Attached file: {file_path}]\n"
        effective_content = prefix + (content or "I've attached a file.")

    # Mark busy *before* spawning the task so that a second chat message
    # arriving immediately is rejected.
    server._busy = True
    server._broadcast({"type": "status", "status": "thinking"})
    server._thinking_callback("Thinking\u2026", phase="reasoning")

    # Echo user message to all clients
    server._broadcast({
        "type": "chat_message",
        "role": "user",
        "content": effective_content,
        "timestamp": datetime.now(UTC).isoformat(),
    })

    # ── Task naming (runs concurrently with the agent) ────────────
    # Maintain a short history of user prompts so continuations can
    # be named in context.
    if not hasattr(server, "_recent_prompts"):
        server._recent_prompts: list[str] = []

    # Use the same provider/model/key/base_url the agent is configured with.
    _naming_model = getattr(server.agent.provider, "model", "")
    _naming_api_key = getattr(server.agent.provider, "api_key", None)
    _naming_base_url = getattr(server.agent.provider, "base_url", None)
    _naming_extra_headers = getattr(server.agent.provider, "extra_headers", None)

    log.info(
        "Task naming: setup",
        model=_naming_model,
        has_key=bool(_naming_api_key),
        key_prefix=(_naming_api_key[:8] + "...") if _naming_api_key else "none",
    )

    # Kick off the naming coroutine as a fire-and-forget task.
    # It writes the result to server.agent._current_task_name which
    # _record_usage_to_db reads when persisting.
    async def _name_and_store() -> None:
        name = await _generate_task_name(
            content, server._recent_prompts, _naming_model, _naming_api_key,
            _naming_base_url, _naming_extra_headers,
        )
        if server.agent:
            server.agent._current_task_name = name

    naming_task = asyncio.create_task(_name_and_store())

    # Update prompt history (only keep the non-continuation ones as
    # meaningful context; always append the raw text for detection).
    if not _is_continuation(content):
        server._recent_prompts.append(content[:500])
        if len(server._recent_prompts) > _MAX_RECENT_PROMPTS:
            server._recent_prompts.pop(0)

    # Launch the heavy work as a background task — do NOT await it here
    # so the caller (the WebSocket read-loop) can keep processing
    # incoming messages such as cancel/stop.
    task = asyncio.create_task(_run_agent(server, effective_content, naming_task))

    # Store a reference so it isn't garbage-collected.
    server._active_task = task


async def _run_agent(
    server: WebServer,
    content: str,
    naming_task: asyncio.Task | None = None,
) -> None:
    """Background coroutine that drives the agent and finalises the turn."""
    try:
        # Wait for the task naming micro-call to finish (should be fast).
        if naming_task is not None:
            try:
                await asyncio.wait_for(naming_task, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass  # naming is best-effort

        # Capture model details before running the agent.
        model_details = server.agent.get_runtime_model_details() if server.agent else {}
        model_label = f"{model_details.get('provider', '')}:{model_details.get('model', '')}" if model_details else ""

        # Route /orchestrate requests to the orchestrator.
        stripped = content.strip()
        if stripped.lower().startswith("/orchestrate ") and server._orchestrator:
            orchestrate_input = stripped[len("/orchestrate "):].strip()
            if not orchestrate_input:
                server._broadcast({
                    "type": "error",
                    "message": "Usage: /orchestrate <request>",
                })
            else:
                response = await server._orchestrator.orchestrate(orchestrate_input)
                server._broadcast({
                    "type": "chat_message",
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "model": model_label,
                })
        else:
            # Use complete() which handles tool calls and guards
            response = await server.agent.complete(content)

            server._broadcast({
                "type": "chat_message",
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now(UTC).isoformat(),
                "model": model_label,
            })

            # Extract and broadcast suggested next steps.
            try:
                steps = await extract_next_steps(server.agent.provider, response)
                if steps:
                    server._broadcast({
                        "type": "next_steps",
                        "options": next_steps_to_dicts(steps),
                    })
            except Exception as ns_err:
                log.debug("Next steps extraction error", error=str(ns_err))

        # Send updated usage/session info
        server._broadcast({
            "type": "usage",
            "last": server.agent.last_usage,
            "total": server.agent.total_usage,
            "context_window": server.agent.last_context_window,
        })
        server._broadcast({
            "type": "session_info",
            **server._session_info(),
        })

    except Exception as e:
        log.error("Chat error", error=str(e))
        server._broadcast({
            "type": "error",
            "message": f"Error: {str(e)}",
        })
    finally:
        server._busy = False
        server._active_task = None
        server._broadcast({"type": "status", "status": "ready"})
