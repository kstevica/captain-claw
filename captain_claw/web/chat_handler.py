"""Chat message handler for the web UI."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

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
    image_paths: list[str] | None = None,
    file_paths: list[str] | None = None,
    rewind_to: str | None = None,
) -> None:
    """Process a chat message through the agent.

    The actual work is launched as a background asyncio task so that the
    WebSocket read-loop stays free to process incoming messages (most
    importantly ``cancel`` signals) while the agent is running.

    If *rewind_to* is an ISO-8601 timestamp string (from Computer history
    branching), the session's message list is truncated to only include
    messages whose timestamp is ≤ that value before the new message is
    processed.  This lets the user "fork" from an earlier point in the
    conversation.
    """
    if not server.agent:
        await server._send(ws, {"type": "error", "message": "Agent not initialized"})
        return

    # ── Resolve the agent to use ─────────────────────────────────
    public_session_id: str | None = getattr(ws, "_public_session_id", None)
    is_public = bool(public_session_id)

    if is_public:
        # Per-session agent for public users — no global busy check.
        # Each public session has its own agent so multiple users can
        # chat concurrently.
        try:
            agent = await server._get_public_agent(public_session_id)
        except Exception as e:
            await server._send(ws, {"type": "error", "message": f"Session error: {e}"})
            return
        # Check if this specific agent is busy.
        if getattr(agent, "_public_busy", False):
            await server._send(ws, {
                "type": "error",
                "message": "Your session is busy processing. Please wait.",
            })
            return
        # Register the WS for this session so callbacks can reach it.
        server._public_active_ws[public_session_id] = ws
    else:
        # Admin / normal mode — use the main shared agent.
        if server._busy:
            await server._send(ws, {
                "type": "error",
                "message": "Agent is busy processing another request. Please wait.",
            })
            return
        agent = server.agent

    # ── History branching: rewind session to a prior point ──
    if rewind_to and agent.session:
        session = agent.session
        before = len(session.messages)
        session.messages = [
            m for m in session.messages
            if (m.get("timestamp") or "") <= rewind_to
        ]
        after = len(session.messages)
        if before != after:
            log.info(
                "Session rewound for history branch",
                before=before, after=after, rewind_to=rewind_to,
            )
            try:
                from captain_claw.session import get_session_manager
                sm = get_session_manager()
                await sm.save_session(session)
            except Exception as e:
                log.warning("Failed to persist rewound session", error=str(e))

    # Build attachment prefix — supports single or multiple files.
    effective_content = content
    attachment_lines: list[str] = []

    # Single image (backward compat)
    if image_path:
        attachment_lines.append(f"[Attached image: {image_path}]")
    # Multiple images
    if image_paths:
        for p in image_paths:
            attachment_lines.append(f"[Attached image: {p}]")
    # Single data file (backward compat)
    if file_path:
        attachment_lines.append(f"[Attached file: {file_path}]")
    # Multiple data files
    if file_paths:
        for p in file_paths:
            attachment_lines.append(f"[Attached file: {p}]")

    if attachment_lines:
        prefix = "\n".join(attachment_lines) + "\n"
        default_msg = (
            "Please analyze these files."
            if len(attachment_lines) > 1
            else ("Please analyze this image." if (image_path or image_paths) else "I've attached a file.")
        )
        effective_content = prefix + (content or default_msg)

    # ── Send to the right targets ────────────────────────────────
    # For public users we send directly to their WS; for admin we
    # broadcast to all admin connections.
    if is_public:
        import json as _json_mod
        def _send_msg(msg: dict) -> None:
            if not ws.closed:
                asyncio.ensure_future(ws.send_str(
                    _json_mod.dumps(msg, default=str)
                ))
        _send_msg({"type": "status", "status": "thinking"})
        _send_msg({
            "type": "chat_message", "role": "user",
            "content": effective_content,
            "timestamp": datetime.now(UTC).isoformat(),
        })
    else:
        server._busy = True
        server._broadcast({"type": "status", "status": "thinking"})
        server._thinking_callback("Thinking\u2026", phase="reasoning")
        server._broadcast({
            "type": "chat_message", "role": "user",
            "content": effective_content,
            "timestamp": datetime.now(UTC).isoformat(),
        })

    # ── Task naming (runs concurrently with the agent) ────────────
    if not hasattr(server, "_recent_prompts"):
        server._recent_prompts: list[str] = []

    _naming_model = getattr(agent.provider, "model", "")
    _naming_provider = getattr(agent.provider, "provider", "")
    if _naming_model and "/" not in _naming_model and _naming_provider:
        _naming_model = f"{_naming_provider}/{_naming_model}"
    _naming_api_key = getattr(agent.provider, "api_key", None)
    _naming_base_url = getattr(agent.provider, "base_url", None)
    _naming_extra_headers = getattr(agent.provider, "extra_headers", None)

    log.info(
        "Task naming: setup",
        model=_naming_model,
        has_key=bool(_naming_api_key),
        key_prefix=(_naming_api_key[:8] + "...") if _naming_api_key else "none",
    )

    async def _name_and_store() -> None:
        name = await _generate_task_name(
            content, server._recent_prompts, _naming_model, _naming_api_key,
            _naming_base_url, _naming_extra_headers,
        )
        agent._current_task_name = name

    naming_task = asyncio.create_task(_name_and_store())

    if not _is_continuation(content):
        server._recent_prompts.append(content[:500])
        if len(server._recent_prompts) > _MAX_RECENT_PROMPTS:
            server._recent_prompts.pop(0)

    # Launch the heavy work as a background task.
    task = asyncio.create_task(_run_agent(
        server, ws, agent, effective_content, naming_task,
        is_public=is_public,
        public_session_id=public_session_id,
    ))

    if is_public:
        # Store per-session so it isn't garbage-collected.
        agent._public_task = task  # type: ignore[attr-defined]
    else:
        server._active_task = task


async def _run_agent(
    server: WebServer,
    ws: web.WebSocketResponse,
    agent: Any,
    content: str,
    naming_task: asyncio.Task | None = None,
    *,
    is_public: bool = False,
    public_session_id: str | None = None,
) -> None:
    """Background coroutine that drives the agent and finalises the turn."""
    import json as _json

    def _send_to_ws(msg: dict) -> None:
        """Send directly to this user's WebSocket."""
        if not ws.closed:
            asyncio.ensure_future(ws.send_str(_json.dumps(msg, default=str)))

    # Choose the right send function.
    send = _send_to_ws if is_public else (lambda msg: server._broadcast(msg))

    if is_public:
        agent._public_busy = True  # type: ignore[attr-defined]

    try:
        if naming_task is not None:
            try:
                await asyncio.wait_for(naming_task, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass

        model_details = agent.get_runtime_model_details() if agent else {}
        model_label = f"{model_details.get('provider', '')}:{model_details.get('model', '')}" if model_details else ""

        # Route /orchestrate requests to the orchestrator (admin only).
        stripped = content.strip()
        if not is_public and stripped.lower().startswith("/orchestrate ") and server._orchestrator:
            orchestrate_input = stripped[len("/orchestrate "):].strip()
            if not orchestrate_input:
                send({"type": "error", "message": "Usage: /orchestrate <request>"})
            else:
                response = await server._orchestrator.orchestrate(orchestrate_input)
                send({
                    "type": "chat_message",
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "model": model_label,
                })
        else:
            response = await agent.complete(content)

            log.info(
                "Agent complete() returned",
                response_len=len(response) if response else 0,
                response_preview=(response[:200] if response else "<empty>"),
                public=is_public,
            )

            send({
                "type": "chat_message",
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now(UTC).isoformat(),
                "model": model_label,
            })

            # Extract and broadcast suggested next steps.
            try:
                steps = await extract_next_steps(agent.provider, response)
                if steps:
                    send({
                        "type": "next_steps",
                        "options": next_steps_to_dicts(steps),
                    })
            except Exception as ns_err:
                log.debug("Next steps extraction error", error=str(ns_err))

        # Send updated usage/session info.
        send({
            "type": "usage",
            "last": agent.last_usage,
            "total": agent.total_usage,
            "context_window": agent.last_context_window,
        })

        if not is_public:
            server._broadcast({
                "type": "session_info",
                **server._session_info(),
            })

        # Auto-reflection (admin only).
        if not is_public:
            try:
                import asyncio as _asyncio
                from captain_claw.reflections import maybe_auto_reflect
                _asyncio.create_task(maybe_auto_reflect(agent))
            except Exception:
                pass

    except Exception as e:
        log.error("Chat error", error=str(e), public=is_public)
        send({"type": "error", "message": f"Error: {str(e)}"})
    finally:
        if is_public:
            agent._public_busy = False  # type: ignore[attr-defined]
        else:
            server._busy = False
            server._active_task = None
        # Clear any /btw instructions accumulated during this task.
        if hasattr(agent, "_btw_instructions"):
            agent._btw_instructions = []
        send({"type": "status", "status": "ready"})
