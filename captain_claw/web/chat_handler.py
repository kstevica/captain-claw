"""Chat message handler for the web UI."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.ws_utils import fire_and_forget_send
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
        # litellm has no provider mapping for ``litert/...`` model
        # strings, so calling acompletion with one raises BadRequestError
        # every single turn and spams the log. Skip task naming entirely
        # for litert — it's a cosmetic feature and not worth the noise.
        if (model or "").strip().lower().startswith("litert/"):
            log.info("Task naming: skipped (litert provider)", model=model)
            return ""

        # ChatGPTResponsesProvider authenticates via OAuth headers from
        # ~/.codex/auth.json (no api_key) and talks to chatgpt.com's
        # Responses endpoint, which litellm doesn't know about. Skip
        # task naming in that case rather than spamming auth errors.
        _has_oauth_header = bool(
            extra_headers and any(
                str(k).lower() == "authorization" for k in extra_headers.keys()
            )
        )
        if not api_key and (_has_oauth_header or "chatgpt.com" in (base_url or "")):
            log.info("Task naming: skipped (chatgpt oauth provider)", model=model)
            return ""

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
    whatsapp_waid: str | None = None,
    no_flow: bool = False,
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

    # ── Remember the originating WhatsApp chat (if any) ──
    # Lets tools like whatsapp_send_file default to "the current chat".
    if whatsapp_waid and getattr(agent, "session", None) is not None:
        try:
            agent.session.metadata["whatsapp_waid"] = whatsapp_waid
        except Exception:
            pass

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
    _has_image = False
    _has_video = False
    _VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v")
    video_attachments: list[str] = []  # auto-analyzed server-side before the turn

    # Single image (backward compat)
    if image_path:
        attachment_lines.append(f"[Attached image: {image_path}]")
        _has_image = True
    # Multiple images
    if image_paths:
        for p in image_paths:
            attachment_lines.append(f"[Attached image: {p}]")
            _has_image = True
    # Single data file (backward compat)
    if file_path:
        attachment_lines.append(f"[Attached file: {file_path}]")
        if str(file_path).lower().endswith(_VIDEO_EXTS):
            _has_video = True
            video_attachments.append(str(file_path))
    # Multiple data files
    if file_paths:
        for p in file_paths:
            attachment_lines.append(f"[Attached file: {p}]")
            if str(p).lower().endswith(_VIDEO_EXTS):
                _has_video = True
                video_attachments.append(str(p))
    # Tell the model how to actually view an image — it must call image_vision,
    # not read (binary) and not give up based on earlier failed turns.
    if _has_image:
        attachment_lines.append(
            "(To view the image(s) above you MUST call a tool: image_vision with the "
            "path, or — if you can't see images — delegate it to a multimodal peer "
            "via flight_deck with file=<path>. Never use read on an image. Never say "
            "you sent/delegated/described it unless you actually called the tool this turn.)"
        )
    # Video is analyzed deterministically server-side (see _run_agent) and the
    # analysis is injected into this turn — so we do NOT ask the model to call
    # video_vision or (worse) write its own extraction script.
    if _has_video:
        attachment_lines.append(
            "(The attached video(s) are being analyzed automatically — frames + audio "
            "transcript — and the analysis is included in this message. Use it to answer; "
            "do NOT call video_vision yourself and do NOT write any extraction script.)"
        )

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
            fire_and_forget_send(ws, _json_mod.dumps(msg, default=str))
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
    # Mark provider class so the namer can skip litellm entirely for
    # the ChatGPT/Codex OAuth path (no api_key, OAuth headers attached
    # only just-in-time inside complete()).
    _naming_provider_class = type(agent.provider).__name__

    log.info(
        "Task naming: setup",
        model=_naming_model,
        has_key=bool(_naming_api_key),
        key_prefix=(_naming_api_key[:8] + "...") if _naming_api_key else "none",
    )

    async def _name_and_store() -> None:
        if _naming_provider_class == "ChatGPTResponsesProvider":
            log.info(
                "Task naming: skipped (chatgpt oauth provider)",
                model=_naming_model,
            )
            agent._current_task_name = ""
            return
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
        video_attachments=video_attachments,
        no_flow=no_flow,
        flow_text=content,
    ))

    if is_public:
        # Store per-session so it isn't garbage-collected.
        agent._public_task = task  # type: ignore[attr-defined]
    else:
        server._active_task = task


async def _prefix_video_analysis(
    agent: Any, content: str, video_paths: list[str], send: Any,
) -> str:
    """Run video_vision server-side for each attached video and prepend the
    constructed analysis to the user message. Deterministic — the agent only
    consumes the result, it never extracts frames itself."""
    from pathlib import Path as _Path

    blocks: list[str] = []
    for vp in video_paths:
        name = _Path(vp).name
        try:
            send({"type": "status", "status": f"\U0001F3AC Analyzing attached video {name}…"})
            res = await agent._execute_tool_with_guard(
                "video_vision", {"path": vp}, interaction_label="video_autorun",
            )
        except Exception as exc:
            log.warning("Auto video analysis failed", path=vp, error=str(exc))
            blocks.append(f"[Video {name}: automatic analysis failed: {exc}]")
            continue
        if res is not None and getattr(res, "success", False):
            blocks.append(f"[Automatic analysis of attached video {name}]\n{res.content}")
        else:
            err = getattr(res, "error", "unknown error") if res is not None else "no result"
            blocks.append(f"[Video {name}: automatic analysis failed: {err}]")

    if not blocks:
        return content
    analysis = "\n\n".join(blocks)
    return (
        f"{analysis}\n\n---\n"
        "The attached video(s) have ALREADY been fully analyzed above (frames + "
        "audio transcript). Reply to the user in plain text using ONLY that "
        "analysis. Do NOT call video_vision again, do NOT write or run any script "
        "(no cv2, no ffmpeg, no shell), and do NOT save anything to a file unless "
        "the user explicitly asked you to.\n\n"
        f"{content}"
    )


async def _maybe_run_flow(agent: Any, text: str, *, is_public: bool) -> str | None:
    """Ask Flight Deck whether a Flow matches this message; return its output text
    (to relay), or None to take a normal agent turn. Best-effort; never raises."""
    text = (text or "").strip()
    if not text:
        return None
    import os as _os
    meta = getattr(getattr(agent, "session", None), "metadata", {}) or {}
    # Prefer the loopback FD_URL (set at spawn) over the public metadata URL —
    # the agent and FD share a host, so skip Caddy/TLS.
    fd_url = str(_os.environ.get("FD_URL") or _os.environ.get("FD_INTERNAL_URL") or meta.get("fd_url") or "").rstrip("/")
    if not fd_url:
        return None
    channel = "glasses" if is_public else "web"
    # Tell FD which agent this turn arrived at, so a step's `on: origin` targets
    # THIS agent (not a random pool member).
    fid = meta.get("fleet_identity") or {}
    origin_port = int(fid.get("port") or 0)
    try:
        if not origin_port:
            from captain_claw.config import get_config as _gc
            origin_port = int(getattr(_gc().web, "port", 0) or 0)
    except Exception:
        pass
    body = {
        "channel": channel, "text": text,
        "origin_host": "localhost", "origin_port": origin_port,
        "origin_name": str(fid.get("name") or ""),
    }
    try:
        import httpx
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.post(f"{fd_url}/fd/flows/evaluate", json=body)
        if r.status_code != 200:
            return None
        data = r.json() or {}
        if data.get("matched") and str(data.get("output") or "").strip():
            return str(data["output"])
    except Exception as exc:
        log.debug("flow evaluate skipped: %s", exc)
    return None


async def _run_agent(
    server: WebServer,
    ws: web.WebSocketResponse,
    agent: Any,
    content: str,
    naming_task: asyncio.Task | None = None,
    *,
    is_public: bool = False,
    public_session_id: str | None = None,
    video_attachments: list[str] | None = None,
    no_flow: bool = False,
    flow_text: str = "",
) -> None:
    """Background coroutine that drives the agent and finalises the turn."""
    import json as _json

    def _send_to_ws(msg: dict) -> None:
        """Send directly to this user's WebSocket."""
        fire_and_forget_send(ws, _json.dumps(msg, default=str))

    # Choose the right send function.
    send = _send_to_ws if is_public else (lambda msg: server._broadcast(msg))

    if is_public:
        agent._public_busy = True  # type: ignore[attr-defined]

    _video_policy_slug = None  # set when a video turn restricts script/shell tools
    try:
        if naming_task is not None:
            try:
                await asyncio.wait_for(naming_task, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass

        model_details = agent.get_runtime_model_details() if agent else {}
        model_label = f"{model_details.get('provider', '')}:{model_details.get('model', '')}" if model_details else ""

        # Flow engine: agent-handled channels (web/glasses) ask Flight Deck
        # whether a Flow trigger matches this message. If one does, FD runs it
        # and we relay its output instead of taking a normal agent turn.
        if not no_flow:
            _flow_out = await _maybe_run_flow(agent, flow_text or content, is_public=is_public)
            if _flow_out is not None:
                send({
                    "type": "chat_message", "role": "assistant",
                    "content": _flow_out, "timestamp": datetime.now(UTC).isoformat(),
                    "model": "flow",
                })
                return  # the `finally` resets busy + emits "ready"

        # Deterministic video preprocessing: when a video was attached, run
        # video_vision server-side (fixed-cadence frames + transcript + synthesis)
        # and feed the constructed analysis into the agent's turn. The agent never
        # chooses tools or counts frames. Mirrors the audio-transcription path.
        if video_attachments:
            content = await _prefix_video_analysis(agent, content, video_attachments, send)
            # The analysis is already done — block the script/shell tools for this
            # turn so the agent can't burn time + tokens writing & running its own
            # extraction or "save-to-file" scripts. Cleared in `finally` below.
            try:
                _video_policy_slug = agent._current_session_slug()
                agent.tools.set_session_policy(_video_policy_slug, {"deny": ["scripts", "shell"]})
            except Exception as exc:
                log.warning("Could not set video-turn tool policy", error=str(exc))
                _video_policy_slug = None

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
            if get_config().ui.next_steps:
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

        # Auto-extract insights (periodic trigger).
        try:
            import asyncio as _asyncio2
            from captain_claw.insights import maybe_extract_insights
            _asyncio2.create_task(maybe_extract_insights(agent, trigger="periodic"))
        except Exception:
            pass

        # Nervous system dreaming (background synthesis).
        try:
            import asyncio as _asyncio3
            from captain_claw.nervous_system import maybe_dream
            _asyncio3.create_task(maybe_dream(agent))
        except Exception:
            pass

        # Proactive intentions generator (admin only; opt-in via config).
        if not is_public:
            try:
                import asyncio as _asyncio4
                from captain_claw.intentions_generator import maybe_auto_propose
                _asyncio4.create_task(maybe_auto_propose(agent, trigger="periodic"))
            except Exception:
                pass

        # Record cognitive tempo metric (non-blocking).
        try:
            tempo = getattr(agent, "_cognitive_tempo", None)
            if tempo:
                import asyncio as _asyncio4
                from captain_claw.cognitive_metrics import get_cognitive_metrics_manager
                cm = get_cognitive_metrics_manager()
                _asyncio4.create_task(cm.record_event(
                    "tempo_detected", "tempo",
                    session_id=str(agent.session.id) if agent.session else None,
                    payload={"tempo": tempo.combined_tempo, "mode": tempo.mode,
                             "signals": tempo.signals},
                ))
        except Exception:
            pass

    except Exception as e:
        log.error("Chat error", error=str(e), public=is_public)
        send({"type": "error", "message": f"Error: {str(e)}"})
    finally:
        if _video_policy_slug is not None:
            try:
                agent.tools.clear_session_policy(_video_policy_slug)
            except Exception:
                pass
        if is_public:
            agent._public_busy = False  # type: ignore[attr-defined]
        else:
            server._busy = False
            server._active_task = None
        # Clear any /btw instructions accumulated during this task.
        if hasattr(agent, "_btw_instructions"):
            agent._btw_instructions = []
        send({"type": "status", "status": "ready"})
        # Inbound peer notifications are now drained by the serialized
        # _inbound_queue_consumer (web_server.py), which waits for _busy to
        # clear — no ad-hoc draining needed here.
