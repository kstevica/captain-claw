"""Chat message handler for the web UI."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw import mail_authority
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
    origin: dict | None = None,
    surface: str | None = None,
    no_flow: bool = False,
    deny_tools: list[str] | None = None,
    no_tools: bool = False,
    no_broadcast: bool = False,
    no_next_steps: bool = False,
    no_rephrase: bool = False,
    automation: mail_authority.Authority | None = None,
    speaker_turn: str | None = None,
    speaker_grant: str | None = None,
) -> bool:
    """Process a chat message through the agent.

    The actual work is launched as a background asyncio task so that the
    WebSocket read-loop stays free to process incoming messages (most
    importantly ``cancel`` signals) while the agent is running.

    If *rewind_to* is an ISO-8601 timestamp string (from Computer history
    branching), the session's message list is truncated to only include
    messages whose timestamp is ≤ that value before the new message is
    processed.  This lets the user "fork" from an earlier point in the
    conversation.

    Returns True iff the ``_run_agent`` task was launched (it then owns the
    turn's final ``ready`` frame). Every early return is False and emits no
    ready frame — on a shared-agent member socket the speaker gate owns it.
    Existing callers ignore the return value.

    *automation* (an automated turn's :class:`mail_authority.Authority`, from
    the frame's ``automation`` marker) is bound for the turn; ``None`` is a
    human turn. A member's turn ignores it.
    """
    from captain_claw.speaker import speaker_error, speaker_key_of

    speaker_key = speaker_key_of(ws)
    if not server.agent:
        await server._send(ws, speaker_error("invalid", "Agent not initialized")
                           if speaker_key is not None
                           else {"type": "error", "message": "Agent not initialized"})
        return False

    # ── Shared-agent member: own instance, own busy flag, own sockets ──
    # Handled entirely apart from the owner path below, so a member turn
    # never reads WhatsApp/origin stamping, attachments or flows.
    if speaker_key is not None:
        return await _handle_speaker_chat(
            server, ws, content, speaker_key,
            rewind_to=rewind_to,
            no_next_steps=no_next_steps,
            no_rephrase=no_rephrase,
            speaker_turn=speaker_turn or "",
            speaker_grant=speaker_grant or "",
        )

    # ── Resolve the agent to use ─────────────────────────────────
    public_session_id: str | None = getattr(ws, "_public_session_id", None)
    is_public = bool(public_session_id)
    # Lane A is the main agent, so `lane` only changes anything for B, C, …
    lane = server.normalize_lane(getattr(ws, "_lane", ""))

    if is_public:
        # Per-session agent for public users — no global busy check.
        # Each public session has its own agent so multiple users can
        # chat concurrently.
        try:
            agent = await server._get_public_agent(public_session_id)
        except Exception as e:
            await server._send(ws, {"type": "error", "message": f"Session error: {e}"})
            return False
        # Check if this specific agent is busy.
        if getattr(agent, "_public_busy", False):
            await server._send(ws, {
                "type": "error",
                "message": "Your session is busy processing. Please wait.",
            })
            return False
        # Register the WS for this session so callbacks can reach it.
        server._public_active_ws[public_session_id] = ws
    elif lane != server.LANE_MAIN:
        # A parallel lane: its own agent, its own busy flag, its own sockets.
        # Runs concurrently with lane A and every other lane.
        try:
            agent = await server._get_lane_agent(lane)
        except Exception as e:
            await server._send(ws, {"type": "error", "message": f"Lane error: {e}"})
            return False
        if getattr(agent, "_lane_busy", False):
            await server._send(ws, {
                "type": "error",
                "message": f"Lane {lane} is busy processing another request. Please wait.",
            })
            return False
    else:
        # Admin / normal mode — use the main shared agent (lane A).
        if server._busy:
            await server._send(ws, {
                "type": "error",
                "message": "Agent is busy processing another request. Please wait.",
            })
            return False
        agent = server.agent

    # ── Remember the originating WhatsApp chat (if any) ──
    # Lets tools like whatsapp_send_file default to "the current chat".
    if whatsapp_waid and getattr(agent, "session", None) is not None:
        try:
            agent.session.metadata["whatsapp_waid"] = whatsapp_waid
        except Exception:
            pass

    # ── Stamp the durable origin so async/cron results can route back here ──
    # Explicit origin from the bridge wins; otherwise synthesize from the WAID.
    if getattr(agent, "session", None) is not None:
        try:
            from captain_claw.origin import (
                KIND_WHATSAPP,
                normalize_origin,
                set_session_origin,
            )
            norm = normalize_origin(origin)
            if norm:
                set_session_origin(agent.session, norm["kind"], norm["address"])
            elif whatsapp_waid:
                set_session_origin(agent.session, KIND_WHATSAPP, whatsapp_waid)
        except Exception:
            pass

    # ── History branching: rewind session to a prior point ──
    await _rewind_session(agent, rewind_to)

    # Build attachment prefix — supports single or multiple files.
    effective_content = content
    attachment_lines: list[str] = []
    _has_image = False
    _has_video = False
    _VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v")
    video_attachments: list[str] = []  # auto-analyzed server-side before the turn
    image_attachments: list[str] = []  # non-inline images auto-analyzed server-side
    _all_images: list[str] = []

    # Single image (backward compat)
    if image_path:
        attachment_lines.append(f"[Attached image: {image_path}]")
        _has_image = True
        _all_images.append(str(image_path))
    # Multiple images
    if image_paths:
        for p in image_paths:
            attachment_lines.append(f"[Attached image: {p}]")
            _has_image = True
            _all_images.append(str(p))
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
    # Image guidance is capability-aware. Ollama-backed agents receive the image
    # INLINE (see _convert_messages_for_ollama) — i.e. they can SEE it directly,
    # so telling them to call image_vision makes them flail on a tool they may
    # not even have. Other providers must use image_vision or delegate.
    if _has_image:
        _sees_inline = str(getattr(getattr(agent, "provider", None), "provider", "")).lower() == "ollama"
        _this_msg_only = (
            " Describe the image attached in THIS message only — do NOT reuse earlier "
            "images or remembered/previous descriptions from the conversation."
        )
        if _sees_inline:
            attachment_lines.append(
                "(The image(s) above are attached and you can SEE them directly — "
                "describe/analyze from what you see. Do NOT call image_vision, do NOT "
                "delegate, and do NOT use read; just look and answer." + _this_msg_only + ")"
            )
        else:
            # Auto-analyze server-side (a vision model or a multimodal peer) and inject
            # the description — the image mirror of the video path (_prefix_image_analysis
            # in _run_agent). The model no longer has to pick the right tool: it kept
            # grabbing the always-on `cv` tool and returning pixel stats instead of a
            # description. Now the answer is already in the turn.
            image_attachments = list(_all_images)
            attachment_lines.append(
                "(An automatic visual description of the image(s) — including any visible "
                "text — is included below. Answer the user from it; you usually need no tool. "
                "Only call image_ocr if the user needs exact/complete text extraction, or cv "
                "for a pixel task they explicitly asked for (QR, blur, diff)." + _this_msg_only + ")"
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
    if is_public or lane != server.LANE_MAIN:
        import json as _json_mod
        # A lane echoes to every socket watching it; a public session to the
        # one socket it owns.
        if is_public:
            def _send_msg(msg: dict) -> None:
                fire_and_forget_send(ws, _json_mod.dumps(msg, default=str))
        else:
            _send_msg = server._lane_send(lane)
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

    naming_task = _start_task_naming(agent, content, server._recent_prompts)
    _remember_prompt(server._recent_prompts, content)

    # The surface this turn arrived on, recorded on its opening message
    # (a bridge's socket remembers it for frames routed here without one).
    surface = surface or getattr(ws, "_claw_surface", None)
    _origin_kind = str((origin or {}).get("kind", "") or "").strip().lower()
    if surface:
        turn_channel = surface
    elif whatsapp_waid or _origin_kind == "whatsapp":
        turn_channel = "whatsapp"
    elif is_public:
        turn_channel = "public"
    else:
        turn_channel = "web"

    # Launch the heavy work as a background task.
    task = asyncio.create_task(_run_agent(
        server, ws, agent, effective_content, naming_task,
        is_public=is_public,
        turn_channel=turn_channel,
        lane=lane,
        public_session_id=public_session_id,
        video_attachments=video_attachments,
        image_attachments=image_attachments,
        no_flow=no_flow,
        deny_tools=deny_tools,
        no_tools=no_tools,
        no_broadcast=no_broadcast,
        no_next_steps=no_next_steps,
        no_rephrase=no_rephrase,
        automation=automation,
        flow_text=content,
        flow_attach={
            "image_path": image_path or (image_paths[0] if image_paths else ""),
            "video_path": video_attachments[0] if video_attachments else "",
            "file_path": file_path or (file_paths[0] if file_paths else ""),
        },
    ))

    if is_public or lane != server.LANE_MAIN:
        # Store per-session/per-lane so it isn't garbage-collected.
        agent._public_task = task  # type: ignore[attr-defined]
    else:
        server._active_task = task
    return True


async def _rewind_session(agent: Any, rewind_to: str | None) -> None:
    """History branching: truncate *agent*'s session to messages at or before
    *rewind_to* (an ISO-8601 timestamp) and persist it."""
    if not rewind_to or not agent.session:
        return
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


def _start_task_naming(agent: Any, content: str, recent_prompts: list[str]) -> asyncio.Task:
    """Name the task with a micro LLM call, concurrently with the turn.

    *recent_prompts* is read when the naming call runs (after the caller
    has recorded *content* in it), so continuations get their context.
    """
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
        # A headless FD worker (Basna/Vatra/Council/Code, or an Iskra being) has
        # no conversation to name — skip the extra, concurrent naming LLM call.
        from captain_claw.agent_reasoning_mixin import _is_fd_spawned_worker
        if _is_fd_spawned_worker() or _naming_provider_class == "ChatGPTResponsesProvider":
            agent._current_task_name = ""
            return
        name = await _generate_task_name(
            content, recent_prompts, _naming_model, _naming_api_key,
            _naming_base_url, _naming_extra_headers,
        )
        agent._current_task_name = name

    return asyncio.create_task(_name_and_store())


def _remember_prompt(recent_prompts: list[str], content: str) -> None:
    """Keep the last few non-continuation prompts for naming continuations."""
    if not _is_continuation(content):
        recent_prompts.append(content[:500])
        if len(recent_prompts) > _MAX_RECENT_PROMPTS:
            recent_prompts.pop(0)


async def _handle_speaker_chat(
    server: WebServer,
    ws: web.WebSocketResponse,
    content: str,
    speaker_key: tuple[str, str],
    *,
    rewind_to: str | None,
    no_next_steps: bool,
    no_rephrase: bool,
    speaker_turn: str,
    speaker_grant: str = "",
) -> bool:
    """A shared-agent member's chat turn on their own instance.

    Returns True iff ``_run_agent`` was launched (it then sends the single
    ``ready``/``turn_end`` frame). Busy/capacity/instance errors go out as
    ``error`` frames and return False — the speaker gate sends turn_end.
    Those early returns never touch ``agent._turn_grant``: a rejected
    second chat must not overwrite the running turn's grant.
    """
    from captain_claw.speaker import speaker_error
    from captain_claw.web_server import SpeakerCapacityError

    try:
        agent = await server._get_speaker_agent(ws._speaker_principal)
    except SpeakerCapacityError:
        await server._send(ws, speaker_error(
            "capacity", "This agent is at member capacity. Try again later.",
        ))
        return False
    except Exception as e:
        log.error("Speaker instance error", error=str(e))
        await server._send(ws, speaker_error(
            "invalid", "Couldn't open your conversation on this agent.",
        ))
        return False
    if getattr(agent, "_lane_busy", False):
        await server._send(ws, speaker_error(
            "busy", "Your previous message is still being answered.",
        ))
        return False
    # Claim the instance synchronously — no await between the check and the
    # claim, so a second frame (or a second tab) can't start a parallel turn
    # before the task below gets to run.
    agent._lane_busy = True  # type: ignore[attr-defined]
    try:
        main = server.agent
        agent._fleet_identity = getattr(main, "_fleet_identity", None)  # type: ignore[attr-defined]
        agent._fleet_instructions = getattr(main, "_fleet_instructions", "") or ""  # type: ignore[attr-defined]
        agent._fd_url = getattr(main, "_fd_url", "") or ""  # type: ignore[attr-defined]

        await _rewind_session(agent, rewind_to)

        send = server._speaker_send(speaker_key)
        send({"type": "status", "status": "thinking"})
        send({
            "type": "chat_message", "role": "user",
            "content": content,
            "timestamp": datetime.now(UTC).isoformat(),
        })

        recent = getattr(agent, "_recent_prompts", None)
        if not isinstance(recent, list):
            recent = []
            agent._recent_prompts = recent  # type: ignore[attr-defined]
        naming_task = _start_task_naming(agent, content, recent)
        _remember_prompt(recent, content)

        task = asyncio.create_task(_run_agent(
            server, ws, agent, content, naming_task,
            lane=speaker_key[1],
            no_flow=True,
            no_next_steps=no_next_steps,
            no_rephrase=no_rephrase,
            flow_text=content,
            speaker_key=speaker_key,
            speaker_turn=speaker_turn,
            speaker_grant=speaker_grant,
            turn_channel="member",
        ))
        agent._public_task = task  # type: ignore[attr-defined]
        return True
    except BaseException:
        agent._lane_busy = False  # type: ignore[attr-defined]
        raise


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


async def _prefix_image_analysis(
    agent: Any, content: str, image_paths: list[str], send: Any,
) -> str:
    """Describe attached image(s) server-side and prepend it to the user message —
    the image mirror of ``_prefix_video_analysis``. Routes like ``video_vision``
    does internally: a locally-configured vision model if there is one, otherwise a
    multimodal peer over Flight Deck. When neither exists it injects an explicit
    "couldn't see it" note so the model tells the truth instead of hallucinating.

    This is what actually fixes the failure the naming/prompt work only nudged: a
    weak, non-vision model no longer has to *choose* image_vision over the always-on
    `cv` tool — the description is already in the turn.
    """
    from pathlib import Path as _Path

    from captain_claw.tools.image_ocr import ImageVisionTool

    kwargs = {"_agent": agent, "_session": getattr(agent, "session", None)}
    prompt = (
        "Describe this image in detail for someone who cannot see it. Include: how "
        "many people are present (count them), the main objects, any visible text "
        "(quote it), and what is happening."
    )

    # Resolve the vision path once (not per image): local model, else a peer.
    has_local = ImageVisionTool()._find_model() is not None
    peer = fdt = fd_url = None
    if not has_local:
        from captain_claw.tools.video_vision import _find_vision_peer

        peer = _find_vision_peer(kwargs)
        if peer:
            from captain_claw.tools.flight_deck import FlightDeckTool

            fdt = FlightDeckTool()
            fd_url = fdt._get_fd_url(**kwargs)
            if not fd_url:
                peer = None  # no way to reach the peer → fall through to the note

    blocks: list[str] = []
    for ip in image_paths:
        name = _Path(ip).name
        try:
            send({"type": "status", "status": f"\U0001F5BC️ Analyzing attached image {name}…"})
        except Exception:
            pass
        try:
            if has_local:
                res = await agent._execute_tool_with_guard(
                    "image_vision", {"path": ip, "prompt": prompt},
                    interaction_label="image_autorun",
                )
                if res is not None and getattr(res, "success", False):
                    desc = res.content
                else:
                    err = getattr(res, "error", "unknown error") if res is not None else "no result"
                    desc = f"(automatic analysis failed: {err})"
            elif peer:
                from captain_claw.tools.video_vision import _describe_frame_via_peer

                desc = await _describe_frame_via_peer(fdt, fd_url, peer, _Path(ip), prompt, kwargs)
            else:
                desc = (
                    "(could not be analyzed — this session has no vision model or multimodal "
                    "peer, so the image can't be seen here. Tell the user that plainly; do NOT "
                    "guess what it shows.)"
                )
        except Exception as exc:
            log.warning("Auto image analysis failed", path=ip, error=str(exc))
            desc = f"(automatic analysis failed: {exc})"
        blocks.append(f"[Automatic analysis of attached image {name}]\n{desc}")

    if not blocks:
        return content
    analysis = "\n\n".join(blocks)
    return (
        f"{analysis}\n\n---\n"
        "The attached image(s) were described above by a vision model (the description "
        "includes any visible text). Answer the user from that description — for "
        "'what/who/how many/what does it say' you need no further tool. Only call "
        "image_ocr if the user needs exact/complete text beyond what's quoted, or cv "
        "for an explicit pixel task (QR decode, blur/quality, diff). Do NOT re-describe "
        "via image_vision, and do NOT use the cv tool to 'read' or 'understand' it.\n\n"
        f"{content}"
    )


async def _maybe_run_flow(
    agent: Any, text: str, *, is_public: bool, attach: dict | None = None, automated: str = "",
    automated_mail_write: str = "",
) -> dict | None:
    """Ask Flight Deck whether a Flow matches this message.

    Returns None to take a normal agent turn, or a dict:
      {"output": text}  → relay this text, end the turn (inline simple flow);
                           plus ``"member_private"`` when the run returned
                           members' private data (PR D)
      {"deferred": True} → flow took over (runs in FD bg, delivers via channel);
                           end the turn silently. Also covers resuming a paused
                           input step. Best-effort; never raises.

    *automated* (the bound automation kind; "" for a human turn) tells FD the
    trigger text wasn't typed by a person, so it never counts as a request
    for email. *automated_mail_write* (that turn's ``mail_write``) goes with
    it: a flow started by a ``deny`` turn may not write email in any step."""
    attach = attach or {}
    has_attach = any(attach.get(k) for k in ("image_path", "video_path", "audio_path", "file_path"))
    # Need either text or an attachment to be worth evaluating.
    text = (text or "").strip()
    if not text and not has_attach:
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
        "image_path": str(attach.get("image_path") or ""),
        "video_path": str(attach.get("video_path") or ""),
        "audio_path": str(attach.get("audio_path") or ""),
    }
    if not automated:
        # A caller that didn't say (the /flow slash command): the bound
        # authority — in a worker or a being that's the deny default.
        _cur = mail_authority.current()
        if _cur.mode == "automated":
            automated, automated_mail_write = _cur.kind, _cur.mail_write
    if automated:
        body["automated"] = automated
        body["automated_mail_write"] = automated_mail_write or "deny"
    try:
        import httpx
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.post(f"{fd_url}/fd/flows/evaluate", json=body)
        if r.status_code != 200:
            return None
        data = r.json() or {}
        if not data.get("matched"):
            return None
        # Deferred: the flow runs in FD's background and delivers via the channel
        # (agent chat-push). The agent must end its turn WITHOUT a normal reply.
        if data.get("deferred"):
            return {"deferred": True}
        out = str(data.get("output") or "").strip()
        if out:
            _private = data.get("member_private")
            if _private in ("data", "content"):
                return {"output": out, "member_private": _private}
            return {"output": out}
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
    lane: str = "A",
    public_session_id: str | None = None,
    video_attachments: list[str] | None = None,
    image_attachments: list[str] | None = None,
    no_flow: bool = False,
    deny_tools: list[str] | None = None,
    no_tools: bool = False,
    no_broadcast: bool = False,
    no_next_steps: bool = False,
    no_rephrase: bool = False,
    automation: mail_authority.Authority | None = None,
    flow_text: str = "",
    flow_attach: dict | None = None,
    speaker_key: tuple[str, str] | None = None,
    speaker_turn: str = "",
    speaker_grant: str = "",
    turn_channel: str | None = None,
) -> None:
    """Background coroutine that drives the agent and finalises the turn.

    With *speaker_key* (a shared-agent member's turn) the member's principal
    is bound for the whole turn, output goes only to that member's sockets,
    flows never run, and the final ``ready`` frame carries
    ``turn_end=speaker_turn`` — exactly one per launched turn, on every path.

    A2: *speaker_grant* (Flight Deck's per-turn grant for this message) is
    bound for the turn only — cleared before the post-turn jobs and again in
    ``finally`` before the lane is freed — and the turn and its post-turn
    jobs count as member work in flight (``speaker.identity_lost``).

    The mail-write authority is bound for the whole turn: *automation* when
    the frame carried one, else ``interactive(<the message as received>)``
    (a human turn, or a worker's / being's automated deny default).
    """
    import json as _json

    def _send_to_ws(msg: dict) -> None:
        """Send directly to this user's WebSocket."""
        fire_and_forget_send(ws, _json.dumps(msg, default=str))

    # Choose the right send function.
    # no_broadcast (flow consult): reply ONLY to the requesting socket, never
    # broadcast to the agent's channels/UI — prevents double-delivery when the
    # step runs on a channel-connected agent (e.g. the WhatsApp origin agent).
    # A lane streams to every socket watching that lane; lane A (and anything
    # with no lane) still broadcasts, because lane A IS the main agent.
    # A member's turn is a side lane of its own, whatever its lane letter.
    _is_side_lane = bool(speaker_key) or lane != server.LANE_MAIN
    if speaker_key:
        send = server._speaker_send(speaker_key)
        no_flow = True
    elif is_public or no_broadcast:
        send = _send_to_ws
    elif _is_side_lane:
        send = server._lane_send(lane)
    else:
        send = lambda msg: server._broadcast(msg)

    if no_rephrase:
        agent._suppress_rephrase = True  # type: ignore[attr-defined]

    if is_public:
        agent._public_busy = True  # type: ignore[attr-defined]
    elif _is_side_lane:
        agent._lane_busy = True  # type: ignore[attr-defined]

    # Who started this turn — read by the mail-write guard and soft checks.
    # A frame without a marker is a person typing — except in an FD-spawned
    # worker or a being, where the process default (deny) stays (J7).
    _auth_tok = mail_authority.bind(
        automation if automation is not None else mail_authority.interactive(flow_text or content)
    )
    _video_policy_slug = None  # set when a video turn restricts script/shell tools
    _speaker_tok = None
    _grant_tok = None
    _counted = False
    try:
        if speaker_key:
            from captain_claw import speaker as _speaker
            _speaker_tok = _speaker.bind(getattr(agent, "_speaker_principal", None))
            _grant_tok = _speaker.bind_grant(speaker_grant)
            agent._turn_grant = _speaker.sanitize_grant(speaker_grant)  # type: ignore[attr-defined]
            _speaker.turn_started()
            _counted = True
            # Commons caches (insights, intuitions) are refreshed for member
            # turns when stale; the owner's caches never are.
            import time as _time
            _now = _time.monotonic()
            _refreshed = float(getattr(agent, "_speaker_cache_refreshed_at", 0.0) or 0.0)
            if _now - _refreshed > _speaker.SPEAKER_CACHE_REFRESH_S:
                for _refresh in ("_refresh_insights_context_cache", "_refresh_nervous_system_cache"):
                    try:
                        await getattr(agent, _refresh)()
                    except Exception:
                        pass
                agent._speaker_cache_refreshed_at = _now  # type: ignore[attr-defined]

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
            _turn_auth = mail_authority.current()
            _automated = _turn_auth.mode == "automated"
            _flow = await _maybe_run_flow(
                agent, flow_text or content, is_public=is_public, attach=flow_attach,
                automated=_turn_auth.kind if _automated else "",
                automated_mail_write=_turn_auth.mail_write if _automated else "",
            )
            if _flow is not None:
                # Inline output → relay it. Deferred → the flow delivers its own
                # messages asynchronously via /api/chat/push; end the turn quietly.
                if _flow.get("output"):
                    # PR D: the level marks the frame (FD's consult / delegate
                    # relays add the header); the text a person reads stays plain.
                    send({
                        "type": "chat_message", "role": "assistant",
                        "content": _flow["output"], "timestamp": datetime.now(UTC).isoformat(),
                        "model": "flow",
                        **({"member_private": _flow["member_private"]} if _flow.get("member_private") else {}),
                    })
                return  # the `finally` resets busy + emits "ready"

        # Provenance: an automated turn is marked as not typed by the user.
        if automation is not None:
            content = mail_authority.automated_prefix() + "\n" + content

        # Deterministic video preprocessing: when a video was attached, run
        # video_vision server-side (fixed-cadence frames + transcript + synthesis)
        # and feed the constructed analysis into the agent's turn. The agent never
        # chooses tools or counts frames. Mirrors the audio-transcription path.
        if video_attachments:
            content = await _prefix_video_analysis(agent, content, video_attachments, send)

        # Deterministic image preprocessing (mirror of video): when a non-inline
        # image was attached, describe it server-side (a vision model, else a
        # multimodal peer) and inject the description. The agent never has to pick
        # image_vision vs the pixel-only `cv` tool — the answer is already in-turn.
        if image_attachments:
            content = await _prefix_image_analysis(agent, content, image_attachments, send)

        # Deterministic per-turn tool denials (guardrails that do NOT rely on the
        # model obeying instructions). Cleared in `finally` below.
        #   • video turn → no scripts/shell (the analysis is already injected)
        #   • relaying a delegated result → no flight_deck/consult_peer, so the
        #     originating agent CANNOT auto-resend the task. This is the gate that
        #     stops the inter-agent resend flood: once a result (even an error)
        #     comes back, the relay turn can only relay it, never re-delegate.
        # Image-describe turn: suppress memory/insights injection so the model
        # describes the freshly-attached image instead of regurgitating a
        # remembered description of an earlier one (rich-session contamination).
        _img_turn = isinstance(content, str) and "[Attached image:" in content
        if _img_turn:
            try:
                agent._suppress_memory_context = True  # type: ignore[attr-defined]
            except Exception:
                pass

        _deny_tools: list[str] = list(deny_tools or [])  # caller-requested (e.g. consult)
        if video_attachments:
            _deny_tools += ["scripts", "shell"]
        # NB: image turns deliberately do NOT deny tools. The description is injected
        # (so describe/understand questions are already answered in-context), but a
        # user may still legitimately want image_ocr (precise text) or cv (QR, blur,
        # diff) on the same image — denying them would break those. The injection,
        # not a deny, is what stops the "grabbed cv for a describe" failure.
        if isinstance(content, str) and "[Delegated result from" in content:
            _deny_tools += ["flight_deck", "consult_peer"]
        # no_tools wins: an empty allow-list filters every tool away, so the agent
        # can only answer in text (used by reflection-only turns like the Council
        # action-points extraction, which must describe work, never execute it).
        _turn_policy: dict | None = None
        if no_tools:
            _turn_policy = {"allow": []}
        elif _deny_tools:
            _turn_policy = {"deny": sorted(set(_deny_tools))}
        if _turn_policy is not None:
            try:
                _video_policy_slug = agent._current_session_slug()
                agent.tools.set_session_policy(_video_policy_slug, _turn_policy)
            except Exception as exc:
                log.warning("Could not set per-turn tool policy", error=str(exc))
                _video_policy_slug = None

        # PR D: the level of a turn that read members' private data ("" when
        # none) — marks the final frame (FD's peer relays keep the header on
        # the relayed text) and skips the post-turn learning jobs below.
        from captain_claw import member_privacy as _member_privacy
        _private_level = ""

        # Route /orchestrate requests to the orchestrator (admin only).
        stripped = content.strip()
        if (not is_public and not speaker_key
                and stripped.lower().startswith("/orchestrate ") and server._orchestrator):
            orchestrate_input = stripped[len("/orchestrate "):].strip()
            if not orchestrate_input:
                send({"type": "error", "message": "Usage: /orchestrate <request>"})
            else:
                response = await server._orchestrator.orchestrate(orchestrate_input)
                _private_level = _member_privacy.header_level(response)
                send({
                    "type": "chat_message",
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "model": model_label,
                    **({"member_private": _private_level} if _private_level else {}),
                })
        else:
            from captain_claw import msg_origin as _msg_origin

            _msg_origin.hint_turn_provenance(agent, channel=turn_channel)
            response = await agent.complete(content)
            # Read right away, before any other await: a concurrent turn on
            # this Agent object could reset it (contract part 0c NB7). The
            # output level also covers a reply that carry-over flagged (it
            # restated member text from context without reading it again).
            _private_level = _member_privacy.output_level(agent)

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
                **({"member_private": _private_level} if _private_level else {}),
            })

            # Extract and broadcast suggested next steps — skip for FD-spawned
            # workers (Basna/Vatra/Council/Code): they're orchestrated, headless,
            # and have no interactive user to offer follow-ups to (each call is
            # also an extra LLM round-trip we don't want to spend per worker turn).
            #
            # A queue-dispatched turn is the same situation wearing a different
            # hat: the next message is already written and waiting, so asking
            # the model "what next?" buys nothing and costs a round-trip per
            # queued item. The client sets no_next_steps for those.
            from captain_claw.agent_reasoning_mixin import _is_fd_spawned_worker
            if get_config().ui.next_steps and not no_next_steps and not _is_fd_spawned_worker():
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
            # Built from the agent that just ran, and delivered to the lane
            # that ran it — lane B's header must not describe lane A.
            _info = {"type": "session_info", **server._session_info(agent)}
            if _is_side_lane:
                send(_info)
            else:
                server._broadcast(_info)

        # A member turn's grant ends HERE: the post-turn jobs below copy this
        # context (create_task) and carry the principal but never the grant —
        # they can't reach the member's Google or deep memory.
        if speaker_key:
            _speaker.clear_grant()
            agent._turn_grant = ""  # type: ignore[attr-defined]

        def _post_turn(coro: Any) -> asyncio.Task:
            """create_task for a post-turn job; a member turn's jobs count as
            member work in flight until they finish (fail closed on a lost
            context in any bare thread they spawn)."""
            task = asyncio.create_task(coro)
            if speaker_key:
                _speaker.track_member_task(task)
            return task

        # Consciousness background jobs — each an EXTRA, CONCURRENT LLM call
        # fired after the turn (create_task, not awaited).
        import os as _os_w
        from captain_claw.agent_reasoning_mixin import _is_fd_spawned_worker
        _worker = _is_fd_spawned_worker()
        _being_worker = str(_os_w.environ.get("CLAW_BEING_WORKER", "")).strip(
            ).lower() in ("1", "true", "yes")

        # Reflection / insight / dreaming / topic-classification feed each agent's
        # OWN memory and would just be parallel generations for a headless worker
        # — skip for ALL FD workers (Basna/Vatra/Council/Code + beings). A being
        # already dreams and reflects through its tick engine.
        # On a member's turn these still run (open commons, A2 part 0 N8).
        # PR D: a turn that read members' private data feeds no shared learnings (U3).
        _private = bool(_private_level)
        if not _worker and not _private:
            # Auto-reflection (admin only).
            if not is_public:
                try:
                    from captain_claw.reflections import maybe_auto_reflect
                    _post_turn(maybe_auto_reflect(agent))
                except Exception:
                    pass

            # Auto-extract insights (periodic trigger).
            try:
                from captain_claw.insights import maybe_extract_insights
                _post_turn(maybe_extract_insights(agent, trigger="periodic"))
            except Exception:
                pass

            # Nervous system dreaming (background synthesis).
            try:
                from captain_claw.nervous_system import maybe_dream
                _post_turn(maybe_dream(agent))
            except Exception:
                pass

            # Conversation topic classification (background; clusters comms
            # traffic into persistent topics recalled via the `topics` tool).
            try:
                from captain_claw.conversation_topics import maybe_classify_topics
                _post_turn(maybe_classify_topics(agent))
            except Exception:
                pass

        # Proactive intentions — a BEING's autonomous nervous system: the way it
        # surfaces observations/proposals to its parent on its own. Kept for
        # beings (and normal agents) but NOT other FD task workers. Internally
        # throttled (cooldown + max/day + quiet hours), so it fires rarely, not
        # the per-faculty-call thrash — one occasional generation is acceptable.
        # Never for a member's turn: proposals go to the owner's channels.
        if (not _worker or _being_worker) and not is_public and not speaker_key and not _private:
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
        log.error("Chat error", error=str(e), public=is_public, speaker=bool(speaker_key))
        if speaker_key:
            # A member never sees the exception text: it can carry the
            # owner's provider errors, LLM base URLs or key fragments.
            from captain_claw.speaker import TURN_FAILED_MESSAGE, speaker_error

            send(speaker_error("invalid", TURN_FAILED_MESSAGE))
        else:
            send({"type": "error", "message": f"Error: {str(e)}"})
    finally:
        try:
            agent._suppress_memory_context = False  # type: ignore[attr-defined]
            agent._suppress_rephrase = False  # type: ignore[attr-defined]
        except Exception:
            pass
        if _video_policy_slug is not None:
            try:
                agent.tools.clear_session_policy(_video_policy_slug)
            except Exception:
                pass
        # A member turn's grant and in-flight count end BEFORE the lane is
        # freed: once it is, a new chat may set `_turn_grant` for its turn.
        if speaker_key:
            from captain_claw import speaker as _spk_end

            try:
                agent._turn_grant = ""  # type: ignore[attr-defined]
            except Exception:
                pass
            if _grant_tok is not None:
                _spk_end.reset_grant(_grant_tok)
            if _counted:
                _spk_end.turn_ended()
        if is_public:
            agent._public_busy = False  # type: ignore[attr-defined]
        elif _is_side_lane:
            agent._lane_busy = False  # type: ignore[attr-defined]
        else:
            server._busy = False
            server._active_task = None
        # Clear any /btw instructions accumulated during this task.
        if hasattr(agent, "_btw_instructions"):
            agent._btw_instructions = []
        send({
            "type": "status", "status": "ready",
            **({"turn_end": speaker_turn} if speaker_key else {}),
        })
        if _speaker_tok is not None:
            _speaker.reset(_speaker_tok)
        mail_authority.reset(_auth_tok)
        # Inbound peer notifications are now drained by the serialized
        # _inbound_queue_consumer (web_server.py), which waits for _busy to
        # clear — no ad-hoc draining needed here.
