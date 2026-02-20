"""Prompt execution and queue dispatch helpers.

All functions receive a :class:`RuntimeContext` instead of closing over
the mutable state that used to live inside ``run_interactive()``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from captain_claw.config import get_config
from captain_claw.execution_queue import (
    CommandLane,
    FollowupRun,
    QueueSettings,
    normalize_queue_drop_policy,
    normalize_queue_mode,
    resolve_global_lane,
    resolve_session_lane,
)
from captain_claw.logging import log

if TYPE_CHECKING:
    from captain_claw.runtime_context import RuntimeContext


# ---------------------------------------------------------------------------
# Queue / lane helpers
# ---------------------------------------------------------------------------

async def enqueue_agent_task(
    ctx: RuntimeContext,
    session_id: str | None,
    task: Callable[[], Awaitable[Any]],
    *,
    lane: str = CommandLane.MAIN,
    warn_after_ms: int = 2_000,
) -> Any:
    resolved_session_lane = resolve_session_lane((session_id or "").strip() or "default")
    resolved_global_lane = resolve_global_lane(lane)
    return await ctx.command_queue.enqueue_in_lane(
        resolved_session_lane,
        lambda: ctx.command_queue.enqueue_in_lane(
            resolved_global_lane,
            lambda: ctx.command_queue.enqueue_in_lane(
                CommandLane.AGENT_RUNTIME,
                task,
                warn_after_ms=warn_after_ms,
            ),
            warn_after_ms=warn_after_ms,
        ),
        warn_after_ms=warn_after_ms,
    )


def _safe_int(value: object, default: int, minimum: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except Exception:
        return default
    return max(minimum, parsed)


async def resolve_queue_settings_for_session(
    ctx: RuntimeContext, session_id: str,
) -> QueueSettings:
    cfg_queue = get_config().execution_queue
    agent = ctx.agent
    session = (
        agent.session
        if agent.session and agent.session.id == session_id
        else await agent.session_manager.load_session(session_id)
    )
    queue_meta: dict[str, object] = {}
    if session and isinstance(session.metadata.get("queue"), dict):
        queue_meta = dict(session.metadata.get("queue") or {})
    mode = (
        normalize_queue_mode(str(queue_meta.get("mode", "")).strip())
        or normalize_queue_mode(str(getattr(cfg_queue, "mode", "")).strip())
        or "collect"
    )
    drop_policy = (
        normalize_queue_drop_policy(str(queue_meta.get("drop_policy", "")).strip())
        or normalize_queue_drop_policy(str(getattr(cfg_queue, "drop", "")).strip())
        or "summarize"
    )
    debounce_ms = _safe_int(
        queue_meta.get("debounce_ms", getattr(cfg_queue, "debounce_ms", 1000)),
        default=1000, minimum=0,
    )
    cap = _safe_int(
        queue_meta.get("cap", getattr(cfg_queue, "cap", 20)),
        default=20, minimum=1,
    )
    return QueueSettings(mode=mode, debounce_ms=debounce_ms, cap=cap, drop_policy=drop_policy)


async def wait_until_session_idle(ctx: RuntimeContext, session_id: str) -> None:
    session_lane = resolve_session_lane(session_id)
    while ctx.command_queue.get_queue_size(session_lane) > 0:
        await asyncio.sleep(0.05)


def queue_meta(session_obj: object) -> dict[str, object]:
    if not hasattr(session_obj, "metadata"):
        return {}
    metadata = getattr(session_obj, "metadata", {})
    if not isinstance(metadata, dict):
        return {}
    raw = metadata.get("queue")
    if isinstance(raw, dict):
        return raw
    return {}


async def update_active_session_queue_settings(
    ctx: RuntimeContext,
    *,
    mode: str | None = None,
    debounce_ms: int | None = None,
    cap: int | None = None,
    drop_policy: str | None = None,
) -> tuple[bool, str]:
    agent = ctx.agent
    if not agent.session:
        return False, "No active session"
    q_meta = dict(queue_meta(agent.session))
    if mode is not None:
        normalized_mode = normalize_queue_mode(mode)
        if not normalized_mode:
            return False, "Invalid queue mode. Use steer|followup|collect|steer-backlog|interrupt|queue."
        q_meta["mode"] = normalized_mode
    if debounce_ms is not None:
        q_meta["debounce_ms"] = max(0, int(debounce_ms))
    if cap is not None:
        q_meta["cap"] = max(1, int(cap))
    if drop_policy is not None:
        normalized_drop = normalize_queue_drop_policy(drop_policy)
        if not normalized_drop:
            return False, "Invalid queue drop policy. Use old|new|summarize."
        q_meta["drop_policy"] = normalized_drop
    q_meta["updated_at"] = datetime.now().isoformat()
    agent.session.metadata["queue"] = q_meta
    await agent.session_manager.save_session(agent.session)
    settings = await resolve_queue_settings_for_session(ctx, agent.session.id)
    return (
        True,
        (
            "Session queue settings updated: "
            f"mode={settings.mode} debounce_ms={settings.debounce_ms} "
            f"cap={settings.cap} drop={settings.drop_policy}"
        ),
    )


# ---------------------------------------------------------------------------
# Prompt execution
# ---------------------------------------------------------------------------

async def run_cancellable(ui: Any, work: Awaitable[Any]) -> tuple[Any, bool]:
    """Run work and cancel on ESC."""
    work_task = asyncio.create_task(work)
    esc_task = asyncio.create_task(ui.wait_for_escape()) if ui.can_capture_escape() else None
    try:
        if esc_task is None:
            return await work_task, False

        done, _ = await asyncio.wait(
            {work_task, esc_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if esc_task in done:
            ui.append_system_line("ESC pressed, cancelling current action")
            ui.set_runtime_status("waiting")
            work_task.cancel()
            try:
                await work_task
            except asyncio.CancelledError:
                pass
            return None, True

        esc_task.cancel()
        try:
            await esc_task
        except asyncio.CancelledError:
            pass
        return await work_task, False
    finally:
        if esc_task and not esc_task.done():
            esc_task.cancel()


async def run_prompt_in_active_session(
    ctx: RuntimeContext,
    prompt_text: str,
    *,
    display_prompt: str | None = None,
    cron_job_id: str | None = None,
    cron_trigger: str | None = None,
    cron_source: str | None = None,
    raise_on_error: bool = False,
    lane: str = CommandLane.MAIN,
    queue: bool = True,
    on_assistant_text: Callable[[str], Awaitable[None]] | None = None,
    after_turn: Callable[[int, str, str], Awaitable[None]] | None = None,
) -> None:
    """Execute one user prompt using the currently selected session."""
    if not prompt_text.strip():
        return

    # Lazy import to avoid circular dependency
    from captain_claw.cron_dispatch import cron_chat_event, cron_monitor_event

    agent = ctx.agent
    ui = ctx.ui

    async def _execute() -> None:
        shown_prompt = display_prompt if display_prompt is not None else prompt_text
        ui.print_message("user", shown_prompt)
        ui.print_blank_line()
        turn_start_idx = len(agent.session.messages) if agent.session else 0
        if cron_job_id:
            await cron_chat_event(
                ctx, cron_job_id, "user", prompt_text,
                trigger=cron_trigger or "", source=cron_source or "",
            )

        started = time.perf_counter()
        assistant_text = ""
        try:
            ui.set_runtime_status("thinking")
            if get_config().ui.streaming:
                ui.begin_assistant_stream()
                ui.set_runtime_status("streaming")
                chunks: list[str] = []

                async def _consume_stream() -> None:
                    async for chunk in agent.stream(prompt_text):
                        chunks.append(chunk)
                        ui.print_streaming(chunk)
                    ui.complete_stream_line()

                try:
                    _, cancelled = await run_cancellable(ui, _consume_stream())
                finally:
                    ui.end_assistant_stream()
                if cancelled:
                    if cron_job_id:
                        await cron_chat_event(
                            ctx, cron_job_id, "system", "[CRON] prompt cancelled",
                            trigger=cron_trigger or "", source=cron_source or "",
                        )
                    ui.print_blank_line()
                    return
                assistant_text = "".join(chunks)
            else:
                response, cancelled = await run_cancellable(ui, agent.complete(prompt_text))
                if cancelled:
                    if cron_job_id:
                        await cron_chat_event(
                            ctx, cron_job_id, "system", "[CRON] prompt cancelled",
                            trigger=cron_trigger or "", source=cron_source or "",
                        )
                    ui.print_blank_line()
                    return
                assistant_text = response or ""
                ui.print_message("assistant", response)
                ui.print_blank_line()

            if cron_job_id:
                from captain_claw.session_export import truncate_history_text

                await cron_chat_event(
                    ctx, cron_job_id, "assistant", assistant_text,
                    trigger=cron_trigger or "", source=cron_source or "",
                )
                await cron_monitor_event(
                    ctx, "prompt_assistant_output",
                    history_job_id=cron_job_id,
                    trigger=cron_trigger or "", source=cron_source or "",
                    output=truncate_history_text(assistant_text),
                )
            if on_assistant_text:
                outbound_text = assistant_text.strip()
                if not outbound_text and agent.session:
                    for msg in reversed(agent.session.messages):
                        if str(msg.get("role", "")).strip().lower() != "assistant":
                            continue
                        candidate = str(msg.get("content", "")).strip()
                        if candidate:
                            outbound_text = candidate
                            break
                if not outbound_text:
                    outbound_text = "Task completed. Check monitor output for details."
                await on_assistant_text(outbound_text)
            if after_turn:
                await after_turn(turn_start_idx, prompt_text, assistant_text)
            ctx.last_exec_seconds = time.perf_counter() - started
            ctx.last_completed_at = datetime.now()
            ui.set_runtime_status("waiting")
        except Exception as e:
            ctx.last_exec_seconds = time.perf_counter() - started
            ctx.last_completed_at = datetime.now()
            ui.set_runtime_status("waiting")
            if cron_job_id:
                await cron_chat_event(
                    ctx, cron_job_id, "system", f"[CRON] prompt failed: {e}",
                    trigger=cron_trigger or "", source=cron_source or "",
                )
            ui.print_error(str(e))
            log.error("Error in agent", error=str(e))
            if on_assistant_text:
                try:
                    await on_assistant_text(f"Error: {e}")
                except Exception:
                    pass
            if raise_on_error:
                raise

    if not queue:
        await _execute()
        return

    session_id = agent.session.id if agent.session else "default"
    await enqueue_agent_task(ctx, session_id, _execute, lane=lane)


# ---------------------------------------------------------------------------
# Followup queue dispatch
# ---------------------------------------------------------------------------

async def run_queued_followup_prompt(ctx: RuntimeContext, run: FollowupRun) -> None:
    payload = dict(run.metadata)
    session_id = str(payload.get("session_id", "")).strip()
    prompt_text = str(run.prompt or "").strip()
    if not session_id or not prompt_text:
        return
    await run_prompt_in_session(
        ctx,
        session_id=session_id,
        prompt_text=prompt_text,
        source=str(payload.get("source", "followup")).strip() or "followup",
        cron_job_id=str(payload.get("cron_job_id", "")).strip() or None,
        trigger=str(payload.get("trigger", "scheduled")).strip() or "scheduled",
    )


async def dispatch_prompt_in_session(
    ctx: RuntimeContext,
    session_id: str,
    prompt_text: str,
    source: str,
    *,
    cron_job_id: str | None = None,
    trigger: str = "scheduled",
    dedupe_mode: str = "prompt",
) -> str:
    from captain_claw.cron_dispatch import cron_chat_event, cron_monitor_event

    session_lane = resolve_session_lane(session_id)
    is_busy = ctx.command_queue.get_queue_size(session_lane) > 0
    has_followup_backlog = ctx.followup_queue.get_queue_depth(session_id) > 0
    if not is_busy and not has_followup_backlog:
        await run_prompt_in_session(
            ctx,
            session_id=session_id,
            prompt_text=prompt_text,
            source=source,
            cron_job_id=cron_job_id,
            trigger=trigger,
        )
        return "executed"

    settings = await resolve_queue_settings_for_session(ctx, session_id)
    if settings.mode == "interrupt":
        lane_cleared = ctx.command_queue.clear_lane(session_lane)
        followup_cleared = ctx.followup_queue.clear_queue(session_id)
        await cron_monitor_event(
            ctx, "followup_interrupt",
            history_job_id=cron_job_id,
            session_id=session_id,
            lane_cleared=lane_cleared,
            followup_cleared=followup_cleared,
            source=source,
        )
        await cron_chat_event(
            ctx, cron_job_id, "system",
            (
                f"[CRON] followup interrupt requested for session={session_id} "
                f"(cleared lane={lane_cleared}, followup={followup_cleared})"
            ),
            trigger=trigger, source=source,
        )

    queued = ctx.followup_queue.enqueue_followup(
        session_id,
        FollowupRun(
            prompt=prompt_text,
            enqueued_at_ms=int(asyncio.get_running_loop().time() * 1000),
            message_id=(cron_job_id or ""),
            summary_line=prompt_text[:180],
            metadata={
                "session_id": session_id,
                "source": source,
                "trigger": trigger,
                "cron_job_id": cron_job_id or "",
            },
        ),
        settings,
        dedupe_mode=dedupe_mode if dedupe_mode in {"message-id", "prompt", "none"} else "prompt",
    )
    if not queued:
        await cron_monitor_event(
            ctx, "followup_skipped",
            history_job_id=cron_job_id,
            session_id=session_id,
            source=source,
            reason="deduplicated_or_drop_policy",
        )
        return "skipped"

    ctx.followup_queue.schedule_drain(
        session_id,
        lambda run: run_queued_followup_prompt(ctx, run),
        wait_until_ready=lambda: wait_until_session_idle(ctx, session_id),
    )
    await cron_monitor_event(
        ctx, "followup_queued",
        history_job_id=cron_job_id,
        session_id=session_id,
        source=source,
        mode=settings.mode,
        queued_depth=ctx.followup_queue.get_queue_depth(session_id),
    )
    await cron_chat_event(
        ctx, cron_job_id, "system",
        f"[CRON] queued followup for busy session={session_id} mode={settings.mode}",
        trigger=trigger, source=source,
    )
    return "queued"


async def run_prompt_in_session(
    ctx: RuntimeContext,
    session_id: str,
    prompt_text: str,
    source: str,
    *,
    cron_job_id: str | None = None,
    trigger: str = "scheduled",
) -> None:
    from captain_claw.cron_dispatch import cron_chat_event, cron_monitor_event

    agent = ctx.agent
    selected = await agent.session_manager.load_session(session_id)
    if not selected:
        raise ValueError(f"Session not found: {session_id}")

    async def _execute() -> None:
        previous_session = agent.session
        previous_session_id = previous_session.id if previous_session else None
        switched = previous_session_id != selected.id
        if switched:
            agent.session = selected
            agent.refresh_session_runtime_flags()
        try:
            await cron_monitor_event(
                ctx, "prompt_start",
                history_job_id=cron_job_id,
                source=source, session_id=selected.id,
            )
            ctx.ui.print_message(
                "system",
                f"[CRON] {trigger} prompt run in session={selected.id} job={cron_job_id or 'oneoff'}",
            )
            if agent.session:
                start_note = f"[CRON] {trigger} prompt start: {source}"
                agent.session.add_message(
                    role="tool", content=start_note, tool_name="cron",
                    tool_arguments={
                        "step": "prompt_start", "trigger": trigger,
                        "source": source, "job_id": cron_job_id or "",
                    },
                    token_count=agent._count_tokens(start_note),
                )
                await agent.session_manager.save_session(agent.session)
            await run_prompt_in_active_session(
                ctx, prompt_text,
                display_prompt=f"[CRON job={cron_job_id or 'oneoff'}] {prompt_text}",
                cron_job_id=cron_job_id,
                cron_trigger=trigger,
                cron_source=source,
                raise_on_error=bool(cron_job_id),
                lane=CommandLane.CRON,
                queue=False,
            )
            await cron_monitor_event(
                ctx, "prompt_done",
                history_job_id=cron_job_id,
                source=source, session_id=selected.id,
            )
            if agent.session:
                done_note = f"[CRON] {trigger} prompt done: {source}"
                agent.session.add_message(
                    role="tool", content=done_note, tool_name="cron",
                    tool_arguments={
                        "step": "prompt_done", "trigger": trigger,
                        "source": source, "job_id": cron_job_id or "",
                    },
                    token_count=agent._count_tokens(done_note),
                )
                await agent.session_manager.save_session(agent.session)
        except Exception as e:
            if agent.session:
                fail_note = f"[CRON] {trigger} prompt failed: {e}"
                agent.session.add_message(
                    role="tool", content=fail_note, tool_name="cron",
                    tool_arguments={
                        "step": "prompt_failed", "trigger": trigger,
                        "source": source, "job_id": cron_job_id or "",
                    },
                    token_count=agent._count_tokens(fail_note),
                )
                await agent.session_manager.save_session(agent.session)
            raise
        finally:
            if switched and previous_session is not None:
                restored = await agent.session_manager.load_session(previous_session_id)
                agent.session = restored or previous_session
                agent.refresh_session_runtime_flags()

    await enqueue_agent_task(ctx, selected.id, _execute, lane=CommandLane.CRON)
