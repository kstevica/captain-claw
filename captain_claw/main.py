"""Main entry point for Captain Claw."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import TypeVar

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from captain_claw.config import Config, get_config, set_config
from captain_claw.cron import compute_next_run, now_utc, parse_schedule_tokens, schedule_to_text, to_utc_iso
from captain_claw.logging import configure_logging, log, set_system_log_sink
from captain_claw.cli import TerminalUI, get_ui
from captain_claw.execution_queue import (
    CommandLane,
    CommandQueueManager,
    FollowupQueueManager,
    FollowupRun,
    QueueSettings,
    normalize_queue_drop_policy,
    normalize_queue_mode,
    resolve_global_lane,
    resolve_session_lane,
)
from captain_claw.agent import Agent

T = TypeVar("T")


async def _run_cancellable(ui: TerminalUI, work: Awaitable[T]) -> tuple[T | None, bool]:
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


def main(
    config: str = "",
    model: str = "",
    provider: str = "",
    no_stream: bool = False,
    verbose: bool = False,
) -> None:
    """Start Captain Claw interactive session."""
    ui = get_ui()
    set_system_log_sink(ui.append_system_line if ui.has_sticky_layout() else None)

    # Configure logging first
    if verbose:
        os.environ["CLAW_LOGGING__LEVEL"] = "DEBUG"
    configure_logging()
    
    # Load configuration
    if config:
        try:
            cfg = Config.from_yaml(Path(config))
        except Exception as e:
            log.error("Failed to load config", error=str(e))
            cfg = Config.load()
    else:
        cfg = Config.load()
    
    # Apply CLI overrides
    if model:
        cfg.model.model = model
    if provider:
        cfg.model.provider = provider
    if no_stream:
        cfg.ui.streaming = False
    
    # Set global config
    set_config(cfg)
    
    # Ensure session directory exists
    session_path = Path(cfg.session.path).expanduser()
    session_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_path = cfg.resolved_workspace_path(Path.cwd())
    workspace_path.mkdir(parents=True, exist_ok=True)
    
    # Run the interactive loop
    try:
        asyncio.run(run_interactive())
    except KeyboardInterrupt:
        log.info("Shutting down...")
        sys.exit(0)
    except Exception as e:
        log.error("Fatal error", error=str(e))
        sys.exit(1)


async def run_interactive() -> None:
    """Run the interactive agent loop."""
    ui = get_ui()
    ui.set_monitor_full_output(bool(get_config().ui.monitor_full_output))
    agent = Agent(
        status_callback=ui.set_runtime_status,
        tool_output_callback=ui.append_tool_output,
        approval_callback=ui.confirm,
    )
    
    # Show welcome
    ui.print_welcome()
    
    # Initialize agent
    await agent.initialize()
    ui.set_monitor_mode(True)
    if agent.session:
        ui.load_monitor_tool_output_from_session(agent.session.messages)
    ui.set_runtime_status("user input")
    last_exec_seconds: float | None = None
    last_completed_at: datetime | None = None
    command_queue = CommandQueueManager()
    followup_queue = FollowupQueueManager()
    cron_running_job_ids: set[str] = set()
    cron_poll_seconds = 2.0

    def _normalize_session_id(raw: str) -> str:
        safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in (raw or "").strip())
        safe = safe.strip("-")
        return safe or "default"

    async def _enqueue_agent_task(
        session_id: str | None,
        task: Callable[[], Awaitable[T]],
        *,
        lane: str = CommandLane.MAIN,
        warn_after_ms: int = 2_000,
    ) -> T:
        resolved_session_lane = resolve_session_lane((session_id or "").strip() or "default")
        resolved_global_lane = resolve_global_lane(lane)
        return await command_queue.enqueue_in_lane(
            resolved_session_lane,
            lambda: command_queue.enqueue_in_lane(
                resolved_global_lane,
                lambda: command_queue.enqueue_in_lane(
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

    async def _resolve_queue_settings_for_session(session_id: str) -> QueueSettings:
        cfg_queue = get_config().execution_queue
        session = agent.session if agent.session and agent.session.id == session_id else await agent.session_manager.load_session(session_id)
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
            default=1000,
            minimum=0,
        )
        cap = _safe_int(
            queue_meta.get("cap", getattr(cfg_queue, "cap", 20)),
            default=20,
            minimum=1,
        )
        return QueueSettings(
            mode=mode,
            debounce_ms=debounce_ms,
            cap=cap,
            drop_policy=drop_policy,
        )

    async def _wait_until_session_idle(session_id: str) -> None:
        session_lane = resolve_session_lane(session_id)
        while command_queue.get_queue_size(session_lane) > 0:
            await asyncio.sleep(0.05)

    async def _run_queued_followup_prompt(run: FollowupRun) -> None:
        payload = dict(run.metadata)
        session_id = str(payload.get("session_id", "")).strip()
        prompt_text = str(run.prompt or "").strip()
        if not session_id or not prompt_text:
            return
        await _run_prompt_in_session(
            session_id=session_id,
            prompt_text=prompt_text,
            source=str(payload.get("source", "followup")).strip() or "followup",
            cron_job_id=str(payload.get("cron_job_id", "")).strip() or None,
            trigger=str(payload.get("trigger", "scheduled")).strip() or "scheduled",
        )

    async def _dispatch_prompt_in_session(
        session_id: str,
        prompt_text: str,
        source: str,
        *,
        cron_job_id: str | None = None,
        trigger: str = "scheduled",
        dedupe_mode: str = "prompt",
    ) -> str:
        session_lane = resolve_session_lane(session_id)
        is_busy = command_queue.get_queue_size(session_lane) > 0
        has_followup_backlog = followup_queue.get_queue_depth(session_id) > 0
        if not is_busy and not has_followup_backlog:
            await _run_prompt_in_session(
                session_id=session_id,
                prompt_text=prompt_text,
                source=source,
                cron_job_id=cron_job_id,
                trigger=trigger,
            )
            return "executed"

        settings = await _resolve_queue_settings_for_session(session_id)
        if settings.mode == "interrupt":
            lane_cleared = command_queue.clear_lane(session_lane)
            followup_cleared = followup_queue.clear_queue(session_id)
            await _cron_monitor_event(
                "followup_interrupt",
                history_job_id=cron_job_id,
                session_id=session_id,
                lane_cleared=lane_cleared,
                followup_cleared=followup_cleared,
                source=source,
            )
            await _cron_chat_event(
                cron_job_id,
                "system",
                (
                    f"[CRON] followup interrupt requested for session={session_id} "
                    f"(cleared lane={lane_cleared}, followup={followup_cleared})"
                ),
                trigger=trigger,
                source=source,
            )

        queued = followup_queue.enqueue_followup(
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
            await _cron_monitor_event(
                "followup_skipped",
                history_job_id=cron_job_id,
                session_id=session_id,
                source=source,
                reason="deduplicated_or_drop_policy",
            )
            return "skipped"

        followup_queue.schedule_drain(
            session_id,
            _run_queued_followup_prompt,
            wait_until_ready=lambda: _wait_until_session_idle(session_id),
        )
        await _cron_monitor_event(
            "followup_queued",
            history_job_id=cron_job_id,
            session_id=session_id,
            source=source,
            mode=settings.mode,
            queued_depth=followup_queue.get_queue_depth(session_id),
        )
        await _cron_chat_event(
            cron_job_id,
            "system",
            f"[CRON] queued followup for busy session={session_id} mode={settings.mode}",
            trigger=trigger,
            source=source,
        )
        return "queued"

    def _cron_monitor(step: str, **args: object) -> None:
        payload: dict[str, object] = {"step": step}
        payload.update(args)
        lines = [f"step={step}"]
        for key, value in args.items():
            lines.append(f"{key}={value}")
        ui.append_tool_output("cron", payload, "\n".join(lines))

    async def _append_cron_history(
        job_id: str | None = None,
        *,
        chat_event: dict[str, object] | None = None,
        monitor_event: dict[str, object] | None = None,
    ) -> None:
        if not job_id:
            return
        await agent.session_manager.append_cron_job_history(
            job_id,
            chat_event=chat_event,
            monitor_event=monitor_event,
        )

    async def _cron_monitor_event(step: str, history_job_id: str | None = None, **args: object) -> None:
        _cron_monitor(step, **args)
        monitor_event: dict[str, object] = {"timestamp": to_utc_iso(now_utc()), "step": step}
        monitor_event.update(args)
        await _append_cron_history(history_job_id, monitor_event=monitor_event)

    async def _cron_chat_event(
        job_id: str | None,
        role: str,
        content: str,
        **meta: object,
    ) -> None:
        event: dict[str, object] = {
            "timestamp": to_utc_iso(now_utc()),
            "role": role,
            "content": content,
        }
        event.update(meta)
        await _append_cron_history(job_id, chat_event=event)

    def _queue_meta(session_obj: object) -> dict[str, object]:
        if not hasattr(session_obj, "metadata"):
            return {}
        metadata = getattr(session_obj, "metadata", {})
        if not isinstance(metadata, dict):
            return {}
        raw = metadata.get("queue")
        if isinstance(raw, dict):
            return raw
        return {}

    async def _update_active_session_queue_settings(
        *,
        mode: str | None = None,
        debounce_ms: int | None = None,
        cap: int | None = None,
        drop_policy: str | None = None,
    ) -> tuple[bool, str]:
        if not agent.session:
            return False, "No active session"
        queue_meta = dict(_queue_meta(agent.session))
        if mode is not None:
            normalized_mode = normalize_queue_mode(mode)
            if not normalized_mode:
                return False, "Invalid queue mode. Use steer|followup|collect|steer-backlog|interrupt|queue."
            queue_meta["mode"] = normalized_mode
        if debounce_ms is not None:
            queue_meta["debounce_ms"] = max(0, int(debounce_ms))
        if cap is not None:
            queue_meta["cap"] = max(1, int(cap))
        if drop_policy is not None:
            normalized_drop = normalize_queue_drop_policy(drop_policy)
            if not normalized_drop:
                return False, "Invalid queue drop policy. Use old|new|summarize."
            queue_meta["drop_policy"] = normalized_drop
        queue_meta["updated_at"] = datetime.now().isoformat()
        agent.session.metadata["queue"] = queue_meta
        await agent.session_manager.save_session(agent.session)
        settings = await _resolve_queue_settings_for_session(agent.session.id)
        return (
            True,
            (
                "Session queue settings updated: "
                f"mode={settings.mode} debounce_ms={settings.debounce_ms} "
                f"cap={settings.cap} drop={settings.drop_policy}"
            ),
        )

    def _truncate_history_text(text: str, max_chars: int = 8000) -> str:
        cleaned = str(text or "")
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "... [truncated]"

    def _render_chat_export_markdown(
        session_id: str,
        session_name: str,
        messages: list[dict[str, object]],
    ) -> str:
        chat_messages = [msg for msg in messages if str(msg.get("role", "")).lower() in {"user", "assistant", "system"}]
        lines = [
            "# Session Chat Export",
            f"- Exported at (UTC): {to_utc_iso(now_utc())}",
            f"- Session ID: {session_id}",
            f"- Session name: {session_name}",
            f"- Messages: {len(chat_messages)}",
            "",
        ]
        if not chat_messages:
            lines.append("(no chat messages found)")
            lines.append("")
            return "\n".join(lines)

        for idx, msg in enumerate(chat_messages, start=1):
            role = str(msg.get("role", "unknown")).strip() or "unknown"
            timestamp = str(msg.get("timestamp", "")).strip()
            content = str(msg.get("content", ""))
            lines.append(f"## {idx}. role={role} timestamp={timestamp or '-'}")
            lines.append(content if content else "(empty)")
            lines.append("")
        return "\n".join(lines)

    def _render_monitor_export_markdown(
        session_id: str,
        session_name: str,
        messages: list[dict[str, object]],
    ) -> str:
        monitor_messages = [msg for msg in messages if str(msg.get("role", "")).lower() == "tool"]
        lines = [
            "# Session Monitor Export",
            f"- Exported at (UTC): {to_utc_iso(now_utc())}",
            f"- Session ID: {session_id}",
            f"- Session name: {session_name}",
            f"- Monitor entries: {len(monitor_messages)}",
            "",
        ]
        if not monitor_messages:
            lines.append("(no monitor/tool messages found)")
            lines.append("")
            return "\n".join(lines)

        for idx, msg in enumerate(monitor_messages, start=1):
            tool_name = str(msg.get("tool_name") or "tool")
            timestamp = str(msg.get("timestamp", "")).strip()
            args = msg.get("tool_arguments")
            if isinstance(args, dict):
                try:
                    args_text = json.dumps(args, ensure_ascii=True, sort_keys=True)
                except Exception:
                    args_text = str(args)
            else:
                args_text = "{}"
            content = str(msg.get("content", ""))
            lines.append(f"## {idx}. tool={tool_name} timestamp={timestamp or '-'}")
            lines.append(f"args={args_text}")
            lines.append(content if content else "(empty)")
            lines.append("")
        return "\n".join(lines)

    def _render_pipeline_export_jsonl(
        session_id: str,
        session_name: str,
        messages: list[dict[str, object]],
    ) -> str:
        entries = _collect_pipeline_trace_entries(session_id, session_name, messages)
        return "\n".join(json.dumps(item, ensure_ascii=True, sort_keys=True) for item in entries)

    def _collect_pipeline_trace_entries(
        session_id: str,
        session_name: str,
        messages: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        pipeline_messages = [
            msg
            for msg in messages
            if str(msg.get("role", "")).lower() == "tool"
            and str(msg.get("tool_name", "")).strip().lower() == "pipeline_trace"
        ]

        entries: list[dict[str, object]] = []
        for idx, msg in enumerate(pipeline_messages, start=1):
            args = msg.get("tool_arguments")
            payload = dict(args) if isinstance(args, dict) else {}
            payload["seq"] = idx
            payload["timestamp"] = str(msg.get("timestamp", "")).strip()
            payload["session_id"] = session_id
            payload["session_name"] = session_name
            entries.append(payload)

        if not entries:
            fallback_sources = {"planning", "task_contract", "completion_gate"}
            for idx, msg in enumerate(messages, start=1):
                if str(msg.get("role", "")).lower() != "tool":
                    continue
                source = str(msg.get("tool_name", "")).strip().lower()
                if source not in fallback_sources:
                    continue
                args = msg.get("tool_arguments")
                payload = dict(args) if isinstance(args, dict) else {}
                payload["source"] = source
                payload["seq"] = idx
                payload["timestamp"] = str(msg.get("timestamp", "")).strip()
                payload["session_id"] = session_id
                payload["session_name"] = session_name
                entries.append(payload)

        return entries

    def _render_pipeline_summary_markdown(
        session_id: str,
        session_name: str,
        messages: list[dict[str, object]],
    ) -> str:
        entries = _collect_pipeline_trace_entries(session_id, session_name, messages)
        lines = [
            "# Session Pipeline Trace Summary",
            f"- Exported at (UTC): {to_utc_iso(now_utc())}",
            f"- Session ID: {session_id}",
            f"- Session name: {session_name}",
            f"- Trace entries: {len(entries)}",
            "",
        ]
        if not entries:
            lines.append("(no pipeline trace entries found)")
            lines.append("")
            return "\n".join(lines)

        by_source: dict[str, int] = {}
        by_step: dict[str, int] = {}
        by_event: dict[str, int] = {}
        first_ts = str(entries[0].get("timestamp", "")).strip()
        last_ts = str(entries[-1].get("timestamp", "")).strip()
        for entry in entries:
            source = str(entry.get("source", "")).strip() or "unknown"
            by_source[source] = by_source.get(source, 0) + 1
            step = str(entry.get("step", "")).strip()
            if step:
                by_step[step] = by_step.get(step, 0) + 1
            event = str(entry.get("event", "")).strip()
            if event:
                by_event[event] = by_event.get(event, 0) + 1

        lines.append(f"- First entry timestamp: {first_ts or '-'}")
        lines.append(f"- Last entry timestamp: {last_ts or '-'}")
        lines.append("")
        lines.append("## Sources")
        for source, count in sorted(by_source.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- {source}: {count}")
        lines.append("")
        if by_event:
            lines.append("## Planning Events")
            for event, count in sorted(by_event.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"- {event}: {count}")
            lines.append("")
        if by_step:
            lines.append("## Completion/Contract Steps")
            for step, count in sorted(by_step.items(), key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"- {step}: {count}")
            lines.append("")

        lines.append("## Timeline")
        for entry in entries:
            seq = int(entry.get("seq", 0))
            timestamp = str(entry.get("timestamp", "")).strip() or "-"
            source = str(entry.get("source", "")).strip() or "unknown"
            item = f"{seq}. [{timestamp}] source={source}"
            event = str(entry.get("event", "")).strip()
            step = str(entry.get("step", "")).strip()
            if event:
                item += f" event={event}"
            if step:
                item += f" step={step}"
            leaf_index = entry.get("leaf_index")
            leaf_tasks = entry.get("leaf_tasks")
            leaf_remaining = entry.get("leaf_remaining")
            if isinstance(leaf_index, int) and isinstance(leaf_tasks, int):
                item += f" progress={leaf_index}/{leaf_tasks}"
            if isinstance(leaf_remaining, int):
                item += f" remaining={leaf_remaining}"
            current_path = str(entry.get("current_path", "")).strip()
            if current_path:
                item += f" path={current_path}"
            eta_text = str(entry.get("eta_text", "")).strip()
            if eta_text:
                item += f" eta={eta_text}"
            lines.append(item)
        lines.append("")
        return "\n".join(lines)

    def _export_active_session_history(mode: str) -> list[Path]:
        if not agent.session:
            return []
        mode_key = (mode or "all").strip().lower()
        if mode_key not in {"chat", "monitor", "pipeline", "pipeline-summary", "all"}:
            mode_key = "all"

        session_id = str(agent.session.id)
        session_name = str(agent.session.name)
        safe_session = _normalize_session_id(session_id)
        snapshot: list[dict[str, object]] = [dict(msg) for msg in agent.session.messages]

        saved_root = agent.tools.get_saved_base_path(create=True)
        export_root = (saved_root / "showcase" / safe_session / "exports").resolve()
        export_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        written: list[Path] = []

        if mode_key in {"chat", "all"}:
            chat_path = export_root / f"chat-{stamp}.md"
            chat_path.write_text(
                _render_chat_export_markdown(session_id, session_name, snapshot) + "\n",
                encoding="utf-8",
            )
            written.append(chat_path)

        if mode_key in {"monitor", "all"}:
            monitor_path = export_root / f"monitor-{stamp}.md"
            monitor_path.write_text(
                _render_monitor_export_markdown(session_id, session_name, snapshot) + "\n",
                encoding="utf-8",
            )
            written.append(monitor_path)

        if mode_key in {"pipeline", "all"}:
            pipeline_path = export_root / f"pipeline-{stamp}.jsonl"
            pipeline_path.write_text(
                _render_pipeline_export_jsonl(session_id, session_name, snapshot) + "\n",
                encoding="utf-8",
            )
            written.append(pipeline_path)
        if mode_key in {"pipeline-summary", "all"}:
            pipeline_summary_path = export_root / f"pipeline-summary-{stamp}.md"
            pipeline_summary_path.write_text(
                _render_pipeline_summary_markdown(session_id, session_name, snapshot) + "\n",
                encoding="utf-8",
            )
            written.append(pipeline_summary_path)

        return written

    def _resolve_saved_file_for_kind(kind: str, session_id: str, path_text: str) -> Path:
        saved_root = agent.tools.get_saved_base_path(create=True)
        requested = Path(path_text).expanduser()
        safe_session = _normalize_session_id(session_id)
        categories = {"downloads", "media", "scripts", "showcase", "skills", "tmp", "tools"}

        if requested.is_absolute():
            candidate = requested.resolve()
        else:
            if requested.parts and requested.parts[0] in categories:
                requested_parts = [part for part in requested.parts if part not in ("", ".", "..")]
                if len(requested_parts) >= 2 and requested_parts[1] == safe_session:
                    scoped_rel = Path(*requested_parts)
                else:
                    scoped_rel = Path(requested_parts[0]) / safe_session
                    if len(requested_parts) > 1:
                        scoped_rel = scoped_rel.joinpath(*requested_parts[1:])
                candidate = (saved_root / scoped_rel).resolve()
            else:
                default_category = "scripts" if kind == "script" else "tools"
                candidate = (saved_root / default_category / safe_session / requested).resolve()

        try:
            relative_candidate = candidate.relative_to(saved_root)
        except ValueError as e:
            raise ValueError(f"Path must be inside saved root: {saved_root}") from e

        relative_parts = [part for part in relative_candidate.parts if part not in ("", ".", "..")]
        if relative_parts and relative_parts[0] in categories:
            if len(relative_parts) < 2 or relative_parts[1] != safe_session:
                expected_prefix = f"{relative_parts[0]}/{safe_session}"
                raise ValueError(f"{kind} path must be inside saved/{expected_prefix}/...")

        if not candidate.exists() or not candidate.is_file():
            raise ValueError(f"{kind} file not found: {candidate}")
        return candidate

    async def _run_script_or_tool_in_session(
        target_session_id: str,
        kind: str,
        path_text: str,
        trigger: str,
        cron_job_id: str | None = None,
    ) -> None:
        target_session = await agent.session_manager.load_session(target_session_id)
        if not target_session:
            raise ValueError(f"Session not found: {target_session_id}")

        file_path = _resolve_saved_file_for_kind(kind=kind, session_id=target_session_id, path_text=path_text)
        command = agent._build_script_runner_command(file_path)
        if not command:
            command = f"cd {shlex.quote(str(file_path.parent))} && ./{shlex.quote(file_path.name)}"

        async def _execute() -> None:
            previous_session = agent.session
            previous_session_id = previous_session.id if previous_session else None
            switched = previous_session_id != target_session.id
            if switched:
                agent.session = target_session
                agent.refresh_session_runtime_flags()

            try:
                ui.print_message(
                    "system",
                    f"[CRON] {trigger} {kind} run in session={target_session.id} path={file_path}",
                )
                if agent.session:
                    start_note = f"[CRON] {trigger} {kind} start: {file_path}"
                    agent.session.add_message(
                        role="tool",
                        content=start_note,
                        tool_name="cron",
                        tool_arguments={
                            "step": "run_script_tool_start",
                            "trigger": trigger,
                            "kind": kind,
                            "path": str(file_path),
                            "job_id": cron_job_id or "",
                        },
                        token_count=agent._count_tokens(start_note),
                    )
                    await agent.session_manager.save_session(agent.session)
                await _cron_chat_event(
                    cron_job_id,
                    "system",
                    f"[CRON] {trigger} {kind} run start: {file_path}",
                    trigger=trigger,
                    kind=kind,
                    session_id=target_session.id,
                )
                await _cron_monitor_event(
                    "run_script_tool_start",
                    history_job_id=cron_job_id,
                    trigger=trigger,
                    kind=kind,
                    path=str(file_path),
                    session_id=target_session.id,
                )
                result = await agent._execute_tool_with_guard(
                    name="shell",
                    arguments={"command": command},
                    interaction_label=f"cron_{kind}_{trigger}",
                )
                shell_output = result.content if result.success else f"Error: {result.error}"
                if agent.session:
                    agent.session.add_message(
                        role="tool",
                        content=shell_output,
                        tool_name="shell",
                        tool_arguments={"command": command, "cron": True, "job_id": cron_job_id or ""},
                        token_count=agent._count_tokens(shell_output),
                    )
                    await agent.session_manager.save_session(agent.session)
                ui.append_tool_output("shell", {"command": command, "cron": True}, shell_output)
                await _cron_monitor_event(
                    "run_script_tool_output",
                    history_job_id=cron_job_id,
                    trigger=trigger,
                    kind=kind,
                    path=str(file_path),
                    session_id=target_session.id,
                    output=_truncate_history_text(shell_output),
                )
                await _cron_chat_event(
                    cron_job_id,
                    "tool",
                    shell_output,
                    trigger=trigger,
                    kind=kind,
                    session_id=target_session.id,
                    path=str(file_path),
                )
                if not result.success:
                    raise RuntimeError(shell_output)
                await _cron_monitor_event(
                    "run_script_tool_done",
                    history_job_id=cron_job_id,
                    trigger=trigger,
                    kind=kind,
                    path=str(file_path),
                    session_id=target_session.id,
                )
                await _cron_chat_event(
                    cron_job_id,
                    "system",
                    f"[CRON] {trigger} {kind} run complete: {file_path}",
                    trigger=trigger,
                    kind=kind,
                    session_id=target_session.id,
                )
                if agent.session:
                    done_note = f"[CRON] {trigger} {kind} complete: {file_path}"
                    agent.session.add_message(
                        role="tool",
                        content=done_note,
                        tool_name="cron",
                        tool_arguments={
                            "step": "run_script_tool_done",
                            "trigger": trigger,
                            "kind": kind,
                            "path": str(file_path),
                            "job_id": cron_job_id or "",
                        },
                        token_count=agent._count_tokens(done_note),
                    )
                    await agent.session_manager.save_session(agent.session)
            except Exception as e:
                if agent.session:
                    failed_note = f"[CRON] {trigger} {kind} failed: {str(e)}"
                    agent.session.add_message(
                        role="tool",
                        content=failed_note,
                        tool_name="cron",
                        tool_arguments={
                            "step": "run_script_tool_failed",
                            "trigger": trigger,
                            "kind": kind,
                            "path": str(file_path),
                            "job_id": cron_job_id or "",
                        },
                        token_count=agent._count_tokens(failed_note),
                    )
                    await agent.session_manager.save_session(agent.session)
                raise
            finally:
                if switched and previous_session is not None:
                    restored = await agent.session_manager.load_session(previous_session_id)
                    agent.session = restored or previous_session
                    agent.refresh_session_runtime_flags()

        await _enqueue_agent_task(target_session.id, _execute, lane=CommandLane.CRON)

    def _parse_cron_add_args(raw_add: str) -> tuple[dict[str, object], str, dict[str, str]]:
        tokens = shlex.split(raw_add)
        schedule, consumed = parse_schedule_tokens(tokens)
        schedule["_text"] = schedule_to_text(schedule)
        remaining = tokens[consumed:]
        if not remaining:
            raise ValueError("Usage: /cron add every <Nm|Nh> <task|script|tool ...>")

        kind_head = remaining[0].strip().lower()
        if kind_head in {"script", "tool"}:
            if len(remaining) < 2:
                raise ValueError(f"Usage: /cron add ... {kind_head} <path>")
            path_text = " ".join(remaining[1:]).strip()
            if not path_text:
                raise ValueError(f"Usage: /cron add ... {kind_head} <path>")
            return schedule, kind_head, {"path": path_text}

        prompt_text = " ".join(remaining).strip()
        if not prompt_text:
            raise ValueError("Usage: /cron add ... \"<task>\"")
        return schedule, "prompt", {"text": prompt_text}

    async def _run_prompt_in_active_session(
        prompt_text: str,
        *,
        display_prompt: str | None = None,
        cron_job_id: str | None = None,
        cron_trigger: str | None = None,
        cron_source: str | None = None,
        raise_on_error: bool = False,
        lane: str = CommandLane.MAIN,
        queue: bool = True,
    ) -> None:
        """Execute one user prompt using the currently selected session."""
        nonlocal last_exec_seconds
        nonlocal last_completed_at

        if not prompt_text.strip():
            return

        async def _execute() -> None:
            shown_prompt = display_prompt if display_prompt is not None else prompt_text
            ui.print_message("user", shown_prompt)
            ui.print_blank_line()
            if cron_job_id:
                await _cron_chat_event(
                    cron_job_id,
                    "user",
                    prompt_text,
                    trigger=cron_trigger or "",
                    source=cron_source or "",
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
                        _, cancelled = await _run_cancellable(ui, _consume_stream())
                    finally:
                        ui.end_assistant_stream()
                    if cancelled:
                        if cron_job_id:
                            await _cron_chat_event(
                                cron_job_id,
                                "system",
                                "[CRON] prompt cancelled",
                                trigger=cron_trigger or "",
                                source=cron_source or "",
                            )
                        ui.print_blank_line()
                        return
                    assistant_text = "".join(chunks)
                else:
                    response, cancelled = await _run_cancellable(ui, agent.complete(prompt_text))
                    if cancelled:
                        if cron_job_id:
                            await _cron_chat_event(
                                cron_job_id,
                                "system",
                                "[CRON] prompt cancelled",
                                trigger=cron_trigger or "",
                                source=cron_source or "",
                            )
                        ui.print_blank_line()
                        return
                    assistant_text = response or ""
                    ui.print_message("assistant", response)
                    ui.print_blank_line()

                if cron_job_id:
                    await _cron_chat_event(
                        cron_job_id,
                        "assistant",
                        assistant_text,
                        trigger=cron_trigger or "",
                        source=cron_source or "",
                    )
                    await _cron_monitor_event(
                        "prompt_assistant_output",
                        history_job_id=cron_job_id,
                        trigger=cron_trigger or "",
                        source=cron_source or "",
                        output=_truncate_history_text(assistant_text),
                    )
                last_exec_seconds = time.perf_counter() - started
                last_completed_at = datetime.now()
                ui.set_runtime_status("waiting")
            except Exception as e:
                last_exec_seconds = time.perf_counter() - started
                last_completed_at = datetime.now()
                ui.set_runtime_status("waiting")
                if cron_job_id:
                    await _cron_chat_event(
                        cron_job_id,
                        "system",
                        f"[CRON] prompt failed: {str(e)}",
                        trigger=cron_trigger or "",
                        source=cron_source or "",
                    )
                ui.print_error(str(e))
                log.error("Error in agent", error=str(e))
                if raise_on_error:
                    raise

        if not queue:
            await _execute()
            return

        session_id = agent.session.id if agent.session else "default"
        await _enqueue_agent_task(session_id, _execute, lane=lane)

    async def _run_prompt_in_session(
        session_id: str,
        prompt_text: str,
        source: str,
        *,
        cron_job_id: str | None = None,
        trigger: str = "scheduled",
    ) -> None:
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
                await _cron_monitor_event(
                    "prompt_start",
                    history_job_id=cron_job_id,
                    source=source,
                    session_id=selected.id,
                )
                ui.print_message(
                    "system",
                    f"[CRON] {trigger} prompt run in session={selected.id} job={cron_job_id or 'oneoff'}",
                )
                if agent.session:
                    start_note = f"[CRON] {trigger} prompt start: {source}"
                    agent.session.add_message(
                        role="tool",
                        content=start_note,
                        tool_name="cron",
                        tool_arguments={
                            "step": "prompt_start",
                            "trigger": trigger,
                            "source": source,
                            "job_id": cron_job_id or "",
                        },
                        token_count=agent._count_tokens(start_note),
                    )
                    await agent.session_manager.save_session(agent.session)
                await _run_prompt_in_active_session(
                    prompt_text,
                    display_prompt=f"[CRON job={cron_job_id or 'oneoff'}] {prompt_text}",
                    cron_job_id=cron_job_id,
                    cron_trigger=trigger,
                    cron_source=source,
                    raise_on_error=bool(cron_job_id),
                    lane=CommandLane.CRON,
                    queue=False,
                )
                await _cron_monitor_event(
                    "prompt_done",
                    history_job_id=cron_job_id,
                    source=source,
                    session_id=selected.id,
                )
                if agent.session:
                    done_note = f"[CRON] {trigger} prompt done: {source}"
                    agent.session.add_message(
                        role="tool",
                        content=done_note,
                        tool_name="cron",
                        tool_arguments={
                            "step": "prompt_done",
                            "trigger": trigger,
                            "source": source,
                            "job_id": cron_job_id or "",
                        },
                        token_count=agent._count_tokens(done_note),
                    )
                    await agent.session_manager.save_session(agent.session)
            except Exception as e:
                if agent.session:
                    fail_note = f"[CRON] {trigger} prompt failed: {str(e)}"
                    agent.session.add_message(
                        role="tool",
                        content=fail_note,
                        tool_name="cron",
                        tool_arguments={
                            "step": "prompt_failed",
                            "trigger": trigger,
                            "source": source,
                            "job_id": cron_job_id or "",
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

        await _enqueue_agent_task(selected.id, _execute, lane=CommandLane.CRON)

    async def _execute_cron_job(job: object, trigger: str = "scheduled") -> None:
        job_id = str(getattr(job, "id", "")).strip()
        if not job_id or job_id in cron_running_job_ids:
            return

        cron_running_job_ids.add(job_id)
        started_at_iso = to_utc_iso(now_utc())
        success = False
        queued_for_followup = False
        error_text = ""
        try:
            kind = str(getattr(job, "kind", "prompt")).strip().lower()
            payload = getattr(job, "payload", {})
            session_id = str(getattr(job, "session_id", "")).strip()
            schedule = getattr(job, "schedule", {})

            await _cron_monitor_event("job_start", history_job_id=job_id, trigger=trigger, job_id=job_id, kind=kind, session_id=session_id)
            await _cron_chat_event(
                job_id,
                "system",
                f"[CRON] job start trigger={trigger} kind={kind} session={session_id}",
                trigger=trigger,
                kind=kind,
                session_id=session_id,
            )
            if not session_id:
                raise ValueError(f"Cron job {job_id} has no session_id")

            if kind == "prompt":
                prompt_text = str(payload.get("text", "")).strip() if isinstance(payload, dict) else ""
                if not prompt_text:
                    raise ValueError(f"Cron job {job_id} has empty prompt payload")
                dispatch_status = await _dispatch_prompt_in_session(
                    session_id=session_id,
                    prompt_text=prompt_text,
                    source=f"cron:{trigger}:{job_id}",
                    cron_job_id=job_id,
                    trigger=trigger,
                )
                queued_for_followup = dispatch_status == "queued"
            elif kind in {"script", "tool"}:
                path_text = str(payload.get("path", "")).strip() if isinstance(payload, dict) else ""
                if not path_text:
                    raise ValueError(f"Cron job {job_id} has empty {kind} path payload")
                await _run_script_or_tool_in_session(
                    target_session_id=session_id,
                    kind=kind,
                    path_text=path_text,
                    trigger=trigger,
                    cron_job_id=job_id,
                )
            else:
                raise ValueError(f"Unsupported cron job kind: {kind}")

            success = True
            await _cron_monitor_event(
                "job_queued" if queued_for_followup else "job_done",
                history_job_id=job_id,
                trigger=trigger,
                job_id=job_id,
                kind=kind,
                session_id=session_id,
            )
            await _cron_chat_event(
                job_id,
                "system",
                (
                    f"[CRON] job queued trigger={trigger} kind={kind} session={session_id}"
                    if queued_for_followup
                    else f"[CRON] job done trigger={trigger} kind={kind} session={session_id}"
                ),
                trigger=trigger,
                kind=kind,
                session_id=session_id,
            )

            next_run_at_iso = to_utc_iso(compute_next_run(schedule))
            await agent.session_manager.update_cron_job(
                job_id,
                last_run_at=started_at_iso,
                next_run_at=next_run_at_iso,
                last_status="queued" if queued_for_followup else "ok",
                last_error="",
            )
        except Exception as e:
            error_text = str(e)
            await _cron_monitor_event("job_failed", history_job_id=job_id, trigger=trigger, job_id=job_id, error=error_text)
            await _cron_chat_event(
                job_id,
                "system",
                f"[CRON] job failed: {error_text}",
                trigger=trigger,
            )
            try:
                next_run_at_iso = to_utc_iso(compute_next_run(getattr(job, "schedule", {})))
            except Exception:
                next_run_at_iso = to_utc_iso(now_utc())
            await agent.session_manager.update_cron_job(
                job_id,
                last_run_at=started_at_iso,
                next_run_at=next_run_at_iso,
                last_status="failed",
                last_error=error_text,
            )
        finally:
            cron_running_job_ids.discard(job_id)
            if trigger == "manual" and not success:
                ui.print_error(error_text or f"Cron job failed: {job_id}")

    async def _cron_scheduler_loop() -> None:
        """Background pseudo-cron runner (Captain Claw cron, not system cron)."""
        while True:
            await asyncio.sleep(cron_poll_seconds)
            due_jobs = await agent.session_manager.get_due_cron_jobs(
                now_iso=to_utc_iso(now_utc()),
                limit=10,
            )
            for job in due_jobs:
                await _execute_cron_job(job, trigger="scheduled")

    cron_worker = asyncio.create_task(_cron_scheduler_loop())
    try:
        # Main loop
        while True:
            try:
                ui.print_status_line(
                    last_usage=agent.last_usage,
                    total_usage=agent.total_usage,
                    last_exec_seconds=last_exec_seconds,
                    last_completed_at=last_completed_at,
                    session_id=agent.session.id if agent.session else None,
                    context_window=agent.last_context_window,
                    model_details=agent.get_runtime_model_details(),
                )
                ui.set_runtime_status("user input")
                # Get user input (threaded so event loop keeps servicing Captain Claw cron).
                user_input = await asyncio.to_thread(ui.prompt)
            
                # Handle special commands
                result = ui.handle_special_command(user_input)
            
                if result is None:
                    continue
                elif result == "EXIT":
                    log.info("User requested exit")
                    break
                elif result == "CLEAR":
                    if agent.session:
                        if agent.is_session_memory_protected():
                            ui.print_error(
                                "Session memory is protected. Disable it with '/session protect off' first."
                            )
                            continue
                        agent.session.messages = []
                        await agent.session_manager.save_session(agent.session)
                        ui.clear_monitor_tool_output()
                        ui.print_success("Session cleared")
                    continue
                elif result == "NEW" or result.startswith("NEW:"):
                    session_name = "default"
                    if result.startswith("NEW:"):
                        session_name = result.split(":", 1)[1].strip() or "default"
                    agent.session = await agent.session_manager.create_session(name=session_name)
                    agent.refresh_session_runtime_flags()
                    if agent.session:
                        await agent.session_manager.set_last_active_session(agent.session.id)
                    if agent.session:
                        ui.load_monitor_tool_output_from_session(agent.session.messages)
                        ui.print_session_info(agent.session)
                    ui.print_success("Started new session")
                    continue
                elif result == "SESSIONS":
                    sessions = await agent.session_manager.list_sessions(limit=20)
                    ui.print_session_list(
                        sessions,
                        current_session_id=agent.session.id if agent.session else None,
                    )
                    continue
                elif result == "MODELS":
                    ui.print_model_list(
                        agent.get_allowed_models(),
                        active_model=agent.get_runtime_model_details(),
                    )
                    continue
                elif result == "SESSION_INFO":
                    if agent.session:
                        ui.print_session_info(agent.session)
                    else:
                        ui.print_error("No active session")
                    continue
                elif result == "SESSION_MODEL_INFO":
                    details = agent.get_runtime_model_details()
                    ui.print_success(
                        "Active model: "
                        f"{details.get('provider')}/{details.get('model')} "
                        f"(source={details.get('source') or 'unknown'}, id={details.get('id') or '-'})"
                    )
                    continue
                elif result in {"SESSION_PROTECT_ON", "SESSION_PROTECT_OFF"}:
                    enabled = result.endswith("_ON")
                    ok, message = await agent.set_session_memory_protection(enabled, persist=True)
                    if ok:
                        if agent.session:
                            ui.print_session_info(agent.session)
                        ui.print_success(message)
                    else:
                        ui.print_error(message)
                    continue
                elif result.startswith("SESSION_MODEL_SET:"):
                    selector = result.split(":", 1)[1].strip()
                    ok, message = await agent.set_session_model(selector, persist=True)
                    if ok:
                        if agent.session:
                            ui.print_session_info(agent.session)
                        ui.print_success(message)
                    else:
                        ui.print_error(message)
                    continue
                elif result == "SESSION_QUEUE_INFO":
                    if not agent.session:
                        ui.print_error("No active session")
                        continue
                    settings = await _resolve_queue_settings_for_session(agent.session.id)
                    ui.print_success(
                        "Session queue settings: "
                        f"mode={settings.mode} debounce_ms={settings.debounce_ms} "
                        f"cap={settings.cap} drop={settings.drop_policy} "
                        f"pending={followup_queue.get_queue_depth(agent.session.id)}"
                    )
                    continue
                elif result.startswith("SESSION_QUEUE_MODE:"):
                    mode_value = result.split(":", 1)[1].strip()
                    ok, message = await _update_active_session_queue_settings(mode=mode_value)
                    if ok:
                        ui.print_success(message)
                    else:
                        ui.print_error(message)
                    continue
                elif result.startswith("SESSION_QUEUE_DEBOUNCE:"):
                    raw_value = result.split(":", 1)[1].strip()
                    try:
                        parsed = int(raw_value)
                    except Exception:
                        ui.print_error("Usage: /session queue debounce <ms>")
                        continue
                    ok, message = await _update_active_session_queue_settings(debounce_ms=parsed)
                    if ok:
                        ui.print_success(message)
                    else:
                        ui.print_error(message)
                    continue
                elif result.startswith("SESSION_QUEUE_CAP:"):
                    raw_value = result.split(":", 1)[1].strip()
                    try:
                        parsed = int(raw_value)
                    except Exception:
                        ui.print_error("Usage: /session queue cap <n>")
                        continue
                    ok, message = await _update_active_session_queue_settings(cap=parsed)
                    if ok:
                        ui.print_success(message)
                    else:
                        ui.print_error(message)
                    continue
                elif result.startswith("SESSION_QUEUE_DROP:"):
                    drop_value = result.split(":", 1)[1].strip()
                    ok, message = await _update_active_session_queue_settings(drop_policy=drop_value)
                    if ok:
                        ui.print_success(message)
                    else:
                        ui.print_error(message)
                    continue
                elif result == "SESSION_QUEUE_CLEAR":
                    if not agent.session:
                        ui.print_error("No active session")
                        continue
                    session_id = agent.session.id
                    followup_cleared = followup_queue.clear_queue(session_id)
                    lane_cleared = command_queue.clear_lane(resolve_session_lane(session_id))
                    ui.print_success(
                        f"Cleared session queue: followup={followup_cleared} lane={lane_cleared}"
                    )
                    continue
                elif result.startswith("SESSION_SELECT:"):
                    selector = result.split(":", 1)[1].strip()
                    selected = await agent.session_manager.select_session(selector)
                    if not selected:
                        ui.print_error(f"Session not found: {selector}")
                        continue
                    agent.session = selected
                    agent.refresh_session_runtime_flags()
                    await agent.session_manager.set_last_active_session(agent.session.id)
                    ui.load_monitor_tool_output_from_session(agent.session.messages)
                    ui.print_session_info(agent.session)
                    ui.print_success("Loaded session")
                    continue
                elif result.startswith("SESSION_RENAME:"):
                    new_name = result.split(":", 1)[1].strip()
                    if not new_name:
                        ui.print_error("Usage: /session rename <new-name>")
                        continue
                    if not agent.session:
                        ui.print_error("No active session")
                        continue
                    old_name = agent.session.name
                    agent.session.name = new_name
                    await agent.session_manager.save_session(agent.session)
                    ui.print_session_info(agent.session)
                    ui.print_success(f'Session renamed: "{old_name}" -> "{new_name}"')
                    continue
                elif result == "SESSION_DESCRIPTION_INFO":
                    if not agent.session:
                        ui.print_error("No active session")
                        continue
                    description = str(agent.session.metadata.get("description", "")).strip()
                    if description:
                        ui.print_success(f"Session description: {description}")
                    else:
                        ui.print_warning("Session has no description yet")
                    continue
                elif result == "SESSION_DESCRIPTION_AUTO":
                    if not agent.session:
                        ui.print_error("No active session")
                        continue
                    generated = await agent.generate_session_description(agent.session, max_sentences=5)
                    description = agent.sanitize_session_description(generated, max_sentences=5)
                    if not description:
                        ui.print_error("Could not generate a session description")
                        continue
                    agent.session.metadata["description"] = description
                    agent.session.metadata["description_source"] = "auto"
                    agent.session.metadata["description_updated_at"] = datetime.now().isoformat()
                    await agent.session_manager.save_session(agent.session)
                    ui.print_session_info(agent.session)
                    ui.print_success("Session description auto-generated")
                    continue
                elif result.startswith("SESSION_DESCRIPTION_SET:"):
                    if not agent.session:
                        ui.print_error("No active session")
                        continue
                    payload_raw = result.split(":", 1)[1].strip()
                    try:
                        payload = json.loads(payload_raw)
                    except Exception:
                        ui.print_error("Invalid /session description payload")
                        continue
                    raw_description = str(payload.get("description", "")).strip()
                    description = agent.sanitize_session_description(raw_description, max_sentences=5)
                    if not description:
                        ui.print_error("Usage: /session description <text> | /session description auto")
                        continue
                    agent.session.metadata["description"] = description
                    agent.session.metadata["description_source"] = "manual"
                    agent.session.metadata["description_updated_at"] = datetime.now().isoformat()
                    await agent.session_manager.save_session(agent.session)
                    ui.print_session_info(agent.session)
                    ui.print_success("Session description updated")
                    continue
                elif result.startswith("SESSION_EXPORT:"):
                    if not agent.session:
                        ui.print_error("No active session")
                        continue
                    mode = result.split(":", 1)[1].strip().lower() or "all"
                    session_id = agent.session.id

                    async def _export_task() -> list[Path]:
                        return _export_active_session_history(mode)

                    written_paths = await _enqueue_agent_task(
                        session_id,
                        _export_task,
                        lane=CommandLane.NESTED,
                    )
                    if not written_paths:
                        ui.print_error("Failed to export session history")
                        continue
                    ui.append_tool_output(
                        "session_export",
                        {
                            "session_id": agent.session.id,
                            "mode": mode,
                            "count": len(written_paths),
                        },
                        "\n".join(f"path={path}" for path in written_paths),
                    )
                    for path in written_paths:
                        ui.print_success(f"Exported: {path}")
                    continue
                elif result.startswith("SESSION_RUN:"):
                    payload_raw = result.split(":", 1)[1].strip()
                    try:
                        payload = json.loads(payload_raw)
                    except Exception:
                        ui.print_error("Invalid /session run payload")
                        continue

                    selector = str(payload.get("selector", "")).strip()
                    prompt = str(payload.get("prompt", "")).strip()
                    if not selector or not prompt:
                        ui.print_error("Usage: /session run <id|name|#index> <prompt>")
                        continue

                    selected = await agent.session_manager.select_session(selector)
                    if not selected:
                        ui.print_error(f"Session not found: {selector}")
                        continue

                    async def _run_selected_session_prompt() -> None:
                        previous_session = agent.session
                        previous_session_id = previous_session.id if previous_session else None
                        switched_temporarily = previous_session_id != selected.id

                        if switched_temporarily:
                            agent.session = selected
                            agent.refresh_session_runtime_flags()
                            ui.load_monitor_tool_output_from_session(agent.session.messages)
                            ui.print_success(f'Running in session "{agent.session.name}"')

                        try:
                            await _run_prompt_in_active_session(
                                prompt,
                                lane=CommandLane.NESTED,
                                queue=False,
                            )
                        finally:
                            if switched_temporarily and previous_session is not None:
                                restored = await agent.session_manager.load_session(previous_session_id)
                                agent.session = restored or previous_session
                                agent.refresh_session_runtime_flags()
                                ui.load_monitor_tool_output_from_session(agent.session.messages)
                                ui.print_success(f'Restored session "{agent.session.name}"')

                    await _enqueue_agent_task(
                        selected.id,
                        _run_selected_session_prompt,
                        lane=CommandLane.NESTED,
                    )
                    continue
                elif result.startswith("SESSION_PROCREATE:"):
                    payload_raw = result.split(":", 1)[1].strip()
                    try:
                        payload = json.loads(payload_raw)
                    except Exception:
                        ui.print_error("Invalid /session procreate payload")
                        continue

                    parent_one_selector = str(payload.get("parent_one", "")).strip()
                    parent_two_selector = str(payload.get("parent_two", "")).strip()
                    new_name = str(payload.get("new_name", "")).strip()
                    if not parent_one_selector or not parent_two_selector or not new_name:
                        ui.print_error(
                            "Usage: /session procreate <id|name|#index> <id|name|#index> <new-name>"
                        )
                        continue

                    ui.append_tool_output(
                        "session_procreate",
                        {"step": "resolve_parents", "parent_one_selector": parent_one_selector, "parent_two_selector": parent_two_selector},
                        "step=resolve_parents\nstatus=locating_parent_sessions",
                    )
                    parent_one = await agent.session_manager.select_session(parent_one_selector)
                    if not parent_one:
                        ui.print_error(f"Session not found: {parent_one_selector}")
                        continue
                    parent_two = await agent.session_manager.select_session(parent_two_selector)
                    if not parent_two:
                        ui.print_error(f"Session not found: {parent_two_selector}")
                        continue

                    if parent_one.id == parent_two.id:
                        ui.print_error("Choose two different sessions for /session procreate")
                        continue

                    try:
                        child_session, stats = await agent.procreate_sessions(
                            parent_one=parent_one,
                            parent_two=parent_two,
                            new_name=new_name,
                            persist=True,
                        )
                    except ValueError as e:
                        ui.print_error(str(e))
                        continue

                    ui.append_tool_output(
                        "session_procreate",
                        {"step": "switch_to_child", "session_id": child_session.id},
                        (
                            "step=switch_to_child\n"
                            f'session_id="{child_session.id}"\n'
                            f'session_name="{child_session.name}"'
                        ),
                    )
                    agent.session = child_session
                    agent.refresh_session_runtime_flags()
                    await agent.session_manager.set_last_active_session(agent.session.id)
                    ui.load_monitor_tool_output_from_session(agent.session.messages)
                    ui.append_tool_output(
                        "session_procreate",
                        {"step": "complete", "session_id": child_session.id},
                        (
                            "step=complete\n"
                            f'session_id="{child_session.id}"\n'
                            f"merged_messages={stats.get('merged_messages', 0)}"
                        ),
                    )
                    ui.print_session_info(agent.session)
                    ui.print_success(
                        f'Procreated session "{child_session.name}" '
                        f"(merged_messages={stats.get('merged_messages', 0)}, "
                        f"compacted={stats.get('parent_one_compacted', 0)}+{stats.get('parent_two_compacted', 0)})"
                    )
                    continue
                elif result == "CRON_LIST":
                    jobs = await agent.session_manager.list_cron_jobs(limit=200, active_only=True)
                    for job in jobs:
                        if isinstance(job.schedule, dict):
                            job.schedule["_text"] = schedule_to_text(job.schedule)
                    ui.print_cron_jobs(jobs)
                    continue
                elif result.startswith("CRON_HISTORY:"):
                    selector = result.split(":", 1)[1].strip()
                    job = await agent.session_manager.select_cron_job(selector, active_only=False)
                    if not job:
                        ui.print_error(f"Cron job not found: {selector}")
                        continue
                    ui.print_cron_job_history(job)
                    continue
                elif result.startswith("CRON_ONEOFF:"):
                    payload_raw = result.split(":", 1)[1].strip()
                    try:
                        payload = json.loads(payload_raw)
                    except Exception:
                        ui.print_error("Invalid /cron payload")
                        continue
                    prompt = str(payload.get("prompt", "")).strip()
                    if not prompt:
                        ui.print_error("Usage: /cron \"<task>\"")
                        continue
                    if not agent.session:
                        ui.print_error("No active session for /cron")
                        continue
                    _cron_monitor("oneoff_prompt", session_id=agent.session.id if agent.session else "", chars=len(prompt))
                    status = await _dispatch_prompt_in_session(
                        session_id=agent.session.id,
                        prompt_text=prompt,
                        source="cron:oneoff",
                        cron_job_id=None,
                        trigger="oneoff",
                    )
                    if status == "queued":
                        ui.print_success("Cron one-off queued as follow-up (session busy)")
                    continue
                elif result.startswith("CRON_ADD:"):
                    if not agent.session:
                        ui.print_error("No active session for /cron add")
                        continue
                    raw_add = result.split(":", 1)[1].strip()
                    try:
                        schedule, kind, payload = _parse_cron_add_args(raw_add)
                    except ValueError as e:
                        ui.print_error(str(e))
                        continue

                    if kind in {"script", "tool"}:
                        try:
                            _ = _resolve_saved_file_for_kind(
                                kind=kind,
                                session_id=agent.session.id,
                                path_text=str(payload.get("path", "")),
                            )
                        except ValueError as e:
                            ui.print_error(str(e))
                            continue

                    next_run_at_iso = to_utc_iso(compute_next_run(schedule))
                    job = await agent.session_manager.create_cron_job(
                        kind=kind,
                        payload=payload,
                        schedule=schedule,
                        session_id=agent.session.id,
                        next_run_at=next_run_at_iso,
                        enabled=True,
                    )
                    await _cron_monitor_event(
                        "job_added",
                        history_job_id=job.id,
                        job_id=job.id,
                        session_id=job.session_id,
                        kind=job.kind,
                        schedule=schedule_to_text(schedule),
                        next_run_at=next_run_at_iso,
                    )
                    ui.print_success(
                        f"Cron job added: id={job.id} kind={job.kind} "
                        f"schedule={schedule_to_text(schedule)} next={next_run_at_iso}"
                    )
                    continue
                elif result.startswith("CRON_REMOVE:"):
                    selector = result.split(":", 1)[1].strip()
                    job = await agent.session_manager.select_cron_job(selector, active_only=False)
                    if not job:
                        ui.print_error(f"Cron job not found: {selector}")
                        continue
                    job_id = job.id
                    deleted = await agent.session_manager.delete_cron_job(job_id)
                    if not deleted:
                        ui.print_error(f"Cron job not found: {job_id}")
                        continue
                    _cron_monitor("job_removed", job_id=job_id)
                    ui.print_success(f"Removed cron job: {job_id}")
                    continue
                elif result.startswith("CRON_PAUSE:"):
                    selector = result.split(":", 1)[1].strip()
                    job = await agent.session_manager.select_cron_job(selector, active_only=False)
                    if not job:
                        ui.print_error(f"Cron job not found: {selector}")
                        continue
                    job_id = job.id
                    updated = await agent.session_manager.update_cron_job(
                        job_id,
                        enabled=False,
                        last_status="paused",
                    )
                    if not updated:
                        ui.print_error(f"Cron job not found: {job_id}")
                        continue
                    await _cron_monitor_event("job_paused", history_job_id=job_id, job_id=job_id)
                    ui.print_success(f"Paused cron job: {job_id}")
                    continue
                elif result.startswith("CRON_RESUME:"):
                    selector = result.split(":", 1)[1].strip()
                    job = await agent.session_manager.select_cron_job(selector, active_only=False)
                    if not job:
                        ui.print_error(f"Cron job not found: {selector}")
                        continue
                    job_id = job.id
                    updated = await agent.session_manager.update_cron_job(
                        job_id,
                        enabled=True,
                        last_status="scheduled",
                    )
                    if not updated:
                        ui.print_error(f"Cron job not found: {job_id}")
                        continue
                    await _cron_monitor_event("job_resumed", history_job_id=job_id, job_id=job_id)
                    ui.print_success(f"Resumed cron job: {job_id}")
                    continue
                elif result.startswith("CRON_RUN:"):
                    payload_raw = result.split(":", 1)[1].strip()
                    try:
                        payload = json.loads(payload_raw)
                    except Exception:
                        ui.print_error("Invalid /cron run payload")
                        continue
                    raw_args = str(payload.get("args", "")).strip()
                    if not raw_args:
                        ui.print_error("Usage: /cron run <job-id|#index> | /cron run script <path> | /cron run tool <path>")
                        continue
                    try:
                        run_tokens = shlex.split(raw_args)
                    except ValueError:
                        ui.print_error("Invalid /cron run arguments")
                        continue
                    if not run_tokens:
                        ui.print_error("Usage: /cron run <job-id|#index> | /cron run script <path> | /cron run tool <path>")
                        continue

                    head = run_tokens[0].strip().lower()
                    if head in {"script", "tool"}:
                        if not agent.session:
                            ui.print_error("No active session for /cron run script|tool")
                            continue
                        path_text = " ".join(run_tokens[1:]).strip()
                        if not path_text:
                            ui.print_error(f"Usage: /cron run {head} <path>")
                            continue
                        try:
                            await _run_script_or_tool_in_session(
                                target_session_id=agent.session.id,
                                kind=head,
                                path_text=path_text,
                                trigger="manual",
                            )
                            ui.print_success(f"Cron manual {head} run completed")
                        except Exception as e:
                            ui.print_error(str(e))
                        continue

                    selector = run_tokens[0].strip()
                    job = await agent.session_manager.select_cron_job(selector, active_only=False)
                    if not job:
                        ui.print_error(f"Cron job not found: {selector}")
                        continue
                    job_id = job.id
                    await _execute_cron_job(job, trigger="manual")
                    ui.print_success(f"Manual cron run finished: {job_id}")
                    continue
                elif result == "CONFIG":
                    ui.print_config(get_config())
                    continue
                elif result == "HISTORY":
                    if agent.session:
                        ui.print_history(agent.session.messages)
                    continue
                elif result == "COMPACT":
                    compacted, stats = await agent.compact_session(force=True, trigger="manual")
                    if compacted:
                        if agent.session:
                            ui.load_monitor_tool_output_from_session(agent.session.messages)
                        ui.print_success(
                            "Session compacted "
                            f"({int(stats.get('before_tokens', 0))} -> {int(stats.get('after_tokens', 0))} tokens)"
                        )
                    else:
                        reason = str(stats.get("reason", "not_needed"))
                        ui.print_warning(f"Compaction skipped: {reason}")
                    continue
                elif result == "PLANNING_ON":
                    await agent.set_pipeline_mode("contracts")
                    ui.print_success("Pipeline mode set to contracts")
                    continue
                elif result == "PLANNING_OFF":
                    await agent.set_pipeline_mode("loop")
                    ui.print_success("Pipeline mode set to loop")
                    continue
                elif result == "PIPELINE_INFO":
                    ui.print_success(
                        "Pipeline mode: "
                        f"{agent.pipeline_mode} "
                        "(loop=fast/simple, contracts=planner+completion gate)"
                    )
                    continue
                elif result.startswith("PIPELINE_MODE:"):
                    mode = result.split(":", 1)[1].strip().lower()
                    try:
                        await agent.set_pipeline_mode(mode)
                    except ValueError:
                        ui.print_error("Invalid pipeline mode. Use /pipeline loop|contracts")
                        continue
                    ui.print_success(
                        "Pipeline mode set to "
                        f"{agent.pipeline_mode} "
                        "(loop=fast/simple, contracts=planner+completion gate)"
                    )
                    continue
                elif result == "MONITOR_ON":
                    ui.set_monitor_mode(True)
                    if agent.session:
                        ui.load_monitor_tool_output_from_session(agent.session.messages)
                    ui.print_success("Monitor enabled")
                    continue
                elif result == "MONITOR_OFF":
                    ui.set_monitor_mode(False)
                    ui.print_success("Monitor disabled")
                    continue
                elif result == "MONITOR_TRACE_ON":
                    await agent.set_monitor_trace_llm(True)
                    ui.print_success("Monitor trace enabled (full intermediate LLM responses will be logged)")
                    continue
                elif result == "MONITOR_TRACE_OFF":
                    await agent.set_monitor_trace_llm(False)
                    ui.print_success("Monitor trace disabled")
                    continue
                elif result == "MONITOR_PIPELINE_ON":
                    await agent.set_monitor_trace_pipeline(True)
                    ui.print_success("Pipeline trace enabled (compact pipeline-only events will be logged)")
                    continue
                elif result == "MONITOR_PIPELINE_OFF":
                    await agent.set_monitor_trace_pipeline(False)
                    ui.print_success("Pipeline trace disabled")
                    continue
                elif result == "MONITOR_FULL_ON":
                    ui.set_monitor_full_output(True)
                    ui.print_success("Monitor full output rendering enabled")
                    continue
                elif result == "MONITOR_FULL_OFF":
                    ui.set_monitor_full_output(False)
                    ui.print_success("Monitor compact output rendering enabled")
                    continue
                elif result == "MONITOR_SCROLL_STATUS":
                    ui.print_success(f"Monitor scroll: {ui.describe_monitor_scroll()}")
                    continue
                elif result.startswith("MONITOR_SCROLL:"):
                    payload_raw = result.split(":", 1)[1].strip()
                    try:
                        payload = json.loads(payload_raw)
                    except Exception:
                        ui.print_error("Invalid /monitor scroll payload")
                        continue
                    pane = str(payload.get("pane", "")).strip().lower()
                    action = str(payload.get("action", "")).strip().lower()
                    amount_raw = payload.get("amount", 1)
                    try:
                        amount = int(amount_raw)
                    except Exception:
                        ui.print_error("Invalid scroll amount")
                        continue
                    ok, message = ui.scroll_monitor_pane(pane=pane, action=action, amount=amount)
                    if ok:
                        ui.print_success(message)
                    else:
                        ui.print_error(message)
                    continue
            
                # Skip empty input
                if not user_input.strip():
                    continue

                await _run_prompt_in_active_session(user_input)
            
            except KeyboardInterrupt:
                log.info("Interrupted by user")
                break
            except EOFError:
                log.info("EOF received")
                break
            except Exception as e:
                ui.print_error(str(e))
                log.error("Error in interactive loop", error=str(e))
    finally:
        cron_worker.cancel()
        try:
            await cron_worker
        except asyncio.CancelledError:
            pass


def version() -> None:
    """Show version information."""
    from captain_claw import __version__
    print(f"Captain Claw v{__version__}")


if __name__ == "__main__":
    import typer
    
    cli = typer.Typer(help="Captain Claw - A powerful console-based AI agent")
    
    @cli.command()
    def run(
        config: str = typer.Option("", "-c", "--config", help="Path to config file"),
        model: str = typer.Option("", "-m", "--model", help="Override model"),
        provider: str = typer.Option("", "-p", "--provider", help="Override provider"),
        no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming"),
        verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug logging"),
    ) -> None:
        main(config, model, provider, no_stream, verbose)
    
    @cli.command()
    def ver() -> None:
        version()
    
    cli()
