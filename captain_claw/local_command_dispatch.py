"""Local console command dispatch.

Extracts the giant ``elif`` chain that was inside the ``while True`` main
loop of ``run_interactive()``.  Returns ``"break"`` to exit, ``"continue"``
to skip to next prompt, or ``None`` to fall through to default execution.
"""

from __future__ import annotations

import json
import shlex
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from captain_claw.config import get_config
from captain_claw.cron import compute_next_run, schedule_to_text, to_utc_iso
from captain_claw.cron_dispatch import (
    cron_monitor,
    cron_monitor_event,
    execute_cron_job,
    parse_cron_add_args,
    resolve_saved_file_for_kind,
    run_script_or_tool_in_session,
)
from captain_claw.execution_queue import (
    CommandLane,
    resolve_session_lane,
)
from captain_claw.logging import log
from captain_claw.platform_adapter import approve_chat_pairing_token
from captain_claw.prompt_execution import (
    dispatch_prompt_in_session,
    enqueue_agent_task,
    resolve_queue_settings_for_session,
    run_cancellable,
    run_prompt_in_active_session,
    update_active_session_queue_settings,
)
from captain_claw.remote_command_handler import (
    format_active_configuration_text,
)
from captain_claw.session_export import export_session_history

if TYPE_CHECKING:
    from captain_claw.runtime_context import RuntimeContext


async def dispatch_local_command(
    ctx: RuntimeContext, result: str, user_input: str,
) -> str | None:
    """Handle a parsed command result from the local console.

    Returns:
        ``"break"``    – caller should exit the main loop
        ``"continue"`` – caller should skip to next iteration
        ``None``       – unhandled, caller should fall through to prompt execution
    """
    agent = ctx.agent
    ui = ctx.ui

    if result == "EXIT":
        log.info("User requested exit")
        return "break"

    if result.startswith("APPROVE_CHAT_USER:"):
        parts = result.split(":", 2)
        platform = parts[1].strip().lower() if len(parts) > 1 else ""
        token = parts[2].strip() if len(parts) > 2 else ""
        enabled_map = {
            "telegram": ctx.telegram.enabled,
            "slack": ctx.slack.enabled,
            "discord": ctx.discord.enabled,
        }
        if not enabled_map.get(platform, False):
            ui.print_error(f"{platform.title() if platform else 'Target'} integration is not enabled.")
            return "continue"
        ok, message = await approve_chat_pairing_token(ctx, platform, token)
        if ok:
            ui.print_success(message)
        else:
            ui.print_error(message)
        return "continue"

    if result.startswith("APPROVE_TELEGRAM_USER:"):
        token = result.split(":", 1)[1].strip()
        ok, message = await approve_chat_pairing_token(ctx, "telegram", token)
        if ok:
            ui.print_success(message)
        else:
            ui.print_error(message)
        return "continue"

    if result == "CLEAR":
        if agent.session:
            if agent.is_session_memory_protected():
                ui.print_error("Session memory is protected. Disable it with '/session protect off' first.")
                return "continue"
            agent.session.messages = []
            await agent.session_manager.save_session(agent.session)
            ui.clear_monitor_tool_output()
            ui.print_success("Session cleared")
        return "continue"

    if result == "NEW" or result.startswith("NEW:"):
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
        return "continue"

    if result == "SESSIONS":
        sessions = await agent.session_manager.list_sessions(limit=20)
        ui.print_session_list(
            sessions,
            current_session_id=agent.session.id if agent.session else None,
        )
        return "continue"

    if result == "MODELS":
        ui.print_model_list(
            agent.get_allowed_models(),
            active_model=agent.get_runtime_model_details(),
        )
        return "continue"

    if result == "SESSION_INFO":
        if agent.session:
            ui.print_session_info(agent.session)
        else:
            ui.print_error("No active session")
        return "continue"

    if result == "SESSION_MODEL_INFO":
        details = agent.get_runtime_model_details()
        ui.print_success(
            f"Active model: {details.get('provider')}/{details.get('model')} "
            f"(source={details.get('source') or 'unknown'}, id={details.get('id') or '-'})"
        )
        return "continue"

    if result in {"SESSION_PROTECT_ON", "SESSION_PROTECT_OFF"}:
        enabled = result.endswith("_ON")
        ok, message = await agent.set_session_memory_protection(enabled, persist=True)
        if ok:
            if agent.session:
                ui.print_session_info(agent.session)
            ui.print_success(message)
        else:
            ui.print_error(message)
        return "continue"

    if result.startswith("SESSION_MODEL_SET:"):
        selector = result.split(":", 1)[1].strip()
        ok, message = await agent.set_session_model(selector, persist=True)
        if ok:
            if agent.session:
                ui.print_session_info(agent.session)
            ui.print_success(message)
        else:
            ui.print_error(message)
        return "continue"

    if result == "SESSION_QUEUE_INFO":
        if not agent.session:
            ui.print_error("No active session")
            return "continue"
        settings = await resolve_queue_settings_for_session(ctx, agent.session.id)
        ui.print_success(
            f"Session queue settings: mode={settings.mode} debounce_ms={settings.debounce_ms} "
            f"cap={settings.cap} drop={settings.drop_policy} "
            f"pending={ctx.followup_queue.get_queue_depth(agent.session.id)}"
        )
        return "continue"

    if result.startswith("SESSION_QUEUE_MODE:"):
        mode_value = result.split(":", 1)[1].strip()
        ok, message = await update_active_session_queue_settings(ctx, mode=mode_value)
        if ok:
            ui.print_success(message)
        else:
            ui.print_error(message)
        return "continue"

    if result.startswith("SESSION_QUEUE_DEBOUNCE:"):
        raw_value = result.split(":", 1)[1].strip()
        try:
            parsed = int(raw_value)
        except Exception:
            ui.print_error("Usage: /session queue debounce <ms>")
            return "continue"
        ok, message = await update_active_session_queue_settings(ctx, debounce_ms=parsed)
        if ok:
            ui.print_success(message)
        else:
            ui.print_error(message)
        return "continue"

    if result.startswith("SESSION_QUEUE_CAP:"):
        raw_value = result.split(":", 1)[1].strip()
        try:
            parsed = int(raw_value)
        except Exception:
            ui.print_error("Usage: /session queue cap <n>")
            return "continue"
        ok, message = await update_active_session_queue_settings(ctx, cap=parsed)
        if ok:
            ui.print_success(message)
        else:
            ui.print_error(message)
        return "continue"

    if result.startswith("SESSION_QUEUE_DROP:"):
        drop_value = result.split(":", 1)[1].strip()
        ok, message = await update_active_session_queue_settings(ctx, drop_policy=drop_value)
        if ok:
            ui.print_success(message)
        else:
            ui.print_error(message)
        return "continue"

    if result == "SESSION_QUEUE_CLEAR":
        if not agent.session:
            ui.print_error("No active session")
            return "continue"
        session_id = agent.session.id
        followup_cleared = ctx.followup_queue.clear_queue(session_id)
        lane_cleared = ctx.command_queue.clear_lane(resolve_session_lane(session_id))
        ui.print_success(f"Cleared session queue: followup={followup_cleared} lane={lane_cleared}")
        return "continue"

    if result.startswith("SESSION_SELECT:"):
        selector = result.split(":", 1)[1].strip()
        selected = await agent.session_manager.select_session(selector)
        if not selected:
            ui.print_error(f"Session not found: {selector}")
            return "continue"
        agent.session = selected
        agent.refresh_session_runtime_flags()
        await agent.session_manager.set_last_active_session(agent.session.id)
        ui.load_monitor_tool_output_from_session(agent.session.messages)
        ui.print_session_info(agent.session)
        ui.print_success("Loaded session")
        return "continue"

    if result.startswith("SESSION_RENAME:"):
        new_name = result.split(":", 1)[1].strip()
        if not new_name:
            ui.print_error("Usage: /session rename <new-name>")
            return "continue"
        if not agent.session:
            ui.print_error("No active session")
            return "continue"
        old_name = agent.session.name
        agent.session.name = new_name
        await agent.session_manager.save_session(agent.session)
        ui.print_session_info(agent.session)
        ui.print_success(f'Session renamed: "{old_name}" -> "{new_name}"')
        return "continue"

    if result == "SESSION_DESCRIPTION_INFO":
        if not agent.session:
            ui.print_error("No active session")
            return "continue"
        description = str(agent.session.metadata.get("description", "")).strip()
        if description:
            ui.print_success(f"Session description: {description}")
        else:
            ui.print_warning("Session has no description yet")
        return "continue"

    if result == "SESSION_DESCRIPTION_AUTO":
        if not agent.session:
            ui.print_error("No active session")
            return "continue"
        generated = await agent.generate_session_description(agent.session, max_sentences=5)
        description = agent.sanitize_session_description(generated, max_sentences=5)
        if not description:
            ui.print_error("Could not generate a session description")
            return "continue"
        agent.session.metadata["description"] = description
        agent.session.metadata["description_source"] = "auto"
        agent.session.metadata["description_updated_at"] = datetime.now().isoformat()
        await agent.session_manager.save_session(agent.session)
        ui.print_session_info(agent.session)
        ui.print_success("Session description auto-generated")
        return "continue"

    if result.startswith("SESSION_DESCRIPTION_SET:"):
        if not agent.session:
            ui.print_error("No active session")
            return "continue"
        payload_raw = result.split(":", 1)[1].strip()
        try:
            payload = json.loads(payload_raw)
        except Exception:
            ui.print_error("Invalid /session description payload")
            return "continue"
        raw_description = str(payload.get("description", "")).strip()
        description = agent.sanitize_session_description(raw_description, max_sentences=5)
        if not description:
            ui.print_error("Usage: /session description <text> | /session description auto")
            return "continue"
        agent.session.metadata["description"] = description
        agent.session.metadata["description_source"] = "manual"
        agent.session.metadata["description_updated_at"] = datetime.now().isoformat()
        await agent.session_manager.save_session(agent.session)
        ui.print_session_info(agent.session)
        ui.print_success("Session description updated")
        return "continue"

    if result.startswith("SESSION_EXPORT:"):
        if not agent.session:
            ui.print_error("No active session")
            return "continue"
        mode = result.split(":", 1)[1].strip().lower() or "all"
        session_id = agent.session.id

        async def _export_task() -> list[Path]:
            return export_session_history(
                mode=mode,
                session_id=agent.session.id,
                session_name=agent.session.name,
                messages=agent.session.messages,
                saved_base_path=agent.tools.get_saved_base_path(create=True),
            )

        written_paths = await enqueue_agent_task(
            ctx, session_id, _export_task, lane=CommandLane.NESTED,
        )
        if not written_paths:
            ui.print_error("Failed to export session history")
            return "continue"
        ui.append_tool_output(
            "session_export",
            {"session_id": agent.session.id, "mode": mode, "count": len(written_paths)},
            "\n".join(f"path={path}" for path in written_paths),
        )
        for path in written_paths:
            ui.print_success(f"Exported: {path}")
        return "continue"

    if result.startswith("SESSION_RUN:"):
        payload_raw = result.split(":", 1)[1].strip()
        try:
            payload = json.loads(payload_raw)
        except Exception:
            ui.print_error("Invalid /session run payload")
            return "continue"
        selector = str(payload.get("selector", "")).strip()
        prompt = str(payload.get("prompt", "")).strip()
        if not selector or not prompt:
            ui.print_error("Usage: /session run <id|name|#index> <prompt>")
            return "continue"
        selected = await agent.session_manager.select_session(selector)
        if not selected:
            ui.print_error(f"Session not found: {selector}")
            return "continue"

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
                await run_prompt_in_active_session(
                    ctx, prompt, lane=CommandLane.NESTED, queue=False,
                )
            finally:
                if switched_temporarily and previous_session is not None:
                    restored = await agent.session_manager.load_session(previous_session_id)
                    agent.session = restored or previous_session
                    agent.refresh_session_runtime_flags()
                    ui.load_monitor_tool_output_from_session(agent.session.messages)
                    ui.print_success(f'Restored session "{agent.session.name}"')

        await enqueue_agent_task(
            ctx, selected.id, _run_selected_session_prompt, lane=CommandLane.NESTED,
        )
        return "continue"

    if result.startswith("SESSION_PROCREATE:"):
        payload_raw = result.split(":", 1)[1].strip()
        try:
            payload = json.loads(payload_raw)
        except Exception:
            ui.print_error("Invalid /session procreate payload")
            return "continue"
        parent_one_selector = str(payload.get("parent_one", "")).strip()
        parent_two_selector = str(payload.get("parent_two", "")).strip()
        new_name = str(payload.get("new_name", "")).strip()
        if not parent_one_selector or not parent_two_selector or not new_name:
            ui.print_error("Usage: /session procreate <id|name|#index> <id|name|#index> <new-name>")
            return "continue"
        ui.append_tool_output(
            "session_procreate",
            {"step": "resolve_parents", "parent_one_selector": parent_one_selector, "parent_two_selector": parent_two_selector},
            "step=resolve_parents\nstatus=locating_parent_sessions",
        )
        parent_one = await agent.session_manager.select_session(parent_one_selector)
        if not parent_one:
            ui.print_error(f"Session not found: {parent_one_selector}")
            return "continue"
        parent_two = await agent.session_manager.select_session(parent_two_selector)
        if not parent_two:
            ui.print_error(f"Session not found: {parent_two_selector}")
            return "continue"
        if parent_one.id == parent_two.id:
            ui.print_error("Choose two different sessions for /session procreate")
            return "continue"
        try:
            child_session, stats = await agent.procreate_sessions(
                parent_one=parent_one, parent_two=parent_two,
                new_name=new_name, persist=True,
            )
        except ValueError as e:
            ui.print_error(str(e))
            return "continue"
        ui.append_tool_output(
            "session_procreate",
            {"step": "switch_to_child", "session_id": child_session.id},
            f'step=switch_to_child\nsession_id="{child_session.id}"\nsession_name="{child_session.name}"',
        )
        agent.session = child_session
        agent.refresh_session_runtime_flags()
        await agent.session_manager.set_last_active_session(agent.session.id)
        ui.load_monitor_tool_output_from_session(agent.session.messages)
        ui.append_tool_output(
            "session_procreate",
            {"step": "complete", "session_id": child_session.id},
            f'step=complete\nsession_id="{child_session.id}"\nmerged_messages={stats.get("merged_messages", 0)}',
        )
        ui.print_session_info(agent.session)
        ui.print_success(
            f'Procreated session "{child_session.name}" '
            f"(merged_messages={stats.get('merged_messages', 0)}, "
            f"compacted={stats.get('parent_one_compacted', 0)}+{stats.get('parent_two_compacted', 0)})"
        )
        return "continue"

    if result == "CRON_LIST":
        jobs = await agent.session_manager.list_cron_jobs(limit=200, active_only=True)
        for job in jobs:
            if isinstance(job.schedule, dict):
                job.schedule["_text"] = schedule_to_text(job.schedule)
        ui.print_cron_jobs(jobs)
        return "continue"

    if result.startswith("CRON_HISTORY:"):
        selector = result.split(":", 1)[1].strip()
        job = await agent.session_manager.select_cron_job(selector, active_only=False)
        if not job:
            ui.print_error(f"Cron job not found: {selector}")
            return "continue"
        ui.print_cron_job_history(job)
        return "continue"

    if result.startswith("CRON_ONEOFF:"):
        payload_raw = result.split(":", 1)[1].strip()
        try:
            payload = json.loads(payload_raw)
        except Exception:
            ui.print_error("Invalid /cron payload")
            return "continue"
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            ui.print_error('Usage: /cron "<task>"')
            return "continue"
        if not agent.session:
            ui.print_error("No active session for /cron")
            return "continue"
        cron_monitor(ctx, "oneoff_prompt", session_id=agent.session.id if agent.session else "", chars=len(prompt))
        status = await dispatch_prompt_in_session(
            ctx,
            session_id=agent.session.id,
            prompt_text=prompt,
            source="cron:oneoff",
            cron_job_id=None,
            trigger="oneoff",
        )
        if status == "queued":
            ui.print_success("Cron one-off queued as follow-up (session busy)")
        return "continue"

    if result.startswith("CRON_ADD:"):
        if not agent.session:
            ui.print_error("No active session for /cron add")
            return "continue"
        raw_add = result.split(":", 1)[1].strip()
        try:
            schedule, kind, payload = parse_cron_add_args(raw_add)
        except ValueError as e:
            ui.print_error(str(e))
            return "continue"
        if kind in {"script", "tool"}:
            try:
                saved_base_path = agent.tools.get_saved_base_path(create=True)
                _ = resolve_saved_file_for_kind(
                    kind=kind, session_id=agent.session.id,
                    path_text=str(payload.get("path", "")),
                    saved_base_path=saved_base_path,
                )
            except ValueError as e:
                ui.print_error(str(e))
                return "continue"
        next_run_at_iso = to_utc_iso(compute_next_run(schedule))
        job = await agent.session_manager.create_cron_job(
            kind=kind, payload=payload, schedule=schedule,
            session_id=agent.session.id, next_run_at=next_run_at_iso, enabled=True,
        )
        await cron_monitor_event(
            ctx, "job_added", history_job_id=job.id,
            job_id=job.id, session_id=job.session_id, kind=job.kind,
            schedule=schedule_to_text(schedule), next_run_at=next_run_at_iso,
        )
        ui.print_success(
            f"Cron job added: id={job.id} kind={job.kind} "
            f"schedule={schedule_to_text(schedule)} next={next_run_at_iso}"
        )
        return "continue"

    if result.startswith("CRON_REMOVE:"):
        selector = result.split(":", 1)[1].strip()
        job = await agent.session_manager.select_cron_job(selector, active_only=False)
        if not job:
            ui.print_error(f"Cron job not found: {selector}")
            return "continue"
        job_id = job.id
        deleted = await agent.session_manager.delete_cron_job(job_id)
        if not deleted:
            ui.print_error(f"Cron job not found: {job_id}")
            return "continue"
        cron_monitor(ctx, "job_removed", job_id=job_id)
        ui.print_success(f"Removed cron job: {job_id}")
        return "continue"

    if result.startswith("CRON_PAUSE:"):
        selector = result.split(":", 1)[1].strip()
        job = await agent.session_manager.select_cron_job(selector, active_only=False)
        if not job:
            ui.print_error(f"Cron job not found: {selector}")
            return "continue"
        job_id = job.id
        updated = await agent.session_manager.update_cron_job(
            job_id, enabled=False, last_status="paused",
        )
        if not updated:
            ui.print_error(f"Cron job not found: {job_id}")
            return "continue"
        await cron_monitor_event(ctx, "job_paused", history_job_id=job_id, job_id=job_id)
        ui.print_success(f"Paused cron job: {job_id}")
        return "continue"

    if result.startswith("CRON_RESUME:"):
        selector = result.split(":", 1)[1].strip()
        job = await agent.session_manager.select_cron_job(selector, active_only=False)
        if not job:
            ui.print_error(f"Cron job not found: {selector}")
            return "continue"
        job_id = job.id
        updated = await agent.session_manager.update_cron_job(
            job_id, enabled=True, last_status="scheduled",
        )
        if not updated:
            ui.print_error(f"Cron job not found: {job_id}")
            return "continue"
        await cron_monitor_event(ctx, "job_resumed", history_job_id=job_id, job_id=job_id)
        ui.print_success(f"Resumed cron job: {job_id}")
        return "continue"

    if result.startswith("CRON_RUN:"):
        payload_raw = result.split(":", 1)[1].strip()
        try:
            payload = json.loads(payload_raw)
        except Exception:
            ui.print_error("Invalid /cron run payload")
            return "continue"
        raw_args = str(payload.get("args", "")).strip()
        if not raw_args:
            ui.print_error("Usage: /cron run <job-id|#index> | /cron run script <path> | /cron run tool <path>")
            return "continue"
        try:
            run_tokens = shlex.split(raw_args)
        except ValueError:
            ui.print_error("Invalid /cron run arguments")
            return "continue"
        if not run_tokens:
            ui.print_error("Usage: /cron run <job-id|#index> | /cron run script <path> | /cron run tool <path>")
            return "continue"
        head = run_tokens[0].strip().lower()
        if head in {"script", "tool"}:
            if not agent.session:
                ui.print_error("No active session for /cron run script|tool")
                return "continue"
            path_text = " ".join(run_tokens[1:]).strip()
            if not path_text:
                ui.print_error(f"Usage: /cron run {head} <path>")
                return "continue"
            try:
                await run_script_or_tool_in_session(
                    ctx,
                    target_session_id=agent.session.id,
                    kind=head, path_text=path_text, trigger="manual",
                )
                ui.print_success(f"Cron manual {head} run completed")
            except Exception as e:
                ui.print_error(str(e))
            return "continue"
        selector = run_tokens[0].strip()
        job = await agent.session_manager.select_cron_job(selector, active_only=False)
        if not job:
            ui.print_error(f"Cron job not found: {selector}")
            return "continue"
        job_id = job.id
        await execute_cron_job(ctx, job, trigger="manual")
        ui.print_success(f"Manual cron run finished: {job_id}")
        return "continue"

    if result == "CONFIG":
        ui.print_message("system", format_active_configuration_text(ctx))
        return "continue"

    if result == "HISTORY":
        if agent.session:
            ui.print_history(agent.session.messages)
        return "continue"

    if result == "COMPACT":
        compacted, stats = await agent.compact_session(force=True, trigger="manual")
        if compacted:
            if agent.session:
                ui.load_monitor_tool_output_from_session(agent.session.messages)
            ui.print_success(
                f"Session compacted ({int(stats.get('before_tokens', 0))} -> {int(stats.get('after_tokens', 0))} tokens)"
            )
        else:
            reason = str(stats.get("reason", "not_needed"))
            ui.print_warning(f"Compaction skipped: {reason}")
        return "continue"

    if result == "PLANNING_ON":
        await agent.set_pipeline_mode("contracts")
        ui.print_success("Pipeline mode set to contracts")
        return "continue"

    if result == "PLANNING_OFF":
        await agent.set_pipeline_mode("loop")
        ui.print_success("Pipeline mode set to loop")
        return "continue"

    if result.startswith("ORCHESTRATE:"):
        payload_raw = result.split(":", 1)[1].strip()
        try:
            payload = json.loads(payload_raw)
        except Exception:
            ui.print_error("Invalid /orchestrate payload")
            return "continue"
        request_text = str(payload.get("request", "")).strip()
        if not request_text:
            ui.print_error("Usage: /orchestrate <request>")
            return "continue"
        ui.print_message("system", f"Orchestrating: {request_text}")
        try:
            from captain_claw.session_orchestrator import SessionOrchestrator
            orchestrator = SessionOrchestrator(
                main_agent=agent,
                provider=agent.provider,
                status_callback=agent.status_callback,
                tool_output_callback=agent.tool_output_callback,
            )
            orch_response, cancelled = await run_cancellable(
                ui,
                enqueue_agent_task(
                    ctx,
                    agent.session.id if agent.session else None,
                    lambda: orchestrator.orchestrate(request_text),
                ),
            )
            if cancelled:
                ui.print_warning("Orchestration cancelled.")
                await orchestrator.shutdown()
                return "continue"
            if orch_response:
                ui.print_message("assistant", str(orch_response))
            await orchestrator.shutdown()
        except Exception as e:
            ui.print_error(f"Orchestration failed: {e}")
        return "continue"

    if result == "PIPELINE_INFO":
        ui.print_success(
            f"Pipeline mode: {agent.pipeline_mode} "
            "(loop=fast/simple, contracts=planner+completion gate)"
        )
        return "continue"

    if result.startswith("PIPELINE_MODE:"):
        mode = result.split(":", 1)[1].strip().lower()
        try:
            await agent.set_pipeline_mode(mode)
        except ValueError:
            ui.print_error("Invalid pipeline mode. Use /pipeline loop|contracts")
            return "continue"
        ui.print_success(
            f"Pipeline mode set to {agent.pipeline_mode} "
            "(loop=fast/simple, contracts=planner+completion gate)"
        )
        return "continue"

    if result == "SKILLS_LIST":
        skills = agent.list_user_invocable_skills()
        if not skills:
            ui.print_warning("No user-invocable skills available.")
            return "continue"
        lines = ["Available skills:", "Use `/skill <name> [args]` to run one:"]
        for command in skills:
            lines.append(f"- /skill {command.name} - {command.description}")
        lines.append("Search catalog: `/skill search <criteria>`")
        lines.append("Install from GitHub: `/skill install <github-url>`")
        lines.append("Install skill deps: `/skill install <skill-name> [install-id]`")
        ui.print_message("system", "\n".join(lines))
        return "continue"

    if result.startswith("SKILL_SEARCH:"):
        payload_raw = result.split(":", 1)[1].strip()
        try:
            payload = json.loads(payload_raw)
        except Exception:
            ui.print_error("Invalid /skill search payload")
            return "continue"
        query = str(payload.get("query", "")).strip()
        if not query:
            ui.print_error("Usage: /skill search <criteria>")
            return "continue"
        ui.print_message("system", f'Searching skills catalog for: "{query}"')
        search_result = await agent.search_skill_catalog(query)
        if not bool(search_result.get("ok", False)):
            ui.print_error(str(search_result.get("error", "Skill search failed.")))
            return "continue"
        source = str(search_result.get("source", "")).strip()
        items = list(search_result.get("results", []))
        lines = [f'Top skills for "{query}":']
        if source:
            lines.append(f"Source: {source}")
        if not items:
            lines.append("No matching skills found.")
        for idx, item in enumerate(items, start=1):
            name = str(item.get("name", "")).strip() or "Unnamed"
            desc = str(item.get("description", "")).strip()
            url = str(item.get("url", "")).strip()
            line = f"{idx}. {name}"
            if desc:
                line += f" - {desc}"
            lines.append(line)
            if url:
                lines.append(f"   {url}")
        ui.print_message("system", "\n".join(lines))
        return "continue"

    if result.startswith("SKILL_INSTALL:"):
        payload_raw = result.split(":", 1)[1].strip()
        try:
            payload = json.loads(payload_raw)
        except Exception:
            ui.print_error("Invalid /skill install payload")
            return "continue"
        skill_url = str(payload.get("url", "")).strip()
        skill_name = str(payload.get("name", "")).strip()
        install_id = str(payload.get("install_id", "")).strip()
        if skill_url:
            ui.print_message("system", f"Installing skill from GitHub: {skill_url}")
            install_result = await agent.install_skill_from_github(skill_url)
            if not bool(install_result.get("ok", False)):
                ui.print_error(str(install_result.get("error", "Skill install failed.")))
                return "continue"
            skill_name = str(install_result.get("skill_name", "")).strip() or "unknown"
            destination = str(install_result.get("destination", "")).strip()
            aliases = list(install_result.get("aliases", []))
            ui.print_success(f'Installed skill "{skill_name}"')
            if destination:
                ui.print_message("system", f"Path: {destination}")
            if aliases:
                ui.print_message("system", f"Invoke with: /skill {aliases[0]}")
            return "continue"
        if not skill_name:
            ui.print_error("Usage: /skill install <github-url> | /skill install <skill-name> [install-id]")
            return "continue"
        ui.print_message("system", f'Installing dependencies for skill "{skill_name}"')
        install_result = await agent.install_skill_dependencies(
            skill_name=skill_name, install_id=install_id or None,
        )
        if not bool(install_result.get("ok", False)):
            ui.print_error(str(install_result.get("error", "Skill dependency install failed.")))
            return "continue"
        ui.print_success(str(install_result.get("message", "Dependencies installed.")))
        command = str(install_result.get("command", "")).strip()
        if command:
            ui.print_message("system", f"Command: {command}")
        return "continue"

    if result.startswith("SKILL_INVOKE:") or result.startswith("SKILL_ALIAS_INVOKE:"):
        payload_raw = result.split(":", 1)[1].strip()
        try:
            payload = json.loads(payload_raw)
        except Exception:
            ui.print_error("Invalid /skill payload")
            return "continue"
        skill_name = str(payload.get("name", "")).strip()
        skill_args = str(payload.get("args", "")).strip()
        if not skill_name:
            ui.print_error(
                "Usage: /skill <name> [args] | /skill search <criteria> | "
                "/skill install <github-url> | /skill install <skill-name> [install-id]"
            )
            return "continue"
        invocation = await agent.invoke_skill_command(skill_name, args=skill_args)
        if not bool(invocation.get("ok", False)):
            if result.startswith("SKILL_ALIAS_INVOKE:"):
                ui.print_error(f"Unknown command: /{skill_name}")
            else:
                ui.print_error(str(invocation.get("error", "Skill invocation failed.")))
            return "continue"
        mode = str(invocation.get("mode", "")).strip().lower()
        if mode == "dispatch":
            text = str(invocation.get("text", "")).strip() or "Done."
            ui.print_message("assistant", text)
            return "continue"
        rewritten_prompt = str(invocation.get("prompt", "")).strip()
        if not rewritten_prompt:
            ui.print_error("Skill invocation did not return a prompt.")
            return "continue"
        display_prompt = f"/skill {skill_name}"
        if skill_args:
            display_prompt += f" {skill_args}"
        await run_prompt_in_active_session(ctx, rewritten_prompt, display_prompt=display_prompt)
        return "continue"

    if result == "MONITOR_ON":
        ui.set_monitor_mode(True)
        if agent.session:
            ui.load_monitor_tool_output_from_session(agent.session.messages)
        ui.print_success("Monitor enabled")
        return "continue"

    if result == "MONITOR_OFF":
        ui.set_monitor_mode(False)
        ui.print_success("Monitor disabled")
        return "continue"

    if result == "MONITOR_TRACE_ON":
        await agent.set_monitor_trace_llm(True)
        ui.print_success("Monitor trace enabled (full intermediate LLM responses will be logged)")
        return "continue"

    if result == "MONITOR_TRACE_OFF":
        await agent.set_monitor_trace_llm(False)
        ui.print_success("Monitor trace disabled")
        return "continue"

    if result == "MONITOR_PIPELINE_ON":
        await agent.set_monitor_trace_pipeline(True)
        ui.print_success("Pipeline trace enabled (compact pipeline-only events will be logged)")
        return "continue"

    if result == "MONITOR_PIPELINE_OFF":
        await agent.set_monitor_trace_pipeline(False)
        ui.print_success("Pipeline trace disabled")
        return "continue"

    if result == "MONITOR_FULL_ON":
        ui.set_monitor_full_output(True)
        ui.print_success("Monitor full output rendering enabled")
        return "continue"

    if result == "MONITOR_FULL_OFF":
        ui.set_monitor_full_output(False)
        ui.print_success("Monitor compact output rendering enabled")
        return "continue"

    if result == "MONITOR_SCROLL_STATUS":
        ui.print_success(f"Monitor scroll: {ui.describe_monitor_scroll()}")
        return "continue"

    if result.startswith("MONITOR_SCROLL:"):
        payload_raw = result.split(":", 1)[1].strip()
        try:
            payload = json.loads(payload_raw)
        except Exception:
            ui.print_error("Invalid /monitor scroll payload")
            return "continue"
        pane = str(payload.get("pane", "")).strip().lower()
        action = str(payload.get("action", "")).strip().lower()
        amount_raw = payload.get("amount", 1)
        try:
            amount = int(amount_raw)
        except Exception:
            ui.print_error("Invalid scroll amount")
            return "continue"
        ok, message = ui.scroll_monitor_pane(pane=pane, action=action, amount=amount)
        if ok:
            ui.print_success(message)
        else:
            ui.print_error(message)
        return "continue"

    # Not a recognized special command -> fall through to prompt execution
    return None
