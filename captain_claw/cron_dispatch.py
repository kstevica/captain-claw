"""Cron scheduling, job execution, and history management.

All functions receive a :class:`RuntimeContext` instead of closing over
the mutable state that used to live inside ``run_interactive()``.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from captain_claw.logging import get_logger

from captain_claw.cron import (
    compute_next_run,
    now_utc,
    parse_schedule_tokens,
    schedule_to_text,
    to_utc_iso,
)
from captain_claw.execution_queue import CommandLane
from captain_claw.session_export import normalize_session_id, truncate_history_text

if TYPE_CHECKING:
    from captain_claw.runtime_context import RuntimeContext

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cron monitoring / history
# ---------------------------------------------------------------------------

def cron_monitor(ctx: RuntimeContext, step: str, **args: object) -> None:
    payload: dict[str, object] = {"step": step}
    payload.update(args)
    lines = [f"step={step}"]
    for key, value in args.items():
        lines.append(f"{key}={value}")
    ctx.ui.append_tool_output("cron", payload, "\n".join(lines))


async def append_cron_history(
    ctx: RuntimeContext,
    job_id: str | None = None,
    *,
    chat_event: dict[str, object] | None = None,
    monitor_event: dict[str, object] | None = None,
) -> None:
    if not job_id:
        return
    await ctx.agent.session_manager.append_cron_job_history(
        job_id,
        chat_event=chat_event,
        monitor_event=monitor_event,
    )


async def cron_monitor_event(
    ctx: RuntimeContext, step: str,
    history_job_id: str | None = None,
    **args: object,
) -> None:
    cron_monitor(ctx, step, **args)
    event: dict[str, object] = {"timestamp": to_utc_iso(now_utc()), "step": step}
    event.update(args)
    await append_cron_history(ctx, history_job_id, monitor_event=event)


async def cron_chat_event(
    ctx: RuntimeContext,
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
    await append_cron_history(ctx, job_id, chat_event=event)


# ---------------------------------------------------------------------------
# Saved file resolution
# ---------------------------------------------------------------------------

def resolve_saved_file_for_kind(
    kind: str, session_id: str, path_text: str,
    saved_base_path: Path,
) -> Path:
    safe_session = normalize_session_id(session_id)
    categories = {"downloads", "media", "scripts", "showcase", "skills", "tmp", "tools"}
    requested = Path(path_text).expanduser()

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
            candidate = (saved_base_path / scoped_rel).resolve()
        else:
            default_category = "scripts" if kind == "script" else "tools"
            candidate = (saved_base_path / default_category / safe_session / requested).resolve()

    try:
        relative_candidate = candidate.relative_to(saved_base_path)
    except ValueError as e:
        raise ValueError(f"Path must be inside saved root: {saved_base_path}") from e

    relative_parts = [part for part in relative_candidate.parts if part not in ("", ".", "..")]
    if relative_parts and relative_parts[0] in categories:
        if len(relative_parts) < 2 or relative_parts[1] != safe_session:
            expected_prefix = f"{relative_parts[0]}/{safe_session}"
            raise ValueError(f"{kind} path must be inside saved/{expected_prefix}/...")

    if not candidate.exists() or not candidate.is_file():
        raise ValueError(f"{kind} file not found: {candidate}")
    return candidate


# ---------------------------------------------------------------------------
# Script / tool execution in session
# ---------------------------------------------------------------------------

async def run_script_or_tool_in_session(
    ctx: RuntimeContext,
    target_session_id: str,
    kind: str,
    path_text: str,
    trigger: str,
    cron_job_id: str | None = None,
) -> None:
    from captain_claw.prompt_execution import enqueue_agent_task

    agent = ctx.agent
    target_session = await agent.session_manager.load_session(target_session_id)
    if not target_session:
        raise ValueError(f"Session not found: {target_session_id}")

    saved_base_path = agent.tools.get_saved_base_path(create=True)
    file_path = resolve_saved_file_for_kind(
        kind=kind, session_id=target_session_id,
        path_text=path_text, saved_base_path=saved_base_path,
    )
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
            ctx.ui.print_message(
                "system",
                f"[CRON] {trigger} {kind} run in session={target_session.id} path={file_path}",
            )
            if agent.session:
                start_note = f"[CRON] {trigger} {kind} start: {file_path}"
                agent.session.add_message(
                    role="tool", content=start_note, tool_name="cron",
                    tool_arguments={
                        "step": "run_script_tool_start", "trigger": trigger,
                        "kind": kind, "path": str(file_path),
                        "job_id": cron_job_id or "",
                    },
                    token_count=agent._count_tokens(start_note),
                )
                await agent.session_manager.save_session(agent.session)
            await cron_chat_event(
                ctx, cron_job_id, "system",
                f"[CRON] {trigger} {kind} run start: {file_path}",
                trigger=trigger, kind=kind, session_id=target_session.id,
            )
            await cron_monitor_event(
                ctx, "run_script_tool_start",
                history_job_id=cron_job_id,
                trigger=trigger, kind=kind,
                path=str(file_path), session_id=target_session.id,
            )
            result = await agent._execute_tool_with_guard(
                name="shell",
                arguments={"command": command},
                interaction_label=f"cron_{kind}_{trigger}",
            )
            shell_output = result.content if result.success else f"Error: {result.error}"
            if agent.session:
                agent.session.add_message(
                    role="tool", content=shell_output, tool_name="shell",
                    tool_arguments={"command": command, "cron": True, "job_id": cron_job_id or ""},
                    token_count=agent._count_tokens(shell_output),
                )
                await agent.session_manager.save_session(agent.session)
            ctx.ui.append_tool_output("shell", {"command": command, "cron": True}, shell_output)
            await cron_monitor_event(
                ctx, "run_script_tool_output",
                history_job_id=cron_job_id,
                trigger=trigger, kind=kind,
                path=str(file_path), session_id=target_session.id,
                output=truncate_history_text(shell_output),
            )
            await cron_chat_event(
                ctx, cron_job_id, "tool", shell_output,
                trigger=trigger, kind=kind,
                session_id=target_session.id, path=str(file_path),
            )
            if not result.success:
                raise RuntimeError(shell_output)
            await cron_monitor_event(
                ctx, "run_script_tool_done",
                history_job_id=cron_job_id,
                trigger=trigger, kind=kind,
                path=str(file_path), session_id=target_session.id,
            )
            await cron_chat_event(
                ctx, cron_job_id, "system",
                f"[CRON] {trigger} {kind} run complete: {file_path}",
                trigger=trigger, kind=kind, session_id=target_session.id,
            )
            if agent.session:
                done_note = f"[CRON] {trigger} {kind} complete: {file_path}"
                agent.session.add_message(
                    role="tool", content=done_note, tool_name="cron",
                    tool_arguments={
                        "step": "run_script_tool_done", "trigger": trigger,
                        "kind": kind, "path": str(file_path),
                        "job_id": cron_job_id or "",
                    },
                    token_count=agent._count_tokens(done_note),
                )
                await agent.session_manager.save_session(agent.session)
        except Exception as e:
            if agent.session:
                failed_note = f"[CRON] {trigger} {kind} failed: {e}"
                agent.session.add_message(
                    role="tool", content=failed_note, tool_name="cron",
                    tool_arguments={
                        "step": "run_script_tool_failed", "trigger": trigger,
                        "kind": kind, "path": str(file_path),
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

    await enqueue_agent_task(ctx, target_session.id, _execute, lane=CommandLane.CRON)


# ---------------------------------------------------------------------------
# Cron add argument parsing
# ---------------------------------------------------------------------------

def parse_cron_add_args(raw_add: str) -> tuple[dict[str, object], str, dict[str, str]]:
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

    if kind_head == "orchestrate":
        # /cron add every 1h orchestrate <workflow-name>
        if len(remaining) < 2:
            raise ValueError("Usage: /cron add ... orchestrate <workflow-name>")
        wf_name = " ".join(remaining[1:]).strip()
        if not wf_name:
            raise ValueError("Usage: /cron add ... orchestrate <workflow-name>")
        return schedule, "orchestrate", {"workflow": wf_name}

    prompt_text = " ".join(remaining).strip()
    if not prompt_text:
        raise ValueError('Usage: /cron add ... "<task>"')
    return schedule, "prompt", {"text": prompt_text}


# ---------------------------------------------------------------------------
# Orchestrate workflow execution (cron kind)
# ---------------------------------------------------------------------------

async def _run_orchestrate_cron(
    ctx: RuntimeContext,
    workflow_name: str,
    trigger: str,
    cron_job_id: str | None = None,
) -> None:
    """Run a saved orchestrator workflow as a cron job.

    Creates a fresh SessionOrchestrator, loads the named workflow,
    executes it, and logs the result.  Runs inline (not queued) because
    orchestration manages its own parallelism via AgentPool.
    """
    from captain_claw.config import get_config
    from captain_claw.session_orchestrator import SessionOrchestrator

    cfg = get_config()

    ctx.ui.print_message(
        "system",
        f"[CRON] {trigger} orchestrate workflow={workflow_name}",
    )

    await cron_monitor_event(
        ctx, "orchestrate_start",
        history_job_id=cron_job_id,
        trigger=trigger, workflow=workflow_name,
    )
    await cron_chat_event(
        ctx, cron_job_id, "system",
        f"[CRON] {trigger} orchestrate start: {workflow_name}",
        trigger=trigger, kind="orchestrate", workflow=workflow_name,
    )

    # Create a temporary orchestrator for this run.
    orchestrator = SessionOrchestrator(
        main_agent=ctx.agent,
        max_parallel=cfg.orchestrator.max_parallel,
        max_agents=cfg.orchestrator.max_agents,
        provider=ctx.agent.provider,
        status_callback=ctx.ui.set_runtime_status,
        tool_output_callback=ctx.ui.append_tool_output,
    )

    try:
        load_result = await orchestrator.load_workflow(workflow_name)
        if not load_result.get("ok"):
            raise ValueError(load_result.get("error", f"Failed to load workflow: {workflow_name}"))

        task_count = len(load_result.get("tasks", []))
        await cron_monitor_event(
            ctx, "orchestrate_executing",
            history_job_id=cron_job_id,
            trigger=trigger, workflow=workflow_name, task_count=task_count,
        )

        synthesis = await orchestrator.execute()

        await cron_monitor_event(
            ctx, "orchestrate_done",
            history_job_id=cron_job_id,
            trigger=trigger, workflow=workflow_name,
            result_preview=truncate_history_text(synthesis),
        )
        await cron_chat_event(
            ctx, cron_job_id, "assistant",
            truncate_history_text(synthesis),
            trigger=trigger, kind="orchestrate", workflow=workflow_name,
        )
        ctx.ui.print_message(
            "system",
            f"[CRON] {trigger} orchestrate complete: {workflow_name}",
        )
    finally:
        await orchestrator.shutdown()


# ---------------------------------------------------------------------------
# Cron job execution
# ---------------------------------------------------------------------------

async def execute_cron_job(ctx: RuntimeContext, job: Any, trigger: str = "scheduled") -> None:
    from captain_claw.prompt_execution import dispatch_prompt_in_session

    job_id = str(getattr(job, "id", "")).strip()
    if not job_id or job_id in ctx.cron_running_job_ids:
        log.debug("cron_job_skipped", job_id=job_id, already_running=job_id in ctx.cron_running_job_ids)
        return
    log.info("cron_job_execute_start", job_id=job_id, trigger=trigger)

    ctx.cron_running_job_ids.add(job_id)
    started_at_iso = to_utc_iso(now_utc())
    success = False
    queued_for_followup = False
    error_text = ""
    try:
        kind = str(getattr(job, "kind", "prompt")).strip().lower()
        payload = getattr(job, "payload", {})
        session_id = str(getattr(job, "session_id", "")).strip()
        schedule = getattr(job, "schedule", {})

        await cron_monitor_event(
            ctx, "job_start", history_job_id=job_id,
            trigger=trigger, job_id=job_id, kind=kind, session_id=session_id,
        )
        await cron_chat_event(
            ctx, job_id, "system",
            f"[CRON] job start trigger={trigger} kind={kind} session={session_id}",
            trigger=trigger, kind=kind, session_id=session_id,
        )
        if not session_id:
            raise ValueError(f"Cron job {job_id} has no session_id")

        if kind == "prompt":
            prompt_text = str(payload.get("text", "")).strip() if isinstance(payload, dict) else ""
            if not prompt_text:
                raise ValueError(f"Cron job {job_id} has empty prompt payload")
            dispatch_status = await dispatch_prompt_in_session(
                ctx,
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
            await run_script_or_tool_in_session(
                ctx,
                target_session_id=session_id,
                kind=kind,
                path_text=path_text,
                trigger=trigger,
                cron_job_id=job_id,
            )
        elif kind == "orchestrate":
            workflow_name = str(payload.get("workflow", "")).strip() if isinstance(payload, dict) else ""
            if not workflow_name:
                raise ValueError(f"Cron job {job_id} has empty orchestrate workflow name")
            await _run_orchestrate_cron(
                ctx,
                workflow_name=workflow_name,
                trigger=trigger,
                cron_job_id=job_id,
            )
        else:
            raise ValueError(f"Unsupported cron job kind: {kind}")

        success = True
        log.info("cron_job_execute_done", job_id=job_id, trigger=trigger, kind=kind)
        await cron_monitor_event(
            ctx,
            "job_queued" if queued_for_followup else "job_done",
            history_job_id=job_id,
            trigger=trigger, job_id=job_id, kind=kind, session_id=session_id,
        )
        await cron_chat_event(
            ctx, job_id, "system",
            (
                f"[CRON] job queued trigger={trigger} kind={kind} session={session_id}"
                if queued_for_followup
                else f"[CRON] job done trigger={trigger} kind={kind} session={session_id}"
            ),
            trigger=trigger, kind=kind, session_id=session_id,
        )

        next_run_at_iso = to_utc_iso(compute_next_run(schedule))
        await ctx.agent.session_manager.update_cron_job(
            job_id,
            last_run_at=started_at_iso,
            next_run_at=next_run_at_iso,
            last_status="queued" if queued_for_followup else "ok",
            last_error="",
        )
    except Exception as e:
        error_text = str(e)
        log.error("cron_job_execute_failed", job_id=job_id, trigger=trigger, error=error_text)
        await cron_monitor_event(
            ctx, "job_failed", history_job_id=job_id,
            trigger=trigger, job_id=job_id, error=error_text,
        )
        await cron_chat_event(
            ctx, job_id, "system",
            f"[CRON] job failed: {error_text}",
            trigger=trigger,
        )
        try:
            next_run_at_iso = to_utc_iso(compute_next_run(getattr(job, "schedule", {})))
        except Exception:
            next_run_at_iso = to_utc_iso(now_utc())
        await ctx.agent.session_manager.update_cron_job(
            job_id,
            last_run_at=started_at_iso,
            next_run_at=next_run_at_iso,
            last_status="failed",
            last_error=error_text,
        )
    finally:
        ctx.cron_running_job_ids.discard(job_id)
        if trigger == "manual" and not success:
            ctx.ui.print_error(error_text or f"Cron job failed: {job_id}")


async def cron_scheduler_loop(ctx: RuntimeContext) -> None:
    """Background pseudo-cron runner."""
    import asyncio

    log.info("cron_scheduler_loop_started", poll_seconds=ctx.cron_poll_seconds)
    while True:
        await asyncio.sleep(ctx.cron_poll_seconds)
        try:
            now_iso = to_utc_iso(now_utc())
            due_jobs = await ctx.agent.session_manager.get_due_cron_jobs(
                now_iso=now_iso,
                limit=10,
            )
            if due_jobs:
                log.info("cron_due_jobs_found", count=len(due_jobs), now=now_iso)
            for job in due_jobs:
                job_id = getattr(job, "id", "?")
                kind = getattr(job, "kind", "?")
                log.info("cron_executing_job", job_id=job_id, kind=kind)
                await execute_cron_job(ctx, job, trigger="scheduled")
        except asyncio.CancelledError:
            log.info("cron_scheduler_loop_cancelled")
            raise
        except Exception:
            log.exception("cron_scheduler_loop_error")
