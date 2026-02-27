"""REST handlers for orchestrator control, task management, and workflows."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw.instructions import InstructionLoader
from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


async def get_orchestrator_status(server: WebServer, request: web.Request) -> web.Response:
    """REST endpoint: current orchestrator graph state."""
    if not server._orchestrator:
        return web.json_response({"status": None})
    status = server._orchestrator.get_status()
    return web.json_response({"status": status}, dumps=lambda obj: json.dumps(obj, default=str))


async def reset_orchestrator(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/orchestrator/reset — cancel work and reset to idle."""
    if not server._orchestrator:
        return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
    try:
        await server._orchestrator.reset()
    except Exception as e:
        log.error("Orchestrator reset error", error=str(e))
        return web.json_response({"ok": False, "error": str(e)}, status=500)
    return web.json_response({"ok": True})


async def get_orchestrator_skills(server: WebServer, request: web.Request) -> web.Response:
    """REST endpoint: list available skills for orchestrator workers."""
    if not server.agent:
        return web.json_response({"skills": []})
    try:
        commands = server.agent.list_user_invocable_skills()
        skills = [
            {"name": cmd.name, "skill_name": cmd.skill_name, "description": cmd.description}
            for cmd in commands
        ]
    except Exception:
        skills = []

    try:
        tool_names = server.agent.tools.list_tools()
    except Exception:
        tool_names = []

    return web.json_response({"skills": skills, "tools": tool_names})


async def rephrase_orchestrator_input(server: WebServer, request: web.Request) -> web.Response:
    """Rephrase a casual user request into a structured orchestrator prompt."""
    from captain_claw.llm import Message

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    user_input = str(body.get("input", "")).strip()
    if not user_input:
        return web.json_response({"error": "Empty input"}, status=400)

    model_selector = str(body.get("model", "")).strip() or None

    provider = server.agent.provider if server.agent else None
    if provider is None:
        from captain_claw.llm import get_provider
        provider = get_provider()

    # If a model override was requested, resolve it and create a temp provider.
    if model_selector and server.agent:
        resolved = server.agent._resolve_allowed_model(model_selector)
        if resolved:
            from captain_claw.llm import create_provider
            from captain_claw.config import get_config as _cfg
            _c = _cfg()
            provider = create_provider(
                provider=str(resolved.get("provider", _c.model.provider)),
                model=str(resolved.get("model", _c.model.model)),
                api_key=server.agent._resolve_provider_api_key(
                    server.agent._normalize_provider_key(str(resolved.get("provider", "")))
                ) or _c.model.api_key or None,
                base_url=str(resolved.get("base_url", "") or "") or _c.model.base_url or None,
                temperature=float(resolved.get("temperature") if resolved.get("temperature") is not None else _c.model.temperature),
                max_tokens=int(resolved.get("max_tokens") if resolved.get("max_tokens") is not None else _c.model.max_tokens),
                tokens_per_minute=_c.model.tokens_per_minute,
            )
            log.info("Rephrase using model override", model=str(resolved.get("model", "")))

    loader = InstructionLoader()
    prompt = loader.render(
        "orchestrator_rephrase_prompt.md",
        user_input=user_input,
    )

    rephrase_messages = [Message(role="user", content=prompt)]
    try:
        import asyncio as _asyncio

        response = await _asyncio.wait_for(
            provider.complete(
                messages=rephrase_messages,
                tools=None,
                max_tokens=2000,
            ),
            timeout=60.0,
        )
        rephrased = str(getattr(response, "content", "") or "").strip()
        if not rephrased:
            rephrased = user_input
        # File-based session logging
        from captain_claw.config import get_config as _get_config
        if _get_config().logging.llm_session_logging:
            try:
                from captain_claw.llm_session_logger import get_llm_session_logger
                llm_log = get_llm_session_logger()
                llm_log.set_session("orchestrator")
                llm_log.log_call(
                    interaction_label="orchestrator_rephrase",
                    model=str(getattr(provider, "model", "") or ""),
                    messages=rephrase_messages,
                    response=response,
                    instruction_files=["orchestrator_rephrase_prompt.md"],
                    max_tokens=2000,
                )
            except Exception:
                pass
    except Exception as e:
        log.error("Rephrase failed", error=str(e))
        rephrased = user_input

    return web.json_response({"rephrased": rephrased, "original": user_input})


# ── Task control endpoints ───────────────────────────────────────────


async def edit_orchestrator_task(server: WebServer, request: web.Request) -> web.Response:
    """Put a task into edit mode."""
    if not server._orchestrator:
        return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)
    task_id = str(body.get("task_id", "")).strip()
    if not task_id:
        return web.json_response({"ok": False, "error": "Missing task_id"}, status=400)
    result = await server._orchestrator.edit_task(task_id)
    return web.json_response(result)


async def update_orchestrator_task(server: WebServer, request: web.Request) -> web.Response:
    """Update task instructions (description)."""
    if not server._orchestrator:
        return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)
    task_id = str(body.get("task_id", "")).strip()
    description = str(body.get("description", ""))
    if not task_id:
        return web.json_response({"ok": False, "error": "Missing task_id"}, status=400)
    result = await server._orchestrator.update_task(task_id, description)
    return web.json_response(result)


async def restart_orchestrator_task(server: WebServer, request: web.Request) -> web.Response:
    """Restart a failed/completed/paused task."""
    if not server._orchestrator:
        return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)
    task_id = str(body.get("task_id", "")).strip()
    if not task_id:
        return web.json_response({"ok": False, "error": "Missing task_id"}, status=400)
    result = await server._orchestrator.restart_task(task_id)
    return web.json_response(result)


async def pause_orchestrator_task(server: WebServer, request: web.Request) -> web.Response:
    """Pause a running task."""
    if not server._orchestrator:
        return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)
    task_id = str(body.get("task_id", "")).strip()
    if not task_id:
        return web.json_response({"ok": False, "error": "Missing task_id"}, status=400)
    result = await server._orchestrator.pause_task(task_id)
    return web.json_response(result)


async def resume_orchestrator_task(server: WebServer, request: web.Request) -> web.Response:
    """Resume a paused/editing task."""
    if not server._orchestrator:
        return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)
    task_id = str(body.get("task_id", "")).strip()
    if not task_id:
        return web.json_response({"ok": False, "error": "Missing task_id"}, status=400)
    result = await server._orchestrator.resume_task(task_id)
    return web.json_response(result)


async def postpone_orchestrator_task(server: WebServer, request: web.Request) -> web.Response:
    """Postpone a task's timeout warning."""
    if not server._orchestrator:
        return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)
    task_id = str(body.get("task_id", "")).strip()
    if not task_id:
        return web.json_response({"ok": False, "error": "Missing task_id"}, status=400)
    result = await server._orchestrator.postpone_task(task_id)
    return web.json_response(result)


# ── Prepare / workflows / sessions / models ──────────────────────────


async def prepare_orchestrator(server: WebServer, request: web.Request) -> web.Response:
    """Decompose a request into tasks without executing (preview)."""
    if not server._orchestrator:
        log.warning("_prepare_orchestrator: no orchestrator available")
        return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
    try:
        body = await request.json()
    except Exception as e:
        log.error("_prepare_orchestrator: invalid JSON body", error=str(e))
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)
    user_input = str(body.get("input", "")).strip()
    if not user_input:
        log.warning("_prepare_orchestrator: empty input")
        return web.json_response({"ok": False, "error": "Missing input"}, status=400)
    model = str(body.get("model", "")).strip() or None
    auto_select_model = bool(body.get("auto_select_model", False))
    log.info("_prepare_orchestrator: calling prepare",
             input_len=len(user_input), input_preview=user_input[:150],
             model=model, auto_select_model=auto_select_model)
    try:
        result = await server._orchestrator.prepare(
            user_input, model=model, auto_select_model=auto_select_model,
        )
    except Exception as e:
        log.error("_prepare_orchestrator: prepare() raised exception",
                  error=str(e), error_type=type(e).__name__)
        return web.json_response(
            {"ok": False, "error": f"Prepare failed: {e}"}, status=500)
    log.info("_prepare_orchestrator: result",
             ok=result.get("ok"), task_count=len(result.get("tasks", [])),
             error=result.get("error"))
    return web.json_response(result, dumps=lambda obj: json.dumps(obj, default=str))


async def get_orchestrator_sessions(server: WebServer, request: web.Request) -> web.Response:
    """List sessions for per-task session selection."""
    if not server.agent:
        return web.json_response({"sessions": []})
    try:
        sessions = await server.agent.session_manager.list_sessions(limit=30)
        result = [
            {"id": s.id, "name": s.name}
            for s in sessions
        ]
    except Exception:
        result = []
    return web.json_response({"sessions": result})


async def get_orchestrator_models(server: WebServer, request: web.Request) -> web.Response:
    """List allowed models for per-task model selection."""
    if not server.agent:
        return web.json_response({"models": []})
    try:
        models = server.agent.get_allowed_models()
    except Exception:
        models = []
    return web.json_response({"models": models})


async def list_workflows(server: WebServer, request: web.Request) -> web.Response:
    """List saved workflows."""
    if not server._orchestrator:
        return web.json_response({"workflows": []})
    workflows = await server._orchestrator.list_workflows()
    return web.json_response({"workflows": workflows})


async def save_workflow(server: WebServer, request: web.Request) -> web.Response:
    """Save the current graph as a workflow."""
    if not server._orchestrator:
        return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)
    name = str(body.get("name", "")).strip() or None
    task_overrides = body.get("task_overrides") or None
    model = body.get("model")
    if model is not None:
        model = str(model).strip()
    result = await server._orchestrator.save_workflow(name, task_overrides=task_overrides, model=model)
    return web.json_response(result)


async def load_workflow(server: WebServer, request: web.Request) -> web.Response:
    """Load a saved workflow for preview."""
    if not server._orchestrator:
        return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)
    name = str(body.get("name", "")).strip()
    if not name:
        return web.json_response({"ok": False, "error": "Missing name"}, status=400)
    result = await server._orchestrator.load_workflow(name)
    return web.json_response(result, dumps=lambda obj: json.dumps(obj, default=str))


async def delete_workflow(server: WebServer, request: web.Request) -> web.Response:
    """Delete a saved workflow."""
    if not server._orchestrator:
        return web.json_response({"ok": False, "error": "No orchestrator"}, status=400)
    name = request.match_info.get("name", "").strip()
    if not name:
        return web.json_response({"ok": False, "error": "Missing name"}, status=400)
    result = await server._orchestrator.delete_workflow(name)
    return web.json_response(result)
