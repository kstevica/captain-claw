"""Slash command handling for the web UI (WebSocket commands)."""

from __future__ import annotations

import json
import shlex as _shlex
from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.config import get_config

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer


async def handle_command(server: WebServer, ws: web.WebSocketResponse, raw: str) -> None:
    """Handle slash commands."""
    if not server.agent:
        return

    parts = raw.strip().split(None, 1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    result = ""

    try:
        if cmd in ("/help", "/h"):
            result = format_help()

        elif cmd in ("/clear",):
            if server.agent.session:
                server.agent.session.messages.clear()
                await server.agent.session_manager.save_session(server.agent.session)
                server.agent.last_usage = server.agent._empty_usage()
                server.agent.last_context_window = {}
            result = "Session messages cleared."

        elif cmd in ("/config",):
            cfg = get_config()
            details = server.agent.get_runtime_model_details()
            result = (
                f"**Model:** {details.get('provider', '')}:{details.get('model', '')}\n"
                f"**Temperature:** {details.get('temperature', '')}\n"
                f"**Max tokens:** {details.get('max_tokens', '')}\n"
                f"**Session:** {server.agent.session.name if server.agent.session else 'none'}\n"
                f"**Pipeline:** {server.agent.pipeline_mode}\n"
                f"**Planning:** {'on' if server.agent.planning_enabled else 'off'}\n"
            )

        elif cmd in ("/stop", "/cancel"):
            if hasattr(server.agent, "cancel_event"):
                server.agent.cancel_event.set()
                result = "Stop signal sent. The agent will stop after the current step completes."
            else:
                result = "No active processing to stop."

        elif cmd in ("/history",):
            if server.agent.session:
                msgs = server.agent.session.messages[-20:]
                lines = []
                for m in msgs:
                    role = m.get("role", "?")
                    content = str(m.get("content", ""))[:120]
                    lines.append(f"**{role}**: {content}")
                result = "\n".join(lines) if lines else "No messages in session."
            else:
                result = "No active session."

        elif cmd in ("/compact",):
            if server.agent.session:
                await server.agent.compact_session(force=True, trigger="web_manual")
                result = "Session compacted."
            else:
                result = "No active session."

        elif cmd in ("/new",):
            name = args.strip() or None
            session = await server.agent.session_manager.create_session(
                name=name or "web-session"
            )
            server.agent.session = session
            server.agent.refresh_session_runtime_flags()
            await server.agent.session_manager.set_last_active_session(session.id)
            server.agent.last_usage = server.agent._empty_usage()
            server.agent.last_context_window = {}
            result = f"New session created: **{session.name}** (`{session.id[:8]}`)"
            server._broadcast({"type": "session_info", **server._session_info()})

        elif cmd in ("/session",):
            if not args.strip():
                info = server._session_info()
                result = (
                    f"**Session:** {info.get('name', '?')}\n"
                    f"**ID:** {info.get('id', '?')}\n"
                    f"**Model:** {info.get('provider', '')}:{info.get('model', '')}\n"
                    f"**Messages:** {info.get('message_count', 0)}\n"
                    f"**Description:** {info.get('description', 'none')}"
                )
            else:
                result = await handle_session_subcommand(server, args.strip())

        elif cmd in ("/sessions",):
            sessions = await server.agent.session_manager.list_sessions(limit=20)
            if sessions:
                lines = []
                for i, s in enumerate(sessions, 1):
                    active = " (active)" if (server.agent.session and s.id == server.agent.session.id) else ""
                    desc = (s.metadata or {}).get("description", "")
                    desc_str = f" - {desc}" if desc else ""
                    lines.append(f"{i}. **{s.name}**{active}{desc_str}")
                result = "\n".join(lines)
            else:
                result = "No sessions found."

        elif cmd in ("/models",):
            models = server.agent.get_allowed_models()
            current = server.agent.get_runtime_model_details()
            lines = []
            for m in models:
                active = " (active)" if m.get("model") == current.get("model") else ""
                lines.append(f"- **{m.get('id', '?')}**: {m.get('provider', '')}:{m.get('model', '')}{active}")
            result = "\n".join(lines) if lines else "No models configured."

        elif cmd in ("/pipeline",):
            if args.strip():
                mode = args.strip().lower()
                if mode in ("loop", "contracts"):
                    server.agent.pipeline_mode = mode
                    if mode == "contracts":
                        server.agent.planning_enabled = True
                    result = f"Pipeline mode set to **{mode}**."
                else:
                    result = "Invalid mode. Use `loop` or `contracts`."
            else:
                result = f"Pipeline mode: **{server.agent.pipeline_mode}**"

        elif cmd in ("/planning",):
            if args.strip().lower() == "on":
                server.agent.planning_enabled = True
                server.agent.pipeline_mode = "contracts"
                result = "Planning enabled (contracts mode)."
            elif args.strip().lower() == "off":
                server.agent.planning_enabled = False
                server.agent.pipeline_mode = "loop"
                result = "Planning disabled (loop mode)."
            else:
                result = f"Planning: **{'on' if server.agent.planning_enabled else 'off'}** (mode: {server.agent.pipeline_mode})"

        elif cmd in ("/skills",):
            skills = server.agent.discover_available_skills()
            if skills:
                lines = []
                for sk in skills:
                    name = getattr(sk, "name", "?")
                    desc = getattr(sk, "description", "")[:80]
                    lines.append(f"- **{name}**: {desc}")
                result = "\n".join(lines)
            else:
                result = "No skills available."

        elif cmd in ("/monitor",):
            result = "Monitor is always visible in the web UI. Use the monitor panel on the right."

        elif cmd in ("/exit", "/quit"):
            result = "Use Ctrl+C on the server terminal or close this browser tab."

        elif cmd in ("/approve",):
            from captain_claw.web.telegram import handle_approve_command
            result = await handle_approve_command(server, args.strip())

        elif cmd in ("/orchestrate",):
            if not args.strip():
                result = "Usage: `/orchestrate <request>`"
            else:
                from captain_claw.web.chat_handler import handle_chat
                await handle_chat(server, ws, raw)
                return

        elif cmd in ("/orchestrate-execute",):
            if not server._orchestrator:
                result = "Orchestrator not available."
            else:
                task_overrides = None
                variable_values = None
                if args.strip():
                    try:
                        parsed = json.loads(args.strip())
                        if isinstance(parsed, dict) and (
                            "variable_values" in parsed or "task_overrides" in parsed
                        ):
                            task_overrides = parsed.get("task_overrides")
                            variable_values = parsed.get("variable_values")
                        else:
                            task_overrides = parsed
                    except json.JSONDecodeError:
                        task_overrides = None
                try:
                    response = await server._orchestrator.execute(
                        task_overrides, variable_values=variable_values,
                    )
                    server._broadcast({
                        "type": "chat_message",
                        "role": "assistant",
                        "content": response,
                    })
                except Exception as e:
                    server._broadcast({
                        "type": "error",
                        "message": f"Orchestrator execute failed: {e}",
                    })
                return

        elif cmd in ("/todo",):
            result = await handle_todo_command(server, args.strip())

        elif cmd in ("/contacts",):
            result = await handle_contacts_command(server, args.strip())

        elif cmd in ("/scripts",):
            result = await handle_scripts_command(server, args.strip())

        elif cmd in ("/apis",):
            result = await handle_apis_command(server, args.strip())

        else:
            result = f"Unknown command: `{cmd}`. Type `/help` for available commands."

    except Exception as e:
        result = f"Command error: {str(e)}"

    await server._send(ws, {
        "type": "command_result",
        "command": raw,
        "content": result,
    })


async def handle_session_subcommand(server: WebServer, args: str) -> str:
    """Handle /session subcommands."""
    parts = args.split(None, 1)
    subcmd = parts[0].lower()
    subargs = parts[1].strip() if len(parts) > 1 else ""

    if subcmd in ("list",):
        sessions = await server.agent.session_manager.list_sessions(limit=20)
        lines = []
        for i, s in enumerate(sessions, 1):
            active = " (active)" if (server.agent.session and s.id == server.agent.session.id) else ""
            lines.append(f"{i}. **{s.name}**{active}")
        return "\n".join(lines) if lines else "No sessions."

    elif subcmd in ("switch", "load"):
        if not subargs:
            return "Usage: `/session switch <id|name|#N>`"
        session = await server.agent.session_manager.select_session(subargs)
        if session:
            server.agent.session = session
            await server.agent.session_manager.set_last_active_session(session.id)
            server.agent._sync_runtime_flags_from_session()
            server._broadcast({"type": "session_info", **server._session_info()})
            server._broadcast({"type": "session_switched"})
            return f"Switched to session **{session.name}**."
        return f"Session not found: `{subargs}`"

    elif subcmd in ("new",):
        name = subargs or "web-session"
        session = await server.agent.session_manager.create_session(name=name)
        server.agent.session = session
        server.agent.refresh_session_runtime_flags()
        await server.agent.session_manager.set_last_active_session(session.id)
        server.agent.last_usage = server.agent._empty_usage()
        server.agent.last_context_window = {}
        server._broadcast({"type": "session_info", **server._session_info()})
        server._broadcast({"type": "session_switched"})
        return f"Created and switched to session **{session.name}**."

    elif subcmd in ("rename",):
        if not subargs:
            return "Usage: `/session rename <new-name>`"
        if server.agent.session:
            server.agent.session.name = subargs
            await server.agent.session_manager.save_session(server.agent.session)
            server._broadcast({"type": "session_info", **server._session_info()})
            return f"Session renamed to **{subargs}**."
        return "No active session."

    elif subcmd in ("description",):
        if not subargs:
            return "Usage: `/session description <text>` or `/session description auto`"
        if server.agent.session:
            if subargs.lower() == "auto":
                desc = await server.agent._auto_generate_session_description()
                return f"Auto-generated description: {desc}"
            server.agent.session.metadata = server.agent.session.metadata or {}
            server.agent.session.metadata["description"] = subargs
            await server.agent.session_manager.save_session(server.agent.session)
            return f"Description set to: {subargs}"
        return "No active session."

    elif subcmd in ("model",):
        if not subargs:
            details = server.agent.get_runtime_model_details()
            return f"Active model: **{details.get('provider', '')}:{details.get('model', '')}**"
        await server.agent.set_session_model(subargs, persist=True)
        details = server.agent.get_runtime_model_details()
        server._broadcast({"type": "session_info", **server._session_info()})
        return f"Model set to **{details.get('provider', '')}:{details.get('model', '')}**"

    elif subcmd in ("protect",):
        if server.agent.session:
            if subargs.lower() == "on":
                server.agent.session.metadata = server.agent.session.metadata or {}
                server.agent.session.metadata["memory_protection"] = True
                await server.agent.session_manager.save_session(server.agent.session)
                return "Memory protection enabled."
            elif subargs.lower() == "off":
                server.agent.session.metadata = server.agent.session.metadata or {}
                server.agent.session.metadata["memory_protection"] = False
                await server.agent.session_manager.save_session(server.agent.session)
                return "Memory protection disabled."
            return "Usage: `/session protect on|off`"
        return "No active session."

    elif subcmd in ("export",):
        if not server.agent.session:
            return "No active session."
        mode = subargs.strip().lower() or "all"
        from captain_claw.session_export import export_session_history

        try:
            written = export_session_history(
                mode=mode,
                session_id=server.agent.session.id,
                session_name=server.agent.session.name,
                messages=server.agent.session.messages,
                saved_base_path=server.agent.tools.get_saved_base_path(create=True),
            )
        except Exception as e:
            return f"Export failed: {e}"
        if not written:
            return "No files exported."
        lines = [f"Exported **{len(written)}** file(s):"]
        for p in written:
            lines.append(f"- `{p}`")
        return "\n".join(lines)

    return f"Unknown session subcommand: `{subcmd}`"


# ── Entity command handlers ──────────────────────────────────────────


async def handle_todo_command(server: WebServer, args: str) -> str:
    """Handle /todo subcommands in the web UI."""
    sm = server.agent.session_manager
    if not args or args.lower() in ("list", "ls"):
        items = await sm.list_todos(limit=50)
        if not items:
            return "No to-do items."
        lines: list[str] = []
        for idx, item in enumerate(items, 1):
            tag_suffix = f" [{item.tags}]" if item.tags else ""
            status_icon = {" ": " ", "pending": " ", "in_progress": ">", "done": "x", "cancelled": "-"}.get(item.status, " ")
            lines.append(
                f"[{status_icon}] #{idx} [{item.priority}/{item.responsible}] "
                f"{item.content} ({item.status}){tag_suffix}  `{item.id[:8]}`"
            )
        return "**To-do items:**\n" + "\n".join(lines)

    parts = args.split(None, 1)
    subcmd = parts[0].lower()
    subargs = parts[1].strip() if len(parts) > 1 else ""

    if subcmd == "add":
        if not subargs:
            return "Usage: `/todo add <text>`"
        session_id = server.agent.session.id if server.agent.session else None
        item = await sm.create_todo(
            content=subargs, responsible="human", source_session=session_id,
        )
        return f"Added todo: **{subargs}** (`{item.id[:8]}`)"

    if subcmd in ("done", "complete", "finish"):
        if not subargs:
            return "Usage: `/todo done <id|#index>`"
        item = await sm.select_todo(subargs)
        if not item:
            return f"Todo not found: `{subargs}`"
        await sm.update_todo(item.id, status="done")
        return f"Marked done: **{item.content}**"

    if subcmd in ("remove", "rm", "delete", "del"):
        if not subargs:
            return "Usage: `/todo remove <id|#index>`"
        item = await sm.select_todo(subargs)
        if not item:
            return f"Todo not found: `{subargs}`"
        await sm.delete_todo(item.id)
        return f"Removed: **{item.content}**"

    if subcmd == "assign":
        assign_parts = subargs.split(None, 1)
        if len(assign_parts) < 2 or assign_parts[0] not in ("bot", "human"):
            return "Usage: `/todo assign bot|human <id|#index>`"
        responsible, selector = assign_parts
        item = await sm.select_todo(selector)
        if not item:
            return f"Todo not found: `{selector}`"
        await sm.update_todo(item.id, responsible=responsible)
        return f"Assigned **{item.content}** to {responsible}"

    # Fallback: treat entire args as an add
    session_id = server.agent.session.id if server.agent.session else None
    item = await sm.create_todo(
        content=args, responsible="human", source_session=session_id,
    )
    return f"Added todo: **{args}** (`{item.id[:8]}`)"


async def handle_contacts_command(server: WebServer, args: str) -> str:
    """Handle /contacts subcommands in the web UI."""
    sm = server.agent.session_manager
    if not args or args.lower() in ("list", "ls"):
        items = await sm.list_contacts(limit=50)
        if not items:
            return "No contacts."
        lines: list[str] = []
        for idx, c in enumerate(items, 1):
            org_part = f" @ {c.organization}" if c.organization else ""
            pos_part = f" ({c.position})" if c.position else ""
            lines.append(
                f"#{idx} [{c.importance}] {c.name}{pos_part}{org_part}"
                f" [{c.relation or '-'}]  `{c.id[:8]}`"
            )
        return "**Contacts:**\n" + "\n".join(lines)

    parts = args.split(None, 1)
    subcmd = parts[0].lower()
    subargs = parts[1].strip() if len(parts) > 1 else ""

    if subcmd == "add":
        if not subargs:
            return "Usage: `/contacts add <name>`"
        session_id = server.agent.session.id if server.agent.session else None
        item = await sm.create_contact(
            name=subargs, source_session=session_id,
        )
        return f"Added contact: **{subargs}** (`{item.id[:8]}`)"

    if subcmd == "info":
        if not subargs:
            return "Usage: `/contacts info <id|#index|name>`"
        item = await sm.select_contact(subargs)
        if not item:
            return f"Contact not found: `{subargs}`"
        parts_out = [f"**{item.name}**  `{item.id}`"]
        if item.position:
            parts_out.append(f"Position: {item.position}")
        if item.organization:
            parts_out.append(f"Organization: {item.organization}")
        if item.relation:
            parts_out.append(f"Relation: {item.relation}")
        if item.email:
            parts_out.append(f"Email: {item.email}")
        if item.phone:
            parts_out.append(f"Phone: {item.phone}")
        parts_out.append(f"Importance: {item.importance} (pinned={item.importance_pinned})")
        parts_out.append(f"Mentions: {item.mention_count}")
        if item.last_seen_at:
            parts_out.append(f"Last seen: {item.last_seen_at}")
        if item.tags:
            parts_out.append(f"Tags: {item.tags}")
        if item.description:
            parts_out.append(f"Description: {item.description}")
        if item.notes:
            parts_out.append(f"Notes: {item.notes}")
        parts_out.append(f"Privacy: {item.privacy_tier}")
        return "\n".join(parts_out)

    if subcmd == "search":
        if not subargs:
            return "Usage: `/contacts search <query>`"
        items = await sm.search_contacts(subargs, limit=20)
        if not items:
            return f"No contacts matching: `{subargs}`"
        lines = []
        for idx, c in enumerate(items, 1):
            org_part = f" @ {c.organization}" if c.organization else ""
            lines.append(
                f"#{idx} [{c.importance}] {c.name}{org_part}  `{c.id[:8]}`"
            )
        return "**Search results:**\n" + "\n".join(lines)

    if subcmd in ("remove", "rm", "delete", "del"):
        if not subargs:
            return "Usage: `/contacts remove <id|#index|name>`"
        item = await sm.select_contact(subargs)
        if not item:
            return f"Contact not found: `{subargs}`"
        await sm.delete_contact(item.id)
        return f"Removed: **{item.name}**"

    if subcmd == "importance":
        imp_parts = subargs.rsplit(None, 1)
        if len(imp_parts) < 2:
            return "Usage: `/contacts importance <id|#index|name> <1-10>`"
        selector, score_str = imp_parts
        try:
            score = max(1, min(10, int(score_str)))
        except ValueError:
            return "Importance must be a number 1-10."
        item = await sm.select_contact(selector)
        if not item:
            return f"Contact not found: `{selector}`"
        await sm.update_contact(item.id, importance=score, importance_pinned=True)
        return f"Set importance={score} (pinned) for **{item.name}**"

    if subcmd == "update":
        if not subargs:
            return "Usage: `/contacts update <id|#index|name> <field=value ...>`"
        update_parts = subargs.split(None, 1)
        if len(update_parts) < 2:
            return "Usage: `/contacts update <id|#index|name> <field=value ...>`"
        selector = update_parts[0]
        fields_str = update_parts[1]
        item = await sm.select_contact(selector)
        if not item:
            return f"Contact not found: `{selector}`"
        kwargs: dict[str, Any] = {}
        valid_fields = {"name", "description", "position", "organization", "relation",
                        "email", "phone", "tags", "notes", "privacy_tier"}
        for token in _shlex.split(fields_str):
            if "=" not in token:
                continue
            key, _, value = token.partition("=")
            key = key.strip().lower()
            if key in valid_fields:
                if key == "notes":
                    existing = item.notes or ""
                    kwargs["notes"] = (existing.rstrip() + "\n" + value) if existing else value
                else:
                    kwargs[key] = value
        if not kwargs:
            return "No valid fields to update. Use `field=value` syntax."
        ok = await sm.update_contact(item.id, **kwargs)
        return f"Updated contact: **{item.name}**" if ok else "Update failed."

    # Fallback: treat as search
    items = await sm.search_contacts(args, limit=20)
    if not items:
        return f"No contacts matching: `{args}`"
    lines = []
    for idx, c in enumerate(items, 1):
        org_part = f" @ {c.organization}" if c.organization else ""
        lines.append(
            f"#{idx} [{c.importance}] {c.name}{org_part}  `{c.id[:8]}`"
        )
    return "**Search results:**\n" + "\n".join(lines)


async def handle_scripts_command(server: WebServer, args: str) -> str:
    """Handle /scripts subcommands in the web UI."""
    sm = server.agent.session_manager
    if not args or args.lower() in ("list", "ls"):
        items = await sm.list_scripts(limit=50)
        if not items:
            return "No scripts."
        lines: list[str] = []
        for idx, s in enumerate(items, 1):
            lang_part = f" ({s.language})" if s.language else ""
            lines.append(
                f"#{idx} {s.name}{lang_part} [{s.file_path}]"
                f"  uses={s.use_count}  `{s.id[:8]}`"
            )
        return "**Scripts:**\n" + "\n".join(lines)

    parts = args.split(None, 1)
    subcmd = parts[0].lower()
    subargs = parts[1].strip() if len(parts) > 1 else ""

    if subcmd == "add":
        add_parts = subargs.split(None, 1)
        if len(add_parts) < 2:
            return "Usage: `/scripts add <name> <path>`"
        session_id = server.agent.session.id if server.agent.session else None
        item = await sm.create_script(
            name=add_parts[0], file_path=add_parts[1], source_session=session_id,
        )
        return f"Added script: **{add_parts[0]}** at {add_parts[1]} (`{item.id[:8]}`)"

    if subcmd == "info":
        if not subargs:
            return "Usage: `/scripts info <id|#index|name>`"
        item = await sm.select_script(subargs)
        if not item:
            return f"Script not found: `{subargs}`"
        parts_out = [f"**{item.name}**  `{item.id}`", f"Path: {item.file_path}"]
        if item.language:
            parts_out.append(f"Language: {item.language}")
        if item.description:
            parts_out.append(f"Description: {item.description}")
        if item.purpose:
            parts_out.append(f"Purpose: {item.purpose}")
        if item.created_reason:
            parts_out.append(f"Created reason: {item.created_reason}")
        if item.tags:
            parts_out.append(f"Tags: {item.tags}")
        parts_out.append(f"Uses: {item.use_count}")
        if item.last_used_at:
            parts_out.append(f"Last used: {item.last_used_at}")
        return "\n".join(parts_out)

    if subcmd == "search":
        if not subargs:
            return "Usage: `/scripts search <query>`"
        items = await sm.search_scripts(subargs, limit=20)
        if not items:
            return f"No scripts matching: `{subargs}`"
        lines = []
        for idx, s in enumerate(items, 1):
            lang_part = f" ({s.language})" if s.language else ""
            lines.append(f"#{idx} {s.name}{lang_part} [{s.file_path}]  `{s.id[:8]}`")
        return "**Search results:**\n" + "\n".join(lines)

    if subcmd in ("remove", "rm", "delete", "del"):
        if not subargs:
            return "Usage: `/scripts remove <id|#index|name>`"
        item = await sm.select_script(subargs)
        if not item:
            return f"Script not found: `{subargs}`"
        await sm.delete_script(item.id)
        return f"Removed: **{item.name}**"

    if subcmd == "update":
        if not subargs:
            return "Usage: `/scripts update <id|#index|name> <field=value ...>`"
        update_parts = subargs.split(None, 1)
        if len(update_parts) < 2:
            return "Usage: `/scripts update <id|#index|name> <field=value ...>`"
        selector = update_parts[0]
        fields_str = update_parts[1]
        item = await sm.select_script(selector)
        if not item:
            return f"Script not found: `{selector}`"
        kwargs: dict[str, Any] = {}
        valid_fields = {"name", "file_path", "description", "purpose", "language",
                        "created_reason", "tags"}
        for token in _shlex.split(fields_str):
            if "=" not in token:
                continue
            key, _, value = token.partition("=")
            key = key.strip().lower()
            if key in valid_fields:
                kwargs[key] = value
        if not kwargs:
            return "No valid fields to update. Use `field=value` syntax."
        ok = await sm.update_script(item.id, **kwargs)
        return f"Updated script: **{item.name}**" if ok else "Update failed."

    # Fallback: search
    items = await sm.search_scripts(args, limit=20)
    if not items:
        return f"No scripts matching: `{args}`"
    lines = []
    for idx, s in enumerate(items, 1):
        lang_part = f" ({s.language})" if s.language else ""
        lines.append(f"#{idx} {s.name}{lang_part} [{s.file_path}]  `{s.id[:8]}`")
    return "**Search results:**\n" + "\n".join(lines)


async def handle_apis_command(server: WebServer, args: str) -> str:
    """Handle /apis subcommands in the web UI."""
    sm = server.agent.session_manager
    if not args or args.lower() in ("list", "ls"):
        items = await sm.list_apis(limit=50)
        if not items:
            return "No APIs."
        lines: list[str] = []
        for idx, a in enumerate(items, 1):
            auth_part = f" [{a.auth_type}]" if a.auth_type else ""
            lines.append(
                f"#{idx} {a.name}{auth_part} ({a.base_url})"
                f"  uses={a.use_count}  `{a.id[:8]}`"
            )
        return "**APIs:**\n" + "\n".join(lines)

    parts = args.split(None, 1)
    subcmd = parts[0].lower()
    subargs = parts[1].strip() if len(parts) > 1 else ""

    if subcmd == "add":
        add_parts = subargs.split(None, 1)
        if len(add_parts) < 2:
            return "Usage: `/apis add <name> <base_url>`"
        session_id = server.agent.session.id if server.agent.session else None
        item = await sm.create_api(
            name=add_parts[0], base_url=add_parts[1], source_session=session_id,
        )
        return f"Added API: **{add_parts[0]}** ({add_parts[1]}) (`{item.id[:8]}`)"

    if subcmd == "info":
        if not subargs:
            return "Usage: `/apis info <id|#index|name>`"
        item = await sm.select_api(subargs)
        if not item:
            return f"API not found: `{subargs}`"
        parts_out = [f"**{item.name}**  `{item.id}`", f"Base URL: {item.base_url}"]
        if item.auth_type:
            parts_out.append(f"Auth type: {item.auth_type}")
        if item.credentials:
            parts_out.append(f"Credentials: {item.credentials}")
        if item.endpoints:
            parts_out.append(f"Endpoints: {item.endpoints}")
        if item.description:
            parts_out.append(f"Description: {item.description}")
        if item.purpose:
            parts_out.append(f"Purpose: {item.purpose}")
        if item.tags:
            parts_out.append(f"Tags: {item.tags}")
        parts_out.append(f"Uses: {item.use_count}")
        if item.last_used_at:
            parts_out.append(f"Last used: {item.last_used_at}")
        return "\n".join(parts_out)

    if subcmd == "search":
        if not subargs:
            return "Usage: `/apis search <query>`"
        items = await sm.search_apis(subargs, limit=20)
        if not items:
            return f"No APIs matching: `{subargs}`"
        lines = []
        for idx, a in enumerate(items, 1):
            auth_part = f" [{a.auth_type}]" if a.auth_type else ""
            lines.append(f"#{idx} {a.name}{auth_part} ({a.base_url})  `{a.id[:8]}`")
        return "**Search results:**\n" + "\n".join(lines)

    if subcmd in ("remove", "rm", "delete", "del"):
        if not subargs:
            return "Usage: `/apis remove <id|#index|name>`"
        item = await sm.select_api(subargs)
        if not item:
            return f"API not found: `{subargs}`"
        await sm.delete_api(item.id)
        return f"Removed: **{item.name}**"

    if subcmd == "update":
        if not subargs:
            return "Usage: `/apis update <id|#index|name> <field=value ...>`"
        update_parts = subargs.split(None, 1)
        if len(update_parts) < 2:
            return "Usage: `/apis update <id|#index|name> <field=value ...>`"
        selector = update_parts[0]
        fields_str = update_parts[1]
        item = await sm.select_api(selector)
        if not item:
            return f"API not found: `{selector}`"
        kwargs: dict[str, Any] = {}
        valid_fields = {"name", "base_url", "endpoints", "auth_type", "credentials",
                        "description", "purpose", "tags"}
        for token in _shlex.split(fields_str):
            if "=" not in token:
                continue
            key, _, value = token.partition("=")
            key = key.strip().lower()
            if key in valid_fields:
                kwargs[key] = value
        if not kwargs:
            return "No valid fields to update. Use `field=value` syntax."
        ok = await sm.update_api(item.id, **kwargs)
        return f"Updated API: **{item.name}**" if ok else "Update failed."

    # Fallback: search
    items = await sm.search_apis(args, limit=20)
    if not items:
        return f"No APIs matching: `{args}`"
    lines = []
    for idx, a in enumerate(items, 1):
        auth_part = f" [{a.auth_type}]" if a.auth_type else ""
        lines.append(f"#{idx} {a.name}{auth_part} ({a.base_url})  `{a.id[:8]}`")
    return "**Search results:**\n" + "\n".join(lines)


def format_help() -> str:
    """Format help text for the /help command."""
    from captain_claw.web_server import COMMANDS

    categories: dict[str, list[dict[str, str]]] = {}
    for cmd in COMMANDS:
        cat = cmd["category"]
        categories.setdefault(cat, []).append(cmd)
    lines = ["## Captain Claw Commands\n"]
    for cat, cmds in categories.items():
        lines.append(f"### {cat}")
        for c in cmds:
            lines.append(f"- `{c['command']}` - {c['description']}")
        lines.append("")
    lines.append("**Tip:** Type `/` to see command suggestions. Press `Ctrl+K` for the command palette.")
    return "\n".join(lines)
