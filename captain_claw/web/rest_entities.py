"""REST handlers for entity CRUD (Todos, Contacts, Scripts, APIs)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

_JSON_DUMPS = lambda obj: json.dumps(obj, default=str)


# ── Todos ────────────────────────────────────────────────────────────


async def list_todos(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/todos — list todo items."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    sm = server.agent.session_manager
    status_filter = request.query.get("status")
    responsible_filter = request.query.get("responsible")
    session_filter = request.query.get("session_id")
    items = await sm.list_todos(
        limit=200,
        status_filter=status_filter or None,
        responsible_filter=responsible_filter or None,
        session_filter=session_filter or None,
    )
    return web.json_response(
        [item.to_dict() for item in items],
        dumps=_JSON_DUMPS,
    )


async def create_todo(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/todos — create a todo item."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    content = str(body.get("content", "")).strip()
    if not content:
        return web.json_response({"error": "content is required"}, status=400)
    sm = server.agent.session_manager
    item = await sm.create_todo(
        content=content,
        responsible=str(body.get("responsible", "human")).strip() or "human",
        priority=str(body.get("priority", "normal")).strip() or "normal",
        source_session=str(body.get("source_session", "")).strip() or None,
        target_session=str(body.get("target_session", "")).strip() or None,
        tags=str(body.get("tags", "")).strip() or None,
    )
    return web.json_response(item.to_dict(), status=201)


async def update_todo(server: WebServer, request: web.Request) -> web.Response:
    """PATCH /api/todos/{id} — update a todo item."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    todo_id = request.match_info.get("id", "")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    sm = server.agent.session_manager
    kwargs: dict[str, Any] = {}
    for field in ("content", "status", "responsible", "priority", "target_session", "tags"):
        if field in body:
            kwargs[field] = str(body[field]).strip() if body[field] is not None else None
    ok = await sm.update_todo(todo_id, **kwargs)
    if not ok:
        return web.json_response({"error": "Todo not found"}, status=404)
    item = await sm.load_todo(todo_id)
    return web.json_response(item.to_dict() if item else {"ok": True})


async def delete_todo(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/todos/{id} — delete a todo item."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    todo_id = request.match_info.get("id", "")
    sm = server.agent.session_manager
    ok = await sm.delete_todo(todo_id)
    if not ok:
        return web.json_response({"error": "Todo not found"}, status=404)
    return web.json_response({"ok": True})


# ── Contacts ─────────────────────────────────────────────────────────


async def list_contacts(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/contacts — list contacts."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    sm = server.agent.session_manager
    items = await sm.list_contacts(limit=200)
    return web.json_response(
        [c.to_dict() for c in items],
        dumps=_JSON_DUMPS,
    )


async def search_contacts(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/contacts/search?q= — search contacts."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    query = request.query.get("q", "").strip()
    if not query:
        return web.json_response({"error": "q parameter required"}, status=400)
    sm = server.agent.session_manager
    items = await sm.search_contacts(query, limit=50)
    return web.json_response(
        [c.to_dict() for c in items],
        dumps=_JSON_DUMPS,
    )


async def create_contact(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/contacts — create a contact."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    name = str(body.get("name", "")).strip()
    if not name:
        return web.json_response({"error": "name is required"}, status=400)
    sm = server.agent.session_manager
    item = await sm.create_contact(
        name=name,
        description=str(body.get("description", "")).strip() or None,
        position=str(body.get("position", "")).strip() or None,
        organization=str(body.get("organization", "")).strip() or None,
        relation=str(body.get("relation", "")).strip() or None,
        email=str(body.get("email", "")).strip() or None,
        phone=str(body.get("phone", "")).strip() or None,
        importance=int(body.get("importance", 1)),
        tags=str(body.get("tags", "")).strip() or None,
        notes=str(body.get("notes", "")).strip() or None,
        privacy_tier=str(body.get("privacy_tier", "normal")).strip() or "normal",
        source_session=str(body.get("source_session", "")).strip() or None,
    )
    return web.json_response(
        item.to_dict(), status=201,
        dumps=_JSON_DUMPS,
    )


async def get_contact(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/contacts/{id} — get a single contact."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    contact_id = request.match_info.get("id", "")
    sm = server.agent.session_manager
    item = await sm.load_contact(contact_id)
    if not item:
        return web.json_response({"error": "Contact not found"}, status=404)
    return web.json_response(
        item.to_dict(),
        dumps=_JSON_DUMPS,
    )


async def update_contact(server: WebServer, request: web.Request) -> web.Response:
    """PATCH /api/contacts/{id} — update a contact."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    contact_id = request.match_info.get("id", "")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    sm = server.agent.session_manager
    kwargs: dict[str, Any] = {}
    for field in ("name", "description", "position", "organization", "relation",
                   "email", "phone", "tags", "notes", "privacy_tier"):
        if field in body:
            kwargs[field] = str(body[field]).strip() if body[field] is not None else None
    if "importance" in body:
        kwargs["importance"] = max(1, min(10, int(body["importance"])))
        kwargs["importance_pinned"] = True
    ok = await sm.update_contact(contact_id, **kwargs)
    if not ok:
        return web.json_response({"error": "Contact not found"}, status=404)
    item = await sm.load_contact(contact_id)
    return web.json_response(
        item.to_dict() if item else {"ok": True},
        dumps=_JSON_DUMPS,
    )


async def delete_contact(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/contacts/{id} — delete a contact."""
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    contact_id = request.match_info.get("id", "")
    sm = server.agent.session_manager
    ok = await sm.delete_contact(contact_id)
    if not ok:
        return web.json_response({"error": "Contact not found"}, status=404)
    return web.json_response({"ok": True})


# ── Scripts ──────────────────────────────────────────────────────────


async def list_scripts(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    items = await server.agent.session_manager.list_scripts(limit=200)
    return web.json_response(
        [s.to_dict() for s in items],
        dumps=_JSON_DUMPS,
    )


async def search_scripts(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    query = request.query.get("q", "").strip()
    if not query:
        return web.json_response({"error": "q parameter required"}, status=400)
    items = await server.agent.session_manager.search_scripts(query, limit=50)
    return web.json_response(
        [s.to_dict() for s in items],
        dumps=_JSON_DUMPS,
    )


async def create_script(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    name = str(body.get("name", "")).strip()
    file_path = str(body.get("file_path", "")).strip()
    if not name or not file_path:
        return web.json_response({"error": "name and file_path are required"}, status=400)
    item = await server.agent.session_manager.create_script(
        name=name, file_path=file_path,
        description=str(body.get("description", "")).strip() or None,
        purpose=str(body.get("purpose", "")).strip() or None,
        language=str(body.get("language", "")).strip() or None,
        created_reason=str(body.get("created_reason", "")).strip() or None,
        tags=str(body.get("tags", "")).strip() or None,
        source_session=str(body.get("source_session", "")).strip() or None,
    )
    return web.json_response(
        item.to_dict(), status=201,
        dumps=_JSON_DUMPS,
    )


async def get_script(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    script_id = request.match_info.get("id", "")
    item = await server.agent.session_manager.load_script(script_id)
    if not item:
        return web.json_response({"error": "Script not found"}, status=404)
    return web.json_response(
        item.to_dict(), dumps=_JSON_DUMPS,
    )


async def update_script(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    script_id = request.match_info.get("id", "")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    kwargs: dict[str, Any] = {}
    for fld in ("name", "file_path", "description", "purpose", "language",
                 "created_reason", "tags"):
        if fld in body:
            kwargs[fld] = str(body[fld]).strip() if body[fld] is not None else None
    ok = await server.agent.session_manager.update_script(script_id, **kwargs)
    if not ok:
        return web.json_response({"error": "Script not found"}, status=404)
    item = await server.agent.session_manager.load_script(script_id)
    return web.json_response(
        item.to_dict() if item else {"ok": True},
        dumps=_JSON_DUMPS,
    )


async def delete_script(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    script_id = request.match_info.get("id", "")
    ok = await server.agent.session_manager.delete_script(script_id)
    if not ok:
        return web.json_response({"error": "Script not found"}, status=404)
    return web.json_response({"ok": True})


# ── APIs ─────────────────────────────────────────────────────────────


async def list_apis(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    items = await server.agent.session_manager.list_apis(limit=200)
    return web.json_response(
        [a.to_dict() for a in items],
        dumps=_JSON_DUMPS,
    )


async def search_apis(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    query = request.query.get("q", "").strip()
    if not query:
        return web.json_response({"error": "q parameter required"}, status=400)
    items = await server.agent.session_manager.search_apis(query, limit=50)
    return web.json_response(
        [a.to_dict() for a in items],
        dumps=_JSON_DUMPS,
    )


async def create_api(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    name = str(body.get("name", "")).strip()
    base_url = str(body.get("base_url", "")).strip()
    if not name or not base_url:
        return web.json_response({"error": "name and base_url are required"}, status=400)
    item = await server.agent.session_manager.create_api(
        name=name, base_url=base_url,
        endpoints=str(body.get("endpoints", "")).strip() or None,
        auth_type=str(body.get("auth_type", "")).strip() or None,
        credentials=str(body.get("credentials", "")).strip() or None,
        description=str(body.get("description", "")).strip() or None,
        purpose=str(body.get("purpose", "")).strip() or None,
        tags=str(body.get("tags", "")).strip() or None,
        source_session=str(body.get("source_session", "")).strip() or None,
    )
    return web.json_response(
        item.to_dict(), status=201,
        dumps=_JSON_DUMPS,
    )


async def get_api(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    api_id = request.match_info.get("id", "")
    item = await server.agent.session_manager.load_api(api_id)
    if not item:
        return web.json_response({"error": "API not found"}, status=404)
    return web.json_response(
        item.to_dict(), dumps=_JSON_DUMPS,
    )


async def update_api(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    api_id = request.match_info.get("id", "")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    kwargs: dict[str, Any] = {}
    for fld in ("name", "base_url", "endpoints", "auth_type", "credentials",
                 "description", "purpose", "tags"):
        if fld in body:
            kwargs[fld] = str(body[fld]).strip() if body[fld] is not None else None
    ok = await server.agent.session_manager.update_api(api_id, **kwargs)
    if not ok:
        return web.json_response({"error": "API not found"}, status=404)
    item = await server.agent.session_manager.load_api(api_id)
    return web.json_response(
        item.to_dict() if item else {"ok": True},
        dumps=_JSON_DUMPS,
    )


async def delete_api(server: WebServer, request: web.Request) -> web.Response:
    if not server.agent:
        return web.json_response({"error": "Agent not initialized"}, status=503)
    api_id = request.match_info.get("id", "")
    ok = await server.agent.session_manager.delete_api(api_id)
    if not ok:
        return web.json_response({"error": "API not found"}, status=404)
    return web.json_response({"ok": True})
