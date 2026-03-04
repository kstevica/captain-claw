"""REST handlers for Direct API Calls CRUD + execution."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from aiohttp import web

from captain_claw.session import get_session_manager
from captain_claw.tools.browser_api_replay import ApiReplayEngine

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

_JSON_DUMPS = lambda obj: json.dumps(obj, default=str)

_ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH"}


async def list_calls(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/direct-api-calls — list all registered API calls."""
    sm = get_session_manager()
    app_name = request.query.get("app_name") or None
    items = await sm.list_direct_api_calls(limit=200, app_name=app_name)
    return web.json_response(
        [_call_dict(c) for c in items],
        dumps=_JSON_DUMPS,
    )


async def get_call(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/direct-api-calls/{id} — get one API call by ID."""
    sm = get_session_manager()
    cid = request.match_info.get("id", "")
    item = await sm.load_direct_api_call(cid)
    if not item:
        return web.json_response({"error": "API call not found"}, status=404)
    return web.json_response(_call_dict(item), dumps=_JSON_DUMPS)


async def create_call(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/direct-api-calls — create a new API call."""
    sm = get_session_manager()
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    name = str(body.get("name", "")).strip()
    if not name:
        return web.json_response({"error": "name is required"}, status=400)

    url = str(body.get("url", "")).strip()
    if not url:
        return web.json_response({"error": "url is required"}, status=400)

    method = str(body.get("method", "GET")).strip().upper()
    if method == "DELETE":
        return web.json_response(
            {"error": "DELETE method is not allowed for safety"},
            status=400,
        )
    if method not in _ALLOWED_METHODS:
        return web.json_response(
            {"error": f"method must be one of: {', '.join(sorted(_ALLOWED_METHODS))}"},
            status=400,
        )

    item = await sm.create_direct_api_call(
        name=name,
        url=url,
        method=method,
        description=str(body.get("description", "")).strip(),
        input_payload=str(body.get("input_payload", "")).strip(),
        result_payload=str(body.get("result_payload", "")).strip(),
        headers=str(body.get("headers", "")).strip() or None,
        auth_type=str(body.get("auth_type", "")).strip() or None,
        auth_token=str(body.get("auth_token", "")).strip() or None,
        auth_source=str(body.get("auth_source", "")).strip() or None,
        app_name=str(body.get("app_name", "")).strip() or None,
        tags=str(body.get("tags", "")).strip() or None,
    )
    return web.json_response(_call_dict(item), status=201, dumps=_JSON_DUMPS)


async def update_call(server: WebServer, request: web.Request) -> web.Response:
    """PATCH /api/direct-api-calls/{id} — update an API call."""
    sm = get_session_manager()
    cid = request.match_info.get("id", "")
    item = await sm.load_direct_api_call(cid)
    if not item:
        return web.json_response({"error": "API call not found"}, status=404)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    # Reject DELETE method
    if "method" in body:
        m = str(body["method"]).strip().upper()
        if m == "DELETE":
            return web.json_response(
                {"error": "DELETE method is not allowed for safety"},
                status=400,
            )
        if m not in _ALLOWED_METHODS:
            return web.json_response(
                {"error": f"method must be one of: {', '.join(sorted(_ALLOWED_METHODS))}"},
                status=400,
            )

    fields: dict[str, Any] = {}
    for key in (
        "name", "url", "method", "description",
        "input_payload", "result_payload",
        "headers", "auth_type", "auth_token", "auth_source",
        "app_name", "tags",
    ):
        if key in body:
            val = body[key]
            if key == "method" and val is not None:
                fields[key] = str(val).strip().upper()
            elif val is not None:
                fields[key] = str(val).strip()
            else:
                fields[key] = None

    if not fields:
        return web.json_response({"error": "No fields to update"}, status=400)

    ok = await sm.update_direct_api_call(item.id, **fields)
    if not ok:
        return web.json_response({"error": "Update failed"}, status=500)

    updated = await sm.load_direct_api_call(item.id)
    return web.json_response(
        _call_dict(updated) if updated else {"ok": True},
        dumps=_JSON_DUMPS,
    )


async def delete_call(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/direct-api-calls/{id} — remove an API call."""
    sm = get_session_manager()
    cid = request.match_info.get("id", "")
    item = await sm.load_direct_api_call(cid)
    if not item:
        return web.json_response({"error": "API call not found"}, status=404)
    ok = await sm.delete_direct_api_call(item.id)
    if not ok:
        return web.json_response({"error": "Delete failed"}, status=500)
    return web.Response(status=204)


async def execute_call(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/direct-api-calls/{id}/execute — execute an API call."""
    sm = get_session_manager()
    cid = request.match_info.get("id", "")
    item = await sm.load_direct_api_call(cid)
    if not item:
        return web.json_response({"error": "API call not found"}, status=404)

    # Reject DELETE method at execution time too
    if item.method.upper() == "DELETE":
        return web.json_response(
            {"error": "DELETE method is not allowed for safety"},
            status=400,
        )

    # Parse optional request body
    try:
        body = await request.json()
    except Exception:
        body = {}

    payload = body.get("payload")
    query_params = body.get("query_params")

    # Split full URL into base_url + endpoint
    parsed = urlparse(item.url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    endpoint = parsed.path
    if parsed.query:
        endpoint += f"?{parsed.query}"

    # Resolve auth headers
    auth_headers = ApiReplayEngine.resolve_auth_headers(
        item.auth_type, item.auth_token
    )

    # Parse extra headers from entry
    extra_headers: dict[str, str] = {}
    if item.headers:
        try:
            extra_headers = json.loads(item.headers)
        except (json.JSONDecodeError, TypeError):
            pass

    # Merge headers: extra first, auth overrides
    merged_headers = {**extra_headers, **auth_headers}

    # Parse query params
    qp: dict[str, str] | None = None
    if query_params:
        if isinstance(query_params, str):
            try:
                qp = json.loads(query_params)
            except (json.JSONDecodeError, TypeError):
                qp = None
        elif isinstance(query_params, dict):
            qp = query_params

    # Parse JSON payload
    body_json: Any = None
    if payload:
        if isinstance(payload, str):
            try:
                body_json = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                body_json = None
        elif isinstance(payload, dict):
            body_json = payload

    # Execute
    result = await ApiReplayEngine.replay(
        base_url=base_url,
        endpoint=endpoint,
        method=item.method.upper(),
        headers=merged_headers if merged_headers else None,
        query_params=qp,
        body_json=body_json,
    )

    # Record usage
    preview = result.response_body[:500] if result.response_body else ""
    await sm.record_direct_api_call_usage(
        item.id,
        status_code=result.status_code,
        response_preview=preview,
    )

    return web.json_response(
        {
            "success": result.success,
            "status_code": result.status_code,
            "url": result.url,
            "method": result.method,
            "elapsed_ms": round(result.elapsed_ms, 1),
            "response_body": result.response_body,
            "response_headers": result.response_headers,
            "error": result.error,
        },
        dumps=_JSON_DUMPS,
    )


# ── Helpers ──────────────────────────────────────────────────────────


def _call_dict(c: Any) -> dict[str, Any]:
    """Convert a DirectApiCallEntry to a JSON-friendly dict."""
    return {
        "id": c.id,
        "name": c.name,
        "url": c.url,
        "method": c.method,
        "description": c.description,
        "input_payload": c.input_payload,
        "result_payload": c.result_payload,
        "headers": c.headers,
        "auth_type": c.auth_type,
        "auth_token": c.auth_token,
        "auth_source": c.auth_source,
        "app_name": c.app_name,
        "tags": c.tags,
        "use_count": c.use_count,
        "last_used_at": c.last_used_at,
        "last_status_code": c.last_status_code,
        "last_response_preview": c.last_response_preview,
        "created_at": c.created_at,
        "updated_at": c.updated_at,
    }
