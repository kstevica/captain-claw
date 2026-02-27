"""REST handlers for the Datastore browser UI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.datastore import get_datastore_manager

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

_JSON_DUMPS = lambda obj: json.dumps(obj, default=str)


# ── Tables ──────────────────────────────────────────────────────────


async def list_tables(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/datastore/tables — list all user tables."""
    mgr = get_datastore_manager()
    tables = await mgr.list_tables()
    return web.json_response(
        [
            {
                "name": t.name,
                "columns": [{"name": c.name, "type": c.col_type, "position": c.position} for c in t.columns],
                "row_count": t.row_count,
                "created_at": t.created_at,
                "updated_at": t.updated_at,
            }
            for t in tables
        ],
        dumps=_JSON_DUMPS,
    )


async def describe_table(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/datastore/tables/{name} — describe a single table."""
    name = request.match_info.get("name", "")
    mgr = get_datastore_manager()
    info = await mgr.describe_table(name)
    if not info:
        return web.json_response({"error": f"Table '{name}' not found"}, status=404)
    return web.json_response(
        {
            "name": info.name,
            "columns": [{"name": c.name, "type": c.col_type, "position": c.position} for c in info.columns],
            "row_count": info.row_count,
            "created_at": info.created_at,
            "updated_at": info.updated_at,
        },
        dumps=_JSON_DUMPS,
    )


async def create_table(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/datastore/tables — create a new table."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    name = str(body.get("name", "")).strip()
    columns = body.get("columns")
    if not name:
        return web.json_response({"error": "name is required"}, status=400)
    if not columns or not isinstance(columns, list):
        return web.json_response({"error": "columns array is required"}, status=400)

    mgr = get_datastore_manager()
    try:
        info = await mgr.create_table(name, columns)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response(
        {
            "name": info.name,
            "columns": [{"name": c.name, "type": c.col_type, "position": c.position} for c in info.columns],
            "row_count": info.row_count,
            "created_at": info.created_at,
            "updated_at": info.updated_at,
        },
        status=201,
        dumps=_JSON_DUMPS,
    )


async def drop_table(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/datastore/tables/{name} — drop a table."""
    name = request.match_info.get("name", "")
    mgr = get_datastore_manager()
    try:
        await mgr.drop_table(name)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"ok": True})


# ── Rows ────────────────────────────────────────────────────────────


async def query_rows(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/datastore/tables/{name}/rows — paginated row query."""
    name = request.match_info.get("name", "")
    limit = min(int(request.query.get("limit", "100")), 500)
    offset = int(request.query.get("offset", "0"))
    order_by = request.query.get("order_by", "_id")
    order_dir = request.query.get("order_dir", "ASC")

    # Optional where filter via query param (JSON string)
    where: dict[str, Any] | None = None
    where_raw = request.query.get("where")
    if where_raw:
        try:
            where = json.loads(where_raw)
        except Exception:
            return web.json_response({"error": "Invalid where JSON"}, status=400)

    mgr = get_datastore_manager()
    try:
        result = await mgr.query(
            table_name=name,
            columns=None,
            where=where,
            order_by=[f"{order_by} {order_dir}"],
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)

    # Convert rows from arrays to dicts for easier JS consumption
    col_names = result.get("columns", [])
    dict_rows = [
        {col_names[i]: val for i, val in enumerate(row)}
        for row in result.get("rows", [])
        if isinstance(row, (list, tuple))
    ]
    result["rows"] = dict_rows

    return web.json_response(result, dumps=_JSON_DUMPS)


async def insert_rows(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/datastore/tables/{name}/rows — insert rows."""
    name = request.match_info.get("name", "")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    rows = body.get("rows")
    if not rows or not isinstance(rows, list):
        return web.json_response({"error": "rows array is required"}, status=400)

    mgr = get_datastore_manager()
    try:
        count = await mgr.insert_rows(name, rows)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"inserted": count}, status=201, dumps=_JSON_DUMPS)


async def update_rows(server: WebServer, request: web.Request) -> web.Response:
    """PATCH /api/datastore/tables/{name}/rows — update rows."""
    name = request.match_info.get("name", "")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    set_values = body.get("set_values")
    where = body.get("where")
    if not set_values or not isinstance(set_values, dict):
        return web.json_response({"error": "set_values object is required"}, status=400)
    if not where or not isinstance(where, dict):
        return web.json_response({"error": "where object is required"}, status=400)

    mgr = get_datastore_manager()
    try:
        count = await mgr.update_rows(name, set_values, where)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"updated": count}, dumps=_JSON_DUMPS)


async def delete_rows(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/datastore/tables/{name}/rows — delete rows."""
    name = request.match_info.get("name", "")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    where = body.get("where")
    if not where or not isinstance(where, dict):
        return web.json_response({"error": "where object is required"}, status=400)

    mgr = get_datastore_manager()
    try:
        count = await mgr.delete_rows(name, where)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"deleted": count}, dumps=_JSON_DUMPS)


# ── Schema mutations ────────────────────────────────────────────────


async def add_column(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/datastore/tables/{name}/columns — add a column."""
    name = request.match_info.get("name", "")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    col_name = str(body.get("col_name", "")).strip()
    col_type = str(body.get("col_type", "text")).strip()
    default = body.get("default")
    if not col_name:
        return web.json_response({"error": "col_name is required"}, status=400)

    mgr = get_datastore_manager()
    try:
        await mgr.add_column(name, col_name, col_type, default)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    info = await mgr.describe_table(name)
    return web.json_response(
        {
            "name": info.name,
            "columns": [{"name": c.name, "type": c.col_type, "position": c.position} for c in info.columns],
            "row_count": info.row_count,
        } if info else {"ok": True},
        status=201,
        dumps=_JSON_DUMPS,
    )


async def drop_column(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/datastore/tables/{name}/columns/{col} — drop a column."""
    table_name = request.match_info.get("name", "")
    col_name = request.match_info.get("col", "")

    mgr = get_datastore_manager()
    try:
        await mgr.drop_column(table_name, col_name)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"ok": True})


# ── SQL ─────────────────────────────────────────────────────────────


async def run_sql(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/datastore/sql — execute raw SELECT query."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    sql_query = str(body.get("sql", "")).strip()
    if not sql_query:
        return web.json_response({"error": "sql is required"}, status=400)

    mgr = get_datastore_manager()
    try:
        result = await mgr.raw_select(sql_query)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response(result, dumps=_JSON_DUMPS)
