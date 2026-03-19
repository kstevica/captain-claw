"""REST handlers for the Datastore browser UI."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.datastore import ProtectedError, get_datastore_manager, get_session_datastore_manager
from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)

_JSON_DUMPS = lambda obj: json.dumps(obj, default=str)


def _resolve_manager(request: web.Request) -> Any:
    """Return the appropriate DatastoreManager for this request.

    In public computer mode, each session gets an isolated DB.
    """
    from captain_claw.web.public_auth import get_request_session_id
    is_public, session_id = get_request_session_id(request)
    if is_public and session_id:
        return get_session_datastore_manager(session_id)
    return get_datastore_manager()


# ── Tables ──────────────────────────────────────────────────────────


async def list_tables(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/datastore/tables — list all user tables."""
    mgr = _resolve_manager(request)
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
    mgr = _resolve_manager(request)
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

    mgr = _resolve_manager(request)
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
    mgr = _resolve_manager(request)
    try:
        await mgr.drop_table(name)
    except ProtectedError as exc:
        return web.json_response({"error": str(exc), "protected": True}, status=403)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"ok": True})


async def rename_table(server: WebServer, request: web.Request) -> web.Response:
    """PATCH /api/datastore/tables/{name} — rename a table."""
    name = request.match_info.get("name", "")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    new_name = str(body.get("new_name", "")).strip()
    if not new_name:
        return web.json_response({"error": "new_name is required"}, status=400)

    mgr = _resolve_manager(request)
    try:
        info = await mgr.rename_table(name, new_name)
    except ProtectedError as exc:
        return web.json_response({"error": str(exc), "protected": True}, status=403)
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
        dumps=_JSON_DUMPS,
    )


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

    # Build order_by in the format the query method expects:
    # "colname" for ASC, "-colname" for DESC
    if order_dir.upper() == "DESC":
        order_param = ["-" + order_by]
    else:
        order_param = [order_by]

    mgr = _resolve_manager(request)
    try:
        result = await mgr.query(
            table_name=name,
            columns=None,
            where=where,
            order_by=order_param,
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

    mgr = _resolve_manager(request)
    try:
        count = await mgr.insert_rows(name, rows)
    except ProtectedError as exc:
        return web.json_response({"error": str(exc), "protected": True}, status=403)
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

    mgr = _resolve_manager(request)
    try:
        count = await mgr.update_rows(name, set_values, where)
    except ProtectedError as exc:
        return web.json_response({"error": str(exc), "protected": True}, status=403)
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

    mgr = _resolve_manager(request)
    try:
        count = await mgr.delete_rows(name, where)
    except ProtectedError as exc:
        return web.json_response({"error": str(exc), "protected": True}, status=403)
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

    mgr = _resolve_manager(request)
    try:
        await mgr.add_column(name, col_name, col_type, default)
    except ProtectedError as exc:
        return web.json_response({"error": str(exc), "protected": True}, status=403)
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

    mgr = _resolve_manager(request)
    try:
        await mgr.drop_column(table_name, col_name)
    except ProtectedError as exc:
        return web.json_response({"error": str(exc), "protected": True}, status=403)
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

    mgr = _resolve_manager(request)
    try:
        result = await mgr.raw_select(sql_query)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response(result, dumps=_JSON_DUMPS)


# ── Protections ────────────────────────────────────────────────────


async def list_protections(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/datastore/tables/{name}/protections — list protections."""
    name = request.match_info.get("name", "")
    mgr = _resolve_manager(request)
    try:
        protections = await mgr.list_protections(name)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response(protections, dumps=_JSON_DUMPS)


async def add_protection(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/datastore/tables/{name}/protections — add a protection."""
    name = request.match_info.get("name", "")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    level = str(body.get("level", "")).strip()
    if not level:
        return web.json_response({"error": "level is required"}, status=400)

    row_id = body.get("row_id")
    col_name = body.get("col_name")
    reason = body.get("reason")

    mgr = _resolve_manager(request)
    try:
        result = await mgr.protect(
            name, level, row_id=row_id, col_name=col_name, reason=reason,
        )
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response(result, status=201, dumps=_JSON_DUMPS)


async def remove_protection(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/datastore/tables/{name}/protections — remove a protection."""
    name = request.match_info.get("name", "")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    level = str(body.get("level", "")).strip()
    if not level:
        return web.json_response({"error": "level is required"}, status=400)

    row_id = body.get("row_id")
    col_name = body.get("col_name")

    mgr = _resolve_manager(request)
    try:
        removed = await mgr.unprotect(name, level, row_id=row_id, col_name=col_name)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    if removed:
        return web.json_response({"ok": True})
    return web.json_response(
        {"error": "No matching protection found"}, status=404,
    )


# ── File Upload & Import ──────────────────────────────────────────


def _build_import_chat_message(result: dict[str, Any]) -> str:
    """Build a Markdown summary for the chat broadcast."""
    action = result.get("action", "imported")
    table = result.get("table", "unknown")
    rows = result.get("rows_imported", 0)
    columns = result.get("columns", [])
    warnings = result.get("warnings", [])

    if action == "appended":
        matched = result.get("matched_table", table)
        score = result.get("match_score", 0)
        msg = (
            f"**Imported {rows} rows** into existing table **{matched}** "
            f"(match score: {score:.0%})\n\n"
        )
    else:
        msg = f"**Created table** **{table}** with **{rows} rows**\n\n"

    msg += f"**Columns** ({len(columns)}): `{'`, `'.join(columns)}`"

    if warnings:
        msg += "\n\n**Warnings:**\n"
        for w in warnings:
            msg += f"- {w}\n"

    return msg


async def upload_and_import(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/datastore/upload — upload CSV/XLSX and import into datastore."""
    tmp_path: str | None = None
    try:
        reader = await request.multipart()
        if reader is None:
            return web.json_response({"error": "Multipart body required"}, status=400)

        file_field = None
        table_name_override: str | None = None
        force_new = False

        # Read multipart fields
        while True:
            field = await reader.next()
            if field is None:
                break
            if field.name == "file":
                file_field = field
                break
            elif field.name == "table_name":
                table_name_override = (await field.text()).strip() or None
            elif field.name == "force_new":
                force_new = (await field.text()).strip().lower() in ("true", "1", "yes")

        if file_field is None:
            return web.json_response({"error": "No file field in upload"}, status=400)

        original_name = file_field.filename or "upload"
        file_stem = Path(original_name).stem
        file_ext = Path(original_name).suffix.lower()

        if file_ext not in (".csv", ".xlsx"):
            return web.json_response(
                {"error": f"Unsupported file type '{file_ext}'. Only .csv and .xlsx are accepted."},
                status=400,
            )

        file_type = "csv" if file_ext == ".csv" else "xlsx"

        # Stream file to temp location
        suffix = file_ext
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as tmp_file:
                while True:
                    chunk = await file_field.read_chunk(8192)
                    if not chunk:
                        break
                    tmp_file.write(chunk)
        except Exception:
            os.close(fd)
            raise

        tmp_file_path = Path(tmp_path)
        log.info("File uploaded for import", filename=original_name, temp_path=tmp_path, file_type=file_type)

        mgr = _resolve_manager(request)

        # Parse headers
        try:
            headers = await mgr.parse_headers(tmp_file_path, file_type)
        except Exception as exc:
            return web.json_response({"error": f"Failed to parse file headers: {exc}"}, status=400)

        if not headers:
            return web.json_response({"error": "File contains no columns"}, status=400)

        result: dict[str, Any]
        use_name = table_name_override or file_stem

        # Try to find matching table
        match = None
        if not force_new:
            match = await mgr.find_matching_table(headers, use_name)

        if match:
            # Append to existing table
            log.info(
                "Matched existing table for import",
                table=match["name"],
                match_type=match["match_type"],
                score=match["score"],
            )
            import_result = await mgr.import_to_existing_table(
                tmp_file_path, file_type, match["name"],
            )
            result = {
                "action": "appended",
                "table": match["name"],
                "rows_imported": import_result["rows_imported"],
                "columns": import_result["columns"],
                "warnings": import_result.get("warnings", []),
                "matched_table": match["name"],
                "match_score": match["score"],
            }
        else:
            # Create new table
            log.info("No matching table found, creating new", table_name=use_name)
            if file_type == "csv":
                import_result = await mgr.import_csv(tmp_file_path, use_name)
            else:
                import_result = await mgr.import_xlsx(tmp_file_path, use_name)
            result = {
                "action": "created",
                "table": import_result.get("table", use_name),
                "rows_imported": import_result.get("rows_imported", 0),
                "columns": import_result.get("columns", []),
                "warnings": import_result.get("warnings", []),
            }

        # Broadcast import summary to chat
        chat_msg = _build_import_chat_message(result)
        server._broadcast({
            "type": "chat_message",
            "role": "assistant",
            "content": chat_msg,
        })

        log.info(
            "File import complete",
            action=result["action"],
            table=result["table"],
            rows=result["rows_imported"],
        )

        return web.json_response(result, dumps=_JSON_DUMPS)

    except web.HTTPException:
        raise
    except Exception as exc:
        log.error("File upload/import failed", error=str(exc), exc_info=True)
        return web.json_response({"error": str(exc)}, status=500)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ── Export ──────────────────────────────────────────────────────────


async def export_table(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/datastore/tables/{name}/export?format=csv|json|xlsx"""
    table_name = request.match_info["name"]
    fmt = request.query.get("format", "csv").lower()
    if fmt not in ("csv", "json", "xlsx"):
        return web.json_response(
            {"error": f"Unsupported format: {fmt}. Use csv, json, or xlsx."},
            status=400,
        )

    mgr = _resolve_manager(request)
    cfg_mod = __import__("captain_claw.config", fromlist=["get_config"])
    cfg = cfg_mod.get_config()

    try:
        result = await mgr.query(
            table_name, limit=cfg.datastore.max_export_rows,
        )
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)

    columns = result["columns"]
    rows = result["rows"]
    filename = f"{table_name}.{fmt}"

    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(columns)
        writer.writerows(rows)
        return web.Response(
            body=buf.getvalue().encode("utf-8"),
            content_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    if fmt == "json":
        data = [dict(zip(columns, row)) for row in rows]
        body = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        return web.Response(
            body=body.encode("utf-8"),
            content_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # xlsx — write to temp file, read bytes, delete
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        tmp.close()
        tmp_path = Path(tmp.name)
        mgr._write_xlsx(tmp_path, columns, rows)
        data_bytes = tmp_path.read_bytes()
        return web.Response(
            body=data_bytes,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    finally:
        if tmp and os.path.exists(tmp.name):
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
