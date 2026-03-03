"""Datastore tool for LLM-managed relational data tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from captain_claw.datastore import ProtectedError, get_datastore_manager
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


def _parse_json_str(value: Any | None, label: str) -> Any:
    """Parse a JSON string parameter, returning the decoded object.

    If the LLM already sent a native dict/list (not a JSON string),
    return it directly.
    """
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if not value:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Invalid JSON for '{label}': {e}") from e


def _parse_columns(value: Any | None) -> list[str] | None:
    """Parse a columns parameter that may be a list, JSON string, or CSV."""
    if value is None:
        return None
    if isinstance(value, list):
        return [str(c).strip() for c in value if str(c).strip()]
    value = str(value).strip()
    if not value:
        return None
    if value.startswith("["):
        parsed = json.loads(value)
        return [str(c).strip() for c in parsed if str(c).strip()]
    return [c.strip() for c in value.split(",") if c.strip()]


def _format_table(columns: list[str], rows: list[list[Any]], total: int | None = None) -> str:
    """Format query results as a compact markdown table."""
    if not rows:
        return "No rows returned."

    # Compute column widths
    widths = [len(str(c)) for c in columns]
    str_rows = []
    for row in rows:
        str_row = [str(v) if v is not None else "NULL" for v in row]
        for i, v in enumerate(str_row):
            if i < len(widths):
                widths[i] = max(widths[i], len(v))
        str_rows.append(str_row)

    lines: list[str] = []
    # Header
    header = " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(columns))
    lines.append(f"| {header} |")
    sep = " | ".join("-" * widths[i] for i in range(len(columns)))
    lines.append(f"| {sep} |")
    # Data
    for str_row in str_rows:
        row_str = " | ".join(
            str_row[i].ljust(widths[i]) if i < len(str_row) else " " * widths[i]
            for i in range(len(columns))
        )
        lines.append(f"| {row_str} |")

    result = "\n".join(lines)
    if total is not None:
        result += f"\n\n({len(rows)} of {total} total rows)"
    return result


class DatastoreTool(Tool):
    """Manage user data tables with SQL-like operations."""

    name = "datastore"
    description = (
        "Manage persistent relational data tables in a local database. "
        "Create tables, insert/update/delete rows, query with filters, "
        "run raw SELECT queries, import/export CSV or XLSX files, and "
        "manage data protection rules. "
        "Actions: list_tables, describe, create_table, drop_table, rename_table, "
        "add_column, rename_column, drop_column, change_column_type, "
        "insert, update, update_column, delete, query, sql, import_file, export, "
        "protect, unprotect, list_protections."
    )
    timeout_seconds = 60.0

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_tables", "describe", "create_table", "drop_table", "rename_table",
                    "add_column", "rename_column", "drop_column", "change_column_type",
                    "insert", "update", "update_column", "delete",
                    "query", "sql", "import_file", "export",
                    "protect", "unprotect", "list_protections",
                ],
                "description": "Operation to perform.",
            },
            "table": {
                "type": "string",
                "description": "Target table name.",
            },
            "columns": {
                "type": "string",
                "description": (
                    'For create_table: JSON array of {"name": "col", "type": "text"}. '
                    "For query/export: comma-separated column names."
                ),
            },
            "column": {
                "type": "string",
                "description": "Column name (for add/rename/drop/change_column_type/update_column).",
            },
            "new_name": {
                "type": "string",
                "description": "New name (for rename_column or rename_table).",
            },
            "col_type": {
                "type": "string",
                "description": "Column type: text, integer, real, boolean, date, datetime, json.",
            },
            "default_value": {
                "type": "string",
                "description": "Default value for new column (for add_column).",
            },
            "rows": {
                "type": "string",
                "description": 'JSON array of row objects: [{"name": "Alice", "age": 30}].',
            },
            "set_values": {
                "type": "string",
                "description": 'JSON object of column=value pairs: {"status": "done"}.',
            },
            "value": {
                "type": "string",
                "description": "Value for update_column.",
            },
            "expression": {
                "type": "string",
                "description": "SQL expression for update_column (e.g. 'price * 1.1').",
            },
            "where": {
                "type": "string",
                "description": (
                    'JSON filter: {"age": {"op": ">", "value": 25}, "status": "active"}. '
                    "Simple equality: {\"name\": \"Alice\"}."
                ),
            },
            "order_by": {
                "type": "string",
                "description": "Comma-separated columns for ordering (prefix with - for DESC).",
            },
            "limit": {
                "type": "integer",
                "description": "Max rows to return.",
            },
            "offset": {
                "type": "integer",
                "description": "Rows to skip.",
            },
            "sql_query": {
                "type": "string",
                "description": "Raw SELECT SQL query (for 'sql' action only).",
            },
            "file_path": {
                "type": "string",
                "description": "Path to CSV/XLSX file (for import_file).",
            },
            "sheet": {
                "type": "string",
                "description": "Sheet name for XLSX import (default: first sheet).",
            },
            "append": {
                "type": "boolean",
                "description": "Append to existing table (for import_file).",
            },
            "format": {
                "type": "string",
                "enum": ["csv", "json", "xlsx"],
                "description": "Export format (default: csv).",
            },
            "level": {
                "type": "string",
                "enum": ["table", "column", "row", "cell"],
                "description": "Protection level (for protect/unprotect).",
            },
            "row_id": {
                "type": "integer",
                "description": "Row ID (for row/cell protection).",
            },
            "reason": {
                "type": "string",
                "description": "Reason for protection (optional, for protect).",
            },
        },
        "required": ["action"],
    }

    async def execute(self, action: str, **kwargs: Any) -> ToolResult:
        dm = get_datastore_manager()

        # Log invocation
        _log_args = {k: v for k, v in kwargs.items() if not k.startswith("_") and v is not None}
        log.info("Datastore tool call", action=action, **_log_args)

        try:
            if action == "list_tables":
                result = await self._list_tables(dm)
            elif action == "describe":
                result = await self._describe(dm, kwargs.get("table"))
            elif action == "create_table":
                result = await self._create_table(dm, kwargs.get("table"), kwargs.get("columns"))
            elif action == "drop_table":
                result = await self._drop_table(dm, kwargs.get("table"))
            elif action == "rename_table":
                result = await self._rename_table(dm, kwargs)
            elif action == "add_column":
                result = await self._add_column(dm, kwargs)
            elif action == "rename_column":
                result = await self._rename_column(dm, kwargs)
            elif action == "drop_column":
                result = await self._drop_column(dm, kwargs)
            elif action == "change_column_type":
                result = await self._change_column_type(dm, kwargs)
            elif action == "insert":
                result = await self._insert(dm, kwargs)
            elif action == "update":
                result = await self._update(dm, kwargs)
            elif action == "update_column":
                result = await self._update_column(dm, kwargs)
            elif action == "delete":
                result = await self._delete(dm, kwargs)
            elif action == "query":
                result = await self._query(dm, kwargs)
            elif action == "sql":
                result = await self._sql(dm, kwargs)
            elif action == "import_file":
                result = await self._import_file(dm, kwargs)
            elif action == "export":
                result = await self._export(dm, kwargs)
            elif action == "protect":
                result = await self._protect(dm, kwargs)
            elif action == "unprotect":
                result = await self._unprotect(dm, kwargs)
            elif action == "list_protections":
                result = await self._list_protections(dm, kwargs)
            else:
                result = ToolResult(success=False, error=f"Unknown action: {action}")
        except ProtectedError as e:
            log.warning("Datastore BLOCKED by protection", action=action, error=str(e))
            result = ToolResult(
                success=False,
                error=f"BLOCKED: {e}. The operation was NOT performed. Inform the user that the data is protected.",
            )
        except Exception as e:
            log.error("Datastore tool error", action=action, error=str(e))
            result = ToolResult(success=False, error=str(e))

        # Log result
        if result.success:
            # Truncate long content for log readability
            _content_preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
            log.info("Datastore tool result", action=action, success=True, content=_content_preview)
        else:
            log.warning("Datastore tool result", action=action, success=False, error=result.error)

        return result

    # ── action handlers ──────────────────────────────────────────────

    @staticmethod
    async def _list_tables(dm: Any) -> ToolResult:
        tables = await dm.list_tables()
        if not tables:
            return ToolResult(success=True, content="No tables in the datastore.")
        lines: list[str] = []
        for t in tables:
            cols = ", ".join(f"{c.name} ({c.col_type})" for c in t.columns)
            lines.append(f"- **{t.name}** ({t.row_count} rows): {cols}")
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    async def _describe(dm: Any, table: str | None) -> ToolResult:
        if not table:
            return ToolResult(success=False, error="'table' is required for describe.")
        info = await dm.describe_table(table)
        lines = [f"Table: **{info.name}** ({info.row_count} rows)"]
        lines.append(f"Created: {info.created_at}")
        lines.append(f"Updated: {info.updated_at}")
        lines.append("\nColumns:")
        for c in info.columns:
            lines.append(f"  - {c.name} ({c.col_type})")
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    async def _create_table(dm: Any, table: str | None, columns_raw: str | None) -> ToolResult:
        if not table:
            return ToolResult(success=False, error="'table' is required for create_table.")
        if not columns_raw:
            return ToolResult(success=False, error="'columns' is required for create_table.")
        columns = _parse_json_str(columns_raw, "columns")
        if not isinstance(columns, list):
            return ToolResult(success=False, error="'columns' must be a JSON array.")
        info = await dm.create_table(table, columns)
        col_names = ", ".join(c.name for c in info.columns)
        return ToolResult(
            success=True,
            content=f"Created table **{info.name}** with columns: {col_names}",
        )

    @staticmethod
    async def _drop_table(dm: Any, table: str | None) -> ToolResult:
        if not table:
            return ToolResult(success=False, error="'table' is required for drop_table.")
        await dm.drop_table(table)
        return ToolResult(success=True, content=f"Dropped table **{table}**.")

    @staticmethod
    async def _rename_table(dm: Any, kwargs: dict[str, Any]) -> ToolResult:
        table = kwargs.get("table")
        new_name = kwargs.get("new_name")
        if not table:
            return ToolResult(success=False, error="'table' is required.")
        if not new_name:
            return ToolResult(success=False, error="'new_name' is required.")
        info = await dm.rename_table(table, new_name)
        return ToolResult(
            success=True,
            content=f"Renamed table **{table}** to **{info.name}**.",
        )

    @staticmethod
    async def _add_column(dm: Any, kwargs: dict[str, Any]) -> ToolResult:
        table = kwargs.get("table")
        column = kwargs.get("column")
        col_type = kwargs.get("col_type", "text")
        default = kwargs.get("default_value")
        if not table:
            return ToolResult(success=False, error="'table' is required.")
        if not column:
            return ToolResult(success=False, error="'column' is required.")
        await dm.add_column(table, column, col_type, default)
        return ToolResult(
            success=True, content=f"Added column **{column}** ({col_type}) to **{table}**."
        )

    @staticmethod
    async def _rename_column(dm: Any, kwargs: dict[str, Any]) -> ToolResult:
        table = kwargs.get("table")
        column = kwargs.get("column")
        new_name = kwargs.get("new_name")
        if not table or not column or not new_name:
            return ToolResult(success=False, error="'table', 'column', and 'new_name' are required.")
        await dm.rename_column(table, column, new_name)
        return ToolResult(
            success=True, content=f"Renamed column **{column}** to **{new_name}** in **{table}**."
        )

    @staticmethod
    async def _drop_column(dm: Any, kwargs: dict[str, Any]) -> ToolResult:
        table = kwargs.get("table")
        column = kwargs.get("column")
        if not table or not column:
            return ToolResult(success=False, error="'table' and 'column' are required.")
        await dm.drop_column(table, column)
        return ToolResult(success=True, content=f"Dropped column **{column}** from **{table}**.")

    @staticmethod
    async def _change_column_type(dm: Any, kwargs: dict[str, Any]) -> ToolResult:
        table = kwargs.get("table")
        column = kwargs.get("column")
        col_type = kwargs.get("col_type")
        if not table or not column or not col_type:
            return ToolResult(success=False, error="'table', 'column', and 'col_type' are required.")
        await dm.change_column_type(table, column, col_type)
        return ToolResult(
            success=True,
            content=f"Changed **{column}** in **{table}** to type **{col_type}**.",
        )

    @staticmethod
    async def _insert(dm: Any, kwargs: dict[str, Any]) -> ToolResult:
        table = kwargs.get("table")
        rows_raw = kwargs.get("rows")
        if not table:
            return ToolResult(success=False, error="'table' is required.")
        if not rows_raw:
            return ToolResult(success=False, error="'rows' is required (JSON array).")
        rows = _parse_json_str(rows_raw, "rows")
        if not isinstance(rows, list):
            return ToolResult(success=False, error="'rows' must be a JSON array of objects.")
        count = await dm.insert_rows(table, rows)
        return ToolResult(success=True, content=f"Inserted {count} row(s) into **{table}**.")

    @staticmethod
    async def _update(dm: Any, kwargs: dict[str, Any]) -> ToolResult:
        table = kwargs.get("table")
        set_raw = kwargs.get("set_values")
        where_raw = kwargs.get("where")
        if not table:
            return ToolResult(success=False, error="'table' is required.")
        if not set_raw:
            return ToolResult(success=False, error="'set_values' is required (JSON object).")
        set_values = _parse_json_str(set_raw, "set_values")
        where = _parse_json_str(where_raw, "where") if where_raw else None
        count = await dm.update_rows(table, set_values, where)
        return ToolResult(success=True, content=f"Updated {count} row(s) in **{table}**.")

    @staticmethod
    async def _update_column(dm: Any, kwargs: dict[str, Any]) -> ToolResult:
        table = kwargs.get("table")
        column = kwargs.get("column")
        value = kwargs.get("value")
        expression = kwargs.get("expression")
        if not table or not column:
            return ToolResult(success=False, error="'table' and 'column' are required.")
        if value is None and not expression:
            return ToolResult(success=False, error="'value' or 'expression' is required.")
        count = await dm.update_column(table, column, value=value, expression=expression)
        return ToolResult(
            success=True, content=f"Updated column **{column}** in {count} row(s) of **{table}**."
        )

    @staticmethod
    async def _delete(dm: Any, kwargs: dict[str, Any]) -> ToolResult:
        table = kwargs.get("table")
        where_raw = kwargs.get("where")
        if not table:
            return ToolResult(success=False, error="'table' is required.")
        where = _parse_json_str(where_raw, "where") if where_raw else None
        count = await dm.delete_rows(table, where)
        return ToolResult(success=True, content=f"Deleted {count} row(s) from **{table}**.")

    @staticmethod
    async def _query(dm: Any, kwargs: dict[str, Any]) -> ToolResult:
        table = kwargs.get("table")
        if not table:
            return ToolResult(success=False, error="'table' is required.")

        columns = _parse_columns(kwargs.get("columns"))

        where_raw = kwargs.get("where")
        where = _parse_json_str(where_raw, "where") if where_raw else None

        order_raw = kwargs.get("order_by")
        order_by: list[str] | None = None
        if order_raw:
            if isinstance(order_raw, list):
                order_by = [str(o).strip() for o in order_raw if str(o).strip()]
            else:
                order_by = [o.strip() for o in str(order_raw).split(",") if o.strip()]

        limit = kwargs.get("limit")
        if isinstance(limit, str):
            limit = int(limit)
        offset = kwargs.get("offset", 0)
        if isinstance(offset, str):
            offset = int(offset)

        result = await dm.query(table, columns, where, order_by, limit, offset)
        return ToolResult(
            success=True,
            content=_format_table(result["columns"], result["rows"], result["total"]),
        )

    @staticmethod
    async def _sql(dm: Any, kwargs: dict[str, Any]) -> ToolResult:
        sql_query = kwargs.get("sql_query")
        if not sql_query:
            return ToolResult(success=False, error="'sql_query' is required.")
        result = await dm.raw_select(sql_query)
        return ToolResult(
            success=True,
            content=_format_table(result["columns"], result["rows"], result.get("total")),
        )

    @staticmethod
    async def _import_file(dm: Any, kwargs: dict[str, Any]) -> ToolResult:
        file_path_str = kwargs.get("file_path")
        if not file_path_str:
            return ToolResult(success=False, error="'file_path' is required.")

        # Resolve path relative to runtime base
        base = kwargs.get("_runtime_base_path")
        fp = Path(file_path_str)
        if not fp.is_absolute() and base:
            fp = Path(base) / fp
        fp = fp.resolve()

        table_name = kwargs.get("table")
        append = kwargs.get("append", False)
        if isinstance(append, str):
            append = append.lower() in ("true", "1", "yes")

        ext = fp.suffix.lower()
        if ext == ".csv":
            result = await dm.import_csv(fp, table_name, append)
        elif ext in (".xlsx", ".xls"):
            sheet = kwargs.get("sheet")
            result = await dm.import_xlsx(fp, table_name, sheet, append)
        else:
            return ToolResult(success=False, error=f"Unsupported file type: {ext}. Use .csv or .xlsx.")

        return ToolResult(
            success=True,
            content=(
                f"Imported {result['rows_imported']} rows into **{result['table']}**. "
                f"Columns: {', '.join(result['columns'])}"
            ),
        )

    @staticmethod
    async def _export(dm: Any, kwargs: dict[str, Any]) -> ToolResult:
        table = kwargs.get("table")
        if not table:
            return ToolResult(success=False, error="'table' is required.")

        fmt = kwargs.get("format", "csv").lower()
        if fmt not in ("csv", "json", "xlsx"):
            return ToolResult(success=False, error=f"Unsupported format: {fmt}")

        columns = _parse_columns(kwargs.get("columns"))

        where_raw = kwargs.get("where")
        where = _parse_json_str(where_raw, "where") if where_raw else None

        # Build output path in saved/ directory.
        # _saved_base_path already points to <runtime>/saved so we must NOT
        # add another "saved" segment; _runtime_base_path needs it appended.
        saved_base = kwargs.get("_saved_base_path")
        runtime_base = kwargs.get("_runtime_base_path")
        session_id = kwargs.get("_session_id", "default")
        if saved_base:
            output_dir = Path(saved_base) / "output" / str(session_id)
        elif runtime_base:
            output_dir = Path(runtime_base) / "saved" / "output" / str(session_id)
        else:
            output_dir = Path(".") / "saved" / "output" / str(session_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{table}.{fmt}"

        if fmt == "csv":
            path = await dm.export_csv(table, output_path, columns, where)
        elif fmt == "json":
            path = await dm.export_json(table, output_path, columns, where)
        else:
            path = await dm.export_xlsx(table, output_path, columns, where)

        return ToolResult(
            success=True,
            content=f"Exported **{table}** to {path}",
        )

    # ── protection handlers ──────────────────────────────────────────

    @staticmethod
    async def _protect(dm: Any, kwargs: dict[str, Any]) -> ToolResult:
        table = kwargs.get("table")
        level = kwargs.get("level")
        if not table:
            return ToolResult(success=False, error="'table' is required.")
        if not level:
            return ToolResult(success=False, error="'level' is required (table, column, row, cell).")

        row_id = kwargs.get("row_id")
        if isinstance(row_id, str):
            row_id = int(row_id)
        col_name = kwargs.get("column")
        reason = kwargs.get("reason")

        result = await dm.protect(
            table, level, row_id=row_id, col_name=col_name, reason=reason,
        )
        parts = [f"Protected **{table}** at level **{level}**"]
        if result.get("row_id") is not None:
            parts.append(f"row_id={result['row_id']}")
        if result.get("col_name"):
            parts.append(f"column={result['col_name']}")
        if reason:
            parts.append(f"reason: {reason}")
        return ToolResult(success=True, content=", ".join(parts))

    @staticmethod
    async def _unprotect(dm: Any, kwargs: dict[str, Any]) -> ToolResult:
        table = kwargs.get("table")
        level = kwargs.get("level")
        if not table:
            return ToolResult(success=False, error="'table' is required.")
        if not level:
            return ToolResult(success=False, error="'level' is required (table, column, row, cell).")

        row_id = kwargs.get("row_id")
        if isinstance(row_id, str):
            row_id = int(row_id)
        col_name = kwargs.get("column")

        removed = await dm.unprotect(table, level, row_id=row_id, col_name=col_name)
        if removed:
            return ToolResult(success=True, content=f"Removed {level}-level protection from **{table}**.")
        return ToolResult(
            success=False,
            error=f"No matching {level}-level protection found on **{table}**.",
        )

    @staticmethod
    async def _list_protections(dm: Any, kwargs: dict[str, Any]) -> ToolResult:
        table = kwargs.get("table")
        protections = await dm.list_protections(table)
        if not protections:
            scope = f" for **{table}**" if table else ""
            return ToolResult(success=True, content=f"No protections{scope}.")

        lines: list[str] = []
        for p in protections:
            parts = [f"- **{p['table_name']}** [{p['level']}]"]
            if p.get("row_id") is not None:
                parts.append(f"row_id={p['row_id']}")
            if p.get("col_name"):
                parts.append(f"column={p['col_name']}")
            if p.get("reason"):
                parts.append(f"({p['reason']})")
            lines.append(" ".join(parts))
        return ToolResult(success=True, content="\n".join(lines))
