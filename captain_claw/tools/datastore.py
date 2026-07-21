"""Datastore tool for LLM-managed relational data tables."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from captain_claw.config import get_config
from captain_claw.datastore import (
    ProtectedError,
    resolve_datastore_manager,
)
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


def _resolve_datastore_manager(session_id: str | None) -> Any:
    """The datastore this agent's tool reads/writes — shared VFS store when a run
    injects ``CLAW_DATASTORE_VFS``, else session/global. Delegates to the single
    shared resolver so the completion-gate verifier checks the SAME database."""
    return resolve_datastore_manager(session_id)


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


def _parse_where(value: Any | None) -> tuple[Any, str | None]:
    """Parse a `where` filter, rescuing repeated keys.

    A model asking for a range writes the only thing that looks right:

        {"id": {"op": ">=", "value": 370}, "id": {"op": "<=", "value": 379}}

    JSON cannot hold two "id" keys. Standard parsing keeps the LAST one, so
    the lower bound vanishes without a sound and the filter silently becomes
    `id <= 379` — one observed call returned 379 rows of prose instead of 10
    and swallowed 1.38M characters of the agent's context.

    Repeated keys are collected into a list, which `_build_where` ANDs
    together — exactly what was meant.
    """
    if value is None or value == "":
        return None, None
    if isinstance(value, (dict, list)):
        return value, None

    repeated: list[str] = []

    def _hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in pairs:
            if k in out:
                repeated.append(k)
                prev = out[k]
                out[k] = [*prev, v] if isinstance(prev, list) else [prev, v]
            else:
                out[k] = v
        return out

    try:
        parsed = json.loads(value, object_pairs_hook=_hook)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Invalid JSON for 'where': {e}") from e
    if not repeated:
        return parsed, None
    cols = ", ".join(f"`{c}`" for c in dict.fromkeys(repeated))
    return parsed, (f"(`where` repeated {cols} — a JSON object can only hold a key once, so "
                    "the earlier condition would have been dropped. Combined them with AND. "
                    'For a range, send one key with a list: {"id": [{"op": ">=", "value": 370}, '
                    '{"op": "<=", "value": 379}]}.)')


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


# ── argument aliases ─────────────────────────────────────────────────
#
# Models reach for the name the rest of the world uses — `data` for a
# payload, `table_name` for a table, `select` for columns. One agent's log
# over 190 datastore calls: data×3, table_name×2, select×2, order×3, sql×1,
# column_type×1, column_name×1.
#
# Half of those raise a confusing error ("'set_values' is required" when the
# model plainly sent the values, under `data`). The other half are worse:
# `select` and `order` are simply dropped, so the call SUCCEEDS and quietly
# returns every column in arbitrary order. Nothing tells the model its
# filter went missing.
#
# So: accept the synonym, rename it to the canonical parameter, and say so
# in the result. Never override a canonical value the model did send.
_ALIASES: dict[str, tuple[str, ...]] = {
    "table": ("table_name", "tableName", "table_id"),
    "where": ("filter", "filters", "condition", "conditions", "criteria"),
    "sql_query": ("sql", "statement", "query_sql"),
    "columns": ("select", "cols", "column_names"),
    "order_by": ("order", "sort", "sort_by", "orderBy"),
    "col_type": ("column_type", "coltype", "type"),
    "column": ("column_name", "col"),
    "file_path": ("path", "file", "filename"),
    "new_name": ("rename_to", "to"),
    "set_values": ("values", "set", "fields", "updates", "changes"),
    "rows": ("records", "items", "row"),
}

# `data` means different things per action, so it can't live in the table above.
_DATA_TARGET_BY_ACTION: dict[str, str] = {
    "update": "set_values", "update_column": "set_values",
    "insert": "rows", "upsert": "rows", "import_file": "rows",
}


def _normalize_arg_aliases(action: str, kwargs: dict[str, Any]) -> str | None:
    """Rename known synonyms onto canonical parameters. Returns a note if any moved."""
    renamed: list[str] = []

    def _move(src: str, dst: str) -> None:
        if kwargs.get(dst) not in (None, ""):
            return                      # the model sent the real one — leave it
        val = kwargs.get(src)
        if val in (None, ""):
            return
        kwargs[dst] = val
        kwargs.pop(src, None)
        renamed.append(f"`{src}` → `{dst}`")

    for canonical, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in kwargs:
                _move(alias, canonical)
    if "data" in kwargs:
        _move("data", _DATA_TARGET_BY_ACTION.get(action, "rows"))

    # Shape slips that cost a round-trip: one row sent bare, or a set of values
    # wrapped in a pointless list.
    rows = kwargs.get("rows")
    if isinstance(rows, dict):
        kwargs["rows"] = [rows]
        renamed.append("wrapped a single `rows` object in an array")
    sv = kwargs.get("set_values")
    if isinstance(sv, list) and len(sv) == 1 and isinstance(sv[0], dict):
        kwargs["set_values"] = sv[0]
        renamed.append("unwrapped `set_values` from a single-item array")

    if not renamed:
        return None
    return ("(Adjusted arguments: " + "; ".join(renamed)
            + ". Use the documented parameter names next time.)")


# ── `table` pre-flight ───────────────────────────────────────────────
#
# "Error: 'table' is required." is the most-seen datastore failure. It shows
# up mid-loop: the model upserts into the same table twenty times, then one
# call comes back without `table` — usually because a large `rows` payload
# crowded the argument out. The old error was a dead end. It named no tables,
# showed no way back, and discarded the payload the model had just built, so
# a weak model's next move was to re-send the same broken call.
#
# Two answers, split by blast radius:
#   - Where a wrong guess is harmless or self-evident (reads, and appends that
#     land in a table the model was already writing to), infer the table and
#     say so in the result, so the model sees the correction.
#   - Where a wrong guess would mangle data (update/delete/drop/rename/column
#     surgery), never infer — but fail with the table list and a filled-in
#     example so the retry succeeds on the next turn instead of the fifth.

# Operate on an existing table; a wrong guess costs a re-read or an append.
_INFERABLE_TABLE_ACTIONS = frozenset({
    "describe", "query", "export", "insert", "upsert",
})
# Need no table at all.
_NO_TABLE_ACTIONS = frozenset({"list_tables", "sql"})
# `table` is optional — derived from the filename when omitted.
_OPTIONAL_TABLE_ACTIONS = frozenset({"import_file"})

# Last table each (datastore, session) actually touched. This is what makes
# the mid-loop recovery precise: the model isn't guessing at "the only table",
# it's continuing the table it named on the previous call.
_LAST_TABLE: dict[str, str] = {}
_LAST_TABLE_MAX = 512


def _scope_key(dm: Any, session_id: str | None) -> str:
    return f"{getattr(dm, 'db_path', '?')}::{session_id or 'default'}"


def _remember_table(dm: Any, session_id: str | None, table: str) -> None:
    if len(_LAST_TABLE) >= _LAST_TABLE_MAX:
        _LAST_TABLE.clear()
    _LAST_TABLE[_scope_key(dm, session_id)] = table


def _missing_table_error(action: str, tables: list[Any], kwargs: dict[str, Any]) -> str:
    """The error a model can actually act on: what exists, and the exact retry."""
    if action == "create_table":
        return ("'table' is required for create_table — it names the table to create, "
                'e.g. {"action": "create_table", "table": "fund_portfolio", '
                '"columns": [{"name": "id", "type": "integer"}]}.')
    if not tables:
        return (f"'table' is required for {action}, but this datastore has no tables yet. "
                "Create one first with action=create_table.")
    listing = ", ".join(f"{t.name} ({t.row_count} rows)" for t in tables[:20])
    example = {"action": action, "table": tables[0].name}
    for k in ("rows", "set_values", "where", "column", "new_name"):
        if kwargs.get(k) is not None:
            example[k] = "…"
    return (f"'table' is required for {action}. Tables in this datastore: {listing}. "
            f"Re-send the SAME call with `table` filled in, e.g. "
            f"{json.dumps(example, ensure_ascii=False)} "
            "— the arguments you already built are still valid.")


# ── SQL writes → the structured action that does the same thing ──────
#
# `action="sql"` is SELECT-only, and stays that way: protection rules
# (_ds_protections) are enforced in the structured write paths, so a raw
# UPDATE would drive straight through them. But "Only SELECT queries are
# allowed" tells a model nothing about where to go instead, and it burns a
# turn — sometimes several — rediscovering the `update` action.
#
# So when a write arrives here, translate it and hand back the structured
# call, filled in. We refuse and explain in the same breath.

_SQL_VERB_TO_ACTION: dict[str, str] = {
    "UPDATE": "update", "INSERT": "insert", "REPLACE": "upsert",
    "DELETE": "delete", "CREATE": "create_table", "DROP": "drop_table",
    "ALTER": "add_column",
}


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split on `sep`, ignoring separators inside quotes or parentheses."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    quote: str | None = None
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _sql_literal(token: str) -> Any:
    """Best-effort SQL literal → Python value. Unrecognized tokens stay strings."""
    t = token.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
        return t[1:-1].replace("''", "'")
    if t.upper() == "NULL":
        return None
    if t.upper() in ("TRUE", "FALSE"):
        return t.upper() == "TRUE"
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t


def _sql_where_to_filter(clause: str) -> dict[str, Any] | None:
    """`id = 5 AND status = 'x'` → {"id": 5, "status": "x"}. None if not that simple."""
    out: dict[str, Any] = {}
    for cond in re.split(r"\bAND\b", clause, flags=re.IGNORECASE):
        m = re.match(r"^\s*[\"'`]?(\w+)[\"'`]?\s*=\s*(.+?)\s*$", cond, re.DOTALL)
        if not m:
            return None
        out[m.group(1)] = _sql_literal(m.group(2))
    return out or None


def _sql_write_to_call(sql: str) -> dict[str, Any] | None:
    """Translate a simple single-table write into the equivalent tool call."""
    s = sql.strip().rstrip(";")
    m = re.match(r"^UPDATE\s+[\"'`]?(\w+)[\"'`]?\s+SET\s+(.+?)(?:\s+WHERE\s+(.+))?$",
                 s, re.IGNORECASE | re.DOTALL)
    if m:
        table, sets, where = m.group(1), m.group(2), m.group(3)
        values: dict[str, Any] = {}
        for assign in _split_top_level(sets):
            a = re.match(r"^[\"'`]?(\w+)[\"'`]?\s*=\s*(.+)$", assign, re.DOTALL)
            if not a:
                return None
            values[a.group(1)] = _sql_literal(a.group(2))
        call = {"action": "update", "table": table, "set_values": values}
        if where:
            filt = _sql_where_to_filter(where)
            if filt is None:
                return None
            call["where"] = filt
        return call

    m = re.match(r"^INSERT\s+INTO\s+[\"'`]?(\w+)[\"'`]?\s*\((.+?)\)\s*VALUES\s*\((.+)\)$",
                 s, re.IGNORECASE | re.DOTALL)
    if m:
        cols = [c.strip().strip('"\'`') for c in _split_top_level(m.group(2))]
        vals = [_sql_literal(v) for v in _split_top_level(m.group(3))]
        if len(cols) != len(vals):
            return None
        return {"action": "insert", "table": m.group(1), "rows": [dict(zip(cols, vals))]}

    m = re.match(r"^DELETE\s+FROM\s+[\"'`]?(\w+)[\"'`]?(?:\s+WHERE\s+(.+))?$",
                 s, re.IGNORECASE | re.DOTALL)
    if m:
        call = {"action": "delete", "table": m.group(1)}
        if m.group(2):
            filt = _sql_where_to_filter(m.group(2))
            if filt is None:
                return None
            call["where"] = filt
        return call
    return None


def _sql_write_error(sql: str) -> str:
    """Refuse a write through raw SQL, and say exactly what to send instead."""
    verb = (re.match(r"^\s*(\w+)", sql or "") or [None, ""])[1].upper()
    head = ("`action=\"sql\"` runs SELECT only — writes go through the structured "
            "actions so protection rules still apply.")
    call = _sql_write_to_call(sql)
    if call:
        return (f"{head} This is a {verb}; re-send it as: "
                f"{json.dumps(call, ensure_ascii=False)}")
    action = _SQL_VERB_TO_ACTION.get(verb)
    if action:
        return (f"{head} Use action=\"{action}\" instead of a {verb} statement "
                "(action=\"upsert\" for insert-or-update on a unique key).")
    return f"{head} Send a SELECT, or use one of the structured actions."


def _missing_payload_error(action: str, param: str, kwargs: dict[str, Any], example: str) -> str:
    """A missing payload error that shows what DID arrive — usually the tell."""
    got = sorted(k for k, v in kwargs.items()
                 if not k.startswith("_") and v not in (None, "") and k != "action")
    got_str = ", ".join(f"`{k}`" for k in got) if got else "nothing but `action`"
    return (f"'{param}' is required for {action} — it holds the data. "
            f"This call carried {got_str}. Re-send as: {example}")


def _result_char_budget() -> int:
    try:
        return int(get_config().datastore.max_result_chars)
    except Exception:
        return 20_000


def _format_table(columns: list[str], rows: list[list[Any]], total: int | None = None) -> str:
    """Format query results as a compact markdown table, inside a size budget.

    A read of a wide table can be enormous — rows of prose, padded to column
    width. One 379-row read cost 1.38M characters and took 71% of a 200k
    context in a single message. Rows are dropped once the budget is spent,
    and the result SAYS so along with how to narrow the query: silent
    truncation would read as "that's all of it".
    """
    if not rows:
        return "No rows returned."

    budget = _result_char_budget()
    kept = rows
    if budget > 0:
        used = sum(len(str(c)) + 3 for c in columns) * 2
        for i, row in enumerate(rows):
            used += sum(len(str(v)) + 3 for v in row)
            if used > budget:
                kept = rows[:i]
                break
    dropped = len(rows) - len(kept)
    if dropped and not kept:
        kept = rows[:1]     # always show something to act on
        dropped = len(rows) - 1
    rows = kept

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
    if dropped:
        # ONE concrete next step, not a menu. A list of options reads as "your
        # query was wrong" and sends the model rephrasing the same question in
        # new ways; `offset` is simply where it left off.
        shown = len(rows)
        result += (
            f"\n\n⚠ Shown: the first {shown} matching row(s). {dropped} more matched but "
            f"did not fit the {budget:,}-character display cap — the query was CORRECT and "
            "the data is there. To read the rest, re-run this exact call with "
            f'`offset` = {shown} (and add `columns` if you only need some fields).'
        )
    return result


class DatastoreTool(Tool):
    """Manage user data tables with SQL-like operations."""

    name = "datastore"
    description = (
        "Manage persistent relational data tables in a local database. "
        "Create tables, insert/upsert/update/delete rows, query with filters, "
        "run raw SELECT queries, import/export CSV or XLSX files, and "
        "manage data protection rules. "
        "Use `upsert` (with a table created with a `unique` key) for idempotent "
        "writes — re-running the same item updates its row instead of duplicating it, "
        "which is what makes resumable/continued runs safe. "
        "Actions: list_tables, describe, create_table, drop_table, rename_table, "
        "add_column, rename_column, drop_column, change_column_type, "
        "insert, upsert, update, update_column, delete, query, sql, import_file, export, "
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
                    "insert", "upsert", "update", "update_column", "delete",
                    "query", "sql", "import_file", "export",
                    "protect", "unprotect", "list_protections",
                ],
                "description": "Operation to perform.",
            },
            "table": {
                "type": "string",
                "description": (
                    "Target table name. REQUIRED for every action except list_tables "
                    "and sql (and export, when sql_query is given). Send it on every "
                    "call — do not assume the previous call's table carries over."
                ),
            },
            "project": {
                "type": "string",
                "description": (
                    "READ-ONLY: query ANOTHER run's datastore by its VFS folder name "
                    "(a reference / prior-knowledge folder, e.g. \"vatra-676f1e31\"). "
                    "Only list_tables/describe/query/sql/export accept it. Omit to "
                    "read/write THIS run's own shared datastore."
                ),
            },
            "columns": {
                "type": "string",
                "description": (
                    'For create_table: JSON array of {"name": "col", "type": "text"}. '
                    "For query/export: comma-separated column names."
                ),
            },
            "unique": {
                "type": "string",
                "description": (
                    "For create_table: the column(s) forming a UNIQUE key — the conflict "
                    "target for `upsert`. JSON array or comma-separated names "
                    '(e.g. ["item_id"] or "item_id"). Give resumable data a stable key here.'
                ),
            },
            "key": {
                "type": "string",
                "description": (
                    "For upsert: the unique column(s) to match on. JSON array or "
                    "comma-separated. Optional if the table was created with a `unique` key."
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
                    'Simple equality: {"name": "Alice"}. RANGE on one column — any of '
                    'these work: {"id": {"op": "BETWEEN", "value": [370, 379]}}, '
                    '{"id": {"op": [">=", "<="], "value": [370, 379]}}, or '
                    '{"id": [{"op": ">=", "value": 370}, {"op": "<=", "value": 379}]}. '
                    'Never two "id" keys in one object (JSON keeps only the last, so the '
                    'other bound is silently lost). A list of plain values means IN: '
                    '{"id": [1, 2, 3]}.'
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
                "description": "Raw SELECT SQL query (for 'sql' and 'export' actions). SELECT only — to change data use action=insert/upsert/update/delete, never an UPDATE/INSERT/DELETE statement here. For export with sql_query, the query result is exported directly — useful for JOINs across tables.",
            },
            "file_path": {
                "type": "string",
                "description": "Path to CSV/XLSX file. For import_file: source file. For export: desired output path (relative to saved/ directory). If omitted on export, an auto-generated path is used.",
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
        session_id = str(kwargs.get("_session_id", "") or "").strip() or None
        # `project` targets ANOTHER run's datastore, READ-ONLY (reference / prior-
        # knowledge folders). Omitted → this run's own shared store.
        project = str(kwargs.get("project", "") or "").strip()
        if project:
            _READ_ACTIONS = {"list_tables", "describe", "query", "sql", "export"}
            if action not in _READ_ACTIONS:
                return ToolResult(success=False, error=(
                    f"'{action}' cannot target another folder — `project` is READ-ONLY "
                    "(list_tables, describe, query, sql, export). Omit `project` to write "
                    "to this run's own datastore."))
            from captain_claw.datastore import get_vfs_datastore_manager
            dm = get_vfs_datastore_manager(project, create=False)
            if dm is None:
                return ToolResult(success=True, content=f"Folder '{project}' has no datastore.")
        else:
            dm = _resolve_datastore_manager(session_id)

        # Log invocation
        _log_args = {k: v for k, v in kwargs.items() if not k.startswith("_") and v is not None}
        log.info("Datastore tool call", action=action, **_log_args)

        # Fold synonyms onto the real parameter names (`data` → `set_values`,
        # `select` → `columns`, …) before anything reads them.
        alias_note = _normalize_arg_aliases(action, kwargs)
        if alias_note:
            log.info("Datastore normalized argument aliases", action=action, note=alias_note)

        # Parse `where` centrally so a repeated key is rescued rather than
        # silently dropped by json.loads (see _parse_where).
        try:
            _where, where_note = _parse_where(kwargs.get("where"))
        except ValueError as e:
            return ToolResult(success=False, error=str(e))
        if _where is not None:
            kwargs["where"] = _where
        if where_note:
            log.warning("Datastore repaired a repeated `where` key", action=action)

        # Recover or explain a missing `table` before the action handlers see
        # it (see _INFERABLE_TABLE_ACTIONS above). Mutates kwargs["table"].
        table_note, table_error = await self._preflight_table(dm, action, kwargs, session_id)
        if table_error is not None:
            log.warning("Datastore tool result", action=action, success=False,
                        error=table_error.error)
            return table_error
        notes = [n for n in (alias_note, where_note, table_note) if n]

        try:
            if action == "list_tables":
                result = await self._list_tables(dm)
            elif action == "describe":
                result = await self._describe(dm, kwargs.get("table"))
            elif action == "create_table":
                result = await self._create_table(dm, kwargs.get("table"), kwargs.get("columns"), kwargs.get("unique"))
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
            elif action == "upsert":
                result = await self._upsert(dm, kwargs)
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

        # Tell the model what we assumed or renamed, so the next call arrives
        # correct instead of leaning on the recovery again.
        if result.success and notes:
            head = "\n".join(notes)
            result.content = f"{head}\n{result.content}" if result.content else head

        # Log result
        if result.success:
            # Truncate long content for log readability
            _content_preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
            log.info("Datastore tool result", action=action, success=True, content=_content_preview)
        else:
            log.warning("Datastore tool result", action=action, success=False, error=result.error)

        return result

    @staticmethod
    async def _preflight_table(
        dm: Any, action: str, kwargs: dict[str, Any], session_id: str | None,
    ) -> tuple[str | None, ToolResult | None]:
        """Fill in a missing `table`, or fail with a retryable error.

        Returns ``(note, error)``: ``note`` is a line to prepend to a
        successful result when the table was inferred; ``error`` is a
        ToolResult to return immediately. At most one is ever set.
        """
        table = str(kwargs.get("table") or "").strip()
        if table:
            kwargs["table"] = table
            _remember_table(dm, session_id, table)
            return None, None
        if action in _NO_TABLE_ACTIONS or action in _OPTIONAL_TABLE_ACTIONS:
            return None, None
        # export accepts a raw query instead of a table.
        if action == "export" and kwargs.get("sql_query"):
            return None, None

        try:
            tables = await dm.list_tables()
        except Exception as e:  # a broken store is the handler's problem, not ours
            log.debug("table pre-flight could not list tables", error=str(e))
            return None, None

        if action in _INFERABLE_TABLE_ACTIONS:
            names = [t.name for t in tables]
            last = _LAST_TABLE.get(_scope_key(dm, session_id))
            guess = last if last in names else (names[0] if len(names) == 1 else None)
            if guess:
                kwargs["table"] = guess
                log.info("Datastore inferred missing table", action=action, table=guess,
                         via="last-used" if guess == last else "only-table")
                return (f"(This call omitted `table`; used **{guess}**. "
                        "Include `table` explicitly next time.)"), None

        return None, ToolResult(
            success=False, error=_missing_table_error(action, tables, kwargs))

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
    async def _create_table(dm: Any, table: str | None, columns_raw: str | None,
                            unique_raw: Any = None) -> ToolResult:
        if not table:
            return ToolResult(success=False, error="'table' is required for create_table.")
        if not columns_raw:
            return ToolResult(success=False, error="'columns' is required for create_table.")
        columns = _parse_json_str(columns_raw, "columns")
        if not isinstance(columns, list):
            return ToolResult(success=False, error="'columns' must be a JSON array.")
        unique = _parse_columns(unique_raw)
        info = await dm.create_table(table, columns, unique=unique)
        col_names = ", ".join(c.name for c in info.columns)
        uniq_note = f" · unique key: {', '.join(unique)}" if unique else ""
        return ToolResult(
            success=True,
            content=f"Created table **{info.name}** with columns: {col_names}{uniq_note}",
        )

    @staticmethod
    async def _upsert(dm: Any, kwargs: dict[str, Any]) -> ToolResult:
        _action = "upsert"
        table = kwargs.get("table")
        rows_raw = kwargs.get("rows")
        if not table:
            return ToolResult(success=False, error="'table' is required.")
        if not rows_raw:
            return ToolResult(success=False, error=_missing_payload_error(
                _action, "rows", kwargs,
                '{"action": "%s", "table": "%s", "rows": [{"col": "value"}]}'
                % (_action, table or "your_table")))
        rows = _parse_json_str(rows_raw, "rows")
        if not isinstance(rows, list):
            return ToolResult(success=False, error="'rows' must be a JSON array of objects.")
        key_columns = _parse_columns(kwargs.get("key"))
        count = await dm.upsert_rows(table, rows, key_columns=key_columns)
        return ToolResult(success=True, content=f"Upserted {count} row(s) into **{table}** (insert-or-update on the unique key).")

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
        _action = "insert"
        table = kwargs.get("table")
        rows_raw = kwargs.get("rows")
        if not table:
            return ToolResult(success=False, error="'table' is required.")
        if not rows_raw:
            return ToolResult(success=False, error=_missing_payload_error(
                _action, "rows", kwargs,
                '{"action": "%s", "table": "%s", "rows": [{"col": "value"}]}'
                % (_action, table or "your_table")))
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
            return ToolResult(success=False, error=_missing_payload_error(
                "update", "set_values", kwargs,
                '{"action": "update", "table": "%s", "set_values": {"col": "value"}, '
                '"where": {"id": 1}}' % (table or "your_table")))
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
        # Catch writes before the store's guard does, so the refusal can carry
        # the structured call that would have worked.
        if not re.match(r"^\s*SELECT\b", str(sql_query), re.IGNORECASE):
            return ToolResult(success=False, error=_sql_write_error(str(sql_query)))
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
        sql_query = kwargs.get("sql_query")

        if not table and not sql_query:
            return ToolResult(success=False, error="Either 'table' or 'sql_query' is required.")

        fmt = kwargs.get("format", "csv").lower()
        if fmt not in ("csv", "json", "xlsx"):
            return ToolResult(success=False, error=f"Unsupported format: {fmt}")

        # Resolve output path.
        # If file_path is provided, use it (resolve relative to saved base).
        # Otherwise, auto-generate a path in saved/output/<session_id>/.
        file_path_str = kwargs.get("file_path")
        saved_base = kwargs.get("_saved_base_path")
        runtime_base = kwargs.get("_runtime_base_path")
        session_id = kwargs.get("_session_id", "default")

        if file_path_str:
            # Resolve the requested file_path relative to the saved base.
            fp = Path(file_path_str)
            if not fp.is_absolute():
                if saved_base:
                    # file_path is relative — if it starts with "saved/", strip
                    # that prefix since saved_base already points there.
                    parts = fp.parts
                    if parts and parts[0] == "saved":
                        fp = Path(*parts[1:]) if len(parts) > 1 else fp
                    fp = Path(saved_base) / fp
                elif runtime_base:
                    fp = Path(runtime_base) / file_path_str
                # else keep as-is (relative to cwd)
            output_path = fp.resolve()
            # Ensure the format extension matches
            if output_path.suffix.lower().lstrip(".") != fmt:
                output_path = output_path.with_suffix(f".{fmt}")
        else:
            # Auto-generate output path
            if saved_base:
                output_dir = Path(saved_base) / "output" / str(session_id)
            elif runtime_base:
                output_dir = Path(runtime_base) / "saved" / "output" / str(session_id)
            else:
                output_dir = Path(".") / "saved" / "output" / str(session_id)
            if sql_query:
                file_stem = table or "query_result"
            else:
                file_stem = table
            output_path = output_dir / f"{file_stem}.{fmt}"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # sql_query mode: export the result of a raw SELECT (supports JOINs).
        if sql_query:
            if fmt == "csv":
                path = await dm.export_sql_csv(sql_query, output_path)
            elif fmt == "json":
                path = await dm.export_sql_json(sql_query, output_path)
            else:
                path = await dm.export_sql_xlsx(sql_query, output_path)
            return ToolResult(
                success=True,
                content=f"Exported query result to {path}",
            )

        # Single-table export mode.
        columns = _parse_columns(kwargs.get("columns"))
        where_raw = kwargs.get("where")
        where = _parse_json_str(where_raw, "where") if where_raw else None

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
