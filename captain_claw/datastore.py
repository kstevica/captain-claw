"""User-facing relational datastore backed by a dedicated SQLite database.

Provides structured table management, CRUD operations, import/export,
and read-only raw SQL queries.  Completely separate from the session
and memory databases.
"""

from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from captain_claw.config import get_config
from captain_claw.logging import get_logger

log = get_logger(__name__)

# ── Column type mapping ──────────────────────────────────────────────
# user-facing type → SQLite affinity
TYPE_MAP: dict[str, str] = {
    "text": "TEXT",
    "integer": "INTEGER",
    "real": "REAL",
    "boolean": "INTEGER",    # stored as 0/1
    "date": "TEXT",          # ISO date string
    "datetime": "TEXT",      # ISO datetime string
    "json": "TEXT",          # JSON-encoded string
}

VALID_TYPES = set(TYPE_MAP.keys())

# All user tables are prefixed to avoid clashing with meta tables.
TABLE_PREFIX = "ds_"

# Operators allowed in structured WHERE clauses.
_ALLOWED_OPS = {"=", "!=", "<", ">", "<=", ">=", "LIKE", "NOT LIKE", "IN", "NOT IN", "IS NULL", "IS NOT NULL"}

# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class ColumnDef:
    name: str
    col_type: str  # one of VALID_TYPES
    position: int = 0


@dataclass
class TableInfo:
    name: str
    columns: list[ColumnDef] = field(default_factory=list)
    row_count: int = 0
    created_at: str = ""
    updated_at: str = ""


# ── DatastoreManager ─────────────────────────────────────────────────

class DatastoreManager:
    """Manages the user-facing relational datastore."""

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            cfg = get_config()
            self.db_path = Path(cfg.datastore.path).expanduser()
        else:
            self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: aiosqlite.Connection | None = None

    # ── lifecycle ────────────────────────────────────────────────────

    async def _ensure_db(self) -> None:
        if self._db is not None:
            return
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS _ds_tables (
                name        TEXT PRIMARY KEY,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS _ds_columns (
                table_name  TEXT NOT NULL,
                col_name    TEXT NOT NULL,
                col_type    TEXT NOT NULL DEFAULT 'text',
                position    INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (table_name, col_name),
                FOREIGN KEY (table_name) REFERENCES _ds_tables(name) ON DELETE CASCADE
            )
        """)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _safe_name(name: str) -> str:
        """Sanitize a table/column name to ``[a-z0-9_]``."""
        cleaned = re.sub(r"[^a-z0-9_]", "_", name.strip().lower())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        if not cleaned or cleaned[0].isdigit():
            cleaned = "t_" + cleaned
        return cleaned

    @staticmethod
    def _internal_name(user_name: str) -> str:
        return TABLE_PREFIX + DatastoreManager._safe_name(user_name)

    async def _resolve_table(self, name: str) -> tuple[str, str]:
        """Return ``(safe_name, internal_name)`` and verify the table exists."""
        await self._ensure_db()
        safe = self._safe_name(name)
        assert self._db is not None
        async with self._db.execute(
            "SELECT 1 FROM _ds_tables WHERE name = ?", (safe,)
        ) as cur:
            if not await cur.fetchone():
                raise ValueError(f"Table not found: {name}")
        return safe, self._internal_name(safe)

    async def _table_columns(self, safe_name: str) -> list[ColumnDef]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT col_name, col_type, position FROM _ds_columns "
            "WHERE table_name = ? ORDER BY position",
            (safe_name,),
        ) as cur:
            rows = await cur.fetchall()
        return [ColumnDef(name=r[0], col_type=r[1], position=r[2]) for r in rows]

    async def _row_count(self, internal: str) -> int:
        assert self._db is not None
        try:
            async with self._db.execute(f'SELECT COUNT(*) FROM "{internal}"') as cur:
                row = await cur.fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    def _now(self) -> str:  # noqa: PLR6301
        return datetime.now(UTC).isoformat()

    # ── table management ─────────────────────────────────────────────

    async def list_tables(self) -> list[TableInfo]:
        await self._ensure_db()
        assert self._db is not None
        async with self._db.execute(
            "SELECT name, created_at, updated_at FROM _ds_tables ORDER BY name"
        ) as cur:
            meta_rows = await cur.fetchall()
        tables: list[TableInfo] = []
        for name, created_at, updated_at in meta_rows:
            internal = self._internal_name(name)
            row_count = await self._row_count(internal)
            columns = await self._table_columns(name)
            tables.append(TableInfo(
                name=name, columns=columns, row_count=row_count,
                created_at=created_at, updated_at=updated_at,
            ))
        return tables

    async def describe_table(self, name: str) -> TableInfo:
        safe, internal = await self._resolve_table(name)
        columns = await self._table_columns(safe)
        row_count = await self._row_count(internal)
        assert self._db is not None
        async with self._db.execute(
            "SELECT created_at, updated_at FROM _ds_tables WHERE name = ?", (safe,)
        ) as cur:
            meta = await cur.fetchone()
        return TableInfo(
            name=safe, columns=columns, row_count=row_count,
            created_at=meta[0] if meta else "",
            updated_at=meta[1] if meta else "",
        )

    async def create_table(
        self, name: str, columns: list[dict[str, str]],
    ) -> TableInfo:
        await self._ensure_db()
        assert self._db is not None
        cfg = get_config()

        existing = await self.list_tables()
        if len(existing) >= cfg.datastore.max_tables:
            raise ValueError(f"Table limit ({cfg.datastore.max_tables}) reached")

        safe = self._safe_name(name)
        internal = self._internal_name(safe)

        # Check for duplicate
        async with self._db.execute(
            "SELECT 1 FROM _ds_tables WHERE name = ?", (safe,)
        ) as cur:
            if await cur.fetchone():
                raise ValueError(f"Table already exists: {safe}")

        if not columns:
            raise ValueError("At least one column required")

        col_defs: list[str] = ["_id INTEGER PRIMARY KEY AUTOINCREMENT"]
        col_objects: list[ColumnDef] = []
        seen: set[str] = set()

        for i, col in enumerate(columns):
            col_name = self._safe_name(col.get("name", ""))
            col_type = (col.get("type", "text") or "text").lower().strip()
            if not col_name or col_name.startswith("_"):
                raise ValueError(f"Invalid column name: {col.get('name')}")
            if col_type not in VALID_TYPES:
                raise ValueError(f"Invalid type '{col_type}'. Valid: {sorted(VALID_TYPES)}")
            if col_name in seen:
                raise ValueError(f"Duplicate column: {col_name}")
            seen.add(col_name)
            col_defs.append(f'"{col_name}" {TYPE_MAP[col_type]}')
            col_objects.append(ColumnDef(name=col_name, col_type=col_type, position=i))

        now = self._now()
        await self._db.execute(f'CREATE TABLE "{internal}" ({", ".join(col_defs)})')
        await self._db.execute(
            "INSERT INTO _ds_tables (name, created_at, updated_at) VALUES (?, ?, ?)",
            (safe, now, now),
        )
        for c in col_objects:
            await self._db.execute(
                "INSERT INTO _ds_columns (table_name, col_name, col_type, position) "
                "VALUES (?, ?, ?, ?)",
                (safe, c.name, c.col_type, c.position),
            )
        await self._db.commit()
        return TableInfo(name=safe, columns=col_objects, row_count=0, created_at=now, updated_at=now)

    async def drop_table(self, name: str) -> bool:
        safe, internal = await self._resolve_table(name)
        assert self._db is not None
        await self._db.execute(f'DROP TABLE IF EXISTS "{internal}"')
        await self._db.execute("DELETE FROM _ds_columns WHERE table_name = ?", (safe,))
        await self._db.execute("DELETE FROM _ds_tables WHERE name = ?", (safe,))
        await self._db.commit()
        return True

    # ── schema changes ───────────────────────────────────────────────

    async def add_column(
        self, table_name: str, col_name: str, col_type: str = "text",
        default: Any = None,
    ) -> bool:
        safe, internal = await self._resolve_table(table_name)
        assert self._db is not None
        col_name = self._safe_name(col_name)
        col_type = col_type.lower().strip()
        if col_type not in VALID_TYPES:
            raise ValueError(f"Invalid type: {col_type}")
        if col_name.startswith("_"):
            raise ValueError(f"Column name cannot start with underscore: {col_name}")

        existing = await self._table_columns(safe)
        if any(c.name == col_name for c in existing):
            raise ValueError(f"Column already exists: {col_name}")

        sqlite_type = TYPE_MAP[col_type]
        default_clause = ""
        if default is not None:
            default_clause = f" DEFAULT {self._quote_literal(default)}"
        await self._db.execute(
            f'ALTER TABLE "{internal}" ADD COLUMN "{col_name}" {sqlite_type}{default_clause}'
        )
        position = max((c.position for c in existing), default=-1) + 1
        await self._db.execute(
            "INSERT INTO _ds_columns (table_name, col_name, col_type, position) "
            "VALUES (?, ?, ?, ?)",
            (safe, col_name, col_type, position),
        )
        await self._db.execute(
            "UPDATE _ds_tables SET updated_at = ? WHERE name = ?",
            (self._now(), safe),
        )
        await self._db.commit()
        return True

    async def rename_column(self, table_name: str, old_name: str, new_name: str) -> bool:
        safe, internal = await self._resolve_table(table_name)
        assert self._db is not None
        old_safe = self._safe_name(old_name)
        new_safe = self._safe_name(new_name)
        if new_safe.startswith("_"):
            raise ValueError(f"Column name cannot start with underscore: {new_safe}")

        existing = await self._table_columns(safe)
        if not any(c.name == old_safe for c in existing):
            raise ValueError(f"Column not found: {old_safe}")
        if any(c.name == new_safe for c in existing):
            raise ValueError(f"Column already exists: {new_safe}")

        await self._db.execute(
            f'ALTER TABLE "{internal}" RENAME COLUMN "{old_safe}" TO "{new_safe}"'
        )
        await self._db.execute(
            "UPDATE _ds_columns SET col_name = ? WHERE table_name = ? AND col_name = ?",
            (new_safe, safe, old_safe),
        )
        await self._db.execute(
            "UPDATE _ds_tables SET updated_at = ? WHERE name = ?",
            (self._now(), safe),
        )
        await self._db.commit()
        return True

    async def drop_column(self, table_name: str, col_name: str) -> bool:
        safe, internal = await self._resolve_table(table_name)
        assert self._db is not None
        col_safe = self._safe_name(col_name)

        existing = await self._table_columns(safe)
        if not any(c.name == col_safe for c in existing):
            raise ValueError(f"Column not found: {col_safe}")
        if len(existing) <= 1:
            raise ValueError("Cannot drop the last column")

        await self._db.execute(f'ALTER TABLE "{internal}" DROP COLUMN "{col_safe}"')
        await self._db.execute(
            "DELETE FROM _ds_columns WHERE table_name = ? AND col_name = ?",
            (safe, col_safe),
        )
        await self._db.execute(
            "UPDATE _ds_tables SET updated_at = ? WHERE name = ?",
            (self._now(), safe),
        )
        await self._db.commit()
        return True

    async def change_column_type(
        self, table_name: str, col_name: str, new_type: str,
    ) -> bool:
        """Change a column's type via table rebuild with CAST."""
        safe, internal = await self._resolve_table(table_name)
        assert self._db is not None
        col_safe = self._safe_name(col_name)
        new_type = new_type.lower().strip()
        if new_type not in VALID_TYPES:
            raise ValueError(f"Invalid type: {new_type}")

        existing = await self._table_columns(safe)
        target_col = next((c for c in existing if c.name == col_safe), None)
        if not target_col:
            raise ValueError(f"Column not found: {col_safe}")

        # Build new table schema
        tmp_internal = internal + "__tmp"
        col_defs = ["_id INTEGER PRIMARY KEY AUTOINCREMENT"]
        select_parts = ["_id"]
        for c in existing:
            if c.name == col_safe:
                sqlite_type = TYPE_MAP[new_type]
                col_defs.append(f'"{c.name}" {sqlite_type}')
                select_parts.append(f'CAST("{c.name}" AS {sqlite_type}) AS "{c.name}"')
            else:
                sqlite_type = TYPE_MAP.get(c.col_type, "TEXT")
                col_defs.append(f'"{c.name}" {sqlite_type}')
                select_parts.append(f'"{c.name}"')

        await self._db.execute(f'CREATE TABLE "{tmp_internal}" ({", ".join(col_defs)})')
        await self._db.execute(
            f'INSERT INTO "{tmp_internal}" SELECT {", ".join(select_parts)} FROM "{internal}"'
        )
        await self._db.execute(f'DROP TABLE "{internal}"')
        await self._db.execute(f'ALTER TABLE "{tmp_internal}" RENAME TO "{internal}"')

        await self._db.execute(
            "UPDATE _ds_columns SET col_type = ? WHERE table_name = ? AND col_name = ?",
            (new_type, safe, col_safe),
        )
        await self._db.execute(
            "UPDATE _ds_tables SET updated_at = ? WHERE name = ?",
            (self._now(), safe),
        )
        await self._db.commit()
        return True

    # ── where clause builder ─────────────────────────────────────────

    def _build_where(
        self,
        where: dict[str, Any],
        valid_columns: set[str],
    ) -> tuple[str, list[Any]]:
        """Build a WHERE clause + params from a structured filter dict."""
        clauses: list[str] = []
        params: list[Any] = []

        for key, val in where.items():
            if key == "_all":
                continue
            # _id is the auto-generated primary key — allow it directly
            if key == "_id":
                col = "_id"
            else:
                col = self._safe_name(key)
                if col not in valid_columns:
                    raise ValueError(f"Unknown column in where: {key}")

            if isinstance(val, dict):
                op = str(val.get("op", "=")).upper().strip()
                if op not in _ALLOWED_OPS:
                    raise ValueError(f"Unsupported operator: {op}")
                if op in ("IS NULL", "IS NOT NULL"):
                    clauses.append(f'"{col}" {op}')
                elif op in ("IN", "NOT IN"):
                    values = val.get("value", [])
                    if not isinstance(values, list) or not values:
                        raise ValueError(f"IN operator requires a non-empty list")
                    placeholders = ", ".join("?" for _ in values)
                    clauses.append(f'"{col}" {op} ({placeholders})')
                    params.extend(values)
                else:
                    clauses.append(f'"{col}" {op} ?')
                    params.append(val.get("value"))
            else:
                clauses.append(f'"{col}" = ?')
                params.append(val)

        if not clauses:
            return "", params
        return "WHERE " + " AND ".join(clauses), params

    @staticmethod
    def _quote_literal(value: Any) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, (int, float)):
            return str(value)
        return "'" + str(value).replace("'", "''") + "'"

    # ── data operations ──────────────────────────────────────────────

    async def insert_rows(
        self, table_name: str, rows: list[dict[str, Any]],
    ) -> int:
        safe, internal = await self._resolve_table(table_name)
        assert self._db is not None
        cfg = get_config()

        if not rows:
            return 0

        current_count = await self._row_count(internal)
        if current_count + len(rows) > cfg.datastore.max_rows_per_table:
            raise ValueError(
                f"Would exceed row limit ({cfg.datastore.max_rows_per_table}). "
                f"Current: {current_count}, inserting: {len(rows)}"
            )

        columns = await self._table_columns(safe)
        col_names = {c.name for c in columns}

        inserted = 0
        for row in rows:
            # Filter to known columns only
            filtered = {self._safe_name(k): v for k, v in row.items() if self._safe_name(k) in col_names}
            if not filtered:
                continue
            col_list = list(filtered.keys())
            placeholders = ", ".join("?" for _ in col_list)
            col_clause = ", ".join(f'"{c}"' for c in col_list)
            values = list(filtered.values())
            await self._db.execute(
                f'INSERT INTO "{internal}" ({col_clause}) VALUES ({placeholders})',
                values,
            )
            inserted += 1

        if inserted:
            await self._db.execute(
                "UPDATE _ds_tables SET updated_at = ? WHERE name = ?",
                (self._now(), safe),
            )
            await self._db.commit()
        return inserted

    async def update_rows(
        self, table_name: str,
        set_values: dict[str, Any],
        where: dict[str, Any] | None = None,
    ) -> int:
        safe, internal = await self._resolve_table(table_name)
        assert self._db is not None

        if not set_values:
            raise ValueError("set_values cannot be empty")

        columns = await self._table_columns(safe)
        col_names = {c.name for c in columns}

        set_clauses: list[str] = []
        set_params: list[Any] = []
        for k, v in set_values.items():
            col = self._safe_name(k)
            if col not in col_names:
                raise ValueError(f"Unknown column: {k}")
            set_clauses.append(f'"{col}" = ?')
            set_params.append(v)

        where_clause = ""
        where_params: list[Any] = []
        if where:
            where_clause, where_params = self._build_where(where, col_names)

        sql = f'UPDATE "{internal}" SET {", ".join(set_clauses)} {where_clause}'
        cursor = await self._db.execute(sql, set_params + where_params)
        affected = cursor.rowcount
        if affected:
            await self._db.execute(
                "UPDATE _ds_tables SET updated_at = ? WHERE name = ?",
                (self._now(), safe),
            )
            await self._db.commit()
        return affected

    async def update_column(
        self, table_name: str, col_name: str,
        value: Any = None, expression: str | None = None,
    ) -> int:
        safe, internal = await self._resolve_table(table_name)
        assert self._db is not None
        col_safe = self._safe_name(col_name)

        columns = await self._table_columns(safe)
        if not any(c.name == col_safe for c in columns):
            raise ValueError(f"Column not found: {col_safe}")

        if expression:
            # Raw expression -- only allow simple math/string ops
            sql = f'UPDATE "{internal}" SET "{col_safe}" = {expression}'
            cursor = await self._db.execute(sql)
        else:
            sql = f'UPDATE "{internal}" SET "{col_safe}" = ?'
            cursor = await self._db.execute(sql, (value,))

        affected = cursor.rowcount
        if affected:
            await self._db.execute(
                "UPDATE _ds_tables SET updated_at = ? WHERE name = ?",
                (self._now(), safe),
            )
            await self._db.commit()
        return affected

    async def delete_rows(
        self, table_name: str,
        where: dict[str, Any] | None = None,
    ) -> int:
        safe, internal = await self._resolve_table(table_name)
        assert self._db is not None

        columns = await self._table_columns(safe)
        col_names = {c.name for c in columns}

        if not where:
            raise ValueError("WHERE clause required. Pass {\"_all\": true} to delete all rows.")

        if where.get("_all") is True:
            cursor = await self._db.execute(f'DELETE FROM "{internal}"')
        else:
            where_clause, where_params = self._build_where(where, col_names)
            if not where_clause:
                raise ValueError("Empty WHERE clause. Pass {\"_all\": true} to delete all rows.")
            cursor = await self._db.execute(f'DELETE FROM "{internal}" {where_clause}', where_params)

        affected = cursor.rowcount
        if affected:
            await self._db.execute(
                "UPDATE _ds_tables SET updated_at = ? WHERE name = ?",
                (self._now(), safe),
            )
            await self._db.commit()
        return affected

    # ── query operations ─────────────────────────────────────────────

    async def query(
        self, table_name: str,
        columns: list[str] | None = None,
        where: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        safe, internal = await self._resolve_table(table_name)
        assert self._db is not None
        cfg = get_config()

        table_cols = await self._table_columns(safe)
        col_names = {c.name for c in table_cols}

        # Select clause
        if columns:
            select_cols = []
            for c in columns:
                c_safe = self._safe_name(c)
                if c_safe not in col_names and c_safe != "_id":
                    raise ValueError(f"Unknown column: {c}")
                select_cols.append(f'"{c_safe}"')
            select_clause = ", ".join(select_cols)
            result_col_names = [self._safe_name(c) for c in columns]
        else:
            select_clause = '"_id", ' + ", ".join(f'"{c.name}"' for c in table_cols)
            result_col_names = ["_id"] + [c.name for c in table_cols]

        # Where
        where_clause = ""
        where_params: list[Any] = []
        if where:
            where_clause, where_params = self._build_where(where, col_names)

        # Order by
        order_clause = ""
        if order_by:
            parts = []
            for ob in order_by:
                ob = str(ob).strip()
                if ob.startswith("-"):
                    col = self._safe_name(ob[1:])
                    parts.append(f'"{col}" DESC')
                else:
                    col = self._safe_name(ob)
                    parts.append(f'"{col}" ASC')
            order_clause = "ORDER BY " + ", ".join(parts)

        # Limit
        effective_limit = min(limit or cfg.datastore.max_query_rows, cfg.datastore.max_query_rows)

        sql = (
            f'SELECT {select_clause} FROM "{internal}" '
            f'{where_clause} {order_clause} '
            f'LIMIT {effective_limit} OFFSET {offset}'
        )

        async with self._db.execute(sql, where_params) as cur:
            rows = await cur.fetchall()

        # Total count (for pagination info)
        count_sql = f'SELECT COUNT(*) FROM "{internal}" {where_clause}'
        async with self._db.execute(count_sql, where_params) as cur:
            total_row = await cur.fetchone()
            total = total_row[0] if total_row else 0

        return {
            "columns": result_col_names,
            "rows": [list(r) for r in rows],
            "total": total,
            "offset": offset,
            "limit": effective_limit,
        }

    async def raw_select(self, sql: str) -> dict[str, Any]:
        """Execute a read-only SQL query. Only SELECT is allowed."""
        await self._ensure_db()
        assert self._db is not None
        cfg = get_config()

        stripped = sql.strip()
        # Validate it's a SELECT
        if not re.match(r"^SELECT\b", stripped, re.IGNORECASE):
            raise ValueError("Only SELECT queries are allowed")

        # Block obvious mutation attempts
        danger = re.search(
            r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH)\b",
            stripped, re.IGNORECASE,
        )
        if danger:
            raise ValueError(f"Mutation keyword '{danger.group()}' not allowed in raw SELECT")

        # Replace user table names with internal names.
        # Users may reference tables as-is; we need to add the ds_ prefix.
        async with self._db.execute("SELECT name FROM _ds_tables") as cur:
            known_tables = [r[0] for r in await cur.fetchall()]

        processed = stripped
        for tbl in sorted(known_tables, key=len, reverse=True):
            internal = self._internal_name(tbl)
            # Replace table name when it appears as a word boundary
            processed = re.sub(
                rf'\b{re.escape(tbl)}\b', f'"{internal}"', processed,
            )

        # Enforce LIMIT
        max_rows = cfg.datastore.max_query_rows
        if not re.search(r"\bLIMIT\b", processed, re.IGNORECASE):
            processed = processed.rstrip(";") + f" LIMIT {max_rows}"

        async with self._db.execute(processed) as cur:
            if cur.description:
                col_names = [d[0] for d in cur.description]
            else:
                col_names = []
            rows = await cur.fetchall()

        return {
            "columns": col_names,
            "rows": [list(r) for r in rows],
            "total": len(rows),
        }

    # ── import / export ──────────────────────────────────────────────

    async def import_csv(
        self, file_path: Path, table_name: str | None = None,
        append: bool = False,
    ) -> dict[str, Any]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = file_path.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise ValueError("CSV has no headers")

        headers = [self._safe_name(h) for h in reader.fieldnames]
        all_rows = list(reader)

        if not table_name:
            table_name = file_path.stem

        safe = self._safe_name(table_name)

        if not append:
            # Infer types from data
            col_defs = self._infer_column_types(headers, all_rows)
            try:
                await self.create_table(safe, col_defs)
            except ValueError as e:
                if "already exists" in str(e):
                    raise ValueError(
                        f"Table '{safe}' already exists. Use append=true to add data, "
                        f"or drop the table first."
                    ) from e
                raise

        # Build row dicts with safe column names
        rows_to_insert = []
        for row in all_rows:
            cleaned = {}
            for orig_key, val in row.items():
                safe_key = self._safe_name(orig_key)
                cleaned[safe_key] = self._coerce_value(val)
            rows_to_insert.append(cleaned)

        inserted = await self.insert_rows(safe, rows_to_insert)
        return {"table": safe, "rows_imported": inserted, "columns": headers}

    async def import_xlsx(
        self, file_path: Path, table_name: str | None = None,
        sheet_name: str | None = None, append: bool = False,
    ) -> dict[str, Any]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        headers, all_rows = self._parse_xlsx(file_path, sheet_name)
        if not headers:
            raise ValueError("XLSX sheet has no data")

        safe_headers = [self._safe_name(h) for h in headers]
        if not table_name:
            table_name = file_path.stem
        safe = self._safe_name(table_name)

        if not append:
            col_defs = self._infer_column_types(
                safe_headers,
                [{safe_headers[i]: v for i, v in enumerate(row) if i < len(safe_headers)} for row in all_rows[:100]],
            )
            try:
                await self.create_table(safe, col_defs)
            except ValueError as e:
                if "already exists" in str(e):
                    raise ValueError(
                        f"Table '{safe}' already exists. Use append=true to add data."
                    ) from e
                raise

        rows_to_insert = []
        for row in all_rows:
            cleaned = {}
            for i, val in enumerate(row):
                if i < len(safe_headers):
                    cleaned[safe_headers[i]] = self._coerce_value(val)
            rows_to_insert.append(cleaned)

        inserted = await self.insert_rows(safe, rows_to_insert)
        return {"table": safe, "rows_imported": inserted, "columns": safe_headers}

    async def export_csv(
        self, table_name: str, output_path: Path,
        columns: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> Path:
        cfg = get_config()
        result = await self.query(
            table_name, columns=columns, where=where,
            limit=cfg.datastore.max_export_rows,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(result["columns"])
            writer.writerows(result["rows"])
        return output_path

    async def export_xlsx(
        self, table_name: str, output_path: Path,
        columns: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> Path:
        cfg = get_config()
        result = await self.query(
            table_name, columns=columns, where=where,
            limit=cfg.datastore.max_export_rows,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_xlsx(output_path, result["columns"], result["rows"])
        return output_path

    # ── summary for context injection ────────────────────────────────

    async def get_tables_summary(self) -> list[TableInfo]:
        return await self.list_tables()

    # ── internal: CSV/XLSX helpers ───────────────────────────────────

    @staticmethod
    def _coerce_value(val: Any) -> Any:
        """Coerce a string value to the most appropriate Python type."""
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return None
        if isinstance(val, str):
            v = val.strip()
            if v.lower() in ("true", "yes"):
                return 1
            if v.lower() in ("false", "no"):
                return 0
            try:
                return int(v)
            except ValueError:
                pass
            try:
                return float(v)
            except ValueError:
                pass
        return val

    @staticmethod
    def _infer_column_types(
        headers: list[str], sample_rows: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Infer column types from sample data."""
        col_defs: list[dict[str, str]] = []
        for header in headers:
            values = [r.get(header) for r in sample_rows[:100] if r.get(header) is not None and str(r.get(header)).strip() != ""]
            col_type = "text"
            if values:
                all_int = all(_looks_int(v) for v in values)
                all_float = all(_looks_float(v) for v in values)
                all_bool = all(_looks_bool(v) for v in values)
                if all_bool:
                    col_type = "boolean"
                elif all_int:
                    col_type = "integer"
                elif all_float:
                    col_type = "real"
            col_defs.append({"name": header, "type": col_type})
        return col_defs

    @staticmethod
    def _parse_xlsx(
        file_path: Path, sheet_name: str | None = None,
    ) -> tuple[list[str], list[list[Any]]]:
        """Parse XLSX into headers + rows using ZIP/XML (no external deps)."""
        ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        ns_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

        with zipfile.ZipFile(file_path) as zf:
            # Parse shared strings
            shared: list[str] = []
            if "xl/sharedStrings.xml" in zf.namelist():
                ss_tree = ET.parse(zf.open("xl/sharedStrings.xml"))
                for si in ss_tree.findall(f"{{{ns}}}si"):
                    parts = []
                    for t_elem in si.iter(f"{{{ns}}}t"):
                        if t_elem.text:
                            parts.append(t_elem.text)
                    shared.append("".join(parts))

            # Find sheet
            wb_tree = ET.parse(zf.open("xl/workbook.xml"))
            sheets_el = wb_tree.findall(f"{{{ns}}}sheets/{{{ns}}}sheet")
            if not sheets_el:
                return [], []

            target_sheet = None
            if sheet_name:
                for s in sheets_el:
                    if s.get("name", "").lower() == sheet_name.lower():
                        target_sheet = s
                        break
                if not target_sheet:
                    raise ValueError(f"Sheet not found: {sheet_name}")
            else:
                target_sheet = sheets_el[0]

            # Resolve sheet path via relationships
            rel_tree = ET.parse(zf.open("xl/_rels/workbook.xml.rels"))
            rid = target_sheet.get(f"{{{ns_rel}}}id", "")
            sheet_path = None
            for rel in rel_tree.findall("{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"):
                if rel.get("Id") == rid:
                    sheet_path = "xl/" + rel.get("Target", "")
                    break

            if not sheet_path or sheet_path not in zf.namelist():
                # Fallback: try first worksheet
                candidates = [n for n in zf.namelist() if n.startswith("xl/worksheets/sheet")]
                if not candidates:
                    return [], []
                sheet_path = sorted(candidates)[0]

            sheet_tree = ET.parse(zf.open(sheet_path))
            rows_data: list[list[Any]] = []

            for row_el in sheet_tree.findall(f"{{{ns}}}sheetData/{{{ns}}}row"):
                row_vals: dict[int, Any] = {}
                for cell in row_el.findall(f"{{{ns}}}c"):
                    ref = cell.get("r", "")
                    col_idx = _col_ref_to_index(ref)
                    cell_type = cell.get("t", "")
                    val_el = cell.find(f"{{{ns}}}v")
                    val_text = val_el.text if val_el is not None and val_el.text else ""

                    if cell_type == "s" and val_text:
                        idx = int(val_text)
                        value = shared[idx] if idx < len(shared) else ""
                    elif cell_type == "b":
                        value = val_text
                    else:
                        value = val_text
                    row_vals[col_idx] = value

                if row_vals:
                    max_col = max(row_vals.keys())
                    row_list = [row_vals.get(i, "") for i in range(max_col + 1)]
                    rows_data.append(row_list)

        if not rows_data:
            return [], []

        headers = [str(v) for v in rows_data[0]]
        data_rows = rows_data[1:]
        return headers, data_rows

    @staticmethod
    def _write_xlsx(
        output_path: Path, columns: list[str], rows: list[list[Any]],
    ) -> None:
        """Build a minimal XLSX file from columns + rows using ZIP/XML."""
        ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        ns_r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        ns_ct = "http://schemas.openxmlformats.org/package/2006/content-types"
        ns_pkg = "http://schemas.openxmlformats.org/package/2006/relationships"

        # Collect all unique strings for shared strings table
        all_strings: list[str] = []
        string_index: dict[str, int] = {}
        for col in columns:
            s = str(col)
            if s not in string_index:
                string_index[s] = len(all_strings)
                all_strings.append(s)
        for row in rows:
            for val in row:
                if isinstance(val, str):
                    if val not in string_index:
                        string_index[val] = len(all_strings)
                        all_strings.append(val)

        # Build shared strings XML
        ss_root = ET.Element("sst", xmlns=ns, count=str(len(all_strings)), uniqueCount=str(len(all_strings)))
        for s in all_strings:
            si = ET.SubElement(ss_root, "si")
            t = ET.SubElement(si, "t")
            t.text = s

        # Build worksheet XML
        ws_root = ET.Element("worksheet", xmlns=ns)
        sd = ET.SubElement(ws_root, "sheetData")

        # Header row
        header_row = ET.SubElement(sd, "row", r="1")
        for ci, col_name in enumerate(columns):
            ref = _index_to_col_ref(ci) + "1"
            c = ET.SubElement(header_row, "c", r=ref, t="s")
            v = ET.SubElement(c, "v")
            v.text = str(string_index[str(col_name)])

        # Data rows
        for ri, row in enumerate(rows, start=2):
            row_el = ET.SubElement(sd, "row", r=str(ri))
            for ci, val in enumerate(row):
                ref = _index_to_col_ref(ci) + str(ri)
                if val is None:
                    continue
                if isinstance(val, str):
                    c = ET.SubElement(row_el, "c", r=ref, t="s")
                    v = ET.SubElement(c, "v")
                    v.text = str(string_index[val])
                elif isinstance(val, (int, float)):
                    c = ET.SubElement(row_el, "c", r=ref)
                    v = ET.SubElement(c, "v")
                    v.text = str(val)
                else:
                    s = str(val)
                    if s not in string_index:
                        string_index[s] = len(all_strings)
                        all_strings.append(s)
                    c = ET.SubElement(row_el, "c", r=ref, t="s")
                    v = ET.SubElement(c, "v")
                    v.text = str(string_index[s])

        # Build workbook XML
        wb_root = ET.Element("workbook", xmlns=ns)
        wb_root.set(f"xmlns:r", ns_r)
        sheets = ET.SubElement(wb_root, "sheets")
        sheet = ET.SubElement(sheets, "sheet", name="Sheet1", sheetId="1")
        sheet.set("r:id", "rId1")

        # Build content types
        ct_root = ET.Element("Types", xmlns=ns_ct)
        ET.SubElement(ct_root, "Default", Extension="rels", ContentType="application/vnd.openxmlformats-package.relationships+xml")
        ET.SubElement(ct_root, "Default", Extension="xml", ContentType="application/xml")
        ET.SubElement(ct_root, "Override", PartName="/xl/workbook.xml", ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml")
        ET.SubElement(ct_root, "Override", PartName="/xl/worksheets/sheet1.xml", ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml")
        ET.SubElement(ct_root, "Override", PartName="/xl/sharedStrings.xml", ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml")

        # Build rels
        root_rels = ET.Element("Relationships", xmlns=ns_pkg)
        ET.SubElement(root_rels, "Relationship", Id="rId1", Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", Target="xl/workbook.xml")

        wb_rels = ET.Element("Relationships", xmlns=ns_pkg)
        ET.SubElement(wb_rels, "Relationship", Id="rId1", Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet", Target="worksheets/sheet1.xml")
        ET.SubElement(wb_rels, "Relationship", Id="rId2", Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings", Target="sharedStrings.xml")

        # Write ZIP
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", _et_to_string(ct_root))
            zf.writestr("_rels/.rels", _et_to_string(root_rels))
            zf.writestr("xl/workbook.xml", _et_to_string(wb_root))
            zf.writestr("xl/_rels/workbook.xml.rels", _et_to_string(wb_rels))
            zf.writestr("xl/worksheets/sheet1.xml", _et_to_string(ws_root))
            zf.writestr("xl/sharedStrings.xml", _et_to_string(ss_root))


# ── module-level helpers ─────────────────────────────────────────────

def _col_ref_to_index(ref: str) -> int:
    """Convert Excel column reference (e.g. 'AB3') to 0-based index."""
    col = "".join(c for c in ref if c.isalpha()).upper()
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _index_to_col_ref(idx: int) -> str:
    """Convert 0-based index to Excel column reference (e.g. 0 -> 'A')."""
    result = ""
    idx += 1
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        result = chr(rem + ord("A")) + result
    return result


def _et_to_string(element: ET.Element) -> str:
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n' + ET.tostring(
        element, encoding="unicode",
    )


def _looks_int(val: Any) -> bool:
    try:
        s = str(val).strip()
        int(s)
        return "." not in s
    except (ValueError, TypeError):
        return False


def _looks_float(val: Any) -> bool:
    try:
        float(str(val).strip())
        return True
    except (ValueError, TypeError):
        return False


def _looks_bool(val: Any) -> bool:
    return str(val).strip().lower() in ("true", "false", "yes", "no", "1", "0")


# ── singleton ────────────────────────────────────────────────────────

_manager: DatastoreManager | None = None


def get_datastore_manager() -> DatastoreManager:
    global _manager
    if _manager is None:
        _manager = DatastoreManager()
    return _manager
