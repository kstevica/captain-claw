"""Built-in entity storage for agent apps.

When an app's manifest does not bind to an external MCP server, the
framework still needs *somewhere* to put the data the user creates
through it. This module is that somewhere: a JSON-per-entity store
addressed by ``(agent_id, entity_id)``.

Records are plain dicts. Each record carries an ``id`` (UUID hex, auto-
generated if missing) and ISO-8601 ``created_at`` / ``updated_at``
timestamps. Per-entity files contain the full record list, which keeps
the storage trivial to inspect, back up, and migrate.

Layout on disk::

    ~/.captain-claw-fd/app_data/<agent_id>/<entity_id>.json

The :class:`EntityStore` protocol gives us a swap point for Postgres /
SQLite later without touching renderers or the manifest schema.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


Record = dict[str, Any]


# ── store protocol ────────────────────────────────────────────────────


class EntityStore(Protocol):
    def list(self, agent_id: str, entity_id: str) -> list[Record]: ...
    def get(self, agent_id: str, entity_id: str, record_id: str) -> Record | None: ...
    def create(self, agent_id: str, entity_id: str, data: Record) -> Record: ...
    def update(
        self, agent_id: str, entity_id: str, record_id: str, data: Record
    ) -> Record | None: ...
    def delete(self, agent_id: str, entity_id: str, record_id: str) -> bool: ...


# ── helpers ───────────────────────────────────────────────────────────


def _safe(name: str) -> str:
    out = "".join(c for c in name if c.isalnum() or c in ("-", "_"))
    if not out:
        raise ValueError(f"unsafe id: {name!r}")
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_base() -> Path:
    base = os.environ.get("CAPTAIN_CLAW_FD_HOME") or os.path.expanduser("~/.captain-claw-fd")
    return Path(base) / "app_data"


# ── local filesystem implementation ──────────────────────────────────


class LocalEntityStore:
    def __init__(self, base: Path | None = None) -> None:
        self._base = base or _default_base()

    def _path(self, agent_id: str, entity_id: str) -> Path:
        d = self._base / _safe(agent_id)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{_safe(entity_id)}.json"

    def _read(self, agent_id: str, entity_id: str) -> list[Record]:
        p = self._path(agent_id, entity_id)
        if not p.exists():
            return []
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
        return raw if isinstance(raw, list) else []

    def _write(self, agent_id: str, entity_id: str, records: list[Record]) -> None:
        p = self._path(agent_id, entity_id)
        tmp = p.with_suffix(".json.part")
        tmp.write_text(json.dumps(records, indent=2), encoding="utf-8")
        os.replace(tmp, p)

    def list(self, agent_id: str, entity_id: str) -> list[Record]:
        rows = self._read(agent_id, entity_id)
        rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return rows

    def get(self, agent_id: str, entity_id: str, record_id: str) -> Record | None:
        for r in self._read(agent_id, entity_id):
            if str(r.get("id")) == record_id:
                return r
        return None

    def create(self, agent_id: str, entity_id: str, data: Record) -> Record:
        rows = self._read(agent_id, entity_id)
        rec = dict(data)
        rec.setdefault("id", uuid.uuid4().hex)
        now = _now()
        rec.setdefault("created_at", now)
        rec["updated_at"] = now
        rows.append(rec)
        self._write(agent_id, entity_id, rows)
        return rec

    def update(
        self, agent_id: str, entity_id: str, record_id: str, data: Record
    ) -> Record | None:
        rows = self._read(agent_id, entity_id)
        for i, r in enumerate(rows):
            if str(r.get("id")) == record_id:
                merged = {**r, **data, "id": r.get("id", record_id), "updated_at": _now()}
                rows[i] = merged
                self._write(agent_id, entity_id, rows)
                return merged
        return None

    def delete(self, agent_id: str, entity_id: str, record_id: str) -> bool:
        rows = self._read(agent_id, entity_id)
        kept = [r for r in rows if str(r.get("id")) != record_id]
        if len(kept) == len(rows):
            return False
        self._write(agent_id, entity_id, kept)
        return True


# ── module-level singleton ────────────────────────────────────────────


_store: EntityStore | None = None


def get_store() -> EntityStore:
    global _store
    if _store is None:
        _store = LocalEntityStore()
    return _store


def set_store(store: EntityStore) -> None:
    global _store
    _store = store
