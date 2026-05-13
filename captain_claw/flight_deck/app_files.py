"""File storage for agent-app uploads.

Apps declared via manifest may accept file uploads (images, PDFs,
audio, etc.) that agents subsequently process. Files are addressed by
opaque ``file_id`` (a UUID) — never by their original filename — so
that an agent's tool call can refer to a file unambiguously across the
upload→process→archive lifecycle.

Phase 1 backend is the local filesystem. The :class:`FileStore`
protocol gives us a swap point for S3/R2 later without touching the
manifest schema, the renderer, or the agent.

Layout on disk::

    ~/.captain-claw-fd/app_files/<agent_id>/<file_id>          # raw bytes
    ~/.captain-claw-fd/app_files/<agent_id>/<file_id>.json     # metadata

A sidecar JSON keeps metadata (filename, mime, size, uploaded_by, ts)
next to the blob so a single-file copy is enough to migrate or back up.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel


# ── metadata ──────────────────────────────────────────────────────────


class FileMeta(BaseModel):
    file_id: str
    filename: str
    mime: str
    size: int
    uploaded_by: str | None = None       # user id, when known
    uploaded_at: str                     # ISO 8601 UTC


# ── store protocol ────────────────────────────────────────────────────


class FileStore(Protocol):
    def save(
        self,
        agent_id: str,
        *,
        filename: str,
        mime: str,
        content: bytes,
        uploaded_by: str | None = None,
    ) -> FileMeta: ...

    def get_path(self, agent_id: str, file_id: str) -> Path | None: ...
    def get_meta(self, agent_id: str, file_id: str) -> FileMeta | None: ...
    def list(self, agent_id: str) -> list[FileMeta]: ...
    def delete(self, agent_id: str, file_id: str) -> bool: ...


# ── local filesystem implementation ──────────────────────────────────


def _safe_id(agent_id: str) -> str:
    out = "".join(c for c in agent_id if c.isalnum() or c in ("-", "_"))
    if not out:
        raise ValueError(f"agent_id has no safe characters: {agent_id!r}")
    return out


def _default_base() -> Path:
    base = os.environ.get("CAPTAIN_CLAW_FD_HOME") or os.path.expanduser("~/.captain-claw-fd")
    return Path(base) / "app_files"


class LocalFileStore:
    def __init__(self, base: Path | None = None) -> None:
        self._base = base or _default_base()

    def _dir(self, agent_id: str) -> Path:
        d = self._base / _safe_id(agent_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(
        self,
        agent_id: str,
        *,
        filename: str,
        mime: str,
        content: bytes,
        uploaded_by: str | None = None,
    ) -> FileMeta:
        file_id = uuid.uuid4().hex
        d = self._dir(agent_id)
        blob = d / file_id
        meta = FileMeta(
            file_id=file_id,
            filename=filename,
            mime=mime,
            size=len(content),
            uploaded_by=uploaded_by,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
        )
        tmp = blob.with_suffix(".part")
        tmp.write_bytes(content)
        os.replace(tmp, blob)
        (d / f"{file_id}.json").write_text(
            json.dumps(meta.model_dump(), indent=2), encoding="utf-8"
        )
        return meta

    def get_path(self, agent_id: str, file_id: str) -> Path | None:
        if not _is_safe_file_id(file_id):
            return None
        p = self._dir(agent_id) / file_id
        return p if p.exists() else None

    def get_meta(self, agent_id: str, file_id: str) -> FileMeta | None:
        if not _is_safe_file_id(file_id):
            return None
        sidecar = self._dir(agent_id) / f"{file_id}.json"
        if not sidecar.exists():
            return None
        try:
            return FileMeta.model_validate_json(sidecar.read_text(encoding="utf-8"))
        except Exception:
            return None

    def list(self, agent_id: str) -> list[FileMeta]:
        d = self._dir(agent_id)
        out: list[FileMeta] = []
        for sidecar in d.glob("*.json"):
            try:
                out.append(FileMeta.model_validate_json(sidecar.read_text(encoding="utf-8")))
            except Exception:
                continue
        out.sort(key=lambda m: m.uploaded_at, reverse=True)
        return out

    def delete(self, agent_id: str, file_id: str) -> bool:
        if not _is_safe_file_id(file_id):
            return False
        d = self._dir(agent_id)
        removed = False
        for p in (d / file_id, d / f"{file_id}.json"):
            if p.exists():
                try:
                    p.unlink()
                    removed = True
                except OSError:
                    pass
        return removed


def _is_safe_file_id(file_id: str) -> bool:
    return bool(file_id) and all(c.isalnum() or c in ("-", "_") for c in file_id)


# ── module-level singleton ────────────────────────────────────────────


_store: FileStore | None = None


def get_store() -> FileStore:
    global _store
    if _store is None:
        _store = LocalFileStore()
    return _store


def set_store(store: FileStore) -> None:
    """Override the default store (test hook / future S3 swap)."""
    global _store
    _store = store
