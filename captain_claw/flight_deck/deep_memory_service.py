"""Flight Deck's owner-scoped deep memory — the only thing that talks to Typesense.

Agents do **not** hold a Typesense key and do not reach Typesense directly.
They call Flight Deck (see ``deep_memory_routes``), FD resolves who is asking,
and every read and write is stamped/filtered with that owner's id. Typesense
can therefore bind to loopback on the FD host and the admin key never leaves
this process.

That is a deliberate step past Typesense's own scoped API keys: those embed a
``filter_by`` the client cannot override, but they only cover *search*. Writes
have no equivalent, so a proxy is the only way to get one enforcement point for
both halves.

Indexing is opt-in per VFS project (``.vfs-index.json`` at the user root, the
same sidecar-registry shape as ``.vfs-links.json``) and stays fresh
automatically: writes re-index, deletes unlink, and a content hash makes a
re-index of unchanged bytes a no-op.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from captain_claw import vfs
from captain_claw.config import get_config
from captain_claw.deep_memory import DeepMemoryIndex
from captain_claw.logging import get_logger

log = get_logger(__name__)

_INDEX_FILE = ".vfs-index.json"

# Guardrails for what may be swept into the archive. A VFS project is a working
# folder — it holds build output, images, checkpoints — and indexing it wholesale
# would bury the useful text.
_MAX_BYTES = 2 * 1024 * 1024
_TEXT_SUFFIXES = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".jsonl",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml", ".html", ".htm",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".sh", ".sql", ".log", ".env",
    ".go", ".rs", ".java", ".c", ".h", ".cpp", ".rb", ".php", ".swift",
})
_SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    ".next", ".cache", ".history", ".datastore", ".code",
})

_index: DeepMemoryIndex | None = None
_lock = threading.Lock()

# Connection settings, owned by Flight Deck (Connections → Typesense) and
# persisted in the FD ``system_settings`` table. ``None`` means "not loaded
# yet"; the routes prime it at startup and refresh it on every write. Cached
# because ``get_index()`` is sync while the DB is async.
_settings: dict[str, Any] | None = None

SETTINGS_KEY = "deep_memory.connection"

# The connection is FD's, so it is NOT a fallback to the agent-shaped
# config.yaml block: a half-configured config file silently pointing at the
# wrong Typesense is worse than an honest "not configured".
_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "host": "localhost",
    "port": 8108,
    "protocol": "http",
    "api_key": "",
    "collection_name": "captain_claw_deep_memory",
}


def current_settings() -> dict[str, Any]:
    """Connection settings: FD's stored values, else the config.yaml block.

    config.yaml is the migration path for single-user installs that configured
    deep memory before Flight Deck owned the connection; once anything is saved
    in Connections, that wins.
    """
    if _settings is not None:
        return dict(_settings)
    cfg = getattr(get_config(), "deep_memory", None)
    out = dict(_DEFAULTS)
    if cfg is not None:
        for k in out:
            v = getattr(cfg, k, None)
            if v is not None:
                out[k] = v
    return out


def apply_settings(values: dict[str, Any] | None) -> None:
    """Install new connection settings and drop the cached index."""
    global _settings
    with _lock:
        _settings = dict(values) if values is not None else None
    reset_index()


def configured() -> bool:
    """True when Flight Deck has a Typesense connection it can actually use."""
    s = current_settings()
    return bool(s.get("enabled")) and bool(str(s.get("host") or "").strip())


# ---------------------------------------------------------------------------
# The FD-owned index
# ---------------------------------------------------------------------------


def get_index() -> DeepMemoryIndex | None:
    """Return the process-wide index, or ``None`` when no connection is set up."""
    global _index
    if _index is not None:
        return _index
    with _lock:
        if _index is not None:
            return _index
        if not configured():
            return None
        s = current_settings()
        try:
            from captain_claw.semantic_memory import _build_embedding_chain

            chain = _build_embedding_chain(get_config().memory)
        except Exception as exc:  # pragma: no cover - provider import guard
            log.warning("Deep memory embedding chain unavailable", error=str(exc))
            chain = None
        cfg = getattr(get_config(), "deep_memory", None)
        _index = DeepMemoryIndex(
            host=str(s.get("host") or "localhost"),
            port=int(s.get("port") or 8108),
            protocol=str(s.get("protocol") or "http"),
            api_key=str(s.get("api_key") or ""),
            collection_name=str(s.get("collection_name") or "captain_claw_deep_memory"),
            # Tuning stays in config.yaml — it is not connection detail.
            embedding_dims=int(getattr(cfg, "embedding_dims", 0)) if cfg else 0,
            auto_embed=bool(getattr(cfg, "auto_embed", True)) if cfg else True,
            min_score=float(getattr(cfg, "min_score", 0.12)) if cfg else 0.12,
            embedding_chain=chain,
        )
        return _index


def probe() -> dict[str, Any]:
    """Reach the configured Typesense and report what came back.

    Used by the Connections card's Test button and by the agent-facing error
    path, so a misconfiguration reads the same in both places.
    """
    s = current_settings()
    host = str(s.get("host") or "").strip()
    if not host:
        return {"ok": False, "error": "No Typesense host configured."}
    base = f"{s.get('protocol') or 'http'}://{host}:{int(s.get('port') or 8108)}"
    try:
        import httpx

        r = httpx.get(
            f"{base}/collections",
            headers={"X-TYPESENSE-API-KEY": str(s.get("api_key") or "")},
            timeout=httpx.Timeout(8.0, connect=4.0),
        )
    except Exception as exc:
        return {"ok": False, "error": f"Cannot reach Typesense at {base}: {exc}"}
    if r.status_code == 401:
        return {"ok": False, "error": "Typesense rejected the API key (401)."}
    if r.status_code != 200:
        return {"ok": False, "error": f"Typesense returned {r.status_code}: {r.text[:200]}"}
    try:
        cols = r.json()
        names = [c.get("name") for c in cols]
        target = str(s.get("collection_name") or "")
        hit = next((c for c in cols if c.get("name") == target), None)
    except Exception:
        names, hit = [], None
    return {
        "ok": True,
        "base_url": base,
        "collections": len(names),
        "collection_exists": hit is not None,
        "documents": (hit or {}).get("num_documents", 0),
    }


def reset_index() -> None:
    """Drop the cached index so the next call rebuilds it (tests, config reload)."""
    global _index
    with _lock:
        if _index is not None:
            try:
                _index.close()
            except Exception:
                pass
        _index = None


# ---------------------------------------------------------------------------
# Per-project opt-in registry
# ---------------------------------------------------------------------------


def user_root(owner_id: str) -> Path:
    """A user's VFS root, resolved the way ``vfs_routes`` resolves it.

    There are two independent implementations of the ``<root>/vfs/<user>``
    layout: ``vfs.vfs_base()`` (env cascade, used by agents) and
    ``vfs_routes._user_root()`` (``server.DATA_DIR``, used by the dashboard).
    They agree in a normal deployment, but ``CLAW_VFS_ROOT`` moves only the
    first — and if they ever disagreed, the freshness hooks would index a
    different tree than the one the VFS panel writes to, silently. Since this
    module runs inside Flight Deck, defer to Flight Deck's answer and fall back
    to the agent-side cascade only when the server is not importable.
    """
    try:
        from captain_claw.flight_deck.server import DATA_DIR

        return (DATA_DIR / "vfs" / vfs._sanitize(owner_id or "", fallback="local")).resolve()
    except Exception:
        return vfs.user_root_of(owner_id)


def resolve(owner_id: str, project: str, rel_path: str = "") -> Path | None:
    """Resolve a project-relative path under *owner_id*, or ``None`` if it escapes.

    Mirrors ``vfs.resolve_under`` but anchored on :func:`user_root`, so linked
    folders still resolve to their external target and ``..`` still cannot climb
    out of whichever root actually backs the project.
    """
    root = user_root(owner_id)
    name = vfs._sanitize(project or "", fallback="shared")
    target = vfs.link_target_at(root, name)
    base = (target if target is not None else (root / name)).resolve()
    candidate = base
    for part in (p for p in str(rel_path or "").replace("\\", "/").split("/") if p not in ("", ".")):
        candidate = candidate / part
    candidate = candidate.resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate


def _registry_path(owner_id: str) -> Path:
    return user_root(owner_id) / _INDEX_FILE


def read_registry(owner_id: str) -> dict[str, Any]:
    """Parse ``.vfs-index.json`` for *owner_id*. Never raises."""
    try:
        data = json.loads(_registry_path(owner_id).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_registry(owner_id: str, registry: dict[str, Any]) -> None:
    path = _registry_path(owner_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True))


def indexing_enabled(owner_id: str, project: str) -> bool:
    entry = read_registry(owner_id).get(project)
    return bool(isinstance(entry, dict) and entry.get("enabled"))


def set_indexing(owner_id: str, project: str, enabled: bool) -> dict[str, Any]:
    registry = read_registry(owner_id)
    entry = registry.get(project) if isinstance(registry.get(project), dict) else {}
    entry = dict(entry)
    entry["enabled"] = bool(enabled)
    registry[project] = entry
    write_registry(owner_id, registry)
    return entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def vfs_reference(project: str, rel_path: str) -> str:
    """The canonical archive key for a VFS file: ``vfs:<project>/<rel>``.

    Deliberately the exact string the ``read`` tool accepts, so a search hit is
    a pointer the agent can act on — read the snippet, then open the file — with
    no path translation in between.
    """
    return f"vfs:{project}/{str(rel_path).lstrip('/')}"


def indexable(path: Path) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for whether *path* should be indexed."""
    if not path.is_file():
        return (False, "not a file")
    if any(part in _SKIP_DIRS for part in path.parts):
        return (False, "excluded directory")
    if path.name.startswith("."):
        return (False, "hidden file")
    if path.suffix.lower() not in _TEXT_SUFFIXES:
        return (False, f"unsupported type {path.suffix or '(none)'}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        return (False, f"unreadable: {exc}")
    if size > _MAX_BYTES:
        return (False, f"too large ({size} bytes > {_MAX_BYTES})")
    if size == 0:
        return (False, "empty")
    return (True, "")


def _read_text(path: Path) -> tuple[str, str]:
    """Return ``(text, sha256-of-bytes)``. Hash the *bytes*, not the decoded text,
    so an encoding fallback can never make two different files look identical."""
    raw = path.read_bytes()
    return (raw.decode("utf-8", errors="replace"), hashlib.sha256(raw).hexdigest())


# ---------------------------------------------------------------------------
# Index / refresh / drop
# ---------------------------------------------------------------------------


def index_file(
    owner_id: str,
    project: str,
    rel_path: str,
    *,
    summarize: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Index one VFS file for *owner_id*. Returns a status dict.

    Re-indexing is delete-then-write rather than upsert: chunk ids derive from
    content, so a file that *shrinks* would otherwise leave its old tail chunks
    orphaned in the archive and searchable forever.
    """
    index = get_index()
    if index is None:
        return {"ok": False, "status": "disabled", "reason": "deep memory is not enabled"}

    target = resolve(owner_id, project, rel_path)
    if target is None:
        return {"ok": False, "status": "rejected", "reason": "path escapes the user root"}

    ok, reason = indexable(target)
    if not ok:
        return {"ok": False, "status": "skipped", "reason": reason}

    reference = vfs_reference(project, rel_path)
    try:
        text, digest = _read_text(target)
    except OSError as exc:
        return {"ok": False, "status": "error", "reason": f"read failed: {exc}"}

    if not force and index.stored_hash(reference, owner_id=owner_id) == digest:
        return {"ok": True, "status": "unchanged", "reference": reference, "chunks": 0}

    index.delete_by_reference(reference, owner_id=owner_id)
    chunks = index.index_document(
        doc_id=hashlib.sha1(f"{owner_id}:{reference}".encode()).hexdigest()[:16],
        text=text,
        source="vfs",
        reference=reference,
        path=reference,
        owner_id=owner_id,
        content_hash=digest,
        summarize=summarize,
    )
    log.info(
        "Indexed VFS file into deep memory",
        owner=owner_id, reference=reference, chunks=chunks, summarized=summarize,
    )
    return {"ok": True, "status": "indexed", "reference": reference, "chunks": chunks}


def drop_file(owner_id: str, project: str, rel_path: str) -> dict[str, Any]:
    """Remove a file's chunks from the archive (used on unlink/rename)."""
    index = get_index()
    if index is None:
        return {"ok": False, "status": "disabled", "deleted": 0}
    reference = vfs_reference(project, rel_path)
    deleted = index.delete_by_reference(reference, owner_id=owner_id)
    if deleted:
        log.info("Dropped VFS file from deep memory",
                 owner=owner_id, reference=reference, chunks=deleted)
    return {"ok": True, "status": "dropped", "reference": reference, "deleted": deleted}


def drop_prefix(owner_id: str, project: str, rel_path: str = "") -> dict[str, Any]:
    """Drop a whole directory (or project) from the archive.

    Typesense has no prefix operator in ``filter_by``, so this pages the
    references under the prefix and deletes them individually.
    """
    index = get_index()
    if index is None:
        return {"ok": False, "status": "disabled", "deleted": 0}
    prefix = vfs_reference(project, rel_path).rstrip("/")
    deleted = 0
    try:
        resp = index._get_client().get(  # noqa: SLF001 - same-subsystem access
            f"{index._base_url}/collections/{index._collection_name}/documents/export",
            params={"include_fields": "reference,owner_id"},
        )
        resp.raise_for_status()
        seen: set[str] = set()
        for line in resp.text.splitlines():
            if not line.strip():
                continue
            doc = json.loads(line)
            if doc.get("owner_id") != owner_id:
                continue
            ref = str(doc.get("reference") or "")
            if ref in seen:
                continue
            if ref == prefix or ref.startswith(prefix + "/"):
                seen.add(ref)
        for ref in seen:
            deleted += index.delete_by_reference(ref, owner_id=owner_id)
    except Exception as exc:
        log.warning("Deep memory prefix drop failed", prefix=prefix, error=str(exc))
        return {"ok": False, "status": "error", "reason": str(exc), "deleted": deleted}
    return {"ok": True, "status": "dropped", "prefix": prefix, "deleted": deleted}


def index_project(
    owner_id: str, project: str, *, summarize: bool = False, force: bool = False
) -> dict[str, Any]:
    """Walk a project and index everything eligible. Returns a per-status tally."""
    root = resolve(owner_id, project)
    if root is None or not root.is_dir():
        return {"ok": False, "reason": "project not found"}
    tally: dict[str, int] = {}
    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        ok, _reason = indexable(path)
        if not ok:
            continue
        rel = path.relative_to(root).as_posix()
        result = index_file(owner_id, project, rel, summarize=summarize, force=force)
        status = str(result.get("status", "error"))
        tally[status] = tally.get(status, 0) + 1
        if status == "indexed":
            files.append(rel)
    return {"ok": True, "project": project, "tally": tally, "indexed": files}


# ---------------------------------------------------------------------------
# Automatic freshness — called from the VFS routes
# ---------------------------------------------------------------------------


def on_write(owner_id: str, project: str, rel_path: str) -> None:
    """Re-index after a write. No-op unless the project opted in.

    Best-effort by design: deep memory being unreachable must never turn a
    successful file write into a failed HTTP request.
    """
    if not indexing_enabled(owner_id, project):
        return
    try:
        index_file(owner_id, project, rel_path,
                   summarize=False, force=False)
    except Exception as exc:
        log.warning("Deep memory re-index after write failed",
                    owner=owner_id, project=project, path=rel_path, error=str(exc))


def on_delete(owner_id: str, project: str, rel_path: str, *, is_dir: bool = False) -> None:
    """Unlink from the archive after a delete. Best-effort, same rationale."""
    if not indexing_enabled(owner_id, project):
        return
    try:
        if is_dir:
            drop_prefix(owner_id, project, rel_path)
        else:
            drop_file(owner_id, project, rel_path)
    except Exception as exc:
        log.warning("Deep memory drop after delete failed",
                    owner=owner_id, project=project, path=rel_path, error=str(exc))


def on_rename(owner_id: str, project: str, old_rel: str, new_rel: str) -> None:
    """A rename is a drop of the old reference plus an index of the new one."""
    if not indexing_enabled(owner_id, project):
        return
    try:
        drop_file(owner_id, project, old_rel)
        index_file(owner_id, project, new_rel, summarize=False, force=True)
    except Exception as exc:
        log.warning("Deep memory rename fixup failed",
                    owner=owner_id, project=project, error=str(exc))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search(
    owner_id: str, query: str, *, max_results: int = 10, filter_by: str = ""
) -> list[dict[str, Any]]:
    """Owner-scoped search. The tenant filter is applied inside ``DeepMemoryIndex``
    and ANDed onto *filter_by*, so a caller cannot widen its own scope."""
    index = get_index()
    if index is None:
        return []
    results = index.search(
        query, max_results=max_results, filter_by=filter_by, owner_id=owner_id
    )
    return [
        {
            "reference": r.reference,
            "source": r.source,
            "score": round(r.score, 4),
            "snippet": r.snippet,
            "summary": r.text_l2 or r.text_l1 or "",
            "chunk_index": r.chunk_index,
            "start_line": r.start_line,
            "end_line": r.end_line,
            "updated_at": r.updated_at,
        }
        for r in results
    ]
