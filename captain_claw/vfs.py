"""Shared virtual filesystem (VFS) for cross-agent collaboration.

A persistent, real-on-disk file tree that any agent can read, write, edit,
glob and grep — addressed with a ``vfs:<project>/<path>`` URI scheme. Unlike
the per-agent ``saved/`` hierarchy (showcase/tmp/scripts, scoped by session
slug), the VFS is **shared** between agents and **permanent** across runs, so
several agents working the same task keep a common file context.

Layout on disk::

    <fd-data>/vfs/<fd-user-id>/<project>/...

* ``<fd-data>`` — the Flight Deck data dir (``FD_DATA_DIR`` or ``./fd-data``).
* ``<fd-user-id>`` — the owning Flight Deck user (``FD_OWNER_ID``); isolates
  one user's shared tree from another's. Falls back to ``_local`` when an
  agent runs outside Flight Deck.
* ``<project>`` — a namespace shared by collaborating agents. Multi-agent
  runs (Council / Basna / Vatra) auto-bind one project via
  ``CLAW_VFS_PROJECT`` so co-spawned agents land in the same tree with zero
  coordination; agents can also address any project explicitly.

Path scheme (case-insensitive prefix)::

    vfs:myproj/src/main.py   -> <user>/myproj/src/main.py
    vfs:myproj               -> <user>/myproj          (the project root)
    vfs:/notes.md            -> <user>/<default>/notes.md
    vfs:myproj/**/*.py       -> glob within a project

All paths are sandboxed under the user root — ``..`` traversal that would
escape it is rejected.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_SCHEME = "vfs:"
# Per-project sidecar recording who wrote what (append-only → concurrency-safe;
# many agents writing the same project never race a read-modify-write). Hidden
# from listings. One JSON object per line: {"path", "agent", "ts"}.
_META_FILE = ".vfs-meta.jsonl"
META_FILENAME = _META_FILE  # public: the listing layer hides this sidecar
_DEFAULT_PROJECT = "shared"
# Must match Flight Deck's no-auth user id (auth._LOCAL_USER["id"] == "local")
# so the main interactive agent and the FD browser panel share one tree.
_DEFAULT_USER = "local"


# ──────────────────────────────────────────────────────────────────────
# Identity / location
# ──────────────────────────────────────────────────────────────────────

def _sanitize(segment: str, *, fallback: str) -> str:
    """Make a single path segment filesystem-safe (no separators, no ``..``)."""
    value = (segment or "").strip()
    if not value or value in (".", ".."):
        return fallback
    safe = "".join(c if (c.isalnum() or c in "._-") else "-" for c in value)
    safe = safe.strip("-._") or fallback
    # A segment of all dots collapsed to empty would re-introduce traversal.
    return safe if safe not in (".", "..") else fallback


def _find_fd_data_ancestor() -> Path | None:
    """Walk up from CWD looking for an ``fd-data`` directory.

    Spawned agents run with a CWD like ``<repo>/fd-data/<slug>/data/workspace``
    where ``fd-data`` is an ancestor — this lets them locate the shared tree
    without every spawn site having to inject ``CLAW_VFS_ROOT``.
    """
    here = Path.cwd().resolve()
    for cand in (here, *here.parents):
        if cand.name == "fd-data":
            return cand
        child = cand / "fd-data"
        if child.is_dir():
            return child
    return None


def vfs_base() -> Path:
    """Return the ``<fd-data>/vfs`` root directory (not user-scoped)."""
    # 1. Explicit override (injected by Flight Deck at spawn).
    root = os.environ.get("CLAW_VFS_ROOT", "").strip()
    if root:
        return Path(root).expanduser().resolve()

    # 2. Flight Deck data dir, if the agent inherited it.
    data_dir = os.environ.get("FD_DATA_DIR", "").strip()
    if data_dir:
        return (Path(data_dir).expanduser().resolve() / "vfs")

    # 3. Walk up to an fd-data ancestor (covers local multi-agent spawns).
    ancestor = _find_fd_data_ancestor()
    if ancestor is not None:
        return (ancestor / "vfs").resolve()

    # 4. Standalone (PyInstaller) build default, else repo-local ./fd-data.
    if getattr(sys, "_MEIPASS", None):
        return (Path.home() / ".captain-claw" / "fd-data" / "vfs").resolve()
    return (Path.cwd() / "fd-data" / "vfs").resolve()


def vfs_user() -> str:
    """Return the sanitized owning-user segment.

    Resolution order:
    1. ``CLAW_VFS_USER`` — explicit override; set this when launching the main
       interactive agent against an auth-enabled Flight Deck so it writes under
       the same user id the panel reads (the logged-in user's UUID).
    2. ``FD_OWNER_ID`` — injected by Flight Deck into spawned agents.
    3. ``local`` — matches FD's no-auth user, so the standalone main agent and
       the panel agree out of the box.
    """
    explicit = os.environ.get("CLAW_VFS_USER", "").strip()
    if explicit:
        return _sanitize(explicit, fallback=_DEFAULT_USER)
    return _sanitize(os.environ.get("FD_OWNER_ID", ""), fallback=_DEFAULT_USER)


def default_project() -> str:
    """Return the auto-bound project for this run (or the shared default)."""
    return _sanitize(os.environ.get("CLAW_VFS_PROJECT", ""), fallback=_DEFAULT_PROJECT)


def user_root(*, create: bool = False) -> Path:
    """Return ``<fd-data>/vfs/<user>`` — the root all this user's projects share."""
    root = (vfs_base() / vfs_user()).resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def project_root(project: str = "", *, create: bool = False) -> Path:
    """Return ``<user_root>/<project>``."""
    proj = _sanitize(project or default_project(), fallback=_DEFAULT_PROJECT)
    root = (user_root() / proj).resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


# ──────────────────────────────────────────────────────────────────────
# Scheme parsing / resolution
# ──────────────────────────────────────────────────────────────────────

def is_vfs_path(path: str | None) -> bool:
    """Whether *path* uses the ``vfs:`` scheme (case-insensitive)."""
    return bool(path) and str(path).strip().lower().startswith(_SCHEME)


def split_scheme(path: str) -> tuple[str, str]:
    """Split a ``vfs:`` path into ``(project, relative_path)``.

    The project defaults to :func:`default_project` when omitted
    (``vfs:/foo`` or ``vfs://foo``). Backslashes are normalised to ``/``.
    """
    raw = str(path).strip()
    remainder = raw[len(_SCHEME):] if raw[: len(_SCHEME)].lower() == _SCHEME else raw
    remainder = remainder.replace("\\", "/")
    # ``vfs://proj/..`` — drop the doubled slash, next segment is the project.
    if remainder.startswith("//"):
        remainder = remainder.lstrip("/")
        first_is_default = False
    else:
        first_is_default = remainder.startswith("/")
    remainder = remainder.lstrip("/")

    if first_is_default or "/" not in remainder:
        if first_is_default:
            return default_project(), remainder
        return (remainder or default_project()), ""
    project, rel = remainder.split("/", 1)
    return (project or default_project()), rel


def resolve_vfs_path(path: str, *, create_parents: bool = False) -> Path | None:
    """Resolve a ``vfs:`` path to an absolute on-disk path.

    Returns ``None`` if the path is not a vfs path or would escape the
    user root via ``..`` traversal. When *create_parents* is set, the
    parent directory is created (used by writers).
    """
    if not is_vfs_path(path):
        return None
    project, rel = split_scheme(path)
    base = project_root(project)

    rel_parts = [p for p in rel.replace("\\", "/").split("/") if p not in ("", ".")]
    candidate = base
    for part in rel_parts:
        candidate = candidate / part
    candidate = candidate.resolve()

    # Sandbox: never allow escaping the user root.
    root = user_root().resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None

    if create_parents:
        candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def resolve_under(user_id: str, default_proj: str, path: str) -> Path | None:
    """Resolve a vfs/bare *path* to an on-disk file for an EXPLICIT user + project,
    independent of this process's ambient env.

    :func:`resolve_vfs_path` keys off ``FD_OWNER_ID``/``CLAW_VFS_PROJECT`` in the
    environment — correct inside a spawned worker, but wrong in the Flight Deck
    server process, which serves many users. Endpoints that act on behalf of a
    run's owner use this to read the same files a worker wrote: pass the run's
    owner id and shared project explicitly. A path that already names a project
    (``vfs:proj/file`` or ``proj/file``) uses that; a bare ``file`` falls back to
    *default_proj*. Returns ``None`` on an empty path or one that would escape the
    user's root.
    """
    raw = str(path or "").strip().replace("\\", "/")
    if not raw:
        return None
    if raw[: len(_SCHEME)].lower() == _SCHEME:
        raw = raw[len(_SCHEME):]
    raw = raw.lstrip("/")
    if "/" in raw:
        proj, rel = raw.split("/", 1)
    else:
        proj, rel = (default_proj or ""), raw
    user_root_p = (vfs_base() / _sanitize(user_id or "", fallback=_DEFAULT_USER)).resolve()
    candidate = user_root_p / _sanitize(proj or default_proj, fallback=_DEFAULT_PROJECT)
    # Keep ``..`` so ``.resolve()`` collapses it and the sandbox check below can
    # reject any path that tries to climb out of the user root.
    for part in (p for p in rel.split("/") if p not in ("", ".")):
        candidate = candidate / part
    candidate = candidate.resolve()
    try:
        candidate.relative_to(user_root_p)
    except ValueError:
        return None
    return candidate


def to_display(path: Path) -> str:
    """Render an absolute VFS path back as a ``vfs:<project>/...`` URI."""
    try:
        rel = Path(path).resolve().relative_to(user_root().resolve())
    except ValueError:
        return str(path)
    parts = rel.parts
    if not parts:
        return _SCHEME
    project, *rest = parts
    return f"{_SCHEME}{project}" + ("/" + "/".join(rest) if rest else "")


# ──────────────────────────────────────────────────────────────────────
# Authorship — who wrote each file (best-effort; never raises)
# ──────────────────────────────────────────────────────────────────────

def agent_label() -> str:
    """A human label for the agent writing right now, from its environment.

    ``CLAW_AGENT_LABEL`` is set by the Basna/Vatra spawn paths to the worker's
    role; ``CLAW_VATRA_OWNER`` (the owning archetype) is the Vatra fallback.
    Empty when unknown (e.g. the main interactive agent) — we don't record noise.
    """
    for key in ("CLAW_AGENT_LABEL", "CLAW_VATRA_OWNER"):
        v = os.environ.get(key, "").strip()
        if v:
            return v
    return ""


def record_author(real_path: str | Path) -> None:
    """Append an authorship record for a freshly-written VFS file (best-effort).

    Stamps the current :func:`agent_label` against the file's project-relative
    path in the project's ``.vfs-meta.jsonl``. No-op when the writer is unknown
    or the path is outside a VFS project.
    """
    try:
        label = agent_label()
        if not label:
            return
        root = user_root().resolve()
        parts = Path(real_path).resolve().relative_to(root).parts
        if len(parts) < 2:  # need <project>/<at least one segment>
            return
        proj_root = root / parts[0]
        rel = "/".join(parts[1:])
        line = json.dumps({"path": rel, "agent": label, "ts": time.time()}, ensure_ascii=False)
        with open(proj_root / _META_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001 — authorship is best-effort, never break a write
        pass


def read_authors(proj_root: str | Path) -> dict[str, dict]:
    """Map ``project-relative path -> {agent, ts}`` from a project's sidecar.

    The last record for a path wins (most recent writer). Returns ``{}`` when no
    sidecar exists.
    """
    out: dict[str, dict] = {}
    meta = Path(proj_root) / _META_FILE
    if not meta.is_file():
        return out
    try:
        for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = d.get("path")
            if p:
                out[p] = {"agent": d.get("agent", ""), "ts": d.get("ts", 0)}
    except OSError:
        pass
    return out


def list_projects() -> list[str]:
    """List existing project namespaces for the current user."""
    root = user_root()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


# ──────────────────────────────────────────────────────────────────────
# Server-side helpers (explicit user id; used by the Flight Deck routes)
# ──────────────────────────────────────────────────────────────────────

def safe_name(segment: str, *, fallback: str = "_") -> str:
    """Public wrapper around segment sanitisation (no separators / traversal)."""
    return _sanitize(segment, fallback=fallback)


def safe_join(root: Path, rel: str) -> Path | None:
    """Join *rel* under *root*, returning ``None`` if it escapes *root*.

    Used by the Flight Deck VFS browser, which already knows the user/project
    root and just needs to safely descend into a relative sub-path.
    """
    base = Path(root).resolve()
    cand = base
    for part in str(rel or "").replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        cand = cand / part
    cand = cand.resolve()
    try:
        cand.relative_to(base)
    except ValueError:
        return None
    return cand
