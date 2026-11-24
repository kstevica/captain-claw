"""File library — every file the caller's agents made, summarised and sorted.

The kiosk's "Library": one place for the files scattered across a user's
agents. Flight Deck walks each of the caller's process agents' workspaces on
disk (``<fd-data>/<slug>/data/workspace`` — the same tree the Files panel
merges in, so stopped agents count too), and for every new or changed file
reads its text (PDF / DOCX / XLSX / PPTX through the agent's own extractors)
and asks the owner's fast tier for a title, a short summary, one category
from a fixed list and a few tags. Pictures are shown to a model that can see
when their agent's vision is on (the owner's tier if it sees, else the
agent's own model), and recordings and videos are transcribed with Soniox
(the sound taken out with ffmpeg first) and summarised from the transcript;
a file left for want of one of these is taken up again once it is there.
Results live in ``file_library.db`` beside the agents' folders, keyed by
agent + workspace path, and are re-summarised only when a file's size or
mtime changes.

Scans run in the background, one per user at a time, started when the
Library opens (or by its Rescan button); the list is searchable (filename,
title, summary, tags, the text itself, a picture's description and what is
said in a recording) and filterable by agent, category
and kind. Tokens count toward the owner's daily allowance, and a scan stops
when it is spent. Workers Flight Deck runs itself (a run's agents, a being's
body) are not the user's library and are left out. The pictures the browser
tool saves while it works are listed apart, never summarised (see "Agentic
screenshots" below).

The user's own shared folders (``vfs:<folder>/``) can be sources too, beside
the agents, but only what the person added: a whole folder, or single files
(a run's folder holds batches, scripts and notes, and summarising all of it
would bury what matters). Those are walked, summarised and searched the same
way (see "Shared folders as sources" below).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import io
import json
import mimetypes
import os
import re
import shutil
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response

from captain_claw.flight_deck.auth import get_current_user
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/file-library", tags=["file-library"])

# The categories a file is sorted into: (id, label). The model picks exactly
# one; anything else lands in "other".
CATEGORIES: list[tuple[str, str]] = [
    ("report", "Report"),
    ("research", "Research"),
    ("notes", "Notes"),
    ("plan", "Plan"),
    ("data", "Data & lists"),
    ("finance", "Finance"),
    ("legal", "Legal"),
    ("correspondence", "Correspondence"),
    ("presentation", "Presentation"),
    ("code", "Code"),
    ("media", "Images & media"),
    ("other", "Other"),
]
_CATEGORY_IDS = {c for c, _ in CATEGORIES}

_KIND_BY_EXT: dict[str, str] = {}
for _kind, _exts in {
    "document": (".txt", ".md", ".markdown", ".rst", ".pdf", ".doc", ".docx", ".odt", ".rtf"),
    "spreadsheet": (".csv", ".tsv", ".xls", ".xlsx", ".ods"),
    "presentation": (".ppt", ".pptx", ".odp"),
    "web": (".html", ".htm"),
    "data": (".json", ".jsonl", ".yaml", ".yml", ".toml", ".xml"),
    "code": (".py", ".js", ".ts", ".jsx", ".tsx", ".sh", ".bash", ".sql", ".rb", ".go",
             ".rs", ".java", ".c", ".cpp", ".h", ".css", ".scss"),
    "image": (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".ico"),
    "media": (".mp3", ".wav", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".flac", ".amr",
              ".mp4", ".webm", ".mov", ".m4v", ".mkv", ".avi", ".3gp"),
    "archive": (".zip", ".tar", ".gz"),
    "config": (".env", ".ini", ".cfg", ".conf", ".log"),
}.items():
    for _e in _exts:
        _KIND_BY_EXT[_e] = _kind

KINDS: list[tuple[str, str]] = [
    ("document", "Documents"), ("spreadsheet", "Spreadsheets"), ("presentation", "Presentations"),
    ("web", "Web pages"), ("data", "Data files"), ("code", "Code"), ("image", "Images"),
    ("media", "Audio & video"), ("archive", "Archives"), ("config", "Config & logs"),
]

# Never descended into while walking a workspace (the agent's own list, the
# stills video_vision cuts out of a video while it watches one, and the
# pictures web_fetch keeps for reports — material, one vision call each).
_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__",
              "dist", "build", ".datastore", ".history", ".reports", "video_frames",
              "web-images"}
# Files per agent a walk lists at most — a runaway workspace can't stall a scan.
_MAX_FILES_PER_AGENT = 3000
# Files a single scan summarises at most; the next scan picks up the rest.
_MAX_SUMMARIES_PER_SCAN = 400
_CONCURRENCY = 3
# What the model reads of a file, and what search keeps of its text.
_LLM_CHARS = 7000
_EXCERPT_CHARS = 4000
_MAX_READ_BYTES = 400_000
_MAX_FILE_BYTES = 30 * 1024 * 1024
# Consecutive model failures after which a scan gives up (no key, no network).
_MAX_FAILURE_STREAK = 4
# A file whose summary failed is tried again by the scan that opening the
# library starts once this long has passed (Rescan retries it right away).
_ERROR_RETRY_SECONDS = 6 * 3600
_MAX_TOKENS = 700

# Pictures a model looks at (an SVG is read as text instead; an icon isn't
# worth a look).
_PICTURE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_MAX_PICTURE_BYTES = 40 * 1024 * 1024
_MAX_PICTURE_PIXELS = 40_000_000
# A WebP decodes whole through two full-size buffers (no draft): a lower cap.
_MAX_WEBP_PIXELS = 16_000_000
_PICTURE_EDGE = 1280
# What Pillow may open a picture as — never by guessing from its bytes (a
# ".png" holding PostScript would run Ghostscript).
_PICTURE_FORMATS = ("PNG", "JPEG", "GIF", "WEBP", "BMP")
# One picture decoded at a time: a large one takes a few hundred MB (a
# WebP at its cap most, about 0.7 GB).
_PICTURE_DECODE_LOCK = threading.Lock()
# A picture's reply carries a long description (and text read off it, which
# in CJK runs to many tokens).
_PICTURE_MAX_TOKENS = 3000
# Recordings and videos: what is said in them is transcribed with Soniox.
_VIDEO_EXTS = {".mp4", ".webm", ".mov", ".m4v", ".mkv", ".avi", ".3gp"}
_SOUND_EXTS = {".mp3", ".wav", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".flac", ".amr"} | _VIDEO_EXTS
# What Soniox reads straight from the file when there is no ffmpeg to take
# the sound out first (and how big a file it is sent whole).
_SONIOX_READS_EXTS = {".mp3", ".wav", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".flac", ".amr",
                      ".mp4", ".webm"}
_MAX_SOUND_UPLOAD_BYTES = 200 * 1024 * 1024
# Soniox's limit for one file.
_MAX_SPEECH_SECONDS = 300 * 60
# Seconds to wait for a transcript (Soniox is polled once a second).
_SONIOX_POLL = 900
# Transcriptions at a time (beside the text summaries, never in their way).
_SOUND_CONCURRENCY = 2
# Soniox failures in a row after which a scan leaves the rest for the next.
_MAX_SONIOX_FAILURES = 3
# Soniox statuses that are about the file, not the service: a job that
# couldn't process it (200), or an upload it won't take.
_SONIOX_FILE_ERRORS = {200, 400, 413, 415, 422}
# What search keeps of a transcript or a picture's description.
_TRANSCRIPT_CHARS = 50_000
# The agent's own spoken replies (the audio tool's and Pocket TTS's): it wrote
# the words itself, so they are not sent to be transcribed.
_AGENT_VOICE_RE = re.compile(r"^(speech|pocket-tts)-\d{8}-\d{6}\b")

# Text read straight from disk; the office formats go through the extractors.
_PLAIN_TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".jsonl", ".yaml",
    ".yml", ".toml", ".xml", ".html", ".htm", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".sh", ".bash", ".sql", ".rb", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".css",
    ".scss", ".ini", ".cfg", ".conf", ".log", ".svg",
}
_EXTRACTED_EXTS = {".pdf", ".docx", ".xlsx", ".pptx"}
# Listed, never read: secrets, binaries no extractor opens.
_NOT_READ_REASON = {
    ".env": "Environment file — may hold secrets, so it is never read.",
}


def kind_of(ext: str) -> str:
    return _KIND_BY_EXT.get(ext.lower(), "other")


# ── Agentic screenshots ───────────────────────────────────────────────
#
# The pictures the browser tool saves while it works: browser_screenshot_<ms>
# (tools/browser.py), browser_chrome_<ms> and browser_zoom_<ms> (the person's
# Chrome, tools/browser_chrome.py and browser_page.py), each a .jpg in the
# session's saved/media/<session>/ folder. They are the agent's working
# material, not the person's files, and a busy agent takes hundreds: they stay
# out of the library's lists, counts and search, are listed on their own
# ("Agentic screenshots"), and are never shown to a model (a vision call each).
# Only that exact name in that exact place counts: a file that merely says
# "browser", or a screenshot an agent saved somewhere else on purpose, stays a
# library file.
SCREENSHOTS = "screenshots"
_AGENTIC_SHOT_RE = re.compile(r"^saved/media/[^/]+/browser_[a-z]+_\d{10,}\.(?:jpe?g|png|webp)$")
_SHOT_TITLES = {"screenshot": "Browser screenshot", "chrome": "Chrome screenshot", "zoom": "Browser close-up"}
_SHOT_SUMMARY = "A screenshot the browser tool took while it worked. Not summarised."


def is_agentic_screenshot(rel_path: str) -> bool:
    """Whether a library file (by its workspace path) is a picture the browser tool saved."""
    return bool(_AGENTIC_SHOT_RE.match(str(rel_path or "")))


def _screenshot_title(filename: str) -> str:
    kind = filename.split("_", 2)[1] if filename.count("_") >= 2 else ""
    return _SHOT_TITLES.get(kind, "Browser screenshot")


# ── Store ─────────────────────────────────────────────────────────────


def _data_dir() -> Path:
    from captain_claw.flight_deck.server import DATA_DIR
    return DATA_DIR


class LibraryStore:
    """``file_library.db``: one row per file, keyed by agent + workspace path."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # The browser tool's screenshots, told apart in SQL by the same rule.
        self._conn.create_function("agentic_shot", 1, is_agentic_screenshot, deterministic=True)
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS library_files (
                    id            TEXT PRIMARY KEY,
                    agent_slug    TEXT NOT NULL,
                    rel_path      TEXT NOT NULL,
                    filename      TEXT NOT NULL,
                    extension     TEXT NOT NULL DEFAULT '',
                    kind          TEXT NOT NULL DEFAULT 'other',
                    size          INTEGER NOT NULL DEFAULT 0,
                    modified      REAL NOT NULL DEFAULT 0,
                    fingerprint   TEXT NOT NULL DEFAULT '',
                    status        TEXT NOT NULL DEFAULT 'pending',
                    title         TEXT NOT NULL DEFAULT '',
                    summary       TEXT NOT NULL DEFAULT '',
                    category      TEXT NOT NULL DEFAULT '',
                    tags          TEXT NOT NULL DEFAULT '[]',
                    excerpt       TEXT NOT NULL DEFAULT '',
                    error         TEXT NOT NULL DEFAULT '',
                    model         TEXT NOT NULL DEFAULT '',
                    tokens        INTEGER NOT NULL DEFAULT 0,
                    summarized_at REAL,
                    added_at      REAL NOT NULL DEFAULT 0,
                    waiting_for   TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_library_agent ON library_files(agent_slug, status);
                -- What of a person's shared folders is in the library: a whole
                -- folder (path ''), or one file in it.
                CREATE TABLE IF NOT EXISTS library_picks (
                    owner     TEXT NOT NULL,
                    folder    TEXT NOT NULL,
                    path      TEXT NOT NULL DEFAULT '',
                    added_at  REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (owner, folder, path)
                );
                """
            )
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(library_files)")}
            if "waiting_for" not in cols:
                # A library from before pictures and recordings were read:
                # the ones it set aside unread are read now.
                self._conn.execute("ALTER TABLE library_files ADD COLUMN waiting_for TEXT NOT NULL DEFAULT ''")
                self._conn.execute(
                    "UPDATE library_files SET status = 'pending', summary = '' "
                    "WHERE status = 'skipped' AND kind IN ('image', 'media')")
            self._conn.commit()

    @staticmethod
    def file_id(slug: str, rel: str) -> str:
        return hashlib.sha1(f"{slug}\0{rel}".encode()).hexdigest()[:20]

    def sync_agent(self, slug: str, files: list[dict]) -> int:
        """Bring *slug*'s rows in line with what is on disk now: new files and
        changed ones (size/mtime) become pending, vanished ones go. Returns
        how many are pending for the agent afterwards."""
        now = time.time()
        with self._lock:
            have = {r["rel_path"]: r for r in self._conn.execute(
                "SELECT rel_path, fingerprint, status FROM library_files WHERE agent_slug = ?", (slug,))}
            seen: set[str] = set()
            for f in files:
                rel = f["rel"]
                seen.add(rel)
                old = have.get(rel)
                if old is None:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO library_files (id, agent_slug, rel_path, filename, extension, "
                        "kind, size, modified, fingerprint, status, added_at) VALUES (?,?,?,?,?,?,?,?,?, 'pending', ?)",
                        (self.file_id(slug, rel), slug, rel, f["filename"], f["extension"], f["kind"],
                         f["size"], f["modified"], f["fingerprint"], now))
                elif old["fingerprint"] != f["fingerprint"]:
                    # Changed: the old summary shows until the new one is in.
                    self._conn.execute(
                        "UPDATE library_files SET size = ?, modified = ?, fingerprint = ?, status = 'pending', "
                        "error = '', waiting_for = '' WHERE agent_slug = ? AND rel_path = ?",
                        (f["size"], f["modified"], f["fingerprint"], slug, rel))
            gone = [rel for rel in have if rel not in seen]
            for i in range(0, len(gone), 500):
                chunk = gone[i:i + 500]
                self._conn.execute(
                    f"DELETE FROM library_files WHERE agent_slug = ? AND rel_path IN ({','.join('?' * len(chunk))})",
                    (slug, *chunk))
            self._conn.commit()
            row = self._conn.execute(
                "SELECT COUNT(*) FROM library_files WHERE agent_slug = ? AND status = 'pending'", (slug,)).fetchone()
            return int(row[0])

    def drop_agents_except(self, slugs: set[str]) -> None:
        """Forget the files of agents Flight Deck no longer has at all
        (shared folders are kept — :meth:`drop_folders_except` sees to them)."""
        with self._lock:
            known = [r[0] for r in self._conn.execute("SELECT DISTINCT agent_slug FROM library_files")]
            for slug in known:
                if slug not in slugs and not slug.startswith(FOLDER_PREFIX):
                    self._conn.execute("DELETE FROM library_files WHERE agent_slug = ?", (slug,))
            self._conn.commit()

    def drop_folders_except(self, owner: str, keys: set[str]) -> None:
        """Forget the files of *owner*'s shared folders that are gone."""
        with self._lock:
            known = [r[0] for r in self._conn.execute("SELECT DISTINCT agent_slug FROM library_files")]
            for slug in known:
                folder = folder_of(slug)
                if folder and folder[0] == owner and slug not in keys:
                    self._conn.execute("DELETE FROM library_files WHERE agent_slug = ?", (slug,))
            self._conn.commit()

    # ── what of the shared folders is in ──

    def picks(self, owner: str) -> dict[str, set[str]]:
        """``{folder: {path, …}}`` of *owner*'s shared folders; ``''`` is the whole folder."""
        out: dict[str, set[str]] = {}
        with self._lock:
            for folder, path in self._conn.execute(
                    "SELECT folder, path FROM library_picks WHERE owner = ?", (owner,)):
                out.setdefault(folder, set()).add(path)
        return out

    def add_pick(self, owner: str, folder: str, path: str = "") -> bool:
        """True when it wasn't in already."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO library_picks (owner, folder, path, added_at) VALUES (?, ?, ?, ?)",
                (owner, folder, path, time.time()))
            self._conn.commit()
            return bool(cur.rowcount)

    def drop_pick(self, owner: str, folder: str, path: str = "") -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM library_picks WHERE owner = ? AND folder = ? AND path = ?", (owner, folder, path))
            self._conn.commit()
            return bool(cur.rowcount)

    def drop_folder_picks(self, owner: str, folders: set[str]) -> None:
        """Forget what was added from folders that are gone."""
        with self._lock:
            for folder in folders:
                self._conn.execute("DELETE FROM library_picks WHERE owner = ? AND folder = ?", (owner, folder))
            self._conn.commit()

    def drop_agent(self, slug: str) -> None:
        """Forget one agent's files: it was removed for good."""
        with self._lock:
            self._conn.execute("DELETE FROM library_files WHERE agent_slug = ?", (slug,))
            self._conn.commit()

    def pending(self, slugs: list[str], limit: int, retry_errors_before: float) -> list[dict]:
        """Files waiting for a summary, newest first — with the failed ones
        last tried before *retry_errors_before*."""
        if not slugs:
            return []
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM library_files WHERE agent_slug IN ({','.join('?' * len(slugs))}) "
                "AND (status = 'pending' OR (status = 'error' AND COALESCE(summarized_at, 0) < ?)) "
                "ORDER BY modified DESC LIMIT ?", (*slugs, retry_errors_before, limit)).fetchall()
        return [dict(r) for r in rows]

    def set_version(self, file_id: str, size: int, modified: float, fingerprint: str) -> None:
        """A file seen changed outside a walk: the version to summarise (as
        the walk would have marked it)."""
        with self._lock:
            self._conn.execute(
                "UPDATE library_files SET size = ?, modified = ?, fingerprint = ?, status = 'pending', "
                "error = '', waiting_for = '' WHERE id = ?", (size, modified, fingerprint, file_id))
            self._conn.commit()

    def requeue_waiting(self, need: str, slugs: list[str]) -> int:
        """Files of these agents set aside for want of *need* ("vision",
        "soniox", "ffmpeg") go back in the queue: it is there now."""
        if not slugs:
            return 0
        with self._lock:
            cur = self._conn.execute(
                "UPDATE library_files SET status = 'pending', waiting_for = '', summary = '' "
                f"WHERE status = 'skipped' AND waiting_for = ? AND agent_slug IN ({','.join('?' * len(slugs))})",
                (need, *slugs))
            self._conn.commit()
            return int(cur.rowcount or 0)

    def get(self, file_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM library_files WHERE id = ?", (file_id,)).fetchone()
        return dict(row) if row else None

    def save_result(self, file_id: str, fingerprint: str, **fields: Any) -> None:
        """Store a summary — unless the file changed again meanwhile (its next
        scan summarises the new version)."""
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._conn.execute(
                f"UPDATE library_files SET {cols} WHERE id = ? AND fingerprint = ?",
                (*fields.values(), file_id, fingerprint))
            self._conn.commit()

    def settle_screenshots(self, slugs: list[str]) -> int:
        """The browser tool's screenshots waiting for a summary (or set aside
        for want of one) are filed as they are: titled from their name, never
        shown to a model. One described on request keeps its description."""
        if not slugs:
            return 0
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, filename FROM library_files WHERE agent_slug IN ({','.join('?' * len(slugs))}) "
                "AND (status IN ('pending', 'error') OR waiting_for != '') AND agentic_shot(rel_path)",
                slugs).fetchall()
            for r in rows:
                self._conn.execute(
                    "UPDATE library_files SET status = 'skipped', title = ?, summary = ?, category = 'media', "
                    "tags = '[]', excerpt = '', error = '', model = '', tokens = 0, summarized_at = ?, "
                    "waiting_for = '' WHERE id = ?",
                    (_screenshot_title(r["filename"]), _SHOT_SUMMARY, now, r["id"]))
            self._conn.commit()
        return len(rows)

    def counts(self, slugs: list[str]) -> dict[str, int]:
        """The library's files by status (the browser tool's screenshots aside)."""
        out = {"total": 0, "done": 0, "pending": 0, "error": 0, "skipped": 0}
        if not slugs:
            return out
        with self._lock:
            for status, n in self._conn.execute(
                    f"SELECT status, COUNT(*) FROM library_files WHERE agent_slug IN ({','.join('?' * len(slugs))}) "
                    "AND NOT agentic_shot(rel_path) GROUP BY status", slugs):
                out[status] = out.get(status, 0) + int(n)
                out["total"] += int(n)
        return out

    def search(self, slugs: list[str], terms: list[str]) -> list[dict]:
        """Every row of these agents matching all *terms* (anywhere: name,
        title, summary, tags, category and kind, or the file's text — for a
        picture its description, for a recording its transcript). Matched
        here, not in SQL: SQLite's lower()/LIKE fold ASCII only, and the files
        aren't."""
        if not slugs:
            return []
        with self._lock:
            rows = [dict(r) for r in self._conn.execute(
                f"SELECT * FROM library_files WHERE agent_slug IN ({','.join('?' * len(slugs))})", slugs)]
        if not terms:
            return rows
        fields = ("filename", "rel_path", "title", "summary", "tags", "category", "excerpt")
        category_names, kind_names = dict(CATEGORIES), dict(KINDS)

        def haystack(r: dict) -> str:
            named = (category_names.get(r.get("category") or "", ""), kind_names.get(r.get("kind") or "", ""))
            return " ".join([*(str(r.get(k) or "") for k in fields), *named]).casefold()

        out: list[dict] = []
        for r in rows:
            text = haystack(r)
            if all(t in text for t in terms):
                out.append(r)
        return out


_STORE: LibraryStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> LibraryStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = LibraryStore(_data_dir() / "file_library.db")
        return _STORE


# ── Whose agents ──────────────────────────────────────────────────────


def _auth_enabled() -> bool:
    return os.environ.get("FD_AUTH_ENABLED", "true").lower() in ("true", "1", "yes")


def caller_agents(user: dict) -> dict[str, dict]:
    """The process agents whose files make up *user*'s library: their own
    (every agent, on a deck without sign-in), minus the workers Flight Deck
    runs itself. ``{slug: {name, running}}``."""
    from captain_claw.flight_deck.agent_sharing import is_managed_agent
    from captain_claw.flight_deck.server import _load_process_registry, _process_is_alive
    uid = str(user.get("id") or "")
    out: dict[str, dict] = {}
    for slug, entry in _load_process_registry().items():
        if _auth_enabled() and str(entry.get("owner") or "") != uid:
            continue
        if is_managed_agent(slug, str(entry.get("description") or "")):
            continue
        out[slug] = {
            "name": str(entry.get("name") or slug),
            "running": _process_is_alive(slug),
        }
    return out


# ── Shared folders as sources ─────────────────────────────────────────
#
# The caller's own shared folders — vfs:<folder>/, the ones they or their
# agents made, an agent run's folder too — sit in both libraries beside the
# agents: in the File library only what the person added (a whole folder, or
# single files; see ``library_picks``), walked and summarised like a
# workspace; in the Data library every folder's datastore (vfs:<folder>/.datastore). A folder's key
# carries its owner, since two people may each keep a "reports":
# ``vfs:<owner>/<folder>``. The UI knows it as ``vfs:<folder>`` (what the
# agents call it too); the key never leaves Flight Deck. Folders linked in
# from this computer, Drive and OneDrive folders and folders other people
# shared stay out: a linked repository would be summarised file by file, a
# mount's files are stand-ins until read, and someone else's folder isn't
# the caller's tokens to spend.

FOLDER_PREFIX = "vfs:"


def folder_of(slug: str) -> tuple[str, str] | None:
    """(owner, folder) for a shared folder's key; None for an agent."""
    if not slug.startswith(FOLDER_PREFIX):
        return None
    owner, _, folder = slug[len(FOLDER_PREFIX):].partition("/")
    if not owner or not folder or "/" in folder or folder in (".", ".."):
        return None
    return owner, folder


def _vfs_owner(uid: str) -> str:
    """The caller's folder in the VFS root (as the VFS routes name it)."""
    from captain_claw.vfs import safe_name
    return safe_name(uid, fallback="local")


def own_folder(user: dict, folder: str) -> Path | None:
    """One of the caller's own physical shared folders (not linked in, not a
    mount, not Flight Deck's own), or None."""
    from captain_claw.vfs import read_links_at
    name = str(folder or "")
    if not name or name.startswith(".") or "/" in name or "\\" in name or name in (".", ".."):
        return None
    root = _data_dir() / "vfs" / _vfs_owner(str(user.get("id") or ""))
    try:
        if name in set(read_links_at(root)):
            return None
        p = root / name
        return p if p.is_dir() and not p.is_symlink() else None
    except OSError:
        return None


def caller_folders(user: dict, *, picked_only: bool = True) -> dict[str, dict]:
    """The caller's own shared folders as library sources:
    ``{key: {id, name, running, source, folder}}`` (named by the folder; a
    run's title comes with :func:`caller_sources`). The File library holds only
    folders something was added from (``picked_only``); the Data library all."""
    from captain_claw.vfs import read_links_at
    owner = _vfs_owner(str(user.get("id") or ""))
    root = _data_dir() / "vfs" / owner
    picked = get_store().picks(owner) if picked_only else None
    out: dict[str, dict] = {}
    try:
        entries = sorted(root.iterdir(), key=lambda p: p.name.lower()) if root.is_dir() else []
        linked = set(read_links_at(root))
    except OSError:
        return out
    for p in entries:
        name = p.name
        # Dot folders are Flight Deck's own (.drive mount trees); a linked
        # folder's name points elsewhere.
        if name.startswith(".") or name in linked:
            continue
        if picked is not None and name not in picked:
            continue
        try:
            if p.is_symlink() or not p.is_dir():
                continue
        except OSError:
            continue
        out[f"{FOLDER_PREFIX}{owner}/{name}"] = {
            "id": f"{FOLDER_PREFIX}{name}", "name": name, "running": False,
            "source": "folder", "folder": name,
        }
    return out


async def _bat_titles(user: dict) -> dict[str, str]:
    """``{bat-<id8>: "<title> · <day>"}``: a Bat run's folder (a First Mate
    factory's run, say) named by its run."""
    try:
        from datetime import datetime

        from captain_claw.flight_deck import bat_loop
        store = bat_loop.get_store()
        if store is None:
            return {}
        out = {}
        for r in await store.list_runs(str(user.get("id") or "") or "local", limit=500):
            day = ""
            try:
                at = datetime.fromtimestamp(float(r.get("created_at") or 0))
                day = f"{at.day} {at:%b}"
            except (TypeError, ValueError, OSError):
                pass
            out[f"bat-{str(r['id'])[:8]}"] = f"{r.get('title') or 'Bat run'}" + (f" · {day}" if day else "")
        return out
    except Exception:  # noqa: BLE001 — names are a nicety
        return {}


async def caller_sources(user: dict, *, all_folders: bool = False) -> dict[str, dict]:
    """Everything the caller's libraries hold: their agents (keyed by slug)
    and their shared folders (keyed ``vfs:<owner>/<folder>``): in the File
    library the folders something was added from, in the Data library
    (``all_folders``) every one. Each says the id the UI knows it by, its
    name, whether it runs, and whether it is an agent or a folder — a run's
    folder named by the run's title."""
    agents = caller_agents(user)
    out: dict[str, dict] = {slug: {**a, "id": slug, "source": "agent"} for slug, a in agents.items()}
    folders = await asyncio.to_thread(caller_folders, user, picked_only=not all_folders)
    from captain_claw.flight_deck.vfs_routes import _project_origin, _run_titles
    if any(_project_origin(f["folder"])[1] for f in folders.values()):
        titles = await _run_titles(str(user.get("id") or ""))
        for f in folders.values():
            title = titles.get(_project_origin(f["folder"])[1], "")
            if title:
                f["name"] = title
    if any(f["folder"].startswith("bat-") for f in folders.values()):
        bats = await _bat_titles(user)
        for f in folders.values():
            if f["folder"] in bats:
                f["name"] = bats[f["folder"]]
    out.update(folders)
    return out


def source_key(public: str, sources: dict[str, dict]) -> str:
    """The key of the source the UI calls *public* (an agent's slug is both)."""
    for key, s in sources.items():
        if s.get("id") == public:
            return key
    return public


def public_id(slug: str, sources: dict[str, dict]) -> str:
    """What the UI calls the source with key *slug*."""
    s = sources.get(slug)
    if s and s.get("id"):
        return str(s["id"])
    folder = folder_of(slug)
    return f"{FOLDER_PREFIX}{folder[1]}" if folder else slug


def source_fields(slug: str, sources: dict[str, dict]) -> dict:
    """A row's source as the UI shows it: ``agent_slug`` (its id),
    ``agent_name``, ``agent_running``, ``source`` and, for a folder, ``folder``."""
    s = sources.get(slug, {})
    folder = folder_of(slug)
    out = {
        "agent_slug": public_id(slug, sources),
        "agent_name": s.get("name") or (folder[1] if folder else slug),
        "agent_running": bool(s.get("running")),
        "source": "folder" if folder else "agent",
    }
    if folder:
        out["folder"] = folder[1]
    return out


def where_line(row_slug: str, name: str) -> str:
    """The prompt's line on where a file or table lives."""
    return f"Shared folder: {name}" if folder_of(row_slug) else f"Agent: {name}"


def _workspace(slug: str) -> Path:
    """Where a source's files live: an agent's workspace, or a shared folder."""
    folder = folder_of(slug)
    if folder is not None:
        return _data_dir() / "vfs" / folder[0] / folder[1]
    return _data_dir() / slug / "data" / "workspace"


def _file_entry(p: Path, root: Path) -> dict | None:
    from captain_claw.flight_deck.server import _WORKSPACE_ALLOWED_EXTS
    ext = p.suffix.lower()
    if p.name.startswith(".") or ext not in _WORKSPACE_ALLOWED_EXTS:
        return None
    try:
        st = p.stat()
    except OSError:
        return None
    return {"rel": p.relative_to(root).as_posix(), "filename": p.name, "extension": ext, "kind": kind_of(ext),
            "size": int(st.st_size), "modified": float(st.st_mtime), "fingerprint": f"{st.st_size}:{int(st.st_mtime)}"}


def can_hold(rel: str) -> str:
    """Why a folder's file can't be in the library, or "" when it can."""
    from captain_claw.flight_deck.server import _WORKSPACE_ALLOWED_EXTS
    parts = [s for s in str(rel or "").split("/") if s]
    if not parts or any(s.startswith(".") or s in _SKIP_DIRS for s in parts):
        return "hidden files and Flight Deck's own folders stay out of the library"
    ext = Path(parts[-1]).suffix.lower()
    if ext not in _WORKSPACE_ALLOWED_EXTS:
        return f"the library doesn't read {ext or 'files without a kind'}"
    return ""


def walk_workspace(slug: str) -> list[dict]:
    """The agent's files on disk, as the Files panel lists them; for a shared
    folder, what of it the person added (all of it, or those files)."""
    from captain_claw.flight_deck.server import _WORKSPACE_ALLOWED_EXTS
    from captain_claw.vfs import safe_join
    root = _workspace(slug)
    out: list[dict] = []
    if not root.is_dir():
        return out
    folder = folder_of(slug)
    if folder is not None:
        picked = get_store().picks(folder[0]).get(folder[1], set())
        if "" not in picked:
            for rel in sorted(picked):
                p = safe_join(root, rel)
                entry = _file_entry(p, root) if p is not None and p.is_file() and not p.is_symlink() else None
                if entry is not None and not can_hold(rel):
                    out.append(entry)
            return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            ext = Path(name).suffix.lower()
            if ext not in _WORKSPACE_ALLOWED_EXTS:
                continue
            p = Path(dirpath) / name
            try:
                st = p.stat()
            except OSError:
                continue
            out.append({
                "rel": p.relative_to(root).as_posix(),
                "filename": name,
                "extension": ext,
                "kind": kind_of(ext),
                "size": int(st.st_size),
                "modified": float(st.st_mtime),
                "fingerprint": f"{st.st_size}:{int(st.st_mtime)}",
            })
            if len(out) >= _MAX_FILES_PER_AGENT:
                return out
    return out


def _resolve_path(row: dict) -> Path | None:
    """The row's file on disk — only ever inside its agent's workspace."""
    from captain_claw.vfs import safe_join
    root = _workspace(row["agent_slug"])
    p = safe_join(root, row["rel_path"])
    return p if p is not None and p.is_file() else None


# ── Reading a file ────────────────────────────────────────────────────


_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.S | re.I)


def _squash(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def read_text(path: Path, ext: str) -> tuple[str, str]:
    """(text, why-not): the file's text for the model, or "" and the reason
    it can't be read."""
    if ext in _NOT_READ_REASON:
        return "", _NOT_READ_REASON[ext]
    if ext in _PLAIN_TEXT_EXTS:
        with path.open("rb") as fh:
            raw = fh.read(_MAX_READ_BYTES)
        text = raw.decode("utf-8", errors="replace")
        if ext in (".html", ".htm", ".svg"):
            text = _TAG_RE.sub(" ", text)
        return _squash(text), ""
    if ext in _EXTRACTED_EXTS:
        from captain_claw.tools import document_extract as dx
        if ext == ".pdf":
            text, err = dx._extract_pdf_markdown(path, 10)
            if err:
                return "", err
            text = text or ""
        elif ext == ".docx":
            text = dx._extract_docx_markdown(path)
        elif ext == ".xlsx":
            text = dx._extract_xlsx_markdown(path, 60)
        else:
            text = dx._extract_pptx_markdown(path, 30)
        text = _squash(text)
        if "No extractable text found" in text and len(text) < 300:
            return "", "No text could be read from it (a scan or an image-only file)."
        return text, ""
    return "", ""


def _humanize(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[-_]+", " ", stem)
    stem = re.sub(r"\s*\b\d{8}[- ]?\d{4,6}\b", "", stem).strip()
    return (stem[:1].upper() + stem[1:]) if stem else filename


def _metadata_only(row: dict, reason: str = "", waiting: str = "") -> dict:
    """A file the model doesn't read: titled from its name, sorted by kind.
    *waiting*: what it waits for ("vision", "soniox", "ffmpeg") — once that
    is there, a scan takes it up again."""
    kind = row["kind"]
    category = "media" if kind in ("image", "media") else "code" if kind == "code" else "other"
    default = {
        "image": "Image — nothing to read, so not summarised.",
        "media": "Audio or video — not summarised.",
        "archive": "Archive — not summarised.",
    }.get(kind, "This file type can't be read for a summary.")
    return {
        "status": "skipped",
        "title": _humanize(row["filename"]),
        "summary": reason or default,
        "category": category,
        "tags": "[]",
        "excerpt": "",
        "error": "",
        "model": "",
        "tokens": 0,
        "summarized_at": time.time(),
        "waiting_for": waiting,
    }


# ── The model ─────────────────────────────────────────────────────────


_SYSTEM = """You catalogue files that AI agents produced for the person they work for.
Read the file and reply with ONLY a JSON object, no prose, no code fence:
{"title": "...", "summary": "...", "category": "...", "tags": ["...", "..."]}

- title: a clear human title for the file, at most 80 characters.
- summary: 2-3 sentences, at most 420 characters: what the file is and its key
  takeaways (names, numbers, conclusions). No "This file..." filler.
- category: exactly one of: %s
- tags: 2-5 short lowercase tags (topics, companies, people, places).
Write the title and summary in the language the file is written in."""

_PICTURE_SYSTEM = """You catalogue pictures that AI agents saved for the person they work for.
Look at the picture and reply with ONLY a JSON object, no prose, no code fence:
{"title": "...", "summary": "...", "category": "...", "tags": ["...", "..."], "description": "..."}

- title: a clear human title for the picture, at most 80 characters.
- summary: 1-2 sentences, at most 300 characters: what it shows and what it is
  for. No "This image..." filler.
- category: exactly one of: %s — a photo, drawing or illustration is
  "media"; a chart, a scanned page or a screenshot goes where its content belongs.
- tags: 2-5 short lowercase tags (subjects, places, people, brands).
- description: at most 1200 characters: what is in it, in detail, and every
  piece of text you can read in it, word for word.
Write in the language of the text in the picture; with no text, in English."""


async def _resolve_tier(user_id: str, prefer: tuple[str, ...] = ("fast", "balanced")) -> dict | None:
    """The owner's fast tier (else balanced, else any): the model that
    writes their library — or the first of ``prefer`` they have. None when
    they have no tier set at all."""
    from captain_claw.flight_deck.basna_routes import _load_owner_tiers
    tiers: dict = {}
    try:
        from captain_claw.flight_deck.auth import get_db
        tiers, _env = await _load_owner_tiers(get_db(), user_id)
    except Exception:
        tiers = {}
    if not tiers:
        try:
            from captain_claw.flight_deck.basna_routes import _load_registry
            tiers = _load_registry().get("tiers") or {}
        except Exception:
            tiers = {}
    cfg = next((tiers[t] for t in prefer if tiers.get(t)), None) or next(iter(tiers.values()), None)
    return cfg if isinstance(cfg, dict) and cfg.get("model") else None


def reply_json(content: str) -> dict:
    """The JSON object a model replied with (fenced or not), or raise."""
    text = (content or "").strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```"))
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("the model did not answer with JSON")
    data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("the model did not answer with a JSON object")
    return data


def clean_tags(raw: Any, limit: int = 5) -> list[str]:
    tags: list[str] = []
    for t in raw if isinstance(raw, list) else []:
        t = " ".join(str(t).lower().split())[:32]
        if t and t not in tags:
            tags.append(t)
    return tags[:limit]


def _parse_reply(content: str) -> dict:
    data = reply_json(content)
    category = str(data.get("category") or "").strip().lower()
    return {
        "title": " ".join(str(data.get("title") or "").split())[:120],
        "summary": " ".join(str(data.get("summary") or "").split())[:600],
        "category": category if category in _CATEGORY_IDS else "other",
        "tags": clean_tags(data.get("tags")),
        "description": str(data.get("description") or "").strip()[:4000],
    }


class PictureRefusedError(RuntimeError):
    """The model's server wouldn't take the picture: the call went through
    without it, so the reply says nothing about the picture."""


class ModelReply(NamedTuple):
    """One model call's answer, and what it says about how it went."""

    text: str
    tokens: int                # spent, prompt and reply: toward the owner's day
    prompt_tokens: int = 0     # the server's own count of what it was sent (0: it didn't say)
    finish_reason: str = ""    # "length" / "max_tokens": cut off at max_tokens
    reasoning: str = ""        # a thinking model's thoughts, when its server sends them apart


def _quiet() -> contextlib.AbstractContextManager:
    """A description call runs with thinking off where the model can switch it
    off: a thinking model counts its thinking against max_tokens, and on a
    small budget it can think through all of it and never write the JSON."""
    from captain_claw.llm import reasoning
    return reasoning.quiet_call()


def cut_off(reply: ModelReply) -> bool:
    """The answer stopped at its max_tokens, not where the model ended it."""
    return reply.finish_reason.strip().lower() in ("length", "max_tokens")


async def call_model(tier: dict, system: str, user: str, owner_id: str,
                     max_tokens: int = _MAX_TOKENS,
                     request_timeout: float | None = None,
                     images: list | None = None) -> tuple[str, int]:
    """One call to *tier*'s model: (reply text, tokens spent). The tokens
    count toward the owner's daily allowance. ``request_timeout`` overrides
    the provider's own (Ollama's is 120s) for a long background call.
    ``images`` go with the user message (``EncodedImage``); raises
    :class:`PictureRefusedError` when the server would not take them."""
    reply = await model_reply(tier, system, user, owner_id, max_tokens=max_tokens,
                              request_timeout=request_timeout, images=images)
    return reply.text, reply.tokens


async def model_reply(tier: dict, system: str, user: str, owner_id: str,
                      max_tokens: int = _MAX_TOKENS,
                      request_timeout: float | None = None,
                      images: list | None = None) -> ModelReply:
    """:func:`call_model`, with what the server said about the call. A
    reasoning level set around it (``reasoning.call_effort``) reaches the
    request."""
    from captain_claw.flight_deck.basna_routes import tier_api_key
    from captain_claw.llm import Message, create_provider, is_reasoning_backfill_placeholder
    provider = create_provider(
        provider=tier.get("provider", "anthropic"), model=tier.get("model", ""),
        base_url=tier.get("base_url") or None, api_key=tier_api_key(tier),
        temperature=0.2, max_tokens=max_tokens)
    if request_timeout:
        provider.request_timeout = request_timeout
    if images:
        # A server that refuses pictures is retried without them, and the
        # provider turns this off — which is how the refusal shows here.
        provider.sees_images = True
    resp = await provider.complete(
        messages=[Message(role="system", content=system),
                  Message(role="user", content=user, images=images or None)],
        temperature=0.2, max_tokens=max_tokens)
    usage = dict(getattr(resp, "usage", None) or {})
    tokens = int(usage.get("total_tokens") or 0) or (
        int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0))
    if owner_id and _auth_enabled() and tokens:
        try:
            from captain_claw.flight_deck import usage_quota
            from captain_claw.flight_deck.auth import get_db
            await usage_quota.record_tokens(get_db(), owner_id, tokens, byok=bool(tier.get("byok")))
        except Exception:
            log.debug("library: token report failed", exc_info=True)
    if images and getattr(provider, "sees_images", None) is False:
        raise PictureRefusedError(f"{tier.get('model') or 'the model'} would not take the picture")
    reasoning = str(getattr(resp, "reasoning_content", "") or "")
    return ModelReply(resp.content or "", tokens, int(usage.get("prompt_tokens") or 0),
                      str(getattr(resp, "finish_reason", "") or ""),
                      "" if is_reasoning_backfill_placeholder(reasoning) else reasoning)


# ── Pictures, recordings and videos ───────────────────────────────────

# Where ffmpeg lives when Flight Deck's PATH doesn't say: an app started
# from the Dock gets launchd's short PATH, without Homebrew's.
_TOOL_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin")


def _find_tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    for d in _TOOL_DIRS:
        p = Path(d) / name
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return ""


def _endpoint(provider: str, base_url: str) -> str:
    """The endpoint a provider built from this config calls — what a refused
    picture is remembered under (Ollama fills in its own when blank)."""
    base_url = str(base_url or "").strip()
    if provider == "ollama" and not base_url:
        from captain_claw.llm import OLLAMA_NATIVE_BASE_URL
        return os.getenv("OLLAMA_BASE_URL") or OLLAMA_NATIVE_BASE_URL
    return base_url


# Providers create_provider runs through LiteLLM: their model is named with
# the provider's prefix and without an effort suffix (gpt-5-high → openai/gpt-5).
_LITELLM_PROVIDERS = {"openai", "anthropic", "gemini", "xai", "openrouter"}


def _vision_id(provider: str, model: str, base_url: str) -> tuple[str, str, str]:
    """(provider, model, endpoint) as the provider built from this config
    names itself — what the agent decides vision by, and what a refused
    picture is remembered under."""
    from captain_claw.llm import (
        _extract_reasoning_effort,
        _normalize_provider_name,
        _provider_model_name,
    )
    provider = _normalize_provider_name(provider)
    model = str(model or "").strip()
    if provider in _LITELLM_PROVIDERS and model:
        stripped, _effort = _extract_reasoning_effort(model)
        model = _provider_model_name(provider, stripped or model)
    return provider, model, _endpoint(provider, base_url).rstrip("/")


def _sees(provider: str, model: str, base_url: str, switch: Any) -> bool:
    from captain_claw.llm.vision import resolve_sees_images
    try:
        return resolve_sees_images(*_vision_id(provider, model, base_url), switch)
    except Exception:  # noqa: BLE001 — an unknown model is one that doesn't see
        return False


def _refused_pictures(provider: str, model: str, base_url: str) -> bool:
    """Did this model's server turn pictures down lately (remembered a day)?"""
    from captain_claw.llm.vision import vision_rejected
    try:
        return vision_rejected(*_vision_id(provider, model, base_url))
    except Exception:  # noqa: BLE001
        return False


def _agent_model_key(slug: str, provider: str, model: dict, provider_keys: dict) -> str:
    """The key the agent calls its model with: its config's, else its .env's
    ("" lets the provider fall back to Flight Deck's own environment)."""
    from captain_claw.flight_deck import server
    raw = str(model.get("provider") or "")
    key = (str(model.get("api_key") or "").strip()
           or str(provider_keys.get(raw) or provider_keys.get(provider) or "").strip())
    if key:
        return key
    try:
        from captain_claw.config import Config
        dotenv = Config._read_dotenv_file(server.DATA_DIR / slug / ".env")
    except Exception:  # noqa: BLE001
        return ""
    for name in server._provider_key_env_names(provider):
        value = str(dotenv.get(name) or "").strip()
        if value:
            return value
    return ""


_VISION_OFF = ("Not described: its agent's vision is off. Turn it on in the agent's settings to "
               "have its pictures described.")
_VISION_UNSEEN = ("Not described: its agent's model doesn't see pictures. If it does, turn vision "
                  "on in the agent's settings.")
_VISION_REFUSED = ("Not described: the model turned pictures down. It is asked again in a day, or "
                   "sooner with a model that sees pictures.")
_VISION_SIGN_IN = ("Not described: its agent sees pictures through its ChatGPT sign-in, which the "
                   "library can't use — a library model that sees pictures can.")
_VISION_TEXT_ONLY = ("Not described: its agent's model takes text only — a library model that sees "
                     "pictures can describe them.")
_VISION_FOLDER = ("Not described: the library's model doesn't see pictures. Pick one that does in "
                  "your model settings to have the pictures in your shared folders described.")


def _tier_id(tier: dict) -> tuple[str, str, str]:
    return _vision_id(str(tier.get("provider") or ""), str(tier.get("model") or ""),
                      str(tier.get("base_url") or ""))


def _eyes_for(slug: str, tier: dict | None) -> tuple[list[dict], str]:
    """(models, why-none): the models that can look at *slug*'s pictures,
    best first — the owner's library tier when it sees pictures, then the
    agent's own model — or [] and why, when the agent's vision is off or
    nothing can look. The agent's switch decides: on, or auto with a model
    that sees."""
    from captain_claw.flight_deck import server
    from captain_claw.llm.vision import normalize_vision_switch
    if folder_of(slug) is not None:
        # A shared folder has no agent of its own: the library's tier, if it sees.
        if tier is not None and _sees(str(tier.get("provider") or ""), str(tier.get("model") or ""),
                                      str(tier.get("base_url") or ""),
                                      "on" if tier.get("vision") is True else "auto"):
            return [tier], ""
        return [], _VISION_FOLDER
    loaded = server._agent_config_model(slug)
    if loaded is None:
        return [], _VISION_OFF
    model, provider_keys = loaded
    name = str(model.get("model") or "").strip()
    base_url = str(model.get("base_url") or "").strip()
    own_id = _vision_id(str(model.get("provider") or ""), name, base_url)
    provider = own_id[0]
    switch = normalize_vision_switch(model.get("vision", "auto"))
    if not name or switch == "off":
        return [], _VISION_OFF
    if switch == "auto" and not _sees(provider, name, base_url, "auto"):
        return [], _VISION_REFUSED if _refused_pictures(provider, name, base_url) else _VISION_UNSEEN
    eyes: list[dict] = []
    if tier is not None and _sees(str(tier.get("provider") or ""), str(tier.get("model") or ""),
                                  str(tier.get("base_url") or ""),
                                  "on" if tier.get("vision") is True else "auto"):
        eyes.append(tier)
    same_as_tier = bool(eyes) and _tier_id(tier) == own_id
    # A model on the ChatGPT sign-in can't be called with a key from here.
    chatgpt = server._signs_in_through_chatgpt(provider, name)
    if not same_as_tier and not chatgpt and _sees(provider, name, base_url, switch):
        eyes.append({"provider": provider, "model": name, "base_url": base_url,
                     "api_key": _agent_model_key(slug, provider, model, provider_keys)})
    if eyes:
        return eyes, ""
    if chatgpt:
        return [], _VISION_SIGN_IN
    if _refused_pictures(provider, name, base_url):
        return [], _VISION_REFUSED
    return [], _VISION_TEXT_ONLY if switch == "on" else _VISION_UNSEEN


@dataclass
class MediaReach:
    """What a scan can do with pictures and recordings, worked out once."""
    owner: str
    tier: dict | None
    soniox: bool = False         # the owner has a Soniox key (own, team or env)
    ffmpeg: str = ""
    ffprobe: str = ""
    soniox_refused: bool = False  # Soniox turned the key down (until it is replaced)
    soniox_failures: int = 0      # Soniox failures in a row
    soniox_stopped: bool = False  # …too many: the rest wait for the next scan
    # Models that turned a picture down during this scan: never sent another.
    refused: set[tuple[str, str, str]] = field(default_factory=set)
    # …and the ones that were sent one (whatever came of it): sent the rest
    # side by side.
    probed: set[tuple[str, str, str]] = field(default_factory=set)
    _probes: dict[tuple[str, str, str], asyncio.Lock] = field(default_factory=dict)
    _eyes: dict[str, tuple[list[dict], str]] = field(default_factory=dict)

    def probe(self, model_id: tuple[str, str, str]) -> contextlib.AbstractAsyncContextManager:
        """Until a model has been sent a picture this scan, pictures go to it
        one at a time: one that turns pictures down does so once, not once
        per picture in flight."""
        if model_id in self.probed:
            return contextlib.nullcontext()
        return self._probes.setdefault(model_id, asyncio.Lock())

    def eyes(self, slug: str) -> tuple[list[dict], str]:
        """(models, why-none) for *slug*'s pictures, less the ones that
        turned a picture down during this scan."""
        if slug not in self._eyes:
            self._eyes[slug] = _eyes_for(slug, self.tier)
        eyes, why = self._eyes[slug]
        left = [m for m in eyes if _tier_id(m) not in self.refused]
        return (left, "") if left else ([], why or _VISION_REFUSED)


async def media_reach(owner_id: str, tier: dict | None) -> MediaReach:
    from captain_claw.flight_deck import soniox_routes
    try:
        status = await soniox_routes.status_for_owner(owner_id or None)
    except Exception:  # noqa: BLE001 — no settings store: no key
        status = {}
    return MediaReach(owner=owner_id, tier=tier, soniox=bool(status.get("key_set")),
                      soniox_refused=status.get("verify_error") == "refused",
                      ffmpeg=_find_tool("ffmpeg"), ffprobe=_find_tool("ffprobe"))


def _encode_picture(path: Path):
    """The picture as a model gets it: upright, the first frame of an
    animation, flattened on white, at most _PICTURE_EDGE a side, as JPEG."""
    from PIL import Image, ImageOps

    from captain_claw.llm.vision import EncodedImage
    with _PICTURE_DECODE_LOCK, Image.open(path, formats=_PICTURE_FORMATS) as img:
        cap = _MAX_WEBP_PIXELS if img.format == "WEBP" else _MAX_PICTURE_PIXELS
        if img.width * img.height > cap:
            raise ValueError("it has too many pixels to look at here")
        # A JPEG decodes straight at a fraction of its size (no-op for others).
        img.draft("RGB", (_PICTURE_EDGE * 2, _PICTURE_EDGE * 2))
        img.seek(0)
        # Pillow shrinks a palette picture pixel by pixel (text in a
        # screenshot breaks up): in full colour first.
        small = img.convert("RGBA") if img.mode in ("P", "1") else img
        # Made small first (one full-size decode, no other full-size copy),
        # then turned upright: the EXIF orientation outlives the resize.
        small.thumbnail((_PICTURE_EDGE, _PICTURE_EDGE))
        pic = ImageOps.exif_transpose(small)
        if pic.mode.startswith("I"):
            # 16-bit grey (a scan, a depth map): scaled to 8 bits, not clipped white.
            pic = pic.point(lambda v: v * (1 / 256)).convert("L")
        if pic.mode in ("RGBA", "LA", "PA") or (pic.mode == "P" and "transparency" in pic.info):
            rgba = pic.convert("RGBA")
            pic = Image.new("RGB", rgba.size, (255, 255, 255))
            pic.paste(rgba, mask=rgba.getchannel("A"))
        else:
            pic = pic.convert("RGB")
        buf = io.BytesIO()
        pic.save(buf, "JPEG", quality=85)
    return EncodedImage(path=str(path), mime="image/jpeg",
                        data=base64.b64encode(buf.getvalue()).decode("ascii"),
                        width=pic.width, height=pic.height)


async def _summarize_picture(row: dict, path: Path, owner_id: str, reach: MediaReach) -> dict:
    if row["size"] > _MAX_PICTURE_BYTES:
        return _metadata_only(row, "Too large to look at here.")
    eyes, why = reach.eyes(row["agent_slug"])
    if not eyes:
        return _metadata_only(row, why, waiting="vision")
    try:
        picture = await asyncio.to_thread(_encode_picture, path)
    except Exception as e:  # noqa: BLE001 — a broken picture is a skipped file
        return _metadata_only(row, f"Could not be opened: {str(e)[:160]}")
    user_msg = (f"File: {row['rel_path']}\n{where_line(row['agent_slug'], row.get('agent_name') or row['agent_slug'])}\n"
                f"Picture: {picture.width}×{picture.height} pixels, attached.")
    failed: Exception | None = None
    for i, model in enumerate(eyes):
        model_id = _tier_id(model)
        try:
            async with reach.probe(model_id):
                if model_id in reach.refused:
                    continue
                try:
                    with _quiet():
                        content, tokens = await call_model(
                            model, _PICTURE_SYSTEM % ", ".join(c for c, _ in CATEGORIES), user_msg, owner_id,
                            max_tokens=_PICTURE_MAX_TOKENS, images=[picture])
                finally:
                    reach.probed.add(model_id)
            parsed = _parse_reply(content)
        except PictureRefusedError:
            reach.refused.add(model_id)
            continue
        except Exception as e:  # noqa: BLE001 — the next model may do; the last one's failure is the scan's
            if i == len(eyes) - 1:
                raise
            log.info("library: picture model failed, trying the next", model=model.get("model"), error=str(e)[:160])
            failed = e
            continue
        return {
            "status": "done",
            "title": parsed["title"] or _humanize(row["filename"]),
            "summary": parsed["summary"],
            "category": parsed["category"],
            "tags": json.dumps(parsed["tags"], ensure_ascii=False),
            "excerpt": parsed["description"][:_TRANSCRIPT_CHARS],
            "error": "",
            "model": str(model.get("model") or ""),
            "tokens": tokens,
            "summarized_at": time.time(),
            "waiting_for": "",
        }
    if failed is not None:
        raise failed
    return _metadata_only(row, _VISION_REFUSED, waiting="vision")


def _fingerprint(path: Path) -> str:
    """The file's version as a walk records it (size and whole-second mtime)."""
    try:
        st = path.stat()
    except OSError:
        return ""
    return f"{st.st_size}:{int(st.st_mtime)}"


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


async def _sound_streams(ffprobe: str, path: Path) -> int | None:
    """How many sound tracks the file has (None: ffprobe couldn't tell)."""
    from captain_claw.tools.video_vision import _run
    rc, out, _ = await _run([ffprobe, "-v", "error", "-select_streams", "a",
                             "-show_entries", "stream=index", "-of", "csv=p=0", str(path)], timeout=30.0)
    if rc != 0:
        return None
    return len([line for line in out.decode(errors="replace").splitlines() if line.strip()])


async def _sound_of(row: dict, path: Path, reach: MediaReach) -> tuple[bytes, str, str, float] | dict:
    """(audio, filename, mime, seconds) to send to Soniox — the sound taken
    out with ffmpeg (small Opus), else the file itself when Soniox reads it
    — or the columns to store when there is nothing to send."""
    from captain_claw.tools.video_vision import _extract_audio, _ffprobe_duration
    ext = row["extension"]
    seconds = 0.0
    if reach.ffmpeg:
        if reach.ffprobe:
            if await _sound_streams(reach.ffprobe, path) == 0:
                return _metadata_only(row, "No sound in it — nothing to transcribe.")
            seconds = await _ffprobe_duration(reach.ffprobe, str(path))
            if seconds > _MAX_SPEECH_SECONDS:
                return _metadata_only(row, "Longer than five hours — too long to transcribe.")
        try:
            audio, filename, mime = await _extract_audio(reach.ffmpeg, str(path))
        except Exception as e:  # noqa: BLE001 — ffmpeg missing after all, or broken
            log.debug("library: sound extraction failed", file=row["rel_path"], error=str(e))
            audio, filename, mime = b"", "", ""
        if audio:
            return audio, filename, mime, seconds
        # ffmpeg can't read it: Soniox won't either (a damaged file).
        return _metadata_only(row, "The sound couldn't be taken out of it — the file may be damaged.")
    if ext not in _SONIOX_READS_EXTS:
        return _metadata_only(
            row, "Not transcribed: the sound of this kind of video is taken out with ffmpeg, "
                 "which isn't installed on the Flight Deck computer.", waiting="ffmpeg")
    too_large = _metadata_only(
        row, "Not transcribed: too large to send whole — with ffmpeg installed on the Flight "
             "Deck computer, only its sound is sent.", waiting="ffmpeg")
    if row["size"] > _MAX_SOUND_UPLOAD_BYTES:
        return too_large
    from captain_claw.tools.audio import _AUDIO_MIME

    def read() -> bytes:
        # At most the cap, whatever the file has grown to since the walk.
        with path.open("rb") as fh:
            return fh.read(_MAX_SOUND_UPLOAD_BYTES + 1)

    audio = await asyncio.to_thread(read)
    if len(audio) > _MAX_SOUND_UPLOAD_BYTES:
        return too_large
    return audio, row["filename"], _AUDIO_MIME.get(ext, "application/octet-stream"), seconds


async def _summarize_recording(row: dict, path: Path, tier: dict, owner_id: str,
                               reach: MediaReach) -> dict | None:
    """Transcribe what is said in a recording or a video, then have the model
    catalogue it from the transcript. None: Soniox keeps failing this scan,
    so the file waits for the next one."""
    from captain_claw.flight_deck.soniox_routes import SonioxNotConnected
    from captain_claw.soniox_api import SonioxError
    if _AGENT_VOICE_RE.match(row["filename"]):
        return _metadata_only(row, "The agent's own spoken reply — not transcribed.")
    if (row.get("excerpt") and (row.get("waiting_for") == "summary" or row.get("status") == "done")
            and _fingerprint(path) == row["fingerprint"]):
        # Transcribed before (only the summary failed, or it is summarised
        # again): the same file says the same thing.
        return await _summarize_transcript(row, row["excerpt"], 0.0, tier, owner_id)
    if not reach.soniox:
        return _metadata_only(
            row, "Not transcribed: connect Soniox (Settings → Connections) to have recordings "
                 "and videos transcribed and summarised.", waiting="soniox")
    if reach.soniox_refused:
        return _metadata_only(row, "Not transcribed: Soniox turned the key down — check it in "
                                   "Settings → Connections.", waiting="soniox")
    if reach.soniox_stopped:
        return None
    try:
        sound = await _sound_of(row, path, reach)
    except OSError as e:  # unreadable, or gone since the walk
        return _metadata_only(row, f"Could not be read: {str(e)[:160]}")
    if isinstance(sound, dict):
        return sound
    audio, filename, mime, seconds = sound
    from captain_claw.flight_deck import soniox_routes
    try:
        out = await soniox_routes.transcribe_for(
            owner_id or None, audio, mime=mime, filename=filename, poll_max=_SONIOX_POLL,
            source="library", agent=row["agent_slug"])
    except SonioxNotConnected:
        reach.soniox = False
        return _metadata_only(row, "Not transcribed: connect Soniox (Settings → Connections).",
                              waiting="soniox")
    except SonioxError as e:
        if e.refused:
            reach.soniox_refused = True
            return _metadata_only(row, "Not transcribed: Soniox turned the key down — check it in "
                                       "Settings → Connections.", waiting="soniox")
        if e.status in _SONIOX_FILE_ERRORS:
            # This file, not Soniox: left as it is until it changes.
            return _metadata_only(row, f"Soniox couldn't transcribe it: {str(e)[:160]}")
        reach.soniox_failures += 1
        if reach.soniox_failures >= _MAX_SONIOX_FAILURES:
            reach.soniox_stopped = True
        return {"status": "error", "error": f"Transcription failed: {str(e)[:240]}",
                "summarized_at": time.time(), "waiting_for": ""}
    reach.soniox_failures = 0
    text = (out.text or "").strip()
    if not text:
        return _metadata_only(row, "No speech in it.")
    return await _summarize_transcript(row, text, seconds, tier, owner_id)


async def _summarize_transcript(row: dict, text: str, seconds: float, tier: dict, owner_id: str) -> dict:
    """Have the model catalogue a recording or a video from its transcript."""
    what = "a video" if row["extension"] in _VIDEO_EXTS else "a recording"
    length = f", {_clock(seconds)} long" if seconds else ""
    body = text if len(text) <= _LLM_CHARS else text[:_LLM_CHARS] + "\n\n[… truncated]"
    user_msg = (f"File: {row['rel_path']}\n{where_line(row['agent_slug'], row.get('agent_name') or row['agent_slug'])}\n"
                f"This is {what}{length}; below is the transcript of what is said in it.\n\n---\n{body}")
    try:
        with _quiet():
            content, tokens = await call_model(
                tier, _SYSTEM % ", ".join(c for c, _ in CATEGORIES), user_msg, owner_id)
        parsed = _parse_reply(content)
    except Exception:
        # Keep the transcript (search finds it meanwhile): the retry only
        # asks the model again, it doesn't transcribe the file again.
        get_store().save_result(row["id"], row["fingerprint"],
                                excerpt=text[:_TRANSCRIPT_CHARS], waiting_for="summary")
        raise
    return {
        "status": "done",
        "title": parsed["title"] or _humanize(row["filename"]),
        "summary": parsed["summary"],
        "category": parsed["category"],
        "tags": json.dumps(parsed["tags"], ensure_ascii=False),
        "excerpt": text[:_TRANSCRIPT_CHARS],
        "error": "",
        "model": str(tier.get("model") or ""),
        "tokens": tokens,
        "summarized_at": time.time(),
        "waiting_for": "",
    }


async def summarize_row(row: dict, tier: dict, owner_id: str,
                        reach: MediaReach | None = None) -> dict | None:
    """Read one file and have the model catalogue it. Returns the columns to
    store (never raises for a file it can't read), or None when it is left
    for the next scan. *reach*: what this scan can do with pictures and
    recordings (worked out here when not given)."""
    path = _resolve_path(row)
    if path is None:
        return {"status": "error", "error": "The file is gone.", "summarized_at": time.time()}
    if row["size"] == 0:
        return _metadata_only(row, "Empty file.")
    ext = row["extension"]
    if ext in _PICTURE_EXTS or ext in _SOUND_EXTS:
        if reach is None:
            reach = await media_reach(owner_id, tier)
        if ext in _PICTURE_EXTS:
            return await _summarize_picture(row, path, owner_id, reach)
        return await _summarize_recording(row, path, tier, owner_id, reach)
    if row["size"] > _MAX_FILE_BYTES and row["extension"] not in _PLAIN_TEXT_EXTS:
        return _metadata_only(row, "Too large to read here.")
    try:
        text, why_not = await asyncio.to_thread(read_text, path, row["extension"])
    except Exception as e:  # noqa: BLE001 — a broken file is a skipped file
        return _metadata_only(row, f"Could not be read: {str(e)[:160]}")
    if not text:
        return _metadata_only(row, why_not)

    body = text if len(text) <= _LLM_CHARS else text[:_LLM_CHARS] + "\n\n[… truncated]"
    user_msg = (f"File: {row['rel_path']}\n{where_line(row['agent_slug'], row.get('agent_name') or row['agent_slug'])}\n"
                f"Size: {row['size']} bytes\n\n---\n{body}")
    with _quiet():
        content, tokens = await call_model(
            tier, _SYSTEM % ", ".join(c for c, _ in CATEGORIES), user_msg, owner_id)
    parsed = _parse_reply(content)
    return {
        "status": "done",
        "title": parsed["title"] or _humanize(row["filename"]),
        "summary": parsed["summary"],
        "category": parsed["category"],
        "tags": json.dumps(parsed["tags"], ensure_ascii=False),
        "excerpt": text[:_EXCERPT_CHARS],
        "error": "",
        "model": str(tier.get("model") or ""),
        "tokens": tokens,
        "summarized_at": time.time(),
        "waiting_for": "",
    }


# ── Scans ─────────────────────────────────────────────────────────────


@dataclass
class ScanJob:
    running: bool = False
    phase: str = "idle"          # idle | scanning | summarizing
    total: int = 0               # files this scan set out to summarise
    done: int = 0
    failed: int = 0
    current: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    message: str = ""            # why it stopped early, if it did
    task: asyncio.Task | None = field(default=None, repr=False)

    def view(self) -> dict:
        return {
            "running": self.running, "phase": self.phase, "total": self.total,
            "done": self.done, "failed": self.failed, "current": self.current,
            "started_at": self.started_at or None, "finished_at": self.finished_at or None,
            "message": self.message,
        }


_JOBS: dict[str, ScanJob] = {}


def _job_for(uid: str) -> ScanJob:
    return _JOBS.setdefault(uid, ScanJob())


async def _quota_spent(owner_id: str) -> bool:
    if not (owner_id and _auth_enabled()):
        return False
    try:
        from captain_claw.flight_deck import usage_quota
        return await usage_quota.automation_blocked(owner_id) is not None
    except Exception:
        return False


async def run_scan(user: dict, job: ScanJob, retry_errors: bool = False) -> None:
    uid = str(user.get("id") or "")
    store = get_store()
    try:
        agents = await caller_sources(user)
        # Forget removed agents only on a registry read that worked: the
        # lenient reader returns {} on a failed read, which would drop them all.
        from captain_claw.flight_deck.server import _load_process_registry_strict
        try:
            store.drop_agents_except(set(_load_process_registry_strict().keys()))
        except (OSError, ValueError):
            pass
        owner = _vfs_owner(uid)
        gone = {f for f in store.picks(owner) if own_folder(user, f) is None}
        if gone:  # a folder deleted since: what was added from it goes too
            store.drop_folder_picks(owner, gone)
            agents = {k: v for k, v in agents.items() if not (folder_of(k) and folder_of(k)[1] in gone)}
        store.drop_folders_except(owner, {k for k in agents if folder_of(k)})
        for slug in agents:
            files = await asyncio.to_thread(walk_workspace, slug)
            store.sync_agent(slug, files)
        tier = await _resolve_tier(uid)
        reach = await media_reach(uid, tier)
        # Files set aside for want of something that is there now go back in.
        slugs = list(agents)
        if reach.soniox and not reach.soniox_refused:
            store.requeue_waiting("soniox", slugs)
        if reach.ffmpeg:
            store.requeue_waiting("ffmpeg", slugs)
        store.requeue_waiting("vision", [s for s in slugs if reach.eyes(s)[0]])
        # The browser tool's screenshots are filed without a model.
        store.settle_screenshots(slugs)
        todo = store.pending(slugs, _MAX_SUMMARIES_PER_SCAN,
                             time.time() if retry_errors else time.time() - _ERROR_RETRY_SECONDS)
        if not todo:
            return
        job.phase, job.total = "summarizing", len(todo)
        if tier is None:
            job.message = "No model is set up to write summaries — add one in your model settings."
            return
        if await _quota_spent(uid):
            job.message = "Today's token allowance is used up — summaries continue tomorrow."
            return

        sem = asyncio.Semaphore(_CONCURRENCY)
        # Transcriptions take minutes: they queue on their own, never
        # holding up the files that take seconds.
        sound_sem = asyncio.Semaphore(_SOUND_CONCURRENCY)
        streak = 0
        stop = asyncio.Event()

        async def one(row: dict) -> None:
            nonlocal streak
            async with (sound_sem if row["extension"] in _SOUND_EXTS else sem):
                if stop.is_set():
                    return
                if await _quota_spent(uid):
                    job.message = "Today's token allowance ran out — the rest continue tomorrow."
                    stop.set()
                    return
                row["agent_name"] = agents.get(row["agent_slug"], {}).get("name", row["agent_slug"])
                job.current = row["filename"]
                try:
                    result = await summarize_row(row, tier, uid, reach)
                    streak = 0
                except Exception as e:  # noqa: BLE001 — recorded on the row
                    streak += 1
                    job.failed += 1
                    store.save_result(row["id"], row["fingerprint"], status="error",
                                      error=str(e)[:300], summarized_at=time.time())
                    if streak >= _MAX_FAILURE_STREAK:
                        job.message = f"Stopped: the model keeps failing ({str(e)[:160]})."
                        stop.set()
                    return
                if result is None:
                    job.message = job.message or "Soniox isn't answering — the other recordings wait for the next scan."
                    return
                store.save_result(row["id"], row["fingerprint"], **result)
                if result.get("status") == "error":
                    job.failed += 1
                else:
                    job.done += 1

        await asyncio.gather(*(one(r) for r in todo))
    except Exception as e:  # noqa: BLE001 — a scan never takes FD down
        log.warning("file library scan failed", user=uid, error=str(e))
        job.message = f"The scan failed: {str(e)[:200]}"
    finally:
        job.running, job.phase, job.current = False, "idle", ""
        job.finished_at = time.time()
        job.task = None


def start_scan(user: dict, retry_errors: bool = False) -> ScanJob:
    uid = str(user.get("id") or "")
    job = _job_for(uid)
    if job.running:
        return job
    job.running, job.phase = True, "scanning"
    job.total = job.done = job.failed = 0
    job.current = job.message = ""
    job.started_at, job.finished_at = time.time(), 0.0
    job.task = asyncio.create_task(run_scan(user, job, retry_errors))
    return job


# ── Rows → what the UI shows ──────────────────────────────────────────


def _item(row: dict, agents: dict[str, dict], terms: list[str] | None = None) -> dict:
    try:
        tags = json.loads(row.get("tags") or "[]")
    except (TypeError, ValueError):
        tags = []
    mt, _ = mimetypes.guess_type(row["filename"])
    source = source_fields(row["agent_slug"], agents)
    item = {
        "id": row["id"],
        **source,
        "rel_path": row["rel_path"],
        # Named as the Files panel names a file on disk (the agent-files
        # proxy's workspace scan), so the agent's view, save, deck and
        # public-link routes take a library file as they take one from there.
        # A shared folder's file is named as its agents name it.
        "physical": str(_workspace(row["agent_slug"]) / row["rel_path"]),
        "logical": (f"{FOLDER_PREFIX}{source['folder']}/{row['rel_path']}" if "folder" in source
                    else "workspace/" + row["rel_path"]),
        "filename": row["filename"],
        "extension": row["extension"],
        "kind": row["kind"],
        "mime_type": mt or "application/octet-stream",
        "size": row["size"],
        "modified": row["modified"],
        "status": row["status"],
        "title": row["title"],
        "summary": row["summary"],
        "category": row["category"],
        "tags": tags if isinstance(tags, list) else [],
        "error": row["error"],
        "model": row["model"],
        "summarized_at": row["summarized_at"],
        "added_at": row["added_at"],
        "snippet": "",
    }
    if terms:
        item["snippet"] = _snippet(row, terms)
    return item


def _unfolded(text: str, start: int, end: int) -> tuple[int, int]:
    """*start*..*end* in ``text.casefold()`` as positions in *text* (a
    letter like "ß" folds to two)."""
    folded, a = 0, None
    for j, ch in enumerate(text):
        folded += len(ch.casefold())
        if a is None and folded > start:
            a = j
        if folded >= end:
            return a if a is not None else j, j + 1
    return a if a is not None else len(text), len(text)


def _snippet(row: dict, terms: list[str]) -> str:
    """A line of the file's text around the first term the visible fields
    don't already show — why a search found it."""
    visible = " ".join(str(row.get(k) or "") for k in ("filename", "rel_path", "title", "summary", "tags", "category")).casefold()
    text = row.get("excerpt") or ""
    low = text.casefold()
    for t in terms:
        if t in visible:
            continue
        i = low.find(t)
        if i < 0:
            continue
        start, end = (i, i + len(t)) if len(low) == len(text) else _unfolded(text, i, i + len(t))
        a, b = max(0, start - 70), min(len(text), end + 110)
        return ("…" if a else "") + " ".join(text[a:b].split()) + ("…" if b < len(text) else "")
    return ""


_SORTS = {
    "recent": lambda r: -float(r["modified"] or 0),
    "name": lambda r: (r["title"] or r["filename"]).lower(),
    "size": lambda r: -int(r["size"] or 0),
    "agent": lambda r: (r["agent_slug"], -float(r["modified"] or 0)),
}


# ── Routes ────────────────────────────────────────────────────────────


def source_facets(counts: dict[str, int], sources: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """The agents and the shared folders a filter offers, the fullest first."""
    agents: list[dict] = []
    folders: list[dict] = []
    for key, n in counts.items():
        pid = public_id(key, sources)
        f = {"id": pid, "label": sources.get(key, {}).get("name") or pid, "count": n}
        (folders if folder_of(key) else agents).append(f)

    def order(x: dict) -> tuple:
        return -x["count"], x["label"].lower()
    return sorted(agents, key=order), sorted(folders, key=order)


async def _status_payload(user: dict) -> dict:
    sources = await caller_sources(user)
    job = _job_for(str(user.get("id") or ""))
    counts = await asyncio.to_thread(get_store().counts, list(sources))
    return {"job": job.view(), "counts": counts,
            "agents": sum(1 for k in sources if not folder_of(k)),
            "folders": sum(1 for k in sources if folder_of(k))}


@router.get("/status")
async def library_status(user: dict = Depends(get_current_user)) -> dict:
    """Where the caller's library stands: file counts and the running scan."""
    return await _status_payload(user)


# ── Adding a shared folder, or files of one ───────────────────────────

def picks_view(user: dict) -> list[dict]:
    """What the caller added: ``[{folder, path}]`` (``path`` '' = the whole folder)."""
    picked = get_store().picks(_vfs_owner(str(user.get("id") or "")))
    return [{"folder": f, "path": p} for f in sorted(picked) for p in sorted(picked[f])]


def _refresh_folder(user: dict, folder: str) -> None:
    """The folder's rows as its picks say now: what was taken out goes at once."""
    slug = f"{FOLDER_PREFIX}{_vfs_owner(str(user.get('id') or ''))}/{folder}"
    get_store().sync_agent(slug, walk_workspace(slug))


def add_pick(user: dict, folder: str, path: str = "") -> bool:
    """A whole folder of the caller's own (``path`` ''), or one file in it, into
    the File library. Raises HTTPException when it can't be."""
    from captain_claw.vfs import safe_join
    root = own_folder(user, folder)
    if root is None:
        raise HTTPException(404, "Only your own folders go in the library: not linked, mounted or shared ones.")
    path = str(path or "").strip().strip("/")
    if path:
        why = can_hold(path)
        if why:
            raise HTTPException(400, f"{Path(path).name} can't go in the library: {why}.")
        target = safe_join(root, path)
        if target is None or not target.is_file() or target.is_symlink():
            raise HTTPException(404, f"{Path(path).name} isn't in that folder.")
    added = get_store().add_pick(_vfs_owner(str(user.get("id") or "")), folder, path)
    _refresh_folder(user, folder)
    return added


def drop_pick(user: dict, folder: str, path: str = "") -> bool:
    path = str(path or "").strip().strip("/")
    dropped = get_store().drop_pick(_vfs_owner(str(user.get("id") or "")), folder, path)
    if dropped:
        _refresh_folder(user, folder)
    return dropped


@router.get("/picks")
async def library_picks(user: dict = Depends(get_current_user)) -> dict:
    """The shared folders, and single files of them, the caller added."""
    return {"picks": await asyncio.to_thread(picks_view, user)}


@router.post("/picks")
async def library_add_pick(body: dict, user: dict = Depends(get_current_user)) -> dict:
    """``{folder, path}``: into the library (``path`` '' = the whole folder);
    a scan summarises it."""
    folder, path = str((body or {}).get("folder") or ""), str((body or {}).get("path") or "")
    added = await asyncio.to_thread(add_pick, user, folder, path)
    start_scan(user)
    return {"added": added, "picks": await asyncio.to_thread(picks_view, user)}


@router.delete("/picks")
async def library_drop_pick(folder: str, path: str = "", user: dict = Depends(get_current_user)) -> dict:
    """Out of the library (the file stays in its folder)."""
    dropped = await asyncio.to_thread(drop_pick, user, folder, path)
    return {"dropped": dropped, "picks": await asyncio.to_thread(picks_view, user)}


@router.post("/scan")
async def library_scan(retry_errors: bool = False, user: dict = Depends(get_current_user)) -> dict:
    """Look for new and changed files and summarise them, in the background
    (``retry_errors``: the files whose summary failed too, now). A scan
    already running is left to finish."""
    start_scan(user, retry_errors)
    return await _status_payload(user)


@router.get("/files")
async def library_files(
    q: str = "", agent: str = "", category: str = "", kind: str = "",
    sort: str = "recent", limit: int = 60, offset: int = 0, section: str = "",
    user: dict = Depends(get_current_user),
) -> dict:
    """The caller's files, searched and filtered, with the counts each
    filter would leave (computed under the other filters). ``agent`` is an
    agent's slug or a shared folder (``vfs:<folder>``). The browser tool's
    screenshots are listed only with ``section=screenshots`` (and then only
    they, category and kind aside); ``sections`` counts both sides under
    the search and the agent filter."""
    agents = await caller_sources(user)
    agent = source_key(agent, agents) if agent else ""
    terms = [t for t in q.casefold().split() if t][:8]
    found = await asyncio.to_thread(get_store().search, list(agents), terms)
    shots = section == SCREENSHOTS
    if shots:
        category = kind = ""
    sections = {"files": 0, SCREENSHOTS: 0}
    rows: list[dict] = []
    for r in found:
        shot = is_agentic_screenshot(r["rel_path"])
        if not agent or r["agent_slug"] == agent:
            sections[SCREENSHOTS if shot else "files"] += 1
        if shot == shots:
            rows.append(r)

    def keep(r: dict, skip: str = "") -> bool:
        return ((skip == "agent" or not agent or r["agent_slug"] == agent)
                and (skip == "category" or not category or (r["category"] or "pending") == category)
                and (skip == "kind" or not kind or r["kind"] == kind))

    def facet(key: str, field_of) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            if keep(r, key):
                k = field_of(r)
                out[k] = out.get(k, 0) + 1
        return out

    agent_counts = facet("agent", lambda r: r["agent_slug"])
    category_counts = facet("category", lambda r: r["category"] or "pending")
    kind_counts = facet("kind", lambda r: r["kind"])
    shown = [r for r in rows if keep(r)]
    shown.sort(key=_SORTS.get(sort, _SORTS["recent"]))
    limit = max(1, min(limit, 1000))
    page = shown[max(0, offset):max(0, offset) + limit]
    agent_facets, folder_facets = source_facets(agent_counts, agents)
    return {
        "items": [_item(r, agents, terms) for r in page],
        "total": len(shown),
        "sections": sections,
        "facets": {
            "agents": agent_facets,
            "folders": folder_facets,
            "categories": [{"id": c, "label": label, "count": category_counts[c]}
                           for c, label in [*CATEGORIES, ("pending", "Not summarised yet")]
                           if category_counts.get(c)],
            "kinds": [{"id": k, "label": label, "count": kind_counts[k]}
                      for k, label in [*KINDS, ("other", "Other")] if kind_counts.get(k)],
        },
    }


async def _owned_row(file_id: str, user: dict) -> tuple[dict, dict[str, dict]]:
    """The caller's file (one of their agents' or their shared folders'),
    and their sources."""
    row = get_store().get(file_id)
    agents = await caller_sources(user)
    if row is None or row["agent_slug"] not in agents:
        raise HTTPException(404, "file not found")
    return row, agents


@router.post("/files/{file_id}/summarize")
async def library_resummarize(file_id: str, user: dict = Depends(get_current_user)) -> dict:
    """Summarise one file again, now."""
    row, agents = await _owned_row(file_id, user)
    uid = str(user.get("id") or "")
    if await _quota_spent(uid):
        raise HTTPException(429, "Today's token allowance is used up.")
    tier = await _resolve_tier(uid)
    if tier is None:
        raise HTTPException(409, "No model is set up to write summaries.")
    path = _resolve_path(row)
    current = _fingerprint(path) if path is not None else ""
    if current and current != row["fingerprint"]:
        # Changed since the last walk: summarised (and stored) as it is now.
        st = path.stat()
        get_store().set_version(file_id, int(st.st_size), float(st.st_mtime), current)
        row = get_store().get(file_id) or row
    row["agent_name"] = agents[row["agent_slug"]]["name"]
    try:
        result = await summarize_row(row, tier, uid)
    except Exception as e:  # noqa: BLE001
        get_store().save_result(row["id"], row["fingerprint"], status="error",
                                error=str(e)[:300], summarized_at=time.time())
        raise HTTPException(502, f"The model failed: {str(e)[:200]}")
    if result is None:
        raise HTTPException(503, "Soniox isn't answering — try again in a while.")
    get_store().save_result(row["id"], row["fingerprint"], **result)
    return _item(get_store().get(file_id) or row, agents)


async def _serve(file_id: str, user: dict, inline: bool):
    row, _agents = await _owned_row(file_id, user)
    path = _resolve_path(row)
    if path is None:
        raise HTTPException(404, "file not found")
    mt, _ = mimetypes.guess_type(row["filename"])
    headers = {"X-Content-Type-Options": "nosniff"}
    if inline:
        from captain_claw.web import active_content
        if active_content.active_view_type(mt or ""):
            headers["Content-Security-Policy"] = active_content.ACTIVE_VIEW_CSP
    return FileResponse(
        path, media_type=mt or "application/octet-stream", headers=headers,
        filename=row["filename"], content_disposition_type="inline" if inline else "attachment")


@router.get("/files/{file_id}/view")
async def library_view(file_id: str, user: dict = Depends(get_current_user)):
    """The file itself, inline — straight from the agent's folder on disk, so
    it opens whether the agent is running or not."""
    return await _serve(file_id, user, inline=True)


@router.get("/files/{file_id}/download")
async def library_download(file_id: str, user: dict = Depends(get_current_user)):
    return await _serve(file_id, user, inline=False)


# ── Thumbnails (the library's preview of a PDF) ──
#
# The first page of a PDF as a PNG: a browser shows a PDF inline only with
# its viewer plugin, which the desktop app (Electron) and most phones don't
# have. Rendered with pypdfium2 (the payments extra) when it is installed —
# without it the route says 404 and the preview offers Open instead. Kept
# for a while per file version, so going back to a file doesn't render again.
#
# One thread draws them (pdfium is serialised anyway): requests waiting for it
# queue as futures instead of each holding a thread of the shared pool, and
# everyone asking for the same file version waits on the same render. A
# preview gives up after _THUMB_TIMEOUT_S (a page can be built to take hours
# to draw, and pdfium can't be interrupted); a version that failed or was too
# slow is refused at once afterwards rather than queued again.

_THUMB_WIDTH = 640
_THUMB_TIMEOUT_S = 15.0
_THUMB_FAILED_MAX = 200
# A page longer (or wider) than this many times its other side shows only its
# top (or left) part: a crafted page a million points tall would otherwise
# render to gigabytes.
_THUMB_MAX_ASPECT = 2.0
_THUMB_CACHE_BYTES = 24 * 1024 * 1024
_thumbs: dict[tuple[str, int, int], bytes] = {}
_thumb_failed: dict[tuple[str, int, int], tuple[int, str]] = {}
_thumb_inflight: dict[tuple[str, int, int], asyncio.Future] = {}
_thumbs_lock = threading.Lock()
_thumb_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fd-pdf-thumb")
_THUMB_SLOW = "This PDF takes too long to draw for a preview."


def _pdf_first_page_png(path: Path) -> bytes | None:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return None
    import io

    from captain_claw.pdfium_lock import PDFIUM_LOCK

    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(str(path))
        try:
            if len(pdf) == 0:
                raise ValueError("it has no pages")
            page = pdf[0]
            width, height = page.get_size()
            if width <= 0 or height <= 0:
                raise ValueError("its first page has no size")
            # Points to cut off the bottom / right so the page is at most
            # _THUMB_MAX_ASPECT times as long as it is wide (and vice versa).
            cut_bottom = max(0.0, height - width * _THUMB_MAX_ASPECT)
            cut_right = max(0.0, width - height * _THUMB_MAX_ASPECT)
            shown_width = width - cut_right
            scale = min(2.0, _THUMB_WIDTH / shown_width)
            image = page.render(scale=scale, crop=(0, cut_bottom, cut_right, 0)).to_pil()
        finally:
            pdf.close()
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def _remember_failure(key: tuple[str, int, int], status: int, detail: str) -> None:
    with _thumbs_lock:
        _thumb_failed[key] = (status, detail)
        while len(_thumb_failed) > _THUMB_FAILED_MAX:
            _thumb_failed.pop(next(iter(_thumb_failed)))


def _thumb_settled(key: tuple[str, int, int], fut: asyncio.Future) -> None:
    """A render finished (perhaps after its requests gave up): keep the
    picture, or remember why there is none."""
    _thumb_inflight.pop(key, None)
    if fut.cancelled():
        return
    exc = fut.exception()
    if exc is not None:
        _remember_failure(key, 422, f"Couldn't read the PDF: {str(exc)[:200]}")
        return
    png = fut.result()
    if png is None:
        _remember_failure(key, 404, "no PDF renderer on this install")
        return
    with _thumbs_lock:
        _thumb_failed.pop(key, None)
        _thumbs[key] = png
        while len(_thumbs) > 1 and sum(len(v) for v in _thumbs.values()) > _THUMB_CACHE_BYTES:
            _thumbs.pop(next(iter(_thumbs)))


@router.get("/files/{file_id}/thumb")
async def library_thumb(file_id: str, user: dict = Depends(get_current_user)):
    """A PNG of a PDF's first page. 404 for anything else, when the file is
    gone, or when this install has no PDF renderer; 422 for a PDF that won't
    open (damaged, or locked with a password)."""
    row, _agents = await _owned_row(file_id, user)
    if not row["filename"].lower().endswith(".pdf"):
        raise HTTPException(404, "no thumbnail for this kind of file")
    path = _resolve_path(row)
    if path is None:
        raise HTTPException(404, "file not found")
    st = path.stat()
    return await pdf_thumb_response((file_id, int(st.st_mtime_ns), st.st_size), path)


async def pdf_thumb_response(key: tuple[str, int, int], path: Path) -> Response:
    """The PNG of *path*'s first page — cached under *key* (an id for the file
    plus its version), drawn once however many ask. The shared folders' thumb
    route (vfs_routes) uses it too."""
    with _thumbs_lock:
        png = _thumbs.get(key)
        failed = _thumb_failed.get(key)
    if png is None and failed:
        raise HTTPException(*failed)
    if png is None:
        fut = _thumb_inflight.get(key)
        if fut is None:
            fut = asyncio.wrap_future(_thumb_executor.submit(_pdf_first_page_png, path))
            fut.add_done_callback(lambda f, k=key: _thumb_settled(k, f))
            _thumb_inflight[key] = fut
        try:
            await asyncio.wait_for(asyncio.shield(fut), _THUMB_TIMEOUT_S)
        except asyncio.TimeoutError:
            _remember_failure(key, 422, _THUMB_SLOW)
            raise HTTPException(422, _THUMB_SLOW)
        except Exception as e:  # noqa: BLE001 — pdfium raises its own errors for damaged or locked files
            raise HTTPException(422, f"Couldn't read the PDF: {str(e)[:200]}")
        png = fut.result()
        if png is None:
            raise HTTPException(404, "no PDF renderer on this install")
    return Response(content=png, media_type="image/png", headers={
        "X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=300",
    })
