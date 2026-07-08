"""R1 — the Research Map: Code Map's blackboard, generalized to research folders.

A Basna/Vatra run accumulates prose artifacts (findings, drafts, sources) in one
shared VFS folder. Today every continuation round re-reads that whole folder and
the Vatra reporter inlines only a capped slice of it — the more a chain grows,
the more tokens are burned re-reading what's already there, and the reporter can
miss material past its cap.

This is the prose analogue of ``code_map``: a per-folder FTS index over the
research files (chunked by heading) plus a one-file-per-purpose table and an
``overview.md``. Workers query it with the ``researchmap`` tool to LOCATE a prior
claim/source instead of re-reading everything; the reporter searches it instead
of trusting an inline cap. Indexing is deterministic and free; it only pays off
(and only runs) when a run opts into ``quality.research_map``.

Storage lives under ``<project>/.researchmap/`` (``map.db`` + ``overview.md``).
Staleness is tracked by an mtime+size pseudo-hash so it works whether or not the
folder is a git repo (git snapshots are R6, and optional).
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from captain_claw.logging import get_logger

log = get_logger(__name__)

_MAP_DIRNAME = ".researchmap"
_SKIP_DIRS = {".git", ".code", ".codemap", ".researchmap", ".captain-claw",
              "saved", "node_modules", ".uploads", "__pycache__"}
_TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".json", ".csv", ".yaml", ".yml"}
_MAX_FILE_BYTES = 2_000_000
_CHUNK_CHARS = 1500       # target chunk size when a file has no headings
_MAX_CHUNKS_PER_FILE = 60

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def map_dir(project: Path | str) -> Path:
    d = Path(project) / _MAP_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _db(project: Path | str) -> sqlite3.Connection:
    con = sqlite3.connect(map_dir(project) / "map.db")
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS files(
        path TEXT PRIMARY KEY, sig TEXT, purpose TEXT, ts REAL)""")
    con.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
        path, heading, body)""")
    return con


# ── candidate discovery + staleness ──────────────────────────────────

def _candidate_files(project: Path) -> list[str]:
    out: list[str] = []
    for p in project.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        if any(part in _SKIP_DIRS for part in p.relative_to(project).parts):
            continue
        try:
            if p.stat().st_size > _MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        out.append(p.relative_to(project).as_posix())
    return out


def _sig(project: Path, rel: str) -> str:
    try:
        st = (project / rel).stat()
        return f"{int(st.st_mtime)}-{st.st_size}"
    except OSError:
        return ""


# ── chunking ─────────────────────────────────────────────────────────

def _chunk(text: str) -> list[tuple[str, str]]:
    """Split a document into (heading, body) chunks — by markdown heading when
    present, else by fixed-size windows. Bounded so a huge file can't explode."""
    chunks: list[tuple[str, str]] = []
    matches = list(_HEADING_RE.finditer(text))
    if matches:
        for i, m in enumerate(matches):
            heading = m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body or heading:
                chunks.append((heading, body[:_CHUNK_CHARS * 3]))
    else:
        for i in range(0, len(text), _CHUNK_CHARS):
            chunks.append(("", text[i:i + _CHUNK_CHARS]))
    return chunks[:_MAX_CHUNKS_PER_FILE]


def _first_meaningful_line(text: str) -> str:
    for line in text.splitlines():
        s = line.strip().lstrip("#").strip()
        if s:
            return s[:200]
    return ""


# ── reindex ──────────────────────────────────────────────────────────

def reindex(project: Path | str) -> dict:
    """(Re)index the folder's text files. Incremental: only re-chunk files whose
    mtime+size changed. Returns ``{files, changed, chunks}``. Never raises."""
    project = Path(project)
    try:
        con = _db(project)
    except Exception as e:  # noqa: BLE001
        log.warning("research_map: db open failed", error=str(e))
        return {"files": 0, "changed": [], "chunks": 0}
    changed: list[str] = []
    try:
        current = _candidate_files(project)
        known = {r["path"]: r["sig"] for r in con.execute("SELECT path, sig FROM files")}
        # Drop files that disappeared.
        gone = set(known) - set(current)
        for rel in gone:
            con.execute("DELETE FROM files WHERE path = ?", (rel,))
            con.execute("DELETE FROM chunks WHERE path = ?", (rel,))
        for rel in current:
            sig = _sig(project, rel)
            if known.get(rel) == sig and sig:
                continue  # unchanged
            changed.append(rel)
            try:
                text = (project / rel).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            con.execute("DELETE FROM chunks WHERE path = ?", (rel,))
            for heading, body in _chunk(text):
                con.execute("INSERT INTO chunks(path, heading, body) VALUES (?, ?, ?)",
                            (rel, heading, body))
            con.execute(
                "INSERT INTO files(path, sig, purpose, ts) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(path) DO UPDATE SET sig=excluded.sig, ts=excluded.ts",
                (rel, sig, _first_meaningful_line(text), 0.0))
        con.commit()
        nchunks = con.execute("SELECT count(*) c FROM chunks").fetchone()["c"]
        return {"files": len(current), "changed": changed, "chunks": nchunks}
    except Exception as e:  # noqa: BLE001
        log.warning("research_map: reindex failed", error=str(e))
        return {"files": 0, "changed": changed, "chunks": 0}
    finally:
        con.close()


# ── query ────────────────────────────────────────────────────────────

def _fts_query(q: str) -> str:
    """Make a forgiving FTS5 MATCH string: OR the bare terms, prefix-match each."""
    terms = re.findall(r"[A-Za-z0-9_]+", q)
    if not terms:
        return ""
    return " OR ".join(f'{t}*' for t in terms[:12])


def search(project: Path | str, query: str, limit: int = 15) -> list[dict]:
    project = Path(project)
    match = _fts_query(query)
    if not match:
        return []
    try:
        con = _db(project)
    except Exception:  # noqa: BLE001
        return []
    try:
        rows = con.execute(
            "SELECT path, heading, snippet(chunks, 2, '', '', ' … ', 18) AS snip "
            "FROM chunks WHERE chunks MATCH ? ORDER BY rank LIMIT ?",
            (match, int(limit or 15)),
        ).fetchall()
        return [{"path": r["path"], "heading": r["heading"], "snippet": r["snip"]} for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        con.close()


def stats(project: Path | str) -> dict:
    try:
        con = _db(project)
    except Exception:  # noqa: BLE001
        return {"files": 0, "chunks": 0}
    try:
        f = con.execute("SELECT count(*) c FROM files").fetchone()["c"]
        c = con.execute("SELECT count(*) c FROM chunks").fetchone()["c"]
        return {"files": f, "chunks": c}
    finally:
        con.close()


def list_files(project: Path | str) -> list[dict]:
    try:
        con = _db(project)
    except Exception:  # noqa: BLE001
        return []
    try:
        return [{"path": r["path"], "purpose": r["purpose"]}
                for r in con.execute("SELECT path, purpose FROM files ORDER BY path")]
    finally:
        con.close()


# ── overview (cartographer-written semantic layer) ────────────────────

def read_overview(project: Path | str) -> str:
    p = map_dir(project) / "overview.md"
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return ""


def write_overview(project: Path | str, text: str) -> None:
    (map_dir(project) / "overview.md").write_text(text or "", encoding="utf-8")


def preamble(project: Path | str) -> str:
    """A prompt preamble that points workers at the map instead of re-reading.

    Empty when the folder has nothing indexed yet, so it never adds noise to a
    fresh run's first round."""
    try:
        st = stats(project)
    except Exception:  # noqa: BLE001
        return ""
    if not st.get("chunks"):
        return ""
    ov = read_overview(project).strip()
    head = (
        "## Research Map available (search it before re-reading files)\n"
        f"This shared folder is indexed ({st['files']} files, {st['chunks']} sections). "
        "Use the `researchmap` tool to find prior findings/sources fast: "
        "`overview`, `search <query>`, `files`, `file <path>`. It returns pointers and "
        "snippets — open the actual file only when you need the full passage. Do NOT "
        "re-read the whole folder; build on what's already there.\n"
    )
    if ov:
        head += "\n### What's been established so far\n" + ov[:2500] + "\n"
    return head + "\n---\n\n"
