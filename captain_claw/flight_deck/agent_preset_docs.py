"""The documents of an agent's "way of working" setup (see agent_preset).

A document is read in full the moment it is added — uploaded, or picked from
the person's File library — and waits here ("staged") until the setup reads
it, so the dialog can say how many pages it has and what reading it in full
costs before anything is spent. Its whole text reaches the agent's knowledge,
split into parts its semantic memory indexes (it skips files over 256 KB);
the model reads it in chunks sized to its context window (``split_text``
serves both). A spreadsheet (.xlsx, .csv) is kept as it came, too: its
sheets become tables in the agent's datastore (agent_preset_tables).
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import shutil
import tempfile
import time
from pathlib import Path

from captain_claw.flight_deck import agent_preset_tables as ptables
from captain_claw.flight_deck.agent_preset_outline import outline
from captain_claw.logging import get_logger

log = get_logger(__name__)

SUPPORTED = (".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md", ".csv")
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_CHARS = 6_000_000          # a document's text, at most (some 2,000 pages)
CHARS_PER_PAGE = 3000          # a page of a document without pages, roughly
CHARS_PER_TOKEN = 4
TTL = 24 * 3600                # a staged document nobody read waits this long
_MAX_PAGES = 3000
_MAX_ROWS = 20000
_MAX_SLIDES = 1000
_PLAIN = (".txt", ".md", ".csv")


# ── Reading a document in full ────────────────────────────────────────


def extract(path: Path) -> tuple[str, str]:
    """The whole text of the document at *path* as Markdown (pages, slides
    and sheets under their own ``##`` headings), or "" and why not."""
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        return "", "not a kind of document the setup reads"
    try:
        if ext in _PLAIN:
            with path.open("rb") as fh:
                return fh.read(MAX_CHARS * 2).decode("utf-8", errors="replace"), ""
        from captain_claw.tools import document_extract as dx
        if ext == ".pdf":
            text, err = dx._extract_pdf_markdown(path, max_pages=_MAX_PAGES)
            if err:
                return "", err
            if "_(No extractable text found in selected pages.)_" in (text or ""):
                return "", "no text in it (a scan or an image-only file)"
        elif ext == ".docx":
            text = dx._extract_docx_markdown(path)
        elif ext == ".xlsx":
            text = dx._extract_xlsx_markdown(path, max_rows=_MAX_ROWS)
        else:
            text = dx._extract_pptx_markdown(path, max_slides=_MAX_SLIDES)
        return text or "", ""
    except Exception as exc:
        return "", str(exc)


def extract_bytes(raw: bytes, suffix: str) -> tuple[str, str]:
    """``extract`` for an upload."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".txt") as fh:
            fh.write(raw)
            tmp = Path(fh.name)
        return extract(tmp)
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def pages_of(text: str) -> tuple[int, bool]:
    """How many pages (or slides) the text has, and whether that is counted
    (a PDF's pages, a deck's slides) or estimated from its length."""
    counted = len(re.findall(r"(?m)^## (?:Page|Slide) \d+$", text))
    if counted:
        return counted, True
    return max(1, math.ceil(len(text) / CHARS_PER_PAGE)), False


def tokens_of(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN)


# ── Splitting ─────────────────────────────────────────────────────────


_HEADING = re.compile(r"(?m)^#{1,3} ")
_MARK = re.compile(r"(?m)^## (Page|Slide) (\d+)$")


def _size(s: str) -> int:
    return len(s.encode("utf-8"))


def _pieces(block: str, limit: int) -> list[str]:
    """*block* in pieces of at most *limit* bytes: whole when it fits, else
    at paragraph breaks, else cut."""
    if _size(block) <= limit:
        return [block]
    out: list[str] = []
    cur, cur_size = "", 0
    for para in re.split(r"(?<=\n\n)", block):
        size = _size(para)
        if size > limit:
            if cur:
                out.append(cur)
                cur, cur_size = "", 0
            step = max(1, limit // 4)  # 4 bytes per character at most
            out += [para[i:i + step] for i in range(0, len(para), step)]
            continue
        if cur and cur_size + size > limit:
            out.append(cur)
            cur, cur_size = "", 0
        cur += para
        cur_size += size
    if cur:
        out.append(cur)
    return out


def _label(text: str, last: tuple[str, int] | None) -> tuple[str, tuple[str, int] | None]:
    marks = [(m.group(1), int(m.group(2))) for m in _MARK.finditer(text)]
    if marks:
        kind, first = marks[0]
        # A part that starts mid-page continues the page before it.
        if last and not text.lstrip().startswith(f"## {kind}"):
            first = last[1]
        end = marks[-1][1]
        noun = "pages" if kind == "Page" else "slides"
        label = f"{noun[:-1]} {first}" if first == end else f"{noun} {first}–{end}"
        return label, (kind, end)
    if last:
        return f"{'page' if last[0] == 'Page' else 'slide'} {last[1]}", last
    heading = next((ln.lstrip("# ").strip() for ln in text.splitlines() if _HEADING.match(ln)), "")
    return heading[:60], None


def split_text(text: str, limit: int) -> list[dict]:
    """*text* in parts of at most *limit* bytes (UTF-8), cut at headings —
    a PDF's pages, a deck's slides, a sheet, a Word heading — where it can
    be. Each part: ``text`` and a ``label`` ("pages 81–160", or its first
    heading)."""
    starts = sorted({0, *(m.start() for m in _HEADING.finditer(text))})
    blocks = [text[a:b] for a, b in zip(starts, [*starts[1:], len(text)])]
    parts: list[str] = []
    cur, cur_size = "", 0
    for block in blocks:
        for piece in _pieces(block, limit):
            size = _size(piece)
            if cur and cur_size + size > limit:
                parts.append(cur)
                cur, cur_size = "", 0
            cur += piece
            cur_size += size
    if cur.strip():
        parts.append(cur)
    out: list[dict] = []
    last = None
    for p in parts:
        label, last = _label(p, last)
        out.append({"text": p, "label": label})
    return out


# ── Staged documents ──────────────────────────────────────────────────


def _id_ok(doc_id: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{24}", str(doc_id or "")))


def _staged_dir(root: Path, doc_id: str) -> Path | None:
    return root / "staged" / doc_id if _id_ok(doc_id) else None


def public(meta: dict) -> dict:
    """What the dialog sees of a staged document."""
    out = {k: meta[k] for k in ("id", "name", "origin", "pages", "pages_exact", "chars", "tokens", "truncated")}
    out["sections"] = int(meta.get("sections") or 0)
    if meta.get("agent"):
        out["agent"] = meta["agent"]
    if meta.get("source"):
        out["tables"] = int(meta.get("tables") or 0)
    return out


def stage(root: Path, name: str, text: str, origin: str, agent: str = "", raw: bytes | None = None) -> dict:
    """Keep a document's text under *root* (the owner's folder) until the
    setup reads it — a spreadsheet's own bytes (*raw*) too, for its tables —
    and its public description (a spreadsheet's: how many tables it gives)."""
    truncated = len(text) > MAX_CHARS
    text = text[:MAX_CHARS]
    pages, exact = pages_of(text)
    doc_id = secrets.token_hex(12)
    folder = _staged_dir(root, doc_id)
    assert folder is not None
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "text.txt").write_text(text, encoding="utf-8")
    meta = {"id": doc_id, "name": name, "origin": origin, "created": time.time(),
            "pages": pages, "pages_exact": exact, "chars": len(text), "tokens": tokens_of(len(text)),
            "sections": len(outline(text.strip())),
            "truncated": truncated, **({"agent": agent} if agent else {})}
    if raw and ptables.is_spreadsheet(name):
        source = folder / f"source{Path(name).suffix.lower()}"
        source.write_bytes(raw)
        meta.update(source=source.name, tables=ptables.count_tables(source))
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return public(meta)


def load_staged(root: Path, doc_id: str) -> dict | None:
    """A staged document (its meta, ``text``, and a spreadsheet's
    ``source`` file); None when it is gone, expired, or not under *root*."""
    folder = _staged_dir(root, doc_id)
    if folder is None or not (folder / "meta.json").is_file():
        return None
    try:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        if time.time() - float(meta.get("created") or 0) > TTL:
            return None
        source = folder / str(meta.get("source") or "")
        ok = bool(re.fullmatch(r"source\.(xlsx|csv)", source.name)) and source.is_file()
        return {**meta, "text": (folder / "text.txt").read_text(encoding="utf-8"),
                "source": str(source) if ok else ""}
    except Exception:
        log.warning("agent preset: could not read a staged document", doc=doc_id, exc_info=True)
        return None


def prune_staged(root: Path, now: float) -> int:
    """Drop the staged documents under one owner's *root* that waited a day."""
    gone = 0
    staged = root / "staged"
    if not staged.is_dir():
        return 0
    for folder in staged.iterdir():
        try:
            created = json.loads((folder / "meta.json").read_text(encoding="utf-8")).get("created")
        except Exception:
            created = folder.stat().st_mtime if folder.exists() else now
        if now - float(created or 0) > TTL:
            shutil.rmtree(folder, ignore_errors=True)
            gone += 1
    return gone
