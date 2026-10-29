"""Knowledge tool — find things in the documents the agent's owner gave it.

Flight Deck's setup (flight_deck/agent_preset) puts each document whole in
the workspace's knowledge/ folder: one file, or an overview and its parts
(``<name>-part-NN.md``), with a section map (``<name>.map.md``), all listed
in INDEX.md. Semantic memory indexes them, but only its passive note
reached that index: three short snippets a turn, searched with the user's
words, among past conversations. This tool searches the knowledge alone,
with the agent's own words — keywords and meaning together, the exact code,
amount or phrase first — and gives each hit as a whole passage with its
document, section and lines; it reads a whole section by its title, and a
document's outline. No model is called. Without the index (semantic memory
off, or not caught up yet) it reads the files itself.

It also takes new documents in (``add``): a file the user hands the agent in
a chat — PDF, Word, Excel, PowerPoint, CSV, Markdown, text, HTML — goes into
its knowledge as a setup would put it there (Flight Deck writes it: parts, a
section map, a spreadsheet's tables, the index), not read for rules; a
picture or a scan the agent reads first and adds as text.

The owner's documents are the owner's: a shared agent's member gets no
knowledge tool (``speaker.allowed_tools``), and a member turn is refused.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from captain_claw import retrieval
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

FOLDER = "knowledge"
PASSAGE_CHARS = 1800     # one search hit, as a passage
SECTION_CHARS = 12_000   # one page of a section
OUTLINE_ENTRIES = 400
MAX_RESULTS = 10
_GROW_LINES = 12         # a paragraph, at most this many lines each way of a line
_WEAKEST = 0.5           # a hit under this share of the best one's score is left out
# INDEX.md's lines (agent_preset.write_knowledge): "- [Title](file.md)", then " — summary", " (how it is
# used)", " — N parts", " — section map: name.map.md", each when it has one.
_INDEX_LINE = re.compile(r"^- \[(?P<title>.+?)\]\((?P<file>[^)\s]+\.md)\)(?P<rest>.*)$")
_NOT_DOCS = {"INDEX.md", "glossary.md", "owner-rules.md"}
_EXTRAS = ("glossary.md", "owner-rules.md")   # searched too, though not documents
_TEXT_EXTS = {".md", ".txt"}
_QUOTED = re.compile(r"[\"“”']([^\"“”']{3,})[\"“”']")
_ADDABLE = (".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md", ".csv", ".html", ".htm")
_PICTURES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic")
_SCAN_PAGES = 12          # a scanned PDF's pages turned into pictures for the agent to read, at most
_USES = ("auto", "reference", "rules", "procedures", "templates")
# What a chat upload's saved name adds to the user's (web/rest_file_upload): "-20261028-215500-a1b2c3".
_UPLOAD_STAMP = re.compile(r"-\d{8}-\d{6}-[0-9a-f]{6}(?=\.[A-Za-z0-9]+$)")


@dataclass
class _Doc:
    """A document in knowledge/: its title and the files its text is in,
    in order (one, or its parts); its text and sections once read."""
    title: str
    file: str
    summary: str = ""
    files: list[str] = field(default_factory=list)
    text: str = ""
    starts: dict[str, tuple[int, int]] = field(default_factory=dict)  # file → (offset in text, file line of its body)
    sections: list[dict] = field(default_factory=list)
    loaded: bool = False


def _body_start(raw: str) -> int:
    """Where a setup file's text begins: after the head it was given
    (``# Title`` … ``---``); a file of one's own is all text."""
    if raw.startswith("# "):
        sep = raw.find("\n---\n", 0, 4000)
        if sep >= 0:
            at = sep + len("\n---\n")
            while raw.startswith("\n", at):
                at += 1
            return at
    return 0


def _norm(s: str) -> str:
    return " ".join(str(s or "").casefold().split())


def _marks(query: str) -> list[str]:
    """The query's exact things — a code, an amount, a name in capitals
    (``LITELLM_PROXY_URL``, ``14``, ``RMA``) — that a passage holding them
    should rank for, above passages that only share its words."""
    return [t for t in re.findall(r"[\w./-]+", query)
            if re.search(r"\d|_", t) or re.fullmatch(r"[A-Z]{2,}[\w-]*", t)]


class KnowledgeTool(Tool):
    """Search and read the documents in the workspace's knowledge/ folder."""

    name = "knowledge"
    timeout_seconds = 300.0  # adding a long PDF; a long document's first read
    description = (
        "Search and read the documents your owner gave you — your knowledge/ folder (the setup's documents, "
        "however long) — and keep new ones there. Use it before you answer from memory, and when you need the "
        "exact wording, number or steps. The rules, skills, facts and terms your owner's setup gave you are a "
        "short version that names the document and section each came from: read that section here to check "
        "one or get its details. "
        "Actions: 'search' (query, optionally one doc: the best passages, each with its document, "
        "section and lines; put an exact phrase or code in quotes to require it), 'section' (doc + section "
        "title: that section's whole text), 'outline' (doc: its sections; with section, just that part), "
        "'list' (the documents), 'add' (when the user gives you a file and asks you to keep it, learn it or "
        "put it in your knowledge: path — PDF, Word, Excel, PowerPoint, CSV, Markdown, text or HTML; for a "
        "picture, screenshot or scan read its text first and pass it as text with a name; optional title and "
        "note on what it is. It is searchable at once; it isn't read for rules or skills — your owner can do "
        "that in Way of working)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["search", "section", "outline", "list", "add"],
                       "description": "search = find passages; section = read one section whole; "
                                      "outline = a document's sections; list = the documents; "
                                      "add = keep a file (or a picture's text) in your knowledge."},
            "query": {"type": "string", "description": "What to find, in your own words (search)."},
            "doc": {"type": "string",
                    "description": "A document's title or file name: search only it, or read its section/outline."},
            "section": {"type": "string", "description": "A section's title, as the outline or a search hit gives it."},
            "page": {"type": "integer", "description": "A long section's page (section; default 1)."},
            "limit": {"type": "integer", "description": f"Passages for search (default 5, at most {MAX_RESULTS})."},
            "path": {"type": "string", "description": "add: the file to keep (one the user gave you)."},
            "text": {"type": "string", "description": "add: the text to keep instead of a file — what a picture "
                                                      "or scan says, as you read it (Markdown)."},
            "name": {"type": "string", "description": "add: its file name (with text: required, e.g. receipt.md; "
                                                      "with a path: to keep it under another name)."},
            "title": {"type": "string", "description": "add: its title (default: from its name)."},
            "note": {"type": "string", "description": "add: what it is, in a sentence, as the user put it."},
            "use": {"type": "string", "enum": list(_USES),
                    "description": "add: how to use it — reference (look things up), rules, procedures, "
                                   "templates, or auto."},
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        super().__init__()
        self._docs: dict[str, tuple[tuple, _Doc]] = {}   # file → (its files' signature, the document)

    # ── Where the knowledge is ────────────────────────────────────────

    def _folder(self) -> Path | None:
        agent = getattr(self, "_agent", None)
        semantic = getattr(getattr(agent, "memory", None), "semantic", None)
        for base in (getattr(semantic, "workspace_path", None), getattr(agent, "workspace_base_path", None)):
            if base and (Path(base) / FOLDER).is_dir():
                return Path(base) / FOLDER
        return None

    def has_knowledge(self) -> bool:
        """Whether the tool is any use (the registry shows it only then):
        under Flight Deck always — it takes documents in — else when there is
        something to search."""
        from captain_claw import fd_client

        if fd_client.is_under_flight_deck():
            return True
        folder = self._folder()
        try:
            return folder is not None and any(p.suffix in _TEXT_EXTS for p in folder.iterdir())
        except OSError:
            return False

    def documents(self, folder: Path) -> list[_Doc]:
        """The documents in *folder* (``_documents``), for the knowledge browser."""
        return self._documents(folder)

    def load(self, folder: Path, doc: _Doc) -> _Doc:
        """A document's text and sections (``_load``), for the knowledge browser."""
        return self._load(folder, doc)

    def where(self, doc: _Doc, offset: int) -> tuple[str, int]:
        """The file and line of *offset* in *doc*'s text."""
        return self._where(doc, offset)

    def _documents(self, folder: Path) -> list[_Doc]:
        """The documents: INDEX.md's, in its order, then any other file put there."""
        listed: list[_Doc] = []
        try:
            index = (folder / "INDEX.md").read_text(encoding="utf-8")
        except OSError:
            index = ""
        for line in index.splitlines():
            m = _INDEX_LINE.match(line.strip())
            if not m or not (folder / m["file"]).is_file():
                continue
            stem = Path(m["file"]).stem
            parts = sorted(folder.glob(f"{stem}-part-*.md"))
            rest = re.sub(r" — section map: \S+$", "", m["rest"].rstrip())
            rest = re.sub(r" \([^()]*\)$", "", re.sub(r" — \d+ parts$", "", rest))
            listed.append(_Doc(title=m["title"], file=m["file"], summary=rest[3:] if rest.startswith(" — ") else "",
                               files=[p.name for p in parts] or [m["file"]]))
        owned = {f for d in listed for f in (d.file, *d.files)}
        for p in sorted(folder.iterdir()):
            if (p.suffix in _TEXT_EXTS and p.is_file() and p.name not in owned and p.name not in _NOT_DOCS
                    and not p.name.endswith(".map.md") and not re.search(r"-part-\d+\.md$", p.name)):
                listed.append(_Doc(title=p.stem, file=p.name, files=[p.name]))
        return [self._cached(folder, d) for d in listed]

    def _cached(self, folder: Path, doc: _Doc) -> _Doc:
        sig = (doc.title, doc.summary,
               *((f, (folder / f).stat().st_mtime_ns) for f in doc.files if (folder / f).is_file()))
        hit = self._docs.get(doc.file)
        if hit and hit[0] == sig:
            return hit[1]
        self._docs[doc.file] = (sig, doc)
        return doc

    def _load(self, folder: Path, doc: _Doc) -> _Doc:
        """The document's text, from its files, and its sections."""
        if doc.loaded:
            return doc
        from captain_claw.flight_deck.agent_preset_outline import outline

        pieces: list[str] = []
        at = 0
        for name in doc.files:
            try:
                raw = (folder / name).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            start = _body_start(raw)
            body = raw[start:].rstrip()
            doc.starts[name] = (at, raw.count("\n", 0, start) + 1)
            pieces.append(body)
            at += len(body) + 2
        doc.text = "\n\n".join(pieces)
        parents: list[dict] = []
        for s in outline(doc.text):
            while parents and parents[-1]["level"] >= s["level"]:
                parents.pop()
            doc.sections.append({**s, "path": [p["title"] for p in parents]})
            parents.append(s)
        doc.loaded = True
        return doc

    def _find_doc(self, docs: list[_Doc], name: str) -> _Doc | None:
        want = _norm(Path(name).stem if name.lower().endswith(".md") else name)
        if not want:
            return None
        # A part's own file (a search hit names "knowledge/<doc>-part-3.md") is its document too.
        for test in (lambda d: _norm(d.title) == want or _norm(Path(d.file).stem) == want
                     or any(_norm(Path(f).stem) == want for f in d.files),
                     lambda d: _norm(d.title).startswith(want) or _norm(d.file).startswith(want),
                     lambda d: want in _norm(d.title) or want in _norm(d.file)):
            found = [d for d in docs if test(d)]
            if found:
                return found[0]
        return None

    # ── Positions ─────────────────────────────────────────────────────

    def _offset(self, doc: _Doc, file: str, line: int) -> int | None:
        """Where *line* (1-based) of *file* is in the document's text."""
        if file not in doc.starts:
            return None
        at, first = doc.starts[file]
        rel = max(0, line - first)
        pos, n = at, 0
        while n < rel:
            nxt = doc.text.find("\n", pos)
            if nxt < 0:
                break
            pos, n = nxt + 1, n + 1
        return pos

    def _where(self, doc: _Doc, offset: int) -> tuple[str, int]:
        """The file and line (1-based) of *offset* in the document."""
        file = next((f for f, (at, _) in sorted(doc.starts.items(), key=lambda kv: -kv[1][0]) if at <= offset),
                    doc.files[0])
        at, first = doc.starts.get(file, (0, 1))
        return file, first + doc.text.count("\n", at, offset)

    @staticmethod
    def _section_at(doc: _Doc, offset: int) -> dict | None:
        """The deepest section *offset* is in."""
        best = None
        for s in doc.sections:
            if s["start"] <= offset < s["end"] and (best is None or s["level"] >= best["level"]):
                best = s
        return best

    # ── Actions ───────────────────────────────────────────────────────

    async def execute(self, action: str, query: str | None = None, doc: str | None = None,
                      section: str | None = None, page: int | None = None, limit: int | None = None,
                      path: str | None = None, text: str | None = None, name: str | None = None,
                      title: str | None = None, note: str | None = None, use: str | None = None,
                      **kwargs: Any) -> ToolResult:
        from captain_claw import speaker

        if speaker.member_bound():
            return ToolResult(success=False, error="The owner's documents aren't available in this conversation.")
        if action == "add":
            try:
                return await self._add(path or "", text or "", name or "", title or "", note or "",
                                       use or "auto", kwargs.get("_runtime_base_path"))
            except Exception as exc:
                log.warning("knowledge add failed", exc_info=True)
                return ToolResult(success=False, error=f"Couldn't add it to your knowledge: {exc}")
        folder = self._folder()
        if folder is None:
            return ToolResult(success=False, error="You have no knowledge/ folder: your owner hasn't given you documents.")
        docs = await asyncio.to_thread(self._documents, folder)
        try:
            if action == "list":
                return ToolResult(success=True, content=self._list(docs))
            if action == "search":
                if not (query or "").strip():
                    return ToolResult(success=False, error="'query' is required for search.")
                return ToolResult(success=True, content=await self._search(
                    folder, docs, query.strip(), doc or "", max(1, min(MAX_RESULTS, int(limit or 5)))))
            if action in ("section", "outline"):
                target = self._find_doc(docs, doc or "")
                if target is None:
                    return ToolResult(success=False, error=(f"No document matches {doc!r}. " if doc else
                                                            "Name a document ('doc'). ") + self._list(docs))
                await asyncio.to_thread(self._load, folder, target)
                if action == "outline":
                    return ToolResult(success=True, content=self._outline(target, section or ""))
                if not (section or "").strip():
                    return ToolResult(success=False, error="'section' is required — a title from the outline.")
                return ToolResult(success=True, content=self._section(target, section.strip(), int(page or 1)))
            return ToolResult(success=False, error=f"Unknown action {action!r}: search, section, outline, list or add.")
        except Exception as exc:
            log.warning("knowledge tool failed", action=action, error=str(exc), exc_info=True)
            return ToolResult(success=False, error=f"Knowledge {action} failed: {exc}")

    # ── Taking a document in ──────────────────────────────────────────

    async def _add(self, path: str, text: str, name: str, title: str, note: str, use: str,
                   runtime_base: Any) -> ToolResult:
        """Keep a file the user gave (or the text of a picture the agent
        read) in the agent's knowledge: Flight Deck writes it, as a setup
        would, not read for rules."""
        from captain_claw import fd_client

        if not fd_client.is_under_flight_deck():
            return ToolResult(success=False, error="Adding to your knowledge works when Flight Deck runs you.")
        use = use if use in _USES else "auto"
        raw = b""
        sent_as = ""
        if text.strip():
            if not name.strip():
                return ToolResult(success=False, error="Give it a name (e.g. receipt.md) to keep the text under.")
            label = name
        elif path.strip():
            from captain_claw.tools.document_extract import _resolve_readable_file

            file_path, error = await _resolve_readable_file(path, runtime_base_path=runtime_base)
            if error or file_path is None:
                return ToolResult(success=False, error=error or f"Can't read {path}.")
            ext = file_path.suffix.lower()
            if ext in _PICTURES:
                return ToolResult(success=False, error=(
                    f"{file_path.name} is a picture: knowledge keeps text. Read its text first (look at it — "
                    f"image_ocr shows it to you), then call knowledge with action=add, text=<all of its text, as "
                    f"Markdown>, name=\"{file_path.stem}.md\"" + (f", title=\"{title}\"" if title else "") + "."))
            if ext not in _ADDABLE:
                return ToolResult(success=False, error=(
                    f"{file_path.name}: knowledge takes {', '.join(_ADDABLE)} — or the text of a picture or scan."))
            raw = await asyncio.to_thread(file_path.read_bytes)
            label = name.strip() or _UPLOAD_STAMP.sub("", file_path.name)
            if not Path(label).suffix:
                label += ext
            sent_as = _UPLOAD_STAMP.sub("", file_path.name)   # what it is: Flight Deck reads it by its own kind
        else:
            return ToolResult(success=False, error="Say which file to keep (path) — or the text, with a name.")

        import httpx

        data = {"name": label, "title": title, "note": note, "use": use, **({} if raw else {"text": text})}
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_seconds - 10, connect=10.0),
                                     trust_env=False) as client:
            resp = await client.post(f"{fd_client.flight_deck_base()}/fd/agent-knowledge/agent/add",
                                     data=data, files={"file": (sent_as, raw, "application/octet-stream")} if raw else None,
                                     headers=fd_client.flight_deck_headers())
        try:
            body = resp.json()
        except ValueError:
            body = {}
        if resp.status_code == 422 and isinstance(body.get("detail"), dict) and body["detail"].get("reason") == "scan":
            return await self._scan_pages(path, label, title, runtime_base)
        if resp.status_code != 200 or not isinstance(body, dict) or "doc" not in body:
            detail = body.get("detail") if isinstance(body, dict) else ""
            detail = detail.get("detail") if isinstance(detail, dict) else detail
            return ToolResult(success=False, error=f"Flight Deck didn't keep it: {detail or resp.status_code}")
        return ToolResult(success=True, content=self._added(body))

    @staticmethod
    def _added(b: dict) -> str:
        bits = [f"{int(b.get('chars') or 0):,} characters"]
        if int(b.get("parts") or 1) > 1:
            bits.append(f"in {b['parts']} parts" + (f" with a map of its {b['sections']} sections" if b.get("sections") else ""))
        out = (f"Kept “{b.get('title')}” ({b.get('doc')}) in your knowledge"
               + (" in place of its earlier copy" if b.get("replaced") else "") + f": {', '.join(bits)}.")
        if b.get("truncated"):
            out += " It was very long, so only its start is kept."
        if b.get("tables"):
            out += f" Its sheets are also tables you can query (datastore): {', '.join(b['tables'])}."
        out += (" It's searchable now — knowledge search finds it. It isn't read for rules or skills; your owner "
                "can do that in Way of working.")
        for w in b.get("warnings") or []:
            out += f" ({w})"
        return out

    async def _scan_pages(self, path: str, label: str, title: str, runtime_base: Any) -> ToolResult:
        """A PDF with no text — a scan: its first pages as pictures in the
        workspace, for the agent to read and add as text."""
        from captain_claw.tools.document_extract import _resolve_readable_file

        file_path, _ = await _resolve_readable_file(path, runtime_base_path=runtime_base)
        agent = getattr(self, "_agent", None)
        base = Path(runtime_base or getattr(agent, "workspace_base_path", "") or ".")
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(label).stem).strip("-.")[:60] or "scan"
        folder = base / "scans" / stem

        def render() -> tuple[list[Path], int]:
            import pypdfium2 as pdfium

            from captain_claw.pdfium_lock import PDFIUM_LOCK

            folder.mkdir(parents=True, exist_ok=True)
            out: list[Path] = []
            with PDFIUM_LOCK:
                doc = pdfium.PdfDocument(str(file_path))
                try:
                    total = len(doc)
                    for n in range(min(total, _SCAN_PAGES)):
                        target = folder / f"page-{n + 1:02d}.png"
                        doc[n].render(scale=2.0).to_pil().save(target)
                        out.append(target)
                finally:
                    doc.close()
            return out, total

        then = (f"Read each page (image_ocr shows it to you), then keep what they say: knowledge with action=add, "
                f"text=<all of it, as Markdown, page by page>, name=\"{stem}.md\""
                + (f", title=\"{title}\"" if title else "") + ".")
        if file_path is None:
            return ToolResult(success=False, error=f"{label} is a scan — there's no text in it to keep.")
        try:
            pages, total = await asyncio.to_thread(render)
        except ImportError:
            return ToolResult(success=False, error=(
                f"{label} is a scan — it has no text, and this Flight Deck can't turn its pages into pictures. "
                "Ask the user for its pages as pictures, or its text."))
        except Exception as exc:
            return ToolResult(success=False, error=f"{label} is a scan, and its pages couldn't be read: {exc}")
        shown = "\n".join(f"- {p.relative_to(base) if p.is_relative_to(base) else p}" for p in pages)
        more = f" (its first {len(pages)} of {total})" if total > len(pages) else ""
        return ToolResult(success=False, error=(
            f"{label} is a scan — there's no text in it to keep. Its pages{more} are pictures now:\n{shown}\n{then}"))

    def _list(self, docs: list[_Doc]) -> str:
        if not docs:
            return "Your knowledge/ folder has no documents."
        lines = [f"Your knowledge — {len(docs)} document{'s' if len(docs) != 1 else ''}:"]
        for d in docs:
            parts = f" ({len(d.files)} parts)" if len(d.files) > 1 else ""
            lines.append(f"- {d.title} — knowledge/{d.file}{parts}" + (f": {d.summary}" if d.summary else ""))
        return "\n".join(lines)

    def _outline(self, doc: _Doc, under: str) -> str:
        sections = doc.sections
        if under:
            top = self._pick_section(doc, under)
            if top is None:
                return f"No section of {doc.title} matches {under!r}."
            sections = [s for s in doc.sections if top["start"] <= s["start"] < top["end"]]
        if not sections:
            return f"{doc.title} has no headings to outline; search it, or read knowledge/{doc.file}."
        base = min(s["level"] for s in sections)
        lines = [f"{doc.title} — {len(sections)} section{'s' if len(sections) != 1 else ''}"
                 + (" (read one with action=section):" if sections else ":")]
        for s in sections[:OUTLINE_ENTRIES]:
            file, line = self._where(doc, s["start"])
            lines.append("  " * (s["level"] - base) + f"- {s['title']} ({file}, line {line})")
        if len(sections) > OUTLINE_ENTRIES:
            lines.append(f"… and {len(sections) - OUTLINE_ENTRIES} more: outline one part (section=…).")
        return "\n".join(lines)

    def _pick_section(self, doc: _Doc, title: str) -> dict | None:
        want = _norm(title.split(" › ")[-1])
        for test in (lambda s: _norm(s["title"]) == want,
                     lambda s: want in _norm(s["title"]),
                     lambda s: _norm(s["title"]) in want):
            found = [s for s in doc.sections if test(s)]
            if found:
                return found[0]
        terms = retrieval.query_terms(title, max_terms=12)
        if not terms:
            return None
        scored = [(len(retrieval.matched_terms(terms, s["title"])), s) for s in doc.sections]
        best = max(scored, key=lambda x: x[0], default=(0, None))
        return best[1] if best[0] >= max(1, (len(terms) + 1) // 2) else None

    def _section(self, doc: _Doc, title: str, page: int) -> str:
        s = self._pick_section(doc, title)
        if s is None:
            return f"No section of {doc.title} matches {title!r}. Its outline:\n" + self._outline(doc, "")
        text = doc.text[s["start"]:s["end"]].strip()
        pages = max(1, -(-len(text) // SECTION_CHARS))
        page = max(1, min(page, pages))
        chunk = text[(page - 1) * SECTION_CHARS:page * SECTION_CHARS]
        file, line = self._where(doc, s["start"])
        head = (f"{doc.title} › {' › '.join([*s['path'], s['title']])} — knowledge/{file}, from line {line}"
                + (f" — page {page} of {pages}" if pages > 1 else ""))
        more = f"\n\n(Continues: page={page + 1}.)" if page < pages else ""
        return f"{head}\n\n{chunk}{more}"

    # ── Search ────────────────────────────────────────────────────────

    async def find(self, folder: Path, docs: list[_Doc], query: str, only: _Doc | None = None,
                   limit: int = 5) -> list[dict]:
        """The best passages for *query* in the knowledge (or one document):
        by the index when the agent has one, else by reading the files —
        each ``doc``, ``file``, ``first``/``last`` lines, ``text``, ``where``
        (document › section path), ``section`` and ``offset`` (in its
        document's text, None outside one). Flight Deck's knowledge browser
        finds with it too."""
        scope = only.files if only else None
        semantic = getattr(getattr(getattr(self, "_agent", None), "memory", None), "semantic", None)
        hits: list[tuple[str, int, int, float]] = []   # (file, first line, last line, score)
        if semantic is not None and hasattr(semantic, "search_files"):
            try:
                found = await asyncio.to_thread(
                    semantic.search_files, query, prefix=f"{FOLDER}/",
                    paths=[f"{FOLDER}/{f}" for f in scope] if scope else None,
                    exclude_suffix=".map.md", max_results=limit * 4)
                hits = [(Path(r.path).name, r.start_line, r.end_line, r.score) for r in found
                        if Path(r.path).name != "INDEX.md"]
            except Exception:
                log.warning("knowledge search: the index failed, reading the files instead", exc_info=True)
        if not hits:
            files = scope or [f for d in docs for f in d.files] + [f for f in _EXTRAS if (folder / f).is_file()]
            hits = await asyncio.to_thread(self._scan, folder, files, query, limit * 4)
        passages = await asyncio.to_thread(self._passages, folder, docs, hits, query)
        return passages[:limit]

    async def _search(self, folder: Path, docs: list[_Doc], query: str, doc_name: str, limit: int) -> str:
        only = self._find_doc(docs, doc_name) if doc_name else None
        if doc_name and only is None:
            return f"No document matches {doc_name!r}. " + self._list(docs)
        passages = await self.find(folder, docs, query, only, limit)
        if not passages:
            where = f" in {only.title}" if only else ""
            return (f"Nothing in your knowledge{where} matches {query!r}. Try other words, or look at a "
                    "document's outline (action=outline).")
        lines = [f"{len(passages)} passage{'s' if len(passages) != 1 else ''} for {query!r} in your knowledge"
                 + (f" ({only.title})" if only else "") + ":"]
        for n, p in enumerate(passages, 1):
            lines += ["", f"[{n}] {p['where']}", f"    knowledge/{p['file']}, lines {p['first']}–{p['last']}",
                      p["text"]]
        named = next((p for p in passages if p.get("section")), None)
        if named:
            lines += ["", f"Read a whole section: action=section, doc={named['doc']!r}, section={named['section']!r}."]
        return "\n".join(lines)

    @staticmethod
    def _scan(folder: Path, files: list[str], query: str, limit: int) -> list[tuple[str, int, int, float]]:
        """Without the index: each paragraph of *files* scored by how many of
        the query's words it holds."""
        terms = retrieval.query_terms(query, max_terms=16)
        phrase = _norm(query)
        marks = [m.casefold() for m in _marks(query)]
        found: list[tuple[str, int, int, float]] = []
        for name in files:
            try:
                lines = (folder / name).read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            start = 0
            for i in range(len(lines) + 1):
                if i < len(lines) and lines[i].strip():
                    continue
                if i > start:
                    para = "\n".join(lines[start:i])
                    folded = _norm(para)
                    got = len(retrieval.matched_terms(terms, para)) / len(terms) if terms else 0.0
                    score = (got + (0.5 if phrase and phrase in folded else 0.0)
                             + (sum(1 for m in marks if m in folded) / len(marks) if marks else 0.0))
                    if score >= 0.5:
                        found.append((name, start + 1, i, score))
                start = i + 1
        found.sort(key=lambda h: -h[3])
        return found[:limit]

    def _passages(self, folder: Path, docs: list[_Doc], hits: list[tuple[str, int, int, float]],
                  query: str) -> list[dict]:
        """The hits as passages: each around its line that best matches the
        query, grown by whole paragraphs within the hit, with the section
        that line is in; a hit far weaker than the best, or overlapping a
        better one, left out. The exact phrase, code or amount asked for
        ranks first, and a quoted phrase is required."""
        if not hits:
            return []
        doc_of = {f: d for d in docs for f in (d.file, *d.files)}
        required = [_norm(q) for q in _QUOTED.findall(query)]
        phrase = _norm(_QUOTED.sub(" ", query)) if not required else ""
        marks = [m.casefold() for m in _marks(query)]
        terms = retrieval.query_terms(query, max_terms=16)
        top = max(h[3] for h in hits)
        file_lines: dict[str, list[str]] = {}
        taken: dict[str, list[tuple[int, int]]] = {}
        out: list[dict] = []
        for file, first, last, score in sorted(hits, key=lambda h: -h[3]):
            if score < top * _WEAKEST:
                break
            if file not in file_lines:
                try:
                    file_lines[file] = (folder / file).read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    continue
            lines = file_lines[file]
            first, last = max(1, first), min(len(lines), max(first, last))
            if first > last:
                continue
            focus = _focus(lines, first, last, terms, marks, phrase)
            a, b = _around(lines, focus, first, last)
            if any(a <= y and b >= x for x, y in taken.get(file, [])):
                continue
            text = "\n".join(lines[a - 1:b]).strip()
            folded = _norm(text)
            if any(q not in folded for q in required):
                continue
            if phrase and len(phrase) > 3 and phrase in folded:
                score += 0.3
            if marks:
                score += 0.2 * sum(1 for m in marks if m in folded) / len(marks)
            taken.setdefault(file, []).append((a, b))
            doc = doc_of.get(file)
            where, section, at = (doc.title if doc else Path(file).stem), "", None
            if doc is not None:
                self._load(folder, doc)
                at = self._offset(doc, file, focus)
                s = self._section_at(doc, at) if at is not None else None
                if s:
                    section = s["title"]
                    where = " › ".join([doc.title, *s["path"], s["title"]])
            out.append({"file": file, "first": a, "last": b, "score": score, "text": text, "where": where,
                        "doc": doc.title if doc else "", "doc_file": doc.file if doc else "",
                        "section": section, "offset": at, "focus": focus})
        out.sort(key=lambda p: -p["score"])
        return out


def _focus(lines: list[str], first: int, last: int, terms: list[str], marks: list[str], phrase: str) -> int:
    """The line of ``first..last`` (1-based) that best matches the query —
    its words, a code or amount it names, the whole phrase; the middle when
    none does."""
    best, best_score = (first + last) // 2, 0.0
    for i in range(first, last + 1):
        line = lines[i - 1]
        if not line.strip():
            continue
        folded = _norm(line)
        score = (len(retrieval.matched_terms(terms, line)) + 2.0 * sum(1 for m in marks if m in folded)
                 + (2.0 if phrase and len(phrase) > 3 and phrase in folded else 0.0))
        if score > best_score:
            best, best_score = i, score
    return best


def _paragraph(lines: list[str], i: int) -> tuple[int, int]:
    """The paragraph line *i* (1-based) is in, at most _GROW_LINES each way."""
    a = b = i
    for _ in range(_GROW_LINES):
        if a <= 1 or not lines[a - 2].strip():
            break
        a -= 1
    for _ in range(_GROW_LINES):
        if b >= len(lines) or not lines[b].strip():
            break
        b += 1
    return a, b


def _around(lines: list[str], focus: int, first: int, last: int) -> tuple[int, int]:
    """The passage around *focus*: its paragraph, then the paragraphs on
    either side while they are within ``first..last`` and it stays under
    PASSAGE_CHARS — never above the heading of the section *focus* is in
    (the end of the one before is another subject)."""
    first = next((i for i in range(focus, first - 1, -1) if lines[i - 1].startswith("#")), first)
    a, b = _paragraph(lines, focus)
    a = max(a, first)
    size = sum(len(x) + 1 for x in lines[a - 1:b])
    while True:
        grew = False
        above = a - 1 if a - 1 >= 1 and lines[a - 2].strip() else a - 2   # the line above, past a blank one
        if above >= first:
            pa = max(first, _paragraph(lines, above)[0])
            more = sum(len(x) + 1 for x in lines[pa - 1:a - 1])
            if size + more <= PASSAGE_CHARS:
                a, size, grew = pa, size + more, True
        below = b + 1 if b < len(lines) and lines[b].strip() else b + 2
        if below <= last:
            pb = min(last, _paragraph(lines, below)[1])
            more = sum(len(x) + 1 for x in lines[b:pb])
            if size + more <= PASSAGE_CHARS:
                b, size, grew = pb, size + more, True
        if not grew:
            break
    return a, b
