"""An agent's knowledge, to browse — the Knowledge item in the kiosk's agent
menu (KioskKnowledgeBrowser).

The documents an owner's setup gave the agent sit whole in its workspace's
knowledge/ folder (agent_preset.write_knowledge). This reads them as the
agent's own knowledge tool does (tools/knowledge.KnowledgeTool), so it works
while the agent is off or asleep:

- ``GET /fd/agent-knowledge/{slug}`` — its documents, each with how much of
  it the model read (the setup record's ``read``), and the glossary and the
  owner's rules;
- ``…/{slug}/doc?id=`` — a document's sections (agent_preset_select.spans,
  the keys reads are kept by), each marked read or not;
- ``…/{slug}/section?id=&i=`` — one section's own text, in pages, with where
  it is in the files and the sections around it;
- ``…/{slug}/extra?id=`` — the glossary or the owner's rules, whole;
- ``…/{slug}/search?q=`` — the passages a search finds, each with the
  section to open.

Those are read-only, and the owner's alone (or an admin's acting for them).
And for the agent itself — its knowledge tool's ``add``, when its user hands
it a file in a chat and says to keep it: ``POST …/agent/add`` takes a
document (or the text of a picture the agent read) into the calling agent's
own knowledge, as a setup would, not read (agent_preset.add_document).
"""

from __future__ import annotations

import asyncio
import types
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile

from captain_claw.flight_deck import agent_preset as ap
from captain_claw.flight_deck import agent_preset_docs as pdocs
from captain_claw.flight_deck import agent_preset_select as psel
from captain_claw.flight_deck.agent_identity import require_agent_caller
from captain_claw.flight_deck.auth import get_managed_user
from captain_claw.logging import get_logger
from captain_claw.tools.knowledge import KnowledgeTool, _Doc

log = get_logger(__name__)

router = APIRouter(prefix="/fd/agent-knowledge", tags=["agent-knowledge"])

PAGE_CHARS = 30_000      # one page of a section
SNIPPET_CHARS = 700      # a search hit, as the list shows it
SEARCH_RESULTS = 20
_KEEP_AGENTS = 4         # agents whose documents stay read in memory
_EXTRAS = {"glossary.md": "Glossary", "owner-rules.md": "Your rules"}

# Per agent's workspace, a knowledge tool holding its documents' text, and
# each document's sections: the agents browsed last stay read.
_TOOLS: OrderedDict[str, KnowledgeTool] = OrderedDict()
_SPANS: dict[int, tuple[_Doc, list[dict]]] = {}


def _place(slug: str, user: dict) -> tuple[dict, Path, Path]:
    """The caller's own agent: its registry entry, folder and workspace (404 for anyone else's)."""
    from captain_claw.flight_deck import server as srv
    entry = ap._own_process(slug, ap._owner(user))
    agent_dir = srv.DATA_DIR / slug
    workspace = Path(entry["workspace"]) if entry.get("workspace") else agent_dir / "data" / "workspace"
    return entry, agent_dir, workspace


def _tool(workspace: Path) -> KnowledgeTool:
    key = str(workspace)
    tool = _TOOLS.pop(key, None)
    if tool is None:
        tool = KnowledgeTool()
        tool._agent = types.SimpleNamespace(memory=None, workspace_base_path=workspace)
    _TOOLS[key] = tool
    while len(_TOOLS) > _KEEP_AGENTS:
        _TOOLS.popitem(last=False)
    return tool


def _spans(tool: KnowledgeTool, folder: Path, doc: _Doc) -> list[dict]:
    """The document's sections, its text read first."""
    hit = _SPANS.get(id(doc))
    if hit and hit[0] is doc:
        return hit[1]
    tool.load(folder, doc)
    spans = psel.spans(doc.text)
    for s in spans:
        s["level"] = len(s["path"]) + 1
        s["sized"] = len(doc.text[s["start"]:s["end"]].strip()) >= psel.MIN_UNIT_CHARS
    if len(_SPANS) > 32:
        _SPANS.clear()
    _SPANS[id(doc)] = (doc, spans)
    return spans


def _entries(agent_dir: Path, workspace: Path) -> tuple[dict[str, dict], list[dict]]:
    """The setup record's documents by their file, and its tables."""
    record = ap.earlier_setup(agent_dir, workspace)
    by_file = {e["file"]: e for e in record.get("knowledge") or [] if isinstance(e, dict) and e.get("file")}
    tables = [t for t in record.get("tables") or [] if isinstance(t, dict)]
    return by_file, tables


def _read_marks(entry: dict) -> tuple[bool, set[str], bool]:
    """(known, keys read, all read) from a record entry's ``read``."""
    read = entry.get("read") if isinstance(entry.get("read"), dict) else None
    if read is None:
        return False, set(), False
    return True, set(read.get("keys") or []), bool(read.get("all"))


def _doc_view(folder: Path, doc: _Doc, entry: dict, tables: list[dict]) -> dict:
    size = sum((folder / f).stat().st_size for f in doc.files if (folder / f).is_file())
    return {"id": doc.file, "doc": entry.get("doc") or doc.file, "title": doc.title,
            "summary": entry.get("summary") or doc.summary, "use": entry.get("use") or "",
            "parts": len(doc.files), "bytes": size, "map": entry.get("map") or "",
            "from_setup": bool(entry), "read": ap._read_state(entry.get("read")),
            "tables": [t.get("name") for t in tables if t.get("doc") == entry.get("doc") and t.get("name")]}


async def _docs(workspace: Path) -> tuple[Path, KnowledgeTool, list[_Doc]]:
    folder = workspace / "knowledge"
    tool = _tool(workspace)
    docs = await asyncio.to_thread(tool.documents, folder) if folder.is_dir() else []
    return folder, tool, docs


def _pick(docs: list[_Doc], doc_id: str) -> _Doc:
    doc = next((d for d in docs if d.file == doc_id), None)
    if doc is None:
        raise HTTPException(404, "That document isn't in the agent's knowledge any more.")
    return doc


@router.get("/{slug}")
async def overview(slug: str, user: dict = Depends(get_managed_user)) -> dict:
    """The agent's documents — each with its summary, size, parts and how
    much of it the model read — and its glossary and the owner's rules."""
    entry, agent_dir, workspace = _place(slug, user)
    folder, _, docs = await _docs(workspace)
    by_file, tables = await asyncio.to_thread(_entries, agent_dir, workspace)
    documents = [await asyncio.to_thread(_doc_view, folder, d, by_file.get(d.file, {}), tables) for d in docs]
    extras = [{"id": name, "title": title} for name, title in _EXTRAS.items() if (folder / name).is_file()]
    known = [d["read"] for d in documents if d["read"]["state"] != "unknown"]
    return {
        "agent": str(entry.get("name") or slug),
        "documents": documents,
        "extras": extras,
        "totals": {"documents": len(documents), "bytes": sum(d["bytes"] for d in documents),
                   "sections": sum(r["of"] for r in known), "read": sum(r["sections"] for r in known),
                   "tracked": len(known)},
    }


@router.get("/{slug}/doc")
async def document(slug: str, id: str = Query(..., max_length=200), user: dict = Depends(get_managed_user)) -> dict:
    """One document's sections, in order: each one's ``level``, ``title``,
    ``path`` (its parents), size and whether the model read it (None: a
    heading with nothing of its own, or a document from before reads were
    kept)."""
    _, agent_dir, workspace = _place(slug, user)
    folder, tool, docs = await _docs(workspace)
    doc = _pick(docs, id)
    spans = await asyncio.to_thread(_spans, tool, folder, doc)
    by_file, tables = await asyncio.to_thread(_entries, agent_dir, workspace)
    entry = by_file.get(doc.file, {})
    known, keys, everything = _read_marks(entry)
    return {
        **_doc_view(folder, doc, entry, tables),
        "chars": len(doc.text),
        "sections": [{"i": n, "level": s["level"], "title": s["title"], "path": s["path"],
                      "chars": s["end"] - s["start"],
                      "read": (everything or s["key"] in keys) if known and s["sized"] else None}
                     for n, s in enumerate(spans)],
    }


@router.get("/{slug}/section")
async def section(slug: str, id: str = Query(..., max_length=200), i: int = Query(..., ge=0),
                  page: int = Query(1, ge=1), user: dict = Depends(get_managed_user)) -> dict:
    """One section's own text — its heading to its first subsection — a
    page at a time, with where it starts in the files, its subsections, and
    the sections before and after it."""
    _, agent_dir, workspace = _place(slug, user)
    folder, tool, docs = await _docs(workspace)
    doc = _pick(docs, id)
    spans = await asyncio.to_thread(_spans, tool, folder, doc)
    if i >= len(spans):
        raise HTTPException(404, "That section isn't in the document any more.")
    s = spans[i]
    text = doc.text[s["start"]:s["end"]].strip()
    pages = max(1, -(-len(text) // PAGE_CHARS))
    page = min(page, pages)
    file, line = tool.where(doc, s["start"])
    children: list[dict] = []
    for n in range(i + 1, len(spans)):
        if spans[n]["level"] <= s["level"]:
            break
        if spans[n]["level"] == s["level"] + 1:
            children.append({"i": n, "title": spans[n]["title"]})
    by_file, _ = await asyncio.to_thread(_entries, agent_dir, workspace)
    known, keys, everything = _read_marks(by_file.get(doc.file, {}))

    def near(n: int) -> dict | None:
        return {"i": n, "title": spans[n]["title"]} if 0 <= n < len(spans) else None

    return {
        "id": doc.file, "title": doc.title, "i": i, "level": s["level"], "section": s["title"], "path": s["path"],
        "text": text[(page - 1) * PAGE_CHARS:page * PAGE_CHARS], "page": page, "pages": pages, "chars": len(text),
        "file": file, "line": line, "children": children[:60], "prev": near(i - 1), "next": near(i + 1),
        "read": (everything or s["key"] in keys) if known and s["sized"] else None,
    }


@router.get("/{slug}/extra")
async def extra(slug: str, id: str = Query(..., max_length=40), user: dict = Depends(get_managed_user)) -> dict:
    """The glossary or the owner's rules, whole."""
    _, _, workspace = _place(slug, user)
    if id not in _EXTRAS:
        raise HTTPException(404, "There's no such page.")
    path = workspace / "knowledge" / id
    if not path.is_file():
        raise HTTPException(404, "The agent has none.")
    return {"id": id, "title": _EXTRAS[id], "text": await asyncio.to_thread(path.read_text, encoding="utf-8")}


@router.get("/{slug}/search")
async def search(slug: str, q: str = Query(..., max_length=300), id: str = Query("", max_length=200),
                 user: dict = Depends(get_managed_user)) -> dict:
    """The passages that best match *q* — in one document (``id``) or all —
    each with its document, the section to open (``i``) and lines."""
    _, _, workspace = _place(slug, user)
    query = q.strip()
    if len(query) < 2:
        return {"results": []}
    folder, tool, docs = await _docs(workspace)
    only = _pick(docs, id) if id else None
    passages = await tool.find(folder, docs, query, only, SEARCH_RESULTS)
    by_file = {d.file: d for d in docs}
    results: list[dict[str, Any]] = []
    for p in passages:
        doc = by_file.get(p.get("doc_file") or "")
        hit: dict[str, Any] = {"file": p["file"], "first": p["first"], "last": p["last"],
                               "text": p["text"][:SNIPPET_CHARS], "where": p["where"]}
        if doc is not None and p.get("offset") is not None:
            spans = await asyncio.to_thread(_spans, tool, folder, doc)
            n = next((k for k in range(len(spans) - 1, -1, -1) if spans[k]["start"] <= p["offset"]), 0)
            hit.update(id=doc.file, title=doc.title, i=n, section=spans[n]["title"] if spans else "",
                       path=spans[n]["path"] if spans else [])
        elif p["file"] in _EXTRAS:
            hit.update(extra=p["file"], title=_EXTRAS[p["file"]])
        results.append(hit)
    return {"results": results}


# ── An agent adding to its own knowledge ──────────────────────────────

_ADDING: dict[str, asyncio.Lock] = {}   # one add at a time per agent: each writes its setup record


@router.post("/agent/add")
async def add_from_agent(
    request: Request,
    name: str = Form("", max_length=200),
    file: UploadFile | None = File(None),
    text: str = Form(""),
    title: str = Form("", max_length=200),
    note: str = Form("", max_length=2000),
    use: str = Form("auto", max_length=20),
) -> dict:
    """The calling agent keeps a file in its own knowledge — what its user
    handed it in a chat: a document (``file``: PDF, Word, Excel, PowerPoint,
    CSV, Markdown, text, HTML), or the ``text`` it read off a picture or a
    scan. Whole and searchable, not read by a model (agent_preset.add_document).
    Only this deck's agents, each into its own knowledge."""
    caller = require_agent_caller(request, what="adding to an agent's knowledge")
    if not caller.slug:
        raise HTTPException(403, "Flight Deck can't tell which agent is calling.")
    # Its name: as the agent gives it, else the file's own.
    given = str(name or (file.filename if file is not None else "") or "")
    name = Path(given.replace("\\", "/")).name.strip()[:120] or "document"
    raw = await file.read() if file is not None else b""
    if raw:
        # What it is comes from the file itself; the name the agent chose keeps its kind ("notes.md" for a
        # web page would be read as Markdown, and a spreadsheet renamed so would lose its tables).
        suffix = Path(file.filename or "").suffix.lower() or Path(name).suffix.lower()
        if len(raw) > pdocs.MAX_FILE_BYTES:
            raise HTTPException(413, f"{name} is larger than 25 MB.")
        if suffix not in pdocs.SUPPORTED:
            raise HTTPException(415, f"{name}: not a kind of document knowledge takes ({', '.join(pdocs.SUPPORTED)}). "
                                     "A picture or a scan: read its text, and add that as text.")
        name = (Path(name).stem if Path(name).suffix.lower() in pdocs.SUPPORTED else name) + suffix
        body, why = await asyncio.to_thread(pdocs.extract_bytes, raw, suffix)
        if not body.strip():
            scan = why == pdocs.NO_TEXT_SCAN
            raise HTTPException(422, {"reason": "scan" if scan else "unreadable",
                                      "detail": f"{name}: " + (why or "no text could be read from it.")})
    elif text.strip():
        body = text
        if not Path(name).suffix:
            name += ".md"
    else:
        raise HTTPException(400, "Send the file, or the text to keep.")
    from captain_claw.flight_deck.file_library import _auth_enabled
    owner = caller.owner if _auth_enabled() else ""
    async with _ADDING.setdefault(caller.slug, asyncio.Lock()):
        return await ap.add_document(caller.slug, owner, name, body[:pdocs.MAX_CHARS], raw, title, note, use)
