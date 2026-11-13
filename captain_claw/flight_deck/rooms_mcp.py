"""The data-room connector (``docs/data-room-plan.md`` P2): Claude reads the
released files of the rooms a person granted it — anonymized text only.

``POST /fd/mcp-rooms`` speaks MCP (Streamable HTTP, JSON responses, no
sessions). It has its own OAuth resource and scope (``rooms``): a fleet token
isn't accepted here, a rooms token isn't accepted anywhere else, and each
connected app reads only the rooms the person ticked on the consent page
(``rooms_store`` grants, changeable on the Data rooms page).

Nothing restored ever leaves through here: every answer is built from a
room's released text views, its roles (checked for real values when saved)
and counts. A real vault value showing up in Claude's arguments means it
reached Claude some other way; the call still answers (it's already out) and
the room shows an alarm. Every read is logged (tool, file, page, sizes —
never text or queries) and goes into the tokens' history as "out".
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import PurePosixPath
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from captain_claw.anonymize import usage
from captain_claw.flight_deck import anonymize_vault as av
from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import rooms_job as rj
from captain_claw.flight_deck import rooms_store as rs
from captain_claw.flight_deck.mcp_oauth_routes import ROOMS_PATH, ROOMS_SCOPE, _public_base
from captain_claw.logging import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/fd", tags=["data-rooms-mcp"])

SERVER_NAME = "captain-data-rooms"
SERVER_VERSION = "1.0.0"
PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26")  # newest first; a client's own is answered when known
READ_PAGE = 50_000      # P0: whole results reach Claude up to ~100k characters
MAX_SEARCH = 25
TO_MODEL = "Claude (data-room connector)"

INSTRUCTIONS = (
    "This connector serves anonymized data rooms. Every real name, company, address and identifying number "
    "in them is replaced by a token like ⟦ORG_7⟧ or ⟦PERSON_2⟧; ⟦ORG_7:2⟧ is another spelling of ⟦ORG_7⟧. "
    "Numbers, dates and the story are kept. Rules:\n"
    "1. Keep every token exactly as written, brackets included, in answers and in every file you make — "
    "titles, sheet names, notes and file names too. Never replace a token with a codename, a role or a "
    "description, and never guess who a token is.\n"
    "2. Start with room_list, then room_overview: it gives each token's role (Target, Buyer, CFO …) — use "
    "the roles to reason, keep the tokens when you write.\n"
    "3. room_search finds passages; room_read reads a file page by page (about 50,000 characters a page). "
    "Cite files by path and page.\n"
    "4. Only released files are here. If something seems missing, say so; don't fill gaps with outside "
    "knowledge about who the parties might be.\n"
    "5. To give the person a result (a memo, an answer, a summary), call room_deliver with markdown in tokens. "
    "Up to about 20,000 characters a call; longer → parts with the same title (part 1 of N, part 2 of N …), "
    "joined exactly as sent, so end each part at a line break. The person reads it in Flight Deck with the real "
    "names; you get a receipt, never the restored text. Don't put pictures or links in a delivery.\n"
    "6. When the person wants a file, don't make it yourself — the names would be missing: send its content with "
    "room_deliver_workbook (Excel: sheets, rows, formulas, formats, charts), room_deliver_deck (PowerPoint) or "
    "room_deliver_document (Word, from markdown). The file is built and the names put back in Flight Deck. Keep "
    "tokens in every cell, sheet name, title and note; formulas stay inside the workbook (no web addresses, other "
    "files, WEBSERVICE, IMAGE or HYPERLINK).\n"
    "7. To work on a released file with code (an Excel model, say), room_get_file gives you the anonymized file "
    "itself as base64 — decode it in your workspace. If you do make a file yourself, keep the tokens in it (never a "
    "codename or a description) and send it with room_deliver_file (base64; about 15 KB fits in one call, more in "
    "parts), or give it to the person to drop into Flight Deck → Data rooms → From Claude.\n"
    "8. The person can leave you tasks in a room (room_list counts them). When they ask you to check Captain or "
    "their tasks, call room_tasks, do each one, and deliver each answer with task=<its number> on the deliver "
    "tool, so it lands next to the task. A task may be about an earlier delivery: it comes along in tokens "
    "(room_get_delivery has the rest)."
)
_TASK = {"type": "integer", "description": "The task this answers (from room_tasks), if it answers one."}

_ANNOT = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
TOOLS = [
    {"name": "room_list", "description": "The data rooms this connector may read: id, codename and how many files "
     "are released. Call this first.", "inputSchema": {"type": "object", "properties": {}, "required": []},
     "annotations": _ANNOT},
    {"name": "room_overview", "description": "One room: the role of each token (Target, Buyer, CFO …), its folders, "
     "file kinds and size. Read it before anything else in the room.",
     "inputSchema": {"type": "object", "properties": {
         "room": {"type": "string", "description": "The room's id or codename (from room_list)."}},
         "required": ["room"]}, "annotations": _ANNOT},
    {"name": "room_list_files", "description": "The released files of a room (or of one folder in it): path, kind, "
     "characters and pages.", "inputSchema": {"type": "object", "properties": {
         "room": {"type": "string"},
         "folder": {"type": "string", "description": "Only files under this folder (optional)."}},
         "required": ["room"]}, "annotations": _ANNOT},
    {"name": "room_read", "description": "Read a file of a room, one page at a time (about 50,000 characters a "
     "page; the answer says how many pages there are). Word, Excel, PowerPoint and PDF come as markdown.",
     "inputSchema": {"type": "object", "properties": {
         "room": {"type": "string"},
         "path": {"type": "string", "description": "The file's path, as room_list_files gives it."},
         "page": {"type": "integer", "description": "Which page (1 = the first; default 1)."}},
         "required": ["room", "path"]}, "annotations": _ANNOT},
    {"name": "room_search", "description": "Search a room's released files: passages with the file, the page to "
     "read with room_read, and a snippet. Words, numbers and tokens like ⟦ORG_7⟧ all work.",
     "inputSchema": {"type": "object", "properties": {
         "room": {"type": "string"},
         "query": {"type": "string"},
         "limit": {"type": "integer", "description": f"At most this many passages (default 10, max {MAX_SEARCH})."}},
         "required": ["room", "query"]}, "annotations": _ANNOT},
    {"name": "room_deliver", "description": "Give the person a result from this room: markdown (or plain text) in "
     "tokens, with a title. The person reads it in Flight Deck with the real names put back; you get a receipt "
     "with a number and counts, never the restored text. Up to about 20,000 characters a call — longer goes in "
     "parts (same title; part 1 of N …), joined exactly as sent.",
     "inputSchema": {"type": "object", "properties": {
         "room": {"type": "string"},
         "title": {"type": "string", "description": "A short title, in tokens like the rest."},
         "content": {"type": "string", "description": "The result: markdown with tokens, no pictures."},
         "format": {"type": "string", "enum": ["markdown", "text"]},
         "part": {"type": "integer", "description": "This part's number, when sending in parts."},
         "of": {"type": "integer", "description": "How many parts in all, when sending in parts."},
         "task": _TASK},
         "required": ["room", "title", "content"]},
     "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False,
                     "openWorldHint": False}},
    {"name": "room_tasks", "description": "Tasks the person left for you in a room (or in all your rooms): what "
     "to do, in tokens, and the earlier delivery it is about. Deliver each answer with task=<its number>.",
     "inputSchema": {"type": "object", "properties": {
         "room": {"type": "string", "description": "One room's id or codename (optional; default: all)."}},
         "required": []}, "annotations": _ANNOT},
    {"name": "room_get_delivery", "description": "One of your earlier deliveries to a room, as you wrote it "
     "(tokens), page by page — the context a task refers to.",
     "inputSchema": {"type": "object", "properties": {
         "room": {"type": "string"},
         "delivery": {"type": "integer", "description": "The delivery's number (#n in its receipt or a task)."},
         "page": {"type": "integer", "description": "Which page (1 = the first; default 1)."}},
         "required": ["room", "delivery"]}, "annotations": _ANNOT},
]


class ToolError(Exception):
    """Told to Claude as the tool's answer (isError), never a real value."""


# ── who is calling ──────────────────────────────────────────────────


def _unauth(request: Request, detail: str) -> HTTPException:
    prm = f"{_public_base(request)}/.well-known/oauth-protected-resource{ROOMS_PATH}"
    return HTTPException(401, detail, headers={"WWW-Authenticate": f'Bearer resource_metadata="{prm}"'})


async def caller(request: Request) -> tuple[dict[str, Any], str]:
    """(the person, the connected app's client id) — a rooms-scope token only."""
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not token:
        raise _unauth(request, "Not authenticated")
    payload = fd_auth.decode_mcp_access_token(token)
    if payload is None or payload.get("scope") != ROOMS_SCOPE:
        raise _unauth(request, "This connector takes a data-room sign-in only")
    user = await fd_auth.get_db().get_user_by_id(str(payload.get("sub") or ""))
    if not user:
        raise _unauth(request, "User not found")
    return user, str(payload.get("client_id") or "")


# ── rooms this app may read ─────────────────────────────────────────


async def allowed_rooms(owner: str, client_id: str) -> list[dict[str, Any]]:
    store = rs.store()
    grant = await store.grant(owner, client_id)
    if grant is None:
        return []
    rooms = await store.list_rooms(owner)
    return [r for r in rooms if grant["all_rooms"] or r["id"] in grant["rooms"]]


async def _room(owner: str, client_id: str, selector: Any) -> dict[str, Any]:
    want = str(selector or "").strip()
    if not want:
        raise ToolError("room is required — call room_list for the rooms you can read.")
    rooms = await allowed_rooms(owner, client_id)
    by_id = [r for r in rooms if r["id"] == want]
    by_name = [r for r in rooms if r["codename"].casefold() == want.casefold()]
    found = by_id or by_name
    if len(found) != 1:
        raise ToolError(f"No room '{want[:80]}' is shared with this connector"
                        + (" (two rooms have that codename — use its id)" if len(found) > 1 else "")
                        + ". Call room_list.")
    return found[0]


async def _released(room_id: str) -> list[dict[str, Any]]:
    return [f for f in await rs.store().files(room_id, released_only=True) if f["state"] == "done"]


def _text(room_id: str, f: dict[str, Any]) -> str:
    path = rs.store().text_file(room_id, f["id"])
    return path.read_text("utf-8") if path.exists() else ""


def page_bounds(text: str) -> list[tuple[int, int]]:
    """Pages of about READ_PAGE characters, ending at a line break when one is near."""
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + READ_PAGE, len(text))
        if end < len(text):
            cut = text.rfind("\n", start + READ_PAGE - 4000, end)
            if cut > start:
                end = cut + 1
        bounds.append((start, end))
        start = end
    return bounds or [(0, 0)]


def page_at(bounds: list[tuple[int, int]], offset: int) -> int:
    for i, (s, e) in enumerate(bounds, 1):
        if s <= offset < e:
            return i
    return len(bounds)


def label_offset(text: str, label: str) -> int:
    """Where an index piece starts in the text: a "Page 3" / "Slide 3" / "Sheet …"
    heading, or a numbered piece of PAGE_CHARS."""
    label = str(label or "")
    if label.isdigit():
        return (int(label) - 1) * rj.PAGE_CHARS
    m = re.search(rf"(?m)^## {re.escape(label)}\s*$", text)
    return m.start() if m else 0


# ── the vault behind a room: alarms and history ─────────────────────

_VAULT_CACHE: dict[str, tuple[float, list[tuple[str, str]], set[str]]] = {}


async def _vault(room: dict[str, Any]) -> tuple[dict[str, Any] | None, list[tuple[str, str]], set[str]]:
    vaults = av.store()
    row = await vaults.get(room["owner_id"], room["vault_id"])
    if row is None:
        return None, [], set()
    cached = _VAULT_CACHE.get(room["id"])
    if cached and cached[0] == row["updated_at"]:
        return row, cached[1], cached[2]
    state = await vaults.load_state(row)
    needle_list = rj.needles(state)
    tokens = set(state.by_token)
    _VAULT_CACHE[room["id"]] = (row["updated_at"], needle_list, tokens)
    return row, needle_list, tokens


async def _alarm(room: dict[str, Any], client_id: str, tool: str, args: dict[str, Any]) -> bool:
    """A real value in what Claude sent: it got there some other way. Kept as
    tokens on the room; the call still answers."""
    text = "\n".join(str(v) for v in args.values() if isinstance(v, str | int))
    if not text.strip():
        return False
    _row, needle_list, _tokens = await _vault(room)
    hits = rj.leaks(text, needle_list)
    if hits:
        await rs.store().add_alarm(room["id"], room["owner_id"], client_id, tool, hits)
        log.warning("Data room alarm: a real value in Claude's arguments", room=room["id"], tool=tool,
                    tokens=len(hits))
    return bool(hits)


async def _history(room: dict[str, Any], client_id: str, texts: list[str], label: str) -> None:
    row, _needles, tokens = await _vault(room)
    if row is None or not tokens:
        return
    found = usage.token_contexts(texts, tokens)
    if found:
        await av.store().record_uses(row, [av.TokenUse(t, "out", n, c, model=TO_MODEL) for t, (n, c) in found.items()],
                                     channel="data-room:connector", actor=f"mcp:{client_id}", label=label[:120])


# ── tools ───────────────────────────────────────────────────────────


async def _room_list(owner: str, client_id: str, args: dict[str, Any]) -> dict[str, Any]:
    out = []
    for r in await allowed_rooms(owner, client_id):
        files = await _released(r["id"])
        entry = {"room": r["id"], "codename": r["codename"], "released_files": len(files),
                 "released_at": time.strftime("%Y-%m-%d", time.gmtime(r["released_at"])) if r["released_at"]
                 else None}
        waiting = len(await rs.store().open_tasks(r["id"]))
        if waiting:
            entry["tasks_waiting"] = waiting
        out.append(entry)
    note = "" if any(r["released_files"] for r in out) else (
        "Nothing is released to this connector yet — the person releases rooms in Flight Deck → Data rooms.")
    if any(r.get("tasks_waiting") for r in out):
        note = (note + " " if note else "") + "The person left you tasks: call room_tasks."
    return {"rooms": out, **({"note": note} if note else {})}


async def _room_tasks(owner: str, client_id: str, args: dict[str, Any]) -> dict[str, Any]:
    from captain_claw.flight_deck import rooms_tasks

    rooms = [await _room(owner, client_id, args["room"])] if args.get("room") else await allowed_rooms(owner, client_id)
    out = []
    for room in rooms:
        tasks = await rooms_tasks.for_claude(room, client_id)
        if tasks:
            out.append({"room": room["id"], "codename": room["codename"], "tasks": tasks})
            await rs.store().log_read(room["id"], owner, client_id, "room_tasks", chars=sum(len(t["prompt"]) for t in tasks))
    if not out:
        return {"tasks": [], "note": "No tasks are waiting."}
    return {"rooms": out, "note": "Do each task and deliver the answer with task=<its number>. Keep every token "
                                  "exactly as written."}


async def _room_get_delivery(owner: str, client_id: str, args: dict[str, Any]) -> str:
    from captain_claw.flight_deck import rooms_inbox

    room = await _room(owner, client_id, args.get("room"))
    try:
        seq = int(args.get("delivery"))
    except (TypeError, ValueError):
        raise ToolError("delivery is a number (#n from a receipt or a task).") from None
    d = await rs.store().delivery(room["id"], seq)
    row, _needles, _tokens = await _vault(room)
    if d is None or row is None:
        raise ToolError(f"There is no delivery #{seq} in this room.")
    try:
        p = rooms_inbox.open_payload(av.store().cipher(row), d)
    except Exception:  # noqa: BLE001
        raise ToolError(f"Delivery #{seq} can't be opened.") from None
    text = f"# {p.get('title_tokens') or ''}\n\n{p.get('body_tokens') or ''}"
    bounds = page_bounds(text)
    page = max(1, int(args.get("page") or 1))
    if page > len(bounds):
        raise ToolError(f"Delivery #{seq} has {len(bounds)} page(s).")
    start, end = bounds[page - 1]
    await rs.store().log_read(room["id"], owner, client_id, "room_get_delivery", page=page, chars=end - start)
    head = f"Delivery #{seq} · page {page} of {len(bounds)} · characters {start + 1:,}–{end:,} of {len(text):,}\n\n"
    return head + text[start:end]


async def _close_task(room: dict[str, Any], client_id: str, args: dict[str, Any], receipt: dict[str, Any]) -> None:
    """A delivery that answers a task closes it (when it is in — not for a part)."""
    from captain_claw.flight_deck import rooms_tasks

    if args.get("task") in (None, "") or "delivered" not in receipt:
        return
    seq = int(str(receipt["delivered"]).lstrip("#"))
    receipt["task"] = await rooms_tasks.close(room, args.get("task"), seq, client_id)


async def _room_overview(owner: str, client_id: str, args: dict[str, Any]) -> dict[str, Any]:
    room = await _room(owner, client_id, args.get("room"))
    alarm = await _alarm(room, client_id, "room_overview", args)
    files = await _released(room["id"])
    folders = Counter(str(PurePosixPath(f["clean_path"]).parent) for f in files)
    kinds = Counter(f["kind"] for f in files)
    await rs.store().log_read(room["id"], owner, client_id, "room_overview", alarm=alarm)
    return {
        "room": room["id"], "codename": room["codename"], "released_files": len(files),
        "characters": sum(f["text_chars"] for f in files),
        "roles": room["roles"] or {},
        "folders": {("(top)" if k == "." else k): v for k, v in sorted(folders.items())},
        "kinds": dict(kinds),
        "read_page_characters": READ_PAGE,
        "note": "Tokens stand for real names the person can see and you can't — keep them exactly, brackets "
                "included, in everything you write." + ("" if room["roles"] else " No roles are set for this room."),
    }


async def _room_list_files(owner: str, client_id: str, args: dict[str, Any]) -> dict[str, Any]:
    room = await _room(owner, client_id, args.get("room"))
    alarm = await _alarm(room, client_id, "room_list_files", args)
    folder = str(args.get("folder") or "").strip().strip("/")
    out = []
    for f in await _released(room["id"]):
        if folder and not (f["clean_path"] == folder or f["clean_path"].startswith(folder + "/")):
            continue
        pages = max(1, -(-f["text_chars"] // READ_PAGE)) if f["text_chars"] else 1
        out.append({"path": f["clean_path"], "kind": f["kind"], "characters": f["text_chars"], "pages": pages})
    await rs.store().log_read(room["id"], owner, client_id, "room_list_files", alarm=alarm)
    return {"room": room["id"], "files": out}


def _find_file(files: list[dict[str, Any]], path: str) -> dict[str, Any]:
    want = str(path or "").strip().strip("/")
    exact = [f for f in files if f["clean_path"] == want]
    if exact:
        return exact[0]
    by_name = [f for f in files if PurePosixPath(f["clean_path"]).name == PurePosixPath(want).name]
    if len(by_name) == 1:
        return by_name[0]
    raise ToolError(f"No released file '{want[:200]}' in this room — call room_list_files for the paths.")


async def _room_read(owner: str, client_id: str, args: dict[str, Any]) -> str:
    room = await _room(owner, client_id, args.get("room"))
    alarm = await _alarm(room, client_id, "room_read", args)
    f = _find_file(await _released(room["id"]), str(args.get("path") or ""))
    text = _text(room["id"], f)
    bounds = page_bounds(text)
    try:
        page = int(args.get("page") or 1)
    except (TypeError, ValueError):
        raise ToolError("page is a number (1 = the first)") from None
    if page < 1 or page > len(bounds):
        raise ToolError(f"{f['clean_path']} has {len(bounds)} page(s).")
    s, e = bounds[page - 1]
    body = text[s:e]
    head = f"File: {f['clean_path']} · page {page} of {len(bounds)} · characters {s + 1:,}–{e:,} of {len(text):,}"
    await rs.store().log_read(room["id"], owner, client_id, "room_read", file_id=f["id"], page=page,
                              chars=len(body), alarm=alarm)
    await _history(room, client_id, [body], f"read {f['clean_path']} p{page}")
    return f"{head}\n\n{body}" + ("" if page == len(bounds) else f"\n\n[continues on page {page + 1}]")


async def _room_search(owner: str, client_id: str, args: dict[str, Any]) -> dict[str, Any]:
    room = await _room(owner, client_id, args.get("room"))
    query = str(args.get("query") or "").strip()
    if not query:
        raise ToolError("query is required")
    alarm = await _alarm(room, client_id, "room_search", args)
    try:
        limit = max(1, min(int(args.get("limit") or 10), MAX_SEARCH))
    except (TypeError, ValueError):
        limit = 10
    hits = await rs.store().search(room["id"], query, limit=limit, released_only=True)
    files = {f["id"]: f for f in await _released(room["id"])}
    out = []
    texts: dict[str, str] = {}
    for h in hits:
        f = files.get(h["file_id"])
        if f is None:
            continue
        text = texts.setdefault(f["id"], _text(room["id"], f))
        page = page_at(page_bounds(text), label_offset(text, h["page"]))
        out.append({"path": f["clean_path"], "page": page, "where": h["page"], "snippet": h["snippet"]})
    await rs.store().log_read(room["id"], owner, client_id, "room_search", chars=sum(len(o["snippet"]) for o in out),
                              query_chars=len(query), alarm=alarm)
    await _history(room, client_id, [o["snippet"] for o in out], "search")
    return {"room": room["id"], "results": out,
            **({} if out else {"note": "Nothing found. Every word has to appear in a passage, and names are tokens here — search for ⟦ORG_7⟧, not a company name. Try fewer or other words."})}


async def _room_deliver(owner: str, client_id: str, args: dict[str, Any]) -> dict[str, Any]:
    from captain_claw.flight_deck import rooms_inbox

    room = await _room(owner, client_id, args.get("room"))
    try:
        receipt = await rooms_inbox.deliver(room, client_id, title=args.get("title"), content=args.get("content"),
                                            fmt=args.get("format"), part=args.get("part"), of=args.get("of"))
    except rooms_inbox.DeliveryError as exc:
        raise ToolError(str(exc)) from None
    await rs.store().log_read(room["id"], owner, client_id, "room_deliver",
                              chars=len(str(args.get("content") or "")))
    await _close_task(room, client_id, args, receipt)
    return receipt


async def _room_deliver_file(owner: str, client_id: str, args: dict[str, Any]) -> dict[str, Any]:
    from captain_claw.flight_deck import rooms_inbox

    room = await _room(owner, client_id, args.get("room"))
    try:
        receipt = await rooms_inbox.deliver_claude_file(room, client_id, name=args.get("name"),
                                                        content_base64=args.get("content_base64"),
                                                        part=args.get("part"), of=args.get("of"))
    except rooms_inbox.DeliveryError as exc:
        raise ToolError(str(exc)) from None
    await rs.store().log_read(room["id"], owner, client_id, "room_deliver_file",
                              chars=len(str(args.get("content_base64") or "")))
    await _close_task(room, client_id, args, receipt)
    return receipt


MAX_GET_FILE = 10 * 1024 * 1024


async def _room_get_file(owner: str, client_id: str, args: dict[str, Any]) -> dict[str, Any]:
    """A released clean copy, whole. The text Claude reads was leak-checked when
    the room ran; a file carries more (charts' caches, comments, properties), so
    every piece of text in it is checked again before it goes out."""
    import asyncio
    import base64

    room = await _room(owner, client_id, args.get("room"))
    alarm = await _alarm(room, client_id, "room_get_file", args)
    f = _find_file(await _released(room["id"]), str(args.get("path") or ""))
    path = rs.store().clean_file(room["id"], f["id"], PurePosixPath(f["clean_path"]).suffix)
    if not path.exists():
        raise ToolError("That file isn't available as a file — read it with room_read.")
    blob = await asyncio.to_thread(path.read_bytes)
    if len(blob) > MAX_GET_FILE:
        raise ToolError(f"{f['clean_path']} is over {MAX_GET_FILE // (1024 * 1024)} MB — read it with room_read.")
    ext = PurePosixPath(f["clean_path"]).suffix.lower().lstrip(".")
    if ext in ("docx", "xlsx", "pptx"):
        from captain_claw.anonymize import office

        def every_text() -> str:
            return "\n".join(s.text for s in office.prepare(blob, ext, scrub=False).segments)

        text = await asyncio.to_thread(every_text)
    else:
        text = blob.decode("utf-8", errors="replace")
    _row, needle_list, _tokens = await _vault(room)
    hits = await asyncio.to_thread(rj.leaks, text + "\n" + f["clean_path"], needle_list)
    if hits:
        await rs.store().add_alarm(room["id"], room["owner_id"], client_id, "room_get_file (held back)", hits)
        raise ToolError(f"{f['clean_path']} can't be handed out as a file — read it with room_read.")
    await rs.store().log_read(room["id"], owner, client_id, "room_get_file", file_id=f["id"], chars=len(blob),
                              alarm=alarm)
    await _history(room, client_id, [text], f"file {f['clean_path']}")
    return {"path": f["clean_path"], "kind": ext, "bytes": len(blob),
            "base64": base64.b64encode(blob).decode("ascii"),
            "note": "Decode the base64 in your workspace. Tokens are inside, as in the text — keep them."}


def _file_handler(kind: str, spec_keys: tuple[str, ...]):
    async def handler(owner: str, client_id: str, args: dict[str, Any]) -> dict[str, Any]:
        from captain_claw.flight_deck import rooms_inbox

        room = await _room(owner, client_id, args.get("room"))
        spec = {k: args[k] for k in spec_keys if k in args}
        try:
            receipt = await rooms_inbox.deliver_file(room, client_id, kind=kind, title=args.get("title"), spec=spec)
        except rooms_inbox.DeliveryError as exc:
            raise ToolError(str(exc)) from None
        await rs.store().log_read(room["id"], owner, client_id, f"room_deliver_{kind}",
                                  chars=len(json.dumps(spec, ensure_ascii=False, default=str)))
        await _close_task(room, client_id, args, receipt)
        return receipt
    return handler


_WRITE = {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False}
_FILE_TOOLS: list[dict[str, Any]] = []


def file_tools() -> list[dict[str, Any]]:
    """The three file tools, their specs taken from the agents' own xlsx and
    pptx tools (pictures removed) so one spec is documented once."""
    if _FILE_TOOLS:
        return _FILE_TOOLS
    import copy

    from captain_claw.tools.pptx_tool import PptxTool
    from captain_claw.tools.xlsx_tool import XlsxTool

    xp = XlsxTool.parameters["properties"]
    slides = copy.deepcopy(PptxTool.parameters["properties"]["slides"])
    for key in ("image", "images"):
        slides["items"]["properties"].pop(key, None)
    slides["description"] = "The slides, in order. No pictures."
    look = {k: {"type": "string", "description": d} for k, d in (
        ("font", "Body font"), ("heading_font", "Heading font"), ("heading_color", "Heading colour, hex"),
        ("accent_color", "Accent colour, hex"))}
    room = {"type": "string", "description": "The room's id or codename."}
    title = {"type": "string", "description": "The file's title, in tokens like the rest."}
    _FILE_TOOLS.extend([
        {"name": "room_deliver_workbook", "description": "Give the person an Excel workbook built from your spec, in "
         "tokens: sheets with rows (values; =… is a formula) and optional edits (formats, number formats, column "
         "widths, charts, tables). The person gets the .xlsx with the real names put back; you get a receipt with "
         "any formula errors, never the restored file. Keep tokens exactly, sheet names too.",
         "inputSchema": {"type": "object", "properties": {"room": room, "title": title, "sheets": xp["sheets"],
                                                          "edits": xp["edits"], "task": _TASK},
                         "required": ["room", "title", "sheets"]}, "annotations": _WRITE},
        {"name": "room_deliver_deck", "description": "Give the person a PowerPoint deck built from your spec, in "
         "tokens: slides with title, body (- for bullets), table, chart and notes. No pictures. The person gets the "
         ".pptx with the real names put back; you get a receipt with anything that doesn't fit.",
         "inputSchema": {"type": "object", "properties": {"room": room, "title": title, "slides": slides,
                                                          "page_numbers": {"type": "boolean"}, "task": _TASK, **look},
                         "required": ["room", "title", "slides"]}, "annotations": _WRITE},
        {"name": "room_deliver_document", "description": "Give the person a Word document built from markdown in "
         "tokens (headings, lists, tables, bold). No pictures. The person gets the .docx with the real names put "
         "back; you get a receipt.",
         "inputSchema": {"type": "object", "properties": {"room": room, "title": title,
                                                          "markdown": {"type": "string"},
                                                          "page_numbers": {"type": "boolean"}, "task": _TASK},
                         "required": ["room", "title", "markdown"]}, "annotations": _WRITE},
        {"name": "room_deliver_file", "description": "Give the person a file you made yourself (Word, Excel, "
         "PowerPoint, markdown, text, CSV or JSON), with tokens inside, as base64. About 15 KB fits in one call; "
         "send bigger ones in parts (same name; part 1 of N …), or give the file to the person to drop into Flight "
         "Deck. The names are put back inside it there; you get a receipt.",
         "inputSchema": {"type": "object", "properties": {
             "room": room,
             "name": {"type": "string", "description": "The file's name with its extension, e.g. model.xlsx."},
             "content_base64": {"type": "string", "description": "The file (or this part of it), base64."},
             "part": {"type": "integer"}, "of": {"type": "integer"}, "task": _TASK},
             "required": ["room", "name", "content_base64"]}, "annotations": _WRITE},
        {"name": "room_get_file", "description": "A released file of a room as the file itself (base64) — the "
         "anonymized copy, tokens inside — to open with code in your workspace (an Excel model, say). Big files "
         "come back long: decode the base64 from wherever your tools put the result.",
         "inputSchema": {"type": "object", "properties": {
             "room": room, "path": {"type": "string", "description": "The file's path, as room_list_files gives it."}},
             "required": ["room", "path"]}, "annotations": _ANNOT},
    ])
    return _FILE_TOOLS


_HANDLERS = {
    "room_list": _room_list,
    "room_overview": _room_overview,
    "room_list_files": _room_list_files,
    "room_read": _room_read,
    "room_search": _room_search,
    "room_deliver": _room_deliver,
    "room_deliver_workbook": _file_handler("workbook", ("sheets", "edits")),
    "room_deliver_deck": _file_handler("deck", ("slides", "page_numbers", "font", "heading_font", "heading_color",
                                                "accent_color")),
    "room_deliver_document": _file_handler("document", ("markdown", "page_numbers")),
    "room_deliver_file": _room_deliver_file,
    "room_get_file": _room_get_file,
    "room_tasks": _room_tasks,
    "room_get_delivery": _room_get_delivery,
}


# ── JSON-RPC ────────────────────────────────────────────────────────


def _result(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


async def _call(user: dict[str, Any], client_id: str, req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    args = params.get("arguments") or {}
    handler = _HANDLERS.get(str(name))
    if handler is None:
        return _error(req_id, -32602, f"Unknown tool: {name}")
    if not isinstance(args, dict):
        return _error(req_id, -32602, "arguments must be an object")
    try:
        out = await handler(str(user["id"]), client_id, args)
    except ToolError as exc:
        return _result(req_id, {"content": [{"type": "text", "text": str(exc)}], "isError": True})
    except Exception as exc:  # noqa: BLE001 — in-band, and never with a value in it
        log.warning("Data room connector tool failed", tool=name, error=type(exc).__name__)
        return _result(req_id, {"content": [{"type": "text", "text": f"The tool failed ({type(exc).__name__})."}],
                                "isError": True})
    text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, indent=1)
    return _result(req_id, {"content": [{"type": "text", "text": text}], "isError": False})


async def _dispatch(msg: Any, user: dict[str, Any], client_id: str) -> dict[str, Any] | None:
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return _error(msg.get("id") if isinstance(msg, dict) else None, -32600, "Invalid Request")
    method, req_id = str(msg.get("method") or ""), msg.get("id")
    params = msg.get("params") or {}
    if req_id is None and method.startswith("notifications/"):
        return None
    if method == "initialize":
        asked = str((params or {}).get("protocolVersion") or "")
        return _result(req_id, {
            "protocolVersion": asked if asked in PROTOCOLS else PROTOCOLS[-1],
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "title": "Captain Data Rooms", "version": SERVER_VERSION},
            "instructions": INSTRUCTIONS,
        })
    if method == "tools/list":
        return _result(req_id, {"tools": TOOLS + file_tools()})
    if method == "tools/call":
        return await _call(user, client_id, req_id, params if isinstance(params, dict) else {})
    if method == "ping":
        return _result(req_id, {})
    return _error(req_id, -32601, f"Method not found: {method}")


@router.api_route(ROOMS_PATH.removeprefix("/fd"), methods=["GET", "DELETE"])
async def rooms_mcp_no_stream() -> Response:
    """No server-initiated stream and no sessions: 405, as Streamable HTTP asks."""
    return Response(status_code=405, headers={"Allow": "POST"})


@router.post(ROOMS_PATH.removeprefix("/fd"))
async def rooms_mcp(request: Request) -> Response:
    user, client_id = await caller(request)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(_error(None, -32700, "Parse error"), status_code=400)
    if isinstance(body, list):
        replies = [r for r in [await _dispatch(m, user, client_id) for m in body] if r is not None]
        return JSONResponse(replies) if replies else Response(status_code=202)
    reply = await _dispatch(body, user, client_id)
    return JSONResponse(reply) if reply is not None else Response(status_code=202)
