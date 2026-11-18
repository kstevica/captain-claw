"""Crew — context a person hands a worker from their own agents.

Two kinds, both kept as uploads (:mod:`crew_attach`) so they go with a new task or a
message exactly like a file someone attached, and the worker finds them in its
``.crew-attachments`` folder, told that what they say is information, not instructions:

- **Files an agent made**: from its workspace folder on this computer (``saved/`` is in
  it), read from disk so an agent that's asleep needn't be woken. Only the person's own
  agents, never a path outside that folder, never a file that looks like keys.
- **What was said with an agent**: from Flight Deck's chat log, which is the person's own
  (it's kept only when sign-in is on). A range of one conversation, the person's messages
  and the agent's final answers only, written out as a Markdown file.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from captain_claw.crew_client import MAX_TASK_FILES

MAX_LISTED = 2000
MAX_TURNS = 3000
_PREVIEW = 400
_AUTO_LANE = "AUTO"  # the automation lane: scheduled runs and plans, nobody talking
_SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv", ".cache", ".pytest_cache", ".mypy_cache"})


# ── Files an agent made ──

def _server():
    from captain_claw.flight_deck import server
    return server


def own_agents(user: dict) -> list[dict[str, str]]:
    """The agents whose files this person may hand over: their own. An agent with no owner
    (made before sign-in was on) is an admin's to use."""
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    server = _server()
    uid = str(user.get("id") or "")
    admin = user.get("role") == "admin"
    signed_in = _fd_auth_enabled()
    out = []
    for slug, entry in server._load_process_registry().items():
        if not isinstance(entry, dict):
            continue
        owner = str(entry.get("owner") or "")
        if signed_in and owner != uid and not (admin and not owner):
            continue
        if not _workspace(slug).is_dir():
            continue
        out.append({"id": str(slug), "name": str(entry.get("name") or slug)})
    return sorted(out, key=lambda a: a["name"].lower())


def _workspace(slug: str) -> Path:
    return _server().DATA_DIR / slug / "data" / "workspace"


def _root_of(user: dict, slug: str) -> Path:
    if not any(a["id"] == slug for a in own_agents(user)):
        raise HTTPException(404, "You have no agent like that here.")
    return _workspace(slug).resolve()


def _agent_name(user: dict, slug: str) -> str:
    return next((a["name"] for a in own_agents(user) if a["id"] == slug), slug)


def why_not(path: Path) -> str:
    """Why a file mustn't go to a worker, or "": the same rules as an agent's mail."""
    from captain_claw import mail_attachments as rules

    if any(p.lower() in rules._SECRET_DIRS for p in path.parts[:-1]):
        return "it's in a folder that holds keys or credentials"
    if rules._SECRET_NAME_RE.search(path.name):
        return "it looks like a key or a secrets file"
    try:
        with path.open("rb") as fh:
            if fh.read(len(rules._SQLITE_MAGIC)) == rules._SQLITE_MAGIC:
                return "it's a database, which can hold keys: export what's needed to a CSV and attach that"
    except OSError:
        return "it can't be read"
    return ""


def list_files(user: dict, slug: str) -> dict[str, Any]:
    """What's in the agent's workspace, newest first. Hidden files and folders, and the
    heavy ones a project keeps (``node_modules``, ``.git``), are left out; nothing is
    followed through a link."""
    root = _root_of(user, slug)
    found: list[dict[str, Any]] = []
    for here, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _SKIP_DIRS]
        for name in names:
            path = Path(here) / name
            if name.startswith(".") or path.is_symlink():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            found.append({"path": path.relative_to(root).as_posix(), "name": name,
                          "size": stat.st_size, "modified": stat.st_mtime})
    found.sort(key=lambda f: -f["modified"])
    return {"files": found[:MAX_LISTED], "more": len(found) > MAX_LISTED}


def stage_files(user: dict, slug: str, paths: list[str]) -> list[dict[str, Any]]:
    """The chosen files, kept for the person like uploads. All or none: a file that can't
    go stops the lot, named, before anything is kept."""
    from captain_claw.flight_deck import crew_attach
    from captain_claw.vfs import safe_join

    root = _root_of(user, slug)
    wanted = list(dict.fromkeys(p for p in paths if str(p or "").strip()))
    if not wanted:
        raise HTTPException(400, "Choose a file.")
    if len(wanted) > MAX_TASK_FILES:
        raise HTTPException(413, f"A task can have {MAX_TASK_FILES} attached files.")
    sources = []
    for rel in wanted:
        path = safe_join(root, rel)
        if path is None or path == root or not path.is_file():
            raise HTTPException(404, f"{Path(str(rel)).name or rel} isn't in that agent's files any more.")
        reason = why_not(path)
        if reason:
            raise HTTPException(400, f"{path.name} can't be attached: {reason}.")
        sources.append(path)
    made: list[dict[str, Any]] = []
    origin = f"Files of {_agent_name(user, slug)}"
    try:
        for path in sources:
            made.append(crew_attach.stage_copy(str(user["id"]), path, origin=origin))
    except BaseException:
        for meta in made:
            crew_attach.drop_upload(str(user["id"]), meta["id"])
        raise
    return made


# ── What was said with an agent ──

def _meta(row: dict) -> dict:
    try:
        meta = json.loads(row.get("metadata") or "{}")
    except (TypeError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def said(row: dict) -> bool:
    """A message the person wrote, or one of the agent's final answers. Not a slash command,
    nor what the runtime wrote in the person's name, nor an automated run's result."""
    from captain_claw import msg_origin

    text = str(row.get("content") or "").strip()
    if not text:
        return False
    meta = _meta(row)
    if row.get("role") == "user":
        return not text.startswith("/") and not meta.get("note") and not msg_origin.transcript_origin(text)
    if row.get("role") == "assistant":
        return not (meta.get("automation") or meta.get("task") or meta.get("narration"))
    return False


def _store_id(lane: str, uid: str) -> str:
    from captain_claw.flight_deck.chat_routes import _store_id
    return _store_id(lane, uid)


def _lane_parts(lane: str) -> tuple[str, str]:
    base, _, sub = lane.partition("::")
    return base, sub


async def conversations(db, user: dict) -> list[dict[str, Any]]:
    """Each chat this person has had with an agent, latest first. Their sessions are asked
    for one chat at a time: a long log takes a moment to count."""
    from captain_claw.flight_deck.chat_routes import _client_id

    uid = str(user["id"])
    out = []
    for row in await db.list_chat_sessions(uid):
        store_id = str(row.get("id") or "")
        lane = _client_id(store_id, uid)
        base, sub = _lane_parts(lane)
        if sub == _AUTO_LANE or not await db.get_chat_turns(store_id, uid, limit=1):
            continue
        out.append({"id": lane, "agent": str(row.get("agent_name") or "") or base,
                    "lane": sub, "updated_at": str(row.get("updated_at") or "")})
    return out


async def _topics(db, store_id: str, uid: str) -> list[dict]:
    return await db.list_chat_topics(store_id, uid)


async def sessions(db, user: dict, lane: str) -> list[dict[str, Any]]:
    """One chat's sessions (the timeline's topics), latest first. Turns from before the
    first divider, in a log older than dividers, are a session of their own."""
    uid = str(user["id"])
    store_id = _store_id(lane, uid)
    if not await db.get_chat_session(store_id, uid):
        raise HTTPException(404, "That conversation isn't here.")
    topics = await _topics(db, store_id, uid)
    # Counted as they'd be sent, so the number matches the list a person picks from.
    turns = [r for r in await db.get_chat_turns(store_id, uid, limit=20000) if said(r)]
    starts = [0, *(int(t["row_id"]) for t in topics)]
    counts = dict.fromkeys(starts, 0)
    began: dict[int, str] = {}
    at = 0
    for r in turns:
        while at + 1 < len(starts) and int(r["id"]) > starts[at + 1]:
            at += 1
        counts[starts[at]] += 1
        began.setdefault(starts[at], str(r.get("created_at") or ""))
    out = []
    if counts[0]:
        out.append({"id": 0, "title": "Earlier messages" if topics else "The whole conversation",
                    "started_at": began.get(0, ""), "count": counts[0]})
    for t in topics:
        row_id = int(t["row_id"])
        if not counts.get(row_id):
            continue
        title = str(t.get("title") or "") or " ".join(str(t.get("first_user_message") or "").split())[:80]
        out.append({"id": row_id, "title": title or "Untitled", "started_at": str(t.get("started_at") or ""),
                    "count": counts[row_id]})
    out.reverse()
    return out


async def _bounds(db, store_id: str, uid: str, session: int) -> tuple[int, int | None, str]:
    """Where a session starts and ends in the log (rows between, exclusive), and its title."""
    topics = await _topics(db, store_id, uid)
    if session == 0:
        return 0, (topics[0]["row_id"] if topics else None), ""
    for i, t in enumerate(topics):
        if int(t["row_id"]) == session:
            end = int(topics[i + 1]["row_id"]) if i + 1 < len(topics) else None
            return session, end, str(t.get("title") or "")
    raise HTTPException(404, "That session isn't in this conversation any more.")


async def messages(db, user: dict, lane: str, session: int) -> dict[str, Any]:
    """A session's messages as a person picks from them: what would be sent, each cut short."""
    uid = str(user["id"])
    store_id = _store_id(lane, uid)
    if not await db.get_chat_session(store_id, uid):
        raise HTTPException(404, "That conversation isn't here.")
    after, before, _ = await _bounds(db, store_id, uid, session)
    rows = [r for r in await db.get_chat_turns(store_id, uid, after, before, limit=MAX_TURNS * 3) if said(r)]
    out = []
    for r in rows[:MAX_TURNS]:
        text = str(r.get("content") or "").strip()
        out.append({"id": int(r["id"]), "role": r["role"], "at": str(r.get("created_at") or ""),
                    "text": text[:_PREVIEW], "chars": len(text)})
    return {"messages": out, "more": len(rows) > MAX_TURNS}


def _when(stamp: str) -> str:
    try:
        moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return stamp
    return moment.strftime("%Y-%m-%d %H:%M") + (" UTC" if moment.utcoffset() is not None and not moment.utcoffset() else "")


def write_out(agent: str, title: str, rows: list[dict]) -> str:
    """A conversation as a worker reads it: who said what, when, and nothing else."""
    name = agent or "the agent"
    head = f"# Conversation with {name}" + (f": {title}" if title else "")
    span = f"from {_when(rows[0]['created_at'])} to {_when(rows[-1]['created_at'])}" if rows else ""
    lines = [head, "",
             f"{len(rows)} messages {span}. The person's messages and {name}'s final answers only: "
             "its steps, tool calls and automated messages are left out.", ""]
    for r in rows:
        who = "User" if r["role"] == "user" else name
        lines += ["---", "", f"**{who}** · {_when(str(r.get('created_at') or ''))}", "",
                  str(r.get("content") or "").strip(), ""]
    return "\n".join(lines).rstrip() + "\n"


def _file_name(agent: str, rows: list[dict]) -> str:
    day = str(rows[0].get("created_at") or "")[:10] if rows else ""
    stem = re.sub(r"\s+", "-", agent.strip()) or "agent"
    return f"conversation-{stem}-{day}.md" if day else f"conversation-{stem}.md"


async def stage_conversation(db, user: dict, lane: str, session: int, first: int, last: int) -> dict[str, Any]:
    """Messages ``first``..``last`` of one session, written out and kept like an upload."""
    from captain_claw.flight_deck import crew_attach

    uid = str(user["id"])
    store_id = _store_id(lane, uid)
    chat = await db.get_chat_session(store_id, uid)
    if not chat:
        raise HTTPException(404, "That conversation isn't here.")
    after, before, title = await _bounds(db, store_id, uid, session)
    first, last = min(first, last), max(first, last)
    low = max(after, first - 1)
    high = min(before, last + 1) if before is not None else last + 1
    rows = [r for r in await db.get_chat_turns(store_id, uid, low, high, limit=20000) if said(r)]
    if not rows:
        raise HTTPException(400, "There's nothing to send in those messages.")
    agent = str(chat.get("agent_name") or "") or _lane_parts(lane)[0]
    origin = f"Conversation with {agent}, {len(rows)} message{'' if len(rows) == 1 else 's'}"
    return crew_attach.stage_text(uid, _file_name(agent, rows), write_out(agent, title, rows), origin=origin)
