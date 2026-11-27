"""Vatra run ledger — the rule-based map behind `vatra(action="map")`.

Nothing told a working Vatra agent who is covering what right now, which sources
a teammate already read, which files exist, or what has been found so far
(one measured run: 8 of 26 page fetches repeated a teammate's URL). Flight Deck
already receives every worker's narration and tool calls live, so the ledger is
built from those events — no LLM, no GPU, no prompt-cache cost — and a worker
pulls a bounded map when it wants one.

Kinds (one row per (session, kind, key); a repeat bumps `count` and names the
other actor):

* ``focus``      — an owner's latest narration line (key = subtask id, replaced)
* ``source``     — a page fetched (key = URL) or a document extracted (``doc:<name>``)
* ``file``       — a file written or edited (key = its path)
* ``finding``    — a board note or finished output (key = ``board:<id>``)
* ``ask_answer`` — an answered ask (key = ``ask:<id>``)
* ``course``     — an owner's latest Jev course check (key = subtask id, replaced;
  run_course): stage, on track, next move
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

_SOURCE_TOOLS = frozenset({"web_fetch", "web_get"})
_DOC_TOOLS = frozenset({"pdf_extract", "docx_extract", "xlsx_extract", "pptx_extract"})
_FILE_TOOLS = frozenset({"write", "edit"})
MAP_MAX_CHARS = 5_000
_RECENT_SOURCES = 15
_PER_OWNER = 5
# Keep fire-and-forget writes referenced until they finish.
_pending: set[asyncio.Task] = set()
# A tool result that is a refusal or an error, not something read or written.
_FAILED_HEADS = ("error", "❌", "duplicate call blocked")
_BATCH_FAILED_RE = re.compile(r"\[URL: (\S+)\]\n\[Status: [^\]\n]*\]\n\[FAILED:")


def ledger_args(arguments: Any, output: Any = None) -> dict[str, Any] | None:
    """The tool arguments the ledger keys on (exact URL / path / query), small;
    ``failed`` when the call's output says it did not happen, and a batch keeps
    only the URLs it actually fetched."""
    if not isinstance(arguments, dict):
        return None
    out: dict[str, Any] = {}
    for key in ("url", "path", "query", "action", "title", "file_id"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()[:500]
    urls = arguments.get("urls")
    if isinstance(urls, list):
        out["urls"] = [str(u).strip()[:500] for u in urls[:10] if str(u).strip()]
    text = str(output or "")
    if text.lstrip()[:40].lower().startswith(_FAILED_HEADS):
        out["failed"] = True
    elif out.get("urls") and "[FAILED:" in text:
        bad = set(_BATCH_FAILED_RE.findall(text))
        out["urls"] = [u for u in out["urls"] if u not in bad]
    return out or None


def _schedule(coro) -> None:
    try:
        task = asyncio.get_running_loop().create_task(coro)
    except RuntimeError:
        coro.close()
        return
    _pending.add(task)
    task.add_done_callback(_pending.discard)


async def _upsert(sid: str, kind: str, key: str, **fields: Any) -> None:
    try:
        from captain_claw.flight_deck.auth import get_db

        await get_db().upsert_vatra_ledger(sid, kind, key, **fields)
    except Exception as e:  # noqa: BLE001 — the ledger never breaks a run
        log.debug("Vatra ledger write failed", kind=kind, error=str(e))


def record_action(sid: str, owner: str, subtask: str, label: str, act: dict) -> None:
    """Fold one streamed worker event into the ledger (fire-and-forget)."""
    if not sid:
        return
    tool = str(act.get("tool") or "")
    args = act.get("args") or {}
    who = {"owner": owner, "subtask_id": subtask, "label": label}
    if tool == "narration":
        detail = " ".join(str(act.get("detail") or "").split())[:200]
        if detail:
            _schedule(_upsert(sid, "focus", subtask or owner, detail=detail, replace=True, **who))
        return
    course = args.get("course")
    if tool == "completion_gate" and isinstance(course, dict):
        from captain_claw.flight_deck.run_course import describe

        _schedule(_upsert(sid, "course", subtask or owner, detail=describe(course)[:200],
                          replace=True, **who))
        return
    if args.get("failed"):
        return  # a refused or failed call read or wrote nothing
    if tool in _SOURCE_TOOLS and args.get("url"):
        _schedule(_upsert(sid, "source", args["url"], **who))
    elif tool == "web_fetch_batch" and args.get("urls"):
        for url in args["urls"]:
            _schedule(_upsert(sid, "source", url, **who))
    elif tool in _DOC_TOOLS and args.get("path"):
        name = os.path.basename(str(args["path"]).rstrip("/\\")) or str(args["path"])
        _schedule(_upsert(sid, "source", f"doc:{name}", **who))
    elif tool in _FILE_TOOLS and args.get("path"):
        _schedule(_upsert(sid, "file", args["path"], **who))


def record_finding(sid: str, owner: str, subtask: str, label: str, board_id: Any,
                   title: str, text: str = "", kind: str = "note") -> None:
    """A board note or finished output becomes a finding row."""
    if not sid or board_id is None:
        return
    head = (title or " ".join(str(text or "").split())[:120]).strip()
    detail = f"{'output' if kind == 'output' else 'note'}: {head}"[:200]
    _schedule(_upsert(sid, "finding", f"board:{board_id}", owner=owner, subtask_id=subtask,
                      label=label, detail=detail))


def record_ask_answer(sid: str, ask_id: Any, question: str, label: str, owner: str = "") -> None:
    """An answered ask, so a teammate doesn't ask the same thing again."""
    if not sid or ask_id is None:
        return
    _schedule(_upsert(sid, "ask_answer", f"ask:{ask_id}", owner=owner, label=label,
                      detail=" ".join(str(question or "").split())[:200]))


def _who(row: dict) -> str:
    others = [o for o in str(row.get("others") or "").split(",") if o]
    who = row.get("label") or row.get("owner") or "?"
    return f"{who} (+{len(others)} more)" if others else who


def _clip(text: Any, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _natural(subtask: str) -> list:
    """s2 before s10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", subtask)]


async def render_map(sid: str, caller_subtask: str = "", activity: dict | None = None) -> str:
    """The bounded run map a worker reads: per owner focus / files / findings,
    the most recent sources across the team, answered asks.

    Each part has its own budget: the caller's own line and the team-wide
    source list are always there; every other owner gets one line first, then
    files and findings while room remains, then "+N more owners"."""
    from captain_claw.flight_deck.auth import get_db

    rows = await get_db().list_vatra_ledger(sid, limit=800)
    activity = activity or {}
    by_kind: dict[str, list[dict]] = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row)
    subtasks = sorted({st for st in activity if st}
                      | {r.get("subtask_id") or "" for r in rows if r.get("subtask_id")}, key=_natural)
    crowded = len(subtasks) > 6

    # Team-wide sources and answered asks: reserved room (less of it in a big team).
    sources = by_kind.get("source", [])
    src_lines: list[str] = []
    if sources:
        n_lines = min(_RECENT_SOURCES, max(5, 20 - len(subtasks)))
        src_lines.append(f"Sources read across the team: {len(sources)}. Most recent:")
        src_lines += [f"- {_clip(r['key'], 100 if crowded else 160)} — {_clip(_who(r), 60)}"
                      for r in sources[:n_lines]]
        src_lines.append("A page a teammate already fetched comes back from web_fetch as the saved "
                         "copy (pointer + preview) — no need to avoid it, but don't re-research it.")
    ask_lines: list[str] = []
    asks = by_kind.get("ask_answer", [])
    if asks:
        ask_lines.append("Answered asks (see `vatra inbox`/`search`):")
        ask_lines += [f"- #{r['key'].split(':', 1)[-1]} {_clip(r.get('detail'), 140)}"
                      for r in asks[:_PER_OWNER]]
    header = "Run map — who is covering what (live, rule-based)."
    footer = "Details: `vatra(action=\"search\", query=…)` for findings, or read a teammate's file."
    tail = "\n\n".join(part for part in ("\n".join(src_lines), "\n".join(ask_lines), footer) if part)

    def _label(st: str, mine: list[dict]) -> str:
        rec = activity.get(st) or {}
        return _clip(rec.get("role") or next((r.get("label") for r in mine if r.get("label")), "") or st, 80)

    files_of = {st: [r["key"] for r in by_kind.get("file", []) if r.get("subtask_id") == st] for st in subtasks}
    you = ""
    if caller_subtask:
        mine = [r for r in rows if r.get("subtask_id") == caller_subtask]
        n_src = sum(1 for r in sources if r.get("subtask_id") == caller_subtask)
        you = (f"## YOU — {_label(caller_subtask, mine)} (piece {caller_subtask}): "
               f"{len(files_of.get(caller_subtask, []))} file(s), {n_src} source(s) so far")

    # Every other owner: one line first …
    others = [st for st in subtasks if st != caller_subtask]
    heads: dict[str, str] = {}
    details: dict[str, list[str]] = {}
    for st in others:
        mine = [r for r in rows if r.get("subtask_id") == st]
        state = (activity.get(st) or {}).get("state") or ("active" if mine else "not started")
        focus = next((r for r in by_kind.get("focus", []) if r.get("key") == st), None)
        head = f"## {_label(st, mine)} · {state} (piece {st})"
        course = next((r for r in by_kind.get("course", []) if r.get("key") == st), None)
        if course and course.get("detail"):
            head += f" · {_clip(course['detail'], 80)}"
        if focus and focus.get("detail"):
            head += f" — {_clip(focus['detail'], 120 if crowded else 200)}"
        heads[st] = head
        extra: list[str] = []
        files = files_of.get(st, [])
        if files:
            extra.append("files: " + ", ".join(_clip(f, 120) for f in files[:3])
                         + (f" (+{len(files) - 3} more)" if len(files) > 3 else ""))
        finds = [f"#{r['key'].split(':', 1)[-1]} {_clip(r.get('detail'), 100)}"
                 for r in by_kind.get("finding", []) if r.get("subtask_id") == st]
        if finds:
            extra.append("findings: " + " | ".join(finds[:3])
                         + (f" (+{len(finds) - 3} more)" if len(finds) > 3 else ""))
        n_src = sum(1 for r in sources if r.get("subtask_id") == st)
        if n_src:
            extra.append(f"sources read: {n_src}")
        details[st] = extra
    room = MAP_MAX_CHARS - len(header) - len(you) - len(tail) - 120
    shown: list[str] = []
    used = 0
    for st in others:
        if used + len(heads[st]) + 2 > room:
            break
        shown.append(st)
        used += len(heads[st]) + 2
    # … then their files and findings while room remains.
    blocks: dict[str, list[str]] = {st: [heads[st]] for st in shown}
    for st in shown:
        for line in details[st]:
            if used + len(line) + 1 <= room:
                blocks[st].append(line)
                used += len(line) + 1
    owner_text = "\n\n".join("\n".join(blocks[st]) for st in shown)
    if len(shown) < len(others):
        owner_text += (f"\n\n(+{len(others) - len(shown)} more owner(s) — "
                       "`vatra search` for their findings)")
    text = "\n\n".join(part for part in (header, you, owner_text, tail) if part)
    return text[:MAP_MAX_CHARS]


async def coverage_appendix(sid: str, limit_chars: int = 3_000) -> str:
    """A short coverage summary for the reporter: per owner, files and sources."""
    from captain_claw.flight_deck.auth import get_db

    try:
        rows = await get_db().list_vatra_ledger(sid, kinds=["file", "source"], limit=800)
    except Exception:  # noqa: BLE001
        return ""
    if not rows:
        return ""
    per: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        who = row.get("label") or row.get("owner") or "?"
        per.setdefault(who, {"file": [], "source": []})[row["kind"]].append(row["key"])
    lines = ["## Team coverage (from the run ledger)"]
    for who, kinds in sorted(per.items()):
        lines.append(f"- {who}: {len(kinds['source'])} source(s) read; files: "
                     + (", ".join(kinds["file"][:8]) or "none"))
    text = "\n".join(lines)
    return text[:limit_chars]
