"""The tool-call register: every tool call this deck's agents make.

  agent  POST /fd/tool-audit/agent/calls    an agent's batch of records (tool_audit.py)
         GET  /fd/tool-audit/agent/{mine,digest}   its own calls (the activity tool, reflections)
  user   GET  /fd/tool-audit/{calls,facets,summary,export}
  admin  GET  /fd/admin/tool-audit/{calls,facets,summary,export,verify,events}
         GET/PUT /fd/admin/tool-audit/settings   (retention)

Who sees what:

* A user sees the calls of their own agents, and the calls made in their own
  turns on someone else's shared agent.
* A call made in a shared-agent member's turn shows its owner the agent, the
  time, the tool and the outcome — not its parameters or its error text, which
  can hold the member's own data. The member and the admins see all of it.
* Admins see every user's calls.
* An agent reads only its own calls made in its owner's turns — never another
  agent's, never a member's, never one made while its turn held members'
  private data (``private``) — shaped by ``tool_audit_digest`` (no parameter
  values beyond what identifies a call, no guard reasons). It can't write,
  change or delete anything here.

Identity comes from Flight Deck's own records: the calling agent is the one
that holds the ``X-Agent-Auth`` token, its owner the one recorded at spawn. A
member named in a record counts only while they are a member of that agent;
otherwise the row stays with the owner, marked unverified. The risk class is
Flight Deck's own (``tool_risk.py``), and parameters are masked again here.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from captain_claw.flight_deck.admin_routes import require_admin
from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.flight_deck.tool_audit_store import (
    MAX_RETENTION_DAYS,
    MIN_RETENTION_DAYS,
    ToolAuditStore,
)
from captain_claw.flight_deck.tool_risk import RISK_LABELS, RISKS, risk_for
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["tool-audit"])

MAX_BATCH = 500
PAGE_MAX = 200
GROUP_SPEAKER_PREFIX = "tg-group:"   # speaker.GROUP_SPEAKER_PREFIX
OUTCOMES = ("running", "ok", "error", "blocked", "timeout", "aborted", "interrupted")
HIDDEN_NOTE = "Hidden: a member's call on your shared agent"
_IDENTITY_TTL_S = 60.0
_PRUNE_EVERY_S = 6 * 3600

_data_dir: Path | None = None
_store: ToolAuditStore | None = None
_identities: dict[str, tuple[float, dict[str, str]]] = {}


def configure(*, data_dir: Path) -> None:
    global _data_dir
    _data_dir = Path(data_dir)


def store() -> ToolAuditStore:
    global _store
    if _store is None:
        if _data_dir is None:
            raise HTTPException(503, "The tool-call register is not configured on this deck")
        _store = ToolAuditStore(_data_dir / "tool_audit.db")
    return _store


def set_store(value: ToolAuditStore | None) -> None:
    """Tests: point the routes at another store."""
    global _store
    _store = value


async def close() -> None:
    if _store is not None:
        await _store.close()


async def retention_loop(stop: asyncio.Event) -> None:
    """Prune past the retention a minute after start, then every 6 hours."""
    delay = 60.0
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=delay)
            return
        except TimeoutError:
            pass
        delay = _PRUNE_EVERY_S
        try:
            n = await store().prune()
            if n:
                log.info("Tool-call register pruned", rows=n)
        except Exception as exc:
            log.warning("Tool-call register prune failed", error=str(exc))


# ── Agent ingest ──────────────────────────────────────────────────────


class CallRecord(BaseModel):
    boot: str = Field(pattern=r"^[0-9a-f]{16}$")
    seq: int = Field(ge=1, le=10**12)
    phase: Literal["start", "full", "end"]
    started_at: float
    tool: str = Field("", max_length=128)
    params: dict[str, Any] = Field(default_factory=dict)
    session_id: str = Field("", max_length=128)
    source: str = Field("", max_length=64)
    source_label: str = Field("", max_length=80)
    speaker_id: str = Field("", max_length=128)
    speaker_name: str = Field("", max_length=80)
    speaker_ref: str = Field("", max_length=160)
    outcome: str = Field("", max_length=16)
    reason: str = Field("", max_length=1000)
    duration_ms: int | None = Field(None, ge=0)
    ended_at: float | None = None
    private: bool = False


class CallBatch(BaseModel):
    records: list[CallRecord] = Field(max_length=MAX_BATCH)


def _identify_sync(token: str) -> dict[str, str] | None:
    from captain_claw.flight_deck import server as srv

    matched, owner, slug = srv._find_agent_by_auth(token)
    if not matched or not slug:
        return None
    endpoint = srv._find_agent_endpoint_by_auth(token) or {}
    return {"slug": slug, "owner": owner or "", "name": str(endpoint.get("name") or slug),
            "ref": str(endpoint.get("agent_ref") or f"proc-{slug}")}


async def _identify(request: Request) -> dict[str, str]:
    """The calling agent, from this deck's own records (403 when it isn't one)."""
    from captain_claw.flight_deck.context_packs import agent_transport_gate

    agent_transport_gate(request)
    token = request.headers.get("X-Agent-Auth", "") or ""
    if not token:
        raise HTTPException(403, "Unknown agent")
    key = hashlib.sha256(token.encode("utf-8", "replace")).hexdigest()
    now = time.monotonic()
    hit = _identities.get(key)
    if hit and now - hit[0] < _IDENTITY_TTL_S:
        return hit[1]
    who = await asyncio.to_thread(_identify_sync, token)
    if who is None:
        raise HTTPException(403, "Unknown agent")
    if len(_identities) > 1024:
        _identities.clear()
    _identities[key] = (now, who)
    return who


async def _speaker(rec: CallRecord, who: dict[str, str], token: str,
                   cache: dict[str, Any]) -> tuple[int, str, str]:
    """``(member, speaker_id, speaker_label)`` for a record, the member
    checked against this agent's shares."""
    sid = rec.speaker_id.strip()
    name = rec.speaker_name.strip()
    if sid.startswith(GROUP_SPEAKER_PREFIX):
        return 0, "", name or "Telegram group"
    if not sid:
        return (1, "", name or "Member") if (name or rec.source == "member") else (0, "", "")
    if sid not in cache:
        ok = False
        try:
            from captain_claw.flight_deck import agent_sharing
            from captain_claw.flight_deck.context_packs import agent_ref_for_auth

            if "ref" not in cache:
                cache["ref"] = await asyncio.to_thread(agent_ref_for_auth, token)
            if cache["ref"] and who["owner"]:
                ok = await agent_sharing.member_check(get_db(), cache["ref"], who["owner"], sid)
        except Exception as exc:
            log.warning("Tool-call register: member check failed", error=type(exc).__name__)
        cache[sid] = ok
    if cache[sid]:
        return 1, sid, name or "Member"
    return 1, "", f"{name or 'Member'} (unverified)"


@router.post("/fd/tool-audit/agent/calls")
async def agent_calls(body: CallBatch, request: Request) -> dict:
    """An agent reports tool calls it made (or is making)."""
    from captain_claw.tool_audit import mask_text, mask_tree

    who = await _identify(request)
    token = request.headers.get("X-Agent-Auth", "") or ""
    cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for rec in body.records:
        member, speaker_id, speaker_label = await _speaker(rec, who, token, cache)
        params = mask_tree(rec.params)
        tool = rec.tool.strip() or "(unknown)"
        outcome = rec.outcome if rec.outcome in OUTCOMES and rec.outcome != "running" else "error"
        rows.append({
            "phase": rec.phase, "owner_id": who["owner"], "member": member,
            "speaker_id": speaker_id, "speaker_label": speaker_label,
            "agent_slug": who["slug"], "agent_name": who["name"], "agent_ref": who["ref"],
            "boot": rec.boot, "seq": rec.seq, "started_at": rec.started_at, "tool": tool,
            "risk": risk_for(tool, params),
            "params": json.dumps(params, ensure_ascii=False, separators=(",", ":"), default=str),
            "session_id": rec.session_id, "source": rec.source, "source_label": rec.source_label,
            "outcome": outcome, "reason": mask_text(rec.reason)[:1000],
            "duration_ms": rec.duration_ms, "ended_at": rec.ended_at, "private": 1 if rec.private else 0,
        })
    written = await store().record(rows)
    return {"ok": True, "accepted": len(rows), "written": written}


# ── The agent's own record (tools/activity.py, reflections) ───────────

AGENT_PAGE_MAX = 50
DIGEST_DEFAULT_S = 7 * 86400
DIGEST_MAX_S = 30 * 86400
DIGEST_ROWS = 5000


def _agent_scope(who: dict[str, str]) -> dict[str, Any]:
    """This agent's calls in its owner's turns, none made while the turn held
    members' private data (their parameters may restate it, and what the
    agent reads here reaches its reflections), nor in an onboarding exam's
    questions (which keep nothing). The owner too: a slug reused after an
    agent was removed doesn't inherit the old agent's record."""
    return {"agent_slug": who["slug"], "owner_id": who["owner"], "member": 0, "private": 0, "not_source": "exam"}


@router.get("/fd/tool-audit/agent/mine")
async def agent_mine(
    request: Request, tool: str = "", outcome: str = "", source: str = "", session: str = "",
    q: str = "", since: float | None = None, until: float | None = None,
    before_id: int | None = Query(None, ge=1), limit: int = Query(20, ge=1, le=AGENT_PAGE_MAX),
) -> dict:
    """An agent's own calls, newest first, the way it may read them."""
    from captain_claw import tool_audit_digest

    who = await _identify(request)
    scope = _agent_scope(who)
    f = {**_filters("", "", tool, outcome, "", source, session, q, since, until), **scope,
         "q_skip_reason": True}
    rows = await store().list({**f, "before_id": before_id}, limit=limit)
    counts = await store().summary(f)
    window = {"since": since, "until": until}
    members = await store().summary({**scope, **window, "member": 1, "private": None})
    private = await store().summary({**scope, **window, "private": 1})
    return {"calls": [tool_audit_digest.agent_row(r) for r in rows],
            "next_before": rows[-1]["id"] if len(rows) == limit else None,
            "total": counts["total"], "outcomes": counts["outcomes"],
            "member_calls": members["total"], "private_calls": private["total"]}


@router.get("/fd/tool-audit/agent/digest")
async def agent_digest(request: Request, since: float | None = None) -> dict:
    """Counts and patterns over an agent's own calls since *since* (default a
    week, at most 30 days back)."""
    from captain_claw import tool_audit_digest

    who = await _identify(request)
    now = time.time()
    start = max(float(since) if since else now - DIGEST_DEFAULT_S, now - DIGEST_MAX_S)
    rows = await store().list({**_agent_scope(who), "since": start}, limit=DIGEST_ROWS)
    return tool_audit_digest.digest(rows, since=start, until=now, capped=len(rows) == DIGEST_ROWS)


# ── Reading ───────────────────────────────────────────────────────────


def _filters(agent: str, ref: str, tool: str, outcome: str, risk: str, source: str, session: str,
             q: str, since: float | None, until: float | None) -> dict[str, Any]:
    if outcome and outcome not in OUTCOMES:
        raise HTTPException(400, "Unknown outcome")
    if risk and risk not in RISKS:
        raise HTTPException(400, "Unknown risk")
    return {"agent_slug": agent.strip(), "agent_ref": ref.strip(), "tool": tool.strip(),
            "outcome": outcome, "risk": risk,
            "source": source.strip(), "session_id": session.strip(), "q": q.strip()[:200],
            "since": since, "until": until}


async def _names(ids: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    db = get_db()
    for uid in ids:
        if not uid:
            continue
        try:
            u = await db.get_user_by_id(uid)
        except Exception:
            u = None
        out[uid] = (u.get("display_name") or u.get("email") or uid) if u else uid
    return out


def _shape(row: dict[str, Any], viewer: str, full: bool, names: dict[str, str]) -> dict[str, Any]:
    hidden = not full and bool(row["member"]) and row["speaker_id"] != viewer
    try:
        params = None if hidden else json.loads(row["params"] or "{}")
    except ValueError:
        params = {}
    return {
        "id": row["id"], "at": row["started_at"], "received_at": row["received_at"],
        "agent": {"slug": row["agent_slug"], "name": row["agent_name"] or row["agent_slug"],
                  "ref": row["agent_ref"]},
        "owner": {"id": row["owner_id"], "name": names.get(row["owner_id"], row["owner_id"]),
                  "you": row["owner_id"] == viewer},
        "member": bool(row["member"]),
        "speaker": {"id": row["speaker_id"], "label": row["speaker_label"],
                    "you": bool(row["speaker_id"]) and row["speaker_id"] == viewer},
        "tool": row["tool"], "risk": row["risk"], "params": params, "hidden": hidden,
        "session_id": row["session_id"], "source": row["source"], "source_label": row["source_label"],
        "outcome": row["outcome"], "reason": HIDDEN_NOTE if hidden and row["reason"] else row["reason"],
        "duration_ms": row["duration_ms"], "ended_at": row["ended_at"],
        "seq": row["seq"], "boot": row["boot"], "hash": row["hash"], "private": bool(row.get("private")),
    }


async def _page(filters: dict[str, Any], viewer: str, full: bool, before_id: int | None,
                limit: int) -> dict[str, Any]:
    limit = max(1, min(PAGE_MAX, int(limit)))
    rows = await store().list({**filters, "before_id": before_id}, limit=limit)
    names = await _names({r["owner_id"] for r in rows})
    return {"calls": [_shape(r, viewer, full, names) for r in rows],
            "next_before": rows[-1]["id"] if len(rows) == limit else None,
            "risk_labels": RISK_LABELS}


def _csv_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    # A spreadsheet must not run what an agent wrote as a formula.
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


_CSV_HEAD = ["id", "time_utc", "agent", "agent_slug", "owner", "speaker", "source", "tool", "risk",
             "outcome", "duration_ms", "reason", "params", "session_id", "boot", "seq", "hash"]


async def _export(filters: dict[str, Any], viewer: str, full: bool, fmt: str) -> StreamingResponse:
    if fmt not in ("csv", "jsonl"):
        raise HTTPException(400, "Pick csv or jsonl")
    names: dict[str, str] = {}

    async def rows():
        async for r in store().iterate(filters):
            if r["owner_id"] not in names:
                names.update(await _names({r["owner_id"]}))
            yield _shape(r, viewer, full, names)

    def iso(ts: Any) -> str:
        return datetime.fromtimestamp(float(ts or 0), tz=UTC).isoformat() if ts else ""

    async def body():
        if fmt == "jsonl":
            async for s in rows():
                yield json.dumps(s, ensure_ascii=False, default=str) + "\n"
            return
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(_CSV_HEAD)
        yield buf.getvalue()
        async for s in rows():
            buf.seek(0)
            buf.truncate()
            w.writerow([_csv_cell(v) for v in (
                s["id"], iso(s["at"]), s["agent"]["name"], s["agent"]["slug"], s["owner"]["name"],
                s["speaker"]["label"], s["source_label"] or s["source"], s["tool"], s["risk"],
                s["outcome"], s["duration_ms"], s["reason"],
                "" if s["params"] is None else json.dumps(s["params"], ensure_ascii=False, default=str),
                s["session_id"], s["boot"], s["seq"], s["hash"],
            )])
            yield buf.getvalue()

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    media = "text/csv; charset=utf-8" if fmt == "csv" else "application/x-ndjson"
    return StreamingResponse(body(), media_type=media, headers={
        "Content-Disposition": f'attachment; filename="tool-calls-{stamp}.{fmt}"'})


def _common(
    agent: str = "", ref: str = "", tool: str = "", outcome: str = "", risk: str = "", source: str = "",
    session: str = "", q: str = "", since: float | None = None, until: float | None = None,
) -> dict[str, Any]:
    """The filters every list, summary and export takes: ``agent`` is the
    agent's slug, ``ref`` the id the app keys it by (a container's id)."""
    return _filters(agent, ref, tool, outcome, risk, source, session, q, since, until)


# The user's own view. A deck without sign-in has one person: everything is theirs.

def _scope(user: dict) -> tuple[dict[str, Any], str, bool]:
    """``(filter, viewer id, full)`` for the user's own view."""
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    uid = str(user["id"])
    if not _fd_auth_enabled():
        return {}, uid, True
    return {"viewer": uid}, uid, False


@router.get("/fd/tool-audit/calls")
async def my_calls(
    f: dict = Depends(_common), before_id: int | None = Query(None, ge=1),
    limit: int = Query(100, ge=1, le=PAGE_MAX), user: dict = Depends(get_current_user),
) -> dict:
    scope, uid, full = _scope(user)
    return await _page({**f, **scope}, uid, full, before_id, limit)


@router.get("/fd/tool-audit/facets")
async def my_facets(since: float | None = None, user: dict = Depends(get_current_user)) -> dict:
    return await store().facets({**_scope(user)[0], "since": since})


@router.get("/fd/tool-audit/summary")
async def my_summary(f: dict = Depends(_common), user: dict = Depends(get_current_user)) -> dict:
    return await store().summary({**f, **_scope(user)[0]})


@router.get("/fd/tool-audit/export")
async def my_export(f: dict = Depends(_common), format: str = "csv",
                    user: dict = Depends(get_current_user)) -> StreamingResponse:
    scope, uid, full = _scope(user)
    return await _export({**f, **scope}, uid, full, format)


# Admins: every user's calls.

@router.get("/fd/admin/tool-audit/calls")
async def admin_calls(
    f: dict = Depends(_common), user_id: str = "", before_id: int | None = Query(None, ge=1),
    limit: int = Query(100, ge=1, le=PAGE_MAX), admin: dict = Depends(require_admin),
) -> dict:
    return await _page({**f, "user": user_id.strip()}, str(admin["id"]), True, before_id, limit)


@router.get("/fd/admin/tool-audit/facets")
async def admin_facets(since: float | None = None, user_id: str = "",
                       _admin: dict = Depends(require_admin)) -> dict:
    return await store().facets({"user": user_id.strip(), "since": since})


@router.get("/fd/admin/tool-audit/summary")
async def admin_summary(f: dict = Depends(_common), user_id: str = "",
                        _admin: dict = Depends(require_admin)) -> dict:
    return await store().summary({**f, "user": user_id.strip()})


@router.get("/fd/admin/tool-audit/export")
async def admin_export(f: dict = Depends(_common), user_id: str = "", format: str = "csv",
                       admin: dict = Depends(require_admin)) -> StreamingResponse:
    await store().add_event("exported", str(admin["id"]), {"format": format, "user": user_id})
    return await _export({**f, "user": user_id.strip()}, str(admin["id"]), True, format)


@router.get("/fd/admin/tool-audit/verify")
async def admin_verify(admin: dict = Depends(require_admin)) -> dict:
    result = await store().verify()
    await store().add_event("verified", str(admin["id"]), {
        "ok": result["ok"], "checked": result["checked"], "problems": len(result["problems"]),
        "gaps": len(result["gaps"])})
    if result["gaps"]:
        names = await _names({g["owner_id"] for g in result["gaps"]})
        for g in result["gaps"]:
            g["owner_name"] = names.get(g["owner_id"], g["owner_id"])
    return result


@router.get("/fd/admin/tool-audit/events")
async def admin_events(_admin: dict = Depends(require_admin)) -> dict:
    events = await store().events()
    names = await _names({e["actor"] for e in events if e.get("actor") and e["actor"] != "retention"})
    for e in events:
        e["actor_name"] = names.get(e["actor"], e["actor"])
    return {"events": events}


class AuditSettings(BaseModel):
    retention_days: int = Field(ge=MIN_RETENTION_DAYS, le=MAX_RETENTION_DAYS)


@router.get("/fd/admin/tool-audit/settings")
async def admin_settings(_admin: dict = Depends(require_admin)) -> dict:
    return {"retention_days": await store().retention_days(),
            "min_days": MIN_RETENTION_DAYS, "max_days": MAX_RETENTION_DAYS}


@router.put("/fd/admin/tool-audit/settings")
async def admin_save_settings(body: AuditSettings, admin: dict = Depends(require_admin)) -> dict:
    days = await store().set_retention_days(body.retention_days, actor=str(admin["id"]))
    return {"retention_days": days, "min_days": MIN_RETENTION_DAYS, "max_days": MAX_RETENTION_DAYS}
