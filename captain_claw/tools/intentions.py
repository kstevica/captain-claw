"""Intentions tool — record and manage future actions under consideration.

Two kinds of intention:
  * ``user``  — a note-to-self for the person; surfaced back to them later.
  * ``agent`` — a proactive action the agent intends to take. Low-risk ones
    are announced; anything that sends/changes data is asked first.

Phase 1 exposes create/list/update/snooze/cancel/done. Surfacing and the
approve/announce flow over WhatsApp / Flight Deck / glasses arrive in Phase 2;
agent intentions created now sit as ``proposed`` until that lands.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from captain_claw.intentions import (
    CATEGORIES,
    OPEN_STATUSES,
    RISKS,
    get_intentions_manager,
)
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


def _fmt(it: dict[str, Any]) -> str:
    bits = [f"[{it['id']}]", f"({it['origin']}/{it['status']})", it.get("title", "")]
    why = (it.get("why") or "").strip()
    line = " ".join(b for b in bits if b)
    if why:
        line += f" — why: {why}"
    repeat = (it.get("repeat") or "").strip()
    if repeat:
        line += f" [repeat: {repeat}]"
    return line


class IntentionsTool(Tool):
    """Record and manage user notes-to-self and the agent's proactive intentions."""

    name = "intentions"
    description = (
        "Record and manage 'intentions' — future actions under consideration. "
        "Use origin='user' to save a note-to-self for the person (a reminder of "
        "something they want to do). Use origin='agent' for something YOU intend "
        "to do for them later (e.g. 'send a Monday portfolio brief'); set "
        "risk='low' for read-only/no-send actions (will be announced) or "
        "risk='normal'/'high' for anything that sends or changes data (will ask "
        "permission). Actions: create, list, update, snooze, cancel, done. "
        "Always capture 'why' (the motivation) so it can be judged later."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "update", "snooze", "cancel", "done"],
                "description": "Operation to perform.",
            },
            "origin": {
                "type": "string",
                "enum": ["user", "agent"],
                "description": "'user' = note-to-self; 'agent' = a proactive action you intend to take.",
            },
            "title": {"type": "string", "description": "Short summary of the intention."},
            "body": {"type": "string", "description": "Fuller description / details."},
            "why": {"type": "string", "description": "Motivation — why this matters (recommended)."},
            "category": {
                "type": "string",
                "enum": sorted(CATEGORIES),
                "description": "Kind of intention.",
            },
            "risk": {
                "type": "string",
                "enum": sorted(RISKS),
                "description": "Agent intentions: 'low' read-only (announced) vs 'normal'/'high' (asks permission).",
            },
            "repeat": {
                "type": "string",
                "description": "Optional recurrence, e.g. 'daily 09:00', 'weekly fri 17:00', 'every 15m'.",
            },
            "intention_id": {
                "type": "string",
                "description": "Target id (for update/snooze/cancel/done).",
            },
            "snooze_days": {
                "type": "number",
                "description": "For 'snooze': days to defer surfacing (default 1).",
            },
            "status_filter": {
                "type": "string",
                "description": "For 'list': filter by status (e.g. 'active', 'proposed'). Default: open ones.",
            },
            "limit": {"type": "integer", "description": "Max results for 'list' (default 20)."},
        },
        "required": ["action"],
    }

    async def execute(self, action: str, **kwargs: Any) -> ToolResult:
        mgr = get_intentions_manager()
        session_id = str(kwargs.get("_session_id", "") or "").strip()
        try:
            if action == "create":
                return await self._create(mgr, session_id, kwargs)
            if action == "list":
                return await self._list(mgr, kwargs)
            if action in ("cancel", "done"):
                return await self._set_status(mgr, action, kwargs)
            if action == "snooze":
                return await self._snooze(mgr, kwargs)
            if action == "update":
                return await self._update(mgr, kwargs)
            return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as exc:
            log.warning("intentions tool error: %s", exc)
            return ToolResult(success=False, error=str(exc))

    async def _create(self, mgr: Any, session_id: str, kw: dict[str, Any]) -> ToolResult:
        title = str(kw.get("title") or "").strip()
        if not title:
            return ToolResult(success=False, error="'title' is required for create.")
        origin = str(kw.get("origin") or "agent").strip()
        it = await mgr.create(
            origin=origin,
            title=title,
            body=str(kw.get("body") or ""),
            why=str(kw.get("why") or ""),
            category=str(kw.get("category") or "other"),
            risk=str(kw.get("risk") or "normal"),
            repeat=(str(kw.get("repeat")).strip() or None) if kw.get("repeat") else None,
            source_session=session_id,
        )
        note = ""
        if origin == "agent" and it["approval_mode"] == "ask":
            note = " I'll ask the user before acting (pending the approval flow)."
        elif origin == "agent" and it["approval_mode"] == "announce":
            note = " I'll announce this before acting (pending the announce flow)."
        return ToolResult(
            success=True,
            content=f"Recorded intention {_fmt(it)}.{note}",
        )

    async def _list(self, mgr: Any, kw: dict[str, Any]) -> ToolResult:
        limit = int(kw.get("limit") or 20)
        status_filter = str(kw.get("status_filter") or "").strip()
        origin = str(kw.get("origin") or "").strip() or None
        if status_filter:
            items = await mgr.list(origin=origin, status=status_filter, limit=limit)
        else:
            items = await mgr.list(origin=origin, statuses=list(OPEN_STATUSES), limit=limit)
        if not items:
            return ToolResult(success=True, content="No matching intentions.")
        return ToolResult(
            success=True,
            content="Intentions:\n" + "\n".join(f"- {_fmt(i)}" for i in items),
        )

    async def _set_status(self, mgr: Any, action: str, kw: dict[str, Any]) -> ToolResult:
        iid = str(kw.get("intention_id") or "").strip()
        if not iid:
            return ToolResult(success=False, error="'intention_id' is required.")
        status = "cancelled" if action == "cancel" else "done"
        ok = await mgr.set_status(iid, status, decided_at=datetime.now(UTC).isoformat())
        if not ok:
            return ToolResult(success=False, error=f"No intention with id {iid}.")
        return ToolResult(success=True, content=f"Marked {iid} as {status}.")

    async def _snooze(self, mgr: Any, kw: dict[str, Any]) -> ToolResult:
        iid = str(kw.get("intention_id") or "").strip()
        if not iid:
            return ToolResult(success=False, error="'intention_id' is required.")
        days = float(kw.get("snooze_days") or 1)
        until = (datetime.now(UTC) + timedelta(days=days)).isoformat()
        ok = await mgr.set_status(iid, "snoozed", next_surface_at=until)
        if not ok:
            return ToolResult(success=False, error=f"No intention with id {iid}.")
        return ToolResult(success=True, content=f"Snoozed {iid} until {until}.")

    async def _update(self, mgr: Any, kw: dict[str, Any]) -> ToolResult:
        iid = str(kw.get("intention_id") or "").strip()
        if not iid:
            return ToolResult(success=False, error="'intention_id' is required.")
        fields = {
            k: kw[k]
            for k in ("title", "body", "why", "category", "risk", "repeat")
            if k in kw and kw[k] is not None
        }
        if not fields:
            return ToolResult(success=False, error="Nothing to update.")
        ok = await mgr.update(iid, **fields)
        if not ok:
            return ToolResult(success=False, error=f"No intention with id {iid}.")
        return ToolResult(success=True, content=f"Updated {iid}.")
