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
    DECISION_RESOLUTIONS,
    OPEN_STATUSES,
    RISKS,
    follow_through,
    get_intentions_manager,
)
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

# Map freeform verdict words → canonical resolutions so the agent can pass
# whatever the user actually said.
_VERDICT = {
    "yes": "approved", "y": "approved", "approve": "approved", "approved": "approved",
    "ok": "approved", "okay": "approved", "sure": "approved", "go": "approved",
    "da": "approved", "može": "approved", "moze": "approved",
    "no": "declined", "n": "declined", "decline": "declined", "declined": "declined",
    "ne": "declined", "nope": "declined",
    "later": "snoozed", "snooze": "snoozed", "snoozed": "snoozed", "kasnije": "snoozed",
    "stop": "undone", "cancel": "undone", "undo": "undone", "undone": "undone",
}


def _current_waid(kw: dict[str, Any]) -> str:
    """Pull the current WhatsApp chat's WAID from the session, if any."""
    s = kw.get("_session")
    md = getattr(s, "metadata", None) if s is not None else None
    if isinstance(md, dict):
        return str(md.get("whatsapp_waid") or "").strip()
    return ""


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
                "enum": ["create", "list", "update", "snooze", "cancel", "done", "resolve"],
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
            "action_prompt": {
                "type": "string",
                "description": "For agent intentions: the prompt to run when this executes/recurs (e.g. 'Compile and send the weekly portfolio brief').",
            },
            "verdict": {
                "type": "string",
                "description": "For 'resolve': the user's answer — yes/no/later/stop (freeform ok, e.g. 'sure', 'ne', 'kasnije').",
            },
            "decision_id": {
                "type": "string",
                "description": "For 'resolve': which pending decision (omit to resolve the most recent).",
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
            if action == "resolve":
                return await self._resolve(mgr, kwargs)
            return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as exc:
            log.warning("intentions tool error: %s", exc)
            return ToolResult(success=False, error=str(exc))

    async def _create(self, mgr: Any, session_id: str, kw: dict[str, Any]) -> ToolResult:
        title = str(kw.get("title") or "").strip()
        if not title:
            return ToolResult(success=False, error="'title' is required for create.")
        origin = str(kw.get("origin") or "agent").strip()
        action_prompt = str(kw.get("action_prompt") or "").strip()
        it = await mgr.create(
            origin=origin,
            title=title,
            body=str(kw.get("body") or ""),
            why=str(kw.get("why") or ""),
            category=str(kw.get("category") or "other"),
            risk=str(kw.get("risk") or "normal"),
            repeat=(str(kw.get("repeat")).strip() or None) if kw.get("repeat") else None,
            action_type="run_prompt" if action_prompt else "nudge",
            action_spec={"prompt": action_prompt} if action_prompt else None,
            source_session=session_id,
        )
        # For agent intentions, emit a decision the user resolves (any channel).
        if origin == "agent" and it["approval_mode"] in ("ask", "announce"):
            waid = _current_waid(kw)
            hint = {"waid": waid} if waid else None
            if it["approval_mode"] == "ask":
                q = f"Should I {title}?"
                dec = await mgr.create_decision(
                    intention_id=it["id"], kind="approval", prompt_text=q,
                    options=["yes", "no", "later"], target_hint=hint,
                )
                return ToolResult(success=True, content=(
                    f"Recorded {_fmt(it)}. Ask the user: \"{q}\" — then when they "
                    f"reply, call intentions(action='resolve', decision_id='{dec['id']}', "
                    f"verdict=<their answer>)."
                ))
            q = f"I'll {title} unless you say stop."
            dec = await mgr.create_decision(
                intention_id=it["id"], kind="announce_undo", prompt_text=q,
                options=["stop"], target_hint=hint,
            )
            return ToolResult(success=True, content=(
                f"Recorded {_fmt(it)}. Announce to the user: \"{q}\" — if they "
                f"object, call intentions(action='resolve', decision_id='{dec['id']}', "
                f"verdict='stop')."
            ))
        return ToolResult(success=True, content=f"Recorded intention {_fmt(it)}.")

    async def _resolve(self, mgr: Any, kw: dict[str, Any]) -> ToolResult:
        verdict = str(kw.get("verdict") or "").strip().lower()
        resolution = _VERDICT.get(verdict) or (verdict if verdict in DECISION_RESOLUTIONS else "")
        if not resolution:
            return ToolResult(
                success=False,
                error="verdict must map to approved/declined/snoozed/undone (e.g. yes/no/later/stop).",
            )
        decision_id = str(kw.get("decision_id") or "").strip()
        if not decision_id:
            pend = await mgr.list_pending_decisions(limit=1)
            if not pend:
                return ToolResult(success=False, error="No pending decision to resolve.")
            decision_id = pend[0]["id"]
        dec = await mgr.resolve_decision(decision_id, resolution, via="agent")
        if not dec:
            return ToolResult(success=False, error=f"No pending decision '{decision_id}'.")
        res = await follow_through(
            dec["intention_id"], resolution,
            source_waid=_current_waid(kw),
            source_session=str(kw.get("_session_id") or ""),
        )
        if not res.get("ok"):
            return ToolResult(success=False, error=res.get("error", "follow-through failed"))
        extra = f" (scheduler job {res['job_id']})" if res.get("job_id") else ""
        return ToolResult(success=True, content=f"Resolved as {res['outcome']}{extra}.")

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
