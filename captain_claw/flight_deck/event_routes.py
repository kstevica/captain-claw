"""Flight Deck event-spine API (#2). Intake + inspection for external events.

  * POST /fd/events/ingest — normalize one signal into the spine (manual exerciser
    now; the generic webhook receiver later). On a genuinely new event it nudges
    the arbiter (debounced force-pulse) so the loop reacts promptly.
  * GET  /fd/events        — list events (for the UI / testing).
"""

from __future__ import annotations

import asyncio
import os
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status

from captain_claw.flight_deck.auth import get_optional_user
from captain_claw.flight_deck.events import get_store

router = APIRouter(prefix="/fd/events", tags=["events"])

# Per-user debounce so a burst of events triggers at most one pulse per window.
_PULSE_DEBOUNCE_SECONDS = 20.0
_last_pulse: dict[str, float] = {}
_pulse_tasks: set = set()


def _auth_enabled() -> bool:
    return os.environ.get("FD_AUTH_ENABLED", "true").lower() in ("true", "1", "yes")


def _user_id(request: Request) -> str:
    uid = getattr(request.state, "user_id", "") or ""
    if _auth_enabled() and not uid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return uid


def _maybe_nudge_arbiter(user_id: str) -> None:
    """Debounced background force-pulse so a new event reaches the arbiter without
    waiting for the 180s heartbeat. Only when the loop is enabled for the user."""
    try:
        from captain_claw.flight_deck.autonomy import resolve_config
        if not resolve_config(user_id).get("enabled"):
            return
    except Exception:
        return
    now = time.monotonic()
    if now - _last_pulse.get(user_id, 0.0) < _PULSE_DEBOUNCE_SECONDS:
        return
    _last_pulse[user_id] = now
    try:
        from captain_claw.flight_deck.consciousness import pulse
        t = asyncio.create_task(pulse(user_id, force=True))
        _pulse_tasks.add(t)
        t.add_done_callback(_pulse_tasks.discard)
    except Exception:
        pass


@router.post("/ingest")
async def ingest_event(request: Request, _user: dict | None = Depends(get_optional_user)):
    """Insert a normalized event. Body: {source, event_type?, summary, body?,
    metadata?, dedup_key?}. Returns the event (or deduped=True if already seen)."""
    uid = _user_id(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be an object")
    source = str(body.get("source") or "").strip()
    summary = str(body.get("summary") or "").strip()
    if not source or not summary:
        raise HTTPException(status_code=400, detail="source and summary are required")
    evt = get_store().add_event(
        uid,
        source=source,
        event_type=str(body.get("event_type") or "").strip(),
        summary=summary,
        body=str(body.get("body") or ""),
        metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else {},
        dedup_key=str(body.get("dedup_key") or "").strip(),
    )
    if evt is None:
        return {"ok": True, "deduped": True}
    _maybe_nudge_arbiter(uid)
    return {"ok": True, "event": evt}


@router.post("/webhook")
async def webhook_ingest(request: Request):
    """Token-gated, SESSIONLESS push path for external systems (Gmail Pub/Sub,
    Zapier, a calendar push channel…) — sub-minute latency vs the 5-min poll.
    Disabled unless ``FD_EVENTS_WEBHOOK_TOKEN`` is set; the caller supplies it via
    the ``X-Webhook-Token`` header (or ``?token=``) and names the target user_id
    in the body. Body: {user_id?, source, summary, event_type?, body?, metadata?,
    dedup_key?}."""
    token = os.environ.get("FD_EVENTS_WEBHOOK_TOKEN", "").strip()
    if not token:
        raise HTTPException(status_code=404, detail="webhook ingest not enabled")
    supplied = request.headers.get("X-Webhook-Token") or request.query_params.get("token") or ""
    if supplied != token:
        raise HTTPException(status_code=403, detail="bad webhook token")
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be an object")
    uid = str(body.get("user_id") or "").strip() or "local"
    source = str(body.get("source") or "").strip()
    summary = str(body.get("summary") or "").strip()
    if not source or not summary:
        raise HTTPException(status_code=400, detail="source and summary are required")
    evt = get_store().add_event(
        uid, source=source,
        event_type=str(body.get("event_type") or "").strip(),
        summary=summary, body=str(body.get("body") or ""),
        metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else {},
        dedup_key=str(body.get("dedup_key") or "").strip(),
    )
    if evt is None:
        return {"ok": True, "deduped": True}
    _maybe_nudge_arbiter(uid)
    return {"ok": True, "event_id": evt["id"]}


@router.get("")
async def list_events(
    request: Request,
    status_filter: str | None = None,
    limit: int = 100,
    _user: dict | None = Depends(get_optional_user),
):
    uid = _user_id(request)
    return {"events": get_store().list_events(uid, status=status_filter, limit=limit)}
