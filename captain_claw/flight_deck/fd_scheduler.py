"""Flight Deck scheduler — a cron that runs agent prompts on a timer and
delivers the reply to WhatsApp or a channel (glasses HUD, etc.).

Why this lives in Flight Deck and not in the agent
--------------------------------------------------
The captain-claw agent is request/response: it only produces a reply when
something injects a prompt. It has its own cron, but a cron job firing
*inside* the agent emits a generic ``chat_message`` with no idea which
WhatsApp number it should reach — addressing (channel ↔ WAID) and agent
binding both live in Flight Deck. So FD owns proactive push.

Execution model
---------------
Each due job:

  1. Resolves its target agent by **slug** against FD's process registry
     (survives port reassignment).
  2. Opens an **ephemeral channel** ``sched:<token>`` and binds the agent
     to it — reusing the exact pump/binding machinery the glasses + WhatsApp
     bridges use for inbound messages.
  3. Injects the job's prompt as a ``chat`` message and captures the first
     ``agent`` reply that comes back over the channel bus.
  4. Delivers that reply: to a WhatsApp WAID (via the bridge's mute-aware
     push) or broadcast onto a channel (glasses HUD + any bound bridge).
  5. Tears the ephemeral channel down.

Because the channel is unique per run, "the next agent message on this
channel" is unambiguously the reply to our prompt — no races with other
traffic.

Schedules
---------
``every <N>{m|h|d}``  · ``daily HH:MM``  · ``weekly <day> HH:MM``
``in <N>{m|h|d}`` (one-shot)  · ``once <ISO-8601>`` (one-shot)

Quiet hours
-----------
``FD_SCHEDULER_QUIET_HOURS=22-08`` suppresses delivery during that window
(wraps midnight). A job with ``ignore_quiet_hours`` runs anyway. Jobs
skipped for quiet hours are retried when the window ends rather than
dropped — so a one-shot scheduled at 2 a.m. fires at 8 a.m.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from captain_claw.flight_deck.glasses_bridge import (
    _broadcast,
    _check_token,
    _ensure_agent_binding,
    _get_or_create_channel,
    _remove_channel,
)

_log = logging.getLogger(__name__)

router = APIRouter()

_NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

# How long to wait for the agent's reply to a scheduled prompt.
_REPLY_TIMEOUT_SECONDS = 150.0

_WEEKDAYS = {
    "mon": 0, "monday": 0,
    "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "weds": 2, "wednesday": 2,
    "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}

_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400}


# ── Schedule parsing ──────────────────────────────────────────────────


class ScheduleError(ValueError):
    """Raised when a schedule string can't be parsed."""


def is_one_shot(schedule: str) -> bool:
    s = (schedule or "").strip().lower()
    return s.startswith("in ") or s.startswith("once ")


def validate_schedule(schedule: str) -> None:
    """Raise ScheduleError if the schedule string is malformed.

    Validates by attempting to compute a next run from a fixed base; any
    parse failure surfaces as ScheduleError with a human message.
    """
    compute_next_run(schedule, base=datetime(2026, 1, 1, 12, 0, 0))


def compute_next_run(schedule: str, base: datetime | None = None) -> float | None:
    """Return the next run time as an epoch float (local), or None when the
    schedule has no future occurrence (consumed one-shots).

    ``base`` defaults to now; pass an explicit value for deterministic
    tests. All wall-clock math is in local time (``datetime`` naive).
    """
    s = (schedule or "").strip()
    if not s:
        raise ScheduleError("empty schedule")
    now = base or datetime.now()
    low = s.lower()

    # every <N><unit>
    m = re.fullmatch(r"every\s+(\d+)\s*([mhd])", low)
    if m:
        secs = int(m.group(1)) * _UNIT_SECONDS[m.group(2)]
        if secs <= 0:
            raise ScheduleError("interval must be > 0")
        return (now + timedelta(seconds=secs)).timestamp()

    # in <N><unit>  (one-shot)
    m = re.fullmatch(r"in\s+(\d+)\s*([mhd])", low)
    if m:
        secs = int(m.group(1)) * _UNIT_SECONDS[m.group(2)]
        return (now + timedelta(seconds=secs)).timestamp()

    # daily HH:MM
    m = re.fullmatch(r"daily\s+(\d{1,2}):(\d{2})", low)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        _check_hm(hh, mm)
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target.timestamp()

    # weekly <day> HH:MM
    m = re.fullmatch(r"weekly\s+([a-z]+)\s+(\d{1,2}):(\d{2})", low)
    if m:
        day = _WEEKDAYS.get(m.group(1))
        if day is None:
            raise ScheduleError(f"unknown weekday: {m.group(1)}")
        hh, mm = int(m.group(2)), int(m.group(3))
        _check_hm(hh, mm)
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        days_ahead = (day - now.weekday()) % 7
        target += timedelta(days=days_ahead)
        if target <= now:
            target += timedelta(days=7)
        return target.timestamp()

    # once <ISO-8601>  (one-shot)
    m = re.fullmatch(r"once\s+(.+)", s, re.I)
    if m:
        raw = m.group(1).strip()
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ScheduleError(f"bad ISO datetime: {raw}") from exc
        # tz-aware → epoch directly; naive → assume local.
        return dt.timestamp()

    raise ScheduleError(
        f"unrecognized schedule: {schedule!r} "
        "(use: every <N>m|h|d, daily HH:MM, weekly <day> HH:MM, "
        "in <N>m|h|d, once <ISO>)"
    )


def _check_hm(hh: int, mm: int) -> None:
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ScheduleError(f"bad time {hh:02d}:{mm:02d}")


# ── Quiet hours ───────────────────────────────────────────────────────


def _quiet_window() -> tuple[int, int] | None:
    spec = os.environ.get("FD_SCHEDULER_QUIET_HOURS", "").strip()
    if not spec or "-" not in spec:
        return None
    try:
        start_s, end_s = spec.split("-", 1)
        start, end = int(start_s), int(end_s)
    except ValueError:
        return None
    if not (0 <= start <= 23 and 0 <= end <= 23) or start == end:
        return None
    return start, end


def _in_quiet_hours(now: datetime | None = None) -> bool:
    win = _quiet_window()
    if not win:
        return False
    start, end = win
    h = (now or datetime.now()).hour
    if start < end:
        return start <= h < end
    return h >= start or h < end  # wraps midnight


def _quiet_hours_end_epoch(now: datetime | None = None) -> float:
    """Epoch of the next time quiet hours end. Used to retry skipped jobs."""
    win = _quiet_window()
    now = now or datetime.now()
    if not win:
        return now.timestamp()
    _, end = win
    boundary = now.replace(hour=end, minute=0, second=0, microsecond=0)
    if boundary <= now:
        boundary += timedelta(days=1)
    return boundary.timestamp()


# ── SQLite store ──────────────────────────────────────────────────────


def _db_path() -> Path:
    # Mirror server.py's DATA_DIR resolution without importing it at module
    # load (avoids an import cycle: server imports this router).
    base = os.environ.get("FD_DATA_DIR", "").strip()
    if base:
        return Path(base).expanduser().resolve() / "scheduler.db"
    return Path("~/.captain-claw/scheduler.db").expanduser()


def _utcnow_iso() -> str:
    from datetime import timezone
    return datetime.now(timezone.utc).isoformat()


def _new_job_id() -> str:
    return "job_" + secrets.token_hex(5)


_VALID_DELIVERY = {"whatsapp", "channel"}


class SchedulerStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = (db_path or _db_path())
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._ensure_db()

    def _conn_or_open(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._conn = conn
        return self._conn

    def _ensure_db(self) -> None:
        with self._lock:
            self._conn_or_open().executescript(
                """
                CREATE TABLE IF NOT EXISTS scheduler_jobs (
                    id                 TEXT PRIMARY KEY,
                    name               TEXT NOT NULL DEFAULT '',
                    schedule           TEXT NOT NULL,
                    agent_slug         TEXT NOT NULL DEFAULT '',
                    agent_auth         TEXT NOT NULL DEFAULT '',
                    prompt             TEXT NOT NULL,
                    delivery_kind      TEXT NOT NULL,
                    delivery_target    TEXT NOT NULL,
                    enabled            INTEGER NOT NULL DEFAULT 1,
                    ignore_quiet_hours INTEGER NOT NULL DEFAULT 0,
                    created_at         TEXT NOT NULL,
                    updated_at         TEXT NOT NULL,
                    next_run_at        REAL,
                    last_run_at        REAL,
                    last_status        TEXT NOT NULL DEFAULT '',
                    last_result        TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_due
                    ON scheduler_jobs(enabled, next_run_at);
                """
            )
            self._conn_or_open().commit()

    def create(self, **fields: Any) -> dict[str, Any]:
        schedule = str(fields.get("schedule", "")).strip()
        validate_schedule(schedule)  # raises ScheduleError
        kind = str(fields.get("delivery_kind", "")).strip().lower()
        if kind not in _VALID_DELIVERY:
            raise ValueError(f"delivery_kind must be one of {sorted(_VALID_DELIVERY)}")
        target = str(fields.get("delivery_target", "")).strip()
        if not target:
            raise ValueError("delivery_target required")
        prompt = str(fields.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("prompt required")

        jid = _new_job_id()
        now_iso = _utcnow_iso()
        enabled = 1 if fields.get("enabled", True) else 0
        next_run = compute_next_run(schedule) if enabled else None
        row = {
            "id": jid,
            "name": str(fields.get("name", "")).strip(),
            "schedule": schedule,
            "agent_slug": str(fields.get("agent_slug", "")).strip(),
            "agent_auth": str(fields.get("agent_auth", "")).strip(),
            "prompt": prompt,
            "delivery_kind": kind,
            "delivery_target": target.lstrip("+") if kind == "whatsapp" else target,
            "enabled": enabled,
            "ignore_quiet_hours": 1 if fields.get("ignore_quiet_hours", False) else 0,
            "created_at": now_iso,
            "updated_at": now_iso,
            "next_run_at": next_run,
            "last_run_at": None,
            "last_status": "",
            "last_result": "",
        }
        with self._lock:
            conn = self._conn_or_open()
            conn.execute(
                """
                INSERT INTO scheduler_jobs
                  (id, name, schedule, agent_slug, agent_auth, prompt,
                   delivery_kind, delivery_target, enabled, ignore_quiet_hours,
                   created_at, updated_at, next_run_at, last_run_at,
                   last_status, last_result)
                VALUES
                  (:id, :name, :schedule, :agent_slug, :agent_auth, :prompt,
                   :delivery_kind, :delivery_target, :enabled, :ignore_quiet_hours,
                   :created_at, :updated_at, :next_run_at, :last_run_at,
                   :last_status, :last_result)
                """,
                row,
            )
            conn.commit()
        return row

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            r = self._conn_or_open().execute(
                "SELECT * FROM scheduler_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            return dict(r) if r else None

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn_or_open().execute(
                "SELECT * FROM scheduler_jobs ORDER BY created_at"
            ).fetchall()
            return [dict(r) for r in rows]

    def list_due(self, now_epoch: float) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn_or_open().execute(
                """
                SELECT * FROM scheduler_jobs
                WHERE enabled = 1 AND next_run_at IS NOT NULL AND next_run_at <= ?
                ORDER BY next_run_at
                """,
                (now_epoch,),
            ).fetchall()
            return [dict(r) for r in rows]

    def update(self, job_id: str, **fields: Any) -> dict[str, Any] | None:
        existing = self.get(job_id)
        if not existing:
            return None
        allowed = {
            "name", "schedule", "agent_slug", "agent_auth", "prompt",
            "delivery_kind", "delivery_target", "enabled", "ignore_quiet_hours",
        }
        updates: dict[str, Any] = {}
        for k, v in fields.items():
            if k not in allowed:
                continue
            if k == "schedule":
                validate_schedule(str(v).strip())
                updates[k] = str(v).strip()
            elif k == "delivery_kind":
                kind = str(v).strip().lower()
                if kind not in _VALID_DELIVERY:
                    raise ValueError(f"delivery_kind must be one of {sorted(_VALID_DELIVERY)}")
                updates[k] = kind
            elif k in ("enabled", "ignore_quiet_hours"):
                updates[k] = 1 if v else 0
            else:
                updates[k] = str(v).strip()
        if not updates:
            return existing
        updates["updated_at"] = _utcnow_iso()
        # Recompute next_run if schedule changed or job (re)enabled.
        recompute = "schedule" in updates or updates.get("enabled") == 1
        if recompute:
            sched = updates.get("schedule", existing["schedule"])
            enabled = updates.get("enabled", existing["enabled"])
            updates["next_run_at"] = compute_next_run(sched) if enabled else None
        if updates.get("enabled") == 0:
            updates["next_run_at"] = None

        sets = ", ".join(f"{k} = :{k}" for k in updates)
        updates["id"] = job_id
        with self._lock:
            conn = self._conn_or_open()
            conn.execute(f"UPDATE scheduler_jobs SET {sets} WHERE id = :id", updates)
            conn.commit()
        return self.get(job_id)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            conn = self._conn_or_open()
            cur = conn.execute("DELETE FROM scheduler_jobs WHERE id = ?", (job_id,))
            conn.commit()
            return cur.rowcount > 0

    def mark_run(
        self,
        job_id: str,
        *,
        status: str,
        result: str,
        last_run_at: float,
        next_run_at: float | None,
        enabled: int | None = None,
    ) -> None:
        with self._lock:
            conn = self._conn_or_open()
            if enabled is None:
                conn.execute(
                    """
                    UPDATE scheduler_jobs
                    SET last_status = ?, last_result = ?, last_run_at = ?,
                        next_run_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (status, result[:2000], last_run_at, next_run_at,
                     _utcnow_iso(), job_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE scheduler_jobs
                    SET last_status = ?, last_result = ?, last_run_at = ?,
                        next_run_at = ?, enabled = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (status, result[:2000], last_run_at, next_run_at,
                     enabled, _utcnow_iso(), job_id),
                )
            conn.commit()


_STORE: SchedulerStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> SchedulerStore:
    global _STORE
    if _STORE is not None:
        return _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = SchedulerStore()
        return _STORE


# ── Agent resolution + prompt execution ───────────────────────────────


def resolve_agent_by_slug(slug: str, auth_override: str = "") -> tuple[str, int, str]:
    """Resolve a process slug → (host, port, auth) from FD's registry.

    ``auth_override`` (the agent's config.yaml web.auth_token) wins over the
    registry's ``web_auth`` — needed for agents whose registry entry has no
    auth recorded. Returns port 0 when the slug isn't a running agent.
    """
    try:
        from captain_claw.flight_deck.server import (
            _load_process_registry,
            _process_is_alive,
        )
        registry = _load_process_registry()
    except Exception:
        return "localhost", 0, auth_override
    entry = registry.get(slug)
    if not entry or not _process_is_alive(slug):
        return "localhost", 0, auth_override
    try:
        port = int(entry.get("web_port", 0) or 0)
    except (TypeError, ValueError):
        port = 0
    auth = auth_override or str(entry.get("web_auth", "") or "")
    return "localhost", port, auth


async def run_prompt_and_capture(
    *,
    host: str,
    port: int,
    auth: str,
    prompt: str,
    timeout: float = _REPLY_TIMEOUT_SECONDS,
) -> str | None:
    """Inject ``prompt`` into a fresh ephemeral channel bound to the agent
    and return the first ``agent`` reply text, or None on timeout/failure.

    The ephemeral channel is torn down in all cases.
    """
    run_channel = "sched:" + secrets.token_hex(6)
    ch = await _get_or_create_channel(run_channel)
    loop = asyncio.get_event_loop()
    fut: asyncio.Future[str] = loop.create_future()

    async def _capture(payload: dict) -> None:
        if payload.get("type") == "agent" and payload.get("text") and not fut.done():
            fut.set_result(str(payload["text"]))

    ch.callback_subscribers.append(_capture)
    try:
        await _ensure_agent_binding(ch, host, port, auth)
        # Wait for the pump's WS to come up (first send after binding).
        for _ in range(50):  # ~5 s
            if ch.agent_ws is not None:
                break
            await asyncio.sleep(0.1)
        if ch.agent_ws is None:
            return None
        async with ch.send_lock:
            await ch.agent_ws.send(json.dumps({"type": "chat", "content": prompt}))
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            return None
    except Exception as exc:
        _log.warning("run_prompt_and_capture failed: %s", exc)
        return None
    finally:
        await _remove_channel(run_channel)


async def _deliver(kind: str, target: str, text: str) -> tuple[bool, str]:
    """Deliver ``text`` per the job's delivery config.

    Returns ``(delivered, status_note)``. ``delivered=False`` with a note
    like "muted" means the message was intentionally suppressed.
    """
    if kind == "whatsapp":
        from captain_claw.flight_deck.whatsapp_bridge import push_to_waid
        sent = await push_to_waid(target, text)
        return (sent, "ok" if sent else "skipped:muted-or-not-allowed")
    if kind == "channel":
        ch = await _get_or_create_channel(target)
        from datetime import timezone
        await _broadcast(ch, {
            "type": "agent",
            "text": text,
            "source": "scheduler",
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        return (True, "ok")
    return (False, f"error:unknown delivery_kind {kind}")


async def execute_job(job: dict[str, Any], *, force: bool = False) -> tuple[str, str]:
    """Run one job end-to-end. Returns ``(status, result_text)``.

    ``force=True`` (manual run-now) bypasses quiet hours.
    """
    if not force and not job.get("ignore_quiet_hours") and _in_quiet_hours():
        return ("skipped:quiet", "")

    slug = str(job.get("agent_slug") or "")
    host, port, auth = resolve_agent_by_slug(slug, str(job.get("agent_auth") or ""))
    if not port:
        return ("error:agent-not-running", f"agent slug '{slug}' not running")

    reply = await run_prompt_and_capture(
        host=host, port=port, auth=auth, prompt=str(job.get("prompt") or ""),
    )
    if reply is None:
        return ("error:no-reply", "agent produced no reply within timeout")

    delivered, note = await _deliver(
        str(job.get("delivery_kind") or ""),
        str(job.get("delivery_target") or ""),
        reply,
    )
    status = note if delivered else note
    return (status, reply)


# ── Poll loop ─────────────────────────────────────────────────────────


async def _run_due_job(store: SchedulerStore, job: dict[str, Any]) -> None:
    now_epoch = time.time()
    status, result = await execute_job(job)

    if status == "skipped:quiet":
        # Retry when quiet hours end; don't consume a one-shot.
        store.mark_run(
            job["id"], status=status, result="",
            last_run_at=now_epoch, next_run_at=_quiet_hours_end_epoch(),
        )
        return

    # Consumed. One-shots disable; recurring recompute.
    if is_one_shot(job["schedule"]):
        store.mark_run(
            job["id"], status=status, result=result,
            last_run_at=now_epoch, next_run_at=None, enabled=0,
        )
    else:
        try:
            nxt = compute_next_run(job["schedule"])
        except ScheduleError:
            nxt = None
        store.mark_run(
            job["id"], status=status, result=result,
            last_run_at=now_epoch, next_run_at=nxt,
        )


async def scheduler_loop(stop_event: asyncio.Event) -> None:
    """Background poll loop. Started from FD's lifespan."""
    store = get_store()
    try:
        poll = float(os.environ.get("FD_SCHEDULER_POLL_SECONDS", "30") or 30)
    except ValueError:
        poll = 30.0
    poll = max(5.0, poll)
    _log.info("FD scheduler loop started (poll=%.0fs)", poll)
    while not stop_event.is_set():
        try:
            due = store.list_due(time.time())
            for job in due:
                try:
                    await _run_due_job(store, job)
                except Exception as exc:
                    _log.warning("scheduler job %s failed: %s", job.get("id"), exc)
                    store.mark_run(
                        job["id"], status=f"error:{exc}", result="",
                        last_run_at=time.time(),
                        next_run_at=(
                            None if is_one_shot(job["schedule"])
                            else compute_next_run(job["schedule"])
                        ),
                        enabled=(0 if is_one_shot(job["schedule"]) else None),
                    )
        except Exception as exc:
            _log.warning("scheduler loop iteration error: %s", exc)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll)
        except asyncio.TimeoutError:
            pass
    _log.info("FD scheduler loop stopped")


# ── REST API ──────────────────────────────────────────────────────────


@router.get("/scheduler/jobs")
async def list_jobs(request: Request) -> JSONResponse:
    _check_token(request)
    return JSONResponse(get_store().list(), headers=_NO_CACHE)


@router.post("/scheduler/jobs")
async def create_job(request: Request) -> JSONResponse:
    _check_token(request)
    body = await request.json()
    try:
        row = get_store().create(**body)
    except (ScheduleError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(row, headers=_NO_CACHE)


@router.get("/scheduler/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> JSONResponse:
    _check_token(request)
    row = get_store().get(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse(row, headers=_NO_CACHE)


@router.patch("/scheduler/jobs/{job_id}")
async def update_job(job_id: str, request: Request) -> JSONResponse:
    _check_token(request)
    body = await request.json()
    try:
        row = get_store().update(job_id, **body)
    except (ScheduleError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not row:
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse(row, headers=_NO_CACHE)


@router.delete("/scheduler/jobs/{job_id}")
async def delete_job(job_id: str, request: Request) -> JSONResponse:
    _check_token(request)
    if not get_store().delete(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse({"ok": True}, headers=_NO_CACHE)


@router.post("/scheduler/jobs/{job_id}/run")
async def run_job_now(job_id: str, request: Request) -> JSONResponse:
    """Manually fire a job immediately, bypassing schedule + quiet hours.

    Does NOT change the job's next_run_at — it's an out-of-band test fire.
    """
    _check_token(request)
    job = get_store().get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    status, result = await execute_job(job, force=True)
    get_store().mark_run(
        job_id, status=status, result=result,
        last_run_at=time.time(), next_run_at=job.get("next_run_at"),
    )
    return JSONResponse(
        {"ok": status.startswith("ok"), "status": status,
         "result_preview": result[:500]},
        headers=_NO_CACHE,
    )
