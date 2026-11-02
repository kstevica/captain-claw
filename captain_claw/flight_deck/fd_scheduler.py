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
``every <N>{m|h|d}``  · ``daily HH:MM``  · ``weekdays HH:MM`` (Mon–Fri)
``weekly <day>[,<day>…] HH:MM``  · ``in <N>{m|h|d}`` (one-shot)
``once <ISO-8601>`` (one-shot)

Wall-clock times are the job's own time zone (``tz``, an IANA name the
browser reports when the job is made); a job without one uses the deck's
local time, as before.

Tasks (the kiosk's Scheduled tasks)
-----------------------------------
A job with ``delivery_kind="agent"`` delivers per its ``delivery`` JSON:
``chat`` (the result lands in the agent's own chat — the kiosk timeline and
the chat bridges bound to it), ``email`` (from the owner's connected Gmail or
Outlook, to them or to ``email_to``) and ``telegram`` (the owner's chat with
the deck's shared bot). Every prompt job runs in its own agent session (one
per job, kept across runs — its runs read as one conversation, and the agent
sees what it said last time), and every run is logged in ``scheduler_runs``
with its result and how each delivery went — the history the kiosk shows and
searches.

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
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from captain_claw.flight_deck import agent_idle
from captain_claw.flight_deck.glasses_bridge import (
    _broadcast,
    _check_token,
    _ensure_agent_binding,
    _get_or_create_channel,
    _remove_channel,
    await_agent_link,
)

_log = logging.getLogger(__name__)

router = APIRouter()

_NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

# How long to wait for the agent's reply to a scheduled prompt — a task's
# (delivery_kind "agent") may do real work first (mail, research, files).
_REPLY_TIMEOUT_SECONDS = 150.0
_TASK_REPLY_TIMEOUT_SECONDS = 1200.0

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


def validate_schedule(schedule: str, tz: str = "") -> None:
    """Raise ScheduleError if the schedule string (or its time zone) is
    malformed.

    Validates by attempting to compute a next run from a fixed base; any
    parse failure surfaces as ScheduleError with a human message.
    """
    compute_next_run(schedule, base=datetime(2026, 1, 1, 12, 0, 0), tz=tz)


def _zone(tz: str) -> ZoneInfo | None:
    """The IANA zone ``tz`` names, or None for "" (the deck's local time)."""
    name = (tz or "").strip()
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ScheduleError(f"unknown time zone: {name}") from exc


def _parse_days(text: str) -> list[int]:
    """``mon,wed fri`` → [0, 2, 4] (order kept, duplicates dropped)."""
    days: list[int] = []
    for part in re.split(r"[,\s]+", text.strip()):
        if not part:
            continue
        day = _WEEKDAYS.get(part)
        if day is None:
            raise ScheduleError(f"unknown weekday: {part}")
        if day not in days:
            days.append(day)
    if not days:
        raise ScheduleError("no weekday given")
    return days


def _next_on_days(now: datetime, days: list[int], hh: int, mm: int) -> float:
    """The first HH:MM after ``now`` on one of ``days`` (Mon=0)."""
    _check_hm(hh, mm)
    today = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    for ahead in range(8):
        target = today + timedelta(days=ahead)
        if target.weekday() in days and target > now:
            return target.timestamp()
    raise ScheduleError("no next run")  # unreachable: 8 days cover every weekday


def compute_next_run(schedule: str, base: datetime | None = None, tz: str = "") -> float | None:
    """Return the next run time as an epoch float, or None when the
    schedule has no future occurrence (consumed one-shots).

    ``base`` defaults to now; pass an explicit value for deterministic
    tests. Wall-clock math is in ``tz`` (an IANA zone) when given, else in
    local time (``datetime`` naive); a naive ``base`` is read in ``tz``.
    """
    s = (schedule or "").strip()
    if not s:
        raise ScheduleError("empty schedule")
    zone = _zone(tz)
    if zone is None:
        now = base or datetime.now()
    elif base is None:
        now = datetime.now(zone)
    else:
        now = base.astimezone(zone) if base.tzinfo else base.replace(tzinfo=zone)
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

    # weekdays HH:MM  (Monday to Friday)
    m = re.fullmatch(r"weekdays\s+(\d{1,2}):(\d{2})", low)
    if m:
        return _next_on_days(now, [0, 1, 2, 3, 4], int(m.group(1)), int(m.group(2)))

    # weekly <day>[,<day>…] HH:MM
    m = re.fullmatch(r"weekly\s+([a-z][a-z,\s]*?)\s+(\d{1,2}):(\d{2})", low)
    if m:
        return _next_on_days(now, _parse_days(m.group(1)), int(m.group(2)), int(m.group(3)))

    # once <ISO-8601>  (one-shot)
    m = re.fullmatch(r"once\s+(.+)", s, re.I)
    if m:
        raw = m.group(1).strip()
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ScheduleError(f"bad ISO datetime: {raw}") from exc
        # tz-aware → epoch directly; naive → the job's zone, else local.
        if dt.tzinfo is None and zone is not None:
            dt = dt.replace(tzinfo=zone)
        return dt.timestamp()

    raise ScheduleError(
        f"unrecognized schedule: {schedule!r} "
        "(use: every <N>m|h|d, daily HH:MM, weekdays HH:MM, "
        "weekly <day>[,<day>] HH:MM, in <N>m|h|d, once <ISO>)"
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


_VALID_DELIVERY = {"whatsapp", "channel", "telegram", "agent"}

# Runs kept per job (the oldest go first) and the result text kept per run.
_RUNS_KEPT_PER_JOB = 200
_RUN_RESULT_MAX = 60_000
_EMAIL_TO_MAX = 10
_EMAIL_RE = re.compile(r"[^@\s,;<>\"']+@[^@\s,;<>\"']+\.[^@\s,;<>\"']+")


def _clean_addresses(raw: Any) -> list[str]:
    """``"a@x.com, b@y.org"`` (or a list) → valid, de-duplicated addresses.
    Raises ValueError for one that isn't an address or for too many."""
    items = raw if isinstance(raw, (list, tuple)) else re.split(r"[,;\s]+", str(raw or ""))
    out: list[str] = []
    for item in items:
        addr = str(item or "").strip()
        if not addr:
            continue
        if not _EMAIL_RE.fullmatch(addr):
            raise ValueError(f"not an email address: {addr}")
        if addr.lower() not in {a.lower() for a in out}:
            out.append(addr)
    if len(out) > _EMAIL_TO_MAX:
        raise ValueError(f"at most {_EMAIL_TO_MAX} email recipients")
    return out


def norm_delivery(raw: Any) -> dict[str, Any]:
    """A task's delivery as stored: ``{chat, email, telegram, email_to}``.
    ``email_to`` "" means the owner's own address."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except ValueError:
            raw = {}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "chat": bool(raw.get("chat")),
        "email": bool(raw.get("email")),
        "telegram": bool(raw.get("telegram")),
        "email_to": ", ".join(_clean_addresses(raw.get("email_to"))),
    }


def _check_agent_delivery(delivery: dict[str, Any]) -> None:
    if not (delivery["chat"] or delivery["email"] or delivery["telegram"]):
        raise ValueError("pick at least one delivery: chat, email or Telegram")


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
            # A job runs either a `prompt` (text → agent) or a `flow_id`
            # (run a Flow). Migrate existing DBs in place.
            try:
                self._conn_or_open().execute(
                    "ALTER TABLE scheduler_jobs ADD COLUMN flow_id TEXT NOT NULL DEFAULT ''"
                )
            except Exception:
                pass  # column already exists
            # The FD user a job belongs to ("" = system / pre-owner legacy row,
            # which only admins see through the REST API).
            try:
                self._conn_or_open().execute(
                    "ALTER TABLE scheduler_jobs ADD COLUMN owner_id TEXT NOT NULL DEFAULT ''"
                )
            except Exception:
                pass  # column already exists
            # Who wrote the job's prompt: "human" (the SchedulerPage / a user
            # token) or "agent" (an internal caller such as intentions.py). Only a
            # human-written prompt can authorize the turn to write email. Legacy
            # rows count as human-written.
            try:
                self._conn_or_open().execute(
                    "ALTER TABLE scheduler_jobs ADD COLUMN prompt_author TEXT NOT NULL DEFAULT 'human'"
                )
            except Exception:
                pass  # column already exists
            # Tasks: where an "agent" job delivers (JSON, see norm_delivery),
            # the IANA zone its wall-clock times are in ("" = deck local), and
            # the agent session its runs live in (learned from the first run).
            for ddl in (
                "ALTER TABLE scheduler_jobs ADD COLUMN delivery TEXT NOT NULL DEFAULT ''",
                "ALTER TABLE scheduler_jobs ADD COLUMN tz TEXT NOT NULL DEFAULT ''",
                "ALTER TABLE scheduler_jobs ADD COLUMN agent_session_id TEXT NOT NULL DEFAULT ''",
            ):
                try:
                    self._conn_or_open().execute(ddl)
                except Exception:
                    pass  # column already exists
            # Every run: what it said and how each delivery went.
            self._conn_or_open().executescript(
                """
                CREATE TABLE IF NOT EXISTS scheduler_runs (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id      TEXT NOT NULL,
                    owner_id    TEXT NOT NULL DEFAULT '',
                    job_name    TEXT NOT NULL DEFAULT '',
                    agent_slug  TEXT NOT NULL DEFAULT '',
                    trigger     TEXT NOT NULL DEFAULT 'scheduled',
                    started_at  REAL NOT NULL,
                    finished_at REAL,
                    status      TEXT NOT NULL DEFAULT 'running',
                    result      TEXT NOT NULL DEFAULT '',
                    deliveries  TEXT NOT NULL DEFAULT '[]',
                    session_id  TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_runs_job ON scheduler_runs(job_id, id);
                CREATE INDEX IF NOT EXISTS idx_runs_owner ON scheduler_runs(owner_id, id);
                """
            )
            # A run this process didn't finish was cut off by a restart.
            self._conn_or_open().execute(
                "UPDATE scheduler_runs SET status = 'error:interrupted', finished_at = started_at "
                "WHERE status = 'running'"
            )
            self._conn_or_open().commit()

    def create(
        self, *, owner_id: str = "", prompt_author: str = "human", **fields: Any,
    ) -> dict[str, Any]:
        schedule = str(fields.get("schedule", "")).strip()
        tz = str(fields.get("tz", "") or "").strip()
        validate_schedule(schedule, tz)  # raises ScheduleError
        kind = str(fields.get("delivery_kind", "")).strip().lower()
        if kind not in _VALID_DELIVERY:
            raise ValueError(f"delivery_kind must be one of {sorted(_VALID_DELIVERY)}")
        target = str(fields.get("delivery_target", "")).strip()
        delivery = ""
        if kind == "agent":
            d = norm_delivery(fields.get("delivery"))
            _check_agent_delivery(d)
            delivery = json.dumps(d)
            target = ""
        elif not target:
            raise ValueError("delivery_target required")
        prompt = str(fields.get("prompt", "")).strip()
        flow_id = str(fields.get("flow_id", "")).strip()
        if not prompt and not flow_id:
            raise ValueError("prompt or flow_id required")

        jid = _new_job_id()
        now_iso = _utcnow_iso()
        enabled = 1 if fields.get("enabled", True) else 0
        next_run = compute_next_run(schedule, tz=tz) if enabled else None
        if kind == "agent" and next_run is not None and is_one_shot(schedule) \
                and next_run < time.time() - 60:
            raise ScheduleError("that time has already passed")
        row = {
            "id": jid,
            "owner_id": str(owner_id or "").strip(),
            "name": str(fields.get("name", "")).strip(),
            "schedule": schedule,
            "agent_slug": str(fields.get("agent_slug", "")).strip(),
            "agent_auth": str(fields.get("agent_auth", "")).strip(),
            "prompt": prompt,
            "prompt_author": _norm_prompt_author(prompt_author),
            "flow_id": flow_id,
            "delivery_kind": kind,
            "delivery_target": target.lstrip("+") if kind == "whatsapp" else target,
            "delivery": delivery,
            "tz": tz,
            "agent_session_id": "",
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
                  (id, owner_id, name, schedule, agent_slug, agent_auth, prompt, prompt_author,
                   flow_id, delivery_kind, delivery_target, delivery, tz, agent_session_id,
                   enabled, ignore_quiet_hours,
                   created_at, updated_at, next_run_at, last_run_at,
                   last_status, last_result)
                VALUES
                  (:id, :owner_id, :name, :schedule, :agent_slug, :agent_auth, :prompt,
                   :prompt_author, :flow_id,
                   :delivery_kind, :delivery_target, :delivery, :tz, :agent_session_id,
                   :enabled, :ignore_quiet_hours,
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

    def list(self, owner_id: str | None = None) -> list[dict[str, Any]]:
        """Every job, or only ``owner_id``'s when given."""
        with self._lock:
            if owner_id is None:
                rows = self._conn_or_open().execute(
                    "SELECT * FROM scheduler_jobs ORDER BY created_at"
                ).fetchall()
            else:
                rows = self._conn_or_open().execute(
                    "SELECT * FROM scheduler_jobs WHERE owner_id = ? ORDER BY created_at",
                    (owner_id,),
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

    def update(
        self, job_id: str, *, prompt_author: str = "human", **fields: Any,
    ) -> dict[str, Any] | None:
        """Patch a job. A new ``prompt`` also records who wrote it
        (``prompt_author``) — the caller decides that, never the body."""
        existing = self.get(job_id)
        if not existing:
            return None
        allowed = {
            "name", "schedule", "agent_slug", "agent_auth", "prompt", "flow_id",
            "delivery_kind", "delivery_target", "delivery", "tz", "enabled",
            "ignore_quiet_hours",
        }
        updates: dict[str, Any] = {}
        for k, v in fields.items():
            if k not in allowed:
                continue
            if k == "schedule":
                updates[k] = str(v).strip()
            elif k == "delivery_kind":
                kind = str(v).strip().lower()
                if kind not in _VALID_DELIVERY:
                    raise ValueError(f"delivery_kind must be one of {sorted(_VALID_DELIVERY)}")
                updates[k] = kind
            elif k == "delivery":
                updates[k] = json.dumps(norm_delivery(v))
            elif k in ("enabled", "ignore_quiet_hours"):
                updates[k] = 1 if v else 0
            else:
                updates[k] = str(v).strip()
        if not updates:
            return existing
        if "schedule" in updates or "tz" in updates:
            validate_schedule(updates.get("schedule", existing["schedule"]),
                              updates.get("tz", existing.get("tz") or ""))
        if updates.get("delivery_kind", existing.get("delivery_kind")) == "agent":
            _check_agent_delivery(norm_delivery(updates.get("delivery", existing.get("delivery"))))
            updates["delivery_target"] = ""
        if "prompt" in updates:
            updates["prompt_author"] = _norm_prompt_author(prompt_author)
        updates["updated_at"] = _utcnow_iso()
        # Recompute next_run if the schedule (or its zone) changed or the job
        # was (re)enabled.
        recompute = "schedule" in updates or "tz" in updates or updates.get("enabled") == 1
        if recompute:
            sched = updates.get("schedule", existing["schedule"])
            enabled = updates.get("enabled", existing["enabled"])
            updates["next_run_at"] = (
                compute_next_run(sched, tz=updates.get("tz", existing.get("tz") or ""))
                if enabled else None
            )
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
            conn.execute("DELETE FROM scheduler_runs WHERE job_id = ?", (job_id,))
            conn.commit()
            return cur.rowcount > 0

    def delete_for_agent(self, agent_slug: str) -> int:
        """Drop every job (and its runs) that prompts one agent — for an agent
        removed for good, so a later one of the same name inherits none."""
        if not agent_slug:
            return 0
        with self._lock:
            conn = self._conn_or_open()
            conn.execute(
                "DELETE FROM scheduler_runs WHERE job_id IN "
                "(SELECT id FROM scheduler_jobs WHERE agent_slug = ?)", (agent_slug,))
            cur = conn.execute("DELETE FROM scheduler_jobs WHERE agent_slug = ?", (agent_slug,))
            conn.commit()
            return int(cur.rowcount or 0)

    def claim(self, job_id: str, *, next_run_at: float | None, enabled: int | None = None) -> None:
        """Move a due job on before it runs, so the next poll doesn't fire it
        again while this run is still going."""
        with self._lock:
            conn = self._conn_or_open()
            if enabled is None:
                conn.execute("UPDATE scheduler_jobs SET next_run_at = ? WHERE id = ?",
                             (next_run_at, job_id))
            else:
                conn.execute("UPDATE scheduler_jobs SET next_run_at = ?, enabled = ? WHERE id = ?",
                             (next_run_at, enabled, job_id))
            conn.commit()

    def record_result(self, job_id: str, *, status: str, result: str, last_run_at: float) -> None:
        """A run's outcome on its job, leaving its next run alone."""
        with self._lock:
            conn = self._conn_or_open()
            conn.execute(
                "UPDATE scheduler_jobs SET last_status = ?, last_result = ?, last_run_at = ?, "
                "updated_at = ? WHERE id = ?",
                (status, result[:2000], last_run_at, _utcnow_iso(), job_id),
            )
            conn.commit()

    def set_session(self, job_id: str, session_id: str) -> None:
        """The agent session a job's runs live in (reported by its first run)."""
        with self._lock:
            conn = self._conn_or_open()
            conn.execute("UPDATE scheduler_jobs SET agent_session_id = ? WHERE id = ?",
                         (session_id[:128], job_id))
            conn.commit()

    # ── run history ──

    def start_run(self, job: dict[str, Any], trigger: str) -> int:
        with self._lock:
            conn = self._conn_or_open()
            cur = conn.execute(
                "INSERT INTO scheduler_runs (job_id, owner_id, job_name, agent_slug, trigger, "
                "started_at) VALUES (?, ?, ?, ?, ?, ?)",
                (str(job.get("id") or ""), str(job.get("owner_id") or ""),
                 str(job.get("name") or ""), str(job.get("agent_slug") or ""),
                 trigger, time.time()),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    def finish_run(
        self, run_id: int, *, status: str, result: str,
        deliveries: list[dict[str, Any]] | None = None, session_id: str = "",
    ) -> None:
        with self._lock:
            conn = self._conn_or_open()
            conn.execute(
                "UPDATE scheduler_runs SET status = ?, result = ?, deliveries = ?, session_id = ?, "
                "finished_at = ? WHERE id = ?",
                (status, (result or "")[:_RUN_RESULT_MAX], json.dumps(deliveries or []),
                 session_id[:128], time.time(), run_id),
            )
            row = conn.execute("SELECT job_id FROM scheduler_runs WHERE id = ?", (run_id,)).fetchone()
            if row:
                conn.execute(
                    "DELETE FROM scheduler_runs WHERE job_id = ? AND id NOT IN ("
                    "SELECT id FROM scheduler_runs WHERE job_id = ? ORDER BY id DESC LIMIT ?)",
                    (row["job_id"], row["job_id"], _RUNS_KEPT_PER_JOB),
                )
            conn.commit()

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        with self._lock:
            r = self._conn_or_open().execute(
                "SELECT * FROM scheduler_runs WHERE id = ?", (run_id,)
            ).fetchone()
            return _run_row(r) if r else None

    def list_runs(
        self, *, owner_id: str | None = None, job_id: str = "", q: str = "",
        limit: int = 50, before_id: int = 0,
    ) -> list[dict[str, Any]]:
        """Newest first. ``owner_id`` None = everyone's (an admin); ``q``
        matches the result text or the task's name, case-insensitively."""
        where, args = ["1 = 1"], []
        if owner_id is not None:
            where.append("owner_id = ?")
            args.append(owner_id)
        if job_id:
            where.append("job_id = ?")
            args.append(job_id)
        if before_id:
            where.append("id < ?")
            args.append(before_id)
        for word in (q or "").split()[:8]:
            like = "%" + word.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            where.append("(result LIKE ? ESCAPE '\\' OR job_name LIKE ? ESCAPE '\\')")
            args.extend([like, like])
        args.append(max(1, min(int(limit or 50), 200)))
        with self._lock:
            rows = self._conn_or_open().execute(
                f"SELECT * FROM scheduler_runs WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT ?",
                args,
            ).fetchall()
            return [_run_row(r) for r in rows]

    def run_stats(self) -> dict[str, dict[str, Any]]:
        """Per job: how many runs are kept and whether one is going now."""
        with self._lock:
            rows = self._conn_or_open().execute(
                "SELECT job_id, COUNT(*) AS n, SUM(status = 'running') AS running "
                "FROM scheduler_runs GROUP BY job_id"
            ).fetchall()
            return {r["job_id"]: {"runs": int(r["n"] or 0), "running": bool(r["running"])} for r in rows}

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


def _run_row(r: sqlite3.Row) -> dict[str, Any]:
    out = dict(r)
    try:
        out["deliveries"] = json.loads(out.get("deliveries") or "[]")
    except ValueError:
        out["deliveries"] = []
    return out


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
    automation: dict[str, Any] | None = None,
    whatsapp_media_to: str = "",
    meta: dict[str, Any] | None = None,
    identity: dict[str, Any] | None = None,
) -> str | None:
    """Inject ``prompt`` into a fresh ephemeral channel bound to the agent
    and return the first ``agent`` reply text, or None on timeout/failure.
    ``meta`` (when given) receives what the reply said about itself: the
    ``task_session`` a scheduled task ran in.

    ``automation`` (the automated-turn marker) rides on the chat frame when
    given, so the agent knows this turn is a scheduled job, not the user.
    ``whatsapp_media_to`` (a job delivering to a WhatsApp number that may
    receive pushes right now) lets the agent send the files and pictures the
    turn makes to that chat itself; the text still goes through the job's
    delivery. ``identity`` (agent_handshake) goes first, so an agent no chat
    has told since it started still knows its job.

    The ephemeral channel is torn down in all cases.
    """
    run_channel = "sched:" + secrets.token_hex(6)
    ch = await _get_or_create_channel(run_channel)
    loop = asyncio.get_event_loop()
    fut: asyncio.Future[str] = loop.create_future()

    async def _capture(payload: dict) -> None:
        # A result mirrored into the main chat (automation_lane) belongs to
        # another automated turn that ended while this one waited for its lane.
        if (payload.get("type") == "agent" and payload.get("text")
                and not payload.get("automation_lane") and not fut.done()):
            if meta is not None and isinstance(payload.get("task_session"), dict):
                meta["task_session"] = payload["task_session"]
            fut.set_result(str(payload["text"]))

    ch.callback_subscribers.append(_capture)
    try:
        await _ensure_agent_binding(ch, host, port, auth)
        # Wait for the pump's WS to come up (first send after binding).
        if not await await_agent_link(ch):
            return None
        async with ch.send_lock:
            if identity:
                await ch.agent_ws.send(json.dumps(identity))
            frame: dict[str, Any] = {"type": "chat", "content": prompt}
            if automation:
                frame["automation"] = automation
            if whatsapp_media_to:
                frame["whatsapp_media_to"] = whatsapp_media_to
            await ch.agent_ws.send(json.dumps(frame))
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            return None
    except Exception as exc:
        _log.warning("run_prompt_and_capture failed: %s", exc)
        return None
    finally:
        await _remove_channel(run_channel)


def _media_target(kind: str, target: str) -> str:
    """The WhatsApp number a job's files and pictures may go to: only one
    a proactive push may reach now (allowlisted, not muted) — the same gate
    the job's text goes through."""
    if kind != "whatsapp" or not target:
        return ""
    from captain_claw.flight_deck.whatsapp_bridge import _allowed_waids, is_push_muted

    waid = target.lstrip("+").strip()
    return waid if waid in _allowed_waids() and not is_push_muted(waid) else ""


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
    if kind == "telegram":
        # target is a comma-separated list of chat_ids (single or multi).
        # Strip markdown to plain text — Telegram rejects unbalanced markdown.
        from captain_claw.flight_deck.meta_webhook_bridge import strip_markdown
        from captain_claw.flight_deck.telegram_out import send_telegram_multi
        chat_ids = [c.strip() for c in str(target).split(",") if c.strip()]
        if not chat_ids:
            return (False, "error:no telegram chat_ids")
        sent, total = await send_telegram_multi(chat_ids, strip_markdown(text))
        if sent == 0:
            return (False, f"error:telegram 0/{total} delivered")
        return (True, f"ok ({sent}/{total})")
    return (False, f"error:unknown delivery_kind {kind}")


async def _execute_flow_job(job: dict[str, Any], flow_id: str) -> tuple[str, str]:
    """Run a scheduled Flow and deliver its output. Scheduled flows should be
    self-contained (no `input` — there's no user to ask at fire time)."""
    try:
        from captain_claw.flight_deck.server import app
        store = getattr(app.state, "flow_store", None)
        runner = getattr(app.state, "flow_runner", None)
    except Exception:
        store = runner = None
    if store is None or runner is None:
        return ("error:flow-engine", "flow engine not available")
    flow = await store.get_flow(flow_id)
    if not flow:
        return ("error:flow-missing", f"flow '{flow_id}' not found")
    # Neutral payload: the scheduler delivers the flow's output itself, so the
    # flow's own auto-delivery (output -> same with no channel) is a no-op.
    payload = {"channel": "scheduler", "scheduled": True, "scheduler_job": job.get("id", "")}
    # Tenancy: an owned job runs the flow as its owner (the runner refuses a
    # non-admin owner's run of another user's flow); an ownerless legacy job
    # keeps acting as the flow's owner.
    owner = str(job.get("owner_id") or "")
    run_ctx: dict[str, Any] = {}
    if owner:
        is_admin = False
        check = getattr(runner, "user_is_admin", None)
        if check is not None:
            try:
                is_admin = bool(await check(owner))
            except Exception:
                is_admin = False
        run_ctx = {"owner_id": owner, "is_admin": is_admin}
    try:
        result = await runner.run(flow, payload, **run_ctx)
    except Exception as exc:
        return ("error:flow-run", str(exc))
    status = str(result.get("status") or "")
    output = str(result.get("output") or "")
    if status != "done":
        return (f"error:flow-{status or 'failed'}", output or status)
    if not output.strip():
        return ("ok:no-output", "")
    delivered, note = await _deliver(
        str(job.get("delivery_kind") or ""),
        str(job.get("delivery_target") or ""),
        output,
    )
    return (note, output)


# What a run tells execute_job's caller beyond (status, text): the agent
# session it ran in and how each delivery went. A contextvar, so execute_job
# keeps its (job, *, force) shape.
_RUN_REPORT: ContextVar[dict[str, Any] | None] = ContextVar("fd_scheduler_run_report", default=None)

# Ahead of a task's prompt (delivery_kind "agent"): Flight Deck sends the reply.
_TASK_NOTE = (
    "This is a scheduled task. Your reply is delivered to the user for you "
    "(their chat, email or Telegram, as they chose) — write the result itself; "
    "do not send it by email or message yourself. Earlier runs of this task "
    "are above in this session: build on them and say what changed rather "
    "than repeating old items.\n\n"
)


def _task_title(job: dict[str, Any]) -> str:
    """A task's name, or the start of its prompt."""
    name = " ".join(str(job.get("name") or "").split())
    if name:
        return name[:80]
    words = " ".join(str(job.get("prompt") or "").split())
    return (words[:57] + "…") if len(words) > 60 else (words or "Scheduled task")


def _agent_owner(slug: str) -> str | None:
    """Who spawned agent ``slug`` per FD's process registry ("" = nobody
    recorded), or None when FD doesn't know the slug."""
    if not slug:
        return None
    try:
        from captain_claw.flight_deck.server import _load_process_registry
        entry = _load_process_registry().get(slug)
    except Exception:
        return None
    if entry is None:
        return None
    return str(entry.get("owner") or "")


async def _user_is_admin(user_id: str) -> bool:
    try:
        from captain_claw.flight_deck.auth import get_db
        user = await get_db().get_user_by_id(user_id)
    except Exception:
        return False
    return bool(user) and str(user.get("role") or "") == "admin"


async def _owner_may_use_agent(owner: str, slug: str) -> bool:
    """A job runs only on its owner's own agent (an admin's on any). An
    ownerless legacy job, or an agent FD has no owner for, is unchecked."""
    if not owner:
        return True
    agent_owner = _agent_owner(slug)
    if not agent_owner or agent_owner == owner:
        return True
    return await _user_is_admin(owner)


def _owner_telegram_chat(owner: str) -> str:
    """The owner's approved chat with the deck's shared Telegram bot, or ""."""
    if not owner:
        return ""
    try:
        from captain_claw.flight_deck import telegram_shared
        return telegram_shared.owner_chat(owner)
    except Exception:
        return ""


async def _email_route(owner: str) -> dict[str, Any]:
    """How a task of ``owner``'s can email: ``available``, ``provider``
    (google | microsoft), ``from``, ``to`` (their own address) and ``reason``
    when it can't. Stored tokens and scopes only — no network."""
    out: dict[str, Any] = {"available": False, "provider": "", "from": "", "to": "", "reason": ""}
    if not owner:
        out["reason"] = "Email needs a signed-in account."
        return out
    try:
        from captain_claw.flight_deck.auth import get_db
        from captain_claw.flight_deck.invite_routes import mail_options
        db = get_db()
        user = await db.get_user_by_id(owner) or {}
        out["to"] = str(user.get("email") or "")
        options = await mail_options(db, owner)
    except Exception as exc:  # noqa: BLE001 — an options probe never sinks a run
        out["reason"] = f"Email isn't available here ({type(exc).__name__})."
        return out
    for provider in ("google", "microsoft"):
        if options.get(provider, {}).get("available"):
            out.update(available=True, provider=provider, **{"from": options[provider]["from"]})
            return out
    google, microsoft = options.get("google", {}), options.get("microsoft", {})
    out["reason"] = (
        "Connect Google or Microsoft in Connections to get results by email."
        if not google.get("from") and not microsoft.get("from")
        else (google.get("reason") or microsoft.get("reason") or "Email isn't available.")
    )
    return out


def _email_parts(title: str, text: str, when: datetime) -> tuple[str, str, str]:
    """(subject, plain text, HTML) for a task's result."""
    from html import escape

    from captain_claw.flight_deck.public_file_shares import _markdown_html

    stamp = f"{when:%a %d %b %Y, %H:%M}"
    subject = f"⏰ {title} — {when:%a %d %b}"
    footer = f"Scheduled task “{title}” · ran {stamp} · Captain Claw"
    plain = f"{text.strip()}\n\n—\n{footer}\n"
    html_body = (
        '<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;'
        'font-size:15px;line-height:1.55;color:#1f2328;max-width:680px">'
        f'<div style="font-size:12px;color:#6e7781;margin-bottom:12px">⏰ {escape(title)} · {escape(stamp)}</div>'
        f"{_markdown_html(text)}"
        f'<hr style="border:none;border-top:1px solid #d0d7de;margin:20px 0 10px">'
        f'<div style="font-size:12px;color:#6e7781">{escape(footer)}</div></div>'
    )
    return subject, plain, html_body


async def _deliver_email(job: dict[str, Any], delivery: dict[str, Any], text: str) -> dict[str, Any]:
    owner = str(job.get("owner_id") or "")
    route = await _email_route(owner)
    recipients = _clean_addresses(delivery.get("email_to")) or ([route["to"]] if route["to"] else [])
    out: dict[str, Any] = {"via": "email", "ok": False, "note": "", "to": ", ".join(recipients)}
    if not route["available"]:
        out["note"] = route["reason"]
        return out
    if not recipients:
        out["note"] = "No address to send to."
        return out
    from captain_claw.flight_deck.invite_routes import _MailError, _send_google, _send_microsoft

    subject, plain, html_body = _email_parts(_task_title(job), text, datetime.now())
    failed: list[str] = []
    for to in recipients:
        try:
            if route["provider"] == "google":
                await _send_google(owner, to, subject, plain, html_body)
            else:
                await _send_microsoft(owner, to, subject, html_body)
        except _MailError as exc:
            failed.append(f"{to}: {exc}")
    out["ok"] = not failed
    out["note"] = (f"sent from {route['from']}" if not failed else "; ".join(failed))[:500]
    return out


async def _deliver_telegram_owner(owner: str, text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"via": "telegram", "ok": False, "note": ""}
    chat = _owner_telegram_chat(owner)
    if not chat:
        out["note"] = "Telegram isn't linked — link it in Connections → Telegram."
        return out
    from captain_claw.flight_deck import telegram_shared
    from captain_claw.flight_deck.meta_webhook_bridge import strip_markdown

    body = strip_markdown(text).strip()
    try:
        for i in range(0, max(1, len(body)), 3900):
            if not await telegram_shared.send_text(chat, body[i:i + 3900]):
                out["note"] = "Telegram refused the message."
                return out
    except Exception as exc:  # noqa: BLE001
        out["note"] = f"Telegram failed ({type(exc).__name__})."
        return out
    out.update(ok=True, note="sent")
    return out


async def _deliver_task(job: dict[str, Any], delivery: dict[str, Any], text: str) -> list[dict[str, Any]]:
    """Deliver a task's result every way it asks for; one outcome each."""
    results: list[dict[str, Any]] = []
    if delivery["chat"]:
        # The agent posts it into its own chat (the mirror of this turn).
        results.append({"via": "chat", "ok": True, "note": "posted in the agent's chat"})
    if delivery["email"]:
        results.append(await _deliver_email(job, delivery, text))
    if delivery["telegram"]:
        results.append(await _deliver_telegram_owner(str(job.get("owner_id") or ""), text))
    return results


def _task_status(results: list[dict[str, Any]]) -> str:
    failed = [r["via"] for r in results if not r.get("ok")]
    if not failed:
        return "ok"
    if len(failed) == len(results):
        return "error:delivery " + ", ".join(failed)
    return "partial: " + ", ".join(failed) + " failed"


def _fire_time_preamble() -> str:
    """A dated header prepended to every scheduled text prompt at FIRE time.

    A job stores its prompt when created and fires it verbatim, often for weeks
    — so without this the agent reasons about "confirm/cancel this appointment"
    with no idea the appointment (or deadline) may already be days past. This is
    the scheduler's equivalent of the nervous system's tick clock: it anchors
    the fired prompt on the real clock and guards against re-nagging a stale,
    already-passed event. Falls back to plain UTC so a fire is never undated."""
    try:
        from captain_claw.config import get_config
        from captain_claw.system_info import build_datetime_lines
        tz = (get_config().context.timezone or "").strip() or None
        _, micro = build_datetime_lines(tz)
        now_line = micro
    except Exception:  # noqa: BLE001 — a clock line must never sink a fire
        from datetime import timezone
        now_line = f"{datetime.now(timezone.utc):%a %Y-%m-%d %H:%M} UTC"
    return (
        f"RIGHT NOW: {now_line}. Anchor every date/deadline judgement on this.\n"
        "If your message concerns an event, appointment, or deadline that is "
        "already in the PAST relative to the time above, do NOT treat it as "
        "upcoming (e.g. asking to confirm or cancel) — say plainly it has "
        "passed, or stay silent. Do not repeat a nudge you have no fresh reason "
        "to send again.\n\n"
    )


async def execute_job(job: dict[str, Any], *, force: bool = False) -> tuple[str, str]:
    """Run one job end-to-end. Returns ``(status, result_text)``.

    ``force=True`` (manual run-now) bypasses quiet hours.
    """
    if not force and not job.get("ignore_quiet_hours") and _in_quiet_hours():
        return ("skipped:quiet", "")
    # The owner's daily token allowance is spent: no runs until it resets.
    from captain_claw.flight_deck import usage_quota

    if await usage_quota.automation_blocked(str(job.get("owner_id") or "")):
        return ("skipped:limit", "")

    # A job runs either a Flow or a text prompt.
    flow_id = str(job.get("flow_id") or "").strip()
    if flow_id:
        return await _execute_flow_job(job, flow_id)

    slug = str(job.get("agent_slug") or "")
    if not await _owner_may_use_agent(str(job.get("owner_id") or ""), slug):
        return ("error:agent-not-yours", f"agent '{slug}' belongs to someone else")
    # Asleep for being idle and not woken ahead (the task was made or run just
    # now, or Flight Deck restarted): wake it for the run (agent_idle).
    agent_idle.touch(slug)
    await agent_idle.wake(slug)
    host, port, auth = resolve_agent_by_slug(slug, str(job.get("agent_auth") or ""))
    if not port:
        return ("error:agent-not-running", f"agent slug '{slug}' not running")

    # The turn is automated: it may write email only when the job's own
    # human-written prompt asks for it (judged on the raw prompt, no preamble).
    raw_prompt = str(job.get("prompt") or "")
    human_written = str(job.get("prompt_author") or "human") == "human"
    _kind = str(job.get("delivery_kind") or "").strip().lower()
    _target = str(job.get("delivery_target") or "").strip()
    delivery = norm_delivery(job.get("delivery")) if _kind == "agent" else None
    automation: dict[str, Any] = {
        "kind": "fd_scheduler",
        "job_text": (raw_prompt if human_written else "")[:4000],
        # fd_delivers: this run's result goes out through _deliver below, so
        # the agent's mirror of it into the main chat isn't relayed again by
        # the chat bridges (a scheduled WhatsApp job arrived twice). A task
        # delivering to the agent's chat lets the mirror through: that IS
        # its delivery (the kiosk timeline and the bridges bound to it).
        "mail_write": "intent", "fd_delivers": not (delivery and delivery["chat"]),
    }
    if job.get("id"):
        # Each task runs in its own agent session, kept across its runs.
        automation["task"] = {
            "id": str(job["id"]), "name": _task_title(job),
            "session_id": str(job.get("agent_session_id") or ""),
            **({"mirror": False} if delivery is not None and not delivery["chat"] else {}),
        }
    meta: dict[str, Any] = {}
    try:
        from captain_claw.flight_deck.agent_handshake import identity_frame

        identity = await identity_frame(str(job.get("owner_id") or ""), slug)
    except Exception:
        _log.debug("scheduler: no identity for %s", slug, exc_info=True)
        identity = None
    reply = await run_prompt_and_capture(
        host=host, port=port, auth=auth,
        prompt=_fire_time_preamble() + (_TASK_NOTE if delivery is not None else "") + raw_prompt,
        timeout=_TASK_REPLY_TIMEOUT_SECONDS if delivery is not None else _REPLY_TIMEOUT_SECONDS,
        automation=automation,
        whatsapp_media_to=_media_target(_kind, _target),
        meta=meta,
        identity=identity,
    )
    agent_idle.touch(slug)  # the idle time starts when the run ends
    report = _RUN_REPORT.get()
    task_session = meta.get("task_session") if isinstance(meta.get("task_session"), dict) else {}
    if report is not None and task_session.get("id"):
        report["session_id"] = str(task_session["id"])
    if reply is None:
        return ("error:no-reply", "agent produced no reply within timeout")

    if delivery is not None:
        results = await _deliver_task(job, delivery, reply)
        if report is not None:
            report["deliveries"] = results
        return (_task_status(results), reply)

    delivered, note = await _deliver(
        str(job.get("delivery_kind") or ""),
        str(job.get("delivery_target") or ""),
        reply,
    )
    if report is not None:
        report["deliveries"] = [{"via": _kind, "ok": delivered, "note": note}]
    status = note if delivered else note
    return (status, reply)


# ── Poll loop ─────────────────────────────────────────────────────────


# Jobs with a run in flight in this process (scheduled or run-now).
_IN_FLIGHT: set[str] = set()
_BACKGROUND: set[asyncio.Task] = set()
# One run at a time per agent: its automation lane takes one turn at a time
# anyway, and a run's reply timeout must start when its turn does.
_AGENT_LOCKS: dict[str, asyncio.Lock] = {}


def _next_after(job: dict[str, Any]) -> float | None:
    try:
        return compute_next_run(job["schedule"], tz=str(job.get("tz") or ""))
    except ScheduleError:
        return None


async def run_job(
    job: dict[str, Any], *, trigger: str = "scheduled", force: bool = False,
    run_id: int = 0,
) -> tuple[str, str, int]:
    """Run one job and log the run: ``(status, result_text, run_id)``.
    The job's last status/result follow; its next run is the caller's."""
    store = get_store()
    jid = str(job.get("id") or "")
    run_id = run_id or store.start_run(job, trigger)
    report: dict[str, Any] = {}
    token = _RUN_REPORT.set(report)
    _IN_FLIGHT.add(jid)
    status, result = "error:failed", ""
    lock = _AGENT_LOCKS.setdefault(str(job.get("agent_slug") or job.get("flow_id") or jid), asyncio.Lock())
    try:
        async with lock:
            status, result = await execute_job(job, force=force)
    except Exception as exc:  # noqa: BLE001 — a run always ends logged
        _log.warning("scheduler job %s failed: %s", jid, exc)
        status, result = f"error:{exc}", ""
    finally:
        _RUN_REPORT.reset(token)
        _IN_FLIGHT.discard(jid)
        session_id = str(report.get("session_id") or "")
        store.finish_run(run_id, status=status, result=result,
                         deliveries=report.get("deliveries") or [], session_id=session_id)
        if status != "skipped:quiet":
            store.record_result(jid, status=status, result=result, last_run_at=time.time())
        if session_id and session_id != str(job.get("agent_session_id") or ""):
            store.set_session(jid, session_id)
    return status, result, run_id


async def _run_due_job(store: SchedulerStore, job: dict[str, Any]) -> None:
    if not job.get("ignore_quiet_hours") and _in_quiet_hours():
        # Retry when quiet hours end; don't consume a one-shot.
        store.mark_run(
            job["id"], status="skipped:quiet", result="",
            last_run_at=time.time(), next_run_at=_quiet_hours_end_epoch(),
        )
        return

    if is_one_shot(job["schedule"]):
        # Its owner's daily token allowance is spent: retry when it resets,
        # don't consume the one-shot. (A recurring job skips this occurrence.)
        from captain_claw.flight_deck import usage_quota

        if await usage_quota.automation_blocked(str(job.get("owner_id") or "")):
            store.mark_run(
                job["id"], status="skipped:limit", result="",
                last_run_at=time.time(), next_run_at=usage_quota.resets_at().timestamp(),
            )
            return

    # Consumed before it runs (a long run must not fire again on the next
    # poll): one-shots disable, recurring jobs move to their next time.
    if is_one_shot(job["schedule"]):
        store.claim(job["id"], next_run_at=None, enabled=0)
    else:
        store.claim(job["id"], next_run_at=_next_after(job))
    await run_job(job, trigger="scheduled")


def _spawn(coro: Any) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)
    return task


async def scheduler_loop(stop_event: asyncio.Event) -> None:
    """Background poll loop. Started from FD's lifespan. Each due job runs
    in its own task, so a long one (a task can take minutes) doesn't hold up
    the others."""
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
                if job["id"] in _IN_FLIGHT:
                    continue
                try:
                    _spawn(_run_due_job(store, job))
                except Exception as exc:
                    _log.warning("scheduler job %s failed: %s", job.get("id"), exc)
                    store.mark_run(
                        job["id"], status=f"error:{exc}", result="",
                        last_run_at=time.time(),
                        next_run_at=None if is_one_shot(job["schedule"]) else _next_after(job),
                        enabled=(0 if is_one_shot(job["schedule"]) else None),
                    )
            # Claim synchronously what was just spawned: the next poll must
            # not see these as due again (a spawned task claims on its first
            # step, which runs once this loop yields).
            await asyncio.sleep(0)
        except Exception as exc:
            _log.warning("scheduler loop iteration error: %s", exc)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll)
        except asyncio.TimeoutError:
            pass
    _log.info("FD scheduler loop stopped")


# ── REST API ──────────────────────────────────────────────────────────


# What the API shows instead of a job's ``agent_auth`` (an agent's web token —
# never returned). The SchedulerPage edit form sends back what it was shown, so
# an update carrying this placeholder keeps the stored value; "" clears it.
_AGENT_AUTH_MASK = "********"


@dataclass(frozen=True)
class _SchedulerCaller:
    owner: str      # owner stamped on jobs it creates ("" = system)
    sees_all: bool  # an admin, or the one trusted user of an auth-off deck
    # A person (user JWT, or an auth-off request carrying no agent headers) —
    # the prompts it writes are human-written; an internal caller's are not.
    is_human: bool = True


# Headers only an agent / internal caller sends (compared case-insensitively).
_AGENT_HEADERS: tuple[str, ...] = ("x-agent-auth", "x-agent-secret", "x-agent-slug", "x-glasses-token")


def _norm_prompt_author(v: Any) -> str:
    return "agent" if str(v or "").strip().lower() == "agent" else "human"


def _calling_agent_owner(request: Request) -> str:
    """Owner FD recorded for the agent named by ``X-Agent-Auth`` — "" when the
    request names none, an unknown one, or one recorded without a real owner
    (the auth-off deck's synthetic ``local`` user counts as none)."""
    token = request.headers.get("X-Agent-Auth", "").strip()
    if not token:
        return ""
    try:
        from captain_claw.flight_deck.auth import _LOCAL_USER
        from captain_claw.flight_deck.server import _find_agent_by_auth
        matched, owner, _slug = _find_agent_by_auth(token)
    except Exception:  # fail closed: an unidentified caller is "system"
        return ""
    owner = str(owner or "") if matched else ""
    return "" if owner == _LOCAL_USER["id"] else owner


def _require_scheduler_caller(request: Request) -> _SchedulerCaller:
    """Authenticate a scheduler API caller and say whose jobs it may reach.

    Scheduler jobs deliver to the owner's WhatsApp/Telegram, so the routes must
    never be open. Accept any of:
      * a valid Flight Deck user JWT (the SchedulerPage UI) — that user's own
        jobs, or every job for an admin;
      * a trusted internal caller — the shared agent secret (X-Agent-Secret),
        or a loopback request when not in lockdown (intentions.py materializing
        a job on the same host);
      * the legacy glasses shared secret when FD_GLASSES_BRIDGE_TOKEN is set.
    An internal caller acts for the owner of the agent its ``X-Agent-Auth``
    names; one that names none may create (system) jobs but reads none — so
    dropping the header never widens an agent's view.
    Anything else is rejected. This closes the prior hole where an unset
    FD_GLASSES_BRIDGE_TOKEN made ``_check_token`` pass every request.
    """
    from captain_claw.flight_deck.auth import decode_access_token, _fd_auth_enabled

    # Local/standalone mode (auth disabled) — single trusted user.
    if not _fd_auth_enabled():
        hdrs = {k.lower() for k in request.headers.keys()}
        return _SchedulerCaller(
            owner="", sees_all=True, is_human=not any(h in hdrs for h in _AGENT_HEADERS),
        )

    # (a) Flight Deck user JWT.
    auth_hdr = request.headers.get("Authorization", "")
    tok = auth_hdr[7:].strip() if auth_hdr.lower().startswith("bearer ") else ""
    if not tok:
        tok = request.query_params.get("fd_token", "") or ""
    if tok:
        try:
            payload = decode_access_token(tok)
        except Exception:
            payload = None
        if not payload or not payload.get("sub"):
            # A user token that doesn't verify (usually an expired session)
            # must not fall through to the internal branches: on loopback the
            # user would silently become "system" — an empty list, and the jobs
            # they create would land ownerless (admin-only).
            raise HTTPException(status_code=401, detail="invalid or expired session token")
        return _SchedulerCaller(
            owner=str(payload["sub"]),
            sees_all=payload.get("role", "user") == "admin",
        )

    internal = False
    # (b) Trusted internal caller — shared agent secret.
    provided = request.headers.get("X-Agent-Secret", "")
    if provided:
        try:
            from captain_claw.flight_deck.agent_secret import get_or_create_agent_secret
            internal = secrets.compare_digest(provided, get_or_create_agent_secret())
        except Exception:
            pass

    # (c) Legacy glasses shared secret (only when configured).
    required = os.environ.get("FD_GLASSES_BRIDGE_TOKEN", "").strip()
    if not internal and required:
        got = request.query_params.get("t", "") or request.headers.get("x-glasses-token", "")
        internal = bool(got) and secrets.compare_digest(got, required)

    # (d) Loopback fallback for same-host internal callers, unless locked down.
    if not internal and os.environ.get("FD_LOCKDOWN", "").lower() not in ("true", "1", "yes"):
        client_host = request.client.host if request.client else ""
        internal = client_host in ("127.0.0.1", "::1", "localhost")

    if internal:
        return _SchedulerCaller(owner=_calling_agent_owner(request), sees_all=False, is_human=False)
    raise HTTPException(status_code=401, detail="scheduler requires authentication")


def _caller_may_access(caller: _SchedulerCaller, job: dict[str, Any]) -> bool:
    """Admins reach every job; anyone else only jobs stamped with their own
    (non-empty) owner — so ownerless legacy/system jobs are admin-only."""
    if caller.sees_all:
        return True
    return bool(caller.owner) and str(job.get("owner_id") or "") == caller.owner


def _job_for_caller(job_id: str, caller: _SchedulerCaller) -> dict[str, Any]:
    """The job, or 404 — also for another user's job, so ids don't leak."""
    job = get_store().get(job_id)
    if not job or not _caller_may_access(caller, job):
        raise HTTPException(status_code=404, detail="job not found")
    return job


def _public_job(job: dict[str, Any], stats: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """A job as the API returns it: ``agent_auth`` redacted to a placeholder,
    ``delivery`` as an object, and its run count / whether it's running now."""
    out = dict(job)
    out["agent_auth"] = _AGENT_AUTH_MASK if job.get("agent_auth") else ""
    out["delivery"] = norm_delivery(job.get("delivery")) if job.get("delivery_kind") == "agent" else None
    st = (stats if stats is not None else get_store().run_stats()).get(str(job.get("id") or ""), {})
    out["run_count"] = int(st.get("runs") or 0)
    out["running"] = bool(st.get("running")) or str(job.get("id") or "") in _IN_FLIGHT
    return out


def _check_agent_access(caller: _SchedulerCaller, body: dict[str, Any]) -> None:
    """A caller's job may only drive its own agent (an admin's, any)."""
    slug = str(body.get("agent_slug") or "").strip()
    if caller.sees_all or not slug:
        return
    owner = _agent_owner(slug)
    if owner and owner != caller.owner:
        raise HTTPException(status_code=403, detail="that agent isn't yours")


def _limit_agent_caller(caller: _SchedulerCaller, body: dict[str, Any]) -> None:
    """Only a person picks who a task emails: an agent's job emails its
    owner's own address, never one the agent wrote."""
    if caller.is_human or "delivery" not in body:
        return
    d = body.get("delivery")
    if isinstance(d, dict):
        body["delivery"] = {**d, "email_to": ""}


async def _json_object(request: Request) -> dict[str, Any]:
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="expected a JSON object")
    # The owner comes from the caller, never the body; and the redaction
    # placeholder is not a token — on update it means "unchanged".
    body.pop("owner_id", None)
    # Who wrote the prompt is the caller's identity, never a body claim.
    body.pop("prompt_author", None)
    if body.get("agent_auth") == _AGENT_AUTH_MASK:
        body.pop("agent_auth")
    return body


@router.get("/scheduler/jobs")
async def list_jobs(request: Request) -> JSONResponse:
    caller = _require_scheduler_caller(request)
    if caller.sees_all:
        rows = get_store().list()
    elif caller.owner:
        rows = get_store().list(owner_id=caller.owner)
    else:
        rows = []
    stats = get_store().run_stats()
    return JSONResponse([_public_job(r, stats) for r in rows], headers=_NO_CACHE)


@router.get("/scheduler/recipients")
async def list_recipients(request: Request) -> JSONResponse:
    """Named delivery recipients for the UI pickers.

    Currently surfaces Telegram recipients parsed from ``TELEGRAM_RECIPIENTS``.
    WhatsApp uses a single allowlisted number so it doesn't need a picker.
    """
    _require_scheduler_caller(request)
    from captain_claw.flight_deck.telegram_out import list_recipients as _tg
    return JSONResponse({"telegram": _tg()}, headers=_NO_CACHE)


@router.post("/scheduler/jobs")
async def create_job(request: Request) -> JSONResponse:
    caller = _require_scheduler_caller(request)
    body = await _json_object(request)
    _check_agent_access(caller, body)
    _limit_agent_caller(caller, body)
    try:
        row = get_store().create(
            owner_id=caller.owner, prompt_author="human" if caller.is_human else "agent", **body,
        )
    except (ScheduleError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(_public_job(row), headers=_NO_CACHE)


@router.get("/scheduler/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> JSONResponse:
    caller = _require_scheduler_caller(request)
    row = _job_for_caller(job_id, caller)
    return JSONResponse(_public_job(row), headers=_NO_CACHE)


@router.patch("/scheduler/jobs/{job_id}")
async def update_job(job_id: str, request: Request) -> JSONResponse:
    caller = _require_scheduler_caller(request)
    _job_for_caller(job_id, caller)
    body = await _json_object(request)
    _check_agent_access(caller, body)
    _limit_agent_caller(caller, body)
    try:
        if "prompt" in body:
            row = get_store().update(
                job_id, prompt_author="human" if caller.is_human else "agent", **body,
            )
        else:
            row = get_store().update(job_id, **body)
    except (ScheduleError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not row:
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse(_public_job(row), headers=_NO_CACHE)


@router.delete("/scheduler/jobs/{job_id}")
async def delete_job(job_id: str, request: Request) -> JSONResponse:
    caller = _require_scheduler_caller(request)
    _job_for_caller(job_id, caller)
    if not get_store().delete(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse({"ok": True}, headers=_NO_CACHE)


@router.post("/scheduler/jobs/{job_id}/run")
async def run_job_now(job_id: str, request: Request, background: bool = False) -> JSONResponse:
    """Manually fire a job immediately, bypassing schedule + quiet hours.

    Does NOT change the job's next_run_at — it's an out-of-band test fire.
    ``?background=1`` answers at once with the run's id (the run shows as
    ``running`` in /scheduler/runs until it ends); otherwise the call waits
    for the result.
    """
    caller = _require_scheduler_caller(request)
    job = _job_for_caller(job_id, caller)
    if job_id in _IN_FLIGHT:
        raise HTTPException(status_code=409, detail="this task is running right now")
    if background:
        run_id = get_store().start_run(job, "manual")
        _IN_FLIGHT.add(job_id)
        _spawn(run_job(job, trigger="manual", force=True, run_id=run_id))
        return JSONResponse({"ok": True, "status": "running", "run_id": run_id}, headers=_NO_CACHE)
    status, result, run_id = await run_job(job, trigger="manual", force=True)
    return JSONResponse(
        {"ok": status.startswith("ok"), "status": status, "run_id": run_id,
         "result_preview": result[:500]},
        headers=_NO_CACHE,
    )


@router.get("/scheduler/runs")
async def list_runs(
    request: Request, job_id: str = "", q: str = "", limit: int = 50, before: int = 0,
) -> JSONResponse:
    """Run history, newest first — one task's (``job_id``) or every task the
    caller can see; ``q`` searches the results and task names; ``before``
    pages back past a run id."""
    caller = _require_scheduler_caller(request)
    if job_id:
        _job_for_caller(job_id, caller)
    if not caller.sees_all and not caller.owner:
        return JSONResponse([], headers=_NO_CACHE)
    rows = get_store().list_runs(
        owner_id=None if caller.sees_all else caller.owner,
        job_id=job_id, q=q[:200], limit=limit, before_id=before,
    )
    return JSONResponse(rows, headers=_NO_CACHE)


@router.post("/scheduler/preview")
async def preview_schedule(request: Request) -> JSONResponse:
    """The next few times a schedule would fire: ``{schedule, tz}`` →
    ``{ok, next: [epoch…], one_shot, error}``. Nothing is saved."""
    _require_scheduler_caller(request)
    body = await _json_object(request)
    schedule = str(body.get("schedule") or "").strip()
    tz = str(body.get("tz") or "").strip()
    try:
        first = compute_next_run(schedule, tz=tz)
    except ScheduleError as exc:
        return JSONResponse({"ok": False, "next": [], "one_shot": False, "error": str(exc)},
                            headers=_NO_CACHE)
    nxt: list[float] = [first] if first is not None else []
    one_shot = is_one_shot(schedule)
    if one_shot and first is not None and first < time.time() - 60:
        return JSONResponse({"ok": False, "next": nxt, "one_shot": True,
                             "error": "that time has already passed"}, headers=_NO_CACHE)
    while not one_shot and nxt and len(nxt) < 3:
        zone = _zone(tz)
        base = datetime.fromtimestamp(nxt[-1] + 1, zone) if zone else datetime.fromtimestamp(nxt[-1] + 1)
        nxt.append(compute_next_run(schedule, base=base, tz=tz))
    return JSONResponse({"ok": True, "next": nxt, "one_shot": one_shot, "error": ""},
                        headers=_NO_CACHE)


@router.get("/scheduler/delivery-options")
async def delivery_options(request: Request) -> JSONResponse:
    """Where the caller's tasks can deliver: ``chat`` always; ``email`` from
    their connected Gmail / Outlook (to their own address unless they name
    others); ``telegram`` when they've linked the deck's shared bot."""
    caller = _require_scheduler_caller(request)
    email = await _email_route(caller.owner)
    tg = bool(_owner_telegram_chat(caller.owner))
    return JSONResponse({
        "chat": {"available": True},
        "email": email,
        "telegram": {"available": tg,
                     "reason": "" if tg else "Link Telegram in Connections → Telegram to get results there."},
    }, headers=_NO_CACHE)
