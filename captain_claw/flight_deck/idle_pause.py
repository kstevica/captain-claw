"""Scheduled tasks pause when their owner has been away for a week.

A deck's scheduled work kept running for people who had stopped coming: one
owner's two daily jobs ran for sixteen days after their last sign-in, another's
twice-a-day task reported "no change" thirty times into a Line nobody read.

Once an hour, for every account on a deck with sign-in on: when the owner's last
activity is as many days old as the admin's setting (Settings → Agents, "Pause
scheduled tasks for people who are away"; ``scheduled_idle_pause_days`` in the
deck config, default 7), their scheduled tasks are paused — Flight Deck's own (``fd_scheduler``) and the jobs their agents set
up for themselves (``agent_cron_routes``) — and one item on their Line says so.
Nothing is deleted; they resume each one in Scheduled tasks.

A job that runs less often than that waits longer: twice its own period (a
weekly job pauses after two weeks away). Someone who reads a Monday scan each
Monday is not away from it; their weekly job was paused three days after the
Monday they last came (cohort001, 2026-11-20, with the setting at 3 days).

Activity is the later of: signing in or coming back to the deck on a saved
session (``users.last_login_at``), and a message they sent their agents from any
channel (``daily_usage``). A new account counts from when it was made.
A setting of 0 turns it off.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import UTC, datetime
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

DEFAULT_DAYS = 7
MAX_DAYS = 365
CONFIG_KEY = "scheduled_idle_pause_days"
CHECK_EVERY_S = 3600.0
FIRST_CHECK_S = 600.0
_NAMES_SHOWN = 8


async def idle_days() -> int:
    """The admin's setting: days without activity before tasks pause (0 =
    never). Unreadable: 0, so nothing pauses on a guess."""
    try:
        from captain_claw.flight_deck.admin_routes import _load_system_config
        from captain_claw.flight_deck.auth import get_db

        cfg = _load_system_config(await get_db().get_system_setting("fd:system-config"))
        days = int(cfg.get(CONFIG_KEY, DEFAULT_DAYS) or 0)
    except Exception:  # noqa: BLE001
        return 0
    return min(MAX_DAYS, max(0, days))


async def pause_days() -> int:
    """Days before an away owner's tasks pause on this deck; 0 when it doesn't
    (the setting is 0, or there are no accounts to be away from)."""
    from captain_claw.flight_deck.auth import _fd_auth_enabled

    if not _fd_auth_enabled():
        return 0
    return await idle_days()


def _epoch(text: Any) -> float:
    value = str(text or "").strip()
    if not value:
        return 0.0
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def last_activity(user: dict[str, Any], last_typed_day: str = "") -> float:
    """When *user* was last active (epoch): signed in, came back, or typed —
    a day typed counts to its end. Never earlier than the account."""
    typed = 0.0
    if last_typed_day:
        start = _epoch(f"{last_typed_day}T00:00:00+00:00")
        typed = start + 86400.0 if start else 0.0
    return max(_epoch(user.get("last_login_at")), typed, _epoch(user.get("created_at")))


def period_days(schedule: Any) -> float:
    """How many days apart a job's runs are, from its schedule: Flight Deck's
    own (``"weekly mon 08:00"``, ``"every 2d"``…) or an agent's (``{"type":
    "weekly", …}``). 0 for a one-shot or a schedule this can't read."""
    if isinstance(schedule, dict):
        typ = str(schedule.get("type") or "").strip().lower()
        if typ == "interval":
            try:
                n = max(0, int(schedule.get("interval", 0)))
            except (TypeError, ValueError):
                return 0.0
            unit = str(schedule.get("unit") or "minutes").lower()
            return n / (24.0 if unit.startswith("hour") else 1440.0)
        return {"daily": 1.0, "weekly": 7.0}.get(typ, 0.0)
    text = str(schedule or "").strip().lower()
    m = re.fullmatch(r"every\s+(\d+)\s*([mhd])", text)
    if m:
        return int(m.group(1)) / {"m": 1440.0, "h": 24.0, "d": 1.0}[m.group(2)]
    if re.match(r"(?:daily|weekdays)\s", text):
        return 1.0
    m = re.fullmatch(r"weekly\s+([a-z][a-z,\s]*?)\s+\d{1,2}:\d{2}", text)
    if m:
        days = {d for d in re.split(r"[,\s]+", m.group(1)) if d}
        return 7.0 / max(1, len(days))
    return 0.0


def pause_after_days(schedule: Any, days: int) -> float:
    """Days away before a job with *schedule* pauses: the setting, or twice the
    job's own period when that is longer."""
    return max(float(days), 2.0 * period_days(schedule))


def _srv():
    from captain_claw.flight_deck import server

    return server


def _owned_agents(user_id: str) -> dict[str, dict]:
    """slug → registry entry of the user's own agents (not the deck's workers)."""
    from captain_claw.flight_deck.agent_sharing import is_managed_agent

    out: dict[str, dict] = {}
    for slug, entry in _srv()._load_process_registry().items():
        if not isinstance(entry, dict) or str(entry.get("owner") or "") != user_id:
            continue
        if entry.get("role") == "worker" or is_managed_agent(slug, str(entry.get("description") or "")):
            continue
        out[slug] = entry
    return out


def _pause_deck_tasks(user_id: str, away: float, days: int) -> list[str]:
    """Pause the user's enabled Flight Deck tasks whose wait (pause_after_days)
    *away* days are past; their names."""
    from captain_claw.flight_deck.fd_scheduler import get_store

    store = get_store()
    names: list[str] = []
    for job in store.list(owner_id=user_id):
        if not job.get("enabled") or away < pause_after_days(job.get("schedule"), days):
            continue
        try:
            store.update(job["id"], enabled=False)
        except Exception as exc:  # noqa: BLE001 — the others still pause
            log.warning("Idle pause: task not paused", owner=user_id, job=job.get("id"), error=str(exc))
            continue
        names.append(str(job.get("name") or "").strip() or "Scheduled task")
    return names


async def _pause_agent_jobs(user_id: str, away: float, days: int) -> list[str]:
    """Pause the jobs the user's agents set up for themselves whose wait
    (pause_after_days) *away* days are past; their names."""
    from captain_claw.flight_deck import agent_cron_routes as acr

    names: list[str] = []
    for slug, entry in _owned_agents(user_id).items():
        agent = str(entry.get("name") or slug)
        for row in acr._read_rows(slug):
            schedule = acr._loads(row["schedule"], {}) if "schedule" in row.keys() else {}
            if not row["enabled"] or away < pause_after_days(schedule, days):
                continue
            job_id = str(row["id"])
            try:
                if _srv()._process_is_alive(slug):
                    await acr._agent_rest(entry, "POST", f"/api/cron/jobs/{job_id}/pause", agent)
                elif not acr._edit_file(slug, job_id, "pause"):
                    continue
            except Exception as exc:  # noqa: BLE001 — the others still pause
                log.warning("Idle pause: agent job not paused", owner=user_id, agent=slug,
                            job=job_id, error=str(getattr(exc, "detail", exc)))
                continue
            payload = acr._loads(row["payload"], {})
            text = acr._task_text(str(row["kind"] or ""), payload if isinstance(payload, dict) else {})
            names.append(f"{agent}: {text[:60]}" if text else agent)
    return names


def _line_body(names: list[str], days: int) -> str:
    shown = names[:_NAMES_SHOWN]
    more = len(names) - len(shown)
    listed = "\n".join(f"• {n}" for n in shown) + (f"\n• … and {more} more" if more > 0 else "")
    return (f"You haven't been on the deck for {days} days, so your scheduled tasks are paused "
            f"— nothing was deleted:\n{listed}\n\nResume any of them in Scheduled tasks when you "
            "want them running again.")


async def pause_idle_owner(user: dict[str, Any], days: int, away: float | None = None) -> int:
    """Pause an owner's tasks *away* days (default: the setting) away is
    past the wait of, and tell them on their Line; how many."""
    uid = str(user.get("id") or "")
    away = float(days) if away is None else away
    names = _pause_deck_tasks(uid, away, days) + await _pause_agent_jobs(uid, away, days)
    if not names:
        return 0
    log.info("Idle pause: tasks paused", owner=uid, tasks=len(names), idle_days=days, away_days=int(away))
    from captain_claw.flight_deck import line

    await line.post(uid, "deck", f"Scheduled tasks paused ({len(names)})", _line_body(names, int(away)),
                    bell_type="scheduled_paused", needs_you=True, source="Flight Deck")
    return len(names)


async def sweep(db: Any, now: float | None = None) -> int:
    """One pass over every account; how many tasks were paused."""
    days = await pause_days()
    if days <= 0:
        return 0
    now = now if now is not None else time.time()
    typed = await db.message_day_bounds()
    paused = 0
    for user in await db.list_users(limit=100_000):
        uid = str(user.get("id") or "")
        if not uid:
            continue
        away = (now - last_activity(user, str((typed.get(uid) or {}).get("last_day") or ""))) / 86400.0
        if away < days:
            continue
        try:
            paused += await pause_idle_owner(user, days, away)
        except Exception as exc:  # noqa: BLE001 — one owner never stops the others
            log.warning("Idle pause: owner skipped", owner=uid, error=str(exc))
    return paused


async def loop(db: Any, stop: asyncio.Event) -> None:
    """Flight Deck's lifespan: a pass ten minutes after start, then hourly."""
    delay = FIRST_CHECK_S
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=delay)
            return
        except TimeoutError:
            pass
        delay = CHECK_EVERY_S
        try:
            await sweep(db)
        except Exception as exc:  # noqa: BLE001 — the loop outlives any one bad pass
            log.warning("Idle pause pass failed", error=str(exc))

