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

Activity is the later of: signing in or coming back to the deck on a saved
session (``users.last_login_at``), and a message they sent their agents from any
channel (``daily_usage``). A new account counts from when it was made.
A setting of 0 turns it off.
"""

from __future__ import annotations

import asyncio
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


def _pause_deck_tasks(user_id: str) -> list[str]:
    """Pause the user's enabled Flight Deck tasks; their names."""
    from captain_claw.flight_deck.fd_scheduler import get_store

    store = get_store()
    names: list[str] = []
    for job in store.list(owner_id=user_id):
        if not job.get("enabled"):
            continue
        try:
            store.update(job["id"], enabled=False)
        except Exception as exc:  # noqa: BLE001 — the others still pause
            log.warning("Idle pause: task not paused", owner=user_id, job=job.get("id"), error=str(exc))
            continue
        names.append(str(job.get("name") or "").strip() or "Scheduled task")
    return names


async def _pause_agent_jobs(user_id: str) -> list[str]:
    """Pause the jobs the user's agents set up for themselves; their names."""
    from captain_claw.flight_deck import agent_cron_routes as acr

    names: list[str] = []
    for slug, entry in _owned_agents(user_id).items():
        agent = str(entry.get("name") or slug)
        for row in acr._read_rows(slug):
            if not row["enabled"]:
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


async def pause_idle_owner(user: dict[str, Any], days: int) -> int:
    """Pause one away owner's tasks and tell them on their Line; how many."""
    uid = str(user.get("id") or "")
    names = _pause_deck_tasks(uid) + await _pause_agent_jobs(uid)
    if not names:
        return 0
    log.info("Idle pause: tasks paused", owner=uid, tasks=len(names), idle_days=days)
    from captain_claw.flight_deck import line

    await line.post(uid, "deck", f"Scheduled tasks paused ({len(names)})", _line_body(names, days),
                    bell_type="scheduled_paused", needs_you=True, source="Flight Deck")
    return len(names)


async def sweep(db: Any, now: float | None = None) -> int:
    """One pass over every account; how many tasks were paused."""
    days = await pause_days()
    if days <= 0:
        return 0
    cutoff = (now if now is not None else time.time()) - days * 86400.0
    typed = await db.message_day_bounds()
    paused = 0
    for user in await db.list_users(limit=100_000):
        uid = str(user.get("id") or "")
        if not uid:
            continue
        if last_activity(user, str((typed.get(uid) or {}).get("last_day") or "")) > cutoff:
            continue
        try:
            paused += await pause_idle_owner(user, days)
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

