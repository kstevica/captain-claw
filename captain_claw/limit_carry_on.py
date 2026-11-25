"""A turn stopped by the plan's usage limit carries on by itself after the reset.

When the model's plan (the person's ChatGPT subscription, say) is used up,
retrying is pointless until it resets — often hours away. The agent loop asks
:func:`pause_turn` instead: the turn ends with a plain message saying when it
goes on, and a one-shot job in the agent's own scheduled jobs (``cron_jobs``
in its sessions.db) carries on at the reset plus a margin. Kept there, it
survives the agent sleeping or restarting (Flight Deck wakes an agent before
its jobs are due), it's listed in Flight Deck's Scheduled tasks ("Set up by
agents") and can be removed there (``/cron remove <id>`` in the terminal).

The job runs back in the chat the limit stopped (``run_session``), so the
model has what it was doing in front of it. One per chat: a later turn the
limit stops moves the pending job instead of adding another.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.plan_limit import PlanLimit, from_error

log = get_logger(__name__)

# The job's author (payload["author"]): what mail authority it carries is the
# stored text of the request the limit interrupted (mail_authority.cron_job_text).
AUTHOR = "plan_limit"
# payload[MARK] = {"run_session": …, "plan": …, "reset_at": …, "guesses": n}
MARK = "plan_limit"
CARRY_ON_TEXT = (
    "Carry on with what you were doing before the usage limit stopped you. "
    "If it's already done, or the conversation has moved on since, say so in one line "
    "and don't redo anything."
)
GUESS_S = 30 * 60   # it didn't say when the limit resets: try again this often…
GUESSES = 6         # …this many times in a row, then wait for the person


def margin_seconds() -> float:
    """How long after the reset the carry-on runs: the server's clock and ours
    needn't agree to the second."""
    try:
        return max(0.0, float(os.environ.get("CLAW_LIMIT_MARGIN_SECONDS", 90)))
    except ValueError:
        return 90.0


@dataclass(frozen=True)
class CarryOn:
    """The carry-on that's waiting."""

    job_id: str
    at: float          # epoch seconds
    guessed: bool      # no reset time was said: it only tries again then


def marker(payload: Any) -> dict[str, Any] | None:
    """The carry-on mark on a job's payload, if it's one."""
    if not isinstance(payload, dict):
        return None
    mark = payload.get(MARK)
    return mark if isinstance(mark, dict) and payload.get("author") == AUTHOR else None


def run_session(payload: Any) -> str:
    """The session a carry-on job runs in (the chat the limit stopped), or ""."""
    mark = marker(payload)
    return str((mark or {}).get("run_session") or "").strip()


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).isoformat()


def _intent_text(agent: Any) -> str:
    """What the person wrote asking for the interrupted work (mail authority
    for the carry-on is judged on it, as for a job the agent set up)."""
    from captain_claw import mail_authority

    try:
        return mail_authority.recent_intent_text(agent)[:mail_authority.JOB_TEXT_MAX]
    except Exception:  # noqa: BLE001 — none stored: the carry-on may not write mail
        return ""


async def schedule(agent: Any, limit: PlanLimit, now: float | None = None) -> CarryOn | None:
    """Set up (or move) the one carry-on for the chat the agent is on; None
    when this turn doesn't get one: an agent whose scheduled jobs don't run
    here (an orchestrator worker, a sister session, a scoped lane), no
    session, a turn another automation started (its own runner decides), or
    no reset time said too many times."""
    from captain_claw import turn_register
    from captain_claw.cron_dispatch import job_chat_session_id

    # The agent the cron loop runs with is handed its runtime context where
    # the loop starts (web_server, main): only its chats get a carry-on.
    if getattr(agent, "_runtime_context", None) is None:
        return None
    session = getattr(agent, "session", None)
    sm = getattr(agent, "session_manager", None)
    if session is None or not getattr(session, "id", "") or not hasattr(sm, "create_cron_job"):
        return None
    kind, ref = turn_register.automation()
    if kind and kind != "cron":
        return None
    running = ref if kind == "cron" else ""
    now = time.time() if now is None else now

    mine = []
    for job in await sm.list_cron_jobs(limit=500, active_only=True):
        if run_session(job.payload) == session.id:
            mine.append(job)
    # The carry-on running now is done after this turn (a one-shot turns
    # itself off): the next one is a new job.
    current = next((j for j in mine if j.id == running), None)
    pending = [j for j in mine if j.id != running]
    before = current or (pending[0] if pending else None)
    guesses = int((marker(before.payload) or {}).get("guesses") or 0) if before else 0

    guessed = limit.reset_at is None
    if guessed:
        guesses += 1
        if guesses > GUESSES:
            for job in pending:
                await sm.delete_cron_job(job.id)
            log.info("plan limit: no reset time again, not trying by itself", session_id=session.id)
            return None
        at = now + GUESS_S
    else:
        guesses = 0
        at = max(limit.reset_at, now) + margin_seconds()

    payload = {
        "text": CARRY_ON_TEXT,
        "author": AUTHOR,
        "mail_intent_text": _intent_text(agent),
        MARK: {
            "run_session": session.id,
            "plan": limit.plan,
            "reset_at": _iso(limit.reset_at) if limit.reset_at else "",
            "guesses": guesses,
        },
    }
    when = {"type": "once", "target_iso": _iso(at)}
    if pending:
        job_id = pending[0].id
        for extra in pending[1:]:
            await sm.delete_cron_job(extra.id)
        await sm.update_cron_job(job_id, payload=payload, schedule=when, next_run_at=_iso(at), enabled=True)
        log.info("plan limit: carry-on moved", job_id=job_id, at=_iso(at), guessed=guessed)
    else:
        # It reports to the chat a job's own session reports to, else the chat itself.
        reply_to = job_chat_session_id(session) or session.id
        job = await sm.create_cron_job(
            kind="prompt", payload=payload, schedule=when, session_id=reply_to, next_run_at=_iso(at),
        )
        job_id = job.id
        log.info("plan limit: carry-on scheduled", job_id=job_id, at=_iso(at), guessed=guessed)
    return CarryOn(job_id=job_id, at=at, guessed=guessed)


def _zone() -> tzinfo:
    from captain_claw.system_info import _resolve_timezone

    try:
        return _resolve_timezone(None)[1] or UTC
    except Exception:  # noqa: BLE001
        return UTC


def clock(at: float, now: float, zone: tzinfo | None = None) -> str:
    """``17:05 CET`` today, ``Thu 27 Nov 09:15 CET`` another day — rounded up
    to the minute, so it's never earlier than the moment itself."""
    zone = zone or _zone()
    at = math.ceil(at / 60) * 60
    local = datetime.fromtimestamp(at, zone)
    today = datetime.fromtimestamp(now, zone).date()
    label = local.tzname() or "UTC"
    day = "" if local.date() == today else f"{local:%a} {local.day} {local:%b} "
    return f"{day}{local:%H:%M} {label}"


def span(at: float, now: float) -> str:
    """``45 min``, ``2 h 10 min``, ``3 d 4 h``."""
    minutes = max(1, math.ceil((at - now) / 60))
    if minutes < 60:
        return f"{minutes} min"
    hours, mins = divmod(minutes, 60)
    if hours < 24:
        return f"{hours} h {mins} min" if mins else f"{hours} h"
    days, hours = divmod(hours, 24)
    return f"{days} d {hours} h" if hours else f"{days} d"


def message(limit: PlanLimit, carry_on: CarryOn | None, now: float, zone: tzinfo | None = None) -> str:
    """What the person is told — never the provider's own text."""
    plan = f"The {limit.plan} plan's" if limit.plan else "The model plan's"
    head = f"{plan} usage limit is used up"
    if carry_on is not None:
        when = f"{clock(carry_on.at, now, zone)} (in {span(carry_on.at, now)})"
        cancel = "To cancel, remove it under Scheduled tasks."
        if carry_on.guessed:
            return (f"{head}, and it didn't say when it resets; I'll try again by myself at {when}. "
                    f"{cancel}")
        return f"{head}; I'll carry on by myself at {when}. {cancel}"
    if limit.reset_at is not None:
        at = limit.reset_at
        return (f"{head} until {clock(at, now, zone)} (in {span(at, now)}). "
                "Send me a message after that and I'll carry on.")
    return f"{head}, and it didn't say when it resets. Send me a message later and I'll carry on."


async def pause_turn(agent: Any, err: BaseException, now: float | None = None) -> str | None:
    """The message a turn ends with when *err* is the plan's usage limit (its
    carry-on set up), or None when *err* isn't one."""
    now = time.time() if now is None else now
    limit = from_error(err, now=now, local=_zone())
    if limit is None:
        return None
    if not limit.plan:
        provider = type(getattr(agent, "provider", None)).__name__
        plan = {"ChatGPTResponsesProvider": "ChatGPT", "ClaudeCLIProvider": "Claude"}.get(provider, "")
        if plan:
            limit = PlanLimit(reset_at=limit.reset_at, plan=plan)
    carry_on = None
    try:
        carry_on = await schedule(agent, limit, now=now)
    except Exception:  # noqa: BLE001 — the turn still ends with the plain message
        log.warning("plan limit: carry-on not scheduled", exc_info=True)
    log.info("Plan usage limit: turn paused", plan=limit.plan or "?",
             reset_at=_iso(limit.reset_at) if limit.reset_at else "", carry_on=bool(carry_on))
    return message(limit, carry_on, now)
