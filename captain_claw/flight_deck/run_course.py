"""Keep on course for Basna and Vatra workers (docs/jev-course-plan.md §3).

Workers run the same agent loop as a chat, so the task anchor and the Jev
course check run inside each worker. Flight Deck only supplies the anchor and
collects the answers:

* **The frame.** Every worker chat frame of a run carries a ``course`` field:
  the run's goal (the person's words), the worker's own part (its subtask or
  role), whether the run checks its course with Jev, and the worker's share of
  the run's check budget. The worker's anchor then reads "Run goal / Your
  part" instead of its long prompt (``task_course.apply_frame``).
* **The switch.** A run started by an agent's ``vatra``/``basna`` tool follows
  that agent's Keep on course switch; one started from the page is on (it does
  anything only while the owner has Jev). A continuation round inherits the
  run's value from the session config.
* **The answers.** A worker's ``jev_course`` monitor event rides back with its
  stage, ``p_on_track`` and next move (:func:`course_from_monitor`); Vatra keeps
  it as a ``course`` ledger row per subtask (the run map shows each owner's
  stage) and both modes show it in the run's progress feed, with a flag when an
  owner is off track in two checks in a row.

Run-scoped values ride in context variables set at run start (like the
dispatch gate), so no dispatch call site has to thread them.
"""

from __future__ import annotations

import contextvars
import os
import re
from typing import Any

MAX_CHECKS_PER_RUN = 150
MAX_CHECKS_PER_WORKER = 8
OFF_TRACK_P = 0.25
OFF_TRACK_STREAK = 2
_GOAL_CHARS = 600
_PART_CHARS = 600

_RUN: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("run_course", default=None)
_PART: contextvars.ContextVar[str] = contextvars.ContextVar("run_course_part", default="")
# (session, subtask/role) → consecutive off-track checks.
_streaks: dict[tuple[str, str], int] = {}


def _flat(text: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def run_cap() -> int:
    try:
        return max(0, int(os.environ.get("FD_COURSE_MAX_CHECKS_PER_RUN", MAX_CHECKS_PER_RUN)))
    except ValueError:
        return MAX_CHECKS_PER_RUN


def worker_share(team_size: int, cap: int | None = None) -> int:
    """Each worker's checks: the run's cap split over the team, 1..8."""
    cap = run_cap() if cap is None else cap
    if cap <= 0:
        return 0
    return max(1, min(MAX_CHECKS_PER_WORKER, cap // max(1, int(team_size or 1))))


# ── the switch ───────────────────────────────────────────────────────


def caller_switch(request: Any) -> bool | None:
    """The calling agent's Keep on course switch (X-Agent-Auth), or None when
    the run wasn't started by an agent this deck knows."""
    try:
        from captain_claw.flight_deck import server

        token = request.headers.get("X-Agent-Auth", "") if request is not None else ""
        matched, _owner, slug = server._find_agent_by_auth(token)
        if not matched or not slug:
            return None
        kind = "process" if slug in server._load_process_registry() else "docker"
        agent_dir = server.DATA_DIR / slug
        if not agent_dir.is_dir():
            return None
        return bool(server._read_agent_course_check(agent_dir, kind))
    except Exception:
        return None


def resolve_switch(request: Any, cfg: dict[str, Any] | None) -> bool:
    """The run's switch: the starting agent's, else what the session kept
    (a continuation round), else on."""
    switch = caller_switch(request)
    if switch is not None:
        return switch
    kept = (cfg or {}).get("course_check")
    return kept if isinstance(kept, bool) else True


# ── run and part ─────────────────────────────────────────────────────


def start_run(goal: str, *, check: bool, team_size: int) -> None:
    """Bind the run's course for every dispatch made from here on."""
    _RUN.set({
        "goal": _flat(goal, _GOAL_CHARS),
        "check": bool(check),
        "max_checks": worker_share(team_size),
    })


def set_part(text: str) -> contextvars.Token:
    return _PART.set(_flat(text, _PART_CHARS))


def reset_part(token: contextvars.Token) -> None:
    try:
        _PART.reset(token)
    except ValueError:  # reset from another context: leave it
        pass


def subtask_part(st: dict[str, Any]) -> str:
    """A Vatra subtask as the worker's part."""
    title = str(st.get("title") or "").strip()
    desc = str(st.get("description") or st.get("desc") or st.get("brief") or "").strip()
    return f"{title}: {desc}" if title and desc else (title or desc)


def frame() -> dict[str, Any] | None:
    """The ``course`` field for a worker's chat frame, or None outside a run."""
    run = _RUN.get()
    if not run:
        return None
    out = dict(run)
    part = _PART.get()
    if part:
        out["part"] = part
    return out


# ── answers coming back ──────────────────────────────────────────────


def course_from_monitor(msg: dict[str, Any]) -> dict[str, Any] | None:
    """A worker's course check from its ``completion_gate step=jev_course``
    monitor event, or None for anything else."""
    if not isinstance(msg, dict):
        return None
    if str(msg.get("tool_name") or msg.get("tool") or "") != "completion_gate":
        return None
    args = msg.get("arguments")
    if not isinstance(args, dict) or args.get("step") != "jev_course":
        return None
    out = {k: args.get(k) for k in ("stage", "stage_name", "p_on_track", "next", "line", "at_step")
           if args.get(k) is not None}
    return out or None


def describe(course: dict[str, Any]) -> str:
    parts: list[str] = []
    stage, name = course.get("stage"), course.get("stage_name")
    if isinstance(stage, (int, float)) and name:
        parts.append(f"stage {min(5, max(1, int(round(stage))))} of 5 · {name}")
    p = course.get("p_on_track")
    if isinstance(p, (int, float)):
        parts.append(f"on track {p:.2f}")
    if course.get("next"):
        parts.append(f"next: {course['next']}")
    return " · ".join(parts) or "course checked"


def off_track(sid: str, key: str, course: dict[str, Any]) -> bool:
    """True once an owner is off track in two checks in a row (once per streak)."""
    p = course.get("p_on_track")
    k = (sid, key)
    if not isinstance(p, (int, float)) or p > OFF_TRACK_P:
        _streaks.pop(k, None)
        return False
    _streaks[k] = _streaks.get(k, 0) + 1
    return _streaks[k] == OFF_TRACK_STREAK


def forget_run(sid: str) -> None:
    for k in [k for k in _streaks if k[0] == sid]:
        _streaks.pop(k, None)
