"""Course check — Jev measures where a long turn stands against the task anchor.

The task anchor (:mod:`captain_claw.task_course`) keeps the person's request in
front of the model on every call. This module adds a measurement on top, when
the owner has Jev (TypeSafe, :mod:`captain_claw.jev_client`) and the agent's
switch is on:

* **At turn start** one ``choice`` question decides whether the new message is
  a new task, continues the open one, adds to it, or is not a task (thanks,
  ok). It replaces the rule-based guess when Jev is confident.
* **During a long turn** — every :data:`EVERY_STEPS` tool steps, right after a
  fold or compaction, and after :data:`FAIL_STREAK` failed tool results in a
  row — one call asks three questions: the stage (orienting → delivered),
  whether the recent steps serve the anchor, and what should come next.

Every call runs in the background while the next model call generates, so it
adds almost no waiting; the answer applies from the call after. A result is
turned into the note's ``Stage:`` line and, only when Jev is confident, one
fixed corrective line (refocus / finish / ask the person) — Jev classifies, it
never writes text the model reads. Both live in the live block and are never
saved into the session.

Never in the way: off, no key, a shared-agent member's turn, a timeout or any
error means no check, and the anchor note goes on as before. The switch is
``~/.captain-claw/course_check.txt`` ("on"/"off", written by Flight Deck's
agent Settings), else ``tools.jev.course_check`` (default on), so it is on by
default wherever Jev is connected. docs/jev-course-plan.md §2.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from captain_claw import jev_client, task_course
from captain_claw.jev_progress import _flat, _step_lines
from captain_claw.logging import get_logger

log = get_logger(__name__)

FILENAME = "course_check.txt"
SETTING_PATH = Path("~/.captain-claw").expanduser() / FILENAME

# Cadence and caps (tool steps = loop iterations that ran tool calls).
EVERY_STEPS = 6
FAIL_STREAK = 3
MAX_CHECKS_PER_TURN = 8
MAX_LINES_PER_TURN = 2
STALE_STEPS = 6          # a result older than this when it lands is dropped
REUSE_STEPS = 3          # the step-budget judge reuses a check this fresh
TIMEOUT_SECONDS = 15.0

# Thresholds.
RELATION_CONFIDENCE = 0.6
REFOCUS_ON_TRACK_MAX = 0.25
REFOCUS_CONFIDENCE = 0.6
FINISH_STAGE_MIN = 4.5
FINISH_CONFIDENCE = 0.7
ASK_CONFIDENCE = 0.7
EXTEND_ON_TRACK_MIN = 0.6

_REQUEST_CHARS = 1500
_STEPS_CHARS = 9000

STAGES = ("orienting", "gathering", "producing", "checking", "delivered")

Q_RELATION = "task_relation"
Q_STAGE = "stage"
Q_ON_TRACK = "on_track"
Q_NEXT = "next"

NEXT_CONTINUE = "continue"
NEXT_REFOCUS = "refocus"
NEXT_FINISH = "finish"
NEXT_ASK = "ask_person"

LINE_REFOCUS = "refocus"
LINE_FINISH = "finish"
LINE_ASK = "ask_person"
LINES = {
    LINE_REFOCUS: (
        "Your recent steps don't look like they serve this task. Re-read the task "
        "above and take the step it needs next."
    ),
    LINE_FINISH: "The task looks done. If the result is ready, give the person your answer now.",
    LINE_ASK: (
        "If you need something only the person can give (a login, a choice, missing "
        "information), ask them now."
    ),
}

RELATION_QUESTIONS: dict[str, dict[str, Any]] = {
    Q_RELATION: {
        "type": "choice",
        "instructions": (
            "An AI agent was working on the open task below for a person. The person has "
            "just sent a new message. How does the new message relate to the open task?"
        ),
        "options": {
            task_course.NEW_TASK: "A different job: the person moved on to something else.",
            task_course.CONTINUES: (
                "The same job with nothing new: go on, an answer to the agent's question, "
                "a confirmation, a nudge to keep working."
            ),
            task_course.AMENDS: (
                "The same job with a change or an addition: another item, a correction, "
                "a new constraint or detail."
            ),
            task_course.NOT_A_TASK: "Not a request at all: thanks, ok, a greeting, small talk.",
        },
    },
}

CHECK_QUESTIONS: dict[str, dict[str, Any]] = {
    Q_STAGE: {
        "type": "choice",
        "instructions": (
            "An AI agent is working on the person's task. Judging by its steps so far, "
            "which stage of the work is it at?"
        ),
        "options": {
            "orienting": "Still working out what is asked or where to start; little done yet.",
            "gathering": "Collecting what it needs: reading, searching, fetching, inspecting.",
            "producing": "Making the result: writing, building, editing, computing.",
            "checking": "The result exists; it is verifying, testing or fixing it.",
            "delivered": "The result is complete and handed over, or ready to hand over.",
        },
    },
    Q_ON_TRACK: {
        "type": "noul",
        "instructions": (
            "Are the agent's most recent steps serving the person's task as stated "
            "(including what they added since)?"
        ),
        "yes_means": (
            "The recent steps work toward what the task asks: they gather what it needs, "
            "produce or check its result, or take a needed detour (reading docs, fixing a "
            "tool problem) to get there."
        ),
        "no_means": (
            "The recent steps drifted: a side quest the task doesn't need, answering a "
            "different question, redoing work that is already done, or polishing "
            "something nobody asked for."
        ),
    },
    Q_NEXT: {
        "type": "choice",
        "instructions": "What should the agent do next?",
        "options": {
            NEXT_CONTINUE: "Keep going as it is: the work is moving toward the task.",
            NEXT_REFOCUS: "Stop the current line of work and return to what the task asks.",
            NEXT_FINISH: "Stop working and give the person the result: the task is done.",
            NEXT_ASK: (
                "Ask the person: it needs something only they can give (a login, a "
                "choice, missing information) or is blocked."
            ),
        },
    },
}


# ── the switch ───────────────────────────────────────────────────────


def parse(text: object) -> bool | None:
    from captain_claw import task_rephrase

    return task_rephrase.parse(text)


def render(enabled: bool) -> str:
    return "on" if enabled else "off"


def config_enabled() -> bool:
    try:
        from captain_claw.config import get_config

        return bool(getattr(get_config().tools.jev, "course_check", True))
    except Exception:
        return True


def enabled() -> bool:
    """The agent's switch: the switch file, else ``tools.jev.course_check``."""
    try:
        switch = parse(SETTING_PATH.read_text(encoding="utf-8"))
    except OSError:
        switch = None
    return switch if switch is not None else config_enabled()


def clear() -> None:
    try:
        SETTING_PATH.unlink()
    except OSError:
        pass


async def ready(switch: bool | None = None) -> bool:
    """On (``switch``, else the agent's own switch), not a shared-agent
    member's turn, and a usable Jev key. Never raises."""
    if not (enabled() if switch is None else switch):
        return False
    try:
        if jev_client._member_call():
            return False
        return bool(await jev_client.refresh_connected())
    except Exception as exc:
        log.debug("Jev course check unavailable", error=str(exc) or type(exc).__name__)
        return False


async def _ask(state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
    if jev_client.under_flight_deck():
        return await jev_client.ask_via_flight_deck(state, questions, timeout=TIMEOUT_SECONDS)
    return await jev_client.standalone_api().ask(state, questions)


# ── reading answers ──────────────────────────────────────────────────


def _answer(resp: Any, name: str) -> dict[str, Any]:
    answers = resp.get("answers") if isinstance(resp, dict) else None
    a = answers.get(name) if isinstance(answers, dict) else None
    return a if isinstance(a, dict) else {}


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _choice(resp: Any, name: str) -> tuple[str | None, float | None, dict[str, float]]:
    a = _answer(resp, name)
    choice = a.get("choice")
    probs_raw = a.get("probabilities") if isinstance(a.get("probabilities"), dict) else {}
    probs = {str(k): p for k, v in probs_raw.items() if (p := _num(v)) is not None}
    conf = _num(a.get("confidence"))
    if conf is None and isinstance(choice, str):
        conf = probs.get(choice)
    return (str(choice) if isinstance(choice, str) and choice else None), conf, probs


def _expected_stage(choice: str | None, probs: dict[str, float]) -> float | None:
    """1..5: the probability-weighted stage, else the chosen one."""
    total = sum(probs.get(s, 0.0) for s in STAGES)
    if total > 0:
        return round(sum((i + 1) * probs.get(s, 0.0) for i, s in enumerate(STAGES)) / total, 2)
    if choice in STAGES:
        return float(STAGES.index(choice) + 1)
    return None


@dataclass
class Check:
    """One course check's answers."""

    step: int
    stage: float | None = None
    stage_name: str | None = None
    on_track: float | None = None
    next: str | None = None
    next_confidence: float | None = None
    latency_ms: int = 0
    trigger: str = ""
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def record(self, line: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {
            "step": self.step, "at": self.at, "trigger": self.trigger,
            "stage": self.stage, "stage_name": self.stage_name,
            "p_on_track": None if self.on_track is None else round(self.on_track, 3),
            "next": self.next,
            "next_confidence": None if self.next_confidence is None else round(self.next_confidence, 3),
            "latency_ms": self.latency_ms,
        }
        if line:
            out["line"] = line
        return out


def parse_check(resp: Any, *, step: int, latency_ms: int = 0, trigger: str = "") -> Check | None:
    stage_choice, _conf, stage_probs = _choice(resp, Q_STAGE)
    on_track = _num(_answer(resp, Q_ON_TRACK).get("noul"))
    nxt, nxt_conf, _ = _choice(resp, Q_NEXT)
    stage = _expected_stage(stage_choice, stage_probs)
    if stage is None and on_track is None and nxt is None:
        return None
    name = STAGES[min(4, max(0, int(round(stage)) - 1))] if stage is not None else stage_choice
    return Check(
        step=step, stage=stage, stage_name=name, on_track=on_track, next=nxt,
        next_confidence=nxt_conf, latency_ms=latency_ms, trigger=trigger,
    )


def corrective(check: Check, shown: list[str]) -> str:
    """Which fixed line (if any) this check earns: at most two per turn,
    never the same one twice in a row."""
    if len(shown) >= MAX_LINES_PER_TURN:
        return ""
    conf = check.next_confidence or 0.0
    key = ""
    if (check.on_track is not None and check.on_track <= REFOCUS_ON_TRACK_MAX
            and check.next == NEXT_REFOCUS and conf >= REFOCUS_CONFIDENCE):
        key = LINE_REFOCUS
    elif (check.stage is not None and check.stage >= FINISH_STAGE_MIN
            and check.next == NEXT_FINISH and conf >= FINISH_CONFIDENCE):
        key = LINE_FINISH
    elif check.next == NEXT_ASK and conf >= ASK_CONFIDENCE:
        key = LINE_ASK
    if key and shown and shown[-1] == key:
        return ""
    return key


def stage_text(check: Check, steps_ago: int) -> str:
    if check.stage is None or not check.stage_name:
        return ""
    n = min(5, max(1, int(round(check.stage))))
    ago = "just checked" if steps_ago <= 0 else (
        "checked 1 step ago" if steps_ago == 1 else f"checked {steps_ago} steps ago")
    return f"Stage: {check.stage_name} ({n} of 5), {ago}"


def status_text(check: Check) -> str:
    """The person-facing status line."""
    if check.stage is None or not check.stage_name:
        return ""
    n = min(5, max(1, int(round(check.stage))))
    return f"Stage {n} of 5 · {check.stage_name}"


# ── state Jev reads ──────────────────────────────────────────────────


def _cut_steps(lines: list[str]) -> str:
    kept: list[str] = []
    size = 0
    for line in reversed(lines):
        if kept and size + len(line) + 1 > _STEPS_CHARS:
            break
        kept.append(line)
        size += len(line) + 1
    kept.reverse()
    cut = len(lines) - len(kept)
    steps = "\n".join(kept)
    return f"[… {cut} earlier steps cut …]\n{steps}" if cut else steps


def check_state(course: dict[str, Any], turn_messages: list[dict[str, Any]], *, steps_used: int) -> dict[str, Any]:
    state: dict[str, Any] = {}
    if course.get("goal"):
        state["run_goal"] = _flat(course.get("goal"), _REQUEST_CHARS)
        state["task"] = _flat(course.get("anchor"), _REQUEST_CHARS)
    else:
        state["task"] = _flat(course.get("anchor"), _REQUEST_CHARS)
    added = [str(a) for a in course.get("amendments") or [] if str(a).strip()]
    if added:
        state["added_since"] = added
    if course.get("latest"):
        state["latest_message"] = _flat(course.get("latest"), 400)
    if course.get("requirements"):
        state["requirements"] = list(course["requirements"])
    state["steps_used"] = steps_used
    state["recent_steps"] = _cut_steps(_step_lines(turn_messages)) or "(no steps recorded)"
    return state


def relation_state(course_before: dict[str, Any], message: str, last_reply: str = "") -> dict[str, Any]:
    state: dict[str, Any] = {"open_task": _flat(course_before.get("anchor"), _REQUEST_CHARS)}
    added = [str(a) for a in course_before.get("amendments") or [] if str(a).strip()]
    if added:
        state["added_since"] = added
    if last_reply:
        state["agent_last_reply"] = _flat(last_reply, 600)
    state["new_message"] = _flat(message, _REQUEST_CHARS)
    return state


async def ask_relation(state: dict[str, Any]) -> tuple[str | None, float | None]:
    """Jev's relation and its confidence, or (None, None). Never raises."""
    if not await ready():
        return None, None
    try:
        resp = await asyncio.wait_for(_ask(state, RELATION_QUESTIONS), TIMEOUT_SECONDS + 3)
    except Exception as exc:
        log.debug("Jev course relation skipped", error=str(exc) or type(exc).__name__)
        return None, None
    choice, conf, _ = _choice(resp, Q_RELATION)
    if choice not in task_course.RELATIONS:
        return None, None
    return choice, conf


async def ask_check(state: dict[str, Any], *, step: int, trigger: str) -> Check | None:
    """One in-turn course check, or None. Never raises."""
    if not await ready():
        return None
    started = time.monotonic()
    try:
        resp = await asyncio.wait_for(_ask(state, CHECK_QUESTIONS), TIMEOUT_SECONDS + 3)
    except Exception as exc:
        log.debug("Jev course check skipped", error=str(exc) or type(exc).__name__)
        return None
    check = parse_check(resp, step=step, latency_ms=int((time.monotonic() - started) * 1000), trigger=trigger)
    if check is not None:
        log.info("Jev course check", step=step, stage=check.stage, p_on_track=check.on_track,
                 next=check.next, trigger=trigger)
    return check


# ── per-turn bookkeeping ─────────────────────────────────────────────


@dataclass
class TurnWatch:
    """A turn's course checks: when one is due, the one in flight, the
    latest answer and the lines shown so far."""

    max_checks: int = MAX_CHECKS_PER_TURN
    tool_steps: int = 0
    last_check_step: int = 0
    checks_started: int = 0
    fail_streak: int = 0
    folded: bool = False
    disabled: bool = False
    task: asyncio.Task | None = None
    task_step: int = 0
    latest: Check | None = None
    lines_shown: list[str] = field(default_factory=list)
    relation_task: asyncio.Task | None = None
    relation_result: tuple[Any, ...] | None = None
    scan_idx: int | None = None   # session index already counted for failures
    tools_seen: int = 0           # the loop's tool-call count at the last tick

    def note_tool_step(self, failures: int, successes: int) -> None:
        self.tool_steps += 1
        if successes:
            self.fail_streak = failures
        else:
            self.fail_streak += failures

    def due(self) -> str:
        """Why a check should start now ("" when it shouldn't)."""
        if self.disabled or self.task is not None or self.checks_started >= self.max_checks:
            return ""
        if self.tool_steps == 0 or self.tool_steps == self.last_check_step:
            return ""
        if self.folded:
            return "fold"
        if self.fail_streak >= FAIL_STREAK:
            return "failures"
        if self.tool_steps >= EVERY_STEPS and self.tool_steps - self.last_check_step >= EVERY_STEPS:
            return "steps"
        return ""

    def started(self, task: asyncio.Task) -> None:
        self.task = task
        self.task_step = self.tool_steps
        self.last_check_step = self.tool_steps
        self.checks_started += 1
        self.folded = False
        self.fail_streak = 0

    def collect(self) -> Check | None:
        """The finished check's answer, once; None while it runs or when it
        came back empty or stale."""
        task = self.task
        if task is None or not task.done():
            return None
        self.task = None
        try:
            check = task.result()
        except BaseException:  # cancelled or failed: no check this time
            return None
        if not isinstance(check, Check):
            return None
        if self.tool_steps - check.step > STALE_STEPS:
            return None
        self.latest = check
        return check

    def fresh(self) -> Check | None:
        """The latest check when it is at most REUSE_STEPS old."""
        if self.latest is None or self.tool_steps - self.latest.step > REUSE_STEPS:
            return None
        return self.latest

    def cancel(self) -> None:
        """Stop the in-turn check at the end of a turn. The relation check is
        left to finish: its answer still fixes the anchor for the next turn
        (``task_course.redo_turn`` skips it if that turn already began)."""
        if self.task is not None and not self.task.done():
            self.task.cancel()
        self.task = None


def budget_verdict(check: Check) -> bool:
    """Does a fresh check say the turn should get more steps?"""
    return (
        check.on_track is not None and check.on_track >= EXTEND_ON_TRACK_MIN
        and check.next == NEXT_CONTINUE
    )


def failures_in(messages: list[dict[str, Any]]) -> tuple[int, int]:
    """(failed, ok) tool results among *messages*."""
    from captain_claw.msg_origin import is_monitor_only

    failed = ok = 0
    for msg in messages:
        if not isinstance(msg, dict) or str(msg.get("role", "")).lower() != "tool" or is_monitor_only(msg):
            continue
        text = str(msg.get("content", "") or "").strip().lower()
        if text.startswith(("error", "❌")):
            failed += 1
        else:
            ok += 1
    return failed, ok
