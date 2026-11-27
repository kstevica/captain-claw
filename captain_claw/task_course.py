"""Task course — keep the person's request in front of the model during a long turn.

Weak models lose the request in the middle of the work: after a fold or a
compaction the opening message is summarised or far back, a "continue" turn's
request is that one word, and in a long tool chain twenty tool results bury
the line that said what the job was. The plan note only exists when a task
contract was made, and Jev's progress check only runs once the step budget is
used up.

So every session keeps a *task anchor* in ``session.metadata["task_course"]``:
the person's request in their own words (no model call), what they added
since, and their latest message. From the turn's second model call on (after
its first tool call) the anchor rides in the live block at the end of the
request (``agent_context_mixin._build_messages``), which is rebuilt on every
call and never saved into the session — a stale or wrong note can't stay in
the history.

Whether a new message starts a new task, continues the open one or adds to it
is decided here by rules that work in any language (length, "continue" words,
time since the last turn, word overlap); with Jev connected the course check
(:mod:`captain_claw.jev_course`) decides instead and calls :func:`redo_turn`.

docs/jev-course-plan.md §1. Off only through ``context.task_anchor: false``
(a benchmark arm); there is no UI switch.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable

KEY = "task_course"
CLOSED_KEY = "task_course_closed"

ANCHOR_CHARS = 600
AMEND_CHARS = 200
LATEST_CHARS = 200
MAX_AMENDMENTS = 3
MAX_REQUIREMENTS = 5
REQUIREMENT_CHARS = 160
MAX_CHECKS = 20
MAX_CLOSED = 10
# A message this short, soon after the last turn, belongs to the open task.
SHORT_WORDS = 12
GAP_SECONDS = 2 * 3600
# Below this share of shared content words a long message is a new task.
NEW_TASK_OVERLAP = 0.15

NEW_TASK = "new_task"
CONTINUES = "continues"
AMENDS = "amends"
NOT_A_TASK = "not_a_task"
RELATIONS = (NEW_TASK, CONTINUES, AMENDS, NOT_A_TASK)

_LONGER = " (the full request is earlier in the conversation)"

# A whole message that only says "go on", in the languages the agents are used in.
_CONTINUE_WORDS = frozenset({
    "continue", "go on", "carry on", "keep going", "keep on", "proceed", "resume",
    "go ahead", "next", "more", "finish it", "finish", "do it", "yes continue",
    "nastavi", "nastavite", "nastavi dalje", "dalje", "idemo dalje", "samo naprijed",
    "samo nastavi", "ajde", "ajmo", "naprijed", "nastavak", "završi", "zavrsi",
    "weiter", "mach weiter", "weitermachen", "fortfahren",
    "sigue", "continúa", "continua", "adelante", "continuez", "vas-y", "vai avanti",
    "продолжай", "дальше", "继续",
})
# A whole message that only thanks or acknowledges.
_THANKS_WORDS = frozenset({
    "thanks", "thank you", "thx", "ty", "ok", "okay", "ok thanks", "cool", "great",
    "perfect", "nice", "good", "got it", "awesome", "super", "hvala", "hvala ti",
    "ok hvala", "odlično", "odlicno", "fala", "danke", "gracias", "merci", "grazie",
    "👍", "🙏", "spasibo", "спасибо",
})
_FILLER = frozenset({"please", "pls", "molim", "molim te", "bitte", "por favor", "ok", "okay"})
_PUNCT_RE = re.compile(r"[\s.,!?;:…\-–—\"'()]+")
_WORD_RE = re.compile(r"[^\W\d_]{4,}", re.UNICODE)


def enabled() -> bool:
    """``context.task_anchor`` (default on)."""
    try:
        from captain_claw.config import get_config

        return bool(getattr(get_config().context, "task_anchor", True))
    except Exception:
        return True


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        dt = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _cut(text: str, limit: int, tail: str = "…") -> str:
    return text if len(text) <= limit else text[: max(0, limit - len(tail))].rstrip() + tail


def person_words(text: Any) -> str:
    """The person's own words: a bridge's surface rules block stripped,
    whitespace collapsed."""
    raw = str(text or "")
    try:
        from captain_claw import msg_origin

        _block, raw = msg_origin.split_surface_block(raw)
    except Exception:
        pass
    return re.sub(r"\s+", " ", raw).strip()


def _bare(text: str) -> str:
    """Lower-cased, punctuation and polite filler stripped from both ends."""
    words = _PUNCT_RE.sub(" ", text.lower()).split()
    while words and words[0] in _FILLER:
        words.pop(0)
    while words and words[-1] in _FILLER:
        words.pop()
    return " ".join(words) or text.strip().lower()


def is_continue(text: str) -> bool:
    return _bare(text) in _CONTINUE_WORDS


def is_thanks(text: str) -> bool:
    return _bare(text) in _THANKS_WORDS


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "")}


def overlap(a: str, b: str) -> float:
    """Share of the smaller side's content words found in the other side."""
    wa, wb = _content_words(a), _content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def is_open(course: Any) -> bool:
    return isinstance(course, dict) and course.get("status") == "open" and bool(course.get("anchor"))


@dataclass
class PreviousTurn:
    """How the conversation stood before the new message."""

    end_at: datetime | None = None
    answered: bool = False   # the last turn ended on an assistant reply
    asked: bool = False      # ... and that reply asked the person something


def previous_turn(messages: list[dict[str, Any]]) -> PreviousTurn:
    """Read the end of the session (before the new user message)."""
    try:
        from captain_claw.msg_origin import is_monitor_only
    except Exception:  # pragma: no cover — import guard
        def is_monitor_only(_msg: Any) -> bool:
            return False
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or is_monitor_only(msg):
            continue
        role = str(msg.get("role", "")).lower()
        if role == "assistant":
            text = str(msg.get("content", "") or "").strip()
            if not text:
                return PreviousTurn(end_at=parse_time(msg.get("timestamp")))
            return PreviousTurn(
                end_at=parse_time(msg.get("timestamp")),
                answered=True,
                asked=text.rstrip(" )*_").endswith("?"),
            )
        if role in ("user", "tool"):
            return PreviousTurn(end_at=parse_time(msg.get("timestamp")))
    return PreviousTurn()


def guess_relation(
    message: str,
    course: Any,
    *,
    prev: PreviousTurn | None = None,
    now: datetime | None = None,
    topic_switch: Callable[[str, str], bool] | None = None,
) -> str:
    """New task, the same one, an addition, or not a task — by rules only."""
    words = person_words(message)
    if not words:
        return NOT_A_TASK
    if not is_open(course):
        return NOT_A_TASK if is_thanks(words) else NEW_TASK
    if is_continue(words):
        return CONTINUES
    if is_thanks(words):
        return NOT_A_TASK
    prev = prev or PreviousTurn()
    now = now or _now()
    if bool(course.get("automated")):
        return NEW_TASK
    gap = (now - prev.end_at).total_seconds() if prev.end_at else 0.0
    if prev.answered and gap > GAP_SECONDS:
        return NEW_TASK
    anchor = str(course.get("anchor", ""))
    known = " ".join([anchor, *[str(a) for a in course.get("amendments") or []]])
    switched = False
    if topic_switch is not None:
        try:
            switched = bool(topic_switch(words, anchor))
        except Exception:
            switched = False
    if len(words.split()) <= SHORT_WORDS and gap <= GAP_SECONDS:
        if switched:
            return NEW_TASK
        return CONTINUES if prev.asked else AMENDS
    if switched or overlap(words, known) < NEW_TASK_OVERLAP:
        return NEW_TASK
    return AMENDS


def _summary(course: dict[str, Any], status: str) -> dict[str, Any]:
    checks = [c for c in course.get("checks") or [] if isinstance(c, dict)]
    stages = [c.get("stage") for c in checks if isinstance(c.get("stage"), (int, float))]
    out = {
        "id": course.get("id", ""),
        "anchor": _cut(str(course.get("anchor", "")), 200),
        "status": status,
        "opened_at": course.get("opened_at", ""),
        "closed_at": _iso(_now()),
        "turns": int(course.get("turns", 0) or 0),
        "checks": len(checks),
        "final_stage": stages[-1] if stages else None,
        "corrections_shown": sum(1 for c in checks if c.get("line")),
    }
    for key in ("turn_stats", "followed"):
        if course.get(key):
            out[key] = course[key]
    return out


def _new_id(now: datetime) -> str:
    import secrets

    return f"tc_{now.strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(3)}"


@dataclass
class TurnStart:
    """What :func:`begin_turn` did, so a later, better decision can redo it."""

    relation: str
    message: str
    automated: bool = False
    turn_msg: int | None = None
    before: dict[str, Any] = field(default_factory=dict)


def begin_turn(
    metadata: dict[str, Any],
    message: Any,
    *,
    relation: str | None = None,
    prev: PreviousTurn | None = None,
    now: datetime | None = None,
    topic_switch: Callable[[str, str], bool] | None = None,
    automated: bool = False,
    turn_msg: int | None = None,
) -> TurnStart:
    """Update the session's anchor for a new turn's message. Never raises
    on odd metadata; returns what it decided."""
    now = now or _now()
    words = person_words(message)
    before = {
        KEY: copy.deepcopy(metadata.get(KEY)),
        CLOSED_KEY: copy.deepcopy(metadata.get(CLOSED_KEY)),
    }
    course = metadata.get(KEY) if isinstance(metadata.get(KEY), dict) else None
    if not words:
        return TurnStart(NOT_A_TASK, words, automated, turn_msg, before)
    if relation not in RELATIONS:
        relation = NEW_TASK if automated else guess_relation(
            words, course, prev=prev, now=now, topic_switch=topic_switch,
        )
    if relation in (CONTINUES, AMENDS, NOT_A_TASK) and not is_open(course):
        relation = NOT_A_TASK if relation == NOT_A_TASK else NEW_TASK
    if relation == NEW_TASK:
        if is_open(course):
            prev = prev or PreviousTurn()
            closed = metadata.get(CLOSED_KEY)
            closed = [c for c in closed if isinstance(c, dict)] if isinstance(closed, list) else []
            closed.append(_summary(course, "done" if prev.answered else "dropped"))
            metadata[CLOSED_KEY] = closed[-MAX_CLOSED:]
        anchor = _cut(words, ANCHOR_CHARS, "…" + _LONGER) if len(words) > ANCHOR_CHARS else words
        course = {
            "id": _new_id(now),
            "anchor": anchor,
            "amendments": [],
            "latest": "",
            "requirements": [],
            "opened_at": _iso(now),
            "updated_at": _iso(now),
            "status": "open",
            "turns": 1,
            "checks": [],
        }
        if automated:
            course["automated"] = True
        if turn_msg is not None:
            course["source"] = {"turn_msg": turn_msg}
        metadata[KEY] = course
    elif relation in (CONTINUES, AMENDS):
        assert course is not None
        if relation == AMENDS:
            amendments = [str(a) for a in course.get("amendments") or []]
            amendments.append(_cut(words, AMEND_CHARS))
            course["amendments"] = amendments[-MAX_AMENDMENTS:]
        course["latest"] = "" if is_continue(words) else _cut(words, LATEST_CHARS)
        course["turns"] = int(course.get("turns", 0) or 0) + 1
        course["updated_at"] = _iso(now)
        metadata[KEY] = course
    return TurnStart(relation, words, automated, turn_msg, before)


def redo_turn(metadata: dict[str, Any], start: TurnStart, relation: str) -> TurnStart:
    """Undo what :func:`begin_turn` did for this turn and decide again with
    ``relation`` (Jev's answer). A no-op when it agrees."""
    if relation not in RELATIONS or relation == start.relation:
        return start
    current = metadata.get(KEY) if isinstance(metadata.get(KEY), dict) else {}
    kept_checks = list(current.get("checks") or []) if relation != NEW_TASK else []
    for key, value in start.before.items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = copy.deepcopy(value)
    redone = begin_turn(
        metadata, start.message, relation=relation, automated=start.automated,
        turn_msg=start.turn_msg,
    )
    redone.before = start.before
    course = metadata.get(KEY)
    if kept_checks and isinstance(course, dict):
        course["checks"] = kept_checks[-MAX_CHECKS:]
    return redone


def current(metadata: Any) -> dict[str, Any] | None:
    course = metadata.get(KEY) if isinstance(metadata, dict) else None
    return course if is_open(course) else None


def set_requirements(metadata: dict[str, Any], requirements: Any) -> None:
    """The task contract's requirements, short, on the open anchor."""
    course = current(metadata)
    if course is None or not isinstance(requirements, list):
        return
    out: list[str] = []
    for req in requirements:
        if isinstance(req, dict):
            text = req.get("title") or req.get("text") or req.get("description") or ""
        else:
            text = req
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if text:
            out.append(_cut(text, REQUIREMENT_CHARS))
        if len(out) >= MAX_REQUIREMENTS:
            break
    course["requirements"] = out


def record_check(metadata: dict[str, Any], check: dict[str, Any]) -> None:
    """Keep a course check (jev_course) on the open anchor, newest 20."""
    course = current(metadata)
    if course is None or not isinstance(check, dict):
        return
    checks = [c for c in course.get("checks") or [] if isinstance(c, dict)]
    checks.append(check)
    course["checks"] = checks[-MAX_CHECKS:]


def note(course: Any, live: dict[str, Any] | None = None) -> str:
    """The note the model reads. ``live`` adds the course check's lines
    (``stage`` and ``line``)."""
    if not is_open(course):
        return ""
    lines: list[str] = []
    goal = str(course.get("goal", "") or "").strip()
    anchor = str(course.get("anchor", "")).strip()
    if goal:
        lines.append(f"Run goal: {goal}")
        lines.append(f"Your part: {anchor}")
    else:
        lines.append(f"Task (the person's request): {anchor}")
    amendments = [str(a).strip() for a in course.get("amendments") or [] if str(a).strip()]
    if amendments:
        lines.append("Added since: " + "; ".join(amendments))
    latest = str(course.get("latest", "") or "").strip()
    if latest and latest not in amendments and latest != anchor:
        lines.append(f"Their latest message: {latest}")
    if isinstance(live, dict):
        for key in ("stage", "line"):
            text = str(live.get(key, "") or "").strip()
            if text:
                lines.append(text)
    return "\n".join(lines)


def turn_made_tool_call(turn_messages: list[dict[str, Any]]) -> bool:
    from captain_claw.msg_origin import is_monitor_only

    return any(
        isinstance(m, dict) and str(m.get("role", "")).lower() == "tool" and not is_monitor_only(m)
        for m in turn_messages or []
    )


def note_for_turn(
    metadata: Any,
    turn_messages: list[dict[str, Any]],
    live: dict[str, Any] | None = None,
) -> str:
    """The note for this model call: nothing on a turn's first call (no
    tool call yet), nothing without an open anchor or with the anchor off."""
    if not enabled() or not turn_made_tool_call(turn_messages):
        return ""
    return note(current(metadata), live)
