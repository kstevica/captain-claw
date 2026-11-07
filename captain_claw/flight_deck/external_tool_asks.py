"""Custom tools, Phase 7d: a tool asks the person running it.

A tool that declares ``"asks": true`` can stop and ask its runner a question,
or a set of them (``captain.ask``), and carries on with the answers. Flight Deck
handles it directly: the question opens in every Flight Deck tab of that
person (each tab waits on ``GET /fd/external-tools/asks``) and as a Needs-you
item in Line and the bell. Answered in one tab, it closes in the others.

While a question is open the tool waits on its line, doing nothing, and its
time limit stops (``external_tools.RunClock``); the question's own wait applies
instead. A run nobody is watching (a schedule, autonomous work) gets "no
answer" at once. Nothing here is kept after the answer: the questions and
answers are the run's, in the runner's audit like the rest of it.
"""

from __future__ import annotations

import asyncio
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

MAX_ASKS_PER_RUN = 5
MAX_QUESTIONS = 10
MAX_QUESTION = 500
MAX_HELP = 500
MAX_CHOICES = 50
MAX_CHOICE = 200
MAX_ANSWER = 4000
MAX_LINES = 20
DEFAULT_WAIT_S = 600.0
MIN_WAIT_S = 5.0
MAX_WAIT_S = 3600.0
BELL_REF = "custom_tool_ask"

# Turn sources (turn_source.py) with nobody there to answer.
UNATTENDED = frozenset({"scheduled", "flow", "agent", "autonomy", "taskboard", "api", "link", "exam"})

_ID_RE = re.compile(r"^[a-z0-9_]{1,40}$")


class AskError(ValueError):
    """Questions (or answers) Flight Deck refuses; the message says why."""


def attended(origin: str) -> bool:
    """Is someone there to answer a run that came from *origin* (a turn source)?"""
    return str(origin or "").split(":", 1)[0].strip().lower() not in UNATTENDED


def _text(value: Any, limit: int, what: str, *, required: bool = True) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()) or len(value) > limit:
        raise AskError(f"{what} must be text of at most {limit} characters.")
    return value.strip()


def check_questions(raw: Any) -> list[dict[str, Any]]:
    """A tool's questions, checked and put in one shape. Raises :class:`AskError`."""
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_QUESTIONS:
        raise AskError(f"Ask 1–{MAX_QUESTIONS} questions at a time.")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, q in enumerate(raw, 1):
        if not isinstance(q, dict):
            raise AskError(f"Question {i} must be an object.")
        qid = str(q.get("id") or ("answer" if len(raw) == 1 else "")).strip().lower()
        if not _ID_RE.match(qid) or qid in seen:
            raise AskError(f"Question {i} needs its own 'id' (lowercase letters, digits, _).")
        seen.add(qid)
        item: dict[str, Any] = {"id": qid, "question": _text(q.get("question"), MAX_QUESTION, f"Question {i}")}
        if q.get("help") is not None:
            item["help"] = _text(q.get("help"), MAX_HELP, f"Question {i}'s help", required=False)
        item["optional"] = bool(q.get("optional"))
        choices = q.get("choices")
        if choices is not None:
            if (not isinstance(choices, list) or not 1 <= len(choices) <= MAX_CHOICES
                    or not all(isinstance(c, str) and 1 <= len(c.strip()) <= MAX_CHOICE for c in choices)):
                raise AskError(f"Question {i}: 'choices' lists 1–{MAX_CHOICES} options of at most {MAX_CHOICE} characters.")
            choices = list(dict.fromkeys(c.strip() for c in choices))
            item["choices"] = choices
            if q.get("multiple"):
                item["kind"] = "multiple"
                low = q.get("min", 0 if item["optional"] else 1)
                high = q.get("max", len(choices))
                if (isinstance(low, bool) or isinstance(high, bool) or not isinstance(low, int)
                        or not isinstance(high, int) or not 0 <= low <= high <= len(choices) or high < 1):
                    raise AskError(f"Question {i}: 'min' and 'max' must fit its choices.")
                item["min"], item["max"] = low, high
            else:
                item["kind"] = "choice"
        elif q.get("confirm"):
            item["kind"] = "confirm"
        else:
            item["kind"] = "text"
            lines = q.get("lines", 1)
            if isinstance(lines, bool) or not isinstance(lines, int) or not 1 <= lines <= MAX_LINES:
                raise AskError(f"Question {i}: 'lines' is 1–{MAX_LINES}.")
            item["lines"] = lines
        if "default" in q:
            try:
                item["default"] = _one_answer(item, q["default"])
            except AskError:
                raise AskError(f"Question {i}: its 'default' isn't an answer it takes.") from None
        out.append(item)
    return out


def _one_answer(q: dict[str, Any], value: Any) -> Any:
    kind = q["kind"]
    if kind == "text":
        if not isinstance(value, str) or len(value) > MAX_ANSWER:
            raise AskError(f"“{q['question']}” takes text of at most {MAX_ANSWER} characters.")
        if not value.strip() and not q.get("optional"):
            raise AskError(f"“{q['question']}” needs an answer.")
        return value
    if kind == "choice":
        if value not in q["choices"]:
            raise AskError(f"“{q['question']}”: pick one of its choices.")
        return value
    if kind == "multiple":
        if not isinstance(value, list) or len(set(map(str, value))) != len(value) or any(v not in q["choices"] for v in value):
            raise AskError(f"“{q['question']}”: pick from its choices.")
        if not q["min"] <= len(value) <= q["max"]:
            raise AskError(f"“{q['question']}”: pick {q['min']}–{q['max']}.")
        return [c for c in q["choices"] if c in value]
    if not isinstance(value, bool):
        raise AskError(f"“{q['question']}”: answer yes or no.")
    return value


def check_answers(questions: list[dict[str, Any]], answers: Any) -> dict[str, Any]:
    """The person's answers, checked against the questions. Raises :class:`AskError`."""
    if not isinstance(answers, dict):
        raise AskError("Answers come as an object, one per question.")
    out: dict[str, Any] = {}
    for q in questions:
        if q["id"] not in answers or answers[q["id"]] is None:
            if not q.get("optional"):
                raise AskError(f"“{q['question']}” needs an answer.")
            out[q["id"]] = q.get("default", {"text": "", "multiple": []}.get(q["kind"]))
            continue
        out[q["id"]] = _one_answer(q, answers[q["id"]])
    return out


@dataclass
class Ask:
    id: str
    user_id: str
    call_id: str
    title: str
    owner_name: str           # the tool's maker, when it isn't the person's own
    agent_name: str
    label: str                # "csv-total, used by bank-report" for an inner tool
    questions: list[dict[str, Any]]
    created: float
    deadline: float           # time.time()
    state: str = "open"       # open | answered | declined | expired | closed
    answers: dict[str, Any] = field(default_factory=dict)
    event: asyncio.Event = field(default_factory=asyncio.Event)

    def view(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "owner_name": self.owner_name, "agent_name": self.agent_name,
                "label": self.label, "questions": self.questions, "created": self.created,
                "seconds_left": max(0, int(self.deadline - time.time()))}


_ASKS: dict[str, Ask] = {}
_CHANGE: dict[str, asyncio.Event] = {}


def reset_for_tests() -> None:
    _ASKS.clear()
    _CHANGE.clear()


def _changed(user_id: str) -> None:
    ev = _CHANGE.pop(user_id, None)
    if ev is not None:
        ev.set()


def open_for(user_id: str) -> list[dict[str, Any]]:
    """The person's open questions, oldest first."""
    return [a.view() for a in sorted(_ASKS.values(), key=lambda a: a.created)
            if a.user_id == user_id and a.state == "open"]


def asking(user_id: str) -> bool:
    return any(a.user_id == user_id and a.state == "open" for a in _ASKS.values())


async def wait_for_change(user_id: str, known: set[str], wait: float) -> list[dict[str, Any]]:
    """The open questions once they differ from *known* (what the tab shows),
    or after *wait* seconds."""
    deadline = time.monotonic() + max(0.0, wait)
    while True:
        now = open_for(user_id)
        if {a["id"] for a in now} != known:
            return now
        left = deadline - time.monotonic()
        if left <= 0:
            return now
        ev = _CHANGE.setdefault(user_id, asyncio.Event())
        try:
            await asyncio.wait_for(ev.wait(), left)
        except TimeoutError:
            return open_for(user_id)


async def _ring(a: Ask) -> None:
    from captain_claw.flight_deck.auth import get_db

    try:
        await get_db().add_notification(
            a.user_id, "custom_tool", f"{a.title} asks you something"[:200],
            "Answer it in Flight Deck: the question is open there.", ref_type=BELL_REF, ref_id=a.id)
    except Exception as exc:  # noqa: BLE001 — the dialog still shows it
        log.debug("Couldn't ring for a custom tool's question", error=str(exc))


async def _settle(a: Ask) -> None:
    from captain_claw.flight_deck.auth import get_db

    try:
        await get_db().resolve_line_items(a.user_id, BELL_REF, [a.id])
    except Exception:  # noqa: BLE001
        pass


async def ask(user_id: str, *, call_id: str, title: str, owner_name: str, agent_name: str, label: str,
              questions: list[dict[str, Any]], wait_s: float) -> dict[str, Any]:
    """Put *questions* to *user_id* and wait for the answer, at most *wait_s*.
    The tool's answer: the answers, or why there are none."""
    wait_s = max(MIN_WAIT_S, min(float(wait_s or DEFAULT_WAIT_S), MAX_WAIT_S))
    a = Ask(id="xa_" + secrets.token_hex(8), user_id=user_id, call_id=call_id, title=title, owner_name=owner_name,
            agent_name=agent_name, label=label, questions=questions, created=time.time(),
            deadline=time.time() + wait_s)
    _ASKS[a.id] = a
    _changed(user_id)
    await _ring(a)
    try:
        await asyncio.wait_for(a.event.wait(), wait_s)
    except TimeoutError:
        a.state = "expired"
    finally:
        if a.state == "open":
            a.state = "closed"  # the run was stopped while it waited
        _ASKS.pop(a.id, None)
        _changed(user_id)
        await _settle(a)
    if a.state == "answered":
        return {"ok": True, "answers": a.answers}
    if a.state == "declined":
        return {"ok": False, "declined": True, "error": "They chose not to answer."}
    why = "No answer in time." if a.state == "expired" else "The question was closed."
    return {"ok": False, "no_answer": True, "error": why}


def answer(ask_id: str, user_id: str, *, answers: Any = None, declined: bool = False) -> None:
    """The person's answer (or "Don't answer"). Raises KeyError when there is no
    such open question of theirs, :class:`AskError` when the answers don't fit."""
    a = _ASKS.get(str(ask_id or ""))
    if a is None or a.user_id != user_id or a.state != "open":
        raise KeyError(ask_id)
    if declined:
        a.state = "declined"
    else:
        a.answers = check_answers(a.questions, answers)
        a.state = "answered"
    a.event.set()
    _changed(user_id)


def close_call(call_id: str) -> None:
    """A run ended (or was stopped): its open questions close."""
    for a in list(_ASKS.values()):
        if a.call_id == call_id and a.state == "open":
            a.state = "closed"
            a.event.set()
