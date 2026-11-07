"""Questions for the person, answered in their Flight Deck.

Two kinds ask:

- a custom tool that declares ``"asks": true`` (Phase 7d of
  docs/external-tools-plan.md): ``captain.ask`` while it runs, through
  ``external_tools``; the tool waits, and its time stops meanwhile;
- an agent, with its ``ask_user`` tool (``ask_routes``): it needs a decision or
  a detail before it can go on, and follows its question until it's answered.

Either way the question goes to the person the run or turn is for (on a shared
agent, the member whose turn it is) and nobody else. It opens in every Flight
Deck tab of theirs (each tab waits on ``GET /fd/asks``) and as a Needs-you item
in Line and the bell; answered in one tab, it closes in the others. Nothing is
kept after the answer beyond a few minutes for the asker to read it.
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
MAX_OPEN_PER_PERSON = 10
DEFAULT_WAIT_S = 600.0
MIN_WAIT_S = 5.0
MAX_WAIT_S = 3600.0
DONE_KEEP_S = 600.0
BELL_REF = "person_ask"

# Turn sources (turn_source.py) with nobody there to answer.
UNATTENDED = frozenset({"scheduled", "flow", "agent", "autonomy", "taskboard", "api", "link", "exam"})
# Turn sources where the person is in Flight Deck, so the dialog reaches them.
# Elsewhere (Telegram, WhatsApp, glasses, a device, a public link) an agent
# asks in its reply instead. "" is a call with no turn behind it.
IN_FLIGHT_DECK = frozenset({"web", "member", "line", ""})

_ID_RE = re.compile(r"^[a-z0-9_]{1,40}$")
ASK_ID_RE = re.compile(r"^xa_[0-9a-f]{16}$")


class AskError(ValueError):
    """Questions (or answers) Flight Deck refuses; the message says why."""


def _source(origin: str) -> str:
    return str(origin or "").split(":", 1)[0].strip().lower()


def attended(origin: str) -> bool:
    """Is someone there to answer a run or turn that came from *origin*?"""
    return _source(origin) not in UNATTENDED


def in_flight_deck(origin: str) -> bool:
    """Does a dialog in Flight Deck reach the person a turn from *origin* is for?"""
    return _source(origin) in IN_FLIGHT_DECK


def _text(value: Any, limit: int, what: str, *, required: bool = True) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()) or len(value) > limit:
        raise AskError(f"{what} must be text of at most {limit} characters.")
    return value.strip()


def check_questions(raw: Any) -> list[dict[str, Any]]:
    """Questions, checked and put in one shape. Raises :class:`AskError`."""
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
        if q.get("default") is not None:
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
    call_id: str              # the custom tool run that asks ("" for an agent)
    title: str                # who asks: the custom tool's title, or the agent's name
    owner_name: str           # a custom tool's maker, when it isn't the person's own
    agent_name: str
    label: str                # "csv-total, used by bank-report" for an inner tool
    questions: list[dict[str, Any]]
    created: float
    deadline: float           # time.time()
    source: str = "tool"      # "tool" (a custom tool) or "agent" (ask_user)
    state: str = "open"       # open | answered | declined | expired | closed
    answers: dict[str, Any] = field(default_factory=dict)
    event: asyncio.Event = field(default_factory=asyncio.Event)

    def view(self) -> dict[str, Any]:
        return {"id": self.id, "source": self.source, "title": self.title, "owner_name": self.owner_name,
                "agent_name": self.agent_name, "label": self.label, "questions": self.questions,
                "created": self.created, "seconds_left": max(0, int(self.deadline - time.time()))}

    def outcome(self) -> dict[str, Any]:
        if self.state == "answered":
            return {"ok": True, "answers": self.answers}
        if self.state == "declined":
            return {"ok": False, "declined": True, "error": "They chose not to answer."}
        why = "No answer in time." if self.state == "expired" else "The question was closed."
        return {"ok": False, "no_answer": True, "error": why}


_ASKS: dict[str, Ask] = {}
_DONE: dict[str, tuple[str, float, dict[str, Any]]] = {}   # id → (user, when, outcome)
_CANCELLED: dict[str, float] = {}                          # ids cancelled before they opened
_CHANGE: dict[str, asyncio.Event] = {}


def reset_for_tests() -> None:
    _ASKS.clear()
    _DONE.clear()
    _CANCELLED.clear()
    _CHANGE.clear()


def _changed(user_id: str) -> None:
    ev = _CHANGE.pop(user_id, None)
    if ev is not None:
        ev.set()


def open_for(user_id: str) -> list[dict[str, Any]]:
    """The person's open questions, oldest first."""
    now = time.time()
    return [a.view() for a in sorted(_ASKS.values(), key=lambda a: a.created)
            if a.user_id == user_id and a.state == "open" and a.deadline > now]


def asking(user_id: str) -> bool:
    return bool(open_for(user_id))


async def _finish(a: Ask) -> dict[str, Any]:
    """The question is done (answered, declined, out of time, closed): out of
    the person's tabs, its ring settled, its outcome kept a while to read."""
    if a.state == "open":
        a.state = "expired" if a.deadline <= time.time() else "closed"
    if _ASKS.pop(a.id, None) is not None:
        now = time.time()
        for old in [k for k, (_, at, _o) in _DONE.items() if now - at > DONE_KEEP_S]:
            _DONE.pop(old, None)
        _DONE[a.id] = (a.user_id, now, a.outcome())
        _changed(a.user_id)
        await _settle(a)
    return a.outcome()


async def _sweep(user_id: str) -> None:
    """Questions whose time ran out with nobody waiting on them."""
    now = time.time()
    for a in [a for a in _ASKS.values() if a.user_id == user_id and a.deadline <= now]:
        await _finish(a)


async def wait_for_change(user_id: str, known: set[str], wait: float) -> list[dict[str, Any]]:
    """The open questions once they differ from *known* (what the tab shows),
    or after *wait* seconds."""
    deadline = time.monotonic() + max(0.0, wait)
    while True:
        await _sweep(user_id)
        now = open_for(user_id)
        if {a["id"] for a in now} != known:
            return now
        left = deadline - time.monotonic()
        if left <= 0:
            return now
        ev = _CHANGE.setdefault(user_id, asyncio.Event())
        try:
            await asyncio.wait_for(ev.wait(), min(left, 5.0))
        except TimeoutError:
            pass


async def _ring(a: Ask) -> None:
    from captain_claw.flight_deck.auth import get_db

    try:
        await get_db().add_notification(
            a.user_id, BELL_REF, f"{a.title} asks you something"[:200],
            "Answer it in Flight Deck: the question is open there.", ref_type=BELL_REF, ref_id=a.id)
    except Exception as exc:  # noqa: BLE001 — the dialog still shows it
        log.debug("Couldn't ring for a question", error=str(exc))


async def _settle(a: Ask) -> None:
    from captain_claw.flight_deck.auth import get_db

    try:
        await get_db().resolve_line_items(a.user_id, BELL_REF, [a.id])
    except Exception:  # noqa: BLE001
        pass


async def open_ask(user_id: str, *, title: str, questions: list[dict[str, Any]], wait_s: float,
                   source: str = "tool", call_id: str = "", owner_name: str = "", agent_name: str = "",
                   label: str = "", ask_id: str = "") -> Ask:
    """Put *questions* to *user_id*: in their tabs and their bell. *ask_id* is
    the asker's own id for it (an agent's, so it can cancel it before it
    hears back); refused when that id was cancelled already."""
    if ask_id and (not ASK_ID_RE.match(ask_id) or ask_id in _ASKS or ask_id in _DONE):
        raise AskError("That question id is taken.")
    if ask_id and _CANCELLED.pop(ask_id, None) is not None:
        raise AskError("That question was cancelled.")
    if sum(1 for a in _ASKS.values() if a.user_id == user_id) >= MAX_OPEN_PER_PERSON:
        raise AskError(f"They already have {MAX_OPEN_PER_PERSON} questions waiting.")
    wait_s = max(MIN_WAIT_S, min(float(wait_s or DEFAULT_WAIT_S), MAX_WAIT_S))
    a = Ask(id=ask_id or "xa_" + secrets.token_hex(8), user_id=user_id, call_id=call_id, title=title,
            owner_name=owner_name, agent_name=agent_name, label=label, questions=questions, created=time.time(),
            deadline=time.time() + wait_s, source=source)
    _ASKS[a.id] = a
    _changed(user_id)
    await _ring(a)
    return a


async def wait_answer(a: Ask, wait: float) -> dict[str, Any] | None:
    """Wait up to *wait* seconds: the outcome once it's done, else None."""
    if a.state != "open" or a.id not in _ASKS:
        return await _finish(a)
    left = a.deadline - time.time()
    try:
        await asyncio.wait_for(a.event.wait(), max(0.0, min(wait, left)))
    except TimeoutError:
        if a.deadline > time.time():
            return None
    return await _finish(a)


async def follow(ask_id: str, user_id: str, wait: float) -> dict[str, Any]:
    """An agent following its question: the outcome, or ``{"pending": id}``.
    Raises KeyError for a question that isn't theirs (or is long gone)."""
    a = _ASKS.get(ask_id)
    if a is not None:
        if a.user_id != user_id:
            raise KeyError(ask_id)
        done = await wait_answer(a, wait)
        return done if done is not None else {"pending": a.id, "seconds_left": max(0, int(a.deadline - time.time()))}
    kept = _DONE.get(ask_id)
    if kept is None or kept[0] != user_id:
        raise KeyError(ask_id)
    return kept[2]


async def cancel(ask_id: str, user_id: str) -> bool:
    """The asker doesn't need the answer any more (its turn was stopped)."""
    a = _ASKS.get(ask_id)
    if a is None:
        if ASK_ID_RE.match(ask_id) and ask_id not in _DONE and len(_CANCELLED) < 10_000:
            _CANCELLED[ask_id] = time.time()  # it may still be on its way: it won't open
        return False
    if a.user_id != user_id:
        return False
    a.state = "closed"
    a.event.set()
    await _finish(a)
    return True


async def ask(user_id: str, *, call_id: str, title: str, owner_name: str, agent_name: str, label: str,
              questions: list[dict[str, Any]], wait_s: float) -> dict[str, Any]:
    """A custom tool's question: put it and wait for the outcome."""
    a = await open_ask(user_id, title=title, questions=questions, wait_s=wait_s, call_id=call_id,
                       owner_name=owner_name, agent_name=agent_name, label=label)
    try:
        while True:
            done = await wait_answer(a, MAX_WAIT_S)
            if done is not None:
                return done
    finally:
        if a.id in _ASKS:  # the run was stopped while it waited
            a.state = "closed"
            await _finish(a)


def answer(ask_id: str, user_id: str, *, answers: Any = None, declined: bool = False) -> None:
    """The person's answer (or "Don't answer"). Raises KeyError when there is no
    such open question of theirs, :class:`AskError` when the answers don't fit."""
    a = _ASKS.get(str(ask_id or ""))
    if a is None or a.user_id != user_id or a.state != "open" or a.deadline <= time.time():
        raise KeyError(ask_id)
    if declined:
        a.state = "declined"
    else:
        a.answers = check_answers(a.questions, answers)
        a.state = "answered"
    a.event.set()
    _changed(user_id)


def close_call(call_id: str) -> None:
    """A custom tool run ended (or was stopped): its open questions close."""
    for a in list(_ASKS.values()):
        if call_id and a.call_id == call_id and a.state == "open":
            a.state = "closed"
            a.event.set()
