"""An MCP server asks the person before it goes on (MCP elicitation).

While one of its tools runs, an MCP server may send ``elicitation/create``: a
message and a small form of plain fields (text, a number, yes/no, one or
several of a list). Some servers make a change only after the person approves
it this way — the model must never answer for them. Flight Deck puts the
question on a card (``person_asks``, source ``"mcp"``), in every tab of the
person and the bell, exactly as an agent's ``ask_user``, and sends their answer
back: ``accept`` with the fields, ``decline`` ("Don't answer"), or ``cancel``
(no answer in time, or the call ended).

Who is asked: the person the tool call is for (:class:`Asker`), given by the
caller — the agent's owner, or on a shared agent the member whose turn it is.
One server process serves every call, and over stdio a server's request
doesn't say which call it belongs to, so the question goes out only when every
call in flight on that server is for the same person; otherwise (or when a
call is for nobody we may ask, or every one is a run nobody attends) it is
cancelled at once and the server changes nothing. Nothing of the form is ever
shown to the model.

Only form mode is offered (``client_capabilities``). The schema is the MCP
"restricted" one — a flat object of primitive properties; anything else is
refused with "invalid params".
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from captain_claw.flight_deck import person_asks as asks
from captain_claw.flight_deck.mcp_transport import INVALID_PARAMS, MCPRequestError
from captain_claw.logging import get_logger

log = get_logger(__name__)

WAIT_S = 20 * 60.0          # how long a card waits for the person
MAX_MESSAGE = 8000          # the server's message, as shown on the card
CANCEL = {"action": "cancel"}
DECLINE = {"action": "decline"}


@dataclass(eq=False)
class Asker:
    """The person a tool call is for, and its open cards while it runs."""

    user_id: str
    agent: str = ""            # the agent's slug ("" for a person's own call)
    agent_name: str = ""
    origin: str = ""           # the turn's source (turn_source.py), "" for none
    session_id: str = ""
    cards: set[str] = field(default_factory=set)
    # Last sign of life (start, progress, a card opening or closing), for
    # "the call has gone quiet" (mcp_calls).
    last_activity: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    @property
    def asking(self) -> bool:
        return bool(self.cards)


# card id → the calls (Asker ids) it was put to; it closes when none is left.
_CARD_CALLS: dict[str, set[int]] = {}


def reset_for_tests() -> None:
    _CARD_CALLS.clear()


# ── The form → questions on a card ────────────────────────────────────


@dataclass
class _Field:
    name: str                  # the property, as the server named it
    qid: str                   # the question's id on the card
    kind: str                  # string | number | integer | boolean | one | many
    values: dict[str, Any] = field(default_factory=dict)   # choice label → value


def _clip(text: Any, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def _labelled(options: list[tuple[Any, Any]], name: str) -> dict[str, Any]:
    """``{label: value}`` for a list's options, labels made unique and short."""
    if not 1 <= len(options) <= asks.MAX_CHOICES:
        raise MCPRequestError(INVALID_PARAMS, f"'{name}' needs 1–{asks.MAX_CHOICES} options")
    out: dict[str, Any] = {}
    for value, title in options:
        if not isinstance(value, str):
            raise MCPRequestError(INVALID_PARAMS, f"'{name}' has an option that isn't text")
        base = _clip(title if isinstance(title, str) and title.strip() else value, asks.MAX_CHOICE - 6) or "(empty)"
        label, n = base, 2
        while label in out:
            label, n = f"{base} ({n})", n + 1
        out[label] = value
    return out


def _options(prop: dict[str, Any], name: str) -> dict[str, Any] | None:
    """A single or multiple choice's options: ``enum`` (with the legacy
    ``enumNames``), or ``oneOf`` / ``anyOf`` of ``{const, title}``."""
    if isinstance(prop.get("enum"), list):
        values = prop["enum"]
        names = prop.get("enumNames")
        titles = names if isinstance(names, list) and len(names) == len(values) else values
        return _labelled(list(zip(values, titles)), name)
    for key in ("oneOf", "anyOf"):
        if isinstance(prop.get(key), list):
            items = prop[key]
            if not all(isinstance(o, dict) and "const" in o for o in items):
                raise MCPRequestError(INVALID_PARAMS, f"'{name}' has options without a value")
            return _labelled([(o["const"], o.get("title")) for o in items], name)
    return None


def _question(name: str, prop: Any, qid: str, required: bool) -> tuple[_Field, dict[str, Any]]:
    if not isinstance(prop, dict):
        raise MCPRequestError(INVALID_PARAMS, f"'{name}' isn't described")
    kind = prop.get("type")
    q: dict[str, Any] = {"id": qid, "question": _clip(prop.get("title") or name, asks.MAX_QUESTION),
                         "optional": not required}
    if prop.get("description"):
        q["help"] = _clip(prop["description"], asks.MAX_HELP)
    default = prop.get("default")
    if kind == "string":
        options = _options(prop, name)
        if options is not None:
            f = _Field(name, qid, "one", options)
            q["choices"] = list(options)
            if default is not None:
                q["default"] = next((label for label, v in options.items() if v == default), None)
        else:
            f = _Field(name, qid, "string")
            if isinstance(prop.get("minLength"), int):
                q["min_length"] = max(0, min(prop["minLength"], asks.MAX_ANSWER))
            if isinstance(prop.get("maxLength"), int):
                q["max_length"] = max(0, min(prop["maxLength"], asks.MAX_ANSWER))
            if prop.get("format") in asks.TEXT_FORMATS:
                q["format"] = prop["format"]
            q["default"] = default if isinstance(default, str) else None
    elif kind == "array":
        items = prop.get("items")
        options = _options(items, name) if isinstance(items, dict) else None
        if options is None:
            raise MCPRequestError(INVALID_PARAMS, f"'{name}' is a list without options")
        f = _Field(name, qid, "many", options)
        q.update(choices=list(options), multiple=True)
        low, high = prop.get("minItems"), prop.get("maxItems")
        if isinstance(low, int) and not isinstance(low, bool):
            q["min"] = max(0, min(low, len(options)))
        if isinstance(high, int) and not isinstance(high, bool):
            q["max"] = max(1, min(high, len(options)))
        if isinstance(default, list):
            q["default"] = [label for label, v in options.items() if v in default]
    elif kind in ("number", "integer"):
        f = _Field(name, qid, kind)
        q.update(number=True, integer=kind == "integer")
        if asks.is_number(prop.get("minimum")):
            q["min_value"] = prop["minimum"]
        if asks.is_number(prop.get("maximum")):
            q["max_value"] = prop["maximum"]
        q["default"] = default if asks.is_number(default) else None
    elif kind == "boolean":
        f = _Field(name, qid, "boolean")
        q.update(confirm=True, default=default if isinstance(default, bool) else None)
    else:
        raise MCPRequestError(INVALID_PARAMS, f"'{name}' has a type Flight Deck can't ask for ({kind!r})")
    if q.get("default") is None:
        q.pop("default", None)
    try:
        return f, asks.check_questions([q])[0]
    except asks.AskError:
        q.pop("default", None)  # a default the field itself wouldn't take
        try:
            return f, asks.check_questions([q])[0]
        except asks.AskError as exc:
            raise MCPRequestError(INVALID_PARAMS, f"'{name}': {exc}") from None


def form_questions(schema: Any) -> tuple[list[_Field], list[dict[str, Any]]]:
    """The requested schema's fields and the card's questions for them.
    Raises :class:`MCPRequestError` (invalid params) for a form we can't show."""
    if not isinstance(schema, dict) or schema.get("type", "object") != "object":
        raise MCPRequestError(INVALID_PARAMS, "requestedSchema must be an object schema")
    props = schema.get("properties")
    if not isinstance(props, dict) or not props:
        raise MCPRequestError(INVALID_PARAMS, "requestedSchema has no fields")
    if len(props) > asks.MAX_QUESTIONS:
        raise MCPRequestError(INVALID_PARAMS, f"Flight Deck shows at most {asks.MAX_QUESTIONS} fields at a time")
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    fields, questions = [], []
    for i, (name, prop) in enumerate(props.items(), 1):
        f, q = _question(str(name), prop, f"f{i}", name in required)
        fields.append(f)
        questions.append(q)
    return fields, questions


def form_content(fields: list[_Field], answers: dict[str, Any]) -> dict[str, Any]:
    """The person's answers as the server's ``content``: its own names and
    values (an option's value, not its label); fields left empty are left out."""
    out: dict[str, Any] = {}
    for f in fields:
        value = answers.get(f.qid)
        if value is None:
            continue
        if f.kind == "string":
            if not str(value).strip():
                continue
            out[f.name] = value
        elif f.kind == "one":
            out[f.name] = f.values[value]
        elif f.kind == "many":
            out[f.name] = [f.values[v] for v in value]
        elif f.kind == "integer":
            out[f.name] = int(value)
        else:
            out[f.name] = value
    return out


# ── Who is asked ──────────────────────────────────────────────────────


def _person(calls: list[Asker | None]) -> list[Asker]:
    """The calls in flight when they are all for one person, else ``[]``."""
    if not calls or any(c is None or not c.user_id for c in calls):
        return []
    if len({c.user_id for c in calls if c is not None}) != 1:
        return []
    return [c for c in calls if c is not None]


async def handle(server: str, params: Any, calls: list[Asker | None]) -> dict[str, Any]:
    """Answer *server*'s ``elicitation/create`` with *params*, sent while
    *calls* were in flight on it: a card for their person, and its outcome."""
    if not isinstance(params, dict):
        raise MCPRequestError(INVALID_PARAMS, "elicitation/create needs params")
    if str(params.get("mode") or "form") != "form":
        raise MCPRequestError(INVALID_PARAMS, "Only form elicitation is supported")
    fields, questions = form_questions(params.get("requestedSchema"))
    message = str(params.get("message") or "").strip()
    if len(message) > MAX_MESSAGE:
        message = message[: MAX_MESSAGE - 1].rstrip() + "…"
    mine = _person(calls)
    if not mine:
        log.info("MCP question cancelled: no single person to ask", server=server, calls=len(calls))
        return CANCEL
    if not any(asks.attended(c.origin) for c in mine):
        log.info("MCP question cancelled: nobody attends the run", server=server)
        return CANCEL
    names = {c.agent_name for c in mine if c.agent_name}
    agent_name = next(iter(names)) if len(names) == 1 else ""
    ring = (_clip(f"{server} asks you before it goes on", 200),
            (f"{agent_name} is using it and waits" if agent_name else "It waits")
            + " for your answer: the card is open in Flight Deck.")
    try:
        card = await asks.open_ask(
            mine[0].user_id, source="mcp", title=server, agent_name=agent_name,
            questions=questions, wait_s=WAIT_S, ring=ring,
            detail={"server": server, "message": message})
    except asks.AskError as exc:
        log.info("MCP question cancelled: the card couldn't open", server=server, error=str(exc))
        return CANCEL
    _CARD_CALLS[card.id] = {id(c) for c in mine}
    for c in mine:
        c.cards.add(card.id)
        c.touch()
    log.info("MCP question on a card", server=server, card=card.id, fields=len(fields))
    try:
        while True:
            done = await asks.wait_answer(card, asks.MAX_WAIT_S)
            if done is not None:
                break
    except asyncio.CancelledError:
        # The server took the question back (its call ended or timed out).
        await asks.cancel(card.id, card.user_id)
        raise
    finally:
        _CARD_CALLS.pop(card.id, None)
        for c in mine:
            c.cards.discard(card.id)
            c.touch()
    if done.get("ok"):
        content = form_content(fields, done.get("answers") or {})
        log.info("MCP question answered", server=server, card=card.id)
        return {"action": "accept", "content": content}
    if done.get("declined"):
        log.info("MCP question declined", server=server, card=card.id)
        return DECLINE
    log.info("MCP question not answered", server=server, card=card.id)
    return CANCEL


async def call_ended(asker: Asker) -> None:
    """A tool call ended (done, failed or stopped): its cards that no other
    call in flight is waiting on close, and their servers hear "cancel"."""
    for card_id in list(asker.cards):
        waiting = _CARD_CALLS.get(card_id)
        if waiting is not None:
            waiting.discard(id(asker))
            if waiting:
                continue
        asker.cards.discard(card_id)
        await asks.cancel(card_id, asker.user_id)
