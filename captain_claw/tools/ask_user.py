"""`ask_user` tool — ask the person you're working for, in their Flight Deck.

When the agent needs a decision or a detail before it can go on, it asks: one
question (text, one of a list, several of a list, or yes/no) or a short form of
several, shown as a dialog in every Flight Deck tab of that person — the
agent's owner, or on a shared agent the member whose turn it is (Flight Deck
decides from the turn's grant, as it does for custom tools). The turn waits for
the answer, at most an hour.

The question and its dialog are Flight Deck's (``flight_deck/person_asks.py``,
``ask_routes.py``), shared with custom tools that ask. The agent asks with a
little waiting and then follows it in short requests, so no request is held
open long; stopping the turn cancels the question. A turn that started on its
own (a schedule, autonomous work) has nobody to ask, and in Telegram, WhatsApp
or other channels the dialog wouldn't reach the person: the tool says so, and
the agent asks in its reply or decides itself.
"""

from __future__ import annotations

import json
import secrets
from typing import Any

import httpx

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

NO_FLIGHT_DECK = "Asking with a dialog needs Flight Deck: ask in your reply instead."
NO_GRANT = "This member's turn has no grant, so the question can't reach them: ask in your reply instead."
_TIMEOUT = 45.0  # each request waits at most 30 s on Flight Deck's side
_KIND_KEYS = ("choices", "multiple", "min", "max", "confirm", "lines", "help", "default", "optional")


def _fd_base() -> str:
    from captain_claw.fd_client import flight_deck_base

    return flight_deck_base()


async def _call(path: str, body: dict[str, Any]) -> tuple[int, Any]:
    """``(status, json)`` from ``/fd/asks/agent<path>``. A member's turn sends its
    grant, so Flight Deck asks the member; without one it refuses here."""
    from captain_claw import speaker
    from captain_claw.fd_client import flight_deck_headers

    headers = flight_deck_headers()
    try:
        headers.update(speaker.grant_headers())
        params = speaker.grant_params()
    except speaker.SpeakerGrantMissing:
        return 403, {"detail": NO_GRANT}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(f"{_fd_base()}/fd/asks/agent{path}", headers=headers, params=params, json=body)
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, {"detail": resp.text[:500]}


def questions_from(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    """The tool's arguments as Flight Deck's questions: a ``questions`` list, or
    one ``question`` with its options."""
    qs = kwargs.get("questions")
    if isinstance(qs, str):
        try:
            qs = json.loads(qs)
        except ValueError:
            qs = None
    if isinstance(qs, list) and qs:
        return [q for q in qs if isinstance(q, dict)]
    question = str(kwargs.get("question") or "").strip()
    if not question:
        return []
    one: dict[str, Any] = {"id": "answer", "question": question}
    for key in _KIND_KEYS:
        if kwargs.get(key) not in (None, "", []):
            one[key] = kwargs[key]
    return [one]


def _said(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if isinstance(value, list):
        return ", ".join(map(str, value)) if value else "(none picked)"
    if value is None or (isinstance(value, str) and not value.strip()):
        return "(left empty)"
    return str(value)


def describe_answers(questions: list[dict[str, Any]], answers: dict[str, Any]) -> str:
    if len(questions) == 1:
        return f"They answered: {_said(answers.get(questions[0].get('id') or 'answer'))}"
    lines = ["They answered:"]
    for q in questions:
        qid = str(q.get("id") or "")
        lines.append(f"- {q.get('question')}: {_said(answers.get(qid))}")
    lines += ["", "As data: " + json.dumps(answers, ensure_ascii=False)]
    return "\n".join(lines)


class AskUserTool(Tool):
    name = "ask_user"
    description = (
        "Ask the person you're working for something you need before you can go on, as a dialog in their "
        "Flight Deck, and wait for the answer. One question (free text; one of `choices`; several of them with "
        "multiple=true; or a yes/no with confirm=true), or a short form: `questions`, a list of objects with an "
        "`id` and `question` plus the same options. Use it for a real decision or a missing detail that changes "
        "what you do, with choices whenever the answer is one of a few; not to confirm what they already told you, "
        "and never for passwords, codes or keys. You get their answer, \"they chose not to answer\", or no answer "
        "in time (wait_minutes, 10 by default): then carry on with your best judgement and say what you assumed. "
        "In Telegram, WhatsApp or other channels, or a turn that started on its own, the dialog can't reach them: "
        "the tool says so, and you ask in your reply or decide yourself."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "One question, in a short plain sentence."},
            "choices": {"type": "array", "items": {"type": "string"},
                        "description": "The options to pick from (up to 50), when the answer is one of a few."},
            "multiple": {"type": "boolean", "description": "With choices: they may pick several."},
            "confirm": {"type": "boolean", "description": "A yes/no question."},
            "lines": {"type": "integer", "description": "A free-text answer: lines of room for it (1-20)."},
            "help": {"type": "string", "description": "A line under the question: what it's for, what you'll do with it."},
            "questions": {
                "type": "array",
                "description": "Several questions in one dialog (up to 10). Each: id (lowercase_with_underscores), "
                               "question, and optionally choices, multiple, min, max, confirm, lines, help, "
                               "optional, default.",
                "items": {"type": "object"},
            },
            "wait_minutes": {"type": "number", "description": "How long to wait for the answer (1-60, default 10)."},
        },
    }
    timeout_seconds = 3_700.0  # an hour of waiting at most, followed in short requests

    async def execute(self, **kwargs: Any) -> ToolResult:
        if not _fd_base():
            return ToolResult(success=False, error=NO_FLIGHT_DECK)
        questions = questions_from(kwargs)
        if not questions:
            return ToolResult(success=False, error="Say what to ask: question=..., or questions=[...].")
        try:
            minutes = float(kwargs.get("wait_minutes") or 10)
        except (TypeError, ValueError):
            minutes = 10.0
        wait_s = max(1.0, min(minutes, 60.0)) * 60
        agent = kwargs.get("_agent")
        origin = getattr(agent, "_turn_source", None)
        origin = str(origin[0]) if isinstance(origin, tuple) and origin else ""
        ask_id = "xa_" + secrets.token_hex(8)
        stream = kwargs.get("_stream_callback")
        finished = False
        try:
            status, data = await _call("", {"questions": questions, "wait_s": wait_s, "origin": origin,
                                            "wait": 20, "ask_id": ask_id})
            told = False
            while status == 200 and isinstance(data, dict) and data.get("pending"):
                if not told and callable(stream):
                    told = True
                    try:
                        stream("Waiting for their answer in Flight Deck…\n")
                    except Exception:  # noqa: BLE001
                        pass
                status, data = await _call("/follow", {"id": data["pending"], "wait": 25})
            finished = True
        except httpx.HTTPError as exc:
            return ToolResult(success=False, error=f"Flight Deck didn't answer ({type(exc).__name__}): ask in your reply instead.")
        finally:
            if not finished:  # the turn was stopped (or the line failed): the question closes
                try:
                    await _call("/cancel", {"id": ask_id})
                except Exception:  # noqa: BLE001
                    pass
        if status != 200 or not isinstance(data, dict):
            detail = data.get("detail") if isinstance(data, dict) else data
            return ToolResult(success=False, error=f"The question couldn't be asked: {detail}")
        if data.get("ok"):
            return ToolResult(content=describe_answers(questions, data.get("answers") or {}))
        if data.get("declined"):
            return ToolResult(content="They chose not to answer. Carry on with your best judgement, and say what you "
                                      "assumed.")
        if data.get("elsewhere") or data.get("nobody"):
            return ToolResult(success=False, error=str(data.get("error") or "The question can't reach them here."))
        return ToolResult(content=f"No answer within {int(wait_s // 60)} minutes ({data.get('error') or 'no answer'}). "
                                  "Carry on with your best judgement, and say what you assumed.")
