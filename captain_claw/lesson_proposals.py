"""Lessons the agent proposes from what its owner tells it — for the owner to approve.

When the owner corrects the agent in a chat ("too long", "stop adding
emojis") or states a standing rule ("always CC Marko"), that is worth
keeping as a lesson (captain_claw/lessons.py). Until now the owner had to
press "Save as a lesson?" themselves; most corrections were never kept.

After an owner's turn, :func:`maybe_propose_lesson` looks at what they just
wrote. A cheap word check (:func:`has_cue`) lets through only messages that
read like a correction or a rule — most turns stop there and cost nothing.
For those, the agent's own model reads the message with the answer it
follows and the lessons it already has near it
(``instructions/lesson_proposal_prompt.md``), and either drafts ONE lesson
or says it teaches nothing. A draft goes to Flight Deck
(``POST /fd/lesson-proposals/agent``), where it waits in the owner's
"Lessons to approve" with the words it came from; Flight Deck drops one the
agent has already, or one the owner turned down. Nothing reaches the
agent's lessons without the owner's yes.

Only under Flight Deck (which keeps the lessons), only for the owner's own
turns (the caller skips members, public chats, automations, exams and
private turns), one draft at a time, a cooldown between drafts and a daily
cap (``lesson_proposals`` in the config).
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.agent import Agent

log = get_logger(__name__)

PROPOSE_PATH = "/fd/lesson-proposals/agent"
PROMPT = "lesson_proposal_prompt.md"
KINDS = ("rule", "ask_first", "fact")
TEXT_MAX = 400
QUOTE_MAX = 200
SAID_MAX = 2500          # a longer message is mostly pasted material, not a correction
REPLY_MAX = 3000         # of the answer it follows, for the model and the owner
NEAR_LESSONS = 8         # lessons it has that the message bears on, so a repeat isn't proposed

_ATTR_RUNNING = "_lesson_proposal_running"
_ATTR_LAST = "_lesson_proposal_last"
_ATTR_DAY = "_lesson_proposal_day"     # (UTC day, drafts that day)

# Words that make a message read like a correction or a standing rule —
# English and Croatian (the owners' languages so far), a few common others.
# Only a gate in front of the model: it misses nothing the model would keep
# often, and lets through plenty the model then drops.
_CUES = re.compile(
    r"\b("
    # English
    r"always|never|from now on|going forward|next time|every time|each time|in future|in the future|"
    r"stop|don'?t|do not|no need|not like that|wrong|instead|rather|prefer|preferably|"
    r"too (long|short|verbose|wordy|formal|informal|casual|much|many|detailed|vague)|shorter|longer|"
    r"remember|keep in mind|make sure|i told you|i said|again|please (don'?t|do not|use|keep|stop)|"
    r"ask me|check with me|without asking|avoid|i want you to|i'd like you to|should(n'?t)? have|"
    # Croatian
    r"uvijek|nikad|nikada|ubuduće|ubuduce|od sada|odsad|sljedeći put|sljedeci put|idući put|iduci put|"
    r"svaki put|prestani|nemoj|nemojte|ne treba|nije trebalo|ne tako|krivo|pogrešno|pogresno|radije|"
    r"previše|previse|predug\w*|prekrat\w*|zapamti|imaj na umu|pazi da|rekao sam|rekla sam|opet|"
    r"pitaj me|provjeri sa mnom|bez pitanja|izbjegavaj|želim da|zelim da|trebao si|trebala si|"
    # a few others
    r"immer|niemals|nie wieder|ab jetzt|siempre|nunca|toujours|jamais"
    r")\b",
    re.IGNORECASE,
)


def has_cue(said: str) -> bool:
    """Whether *said* reads like a correction or a standing rule — the gate
    before any model call."""
    text = " ".join(str(said or "").split())
    if len(text) < 4 or len(text) > SAID_MAX or text.startswith("/"):
        return False
    return bool(_CUES.search(text.replace("’", "'")))


def _content(m: dict[str, Any]) -> str:
    c = m.get("content")
    if isinstance(c, list):  # content blocks
        c = " ".join(str(b.get("text") or "") for b in c if isinstance(b, dict))
    return str(c or "").strip()


def exchange(messages: list[dict[str, Any]]) -> tuple[str, str, str]:
    """(the answer the owner's last message follows, that message, the
    agent's answer to it) from a session's messages — "" where there is none."""
    user_at = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"), -1)
    if user_at < 0:
        return "", "", ""
    said = _content(messages[user_at])
    before = next((_content(m) for m in reversed(messages[:user_at])
                   if m.get("role") == "assistant" and _content(m)), "")
    after = next((_content(m) for m in reversed(messages[user_at + 1:])
                  if m.get("role") == "assistant" and _content(m)), "")
    return before, said, after


def parse(raw: str) -> dict[str, Any] | None:
    """The model's lesson, cleaned — None when it teaches nothing or can't be read."""
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        text = text.split("\n```")[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        got = json.loads(text[start:end + 1])
    except ValueError:
        return None
    if not isinstance(got, dict):
        return None
    lesson = " ".join(str(got.get("text") or "").split())[:TEXT_MAX]
    if not lesson:
        return None
    kind = str(got.get("kind") or "").strip().lower()
    kind = kind if kind in KINDS else "rule"
    return {"kind": kind, "text": lesson, "always": got.get("always") is True and kind != "fact",
            "quote": " ".join(str(got.get("quote") or "").split())[:QUOTE_MAX]}


def _flight_deck_ready() -> bool:
    from captain_claw.fd_client import flight_deck_base, flight_deck_headers, flight_deck_slug

    return bool(flight_deck_base() and flight_deck_slug() and flight_deck_headers().get("X-Agent-Auth"))


def _within_limits(agent: Agent, cfg: Any) -> bool:
    now = time.time()
    if now - float(getattr(agent, _ATTR_LAST, 0.0) or 0.0) < max(0, int(cfg.cooldown_seconds)):
        return False
    day = datetime.now(UTC).date().isoformat()
    seen_day, count = getattr(agent, _ATTR_DAY, ("", 0)) or ("", 0)
    if seen_day == day and count >= max(0, int(cfg.max_per_day)):
        return False
    return True


def _count(agent: Agent) -> None:
    day = datetime.now(UTC).date().isoformat()
    seen_day, count = getattr(agent, _ATTR_DAY, ("", 0)) or ("", 0)
    setattr(agent, _ATTR_DAY, (day, (count if seen_day == day else 0) + 1))
    setattr(agent, _ATTR_LAST, time.time())


def _near_lessons(said: str) -> list[str]:
    try:
        from captain_claw.lessons import get_book

        return [str(x.get("text") or "") for x in get_book().relevant(said, limit=NEAR_LESSONS, budget=4000)]
    except Exception:
        return []


def build_prompt(before: str, said: str, after: str, near: list[str]) -> str:
    parts = []
    if before:
        parts.append(f"## The agent's answer the owner wrote after\n{before[:REPLY_MAX]}")
    parts.append(f"## What the owner wrote\n{said}")
    if after:
        parts.append(f"## How the agent answered that\n{after[:1500]}")
    parts.append("## Lessons the agent already has that bear on this\n"
                 + ("\n".join(f"- {t}" for t in near) if near else "(none)"))
    return "\n\n".join(parts)


async def _send(proposal: dict[str, Any]) -> bool:
    from captain_claw.fd_client import FDClient

    client = FDClient(timeout=10.0)
    try:
        resp = await client.post(PROPOSE_PATH, json=proposal)
        if resp.status_code != 200:
            log.debug("Lesson proposal refused", status=resp.status_code)
            return False
        return True
    except Exception as exc:
        log.debug("Lesson proposal not delivered", error=type(exc).__name__)
        return False
    finally:
        await client.close()


async def maybe_propose_lesson(agent: Agent, *, channel: str = "") -> dict[str, Any] | None:
    """After an owner's turn: draft a lesson from what they just wrote, when
    it teaches one, and hand it to Flight Deck for them to approve. Returns
    what was proposed (None when nothing was). Never raises."""
    try:
        from captain_claw.config import get_config

        cfg = get_config().lesson_proposals
        if not cfg.enabled or getattr(agent, _ATTR_RUNNING, False) or not getattr(agent, "session", None):
            return None
        before, said, after = exchange(list(agent.session.messages))
        if not has_cue(said) or not _within_limits(agent, cfg) or not _flight_deck_ready():
            return None
        setattr(agent, _ATTR_RUNNING, True)
        try:
            _count(agent)
            from captain_claw.llm import Message

            response = await agent._complete_with_guards(
                messages=[Message(role="system", content=agent.instructions.load(PROMPT)),
                          Message(role="user", content=build_prompt(before, said, after, _near_lessons(said)))],
                tools=None,
                interaction_label="lesson_proposal",
                max_tokens=400,
            )
            lesson = parse(getattr(response, "content", "") or "")
            if lesson is None:
                return None
            proposal = {**lesson, "said": said, "reply": before[:REPLY_MAX], "channel": str(channel or "")[:20],
                        "session_id": str(getattr(agent.session, "id", "") or "")}
            if not await _send(proposal):
                return None
            log.info("Lesson proposed to the owner", kind=lesson["kind"], always=lesson["always"])
            return proposal
        finally:
            setattr(agent, _ATTR_RUNNING, False)
    except Exception as exc:
        log.warning("Lesson proposal failed (non-fatal)", error=str(exc))
        return None
