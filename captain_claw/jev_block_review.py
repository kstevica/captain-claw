"""Block review — when a guard refuses a tool call, ask Jev what the agent
should do next instead of handing it one fixed message.

The tool loop (``agent_tool_loop_mixin``) refuses a call that repeats work
already done this turn: a duplicate call, or a full rewrite of a file written
with nothing learned since. The refusal itself is almost always right. What
went wrong on glasses staging (Claude Haiku, 2026-10-22/23) was the fixed text
that came with it: "Do NOT call any more tools" ended turns that still owed the
person a share link and an email, and the model told them it had no tools.

Here Jev (TypeSafe, see :mod:`captain_claw.jev_client`) reads the person's
request, the turn's recent steps and the refused call, and picks one next
step: check the file, carry on with the rest of the request, give the final
answer, or ask the person. For a read-only call it may also judge that the
repeat is legitimate (something changed since, or the person asked to check
again) and let it run once. Each pick maps to a fixed instruction written
here; Jev chooses, it never writes the text.

Never in the way: off, no key, a shared-agent member's turn (Jev is
owner-only), a timeout, an error or a pick below the confidence bar all mean
the guard's usual message. A call that sends, creates or changes something is
never released. Off with ``tools.jev.review_blocks: false``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from captain_claw import jev_client
from captain_claw.jev_progress import _flat, _step_lines
from captain_claw.logging import get_logger

log = get_logger(__name__)

QUESTION_NAME = "next_step"

# The chosen option's probability at or above this replaces the guard's usual
# message with the picked instruction. A wrong pick costs one step: the call
# stays refused either way.
THRESHOLD = 0.55
# Letting a refused call run needs more: a wrong release is the loop the guard
# was there to stop (bounded to one release per call per turn).
RELEASE_THRESHOLD = 0.8

# Jev answers inside the tool loop, so it waits less than the progress check.
TIMEOUT_SECONDS = 8.0

# At most this many reviews per turn: a model that keeps hitting guards gets
# the usual messages after that, without the extra round-trips.
MAX_REVIEWS_PER_TURN = 3

_REQUEST_CHARS = 1500
_STEPS_CHARS = 6000
_CALL_CHARS = 400

DUPLICATE = "duplicate"
BLIND_REWRITE = "blind_rewrite"

RELEASE = "repeat_ok"

# What each option means — Jev's criteria.
OPTION_MEANINGS: dict[str, str] = {
    "verify": (
        "The agent doubts what the target holds (it thinks its earlier result or "
        "the saved file is wrong, empty or a placeholder) and should look at it "
        "once before doing anything else."
    ),
    "continue": (
        "The refused call's work is already done and the request still has other "
        "steps left (sharing or sending what was made, the next item, another "
        "part of the question) that the agent should do now."
    ),
    "finish": (
        "Everything the person asked for is done; the agent should give its final "
        "answer now."
    ),
    "ask_user": (
        "The agent can't go on without something only the person can give: a "
        "choice between options, missing information or a permission."
    ),
    RELEASE: (
        "Repeating this read-only call is legitimate: something changed since the "
        "earlier identical call (a write, a send, a delete, time passing, or the "
        "person asked to check again), so it can return something new."
    ),
}

# What the agent reads for each pick (after the guard's factual first line).
_NEXT_STEP: dict[str, str] = {
    "verify": (
        "Next: read `{target}` once to see what it actually holds, then decide. "
        "Don't redo it from memory."
    ),
    "continue": (
        "Next: use the result you already have and go on with the rest of the "
        "request, for example sharing or sending what you made, or the next item. "
        "Don't repeat this call."
    ),
    "finish": (
        "Next: the request looks done. Give your final answer now: what you did "
        "and where the result is."
    ),
    "ask_user": (
        "Next: you need something only the person can give. Ask them one short, "
        "specific question."
    ),
}

# Tools whose calls only look things up, and the actions that only look
# things up on the tools that can also change something.
READ_ONLY_TOOLS = frozenset({
    "read", "glob", "grep", "web_fetch", "web_get", "web_search",
    "pdf_extract", "docx_extract", "xlsx_extract", "pptx_extract",
    "image_vision", "image_ocr", "ocr_pdf", "ocr_image", "summarize_files",
})
READ_ONLY_ACTIONS = frozenset({
    "list", "search", "get", "read", "view", "show", "status", "info", "check",
    "find", "query", "list_events", "list_messages", "list_threads",
    "get_message", "get_thread", "get_event", "list_jobs", "map", "inbox",
})


@dataclass(frozen=True)
class Review:
    """Jev's pick for one refused call."""

    choice: str
    probability: float
    # True: run the call after all. Otherwise ``message`` replaces the
    # guard's usual text, or is None when the pick isn't confident enough.
    release: bool
    message: str | None


def enabled() -> bool:
    """``tools.jev.review_blocks`` (default on)."""
    try:
        from captain_claw.config import get_config

        return bool(getattr(get_config().tools.jev, "review_blocks", True))
    except Exception:
        return True


async def ready() -> bool:
    """Whether a review can run: on, not a shared-agent member's turn and a
    usable key. Never raises."""
    if not enabled():
        return False
    try:
        if jev_client._member_call():
            return False
        return bool(await jev_client.refresh_connected())
    except Exception as exc:
        log.debug("Jev block review unavailable", error=str(exc) or type(exc).__name__)
        return False


def can_release(tool: str, action: str = "") -> bool:
    """A call that only looks something up — the one kind Jev may let repeat."""
    tool = str(tool or "").strip().lower()
    action = str(action or "").strip().lower()
    if tool in READ_ONLY_TOOLS:
        return True
    return bool(action) and action in READ_ONLY_ACTIONS


def blocked_line(kind: str, tool: str, *, count: int = 1, path: str = "") -> str:
    """The guard's factual first line. Keeps the prefix the loop's other
    checks look for (``DUPLICATE CALL BLOCKED`` / ``BLIND REWRITE BLOCKED``)."""
    if kind == BLIND_REWRITE:
        return (
            f"BLIND REWRITE BLOCKED: You already wrote `{path}` this turn and have "
            "gathered nothing new since. The file is saved on disk — a "
            "`[written to disk: …]` note in your history is a shortened copy of "
            "your call, not the file's content."
        )
    return (
        f"DUPLICATE CALL BLOCKED: `{tool}` already ran with these arguments "
        f"{count} time(s) this turn, so it was not run again."
    )


def options_for(kind: str, *, target: str = "", releasable: bool = False) -> list[str]:
    """The picks that make sense for this refusal."""
    options = ["continue", "finish", "ask_user"]
    if target and (kind == BLIND_REWRITE or not releasable):
        # Something on disk or in a store to look at; a read-only repeat has
        # nothing to verify that the repeat itself wouldn't.
        options.insert(0, "verify")
    if releasable:
        options.append(RELEASE)
    return options


def _call_text(tool: str, arguments: Any) -> str:
    if isinstance(arguments, dict):
        try:
            args = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        except Exception:
            args = str(arguments)
    else:
        args = str(arguments or "")
    return _flat(f"{tool}({args})", _CALL_CHARS)


def build_state(
    request: str,
    turn_messages: list[dict[str, Any]],
    *,
    kind: str,
    tool: str,
    arguments: Any,
    count: int = 1,
) -> dict[str, Any]:
    """What Jev reads: the request, the turn's recent steps, the refused call
    and why it was refused."""
    from captain_claw import msg_origin

    _block, said = msg_origin.split_surface_block(str(request or ""))
    lines = _step_lines(turn_messages)
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
    if cut:
        steps = f"[… {cut} earlier steps cut …]\n{steps}"
    if kind == BLIND_REWRITE:
        why = ("the agent already wrote this file in full this turn and has read, "
               "fetched or run nothing since, so this rewrite would come from memory")
    else:
        why = f"the identical call already ran {count} time(s) this turn"
    return {
        "request": said.strip()[:_REQUEST_CHARS],
        "recent_steps": steps or "(no steps recorded)",
        "blocked_call": _call_text(tool, arguments),
        "why_blocked": why,
    }


def questions_for(options: list[str]) -> dict[str, dict[str, Any]]:
    return {
        QUESTION_NAME: {
            "type": "choice",
            "instructions": (
                "An AI agent's last tool call (blocked_call) was refused by a guard "
                "because it repeated work already done this turn (why_blocked). "
                "Judging by the person's request and the turn's steps, what should "
                "the agent do next?"
            ),
            "options": {name: OPTION_MEANINGS[name] for name in options},
        },
    }


async def _ask(state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
    if jev_client.under_flight_deck():
        return await jev_client.ask_via_flight_deck(state, questions, timeout=TIMEOUT_SECONDS)
    return await jev_client.standalone_api().ask(state, questions)


def _pick(response: dict[str, Any], options: list[str]) -> tuple[str, float] | None:
    answers = response.get("answers") if isinstance(response, dict) else None
    answer = answers.get(QUESTION_NAME) if isinstance(answers, dict) else None
    if not isinstance(answer, dict):
        return None
    choice = str(answer.get("choice") or "").strip()
    if choice not in options:
        return None
    probs = answer.get("probabilities")
    p = probs.get(choice) if isinstance(probs, dict) else None
    if p is None:
        p = answer.get("confidence")
    try:
        return choice, float(p)
    except (TypeError, ValueError):
        return None


async def review(
    request: str,
    turn_messages: list[dict[str, Any]],
    *,
    kind: str,
    tool: str,
    arguments: Any,
    count: int = 1,
    target: str = "",
    releasable: bool = False,
) -> Review | None:
    """Jev's pick for a refused call, or None when the review can't run — off,
    no key, a shared-agent member's turn, a timeout or an error. Never raises.

    The caller checks :func:`ready` first (it also refreshes the key state)."""
    options = options_for(kind, target=target, releasable=releasable)
    try:
        state = build_state(
            request, turn_messages, kind=kind, tool=tool, arguments=arguments, count=count,
        )
        resp = await asyncio.wait_for(_ask(state, questions_for(options)), TIMEOUT_SECONDS + 2)
    except Exception as exc:  # TimeoutError, JevError, JevInputError, anything
        log.debug("Jev block review skipped", error=str(exc) or type(exc).__name__)
        return None
    picked = _pick(resp, options)
    if picked is None:
        return None
    choice, p = picked
    release = choice == RELEASE and p >= RELEASE_THRESHOLD
    message = None
    if not release and choice != RELEASE and p >= THRESHOLD:
        first = blocked_line(kind, tool, count=count, path=target)
        message = f"{first} {_NEXT_STEP[choice].format(target=target or 'it')}"
    log.info("Jev block review", guard=kind, tool=tool, choice=choice,
             p=round(p, 3), released=release, applied=bool(release or message))
    return Review(choice=choice, probability=p, release=release, message=message)
