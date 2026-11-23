"""Drive — how far the harness pushes a turn.

Reasoning effort says how hard the model thinks, a cognitive mode says how the
agent approaches work; Drive says how hard the *harness* works to finish what
was asked: how many steps a turn may take, how often a weak answer is sent
back, whether the result is checked before it is delivered and (at the top
level only) whether the agent may do a little more than was asked.

    relaxed · diligent · persistent · ambitious

``diligent`` is the harness as it was before Drive existed. Plan and the
owner's decisions: ``docs/drive-plan.md``.

Phase 1 (this file): the levels, the turn scope and the agent's default, so
every turn has a level and the usage and turn logs record it. Nothing reads
the level to change behaviour yet — the limits a level sets are added to
:class:`DriveLevel` in the phase that makes the harness read them.

The level reaches the harness the way reasoning effort does, through a context
variable: the chat frame sets the turn's level (:func:`turn_drive`) and tasks
created inside the turn inherit it. A turn nobody picked a level for (a
scheduled job, a hand-over, WhatsApp, Telegram, a link) runs at the agent's
own default (``drive.default`` in its config), else ``diligent`` — a level
picked in one chat never reaches a scheduled run.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

DEFAULT = "diligent"


@dataclass(frozen=True)
class DriveLevel:
    """One level. Fields are added with the harness code that reads them."""

    name: str
    label: str
    summary: str   # one line for the picker: what the person gets, and what it costs


LEVELS: dict[str, DriveLevel] = {
    "relaxed": DriveLevel(
        "relaxed", "Relaxed",
        "Takes the first complete answer. Fastest and cheapest."),
    "diligent": DriveLevel(
        "diligent", "Diligent",
        "The everyday level: works until the task is done, within normal limits."),
    "persistent": DriveLevel(
        "persistent", "Persistent",
        "Keeps going longer and checks the result before delivering. Slower, costs more."),
    "ambitious": DriveLevel(
        "ambitious", "Ambitious",
        "As Persistent, and may add the obvious next step. Slowest, costs the most."),
}

_ALIASES = {"relax": "relaxed", "default": "", "auto": "", "normal": "diligent",
            "stubborn": "persistent", "persist": "persistent"}


def normalize(value: Any) -> str | None:
    """A level name from the wire or a config file, or ``None`` for "not picked"
    (empty, ``auto``, or a name that is no level)."""
    text = str(value or "").strip().lower()
    text = _ALIASES.get(text, text)
    return text if text in LEVELS else None


# ── Turn scope ────────────────────────────────────────────────────────

_TURN: ContextVar[str | None] = ContextVar("cc_drive_turn", default=None)


@contextmanager
def turn_drive(level: Any) -> Iterator[str | None]:
    """The person's pick for this turn. Tasks created inside inherit it."""
    token = _TURN.set(normalize(level))
    try:
        yield _TURN.get()
    finally:
        _TURN.reset(token)


def current_turn_drive() -> str | None:
    """What was picked for the turn that is running, if anything."""
    return _TURN.get()


def agent_default() -> str | None:
    """The agent's own default level (``drive.default``), if it has one."""
    try:
        from captain_claw.config import get_config

        return normalize(get_config().drive.default)
    except Exception:
        return None


def current_name() -> str:
    """The level in effect: the turn's pick, else the agent's default, else
    ``diligent``. Never raises."""
    return _TURN.get() or agent_default() or DEFAULT


def current() -> DriveLevel:
    return LEVELS[current_name()]


def level_to_dict(level: DriveLevel) -> dict[str, Any]:
    return {"name": level.name, "label": level.label, "summary": level.summary}


def list_levels() -> list[DriveLevel]:
    """Lowest to highest, for a picker."""
    return list(LEVELS.values())
