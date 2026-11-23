"""Drive — how far the harness pushes a turn.

Reasoning effort says how hard the model thinks, a cognitive mode says how the
agent approaches work; Drive says how hard the *harness* works to finish what
was asked: how many steps a turn may take, how often a weak answer is sent
back, whether the result is checked before it is delivered and (at the top
level only) whether the agent may do a little more than was asked.

    relaxed · diligent · persistent · ambitious

``diligent`` is the harness as it was before Drive existed. Plan and the
owner's decisions: ``docs/drive-plan.md``.

A level is a fixed set of harness limits (:class:`DriveLevel`), each read by
the code named beside it. Two rules hold at every level: none lowers a guard
about honesty or safety (false-claim retries, the stuck detector, mail
authority and confirmations never read Drive), and ``relaxed`` still has to
deliver a complete answer — it only stops sending a delivered answer back.

The level reaches the harness the way reasoning effort does, through a context
variable: the chat frame sets the turn's level (:func:`turn_drive`) and tasks
created inside the turn inherit it. A turn nobody picked a level for (a
scheduled job, a hand-over, WhatsApp, Telegram, a link) runs at the agent's
own default (``drive.default`` in its config), else at the level its cognitive
mode suggests (Ionian and Phrygian: ``persistent``), else ``diligent`` — a
level picked in one chat never reaches a scheduled run.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

DEFAULT = "diligent"


@dataclass(frozen=True)
class DriveLevel:
    """One level. Every limit here is read by the harness; a test fails for
    one that is not."""

    name: str
    label: str
    summary: str   # one line for the picker: what the person gets, and what it costs

    # Step budget (agent_orchestration_mixin._turn_limits). A turn may run to
    # ``step_ceiling_multiplier`` x its planned budget, at most
    # ``step_ceiling_cap`` steps; up to ``free_extension_multiplier`` x planned,
    # progress alone earns the extra steps.
    step_ceiling_multiplier: int = 4
    step_ceiling_cap: int = 300
    free_extension_multiplier: int = 2

    # Completion gate (agent_completion_mixin): how many times an answer that
    # fails its contract is sent back before it is accepted as it is — in
    # general, and after a list run already did the work item by item.
    validation_retries: int = 2
    validation_retries_after_list: int = 1

    # How many times a reply that announces work without doing it is sent back
    # (agent_orchestration_mixin). Never below 2: that is about finishing.
    stall_retries: int = 2


LEVELS: dict[str, DriveLevel] = {
    "relaxed": DriveLevel(
        "relaxed", "Relaxed",
        "Takes the first complete answer. Fastest and cheapest.",
        step_ceiling_multiplier=2, step_ceiling_cap=100, free_extension_multiplier=1,
        validation_retries=0, validation_retries_after_list=0, stall_retries=2),
    # The harness as it was before Drive: these are the old fixed numbers.
    "diligent": DriveLevel(
        "diligent", "Diligent",
        "The everyday level: works until the task is done, within normal limits."),
    "persistent": DriveLevel(
        "persistent", "Persistent",
        "Keeps going longer and sends a weak answer back more often. Slower, costs more.",
        step_ceiling_multiplier=6, step_ceiling_cap=450, free_extension_multiplier=3,
        validation_retries=3, validation_retries_after_list=2, stall_retries=3),
    # Same limits as Persistent until its extra-work part is built (plan, phase 5).
    "ambitious": DriveLevel(
        "ambitious", "Ambitious",
        "As Persistent for now; doing the obvious next step is still to come.",
        step_ceiling_multiplier=6, step_ceiling_cap=450, free_extension_multiplier=3,
        validation_retries=3, validation_retries_after_list=2, stall_retries=3),
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


def mode_suggestion(agent: Any | None = None) -> str | None:
    """The level the agent's cognitive mode suggests, if any (its
    ``suggested_drive``). With *agent*, the mode its prompt really carries."""
    try:
        from captain_claw.cognitive_mode import MODES, effective_mode_name

        mode = MODES.get(effective_mode_name(agent))
        return normalize(mode.suggested_drive) if mode is not None else None
    except Exception:
        return None


def current_name(agent: Any | None = None) -> str:
    """The level in effect: the turn's pick, else the agent's own default,
    else what its cognitive mode suggests, else ``diligent``. Never raises."""
    return _TURN.get() or agent_default() or mode_suggestion(agent) or DEFAULT


def current(agent: Any | None = None) -> DriveLevel:
    return LEVELS[current_name(agent)]


def level_to_dict(level: DriveLevel) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(level)


def list_levels() -> list[DriveLevel]:
    """Lowest to highest, for a picker."""
    return list(LEVELS.values())
