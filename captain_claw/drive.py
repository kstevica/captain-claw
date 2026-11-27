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
own default, else at the level its cognitive
mode suggests (Ionian and Phrygian: ``persistent``), else ``diligent`` — a
level picked in one chat never reaches a scheduled run.

The agent's own default is ``~/.captain-claw/drive.txt`` (a level name, or
``auto`` for "none of its own"). Flight Deck's agent settings write it and the
agent reads it on every turn, so a change applies from its next message
without a restart. With no file, ``drive.default`` in the config decides.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

DEFAULT = "diligent"
FILENAME = "drive.txt"
SETTING_PATH = Path("~/.captain-claw").expanduser() / FILENAME
# The extra-work switch, live like the default above: ``on`` or ``off``.
INITIATIVE_FILENAME = "drive_initiative.txt"
INITIATIVE_PATH = Path("~/.captain-claw").expanduser() / INITIATIVE_FILENAME
# A level forced from outside for the agent's whole run — a benchmark harness
# (captain-claw-once --lock). It beats the turn's pick, the agent's own
# drive_level tool, its default and its mode; the tool refuses to change it.
LOCK_ENV = "CLAW_LOCK_DRIVE"


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

    # Check before delivering (agent_completion_mixin._delivery_check): one
    # extra model call compares a tool-using turn's reply with what was asked,
    # once, and sends it back when something asked for is missing.
    delivery_check: bool = False

    # Extra work (agent_completion_mixin, agent_guard_mixin): once the asked
    # work is delivered, the agent may add one obvious next step, with
    # reading and file-writing tools only, using at most ``initiative_share``
    # of a turn's base step budget in tool calls. The agent's own switch
    # can turn it off (:func:`initiative_enabled`); it is on by default.
    initiative: bool = False
    initiative_share: float = 0.0

    # What the model is told about the level (agent_context_mixin appends it
    # to the turn's system prompt, after the cached part). The limits above
    # only bite on long turns; this is what makes a short turn go further or
    # stop sooner. Empty at Diligent: its prompt is the prompt as it was.
    guidance: str = ""


# Each says how far to go, in words that fit any kind of work. None changes
# what counts as true or as done, and none outranks what the person asked.
_GUIDE_FRAME = (
    " This sets how far to go on substantial work. It never changes what "
    "counts as true or finished, and what the person asked for outranks it: "
    "a short question still gets a short answer."
)
_GUIDE_RELAXED = (
    "Drive for this turn: Relaxed. The person wants this quick and cheap. "
    "Take the shortest path to a correct, complete answer: use what is "
    "already at hand first, look up only what the answer needs, and stop "
    "when it is answered. Do not widen the search or polish." + _GUIDE_FRAME
)
_GUIDE_PERSISTENT = (
    "Drive for this turn: Persistent. This work has to hold up, and the "
    "person accepts that it takes longer and costs more. Go past the first "
    "adequate result: use more sources or more passes than you would by "
    "default, read the primary material rather than summaries of it, check "
    "the main claims against something independent, and cover the parts a "
    "careful reader would look for." + _GUIDE_FRAME
)
_GUIDE_AMBITIOUS = (
    "Drive for this turn: Ambitious. The person wants the most thorough "
    "result you can produce, and accepts the time and cost. Go well past the "
    "first adequate result: cast a wide net before narrowing, read the "
    "primary material rather than summaries of it, follow the leads that "
    "turn up, check the main claims against something independent, and "
    "include the detail someone would otherwise have to ask for next. Where "
    "a quicker and a deeper way both exist, take the deeper one." + _GUIDE_FRAME
)

LEVELS: dict[str, DriveLevel] = {
    "relaxed": DriveLevel(
        "relaxed", "Relaxed",
        "Takes the first complete answer. Fastest and cheapest.",
        step_ceiling_multiplier=2, step_ceiling_cap=100, free_extension_multiplier=1,
        validation_retries=0, validation_retries_after_list=0, stall_retries=2,
        guidance=_GUIDE_RELAXED),
    # The harness as it was before Drive: these are the old fixed numbers.
    "diligent": DriveLevel(
        "diligent", "Diligent",
        "The everyday level: works until the task is done, within normal limits."),
    "persistent": DriveLevel(
        "persistent", "Persistent",
        "Keeps going longer and checks the result before delivering. Slower, costs more.",
        step_ceiling_multiplier=6, step_ceiling_cap=450, free_extension_multiplier=3,
        validation_retries=3, validation_retries_after_list=2, stall_retries=3,
        delivery_check=True, guidance=_GUIDE_PERSISTENT),
    # Persistent's limits, plus the extra-work step.
    "ambitious": DriveLevel(
        "ambitious", "Ambitious",
        "As Persistent, and may add one obvious next step nobody asked for. "
        "Slowest, costs the most.",
        step_ceiling_multiplier=6, step_ceiling_cap=450, free_extension_multiplier=3,
        validation_retries=3, validation_retries_after_list=2, stall_retries=3,
        delivery_check=True, initiative=True, initiative_share=0.25,
        guidance=_GUIDE_AMBITIOUS),
}

_ALIASES = {"relax": "relaxed", "default": "", "auto": "", "normal": "diligent",
            "stubborn": "persistent", "persist": "persistent"}


def normalize(value: Any) -> str | None:
    """A level name from the wire or a config file, or ``None`` for "not picked"
    (empty, ``auto``, or a name that is no level)."""
    text = str(value or "").strip().lower()
    text = _ALIASES.get(text, text)
    return text if text in LEVELS else None


def locked() -> str | None:
    """The level forced for this run (:data:`LOCK_ENV`), or ``None``."""
    return normalize(os.environ.get(LOCK_ENV, ""))


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


def render(level: Any) -> str:
    """What the setting file holds for *level* (``auto`` = no default of its own)."""
    return normalize(level) or "auto"


def agent_default() -> str | None:
    """The agent's own default level, if it has one: the setting file when
    there is one (``auto`` in it means none), else ``drive.default``."""
    try:
        return normalize(SETTING_PATH.read_text(encoding="utf-8"))
    except OSError:
        pass
    except Exception:
        return None
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
    """The level in effect: one forced for the run, else the turn's pick, else
    the agent's own default, else what its cognitive mode suggests, else
    ``diligent``. Never raises."""
    forced = locked()
    if forced:
        return forced
    override = getattr(agent, "_turn_drive_override", None) if agent is not None else None
    if isinstance(override, str):
        # What the agent set for the rest of the turn (its drive_level tool):
        # a level, or "auto" = no pick, so its default decides.
        if override in LEVELS:
            return override
        if override == "auto":
            return agent_default() or mode_suggestion(agent) or DEFAULT
    return _TURN.get() or agent_default() or mode_suggestion(agent) or DEFAULT


def current(agent: Any | None = None) -> DriveLevel:
    return LEVELS[current_name(agent)]


def parse_switch(value: Any) -> bool | None:
    """``on``/``off`` as a setting file holds it; None for anything else."""
    text = str(value or "").strip().lower()
    if text in ("on", "true", "1", "yes"):
        return True
    if text in ("off", "false", "0", "no"):
        return False
    return None


def render_switch(on: Any) -> str:
    return "on" if on else "off"


def prompt_note(agent: Any | None = None) -> str:
    """The line the turn's system prompt carries for the level in effect
    ("" at Diligent). Never raises."""
    try:
        return current(agent).guidance
    except Exception:
        return ""


def initiative_enabled() -> bool:
    """The agent's own switch for extra work: on unless it was turned off.
    The setting file when it says so, else ``drive.initiative``. Off when
    the setting cannot be read at all. Never raises."""
    try:
        own = parse_switch(INITIATIVE_PATH.read_text(encoding="utf-8"))
        if own is not None:
            return own
    except OSError:
        pass
    except Exception:
        return False
    try:
        from captain_claw.config import get_config

        return bool(get_config().drive.initiative)
    except Exception:
        return False


# ── Extra work: what may run ──────────────────────────────────────────

# File-writing tools an extra-work step may use, beside every tool call that
# only reads. Everything else is refused: nothing that runs code, reaches
# anyone, spends, schedules, or changes a store the person keeps.
EXTRA_WORK_WRITE_TOOLS = frozenset({"write", "edit", "docx", "xlsx", "pptx", "visualize"})
# Reading tools that are still not for an unasked step: they put a question
# to the person.
EXTRA_WORK_NEVER = frozenset({"ask_user", "ask_human", "request_secret"})
NOTHING_TO_ADD = "NOTHING TO ADD"


def extra_work_allows(tool: str, arguments: Any = None) -> bool:
    """Whether an extra-work step may call *tool*: a call that only reads (as
    Flight Deck's tool-call register classes it, from the tool's name and
    action alone), or one of the file-writing tools. Unknown tools are not
    allowed."""
    from captain_claw.flight_deck.tool_risk import risk_for

    name = str(tool or "").strip().lower()
    if name in EXTRA_WORK_NEVER:
        return False
    if name in EXTRA_WORK_WRITE_TOOLS:
        return True
    return risk_for(name, arguments) == "read"


def level_to_dict(level: DriveLevel) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(level)


def list_levels() -> list[DriveLevel]:
    """Lowest to highest, for a picker."""
    return list(LEVELS.values())
