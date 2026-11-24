"""Cognitive Mode — musical-mode-inspired reasoning strategies.

Maps seven musical modes (Ionian through Locrian) plus a neutral default
to distinct cognitive approaches.  Each mode operates at three layers:

  Layer 1 — Prompt injection: the mode's text plus a shared frame
            (``instructions/cognitive_modes/frame.md``) that says when a mode
            applies and what it never overrides.
  Layer 2 — Parameters: only numbers that some subsystem actually reads
            (nervous system, game seats).  How far the harness pushes a
            turn is Drive's job (drive.py); a mode only suggests a level.  Anything about
            *how* the agent works lives in the prompt text, not in a number.

A mode also carries a one-line ``review_lens`` for critics and checkers,
which judge an answer rather than do the work.

The mode is stored as a simple text file at
``~/.captain-claw/cognitive_mode.txt`` containing the mode name (e.g.
"phrygian").  Cached in memory with mtime-based invalidation — same
pattern as ``personality.py``.

Mode #0 "neutra" is the default no-op: no prompt injection, no parameter
overrides, no hooks.  Existing behaviour is 100% preserved unless a mode
is explicitly set and the ``cognitive_mode.enabled`` config flag is True.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)


_CONFIG_DIR = Path("~/.captain-claw").expanduser()
MODE_PATH = _CONFIG_DIR / "cognitive_mode.txt"

# Instruction files live under instructions/cognitive_modes/
_INSTRUCTION_SUBDIR = "cognitive_modes"

# ── Data model ────────────────────────────────────────────────────────


@dataclass
class CognitiveModeParams:
    """Layer 2 — numerical parameters that modify existing subsystems.

    Every field here is read by a subsystem; do not add one that is not.
    Default values are neutral — they produce no change.
    """

    # Multiplier on nervous system dream_interval_messages (1.0 = no change).
    dream_interval_modifier: float = 1.0

    # Multiplier on nervous system dream_cooldown_seconds (1.0 = no change).
    dream_cooldown_modifier: float = 1.0

    # Delta added to maturation_cycles_required (0 = no change).
    maturation_cycles_delta: int = 0

    # Multiplier on intuition surfacing priority per thread_type.
    # e.g. {"unresolved": 2.0, "connection": 0.5}
    # Types not listed default to 1.0.
    intuition_type_weights: dict[str, float] = field(default_factory=dict)

    # LLM temperature for a game seat playing in this mode (games/seats.py).
    game_temperature: float = 0.7


@dataclass
class CognitiveMode:
    """Complete mode definition."""

    id: int                     # 0–7
    name: str                   # "neutra", "ionian", …
    label: str                  # "The Resolver", "The Adversarial Analyst", …
    character: str              # One-line summary of cognitive character
    params: CognitiveModeParams
    instruction_file: str       # Filename in instructions/cognitive_modes/ (empty for neutra)
    review_lens: str = ""       # What a critic in this mode looks for (empty for neutra)
    # The Drive level (drive.py) an agent in this mode runs at when neither a
    # chat nor the agent's own settings picked one.  "" = no suggestion.
    suggested_drive: str = ""


# ── Mode Registry ────────────────────────────────────────────────────

MODES: dict[str, CognitiveMode] = {

    "neutra": CognitiveMode(
        id=0,
        name="neutra",
        label="Default",
        character="Balanced generalist — no cognitive bias in any direction",
        params=CognitiveModeParams(),  # all defaults
        instruction_file="",
    ),

    "ionian": CognitiveMode(
        id=1,
        name="ionian",
        label="The Resolver",
        character="Convergent problem-solving — seeks clear answers and closure",
        params=CognitiveModeParams(
            dream_interval_modifier=1.2,       # dreams slightly less often
            dream_cooldown_modifier=1.2,
            maturation_cycles_delta=0,
            game_temperature=0.5,
            intuition_type_weights={"pattern": 1.3, "hypothesis": 0.7},
        ),
        instruction_file="ionian.md",
        suggested_drive="persistent",
        review_lens=(
            "Does the answer settle the question that was asked: one clear, complete result that meets the stated goal? Look for an open question left behind, a hedge where a conclusion was possible, or a goal only partly met."
        ),
    ),

    "dorian": CognitiveMode(
        id=2,
        name="dorian",
        label="The Pragmatic Empath",
        character="Empathetic pragmatism — acknowledges complexity while finding workable paths",
        params=CognitiveModeParams(
            dream_interval_modifier=0.9,
            dream_cooldown_modifier=0.9,
            maturation_cycles_delta=0,
            game_temperature=0.7,
            intuition_type_weights={"connection": 1.5, "unresolved": 1.3},
        ),
        instruction_file="dorian.md",
        review_lens=(
            "Would this work for the people involved, under their real constraints? Look for ignored trade-offs, steps that are impractical for them, and costs that land on someone the answer does not mention."
        ),
    ),

    "phrygian": CognitiveMode(
        id=3,
        name="phrygian",
        label="The Adversarial Analyst",
        character="Adversarial thinking — assumes things will go wrong, finds the hidden flaw",
        params=CognitiveModeParams(
            dream_interval_modifier=0.8,       # dreams more often
            dream_cooldown_modifier=0.8,
            maturation_cycles_delta=1,          # intuitions need more cycles
            game_temperature=0.5,
            intuition_type_weights={"unresolved": 2.0, "hypothesis": 1.5, "association": 0.6},
        ),
        instruction_file="phrygian.md",
        suggested_drive="persistent",
        review_lens=(
            "Look for what would make the answer fail: a false assumption, an unhandled case, a claim that was not verified, the weakest step in the reasoning."
        ),
    ),

    "lydian": CognitiveMode(
        id=4,
        name="lydian",
        label="The Visionary Explorer",
        character="Divergent creative thinking — expands the solution space, finds unexpected connections",
        params=CognitiveModeParams(
            dream_interval_modifier=0.6,        # dreams much more often
            dream_cooldown_modifier=0.6,
            maturation_cycles_delta=-1,          # intuitions surface faster
            game_temperature=1.0,
            intuition_type_weights={"association": 2.0, "connection": 2.0, "hypothesis": 1.5, "pattern": 0.7},
        ),
        instruction_file="lydian.md",
        review_lens=(
            "Is there a clearly better approach or framing that the answer missed? Look for a narrow reading of the question and for an obvious alternative that was never weighed."
        ),
    ),

    "mixolydian": CognitiveMode(
        id=5,
        name="mixolydian",
        label="The Iterative Builder",
        character="Momentum-focused iteration — ship something, learn, improve, never declare done",
        params=CognitiveModeParams(
            dream_interval_modifier=1.3,         # dreams less (action-focused)
            dream_cooldown_modifier=1.3,
            maturation_cycles_delta=-1,
            game_temperature=0.8,
            intuition_type_weights={"pattern": 1.3, "unresolved": 0.5},
        ),
        instruction_file="mixolydian.md",
        review_lens=(
            "Is the result usable as it stands? Look for parts that are described rather than done, and for anything that would not work when actually tried."
        ),
    ),

    "aeolian": CognitiveMode(
        id=6,
        name="aeolian",
        label="The Depth Researcher",
        character="Deep analytical research — traces root causes, maps full context, values thoroughness",
        params=CognitiveModeParams(
            dream_interval_modifier=0.7,         # dreams more often
            dream_cooldown_modifier=0.7,
            maturation_cycles_delta=1,           # intuitions mature longer
            game_temperature=0.5,
            intuition_type_weights={"pattern": 2.0, "connection": 1.5, "hypothesis": 1.3},
        ),
        instruction_file="aeolian.md",
        review_lens=(
            "Check the grounds: is each important claim supported by evidence, is the cause given the real cause and not a symptom, and is relevant context or history missing?"
        ),
    ),

    "locrian": CognitiveMode(
        id=7,
        name="locrian",
        label="The Deconstructionist",
        character="Radical questioning — challenges the entire framing, questions whether the problem should exist",
        params=CognitiveModeParams(
            dream_interval_modifier=0.7,
            dream_cooldown_modifier=0.7,
            maturation_cycles_delta=0,
            game_temperature=0.9,
            intuition_type_weights={"unresolved": 2.5, "hypothesis": 2.0, "pattern": 0.5, "association": 0.5},
        ),
        instruction_file="locrian.md",
        review_lens=(
            "Question the premise: does the answer solve the right problem, is it more complicated than it needs to be, does it rest on an assumption nobody examined?"
        ),
    ),
}


# ── Lookup helpers ───────────────────────────────────────────────────


def get_mode(name: str) -> CognitiveMode:
    """Look up a mode by name.  Falls back to neutra for unknown names."""
    mode = MODES.get(name.lower().strip())
    if mode is None:
        log.warning("Unknown cognitive mode requested, falling back to neutra", requested=name)
        return MODES["neutra"]
    return mode


def get_mode_params(name: str) -> CognitiveModeParams:
    """Convenience: get just the Layer 2 params for a mode name."""
    return get_mode(name).params


def list_modes() -> list[CognitiveMode]:
    """Return all available modes sorted by id (for UI dropdowns)."""
    return sorted(MODES.values(), key=lambda m: m.id)


def mode_to_dict(mode: CognitiveMode) -> dict[str, Any]:
    """Serialize a mode for JSON APIs (Flight Deck, etc.)."""
    return {
        "id": mode.id,
        "name": mode.name,
        "label": mode.label,
        "character": mode.character,
        "instruction_file": mode.instruction_file,
        "review_lens": mode.review_lens,
        "suggested_drive": mode.suggested_drive,
        "params": {
            "dream_interval_modifier": mode.params.dream_interval_modifier,
            "dream_cooldown_modifier": mode.params.dream_cooldown_modifier,
            "maturation_cycles_delta": mode.params.maturation_cycles_delta,
            "intuition_type_weights": dict(mode.params.intuition_type_weights),
            "game_temperature": mode.params.game_temperature,
        },
    }


# ── Prompt block builder ────────────────────────────────────────────


_FRAME_FILE = "frame.md"


def cognitive_mode_to_prompt_block(
    mode: CognitiveMode,
    instruction_loader: Any | None = None,
) -> str:
    """Build the Layer 1 prompt text for a mode.

    Returns an empty string for neutra (no prompt injection).
    For other modes: a header, the mode's own text from
    ``instructions/cognitive_modes/<file>``, then the shared frame
    (``frame.md``) that scopes every mode the same way.

    *instruction_loader* should be the agent's ``InstructionLoader``
    instance.  Without one (or when the mode's file is missing) there is
    nothing useful to inject, so the result is empty.
    """
    if mode.name == "neutra" or not mode.instruction_file:
        return ""
    if instruction_loader is None:
        return ""

    template_name = f"{_INSTRUCTION_SUBDIR}/{mode.instruction_file}"
    try:
        body = instruction_loader.load(template_name).strip()
    except FileNotFoundError:
        log.warning(
            "Cognitive mode instruction file not found",
            mode=mode.name, file=template_name,
        )
        return ""

    parts = [f"COGNITIVE MODE: {mode.label} ({mode.name.capitalize()})", "", body]
    try:
        frame = instruction_loader.load(f"{_INSTRUCTION_SUBDIR}/{_FRAME_FILE}").strip()
    except FileNotFoundError:
        frame = ""
    if frame:
        parts += ["", frame]
    return "\n".join(parts)


def cognitive_mode_review_block(mode: CognitiveMode) -> str:
    """The mode as a lens for judging an answer (critics, checkers).

    A reviewer does not need the mode's working instructions — only what to
    look for.  Empty for neutra.
    """
    if mode.name == "neutra" or not mode.review_lens:
        return ""
    return f"Review lens: {mode.label}. {mode.review_lens}"


_UNSET: Any = object()


def effective_mode_name(agent: Any | None = None) -> str:
    """The mode in effect, for logs: a mode name, or ``""`` when none applies.

    With an *agent* whose system prompt has been built, this is the mode that
    prompt carried (``""`` when modes are off or the prompt has no mode slot).
    Otherwise it is what this process is set to: the mode file or the config
    default, ``""`` when modes are disabled.  Never raises.
    """
    try:
        if agent is not None:
            held = getattr(agent, "_cognitive_mode", _UNSET)
            if held is None:
                return ""
            if isinstance(held, CognitiveMode):
                return held.name
        from captain_claw.config import get_config
        if not get_config().cognitive_mode.enabled:
            return ""
        name = load_agent_mode()
        return name if name in MODES else ""
    except Exception:
        return ""


# ── File I/O with mtime caching ─────────────────────────────────────

_cached_mode_name: str | None = None
_cached_mtime: float = 0.0


def _config_default_mode() -> str:
    """Read the default mode from config, falling back to neutra."""
    try:
        from captain_claw.config import get_config
        return get_config().cognitive_mode.default_mode or "neutra"
    except Exception:
        return "neutra"


def load_agent_mode() -> str:
    """Load the current mode name from disk, with mtime-based caching.

    Resolution order:
    1. ``~/.captain-claw/cognitive_mode.txt`` (explicit override)
    2. ``cognitive_mode.default_mode`` from config.yaml
    3. ``"neutra"`` (hardcoded fallback)
    """
    global _cached_mode_name, _cached_mtime

    if not MODE_PATH.is_file():
        _cached_mode_name = _config_default_mode()
        return _cached_mode_name

    try:
        mtime = MODE_PATH.stat().st_mtime
    except OSError:
        return _cached_mode_name or "neutra"

    if _cached_mode_name is not None and mtime == _cached_mtime:
        return _cached_mode_name

    try:
        text = MODE_PATH.read_text(encoding="utf-8").strip().lower()
        if text and text in MODES:
            _cached_mode_name = text
        else:
            log.warning("Invalid mode in cognitive_mode.txt, defaulting to neutra", found=text)
            _cached_mode_name = "neutra"
        _cached_mtime = mtime
    except Exception:
        _cached_mode_name = "neutra"
        _cached_mtime = 0.0

    return _cached_mode_name


def save_agent_mode(name: str) -> None:
    """Write the mode name to ``~/.captain-claw/cognitive_mode.txt``."""
    global _cached_mode_name, _cached_mtime

    name = name.lower().strip()
    if name not in MODES:
        raise ValueError(f"Unknown cognitive mode: {name!r}. Valid: {', '.join(MODES)}")

    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    MODE_PATH.write_text(name, encoding="utf-8")

    _cached_mode_name = name
    try:
        _cached_mtime = MODE_PATH.stat().st_mtime
    except OSError:
        _cached_mtime = 0.0

    log.info("Cognitive mode saved", mode=name)
