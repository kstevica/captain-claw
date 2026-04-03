"""Cognitive Mode — musical-mode-inspired reasoning strategies.

Maps seven musical modes (Ionian through Locrian) plus a neutral default
to distinct cognitive approaches.  Each mode operates at three layers:

  Layer 1 — Prompt injection (instruction text shaping *how* the agent thinks)
  Layer 2 — Parameterized config (numerical modifiers for existing subsystems)
  Layer 3 — Behavioral hooks (checked by orchestration, completion, dreaming)

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

    All values are designed to compose with existing config via
    multiplication (modifiers) or addition (deltas).  Default values
    are neutral — they produce no change.
    """

    # Cognitive tempo starting bias: 0.0 = adagio-leaning, 1.0 = allegro-leaning.
    tempo_bias: float = 0.5

    # Minimum read-type tool calls before the agent should write/execute.
    # Enforced as guidance in the prompt, not as a hard gate.
    tool_read_before_act: int = 0

    # How strict the completion gate should be: 0.0 = loose, 1.0 = strict.
    completion_strictness: float = 0.5

    # Max clarifying questions before acting.
    question_budget: int = 2

    # How many alternative approaches to consider before committing.
    exploration_breadth: int = 1

    # Multiplier on response length tendency (1.0 = no change).
    response_length_bias: float = 1.0

    # Minimum confidence to state something as fact.
    confidence_threshold: float = 0.5

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

    # Hint for the reflection system — what should self-assessment focus on.
    reflection_focus: str = ""


@dataclass
class CognitiveMode:
    """Complete mode definition."""

    id: int                     # 0–7
    name: str                   # "neutra", "ionian", …
    label: str                  # "The Resolver", "The Adversarial Analyst", …
    character: str              # One-line summary of cognitive character
    params: CognitiveModeParams
    instruction_file: str       # Filename in instructions/cognitive_modes/ (empty for neutra)


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
            tempo_bias=0.6,
            tool_read_before_act=0,
            completion_strictness=0.9,
            question_budget=1,
            exploration_breadth=1,
            response_length_bias=0.8,
            confidence_threshold=0.7,
            dream_interval_modifier=1.2,       # dreams slightly less often
            dream_cooldown_modifier=1.2,
            maturation_cycles_delta=0,
            intuition_type_weights={"pattern": 1.3, "hypothesis": 0.7},
            reflection_focus="Did I resolve tasks completely? Did I deliver clear, definitive answers?",
        ),
        instruction_file="ionian.md",
    ),

    "dorian": CognitiveMode(
        id=2,
        name="dorian",
        label="The Pragmatic Empath",
        character="Empathetic pragmatism — acknowledges complexity while finding workable paths",
        params=CognitiveModeParams(
            tempo_bias=0.45,
            tool_read_before_act=1,
            completion_strictness=0.6,
            question_budget=2,
            exploration_breadth=2,
            response_length_bias=1.1,
            confidence_threshold=0.5,
            dream_interval_modifier=0.9,
            dream_cooldown_modifier=0.9,
            maturation_cycles_delta=0,
            intuition_type_weights={"connection": 1.5, "unresolved": 1.3},
            reflection_focus="Did I account for human context and real-world constraints? Did I surface tradeoffs honestly?",
        ),
        instruction_file="dorian.md",
    ),

    "phrygian": CognitiveMode(
        id=3,
        name="phrygian",
        label="The Adversarial Analyst",
        character="Adversarial thinking — assumes things will go wrong, finds the hidden flaw",
        params=CognitiveModeParams(
            tempo_bias=0.35,
            tool_read_before_act=2,
            completion_strictness=0.8,
            question_budget=3,
            exploration_breadth=2,
            response_length_bias=1.2,
            confidence_threshold=0.8,
            dream_interval_modifier=0.8,       # dreams more often
            dream_cooldown_modifier=0.8,
            maturation_cycles_delta=1,          # intuitions need more cycles
            intuition_type_weights={"unresolved": 2.0, "hypothesis": 1.5, "association": 0.6},
            reflection_focus="What risks did I miss? What assumptions went unchallenged? What edge cases were ignored?",
        ),
        instruction_file="phrygian.md",
    ),

    "lydian": CognitiveMode(
        id=4,
        name="lydian",
        label="The Visionary Explorer",
        character="Divergent creative thinking — expands the solution space, finds unexpected connections",
        params=CognitiveModeParams(
            tempo_bias=0.4,
            tool_read_before_act=1,
            completion_strictness=0.3,
            question_budget=2,
            exploration_breadth=4,              # consider many alternatives
            response_length_bias=1.3,
            confidence_threshold=0.4,           # comfortable with speculation
            dream_interval_modifier=0.6,        # dreams much more often
            dream_cooldown_modifier=0.6,
            maturation_cycles_delta=-1,          # intuitions surface faster
            intuition_type_weights={"association": 2.0, "connection": 2.0, "hypothesis": 1.5, "pattern": 0.7},
            reflection_focus="Did I explore genuinely different approaches? Did I find unexpected connections? Did I challenge conventional thinking?",
        ),
        instruction_file="lydian.md",
    ),

    "mixolydian": CognitiveMode(
        id=5,
        name="mixolydian",
        label="The Iterative Builder",
        character="Momentum-focused iteration — ship something, learn, improve, never declare done",
        params=CognitiveModeParams(
            tempo_bias=0.75,                    # naturally fast
            tool_read_before_act=0,
            completion_strictness=0.3,           # loose — ship early
            question_budget=0,                   # just start building
            exploration_breadth=1,
            response_length_bias=0.7,            # concise
            confidence_threshold=0.4,
            dream_interval_modifier=1.3,         # dreams less (action-focused)
            dream_cooldown_modifier=1.3,
            maturation_cycles_delta=-1,
            intuition_type_weights={"pattern": 1.3, "unresolved": 0.5},
            reflection_focus="Did I ship something useful? Did I maintain momentum? Did I avoid over-planning?",
        ),
        instruction_file="mixolydian.md",
    ),

    "aeolian": CognitiveMode(
        id=6,
        name="aeolian",
        label="The Depth Researcher",
        character="Deep analytical research — traces root causes, maps full context, values thoroughness",
        params=CognitiveModeParams(
            tempo_bias=0.2,                     # naturally slow/contemplative
            tool_read_before_act=3,             # read extensively before acting
            completion_strictness=0.7,
            question_budget=2,
            exploration_breadth=2,
            response_length_bias=1.5,            # detailed responses
            confidence_threshold=0.6,
            dream_interval_modifier=0.7,         # dreams more often
            dream_cooldown_modifier=0.7,
            maturation_cycles_delta=1,           # intuitions mature longer
            intuition_type_weights={"pattern": 2.0, "connection": 1.5, "hypothesis": 1.3},
            reflection_focus="Did I research thoroughly? Did I trace to root causes? Did I miss important context or history?",
        ),
        instruction_file="aeolian.md",
    ),

    "locrian": CognitiveMode(
        id=7,
        name="locrian",
        label="The Deconstructionist",
        character="Radical questioning — challenges the entire framing, questions whether the problem should exist",
        params=CognitiveModeParams(
            tempo_bias=0.3,
            tool_read_before_act=2,
            completion_strictness=0.2,           # may reject the task itself
            question_budget=5,                   # asks many questions
            exploration_breadth=3,
            response_length_bias=1.0,
            confidence_threshold=0.3,            # low confidence in any single answer
            dream_interval_modifier=0.7,
            dream_cooldown_modifier=0.7,
            maturation_cycles_delta=0,
            intuition_type_weights={"unresolved": 2.5, "hypothesis": 2.0, "pattern": 0.5, "association": 0.5},
            reflection_focus="Did I challenge assumptions? Did I identify accidental complexity? Did I suggest simplification or removal?",
        ),
        instruction_file="locrian.md",
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
        "params": {
            "tempo_bias": mode.params.tempo_bias,
            "tool_read_before_act": mode.params.tool_read_before_act,
            "completion_strictness": mode.params.completion_strictness,
            "question_budget": mode.params.question_budget,
            "exploration_breadth": mode.params.exploration_breadth,
            "response_length_bias": mode.params.response_length_bias,
            "confidence_threshold": mode.params.confidence_threshold,
            "dream_interval_modifier": mode.params.dream_interval_modifier,
            "dream_cooldown_modifier": mode.params.dream_cooldown_modifier,
            "maturation_cycles_delta": mode.params.maturation_cycles_delta,
            "intuition_type_weights": dict(mode.params.intuition_type_weights),
            "reflection_focus": mode.params.reflection_focus,
        },
    }


# ── Prompt block builder ────────────────────────────────────────────


def cognitive_mode_to_prompt_block(
    mode: CognitiveMode,
    instruction_loader: Any | None = None,
) -> str:
    """Build the Layer 1 prompt text for a mode.

    Returns an empty string for neutra (no prompt injection).
    For other modes, loads the instruction file from
    ``instructions/cognitive_modes/<file>`` and wraps it.

    *instruction_loader* should be the agent's ``InstructionLoader``
    instance.  If ``None``, returns only the header (no instruction body).
    """
    if mode.name == "neutra" or not mode.instruction_file:
        return ""

    def _esc(s: str) -> str:
        return s.replace("{", "{{").replace("}", "}}")

    parts: list[str] = []
    parts.append(f"COGNITIVE MODE: {_esc(mode.label)} ({mode.name.upper()})")
    parts.append(f"Character: {_esc(mode.character)}")
    parts.append("")

    # Load mode-specific instruction body.
    if instruction_loader is not None:
        template_name = f"{_INSTRUCTION_SUBDIR}/{mode.instruction_file}"
        try:
            body = instruction_loader.load(template_name)
            parts.append(body)
        except FileNotFoundError:
            log.warning(
                "Cognitive mode instruction file not found",
                mode=mode.name, file=template_name,
            )
            parts.append(f"(Mode instruction file missing: {template_name})")
    else:
        parts.append("(Mode instructions not loaded — no instruction loader available)")

    return "\n".join(parts)


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
