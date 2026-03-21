"""Cognitive Tempo — processing depth detection and user rhythm modeling.

Inspired by how classical music uses tempo (adagio/allegro) and dynamic contour
to create different emotional/cognitive effects.  Analyzes conversation signals
to determine whether the current interaction calls for deep contemplative
processing (adagio) or rapid task-focused execution (allegro).

Pure heuristics — no LLM calls.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from captain_claw.config import get_config
from captain_claw.logging import get_logger

log = get_logger(__name__)

# ── Signal word sets ──────────────────────────────────────────────────

# Words indicating reflective/contemplative thinking.
_REFLECTIVE_MARKERS = frozenset({
    "wonder", "wondering", "think", "thinking", "thought",
    "feel", "feeling", "sense", "meaning", "purpose",
    "understand", "understanding", "insight", "perspective",
    "reflect", "reflecting", "contemplate", "consider",
    "curious", "ponder", "philosophy", "philosophical",
    "why", "deeper", "essence", "nature", "soul",
    "emotion", "emotional", "intuition", "intuitive",
    "pattern", "connection", "relationship", "evolve",
    "meditate", "introspect", "introspection",
    "beauty", "aesthetic", "profound", "nuance",
})

# Words indicating task-oriented/rapid execution.
_TASK_MARKERS = frozenset({
    "run", "running", "do", "doing", "show", "showing",
    "list", "listing", "create", "creating", "delete", "deleting",
    "send", "sending", "fix", "fixing", "build", "building",
    "deploy", "deploying", "install", "update", "updating",
    "check", "test", "testing", "push", "pull", "commit",
    "move", "copy", "rename", "add", "remove", "start", "stop",
    "open", "close", "save", "load", "export", "import",
    "quick", "fast", "now", "asap", "immediately",
    "next", "then", "also", "just", "simply",
})

# Question patterns indicating depth of inquiry.
_DEEP_QUESTION_PATTERN = re.compile(
    r"\b(why|how come|what if|what does .+ mean|"
    r"how .+ relate|what.+think about|"
    r"how .+ influence|what.+understand|"
    r"how can we|what.+implication|"
    r"how .+ connect|what.+pattern)\b",
    re.IGNORECASE,
)

_SHALLOW_QUESTION_PATTERN = re.compile(
    r"\b(what is|where is|how do i|"
    r"can you|show me|list|"
    r"how to|what are the|"
    r"give me|tell me the)\b",
    re.IGNORECASE,
)


# ── Data model ────────────────────────────────────────────────────────

@dataclass
class TempoState:
    """Current cognitive tempo assessment."""

    user_tempo: float           # 0.0 = deep contemplation, 1.0 = rapid execution
    content_depth: float        # 0.0 = philosophical, 1.0 = tactical
    combined_tempo: float       # Weighted blend of both
    mode: str                   # "adagio" | "moderato" | "allegro"
    signals: dict[str, Any] = field(default_factory=dict)
    assessed_at: str = ""


# ── Signal extraction ─────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Sigmoid function clamped to avoid overflow."""
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _extract_user_messages(messages: list[dict], window: int) -> list[dict]:
    """Get the last N user messages."""
    user_msgs = [m for m in messages if m.get("role") == "user"]
    return user_msgs[-window:]


def _compute_message_length_signal(user_msgs: list[dict]) -> float:
    """Average message length → lower values indicate rapid mode.

    Returns 0.0-1.0 where 0.0 = very long messages (contemplative),
    1.0 = very short messages (rapid).
    """
    if not user_msgs:
        return 0.5

    lengths = [len(str(m.get("content", ""))) for m in user_msgs]
    avg_len = sum(lengths) / len(lengths)

    # Normalize: 50 chars → rapid (1.0), 500+ chars → contemplative (0.0)
    normalized = 1.0 - min(1.0, max(0.0, (avg_len - 50) / 450))
    return normalized


def _compute_time_gap_signal(user_msgs: list[dict]) -> float:
    """Average time between messages → longer gaps indicate deliberation.

    Returns 0.0-1.0 where 0.0 = long gaps (contemplative), 1.0 = rapid fire.
    """
    if len(user_msgs) < 2:
        return 0.5

    gaps: list[float] = []
    for i in range(1, len(user_msgs)):
        ts_curr = user_msgs[i].get("timestamp") or user_msgs[i].get("created_at")
        ts_prev = user_msgs[i - 1].get("timestamp") or user_msgs[i - 1].get("created_at")
        if ts_curr and ts_prev:
            try:
                dt_curr = datetime.fromisoformat(str(ts_curr))
                dt_prev = datetime.fromisoformat(str(ts_prev))
                gap_seconds = abs((dt_curr - dt_prev).total_seconds())
                gaps.append(gap_seconds)
            except (ValueError, TypeError):
                continue

    if not gaps:
        return 0.5

    avg_gap = sum(gaps) / len(gaps)
    # Normalize: <10s → rapid (1.0), >120s → contemplative (0.0)
    normalized = 1.0 - min(1.0, max(0.0, (avg_gap - 10) / 110))
    return normalized


def _compute_question_depth_signal(user_msgs: list[dict]) -> float:
    """Ratio of deep questions to shallow questions.

    Returns 0.0-1.0 where 0.0 = deep questions (contemplative),
    1.0 = shallow/command-like (rapid).
    """
    deep_count = 0
    shallow_count = 0

    for m in user_msgs:
        text = str(m.get("content", ""))
        deep_count += len(_DEEP_QUESTION_PATTERN.findall(text))
        shallow_count += len(_SHALLOW_QUESTION_PATTERN.findall(text))

    total = deep_count + shallow_count
    if total == 0:
        return 0.5

    # More deep questions → lower value (contemplative)
    return 1.0 - (deep_count / total)


def _compute_word_complexity_signal(user_msgs: list[dict]) -> float:
    """Average word length as a crude vocabulary complexity proxy.

    Returns 0.0-1.0 where 0.0 = complex vocabulary (contemplative),
    1.0 = simple vocabulary (rapid).
    """
    all_words: list[str] = []
    for m in user_msgs:
        text = str(m.get("content", ""))
        words = re.findall(r"[a-zA-Z]+", text)
        all_words.extend(words)

    if not all_words:
        return 0.5

    avg_word_len = sum(len(w) for w in all_words) / len(all_words)
    # Normalize: 3 chars → simple/rapid (1.0), 8+ chars → complex/contemplative (0.0)
    normalized = 1.0 - min(1.0, max(0.0, (avg_word_len - 3) / 5))
    return normalized


def _compute_reflective_ratio(user_msgs: list[dict]) -> float:
    """Ratio of reflective to task-oriented language.

    Returns 0.0-1.0 where 0.0 = highly reflective (contemplative),
    1.0 = highly task-oriented (rapid).
    """
    reflective_count = 0
    task_count = 0

    for m in user_msgs:
        words = set(re.findall(r"[a-z]+", str(m.get("content", "")).lower()))
        reflective_count += len(words & _REFLECTIVE_MARKERS)
        task_count += len(words & _TASK_MARKERS)

    total = reflective_count + task_count
    if total == 0:
        return 0.5

    return 1.0 - (reflective_count / total)


# ── Main assessment ───────────────────────────────────────────────────

def assess_tempo(
    messages: list[dict],
    *,
    window: int = 5,
) -> TempoState:
    """Analyze recent messages to determine cognitive tempo.

    Returns a TempoState with mode "adagio", "moderato", or "allegro"
    based on user behavior signals and content depth analysis.

    No LLM calls — pure heuristic analysis.
    """
    cfg = get_config()

    user_msgs = _extract_user_messages(messages, window)
    if not user_msgs:
        return TempoState(
            user_tempo=0.5,
            content_depth=0.5,
            combined_tempo=0.5,
            mode="moderato",
            signals={},
            assessed_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    # User tempo signals (behavior-based).
    msg_length = _compute_message_length_signal(user_msgs)
    time_gap = _compute_time_gap_signal(user_msgs)
    question_depth = _compute_question_depth_signal(user_msgs)
    word_complexity = _compute_word_complexity_signal(user_msgs)

    # Weighted average for user tempo.
    user_tempo = (
        0.30 * msg_length +
        0.25 * time_gap +
        0.25 * question_depth +
        0.20 * word_complexity
    )

    # Content depth signals (semantic analysis).
    reflective_ratio = _compute_reflective_ratio(user_msgs)

    # Content depth is primarily driven by reflective ratio
    # but modulated by question depth for cross-validation.
    content_depth = (
        0.70 * reflective_ratio +
        0.30 * question_depth
    )

    # Combined tempo — weighted blend.
    user_weight = cfg.cognitive_tempo.user_tempo_weight
    content_weight = cfg.cognitive_tempo.content_depth_weight
    combined = user_weight * user_tempo + content_weight * content_depth

    # Determine mode.
    adagio_thresh = cfg.cognitive_tempo.adagio_threshold
    allegro_thresh = cfg.cognitive_tempo.allegro_threshold

    if combined < adagio_thresh:
        mode = "adagio"
    elif combined > allegro_thresh:
        mode = "allegro"
    else:
        mode = "moderato"

    signals = {
        "msg_length_signal": round(msg_length, 3),
        "time_gap_signal": round(time_gap, 3),
        "question_depth_signal": round(question_depth, 3),
        "word_complexity_signal": round(word_complexity, 3),
        "reflective_ratio_signal": round(reflective_ratio, 3),
        "user_msgs_analyzed": len(user_msgs),
    }

    state = TempoState(
        user_tempo=round(user_tempo, 3),
        content_depth=round(content_depth, 3),
        combined_tempo=round(combined, 3),
        mode=mode,
        signals=signals,
        assessed_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )

    log.debug("Cognitive tempo assessed",
              mode=mode, combined=round(combined, 3),
              user=round(user_tempo, 3), content=round(content_depth, 3))

    return state
