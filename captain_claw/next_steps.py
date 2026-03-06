"""Extract suggested next steps from LLM responses for interactive selection.

When the agent's final response suggests next steps, this module runs a
quick follow-up LLM call to extract structured options that can be
presented as interactive buttons (Web UI, Telegram) or numbered choices (TUI).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from captain_claw.llm import LLMProvider, LLMResponse, Message
from captain_claw.logging import get_logger

log = get_logger(__name__)


@dataclass
class NextStep:
    """A single extracted next-step option."""

    label: str        # Short caption (max 30 chars, for Telegram buttons)
    action: str       # Rephrased instruction for the agent
    description: str  # One-sentence description (for web/TUI)


# ---------------------------------------------------------------------------
# Heuristic pre-filter
# ---------------------------------------------------------------------------

# Matches bullet points (-, *, •) or numbered items (1., 2)) near end of text.
_BULLET_RE = re.compile(
    r"^[\s]*(?:[-*•]|\d{1,2}[.)]) .+",
    re.MULTILINE,
)


def _looks_like_suggestions(text: str) -> bool:
    """Quick check: does the response likely contain suggested next steps?"""
    if not text or len(text) < 80:
        return False
    # Look for bullet/numbered lists in the last 60% of the text.
    cutoff = max(0, len(text) * 4 // 10)
    tail = text[cutoff:]
    matches = _BULLET_RE.findall(tail)
    return len(matches) >= 2


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = (
    "You are a JSON extraction assistant. You extract suggested next steps "
    "from an AI assistant's response. Return ONLY a JSON array, no markdown "
    "fences, no explanation."
)

_EXTRACTION_USER = """Extract the suggested next steps/options from the assistant's response below.
Return a JSON array of objects. Each object has:
- "label": short button caption (max 30 characters, suitable for a small mobile button)
- "action": the task rephrased as a clear imperative instruction that an AI agent can execute (be specific, include context from the original response)
- "description": one-sentence description of what this option does

Rules:
- Only extract EXPLICIT suggestions/options the assistant offered to the user. Do not invent new ones.
- If there are no suggested next steps or options, return an empty array: []
- Maximum 6 options.
- Labels must be concise — they appear as buttons on a small screen.

Assistant's response:
{response_text}"""


async def extract_next_steps(
    provider: LLMProvider,
    response_text: str,
    *,
    max_tokens: int = 600,
) -> list[NextStep]:
    """Extract suggested next steps from an LLM response.

    Returns a list of NextStep objects, or empty list if none found.
    """
    if not response_text or not response_text.strip():
        return []

    # Quick heuristic gate — avoid LLM call for responses without lists.
    if not _looks_like_suggestions(response_text):
        return []

    try:
        messages = [
            Message(role="system", content=_EXTRACTION_SYSTEM),
            Message(
                role="user",
                content=_EXTRACTION_USER.format(
                    response_text=response_text[-3000:]  # Trim to last 3K chars
                ),
            ),
        ]

        llm_response: LLMResponse = await provider.complete(
            messages=messages,
            tools=None,
            temperature=0,
            max_tokens=max_tokens,
        )

        raw = (llm_response.content or "").strip()
        if not raw:
            return []

        # Strip markdown code fences if present.
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return []

        steps: list[NextStep] = []
        for item in parsed[:6]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()[:30]
            action = str(item.get("action", "")).strip()
            description = str(item.get("description", "")).strip()
            if not label or not action:
                continue
            steps.append(NextStep(label=label, action=action, description=description))

        return steps

    except Exception as e:
        log.debug("Next steps extraction failed", error=str(e))
        return []


def next_steps_to_dicts(steps: list[NextStep]) -> list[dict[str, str]]:
    """Convert NextStep list to plain dicts for JSON serialization."""
    return [
        {"label": s.label, "action": s.action, "description": s.description}
        for s in steps
    ]
