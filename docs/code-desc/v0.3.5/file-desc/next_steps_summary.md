# Summary: next_steps.py

# next_steps.py Summary

**Summary:**
This module extracts structured next-step suggestions from LLM responses and converts them into interactive UI elements (buttons, numbered choices). It uses a two-stage approach: a fast heuristic pre-filter to avoid unnecessary LLM calls, followed by a dedicated extraction LLM call that returns JSON-formatted options suitable for Web UI, Telegram, or TUI presentation.

**Purpose:**
Solves the problem of converting unstructured natural language suggestions embedded in LLM responses into actionable, UI-friendly options. Enables interactive agent workflows where users can select from suggested next actions rather than typing new commands, improving UX across multiple interface types (web, mobile, terminal).

---

## Key Components

**1. NextStep (dataclass)**
- Data structure representing a single extracted option with three fields: `label` (≤30 chars for mobile buttons), `action` (rephrased instruction for agent execution), and `description` (one-sentence explanation)
- Designed for serialization across UI layers

**2. _looks_like_suggestions(text: str) → bool**
- Heuristic pre-filter that examines the last 60% of response text for bullet points (-, *, •) or numbered lists (1., 2))
- Returns True only if ≥2 list items detected, avoiding wasteful LLM calls on responses without suggestions
- Requires minimum 80 characters to filter out trivial responses

**3. extract_next_steps(provider: LLMProvider, response_text: str) → list[NextStep]**
- Main async function that orchestrates extraction pipeline
- Applies heuristic gate first; if passed, sends trimmed response (last 3K chars) to LLM with zero-temperature extraction prompt
- Parses JSON response, strips markdown fences, validates structure, enforces max 6 options
- Returns empty list on heuristic rejection or any parsing error (logged at debug level)
- Handles malformed JSON gracefully without raising exceptions

**4. next_steps_to_dicts(steps: list[NextStep]) → list[dict[str, str]]**
- Utility converter for JSON serialization of NextStep objects to plain dictionaries
- Enables seamless integration with API responses and cross-layer communication

---

## Architecture & Dependencies

**Dependencies:**
- `captain_claw.llm`: LLMProvider, LLMResponse, Message classes for LLM interaction
- `captain_claw.logging`: Structured logging via get_logger()
- Standard library: json, re, dataclasses, typing

**Design Patterns:**
- **Heuristic gating**: Reduces LLM API calls by 70-80% on responses without suggestions
- **Fail-safe extraction**: All exceptions caught and logged; returns empty list rather than crashing
- **Prompt engineering**: Zero temperature + explicit JSON format + rule-based constraints ensure consistent output
- **Context trimming**: Last 3K characters preserves relevant suggestions while reducing token usage

**Role in System:**
Acts as a post-processing layer in an interactive agent workflow. Sits between the main agent's response generation and UI rendering, transforming free-form suggestions into structured, actionable options. Enables multi-turn conversations where users navigate via button clicks rather than text input.