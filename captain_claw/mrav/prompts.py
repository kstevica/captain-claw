"""Mrav step prompts — small, byte-stable, schema-described.

Constraints that shaped these (see docs/mrav-micro-agent-plan.md):
- 2-4B models are extremely format-sensitive: wording stays frozen; any
  change must go through the eval harness.
- Constrained decoding does NOT inject the schema into the prompt, so the
  contract describes the exact JSON shapes itself.
- The ACT contract + toolpack render identically every step (prefix-cache /
  browser KV delta-prefill depend on it); only the tail sections change.
"""

from __future__ import annotations

ACT_CONTRACT = """You are Mrav, a careful operator that works in small steps toward the TASK.

Reply with EXACTLY ONE JSON object — no prose, no markdown, no code fences. Choose ONE action:
{"action":"tool","tool":"<name>","args":{...}}  run one tool from TOOLS
{"action":"open_tool","name":"<name>"}  load the schema of a tool listed in INDEX
{"action":"final","text":"<answer>"}  finish with the complete answer for the user
{"action":"give_up","reason":"<why>"}  only if the task is truly impossible. One failed attempt is NOT impossible — try another tool or approach first

Rules:
- ONE action per reply. Never more than one tool call.
- Tools in TOOLS are ready — call them directly, never open_tool them.
- open_tool is ONLY for tools listed in INDEX.
- Give every required arg (marked *). Use exact arg names.
- Ground statements in OBSERVATIONS. They may be truncated — re-check when unsure.
- If ERROR is shown, fix the cause; never send the same reply again.
- Use "final" as soon as the TASK is done. Answer factually and completely."""

PLAN_CONTRACT = """You are Mrav's planner. Reply with EXACTLY ONE JSON object, no prose:
{"plan":["step 1","step 2",...]}
2-6 short, concrete steps to finish the TASK with the available tools. Each step one line."""

DIGEST_INSTRUCTION = (
    "Condense the {label} below to at most {words} words of facts useful for the TASK. "
    "Keep file paths, numbers, names, commands and error messages exact. "
    "Output only the condensed text, no preamble."
)

COMPRESS_INSTRUCTION = (
    "Rewrite the RUNNING SUMMARY to include the OLDER OBSERVATIONS below. "
    "Keep task-relevant facts, decisions, paths and results; drop noise. "
    "At most {words} words. Output only the new summary, no preamble."
)

# Section headers — single source of truth so runtime and tests agree.
H_TOOLS = "## TOOLS"
H_INDEX = "## INDEX (open_tool to use)"
H_TASK = "## TASK"
H_PLAN = "## PLAN"
H_FACTS = "## FACTS"
H_SUMMARY = "## RUNNING SUMMARY"
H_OBSERVATIONS = "## OBSERVATIONS"
H_ERROR = "## ERROR (fix this)"
H_NOW = "## NOW"

ACT_NOW = "Decide the single next action as one JSON object."
PLAN_NOW = "Write the plan as one JSON object."


def model_options(model: str) -> dict[str, object]:
    """Per-model request quirks for the micro loop.

    Thinking modes multiply output tokens 3-10x — always off in Mrav. The
    Ollama provider handles think-capable detection; we just force it off.
    """
    return {"think": False}
