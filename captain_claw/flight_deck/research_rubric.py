"""R9 — rubric-from-source contract for research deliverables.

The biggest structural gap in the Fable-vs-Vatra comparison was scope: the task
said "ICO template," and the ensemble *self-declared* a reduced 11-field subset
and routed mandatory elements out to "separate policies," while Fable used the
full field set. Each worker guessed what "complete" meant.

R9 fixes that by deriving the completeness rubric ONCE, with the run's best
reasoning tier — from the authoritative standard the task names, not from each
worker's assumption — and injecting it into every worker + the reporter as the
definition of "complete." The coverage judge then scores the deliverable field by
field against that rubric. Derive-once, follow-everywhere is what lets weak worker
models hit a gold-standard structure.

This module is the pure part (prompts + parsing + the injected directive); the
LLM call is made by the caller with whatever provider it already has.
"""

from __future__ import annotations

import json
import re

from captain_claw.logging import get_logger

log = get_logger(__name__)

_MAX_ITEMS = 30


def derive_rubric_prompt(intent: str) -> str:
    """Ask a reasoning model to enumerate the completeness checklist for the task."""
    return (
        "You are scoping a research deliverable. Produce the COMPLETENESS CHECKLIST "
        "that a gold-standard answer must satisfy — the definition of 'complete' for "
        "this task.\n\n"
        "If the task references a specific standard, template, framework, or "
        "specification (a named template, an ISO/RFC/legal article, a required report "
        "structure, etc.), enumerate that standard's ACTUAL required fields / sections "
        "/ elements — the FULL set as the standard defines them, not a simplified "
        "subset. Mandatory elements of the standard belong IN the deliverable; do not "
        "route them out to a 'separate document'.\n"
        "If no standard is named, enumerate the sections and dimensions a rigorous, "
        "complete answer to this task must cover.\n\n"
        f"Task:\n{intent}\n\n"
        "Return ONLY a JSON array of short checklist items (strings), most important "
        "first — covering both structure (what sections/fields must exist) and "
        "substance (what each must contain). 8–25 items. No prose, just the array."
    )


def parse_rubric(output: str) -> list[str]:
    """Extract the checklist (list of strings) from the model's reply. Tolerant of
    fences / prose. Returns [] if nothing usable."""
    if not output:
        return []
    text = output.strip()
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:
        start, end = text.find("["), text.rfind("]")
        blob = text[start:end + 1] if 0 <= start < end else None
    if not blob:
        return []
    try:
        raw = json.loads(blob)
    except (ValueError, TypeError):
        return []
    if not isinstance(raw, list):
        return []
    items: list[str] = []
    for it in raw:
        s = (it if isinstance(it, str) else json.dumps(it)).strip()
        if s:
            items.append(s[:240])
    return items[:_MAX_ITEMS]


def rubric_directive(items: list[str]) -> str:
    """The block injected into worker + reporter prompts — the completeness contract."""
    if not items:
        return ""
    body = "\n".join(f"- {it}" for it in items)
    return (
        "\n\n## Completeness checklist (the deliverable MUST cover ALL of these)\n"
        f"{body}\n"
        "Treat this as the definition of 'complete'. Cover every item that applies; "
        "if one is genuinely not applicable, say so explicitly rather than omitting "
        "it. Do NOT route required elements out to a separate document — include them."
    )


def coverage_prompt(intent: str, rubric: list[str], deliverable: str) -> str:
    """Score the deliverable against the rubric — which checklist items are missing
    or thin. Returns a prompt whose JSON reply matches ``parse_coverage``."""
    checklist = "\n".join(f"- {it}" for it in rubric)
    return (
        "Score this deliverable against a required completeness checklist. For each "
        "checklist item, decide whether the deliverable covers it FULLY, PARTIALLY, or "
        "NOT AT ALL.\n\n"
        f"## Task\n{intent}\n\n## Completeness checklist\n{checklist}\n\n"
        f"## Deliverable\n{deliverable}\n\n"
        "Return ONLY a JSON object:\n"
        '{"missing": ["<checklist item not covered at all>", ...], '
        '"thin": ["<item covered only partially, and what is missing>", ...]}\n'
        "Be strict but fair — only list genuine gaps."
    )


def parse_coverage(output: str) -> dict:
    """Parse the coverage judge's reply into ``{missing:[...], thin:[...]}``."""
    if not output:
        return {"missing": [], "thin": []}
    text = output.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:
        start, end = text.find("{"), text.rfind("}")
        blob = text[start:end + 1] if 0 <= start < end else None
    try:
        raw = json.loads(blob) if blob else {}
    except (ValueError, TypeError):
        raw = {}
    miss = [str(x).strip() for x in (raw.get("missing") or []) if str(x).strip()]
    thin = [str(x).strip() for x in (raw.get("thin") or []) if str(x).strip()]
    return {"missing": miss[:_MAX_ITEMS], "thin": thin[:_MAX_ITEMS]}
