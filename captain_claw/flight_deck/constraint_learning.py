"""R2 — the learning edge as a CONSTRAINT edge (opt-in `constraint_learning`).

The automatic loop today distills only a routing WEIGHT
(db.record_archetype_outcome). The real *rule* path — the playbook distiller
(agent_playbook_mixin._distill_playbook_from_session) — is human-triggered and
main-agent only, and the Code/Vatra splitters and planners get nothing learned.
This module is the automatic, structured-rule analogue: from an accepted (or
fixed-failure) run it distills ONE short reusable CONSTRAINT —
{trigger, constraint, severity, domain} — persisted domain-scoped
(db.learned_constraints) and injected into the briefs that learn nothing today.

Everything here is inert unless the `constraint_learning` flag is on: the route
files only import it inside that gate. Pure module (stdlib only): the distiller
prompt (adapted from the playbook distiller), a tolerant parser, a compact signal
builder, and the bounded injection renderer. The routes own the model call + DB.
"""
from __future__ import annotations

import json
import re

from captain_claw.logging import get_logger

log = get_logger(__name__)

_SEVERITIES = ("critical", "major", "minor")
_MAX_INJECT = 5
_SIGNAL_CAP = 6000


def build_signal(deliverable: str, gaps: list[str] | None = None, *,
                 fixed_notes: str = "") -> str:
    """Compact 'what this run produced / had to fix' text the distiller reads."""
    parts = [(deliverable or "").strip()[:_SIGNAL_CAP]]
    if fixed_notes.strip():
        parts.append("Issues that had to be fixed before it passed:\n"
                     + fixed_notes.strip()[:2000])
    for g in (gaps or [])[:8]:
        if str(g).strip():
            parts.append(f"- gap: {str(g).strip()[:200]}")
    return "\n\n".join(p for p in parts if p).strip()


def distill_prompt(intent: str, signal: str, outcome_kind: str = "accepted") -> str:
    kind = ("was accepted as good" if outcome_kind == "accepted"
            else "passed only after failures were fixed")
    return (
        "You are distilling ONE reusable CONSTRAINT from a finished task that "
        f"{kind}. A constraint is a short, GENERAL rule that should guide FUTURE "
        "similar tasks — an anti-pattern to avoid or a non-negotiable to honor — "
        "NOT a fact about this specific task. Generalize: strip specific names, "
        "paths, numbers, and one-off details.\n\n"
        "Reply ONLY with a JSON object, no prose:\n"
        '{"trigger": "<when this rule applies, one short clause>", '
        '"constraint": "<the rule, one imperative sentence>", '
        '"severity": "critical|major|minor", '
        '"domain": "<one lowercase word for the task area, e.g. web, data, api, '
        "research, finance; '' if unclear>\"}\n"
        "Keep it a SINGLE rule, concrete and checkable in spirit. If nothing "
        'generalizable was learned, reply {"constraint": ""}.\n\n'
        f"## Task\n{intent[:2000]}\n\n"
        f"## What the run produced / had to fix\n{signal[:8000]}"
    )


def parse_constraint(output: str) -> dict | None:
    """Normalize the distiller reply into {trigger, constraint, severity, domain}
    or None when there is nothing usable. Tolerant; never raises."""
    if not output:
        return None
    text = output.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:
        s, e = text.find("{"), text.rfind("}")
        blob = text[s:e + 1] if 0 <= s < e else None
    if not blob:
        return None
    try:
        raw = json.loads(blob)
    except (ValueError, TypeError):
        return None
    if not isinstance(raw, dict):
        return None
    constraint = str(raw.get("constraint") or "").strip()[:300]
    if not constraint:
        return None
    sev = str(raw.get("severity") or "").strip().casefold()
    if sev not in _SEVERITIES:
        sev = "major"
    domain = re.sub(r"[^a-z0-9_-]", "",
                    str(raw.get("domain") or "").strip().casefold())[:40]
    trigger = str(raw.get("trigger") or "").strip()[:200]
    return {"trigger": trigger, "constraint": constraint,
            "severity": sev, "domain": domain}


def format_constraints_block(rows: list[dict] | None, max_n: int = _MAX_INJECT) -> str:
    """Bounded injection block, or '' when there is nothing to inject. Accepts
    both DB rows (trigger_text/constraint_text) and parsed dicts (trigger/
    constraint), so it renders whichever the caller has."""
    rows = [r for r in (rows or [])
            if (r.get("constraint_text") or r.get("constraint"))][:max(0, int(max_n))]
    if not rows:
        return ""
    lines = ["\n\n## Learned constraints (from earlier accepted runs) — honor these",
             "Short rules distilled from past runs in this area. Follow them unless "
             "the task clearly overrides one:"]
    for r in rows:
        sev = str(r.get("severity") or "major")
        trig = str(r.get("trigger_text") or r.get("trigger") or "").strip()
        rule = str(r.get("constraint_text") or r.get("constraint") or "").strip()
        lines.append(f"- [{sev}] " + (f"when {trig}: " if trig else "") + rule)
    return "\n".join(lines)
