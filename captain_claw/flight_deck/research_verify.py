"""R8 — grounded claim verification for research deliverables.

The Horizon closer (Deep mode) verifies a deliverable with tool-LESS critics, so
it can only refute on internal coherence — it cannot catch a stale statute, a
wrong version number, or a fabricated citation, because it has no way to look
anything up. This is C1 for research: every research output makes objectively
checkable claims (citations, dates, versions, figures, named entities), and the
web (plus the run's own source corpus) is the test runner.

This module is the pure, mode-agnostic part: the fact-checker prompt, the parser
for its findings, and the builder that turns confirmed errors into a revision
instruction. The actual agent spawn + the revision call are done by the caller
(Basna after merge, Vatra after the reporter), so this stays testable with no
model or network.
"""

from __future__ import annotations

import json
import re

from captain_claw.logging import get_logger

log = get_logger(__name__)

_VALID_VERDICTS = ("confirmed", "refuted", "unverifiable")


def claim_check_prompt(deliverable: str, question: str, max_claims: int,
                       corpus_hint: bool = False) -> str:
    """Prompt a tool-enabled agent to verify the deliverable's load-bearing claims."""
    corpus = (
        "You also have this run's saved sources: use the `researchmap` tool to "
        "search them and read any `vfs:<project>/sources/` file. "
        if corpus_hint else ""
    )
    return (
        "You are a rigorous fact-checker. Below is a research deliverable. Verify its "
        "most load-bearing VERIFIABLE claims against authoritative sources — do NOT "
        "rewrite or critique its style.\n\n"
        f"1. Identify the {max_claims} most important checkable claims: citations and "
        "their identifier/version numbers, statutes/standards and whether they are "
        "current or superseded, dates, named entities and their attributes, and "
        "quantitative figures. Prefer claims whose error would materially mislead a "
        "reader.\n"
        f"2. Verify EACH by searching the web and reading primary sources. {corpus}"
        "Fetch the authoritative source when a claim hinges on a specific number, "
        "version, or status.\n"
        "3. Report ONLY as a JSON array — no prose before or after:\n"
        '   [{"claim": "<the claim, quoted briefly>", '
        '"verdict": "confirmed|refuted|unverifiable", '
        '"correction": "<the correct fact, ONLY if refuted>", '
        '"evidence": "<url or source name>"}]\n'
        "Be conservative: mark \"refuted\" ONLY when you found a concrete, better "
        "source proving the claim wrong; mark \"unverifiable\" if you genuinely could "
        "not check it. Never invent a source. If everything checks out, return [].\n\n"
        "## Deliverable to verify\n"
        f"{deliverable}"
    )


def parse_findings(output: str) -> list[dict]:
    """Pull the JSON findings array out of the fact-checker's reply. Tolerant of
    code fences and surrounding prose. Returns a normalised list (may be empty)."""
    if not output:
        return []
    text = output.strip()
    # Prefer a fenced block if present.
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:
        # Else the outermost [...] array.
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
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict") or "").strip().lower()
        if verdict not in _VALID_VERDICTS:
            verdict = "unverifiable"
        claim = str(item.get("claim") or "").strip()
        if not claim:
            continue
        out.append({
            "claim": claim[:400],
            "verdict": verdict,
            "correction": str(item.get("correction") or "").strip()[:400],
            "evidence": str(item.get("evidence") or "").strip()[:300],
        })
    return out


def refuted(findings: list[dict]) -> list[dict]:
    """The subset that was verified WRONG with a concrete correction."""
    return [f for f in findings if f.get("verdict") == "refuted" and f.get("correction")]


def fix_instructions(findings: list[dict]) -> str:
    """Turn confirmed errors into an ordered correction list for one revision pass.

    Returns "" when there is nothing to fix, so the caller can skip the revision.
    """
    bad = refuted(findings)
    if not bad:
        return ""
    lines = [
        "The following claims in the deliverable were fact-checked against sources "
        "and found INCORRECT. Correct each one precisely, preserving everything else:",
    ]
    for i, f in enumerate(bad, 1):
        ev = f" [source: {f['evidence']}]" if f.get("evidence") else ""
        lines.append(f"{i}. Claim: {f['claim']}\n   Correct it to: {f['correction']}{ev}")
    return "\n".join(lines)


def summary_line(findings: list[dict]) -> str:
    """A one-line tally for the live log."""
    n = len(findings)
    c = sum(1 for f in findings if f["verdict"] == "confirmed")
    r = sum(1 for f in findings if f["verdict"] == "refuted")
    u = n - c - r
    return f"{n} claim(s) checked · {c} confirmed · {r} refuted · {u} unverifiable"
