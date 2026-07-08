"""R8 — grounded claim verification for research deliverables.

The Horizon closer (Deep mode) verifies a deliverable with tool-LESS critics, so
it can only refute on internal coherence — it cannot catch a stale statute, a
wrong version number, or a fabricated citation, because it has no way to look
anything up. This is C1 for research: every research output makes objectively
checkable claims (citations, dates, versions, figures, named entities), and the
web (plus the run's own source corpus) is the test runner.

Two verdicts are actionable, not one:

* **refuted** — a source proves the claim wrong → correct it.
* **unverifiable-but-asserted** — the deliverable states a load-bearing specific
  (a named individual, a role-holder, an origin, an exact figure/attribution) as
  established fact, but no source confirms it. A tool-less critic cannot catch
  this, and it is the single most dangerous research failure: a *confident
  fabrication* asserted as fact. You cannot "refute" it (you can't prove a
  negative), so the fix is to **hedge** it — qualify it honestly rather than
  leave the bare assertion. This is the exact difference between a "be thorough"
  model that invents a named officer and a careful one that writes "unconfirmed".

This module is the pure, mode-agnostic part: the fact-checker prompt, the parser
for its findings, the builder that turns confirmed errors + unconfirmable
assertions into one revision instruction, and the renderer for a standalone
audit ledger. The actual agent spawn + the revision call are done by the caller
(Basna after merge, Vatra after the reporter), so this stays testable with no
model or network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from captain_claw.logging import get_logger

log = get_logger(__name__)

_VALID_VERDICTS = ("confirmed", "refuted", "unverifiable")

#: Suffix for the standalone fact-check audit document written next to the
#: deliverable, so the user sees WHAT was checked and HOW it was resolved rather
#: than a silently-mutated report.
AUDIT_SUFFIX = ".fact-check.md"


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
        '"hedge": "<see rule below, ONLY if unverifiable>", '
        '"evidence": "<url or source name>"}]\n'
        "Rules:\n"
        "- \"refuted\": use ONLY when you found a concrete, better source proving the "
        "claim wrong. Put the correct fact in \"correction\".\n"
        "- \"unverifiable\": use when you genuinely could not confirm the claim from any "
        "authoritative source. This is the DANGEROUS case when the deliverable states "
        "the claim as unqualified FACT and it is a specific — a named individual or "
        "role-holder (e.g. an appointed officer), an origin or affiliation, an exact "
        "date/figure/identifier, or a specific attribution. For such an "
        "asserted-but-unconfirmable specific you MUST fill \"hedge\": the honestly-"
        "qualified rewrite of that sentence (mark it \"unconfirmed\" / \"not "
        "independently verified\" / \"reportedly\", or attribute it to whoever asserts "
        "it) — never leave a fabricated-looking specific stated as established fact. If "
        "the deliverable ALREADY qualifies the claim, leave \"hedge\" empty.\n"
        "- Never invent a source. Be conservative. If everything checks out and nothing "
        "needs qualifying, return [].\n\n"
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
            "hedge": str(item.get("hedge") or "").strip()[:400],
            "evidence": str(item.get("evidence") or "").strip()[:300],
        })
    return out


def refuted(findings: list[dict]) -> list[dict]:
    """The subset that was verified WRONG with a concrete correction."""
    return [f for f in findings if f.get("verdict") == "refuted" and f.get("correction")]


def unconfirmed(findings: list[dict]) -> list[dict]:
    """Unverifiable specifics the deliverable asserted as fact — carry a ``hedge``.

    These are the fabrication-risk claims: no source confirms them, so they can't
    be *refuted*, but they were stated as established fact and must be qualified
    rather than left standing. The fact-checker only fills ``hedge`` when the
    deliverable currently asserts the claim without qualification, so an already-
    hedged claim never shows up here.
    """
    return [f for f in findings if f.get("verdict") == "unverifiable" and f.get("hedge")]


def fix_instructions(findings: list[dict]) -> str:
    """Turn confirmed errors + unconfirmable assertions into one ordered revision
    instruction. Returns "" when there is nothing to change, so the caller can
    skip the revision pass entirely.
    """
    bad = refuted(findings)
    soft = unconfirmed(findings)
    if not bad and not soft:
        return ""
    lines = [
        "The following claims in the deliverable were fact-checked against sources. "
        "Apply EACH change precisely and preserve everything else — keep all other "
        "content, structure and formatting identical:",
    ]
    n = 0
    for f in bad:
        n += 1
        ev = f" [source: {f['evidence']}]" if f.get("evidence") else ""
        lines.append(f"{n}. INCORRECT — {f['claim']}\n   Correct it to: {f['correction']}{ev}")
    for f in soft:
        n += 1
        ev = f" [checked: {f['evidence']}]" if f.get("evidence") else ""
        lines.append(
            f"{n}. UNCONFIRMED — {f['claim']}\n"
            f"   No authoritative source confirms this; do NOT keep it stated as "
            f"established fact. Rewrite it as: {f['hedge']}{ev}")
    return "\n".join(lines)


def summary_line(findings: list[dict]) -> str:
    """A one-line tally for the live log."""
    n = len(findings)
    c = sum(1 for f in findings if f["verdict"] == "confirmed")
    r = sum(1 for f in findings if f["verdict"] == "refuted")
    u = n - c - r
    line = f"{n} claim(s) checked · {c} confirmed · {r} refuted · {u} unverifiable"
    h = len(unconfirmed(findings))
    if h:
        line += f" · {h} hedged"
    return line


# ── Audit ledger (the standalone "<deliverable>.fact-check.md") ────────

def _cell(text: str) -> str:
    """Make a value safe for a one-line markdown table cell."""
    return (text or "").replace("|", "\\|").replace("\n", " ").strip() or "—"


def audit_markdown(findings: list[dict], *, question: str, revised: bool) -> str:
    """Render the fact-check findings as a standalone audit document.

    This is the visible, non-destructive record: every checked claim with its
    verdict, the source, and the action taken (corrected / hedged / left as-is).
    It is written even when nothing was auto-changed, so unverifiable specifics
    are surfaced rather than silently dropped from the log tally.
    """
    n_bad, n_soft = len(refuted(findings)), len(unconfirmed(findings))
    header = [
        "# Fact-check report",
        "",
        f"**Question:** {(question or '').strip()[:500]}",
        "",
        f"**Result:** {summary_line(findings)}",
        (f"**Action:** {n_bad} correction(s) + {n_soft} hedge(s) "
         + ("applied to the deliverable." if revised else "recommended (revision not applied).")
         if (n_bad or n_soft) else "**Action:** nothing needed correcting."),
        "",
    ]
    if not findings:
        header.append("_No load-bearing claims were flagged._")
        return "\n".join(header)

    def _action(f: dict) -> str:
        if f["verdict"] == "refuted" and f.get("correction"):
            return f"→ corrected to: {f['correction']}"
        if f["verdict"] == "unverifiable" and f.get("hedge"):
            return f"→ hedged: {f['hedge']}"
        if f["verdict"] == "confirmed":
            return "verified"
        return "no source found; left as-is"

    rows = ["| # | Verdict | Claim | Finding / action | Evidence |",
            "|---|---------|-------|------------------|----------|"]
    for i, f in enumerate(findings, 1):
        rows.append(
            f"| {i} | {_cell(f['verdict'])} | {_cell(f['claim'])} | "
            f"{_cell(_action(f))} | {_cell(f.get('evidence', ''))} |")
    return "\n".join(header + rows) + "\n"


def write_audit(dest_dir: Path, findings: list[dict], *, question: str,
                revised: bool, base_name: str = "deliverable") -> dict | None:
    """Write the audit ledger next to the deliverable and return a generated-file
    descriptor the caller registers on the session, or ``None`` if there was
    nothing to record or the write failed. Best-effort — never raises.
    """
    if not findings:
        return None
    name = f"{base_name}{AUDIT_SUFFIX}"
    try:
        md = audit_markdown(findings, question=question, revised=revised)
        p = Path(dest_dir) / name
        p.write_text(md, encoding="utf-8")
        return {"name": name, "mime": "text/markdown", "size": p.stat().st_size,
                "kind": "generated", "agent": "fact-checker"}
    except OSError as e:  # noqa: BLE001 — audit doc is best-effort
        log.warning("fact-check audit write failed", error=str(e))
        return None
