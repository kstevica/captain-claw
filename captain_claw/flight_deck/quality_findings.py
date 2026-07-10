"""Blocking quality gate — unified findings + the bounded fix-until-clean loop.

Increments 2–5 produce findings (consistency criticals, contract failures,
claim-check verdicts) but every one is advisory: recorded, never enforced. This
module is the enforcement half: map each lever's native output into ONE finding
shape, build a single triaged checklist across them, revise the deliverable,
and loop until the criticals clear — bounded by rounds and budget.

The load-bearing rule: **the loop is driven only by findings that plain code
can re-verify from the revised text** — consistency identity/relation criticals.
A contract violation whose values live in the facts ledger, or a text-vs-ledger
mismatch, cannot be fixed by rewriting prose (the revision would just drift the
text away from the ledger, and a re-check against the now-stale ledger would
never converge). Those go INTO the revision checklist (the reviser does its
best) and into the final verdict, but they never drive another round: ground
truth blocks, opinion and stale state only report.

A run NEVER loses work here: cap/budget exhaustion or a collapsed revision ends
the loop with the best text so far and a ``critical_findings_remain`` verdict —
visible, actionable (a fill-gaps round), not discarded.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from captain_claw.flight_deck.quality_profile import TokenBudget
from captain_claw.logging import get_logger

log = get_logger(__name__)

#: Consistency finding kinds the loop may drive on — re-verifiable from the
#: text alone, no external state.
LOOP_KINDS = frozenset({"identity", "relation"})

CompleteFn = Callable[[str], Awaitable[str]]


# ── mappers: each lever's native output → the unified finding shape ────

def from_consistency(findings: list[dict] | None) -> list[dict]:
    """``research_consistency.verify()`` findings → unified findings."""
    return [{"source": "consistency", "kind": str(f.get("kind") or ""),
             "severity": str(f.get("severity") or "minor"),
             "detail": str(f.get("detail") or "")}
            for f in findings or []]


def from_contract(failed: list[dict] | None) -> list[dict]:
    """``research_contract.summarize()['failed']`` entries → unified findings."""
    out = []
    for f in failed or []:
        detail = str(f.get("text") or "")
        if f.get("note"):
            detail += f" — {f['note']}"
        out.append({"source": "contract", "kind": str(f.get("how") or "deterministic"),
                    "severity": str(f.get("severity") or "major"), "detail": detail})
    return out


def from_claim_check(findings: list[dict] | None) -> list[dict]:
    """``research_verify`` findings → unified findings (refuted = critical,
    asserted-but-unconfirmable = major). Advisory-only today — the claim check
    applies its own correction — but the mapping keeps the shape uniform for
    verdicts/metrics."""
    out = []
    for f in findings or []:
        if f.get("verdict") == "refuted":
            detail = f"refuted claim: {f.get('claim', '')}"
            if f.get("correction"):
                detail += f" → {f['correction']}"
            out.append({"source": "claim_check", "kind": "refuted",
                        "severity": "critical", "detail": detail})
        elif f.get("verdict") == "unverifiable" and f.get("hedge"):
            out.append({"source": "claim_check", "kind": "unconfirmed",
                        "severity": "major",
                        "detail": f"asserted but unconfirmable: {f.get('claim', '')}"})
    return out


def criticals(findings: list[dict]) -> list[dict]:
    return [f for f in findings if f.get("severity") == "critical"]


def loop_drivers(findings: list[dict]) -> list[dict]:
    """The subset the loop may block on: consistency criticals re-verifiable
    from text alone."""
    return [f for f in criticals(findings)
            if f.get("source") == "consistency" and f.get("kind") in LOOP_KINDS]


# ── checklist + revision prompt ────────────────────────────────────────

def blocking_checklist(findings: list[dict]) -> str:
    """One deduped, numbered fix list across all sources (the R3 triage shape)."""
    seen: set[str] = set()
    items: list[str] = []
    for f in findings:
        key = f.get("detail", "").strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        hint = {
            "consistency": "Make the values consistent everywhere they appear.",
            "contract": "Recompute/adjust so the rule holds, propagate the change "
                        "everywhere, and note the correction.",
            "claim_check": "Apply the correction (or hedge the claim honestly).",
        }.get(f.get("source", ""), "")
        items.append(f"{len(items) + 1}. [{f.get('source')}] {f['detail']}"
                     + (f"\n   {hint}" if hint else ""))
    if not items:
        return ""
    return ("These CRITICAL findings block finalization. Fix EACH precisely and "
            "keep everything else identical:\n" + "\n".join(items))


def revise_prompt(deliverable: str, checklist: str) -> str:
    return (
        "Below is a deliverable that failed its final quality gate, followed by "
        "the blocking findings. Output the FULL corrected deliverable — apply "
        "exactly these fixes and keep all other content, structure, and "
        "formatting identical.\n\n"
        f"## Blocking findings\n{checklist}\n\n"
        f"## Deliverable\n{deliverable}\n\n"
        "Output the complete corrected deliverable only — no preamble, no "
        "commentary."
    )


# ── the gate ───────────────────────────────────────────────────────────

async def run_gate(
    text: str, *,
    findings: list[dict],
    revise_fn: CompleteFn,
    consistency_recheck_fn: Callable[[str], Awaitable[list[dict]]] | None = None,
    max_rounds: int = 2,
    budget: TokenBudget | None = None,
    est: int = 8192,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Bounded fix loop. Returns ``{text, revised, rounds, remaining, verdict}``
    where ``verdict`` is ``"clean"`` or ``"critical_findings_remain"`` and
    ``remaining`` holds the surviving criticals (non-driver criticals that got a
    kept revision are annotated as applied-but-unverified rather than dropped).

    ``consistency_recheck_fn(text) -> verify() findings`` re-checks a candidate
    revision from the text alone (no ledger — a gate revision may deliberately
    move the text away from stale ledger values). Without it, one revision is
    applied blind and drivers are demoted to applied-unverified.
    """
    def _note(msg: str) -> None:
        if on_progress:
            try:
                on_progress(msg)
            except Exception:  # noqa: BLE001 — progress is cosmetic
                pass

    all_criticals = criticals(findings)
    if not all_criticals:
        return {"text": text, "revised": False, "rounds": 0,
                "remaining": [], "verdict": "clean"}

    drivers = loop_drivers(findings)
    passengers = [f for f in all_criticals if f not in drivers]
    current, rounds, revised_any = text, 0, False

    while rounds < max_rounds and (drivers or (passengers and not revised_any)):
        if budget is not None and not budget.can_afford(est):
            _note("Blocking gate: token budget reached — stopping with findings open")
            break
        if budget is not None:
            budget.add(est)
        rounds += 1
        checklist = blocking_checklist(drivers + passengers)
        _note(f"Blocking gate round {rounds}: revising against "
              f"{len(drivers) + len(passengers)} critical finding(s)…")
        revised = ((await revise_fn(revise_prompt(current, checklist))) or "").strip()
        collapsed = not revised or (len(current) > 800 and len(revised) < 0.5 * len(current))
        if collapsed:
            _note("Blocking gate: revision collapsed — kept the prior text")
            break
        prev_text, prev_revised, prev_drivers = current, revised_any, drivers
        current, revised_any = revised, True
        if drivers:
            if consistency_recheck_fn is None:
                # No way to re-verify — demote drivers to applied-unverified.
                for f in drivers:
                    f["note"] = "revision applied — not re-verified"
                passengers = drivers + passengers
                drivers = []
                break
            new_drivers = loop_drivers(from_consistency(await consistency_recheck_fn(current)))
            if len(new_drivers) > len(prev_drivers):
                # The "fix" made things worse — revert and stop.
                current, revised_any, drivers = prev_text, prev_revised, prev_drivers
                _note("Blocking gate: revision made the check worse — reverted")
                break
            drivers = new_drivers
            _note(f"Blocking gate round {rounds}: "
                  f"{len(drivers)} driver critical(s) remain after re-check")
        else:
            break  # passengers-only: one best-effort revision, no loop

    if revised_any:
        for f in passengers:
            f.setdefault("note", "revision applied — not deterministically re-verifiable")
    remaining = drivers + passengers
    verdict = "clean" if not remaining else "critical_findings_remain"
    return {"text": current, "revised": revised_any, "rounds": rounds,
            "remaining": remaining, "verdict": verdict}
