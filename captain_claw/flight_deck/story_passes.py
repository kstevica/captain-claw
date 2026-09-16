"""Story-integrity MODEL passes (P1 + P2) — the judgment half of the ten passes.

P0 (``story_state``) runs the countable/relational passes in pure code. The passes
here need a model's judgment and read the merged draft + the extracted store on the
strong (reason) tier:

  C  Physical mechanism  — replay the crime against the store: can the victim/antagonist
                           perform each action in the available time, WHY does the victim
                           enter the danger, is a safety escape ignored, does the mechanism
                           work on the researched timescale. (P1)
  F  Research claims     — the story overstates what a source proves: a possibility is not
                           a proof; a present result does not establish a past state;
                           "not found" is not "proved absent"; jargon is not understanding. (P1)
  I  Narrative economy   — repeated summaries, duplicate scenes, merged voices, scenes that
                           don't change the investigation, endings that repeat a function. (P1)
  J  Emotional fairness  — a late revelation must be seeded (≥ 2 earlier signals); the ending
                           must follow from the story, not be appended. (P1)
  CA Claim-attacker      — §9: adversarially test each MATERIAL claim (timings, what a
                           finding proves, jurisdiction, what a current result says about the
                           past) and flag overstatements. (P2)
  PR Professional        — correct authority/jurisdiction/procedural status/warrant/caution/
                           counsel/chain-of-custody; a confession must be procedurally
                           coherent, made of words actually spoken, and never the sole
                           support. (P2)

Each pass returns findings in the SAME shape as story_state (pass/kind/severity/
scene/reason/quotes), so bucket()/blocking_analysis()/the gate treat them uniformly.
Severity returned by the model is CLAMPED to each pass's ceiling, so a soft pass
(I) can never emit a done-blocking hard error.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Awaitable, Callable

from captain_claw.logging import get_logger

log = get_logger(__name__)

CompleteFn = Callable[[str], Awaitable[str]]

DRAFT_CAP = 60_000   # chars of the draft handed to a model pass (context-bounded)
STORE_CAP = 24_000   # chars of the compact store JSON


def _window(draft: str, cap: int = DRAFT_CAP) -> tuple[str, bool]:
    """Fit a long draft into *cap* chars WITHOUT dropping the ending — head + tail,
    middle elided. Pass J (does the ending follow / is a twist seeded) and the
    payoff passes must see BOTH the setup and the resolution, so a plain head cut
    would false-flag the ending. Returns (windowed_text, truncated?)."""
    d = draft or ""
    if len(d) <= cap:
        return d, False
    head = int(cap * 0.6)
    tail = cap - head
    return (d[:head] + "\n\n…[middle of the draft elided for length; "
            "head and ending shown]…\n\n" + d[-tail:]), True


_SEV_SYNONYMS = {
    "critical": "hard", "blocker": "hard", "blocking": "hard", "fatal": "hard",
    "severe": "hard", "high": "hard",
    "medium": "major", "moderate": "major", "warning": "major", "minor": "soft",
    "low": "soft", "info": "soft", "nit": "soft",
}


def _clamp(sev: str, ceiling: str) -> str:
    order = {"soft": 0, "major": 1, "hard": 2}
    raw = str(sev or "").strip().lower()
    raw = _SEV_SYNONYMS.get(raw, raw)  # a weak model may label a blocker 'critical' etc.
    s = order.get(raw, 1)
    c = order.get(ceiling, 2)
    return ("soft", "major", "hard")[min(s, c)]


# ── each pass: id, human name, ceiling severity, prompt builder ───────

def _p_physical(draft: str, store: str) -> str:
    return (
        "You are the PHYSICAL-MECHANISM checker for a mystery/thriller. Replay the crime "
        "minute-by-minute against the structured state and the prose. Flag ONLY concrete "
        "impossibilities you can point to, with a short exact QUOTE. Ask: can the victim "
        "perform every assigned action; WHY does the victim enter the dangerous location "
        "(is it caused and shown, not just convenient); can they escape via any visible "
        "safety mechanism; does one device make another redundant; can the antagonist "
        "complete every action in the available time; would an obvious witness/camera/log "
        "identify them at once; does anyone have to be in two places at once; does the "
        "mechanism operate on a realistic timescale. severity: 'hard' for a true "
        "impossibility, a false essential mechanism, or an UNCAUSED victim entry into the "
        "trap; 'major' for a strained-but-possible mechanism. If nothing is clearly "
        "impossible, return an empty list."
    )


def _p_research(draft: str, store: str) -> str:
    return (
        "You are the RESEARCH-CLAIMS checker. Flag where the story OVERSTATES what a source "
        "or finding proves. Discipline: a possibility is not a proof; a PRESENT result does "
        "not establish a PAST state; 'not found' is not 'proved absent'; technical vocabulary "
        "is not causal understanding. For each, quote the overstatement. severity: 'major' "
        "for an overstated claim; 'hard' only when the overstatement is the ESSENTIAL "
        "mechanism the solution depends on. Empty list if none."
    )


def _p_economy(draft: str, store: str) -> str:
    return (
        "You are the NARRATIVE-ECONOMY checker. Flag repetition and dead weight: repeated "
        "summaries of what the reader already knows, duplicate institutional-pressure scenes, "
        "two characters who read as one merged voice, scenes that do not change the "
        "investigation, and endings that repeat a function already served. severity is "
        "always 'soft' (never blocks). Empty list if the draft is economical."
    )


def _p_fairness(draft: str, store: str) -> str:
    return (
        "You are the EMOTIONAL-FAIRNESS / FAIR-PLAY checker. For each late revelation or "
        "twist, decide: was it SEEDED earlier (at least two earlier signals the reader could "
        "notice in hindsight), and does the ending FOLLOW from the story rather than being "
        "appended? severity: 'hard' when an UNSEEDED fact is REQUIRED to solve the case; "
        "'major' for a late-introduced motive, an under-seeded twist, or an appended ending. "
        "Quote the revelation. Empty list if the payoffs are fair."
    )


def _p_claim_attacker(draft: str, store: str) -> str:
    return (
        "You are the CLAIM-ATTACKER (§9). For each MATERIAL claim the plot depends on — how "
        "long a chemical/biological/mechanical process takes, how a machine or safety device "
        "operates, what a forensic/medical finding proves, what telecom/records capture, what "
        "a lab test confirms, what a missing sample means, what a CURRENT result proves about "
        "the PAST, which authority has jurisdiction, what warrant/order/caution is required — "
        "ATTACK it: what must be true for the scene to work, and does the claim hold? Use your "
        "knowledge adversarially; do not look for supporting material. Flag a claim that is "
        "overstated, impossible on the stated timescale, or that turns 'possible'→'established' "
        "/ 'not found'→'proved absent' / a current finding into a historical one without a "
        "bridge. severity: 'hard' when the false claim is load-bearing for the solution; "
        "'major' otherwise. Quote the claim. Empty list if all material claims hold."
    )


def _p_procedure(draft: str, store: str) -> str:
    return (
        "You are the PROFESSIONAL-PROCEDURE checker. Flag procedural errors that would "
        "invalidate the case: the wrong authority/jurisdiction acting; a warrant/order/caution/"
        "right-to-counsel that is required but missing; a voluntary conversation that turns "
        "accusatory without its procedural status changing; a broken chain of custody used as "
        "key proof; and CONFESSION problems — a confession that is procedurally unusable, is "
        "not made of words actually spoken (it merely echoes the investigator's summary), or "
        "is the SOLE support for the culprit. severity: 'hard' for a procedurally-unusable "
        "confession as sole support or a case-invalidating procedural error; 'major' "
        "otherwise. Quote the passage. Empty list if procedure is sound."
    )


# id → (name, ceiling, prompt_fn)
PASSES: dict[str, tuple] = {
    "C": ("physical mechanism", "hard", _p_physical),
    "F": ("research claims", "hard", _p_research),
    "I": ("narrative economy", "soft", _p_economy),
    "J": ("emotional fairness", "hard", _p_fairness),
    "CA": ("claim-attacker", "hard", _p_claim_attacker),
    "PR": ("professional procedure", "hard", _p_procedure),
}

# The passes each phase enables (P1 = C/F/I/J; P2 adds CA/PR).
P1_PASSES = ("C", "F", "I", "J")
P2_PASSES = ("CA", "PR")
ALL_PASSES = P1_PASSES + P2_PASSES

_INSTR = (
    "\n\nReturn ONLY this JSON (no prose):\n"
    '{"findings": [{"kind": "<short_slug>", "severity": "hard|major|soft", '
    '"scene": "<the earliest scene/chapter it starts in>", "reason": "<one sentence>", '
    '"fix": "<the smallest change that repairs the CAUSE — may simplify the plot>", '
    '"quote": "<exact short quote from the draft>"}]}\n\n'
)


def _parse(output) -> list[dict]:
    """Parse a model pass's reply. Accepts the {\"findings\": [...]} object OR a bare
    top-level [...] array (a weak model often drops the wrapper). Coerces any
    non-string input so a provider returning a parsed value never raises."""
    text = str(output or "").strip()
    if not text:
        return []
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, text, re.DOTALL)
        if not m:
            continue
        try:
            raw = json.loads(m.group(0))
        except (ValueError, TypeError):
            continue
        if isinstance(raw, dict):
            fs = raw.get("findings")
            if isinstance(fs, list):
                return [f for f in fs if isinstance(f, dict)]
        if isinstance(raw, list):
            return [f for f in raw if isinstance(f, dict)]
    return []


async def run_model_pass(pass_id: str, draft: str, store_json: str,
                         model_fn: CompleteFn) -> list[dict]:
    """Run one model pass; return findings tagged + severity-clamped."""
    spec = PASSES.get(pass_id)
    if not spec:
        return []
    name, ceiling, prompt_fn = spec
    windowed, truncated = _window(draft)
    body = prompt_fn(windowed, store_json[:STORE_CAP])
    # When the middle is elided, the fairness/seeding pass (J) cannot reliably tell a
    # genuinely-seeded twist from an unseeded one (seeds live in the middle act), so it
    # must not HARD-block on that basis — downgrade its ceiling to major (surface, don't
    # block). The deterministic clue-payoff pass (H), which reads the store extracted
    # from the FULL draft, still covers unpaid high-salience clues.
    ceil = "major" if (truncated and pass_id == "J") else ceiling
    trunc_note = ("\n\n(NOTE: the draft's middle was elided for length; you can see the "
                  "opening and the ending. Do NOT flag a payoff as missing, and do NOT "
                  "conclude a twist is unseeded, on that basis — the setup you can't see "
                  "may contain the seeds.)" if truncated else "")
    prompt = (body + _INSTR
              + "STORE (structured state):\n" + store_json[:STORE_CAP]
              + "\n\nDRAFT:\n" + windowed + trunc_note)
    try:
        out = await model_fn(prompt)
        parsed = _parse(out)
    except Exception as e:  # noqa: BLE001 — a pass must never crash the run
        log.warning("story model pass failed", pass_id=pass_id, error=str(e))
        return []
    findings = []
    for f in parsed:
        sev = _clamp(f.get("severity", "major"), ceil)
        findings.append({
            "pass": pass_id, "kind": str(f.get("kind") or name.replace(" ", "_")),
            "severity": sev, "source": "story_integrity",
            "reason": str(f.get("reason") or "")[:400],
            "scene": str(f.get("scene") or ""),
            "detail": str(f.get("reason") or "")[:400],
            "fix": str(f.get("fix") or "")[:400],
            "quotes": [str(f.get("quote") or "")] if f.get("quote") else [],
        })
    return findings


async def run_model_passes(draft: str, store_json: str, model_fn: CompleteFn, *,
                           passes: tuple = ALL_PASSES,
                           on_progress: Callable[[str], None] | None = None) -> list[dict]:
    """Run the enabled model passes concurrently; return the merged findings."""
    def _note(m: str) -> None:
        if on_progress:
            try:
                on_progress(m)
            except Exception:  # noqa: BLE001
                pass

    enabled = [p for p in passes if p in PASSES]
    if not (draft or "").strip() or not enabled:
        return []
    _note(f"Integrity model passes: {', '.join(PASSES[p][0] for p in enabled)}…")
    results = await asyncio.gather(
        *[run_model_pass(p, draft, store_json, model_fn) for p in enabled],
        return_exceptions=True)
    out: list[dict] = []
    for r in results:
        if isinstance(r, list):
            out.extend(r)
    return out
