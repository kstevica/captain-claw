"""Story-state store + deterministic story-integrity validator (P0).

The Story Integrity & Finalization Protocol (docs/story-integrity-protocol.md).
A weak model cannot be trusted to self-run a ten-pass audit over ~18k words, so
integrity lives in **structured state + deterministic checks**, not a longer
prompt. This module is P0: a post-draft validator that

  1. EXTRACTS a structured store from the merged draft (one reason-tier model call
     per chunk — the only model work here), then
  2. runs six DETERMINISTIC passes over the store in pure code (no tokens, no
     miscounting, no lost context), then
  3. classifies findings by severity and drives a bounded patch-revise loop.

The store is authoritative; the checks fail SAFE — a contradiction must be clearly
present in the extracted data to fire, so a sparse extraction misses rather than
false-blocks. Later phases (P1/P2) add the pre-draft simulation stages, the model
passes (C/F/I/J), and the claim-attacker research gate; this module is the
deterministic 80/20.

Passes implemented here (deterministic):
  A Chronology     — future-evidence cited before it exists; a character in two
                     places at one time.
  B Knowledge      — a character acts on a fact they do not yet know.
  D Provenance     — an object with two disappearance histories; a record/confession
                     asserting an unestablished fact; a broken custody chain still
                     used as key proof; generic evidence used as individual identity.
  E Hypothesis     — an elimination reused across roles; "the pool is one" asserted
                     while more than one candidate remains.
  G Quantities     — a value/identity repeated with mismatched values; age vs date.
  H Clue payoff    — a high-salience clue neither paid off, disproved, nor left
                     open for a stated reason.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Awaitable, Callable

from captain_claw.flight_deck.research_canon import chunk_text
from captain_claw.logging import get_logger

log = get_logger(__name__)

AUDIT_SUFFIX = ".integrity.md"
FULL_REEMIT_MAX = 20_000

CompleteFn = Callable[[str], Awaitable[str]]

_COLLECTIONS = ("characters", "timeline", "evidence", "claims", "hypotheses", "clues")


# ── extraction ────────────────────────────────────────────────────────

def extract_prompt(chunk: str, n: int = 80) -> str:
    return (
        "You are building a STRUCTURED STATE STORE for a continuity check of a "
        "narrative — not a review. Extract only what the text states, with an exact "
        "short QUOTE for each item so a fix is a precise find/replace. Assign every "
        "timeline event a global chronological `order` integer (1 = earliest). "
        "Output ONLY this JSON (no prose); each list at most "
        f"{n} items:\n"
        "{\n"
        '  "characters": [{"name": "<name>", "age": "<n or empty>", "birth_year": "<yyyy or empty>",'
        ' "location_by_time": [{"order": <int>, "location": "<where>"}],'
        ' "knows": [{"fact": "<fact>", "learned_order": <int>}], "quote": "<exact quote>"}],\n'
        '  "story_year": "<the year the present-day story is set, or empty>",\n'
        '  "timeline": [{"id": "<label>", "order": <int>, "date": "<date or empty>", "time": "<time or empty>",'
        ' "location": "<where>", "participants": ["<name>"], "actor": "<the acting character>",'
        ' "uses_facts": ["<fact the actor relies on>"], "cites_evidence": ["<evidence id used/referenced>"],'
        ' "evidence_created": ["<evidence id created in this event>"], "quote": "<exact quote>"}],\n'
        '  "evidence": [{"id": "<label>", "specificity": "generic|individual", "custody_gap": true|false,'
        ' "key_proof": true|false, "proves": "<what it proves>", "movements": [{"order": <int>, "location": "<where>"}],'
        ' "disappearance_accounts": ["<account>"], "quote": "<exact quote>"}],\n'
        '  "claims": [{"proposition": "<statement>", "class": "observed|tested|documentary|witness|suspect|expert|inference|allegation",'
        ' "presented_as_fact": true|false, "sole_support_for_case": true|false,'
        ' "supporting_fact_ids": ["<evidence/claim id>"], "quote": "<exact quote>"}],\n'
        '  "hypotheses": [{"role": "<crime|leak|murder|motive|access|presence|...>", "candidates": ["<name>"],'
        ' "eliminations": [{"candidate": "<name>", "evidence_id": "<id>"}], "concluded_single": "<name or empty>", "quote": "<exact quote>"}],\n'
        '  "clues": [{"id": "<label>", "salience": "high|low", "status": "paid_off|disproved|open|unresolved",'
        ' "open_reason": "<why deliberately open, or empty>", "quote": "<exact quote>"}]\n'
        "}\n\n"
        "TEXT:\n" + chunk
    )


def parse_store(output: str) -> dict:
    """Parse the extractor's JSON into a normalised store (best-effort, fail-safe)."""
    empty = {k: [] for k in _COLLECTIONS}
    empty["story_year"] = ""
    if not output:
        return empty
    m = re.search(r"\{.*\}", output.strip(), re.DOTALL)
    if not m:
        return empty
    try:
        raw = json.loads(m.group(0))
    except (ValueError, TypeError):
        return empty
    if not isinstance(raw, dict):
        return empty
    out = {k: [] for k in _COLLECTIONS}
    out["story_year"] = str(raw.get("story_year") or "")
    for k in _COLLECTIONS:
        v = raw.get(k)
        if isinstance(v, list):
            out[k] = [e for e in v if isinstance(e, dict)]
    return out


def _max_order(store: dict) -> int:
    mx = 0
    for ev in store.get("timeline") or []:
        if isinstance(ev, dict):
            mx = max(mx, _int(ev.get("order")) or 0)
    for ch in store.get("characters") or []:
        if not isinstance(ch, dict):
            continue
        for lb in ch.get("location_by_time") or []:
            if isinstance(lb, dict):
                mx = max(mx, _int(lb.get("order")) or 0)
        for k in ch.get("knows") or []:
            if isinstance(k, dict):
                mx = max(mx, _int(k.get("learned_order")) or 0)
    for ev in store.get("evidence") or []:
        if isinstance(ev, dict):
            for mv in ev.get("movements") or []:
                if isinstance(mv, dict):
                    mx = max(mx, _int(mv.get("order")) or 0)
    return mx


def _offset_orders(store: dict, off: int) -> None:
    if off <= 0:
        return
    for ev in store.get("timeline") or []:
        if isinstance(ev, dict) and _int(ev.get("order")) is not None:
            ev["order"] = _int(ev.get("order")) + off
    for ch in store.get("characters") or []:
        if not isinstance(ch, dict):
            continue
        for lb in ch.get("location_by_time") or []:
            if isinstance(lb, dict) and _int(lb.get("order")) is not None:
                lb["order"] = _int(lb.get("order")) + off
        for k in ch.get("knows") or []:
            if isinstance(k, dict) and _int(k.get("learned_order")) is not None:
                k["learned_order"] = _int(k.get("learned_order")) + off
    for ev in store.get("evidence") or []:
        if not isinstance(ev, dict):
            continue
        for mv in ev.get("movements") or []:
            if isinstance(mv, dict) and _int(mv.get("order")) is not None:
                mv["order"] = _int(mv.get("order")) + off


def merge(stores: list[dict]) -> dict:
    """Merge per-chunk stores. The extractor numbers `order` PER CHUNK (restarting at
    1), so before concatenating we re-base each successive chunk's orders past the
    running maximum — otherwise event 1 of chunk 2 collides with event 1 of chunk 1
    and the chronology checks false-fire (two-places, future-evidence). Stores must be
    passed in chunk (document) order.
    """
    out = {k: [] for k in _COLLECTIONS}
    out["story_year"] = ""
    off = 0
    for s in stores:
        if not isinstance(s, dict):
            continue
        _offset_orders(s, off)
        off = max(off, _max_order(s))
        if not out["story_year"] and s.get("story_year"):
            out["story_year"] = s["story_year"]
        for k in _COLLECTIONS:
            out[k].extend(s.get(k) or [])
    return out


# ── helpers ───────────────────────────────────────────────────────────

def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _num(s):
    t = re.sub(r"[,$£€\s]", "", str(s or ""))
    try:
        return float(t)
    except (TypeError, ValueError):
        return None


def _year(s):
    m = re.search(r"\b(1[0-9]{3}|2[0-9]{3})\b", str(s or ""))
    return int(m.group(1)) if m else None


def _truthy(v) -> bool:
    """Strict truthiness — a weak model often emits JSON booleans as STRINGS, so the
    string 'false' must NOT read as true (that would false-fire / mis-escalate)."""
    if v is True:
        return True
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return bool(v) if isinstance(v, (int, float)) else False


def _finding(pass_: str, kind: str, severity: str, reason: str,
             scene: str = "", quotes=None) -> dict:
    # origin="deterministic": produced by a pure-code pass over the store, so it is
    # FAIL-SAFE (fires only on a contradiction clearly present in the state). Only these
    # gate `done`. Model-pass findings (story_passes.run_model_pass) carry origin="model"
    # and never block on their own — see blocking_findings().
    return {"pass": pass_, "kind": kind, "severity": severity, "source": "story_integrity",
            "origin": "deterministic",
            "reason": reason, "scene": scene, "detail": reason, "quotes": quotes or []}


# ── Pass A — chronology ───────────────────────────────────────────────

def check_chronology(store: dict) -> list[dict]:
    out: list[dict] = []
    timeline = [ev for ev in (store.get("timeline") or []) if isinstance(ev, dict)]
    # evidence creation order: earliest event whose evidence_created lists it
    created_at: dict[str, int] = {}
    for ev in timeline:
        o = _int(ev.get("order"))
        if o is None:
            continue
        for eid in (ev.get("evidence_created") or []):
            k = _norm(eid)
            if k and (k not in created_at or o < created_at[k]):
                created_at[k] = o
    # future-evidence: a scene cites evidence created at a LATER order
    for ev in timeline:
        o = _int(ev.get("order"))
        if o is None:
            continue
        for cid in (ev.get("cites_evidence") or []):
            k = _norm(cid)
            if k in created_at and created_at[k] > o:
                out.append(_finding(
                    "A", "future_evidence", "hard",
                    f"evidence '{cid}' is used at event {o} but is not created until event {created_at[k]}",
                    scene=str(ev.get("id") or o), quotes=[str(ev.get("quote") or "")]))
    # two-places-at-once: a character at two locations at one order
    where: dict[tuple, set] = {}
    for ev in timeline:
        o = _int(ev.get("order"))
        loc = _norm(ev.get("location"))
        if o is None or not loc:
            continue
        actors = list(ev.get("participants") or [])
        if ev.get("actor"):
            actors.append(ev.get("actor"))
        for p in actors:
            where.setdefault((_norm(p), o), set()).add(loc)
    for ch in (store.get("characters") or []):
        if not isinstance(ch, dict):
            continue
        nm = _norm(ch.get("name"))
        for lb in (ch.get("location_by_time") or []):
            if not isinstance(lb, dict):
                continue
            o = _int(lb.get("order"))
            loc = _norm(lb.get("location"))
            if nm and o is not None and loc:
                where.setdefault((nm, o), set()).add(loc)
    for (nm, o), locs in where.items():
        if nm and len(locs) > 1:
            out.append(_finding(
                "A", "two_places", "hard",
                f"{nm} is in two places at time {o}: {sorted(locs)}", scene=str(o)))
    return out


# ── Pass B — character knowledge ──────────────────────────────────────

def check_knowledge(store: dict) -> list[dict]:
    out: list[dict] = []
    knows: dict[str, dict[str, int]] = {}
    for ch in (store.get("characters") or []):
        if not isinstance(ch, dict):
            continue
        nm = _norm(ch.get("name"))
        if not nm:
            continue
        for k in (ch.get("knows") or []):
            if not isinstance(k, dict):
                continue
            f = _norm(k.get("fact"))
            lo = _int(k.get("learned_order"))
            if f:
                # earliest order the character knows this fact (default: always)
                prev = knows.get(nm, {}).get(f)
                val = lo if lo is not None else 0
                if prev is None or val < prev:
                    knows.setdefault(nm, {})[f] = val
    for ev in (store.get("timeline") or []):
        if not isinstance(ev, dict):
            continue
        o = _int(ev.get("order"))
        actor = _norm(ev.get("actor"))
        if o is None or not actor:
            continue
        for fact in (ev.get("uses_facts") or []):
            f = _norm(fact)
            if not f:
                continue
            learned = knows.get(actor, {}).get(f)
            # fire only when we KNOW the actor learns it strictly later (fail-safe:
            # an unknown fact does not fire — only a recorded later learning does)
            if learned is not None and learned > o:
                out.append(_finding(
                    "B", "knowledge_gap", "hard",
                    f"{actor} acts on '{fact}' at event {o} but does not learn it until event {learned}",
                    scene=str(ev.get("id") or o), quotes=[str(ev.get("quote") or "")]))
    return out


# ── Pass D — evidence provenance ──────────────────────────────────────

# Affirmative individual-identification phrasing (fires) vs a negation/qualifier
# nearby (does NOT fire — a "cannot identify" / "common type" description is correct
# use of generic evidence, not an overclaim). Tight + negation-guarded so a faithful
# extraction of a legitimate description never false-blocks.
_ID_POS = ("identifies the", "identifies him", "identifies her", "identifies them",
           "uniquely identif", "pinpoints", "proves it was ", "this specific person",
           "was definitely ", "must be the ")
_ID_NEG = ("cannot", "can't", "does not", "doesn't", "not ", "no ", "never", "only narrow",
           "common type", "common blood", "shared by", "merely", "narrows")


def check_provenance(store: dict) -> list[dict]:
    out: list[dict] = []
    for ev in (store.get("evidence") or []):
        if not isinstance(ev, dict):
            continue
        eid = ev.get("id") or "an object"
        accounts = [a for a in (ev.get("disappearance_accounts") or []) if _norm(a)]
        if len({_norm(a) for a in accounts}) > 1:
            out.append(_finding(
                "D", "two_disappearances", "hard",
                f"evidence '{eid}' has {len(set(_norm(a) for a in accounts))} conflicting "
                f"disappearance accounts", scene=str(eid), quotes=[str(ev.get("quote") or "")]))
        # generic evidence used as individual identification — affirmative phrasing
        # only, and never when the description negates/qualifies identification.
        proves = _norm(ev.get("proves"))
        if _norm(ev.get("specificity")) == "generic" and proves:
            if (not any(n in proves for n in _ID_NEG)) and any(p in proves for p in _ID_POS):
                out.append(_finding(
                    "D", "generic_as_identity", "hard",
                    f"generic evidence '{eid}' is used as individual identification "
                    f"('{ev.get('proves')}')", scene=str(eid), quotes=[str(ev.get("quote") or "")]))
        # broken custody chain still used as key proof
        if _truthy(ev.get("custody_gap")) and _truthy(ev.get("key_proof")):
            out.append(_finding(
                "D", "custody_broken", "major",
                f"evidence '{eid}' has a broken custody chain yet is used as key proof",
                scene=str(eid), quotes=[str(ev.get("quote") or "")]))
    # a claim/record asserting an unestablished fact
    for c in (store.get("claims") or []):
        if not isinstance(c, dict):
            continue
        cls = _norm(c.get("class"))
        if not _truthy(c.get("presented_as_fact")):
            continue
        supported = [s for s in (c.get("supporting_fact_ids") or []) if _norm(s)]
        if cls in ("suspect", "allegation", "inference", "witness") and not supported:
            sole = _truthy(c.get("sole_support_for_case"))
            sev = "hard" if sole else "major"
            kind = "confession_sole_support" if (cls == "suspect" and sole) \
                else "unsupported_assertion"
            out.append(_finding(
                "D", kind, sev,
                f"a {cls} claim is presented as fact without support: "
                f"'{str(c.get('proposition') or '')[:120]}'", quotes=[str(c.get("quote") or "")]))
    return out


# ── Pass E — hypothesis scope ─────────────────────────────────────────

def check_hypothesis_scope(store: dict) -> list[dict]:
    out: list[dict] = []
    hyps = [h for h in (store.get("hypotheses") or []) if isinstance(h, dict)]
    # Aggregate candidates / eliminations / conclusions PER normalised role across ALL
    # hypothesis entries — the extractor runs per chunk and merge() concatenates, so a
    # role's candidates, its eliminations, and its conclusion can land in separate
    # dicts. Without the union a conclusion entry with empty eliminations would
    # false-fire pool_overclaim from ABSENCE (the exact opposite of the fail-safe rule).
    role_cands: dict[str, set] = {}
    role_elim: dict[str, set] = {}
    role_concl: dict[str, str] = {}
    role_quote: dict[str, str] = {}
    ev_roles: dict[str, set] = {}
    for h in hyps:
        role = _norm(h.get("role"))
        if not role:
            continue
        role_cands.setdefault(role, set()).update(
            _norm(c) for c in (h.get("candidates") or []) if _norm(c))
        for e in (h.get("eliminations") or []):
            if not isinstance(e, dict):
                continue
            cand = _norm(e.get("candidate"))
            eid = _norm(e.get("evidence_id"))
            if cand:
                role_elim.setdefault(role, set()).add(cand)
            if eid:
                ev_roles.setdefault(eid, set()).add(role)
        concl = _norm(h.get("concluded_single"))
        if concl:
            role_concl[role] = concl
            role_quote[role] = str(h.get("quote") or "")
    # cross-role elimination: one evidence_id eliminates candidates in >1 role
    for eid, roles in ev_roles.items():
        if len(roles) > 1:
            out.append(_finding(
                "E", "cross_role_elimination", "hard",
                f"evidence '{eid}' is used to eliminate suspects across {len(roles)} "
                f"different roles ({sorted(roles)}) — elimination is per-role"))
    # "the pool is one" overclaim — only when the pool was ACTUALLY narrowed (at least
    # one elimination captured for the role) yet more than one candidate still remains.
    for role, concluded in role_concl.items():
        elim = role_elim.get(role, set())
        if not elim:
            continue  # no eliminations captured → under-extracted, not an overclaim
        remaining = role_cands.get(role, set()) - elim
        if len(remaining) > 1:
            out.append(_finding(
                "E", "pool_overclaim", "hard",
                f"the {role} role concludes '{concluded}' alone, but {len(remaining)} "
                f"candidates remain un-eliminated: {sorted(remaining)}",
                quotes=[role_quote.get(role, "")]))
    return out


# ── Pass G — quantities & identity ────────────────────────────────────

def check_quantities(store: dict) -> list[dict]:
    out: list[dict] = []
    # age vs date math per character
    story_year = _year(store.get("story_year"))
    for ch in (store.get("characters") or []):
        if not isinstance(ch, dict):
            continue
        age = _num(ch.get("age"))
        birth = _year(ch.get("birth_year"))
        if age is not None and birth is not None and story_year is not None:
            computed = story_year - birth
            if abs(computed - age) > 1:
                out.append(_finding(
                    "G", "age_date_mismatch", "hard",
                    f"{ch.get('name')} is stated as age {age:g} but born {birth} in a story set "
                    f"{story_year} (implies {computed})", quotes=[str(ch.get("quote") or "")]))
    # a character's age stated two DIFFERENT numeric ways. Compare NUMBERS, not raw
    # strings — "34" / "34.0" / "34 years old" are the same age and must not fire; only
    # a real numeric difference (> 1) does. (A name-collision — two different people
    # sharing a first name — can still bucket together; the store has no character id
    # to disambiguate, so that rarer case is a documented P1 limitation, not fixed by a
    # string change.)
    by_name: dict[str, set] = {}
    quotes: dict[str, list] = {}
    for ch in (store.get("characters") or []):
        if not isinstance(ch, dict):
            continue
        nm = _norm(ch.get("name"))
        n = _num(ch.get("age"))
        if nm and n is not None:
            by_name.setdefault(nm, set()).add(round(n, 3))
            quotes.setdefault(nm, []).append(str(ch.get("quote") or ""))
    for nm, ages in by_name.items():
        if len(ages) > 1 and (max(ages) - min(ages)) > 1:
            out.append(_finding(
                "G", "quantity_mismatch", "hard",
                f"{nm}'s age is given as {sorted(ages)}", quotes=quotes.get(nm, [])[:2]))
    return out


# ── Pass H — clue & payoff ────────────────────────────────────────────

def check_clue_payoff(store: dict) -> list[dict]:
    out: list[dict] = []
    for c in (store.get("clues") or []):
        if not isinstance(c, dict):
            continue
        if _norm(c.get("salience")) != "high":
            continue
        status = _norm(c.get("status")).replace(" ", "_")
        if status in ("paid_off", "disproved"):
            continue
        # a stated reason exempts the clue whatever the exact status spelling
        if _norm(c.get("open_reason")):
            continue
        out.append(_finding(
            "H", "unpaid_clue", "major",
            f"high-salience clue '{c.get('id') or '?'}' is neither paid off, disproved, nor "
            f"left open for a stated reason", quotes=[str(c.get("quote") or "")]))
    return out


_CHECKS = (check_chronology, check_knowledge, check_provenance,
           check_hypothesis_scope, check_quantities, check_clue_payoff)


def verify(store: dict) -> list[dict]:
    """Run every deterministic pass over the store; return all findings."""
    out: list[dict] = []
    for chk in _CHECKS:
        try:
            out.extend(chk(store))
        except Exception as e:  # noqa: BLE001 — a check must never crash the run
            log.warning("story_state check failed", check=chk.__name__, error=str(e))
    return out


# ── severity buckets ──────────────────────────────────────────────────

def bucket(findings: list[dict]) -> dict:
    """Split findings into hard / major / soft."""
    return {
        "hard": [f for f in findings if f.get("severity") == "hard"],
        "major": [f for f in findings if f.get("severity") == "major"],
        "soft": [f for f in findings if f.get("severity") not in ("hard", "major")],
    }


# ── deterministic vs model, and the blocking (gating) set ─────────────

# The six pure-code passes over the store. These are fail-safe by construction, so ONLY
# their hard findings gate `done`. Everything else (C/F/I/J/CA/PR in story_passes) is a
# weak-model judgment — surfaced and used to drive revision, but never a blocker on its
# own, because a single hallucinated model objection must not false-fail a good draft.
_DETERMINISTIC_PASSES = frozenset({"A", "B", "D", "E", "G", "H"})


def is_deterministic(f: dict) -> bool:
    """Was this finding produced by a fail-safe pure-code pass? Prefers the explicit
    `origin` tag; falls back to the pass id so older/foreign findings still classify."""
    origin = f.get("origin")
    if origin == "deterministic":
        return True
    if origin == "model":
        return False
    return str(f.get("pass", "")) in _DETERMINISTIC_PASSES


def blocking_findings(findings: list[dict]) -> list[dict]:
    """The hard findings that GATE `done`: deterministic (fail-safe) hard only. Model-pass
    hard findings surface + drive revision but never block a finished draft on their own."""
    return [f for f in findings
            if f.get("severity") == "hard" and is_deterministic(f)]


# ── quality scores (advisory; they never gate) ────────────────────────

# Each pass rolls up into one quality dimension. continuity is the deterministic backbone
# (the trustworthy signal); plausibility/grounding/craft are model-judged and noisier.
_DIMENSION = {
    "A": "continuity", "B": "continuity", "D": "continuity",
    "E": "continuity", "G": "continuity", "H": "continuity",
    "C": "plausibility", "PR": "plausibility",
    "F": "grounding", "CA": "grounding",
    "I": "craft", "J": "craft",
}
_DIMENSIONS = ("continuity", "plausibility", "grounding", "craft")
_SEV_PENALTY = {"hard": 25, "major": 10, "soft": 3}


def _grade(s: int) -> str:
    return "clean" if s >= 85 else "sound" if s >= 70 else "caution" if s >= 50 else "weak"


def score(findings: list[dict], *, passes_run=None) -> dict:
    """0–100 quality scores derived from the surviving findings (higher = cleaner).

    Per-dimension: 100 minus a severity-weighted penalty for that dimension's findings,
    floored at 0. `integrity` is the headline composite, weighted heavily toward the
    deterministic `continuity` backbone (0.7) with the three model dimensions at 0.1 each,
    so a story whose objective backbone is airtight cannot be dragged low by noisy weak-
    model quibbles alone. These are ADVISORY — nothing here gates `done`.

    `passes_run` (pass ids that actually executed) lets a dimension whose passes never ran
    report `null` ("not checked") instead of a perfect 100 we can't stand behind. A
    dimension that ran and found nothing scores 100. Omit it to treat all as checked."""
    per = {d: 100 for d in _DIMENSIONS}
    for f in findings:
        dim = _DIMENSION.get(str(f.get("pass", "")))
        if not dim:
            continue
        per[dim] -= _SEV_PENALTY.get(f.get("severity"), _SEV_PENALTY["soft"])
    per = {d: max(0, v) for d, v in per.items()}
    if passes_run is None:
        checked = set(_DIMENSIONS)
    else:
        run = set(passes_run)
        checked = {dd for pid, dd in _DIMENSION.items() if pid in run}
    # Composite over ONLY the dimensions that were actually checked, with the weights
    # renormalized over that subset — so a dimension reported as null (never ran / provider
    # outage) does NOT sneak its default 100 into the headline. A deterministic-only run
    # thus yields integrity == continuity, not an inflated blend.
    weights = {"continuity": 0.7, "plausibility": 0.1, "grounding": 0.1, "craft": 0.1}
    wsum = sum(weights[d] for d in _DIMENSIONS if d in checked)
    if wsum > 0:
        integrity = round(sum(per[d] * weights[d] for d in _DIMENSIONS if d in checked) / wsum)
    else:
        integrity = None
    dims = {d: (per[d] if d in checked else None) for d in _DIMENSIONS}
    return {"integrity": integrity,
            "grade": (_grade(integrity) if integrity is not None else None), **dims}


# ── patch-mode revision ───────────────────────────────────────────────

def patch_prompt(text: str, findings: list[dict], simplify: bool = False) -> str:
    lines = []
    for f in findings[:20]:
        q = "; ".join(x for x in (f.get("quotes") or []) if x)
        fix = f.get("fix")
        lines.append(f"- [{f['pass']}/{f['kind']} · {f['severity']}] {f['reason']}"
                     + (f" — suggested fix: {fix}" if fix else "")
                     + (f" (quotes: {q})" if q else ""))
    escalate = (
        "An earlier narrow fix did NOT clear these — now SIMPLIFY the plot as needed: "
        "replace an exact time with a range, replace an unreliable mechanism with a "
        "simpler one that works, change a professional action that would invalidate the "
        "case, or seed an unsupported twist earlier or cut it. A simpler true story beats "
        "a complex impossible one. "
    ) if simplify else (
        "You MAY simplify the plot — a time range instead of an exact time, a simpler "
        "mechanism, a changed action, a seeded-or-cut twist. "
    )
    return (
        "A continuity check found these HARD/MAJOR integrity errors in a narrative. "
        "REPAIR THE CAUSE, not the wording: fix the chronology, the state, the "
        "mechanism, or the logic. " + escalate +
        "Do NOT paper over an impossible sequence with one explanatory sentence, and do "
        "NOT rewrite the whole story. Output ONLY a JSON array of anchored find/replace "
        "patches — the `find` must be an EXACT substring of the text:\n"
        '[{"find": "<exact text to replace>", "replace": "<corrected text>"}]\n\n'
        "Errors:\n" + "\n".join(lines) + "\n\nTEXT:\n" + text[:FULL_REEMIT_MAX]
    )


def parse_patches(output: str) -> list[dict]:
    if not output:
        return []
    m = re.search(r"\[.*\]", output.strip(), re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except (ValueError, TypeError):
        return []
    return [p for p in arr if isinstance(p, dict)] if isinstance(arr, list) else []


def apply_patches(text: str, patches: list[dict]) -> tuple[str, int, list[dict]]:
    out = text or ""
    applied = 0
    unapplied: list[dict] = []
    for p in patches or []:
        find = str(p.get("find") or "")
        repl = str(p.get("replace") or "")
        if find and find in out:
            out = out.replace(find, repl, 1)
            applied += 1
        else:
            unapplied.append(p)
    return out, applied, unapplied


# ── the validator run ─────────────────────────────────────────────────

async def run_validator(text: str, *, extract_fn: CompleteFn,
                        revise_fn: CompleteFn | None = None,
                        model_fn: CompleteFn | None = None,
                        model_passes: tuple = (),
                        max_rounds: int = 2, max_items: int = 80,
                        on_progress: Callable[[str], None] | None = None) -> dict:
    """Extract the store → deterministic verify (+ optional MODEL passes) → bounded
    patch-revise loop.

    Returns ``{text, store, revised, findings, initial_findings, rounds, hard,
    major, soft}``. Each round: extract the store from the current text, run the
    deterministic passes AND (when ``model_fn``/``model_passes`` are given) the
    reason-tier model passes (C/F/I/J physical/research/economy/fairness, CA/PR
    claim-attacker/procedure); if hard/major findings remain and a reviser is
    available, patch by anchored find/replace (repair the cause, may simplify),
    re-extract, re-verify, and keep the revision only if it reduces the hard+major
    count. Later rounds escalate to allow structural simplification.
    """
    def _note(m: str) -> None:
        if on_progress:
            try:
                on_progress(m)
            except Exception:  # noqa: BLE001
                pass

    async def _extract(t: str) -> dict:
        parts = []
        chunks = chunk_text(t)
        for i, ch in enumerate(chunks):
            _note(f"Integrity: reading chunk {i + 1}/{len(chunks)}…")
            parts.append(parse_store(await extract_fn(extract_prompt(ch, max_items)) or ""))
        return merge(parts)

    async def _all_findings(t: str) -> tuple[dict, list[dict], set]:
        # passes_run: which passes ACTUALLY executed, so score() can tell a dimension that
        # ran clean (100) from one whose passes never ran / failed (null). The six
        # deterministic passes always run in pure code; model passes run only when a model
        # is wired AND the call completes (a provider outage drops them from the set).
        st = await _extract(t)
        fs = verify(st)  # deterministic passes (A/B/D/E/G/H)
        passes_run = set(_DETERMINISTIC_PASSES)
        if model_fn is not None and model_passes:
            from captain_claw.flight_deck import story_passes
            try:
                mp, ran = await story_passes.run_model_passes(
                    t, json.dumps(st, ensure_ascii=True), model_fn,
                    passes=model_passes, on_progress=on_progress)
                fs = fs + mp
                passes_run |= ran
            except Exception as e:  # noqa: BLE001 — model passes are best-effort
                log.warning("story model passes failed", error=str(e))
        return st, fs, passes_run

    store, findings, passes_run = await _all_findings(text)
    result = {"text": text, "store": store, "revised": False, "findings": findings,
              "initial_findings": findings, "rounds": 0, "patched": 0,
              "blocking": blocking_findings(findings),
              "scores": score(findings, passes_run=passes_run),
              **bucket(findings)}
    if revise_fn is None:
        return result

    cur_text = text
    cur_findings = findings
    cur_passes_run = passes_run
    rounds = 0
    while rounds < max_rounds:
        blockers = [f for f in cur_findings if f.get("severity") in ("hard", "major")]
        if not blockers:
            break
        rounds += 1
        _note(f"Integrity round {rounds}: {len(blockers)} hard/major error(s) — revising…")
        # Escalate to structural simplification once a first round didn't clear it.
        patches = parse_patches((await revise_fn(
            patch_prompt(cur_text, blockers, simplify=rounds > 1)) or ""))
        if not patches:
            break
        revised, applied, _un = apply_patches(cur_text, patches)
        collapsed = not revised.strip() or (len(cur_text) > 800 and len(revised) < 0.9 * len(cur_text))
        if collapsed or applied == 0:
            _note("Integrity: patches did not apply cleanly — kept the draft")
            break
        new_store, new_findings, new_passes_run = await _all_findings(revised)
        new_block = [f for f in new_findings if f.get("severity") in ("hard", "major")]
        # The done-gate blocks only on the DETERMINISTIC hard set (blocking_findings), so
        # the invariant to protect is that count: never accept a revision that grows it (a
        # simplification could introduce a real deterministic contradiction). Otherwise
        # require the total hard+major to strictly drop, so the loop makes progress.
        old_block = len(blocking_findings(cur_findings))
        new_block_hard = len(blocking_findings(new_findings))
        if new_block_hard > old_block or len(new_block) >= len(blockers):
            _note("Integrity: revision did not reduce hard/major errors — kept the draft")
            break
        cur_text, cur_findings, store, cur_passes_run = (
            revised, new_findings, new_store, new_passes_run)
        result["patched"] += applied
        result.update(text=cur_text, store=store, revised=True, findings=cur_findings,
                      rounds=rounds, blocking=blocking_findings(cur_findings),
                      scores=score(cur_findings, passes_run=cur_passes_run),
                      **bucket(cur_findings))
    result["rounds"] = rounds
    return result


# ── audit + summary (kept OUT of the deliverable) ─────────────────────

def summarize(result: dict) -> dict:
    return {
        "rounds": result.get("rounds", 0),
        "hard_fixed": max(0, len([f for f in result.get("initial_findings") or []
                                  if f.get("severity") == "hard"]) - len(result.get("hard") or [])),
        "remaining": len(result.get("hard") or []) + len(result.get("major") or []),
        "hard": len(result.get("hard") or []),
        # `blocking` = the deterministic hard errors that actually gate `done`; a run can
        # have hard > 0 (a model-pass objection) yet blocking == 0 and still finish.
        "blocking": len(result.get("blocking") or []),
        "major": len(result.get("major") or []),
        "soft": len(result.get("soft") or []),
        "scores": result.get("scores"),
        "passes": sorted({f["pass"] for f in (result.get("findings") or [])}),
    }


def summary_line(result: dict) -> str:
    s = summarize(result)
    sc = s.get("scores") or {}
    head = (f"integrity {sc['integrity']} ({sc.get('grade', '')}) · "
            if sc.get("integrity") is not None else "")
    line = (head
            + f"{s['blocking']} blocking · {s['hard']} hard · {s['major']} major · {s['soft']} soft"
            + (f" (after {s['rounds']} round(s), {result.get('patched', 0)} patch(es))"
               if s["rounds"] else ""))
    return line


def blocking_analysis(result: dict) -> dict:
    """`analysis.blocking.hard[]` / `.major[]` — {pass, scene, reason}."""
    def _rows(fs):
        return [{"pass": f.get("pass"), "scene": f.get("scene"), "reason": f.get("reason")}
                for f in fs]
    return {"hard": _rows(result.get("hard") or []), "major": _rows(result.get("major") or [])}


def audit_markdown(result: dict, *, question: str) -> str:
    lines = [f"# Story-integrity check\n\n*Task:* {question[:200]}\n"]
    hard = result.get("hard") or []
    major = result.get("major") or []
    if not hard and not major:
        lines.append("\nNo unresolved hard or major integrity errors.\n")
    else:
        for label, fs in (("Hard errors", hard), ("Major weaknesses", major)):
            if fs:
                lines.append(f"\n## {label} ({len(fs)})\n")
                for f in fs:
                    lines.append(f"- **{f['pass']}/{f['kind']}** (scene {f.get('scene', '?')}) — {f['reason']}")
    return "\n".join(lines) + "\n"


def write_audit(dest_dir: Path, result: dict, *, question: str,
                base_name: str = "deliverable") -> dict | None:
    if not (result.get("hard") or result.get("major")):
        return None
    name = f"{base_name}{AUDIT_SUFFIX}"
    try:
        p = Path(dest_dir) / name
        p.write_text(audit_markdown(result, question=question), encoding="utf-8")
        return {"name": name, "mime": "text/markdown", "size": p.stat().st_size,
                "kind": "generated", "agent": "story-integrity"}
    except OSError as e:  # noqa: BLE001
        log.warning("story-integrity audit write failed", error=str(e))
        return None
