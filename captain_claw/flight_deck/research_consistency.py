"""Deterministic cross-section consistency verification for research deliverables.

The one failure class no LLM opinion catches reliably: a deliverable that
contradicts ITSELF — the same quantity stated with different values in different
sections, or a stated total that its own listed parts don't sum to. Critics and
fact-checkers verify claims against the world; nothing verifies the document
against itself, and the reporter's "resolve it sensibly" pass is prose, not a
check.

The split here is the whole design: an LLM does ONLY extraction (find the
load-bearing values and the arithmetic relationships the text itself asserts —
no judging, no computing), and plain code does ALL the verification (identity
across occurrences, sum/percent/difference/product recomputation with rounding
tolerance). The deterministic side cannot hallucinate a failure: every finding
is anchored to quoted occurrences, so false positives are bounded by extraction
quality while false negatives just mean fewer catches.

This module is the pure, mode-agnostic part (mirrors ``research_verify``): the
extractor prompt, the tolerant parser, the ``verify()`` arithmetic, the fix
checklist + revision prompt, and the audit renderer. Model calls are injected as
async callables, so the whole flow is testable with stubs. Wiring (Basna after
merge, Vatra after the reporter — both BEFORE the R8 claim check, so internal
fixes land before external verification) lives in the route files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Awaitable, Callable

from captain_claw.logging import get_logger

log = get_logger(__name__)

#: Suffix for the standalone audit document written next to the deliverable.
AUDIT_SUFFIX = ".consistency.md"

_VALID_KINDS = ("figure", "date", "identifier")
_VALID_RELATIONS = ("sum", "difference", "product", "percent_of")

#: Chars of deliverable sent to the extractor. Overflow is reported (progress +
#: audit note), never silent — a truncated check must not read as "covered".
DELIVERABLE_CAP = 60_000

CompleteFn = Callable[[str], Awaitable[str]]


# ── Extraction (the only LLM step — it finds, it never computes) ───────

def extract_prompt(deliverable: str, max_values: int) -> str:
    """Prompt an extractor to pull out value occurrences + stated relations."""
    return (
        "You are a meticulous EXTRACTOR. Below is a deliverable. Extract its "
        "load-bearing values and the arithmetic relationships the text itself "
        "STATES — do NOT judge, correct, or compute anything yourself.\n\n"
        f"1. VALUES: up to {max_values} of the most load-bearing quantities, dates, "
        "and identifiers. Record every OCCURRENCE separately — if the same quantity "
        "appears in three places, emit three entries with the same label. Use one "
        "canonical `label` per real-world quantity (every mention of the project's "
        "total budget gets the same label, even when the wording differs between "
        "sections).\n"
        "2. RELATIONS: every arithmetic relationship the text asserts between "
        "labeled values — a total said to be the sum of listed parts, a value said "
        "to be a percentage of another, a difference, a product.\n\n"
        "Reply ONLY with JSON — no prose before or after:\n"
        '{"values": [{"label": "<canonical quantity name>", '
        '"kind": "figure|date|identifier", '
        '"raw": "<the value exactly as written>", '
        '"value": <number for figures; "YYYY-MM-DD" for dates; the exact string for identifiers>, '
        '"unit": "<EUR, %, days, FTE, … or empty>", '
        '"quote": "<up to 120 chars of surrounding text>"}],\n'
        ' "relations": [{"type": "sum|difference|product|percent_of", '
        '"operands": ["<label>", "…"], "result": "<label>", '
        '"percent": <number, ONLY for percent_of>, '
        '"quote": "<up to 120 chars>"}]}\n'
        "Rules:\n"
        "- Extract only what the text states; never infer a relation the text does "
        "not assert.\n"
        "- Figures: `value` is the plain number — no thousands separators or "
        "currency symbols; expand shorthand (\"€1.2M\" → 1200000) and keep the unit "
        "in `unit`.\n"
        "- Dates: `value` is YYYY-MM-DD (or YYYY-MM / YYYY when the text is less "
        "precise).\n"
        "- Identifiers (reference codes, IDs, versions): `value` is the exact string.\n"
        '- Nothing load-bearing to extract → {"values": [], "relations": []}.\n\n'
        "## Deliverable\n"
        f"{deliverable}"
    )


def parse_entries(output: str) -> dict:
    """Pull ``{"values": [...], "relations": [...]}`` out of the extractor's reply.
    Tolerant of code fences and surrounding prose; malformed rows are dropped,
    never fatal. Always returns both keys."""
    empty = {"values": [], "relations": []}
    if not output:
        return empty
    text = output.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:
        start, end = text.find("{"), text.rfind("}")
        blob = text[start:end + 1] if 0 <= start < end else None
    if not blob:
        return empty
    try:
        raw = json.loads(blob)
    except (ValueError, TypeError):
        return empty
    if not isinstance(raw, dict):
        return empty

    values: list[dict] = []
    for item in raw.get("values") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()[:120]
        if not label:
            continue
        kind = str(item.get("kind") or "").strip().lower()
        v = item.get("value")
        if kind not in _VALID_KINDS:
            # Coerce from the value's shape rather than dropping outright.
            kind = "figure" if isinstance(v, (int, float)) else "identifier"
        if kind == "figure":
            try:
                num = float(v)
            except (TypeError, ValueError):
                continue  # a figure we can't compare is useless — drop it
            val: float | str = num
        else:
            val = str(v if v is not None else "").strip()
            if not val:
                continue
        values.append({
            "label": label,
            "kind": kind,
            "raw": str(item.get("raw") or "").strip()[:80],
            "value": val,
            "unit": str(item.get("unit") or "").strip()[:20],
            "quote": str(item.get("quote") or "").strip()[:160],
        })

    relations: list[dict] = []
    for item in raw.get("relations") or []:
        if not isinstance(item, dict):
            continue
        rtype = str(item.get("type") or "").strip().lower()
        if rtype not in _VALID_RELATIONS:
            continue
        operands = [str(o).strip()[:120] for o in (item.get("operands") or [])
                    if str(o).strip()]
        result = str(item.get("result") or "").strip()[:120]
        if not operands or not result:
            continue
        rel: dict = {"type": rtype, "operands": operands, "result": result,
                     "quote": str(item.get("quote") or "").strip()[:160]}
        if rtype == "percent_of":
            try:
                rel["percent"] = float(item.get("percent"))
            except (TypeError, ValueError):
                continue  # a percent relation without its percent can't be checked
        relations.append(rel)

    return {"values": values, "relations": relations}


# ── Verification (pure code — no model, no network) ────────────────────

def _norm(label: str) -> str:
    return re.sub(r"\s+", " ", (label or "").strip().strip(".,:;")).casefold()


def _tol(a: float, b: float) -> float:
    """Rounding tolerance: half a percent of the larger magnitude, at least 1 unit."""
    return max(0.005 * max(abs(a), abs(b)), 1.0)


def _dates_compatible(a: str, b: str) -> bool:
    """"2028-10" and "2028-10-31" are the same date at different precision."""
    a, b = a.strip(), b.strip()
    short, longer = (a, b) if len(a) <= len(b) else (b, a)
    return longer.startswith(short) and (len(longer) == len(short)
                                         or longer[len(short):len(short) + 1] == "-")


def _fmt(n: float) -> str:
    return f"{n:,.2f}".rstrip("0").rstrip(".")


def _occurrences(group: list[dict]) -> str:
    return "; ".join(
        f"{e['raw'] or e['value']}" + (f" (“{e['quote']}”)" if e["quote"] else "")
        for e in group)


def verify(entries: dict, ledger_rows: list[dict] | None = None) -> list[dict]:
    """Deterministically check the extracted entries. Returns findings, each
    ``{severity, kind, label, detail, quotes}`` — severity ``critical`` for a
    same-quantity value conflict, a broken stated relation, or a ledger mismatch.
    """
    findings: list[dict] = []
    values = entries.get("values") or []

    # 1) Identity: every occurrence of the same quantity must carry the same value.
    groups: dict[tuple[str, str], list[dict]] = {}
    for e in values:
        groups.setdefault((_norm(e["label"]), e["kind"]), []).append(e)
    for (label_n, kind), group in groups.items():
        if len(group) < 2:
            continue
        label = group[0]["label"]
        quotes = [e["quote"] for e in group if e["quote"]]
        if kind == "figure":
            # Compare only within a unit (EUR vs days aren't the same quantity);
            # unit-less occurrences join the group's single stated unit, if any.
            units = {e["unit"].casefold() for e in group if e["unit"]}
            if len(units) > 1:
                continue
            nums = [float(e["value"]) for e in group]
            lo, hi = min(nums), max(nums)
            if hi - lo > _tol(lo, hi):
                findings.append({
                    "severity": "critical", "kind": "identity", "label": label,
                    "detail": (f"“{label}” appears with conflicting values: "
                               f"{_occurrences(group)}"),
                    "quotes": quotes,
                })
        elif kind == "date":
            distinct = {str(e["value"]).strip() for e in group}
            base = sorted(distinct, key=len, reverse=True)[0]
            if any(not _dates_compatible(d, base) for d in distinct):
                findings.append({
                    "severity": "critical", "kind": "identity", "label": label,
                    "detail": (f"“{label}” appears with conflicting dates: "
                               f"{_occurrences(group)}"),
                    "quotes": quotes,
                })
        else:  # identifier
            if len({str(e["value"]).strip() for e in group}) > 1:
                findings.append({
                    "severity": "critical", "kind": "identity", "label": label,
                    "detail": (f"“{label}” appears with conflicting identifiers: "
                               f"{_occurrences(group)}"),
                    "quotes": quotes,
                })

    # 2) Relations: recompute every arithmetic relationship the text asserts.
    # Representative value per label = first occurrence (identity findings above
    # already flag conflicted labels, so an arbitrary-but-stable pick is fine).
    rep: dict[str, dict] = {}
    for e in values:
        if e["kind"] == "figure":
            rep.setdefault(_norm(e["label"]), e)
    for rel in entries.get("relations") or []:
        ops = [rep.get(_norm(o)) for o in rel["operands"]]
        res = rep.get(_norm(rel["result"]))
        if res is None or any(o is None for o in ops):
            continue  # an operand was never extracted as a value — nothing to check
        nums = [float(o["value"]) for o in ops]
        actual = float(res["value"])
        if rel["type"] == "sum":
            expected = sum(nums)
            stated = f"the sum of {', '.join(rel['operands'])}"
        elif rel["type"] == "difference":
            if len(nums) != 2:
                continue
            expected = nums[0] - nums[1]
            stated = f"{rel['operands'][0]} minus {rel['operands'][1]}"
        elif rel["type"] == "product":
            expected = 1.0
            for n in nums:
                expected *= n
            stated = f"the product of {', '.join(rel['operands'])}"
        else:  # percent_of
            expected = rel["percent"] / 100.0 * nums[0]
            stated = f"{_fmt(rel['percent'])}% of {rel['operands'][0]}"
        if abs(expected - actual) > _tol(expected, actual):
            findings.append({
                "severity": "critical", "kind": "relation", "label": rel["result"],
                "detail": (f"stated relation does not hold: “{rel['result']}” "
                           f"is {res['raw'] or _fmt(actual)}, but {stated} computes to "
                           f"{_fmt(expected)}"),
                "quotes": [q for q in [rel["quote"]] if q],
            })

    # 3) Ledger cross-check (lights up when a facts ledger rides along): a value
    # in the text must match the canonical value recorded for the same key.
    for row in ledger_rows or []:
        key_n = _norm(str(row.get("key") or ""))
        if not key_n:
            continue
        try:
            canon = float(row.get("value"))
        except (TypeError, ValueError):
            continue
        for e in values:
            if e["kind"] != "figure" or _norm(e["label"]) != key_n:
                continue
            num = float(e["value"])
            if abs(num - canon) > _tol(num, canon):
                findings.append({
                    "severity": "critical", "kind": "ledger", "label": e["label"],
                    "detail": (f"“{e['label']}” is {e['raw'] or _fmt(num)} in the "
                               f"text but the shared ledger records {_fmt(canon)}"),
                    "quotes": [q for q in [e["quote"]] if q],
                })
    return findings


def needs_fix(findings: list[dict]) -> list[dict]:
    """The subset worth a correction pass."""
    return [f for f in findings if f.get("severity") in ("critical", "major")]


# ── Correction (one targeted revision, verified by a re-check) ─────────

def fix_checklist(findings: list[dict]) -> str:
    """Numbered fix instructions (the R3 triage shape). "" when nothing to fix."""
    items = needs_fix(findings)
    if not items:
        return ""
    lines = [
        "Deterministic arithmetic on the deliverable's own stated values found "
        "these internal inconsistencies. Fix EACH precisely and keep everything "
        "else identical:",
    ]
    for i, f in enumerate(items, 1):
        if f["kind"] == "identity":
            lines.append(
                f"{i}. {f['detail']}\n   Determine the correct value from context "
                "and use it EVERYWHERE this quantity appears.")
        elif f["kind"] == "relation":
            lines.append(
                f"{i}. {f['detail']}\n   Make the arithmetic consistent: recompute "
                "the result or correct the operands, and propagate the fix to every "
                "place these values appear.")
        else:  # ledger
            lines.append(
                f"{i}. {f['detail']}\n   The ledger value is canonical — use it, "
                "unless the text is clearly right, in which case say so explicitly.")
    return "\n".join(lines)


def revise_prompt(deliverable: str, checklist: str) -> str:
    return (
        "Below is a deliverable with verified internal inconsistencies, followed "
        "by the required fixes. Output the FULL corrected deliverable — apply "
        "exactly these changes and keep all other content, structure, and "
        "formatting identical.\n\n"
        f"## Required fixes\n{checklist}\n\n"
        f"## Deliverable\n{deliverable}\n\n"
        "Output the complete corrected deliverable only — no preamble, no "
        "commentary."
    )


async def run_check(
    deliverable: str, *,
    extract_fn: CompleteFn,
    revise_fn: CompleteFn | None = None,
    max_values: int = 40,
    ledger_rows: list[dict] | None = None,
    max_chars: int = DELIVERABLE_CAP,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Extract → verify → (one revision) → re-verify. Returns
    ``{text, revised, findings, initial_findings, checked, truncated}`` where
    ``text`` is the (possibly corrected) deliverable and ``findings`` reflects
    its FINAL state. The revision is kept only when the deterministic re-check
    confirms it reduced the critical/major count and it didn't collapse the text.
    """
    def _note(msg: str) -> None:
        if on_progress:
            try:
                on_progress(msg)
            except Exception:  # noqa: BLE001 — progress is cosmetic
                pass

    text = deliverable
    truncated = len(text) > max_chars
    body = text[:max_chars]
    if truncated:
        _note(f"Consistency: deliverable truncated for checking "
              f"({len(text):,} → {max_chars:,} chars) — the tail is unchecked")

    entries = parse_entries(await extract_fn(extract_prompt(body, max_values)) or "")
    findings = verify(entries, ledger_rows=ledger_rows)
    checked = {"values": len(entries["values"]), "relations": len(entries["relations"])}
    result = {"text": text, "revised": False, "findings": findings,
              "initial_findings": findings, "checked": checked, "truncated": truncated}

    fix = fix_checklist(findings)
    if not fix or revise_fn is None:
        return result

    _note(f"Consistency: {len(needs_fix(findings))} inconsistenc(ies) found — revising…")
    revised = ((await revise_fn(revise_prompt(body, fix))) or "").strip()
    collapsed = not revised or (len(body) > 800 and len(revised) < 0.5 * len(body))
    if collapsed:
        _note("Consistency: kept the original (correction pass collapsed)")
        return result
    if truncated:
        # Only the checked head was revised; the unchecked tail rides along
        # unchanged (newline-joined — the revision was stripped above).
        revised = revised + "\n" + text[max_chars:]

    # Deterministic confirmation: the revision must actually reduce the damage,
    # else it is discarded (an LLM "fix" can shuffle numbers instead of fixing them).
    entries2 = parse_entries(await extract_fn(extract_prompt(revised[:max_chars], max_values)) or "")
    findings2 = verify(entries2, ledger_rows=ledger_rows)
    if len(needs_fix(findings2)) >= len(needs_fix(findings)):
        _note("Consistency: revision did not improve the check — kept the original")
        return result
    result.update(text=revised, revised=True, findings=findings2)
    return result


# ── Reporting ──────────────────────────────────────────────────────────

def _sev_counts(findings: list[dict]) -> tuple[int, int]:
    c = sum(1 for f in findings if f.get("severity") == "critical")
    m = sum(1 for f in findings if f.get("severity") == "major")
    return c, m


def summary_line(result: dict) -> str:
    """One-line tally for the live log."""
    c0, m0 = _sev_counts(result["initial_findings"])
    line = (f"{result['checked']['values']} value(s) + "
            f"{result['checked']['relations']} relation(s) checked · "
            f"{c0} critical · {m0} major")
    if result["revised"]:
        c1, m1 = _sev_counts(result["findings"])
        line += f" · after fix: {c1} critical · {m1} major"
    if result.get("truncated"):
        line += " · TAIL UNCHECKED (truncated)"
    return line


def summarize(result: dict) -> dict:
    """The compact record persisted into the session's ``analysis.consistency``."""
    c0, m0 = _sev_counts(result["initial_findings"])
    c1, m1 = _sev_counts(result["findings"])
    return {
        "values_checked": result["checked"]["values"],
        "relations_checked": result["checked"]["relations"],
        "initial_critical": c0, "initial_major": m0,
        "critical": c1, "major": m1,
        "revised": bool(result["revised"]),
        "truncated": bool(result.get("truncated")),
    }


def _cell(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ").strip() or "—"


def audit_markdown(result: dict, *, question: str) -> str:
    """The standalone, non-destructive audit: what was checked, what conflicted,
    and whether the correction pass resolved it."""
    initial = result["initial_findings"]
    c0, m0 = _sev_counts(initial)
    header = [
        "# Consistency check report",
        "",
        f"**Task:** {(question or '').strip()[:500]}",
        "",
        f"**Result:** {summary_line(result)}",
        (f"**Action:** correction applied and re-verified "
         f"({len(needs_fix(initial)) - len(needs_fix(result['findings']))} of "
         f"{len(needs_fix(initial))} resolved)."
         if result["revised"] else
         ("**Action:** correction not applied — original kept." if needs_fix(initial)
          else "**Action:** nothing needed correcting.")),
        "",
    ]
    if result.get("truncated"):
        header.insert(-1, "**Note:** the deliverable exceeded the checking window; "
                          "its tail was NOT checked.")
        header.insert(-1, "")
    if not initial:
        header.append("_No internal inconsistencies were found._")
        return "\n".join(header)
    rows = ["| # | Severity | Type | Finding |",
            "|---|----------|------|---------|"]
    for i, f in enumerate(initial, 1):
        rows.append(f"| {i} | {_cell(f['severity'])} | {_cell(f['kind'])} | "
                    f"{_cell(f['detail'])} |")
    out = header + rows
    remaining = needs_fix(result["findings"])
    if result["revised"] and remaining:
        out += ["", "## Remaining after correction", ""]
        out += [f"- **{f['severity']}** ({f['kind']}): {f['detail']}" for f in remaining]
    return "\n".join(out) + "\n"


def write_audit(dest_dir: Path, result: dict, *, question: str,
                base_name: str = "deliverable") -> dict | None:
    """Write the audit next to the deliverable; returns a generated-file
    descriptor, or None when there was nothing to record or the write failed."""
    if not result["initial_findings"]:
        return None
    name = f"{base_name}{AUDIT_SUFFIX}"
    try:
        md = audit_markdown(result, question=question)
        p = Path(dest_dir) / name
        p.write_text(md, encoding="utf-8")
        return {"name": name, "mime": "text/markdown", "size": p.stat().st_size,
                "kind": "generated", "agent": "consistency-check"}
    except OSError as e:  # noqa: BLE001 — audit doc is best-effort
        log.warning("consistency audit write failed", error=str(e))
        return None
