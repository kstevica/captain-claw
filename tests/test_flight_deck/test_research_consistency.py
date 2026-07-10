"""Increment 2 of the quality-tightening plan: deterministic consistency check.

The verify() fixtures ARE the DIGIT SPARK report's regression suite, generalized:
same-quantity value drift across sections (the €549k-vs-€157k class), budget
closure (Σ parts == total), percentage-of-base recomputation, and rounding
tolerance. The LLM only extracts; every failure asserted here is caught by plain
arithmetic, so none of these can regress with a model change.
"""

from __future__ import annotations

import json

from captain_claw.flight_deck.research_consistency import (
    audit_markdown,
    fix_checklist,
    needs_fix,
    parse_entries,
    run_check,
    summarize,
    summary_line,
    verify,
    write_audit,
)


def _v(label: str, value, *, kind: str = "figure", unit: str = "EUR",
       raw: str = "", quote: str = "") -> dict:
    return {"label": label, "kind": kind, "raw": raw or str(value),
            "value": value, "unit": unit, "quote": quote}


# ── parse_entries ────────────────────────────────────────────────────

def test_parse_handles_fences_prose_and_garbage():
    payload = {"values": [_v("total budget", 300000)], "relations": []}
    fenced = f"Here you go:\n```json\n{json.dumps(payload)}\n```\nDone."
    bare = f"noise before {json.dumps(payload)} noise after"
    assert parse_entries(fenced)["values"][0]["label"] == "total budget"
    assert parse_entries(bare)["values"][0]["value"] == 300000
    assert parse_entries("no json here") == {"values": [], "relations": []}
    assert parse_entries("") == {"values": [], "relations": []}
    assert parse_entries("[1, 2, 3]") == {"values": [], "relations": []}


def test_parse_drops_uncomparable_rows_instead_of_failing():
    raw = json.dumps({"values": [
        {"label": "budget", "kind": "figure", "value": "not-a-number"},
        {"label": "", "kind": "figure", "value": 5},
        {"label": "oib", "kind": "bogus-kind", "value": "52435554303"},  # coerced
        {"label": "employees", "kind": "bogus-kind", "value": 51},       # coerced
    ], "relations": [
        {"type": "sum", "operands": [], "result": "x"},                  # no operands
        {"type": "percent_of", "operands": ["a"], "result": "b"},        # no percent
        {"type": "ratio", "operands": ["a"], "result": "b"},             # bad type
    ]})
    out = parse_entries(raw)
    kinds = {e["label"]: e["kind"] for e in out["values"]}
    assert kinds == {"oib": "identifier", "employees": "figure"}
    assert out["relations"] == []


# ── verify: identity across sections ─────────────────────────────────

def test_same_quantity_with_conflicting_values_is_critical():
    # The DeepSeek failure class: personnel stated as €549,620 in one table
    # and ~€157,000 in another. No fact-checker caught it; arithmetic does.
    entries = {"values": [
        _v("personnel cost", 549620, quote="personnel costs total €549,620"),
        _v("personnel cost", 157000, quote="personnel budget line €157,000"),
    ], "relations": []}
    findings = verify(entries)
    assert len(findings) == 1
    f = findings[0]
    assert f["severity"] == "critical" and f["kind"] == "identity"
    assert "549" in f["detail"] and "157" in f["detail"]


def test_rounding_drift_is_tolerated():
    entries = {"values": [
        _v("total budget", 300000),
        _v("total budget", 300000.4),
    ], "relations": []}
    assert verify(entries) == []


def test_different_units_are_not_compared():
    # "duration" in months in one place and EUR of cost elsewhere is an
    # extraction-labeling artifact, not a numeric contradiction.
    entries = {"values": [
        _v("activity 1", 16, unit="months"),
        _v("activity 1", 150000, unit="EUR"),
    ], "relations": []}
    assert verify(entries) == []


def test_date_precision_is_compatible_but_conflict_is_critical():
    ok = {"values": [
        _v("completion date", "2028-10", kind="date", unit=""),
        _v("completion date", "2028-10-31", kind="date", unit=""),
    ], "relations": []}
    assert verify(ok) == []
    bad = {"values": [
        _v("completion date", "2028-10-31", kind="date", unit=""),
        _v("completion date", "2027-06-30", kind="date", unit=""),
    ], "relations": []}
    assert [f["severity"] for f in verify(bad)] == ["critical"]


def test_identifier_conflict_is_critical():
    entries = {"values": [
        _v("call reference", "DIGIT.2.1.03", kind="identifier", unit=""),
        _v("call reference", "DIGIT.2.1.04", kind="identifier", unit=""),
    ], "relations": []}
    assert [f["kind"] for f in verify(entries)] == ["identity"]


# ── verify: stated relations recomputed in code ──────────────────────

def test_budget_closure_sum_mismatch_is_critical():
    # Regression 2 from the report: Σ(cost lines) must equal the stated total.
    entries = {"values": [
        _v("personnel", 157000), _v("equipment", 50000),
        _v("total eligible cost", 300000),
    ], "relations": [
        {"type": "sum", "operands": ["personnel", "equipment"],
         "result": "total eligible cost", "quote": "total = personnel + equipment"},
    ]}
    findings = verify(entries)
    assert len(findings) == 1
    assert findings[0]["kind"] == "relation" and findings[0]["severity"] == "critical"
    assert "207,000" in findings[0]["detail"]


def test_sum_within_tolerance_passes():
    entries = {"values": [
        _v("a", 100000.4), _v("b", 199999.8), _v("total", 300000),
    ], "relations": [
        {"type": "sum", "operands": ["a", "b"], "result": "total", "quote": ""},
    ]}
    assert verify(entries) == []


def test_percent_of_recomputes():
    # Regression 6: indirect == 20% of Activity 1 direct (not 15%, not of total).
    base = [_v("activity 1 direct costs", 150000)]
    ok = {"values": base + [_v("indirect costs", 30000)], "relations": [
        {"type": "percent_of", "percent": 20, "operands": ["activity 1 direct costs"],
         "result": "indirect costs", "quote": ""}]}
    assert verify(ok) == []
    bad = {"values": base + [_v("indirect costs", 22500)], "relations": [
        {"type": "percent_of", "percent": 20, "operands": ["activity 1 direct costs"],
         "result": "indirect costs", "quote": ""}]}
    findings = verify(bad)
    assert len(findings) == 1 and findings[0]["severity"] == "critical"


def test_difference_and_product_relations():
    entries = {"values": [
        _v("total", 300000), _v("grant", 105000), _v("own contribution", 195000),
    ], "relations": [
        {"type": "difference", "operands": ["total", "grant"],
         "result": "own contribution", "quote": ""},
    ]}
    assert verify(entries) == []
    entries["values"][2] = _v("own contribution", 150000)
    assert len(verify(entries)) == 1


def test_relation_with_missing_operand_is_skipped():
    entries = {"values": [_v("total", 300000)], "relations": [
        {"type": "sum", "operands": ["never extracted"], "result": "total", "quote": ""},
    ]}
    assert verify(entries) == []


def test_label_matching_is_case_and_whitespace_insensitive():
    entries = {"values": [
        _v("Total  Eligible Cost", 300000),
        _v("total eligible cost.", 250000),
    ], "relations": []}
    assert len(verify(entries)) == 1


# ── verify: ledger cross-check ───────────────────────────────────────

def test_ledger_mismatch_is_critical_and_match_is_silent():
    entries = {"values": [_v("total eligible cost", 350000)], "relations": []}
    rows = [{"key": "total eligible cost", "value": 300000}]
    findings = verify(entries, ledger_rows=rows)
    assert len(findings) == 1 and findings[0]["kind"] == "ledger"
    entries["values"][0] = _v("total eligible cost", 300000)
    assert verify(entries, ledger_rows=rows) == []


# ── fix checklist ────────────────────────────────────────────────────

def test_checklist_is_empty_when_clean_and_numbered_when_not():
    assert fix_checklist([]) == ""
    findings = verify({"values": [
        _v("personnel", 549620), _v("personnel", 157000),
    ], "relations": []})
    text = fix_checklist(findings)
    assert text.startswith("Deterministic arithmetic")
    assert "1. " in text and "EVERYWHERE" in text


# ── run_check orchestration (stubbed model calls) ────────────────────

_CLEAN = json.dumps({"values": [_v("total", 300000)], "relations": []})
_BROKEN = json.dumps({"values": [
    _v("personnel", 549620), _v("personnel", 157000)], "relations": []})


def _seq_fn(outputs: list[str], calls: list[str]):
    async def fn(prompt: str) -> str:
        calls.append(prompt)
        return outputs[min(len(calls) - 1, len(outputs) - 1)]
    return fn


async def test_clean_deliverable_never_pays_for_a_revision():
    extract_calls: list[str] = []
    revise_calls: list[str] = []
    res = await run_check(
        "doc " * 100,
        extract_fn=_seq_fn([_CLEAN], extract_calls),
        revise_fn=_seq_fn(["should never run"], revise_calls))
    assert res["findings"] == [] and res["revised"] is False
    assert len(extract_calls) == 1 and revise_calls == []


async def test_confirmed_fix_is_kept():
    calls: list[str] = []

    async def extract(prompt: str) -> str:
        calls.append("x")
        return _BROKEN if len(calls) == 1 else _CLEAN  # re-check comes back clean

    long_doc = "the personnel figure appears twice. " * 40
    res = await run_check(long_doc, extract_fn=extract,
                          revise_fn=_seq_fn([long_doc.replace("twice", "once")], []))
    assert res["revised"] is True
    assert "once" in res["text"]
    assert res["findings"] == [] and len(res["initial_findings"]) == 1
    assert len(calls) == 2  # extract + deterministic re-check


async def test_collapsed_revision_is_rejected():
    long_doc = "line of real content here. " * 60  # > 800 chars
    res = await run_check(long_doc,
                          extract_fn=_seq_fn([_BROKEN], []),
                          revise_fn=_seq_fn(["too short"], []))
    assert res["revised"] is False and res["text"] == long_doc
    assert len(res["initial_findings"]) == 1


async def test_non_improving_revision_is_rejected():
    long_doc = "content " * 200
    res = await run_check(long_doc,
                          extract_fn=_seq_fn([_BROKEN, _BROKEN], []),  # still broken after "fix"
                          revise_fn=_seq_fn([long_doc + " revised"], []))
    assert res["revised"] is False and res["text"] == long_doc
    # Findings reflect the KEPT text (the original), not the discarded revision.
    assert len(res["findings"]) == 1


async def test_truncation_is_flagged_and_tail_survives_revision():
    head = "checked head content. " * 50
    tail = "UNCHECKED-TAIL-MARKER " * 10
    doc = head + tail
    cap = len(head)
    calls: list[str] = []

    async def extract_seq(prompt: str) -> str:
        calls.append("x")
        return _BROKEN if len(calls) == 1 else _CLEAN

    res = await run_check(doc, extract_fn=extract_seq,
                          revise_fn=_seq_fn([head.upper()], []),
                          max_chars=cap)
    assert res["truncated"] is True
    assert res["revised"] is True
    assert res["text"].endswith(tail)          # tail spliced back untouched
    assert res["text"].startswith(head.upper().strip())


# ── reporting ────────────────────────────────────────────────────────

def _result(findings: list[dict], *, revised: bool = False) -> dict:
    return {"text": "doc", "revised": revised, "findings": [] if revised else findings,
            "initial_findings": findings,
            "checked": {"values": 5, "relations": 2}, "truncated": False}


def test_summary_and_summarize_tally():
    findings = verify({"values": [
        _v("personnel", 549620), _v("personnel", 157000)], "relations": []})
    line = summary_line(_result(findings))
    assert "5 value(s)" in line and "1 critical" in line
    s = summarize(_result(findings, revised=True))
    assert s["initial_critical"] == 1 and s["critical"] == 0 and s["revised"] is True


def test_audit_written_only_when_something_was_found(tmp_path):
    clean = _result([])
    assert write_audit(tmp_path, clean, question="q") is None
    findings = verify({"values": [
        _v("personnel", 549620), _v("personnel", 157000)], "relations": []})
    doc = write_audit(tmp_path, _result(findings), question="fill the form")
    assert doc is not None and doc["name"] == "deliverable.consistency.md"
    md = (tmp_path / doc["name"]).read_text()
    assert "Consistency check report" in md and "critical" in md
    assert "fill the form" in md


def test_audit_markdown_notes_remaining_findings_after_partial_fix():
    findings = verify({"values": [
        _v("personnel", 549620), _v("personnel", 157000)], "relations": []})
    partial = {"text": "doc", "revised": True, "findings": findings,
               "initial_findings": findings + findings,
               "checked": {"values": 4, "relations": 0}, "truncated": False}
    md = audit_markdown(partial, question="q")
    assert "Remaining after correction" in md
    assert needs_fix(findings)  # sanity: these are fix-worthy findings
