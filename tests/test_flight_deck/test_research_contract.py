"""Increment 5 of the quality-tightening plan: the constraints contract.

The evaluator is the security- and correctness-critical piece: pure arithmetic
over a value dict, no eval(), nothing resolvable but numbers and ledger keys.
The fixtures mirror the DIGIT SPARK rulebook — grant range, intensity cap via a
relationship, positive co-financing — i.e. exactly the rules every failed run
in the report broke.
"""

from __future__ import annotations

import json

import pytest

from captain_claw.flight_deck.quality_profile import build_quality_metrics
from captain_claw.flight_deck.research_contract import (
    MissingKey,
    apply_judgement,
    contract_directive,
    derive_prompt,
    eval_predicate,
    evaluate_check,
    judge_prompt,
    load,
    parse_contract,
    parse_judgement,
    save,
    summarize,
    validate,
)

_VALS = {"grant_eur": 105000, "total_eligible_cost_eur": 300000,
         "own_contribution_eur": 195000, "intensity": 0.35,
         "duration_months": 16}


# ── the safe evaluator ───────────────────────────────────────────────

def test_arithmetic_precedence_and_parens():
    assert eval_predicate("2 + 3 * 4 == 14", {}) is True
    assert eval_predicate("(2 + 3) * 4 == 20", {}) is True
    assert eval_predicate("-5 + 10 > 0", {}) is True


def test_keys_resolve_case_insensitively():
    assert eval_predicate("grant_eur <= 0.5 * TOTAL_ELIGIBLE_COST_EUR", _VALS) is True
    assert eval_predicate("grant_eur > total_eligible_cost_eur", _VALS) is False


def test_chained_comparison_is_a_range():
    assert eval_predicate("80000 <= grant_eur <= 150000", _VALS) is True
    assert eval_predicate("80000 <= grant_eur <= 100000", _VALS) is False


def test_equality_uses_relative_tolerance():
    assert eval_predicate("own_contribution_eur == total_eligible_cost_eur - grant_eur",
                          _VALS) is True
    assert eval_predicate("grant_eur == 105001", _VALS) is True   # 0.001% drift
    assert eval_predicate("intensity == 0.36", _VALS) is False    # 2.8% is real


def test_missing_key_and_malformed_predicates_raise():
    with pytest.raises(MissingKey):
        eval_predicate("grant_eur <= unknown_key", _VALS)
    with pytest.raises(ValueError):
        eval_predicate("grant_eur + 5", _VALS)          # no comparison
    with pytest.raises(ValueError):
        eval_predicate("grant_eur / 0 > 1", _VALS)      # division by zero
    with pytest.raises(ValueError):
        eval_predicate("__import__('os') > 1", _VALS)   # call syntax is not grammar
    with pytest.raises((ValueError, MissingKey)):
        eval_predicate("grant_eur.__class__ > 1", _VALS)  # no attribute access


def test_evaluate_check_range_equals_and_judge():
    assert evaluate_check({"type": "range", "key": "grant_eur",
                           "min": 80000, "max": 150000}, _VALS) is True
    assert evaluate_check({"type": "range", "key": "duration_months", "max": 12},
                          _VALS) is False
    assert evaluate_check({"type": "equals", "key": "intensity", "value": 0.35},
                          _VALS) is True
    with pytest.raises(MissingKey):
        evaluate_check({"type": "range", "key": "nope", "max": 1}, _VALS)
    with pytest.raises(ValueError):
        evaluate_check({"type": "judge"}, _VALS)


# ── parsing / persistence ────────────────────────────────────────────

_RAW = json.dumps({"constraints": [
    {"id": "c1", "text": "grant between €80k and €150k", "severity": "critical",
     "check": {"type": "range", "key": "Grant (EUR)", "min": 80000, "max": 150000}},
    {"text": "own contribution must be positive", "severity": "CRITICAL",
     "check": {"type": "expr", "expr": "total_eligible_cost_eur - grant_eur > 0"}},
    {"text": "application in Croatian", "severity": "bogus",
     "check": {"type": "language?!"}},
    {"text": "", "check": {"type": "judge"}},                    # dropped: no text
    {"text": "range with no bounds", "check": {"type": "range", "key": "x"}},  # → judge
]})


def test_parse_contract_normalizes_defensively():
    items = parse_contract(f"```json\n{_RAW}\n```")
    assert len(items) == 4
    assert items[0]["check"] == {"type": "range", "key": "grant_eur",
                                 "min": 80000.0, "max": 150000.0}
    assert items[1]["id"] == "c2" and items[1]["severity"] == "critical"
    assert items[2]["severity"] == "major" and items[2]["check"] == {"type": "judge"}
    assert items[3]["check"] == {"type": "judge"}
    assert parse_contract("no json") == []


def test_save_load_roundtrip_and_renormalization(tmp_path):
    items = parse_contract(_RAW)
    save(tmp_path, items, "fill the DIGIT SPARK form")
    loaded = load(tmp_path)
    assert loaded == items
    assert load(tmp_path / "nowhere") is None
    # A hand-edited file with a malformed check degrades to judge, not a crash.
    raw = json.loads((tmp_path / ".contract.json").read_text())
    raw["constraints"][0]["check"] = {"type": "range"}  # key removed
    (tmp_path / ".contract.json").write_text(json.dumps(raw))
    assert load(tmp_path)[0]["check"] == {"type": "judge"}


# ── validation + judge fold-back ─────────────────────────────────────

def _contract() -> list[dict]:
    return parse_contract(json.dumps({"constraints": [
        {"id": "c1", "text": "grant in range", "severity": "critical",
         "check": {"type": "range", "key": "grant_eur", "min": 80000, "max": 150000}},
        {"id": "c2", "text": "intensity at most 50%", "severity": "critical",
         "check": {"type": "expr", "expr": "grant_eur <= 0.5 * total_eligible_cost_eur"}},
        {"id": "c3", "text": "duration cap", "severity": "major",
         "check": {"type": "range", "key": "duration_months", "max": 16}},
        {"id": "c4", "text": "written in Croatian", "severity": "major",
         "check": {"type": "judge"}},
        {"id": "c5", "text": "mystery quantity in range", "severity": "minor",
         "check": {"type": "range", "key": "not_in_ledger", "max": 10}},
    ]}))


def test_validate_splits_deterministic_from_unresolved():
    res = validate(_contract(), _VALS)
    assert [e["id"] for e in res["passed"]] == ["c1", "c2", "c3"]
    assert res["failed"] == []
    assert [c["id"] for c in res["unresolved"]] == ["c4", "c5"]


def test_validate_catches_the_report_failure_classes():
    # Qwen's 100% intensity / €0 co-financing world: grant == total.
    bad = dict(_VALS, grant_eur=300000)
    res = validate(_contract(), bad)
    failed = {e["id"] for e in res["failed"]}
    assert failed == {"c1", "c2"}  # over grant_max AND over the intensity cap
    assert all(e["how"] == "deterministic" for e in res["failed"])


def test_apply_judgement_folds_verdicts_and_never_guesses():
    res = validate(_contract(), _VALS)
    judged = parse_judgement(json.dumps([
        {"id": "c4", "verdict": "fail", "note": "the deliverable is in English"},
        # c5 not mentioned → unclear, not guessed
    ]))
    apply_judgement(res, judged)
    assert res["unresolved"] == []
    assert [e["id"] for e in res["failed"]] == ["c4"]
    assert res["failed"][0]["how"] == "judged"
    assert [e["id"] for e in res["unclear"]] == ["c5"]


def test_summarize_counts_feed_the_metrics():
    res = validate(_contract(), dict(_VALS, grant_eur=300000))
    apply_judgement(res, [])
    s = summarize(res)
    assert s["checked"] == 5 and s["failed_critical"] == 2
    assert s["unclear"] == 2
    assert {f["id"] for f in s["failed"]} == {"c1", "c2"}
    m = build_quality_metrics(contract=s)
    assert m["contract_failed_critical"] == 2 and m["contract_checked"] == 5
    assert m["contract_unclear"] == 2


# ── prompts / directive ──────────────────────────────────────────────

def test_directive_lists_rules_and_ledger_hint_is_optional():
    items = _contract()
    d = contract_directive(items, ledger=True)
    assert "HARD CONSTRAINTS" in d and "[critical] grant in range" in d
    assert "facts ledger" in d
    assert "facts ledger" not in contract_directive(items, ledger=False)
    assert contract_directive([]) == ""


def test_judge_prompt_and_derive_prompt_shapes():
    jp = judge_prompt("the deliverable text", _contract()[3:])
    assert 'id "c4"' in jp and "pass|fail|unclear" in jp
    dp = derive_prompt("fill the form")
    assert "snake_case" in dp and '"constraints": []' in dp
