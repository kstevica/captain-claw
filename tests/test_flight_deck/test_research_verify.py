"""Tests for R8 — the pure part of grounded claim verification (parse + fix build)."""

from __future__ import annotations

from captain_claw.flight_deck import research_verify as rv


def test_parse_plain_json_array():
    out = ('[{"claim":"NN 73/08 is current","verdict":"refuted",'
           '"correction":"NN 73/08 is repealed; current is NN 76/22","evidence":"zakon.hr"}]')
    f = rv.parse_findings(out)
    assert len(f) == 1
    assert f[0]["verdict"] == "refuted"
    assert "76/22" in f[0]["correction"]


def test_parse_fenced_json_with_prose():
    out = ("Here are my findings:\n```json\n"
           '[{"claim":"Founded 1996","verdict":"confirmed","correction":"","evidence":"registry"}]'
           "\n```\nDone.")
    f = rv.parse_findings(out)
    assert len(f) == 1 and f[0]["verdict"] == "confirmed"


def test_parse_invalid_verdict_becomes_unverifiable():
    f = rv.parse_findings('[{"claim":"x","verdict":"maybe"}]')
    assert f[0]["verdict"] == "unverifiable"


def test_parse_garbage_returns_empty():
    assert rv.parse_findings("no json here") == []
    assert rv.parse_findings("") == []
    assert rv.parse_findings("[not, valid, json") == []


def test_refuted_needs_a_correction():
    findings = [
        {"claim": "a", "verdict": "refuted", "correction": "fix a", "evidence": ""},
        {"claim": "b", "verdict": "refuted", "correction": "", "evidence": ""},  # no fix → skip
        {"claim": "c", "verdict": "confirmed", "correction": "", "evidence": ""},
    ]
    bad = rv.refuted(findings)
    assert [f["claim"] for f in bad] == ["a"]


def test_fix_instructions_numbers_only_refuted():
    findings = [
        {"claim": "NN 73/08 current", "verdict": "refuted",
         "correction": "use NN 76/22", "evidence": "zakon.hr"},
        {"claim": "founded 1996", "verdict": "confirmed", "correction": "", "evidence": ""},
    ]
    fix = rv.fix_instructions(findings)
    assert "1." in fix and "use NN 76/22" in fix and "zakon.hr" in fix
    assert "founded 1996" not in fix  # confirmed claims aren't in the fix list


def test_fix_instructions_empty_when_all_ok():
    assert rv.fix_instructions([{"claim": "x", "verdict": "confirmed", "correction": ""}]) == ""
    assert rv.fix_instructions([]) == ""


def test_summary_line_tally():
    findings = [
        {"claim": "a", "verdict": "confirmed", "correction": ""},
        {"claim": "b", "verdict": "refuted", "correction": "y"},
        {"claim": "c", "verdict": "unverifiable", "correction": ""},
    ]
    s = rv.summary_line(findings)
    assert "3 claim(s)" in s and "1 confirmed" in s and "1 refuted" in s and "1 unverifiable" in s


def test_prompt_mentions_corpus_only_when_hinted():
    assert "saved sources" in rv.claim_check_prompt("d", "q", 8, corpus_hint=True)
    assert "saved sources" not in rv.claim_check_prompt("d", "q", 8, corpus_hint=False)
