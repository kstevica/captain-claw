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


def test_prompt_instructs_hedging_of_unconfirmable_specifics():
    p = rv.claim_check_prompt("d", "q", 8)
    assert "hedge" in p
    assert "unverifiable" in p
    # Domain-agnostic framing: it names the CLASS (an appointed officer), not a field.
    assert "appointed officer" in p


# ── unconfirmed / hedging (the DPO-fabrication fix) ──────────────────

def test_parse_keeps_hedge_field():
    out = ('[{"claim":"The DPO is Jane Roe","verdict":"unverifiable",'
           '"hedge":"The DPO is unconfirmed; no public source names one.",'
           '"evidence":"searched registry + site"}]')
    f = rv.parse_findings(out)
    assert len(f) == 1
    assert f[0]["verdict"] == "unverifiable"
    assert "unconfirmed" in f[0]["hedge"]


def test_unconfirmed_needs_a_hedge():
    findings = [
        {"claim": "named DPO", "verdict": "unverifiable", "hedge": "say unconfirmed", "correction": ""},
        {"claim": "vague thing", "verdict": "unverifiable", "hedge": "", "correction": ""},  # no hedge → skip
        {"claim": "checked ok", "verdict": "confirmed", "hedge": "", "correction": ""},
    ]
    soft = rv.unconfirmed(findings)
    assert [f["claim"] for f in soft] == ["named DPO"]


def test_fix_instructions_covers_refuted_and_unconfirmed():
    findings = [
        {"claim": "NN 73/08 current", "verdict": "refuted",
         "correction": "use NN 76/22", "evidence": "zakon.hr", "hedge": ""},
        {"claim": "DPO is Jane Roe", "verdict": "unverifiable",
         "correction": "", "hedge": "state the DPO as unconfirmed", "evidence": "site"},
    ]
    fix = rv.fix_instructions(findings)
    assert "1." in fix and "2." in fix
    assert "use NN 76/22" in fix           # refuted correction
    assert "state the DPO as unconfirmed" in fix  # the hedge
    assert "UNCONFIRMED" in fix            # the unverifiable-asserted class is labelled


def test_fix_instructions_fires_on_hedge_alone():
    # A run with zero refutations but one asserted-but-unconfirmable specific
    # must STILL trigger a revision — this is exactly the fabricated-DPO case.
    findings = [{"claim": "DPO is Jane Roe", "verdict": "unverifiable",
                 "correction": "", "hedge": "mark the DPO unconfirmed", "evidence": ""}]
    assert rv.fix_instructions(findings) != ""


def test_summary_line_reports_hedged_count():
    findings = [
        {"claim": "a", "verdict": "unverifiable", "hedge": "h1", "correction": ""},
        {"claim": "b", "verdict": "unverifiable", "hedge": "", "correction": ""},
    ]
    s = rv.summary_line(findings)
    assert "2 unverifiable" in s and "1 hedged" in s


# ── audit ledger ─────────────────────────────────────────────────────

def test_audit_markdown_lists_every_claim_and_action():
    findings = [
        {"claim": "founded 1996", "verdict": "confirmed", "correction": "", "hedge": "", "evidence": "reg"},
        {"claim": "NN 73/08 current", "verdict": "refuted",
         "correction": "NN 76/22", "hedge": "", "evidence": "zakon.hr"},
        {"claim": "DPO is Jane Roe", "verdict": "unverifiable",
         "correction": "", "hedge": "mark DPO unconfirmed", "evidence": "site"},
    ]
    md = rv.audit_markdown(findings, question="ROPA for X", revised=True)
    assert "# Fact-check report" in md
    assert "ROPA for X" in md
    assert "founded 1996" in md and "NN 76/22" in md and "mark DPO unconfirmed" in md
    assert "3 claim(s) checked" in md


def test_audit_markdown_empty_findings_says_nothing_flagged():
    md = rv.audit_markdown([], question="q", revised=False)
    assert "No load-bearing claims" in md


def test_audit_markdown_escapes_pipes_in_cells():
    findings = [{"claim": "a | b table pipe", "verdict": "confirmed",
                 "correction": "", "hedge": "", "evidence": ""}]
    md = rv.audit_markdown(findings, question="q", revised=False)
    assert "a \\| b" in md  # pipe escaped so it can't break the table


def test_write_audit_writes_file_and_returns_descriptor(tmp_path):
    findings = [{"claim": "x", "verdict": "refuted", "correction": "y",
                 "hedge": "", "evidence": "src"}]
    doc = rv.write_audit(tmp_path, findings, question="q", revised=True, base_name="report")
    assert doc is not None
    assert doc["name"] == "report" + rv.AUDIT_SUFFIX
    assert doc["kind"] == "generated" and doc["mime"] == "text/markdown"
    written = (tmp_path / doc["name"]).read_text()
    assert "# Fact-check report" in written


def test_write_audit_skips_when_no_findings(tmp_path):
    assert rv.write_audit(tmp_path, [], question="q", revised=False) is None
    assert not list(tmp_path.iterdir())
