"""Tests for R9 — the pure parts of the rubric contract (parse + inject + coverage)."""

from __future__ import annotations

from captain_claw.flight_deck import research_rubric as rr


def test_parse_rubric_json_array():
    items = rr.parse_rubric('["Controller details", "Lawful basis", "Retention"]')
    assert items == ["Controller details", "Lawful basis", "Retention"]


def test_parse_rubric_fenced_with_prose():
    out = 'The checklist:\n```json\n["A","B","C"]\n```\ndone'
    assert rr.parse_rubric(out) == ["A", "B", "C"]


def test_parse_rubric_caps_items():
    big = "[" + ",".join(f'"item{i}"' for i in range(50)) + "]"
    assert len(rr.parse_rubric(big)) == 30


def test_parse_rubric_garbage_is_empty():
    assert rr.parse_rubric("no json") == []
    assert rr.parse_rubric("") == []


def test_rubric_directive_lists_items_and_forbids_routing_out():
    d = rr.rubric_directive(["Controller details", "DPIA determination"])
    assert "Controller details" in d and "DPIA determination" in d
    assert "definition of 'complete'" in d
    assert "separate document" in d  # the anti-scope-shrink rule
    assert rr.rubric_directive([]) == ""


def test_coverage_parse_missing_and_thin():
    out = '{"missing": ["Processor annex", "Source of data"], "thin": ["Security measures"]}'
    cov = rr.parse_coverage(out)
    assert cov["missing"] == ["Processor annex", "Source of data"]
    assert cov["thin"] == ["Security measures"]


def test_coverage_parse_garbage_is_empty():
    assert rr.parse_coverage("nope") == {"missing": [], "thin": []}


def test_coverage_prompt_includes_checklist_and_deliverable():
    p = rr.coverage_prompt("do X", ["item A", "item B"], "the deliverable text")
    assert "item A" in p and "the deliverable text" in p and "do X" in p
