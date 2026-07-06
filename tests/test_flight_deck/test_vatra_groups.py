"""Vatra execution-group resolver: archetype presets, the Lead-may-push-later
rule, and phase ordering."""

from __future__ import annotations

from pathlib import Path

import captain_claw
from captain_claw.flight_deck import vatra_groups as g

_LEAD_MD = Path(captain_claw.__file__).resolve().parent / "instructions" / "vatra" / "lead.md"


# ── presets ──────────────────────────────────────────────────────────

def test_research_and_design_are_group_a():
    assert g.archetype_group({"id": "deep-researcher", "family": "Research & Intelligence"}) == 1
    assert g.archetype_group({"id": "market-scanner", "role": "Market & Competitor Scanner"}) == 1
    assert g.archetype_group({"id": "architect", "role": "Software Architect"}) == 1
    assert g.archetype_group({"id": "long-horizon-planner", "role": "Long Horizon Planner"}) == 1


def test_review_and_assembly_are_group_c():
    assert g.archetype_group({"id": "report-builder", "role": "Report Builder"}) == 3
    assert g.archetype_group({"id": "debugger", "role": "Debugger"}) == 3
    assert g.archetype_group({"id": "code-reviewer", "role": "Code Reviewer"}) == 3
    assert g.archetype_group({"id": "security-reviewer", "role": "Security Reviewer"}) == 3


def test_build_and_write_default_to_middle_b():
    assert g.archetype_group({"id": "code-implementer", "role": "Software Implementer"}) == 2
    assert g.archetype_group({"id": "data-analyst", "role": "Data Analyst"}) == 2
    assert g.archetype_group({"id": "editor-writer", "role": "Editor & Long-form Writer"}) == 2


def test_explicit_group_field_overrides_heuristics():
    # A researcher pinned to D by config.
    assert g.archetype_group({"id": "deep-researcher", "family": "Research", "group": "D"}) == 4
    assert g.archetype_group({"id": "report-builder", "group": 1}) == 1


# ── labels ───────────────────────────────────────────────────────────

def test_group_labels():
    assert [g.group_label(o) for o in (1, 2, 3, 4)] == ["A", "B", "C", "D"]
    assert g.group_label(99) == "D" and g.group_label(0) == "A"  # clamped


# ── the Lead-may-push-later rule ─────────────────────────────────────

def test_lead_can_push_later_but_never_earlier():
    arch = {"id": "deep-researcher", "family": "Research"}  # floor = A (1)
    assert g.effective_group({}, arch) == 1                      # no override → floor
    assert g.effective_group({"group": "C"}, arch) == 3          # pushed later
    assert g.effective_group({"group": "A"}, arch) == 1          # can't go below floor
    report = {"id": "report-builder"}  # floor = C (3)
    assert g.effective_group({"group": "A"}, report) == 3        # Lead tries earlier → clamped to floor
    assert g.effective_group({"group": "D"}, report) == 4        # later is fine


def test_clamp_lead_group():
    assert g.clamp_lead_group("A", floor=3) == 3    # can't go below floor
    assert g.clamp_lead_group("D", floor=1) == 4
    assert g.clamp_lead_group(None, floor=2) is None
    assert g.clamp_lead_group("nonsense", floor=2) is None


# ── phase ordering ───────────────────────────────────────────────────

def test_order_groups_runs_distinct_ascending():
    # A team of {A, C} runs two phases, A then C — not four.
    assert g.order_groups([3, 1, 1, 3]) == [1, 3]
    assert g.order_groups([2, 2, 2]) == [2]
    assert g.order_groups([]) == []


# ── clarification loop helpers ───────────────────────────────────────

def test_parse_request_extracts_the_ask():
    out = ("Here is my draft.\n\n"
           "REQUEST: Market & Competitor Scanner — the confirmed list of EV models for Croatia")
    r = g.parse_request(out)
    assert r and "Market & Competitor Scanner" in r and "EV models" in r


def test_parse_request_none_when_absent():
    assert g.parse_request("just a normal answer, no marker") is None
    assert g.parse_request("") is None
    assert g.parse_request(None) is None


def test_parse_clarify_approve_and_deny():
    approve = g.parse_clarify('{"approve": true, "provider": "market-scanner", '
                              '"instruction": "Post the final model list."}')
    assert approve["approve"] is True
    assert approve["provider"] == "market-scanner"
    assert "model list" in approve["instruction"]
    deny = g.parse_clarify('Sorry: {"approve": false, "provider": "", "instruction": ""}')
    assert deny["approve"] is False


def test_parse_clarify_defaults_to_deny_on_garbage():
    for bad in ("", "not json", "{broken", None):
        assert g.parse_clarify(bad)["approve"] is False


def test_clarify_prompt_names_requester_request_and_roster():
    p = g.clarify_prompt("Report Builder", "need the price table",
                         [{"id": "data-analyst", "role": "Data Analyst", "title": "Consolidate"}])
    assert "Report Builder" in p and "need the price table" in p
    assert "data-analyst" in p and "Data Analyst" in p
    assert "approve" in p  # asks for the JSON decision


def test_request_directive_and_cap_exist():
    assert "REQUEST:" in g.REQUEST_DIRECTIVE
    assert g.CLARIFY_CAP == 2


# ── live-panel phase letters (Increment 3) ───────────────────────────

def test_owner_group_letters_for_a_mixed_roster():
    # The exact computation execute_vatra runs inline to tag each owner's live-panel
    # events: effective_group (floor raised by any Lead push) → group_label.
    roster = [
        ({"id": "s1"}, {"id": "deep-researcher", "family": "Research"}),            # A (floor)
        ({"id": "s2"}, {"id": "data-analyst", "role": "Data Analyst"}),             # B (middle)
        ({"id": "s3", "group": "D"}, {"id": "editor-writer", "role": "Editor"}),    # B pushed → D
        ({"id": "s4"}, {"id": "report-builder", "role": "Report Builder"}),         # C (floor)
        ({"id": "s5", "group": "A"}, {"id": "report-builder", "role": "Report"}),   # can't go below C
    ]
    letters = {st["id"]: g.group_label(g.effective_group(st, arch)) for st, arch in roster}
    assert letters == {"s1": "A", "s2": "B", "s3": "D", "s4": "C", "s5": "C"}
    # The phases that actually run are the DISTINCT letters, ascending.
    ords = [g.effective_group(st, arch) for st, arch in roster]
    assert [g.group_label(o) for o in g.order_groups(ords)] == ["A", "B", "C", "D"]


# ── Lead prompt teaches the optional group field (Increment 3) ───────

def test_lead_prompt_teaches_the_optional_group_field():
    text = _LEAD_MD.read_text(encoding="utf-8")
    lo = text.lower()
    assert "group" in lo                       # the field is documented
    assert "optional" in lo                     # framed as optional, not mandatory
    assert '"a"' in lo and '"d"' in lo          # names the A..D letter scale
    assert "later" in lo                        # the push-later-only rule
