"""Vatra execution-group resolver: archetype presets, the Lead-may-push-later
rule, and phase ordering."""

from __future__ import annotations

from captain_claw.flight_deck import vatra_groups as g


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
