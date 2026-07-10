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


# ── dependency repair (resolve_groups): deps out-rank floors + pushes ──
# The observed failure: the Lead pushed the fact-checker (floor A) to group D
# while the group-B data-analyst depends_on it — the analyst then waited at
# runtime for output scheduled to be produced AFTER it.

_ARCHS = {
    "deep-researcher": {"id": "deep-researcher", "family": "Research"},
    "fact-checker": {"id": "fact-checker", "role": "Fact Checker"},
    "data-analyst": {"id": "data-analyst", "role": "Data Analyst"},
    "editor-writer": {"id": "editor-writer", "role": "Editor"},
}


def _st(sid, owner, deps=(), group=None, title=""):
    s = {"id": sid, "owner_archetype_id": owner, "title": title or sid,
         "brief": "x", "depends_on": list(deps)}
    if group is not None:
        s["group"] = group
    return s


def test_repair_pulls_a_pushed_dependency_back_to_its_dependent():
    subtasks = [
        _st("s1", "deep-researcher"),                       # A
        _st("s2", "fact-checker", group="D"),               # floor A, pushed D
        _st("s3", "data-analyst", deps=["s1", "s2"]),       # B — needs s2!
    ]
    notes = g.resolve_groups(subtasks, _ARCHS)
    by_id = {s["id"]: s for s in subtasks}
    assert by_id["s2"]["group_resolved"] == "B"   # pulled D → B (joins its dependent)
    assert by_id["s3"]["group_resolved"] == "B"
    assert by_id["s1"]["group_resolved"] == "A"
    assert notes and "s2" in notes[0] and "s3" in notes[0]
    # effective_group honors the pin — even though the Lead push said D.
    assert g.effective_group(by_id["s2"], _ARCHS["fact-checker"]) == 2


def test_repair_cascades_down_a_dependency_chain():
    subtasks = [
        _st("s1", "editor-writer", group="D"),              # pushed D
        _st("s2", "editor-writer", deps=["s1"], group="C"),  # pushed C, needs s1
        _st("s3", "data-analyst", deps=["s2"]),             # B, needs s2
    ]
    g.resolve_groups(subtasks, _ARCHS)
    got = {s["id"]: s["group_resolved"] for s in subtasks}
    assert got == {"s1": "B", "s2": "B", "s3": "B"}


def test_repair_is_a_noop_when_order_is_already_right():
    subtasks = [
        _st("s1", "deep-researcher"),
        _st("s2", "data-analyst", deps=["s1"]),
    ]
    assert g.resolve_groups(subtasks, _ARCHS) == []
    assert [s["group_resolved"] for s in subtasks] == ["A", "B"]


def test_repair_is_idempotent_and_cycle_safe():
    subtasks = [
        _st("s1", "data-analyst", deps=["s2"]),
        _st("s2", "editor-writer", deps=["s1"], group="D"),
    ]
    g.resolve_groups(subtasks, _ARCHS)
    first = [s["group_resolved"] for s in subtasks]
    g.resolve_groups(subtasks, _ARCHS)   # re-run on already-pinned plan
    assert [s["group_resolved"] for s in subtasks] == first == ["B", "B"]


# ── schedule awareness: worker brief block + wait-query matching ──────

_SCHED = {
    "current": 2,
    "done": {"s1"},
    "owners": [
        {"subtask": "s1", "arch": "deep-researcher", "role": "Deep Researcher",
         "title": "Form Map", "group": 1},
        {"subtask": "s3", "arch": "data-analyst", "role": "Data Analyst",
         "title": "Quantitative Data", "group": 2},
        {"subtask": "s2", "arch": "fact-checker", "role": "Fact Checker",
         "title": "Verify One-Pager", "group": 4},
    ],
}


def test_schedule_block_tells_a_worker_the_three_buckets():
    block = g.schedule_block("s3", _SCHED)
    assert "Already FINISHED" in block and "Deep Researcher" in block
    assert "Runs AFTER you" in block and "Fact Checker" in block
    assert "NEVER wait for it" in block
    assert "Data Analyst" not in block  # never lists the worker itself


def test_schedule_block_empty_for_a_solo_roster():
    solo = {"current": 1, "done": set(),
            "owners": [{"subtask": "s3", "arch": "a", "role": "", "title": "", "group": 1}]}
    assert g.schedule_block("s3", solo) == ""


def test_match_later_owner_catches_the_observed_wait_query():
    # The exact query from the run log that burned 90s.
    hit = g.match_later_owner("fact-checker verified company data SignificoAI", _SCHED)
    assert hit is not None and hit["subtask"] == "s2"
    # Role phrasing (spaces instead of the hyphenated id) also matches.
    assert g.match_later_owner("waiting for the Fact Checker output", _SCHED)["subtask"] == "s2"


def test_match_later_owner_is_conservative():
    # Finished or concurrent owners never match — a live wait on them is fine.
    assert g.match_later_owner("deep-researcher form map", _SCHED) is None
    assert g.match_later_owner("data-analyst numbers", _SCHED) is None
    # Unattributable queries pass through to the normal wait.
    assert g.match_later_owner("application form structure extracted", _SCHED) is None
    assert g.match_later_owner("", _SCHED) is None
    assert g.match_later_owner("fact-checker data", {}) is None
