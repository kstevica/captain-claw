"""Group 0 (Long Horizon Planner) coordination-plan helpers.

The planner's reply is parsed into a structured per-agent plan that is injected into
every worker's prompt. These tests pin the pure transforms: parse (with fences / prose
/ garbage), the pass-through fallback, sanitisation of user-edited plans, and the
per-owner slice block. A dead or malformed planner must never block a run — every
failure path yields a total plan (one entry per subtask)."""

from __future__ import annotations

from captain_claw.flight_deck import vatra_routes as v


_SUBTASKS = [
    {"id": "s0", "owner_archetype_id": "deep-researcher", "title": "Research the org",
     "brief": "Find facts about the org", "depends_on": [], "group_resolved": "A"},
    {"id": "s1", "owner_archetype_id": "fact-checker", "title": "Verify claims",
     "brief": "Check the research", "depends_on": ["s0"], "group_resolved": "B"},
]
_ARCH = {
    "deep-researcher": {"role": "Deep Researcher", "description": "Digs sources"},
    "fact-checker": {"role": "Fact Checker", "description": "Verifies"},
}


def test_parse_good_json_with_fence_and_prose():
    txt = (
        "Here is the plan:\n```json\n"
        '{"overview":"Research then verify.","agents":['
        '{"subtask_id":"s0","mandate":"Gather facts","produces":"facts.md","consumes_from":[],"hand_off_notes":"cite"},'
        '{"subtask_id":"s1","mandate":"Verify","produces":"verified.md","consumes_from":["s0"],"hand_off_notes":"flag"}'
        "]}\n```\ndone"
    )
    p = v._parse_group0_plan(txt, _SUBTASKS, _ARCH)
    assert p["overview"].startswith("Research then verify")
    assert [a["subtask_id"] for a in p["agents"]] == ["s0", "s1"]
    assert p["agents"][1]["consumes_from"] == ["s0"]
    # agent_id + group are backfilled from the subtask, not the LLM.
    assert p["agents"][0]["agent_id"] == "deep-researcher"
    assert p["agents"][0]["group"] == "A"


def test_parse_garbage_falls_back_to_passthrough():
    p = v._parse_group0_plan("not json at all", _SUBTASKS, _ARCH)
    assert [a["subtask_id"] for a in p["agents"]] == ["s0", "s1"]
    assert p["agents"][0]["mandate"] == "Find facts about the org"  # = brief
    assert p["agents"][0]["produces"] == "Research the org"          # = title


def test_parse_partial_plan_backfills_missing_subtask():
    txt = '{"overview":"x","agents":[{"subtask_id":"s0","mandate":"m0","produces":"p0"}]}'
    p = v._parse_group0_plan(txt, _SUBTASKS, _ARCH)
    ids = [a["subtask_id"] for a in p["agents"]]
    assert ids == ["s0", "s1"]  # s1 backfilled
    assert p["agents"][1]["mandate"] == "Check the research"


def test_parse_drops_unknown_and_duplicate_ids_and_bad_deps():
    txt = ('{"overview":"y","agents":['
           '{"subtask_id":"s0","consumes_from":["s99","s0"]},'   # unknown + self dep
           '{"subtask_id":"nope","mandate":"z"},'                # unknown id
           '{"subtask_id":"s0","mandate":"dup"}]}')              # duplicate
    p = v._parse_group0_plan(txt, _SUBTASKS, _ARCH)
    assert [a["subtask_id"] for a in p["agents"]] == ["s0", "s1"]
    # s99 (unknown) and s0 (self) removed → empty → falls back to depends_on ([]).
    assert p["agents"][0]["consumes_from"] == []


def test_sanitize_user_edited_plan():
    edited = {"overview": "edited", "agents": [
        {"subtask_id": "s1", "mandate": "EDITED", "produces": "", "consumes_from": ["s0"],
         "hand_off_notes": "note"}]}
    p = v._sanitize_group0_plan(edited, _SUBTASKS)
    by = {a["subtask_id"]: a for a in p["agents"]}
    assert by["s1"]["mandate"] == "EDITED"
    assert by["s1"]["produces"] == "Verify claims"        # blank → backfilled title
    assert by["s0"]["mandate"] == "Find facts about the org"  # missing entry backfilled


def test_slice_block_renders_and_resolves_consumers():
    plan = v._parse_group0_plan(
        '{"overview":"o","agents":['
        '{"subtask_id":"s0","mandate":"Gather","produces":"facts.md","consumes_from":[]},'
        '{"subtask_id":"s1","mandate":"Verify each fact","produces":"verified.md","consumes_from":["s0"]}]}',
        _SUBTASKS, _ARCH)
    g0 = {a["subtask_id"]: a for a in plan["agents"]}
    blk = v._plan_slice_block(_SUBTASKS[1], g0, _ARCH)
    assert "Your coordination plan (Group 0)" in blk
    assert "Verify each fact" in blk
    # consumes-from resolves to the teammate's role + produced artifact.
    assert "Deep Researcher" in blk and "facts.md" in blk


def test_slice_block_empty_without_plan():
    # No plan / no entry → '' so resume + legacy runs emit byte-identical prompts.
    assert v._plan_slice_block(_SUBTASKS[0], {}, _ARCH) == ""


def test_build_prompt_lists_ids_roles_and_json_schema():
    pr = v._build_group0_prompt("Examine the org", "Use UK English", ["file.pdf"], _SUBTASKS, _ARCH)
    assert "id=s0" in pr and "id=s1" in pr
    assert "Deep Researcher" in pr and "Fact Checker" in pr
    assert '"subtask_id"' in pr and "file.pdf" in pr and "Use UK English" in pr
