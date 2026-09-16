"""Increment 6: gap-closing grant, parse_failed, strict-deps sequencing."""

from captain_claw.flight_deck import deliverable_manifest as dm
from captain_claw.flight_deck import vatra_groups as vg


_SUBTASKS = [
    {"id": "s4", "title": "Part One", "owner_archetype_id": "editor-writer", "depends_on": []},
    {"id": "s5", "title": "Part Two", "owner_archetype_id": "editor-writer", "depends_on": ["s4"]},
]


def _manifest():
    return dm.parse({"path": "fair-measure.md", "kind": "fiction", "parts": [
        {"path": "part-one.md", "owner": "s4", "range": "ch1-5"},
        {"path": "part-two.md", "owner": "s5", "range": "ch6-10"},
    ]}, _SUBTASKS, "proj")


def test_parse_clarify_flags_parse_failure():
    # prose reply → deny + parse_failed
    d = vg.parse_clarify("Sure, I think that teammate can help.")
    assert d["approve"] is False and d["parse_failed"] is True
    # valid JSON deny → parse_failed False
    d2 = vg.parse_clarify('{"approve": false, "provider": "", "instruction": ""}')
    assert d2["approve"] is False and d2["parse_failed"] is False
    # valid JSON approve
    d3 = vg.parse_clarify('{"approve": true, "provider": "s4", "instruction": "do X"}')
    assert d3["approve"] is True and d3["parse_failed"] is False


def test_gap_request_grants_declared_range():
    m = _manifest()
    # s5 (depends on s4) asks for chapters six and seven — within s4's ch1-5? no.
    # Ask for a range inside its OWN dependency provider s4 (ch1-5): "chapters 2-3"
    g = dm.gap_request("please supply Chapters Two and Three", m, "s5", _SUBTASKS)
    assert g is not None and g["provider"] == "s4"
    assert "append=true" in g["instruction"]
    # a vague ask → None
    assert dm.gap_request("can you help me with stuff", m, "s5", _SUBTASKS) is None


def test_gap_request_by_part_filename():
    m = _manifest()
    g = dm.gap_request("I need the content of part-one.md to bridge", m, "s5", _SUBTASKS)
    assert g is not None and g["provider"] == "s4"


def test_resolve_groups_strict_deps_sequences_dependent_later():
    subs = [dict(s, depends_on=list(s["depends_on"])) for s in _SUBTASKS]
    arch = {"editor-writer": {"id": "editor-writer", "role": "Editor"}}
    # default: same group (pulled together)
    vg.resolve_groups(subs, arch, strict_deps=False)
    g_default = {s["id"]: s["group_resolved"] for s in subs}
    # strict: s5 (depends on s4) pushed to a later phase than s4
    subs2 = [dict(s, depends_on=list(s["depends_on"])) for s in _SUBTASKS]
    vg.resolve_groups(subs2, arch, strict_deps=True)
    g_strict = {s["id"]: s["group_resolved"] for s in subs2}
    assert g_strict["s5"] > g_strict["s4"]  # later phase letter


def test_dep_layers_and_match_owner_phrase():
    layers = vg.dep_layers(_SUBTASKS)
    assert layers[0] == ["s4"] and layers[1] == ["s5"]
    owners = [{"role": "Editor One", "title": "Part One"}, {"role": "Fact Checker", "title": "Audit"}]
    assert vg.match_owner_phrase("please ask the Fact Checker", owners)["role"] == "Fact Checker"
    assert vg.match_owner_phrase("nobody named here", owners) is None
