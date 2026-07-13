"""Tests for Phase B — decomposition parsing + dependency-DAG layering.

Pure functions: JSON parse, normalization/validation, topological layering (incl.
cycle safety), and prompt/rendering. No model or filesystem needed.
"""

from __future__ import annotations

import json

from captain_claw.flight_deck import code_plan

# A minimal archetype registry for owner validation.
ARCHES = {"code-implementer": {"id": "code-implementer"},
          "debugger": {"id": "debugger"},
          "quick-dirty": {"id": "quick-dirty"}}


def _slices(*items):
    return json.dumps({"slices": list(items)})


# ── parsing / normalization ──────────────────────────────────────────

def test_parse_empty_and_garbage():
    assert code_plan.parse_slices("", ARCHES) == []
    assert code_plan.parse_slices("no json", ARCHES) == []
    assert code_plan.parse_slices('{"slices": []}', ARCHES) == []


def test_parse_basic_slices():
    out = code_plan.parse_slices(_slices(
        {"id": "s1", "title": "schema", "brief": "models", "owner": "code-implementer",
         "files": ["models.py"], "depends_on": []},
        {"id": "s2", "title": "api", "brief": "handlers", "owner": "code-implementer",
         "files": ["api.py"], "depends_on": ["s1"]},
    ), ARCHES)
    assert [s["id"] for s in out] == ["s1", "s2"]
    assert out[0]["layer"] == 1 and out[1]["layer"] == 2


def test_unknown_owner_defaults_to_implementer():
    out = code_plan.parse_slices(_slices(
        {"id": "s1", "title": "x", "brief": "y", "owner": "wizard", "depends_on": []},
        {"id": "s2", "title": "z", "brief": "w", "owner": "debugger", "depends_on": []},
    ), ARCHES)
    assert out[0]["owner_archetype_id"] == "code-implementer"
    assert out[1]["owner_archetype_id"] == "debugger"


def test_dangling_and_self_deps_dropped():
    out = code_plan.parse_slices(_slices(
        {"id": "s1", "title": "a", "brief": "a", "depends_on": ["s1", "ghost"]},
        {"id": "s2", "title": "b", "brief": "b", "depends_on": ["s1", "nope"]},
    ), ARCHES)
    assert out[0]["depends_on"] == []          # self + ghost dropped
    assert out[1]["depends_on"] == ["s1"]       # nope dropped, s1 kept


def test_duplicate_ids_disambiguated():
    out = code_plan.parse_slices(_slices(
        {"id": "s1", "title": "a", "brief": "a", "depends_on": []},
        {"id": "s1", "title": "b", "brief": "b", "depends_on": []},
    ), ARCHES)
    assert len({s["id"] for s in out}) == 2


def test_cap_enforced():
    many = [{"id": f"s{i}", "title": f"t{i}", "brief": "b", "depends_on": []}
            for i in range(20)]
    out = code_plan.parse_slices(json.dumps({"slices": many}), ARCHES, cap=3)
    assert len(out) == 3


def test_unsafe_files_filtered():
    out = code_plan.parse_slices(_slices(
        {"id": "s1", "title": "a", "brief": "a",
         "files": ["ok.py", "/etc/passwd", "../esc.py"], "depends_on": []},
    ), ARCHES)
    assert out[0]["files"] == ["ok.py"]


def test_titleless_and_briefless_dropped():
    out = code_plan.parse_slices(_slices(
        {"id": "s1", "title": "", "brief": "", "depends_on": []},
        {"id": "s2", "title": "ok", "brief": "", "depends_on": []},
    ), ARCHES)
    assert [s["id"] for s in out] == ["s2"]


# ── DAG layering ─────────────────────────────────────────────────────

def test_layers_linear_chain():
    slices = [{"id": "a", "title": "a", "brief": "", "depends_on": []},
              {"id": "b", "title": "b", "brief": "", "depends_on": ["a"]},
              {"id": "c", "title": "c", "brief": "", "depends_on": ["b"]}]
    n = code_plan.assign_layers(slices)
    assert n == 3
    assert {s["id"]: s["layer"] for s in slices} == {"a": 1, "b": 2, "c": 3}


def test_layers_diamond():
    slices = [{"id": "a", "title": "a", "brief": "", "depends_on": []},
              {"id": "b", "title": "b", "brief": "", "depends_on": ["a"]},
              {"id": "c", "title": "c", "brief": "", "depends_on": ["a"]},
              {"id": "d", "title": "d", "brief": "", "depends_on": ["b", "c"]}]
    code_plan.assign_layers(slices)
    got = {s["id"]: s["layer"] for s in slices}
    assert got == {"a": 1, "b": 2, "c": 2, "d": 3}
    # Two slices share layer 2 → one phase with two slices.
    grouped = code_plan.layers(slices)
    assert [lyr for lyr, _ in grouped] == [1, 2, 3]
    assert len(dict(grouped)[2]) == 2


def test_layers_cycle_is_safe():
    # a→b→a: must terminate, everything lands as early as acyclic deps allow.
    slices = [{"id": "a", "title": "a", "brief": "", "depends_on": ["b"]},
              {"id": "b", "title": "b", "brief": "", "depends_on": ["a"]}]
    n = code_plan.assign_layers(slices)   # must not hang / recurse forever
    assert n >= 1 and all(1 <= s["layer"] <= 6 for s in slices)


def test_roots_are_layer_one():
    slices = [{"id": "a", "title": "a", "brief": "", "depends_on": []},
              {"id": "b", "title": "b", "brief": "", "depends_on": []}]
    code_plan.assign_layers(slices)
    assert all(s["layer"] == 1 for s in slices)


# ── prompts / rendering ──────────────────────────────────────────────

def test_slice_prompt_mentions_files_and_deps():
    slices = code_plan.parse_slices(_slices(
        {"id": "s1", "title": "schema", "brief": "models", "files": ["m.py"], "depends_on": []},
        {"id": "s2", "title": "api", "brief": "handlers", "files": ["api.py"], "depends_on": ["s1"]},
    ), ARCHES)
    by_id = {s["id"]: s for s in slices}
    p = code_plan.slice_prompt("build a thing", slices[1], by_id, facts=True)
    assert "api" in p and "api.py" in p
    assert "schema" in p               # names its foundation dep
    assert "INTERFACE LEDGER" in p     # ledger directive when facts=True


def test_slice_prompt_no_ledger_when_off():
    slices = code_plan.parse_slices(_slices(
        {"id": "s1", "title": "x", "brief": "y", "depends_on": []}), ARCHES) \
        or [{"id": "s1", "title": "x", "brief": "y", "files": [], "depends_on": [], "layer": 1,
             "owner_archetype_id": "code-implementer"}]
    p = code_plan.slice_prompt("t", slices[0], {s["id"]: s for s in slices}, facts=False)
    assert "INTERFACE LEDGER" not in p


def test_coordination_markdown_lists_layers():
    slices = code_plan.parse_slices(_slices(
        {"id": "s1", "title": "schema", "brief": "m", "depends_on": []},
        {"id": "s2", "title": "api", "brief": "h", "depends_on": ["s1"]},
    ), ARCHES)
    md = code_plan.coordination_markdown(slices)
    assert "coordination plan" in md.lower()
    assert "schema" in md and "api" in md
    assert "Layer 1" in md and "Layer 2" in md
