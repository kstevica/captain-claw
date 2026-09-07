"""R4 tests — grouped Vatra edges carry data (push a finished dependency's output).

Pure-function coverage for _dep_output_block plus the push_deps opt-in flag.
"""

from __future__ import annotations

from captain_claw.flight_deck.quality_profile import QualityProfile
from captain_claw.flight_deck.vatra_routes import _DEP_PUSH_CAP, _dep_output_block


def test_pushes_finished_dependency_with_header_and_output():
    st = {"id": "c", "depends_on": ["a"]}
    rbi = {"a": {"output": "AAA facts here", "role": "Researcher", "title": "facts"}}
    block = _dep_output_block(st, rbi)
    assert "### Researcher — facts" in block
    assert "AAA facts here" in block
    assert "already delivered" in block


def test_empty_when_no_dependency_finished():
    assert _dep_output_block({"id": "c", "depends_on": ["a"]}, {}) == ""


def test_empty_when_only_unrelated_result_present():
    # Only depends_on edges push; an unrelated finished result is ignored.
    assert _dep_output_block({"id": "c", "depends_on": ["a"]}, {"z": {"output": "Z"}}) == ""


def test_empty_when_no_deps_declared():
    assert _dep_output_block({"id": "c"}, {"a": {"output": "A"}}) == ""


def test_skips_finished_dep_with_no_text():
    assert _dep_output_block({"id": "c", "depends_on": ["a"]}, {"a": {"output": "   ", "role": "R"}}) == ""


def test_header_fallback_role_then_owner_then_id():
    st = {"id": "c", "depends_on": ["a"]}
    assert "### a" in _dep_output_block(st, {"a": {"output": "X"}})            # id fallback
    assert "### Owner" in _dep_output_block(st, {"a": {"output": "X", "owner": "Owner"}})
    assert "### Role" in _dep_output_block(st, {"a": {"output": "X", "role": "Role", "owner": "Owner"}})


def test_output_truncated_to_cap():
    big = "Y" * (_DEP_PUSH_CAP + 5000)
    block = _dep_output_block({"id": "c", "depends_on": ["a"]}, {"a": {"output": big}})
    assert ("Y" * _DEP_PUSH_CAP) in block
    assert ("Y" * (_DEP_PUSH_CAP + 1)) not in block


def test_multiple_finished_deps_all_push():
    st = {"id": "c", "depends_on": ["a", "b"]}
    rbi = {"a": {"output": "AAA", "role": "RA"}, "b": {"output": "BBB", "role": "RB"}}
    block = _dep_output_block(st, rbi)
    assert "AAA" in block and "BBB" in block
    assert "### RA" in block and "### RB" in block


def test_push_deps_is_off_by_default_and_off_in_every_preset():
    assert QualityProfile.from_dict(None).push_deps is False
    for preset in ("off", "balanced", "thorough"):
        assert QualityProfile.from_dict({"profile": preset}).push_deps is False, preset


def test_push_deps_explicit_opt_in_but_not_in_bool_flags():
    q = QualityProfile.from_dict({"push_deps": True})
    assert q.push_deps is True
    # Vatra-only lever: excluded from _BOOL_FLAGS/any_enabled (read only by Code),
    # exactly like micro_workers.
    assert "push_deps" not in QualityProfile._BOOL_FLAGS
    assert q.any_enabled is False
