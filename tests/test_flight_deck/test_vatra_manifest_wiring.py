"""Increment 3 wiring: _vatra_env strict tier, manifest slice block, byte-identical off-path."""

from captain_claw.flight_deck import deliverable_manifest as dm
from captain_claw.flight_deck import vatra_routes as vr


_SUBTASKS = [
    {"id": "s4", "title": "Part One", "owner_archetype_id": "editor-writer", "depends_on": []},
    {"id": "s5", "title": "Part Two", "owner_archetype_id": "editor-writer", "depends_on": ["s4"]},
]


def _manifest():
    return dm.parse({
        "path": "fair-measure.md", "kind": "fiction", "min_bytes": 60000,
        "parts": [
            {"path": "part-one.md", "order": 1, "range": "ch1-5", "owner": "s4", "min_bytes": 20000},
            {"path": "part-two.md", "order": 2, "range": "ch6-10", "owner": "s5"},
        ],
        "seam_owner": 1,
    }, _SUBTASKS, "vatra-abcd1234")


def test_vatra_env_no_strict_without_flag(monkeypatch):
    sid = "sid-noflag"
    monkeypatch.setitem(vr._run_flags, sid, {"write_guard": False})
    monkeypatch.setitem(vr._run_manifest, sid, _manifest())
    env = vr._vatra_env(sid, "s4", "editor-writer", 0)
    keys = {e["key"] for e in env}
    assert "CLAW_WRITE_STRICT" not in keys
    assert "CLAW_DECLARED_FILES" not in keys


def test_vatra_env_strict_with_flag_and_manifest(monkeypatch):
    sid = "sid-strict"
    monkeypatch.setitem(vr._run_flags, sid, {"write_guard": True})
    monkeypatch.setitem(vr._run_manifest, sid, _manifest())
    env = {e["key"]: e["value"] for e in vr._vatra_env(sid, "s4", "editor-writer", 0)}
    assert env["CLAW_WRITE_STRICT"] == "1"
    assert "part-one.md" in env["CLAW_DECLARED_FILES"]
    assert "fair-measure.md" in env["CLAW_DECLARED_FILES"]
    assert env["CLAW_MY_ARTIFACT"] == "part-one.md"
    assert env["CLAW_WRITE_MIN_BYTES"] == "20000"  # part floor wins


def test_vatra_env_floor_falls_back_to_manifest(monkeypatch):
    sid = "sid-floor"
    monkeypatch.setitem(vr._run_flags, sid, {"write_guard": True})
    monkeypatch.setitem(vr._run_manifest, sid, _manifest())
    env = {e["key"]: e["value"] for e in vr._vatra_env(sid, "s5", "editor-writer", 0)}
    # s5's part has no min_bytes → manifest.min_bytes (60000)
    assert env["CLAW_MY_ARTIFACT"] == "part-two.md"
    assert env["CLAW_WRITE_MIN_BYTES"] == "60000"


def test_manifest_slice_block_has_range_and_seam():
    m = _manifest()
    blk = vr._manifest_slice_block({"id": "s5"}, m)
    assert "part-two.md" in blk
    assert "chapters 6–10" in blk
    assert "seam" in blk.lower()  # s5 is the seam owner
    # s4 is not the seam owner
    blk4 = vr._manifest_slice_block({"id": "s4"}, m)
    assert "part-one.md" in blk4
    assert "chapters 1–5" in blk4


def test_manifest_slice_block_empty_without_manifest():
    assert vr._manifest_slice_block({"id": "s4"}, None) == ""


def test_plan_slice_block_byte_identical_without_manifest():
    # With no group0 entry and no manifest → empty string (legacy behaviour)
    assert vr._plan_slice_block({"id": "s4"}, {}, {}, manifest=None) == ""
    # A group0 entry with no manifest is unchanged by the manifest param
    g0 = {"s4": {"subtask_id": "s4", "mandate": "Write part one", "produces": "Part One",
                 "consumes_from": [], "hand_off_notes": ""}}
    with_none = vr._plan_slice_block({"id": "s4"}, g0, {}, manifest=None)
    assert "Your coordination plan" in with_none
    assert "deliverable file" not in with_none.lower()
