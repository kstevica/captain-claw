"""A factory's goals, each on or off.

Pinned here: goals the owner switches off are kept with the version (``checks.off``)
so they can be switched on again, a goal can't be both on and off, a version with
none off looks as before, and runs only ever see the goals that are on (the brief
and the Bat contract)."""

from __future__ import annotations

import pytest

from captain_claw.flight_deck import first_mate_runs as runs
from captain_claw.flight_deck import first_mate_store as fm


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", s)
    s.set_settings({"enabled": True, "admins_only": False})
    return s


def test_goals_switched_off_are_kept_not_used(store):
    f = store.create("u1", title="Leads", charter="Find 100 small US firms with a reason they fit.",
                     checks={"contract": ["At least 100 rows", "No repeated Email"],
                             "off": ["Exactly 25 per segment", "No repeated Email", ""]})
    v = f["draft"]
    assert v["checks"]["contract"] == ["At least 100 rows", "No repeated Email"]
    assert v["checks"]["off"] == ["Exactly 25 per segment"]  # not both on and off; blanks dropped
    brief = runs.compose_brief(f, v)
    assert "At least 100 rows" in brief and "Exactly 25 per segment" not in brief
    assert [r["text"] for r in runs.bat_contract(v)] == ["At least 100 rows", "No repeated Email"]
    edited = store.edit_draft(f["id"], "u1", checks={"contract": ["Exactly 25 per segment"], "off": ["At least 100 rows"]})
    assert edited["draft"]["checks"]["contract"] == ["Exactly 25 per segment"]
    assert edited["draft"]["checks"]["off"] == ["At least 100 rows"]


def test_no_goals_off_looks_as_before(store):
    assert fm.check_checks({"contract": ["a"]}) == {"contract": ["a"], "satisfaction_target": 0.8}
    with pytest.raises(fm.FirstMateError):
        fm.check_checks({"contract": ["a"], "maybe": ["b"]})
    with pytest.raises(fm.FirstMateError):
        fm.check_checks({"off": "not a list"})
