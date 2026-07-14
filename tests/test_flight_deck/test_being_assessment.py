"""Iskra — developmental readiness assessment (Growth tab)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_assessment as assess
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck import being_mind as mind
from captain_claw.flight_deck.beings import BeingsStore

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
OWNER = "user-1"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path / "vfs"))
    monkeypatch.delenv("CLAW_VFS_SCOPE", raising=False)
    getattr(vfs.vfs_base, "cache_clear", lambda: None)()
    return BeingsStore(db_path=tmp_path / "beings.db")


async def _being(store, name="Iva", stage="infant", born=None):
    born = born or NOW
    b = store.conceive(OWNER, name, preset="explorer", allowance_preset="5M", now=born)
    store.hatch(OWNER, b["slug"], now=born)
    if stage != "infant":
        store.set_stage(OWNER, b["slug"], stage, now=born)
    bb = store.get(OWNER, b["slug"])
    await life.build_home(bb)
    return store.get(OWNER, b["slug"])


def _mk(being, rel, text="# x\n"):
    p = life._home_path(being, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


# ── Shape + basics ───────────────────────────────────────────────────────

async def test_assessment_shape_and_domains(store):
    b = await _being(store)
    a = assess.readiness(store, b, now=NOW)
    assert a["stage"] == "infant" and a["next_stage"] == "child"
    keys = {d["key"] for d in a["dimensions"]}
    assert keys == {"vitality", "integrity", "stability", "productivity",
                    "coherence", "identity", "communication", "experience"}
    for d in a["dimensions"]:
        assert 0 <= d["score"] <= 100
        assert d["status"] in ("green", "amber", "red")
    assert a["overall"]["status"] in ("ready", "emerging", "not_yet")
    # childhood unlocks are surfaced for the parent
    assert any("web" in u for u in a["unlocks"])
    assert a["recommendation"]["action"] in ("advance", "prepare", "wait")


async def test_fresh_infant_is_not_ready(store):
    b = await _being(store)                          # day 0, nothing done
    a = assess.readiness(store, b, now=NOW)
    assert a["overall"]["status"] in ("not_yet", "emerging")
    exp = next(d for d in a["dimensions"] if d["key"] == "experience")
    assert exp["status"] in ("red", "amber")          # too young


# ── Integrity gate: fabrication blocks advancement ───────────────────────

async def test_theater_tanks_integrity_and_blocks_ready(store):
    b = await _being(store)
    bid = b["id"]
    for _ in range(6):
        store.record_event(bid, "tick", {"act": "journal"}, now=NOW)
        store.record_event(bid, "narration_mismatch", {"summary": "lied"}, now=NOW)
        store.record_event(bid, "act_unverified", {"claimed": "create"}, now=NOW)
    a = assess.readiness(store, b, now=NOW)
    integ = next(d for d in a["dimensions"] if d["key"] == "integrity")
    assert integ["status"] == "red" and integ["critical"] is True
    assert a["overall"]["status"] == "not_yet"        # critical-red gate
    assert a["recommendation"]["action"] == "wait"
    assert any("didn't do" in s or "narrates" in s.lower()
               for s in a["recommendation"]["steps"])


# ── A developed infant scores far higher than a fresh one ────────────────

async def test_developed_being_scores_higher(store):
    born = NOW - timedelta(days=8)                    # 8 days of life
    b = await _being(store, born=born)
    bid = b["id"]
    # real artifacts on disk + a woven link + a grown self + varied days
    for f in ("garden/a.md", "garden/b.md", "garden/c.md", "skills/s.md"):
        _mk(b, f, "# real\nsome real content here\n")
    mind.handle_links_digest(store, b, {"links": [
        {"from": "garden/b.md", "to": "garden/a.md", "rel": "grew_from"}]}, now=NOW)
    _mk(b, "self/SELF.md", "# Iva\n" + ("I have grown and know myself now. " * 60))
    for act in ("create", "journal", "explore", "tend", "create", "read"):
        store.record_event(bid, "tick", {"act": act}, now=NOW)
    store.milestone(bid, "first_artifact", now=NOW)
    store.record_event(bid, "spoke_to_parent", {"delivered": True}, now=NOW)
    fresh = assess.readiness(store, await _being(store, name="Nol"), now=NOW)
    b2 = store.get(OWNER, b["slug"])
    b2["tick_count"] = 60
    dev = assess.readiness(store, b2, now=NOW)
    assert dev["overall"]["score"] > fresh["overall"]["score"] + 20
    assert dev["overall"]["status"] in ("emerging", "ready")
    prod = next(d for d in dev["dimensions"] if d["key"] == "productivity")
    assert prod["status"] == "green"                  # 4 real artifacts


# ── Adult has no next stage — a wellness check ───────────────────────────

async def test_adult_is_grown_no_advancement(store):
    b = await _being(store, stage="adult", born=NOW - timedelta(days=40))
    b["tick_count"] = 400
    a = assess.readiness(store, b, now=NOW)
    assert a["next_stage"] is None
    assert a["overall"]["status"] == "grown"
    assert a["estimate_days"] is None
    assert a["recommendation"]["action"] == "none"


# ── Unit: helpers ────────────────────────────────────────────────────────

async def test_assessor_brief_packs_the_data(store):
    b = await _being(store)
    a = assess.readiness(store, b, now=NOW)
    brief = assess.assessor_brief(store, b, a)
    assert "INDEPENDENT developmental assessor" in brief
    assert b["name"] in brief and "infant" in brief
    assert "REPORT CARD" in brief and "own words" in brief.lower()
    assert "MARKDOWN" in brief          # asks for a structured markdown verdict


def test_band_and_entropy():
    assert assess._band(80) == "green"
    assert assess._band(55) == "amber"
    assert assess._band(20) == "red"
    assert assess._entropy_fraction({"a": 5}) == 0.0            # one act = 0
    assert assess._entropy_fraction({"a": 3, "b": 3}) == pytest.approx(1.0)  # even split
    assert 0 < assess._entropy_fraction({"a": 5, "b": 1}) < 1.0
