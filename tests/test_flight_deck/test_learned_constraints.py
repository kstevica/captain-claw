"""R2 tests — the learned_constraints DB layer (add/get/dedup/cap/isolation)."""

from __future__ import annotations

from captain_claw.flight_deck.db import FlightDeckDB


async def test_add_and_get_roundtrip(tmp_path):
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    try:
        res = await db.add_learned_constraint(
            "u1", "web", "code", "when building forms", "always validate input", "major")
        assert res.get("id") and res.get("deduped") is False
        rows = await db.get_learned_constraints("u1", "web")
        assert len(rows) == 1
        assert rows[0]["constraint_text"] == "always validate input"
        assert rows[0]["trigger_text"] == "when building forms"
        assert rows[0]["engine"] == "code"
    finally:
        await db.close()


async def test_domain_filter_and_global(tmp_path):
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    try:
        await db.add_learned_constraint("u1", "web", "code", "", "web rule", "minor")
        await db.add_learned_constraint("u1", "data", "vatra", "", "data rule", "minor")
        assert len(await db.get_learned_constraints("u1", "web")) == 1
        assert len(await db.get_learned_constraints("u1", "data")) == 1
        assert len(await db.get_learned_constraints("u1", None)) == 2   # all domains
    finally:
        await db.close()


async def test_severity_ordering(tmp_path):
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    try:
        await db.add_learned_constraint("u1", "web", "code", "", "minor one", "minor")
        await db.add_learned_constraint("u1", "web", "code", "", "critical one", "critical")
        await db.add_learned_constraint("u1", "web", "code", "", "major one", "major")
        rows = await db.get_learned_constraints("u1", "web", limit=5)
        assert [r["severity"] for r in rows] == ["critical", "major", "minor"]
    finally:
        await db.close()


async def test_dedup_bumps_hits(tmp_path):
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    try:
        a = await db.add_learned_constraint("u1", "web", "code", "t1", "same rule", "major")
        b = await db.add_learned_constraint("u1", "web", "code", "t2", "SAME RULE", "critical")
        assert a["deduped"] is False and b["deduped"] is True
        rows = await db.get_learned_constraints("u1", "web")
        assert len(rows) == 1
        assert rows[0]["hits"] == 1
        assert rows[0]["severity"] == "critical"   # updated on dedup
    finally:
        await db.close()


async def test_cap_prunes_oldest(tmp_path):
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    try:
        for i in range(5):
            await db.add_learned_constraint(
                "u1", "web", "code", "", f"rule {i}", "major", cap_per_domain=3)
        rows = await db.get_learned_constraints("u1", "web", limit=50)
        assert len(rows) == 3   # capped
    finally:
        await db.close()


async def test_empty_constraint_writes_nothing_and_isolation(tmp_path):
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    try:
        assert await db.add_learned_constraint("u1", "web", "code", "", "   ", "major") == {}
        assert await db.get_learned_constraints("u1", "web") == []
        await db.add_learned_constraint("u1", "web", "code", "", "u1 rule", "major")
        assert await db.get_learned_constraints("u2", "web") == []   # user isolation
    finally:
        await db.close()
