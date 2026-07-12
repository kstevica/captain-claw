"""cost_ledger: persisted run costs (Iskra Phase 0) round-trip."""

from __future__ import annotations

from captain_claw.flight_deck.db import FlightDeckDB


async def test_log_and_list_run_costs(tmp_path):
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    try:
        cost = {
            "tokens": {"prompt_tokens": 1000, "completion_tokens": 200},
            "usd": 0.1234,
            "elapsed_seconds": 12.5,
        }
        await db.log_run_cost("u1", "basna", "sess-1", cost)
        await db.log_run_cost("u1", "vatra", "sess-2", {"tokens": {}, "usd": None})
        await db.log_run_cost(
            "u1", "being_tick", "tick-1", cost,
            owner_type="being", owner_ref="iskra-prva-abcd")
        await db.log_run_cost("u2", "code", "p/s", cost)

        mine = await db.list_run_costs("u1")
        assert len(mine) == 3
        assert mine[0]["run_kind"] in ("basna", "vatra", "being_tick")
        basna_only = await db.list_run_costs("u1", run_kind="basna")
        assert len(basna_only) == 1
        assert basna_only[0]["usd"] == 0.1234
        assert basna_only[0]["owner_type"] == "user"
        being_rows = await db.list_run_costs("u1", run_kind="being_tick")
        assert being_rows[0]["owner_ref"] == "iskra-prva-abcd"
        assert await db.list_run_costs("u3") == []
    finally:
        await db.close()
