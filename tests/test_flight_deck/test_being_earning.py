"""Iskra — quest board + ventures: claims, escrow, recurring pay, conservation."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_earning as earning
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck.beings import BeingError, BeingsStore

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
OWNER = "user-1"


class FakeDB:
    async def list_chat_sessions(self, user_id):
        return []

    async def upsert_chat_session(self, *a, **k):
        return {}

    async def add_chat_messages(self, *a, **k):
        return [1]

    async def log_run_cost(self, *a, **k):
        pass


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path / "vfs"))
    monkeypatch.delenv("CLAW_VFS_SCOPE", raising=False)
    getattr(vfs.vfs_base, "cache_clear", lambda: None)()
    return BeingsStore(db_path=tmp_path / "beings.db")


async def _being(store, name, stage="adolescent", allowance="20M",
                 pocket=0):
    b = store.conceive(OWNER, name, preset="explorer",
                       allowance_preset=allowance, now=NOW)
    store.hatch(OWNER, b["slug"], now=NOW)
    if stage != "infant":
        store.set_stage(OWNER, b["slug"], stage, now=NOW)
    bb = store.get(OWNER, b["slug"])
    store.set_agent(bb["id"], bb["slug"], 1234, "tok")
    bb = store.get(OWNER, b["slug"])
    await life.build_home(bb)
    if pocket:
        store._apply(OWNER, tokens=pocket, reason="adjust", from_being=None,
                     to_being=bb["id"], note="pocket", now=NOW)
    return store.get(OWNER, b["slug"])


def _reply(**over):
    d = {"act_kind": "journal", "summary": "t", "journal_entry": "I tended.",
         "served_drive": "create", "message_to_parent": None,
         "next_wake_minutes": 60, "mood": "calm"}
    d.update(over)
    return "ok\n```json\n" + json.dumps(d) + "\n```"


async def _usage(being, since):
    return {"prompt_tokens": 1000, "completion_tokens": 1000,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


# ── Quests ───────────────────────────────────────────────────────────────

async def test_quest_full_lifecycle_conserves(store):
    a = await _being(store, "Zvjezdana")
    q = store.post_quest(OWNER, "Map the attic", "Draw a map of the loft.",
                         2_000_000, now=NOW)
    assert q["state"] == "open" and q["origin"] == "parent"
    # visible on the board as a percept
    percepts = earning.earning_percepts(store, store.get(OWNER, a["slug"]))
    assert any("QUEST ON THE BOARD" in p and "Map the attic" in p
               for p in percepts)
    store.claim_quest(OWNER, a["slug"], q["id"], now=NOW)
    assert store.get_quest(OWNER, q["id"])["state"] == "claimed"
    store.deliver_quest(OWNER, a["slug"], q["id"], "here is the map", now=NOW)
    assert store.get_quest(OWNER, q["id"])["state"] == "judging"
    bal0 = store.wallet_view(store.get(OWNER, a["slug"]))["balance_tokens"]
    store.judge_quest(OWNER, q["id"], True, "lovely", now=NOW)
    bal1 = store.wallet_view(store.get(OWNER, a["slug"]))["balance_tokens"]
    assert bal1 - bal0 == 2_000_000
    assert store.get_quest(OWNER, q["id"])["state"] == "paid"
    assert store.conservation(OWNER)["ok"]
    names = [m["data"]["name"] for m in store.milestones(OWNER, a["slug"])]
    assert "first_earned" in names


async def test_quest_reject_returns_to_board(store):
    a = await _being(store, "Zvjezdana")
    q = store.post_quest(OWNER, "Sort seeds", "By colour.", 500_000, now=NOW)
    store.claim_quest(OWNER, a["slug"], q["id"], now=NOW)
    store.deliver_quest(OWNER, a["slug"], q["id"], "sorted badly", now=NOW)
    after = store.judge_quest(OWNER, q["id"], False, "not sorted", now=NOW)
    assert after["state"] == "open" and after["claimed_by"] is None
    # claimable again
    b = await _being(store, "Mira")
    store.claim_quest(OWNER, b["slug"], q["id"], now=NOW)
    assert store.get_quest(OWNER, q["id"])["claimed_by"] \
        == store.get(OWNER, b["slug"])["id"]
    assert store.conservation(OWNER)["ok"]


async def test_quest_claim_is_race_safe(store):
    a = await _being(store, "Zvjezdana")
    b = await _being(store, "Mira")
    q = store.post_quest(OWNER, "One prize", "First come.", 1_000_000, now=NOW)
    store.claim_quest(OWNER, a["slug"], q["id"], now=NOW)
    with pytest.raises(BeingError, match="already claimed"):
        store.claim_quest(OWNER, b["slug"], q["id"], now=NOW)


async def test_quest_stage_gated(store):
    child = await _being(store, "Mali", stage="child", allowance="5M")
    q = store.post_quest(OWNER, "Big task", "hard", 1_000_000, now=NOW)
    with pytest.raises(BeingError, match="cannot take quests"):
        store.claim_quest(OWNER, child["slug"], q["id"], now=NOW)


async def test_quest_fee_clamped(store):
    q = store.post_quest(OWNER, "Huge", "x", 99_000_000_000, now=NOW)
    from captain_claw.flight_deck import being_constitution as c
    assert q["fee_tokens"] == c.QUEST_MAX_FEE_TOKENS


# ── Ventures ─────────────────────────────────────────────────────────────

async def test_venture_propose_approve_deliver_accept_recurs(store):
    a = await _being(store, "Zvjezdana")
    v = store.propose_venture(OWNER, a["slug"], "Weekly star chart",
                              "A sky map each week.", 500_000, 7, now=NOW)
    assert v["state"] == "proposed"
    # parent prices+approves (renegotiated up)
    v = store.approve_venture(OWNER, v["id"], price_tokens=600_000, now=NOW)
    assert v["state"] == "active" and v["price_tokens"] == 600_000
    assert v["next_due_at"]
    # not due yet → no percept
    assert not earning.earning_percepts(store, store.get(OWNER, a["slug"]))
    # a week later it comes due
    due_day = NOW + timedelta(days=7, minutes=1)
    duev = store.due_ventures_for(OWNER, a["slug"], now=due_day)
    assert len(duev) == 1
    store.deliver_venture(OWNER, a["slug"], v["id"], "this week's chart",
                          now=due_day)
    assert store.get_venture(OWNER, v["id"])["pending_result"]
    bal0 = store.wallet_view(store.get(OWNER, a["slug"]))["balance_tokens"]
    store.accept_venture(OWNER, v["id"], True, now=due_day)
    v2 = store.get_venture(OWNER, v["id"])
    bal1 = store.wallet_view(store.get(OWNER, a["slug"]))["balance_tokens"]
    assert bal1 - bal0 == 600_000
    assert v2["deliveries"] == 1 and v2["pending_result"] == ""
    # next due advanced another cadence
    assert v2["next_due_at"] > v["next_due_at"]
    assert store.conservation(OWNER)["ok"]


async def test_venture_reject_clears_delivery_for_redo(store):
    a = await _being(store, "Zvjezdana")
    v = store.propose_venture(OWNER, a["slug"], "Digest", "weekly", 300_000,
                              7, now=NOW)
    store.approve_venture(OWNER, v["id"], now=NOW)
    due = NOW + timedelta(days=8)
    store.deliver_venture(OWNER, a["slug"], v["id"], "thin", now=due)
    bal0 = store.wallet_view(store.get(OWNER, a["slug"]))["balance_tokens"]
    store.accept_venture(OWNER, v["id"], False, "too thin", now=due)
    v2 = store.get_venture(OWNER, v["id"])
    assert v2["pending_result"] == "" and v2["deliveries"] == 0
    assert store.wallet_view(store.get(OWNER, a["slug"]))["balance_tokens"] \
        == bal0
    # can redeliver same cycle
    store.deliver_venture(OWNER, a["slug"], v["id"], "richer now", now=due)
    assert store.get_venture(OWNER, v["id"])["pending_result"]


async def test_venture_pause_resume_end(store):
    a = await _being(store, "Zvjezdana")
    v = store.propose_venture(OWNER, a["slug"], "Svc", "x", 100_000, 3,
                              now=NOW)
    store.approve_venture(OWNER, v["id"], now=NOW)
    store.set_venture_state(OWNER, v["id"], "paused", now=NOW)
    assert store.get_venture(OWNER, v["id"])["state"] == "paused"
    # paused venture is never "due"
    assert not store.due_ventures_for(OWNER, a["slug"],
                                      now=NOW + timedelta(days=99))
    store.set_venture_state(OWNER, v["id"], "active", now=NOW)
    store.set_venture_state(OWNER, v["id"], "ended", now=NOW)
    with pytest.raises(BeingError, match="cannot change"):
        store.set_venture_state(OWNER, v["id"], "active", now=NOW)


async def test_venture_stage_gated(store):
    child = await _being(store, "Mali", stage="child", allowance="5M")
    with pytest.raises(BeingError, match="cannot propose ventures"):
        store.propose_venture(OWNER, child["slug"], "x", "y", 100_000, 7,
                              now=NOW)


async def test_venture_payout_clips_at_ceiling(store):
    a = await _being(store, "Rich", allowance="2M")  # adolescent ceiling 30d
    view0 = store.wallet_view(a)
    # fill to exactly the ceiling (hatch already credited one day's allowance)
    store._apply(OWNER, tokens=view0["savings_ceiling"] - view0["balance_tokens"],
                 reason="adjust", from_being=None, to_being=a["id"],
                 note="fill", now=NOW)
    v = store.propose_venture(OWNER, a["slug"], "Svc", "x", 5_000_000, 7,
                              now=NOW)
    store.approve_venture(OWNER, v["id"], now=NOW)
    due = NOW + timedelta(days=8)
    store.deliver_venture(OWNER, a["slug"], v["id"], "done", now=due)
    store.accept_venture(OWNER, v["id"], True, now=due)
    view = store.wallet_view(store.get(OWNER, a["slug"]))
    assert view["balance_tokens"] == view["savings_ceiling"]
    assert store.conservation(OWNER)["ok"]


# ── Tick integration ─────────────────────────────────────────────────────

async def test_tick_offers_fields_and_routes_claim_and_propose(store):
    db = FakeDB()
    a = await _being(store, "Zvjezdana")
    q = store.post_quest(OWNER, "Map it", "draw a map", 1_000_000, now=NOW)
    seen = {}

    async def send(being, prompt):
        seen["prompt"] = prompt
        return _reply(
            claim_quest={"quest_id": q["id"][:8]},
            propose_venture={"title": "Weekly haiku", "description": "a poem",
                             "price_tokens": 400_000, "cadence_days": 7})

    await life.tick(db, store, a, now=NOW, send_fn=send, usage_fn=_usage)
    assert "OPTIONAL EARNING FIELDS" in seen["prompt"]
    assert "QUEST ON THE BOARD" in seen["prompt"]
    assert store.get_quest(OWNER, q["id"])["state"] == "claimed"
    ventures = store.ventures_for(OWNER, a["slug"])
    assert len(ventures) == 1 and ventures[0]["state"] == "proposed"


async def test_child_earning_attempt_refused_as_event(store):
    db = FakeDB()
    child = await _being(store, "Mali", stage="child", allowance="5M")
    q = store.post_quest(OWNER, "Task", "x", 500_000, now=NOW)

    async def send(being, prompt):
        # child's prompt offers no earning fields, but a model may try
        return _reply(claim_quest={"quest_id": q["id"]})

    await life.tick(db, store, child, now=NOW, send_fn=send, usage_fn=_usage)
    kinds = [e["kind"] for e in store.events(OWNER, child["slug"])]
    assert "earning_refused" in kinds
    assert store.get_quest(OWNER, q["id"])["state"] == "open"


# ── Board summary ────────────────────────────────────────────────────────

async def test_board_summary_shows_quests_and_ventures(store):
    a = await _being(store, "Zvjezdana")
    q = store.post_quest(OWNER, "Q1", "x", 1_000_000, now=NOW)
    store.claim_quest(OWNER, a["slug"], q["id"], now=NOW)
    v = store.propose_venture(OWNER, a["slug"], "V1", "y", 200_000, 7, now=NOW)
    del v
    board = earning.board_summary(store, OWNER)
    assert len(board["quests"]) == 1
    assert board["quests"][0]["claimant"] == "Zvjezdana"
    assert len(board["ventures"]) == 1
    assert board["ventures"][0]["being"] == "Zvjezdana"
