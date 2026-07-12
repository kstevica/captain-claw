"""Iskra Phase 2 — chores/escrow, house rules, affect, report cards, milestones."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck.beings import BeingError, BeingsStore

NOW = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
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
    getattr(vfs.vfs_base, "cache_clear", lambda: None)()
    return BeingsStore(db_path=tmp_path / "beings.db")


def _child(store, name="Mira", allowance="5M"):
    b = store.conceive(OWNER, name, preset="caretaker",
                       allowance_preset=allowance, now=NOW)
    store.hatch(OWNER, b["slug"], now=NOW)
    store.set_stage(OWNER, b["slug"], "child", now=NOW)
    bb = store.get(OWNER, b["slug"])
    store.set_agent(bb["id"], bb["slug"], 1234, "tok")
    return store.get(OWNER, b["slug"])


def _reply(**over):
    import json
    d = {"act_kind": "tend", "summary": "tended", "journal_entry": "I tended.",
         "served_drive": "create", "message_to_parent": None,
         "next_wake_minutes": 60, "mood": "calm"}
    d.update(over)
    return "ok\n```json\n" + json.dumps(d) + "\n```"


async def _usage(being, since):
    return {"prompt_tokens": 1000, "completion_tokens": 1000,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


# ── Chores: post → percept → done-in-digest → judged → paid ─────────────

def test_chores_are_stage_gated(store):
    b = store.conceive(OWNER, "Beba", preset="scholar", now=NOW)
    store.hatch(OWNER, b["slug"], now=NOW)   # infant
    with pytest.raises(BeingError, match="too young"):
        store.post_chore(OWNER, b["slug"], "tidy garden", 100_000, now=NOW)


async def test_chore_full_lifecycle_conserves(store):
    db = FakeDB()
    b = _child(store)
    job = store.post_chore(OWNER, b["slug"], "write a haiku about maps",
                           500_000, now=NOW)
    assert job["escrow_state"] == "open"

    # The being sees the chore as a percept and claims it in its digest.
    senses = life.percepts_since(store, store.get(OWNER, b["slug"]))
    assert any("CHORE" in s and "haiku" in s for s in senses)

    async def send(being, prompt):
        assert "haiku" in prompt          # chore reached the prompt
        return _reply(chore={"job_id": job["id"][:8], "result": "haiku written"})

    await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    j = store.get_chore(OWNER, job["id"])
    assert j["escrow_state"] == "judging"
    assert j["result_text"] == "haiku written"

    before = store.wallet_view(store.get(OWNER, b["slug"]))["balance_tokens"]
    store.judge_chore(OWNER, job["id"], True, "lovely", now=NOW)
    after = store.wallet_view(store.get(OWNER, b["slug"]))["balance_tokens"]
    assert after - before == 500_000
    assert store.conservation(OWNER)["ok"]
    names = [m["data"]["name"] for m in store.milestones(OWNER, b["slug"])]
    assert "first_earned" in names
    with pytest.raises(BeingError):        # cannot double-pay
        store.judge_chore(OWNER, job["id"], True, now=NOW)


def test_chore_rejection_pays_nothing(store):
    b = _child(store)
    job = store.post_chore(OWNER, b["slug"], "sort the garden", 300_000, now=NOW)
    store.chore_done(OWNER, job["id"], "did it badly", now=NOW)
    before = store.wallet_view(store.get(OWNER, b["slug"]))["balance_tokens"]
    j = store.judge_chore(OWNER, job["id"], False, "not sorted at all", now=NOW)
    assert j["escrow_state"] == "failed"
    assert store.wallet_view(store.get(OWNER, b["slug"]))["balance_tokens"] == before
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "chore_failed" in kinds


def test_chore_payout_clips_at_piggy_bank(store):
    b = _child(store)   # child 5M/day, ceiling 35M; balance 5M (hatch+clamp)
    job = store.post_chore(OWNER, b["slug"], "big job", 90_000_000, now=NOW)
    store.chore_done(OWNER, job["id"], "done", now=NOW)
    store.judge_chore(OWNER, job["id"], True, now=NOW)
    view = store.wallet_view(store.get(OWNER, b["slug"]))
    assert view["balance_tokens"] == view["savings_ceiling"]
    assert store.conservation(OWNER)["ok"]


# ── House rules internalization ──────────────────────────────────────────

async def test_rules_flow_into_prompt_then_clear(store):
    db = FakeDB()
    b = _child(store)
    store.set_house_rules(OWNER, b["slug"], ["No web after dreams",
                                             "Always cite sources"], now=NOW)
    b = store.get(OWNER, b["slug"])
    assert b["rules_pending"] == 1
    seen = {}

    async def send(being, prompt):
        seen["prompt"] = prompt
        return _reply()

    await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=_usage)
    assert "NEW HOUSE RULES" in seen["prompt"]
    assert "Always cite sources" in seen["prompt"]
    fresh = store.get(OWNER, b["slug"])
    assert fresh["rules_pending"] == 0
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "rules_internalized" in kinds

    async def send2(being, prompt):
        seen["prompt2"] = prompt
        return _reply()

    await life.tick(db, store, fresh, now=NOW + timedelta(hours=1),
                    send_fn=send2, usage_fn=_usage)
    assert "NEW HOUSE RULES" not in seen["prompt2"]


# ── Affect ───────────────────────────────────────────────────────────────

def test_affect_derives_from_real_dynamics():
    hungry_wallet = {"enforced": True, "per_day_tokens": 2_000_000,
                     "balance_tokens": 100_000}
    rich_wallet = {"enforced": True, "per_day_tokens": 2_000_000,
                   "balance_tokens": 2_000_000}
    d_hi = {"explore": {"weight": 0.9, "satisfaction": 0.9}}
    d_lo = {"explore": {"weight": 0.9, "satisfaction": 0.5}}
    lonely = {"connect": {"weight": 0.5, "satisfaction": 0.1}}
    assert life.compute_affect(d_hi, d_lo, rich_wallet)["mood"] == "frustrated"
    assert life.compute_affect(d_lo, d_hi, rich_wallet)["mood"] == "bright"
    assert life.compute_affect(d_hi, d_hi, hungry_wallet)["mood"] == "hungry"
    assert life.compute_affect(lonely, lonely, rich_wallet)["mood"] == "lonely"


async def test_affect_lands_in_next_prompt(store):
    db = FakeDB()
    b = _child(store)
    store.set_affect(b["id"], {"mood": "bright", "notes": ["things went well"]})
    b = store.get(OWNER, b["slug"])
    seen = {}

    async def send(being, prompt):
        seen["p"] = prompt
        return _reply()

    await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=_usage)
    assert "You feel bright" in seen["p"]


# ── Report card ──────────────────────────────────────────────────────────

async def test_report_card_counts_and_catches_a_rut(store):
    db = FakeDB()
    b = _child(store)

    async def send(being, prompt):
        return _reply(journal_entry="I watered the same three plants again "
                                    "and wrote the same words again today.")

    for i in range(6):
        await life.tick(db, store, store.get(OWNER, b["slug"]),
                        now=NOW + timedelta(hours=i), send_fn=send,
                        usage_fn=_usage)
    card = life.report_card(store, store.get(OWNER, b["slug"]), days=7,
                            now=NOW + timedelta(hours=6))
    assert card["ticks"] == 6
    assert card["acts"].get("tend") == 6
    assert card["tokens_spent_weighted"] == 6 * 2000
    assert any("monoton" in c or "rut" in c for c in card["concerns"])
    assert card["drives_trail"]                     # trajectory captured
    assert "watered" in card["in_its_own_words"]


def test_rut_score_math():
    same = ["the same words again"] * 4
    fresh = ["maps of tuscany", "a poem about rust", "sorting my seeds"]
    assert life._rut_score(same) == 1.0
    assert life._rut_score(fresh) < 0.2
    assert life._rut_score([]) == 0.0


# ── Milestones ───────────────────────────────────────────────────────────

async def test_milestones_fire_once(store):
    db = FakeDB()
    b = _child(store)

    async def send(being, prompt):
        return _reply(act_kind="create",
                      message_to_parent="I made my first thing!")

    await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=_usage)
    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(hours=1), send_fn=send,
                    usage_fn=_usage)
    names = [m["data"]["name"] for m in store.milestones(OWNER, b["slug"])]
    assert names.count("first_artifact") == 1
    assert names.count("first_word") == 1
