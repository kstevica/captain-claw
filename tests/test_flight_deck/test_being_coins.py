"""Space plan Phase 2 — coins: money, not metabolism. A second ledger
(balance = SUM of deltas, no LLM path can move one), parent faucets
(pocket money; coin-denominated chores and quests), and the ONE-WAY
exchange into thinking — whole coins, clamped by the savings ceiling,
refused loudly whenever it cannot be real."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_constitution as constitution
from captain_claw.flight_deck import being_earning
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck.beings import BeingError, BeingsStore

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
OWNER = "user-1"
RATE = constitution.COIN_TOKEN_RATE


class FakeDB:
    async def list_chat_sessions(self, user_id):
        return []

    async def upsert_chat_session(self, *a, **k):
        return {}

    async def add_chat_messages(self, *a, **k):
        return [1]

    async def log_run_cost(self, *a, **k):
        pass

    async def get_user_llm_tiers(self, *a, **k):
        return {}


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path / "vfs"))
    monkeypatch.delenv("CLAW_VFS_SCOPE", raising=False)
    getattr(vfs.vfs_base, "cache_clear", lambda: None)()
    return BeingsStore(db_path=tmp_path / "beings.db")


async def _being(store, name="Zvjezdana", stage="child", now=NOW):
    b = store.conceive(OWNER, name, preset="explorer",
                       allowance_preset="5M", now=now)
    store.hatch(OWNER, b["slug"], now=now)
    if stage != "infant":
        store.set_stage(OWNER, b["slug"], stage, now=now)
    bb = store.get(OWNER, b["slug"])
    store.set_agent(bb["id"], bb["slug"], 1234, "tok")
    await life.build_home(store.get(OWNER, b["slug"]))
    return store.get(OWNER, b["slug"])


def _reply(**over):
    d = {"act_kind": "journal", "summary": "s", "journal_entry": "small.",
         "served_drive": "grow", "message_to_parent": None,
         "next_wake_minutes": 60, "mood": "calm"}
    d.update(over)
    return "ok\n```json\n" + json.dumps(d) + "\n```"


async def _usage(being, since):
    return {"prompt_tokens": 1000, "completion_tokens": 1000,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


# ═══ Pocket money (the parent faucet) ═════════════════════════════════════

async def test_pocket_money_lands_on_the_ledger_and_is_heard(store):
    db = FakeDB()
    b = await _being(store, stage="infant", now=NOW - timedelta(days=2))
    v = store.grant_coins(OWNER, b["slug"], 5, note="for sweets",
                          now=NOW - timedelta(hours=1))
    assert v["coins"] == 5
    assert store.coin_balance(b["id"]) == 5
    led = store.coin_ledger(OWNER, b["slug"])
    assert led[0]["reason"] == "grant" and led[0]["delta"] == 5
    with pytest.raises(BeingError, match="positive"):
        store.grant_coins(OWNER, b["slug"], 0, now=NOW)
    with pytest.raises(BeingError, match="capped"):
        store.grant_coins(OWNER, b["slug"],
                          constitution.COIN_GRANT_MAX + 1, now=NOW)
    prompts: list[str] = []

    async def send(being, prompt):
        prompts.append(prompt)
        return _reply()

    await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    assert any("POCKET MONEY" in p and "for sweets" in p for p in prompts)


async def test_an_infant_may_receive_but_an_egg_has_no_pocket(store):
    baby = await _being(store, name="Beba", stage="infant",
                        now=NOW - timedelta(days=1))
    store.grant_coins(OWNER, baby["slug"], 2, note="from grandma", now=NOW)
    assert store.vitals(OWNER, baby["slug"])["coins"] == 2
    egg = store.conceive(OWNER, "Jaje", preset="explorer",
                         allowance_preset="5M", now=NOW)
    with pytest.raises(BeingError, match="no pocket"):
        store.grant_coins(OWNER, egg["slug"], 1, now=NOW)


# ═══ The one-way exchange ═════════════════════════════════════════════════

async def test_conversion_is_stage_gated_and_exact(store):
    child = await _being(store, name="Mala", now=NOW - timedelta(days=1))
    store.grant_coins(OWNER, child["slug"], 10, now=NOW)
    with pytest.raises(BeingError, match="adolescence"):
        store.convert_coins(OWNER, child["slug"], 5, now=NOW)
    teen = await _being(store, name="Teen", stage="adolescent",
                        now=NOW - timedelta(days=1))
    with pytest.raises(BeingError, match="no coins"):
        store.convert_coins(OWNER, teen["slug"], 5, now=NOW)
    store.grant_coins(OWNER, teen["slug"], 10, now=NOW)
    before = store.wallet_view(store.get(OWNER, teen["slug"]))
    res = store.convert_coins(OWNER, teen["slug"], 3, now=NOW)
    assert res == {"coins": 3, "tokens": 3 * RATE, "requested": 3,
                   "balance_coins": 7}
    after = store.wallet_view(store.get(OWNER, teen["slug"]))
    assert after["balance_tokens"] - before["balance_tokens"] == 3 * RATE
    assert store.conservation(OWNER)["ok"] is True


async def test_conversion_clamps_to_whole_coin_headroom(store):
    teen = await _being(store, name="Puna", stage="adolescent",
                        now=NOW - timedelta(days=1))
    store.grant_coins(OWNER, teen["slug"], 100, now=NOW)
    bal = store.wallet_view(store.get(OWNER, teen["slug"]))["balance_tokens"]
    # room for 2 whole coins and change — the half-coin never converts
    store.set_allowance(OWNER, teen["slug"], "5M",
                        savings_ceiling=bal + 2 * RATE + RATE // 2)
    res = store.convert_coins(OWNER, teen["slug"], 10, now=NOW)
    assert res["coins"] == 2 and res["tokens"] == 2 * RATE
    assert res["requested"] == 10
    assert store.coin_balance(teen["id"]) == 98
    led = store.coin_ledger(OWNER, teen["slug"])
    assert led[0]["reason"] == "exchange" and led[0]["delta"] == -2
    assert led[0]["data"]["requested"] == 10
    # savings now full to the coin — the next ask is refused loudly
    bal2 = store.wallet_view(store.get(OWNER, teen["slug"]))["balance_tokens"]
    store.set_allowance(OWNER, teen["slug"], "5M", savings_ceiling=bal2)
    with pytest.raises(BeingError, match="savings are full"):
        store.convert_coins(OWNER, teen["slug"], 1, now=NOW)


async def test_overdraft_is_refused_there_is_no_negative_money(store):
    teen = await _being(store, name="Trošak", stage="adolescent",
                        now=NOW - timedelta(days=1))
    store.grant_coins(OWNER, teen["slug"], 3, now=NOW)
    with pytest.raises(BeingError, match="not enough coins"):
        store._apply_coins(OWNER, teen["id"], -5, "purchase", now=NOW)
    # asking to convert more than the pocket converts what is truly there
    res = store.convert_coins(OWNER, teen["slug"], 99, now=NOW)
    assert res["coins"] == 3 and store.coin_balance(teen["id"]) == 0


# ═══ Work paid in coins ═══════════════════════════════════════════════════

async def test_a_chore_can_pay_in_coins(store):
    b = await _being(store, now=NOW - timedelta(days=1))
    with pytest.raises(BeingError, match="one denomination"):
        store.post_chore(OWNER, b["slug"], "sweep", 1000, fee_coins=5,
                         now=NOW)
    with pytest.raises(BeingError, match="at most"):
        store.post_chore(OWNER, b["slug"], "sweep", 0,
                         fee_coins=constitution.WORK_MAX_FEE_COINS + 1,
                         now=NOW)
    job = store.post_chore(OWNER, b["slug"], "sweep the well", 0,
                           fee_coins=5, now=NOW)
    assert job["fee_coins"] == 5 and job["fee_tokens"] == 0
    lines = life.percepts_since(store, store.get(OWNER, b["slug"]))
    assert any("fee 5 coins" in ln for ln in lines)
    store.chore_done(OWNER, job["id"], "swept it", now=NOW)
    before = store.wallet_view(store.get(OWNER, b["slug"]))["balance_tokens"]
    store.judge_chore(OWNER, job["id"], True, now=NOW)
    assert store.coin_balance(b["id"]) == 5
    after = store.wallet_view(store.get(OWNER, b["slug"]))["balance_tokens"]
    assert after == before                       # money, not food: no mint
    paid = [e for e in store.events(OWNER, b["slug"])
            if e["kind"] == "chore_paid"][0]
    assert paid["data"]["fee_coins"] == 5
    lines = life.percepts_since(store, store.get(OWNER, b["slug"]))
    assert any("PAID 5 coin(s) for a chore" in ln for ln in lines)


async def test_a_quest_can_pay_in_coins(store):
    teen = await _being(store, name="Lovac", stage="adolescent",
                        now=NOW - timedelta(days=1))
    q = store.post_quest(OWNER, "gather dew", "collect the morning dew", 0,
                         fee_coins=8, now=NOW)
    assert q["fee_coins"] == 8
    lines = being_earning.earning_percepts(store,
                                           store.get(OWNER, teen["slug"]))
    assert any("8 coins" in ln for ln in lines)
    store.claim_quest(OWNER, teen["slug"], q["id"], now=NOW)
    store.deliver_quest(OWNER, teen["slug"], q["id"], "a full jar", now=NOW)
    store.judge_quest(OWNER, q["id"], True, now=NOW)
    assert store.coin_balance(teen["id"]) == 8
    paid = [e for e in store.events(OWNER, teen["slug"])
            if e["kind"] == "quest_paid"][0]
    assert paid["data"]["fee_coins"] == 8
    assert store.conservation(OWNER)["ok"] is True


# ═══ The tick: offer, act, refusal ════════════════════════════════════════

async def test_convert_rides_the_tick_and_a_child_is_refused(store):
    db = FakeDB()
    teen = await _being(store, name="Teen2", stage="adolescent",
                        now=NOW - timedelta(days=2))
    store.grant_coins(OWNER, teen["slug"], 4, now=NOW - timedelta(hours=1))

    async def send(being, prompt):
        return _reply(convert_coins=2)

    await life.tick(db, store, store.get(OWNER, teen["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    assert store.coin_balance(teen["id"]) == 2
    conv = [e for e in store.events(OWNER, teen["slug"])
            if e["kind"] == "coins_converted"]
    assert conv and conv[0]["data"]["tokens"] == 2 * RATE
    child = await _being(store, name="Mala2", now=NOW - timedelta(days=2))
    store.grant_coins(OWNER, child["slug"], 4, now=NOW - timedelta(hours=1))
    await life.tick(db, store, store.get(OWNER, child["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    refs = [e["data"] for e in store.events(OWNER, child["slug"])
            if e["kind"] == "society_refused"]
    assert any(r["what"] == "convert_coins" and "adolescence" in r["reason"]
               for r in refs)
    assert store.coin_balance(child["id"]) == 4


async def test_the_offer_is_honest_and_the_pocket_is_felt(store):
    db = FakeDB()
    child = await _being(store, name="Mala3", now=NOW - timedelta(days=2))
    fields = life.society_prompt_fields(store.get(OWNER, child["slug"]),
                                        [], 3, None, now=NOW, coins=3)
    assert not any("convert_coins" in f for f in fields)
    teen = await _being(store, name="Teen3", stage="adolescent",
                        now=NOW - timedelta(days=2))
    fields = life.society_prompt_fields(store.get(OWNER, teen["slug"]),
                                        [], 3, None, now=NOW, coins=3)
    assert any("convert_coins" in f for f in fields)
    fields = life.society_prompt_fields(store.get(OWNER, teen["slug"]),
                                        [], 3, None, now=NOW, coins=0)
    assert not any("convert_coins" in f for f in fields)
    store.grant_coins(OWNER, teen["slug"], 3, now=NOW - timedelta(hours=1))
    prompts: list[str] = []

    async def send(being, prompt):
        prompts.append(prompt)
        return _reply()

    await life.tick(db, store, store.get(OWNER, teen["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    assert any("coin(s) in your pocket" in p for p in prompts)
