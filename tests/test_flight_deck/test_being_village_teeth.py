"""Space plan Phase 3 — teeth: the ground favors drives (×1.5, bonus not
gate), first visits feed explore, reading at a read-place mints richer,
co-presence builds contacts + gossip + earned connection, guestbooks hold
one honest line a day, and the market trades REAL files for coins under a
per-stage quota (both sides of the counter count)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_constitution as constitution
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck import being_society as society
from captain_claw.flight_deck import being_world as world
from captain_claw.flight_deck.beings import BeingError, BeingsStore

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)      # a Thursday
SATURDAY = datetime(2026, 7, 18, 10, 0, tzinfo=timezone.utc)
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


def _garden_file(being, rel="garden/poem.md", text="# a poem\nsmall.\n"):
    p = life._home_path(being, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return rel


async def _settle_at(store, slug, place, depart_at):
    """Walk a being somewhere and settle the arrival (hours later)."""
    store.depart(OWNER, slug, place, now=depart_at)
    store.settle_location(store.get(OWNER, slug),
                          now=depart_at + timedelta(hours=6))
    return store.get(OWNER, slug)


# ═══ The ground favors its drive (strong bonus, never a gate) ═════════════

async def test_the_ground_boosts_the_matching_drive(store):
    db = FakeDB()
    born = NOW - timedelta(days=1)
    a = await _being(store, name="Ana", now=born)
    b = await _being(store, name="Bura", now=born)
    world.ensure_village(store, OWNER, now=born)
    await _settle_at(store, a["slug"], "library", NOW - timedelta(hours=8))

    async def send(being, prompt):
        return _reply(served_drive="grow")

    await life.tick(db, store, store.get(OWNER, a["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    sat_a = store.get(OWNER, a["slug"])["drives"]["grow"]["satisfaction"]
    sat_b = store.get(OWNER, b["slug"])["drives"]["grow"]["satisfaction"]
    assert sat_a > sat_b                     # the library favors grow ×1.5


async def test_a_first_visit_feeds_explore_once_per_place(store):
    db = FakeDB()
    b = await _being(store, now=NOW - timedelta(days=1))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    store.depart(OWNER, b["slug"], "meadow", now=NOW - timedelta(hours=8))

    async def send(being, prompt):
        return _reply(served_drive="")

    await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    bb = store.get(OWNER, b["slug"])
    assert bb["drives"]["explore"]["last_served"]      # arrival = discovery
    firsts = [e for e in store.events(OWNER, b["slug"])
              if e["kind"] == "milestone"
              and e["data"].get("name") == "first_visit_meadow"]
    assert len(firsts) == 1
    # away and back — the meadow is no longer a discovery
    store.depart(OWNER, b["slug"], "home", now=NOW + timedelta(minutes=5))
    store.settle_location(store.get(OWNER, b["slug"]),
                          now=NOW + timedelta(hours=6))
    store.depart(OWNER, b["slug"], "meadow", now=NOW + timedelta(hours=7))
    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(hours=14), send_fn=send,
                    usage_fn=_usage)
    firsts = [e for e in store.events(OWNER, b["slug"])
              if e["kind"] == "milestone"
              and e["data"].get("name") == "first_visit_meadow"]
    assert len(firsts) == 1                            # once per life


async def test_reading_finished_at_the_library_mints_richer(store):
    db = FakeDB()
    born = NOW - timedelta(hours=5)                    # same-day allowance
    a = await _being(store, name="Ucena", now=born)
    b = await _being(store, name="Domača", now=born)
    world.ensure_village(store, OWNER, now=born)
    await _settle_at(store, a["slug"], "library", NOW - timedelta(hours=4))
    for who in (a, b):
        store.add_reading(OWNER, who["slug"], "the red atlas",
                          fee_tokens=100_000, now=NOW - timedelta(hours=2))

    def _send_for(slug):
        item = store.get(OWNER, slug)["reading_list"][0]

        async def send(being, prompt):
            p = life._home_path(being, "garden/reports/atlas.md")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# learned\n", encoding="utf-8")
            return _reply(act_kind="create", served_drive="grow",
                          reading_report={"item_id": item["id"],
                                          "path": "garden/reports/atlas.md"})
        return send

    bal_a0 = store.wallet_view(store.get(OWNER, a["slug"]))["balance_tokens"]
    bal_b0 = store.wallet_view(store.get(OWNER, b["slug"]))["balance_tokens"]
    await life.tick(db, store, store.get(OWNER, a["slug"]), now=NOW,
                    send_fn=_send_for(a["slug"]), usage_fn=_usage)
    await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                    send_fn=_send_for(b["slug"]), usage_fn=_usage)
    gain_a = store.wallet_view(
        store.get(OWNER, a["slug"]))["balance_tokens"] - bal_a0
    gain_b = store.wallet_view(
        store.get(OWNER, b["slug"]))["balance_tokens"] - bal_b0
    assert gain_a - gain_b == 25_000       # 100k ×1.25 at the read-place


# ═══ Co-presence: contacts, gossip, earned connection ═════════════════════

async def test_crossed_paths_builds_a_contact_and_is_heard_by_both(store):
    db = FakeDB()
    born = NOW - timedelta(days=1)
    a = await _being(store, name="Ana", now=born)
    b = await _being(store, name="Bura", now=born)
    world.ensure_village(store, OWNER, now=born)
    await _settle_at(store, a["slug"], "well", NOW - timedelta(hours=9))
    await _settle_at(store, b["slug"], "well", NOW - timedelta(hours=9))
    prompts: list[str] = []

    async def send(being, prompt):
        prompts.append(prompt)
        return _reply(served_drive="")

    await life.tick(db, store, store.get(OWNER, a["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    assert any("CROSSED PATHS: Bura is here at the Well" in p
               for p in prompts)
    for slug in (a["slug"], b["slug"]):
        crossed = [e for e in store.events(OWNER, slug)
                   if e["kind"] == "crossed_paths"]
        assert len(crossed) == 1
    contact = store.contacts_for(OWNER, a["slug"])[0]
    assert contact["with"] == "Bura" and contact["met_count"] == 1
    assert contact["strength"] == pytest.approx(0.2)
    # the encounter FED connection — presence, earned
    aa = store.get(OWNER, a["slug"])
    assert aa["drives"]["connect"]["last_served"]
    # same day, same place: one hello per pair per day
    prompts.clear()
    await life.tick(db, store, store.get(OWNER, a["slug"]),
                    now=NOW + timedelta(hours=2), send_fn=send,
                    usage_fn=_usage)
    assert not any("CROSSED PATHS" in p for p in prompts)
    assert len([e for e in store.events(OWNER, a["slug"])
                if e["kind"] == "crossed_paths"]) == 1
    # the OTHER being hears it on waking, and its connect is fed too
    prompts.clear()
    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(hours=3), send_fn=send,
                    usage_fn=_usage)
    assert any("You crossed paths with Ana" in p for p in prompts)
    assert store.get(OWNER, b["slug"])["drives"]["connect"]["last_served"]
    # tomorrow the well is new again — the contact grows asymptotically
    await life.tick(db, store, store.get(OWNER, a["slug"]),
                    now=NOW + timedelta(days=1), send_fn=send,
                    usage_fn=_usage)
    contact = store.contacts_for(OWNER, a["slug"])[0]
    assert contact["met_count"] == 2
    assert contact["strength"] == pytest.approx(0.36)


async def test_homes_are_private_no_encounters_there(store):
    db = FakeDB()
    born = NOW - timedelta(days=1)
    a = await _being(store, name="Ana", now=born)
    await _being(store, name="Bura", now=born)

    async def send(being, prompt):
        return _reply()

    await life.tick(db, store, store.get(OWNER, a["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    assert [e for e in store.events(OWNER, a["slug"])
            if e["kind"] == "crossed_paths"] == []


# ═══ Guestbooks ═══════════════════════════════════════════════════════════

async def test_guestbook_one_honest_line_a_day(store):
    b = await _being(store, now=NOW - timedelta(days=1))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    with pytest.raises(BeingError, match="no guestbook here"):
        society.guestbook_sign(store, store.get(OWNER, b["slug"]),
                               "hello", now=NOW)
    await _settle_at(store, b["slug"], "library", NOW - timedelta(hours=8))
    society.guestbook_sign(store, store.get(OWNER, b["slug"]),
                           "loved the quiet", now=NOW)
    text = society._commons_path(
        OWNER, "places/library/guestbook.md").read_text()
    assert "Zvjezdana: loved the quiet" in text
    with pytest.raises(BeingError, match="today already"):
        society.guestbook_sign(store, store.get(OWNER, b["slug"]),
                               "again!", now=NOW + timedelta(hours=2))
    # a different place is a different book
    await _settle_at(store, b["slug"], "well", NOW + timedelta(hours=3))
    society.guestbook_sign(store, store.get(OWNER, b["slug"]),
                           "cool water", now=NOW + timedelta(hours=10))
    assert "cool water" in society._commons_path(
        OWNER, "places/well/guestbook.md").read_text()


# ═══ The market: real files for coins ═════════════════════════════════════

async def test_a_stall_needs_a_real_file_and_a_sane_price(store):
    b = await _being(store, now=NOW - timedelta(days=1))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    bb = store.get(OWNER, b["slug"])
    with pytest.raises(Exception, match="no such file"):
        society.market_sell(store, bb, "garden/ghost.md", "Ghost", 3,
                            now=NOW)
    rel = _garden_file(bb)
    with pytest.raises(BeingError, match="positive"):
        society.market_sell(store, bb, rel, "Poem", 0, now=NOW)
    with pytest.raises(BeingError, match="caps prices"):
        society.market_sell(
            store, bb, rel, "Poem",
            constitution.MARKET_MAX_PRICE_COINS + 1, now=NOW)
    li = society.market_sell(store, bb, rel, "A Sea Poem", 3, now=NOW)
    assert li["state"] == "open" and li["price_coins"] == 3
    market = society._commons_path(OWNER, "village/MARKET.md").read_text()
    assert "A Sea Poem" in market and "3 coins" in market
    # an infant does not trade
    baby = await _being(store, name="Beba", stage="infant",
                        now=NOW - timedelta(days=1))
    rel2 = _garden_file(store.get(OWNER, baby["slug"]))
    with pytest.raises(BeingError, match="does not trade yet"):
        society.market_sell(store, store.get(OWNER, baby["slug"]),
                            rel2, "Scribble", 1, now=NOW)


async def test_buying_moves_the_coins_and_the_file(store):
    seller = await _being(store, name="Ana", now=NOW - timedelta(days=1))
    buyer = await _being(store, name="Bura", now=NOW - timedelta(days=1))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    rel = _garden_file(store.get(OWNER, seller["slug"]),
                       text="# the sea\nwaves.\n")
    li = society.market_sell(store, store.get(OWNER, seller["slug"]),
                             rel, "A Sea Poem", 3, now=NOW)
    store.grant_coins(OWNER, buyer["slug"], 5, now=NOW)
    # broke third party is refused loudly, nothing moves
    broke = await _being(store, name="Cvrk", now=NOW - timedelta(days=1))
    with pytest.raises(BeingError, match="not enough coins"):
        society.market_buy(store, store.get(OWNER, broke["slug"]),
                           li["id"], now=NOW)
    # self-dealing refused
    with pytest.raises(BeingError, match="already yours"):
        society.market_buy(store, store.get(OWNER, seller["slug"]),
                           li["id"], now=NOW)
    out = society.market_buy(store, store.get(OWNER, buyer["slug"]),
                             li["id"], now=NOW)
    assert store.coin_balance(store.get(OWNER, buyer["slug"])["id"]) == 2
    assert store.coin_balance(store.get(OWNER, seller["slug"])["id"]) == 3
    bought = life._home_path(store.get(OWNER, buyer["slug"]), out["path"])
    text = bought.read_text()
    assert "bought at the market from Ana" in text and "waves." in text
    kinds_seller = [e["kind"] for e in store.events(OWNER, seller["slug"])]
    kinds_buyer = [e["kind"] for e in store.events(OWNER, buyer["slug"])]
    assert "market_sold" in kinds_seller and "market_bought" in kinds_buyer
    # the stall is empty now
    with pytest.raises(BeingError, match="already sold"):
        society.market_buy(store, store.get(OWNER, broke["slug"]),
                           li["id"], now=NOW)
    # circulation, never minting: the village's coins are exactly the grant
    total = sum(store.coin_balance(store.get(OWNER, s)["id"])
                for s in (seller["slug"], buyer["slug"], broke["slug"]))
    assert total == 5


async def test_trades_quota_counts_both_sides_of_the_counter(store):
    assert world.trades_cap("child", NOW) == 3
    assert world.trades_cap("child", SATURDAY) == 5    # market day +2
    assert world.trades_cap("infant", SATURDAY) == 0   # no bonus below cap
    a = await _being(store, name="Ana", now=NOW - timedelta(days=1))
    b = await _being(store, name="Bura", now=NOW - timedelta(days=1))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    aa = store.get(OWNER, a["slug"])
    for i in range(2):
        society.market_sell(store, aa, _garden_file(aa, f"garden/p{i}.md"),
                            f"Poem {i}", 2, now=NOW)
    bb = store.get(OWNER, b["slug"])
    li = society.market_sell(store, bb, _garden_file(bb, "garden/q.md"),
                             "Q", 1, now=NOW)
    store.grant_coins(OWNER, a["slug"], 5, now=NOW)
    society.market_buy(store, store.get(OWNER, a["slug"]), li["id"],
                       now=NOW)                        # Ana's 3rd trade
    with pytest.raises(BeingError, match="trades are spent"):
        society.market_sell(store, store.get(OWNER, a["slug"]),
                            _garden_file(aa, "garden/p9.md"), "P9", 2,
                            now=NOW)


async def test_market_morning_is_richer_when_you_are_there(store):
    db = FakeDB()
    born = SATURDAY - timedelta(days=2)
    there = await _being(store, name="Ana", now=born)
    afar = await _being(store, name="Bura", now=born)
    seller = await _being(store, name="Cvrk", now=born)
    world.ensure_village(store, OWNER, now=born)
    ss = store.get(OWNER, seller["slug"])
    society.market_sell(store, ss, _garden_file(ss), "A Sea Poem", 3,
                        now=SATURDAY - timedelta(days=1))
    await _settle_at(store, there["slug"], "square",
                     SATURDAY - timedelta(hours=9))
    prompts: list[str] = []

    async def send(being, prompt):
        prompts.append(prompt)
        return _reply(served_drive="")

    await life.tick(db, store, store.get(OWNER, there["slug"]),
                    now=SATURDAY, send_fn=send, usage_fn=_usage)
    assert any("A Sea Poem" in p and "3 coins" in p for p in prompts)
    prompts.clear()
    await life.tick(db, store, store.get(OWNER, afar["slug"]),
                    now=SATURDAY, send_fn=send, usage_fn=_usage)
    assert any("the square hums without you" in p for p in prompts)
    assert not any("A Sea Poem" in p for p in prompts)


async def test_sell_and_buy_ride_the_tick_refusals_echo(store):
    db = FakeDB()
    born = NOW - timedelta(days=1)
    seller = await _being(store, name="Ana", now=born)
    buyer = await _being(store, name="Bura", now=born)
    ss = store.get(OWNER, seller["slug"])
    rel = _garden_file(ss)

    async def send_sell(being, prompt):
        return _reply(sell={"path": rel, "title": "A Sea Poem",
                            "price_coins": 3})

    await life.tick(db, store, ss, now=NOW, send_fn=send_sell,
                    usage_fn=_usage)
    lis = store.market_listings(OWNER)
    assert len(lis) == 1 and lis[0]["title"] == "A Sea Poem"
    # a broke buyer through the tick: refused on the record, echoed next
    async def send_buy(being, prompt):
        return _reply(buy={"listing_id": lis[0]["id"][:8]})

    await life.tick(db, store, store.get(OWNER, buyer["slug"]), now=NOW,
                    send_fn=send_buy, usage_fn=_usage)
    refs = [e["data"] for e in store.events(OWNER, buyer["slug"])
            if e["kind"] == "society_refused"]
    assert any(r["what"] == "buy" and "not enough coins" in r["reason"]
               for r in refs)
    # funded, the same wish is real
    store.grant_coins(OWNER, buyer["slug"], 5, now=NOW)
    await life.tick(db, store, store.get(OWNER, buyer["slug"]),
                    now=NOW + timedelta(hours=1), send_fn=send_buy,
                    usage_fn=_usage)
    assert store.market_listings(OWNER) == []          # sold
    assert store.coin_balance(store.get(OWNER, buyer["slug"])["id"]) == 2


async def test_offers_are_honest_about_quota_and_pocket(store):
    b = await _being(store, now=NOW - timedelta(days=1))
    bb = store.get(OWNER, b["slug"])
    fields = life.society_prompt_fields(bb, [], 3, None, now=NOW,
                                        coins=0, trades_left=0)
    assert not any('"sell"' in f for f in fields)
    fields = life.society_prompt_fields(bb, [], 3, None, now=NOW,
                                        coins=0, trades_left=2)
    assert any('"sell"' in f for f in fields)
    assert not any('"buy"' in f for f in fields)       # no coins, no buying
    fields = life.society_prompt_fields(bb, [], 3, None, now=NOW,
                                        coins=4, trades_left=2)
    assert any('"buy"' in f for f in fields)
