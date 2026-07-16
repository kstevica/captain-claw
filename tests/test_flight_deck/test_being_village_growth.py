"""Space plan Phase 5 — growth: introductions (a contact's pen-pal reach,
lent one hop, honestly named), commissioned buildings (coins escrowed into
ONE village fund → parent's word → the architect raises real ground, or
every contributor is refunded to the coin), and the steward's weekly
stipend (a parent knob, ledger-idempotent per ISO week)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_constitution as constitution
from captain_claw.flight_deck import being_federation as federation
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck import being_society as society
from captain_claw.flight_deck import being_world as world
from captain_claw.flight_deck.beings import BeingError, BeingsStore

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
OWNER = "user-1"
TARGET = constitution.COMMISSION_COST_COINS


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


async def _being(store, name="Zvjezdana", stage="child", now=NOW,
                 public=False):
    b = store.conceive(OWNER, name, preset="explorer",
                       allowance_preset="5M", now=now)
    store.hatch(OWNER, b["slug"], now=now)
    if stage != "infant":
        store.set_stage(OWNER, b["slug"], stage, now=now)
    bb = store.get(OWNER, b["slug"])
    store.set_agent(bb["id"], bb["slug"], 1234, "tok")
    if public:
        store.set_public(OWNER, b["slug"], True, now=now)
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


def _make_contacts(store, a, b, when):
    store.touch_contact(OWNER, a["id"], b["id"], now=when)


# ═══ Introductions (a contact's reach, lent one hop) ══════════════════════

async def test_a_contact_with_reach_can_introduce_you(store, monkeypatch):
    born = NOW - timedelta(days=2)
    a = await _being(store, name="Ana", now=born)               # NOT public
    b = await _being(store, name="Bura", now=born, public=True)  # the door
    _make_contacts(store, a, b, NOW - timedelta(days=1))
    store.upsert_visitor(OWNER, "https://far.example", "iskra-mira-1234",
                         name="Mira", profile={}, now=NOW)
    monkeypatch.setattr(federation, "is_linked", lambda vid: True)
    sent: list[dict] = []

    async def fake_send(store_, vid, frm, village, body):
        sent.append({"frm": frm, "body": body})

    monkeypatch.setattr(federation, "send_letter_to_visitor", fake_send)
    reach = life.introduction_reach(store, store.get(OWNER, a["slug"]))
    assert reach and reach[0]["to"] == "Mira" \
        and reach[0]["via"]["name"] == "Bura"
    await life._deliver_introduction(
        store, store.get(OWNER, a["slug"]),
        {"to": "Mira", "body": "hello from a friend of Bura"}, NOW)
    assert sent and sent[0]["frm"] == "Ana"
    assert "Bura, whom we both know" in sent[0]["body"]
    kinds_a = [e["kind"] for e in store.events(OWNER, a["slug"])]
    assert "introduced" in kinds_a and "penpal_sent" in kinds_a
    assert store.penpals_sent_today(a["id"], NOW) == 1     # HER quota spent
    made = [e for e in store.events(OWNER, b["slug"])
            if e["kind"] == "made_introduction"]
    assert made and made[0]["data"]["for"] == "Ana" \
        and made[0]["data"]["to"] == "Mira"
    # the via hears it as a percept on waking
    lines = life.percepts_since(store, store.get(OWNER, b["slug"]))
    assert any("You introduced Ana to Mira" in ln for ln in lines)


async def test_introductions_need_a_true_contact_and_a_live_door(
        store, monkeypatch):
    born = NOW - timedelta(days=2)
    a = await _being(store, name="Ana", now=born)
    b = await _being(store, name="Bura", now=born, public=True)
    store.upsert_visitor(OWNER, "https://far.example", "iskra-mira-1234",
                         name="Mira", profile={}, now=NOW)
    monkeypatch.setattr(federation, "is_linked", lambda vid: True)
    # never met → no reach, refused loudly
    assert life.introduction_reach(store, store.get(OWNER, a["slug"])) == []
    with pytest.raises(BeingError, match="no one you truly know"):
        await life._deliver_introduction(
            store, store.get(OWNER, a["slug"]),
            {"to": "Mira", "body": "hi"}, NOW)
    # met, but the door is dark (link down) → still no reach
    _make_contacts(store, a, b, NOW - timedelta(days=1))
    monkeypatch.setattr(federation, "is_linked", lambda vid: False)
    assert life.introduction_reach(store, store.get(OWNER, a["slug"])) == []


async def test_the_open_door_is_offered_in_the_tick(store, monkeypatch):
    db = FakeDB()
    born = NOW - timedelta(days=2)
    a = await _being(store, name="Ana", now=born)
    b = await _being(store, name="Bura", now=born, public=True)
    _make_contacts(store, a, b, NOW - timedelta(days=1))
    store.upsert_visitor(OWNER, "https://far.example", "iskra-mira-1234",
                         name="Mira", profile={}, now=NOW)
    monkeypatch.setattr(federation, "is_linked", lambda vid: True)
    prompts: list[str] = []

    async def send(being, prompt):
        prompts.append(prompt)
        return _reply()

    await life.tick(db, store, store.get(OWNER, a["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    assert any("AN OPEN DOOR: Bura" in p and "Mira" in p for p in prompts)


# ═══ Commissioned buildings ═══════════════════════════════════════════════

async def test_a_commission_is_skin_first_then_a_pool(store):
    born = NOW - timedelta(days=2)
    teen = await _being(store, name="Teen", stage="adolescent", now=born)
    kid = await _being(store, name="Mala", now=born)
    world.ensure_village(store, OWNER, now=born)
    store.grant_coins(OWNER, teen["slug"], 30, now=NOW)
    store.grant_coins(OWNER, kid["slug"], 40, now=NOW)
    # a child may not PROPOSE…
    with pytest.raises(BeingError, match="adolescence"):
        store.propose_commission(OWNER, kid["slug"], "the Pond", "",
                                 "play", 10, now=NOW)
    c = store.propose_commission(OWNER, teen["slug"], "the Pond",
                                 "somewhere to float things", "play", 30,
                                 now=NOW)
    assert c["state"] == "open" and c["raised_coins"] == 30
    assert store.coin_balance(teen["id"]) == 0             # escrowed out
    # …but any coin-holder may CONTRIBUTE (pooling is the point);
    # the pool clamps to what remains.
    with pytest.raises(BeingError, match="one building at a time"):
        store.propose_commission(OWNER, teen["slug"], "the Tower", "",
                                 "read", 1, now=NOW)
    c = store.contribute_commission(OWNER, kid["slug"], 99, now=NOW)
    assert c["state"] == "funded" and c["raised_coins"] == TARGET
    assert store.coin_balance(kid["id"]) == 40 - (TARGET - 30)
    contributors = store.commission_contributors(OWNER, c["id"])
    assert {x["name"]: x["coins"] for x in contributors} == \
        {"Teen": 30, "Mala": 20}
    with pytest.raises(BeingError, match="fully funded"):
        store.contribute_commission(OWNER, kid["slug"], 1, now=NOW)


async def test_approval_raises_real_ground_and_burns_the_coins(store):
    born = NOW - timedelta(days=2)
    teen = await _being(store, name="Teen", stage="adolescent", now=born)
    world.ensure_village(store, OWNER, now=born)
    store.grant_coins(OWNER, teen["slug"], TARGET, now=NOW)
    store.propose_commission(OWNER, teen["slug"], "the Pond",
                             "somewhere to float things", "play", TARGET,
                             now=NOW)
    n_before = len(store.village_places(OWNER))
    out = store.judge_commission(OWNER, True, "lovely idea", now=NOW)
    place = out["place"]
    assert place["id"] == "the-pond" and place["affordances"] == ["play"]
    assert 40 <= place["x"] <= 960 and 40 <= place["y"] <= 960
    assert len(store.village_places(OWNER)) == n_before + 1
    assert store.coin_balance(teen["id"]) == 0             # burned — the sink
    text = society._commons_path(OWNER, "village/MAP.md").read_text()
    assert "the Pond" in text
    kinds = [e["kind"] for e in store.events(OWNER, teen["slug"])]
    assert "commission_built" in kinds
    assert any(e["data"].get("name") == "first_commission"
               for e in store.events(OWNER, teen["slug"])
               if e["kind"] == "milestone")
    assert store.open_commission(OWNER) is None
    # the new ground is walkable
    bb = store.depart(OWNER, teen["slug"], "the Pond", now=NOW)
    assert bb["location"]["to"] == "the-pond"


async def test_rejection_refunds_every_contributor_to_the_coin(store):
    born = NOW - timedelta(days=2)
    teen = await _being(store, name="Teen", stage="adolescent", now=born)
    kid = await _being(store, name="Mala", now=born)
    world.ensure_village(store, OWNER, now=born)
    store.grant_coins(OWNER, teen["slug"], 12, now=NOW)
    store.grant_coins(OWNER, kid["slug"], 7, now=NOW)
    store.propose_commission(OWNER, teen["slug"], "the Tower", "", "read",
                             12, now=NOW)
    store.contribute_commission(OWNER, kid["slug"], 7, now=NOW)
    # an unfunded fund cannot be approved — but it CAN be declined
    with pytest.raises(BeingError, match="not fully funded"):
        store.judge_commission(OWNER, True, now=NOW)
    store.judge_commission(OWNER, False, "not this season", now=NOW)
    assert store.coin_balance(teen["id"]) == 12
    assert store.coin_balance(kid["id"]) == 7
    assert store.open_commission(OWNER) is None
    lines = life.percepts_since(store, store.get(OWNER, kid["slug"]))
    assert any("came back to your pocket" in ln for ln in lines)


async def test_the_fund_rides_the_tick_and_is_cried_each_morning(store):
    db = FakeDB()
    born = NOW - timedelta(days=2)
    teen = await _being(store, name="Teen", stage="adolescent", now=born)
    kid = await _being(store, name="Mala", now=born)
    world.ensure_village(store, OWNER, now=born)
    store.grant_coins(OWNER, teen["slug"], 25, now=NOW - timedelta(hours=2))
    store.grant_coins(OWNER, kid["slug"], 5, now=NOW - timedelta(hours=2))

    async def send_propose(being, prompt):
        return _reply(commission={"name": "the Pond", "why": "to float",
                                  "affordance": "play", "coins": 25})

    await life.tick(db, store, store.get(OWNER, teen["slug"]), now=NOW,
                    send_fn=send_propose, usage_fn=_usage)
    c = store.open_commission(OWNER)
    assert c and c["raised_coins"] == 25
    # the kid hears the fund next morning and contributes through the tick
    prompts: list[str] = []

    async def send_contribute(being, prompt):
        prompts.append(prompt)
        return _reply(commission={"coins": 5})

    await life.tick(db, store, store.get(OWNER, kid["slug"]),
                    now=NOW + timedelta(hours=1), send_fn=send_contribute,
                    usage_fn=_usage)
    assert any('THE COMMISSION: "the Pond"' in p and "25/50" in p
               for p in prompts)
    assert store.open_commission(OWNER)["raised_coins"] == 30
    # a broke third wheel is refused on the record
    broke = await _being(store, name="Cvrk", now=born)

    async def send_broke(being, prompt):
        return _reply(commission={"coins": 3})

    await life.tick(db, store, store.get(OWNER, broke["slug"]),
                    now=NOW + timedelta(hours=2), send_fn=send_broke,
                    usage_fn=_usage)
    refs = [e["data"] for e in store.events(OWNER, broke["slug"])
            if e["kind"] == "society_refused"]
    assert any(r["what"] == "commission" and "coins first" in r["reason"]
               for r in refs)


def test_commission_spot_is_deterministic_and_off_the_crowd(store):
    world.ensure_village(store, OWNER, now=NOW)
    a = world.commission_spot(store, OWNER, "seed-1")
    assert a == world.commission_spot(store, OWNER, "seed-1")
    assert a != world.commission_spot(store, OWNER, "seed-2")
    assert 80 <= a[0] <= 920 and 80 <= a[1] <= 920
    import math
    nearest = min(math.dist(a, (p["x"], p["y"]))
                  for p in store.village_places(OWNER))
    assert nearest > 60                       # new ground, not a crowd


# ═══ The steward's stipend ════════════════════════════════════════════════

async def test_stipend_pays_once_per_week_and_only_when_set(store):
    born = NOW - timedelta(days=3)
    a = await _being(store, name="Prva", stage="adult", now=born)
    steward_slug = world.current_steward(store, OWNER, NOW)
    assert steward_slug == a["slug"]           # only eligible being
    holder = store.get(OWNER, steward_slug)
    # default off: the note pays nothing
    lines = world.steward_percepts(store, holder, NOW, "wake", True)
    assert lines and "stipend" not in lines[0]
    assert store.coin_balance(holder["id"]) == 0
    # the knob validates and pays once per ISO week
    with pytest.raises(BeingError, match="0–10"):
        store.set_steward_stipend(OWNER, 99)
    store.set_steward_stipend(OWNER, 3)
    lines = world.steward_percepts(store, holder, NOW, "wake", True)
    assert "3 coin(s)" in lines[0]
    assert store.coin_balance(holder["id"]) == 3
    lines = world.steward_percepts(store, holder, NOW + timedelta(days=1),
                                   "wake", True)
    assert lines and "stipend" not in lines[0]         # same week: once
    assert store.coin_balance(holder["id"]) == 3
    lines = world.steward_percepts(store, holder, NOW + timedelta(days=7),
                                   "wake", True)
    assert store.coin_balance(holder["id"]) == 6       # a new week pays
