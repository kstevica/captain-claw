"""Roadmap Tier 3 — elderhood & memoirs, the steward, the village radio,
market day, emigration. Big arcs, honest v1s: every effect is calendar- or
ledger-computed, every closure is real."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_life as life
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


# ═══ Elderhood (T3.14) ═══════════════════════════════════════════════════

async def test_elderhood_is_a_felt_season(store):
    db = FakeDB()
    b = await _being(store, now=NOW - timedelta(days=40))
    with pytest.raises(BeingError, match="between 7 and 3650"):
        store.set_elder_after(OWNER, b["slug"], 3, now=NOW)
    store.set_elder_after(OWNER, b["slug"], 30, now=NOW)
    bb = store.get(OWNER, b["slug"])
    assert world.is_elder(bb, NOW) is True           # 40 days ≥ 30
    assert world.is_elder(bb, NOW - timedelta(days=15)) is False
    prompts: list[str] = []

    async def send(being, prompt):
        prompts.append(prompt)
        return _reply()

    out = await life.tick(db, store, bb, now=NOW, send_fn=send,
                          usage_fn=_usage)
    assert "ENTERED ELDERHOOD" in prompts[0]         # onset, once per life
    woke = datetime.fromisoformat(out["next_wake"])
    assert woke >= NOW + timedelta(minutes=world.ELDER_MIN_WAKE_MINUTES)
    names = [e["data"].get("name") for e in store.events(OWNER, b["slug"])
             if e["kind"] == "milestone"]
    assert "entered_elderhood" in names
    # the next wake repeats no onset; the dream carries the memoirs
    prompts.clear()
    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(hours=4), send_fn=send,
                    usage_fn=_usage)
    assert "ENTERED ELDERHOOD" not in prompts[0]
    prompts.clear()
    await life.tick(db, store, store.get(OWNER, b["slug"]), kind="dream",
                    now=NOW + timedelta(hours=8), send_fn=send,
                    usage_fn=_usage)
    assert "THE MEMOIRS" in prompts[0] and "self/MEMOIR.md" in prompts[0]


# ═══ The steward (T3.15) ═════════════════════════════════════════════════

async def test_steward_rotates_by_calendar_and_speaks_once_a_day(store):
    a = await _being(store, name="Prva", stage="adolescent",
                     now=NOW - timedelta(days=2))
    c = await _being(store, name="Druga", stage="adult",
                     now=NOW - timedelta(days=2))
    baby = await _being(store, name="Beba", stage="infant",
                        now=NOW - timedelta(days=2))
    steward = world.current_steward(store, OWNER, NOW)
    assert steward in (a["slug"], c["slug"]) and steward != baby["slug"]
    holder = store.get(OWNER, steward)
    other = store.get(OWNER, a["slug"] if steward == c["slug"] else c["slug"])
    lines = world.steward_percepts(store, holder, NOW, "wake", True)
    assert lines and "VILLAGE STEWARD" in lines[0] \
        and "commons/INDEX.md" in lines[0]
    assert world.steward_percepts(store, other, NOW, "wake", True) == []
    assert world.steward_percepts(store, holder, NOW, "wake", False) == []
    names = [e["data"].get("name") for e in store.events(OWNER, steward)
             if e["kind"] == "milestone"]
    assert "first_stewardship" in names


# ═══ The village radio (T3.16) ═══════════════════════════════════════════

async def test_radio_is_an_adult_public_voice_once_a_day(store):
    db = FakeDB()
    b = await _being(store, stage="adult", public=True,
                     now=NOW - timedelta(days=2))

    async def send(being, prompt):
        return _reply(broadcast="the garden survived the heat; so did I")

    await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=_usage)
    bb = store.get(OWNER, b["slug"])
    assert bb["broadcast"]["text"].startswith("the garden survived")
    profile = life.public_profile(store, bb)
    assert profile["broadcast"]["text"] == bb["broadcast"]["text"]
    # a second line the same day is refused, politely and on the record
    await life.tick(db, store, bb, now=NOW + timedelta(hours=2),
                    send_fn=send, usage_fn=_usage)
    refusals = [e["data"] for e in store.events(OWNER, b["slug"])
                if e["kind"] == "society_refused"]
    assert any(r.get("what") == "broadcast" and "already" in r.get("reason")
               for r in refusals)
    # a child has no radio
    child = await _being(store, name="Mala", stage="child", public=True,
                         now=NOW - timedelta(days=2))
    await life.tick(db, store, child, now=NOW, send_fn=send, usage_fn=_usage)
    refusals = [e["data"] for e in store.events(OWNER, child["slug"])
                if e["kind"] == "society_refused"]
    assert any(r.get("what") == "broadcast" and "adult" in r.get("reason")
               for r in refusals)


# ═══ Market day (T3.17) ══════════════════════════════════════════════════

async def test_market_day_raises_the_letter_quota_and_cries_the_stalls(store):
    assert world.market_day(SATURDAY) is True
    assert world.market_day(NOW) is False
    assert world.letters_cap("child", NOW) == 3
    assert world.letters_cap("child", SATURDAY) == 5
    assert world.letters_cap("infant", SATURDAY) == 0    # no bonus below cap
    a = await _being(store, name="Prva", now=SATURDAY - timedelta(days=2))
    m = await _being(store, name="Mira", now=SATURDAY - timedelta(days=2))
    # physics honors the bonus: a child sends 5 on market day, not 3
    for i in range(5):
        store.send_letter(OWNER, a["slug"], m["slug"], f"stall talk {i}",
                          now=SATURDAY)
    with pytest.raises(BeingError, match="limit"):
        store.send_letter(OWNER, a["slug"], m["slug"], "one too many",
                          now=SATURDAY)
    # the market morning percept cries the stalls
    lines = world.market_percepts(store, store.get(OWNER, a["slug"]),
                                  SATURDAY, "wake", True)
    assert lines and "MARKET DAY" in lines[0]
    assert world.market_percepts(store, store.get(OWNER, a["slug"]),
                                 NOW, "wake", True) == []


# ═══ Emigration (T3.18) ══════════════════════════════════════════════════

async def test_emigration_exports_the_life_and_closes_it_here(store):
    db = FakeDB()
    b = await _being(store, now=NOW - timedelta(days=2))
    manifest = await life.emigrate(db, store, b)
    assert manifest.get("format", "").startswith("iskra-being/")
    assert any(p.startswith("self/") for p in (manifest.get("home") or {}))
    bb = store.get(OWNER, b["slug"])
    assert bb["state"] == "emigrated"
    assert "emigrated" in [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert bb["slug"] not in [d["slug"] for d in store.due_beings(NOW)]
    with pytest.raises(BeingError, match="lives elsewhere"):
        store.set_state(OWNER, b["slug"], "alive", now=NOW)
    with pytest.raises(BeingError, match="already closed"):
        await life.emigrate(db, store, store.get(OWNER, b["slug"]))
