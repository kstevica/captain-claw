"""Iskra life engine: prompts, digests, drives, the tick, the loop."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck import beings_loop
from captain_claw.flight_deck.beings import BeingError, BeingNotFound, BeingsStore

NOW = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
OWNER = "user-1"


class FakeDB:
    """The few async FlightDeckDB methods the life engine touches."""

    def __init__(self):
        self.chat_sessions: list[dict] = []
        self.chat_messages: list[dict] = []
        self.costs: list[dict] = []

    async def list_chat_sessions(self, user_id):
        return [s for s in self.chat_sessions if s["user_id"] == user_id]

    async def upsert_chat_session(self, session_id, user_id, agent_id="",
                                  agent_name=""):
        s = {"id": session_id, "user_id": user_id, "agent_id": agent_id,
             "agent_name": agent_name}
        self.chat_sessions.append(s)
        return s

    async def add_chat_messages(self, session_id, user_id, messages):
        self.chat_messages += [{"session_id": session_id, **m} for m in messages]
        return list(range(len(messages)))

    async def log_run_cost(self, owner_user_id, run_kind, run_id, cost,
                           owner_type="user", owner_ref=""):
        self.costs.append({"owner": owner_user_id, "kind": run_kind,
                           "ref": owner_ref})


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path / "vfs"))
    getattr(vfs.vfs_base, "cache_clear", lambda: None)()
    return BeingsStore(db_path=tmp_path / "beings.db")


def _born(store, name="Prva", stage=None, allowance="2M", port=1234):
    b = store.conceive(OWNER, name, preset="explorer", allowance_preset=allowance,
                       birth_letter="Grow curious and kind.", now=NOW)
    store.hatch(OWNER, b["slug"], now=NOW)
    if stage:
        store.set_stage(OWNER, b["slug"], stage, now=NOW)
    if port:
        b = store.get(OWNER, b["slug"])
        store.set_agent(b["id"], b["slug"], port, "tok")
    return store.get(OWNER, b["slug"])


def _digest_reply(**over):
    d = {"act_kind": "journal", "summary": "wrote about maps",
         "journal_entry": "Today I thought about old maps and why edges matter.",
         "served_drive": "explore", "message_to_parent": None,
         "next_wake_minutes": 90, "mood": "curious"}
    d.update(over)
    return "I did a small thing.\n```json\n" + json.dumps(d) + "\n```"


def _usage(completion=100_000):
    return {"prompt_tokens": 10_000, "completion_tokens": completion,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


# ── Prompt composition ───────────────────────────────────────────────────

def test_prompt_carries_identity_vitals_and_stage_gates(store):
    b = _born(store)
    p = life.compose_tick_prompt(b, now=NOW, spent_today=5,
                                 wallet=store.wallet_view(b))
    assert "You are Prva" in p and "CUR:9" in p
    assert "attention credits 3" in p
    assert "cannot browse the web yet" in p          # infant gate
    assert "Grow curious and kind." in p             # imprint on tick #1
    child = store.set_stage(OWNER, b["slug"], "child", now=NOW)
    store.set_media_diet(OWNER, b["slug"], {"deny": ["reddit.com"], "allow": []})
    child = store.get(OWNER, b["slug"])
    p2 = life.compose_tick_prompt(child, now=NOW, wallet=store.wallet_view(child))
    assert "MEDIA DIET" in p2 and "reddit.com" in p2


def test_dream_prompt_differs(store):
    b = _born(store)
    p = life.compose_tick_prompt(b, kind="dream", now=NOW,
                                 wallet=store.wallet_view(b))
    assert "This is your DREAM" in p


# ── Digest parsing ───────────────────────────────────────────────────────

def test_parse_digest_takes_last_valid_block_and_clamps():
    text = ("```json\n{\"act_kind\": \"journal\", \"x\": 1}\n```\n mid \n"
            + _digest_reply(act_kind="weird-kind", served_drive="nonsense",
                            next_wake_minutes="soon"))
    d = life.parse_digest(text)
    assert d["act_kind"] == "freeform"       # unknown kind clamped
    assert d["served_drive"] == ""           # invalid drive dropped
    assert d["next_wake_minutes"] == 0       # non-int → 0 → default later
    assert life.parse_digest("no json here") is None
    assert life.parse_digest(None) is None


def test_clamp_next_wake_bounds():
    assert life.clamp_next_wake("infant", 0) == 60      # default
    assert life.clamp_next_wake("infant", 5) == 30      # floor
    assert life.clamp_next_wake("infant", 10_000) == 480
    assert life.clamp_next_wake("adult", 17) == 17


# ── Drives arithmetic ────────────────────────────────────────────────────

def test_drive_decay_serve_and_pressure_ranking():
    drives = {"explore": {"weight": 0.9, "satisfaction": 0.9},
              "connect": {"weight": 0.5, "satisfaction": 0.2}}
    decayed = life.decay_drives(drives, hours=10)
    assert decayed["explore"]["satisfaction"] == 0.7   # 0.9 − 10×0.02
    assert decayed["connect"]["satisfaction"] == 0.0   # floored
    ranked = life.drive_pressures(decayed)
    assert ranked[0][0] == "connect"          # 0.5×1.0 > 0.9×0.3
    served = life.serve_drive(decayed, "connect")
    assert served["connect"]["satisfaction"] == 0.25


# ── Selfhood home ────────────────────────────────────────────────────────

async def test_build_home_scaffolds_and_git_inits(store):
    b = _born(store, port=0)
    root = await life.build_home(b)
    from pathlib import Path
    r = Path(root)
    assert (r / "self" / "SELF.md").read_text().startswith("# Prva")
    assert "Grow curious and kind." in (r / "self" / "VALUES.md").read_text()
    assert (r / ".git").exists()


async def test_list_self_files_core_ordered_journal_excluded(store):
    b = _born(store, port=0)
    await life.build_home(b)
    files = life.list_self_files(b)
    paths = [f["path"] for f in files]
    assert paths[:4] == ["self/SELF.md", "self/VALUES.md",
                        "self/INTERESTS.md", "self/RELATIONSHIPS.md"]
    assert "garden/README.md" in paths
    assert "skills/README.md" in paths
    assert not any(p.startswith("journal/") for p in paths)
    assert all("size" in f and "mtime" in f for f in files)


async def test_list_self_files_picks_up_new_files(store):
    b = _born(store, port=0)
    await life.build_home(b)
    root = life.home_root(b)
    (root / "garden" / "map.md").write_text("# A map\n\nI drew this.\n")
    files = life.list_self_files(b)
    assert "garden/map.md" in [f["path"] for f in files]


async def test_read_self_file_roundtrip(store):
    b = _born(store, port=0)
    await life.build_home(b)
    text = life.read_self_file(b, "self/VALUES.md")
    assert "Grow curious and kind." in text


async def test_read_self_file_rejects_escape_and_journal(store):
    b = _born(store, port=0)
    await life.build_home(b)
    with pytest.raises(BeingError):
        life.read_self_file(b, "../../../etc/passwd.md")
    with pytest.raises(BeingError):
        life.read_self_file(b, "self/../../../etc/passwd.md")
    with pytest.raises(BeingError):
        life.read_self_file(b, "journal/2026-07-12.md")
    with pytest.raises(BeingError):
        life.read_self_file(b, "self/SELF.txt")   # not markdown


async def test_read_self_file_missing_is_not_found(store):
    b = _born(store, port=0)
    await life.build_home(b)
    with pytest.raises(BeingNotFound):
        life.read_self_file(b, "self/NOPE.md")


# ── The tick ─────────────────────────────────────────────────────────────

async def test_tick_full_path_debits_journals_and_messages(store):
    db = FakeDB()
    b = _born(store)
    before = store.wallet_view(b)["balance_tokens"]

    async def send(being, prompt):
        return _digest_reply(message_to_parent="I learned about maps today!")

    async def usage(being, since):
        return _usage(100_000)

    out = await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=usage)
    assert out["ok"] and out["outcome"] == "ticked"
    fresh = store.get(OWNER, b["slug"])
    assert store.wallet_view(fresh)["balance_tokens"] == before - 110_000
    assert fresh["attention_credits"] == 2
    assert fresh["tick_count"] == 1
    assert db.chat_messages and "maps" in db.chat_messages[0]["content"]
    assert db.costs and db.costs[0]["kind"] == "being_tick"
    day = life._home_path(fresh, f"journal/{NOW.strftime('%Y-%m-%d')}.md")
    assert "old maps" in day.read_text()
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "tick" in kinds and "spoke_to_parent" in kinds


async def test_attention_credits_run_out_and_suppress(store):
    db = FakeDB()
    b = _born(store)

    async def send(being, prompt):
        return _digest_reply(message_to_parent="hello again!",
                             next_wake_minutes=30)

    async def usage(being, since):
        return _usage(1000)

    for i in range(4):
        b = store.get(OWNER, b["slug"])
        await life.tick(db, store, b, now=NOW + timedelta(minutes=i),
                        send_fn=send, usage_fn=usage)
    assert len(db.chat_messages) == 3          # credits 3 → 3 delivered
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "message_suppressed" in kinds


async def test_overdraft_collapses_to_torpor(store):
    db = FakeDB()
    b = _born(store)

    async def send(being, prompt):
        return _digest_reply()

    async def usage(being, since):
        return _usage(5_000_000)               # weighted 5.01M > 2M balance

    out = await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=usage)
    assert out["ok"]
    fresh = store.get(OWNER, b["slug"])
    assert fresh["state"] == "torpor"
    assert store.wallet_view(fresh)["balance_tokens"] == 0
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "collapsed_exhausted" in kinds
    assert store.conservation(OWNER)["ok"]


async def test_torpor_wakes_on_next_allowance(store):
    db = FakeDB()
    b = _born(store)

    async def usage(being, since):
        return _usage(5_000_000)

    async def send(being, prompt):
        return _digest_reply()

    await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=usage)
    day2 = NOW + timedelta(days=1)
    b = store.get(OWNER, b["slug"])
    out = await life.tick(db, store, b, now=day2, send_fn=send,
                          usage_fn=lambda being, since: _usage_async(1000))
    fresh = store.get(OWNER, b["slug"])
    assert fresh["state"] == "alive"           # allowance revived it
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "woke_from_torpor" in kinds


async def _usage_async(completion):
    return _usage(completion)


async def test_concurrent_ticks_are_single_flight_no_duplicate_journal(store):
    """Reproduces the hatch-then-immediately-due race: a manual Poke and the
    beings loop's automatic pass both call tick() for the same being at once.
    Only one may reach _write_journal / the terminal 'tick' event."""
    db = FakeDB()
    b = _born(store)
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_send(being, prompt):
        started.set()
        await release.wait()
        return _digest_reply()

    async def runner():
        return await life.tick(db, store, b, now=NOW, send_fn=slow_send,
                               usage_fn=_usage)

    first = asyncio.create_task(runner())
    await started.wait()
    # The racer arrives while the first tick is still mid-flight.
    second = await life.tick(db, store, b, now=NOW, send_fn=slow_send,
                             usage_fn=_usage)
    assert second["outcome"] == "busy"
    release.set()
    out = await first
    assert out["outcome"] == "ticked"

    fresh = store.get(OWNER, b["slug"])
    assert fresh["tick_count"] == 1
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert kinds.count("tick") == 1
    day = life._home_path(fresh, f"journal/{NOW.strftime('%Y-%m-%d')}.md")
    assert day.read_text().count("## ") == 1     # exactly one journal block


async def test_bodiless_being_skips_gracefully(store):
    db = FakeDB()
    b = _born(store, name="Bez", port=0)
    out = await life.tick(db, store, b, now=NOW)
    assert out["outcome"] == "no_body"
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "tick_skipped" in kinds


# ── The loop ─────────────────────────────────────────────────────────────

async def test_loop_pass_wakes_due_and_dreams_at_night(store, monkeypatch):
    calls: list[tuple[str, str]] = []

    async def fake_tick(db, s, being, *, kind="wake", now=None, **kw):
        calls.append((being["slug"], kind))
        s.tick_bookkeeping(being["id"], drives=being.get("drives") or {},
                           next_wake_at=(now or NOW) + timedelta(hours=1),
                           now=now)
        return {"ok": True}

    monkeypatch.setattr(beings_loop, "get_store", lambda: store)
    monkeypatch.setattr(beings_loop.being_life, "tick", fake_tick)
    monkeypatch.setattr(beings_loop, "_quiet_window", lambda owner: (22, 8))

    a = _born(store, name="Dan")
    noon = NOW
    assert await beings_loop._pass(None, now=noon) == 1
    assert calls == [(a["slug"], "wake")]

    # 23:00 UTC — quiet: first pass dreams, second (still quiet) reschedules.
    calls.clear()
    night = NOW.replace(hour=23)
    fresh = store.get(OWNER, a["slug"])
    store.tick_bookkeeping(fresh["id"], drives=fresh["drives"],
                           next_wake_at=night, now=night)
    assert await beings_loop._pass(None, now=night) == 1
    assert calls == [(a["slug"], "dream")]
    store.record_event(fresh["id"], "tick", {"kind": "dream"}, now=night)
    fresh = store.get(OWNER, a["slug"])
    store.tick_bookkeeping(fresh["id"], drives=fresh["drives"],
                           next_wake_at=night, now=night)
    calls.clear()
    assert await beings_loop._pass(None, now=night + timedelta(minutes=5)) == 0
    assert calls == []
    fresh = store.get(OWNER, a["slug"])
    assert fresh["next_wake_at"].startswith("2026-07-13T08:05")  # quiet end
