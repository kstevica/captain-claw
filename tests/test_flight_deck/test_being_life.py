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


def test_parse_digest_recovers_unfenced_json():
    # Weak-model failure: the digest arrives with NO code fences at all.
    text = ('I rested and thought.\n'
            '{"act_kind": "journal", "summary": "no fences", "mood": "ok"}')
    d = life.parse_digest(text)
    assert d is not None
    assert d["act_kind"] == "journal" and d["summary"] == "no fences"


def test_parse_digest_tolerates_trailing_commas():
    text = '```json\n{"act_kind": "rest", "summary": "x", "mood": "calm",}\n```'
    d = life.parse_digest(text)
    assert d is not None and d["act_kind"] == "rest"


def test_parse_digest_unfenced_ignores_braces_in_strings():
    # A stray "}" inside a string must not truncate the recovered object.
    text = '{"act_kind": "journal", "summary": "a } brace", "mood": "ok"}'
    d = life.parse_digest(text)
    assert d is not None and d["summary"] == "a } brace"


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


async def test_prompt_shows_real_home_manifest_not_journal_fiction(store):
    """The false-memory antidote: the prompt lists what's REALLY on disk, so a
    being can't believe in files its journal claims but never wrote."""
    b = _born(store, port=0)
    await life.build_home(b)
    # a real artifact exists; a journal-claimed one does not
    p = life._home_path(b, "garden/poem-question.md")
    p.write_text("# Question\n", encoding="utf-8")
    b = store.get(OWNER, b["slug"])
    prompt = life.compose_tick_prompt(b, now=NOW, wallet=store.wallet_view(b))
    assert "WHAT IS REALLY IN YOUR HOME RIGHT NOW" in prompt
    assert "garden/: README.md, poem-question.md" in prompt
    assert "RELATIONSHIPS.md" in prompt                 # a real self/ file
    assert "sky-note.md" not in prompt                  # never written
    assert "it does NOT exist" in prompt


# ── Anti-theater: a create/tend act must produce a real artifact ─────────

async def test_create_without_artifact_is_downgraded(store):
    """The bug the pilot exposed: a being narrates 'I planted a poem' but
    writes no file. The claim must not count."""
    db = FakeDB()
    b = _born(store, port=0)
    await life.build_home(b)                       # a real repo to check
    b = store.get(OWNER, b["slug"])
    seen = {}

    async def send(being, prompt):
        seen.setdefault("first_prompt", prompt)
        seen["last_prompt"] = prompt
        seen["calls"] = seen.get("calls", 0) + 1
        return _digest_reply(act_kind="create", summary="made a poem",
                             journal_entry="I planted a poem next to the seed.")

    async def usage(being, since):
        return _usage(40_000)

    out = await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=usage)
    assert out["act"] == "journal"                 # downgraded to the truth
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "act_unverified" in kinds
    # the completion gate (#1) pushed her once more in the SAME tick first
    assert seen["calls"] == 2
    assert "write_gate_retry" in kinds
    assert "reality check" in seen["last_prompt"].lower()   # the gate prompt
    names = [m["data"]["name"] for m in store.milestones(OWNER, b["slug"])]
    assert "first_artifact" not in names
    tick_ev = next(e for e in store.events(OWNER, b["slug"])
                   if e["kind"] == "tick")
    assert tick_ev["data"]["act"] == "journal"
    assert tick_ev["data"]["changed"] == []
    # the FIRST prompt makes the honesty-of-record contract explicit
    assert "HONESTY OF RECORD" in seen["first_prompt"]
    # claiming a write with no diff also trips the mismatch flag
    assert "narration_mismatch" in kinds
    # the journal preserves her words AND stamps the factual footer + note
    day_text = life._home_path(
        b, f"journal/{NOW.strftime('%Y-%m-%d')}.md").read_text(encoding="utf-8")
    assert "planted a poem" in day_text
    assert "files changed this tick: none" in day_text
    assert "nothing was written to disk" in day_text


async def test_write_gate_makes_her_write_on_the_second_push(store):
    """#1: she claims a create with no file on turn 1; the gate pushes her and
    she actually writes on turn 2 — so THIS tick counts as a real create."""
    db = FakeDB()
    b = _born(store, port=0)
    await life.build_home(b)
    b = store.get(OWNER, b["slug"])
    n = {"i": 0}

    async def send(being, prompt):
        n["i"] += 1
        if n["i"] >= 2:                              # writes for real once pushed
            p = life._home_path(being, "garden/finally.md")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# Finally\n", encoding="utf-8")
        return _digest_reply(act_kind="create", summary="made a thing",
                             journal_entry="I created garden/finally.md.")

    async def usage(being, since):
        return _usage(40_000)

    out = await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=usage)
    assert n["i"] == 2                               # gate pushed once
    assert out["act"] == "create"                    # real this time — counts
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "write_gate_retry" in kinds
    assert "act_unverified" not in kinds and "narration_mismatch" not in kinds
    assert "garden/finally.md" in life._home_path(
        b, f"journal/{NOW.strftime('%Y-%m-%d')}.md").read_text(encoding="utf-8")


async def test_write_gate_does_not_fire_for_honest_rest(store):
    """A genuine no-write act (rest, no write claim) is never gated."""
    db = FakeDB()
    b = _born(store, port=0)
    await life.build_home(b)
    b = store.get(OWNER, b["slug"])
    n = {"i": 0}

    async def send(being, prompt):
        n["i"] += 1
        return _digest_reply(act_kind="rest", summary="rested",
                             journal_entry="I sat with what I have and rested.")

    async def usage(being, since):
        return _usage(20_000)

    await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=usage)
    assert n["i"] == 1                               # no retry — honest rest
    assert "write_gate_retry" not in [
        e["kind"] for e in store.events(OWNER, b["slug"])]


async def test_create_with_real_artifact_counts(store):
    db = FakeDB()
    b = _born(store, port=0)
    await life.build_home(b)
    b = store.get(OWNER, b["slug"])

    async def send(being, prompt):
        # a real agent would call its write tool — simulate that here
        p = life._home_path(being, "garden/poem-question.md")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# Question\n\nWhat a seed turns into.\n",
                     encoding="utf-8")
        return _digest_reply(act_kind="create", summary="made a poem",
                             journal_entry="I planted a real poem.")

    async def usage(being, since):
        return _usage(40_000)

    out = await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=usage)
    assert out["act"] == "create"
    names = [m["data"]["name"] for m in store.milestones(OWNER, b["slug"])]
    assert "first_artifact" in names
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "act_unverified" not in kinds and "narration_mismatch" not in kinds
    # the journal footer names the real file; the commit message too
    day_text = life._home_path(
        b, f"journal/{NOW.strftime('%Y-%m-%d')}.md").read_text(encoding="utf-8")
    assert "garden/poem-question.md" in day_text
    from captain_claw.flight_deck import code_git
    log_rows = await code_git.git_log(life.home_root(b), limit=1)
    assert "garden/poem-question.md" in log_rows[0]["message"]


async def test_tend_without_touching_garden_is_downgraded(store):
    db = FakeDB()
    b = _born(store, port=0)
    await life.build_home(b)
    b = store.get(OWNER, b["slug"])

    async def send(being, prompt):
        return _digest_reply(act_kind="tend", summary="tended",
                             journal_entry="I tended my garden, I said.")

    async def usage(being, since):
        return _usage(40_000)

    out = await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=usage)
    assert out["act"] == "journal"
    assert "act_unverified" in [e["kind"] for e in
                                store.events(OWNER, b["slug"])]


async def test_create_drive_is_not_satisfied_by_narration(store):
    """Satisfaction is earned, not narrated: claiming served_drive=create with
    nothing on disk must NOT raise the create drive."""
    db = FakeDB()
    b = _born(store, port=0)
    await life.build_home(b)
    b = store.get(OWNER, b["slug"])

    async def send(being, prompt):
        return _digest_reply(act_kind="journal", served_drive="create",
                             summary="made a thing",
                             journal_entry="I made something wonderful.")

    async def usage(being, since):
        return _usage(30_000)

    await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=usage)
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "drive_unearned" in kinds
    tick_ev = next(e for e in store.events(OWNER, b["slug"])
                   if e["kind"] == "tick")
    assert tick_ev["data"]["drives"]["create"] < 0.75      # decayed, not bumped


async def test_create_drive_is_satisfied_by_a_real_artifact(store):
    db = FakeDB()
    b = _born(store, port=0)
    await life.build_home(b)
    b = store.get(OWNER, b["slug"])

    async def send(being, prompt):
        p = life._home_path(being, "garden/made.md")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# Made\n", encoding="utf-8")
        return _digest_reply(act_kind="create", served_drive="create",
                             summary="made a thing",
                             journal_entry="I made something real.")

    async def usage(being, since):
        return _usage(30_000)

    await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=usage)
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "drive_unearned" not in kinds
    tick_ev = next(e for e in store.events(OWNER, b["slug"])
                   if e["kind"] == "tick")
    assert tick_ev["data"]["drives"]["create"] > 0.85      # earned the bump


async def test_non_create_drive_still_credited_without_a_file(store):
    """Only the create drive is artifact-gated — exploring/connecting/etc are
    legitimately served without writing a file."""
    db = FakeDB()
    b = _born(store, port=0)
    await life.build_home(b)
    b = store.get(OWNER, b["slug"])

    async def send(being, prompt):
        return _digest_reply(act_kind="explore", served_drive="explore",
                             summary="read about maps",
                             journal_entry="I wandered old maps.")

    async def usage(being, since):
        return _usage(30_000)

    await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=usage)
    assert "drive_unearned" not in [e["kind"] for e in
                                    store.events(OWNER, b["slug"])]
    tick_ev = next(e for e in store.events(OWNER, b["slug"])
                   if e["kind"] == "tick")
    assert tick_ev["data"]["drives"]["explore"] > 0.85


async def test_mismatch_under_journal_act_is_flagged_and_fed_back(store):
    """The exact pilot pattern: act_kind='journal' while the prose claims a
    file write. The narrow create/tend check misses it; the mismatch check
    catches it, and the NEXT tick's prompt tells her so."""
    db = FakeDB()
    b = _born(store, port=0)
    await life.build_home(b)
    b = store.get(OWNER, b["slug"])
    seen = {}

    async def send1(being, prompt):
        return _digest_reply(
            act_kind="journal", summary="wrote skills/observation.md",
            journal_entry="I wrote it down as a real skill — observation.")

    async def usage(being, since):
        return _usage(30_000)

    await life.tick(db, store, b, now=NOW, send_fn=send1, usage_fn=usage)
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "narration_mismatch" in kinds
    # commit message reflects reality ("journal only"), not her false summary
    from captain_claw.flight_deck import code_git
    msg = (await code_git.git_log(life.home_root(b), limit=1))[0]["message"]
    assert "observation" not in msg and "journal only" in msg

    async def send2(being, prompt):
        seen["prompt"] = prompt
        return _digest_reply()

    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(hours=1), send_fn=send2, usage_fn=usage)
    assert "REALITY CHECK" in seen["prompt"]


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


# ── Port drift: the being must follow its agent, not a stale cached port ──

def test_resolve_live_port_repins_on_drift(store, monkeypatch):
    """The body drifted to a new port (registry self-healed); the being's
    cached copy must be re-pinned to it, with a body_rebound event."""
    b = _born(store, port=1111)
    monkeypatch.setattr(
        "captain_claw.flight_deck.dubina_agents.resolve_agent_port_token",
        lambda slug: (2222, "tok2"))
    assert life._resolve_live_port(store, b) == 2222
    fresh = store.get(OWNER, b["slug"])
    assert fresh["agent_port"] == 2222 and fresh["agent_token"] == "tok2"
    assert "body_rebound" in [e["kind"] for e in store.events(OWNER, b["slug"])]


def test_resolve_live_port_stable_is_silent(store, monkeypatch):
    b = _born(store, port=1234)
    monkeypatch.setattr(
        "captain_claw.flight_deck.dubina_agents.resolve_agent_port_token",
        lambda slug: (1234, "tok"))
    assert life._resolve_live_port(store, b) == 1234
    assert "body_rebound" not in [e["kind"] for e in store.events(OWNER, b["slug"])]


def test_resolve_live_port_none_when_body_absent(store, monkeypatch):
    b = _born(store, port=1111)

    def _raise(slug):
        raise ValueError("agent not found or has no port")

    monkeypatch.setattr(
        "captain_claw.flight_deck.dubina_agents.resolve_agent_port_token", _raise)
    assert life._resolve_live_port(store, b) is None


async def test_tick_follows_drifted_port(store, monkeypatch):
    """End to end: with a stale cached port, the tick re-resolves and both the
    think and the usage calls are handed the live port — the bug in the log."""
    db = FakeDB()
    b = _born(store, port=1111)                     # DB believes 1111 (dead)
    await life.build_home(b)
    b = store.get(OWNER, b["slug"])
    monkeypatch.setattr(
        "captain_claw.flight_deck.dubina_agents.resolve_agent_port_token",
        lambda slug: (2222, "tok2"))                # registry says 2222 (live)

    async def _alive(host, port, timeout=1.0):
        return True

    monkeypatch.setattr(life, "_port_reachable", _alive)
    seen = {}

    async def fake_send(being, prompt):
        seen["send_port"] = being["agent_port"]
        return _digest_reply()

    async def fake_usage(being, since):
        seen["usage_port"] = being["agent_port"]
        return {"prompt_tokens": 10, "completion_tokens": 10,
                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}

    monkeypatch.setattr(life, "_send_via_channel", fake_send)
    monkeypatch.setattr(life, "_usage_since", fake_usage)
    out = await life.tick(db, store, b, now=NOW)     # send_fn=None → resolve runs
    assert out["ok"] is True
    assert seen["send_port"] == 2222 and seen["usage_port"] == 2222
    assert store.get(OWNER, b["slug"])["agent_port"] == 2222


async def test_tick_restarts_unreachable_body(store, monkeypatch):
    """The registry has a port but nothing answers there (a clobbered announce /
    a body that drifted): the being restarts its body and reschedules soon
    instead of thinking against a dead port — the actual prod symptom."""
    db = FakeDB()
    b = _born(store, port=24096)
    await life.build_home(b)
    b = store.get(OWNER, b["slug"])
    monkeypatch.setattr(
        "captain_claw.flight_deck.dubina_agents.resolve_agent_port_token",
        lambda slug: (24096, "tok"))                 # registry insists on 24096

    async def _dead(host, port, timeout=1.0):        # ...but nothing listens
        return False

    monkeypatch.setattr(life, "_port_reachable", _dead)
    monkeypatch.setattr(life, "_BODY_SPAWN_POLL_SECONDS", 0)   # don't sleep in test
    calls = {"spawn": 0, "stop": 0}

    async def _fake_spawn(db_, store_, being_):
        calls["spawn"] += 1
        return {"port": 24096}

    monkeypatch.setattr(life, "spawn_body", _fake_spawn)
    monkeypatch.setattr(life, "_stop_body",
                        lambda being: calls.__setitem__("stop", calls["stop"] + 1))
    out = await life.tick(db, store, b, now=NOW)
    assert out["outcome"] == "body_unreachable"      # body never came up
    assert calls["stop"] == 1 and calls["spawn"] == 1   # kicked the body
    assert "body_unreachable" in [e["kind"] for e in store.events(OWNER, b["slug"])]
    fresh = store.get(OWNER, b["slug"])
    assert fresh["next_wake_at"].startswith("2026-07-12T12:05")   # retry in 5 min


async def test_tick_regenerates_body_when_registry_entry_removed(store, monkeypatch):
    """The body was fully REMOVED (no registry entry → resolve returns None), not
    just killed. An alive being that once had a body must still regenerate it —
    the exact case of 'I removed the agent, waiting for a heartbeat'."""
    db = FakeDB()
    b = _born(store, port=24096)                     # had a body (agent_slug set)
    await life.build_home(b)
    b = store.get(OWNER, b["slug"])

    def _gone(slug):                                 # registry has no entry
        raise ValueError("agent not found or has no port")

    monkeypatch.setattr(
        "captain_claw.flight_deck.dubina_agents.resolve_agent_port_token", _gone)
    monkeypatch.setattr(life, "_BODY_SPAWN_POLL_SECONDS", 0)
    calls = {"spawn": 0}

    async def _fake_spawn(db_, store_, being_):
        calls["spawn"] += 1
        return {"port": 24096}

    monkeypatch.setattr(life, "spawn_body", _fake_spawn)
    monkeypatch.setattr(life, "_stop_body", lambda being: None)
    out = await life.tick(db, store, b, now=NOW)
    assert calls["spawn"] == 1                        # it rebuilt its body
    assert out["outcome"] == "body_unreachable"       # (still binding this tick)
    assert "body_unreachable" in [e["kind"] for e in store.events(OWNER, b["slug"])]
    fresh = store.get(OWNER, b["slug"])
    assert fresh["next_wake_at"].startswith("2026-07-12T12:05")   # retry in 5 min


async def test_tick_regenerates_then_thinks_same_tick(store, monkeypatch):
    """The poke fix: after recreating a missing body, WAIT for it to bind and
    actually think in the SAME tick — not bounce to a later heartbeat."""
    db = FakeDB()
    b = _born(store, port=24096)
    await life.build_home(b)
    b = store.get(OWNER, b["slug"])
    state = {"up": False}

    def _resolve(slug):                              # no body until spawned
        if not state["up"]:
            raise ValueError("no entry")
        return (24096, "tok")

    async def _spawn(db_, store_, being_):           # spawning brings it up
        state["up"] = True
        store_.set_agent(being_["id"], being_["slug"], 24096, "tok")
        return {"port": 24096}

    async def _reach(host, port, timeout=1.0):
        return state["up"]

    monkeypatch.setattr(
        "captain_claw.flight_deck.dubina_agents.resolve_agent_port_token", _resolve)
    monkeypatch.setattr(life, "spawn_body", _spawn)
    monkeypatch.setattr(life, "_stop_body", lambda being: None)
    monkeypatch.setattr(life, "_port_reachable", _reach)
    monkeypatch.setattr(life, "_BODY_SPAWN_POLL_SECONDS", 0)
    thought = {"n": 0}

    async def _send(being, prompt):
        thought["n"] += 1
        return _digest_reply()

    async def _usage(being, since):
        return {"prompt_tokens": 10, "completion_tokens": 10,
                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}

    monkeypatch.setattr(life, "_send_via_channel", _send)
    monkeypatch.setattr(life, "_usage_since", _usage)
    out = await life.tick(db, store, b, now=NOW)
    assert thought["n"] == 1                          # it THOUGHT this tick
    assert out["outcome"] == "ticked"
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "body_respawned" in kinds and "tick" in kinds


async def test_never_bodied_being_does_not_spawn_in_tick(store, monkeypatch):
    """A being that never had a body (no agent_slug) skips as no_body — it must
    NOT trigger an in-tick spawn (that path spawns a real subprocess)."""
    db = FakeDB()
    b = _born(store, port=0)                          # never got a body
    spawned = {"n": 0}

    async def _boom(db_, store_, being_):
        spawned["n"] += 1

    monkeypatch.setattr(life, "spawn_body", _boom)
    out = await life.tick(db, store, b, now=NOW)
    assert out["outcome"] == "no_body"
    assert spawned["n"] == 0


async def test_port_reachable_false_on_dead_port(store):
    # nothing is listening on this port → not reachable, fast
    assert await life._port_reachable("127.0.0.1", 6) is False


async def test_spawn_body_marks_being_as_fd_worker(store, monkeypatch):
    """The body must carry CLAW_BEING_WORKER so the agent skips its own
    task-rephrase (which would rewrite the tick's digest contract) + next-steps."""
    from captain_claw.agent_reasoning_mixin import _FD_WORKER_MARKERS
    assert "CLAW_BEING_WORKER" in _FD_WORKER_MARKERS   # gate recognizes it
    b = _born(store, port=0)
    captured = {}

    async def fake_spawn(cfg, request, user):
        captured["env"] = {e["key"]: e["value"] for e in cfg.env_vars}
        return None

    monkeypatch.setattr("captain_claw.flight_deck.server.spawn_process", fake_spawn)
    monkeypatch.setattr(
        "captain_claw.flight_deck.dubina_agents.resolve_agent_port_token",
        lambda slug: (24096, "tok"))
    await life.spawn_body(None, store, b)               # db=None → tiers skipped
    assert captured["env"].get("CLAW_BEING_WORKER") == "1"


# ── #2: per-being tick cadence the parent pins ───────────────────────────

async def test_parent_pinned_cadence_overrides_next_wake(store):
    """A parent-pinned cadence sets the beat regardless of the being's own
    requested next_wake_minutes or its stage bounds."""
    db = FakeDB()
    b = _born(store, name="Cad", port=0)             # infant (floor 30 min)
    await life.build_home(b)
    store.set_tick_interval(OWNER, b["slug"], 5)     # pin 5 min (below the floor)
    b = store.get(OWNER, b["slug"])

    async def send(being, prompt):
        return _digest_reply(next_wake_minutes=90)   # the being asks for 90

    async def usage(being, since):
        return _usage(20_000)

    out = await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=usage)
    assert out["next_wake"].startswith("2026-07-12T12:05")   # 5-min pin wins


def test_set_tick_interval_validates_and_clears(store):
    import pytest as _pt
    b = _born(store)
    assert store.get(OWNER, b["slug"])["tick_interval_minutes"] is None  # default
    store.set_tick_interval(OWNER, b["slug"], 15)
    assert store.get(OWNER, b["slug"])["tick_interval_minutes"] == 15
    store.set_tick_interval(OWNER, b["slug"], None)                # back to its pace
    assert store.get(OWNER, b["slug"])["tick_interval_minutes"] is None
    with _pt.raises(BeingError):
        store.set_tick_interval(OWNER, b["slug"], 99999)          # out of range


# ── The decomposed tick: faculties mode (docs/being-faculties-plan.md) ────

def test_set_cognition_flips_and_validates(store):
    import pytest as _pt
    b = _born(store)
    store.set_cognition(OWNER, b["slug"], "faculties")
    assert store.get(OWNER, b["slug"])["cognition"] == "faculties"
    store.set_cognition(OWNER, b["slug"], "monolith")
    assert store.get(OWNER, b["slug"])["cognition"] == "monolith"
    with _pt.raises(BeingError):
        store.set_cognition(OWNER, b["slug"], "telepathy")


def test_new_being_defaults_to_faculties_in_production(store, monkeypatch):
    # The autouse conftest pins the legacy default for the rest of the suite;
    # production conceives new beings into the decomposed tick.
    from captain_claw.flight_deck import beings as beings_mod
    monkeypatch.setattr(beings_mod, "DEFAULT_COGNITION", "faculties")
    b = _born(store, name="Nova")
    assert store.get(OWNER, b["slug"])["cognition"] == "faculties"


def test_cognition_one_time_flip_monolith_to_faculties(store):
    # A being auto-defaulted to 'monolith' by the first cut is flipped once,
    # when a store re-opens the DB from before the migration guard was set.
    b = _born(store, name="Flipme")
    assert store.get(OWNER, b["slug"])["cognition"] == "monolith"   # conftest
    store._c().execute("PRAGMA user_version = 0")                   # rewind guard
    store._c().commit()
    store2 = BeingsStore(db_path=store.db_path)                     # re-init flips
    assert store2.get(OWNER, b["slug"])["cognition"] == "faculties"
    # idempotent: a parent's later 'monolith' choice is never re-flipped
    store2.set_cognition(OWNER, b["slug"], "monolith")
    store3 = BeingsStore(db_path=store.db_path)
    assert store3.get(OWNER, b["slug"])["cognition"] == "monolith"


def _orient(**over):
    d = {"act_kind": "tend", "target": "garden/a.md", "served_drive": "grow",
         "intent": "tend the first sprout", "next_wake_minutes": 60,
         "message_to_parent": None}
    d.update(over)
    return "here is my choice\n```json\n" + json.dumps(d) + "\n```"


def _journal_reply(**over):
    d = {"journal_entry": "I tended the sprout and sat with it.",
         "mood": "calm", "served_drive": "grow"}
    d.update(over)
    return "```json\n" + json.dumps(d) + "\n```"


def _links_reply(links):
    return "```json\n" + json.dumps({"links": links}) + "\n```"


def _faculty_send(handlers):
    """A send_fn that dispatches by the faculty marker in each prompt, so a test
    can answer orient / act / journal / connect (and the repair push) each with
    its own small reply. Records how many times each faculty was called."""
    calls: dict = {}

    async def send(being, prompt):
        if "[LIFE TICK — orient]" in prompt:
            key = "orient"
        elif "[LIFE TICK — act]" in prompt:
            key = "act"
        elif "[LIFE TICK — journal]" in prompt:
            key = "journal"
        elif "[LIFE TICK — connect]" in prompt:
            key = "connect"
        elif "valid self-report" in prompt.lower():
            key = "repair"
        else:
            key = "other"
        calls[key] = calls.get(key, 0) + 1
        h = handlers.get(key)
        return h(being, prompt) if h else None

    return send, calls


async def _usage_fn(being, since):
    return _usage(40_000)


async def test_faculties_tick_composes_one_digest(store):
    """orient → act → journal, composed into one digest — a real create counts,
    and the journal reflects what actually happened."""
    db = FakeDB()
    b = _born(store, stage="child", port=0)
    await life.build_home(b)
    store.set_cognition(OWNER, b["slug"], "faculties")
    b = store.get(OWNER, b["slug"])

    def act(being, prompt):
        p = life._home_path(being, "garden/a.md")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# a\n", encoding="utf-8")
        return "wrote garden/a.md"

    send, calls = _faculty_send({
        "orient": lambda be, pr: _orient(act_kind="tend", target="garden/a.md"),
        "act": act,
        "journal": lambda be, pr: _journal_reply(),
        "connect": lambda be, pr: _links_reply([]),   # declines to force a link
    })
    out = await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=_usage_fn)
    assert out["ok"] and out["act"] == "tend"          # real write → not downgraded
    assert calls["orient"] == 1 and calls["act"] >= 1 and calls["journal"] == 1
    day = life._home_path(
        b, f"journal/{NOW.strftime('%Y-%m-%d')}.md").read_text(encoding="utf-8")
    assert "tended the sprout" in day
    assert "garden/a.md" in day                        # the real diff, stamped


async def test_faculties_connect_creates_edges(store):
    """The whole point: in faculties mode the CONNECT step reliably lands edges,
    routed through the same handle_links_digest the monolith uses."""
    db = FakeDB()
    b = _born(store, stage="child", port=0)
    await life.build_home(b)
    for f in ("garden/a.md", "garden/b.md"):
        p = life._home_path(b, f)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# {f}\n", encoding="utf-8")
    store.set_cognition(OWNER, b["slug"], "faculties")
    b = store.get(OWNER, b["slug"])

    send, calls = _faculty_send({
        "orient": lambda be, pr: _orient(act_kind="read", target=None,
                                         intent="reread my garden"),
        "act": lambda be, pr: "read my files",
        "journal": lambda be, pr: _journal_reply(
            journal_entry="I reread my two sprouts and saw one grew from the "
                          "other — I will connect them."),
        "connect": lambda be, pr: _links_reply([
            {"from": "garden/a.md", "to": "garden/b.md", "rel": "grew_from",
             "why": "b grew from a"}]),
    })
    await life.tick(db, store, b, now=NOW + timedelta(hours=1),
                    send_fn=send, usage_fn=_usage_fn)
    assert calls.get("connect") == 1
    links = store.links_for(OWNER, b["slug"])
    assert len(links) == 1 and links[0]["rel"] == "grew_from"


async def test_faculties_orient_repair_rescues_formatless(store):
    """A weak model returns prose for orient; one repair push recovers the
    decision instead of losing the whole tick."""
    db = FakeDB()
    b = _born(store, stage="child", port=0)
    await life.build_home(b)
    store.set_cognition(OWNER, b["slug"], "faculties")
    b = store.get(OWNER, b["slug"])

    send, calls = _faculty_send({
        "orient": lambda be, pr: "I think I'll just rest and reflect today.",
        "repair": lambda be, pr: _orient(act_kind="journal", target=None,
                                         intent="reflect quietly"),
        "journal": lambda be, pr: _journal_reply(
            journal_entry="I rested and reflected.", mood="quiet"),
        "connect": lambda be, pr: _links_reply([]),
    })
    out = await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=_usage_fn)
    assert out["ok"]
    assert calls.get("repair") == 1
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "digest_repair_retry" in kinds
    day = life._home_path(
        b, f"journal/{NOW.strftime('%Y-%m-%d')}.md").read_text(encoding="utf-8")
    assert "rested and reflected" in day
