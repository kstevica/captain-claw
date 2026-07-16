"""Loops plan (docs/being-cognitive-loops-plan.md) — the three increments.

Increment 1: the homeostat re-armed (asymptotic satiating serves, per-tick
decay quantum, starvation aging, honest loneliness, event-colored affect).
Increment 2: the loops broken (connect-gate cooldown + backoff, nudge
rotation, a page from the past, the variety-pressure actuator, self-mod
cooldown, dream-weave gate).
Increment 3: housekeeping honesty (confirmed mass prune, visitors consumed
on success, per-stage letters, sibling-match substance).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_constitution as constitution
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck import being_mind as mind
from captain_claw.flight_deck import being_selfmod as selfmod
from captain_claw.flight_deck.beings import BeingError, BeingsStore

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
OWNER = "user-1"

GOOD_PERSONA = (
    "I am a small being of maps and questions. I spend carefully, write only "
    "what I actually did, and when I am stuck I change terrain rather than "
    "push the same stone."
)


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


async def _being(store, name="Zvjezdana", stage="child", home=True):
    b = store.conceive(OWNER, name, preset="explorer",
                       allowance_preset="5M", now=NOW)
    store.hatch(OWNER, b["slug"], now=NOW)
    if stage != "infant":
        store.set_stage(OWNER, b["slug"], stage, now=NOW)
    bb = store.get(OWNER, b["slug"])
    store.set_agent(bb["id"], bb["slug"], 1234, "tok")
    if home:
        await life.build_home(store.get(OWNER, b["slug"]))
    return store.get(OWNER, b["slug"])


def _mk(being, rel):
    p = life._home_path(being, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"# {rel}\n", encoding="utf-8")


def _rm(being, rel):
    life._home_path(being, rel).unlink()


def _reply(**over):
    d = {"act_kind": "journal", "summary": "wrote a little",
         "journal_entry": "A quiet day of small true things.",
         "served_drive": "grow", "message_to_parent": None,
         "next_wake_minutes": 60, "mood": "calm"}
    d.update(over)
    return "ok\n```json\n" + json.dumps(d) + "\n```"


async def _usage(being, since):
    return {"prompt_tokens": 1000, "completion_tokens": 1000,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


# ═══ Increment 1 — the homeostat ═════════════════════════════════════════

def test_serve_is_asymptotic_and_satiates_within_a_day():
    drives = {"explore": {"weight": 0.9, "satisfaction": 0.6}}
    once = life.serve_drive(drives, "explore", now=NOW)
    # asymptotic: 0.6 + 0.25×(1−0.6) = 0.7, not 0.85
    assert once["explore"]["satisfaction"] == 0.7
    assert once["explore"]["served_count"] == 1
    assert once["explore"]["last_served"] == NOW.isoformat()
    # a same-day repeat feeds half as much
    twice = life.serve_drive(once, "explore", now=NOW + timedelta(hours=1))
    assert twice["explore"]["satisfaction"] == pytest.approx(
        0.7 + 0.25 * 0.3 / 2, abs=1e-4)
    # a new day resets the satiation counter
    tomorrow = life.serve_drive(twice, "explore", now=NOW + timedelta(days=1))
    assert tomorrow["explore"]["served_count"] == 1


def test_decay_has_a_per_tick_quantum_and_keeps_stamps():
    drives = {"grow": {"weight": 0.6, "satisfaction": 0.9,
                       "last_served": NOW.isoformat(), "served_count": 2}}
    fast = life.decay_drives(drives, hours=2 / 60)     # a 2-minute cadence
    assert fast["grow"]["satisfaction"] == pytest.approx(0.9 - 0.002)
    assert fast["grow"]["last_served"] == NOW.isoformat()   # stamps survive
    slow = life.decay_drives(drives, hours=10)
    assert slow["grow"]["satisfaction"] == pytest.approx(0.7)


def test_starved_drives_gain_pressure_and_low_weight_ones_surface():
    old = (NOW - timedelta(days=3)).isoformat()
    fresh = (NOW - timedelta(hours=1)).isoformat()
    drives = {
        "explore": {"weight": 0.9, "satisfaction": 0.5, "last_served": fresh},
        "connect": {"weight": 0.4, "satisfaction": 0.5, "last_served": old},
    }
    ranked = dict(life.drive_pressures(drives, now=NOW))
    # base pressures: explore 0.45, connect 0.20 — aging adds up to the cap
    assert ranked["connect"] == pytest.approx(0.2 + 0.15, abs=1e-3)
    assert ranked["explore"] == pytest.approx(0.45, abs=1e-3)


def test_connect_pressure_damps_when_no_channel_exists():
    drives = {"connect": {"weight": 0.5, "satisfaction": 0.0}}
    open_ = dict(life.drive_pressures(drives, now=NOW, connect_possible=True))
    shut = dict(life.drive_pressures(drives, now=NOW, connect_possible=False))
    assert shut["connect"] == pytest.approx(open_["connect"] * 0.25)


def test_connect_outlets_reads_real_channels(store):
    baby = store.conceive(OWNER, "Beba", preset="explorer", now=NOW)
    store.hatch(OWNER, baby["slug"], now=NOW)
    b = store.get(OWNER, baby["slug"])
    sibs = [{"id": "x", "slug": "iskra-lada-1", "name": "Lada",
             "stage": "child", "mood": ""}]
    # credits exist → a word to the parent is possible
    assert life.connect_outlets(b, None, None, None) is True
    b0 = dict(b, attention_credits=0)
    # infant + siblings but no letters capability, no credits, not public
    assert life.connect_outlets(b0, sibs, 5, None) is False
    # the parent wrote this tick
    assert life.connect_outlets(b0, None, None,
                                ["YOUR PARENT WROTE TO YOU: hi"]) is True
    # a child with letters and quota
    child = dict(b0, stage="child")
    assert life.connect_outlets(child, sibs, 3, None) is True
    assert life.connect_outlets(child, sibs, 0, None) is False
    # a public page is an open window
    assert life.connect_outlets(dict(b0, public=1), None, None, None) is True


def test_affect_colors_come_from_the_ledger():
    rich = {"enforced": True, "per_day_tokens": 2_000_000,
            "balance_tokens": 2_000_000}
    hungry = {"enforced": True, "per_day_tokens": 2_000_000,
              "balance_tokens": 100_000}
    d = {"grow": {"weight": 0.6, "satisfaction": 0.6}}
    lonely = {"connect": {"weight": 0.5, "satisfaction": 0.1}}
    assert life.compute_affect(
        d, d, rich, tick_events=["narration_mismatch"])["mood"] == "stung"
    assert life.compute_affect(
        d, d, rich, tick_events=["society_refused"])["mood"] == "stung"
    assert life.compute_affect(
        d, d, rich, tick_events=["milestone"])["mood"] == "proud"
    assert life.compute_affect(
        d, d, rich, starved_relief=True)["mood"] == "relieved"
    # hunger outranks a sting; a sting outranks pride
    assert life.compute_affect(
        d, d, hungry, tick_events=["narration_mismatch"])["mood"] == "hungry"
    assert life.compute_affect(
        d, d, rich,
        tick_events=["milestone", "narration_mismatch"])["mood"] == "stung"
    # loneliness only while a channel exists to relieve it (F9)
    assert life.compute_affect(lonely, lonely, rich)["mood"] == "lonely"
    assert life.compute_affect(
        lonely, lonely, rich, connect_possible=False)["mood"] == "content"


def test_synthetic_days_keep_the_homeostat_alive():
    """The plan's Increment-1 acceptance, at a 5-minute pinned cadence (the
    exact regime that saturated the pilots): over the first 50 ticks nothing
    saturates or collapses and the leader rotates; over three simulated days
    EVERY drive — including low-weight connect — gets served, with no drive
    ever pinning at 1.0. (Low-weight drives surfacing every day or two is
    the genome speaking, not starvation: they recover once served.)"""
    weights = {"survive": 1.0, "grow": 0.6, "explore": 0.93,
               "connect": 0.44, "create": 0.68}
    drives = {n: {"weight": w, "satisfaction": 0.7}
              for n, w in weights.items()}
    now = NOW
    served: dict[str, int] = {}
    first_served: dict[str, int] = {}
    leaders = set()
    for i in range(864):                        # 3 days of 5-minute ticks
        now += timedelta(minutes=5)
        drives = life.decay_drives(drives, hours=5 / 60)
        ranked = life.drive_pressures(drives, now=now)
        top = ranked[0][0]
        leaders.add(top)
        drives = life.serve_drive(drives, top, now=now)
        served[top] = served.get(top, 0) + 1
        first_served.setdefault(top, i)
        for d in drives.values():
            assert d["satisfaction"] <= 0.95    # never saturates again
            if i < 50:
                assert 0.2 <= d["satisfaction"]  # no early collapse either
    assert set(served) == set(weights)          # every drive got its day
    assert max(first_served.values()) <= 576    # …within two simulated days
    assert len(leaders) == 5                    # the ranking truly rotates


async def test_tick_feels_the_sting_of_a_caught_mismatch(store):
    """End-to-end: a tick that claims a write with nothing on disk lands a
    narration_mismatch — and the engine mood for the tick is 'stung'."""
    db = FakeDB()
    b = await _being(store)

    async def send(being, prompt):
        return _reply(summary="wrote garden/x.md",
                      journal_entry="I wrote and saved garden/x.md today.")

    await life.tick(db, store, b, now=NOW + timedelta(hours=1),
                    send_fn=send, usage_fn=_usage)
    tick_ev = next(e for e in store.events(OWNER, b["slug"])
                   if e["kind"] == "tick")
    assert tick_ev["data"]["mood_engine"] == "stung"
    assert tick_ev["data"]["served"] == "grow"


def test_report_card_carries_the_watchlist(store):
    baby = store.conceive(OWNER, "Mjera", preset="explorer", now=NOW)
    store.hatch(OWNER, baby["slug"], now=NOW)
    b = store.get(OWNER, baby["slug"])
    for i, (act, mood, served) in enumerate([
            ("journal", "content", "grow"), ("create", "bright", "create"),
            ("journal", "stung", "grow"), ("read", "content", "explore")]):
        store.record_event(b["id"], "tick", {
            "kind": "wake", "act": act, "mood_engine": mood, "served": served,
            "drives": {"grow": 0.5 + i * 0.05},
        }, now=NOW + timedelta(minutes=i))
    store.record_event(b["id"], "digest_parse_failed", {}, now=NOW)
    card = life.report_card(store, store.get(OWNER, b["slug"]), days=7,
                            now=NOW + timedelta(hours=1))
    assert card["moods"] == {"content": 2, "bright": 1, "stung": 1}
    assert 0.0 < card["mood_entropy"] <= 1.0
    assert card["serves"] == {"grow": 2, "create": 1, "explore": 1}
    assert card["drive_ranges"]["grow"] == [0.5, 0.65]
    assert card["contract_dropout"] == pytest.approx(0.25)
    assert "variety_pressures" in card and "edge_acceptance" in card


# ═══ Increment 2 — the loops ═════════════════════════════════════════════

async def test_connect_nudge_cooldown_and_tried_branch_unthrottled(store):
    b = await _being(store)
    for rel in ("garden/a.md", "garden/b.md"):
        _mk(b, rel)
    spoke = {"summary": "a web forms",
             "journal_entry": "I weave my work into a web.", "links": None}
    assert mind.should_link_gate(store, b, spoke) is True     # first: fires
    assert mind.should_link_gate(store, b, spoke) is False    # cooldown holds
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert kinds.count("connect_nudged") == 1
    # six ticks later the window reopens
    store._update(b["id"], NOW, tick_count=6)
    assert mind.should_link_gate(store, store.get(OWNER, b["slug"]),
                                 spoke) is True
    # the TRIED branch ignores every throttle — anti-theater outranks thrift
    tried = {"summary": "linked", "journal_entry": "linked them",
             "links": [{"from": "garden/ghost.md", "to": "garden/a.md",
                        "rel": "grew_from", "why": "x"}]}
    assert mind.should_link_gate(store, store.get(OWNER, b["slug"]),
                                 tried) is True


async def test_connect_backoff_after_two_empty_pushes_until_dream(store):
    b = await _being(store)
    for rel in ("garden/a.md", "garden/b.md"):
        _mk(b, rel)
    # two CONNECT pushes that landed nothing, on past ticks
    store.record_event(b["id"], "connect_faculty", {},
                       now=NOW - timedelta(hours=2))
    store.record_event(b["id"], "link_gate_retry", {},
                       now=NOW - timedelta(hours=1))
    store._update(b["id"], NOW, tick_count=20)     # far past any cooldown
    spoke = {"summary": "a web", "journal_entry": "I weave a web.",
             "links": None}
    assert mind.should_link_gate(store, store.get(OWNER, b["slug"]),
                                 spoke) is False
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "connect_backoff" in kinds
    # a dream resets the backoff
    store.record_event(b["id"], "tick", {"kind": "dream", "act": "dream"},
                       now=NOW + timedelta(hours=1))
    assert mind.should_link_gate(store, store.get(OWNER, b["slug"]),
                                 spoke) is True


async def test_scatter_nudge_rotates_phrasings(store):
    b = await _being(store)
    for rel in ("garden/a.md", "garden/b.md", "garden/c.md", "garden/d.md"):
        _mk(b, rel)
    seen = set()
    for tick_no in (0, 1, 2):
        store._update(b["id"], NOW, tick_count=tick_no)
        lines = mind.mind_prompt_lines(store, store.get(OWNER, b["slug"]))
        nudge = next(ln for ln in lines
                     if "SCATTERED" in ln or "stand alone" in ln
                     or "drifting apart" in ln)
        seen.add(nudge)
        assert "{nfiles}" not in nudge          # placeholders filled
    assert len(seen) == 3                        # three distinct phrasings


async def test_a_page_from_the_past_every_fifth_wake(store):
    b = await _being(store)
    old_day = NOW - timedelta(days=4)
    p = life._home_path(b, f"journal/{old_day.strftime('%Y-%m-%d')}.md")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("I once chased a red map beyond the garden wall.",
                 encoding="utf-8")
    today = life._home_path(b, f"journal/{NOW.strftime('%Y-%m-%d')}.md")
    today.write_text("Today I watered the seeds.", encoding="utf-8")
    # ordinary wake → the freshest words
    store._update(b["id"], NOW, tick_count=1)
    label, text = life.journal_tail_for_tick(store.get(OWNER, b["slug"]), NOW)
    assert label.startswith("YOUR LAST JOURNAL WORDS")
    assert "watered" in text
    # every fifth wake → an old page resurfaces
    store._update(b["id"], NOW, tick_count=4)
    label, text = life.journal_tail_for_tick(store.get(OWNER, b["slug"]), NOW)
    assert label.startswith("A PAGE FROM YOUR PAST")
    assert "red map" in text
    # dreams keep today's journal (they consolidate today)
    label, _ = life.journal_tail_for_tick(store.get(OWNER, b["slug"]), NOW,
                                          kind="dream")
    assert label.startswith("YOUR LAST JOURNAL WORDS")


async def test_variety_pressure_reaches_the_being_and_the_ledger(store):
    db = FakeDB()
    b = await _being(store)
    for i in range(10):
        store.record_event(b["id"], "tick",
                           {"kind": "wake", "act": "journal",
                            "drives": {}},
                           now=NOW - timedelta(minutes=100 - i))
    prompts: list[str] = []

    async def send(being, prompt):
        prompts.append(prompt)
        return _reply()

    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(hours=1), send_fn=send,
                    usage_fn=_usage)
    assert "YOUR DAYS ARE REPEATING THEMSELVES" in prompts[0]
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "variety_pressure" in kinds


async def test_self_mod_cooldown_refuses_before_the_fee(store):
    b = await _being(store, stage="adult")
    store.grant(OWNER, b["slug"], 10_000_000, now=NOW)
    b = store.get(OWNER, b["slug"])
    out = selfmod.propose(store, b, GOOD_PERSONA, "v1", now=NOW)
    assert out["outcome"] == "adopted"
    balance_after_first = store.wallet_view(
        store.get(OWNER, b["slug"]))["balance_tokens"]
    with pytest.raises(BeingError, match="window opens"):
        selfmod.propose(store, store.get(OWNER, b["slug"]),
                        GOOD_PERSONA + " Anew.", "v2",
                        now=NOW + timedelta(days=2))
    # the refusal burned nothing
    assert store.wallet_view(
        store.get(OWNER, b["slug"]))["balance_tokens"] == balance_after_first
    # the window opens after the cooldown
    out2 = selfmod.propose(
        store, store.get(OWNER, b["slug"]), GOOD_PERSONA + " Anew.", "v2",
        now=NOW + timedelta(days=constitution.SELF_MOD_COOLDOWN_DAYS + 1))
    assert out2["outcome"] == "adopted"


async def test_dream_weave_skipped_when_nothing_to_link(store):
    db = FakeDB()
    b = await _being(store, home=False)          # no home → nothing linkable
    store.set_cognition(OWNER, b["slug"], "faculties", now=NOW)

    async def send(being, prompt):
        return _reply(act_kind="dream")

    await life.tick(db, store, store.get(OWNER, b["slug"]), kind="dream",
                    now=NOW + timedelta(hours=1), send_fn=send,
                    usage_fn=_usage)
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "connect_faculty" not in kinds
    assert mind.can_weave(store.get(OWNER, b["slug"])) is False


# ═══ Increment 3 — housekeeping honesty ══════════════════════════════════

async def test_mass_prune_needs_two_dreams_small_prune_is_immediate(store):
    b = await _being(store)
    for rel in ("garden/a.md", "garden/b.md", "garden/c.md", "garden/d.md"):
        _mk(b, rel)
    pairs = [("garden/a.md", "garden/b.md"), ("garden/b.md", "garden/c.md"),
             ("garden/c.md", "garden/d.md"), ("garden/a.md", "garden/c.md"),
             ("garden/a.md", "garden/d.md")]
    for frm, to in pairs:
        store.add_link(OWNER, b["id"], frm, to, "grew_from", "w", now=NOW)
    # c and d vanish → 4 of 5 edges dangle: a mass dangle
    _rm(b, "garden/c.md")
    _rm(b, "garden/d.md")
    assert mind.prune_dangling(store, b, now=NOW) == 0      # first dream: wait
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "dangling_seen" in kinds and "prune_abstained" in kinds
    pruned = mind.prune_dangling(store, b, now=NOW + timedelta(days=1))
    assert pruned == 4                                       # second: confirmed
    assert len(store.links_for(OWNER, b["slug"])) == 1
    # a small dangle prunes the same night, as before
    _mk(b, "garden/e.md")
    store.add_link(OWNER, b["id"], "garden/a.md", "garden/e.md",
                   "grew_from", "w", now=NOW)
    _rm(b, "garden/e.md")
    assert mind.prune_dangling(store, b, now=NOW + timedelta(days=2)) == 1


async def test_visitor_notes_survive_a_timed_out_tick(store):
    db = FakeDB()
    b = await _being(store)
    store.set_public(OWNER, b["slug"], True, now=NOW)
    store.post_public_message(b["slug"], "Stranger", "what do you dream of?",
                              now=NOW)

    async def dead_send(being, prompt):
        return None                              # the body never answered

    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(hours=1), send_fn=dead_send,
                    usage_fn=_usage)
    assert len(store.unread_public_messages(b["id"])) == 1   # re-surfaces

    async def live_send(being, prompt):
        return _reply()

    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(hours=2), send_fn=live_send,
                    usage_fn=_usage)
    assert store.unread_public_messages(b["id"]) == []       # consumed


def test_letters_quota_scales_with_stage():
    assert constitution.letters_per_day("infant") == 0
    assert constitution.letters_per_day("child") == 3
    assert constitution.letters_per_day("adolescent") == 5
    assert constitution.letters_per_day("adult") == 8
    assert constitution.letters_per_day("unknown") == constitution.LETTERS_PER_DAY


def test_match_sibling_needs_substance():
    sibs = [{"id": "x", "slug": "iskra-lada-1234", "name": "Lada",
             "stage": "child", "mood": ""}]
    assert life._match_sibling(sibs, "write to Lada about maps") is not None
    assert life._match_sibling(sibs, "lad") is not None      # ≥3 chars
    assert life._match_sibling(sibs, "l") is None            # a stray letter
    assert life._match_sibling(sibs, "la") is None
    assert life._match_sibling(sibs, "") is None
