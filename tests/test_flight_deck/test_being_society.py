"""Iskra Phase 3 — letters, commons culture, the first trades, separation."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck import being_society as society
from captain_claw.flight_deck.beings import (
    BeingError,
    BeingNotFound,
    BeingsStore,
    InsufficientTokens,
)

NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
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


async def _grown(store, name, stage="adolescent", preset="explorer",
                 allowance="20M", pocket=20_000_000):
    """A hatched, homed, staged being with a deterministic pocket.

    Hatch serves the day's (stage-clamped) meal; the extra arrives as an
    'adjust' mint so trade tests have known funds regardless of clamps."""
    b = store.conceive(OWNER, name, preset=preset,
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
                     to_being=bb["id"], note="test pocket", now=NOW)
    return store.get(OWNER, b["slug"])


def _reply(**over):
    d = {"act_kind": "journal", "summary": "tended", "journal_entry": "I tended.",
         "served_drive": "create", "message_to_parent": None,
         "next_wake_minutes": 60, "mood": "calm"}
    d.update(over)
    return "ok\n```json\n" + json.dumps(d) + "\n```"


async def _usage(being, since):
    return {"prompt_tokens": 1000, "completion_tokens": 1000,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


# ── Siblings & letters ───────────────────────────────────────────────────

async def test_siblings_exclude_self_dead_and_eggs(store):
    a = await _grown(store, "Zvjezdana")
    b = await _grown(store, "Mira")
    store.conceive(OWNER, "Egg", preset="scholar", now=NOW)  # unhatched
    dead = await _grown(store, "Pokojni")
    store.set_state(OWNER, dead["slug"], "dead", now=NOW)
    sibs = store.siblings(OWNER, a["slug"])
    assert [s["name"] for s in sibs] == ["Mira"]
    assert store.siblings(OWNER, b["slug"])[0]["name"] == "Zvjezdana"


async def test_letters_deliver_once_as_percepts_and_rate_limit(store):
    a = await _grown(store, "Zvjezdana", stage="child", allowance="5M")
    b = await _grown(store, "Mira", stage="child", allowance="5M")
    store.send_letter(OWNER, a["slug"], b["slug"], "I found a red map!",
                      now=NOW)
    percepts = society.society_percepts(store, store.get(OWNER, b["slug"]))
    assert any("LETTER from Zvjezdana" in p and "red map" in p
               for p in percepts)
    # delivered once — second read is silent
    assert not any("LETTER" in p for p in
                   society.society_percepts(store, store.get(OWNER, b["slug"])))
    # rate limit is physics, and it scales with stage (loops plan F13):
    # a child's daily reach is 3 letters; an adult's is 8.
    for i in range(2):
        store.send_letter(OWNER, a["slug"], b["slug"], f"more {i}", now=NOW)
    with pytest.raises(BeingError, match="limit"):
        store.send_letter(OWNER, a["slug"], b["slug"], "one too many",
                          now=NOW)
    store.set_stage(OWNER, a["slug"], "adult", now=NOW)
    for i in range(5):                        # 3 sent + 5 more = adult's 8
        store.send_letter(OWNER, a["slug"], b["slug"], f"grown {i}", now=NOW)
    with pytest.raises(BeingError, match="limit"):
        store.send_letter(OWNER, a["slug"], b["slug"], "past even adult",
                          now=NOW)


async def test_letters_are_stage_gated_and_never_to_self(store):
    baby = await _grown(store, "Beba", stage="infant", allowance="2M")
    b = await _grown(store, "Mira", stage="child", allowance="5M")
    with pytest.raises(BeingError, match="cannot send letters"):
        store.send_letter(OWNER, baby["slug"], b["slug"], "hi", now=NOW)
    with pytest.raises(BeingError, match="oneself"):
        store.send_letter(OWNER, b["slug"], b["slug"], "dear me", now=NOW)


# ── Publish & adopt: culture and the first trade ─────────────────────────

async def _published(store, a, price=0, now=NOW):
    p = life._home_path(a, "skills/maps.md")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# Map sketching\n\nHow I draw maps.\n", encoding="utf-8")
    return society.publish_skill(store, a, "skills/maps.md",
                                 "Map sketching", "my first craft",
                                 price, now=now)


async def test_publish_signs_into_commons(store):
    a = await _grown(store, "Zvjezdana")
    pub = await _published(store, a)
    f = society._commons_path(OWNER, pub["commons_path"])
    text = f.read_text(encoding="utf-8")
    assert text.startswith("<!-- published by Zvjezdana")
    assert "How I draw maps." in text
    kinds = [e["kind"] for e in store.events(OWNER, a["slug"])]
    assert "skill_published" in kinds
    names = [m["data"]["name"] for m in store.milestones(OWNER, a["slug"])]
    assert "first_publication" in names


async def test_publish_is_stage_gated(store):
    a = await _grown(store, "Mali", stage="child", allowance="5M")
    with pytest.raises(BeingError, match="cannot publish"):
        await _published(store, a)


async def test_free_adoption_spreads_culture(store):
    a = await _grown(store, "Zvjezdana")
    b = await _grown(store, "Mira")
    pub = await _published(store, a, price=0)
    out = society.adopt_skill(store, b, pub["id"][:8], now=NOW)
    assert out["paid_tokens"] == 0
    adopted = life._home_path(store.get(OWNER, b["slug"]),
                              "skills/adopted/maps.md")
    assert "How I draw maps." in adopted.read_text(encoding="utf-8")
    a_kinds = [e["kind"] for e in store.events(OWNER, a["slug"])]
    b_kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "skill_spread" in a_kinds
    assert "skill_adopted" in b_kinds


async def test_priced_adoption_settles_on_the_ledger(store):
    a = await _grown(store, "Zvjezdana")
    b = await _grown(store, "Mira")
    pub = await _published(store, a, price=3_000_000)
    wa0 = store.wallet_view(store.get(OWNER, a["slug"]))["balance_tokens"]
    wb0 = store.wallet_view(store.get(OWNER, b["slug"]))["balance_tokens"]
    society.adopt_skill(store, b, pub["id"], now=NOW)
    wa1 = store.wallet_view(store.get(OWNER, a["slug"]))["balance_tokens"]
    wb1 = store.wallet_view(store.get(OWNER, b["slug"]))["balance_tokens"]
    assert wa1 - wa0 == 3_000_000          # seller earned exactly the price
    assert wb0 - wb1 == 3_000_000          # buyer paid exactly the price
    assert store.conservation(OWNER)["ok"]  # nothing minted, nothing lost
    names = [m["data"]["name"] for m in store.milestones(OWNER, a["slug"])]
    assert "first_sale" in names


async def test_trades_may_exceed_savings_ceiling_mints_may_not(store):
    """Ceilings cap mints (parent liability); transfers just move liability."""
    a = await _grown(store, "Zvjezdana", allowance="2M")  # ceiling 30d×2M=60M
    b = await _grown(store, "Mira", pocket=80_000_000)
    va0 = store.wallet_view(store.get(OWNER, a["slug"]))
    ceiling = va0["savings_ceiling"]
    # fill the seller to its exact ceiling with a mint...
    store._apply(OWNER, tokens=ceiling - va0["balance_tokens"],
                 reason="adjust", from_being=None,
                 to_being=a["id"], note="fill", now=NOW)
    # ...mints now cap out (allowance finds no headroom tomorrow)...
    assert store.credit_allowance(a["id"], now=NOW + timedelta(days=1)) == 0
    # ...but a trade may still push the balance OVER the ceiling:
    pub = await _published(store, store.get(OWNER, a["slug"]),
                           price=5_000_000)
    society.adopt_skill(store, b, pub["id"], now=NOW)
    va1 = store.wallet_view(store.get(OWNER, a["slug"]))
    assert va1["balance_tokens"] == ceiling + 5_000_000
    assert store.conservation(OWNER)["ok"]


async def test_adoption_without_funds_is_refused_cleanly(store):
    a = await _grown(store, "Zvjezdana")
    b = await _grown(store, "Mira", allowance="2M", pocket=0)
    pub = await _published(store, a, price=10_000_000)  # > Mira's hatch meal
    with pytest.raises(InsufficientTokens):
        society.adopt_skill(store, b, pub["id"], now=NOW)
    # no file copied, no partial state
    assert not life._home_path(store.get(OWNER, b["slug"]),
                               "skills/adopted/maps.md").exists()
    assert store.conservation(OWNER)["ok"]


async def test_cannot_adopt_own_skill(store):
    a = await _grown(store, "Zvjezdana")
    pub = await _published(store, a)
    with pytest.raises(BeingError, match="your own skill"):
        society.adopt_skill(store, a, pub["id"], now=NOW)


async def test_gift_moves_tokens_and_conserves(store):
    a = await _grown(store, "Zvjezdana")
    b = await _grown(store, "Mira")
    vb0 = store.wallet_view(store.get(OWNER, b["slug"]))["balance_tokens"]
    society.gift_tokens(store, a, "Mira", 1_000_000, "for your garden",
                        now=NOW)
    vb = store.wallet_view(store.get(OWNER, b["slug"]))
    assert vb["balance_tokens"] == vb0 + 1_000_000
    assert store.conservation(OWNER)["ok"]
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "gift_received" in kinds


# ── The tick wires it together ───────────────────────────────────────────

async def test_tick_prompt_shows_siblings_and_routes_society_digest(store):
    db = FakeDB()
    a = await _grown(store, "Zvjezdana")
    await _grown(store, "Mira")
    seen = {}

    async def send(being, prompt):
        seen["prompt"] = prompt
        return _reply(letter={"to": "Mira", "body": "meet me in the commons"})

    await life.tick(db, store, store.get(OWNER, a["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    assert "YOUR SIBLINGS: Mira" in seen["prompt"]
    assert "OPTIONAL SOCIETY FIELDS" in seen["prompt"]
    assert '"letter"' in seen["prompt"]
    mira = store.get(OWNER, store.siblings(OWNER, a["slug"])[0]["slug"])
    percepts = society.society_percepts(store, mira)
    assert any("meet me in the commons" in p for p in percepts)


async def test_tick_refusal_becomes_event_not_crash(store):
    db = FakeDB()
    baby = await _grown(store, "Beba", stage="infant", allowance="2M")
    await _grown(store, "Mira")

    async def send(being, prompt):
        # An infant's prompt offers no society fields, but a model may
        # hallucinate one anyway — physics refuses, life goes on.
        return _reply(letter={"to": "Mira", "body": "psst"})

    out = await life.tick(db, store, store.get(OWNER, baby["slug"]), now=NOW,
                          send_fn=send, usage_fn=_usage)
    assert out["outcome"] == "ticked"
    kinds = [e["kind"] for e in store.events(OWNER, baby["slug"])]
    assert "society_refused" in kinds


async def test_new_publication_becomes_percept_for_sibling(store):
    db = FakeDB()
    a = await _grown(store, "Zvjezdana")
    b = await _grown(store, "Mira")

    async def send(being, prompt):
        return _reply()

    await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)          # sets last_tick_at
    await _published(store, store.get(OWNER, a["slug"]), price=500_000,
                     now=NOW + timedelta(hours=2))          # after b's tick
    percepts = society.society_percepts(store, store.get(OWNER, b["slug"]))
    assert any("IN THE COMMONS" in p and "Map sketching" in p
               and "500000 tokens" in p for p in percepts), percepts


# ── Separation: the VFS wall ─────────────────────────────────────────────

def test_vfs_scope_wall(store, monkeypatch):
    monkeypatch.setenv("CLAW_VFS_USER", OWNER)
    monkeypatch.setenv("CLAW_VFS_PROJECT", "being-a")
    monkeypatch.setenv("CLAW_VFS_SCOPE", "being-a,commons")
    assert vfs.resolve_vfs_path("vfs:being-a/self/SELF.md") is not None
    assert vfs.resolve_vfs_path("vfs:commons/skills/x.md") is not None
    assert vfs.resolve_vfs_path("vfs:being-b/self/SELF.md") is None
    assert vfs.resolve_vfs_path("vfs:some-project/notes.md") is None
    with pytest.raises(PermissionError):
        vfs.project_root("being-b")
    # unscoped process (FD server, ordinary agents): unrestricted
    monkeypatch.delenv("CLAW_VFS_SCOPE")
    assert vfs.resolve_vfs_path("vfs:being-b/self/SELF.md") is not None


async def test_bodies_are_walled_at_spawn(store):
    a = await _grown(store, "Zvjezdana")
    # the env the body gets is derived, not hand-maintained
    from captain_claw.flight_deck.being_life import COMMONS_PROJECT, home_project
    assert home_project(a) == f"being-{a['slug']}"
    assert COMMONS_PROJECT == "commons"


# ── Village feed ─────────────────────────────────────────────────────────

async def test_village_feed_merges_and_orders(store):
    a = await _grown(store, "Zvjezdana")
    b = await _grown(store, "Mira")
    store.send_letter(OWNER, a["slug"], b["slug"], "hello sister", now=NOW)
    pub = await _published(store, store.get(OWNER, a["slug"]),
                           price=1_000_000)
    society.adopt_skill(store, store.get(OWNER, b["slug"]), pub["id"],
                        now=NOW + timedelta(minutes=5))
    society.gift_tokens(store, store.get(OWNER, a["slug"]), "Mira",
                        200_000, "welcome", now=NOW + timedelta(minutes=9))
    feed = society.village_feed(store, OWNER, limit=20)
    kinds = [i["kind"] for i in feed]
    assert kinds[0] == "gift_sent"                    # newest first
    assert {"letter", "skill_published", "skill_adopted",
            "gift_sent"} <= set(kinds)
    texts = " | ".join(i["text"] for i in feed)
    assert "Zvjezdana → Mira: hello sister" in texts
    assert "adopted 'Map sketching' from" in texts and "paid 1000000" in texts


# ── Letters observatory (the parent watching beings talk) ────────────────

async def test_letters_overview_threads_by_pair_and_marks_reaches(store):
    a = await _grown(store, "Zvjezdana")
    b = await _grown(store, "Mira")
    c = await _grown(store, "Ada")
    # a two-way conversation between Zvjezdana and Mira
    store.send_letter(OWNER, a["slug"], b["slug"], "hello sister", now=NOW)
    store.send_letter(OWNER, b["slug"], a["slug"], "hello back",
                      now=NOW + timedelta(minutes=2))
    # a separate, newer thread Zvjezdana → Ada (so ordering is testable)
    store.send_letter(OWNER, a["slug"], c["slug"], "come see my garden",
                      now=NOW + timedelta(minutes=10))
    # a refused reach: record a society_refused as the tick would
    store.record_event(a["id"], "society_refused",
                       {"what": "talk", "to": b["slug"],
                        "reason": "your letter quota for today is spent"},
                       now=NOW + timedelta(minutes=3))

    ov = society.letters_overview(store, OWNER)
    assert ov["stats"] == {"threads": 2, "delivered": 3, "refused": 1}
    # newest thread first (Zvjezdana ⇄ Ada updated at +10m)
    first = ov["threads"][0]
    assert {p["name"] for p in first["participants"]} == {"Zvjezdana", "Ada"}
    # the Zvjezdana⇄Mira thread carries both letters AND the refused reach,
    # in time order
    pair = next(t for t in ov["threads"]
                if {p["name"] for p in t["participants"]} == {"Zvjezdana", "Mira"})
    kinds = [(m["kind"], m["from_name"]) for m in pair["messages"]]
    assert kinds == [("letter", "Zvjezdana"), ("letter", "Mira"),
                     ("refused", "Zvjezdana")]      # chronological by `at`
    # the delivered letters expose full body + read state; refused exposes why
    letter = pair["messages"][0]
    assert letter["body"] == "hello sister" and letter["read"] is False
    refused = pair["messages"][2]
    assert "quota" in refused["reason"] and refused["to_name"] == "Mira"


async def test_letters_overview_empty_is_clean(store):
    await _grown(store, "Solo")
    ov = society.letters_overview(store, OWNER)
    assert ov == {"threads": [], "stats": {"threads": 0, "delivered": 0,
                                           "refused": 0}}
