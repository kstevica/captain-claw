"""Roadmap T2.8-10 — pen-pals across villages, the games shelf, the naming
rite. Delivery is real or refused loudly; names change once in a life; play
rides the letters physics unchanged."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_federation as federation
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck import being_society as society
from captain_claw.flight_deck.beings import BeingError, BeingsStore

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
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


async def _being(store, name="Zvjezdana", stage="child", public=False):
    b = store.conceive(OWNER, name, preset="explorer",
                       allowance_preset="5M", now=NOW)
    store.hatch(OWNER, b["slug"], now=NOW)
    if stage != "infant":
        store.set_stage(OWNER, b["slug"], stage, now=NOW)
    bb = store.get(OWNER, b["slug"])
    store.set_agent(bb["id"], bb["slug"], 1234, "tok")
    if public:
        store.set_public(OWNER, b["slug"], True, now=NOW)
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


# ═══ The games shelf (T2.9) ══════════════════════════════════════════════

async def test_games_shelf_exists_and_note_is_gated(store):
    b = await _being(store)
    society.ensure_commons(OWNER)    # what birth() runs for every hatchling
    shelf = society._commons_path(OWNER, "games/riddle-chain.md")
    assert shelf.exists()
    assert society._commons_path(OWNER, "games/README.md").exists()
    sibs = [{"id": "x", "slug": "iskra-lada-1", "name": "Lada",
             "stage": "child", "mood": ""}]
    # explorer preset has PLA 7 → whimsy 0.7 ≥ 0.5; note fires on tick%5==2
    store._update(b["id"], NOW, tick_count=2)
    note = society.games_note(store.get(OWNER, b["slug"]), sibs, 3)
    assert note and "games" in note and "riddle" in note
    # …and stays quiet off-cadence, without siblings, or with quota spent
    store._update(b["id"], NOW, tick_count=3)
    assert society.games_note(store.get(OWNER, b["slug"]), sibs, 3) is None
    store._update(b["id"], NOW, tick_count=2)
    assert society.games_note(store.get(OWNER, b["slug"]), None, 3) is None
    assert society.games_note(store.get(OWNER, b["slug"]), sibs, 0) is None


# ═══ The naming rite (T2.10) ═════════════════════════════════════════════

async def test_chosen_name_full_rite(store):
    b = await _being(store, name="Prva", stage="adolescent")
    # a child cannot yet
    child = await _being(store, name="Mala", stage="child")
    with pytest.raises(BeingError, match="adolescence"):
        life._propose_name(store, child, {"name": "Iskrica", "why": "w"}, NOW)
    # the adolescent proposes
    life._propose_name(store, b, {"name": "Zora", "why": "dawn work"}, NOW)
    bb = store.get(OWNER, b["slug"])
    assert bb["pending_name"]["name"] == "Zora"
    with pytest.raises(BeingError, match="awaits your parent"):
        life._propose_name(store, bb, {"name": "Other", "why": "w"}, NOW)
    # the offer line flips to patience while pending
    lines = "\n".join(life.rare_option_lines(bb))
    assert "awaits your parent's blessing" in lines
    # the parent blesses: display name changes, slug doesn't, genome remembers
    old_slug = bb["slug"]
    blessed = store.approve_name(OWNER, b["slug"], now=NOW)
    assert blessed["name"] == "Zora" and blessed["slug"] == old_slug
    assert blessed["genome"]["epigenetics"]["chosen_name"] == "Zora"
    names = [e["data"].get("name") for e in store.events(OWNER, b["slug"])
             if e["kind"] == "milestone"]
    assert "chose_name" in [e["data"].get("name") for e in
                            store.events(OWNER, b["slug"])
                            if e["kind"] == "milestone"] or "Zora" in names
    # the being hears it as a percept, once
    percepts = life.percepts_since(store, store.get(OWNER, b["slug"]))
    assert any("BLESSED YOUR CHOSEN NAME" in p and "Zora" in p
               for p in percepts)
    # once in a life — a second choice is refused, and the offer is gone
    with pytest.raises(BeingError, match="yours for life"):
        life._propose_name(store, store.get(OWNER, b["slug"]),
                           {"name": "Treca", "why": "w"}, NOW)
    assert "naming rite" not in "\n".join(
        life.rare_option_lines(store.get(OWNER, b["slug"])))


async def test_chosen_name_rejection_is_heard(store):
    b = await _being(store, name="Druga", stage="adult")
    life._propose_name(store, b, {"name": "Vjetar", "why": "w"}, NOW)
    store.reject_name(OWNER, b["slug"], "not yet", now=NOW)
    bb = store.get(OWNER, b["slug"])
    assert bb["pending_name"] is None and bb["name"] == "Druga"
    assert any("said not yet to your chosen name" in p
               for p in life.percepts_since(store, bb))


async def test_chosen_name_routes_through_the_tick(store):
    db = FakeDB()
    b = await _being(store, name="Tiha", stage="adolescent")

    async def send(being, prompt):
        return _reply(chosen_name={"name": "Jeka", "why": "echoes stay"})

    await life.tick(db, store, b, now=NOW + timedelta(hours=1),
                    send_fn=send, usage_fn=_usage)
    bb = store.get(OWNER, b["slug"])
    assert bb["pending_name"] and bb["pending_name"]["name"] == "Jeka"


# ═══ Pen-pals (T2.8) ═════════════════════════════════════════════════════

async def test_penpal_host_role_delivers_and_meters(store, monkeypatch):
    b = await _being(store, stage="child", public=True)
    store.upsert_visitor(OWNER, "https://far.example", "iskra-tudja-1",
                         "Tudja", {})
    v = store.visitors_for(OWNER)[0]
    sent = []
    monkeypatch.setattr(federation, "is_linked", lambda vid: True)

    async def fake_send(store_, vid, frm, village, body):
        sent.append({"vid": vid, "frm": frm, "body": body})
    monkeypatch.setattr(federation, "send_letter_to_visitor", fake_send)

    await life._deliver_penpal(store, store.get(OWNER, b["slug"]),
                               {"to": "Tudja", "body": "hello, far one"}, NOW)
    assert sent and sent[0]["vid"] == v["id"] and sent[0]["frm"] == b["name"]
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "penpal_sent" in kinds
    assert store.penpals_sent_today(b["id"], NOW) == 1
    assert any(e["data"].get("name") == "first_penpal"
               for e in store.events(OWNER, b["slug"])
               if e["kind"] == "milestone")


async def test_penpal_needs_an_open_door_and_a_reachable_name(store,
                                                              monkeypatch):
    # not public, not visiting → the door is closed
    shut = await _being(store, name="Zatvorena", stage="child")
    with pytest.raises(BeingError, match="open a door"):
        await life._deliver_penpal(store, shut, {"to": "X", "body": "hi"}, NOW)
    # public but nobody of that name is linked
    b = await _being(store, name="Otvorena", stage="child", public=True)
    with pytest.raises(BeingError, match="within reach"):
        await life._deliver_penpal(store, store.get(OWNER, b["slug"]),
                                   {"to": "Nitko", "body": "hi"}, NOW)
    # an infant has no letters at all
    baby = await _being(store, name="Bebica", stage="infant", public=True)
    with pytest.raises(BeingError, match="cannot send letters"):
        await life._deliver_penpal(store, baby, {"to": "X", "body": "hi"}, NOW)


async def test_penpal_shares_the_letter_quota(store, monkeypatch):
    b = await _being(store, stage="child", public=True)   # child: 3/day
    store.upsert_visitor(OWNER, "https://far.example", "iskra-tudja-1",
                         "Tudja", {})
    monkeypatch.setattr(federation, "is_linked", lambda vid: True)

    async def fake_send(*a, **k):
        pass
    monkeypatch.setattr(federation, "send_letter_to_visitor", fake_send)
    for i in range(3):
        await life._deliver_penpal(store, store.get(OWNER, b["slug"]),
                                   {"to": "Tudja", "body": f"n{i}"}, NOW)
    with pytest.raises(BeingError, match="limit"):
        await life._deliver_penpal(store, store.get(OWNER, b["slug"]),
                                   {"to": "Tudja", "body": "one more"}, NOW)


async def test_penpal_letter_arrives_as_a_percept_once(store):
    b = await _being(store, stage="child")
    # what the far side's link op does on delivery
    out = await federation.handle_link_request(
        store, OWNER, b["slug"], "letter",
        {"frm": "Tudja", "village": "Far Hollow", "body": "greetings"})
    assert out["delivered"] is True
    percepts = life.percepts_since(store, store.get(OWNER, b["slug"]))
    afar = [p for p in percepts if "A LETTER FROM AFAR" in p]
    assert afar and "Tudja" in afar[0] and "Far Hollow" in afar[0]
    assert '"penpal"' in afar[0]                 # the reply affordance


async def test_penpal_sender_role_uses_the_village_link(store, monkeypatch):
    b = await _being(store, stage="child")
    store.set_being_visit(OWNER, b["slug"], "https://far.example", "sec")
    sent = []
    monkeypatch.setattr(federation.village_client, "is_up", lambda s: True)

    async def fake_send(slug, to, frm, body):
        sent.append({"slug": slug, "to": to, "body": body})
    monkeypatch.setattr(federation.village_client, "send_letter", fake_send)
    await life._deliver_penpal(store, store.get(OWNER, b["slug"]),
                               {"to": "Domacin", "body": "hello host"}, NOW)
    assert sent and sent[0]["to"] == "Domacin"
    assert "penpal_sent" in [e["kind"] for e in store.events(OWNER, b["slug"])]


async def test_refused_penpal_is_loud_through_the_tick(store):
    db = FakeDB()
    b = await _being(store, stage="child")     # door closed: not public

    async def send(being, prompt):
        return _reply(penpal={"to": "Tudja", "body": "hi"})

    await life.tick(db, store, b, now=NOW + timedelta(hours=1),
                    send_fn=send, usage_fn=_usage)
    refusals = [e["data"] for e in store.events(OWNER, b["slug"])
                if e["kind"] == "society_refused"]
    assert refusals and refusals[0]["what"] == "penpal"
