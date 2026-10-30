"""The Taskboard's opt-in inboxes (v2, decision #1): the user's Telegram
inbox, WhatsApp and Gmail — each off until they switch it on with consent
(and only once that channel is connected), then read for that user only:
people's messages past a deterministic gate (groups only when addressed to
the user, no broadcasts, no newsletters or automated mail, the user's own
rules) as ``message`` tasks waiting for Jev; the user's own messages as
``sent`` events or tasks (an agent's send never counts as theirs); text
kept 30 days. Switching an inbox off — or the inbox going away where it
lives — deletes what it put on the board, and a task with more on it keeps
the rest."""

from __future__ import annotations

import base64
import json
import sqlite3
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import (
    auth,
    fd_crypto,
    gmail_autoreply,
    google_oauth_routes,
    taskboard,
    taskboard_passive,
    taskboard_triage,
    telegram_inbox,
)
from captain_claw.flight_deck.taskboard_passive import KEPT_S, KEYS, switch_on
from captain_claw.flight_deck.taskboard_sources import run_sweep
from captain_claw.flight_deck.whatsapp_web.link_state import LinkState
from captain_claw.flight_deck.whatsapp_web.store import Store

from .test_taskboard_sources import env  # noqa: F401

PASSIVE_ROWS = ("telegram_inbox", "whatsapp_web", "gmail_inbox")
ME_WA = "385900000000@s.whatsapp.net"
ANA_WA = "385911111111@s.whatsapp.net"
IVO_WA = "385922222222@s.whatsapp.net"
GROUP_WA = "120363000000000001@g.us"
OWN_MAIL = "me.inbox@gmail.com"
SENDER = "alice@example.com"
AR_PASS = ("mx.google.com; dkim=pass header.i=@example.com header.s=s1 header.b=Ab3dEf9h; "
           "spf=pass (google.com: domain of alice@example.com designates 192.0.2.25 as "
           "permitted sender) smtp.mailfrom=alice@example.com; "
           "dmarc=pass (p=REJECT sp=REJECT dis=NONE) header.from=example.com")
MX_HOP = ("from mail.example.com (mail.example.com. [192.0.2.25]) by mx.google.com with ESMTPS "
          "id a640c23a62f3a; Tue, 13 Oct 2026 02:14:07 -0700 (PDT)")


@pytest.fixture()
def board(env, monkeypatch, tmp_path):  # noqa: F811
    """The sources' deck (two users, an empty board) with the inboxes' key
    for sealed text, and triage's kicks recorded instead of run."""
    monkeypatch.setenv("FD_DATA_KEY_FILE", str(tmp_path / "data_key"))
    fd_crypto.reset_for_tests()
    monkeypatch.setattr(auth, "get_db", lambda: env.db)
    monkeypatch.setattr(taskboard_passive, "_LOOP", None)
    kicked: list[str] = []
    monkeypatch.setattr(taskboard_triage, "kick", kicked.append)
    env.kicked = kicked
    yield env
    fd_crypto.reset_for_tests()


@pytest.fixture()
async def api(board, monkeypatch):
    monkeypatch.setattr(taskboard, "_SYNC_GAP_S", 0.0)
    app = FastAPI()
    app.include_router(taskboard.router)
    alice = {"id": board.alice, "email": "alice@x.co", "display_name": "Alice", "role": "user"}
    bob = {"id": board.bob, "email": "bob@x.co", "display_name": "Bob", "role": "user"}
    current = {"user": alice}
    app.dependency_overrides[auth.get_current_user] = lambda: current["user"]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://deck") as client:
        yield SimpleNamespace(client=client, env=board, alice=alice, bob=bob, be=lambda who: current.update(user=who))


async def _get(api, path):
    r = await api.client.get(path)
    assert r.status_code == 200, r.text
    return r.json()


async def _put(api, body):
    return await api.client.put("/fd/taskboard/settings", json=body)


def _all(deck, uid, **filters):
    return deck.board.query(uid, view="all", filters={k: [v] for k, v in filters.items()}, limit=200)["rows"]


def _task(deck, uid, key):
    tid = deck.board.task_for_key(uid, key)
    assert tid, f"no task for {key}"
    return deck.board.get_task(uid, tid)


def _events(deck, uid, task_id):
    return deck.board.events(uid, task_id)


def _parties(deck, uid, task_id):
    return deck.board.parties_for(uid, [task_id])[task_id]


# ── the channels, connected ───────────────────────────────────────────


async def _tg_inbox(deck, uid, *, tg="777", username="alice_tg", monkeypatch=None):
    """*uid*'s Telegram inbox on where it lives: linked to the deck's bot and
    keeping their chats."""
    await deck.db.create_telegram_link(uid, "", "", f"code-{uid}")
    await deck.db.update_telegram_link(uid, status="approved", tg_user_id=tg, tg_username=username)
    await deck.db.set_tg_inbox_settings(uid, enabled=True, consented_at=datetime.now(UTC).isoformat(),
                                       retention_days=30)


async def _tg(deck, uid, chat, mid, text, *, sender="555", name="Ana", outgoing=False, ago=60.0,
              kind="private", title="Ana", reply_to=None):
    at = int(time.time() - ago)
    await deck.db.upsert_tg_inbox_chat(uid, chat, kind, fd_crypto.seal(title), last_at=at)
    await deck.db.add_tg_inbox_message(
        uid, chat, mid, sender_id="777" if outgoing else sender, outgoing=outgoing, sent_at=at,
        body=fd_crypto.seal({"t": text, "s": "" if outgoing else name, "m": "", "f": ""}), reply_to=reply_to)
    return at


def _wa_link(deck, uid) -> Store:
    d = deck.tmp / "whatsapp_web" / uid
    d.mkdir(parents=True)
    LinkState(state="connected", jid=ME_WA, linked_at="2026-10-01T10:00:00+00:00").save(d / "state.json")
    return Store(d / "store.sqlite3")


def _wa(store: Store, mid, text, *, chat=ANA_WA, sender=None, from_me=False, ago=600.0, push="Ana", **extra):
    store.ingest([{"op": "message", "chat": chat, "id": mid,
                   "sender": sender if sender is not None else ("" if from_me else chat), "from_me": from_me,
                   "is_group": chat.endswith("@g.us"), "ts": int(time.time() - ago), "push_name": push,
                   "kind": "text", "text": text, "media": None, "reply_to": "", **extra}], live=True)


def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def _mail(mid, *, frm=f"Ana Kovač <{SENDER}>", to=OWN_MAIL, cc="", subject="Q3 invoice",
          body="Hi, could you send me the Q3 invoice by Friday?", labels=("INBOX", "UNREAD"), ago=600.0,
          thread="", headers=(), auth_ok=True) -> dict:
    """A Gmail API message, ``format=full``."""
    hs = [("Received", MX_HOP), ("Authentication-Results", AR_PASS)] if auth_ok else []
    hs += [("From", frm), ("To", to), *([("Cc", cc)] if cc else []), ("Subject", subject),
           ("Message-ID", f"<{mid}@example.com>"), *headers]
    return {"id": mid, "threadId": thread or f"th-{mid}", "labelIds": list(labels),
            "internalDate": str(int((time.time() - ago) * 1000)),
            "payload": {"mimeType": "text/plain", "headers": [{"name": n, "value": v} for n, v in hs],
                        "body": {"size": len(body), "data": _b64(body)}}}


def _sent_mail(mid, thread, *, to=f"Ana Kovač <{SENDER}>", body="Sure, here it is.", ago=1200.0) -> dict:
    return _mail(mid, frm=f"Alice <{OWN_MAIL}>", to=to, subject="Re: Q3 invoice", body=body, labels=("SENT",),
                 ago=ago, thread=thread, auth_ok=False)


def _gmail_box(monkeypatch, boxes: dict[str, list[dict]]):
    """Gmail as each user's own token reads it: their mailbox only."""
    asked: list[str] = []

    def read(uid, cur, seen):
        asked.append(uid)
        msgs = boxes.get(uid, [])
        known = seen([m["id"] for m in msgs])
        return {"own": [OWN_MAIL], "own_at": time.time(), "messages": [m for m in msgs if m["id"] not in known]}

    monkeypatch.setattr(taskboard_passive, "_gmail_read", read)
    return asked


def _connect_all(deck, monkeypatch, uid):
    """Every channel connected for *uid* (Telegram's bot, their linked
    WhatsApp, their Google with Gmail read)."""
    monkeypatch.setattr(telegram_inbox, "allowed", lambda: True)
    monkeypatch.setattr(telegram_inbox, "_tg_for_user", lambda u: "777" if u == uid else "")

    async def google(u):
        return u == uid

    async def scopes(db, owner):
        return []

    monkeypatch.setattr(google_oauth_routes, "is_google_connected", google)
    monkeypatch.setattr(gmail_autoreply, "missing_scopes", scopes)


# ── off until switched on, with consent, once connected ───────────────


async def test_every_inbox_is_off_by_default_and_reads_nothing(api, monkeypatch):
    deck = api.env
    out = await _get(api, "/fd/taskboard/settings")
    assert set(out["passive"]) == set(KEYS)
    assert all(set(v) == {"on", "consented_at", "available", "note"} for v in out["passive"].values())
    assert {k: (v["on"], v["consented_at"]) for k, v in out["passive"].items()} == {k: (False, None) for k in KEYS}
    rows = {s["key"]: s for s in (await _get(api, "/fd/taskboard/status"))["sources"]}
    assert not any(rows[k]["tracked"] or rows[k]["on"] for k in PASSIVE_ROWS)
    assert all(set(rows[k]) == {"key", "label", "tracked", "on", "note"} for k in PASSIVE_ROWS)
    # Messages wait in every inbox, and none reaches the board.
    await _tg_inbox(deck, deck.alice)
    await _tg(deck, deck.alice, "555", 1, "Can you send me the invoice?")
    _wa(_wa_link(deck, deck.alice), "W1", "Are we still on for Saturday?")
    asked = _gmail_box(monkeypatch, {deck.alice: [_mail("m1")]})
    res = run_sweep()
    assert not set(res["new_events"]) & {"telegram_inbox", "whatsapp_web", "gmail_inbox"}
    assert deck.board.query(deck.alice, view="all")["total"] == 0 and asked == []


async def test_an_inbox_needs_its_channel_and_the_users_consent(api, monkeypatch):
    deck = api.env
    out = await _get(api, "/fd/taskboard/settings")
    assert not any(v["available"] for v in out["passive"].values())
    assert "WhatsApp" in out["passive"]["whatsapp_web"]["note"] and "Google" in out["passive"]["gmail"]["note"]
    r = await _put(api, {"passive": {"whatsapp_web": {"on": True, "consent": True}}})
    assert r.status_code == 400 and "Link your WhatsApp" in r.json()["detail"]

    _connect_all(deck, monkeypatch, deck.alice)
    await _tg_inbox(deck, deck.alice)
    _wa_link(deck, deck.alice)
    out = await _get(api, "/fd/taskboard/settings")
    assert all(v["available"] for v in out["passive"].values()), out["passive"]
    for body in ({"telegram_inbox": {"on": True}}, {"telegram_inbox": {"on": True, "consent": False}},
                 {"telegram_inbox": {"on": "yes", "consent": True}}, {"slack": {"on": True, "consent": True}},
                 {}, {"gmail": {"on": True, "consent": True}, "whatsapp_web": {"on": True}}):
        r = await _put(api, {"passive": body})
        assert r.status_code == 400, body
    out = await _get(api, "/fd/taskboard/settings")
    assert not any(v["on"] for v in out["passive"].values())  # nothing moved, not even the consented one

    before = time.time()
    r = await _put(api, {"passive": {"telegram_inbox": {"on": True, "consent": True}}, "jev_per_day": 20})
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["deleted"] == 0 and got["settings"]["passive"] == got["passive"] and got["jev_per_day"] == 20
    switch = got["passive"]["telegram_inbox"]
    assert switch["on"] is True and before <= switch["consented_at"] <= time.time()
    rows = {s["key"]: s for s in (await _get(api, "/fd/taskboard/status"))["sources"]}
    assert rows["telegram_inbox"]["tracked"] and rows["telegram_inbox"]["on"]
    assert not rows["whatsapp_web"]["tracked"] and not rows["gmail_inbox"]["tracked"]
    # Another user's switches are their own.
    api.be(api.bob)
    out = await _get(api, "/fd/taskboard/settings")
    assert not out["passive"]["telegram_inbox"]["on"] and not out["passive"]["telegram_inbox"]["available"]


# ── Telegram ──────────────────────────────────────────────────────────


async def test_telegram_messages_reach_their_users_board_only(board):
    deck = board
    await _tg_inbox(deck, deck.alice)
    await _tg_inbox(deck, deck.bob, tg="888", username="bob_tg")
    switch_on(deck.alice, "telegram_inbox")
    at = await _tg(deck, deck.alice, "555", 1, "Can you send me the Q3 invoice?\nThanks!", ago=20 * 3600)
    await _tg(deck, deck.alice, "555", 2, "By Friday, please", ago=19.9 * 3600)
    await _tg(deck, deck.bob, "555", 1, "Bob's own chat with Ana", ago=60)  # Bob never switched it on
    res = run_sweep()
    assert res["new_events"].get("telegram_inbox") == 2 and "telegram_inbox" not in res["problems"]

    task = _task(deck, deck.alice, "tg_inbox:555")
    assert (task["kind"], task["direction"], task["channel"], task["source"]) == \
        ("message", "received", "tg_inbox", "telegram_inbox")
    # A maybe until Jev says it is a task — not in Needs me meanwhile; taken
    # to ask the assistant to act until Jev says otherwise.
    assert (task["status"], task["owes_next"], task["triage_state"], task["private"]) == ("maybe", "me", "pending", 1)
    assert taskboard_passive._json(task["flags"], {})["contains_instructions"] is True
    assert task["title"] == "Can you send me the Q3 invoice?" and task["ask"] == "Can you send me the Q3 invoice? Thanks!"
    assert abs(task["retention_until"] - (at + 3600 * 0.1 + KEPT_S)) < 5  # its newest message's 30 days
    events = _events(deck, deck.alice, task["id"])
    assert [e["kind"] for e in events] == ["received", "received"] and all(e["private"] for e in events)
    assert events[0]["actor"] == "Ana" and events[0]["refs"]["retention_until"] == pytest.approx(at + KEPT_S)
    [ana] = _parties(deck, deck.alice, task["id"])
    assert (ana["role"], ana["kind"], ana["address"], ana["display"], ana["verified"]) == \
        ("requester", "tg_id", "555", "Ana", True)
    assert deck.board.lane_of(deck.alice, task["id"]) != "needs_me" and deck.alice in deck.kicked
    assert deck.board.query(deck.bob, view="all")["total"] == 0

    # Read again: nothing new. Later in the day the chat is a new conversation.
    assert run_sweep()["new_events"].get("telegram_inbox") is None
    await _tg(deck, deck.alice, "555", 3, "Also, lunch tomorrow?", ago=3600)
    run_sweep()
    later = _task(deck, deck.alice, "tg_inbox:555")
    assert later["id"] != task["id"] and later["title"] == "Also, lunch tomorrow?"
    assert len(_events(deck, deck.alice, task["id"])) == 2  # the first conversation keeps its own


async def test_a_group_counts_only_when_it_is_addressed_to_the_user(board):
    deck = board
    await _tg_inbox(deck, deck.alice)
    switch_on(deck.alice, "telegram_inbox")
    group = dict(kind="group", title="Team", chat="-100123")
    chat = group.pop("chat")
    await _tg(deck, deck.alice, chat, 10, "Random chatter about lunch", sender="556", name="Ivo", ago=900, **group)
    await _tg(deck, deck.alice, chat, 11, "I can take the slides", outgoing=True, ago=800, **group)
    await _tg(deck, deck.alice, chat, 12, "Great — can you send them by noon?", sender="556", name="Ivo",
              ago=700, reply_to=11, **group)
    await _tg(deck, deck.alice, chat, 13, "@Alice_TG could you also check the numbers?", sender="557",
              name="Mia", ago=600, **group)
    run_sweep()
    [task] = _all(deck, deck.alice)
    assert task["id"] == deck.board.task_for_key(deck.alice, f"tg_inbox:{chat}")
    assert [e["summary"] for e in _events(deck, deck.alice, task["id"])] == [
        "Great — can you send them by noon?", "@Alice_TG could you also check the numbers?"]


async def test_the_users_own_messages_are_sent_events_or_sent_tasks(board):
    deck = board
    await _tg_inbox(deck, deck.alice)
    switch_on(deck.alice, "telegram_inbox")
    deck.board.add_rule(deck.alice, {"sender": "555"}, "always_task")
    await _tg(deck, deck.alice, "555", 1, "Can you call me today?", ago=7200)
    await _tg(deck, deck.alice, "555", 2, "Sure, at five", outgoing=True, ago=3600)
    await _tg(deck, deck.alice, "600", 1, "Don't forget the keys", outgoing=True, ago=1800, title="Ivo")
    run_sweep()
    asked = _task(deck, deck.alice, "tg_inbox:555")
    # Always a task by the user's rule: open, and once they answered it waits on them.
    assert (asked["kind"], asked["status"], asked["owes_next"], asked["evidence"]) == ("message", "open", "them", "verified")
    sent = [e for e in _events(deck, deck.alice, asked["id"]) if e["kind"] == "sent"]
    assert [(e["actor_kind"], e["evidence"], e["summary"]) for e in sent] == [("me", "recorded", "Sure, at five")]
    own = _task(deck, deck.alice, "tg_inbox:600")
    assert (own["kind"], own["direction"], own["status"], own["evidence"], own["closed_by"]) == \
        ("sent", "sent", "done", "verified", "evidence")
    assert own["title"] == "Don't forget the keys" and own["result_text"] == "You wrote to Ivo"
    [ivo] = _parties(deck, deck.alice, own["id"])
    assert (ivo["role"], ivo["kind"], ivo["address"], ivo["display"]) == ("recipient", "tg_id", "600", "Ivo")
    assert len(_all(deck, deck.alice)) == 2


# ── WhatsApp ──────────────────────────────────────────────────────────


async def test_whatsapp_people_groups_broadcasts_and_agents_sends(board):
    deck = board
    wa = _wa_link(deck, deck.alice)
    switch_on(deck.alice, "whatsapp_web")
    wa.put_contacts([{"jid": ANA_WA, "full_name": "Ana Horvat"}])
    _wa(wa, "W1", "Can you book the table for Saturday?", ago=3000)
    _wa(wa, "W2", "Weekly status update", chat="status@broadcast", sender=IVO_WA, ago=2900)
    _wa(wa, "G1", "Who has the projector?", chat=GROUP_WA, sender=IVO_WA, push="Ivo", ago=2800)
    _wa(wa, "G2", "@385900000000 can you bring it?", chat=GROUP_WA, sender=IVO_WA, push="Ivo", ago=2700)
    _wa(wa, "A1", "Booked for 8 pm — sent by my assistant", from_me=True, ago=2600)
    send = wa.start_send(ANA_WA, "helper", "Booked for 8 pm — sent by my assistant", now=time.time())
    wa.finish_send(send, "sent", wa_id="A1")
    _wa(wa, "O1", "Thanks, see you there", from_me=True, ago=2500)
    _wa(wa, "O2", "Running late", chat=IVO_WA, from_me=True, ago=2400)
    _wa(wa, "O3", "Just typed this", chat=IVO_WA, from_me=True, ago=10)  # settles first
    run_sweep()
    ana = _task(deck, deck.alice, f"waweb:{ANA_WA}")
    assert (ana["kind"], ana["channel"], ana["status"], ana["title"]) == \
        ("message", "waweb", "maybe", "Can you book the table for Saturday?")
    assert [(e["kind"], e["summary"]) for e in _events(deck, deck.alice, ana["id"])] == [
        ("received", "Can you book the table for Saturday?"), ("sent", "Thanks, see you there")]
    [who] = _parties(deck, deck.alice, ana["id"])
    assert (who["kind"], who["address"], who["display"]) == ("wa_jid", ANA_WA, "Ana Horvat")
    group = _task(deck, deck.alice, f"waweb:{GROUP_WA}")
    assert [e["summary"] for e in _events(deck, deck.alice, group["id"])] == ["@385900000000 can you bring it?"]
    assert deck.board.task_for_key(deck.alice, "waweb:status@broadcast") is None
    ivo = _task(deck, deck.alice, f"waweb:{IVO_WA}")
    assert (ivo["kind"], ivo["status"]) == ("sent", "done")
    assert [e["summary"] for e in _events(deck, deck.alice, ivo["id"])] == ["Running late"]
    # The message just typed comes once it settled.
    cur = taskboard_passive._cursor(deck.board, "whatsapp_web", deck.alice)
    conn = sqlite3.connect(deck.tmp / "whatsapp_web" / deck.alice / "store.sqlite3")
    conn.execute("UPDATE messages SET ts = ? WHERE id = 'O3'", (int(time.time() - 600),))
    conn.commit()
    conn.close()
    run_sweep()
    assert [e["summary"] for e in _events(deck, deck.alice, ivo["id"])] == ["Running late", "Just typed this"]
    assert taskboard_passive._cursor(deck.board, "whatsapp_web", deck.alice)["rowid"] > cur["rowid"]
    assert deck.board.query(deck.bob, view="all")["total"] == 0


# ── Gmail ─────────────────────────────────────────────────────────────


async def test_gmail_takes_people_not_newsletters_and_the_users_own_replies(board, monkeypatch):
    deck = board
    switch_on(deck.alice, "gmail")
    switch_on(deck.bob, "gmail")
    deck.board.add_rule(deck.alice, {"sender": "bo@example.com"}, "not_task")
    boxes = {deck.alice: [
        _mail("m1", cc="Mia <mia@example.com>", ago=3000),
        _mail("m2", frm="Shop <news@example.com>", subject="This week's deals",
              headers=(("List-Unsubscribe", "<mailto:u@example.com>"),)),
        _mail("m3", frm="Robot <noreply@example.com>", subject="Your receipt"),
        _mail("m4", frm="Bo <bo@example.com>", subject="Lunch?"),
        _mail("m5", to=", ".join(f"p{i}@example.com" for i in range(7)) + f", {OWN_MAIL}", subject="All hands"),
        _sent_mail("s1", "th-m1", ago=2000),
        _sent_mail("s2", "th-m1", body="And the Q4 one later.", ago=30),  # settles first
        _sent_mail("s3", "th-x", to="Shop <orders@example.com>", body="Order confirmed"),
    ], deck.bob: [_mail("b1", subject="For Bob only")]}
    sent = await deck.db.add_gmail_send(deck.alice, "helper", to_addrs="orders@example.com", subject="Order",
                                       gmail_message_id="s3", thread_id="th-x")
    conn = sqlite3.connect(deck.tmp / "flight-deck.db")
    conn.execute("UPDATE gmail_sends SET created_at = ? WHERE id = ?", ("2026-01-01T00:00:00+00:00", sent))
    conn.commit()
    conn.close()
    _gmail_box(monkeypatch, boxes)
    run_sweep()

    task = _task(deck, deck.alice, "gmail:thread:th-m1")
    assert (task["kind"], task["channel"], task["status"], task["source"], task["title"]) == \
        ("message", "gmail", "maybe", "gmail_inbox", "Q3 invoice")
    assert task["ask"] == "Hi, could you send me the Q3 invoice by Friday?"
    assert deck.board.task_for_key(deck.alice, "gmail:msg:m1") == task["id"]
    events = _events(deck, deck.alice, task["id"])
    assert [(e["kind"], e["actor_kind"]) for e in events] == [("received", "party"), ("sent", "me")]
    assert events[0]["summary"] == "Q3 invoice: Hi, could you send me the Q3 invoice by Friday?"
    people = {(p["role"], p["address"]): p for p in _parties(deck, deck.alice, task["id"])}
    assert set(people) == {("requester", SENDER), ("cc", "mia@example.com")}
    assert people[("requester", SENDER)]["verified"] and people[("requester", SENDER)]["display"] == "Ana Kovač"
    # The newsletter, the no-reply mail, the sender the user ruled out and the
    # crowd mail stay off; the agent's own send stays its own.
    titles = {t["title"] for t in _all(deck, deck.alice)}
    assert titles == {"Q3 invoice", "Order"}, titles
    assert _task(deck, deck.alice, "gmail:thread:th-x")["source"] == "gmail_sends"
    assert [t["title"] for t in _all(deck, deck.bob)] == ["For Bob only"]

    # Read again: nothing twice; the reply just sent comes once it settled.
    boxes[deck.alice][6] = _sent_mail("s2", "th-m1", body="And the Q4 one later.", ago=1000)
    run_sweep()
    assert [e["summary"] for e in _events(deck, deck.alice, task["id"])] == [
        "Q3 invoice: Hi, could you send me the Q3 invoice by Friday?", "Re: Q3 invoice: Sure, here it is.",
        "Re: Q3 invoice: And the Q4 one later."]


async def test_an_email_on_a_thread_the_board_knows_lands_on_that_task(board, monkeypatch):
    deck = board
    switch_on(deck.alice, "gmail")
    request, _ = deck.board.ingest(deck.alice, keys=["turn:t1"], extra_keys=["gmail:thread:th-r"],
                                  create={"title": "Ask Ana for the invoice", "kind": "request",
                                          "source": "agent_turns", "status": "done", "owes_next": "nobody"},
                                  event={"source": "agent_turns", "source_ref": "t1:open", "kind": "received"})
    _gmail_box(monkeypatch, {deck.alice: [_mail("r1", thread="th-r", subject="Re: invoice", body="Attached!")]})
    run_sweep()
    task = deck.board.get_task(deck.alice, request)
    assert task["title"] == "Ask Ana for the invoice"
    assert [e["kind"] for e in _events(deck, deck.alice, request)] == ["received", "received"]
    # Someone writing on a done request: triage asks whether it opens it again.
    flags = taskboard_passive._json(task["flags"], {})
    assert flags.get("new_since_done") is True and task["triage_state"] == "pending"
    assert len(_all(deck, deck.alice)) == 1


# ── off deletes what it made; the source going away too ───────────────


async def test_switching_off_deletes_its_tasks_and_mixed_ones_keep_the_rest(api, monkeypatch):
    deck = api.env
    _connect_all(deck, monkeypatch, deck.alice)
    r = await _put(api, {"passive": {"gmail": {"on": True, "consent": True}}})
    assert r.status_code == 200, r.text
    request, _ = deck.board.ingest(deck.alice, keys=["turn:t1"], extra_keys=["gmail:thread:th-r"],
                                  create={"title": "Ask Ana for the invoice", "kind": "request",
                                          "source": "agent_turns", "status": "open", "owes_next": "them"},
                                  event={"source": "agent_turns", "source_ref": "t1:open", "kind": "received"})
    _gmail_box(monkeypatch, {deck.alice: [_mail("m1", thread="th-1"), _mail("m2", thread="th-2", subject="Lunch"),
                                         _mail("r1", thread="th-r", subject="Re: invoice", body="Attached!")]})
    run_sweep()
    mixed = deck.board.task_for_key(deck.alice, "gmail:thread:th-1")
    alone = deck.board.task_for_key(deck.alice, "gmail:thread:th-2")
    # An agent then emails on the first thread (FD's own record of it).
    sent = await deck.db.add_gmail_send(deck.alice, "helper", to_addrs="Ana <ana@x.co>", subject="Re: Q3 invoice",
                                       gmail_message_id="g-1", thread_id="th-1")
    conn = sqlite3.connect(deck.tmp / "flight-deck.db")
    conn.execute("UPDATE gmail_sends SET created_at = ? WHERE id = ?", ("2026-01-01T00:00:00+00:00", sent))
    conn.commit()
    conn.close()
    run_sweep()
    assert sorted(e["source"] for e in _events(deck, deck.alice, mixed)) == ["gmail_inbox", "gmail_sends"]
    assert {p["address"] for p in _parties(deck, deck.alice, request)} >= {SENDER}

    r = await _put(api, {"passive": {"gmail": {"on": False}}})
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] == 1 and r.json()["settings"]["passive"]["gmail"] == {
        **r.json()["settings"]["passive"]["gmail"], "on": False, "consented_at": None}
    assert deck.board.get_task(deck.alice, alone) is None
    # The task an agent's email is on keeps it — and nothing the inbox said.
    kept = deck.board.get_task(deck.alice, mixed)
    assert [e["source"] for e in _events(deck, deck.alice, mixed)] == ["gmail_sends"]
    assert kept["ask"] == "" and kept["retention_until"] is None and "invoice by Friday" not in kept["title"]
    assert {p["address"] for p in _parties(deck, deck.alice, mixed)} == {"ana@x.co"}
    assert deck.board.task_for_key(deck.alice, "gmail:msg:m1") is None
    assert "friday" not in kept["search_text"]
    # The agent's request keeps all of its own, and loses the inbox's message and sender.
    own = deck.board.get_task(deck.alice, request)
    assert own["title"] == "Ask Ana for the invoice"
    assert [e["source"] for e in _events(deck, deck.alice, request)] == ["agent_turns"]
    assert SENDER not in {p["address"] for p in _parties(deck, deck.alice, request)}
    assert deck.board.task_for_key(deck.alice, "gmail:thread:th-r") == request
    # Off, nothing more is read.
    run_sweep()
    assert deck.board.task_for_key(deck.alice, "gmail:thread:th-2") is None


async def test_the_inbox_going_away_where_it_lives_takes_its_tasks(board):
    deck = board
    await _tg_inbox(deck, deck.alice)
    switch_on(deck.alice, "telegram_inbox")
    await _tg(deck, deck.alice, "555", 1, "Can you send me the invoice?")
    wa = _wa_link(deck, deck.alice)
    switch_on(deck.alice, "whatsapp_web")
    _wa(wa, "W1", "Are we still on for Saturday?")
    run_sweep()
    assert len(_all(deck, deck.alice)) == 2
    # The Telegram inbox switched off (it deleted what it kept), WhatsApp unlinked.
    await deck.db.set_tg_inbox_settings(deck.alice, enabled=False, consented_at=None, retention_days=30)
    wa.close()
    (deck.tmp / "whatsapp_web" / deck.alice / "store.sqlite3").unlink()
    run_sweep()
    assert deck.board.query(deck.alice, view="all")["total"] == 0
    now = taskboard_passive.switches(deck.board, deck.alice)
    assert not now["telegram_inbox"]["on"] and not now["whatsapp_web"]["on"]


# ── retention ─────────────────────────────────────────────────────────


async def test_text_goes_after_30_days_and_what_the_user_did_stays(board, monkeypatch):
    deck = board
    await _tg_inbox(deck, deck.alice)
    switch_on(deck.alice, "telegram_inbox")
    taskboard_passive._save_cursor(deck.board, "telegram_inbox", deck.alice, {"since": time.time() - 40 * 86400})
    await _tg(deck, deck.alice, "555", 1, "Old question", ago=29.5 * 86400)
    await _tg(deck, deck.alice, "600", 1, "Another old one", ago=29.6 * 86400, sender="600", title="Ivo")
    await _tg(deck, deck.alice, "700", 1, "Fresh one", ago=60, sender="700", title="Mia")
    await _tg(deck, deck.alice, "800", 1, "Long gone", ago=31 * 86400, sender="800", title="Leo")
    run_sweep()
    assert deck.board.task_for_key(deck.alice, "tg_inbox:800") is None  # past its 30 days already
    renamed = deck.board.task_for_key(deck.alice, "tg_inbox:600")
    deck.board.correct(deck.alice, renamed, "rename", "Call Ivo back")
    real = time.time
    monkeypatch.setattr(taskboard_passive, "time", SimpleNamespace(time=lambda: real() + 86400))
    run_sweep()
    assert deck.board.task_for_key(deck.alice, "tg_inbox:555") is None
    assert deck.board.task_for_key(deck.alice, "tg_inbox:700")
    kept = deck.board.get_task(deck.alice, renamed)
    assert kept["title"] == "Call Ivo back" and kept["ask"] == ""
    assert [e["kind"] for e in _events(deck, deck.alice, renamed)] == ["correction"]
    assert _parties(deck, deck.alice, renamed) == []


async def test_an_inbox_kept_fewer_days_keeps_its_text_on_the_board_no_longer(board):
    deck = board
    await _tg_inbox(deck, deck.alice)
    await deck.db.set_tg_inbox_settings(deck.alice, enabled=True, consented_at=datetime.now(UTC).isoformat(),
                                        retention_days=7)
    switch_on(deck.alice, "telegram_inbox")
    taskboard_passive._save_cursor(deck.board, "telegram_inbox", deck.alice, {"since": time.time() - 20 * 86400})
    at = await _tg(deck, deck.alice, "555", 1, "Six days old", ago=6 * 86400)
    await _tg(deck, deck.alice, "600", 1, "Eight days old", ago=8 * 86400, sender="600", title="Ivo")
    run_sweep()
    task = _task(deck, deck.alice, "tg_inbox:555")
    assert task["retention_until"] == pytest.approx(at + 7 * 86400)
    assert deck.board.task_for_key(deck.alice, "tg_inbox:600") is None


# ── an inbox's text, wherever a merge or a split took it ──────────────

ME = "user-me"
WA = "whatsapp_web"


def _store(tmp_path):
    from captain_claw.flight_deck.taskboard_store import TaskboardStore

    return TaskboardStore(tmp_path / "taskboard.db")


def _inbox_msg(s, conv, ref, text, *, at=None, until=None, number="38591"):
    """Someone's WhatsApp in the user's own inbox, as taskboard_passive writes it."""
    at = time.time() - 60 if at is None else at
    until = at + KEPT_S if until is None else until

    def write():
        return s.ingest(ME, keys=[conv], create={
            "title": text, "ask": text, "kind": "message", "direction": "received", "channel": "waweb",
            "status": "maybe", "owes_next": "me", "source": WA, "private": 1, "triage_state": "pending",
            "triage_after": at}, event={
            "source": WA, "source_ref": f"{ME}:{ref}", "at": at, "kind": "received", "channel": "waweb",
            "actor_kind": "party", "actor": "Ana Kovač", "summary": text, "private": True,
            "refs": {"retention_until": until}}, parties=[
            {"role": "requester", "kind": "wa_jid", "address": f"{number}@s.whatsapp.net", "display": "Ana Kovač",
             "verified": True}])
    return s.passive_write(ME, WA, [conv], write, retention_until=until)[0]


def _agent_request(s, key, title="Book the table", *, number="38591", at=None):
    """Ana's request on the agent's own WhatsApp (no inbox's)."""
    at = time.time() if at is None else at
    return s.ingest(ME, keys=[key], parties=[
        {"role": "requester", "kind": "waid", "address": number, "display": "Ana", "verified": True}],
        create={"title": title, "ask": title, "kind": "request", "direction": "received", "channel": "whatsapp",
                "status": "open", "owes_next": "agent", "triage_state": "ok", "source": "agent_turns",
                "source_ref": key, "agent_slug": "helper", "created_at": at},
        event={"source": "agent_turns", "source_ref": f"{key}:open", "at": at, "kind": "received",
               "channel": "whatsapp", "actor_kind": "party", "actor": "Ana", "summary": title})[0]


def _kept_anywhere(s, text: str) -> list[str]:
    """Every column on the board still holding *text*."""
    places = {"tasks": ("title", "ask", "result_text", "flags", "search_text", "next_steps"),
              "task_events": ("summary", "refs"), "task_decisions": ("applied", "why"),
              "parties": ("display_name", "folded_name"), "task_parties": ("display",)}
    return [f"{t}.{c}" for t, cols in places.items() for c in cols
            if s._conn.execute(f"SELECT 1 FROM {t} WHERE {c} LIKE ?", (f"%{text}%",)).fetchone()]


@pytest.mark.parametrize("how", ["switched_off", "expired"])
def test_an_inbox_task_merged_into_another_goes_with_its_text(tmp_path, how):
    s = _store(tmp_path)
    then = time.time() - 40 * 86400
    inbox = _inbox_msg(s, "waweb:ana", "m1", "SECRETA dinner plans", at=then if how == "expired" else None)
    request = _agent_request(s, "turn:r1")
    s.merge(ME, inbox, request)
    assert s.merged_from(ME, request) == [{"id": inbox, "title": "SECRETA dinner plans"}]
    gone = s.passive_forget(ME, WA) if how == "switched_off" else s.passive_purge(WA, now=time.time(), kept_s=KEPT_S)
    assert gone == 1 and s.get_task(ME, inbox) is None
    assert s.merged_from(ME, request) == [] and _kept_anywhere(s, "SECRETA") == []
    # The request keeps its own, and the user's merge note says what happened, not what it said.
    assert [e["summary"] for e in s.events(ME, request)] == ["Book the table", "Merged another task into this one"]
    # Ana is called what the request's own record calls her, not the inbox's contact name.
    (row,) = s.parties_for(ME, [request])[request]
    assert row["name"] == "Ana" and _kept_anywhere(s, "Kova") == []


def test_an_inbox_task_merged_elsewhere_stays_while_its_text_is_kept(tmp_path):
    s = _store(tmp_path)
    inbox = _inbox_msg(s, "waweb:ana", "m1", "Still fresh")
    request = _agent_request(s, "turn:r1")
    s.merge(ME, inbox, request)
    assert s.passive_purge(WA, now=time.time(), kept_s=KEPT_S) == 0
    assert s.get_task(ME, inbox)["title"] == "Still fresh"
    s.unmerge(ME, inbox)
    assert [e["summary"] for e in s.events(ME, inbox) if e["kind"] == "received"] == ["Still fresh"]


@pytest.mark.parametrize("how", ["switched_off", "expired"])
def test_a_message_split_off_an_inbox_task_is_the_inboxs_and_keeps_none_of_its_text(tmp_path, how):
    s = _store(tmp_path)
    then = time.time() - 40 * 86400 if how == "expired" else time.time() - 120
    first = _inbox_msg(s, "waweb:ana", "m1", "SECRETB first", at=then)
    _inbox_msg(s, "waweb:ana", "m2", "SECRETC second", at=then + 5)
    *_, last = [e["id"] for e in s.events(ME, first)]
    _rest, new = s.split(ME, first, [last])
    assert (new["source"], new["retention_until"]) == (WA, pytest.approx(then + 5 + KEPT_S))
    s.passive_forget(ME, WA) if how == "switched_off" else s.passive_purge(WA, now=time.time(), kept_s=KEPT_S)
    assert _kept_anywhere(s, "SECRETB") == [] and _kept_anywhere(s, "SECRETC") == []
    # What the user did stays, told plainly: the split, on both sides.
    assert s.get_task(ME, first)["title"] == "Split some of it off as a task of its own"
    assert s.get_task(ME, new["id"])["title"] == "Split off another task"


def test_a_rename_undone_after_the_text_went_brings_none_of_it_back(tmp_path):
    s = _store(tmp_path)
    task = _inbox_msg(s, "waweb:ana", "m1", "SECRETD the invoice")
    s.correct(ME, task, "rename", "Call Ana back")
    s.passive_forget(ME, WA)
    assert s.get_task(ME, task)["title"] == "Call Ana back"
    s.correct(ME, task, "undo")
    assert "SECRETD" not in s.get_task(ME, task)["title"] and _kept_anywhere(s, "SECRETD") == []


def test_jev_is_never_asked_whether_a_task_is_an_inbox_one(tmp_path):
    """A "Same as …?" would copy the inbox's title onto another task, where it
    outlives the inbox's 30 days and its switch-off."""
    s = _store(tmp_path)
    now = time.time()
    _inbox_msg(s, "waweb:ana", "m1", "Invoice for the March kitchen project", at=now - 3600)
    older = _agent_request(s, "turn:r0", "Kitchen invoice for March", at=now - 1800)
    newer = _agent_request(s, "turn:r1", "March kitchen invoice", at=now)
    found = s.same_task_candidates(ME, s.get_task(ME, newer), since=now - 7 * 86400)
    assert [t["id"] for t in found] == [older]


def test_a_same_as_naming_a_task_gone_goes_with_it(tmp_path):
    s = _store(tmp_path)
    older = _agent_request(s, "turn:r0", "Kitchen invoice")
    newer = _agent_request(s, "turn:r1", "Kitchen invoice again")
    s.update_fields(ME, newer, {"flags": {"same_as": {"task_id": older, "title": "Kitchen invoice", "p": 0.9}}})
    with s._lock:
        s._drop_tasks([older])
        s._conn.commit()
    assert "same_as" not in json.loads(s.get_task(ME, newer)["flags"])


def test_unmerging_a_message_still_waiting_for_triage_puts_it_back_in_the_queue(tmp_path):
    s = _store(tmp_path)
    waiting = _inbox_msg(s, "waweb:ana", "m1", "Are we on for Saturday?")
    other = _inbox_msg(s, "waweb:ivo", "m2", "Lunch?", number="38592")
    s.update_fields(ME, other, {"triage_state": "ok", "triage_after": None, "extracted_fp": "x"})
    s.merge(ME, waiting, other)
    target = s.get_task(ME, other)   # what it has to go by changed: read again
    assert (target["triage_state"], target["extracted_fp"]) == ("pending", "")
    s.unmerge(ME, waiting)
    back = s.get_task(ME, waiting)
    assert back["triage_state"] == "pending" and back["triage_after"] is not None
    assert waiting in [t["id"] for t in s.triage_queue(ME, now=time.time() + 1)]


# ── Gmail: every email read once, however many came ───────────────────


def _gmail_api(monkeypatch, box: list[dict]) -> list[str]:
    """Gmail as its API answers the user's own token: ``messages.list`` with
    ``after:`` / ``before:`` (epoch seconds), newest first, a page at a time
    behind ``nextPageToken``; every email it fetched, in order."""
    import asyncio
    import concurrent.futures
    import re

    fetched: list[str] = []

    async def token(uid):
        return "tok"

    async def get(client, tok, path, params=None):
        params = params or {}
        if path == "/messages":
            q = params["q"]
            after = int(re.search(r"after:(\d+)", q).group(1))
            before = re.search(r"before:(\d+)", q)
            match = sorted((m for m in box if int(m["internalDate"]) // 1000 > after
                            and (before is None or int(m["internalDate"]) // 1000 < int(before.group(1)))),
                           key=lambda m: -int(m["internalDate"]))
            start, n = int(params.get("pageToken") or 0), int(params["maxResults"])
            out: dict = {"messages": [{"id": m["id"], "threadId": m["threadId"]} for m in match[start:start + n]]}
            if start + n < len(match):
                out["nextPageToken"] = str(start + n)
            return 200, out
        mid = path.rsplit("/", 1)[1]
        if (params or {}).get("format") == "full":
            fetched.append(mid)
        return 200, next(m for m in box if m["id"] == mid)

    async def mailbox(client, tok):
        return gmail_autoreply._Mailbox(OWN_MAIL, "1", [OWN_MAIL], True)

    def on_loop(coro, timeout):
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout)

    monkeypatch.setattr(google_oauth_routes, "get_valid_google_access_token", token)
    monkeypatch.setattr(gmail_autoreply, "_gmail_get", get)
    monkeypatch.setattr(gmail_autoreply, "_mailbox", mailbox)
    monkeypatch.setattr(taskboard_passive, "_on_loop", on_loop)
    monkeypatch.setattr(taskboard_passive, "_GMAIL_LIST_MAX", 100)   # 120 is more than a list holds
    monkeypatch.setattr(taskboard_passive, "_GMAIL_PAGE", 40)
    return fetched


def _on_board(deck, ids) -> set[str]:
    return {i for i in ids if deck.board.task_for_key(deck.alice, f"gmail:msg:{i}")}


async def test_gmail_reads_a_backlog_past_50_and_past_one_list_oldest_first(board, monkeypatch):
    deck = board
    switch_on(deck.alice, "gmail")
    # The first look's day: 120 emails from people, more than one list (100) holds.
    box = [_mail(f"m{i:03d}", thread=f"th-{i:03d}", subject=f"Ask {i}", ago=600 + i * 600) for i in range(120)]
    fetched = _gmail_api(monkeypatch, box)
    ids = [m["id"] for m in box]
    run_sweep()
    # The oldest 50 of what was listed first: the newest 100's older half.
    assert len(fetched) == 50 and set(fetched) == {f"m{i:03d}" for i in range(50, 100)}
    assert taskboard_passive._cursor(deck.board, "gmail", deck.alice).get("before_ms")
    for _ in range(4):
        run_sweep()
    assert _on_board(deck, ids) == set(ids)
    assert len(fetched) == len(set(fetched)) == 120   # each read once
    assert "before_ms" not in taskboard_passive._cursor(deck.board, "gmail", deck.alice)


async def test_gmail_a_burst_between_passes_is_read_in_full(board, monkeypatch):
    deck = board
    switch_on(deck.alice, "gmail")
    box = [_mail("old1", thread="th-old1", ago=7200)]
    fetched = _gmail_api(monkeypatch, box)
    run_sweep()
    burst = [_mail(f"n{i:03d}", thread=f"th-n{i:03d}", subject=f"New {i}", ago=60 + i * 30) for i in range(80)]
    box += burst
    for _ in range(2):
        run_sweep()
    assert _on_board(deck, [m["id"] for m in burst]) == {m["id"] for m in burst}
    assert len(fetched) == len(set(fetched)) == 81
