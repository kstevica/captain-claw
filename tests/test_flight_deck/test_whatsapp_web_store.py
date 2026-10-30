"""WhatsApp Web, Phase 1 (docs/whatsapp-web-plan.md): what's kept, and agents reading it.

* parse — every message kind from its real proto (neonize's, when installed):
  text, replies, captions, voice (``PTT``), documents, reactions, edits,
  revokes, wrappers; sender-key-only events are skipped; live timestamps are
  milliseconds, history seconds; history chunks carry chats, push names and
  LID mappings;
* store — one row per (chat, id) however often it arrives; edits, deletions and
  reactions; LID chats folded into their phone-number chat; nothing readable
  at rest but JIDs; names from the address book first; search; the news line
  counts only live arrivals; retention;
* engine — history and live messages reach the store; the phone unlinking takes
  them from the link (test_whatsapp_web_kept.py: where they go), a disconnect
  told to deletes them; retention through the engine;
* the routes — an agent reads only its own owner's WhatsApp, its look moves its
  news cursor, and nothing is readable on a deck without sign-in.

WhatsApp is never contacted.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from captain_claw.flight_deck import auth, fd_crypto
from captain_claw.flight_deck import whatsapp_web_routes as routes
from captain_claw.flight_deck import whatsapp_web_supervisor as sup
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.flight_deck.whatsapp_web import parse
from captain_claw.flight_deck.whatsapp_web.api import build_app
from captain_claw.flight_deck.whatsapp_web.engine import Engine, LinkError
from captain_claw.flight_deck.whatsapp_web.store import ChatNotFound, Store

ANA = "385911111111@s.whatsapp.net"
IVO = "385922222222@s.whatsapp.net"
ANA_LID = "111111111111111@lid"
GROUP = "120363000000000001@g.us"
ALICE, BOB = "user-alice", "user-bob"


@pytest.fixture(autouse=True)
def data_key(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_DATA_KEY_FILE", str(tmp_path / "data_key"))
    fd_crypto.reset_for_tests()
    yield
    fd_crypto.reset_for_tests()


def _row(mid, text="hi", *, chat=ANA, sender=None, from_me=False, ts=1_790_000_000, op="message",
         kind="text", push="Ana", **extra):
    return {"op": op, "chat": chat, "id": mid, "sender": sender if sender is not None else ("" if from_me else chat),
            "from_me": from_me, "is_group": chat.endswith("@g.us"), "ts": ts, "push_name": push,
            "kind": kind, "text": text, "media": None, "reply_to": "", **extra}


# ── parse, from real protos ─────────────────────────────────────────


@pytest.fixture()
def pb():
    import importlib.util

    if importlib.util.find_spec("neonize") is None:
        pytest.skip("neonize isn't installed (pip install 'captain-claw[whatsapp-web]')")
    from captain_claw.flight_deck.whatsapp_web import neonize_adapter

    neonize_adapter._ensure_magic()  # before neonize's import, which loads libmagic
    from neonize.proto import Neonize_pb2 as N
    from neonize.proto.waCommon.WACommon_pb2 import MessageKey
    from neonize.proto.waE2E import WAWebProtobufsE2E_pb2 as E
    from neonize.proto.waHistorySync import WAWebProtobufsHistorySync_pb2 as H
    from neonize.proto.waWeb.WAWebProtobufsWeb_pb2 import WebMessageInfo

    return type("PB", (), {"N": N, "E": E, "H": H, "Key": MessageKey, "Web": WebMessageInfo})


def test_every_kind_from_its_proto(pb):
    E = pb.E
    c = parse.classify
    assert c(E.Message(conversation="hello")) == {"op": "message", "kind": "text", "text": "hello",
                                                   "media": None, "reply_to": ""}
    reply = c(E.Message(extendedTextMessage=E.ExtendedTextMessage(
        text="yes", contextInfo=E.ContextInfo(stanzaID="ABC"))))
    assert reply["text"] == "yes" and reply["reply_to"] == "ABC"
    img = c(E.Message(imageMessage=E.ImageMessage(caption="look", mimetype="image/jpeg", fileLength=1234)))
    assert (img["kind"], img["text"], img["media"]["mimetype"], img["media"]["size"]) == ("image", "look", "image/jpeg", 1234)
    assert img["media"]["raw"]  # what a later download needs
    assert c(E.Message(audioMessage=E.AudioMessage(PTT=True, seconds=42)))["kind"] == "voice"
    assert c(E.Message(audioMessage=E.AudioMessage(seconds=42)))["kind"] == "audio"
    doc = c(E.Message(documentMessage=E.DocumentMessage(fileName="offer.pdf", caption="")))
    assert doc["kind"] == "document" and doc["media"]["file_name"] == "offer.pdf"
    assert c(E.Message(stickerMessage=E.StickerMessage()))["kind"] == "sticker"
    loc = c(E.Message(locationMessage=E.LocationMessage(degreesLatitude=45.8, degreesLongitude=15.97, name="Zagreb")))
    assert loc["kind"] == "location" and "Zagreb" in loc["text"]


def test_reactions_edits_revokes_and_technical_traffic(pb):
    E, Key = pb.E, pb.Key
    react = parse.classify(E.Message(reactionMessage=E.ReactionMessage(
        key=Key(ID="M1"), text="👍", senderTimestampMS=1_790_000_000_000)))
    assert react == {"op": "react", "target": "M1", "emoji": "👍", "react_ts": 1_790_000_000}
    revoke = parse.classify(E.Message(protocolMessage=E.ProtocolMessage(
        key=Key(ID="M2"), type=E.ProtocolMessage.REVOKE)))
    assert revoke == {"op": "revoke", "target": "M2"}
    edit = parse.classify(E.Message(protocolMessage=E.ProtocolMessage(
        key=Key(ID="M3"), type=E.ProtocolMessage.MESSAGE_EDIT,
        editedMessage=E.Message(conversation="fixed typo"))))
    assert edit == {"op": "edit", "target": "M3", "text": "fixed typo"}
    # A group message with a new sender key arrives once as only the key.
    assert parse.classify(E.Message(senderKeyDistributionMessage=E.SenderKeyDistributionMessage(
        groupID=GROUP)))["op"] == "skip"
    assert parse.classify(E.Message())["op"] == "skip"
    assert parse.classify(None)["op"] == "skip"
    wrapped = E.Message(ephemeralMessage=E.FutureProofMessage(message=E.Message(conversation="gone soon")))
    assert parse.classify(wrapped)["text"] == "gone soon"


def test_live_event_row(pb):
    N, E = pb.N, pb.E
    ev = N.Message(
        Info=N.MessageInfo(ID="L1", Pushname="Ana", Timestamp=1_790_000_123_000, MessageSource=N.MessageSource(
            Chat=N.JID(User="385911111111", Server="s.whatsapp.net"),
            Sender=N.JID(User="385911111111", Server="s.whatsapp.net"), IsFromMe=False, IsGroup=False)),
        Message=E.Message(conversation="live one"))
    row = parse.live_row(ev)
    assert row["op"] == "message" and row["chat"] == ANA and row["ts"] == 1_790_000_123  # ms → s
    assert row["push_name"] == "Ana" and row["id"] == "L1" and row["text"] == "live one"


def test_history_chunk(pb):
    H, E, Key, Web = pb.H, pb.E, pb.Key, pb.Web
    conv = H.Conversation(ID=GROUP, name="Obitelj", pinned=1, messages=[
        H.HistorySyncMsg(message=Web(key=Key(remoteJID=GROUP, ID="H1", participant=IVO, fromMe=False),
                                     message=E.Message(conversation="dinner at 8"), messageTimestamp=1_789_000_000,
                                     pushName="Ivo")),
        H.HistorySyncMsg(message=Web(key=Key(remoteJID=GROUP, ID="H2", fromMe=True),
                                     message=E.Message(senderKeyDistributionMessage=E.SenderKeyDistributionMessage()),
                                     messageTimestamp=1_789_000_100)),
    ])
    hs = H.HistorySync(syncType=H.HistorySync.FULL, progress=40, conversations=[conv],
                       pushnames=[H.Pushname(ID=ANA, pushname="Ana")])
    hs.phoneNumberToLidMappings.add(pnJID=ANA, lidJID=ANA_LID)
    out = parse.history_chunk(hs)
    assert out["sync_type"] == "FULL" and out["progress"] == 40 and out["messages"] == 1
    first, second = out["rows"]
    assert (first["op"], first["sender"], first["ts"], first["push_name"]) == ("message", IVO, 1_789_000_000, "Ivo")
    assert second["op"] == "skip"
    assert out["chats"][0]["name"] == "Obitelj" and out["chats"][0]["pinned"] is True
    assert out["pushnames"] == [{"jid": ANA, "name": "Ana"}]
    assert out["lid_map"] == [{"pn": ANA, "lid": ANA_LID}]


def test_timestamps():
    assert parse.ts_seconds(1_790_000_000) == 1_790_000_000
    assert parse.ts_seconds(1_790_000_000_123) == 1_790_000_000
    assert parse.ts_seconds(None) == 0 and parse.ts_seconds("x") == 0


# ── store ───────────────────────────────────────────────────────────


def _store(tmp_path) -> Store:
    return Store(tmp_path / "store.sqlite3")


def test_one_row_per_message_however_often_it_arrives(tmp_path):
    s = _store(tmp_path)
    s.ingest([_row("M1", op="skip")], live=True)  # the sender-key half
    s.ingest([_row("M1", "the real text")], live=True)
    s.ingest([_row("M1", "the real text")], live=False)  # and again from history
    msgs = s.messages()
    assert len(msgs) == 1 and msgs[0]["text"] == "the real text" and msgs[0]["sender"] == "Ana"
    assert s.stats()["messages"] == 1 and s.stats()["chats"] == 1


def test_edits_deletions_and_reactions(tmp_path):
    s = _store(tmp_path)
    s.ingest([_row("M1", "teh plan"), _row("M2", "secret")], live=True)
    s.ingest([{**_row("", op="edit"), "target": "M1", "text": "the plan"},
              {**_row("", op="revoke"), "target": "M2"},
              {**_row("", op="react", from_me=True), "target": "M1", "emoji": "👍"},
              {**_row("", op="react", sender=IVO), "target": "M1", "emoji": "😂"}], live=True)
    by_id = {m["id"]: m for m in s.messages()}
    assert by_id["M1"]["text"] == "the plan" and by_id["M1"]["edited"]
    assert sorted(by_id["M1"]["reactions"]) == sorted(["👍 you", "😂 +385922222222"])
    assert by_id["M2"]["deleted"] and by_id["M2"]["text"] == ""
    s.ingest([{**_row("", op="react", from_me=True), "target": "M1", "emoji": ""}], live=True)  # un-react
    assert [r for r in {m["id"]: m for m in s.messages()}["M1"]["reactions"] if "you" in r] == []
    # A later copy of a deleted message doesn't bring its text back.
    s.ingest([_row("M2", "secret")], live=False)
    assert {m["id"]: m for m in s.messages()}["M2"]["text"] == ""


def test_nothing_readable_at_rest_but_jids(tmp_path):
    s = _store(tmp_path)
    s.ingest([_row("M1", "meet me at the old mill", push="Ana Secret")], live=True)
    s.put_chats([{"jid": GROUP, "name": "Hidden Group Name", "is_group": True}])
    s.put_contacts([{"jid": IVO, "full_name": "Ivo Ivić"}])
    s.close()
    raw = b"".join(p.read_bytes() for p in tmp_path.glob("store.sqlite3*"))
    for secret in (b"old mill", b"Ana Secret", b"Hidden Group Name", "Ivić".encode()):
        assert secret not in raw, secret
    assert ANA.encode() in raw  # JIDs stay plain (decision 1)


def test_lid_chats_fold_into_the_phone_number_chat(tmp_path):
    s = _store(tmp_path)
    s.ingest([_row("A1", "via lid", chat=ANA_LID)], live=True)
    s.ingest([_row("A2", "via number", chat=ANA)], live=True)
    assert len(s.chats()) == 2
    assert s.put_lid_map([{"pn": ANA, "lid": ANA_LID}]) == 1
    chats = s.chats()
    assert [c["jid"] for c in chats] == [ANA] and chats[0]["messages"] == 2
    s.ingest([_row("A3", "later, still via lid", chat=ANA_LID)], live=True)
    assert {m["chat_jid"] for m in s.messages()} == {ANA}


def test_names_chats_and_search(tmp_path):
    s = _store(tmp_path)
    s.ingest([_row("M1", "dinner at 8", chat=GROUP, sender=IVO, push="Ivo the Great", ts=1_790_000_100),
              _row("M2", "running late", chat=ANA, ts=1_790_000_200)], live=False)
    s.put_contacts([{"jid": IVO, "full_name": "Ivo Ivić", "push_name": "Ivo the Great"}])
    s.put_groups([{"jid": GROUP, "name": "Obitelj"}])
    chats = s.chats()
    assert [c["title"] for c in chats] == ["Ana", "Obitelj"]  # latest first; Ana by push name
    assert s.messages(chat_jid=GROUP)[0]["sender"] == "Ivo Ivić"  # the address book wins
    assert s.find_chat("obit")["jid"] == GROUP
    assert s.find_chat("+385 91 111 1111")["jid"] == ANA
    with pytest.raises(ChatNotFound):
        s.find_chat("nobody")
    assert [m["id"] for m in s.messages(query="DINNER 8")] == ["M1"]
    assert [m["id"] for m in s.messages(query="ivić dinner")] == ["M1"]  # sender names count
    assert s.messages(query="breakfast") == []
    people = s.contacts("ivo")
    assert people[0]["jid"] == IVO and people[0]["name"] == "Ivo Ivić" and people[0]["groups"] == ["Obitelj"]


def test_news_counts_only_live_arrivals(tmp_path):
    s = _store(tmp_path)
    now = 1_790_000_000.0
    s.ingest([_row("H1", "old history", ts=int(now) - 60)], live=False)
    s.ingest([_row("L1", "new!", ts=int(now) - 30), _row("L2", "me", from_me=True, ts=int(now) - 20),
              _row("L3", "yesterday's", ts=int(now) - 200_000)], live=True)
    first = s.news("bot-a", now=now)
    assert first["first_look"] and first["total"] == 1 and first["chats"][0]["new"] == 1
    assert s.news("bot-a", now=now)["total"] == 0  # the look moved the cursor
    assert s.news("bot-b", now=now)["total"] == 1  # each agent has its own place
    s.ingest([_row("L4", "another", ts=int(now))], live=True)
    s.mark_read("bot-b", [m for m in s.messages() if m["id"] == "L4"])
    assert s.news("bot-b", now=now)["total"] == 0  # reading the chat counts as seen
    assert s.news("bot-a", now=now)["total"] == 1


def test_retention(tmp_path):
    s = _store(tmp_path)
    now = time.time()
    s.ingest([_row("OLD", ts=int(now) - 40 * 86400), _row("NEW", ts=int(now) - 86400)], live=False)
    assert s.purge(now=now) == 0  # keep everything is the default
    s.set_meta("retention_days", "30")
    assert s.retention_days() == 30 and s.purge(now=now) == 1
    assert [m["id"] for m in s.messages()] == ["NEW"]


# ── engine ──────────────────────────────────────────────────────────


class FakeAdapter:
    instances: list = []

    def __init__(self, client_id, session_path, sink, *, history_days):
        self.sink, self.session_path = sink, Path(session_path)
        FakeAdapter.instances.append(self)

    async def connect(self):
        self.session_path.write_bytes(b"x")

    async def pair_code(self, phone):
        return "ABCD-EFGH"

    async def logout(self):
        pass

    async def stop(self):
        pass


async def _linked_engine(tmp_path, uid=ALICE):
    FakeAdapter.instances.clear()
    eng = Engine(tmp_path / "waw", adapter_factory=FakeAdapter)
    await eng.link(uid, consented_at="t")
    a = FakeAdapter.instances[-1]
    await a.sink("paired", {"jid": "385976707736:56@s.whatsapp.net"})
    await a.sink("connected", {})
    return eng, a


async def test_history_and_live_messages_reach_the_store(tmp_path):
    eng, a = await _linked_engine(tmp_path)
    await a.sink("history", {"sync_type": "FULL", "progress": 100, "messages": 2,
                             "rows": [_row("H1", "from history", chat=ANA_LID), _row("H2", "", op="skip")],
                             "chats": [{"jid": GROUP, "name": "Obitelj", "is_group": True}],
                             "pushnames": [{"jid": ANA, "name": "Ana"}],
                             "lid_map": [{"pn": ANA, "lid": ANA_LID}]})
    await a.sink("message", _row("L1", "live", chat=ANA))
    v = eng.view(ALICE)
    assert v["stats"]["messages"] == 2 and v["sync"]["chunks"] == 1 and v["retention_days"] is None
    data = await eng.messages(ALICE, chat="Ana", agent="bot", mark_seen=True)
    assert [m["id"] for m in data["messages"]] == ["L1", "H1"] and data["chat"]["jid"] == ANA
    assert (await eng.news(ALICE, agent="bot"))["total"] == 0  # read above
    chats = await eng.chats(ALICE)
    assert {c["title"] for c in chats["chats"]} == {"Ana", "Obitelj"}
    with pytest.raises(LinkError) as e:
        await eng.messages(ALICE, chat="nobody")
    assert e.value.status == 404


async def test_reading_needs_a_link(tmp_path):
    eng = Engine(tmp_path / "waw", adapter_factory=FakeAdapter)
    with pytest.raises(LinkError) as e:
        await eng.chats(BOB)
    assert e.value.status == 409
    assert await eng.news(BOB, agent="x") == {"connected": False}


async def test_unlinked_on_the_phone_takes_the_messages_from_the_link(tmp_path):
    eng, a = await _linked_engine(tmp_path)
    await a.sink("message", _row("L1", "live"))
    store_file = tmp_path / "waw" / ALICE / "store.sqlite3"
    assert store_file.exists()
    await a.sink("logged_out", {})
    for _ in range(5):
        await asyncio.sleep(0)
    assert not store_file.exists() and eng.view(ALICE)["stats"] is None
    with pytest.raises(LinkError):
        await eng.chats(ALICE)


async def test_disconnect_told_to_deletes_everything(tmp_path):
    eng, a = await _linked_engine(tmp_path)
    await a.sink("message", _row("L1", "live"))
    await eng.unlink(ALICE, keep_history=False)
    assert not (tmp_path / "waw" / ALICE).exists()
    assert not (tmp_path / "waw" / ".kept" / ALICE).exists()


async def test_retention_through_the_engine(tmp_path):
    eng, a = await _linked_engine(tmp_path)
    await a.sink("message", _row("OLD", ts=int(time.time()) - 100 * 86400))
    await a.sink("message", _row("NEW", ts=int(time.time())))
    with pytest.raises(LinkError) as e:
        await eng.set_retention(ALICE, 12)
    assert e.value.status == 422
    v = await eng.set_retention(ALICE, 30)
    assert v["retention_days"] == 30 and v["stats"]["messages"] == 1
    v = await eng.set_retention(ALICE, None)
    assert v["retention_days"] is None


async def test_service_read_api(tmp_path):
    eng, a = await _linked_engine(tmp_path)
    await a.sink("message", _row("L1", "hello there"))
    c = TestClient(build_app(eng, "s"))
    h = {"Authorization": "Bearer s"}
    assert c.get(f"/links/{ALICE}/chats", headers=h).json()["chats"][0]["jid"] == ANA
    r = c.post(f"/links/{ALICE}/messages", json={"query": "hello"}, headers=h).json()
    assert [m["id"] for m in r["messages"]] == ["L1"]
    assert c.post(f"/links/{ALICE}/contacts", json={"query": "ana"}, headers=h).json()["contacts"]
    assert c.post(f"/links/{ALICE}/news", json={"agent": "bot"}, headers=h).json()["connected"]
    assert c.put(f"/links/{ALICE}/retention", json={"days": 7}, headers=h).json()["retention_days"] == 7
    assert c.get(f"/links/{BOB}/chats", headers=h).status_code == 409


# ── FD's agent routes ───────────────────────────────────────────────


class Svc:
    """A stand-in for the service: answers what the routes ask, records it."""

    def __init__(self, root):
        self.root = root
        self.calls = []

    def user_has_link(self, uid):
        return (self.root / uid / "state.json").exists()

    def offline_view(self, uid):
        return sup.Supervisor.offline_view(self, uid)

    def has_kept(self, uid):
        return False

    async def request(self, method, path, body=None, **_):
        self.calls.append((method, path, body))
        if path.endswith("/news"):
            return {"connected": True, "chats": [], "total": 0, "first_look": True}
        if "/chats" in path:
            return {"chats": [{"jid": ANA, "title": "Ana"}], "stats": {}, "state": "connected"}
        return {"chat": None, "messages": []}


@pytest.fixture()
async def deck(monkeypatch, tmp_path):
    import captain_claw.flight_deck.server as srv

    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.delenv("FD_WHATSAPP_WEB", raising=False)
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    monkeypatch.delenv("FD_AGENT_SHARED_SECRET", raising=False)
    d = FlightDeckDB(str(tmp_path / "fd.db"))
    await d.init()
    prev = auth._db
    auth.set_auth_db(d)
    now = "2026-01-01T00:00:00Z"
    for uid in (ALICE, BOB):
        await d._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)", (uid, f"{uid}@x.co", "h", uid, "user", now, now))
    await d._db.commit()
    await sup.save_config(d, allowed=True)
    agents = {"tok-bob": (BOB, "bob-bot"), "tok-alice": (ALICE, "alice-bot")}

    def find(token):
        return (True, *agents[token]) if token in agents else (False, "", "")

    monkeypatch.setattr(srv, "_find_agent_by_auth", find)
    monkeypatch.setattr(srv, "_resolve_agent_identity_by_auth", lambda t: find(t)[:2])
    svc = Svc(tmp_path / "waw")
    monkeypatch.setattr(sup, "get", lambda: svc)
    monkeypatch.setattr(sup, "installed", lambda: True)
    app = FastAPI()
    app.include_router(routes.router)
    try:
        yield d, svc, TestClient(app, client=("127.0.0.1", 50123))
    finally:
        auth.set_auth_db(prev)
        await d.close()


def _linked_on_disk(root: Path, uid: str, state="connected"):
    from captain_claw.flight_deck.whatsapp_web.link_state import LinkState

    (root / uid).mkdir(parents=True, exist_ok=True)
    LinkState(state=state, jid="385976707736@s.whatsapp.net", linked_at="2026-10-26T09:00:00+00:00").save(
        root / uid / "state.json")


def test_agents_read_only_their_own_owner(deck):
    _db, svc, c = deck
    bob = {"X-Agent-Auth": "tok-bob"}
    assert c.get("/fd/whatsapp-web/agent/status", headers=bob).json() == {"connected": False}
    r = c.post("/fd/whatsapp-web/agent/chats", headers=bob)
    assert r.status_code == 409 and "linked WhatsApp" in r.json()["detail"]
    _linked_on_disk(svc.root, BOB)
    assert c.get("/fd/whatsapp-web/agent/status", headers=bob).json() == {"connected": True}
    assert c.post("/fd/whatsapp-web/agent/chats", headers=bob).json()["chats"][0]["jid"] == ANA
    c.post("/fd/whatsapp-web/agent/messages", json={"chat": "Ana", "limit": 5}, headers=bob)
    method, path, body = svc.calls[-1]
    assert path == f"/links/{BOB}/messages" and body["agent"] == "bob-bot" and body["mark_seen"] is True
    c.post("/fd/whatsapp-web/agent/news", headers=bob)
    assert svc.calls[-1][:2] == ("POST", f"/links/{BOB}/news") and svc.calls[-1][2] == {"agent": "bob-bot"}
    # Alice's agent never reaches Bob's link, and has none of its own.
    assert c.post("/fd/whatsapp-web/agent/chats", headers={"X-Agent-Auth": "tok-alice"}).status_code == 409
    assert all(f"/links/{ALICE}" not in p for _m, p, _b in svc.calls)
    # Unknown agents and browsers are refused.
    assert c.post("/fd/whatsapp-web/agent/chats", headers={"X-Agent-Auth": "nope"}).status_code == 403
    assert c.post("/fd/whatsapp-web/agent/chats", headers={**bob, "Origin": "http://evil.example"}).status_code == 403


def test_a_link_in_trouble_is_still_readable_but_not_a_dead_one(deck):
    _db, svc, c = deck
    bob = {"X-Agent-Auth": "tok-bob"}
    _linked_on_disk(svc.root, BOB, state="stream_replaced")
    assert c.get("/fd/whatsapp-web/agent/status", headers=bob).json() == {"connected": True}
    _linked_on_disk(svc.root, BOB, state="logged_out")
    assert c.get("/fd/whatsapp-web/agent/status", headers=bob).json() == {"connected": False}
    assert c.post("/fd/whatsapp-web/agent/news", headers=bob).json() == {"connected": False}


def test_no_sign_in_no_owners(deck, monkeypatch):
    _db, svc, c = deck
    _linked_on_disk(svc.root, BOB)
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    bob = {"X-Agent-Auth": "tok-bob"}
    assert c.get("/fd/whatsapp-web/agent/status", headers=bob).json() == {"connected": False}
    r = c.post("/fd/whatsapp-web/agent/chats", headers=bob)
    assert r.status_code == 409 and "sign-in" in r.json()["detail"]
    assert not svc.calls


def test_sealed_store_is_not_sqlite_plaintext_even_for_search_terms(tmp_path):
    # The search index is the sealed text itself: nothing to grep in the file.
    s = _store(tmp_path)
    s.ingest([_row("M1", "unique-needle-word")], live=True)
    s.close()
    con = sqlite3.connect(str(tmp_path / "store.sqlite3"))
    assert not con.execute("SELECT 1 FROM messages WHERE body LIKE '%needle%'").fetchall()
    con.close()


async def test_a_late_event_doesnt_recreate_an_unlinked_folder(tmp_path):
    eng, a = await _linked_engine(tmp_path)
    link = eng._links[ALICE]
    await eng.unlink(ALICE)
    eng._ingest(link, "message", _row("LATE", "on its way during the unlink"))
    assert not (tmp_path / "waw" / ALICE).exists()
