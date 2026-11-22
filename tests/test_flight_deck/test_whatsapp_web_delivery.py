"""WhatsApp Web: whether what an agent sent reached the other phone.

WhatsApp accepting a send isn't delivery: on staging it took two auto-replies
(message ids and all) and no phone ever showed them, while the deck said
"sent". Receipts now say what happened:

* a delivered / read receipt moves the send on, never back; a receipt from the
  account's own devices, or for a message the phone sent, changes nothing;
* a receipt can come in before the send is recorded, and still counts;
* WhatsApp turning a message away rings a bell at once;
* a send no phone confirms within ten minutes rings one bell (then quiet for
  hours), only while connected and not right after a reconnect;
* sends from before receipts were kept read "untracked", never "not delivered";
* an auto-reply's entry says whether its answer arrived;
* a signature the text already carries isn't added twice.

WhatsApp is never contacted.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from captain_claw.flight_deck import fd_crypto
from captain_claw.flight_deck.whatsapp_web import parse
from captain_claw.flight_deck.whatsapp_web.engine import (
    UNDELIVERED_AFTER,
    UNDELIVERED_QUIET,
    Engine,
    LinkError,
)
from captain_claw.flight_deck.whatsapp_web.store import DEFAULT_SIGNATURE, Store, strip_signature

ANA = "385911111111@s.whatsapp.net"
ANA_LID = "200000000000001@lid"
ALICE = "user-alice"
T0 = 1_790_000_000.0


@pytest.fixture(autouse=True)
def data_key(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_DATA_KEY_FILE", str(tmp_path / "data_key"))
    fd_crypto.reset_for_tests()
    yield
    fd_crypto.reset_for_tests()


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t


class FakeAdapter:
    def __init__(self, client_id, session_path, sink, *, history_days):
        self.sink, self.session_path = sink, Path(session_path)
        self.sent: list[tuple[str, str]] = []
        # A receipt WhatsApp sends before send_text has even returned.
        self.receipt_first = ""
        FakeAdapter.last = self

    async def connect(self):
        self.session_path.write_bytes(b"x")

    async def send_text(self, jid, text):
        self.sent.append((jid, text))
        wa_id = f"3EB0{len(self.sent):04d}"
        if self.receipt_first:
            await self.sink("receipt", {"kind": self.receipt_first, "ids": [wa_id], "chat": ANA_LID})
        return wa_id

    async def logout(self):
        pass

    async def stop(self):
        pass


async def _engine(tmp_path, clock):
    eng = Engine(tmp_path / "waw", adapter_factory=FakeAdapter, clock=clock)
    await eng.link(ALICE, consented_at="t")
    a = FakeAdapter.last
    await a.sink("paired", {"jid": "385900000005:5@s.whatsapp.net"})
    await a.sink("connected", {})
    await a.sink("message", {"op": "message", "chat": ANA, "id": "A1", "sender": ANA, "from_me": False,
                             "is_group": False, "ts": int(T0) - 100, "push_name": "Ana", "kind": "text",
                             "text": "hi", "media": None, "reply_to": ""})
    await eng.set_send_policy(ALICE, {"enabled": True, "allowed": [ANA]})
    return eng, a


async def _sends(eng):
    return (await eng.send_policy(ALICE))["sends"]


def _bells(eng):
    return [e for e in eng.events_after(0) if e.get("title")]


# ── receipts ────────────────────────────────────────────────────────


def test_a_receipt_from_neonize():
    src = SimpleNamespace(IsFromMe=False, Chat=SimpleNamespace(User="200000000000001", Server="lid"),
                          Sender=SimpleNamespace(User="200000000000001", Server="lid"))
    ev = SimpleNamespace(Type="DELIVERED", MessageIDs=["3EB0A", "", "3EB0B"], MessageSource=src, Timestamp=T0 * 1000)
    assert parse.receipt(ev) == {"kind": "delivered", "ids": ["3EB0A", "3EB0B"], "chat": ANA_LID,
                                 "sender": ANA_LID, "ts": int(T0)}
    # neonize's enum is a number with a name.
    names = {1: "DELIVERED", 4: "READ", 6: "PLAYED", 3: "RETRY", 8: "SERVER_ERROR", 2: "SENDER", 5: "READ_SELF"}
    kinds = {}
    for number, name in names.items():
        e = SimpleNamespace(Type=number, ReceiptType=SimpleNamespace(Name=names.get), MessageIDs=["x"],
                            MessageSource=src, Timestamp=0)
        kinds[name] = parse.receipt(e)["kind"]
    assert kinds == {"DELIVERED": "delivered", "READ": "read", "PLAYED": "read", "RETRY": "retry",
                     "SERVER_ERROR": "error", "SENDER": "", "READ_SELF": ""}
    # The account's own devices confirming isn't the other side.
    own = SimpleNamespace(Type="READ", MessageIDs=["x"], MessageSource=SimpleNamespace(IsFromMe=True), Timestamp=0)
    assert parse.receipt(own)["kind"] == ""


async def test_receipts_move_a_send_on_never_back(tmp_path):
    clock = Clock()
    eng, a = await _engine(tmp_path, clock)
    out = await eng.send(ALICE, chat="Ana", text="Running late", agent="bot")
    wa_id = out["id"]
    assert (await _sends(eng))[0]["delivery"] == ""
    clock.t += 5
    await a.sink("receipt", {"kind": "delivered", "ids": [wa_id], "chat": ANA_LID})
    s = (await _sends(eng))[0]
    assert s["delivery"] == "delivered" and s["delivered_at"] == int(clock.t) and s["read_at"] is None
    clock.t += 5
    await a.sink("receipt", {"kind": "read", "ids": [wa_id]})
    await a.sink("receipt", {"kind": "delivered", "ids": [wa_id]})  # a late one from another device
    s = (await _sends(eng))[0]
    assert s["delivery"] == "read" and s["read_at"] == int(clock.t) and s["delivered_at"] == int(clock.t) - 5
    # A message the phone sent, a receipt with no kind: nothing changes, nothing breaks.
    await a.sink("receipt", {"kind": "delivered", "ids": ["PHONE1"]})
    await a.sink("receipt", {"kind": "", "ids": [wa_id]})
    await a.sink("receipt", {"kind": "inactive", "ids": [wa_id]})
    assert (await _sends(eng))[0]["delivery"] == "read"


async def test_a_receipt_before_the_send_is_recorded_still_counts(tmp_path):
    eng, a = await _engine(tmp_path, Clock())
    a.receipt_first = "delivered"
    await eng.send(ALICE, chat="Ana", text="fast phone", agent="bot")
    assert (await _sends(eng))[0]["delivery"] == "delivered"


async def test_whatsapp_turning_a_message_away_rings_at_once(tmp_path):
    eng, a = await _engine(tmp_path, Clock())
    out = await eng.send(ALICE, chat="Ana", text="are you there?", agent="bot")
    await a.sink("receipt", {"kind": "error", "ids": [out["id"]]})
    bell = _bells(eng)[-1]
    assert bell["title"] == "WhatsApp turned away bot's message" and "To Ana: “are you there?" in bell["body"]
    assert (await _sends(eng))[0]["delivery"] == "error"
    # Delivered after all: that wins.
    await a.sink("receipt", {"kind": "delivered", "ids": [out["id"]]})
    assert (await _sends(eng))[0]["delivery"] == "delivered"


async def test_an_undelivered_send_rings_once_then_stays_quiet(tmp_path):
    clock = Clock()
    eng, a = await _engine(tmp_path, clock)
    clock.t += 300
    await eng.send(ALICE, chat="Ana", text="Hey, I'm here", agent="Glasses")
    before = len(_bells(eng))
    clock.t += UNDELIVERED_AFTER - 1
    await eng.check_delivery()
    assert len(_bells(eng)) == before
    clock.t += 2
    await eng.check_delivery()
    (bell,) = _bells(eng)[before:]
    assert bell["title"] == "Glasses's WhatsApp message hasn't arrived"
    assert "To Ana, 10 min ago: “Hey, I'm here”" in bell["body"] and "holding back" in bell["body"]
    await eng.check_delivery()  # the same send never rings twice
    assert len(_bells(eng)) == before + 1
    # Another one in the quiet hours: no bell, and it isn't saved for later either.
    await eng.send(ALICE, chat="Ana", text="Yes, it works", agent="Glasses")
    after = len(_bells(eng))
    clock.t += UNDELIVERED_AFTER + 1
    await eng.check_delivery()
    assert len(_bells(eng)) == after
    clock.t += UNDELIVERED_QUIET
    await eng.check_delivery()
    assert len(_bells(eng)) == after


async def test_delivered_sends_and_a_fresh_connection_ring_nothing(tmp_path):
    clock = Clock()
    eng, a = await _engine(tmp_path, clock)
    clock.t += 300
    out = await eng.send(ALICE, chat="Ana", text="one", agent="bot")
    await a.sink("receipt", {"kind": "delivered", "ids": [out["id"]]})
    await eng.send(ALICE, chat="Ana", text="two", agent="bot")
    before = len(_bells(eng))
    # Disconnected: receipts can't come in, so nothing is judged.
    await a.sink("disconnected", {})
    clock.t += UNDELIVERED_AFTER + 1
    await eng.check_delivery()
    assert len(_bells(eng)) == before
    # Just reconnected: the receipts it missed are still on their way.
    await a.sink("connected", {})
    await eng.check_delivery()
    assert len(_bells(eng)) == before
    clock.t += 121
    await eng.check_delivery()
    (bell,) = _bells(eng)[before:]
    assert "“two”" in bell["body"]  # only the one nobody confirmed


async def test_an_auto_reply_entry_says_whether_it_arrived(tmp_path):
    eng, a = await _engine(tmp_path, Clock())
    out = await eng.send(ALICE, chat="Ana", text="Saturday at 8", agent="bot")
    await eng.add_autoreply_log(ALICE, chat=ANA, outcome="sent", details={"sent_id": out["id"], "steps": []})
    await eng.add_autoreply_log(ALICE, chat=ANA, outcome="skip:silent", details={"steps": []})
    silent, sent = (await eng.autoreply_log(ALICE))["entries"]
    assert "delivery" not in silent and sent["delivery"] == {"sent_at": int(T0), "delivery": "", "delivered_at": None,
                                                               "read_at": None}
    await a.sink("receipt", {"kind": "read", "ids": [out["id"]]})
    assert (await eng.autoreply_log(ALICE))["entries"][1]["delivery"]["delivery"] == "read"


def test_sends_from_before_receipts_are_untracked(tmp_path):
    path = tmp_path / "store.sqlite3"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE sends (rowid INTEGER PRIMARY KEY, ts INTEGER NOT NULL, chat_jid TEXT NOT NULL,"
               " agent TEXT NOT NULL DEFAULT '', text_hash TEXT NOT NULL, body TEXT NOT NULL DEFAULT '',"
               " status TEXT NOT NULL, wa_id TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '')")
    db.execute("INSERT INTO sends (ts, chat_jid, text_hash, status, wa_id) VALUES (?, ?, 'h', 'sent', '3EB0OLD')",
               (int(T0) - 7200, ANA))
    db.commit()
    db.close()
    s = Store(path)
    assert s.sends()[0]["delivery"] == "untracked"
    assert s.undelivered(older_than=T0, since=T0 - 86400) == []
    rid = s.start_send(ANA, "bot", "new", now=T0)
    s.finish_send(rid, "sent", wa_id="3EB0NEW")
    assert [x["delivery"] for x in s.sends()] == ["", "untracked"]
    s.close()
    Store(path).close()  # opening again changes nothing


# ── the signature, once ─────────────────────────────────────────────


def test_strip_signature():
    sig = "— sent by LeChuck"
    assert strip_signature("Yes, it works! ⏎ — sent by LeChuck", sig) == "Yes, it works!"
    assert strip_signature("Hi\n\n— sent by LeChuck\n—  sent by  LeChuck ", sig) == "Hi"
    assert strip_signature("sent by LeChuck rocks", sig) == "sent by LeChuck rocks"
    assert strip_signature("Hi ⏎ there", sig) == "Hi ⏎ there"
    assert strip_signature("Hi", "") == "Hi"


async def test_a_text_that_signs_itself_is_signed_once(tmp_path):
    eng, a = await _engine(tmp_path, Clock())
    await eng.send(ALICE, chat="Ana", text=f"Running late\n{DEFAULT_SIGNATURE}", agent="bot")
    assert a.sent[-1] == (ANA, f"Running late\n\n{DEFAULT_SIGNATURE}")
    with pytest.raises(LinkError) as e:
        await eng.send(ALICE, chat="Ana", text=DEFAULT_SIGNATURE)
    assert e.value.status == 422
    # With the signature off, the text is the agent's own: left as it is.
    await eng.set_send_policy(ALICE, {"signature_on": False})
    await eng.send(ALICE, chat="Ana", text=f"quoting: {DEFAULT_SIGNATURE}")
    assert a.sent[-1] == (ANA, f"quoting: {DEFAULT_SIGNATURE}")
