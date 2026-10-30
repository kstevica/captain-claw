"""WhatsApp Web: a link's messages outlive it (kept_history).

* the ways a link ends — WhatsApp logging the device out (how a ban arrives),
  Link again, Dismiss, a disconnect with either answer — and what each leaves;
* reading — agents read the linked account and what's kept as one, a message in
  both once, and with no link at all they still read what's kept;
* sending — an earlier number's chats are nothing the linked number may write to;
* deleting, retention, and one user's history out of another's reach;
* the routes — what the card and the agents get, the deck switched off, a
  service too old to keep history refusing rather than deleting.

WhatsApp is never contacted.
"""

from __future__ import annotations

import asyncio
import stat
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from captain_claw.flight_deck import auth, fd_crypto
from captain_claw.flight_deck import whatsapp_web_routes as routes
from captain_claw.flight_deck import whatsapp_web_supervisor as sup
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.flight_deck.whatsapp_web import kept_history
from captain_claw.flight_deck.whatsapp_web.api import build_app
from captain_claw.flight_deck.whatsapp_web.engine import Engine, LinkError
from captain_claw.flight_deck.whatsapp_web.link_state import LinkState

ALICE, BOB = "user-alice", "user-bob"
OLD_NUMBER = "385911110000:7@s.whatsapp.net"
NEW_NUMBER = "385922220000:7@s.whatsapp.net"
ANA = "385931111111@s.whatsapp.net"
IVO = "385932222222@s.whatsapp.net"
GROUP = "120363000000000001@g.us"


@pytest.fixture(autouse=True)
def data_key(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_DATA_KEY_FILE", str(tmp_path / "data_key"))
    fd_crypto.reset_for_tests()
    yield
    fd_crypto.reset_for_tests()


def _row(mid, text="hi", *, chat=ANA, ts=None, push="Ana"):
    return {"op": "message", "chat": chat, "id": mid, "sender": chat, "from_me": False,
            "is_group": chat.endswith("@g.us"), "ts": ts or int(time.time()), "push_name": push,
            "kind": "text", "text": text, "media": None, "reply_to": ""}


class FakeAdapter:
    instances: list = []

    def __init__(self, client_id, session_path, sink, *, history_days):
        self.sink, self.session_path = sink, Path(session_path)
        self.sent: list = []
        FakeAdapter.instances.append(self)

    async def connect(self):
        self.session_path.write_bytes(b"x")

    async def logout(self):
        pass

    async def stop(self):
        pass

    async def send_text(self, jid, text):
        self.sent.append((jid, text))
        return "SENT1"


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0)


async def _link(eng, uid=ALICE, jid=OLD_NUMBER):
    await eng.link(uid, consented_at="t")
    a = FakeAdapter.instances[-1]
    await a.sink("paired", {"jid": jid})
    await a.sink("connected", {})
    return a


@pytest.fixture()
async def eng(tmp_path):
    FakeAdapter.instances.clear()
    return Engine(tmp_path / "waw", adapter_factory=FakeAdapter)


def _kept(eng, uid=ALICE):
    return kept_history.entries(eng.root, uid)


# ── the ways a link ends ────────────────────────────────────────────


async def test_a_ban_keeps_the_messages_and_says_so(eng):
    a = await _link(eng)
    await a.sink("message", _row("M1", "before the ban"))
    await a.sink("logged_out", {"reason": "403"})
    await _settle()
    # The session is gone, the store moved away from it.
    v = eng.view(ALICE)
    assert v["state"] == "logged_out" and not v["has_session"] and v["stats"] is None
    assert not (eng.root / ALICE / "store.sqlite3").exists()
    entry = _kept(eng)[0]
    assert entry["number"] == "+385911110000" and entry["reason"] == "logged_out"
    assert entry["stats"]["messages"] == 1 and entry["stats"]["chats"] == 1
    bell = eng.events_after(0)[0]
    assert "banned" in bell["body"] and "stay" in bell["body"]
    # Agents read it; what the linked account reads doesn't exist any more.
    found = await eng.messages(ALICE, chat="Ana", with_kept=True)
    assert [m["text"] for m in found["messages"]] == ["before the ban"]
    with pytest.raises(LinkError) as e:
        await eng.chats(ALICE)
    assert e.value.status == 409


async def test_link_again_and_dismiss_after_a_ban_lose_nothing(eng):
    a = await _link(eng)
    await a.sink("message", _row("M1"))
    await a.sink("logged_out", {})
    await _settle()
    await eng.link(ALICE, consented_at="t2")
    assert len(_kept(eng)) == 1 and eng.view(ALICE)["state"] == "pairing"

    a2 = FakeAdapter.instances[-1]
    await a2.sink("paired", {"jid": NEW_NUMBER})
    await a2.sink("connected", {})
    await a2.sink("logged_out", {})
    await _settle()
    await eng.cancel(ALICE)  # the card's Dismiss
    assert eng.view(ALICE)["state"] == "off" and len(_kept(eng)) == 1


async def test_a_failed_link_replaced_keeps_its_messages(eng):
    a = await _link(eng)
    await a.sink("message", _row("M1"))
    await eng._stop_client(eng._links[ALICE])
    eng._links[ALICE].state.state = "failed"
    await eng.link(ALICE, consented_at="t2")
    assert _kept(eng)[0]["stats"]["messages"] == 1
    assert _kept(eng)[0]["reason"] == "replaced"


async def test_a_failed_link_dismissed_keeps_its_messages(eng):
    a = await _link(eng)
    await a.sink("message", _row("M1"))
    await eng._stop_client(eng._links[ALICE])
    eng._links[ALICE].state.state = "failed"
    await eng.cancel(ALICE)
    assert _kept(eng)[0]["stats"]["messages"] == 1


async def test_disconnect_keeps_by_default_and_deletes_when_told(eng):
    a = await _link(eng)
    await a.sink("message", _row("M1"))
    v = await eng.unlink(ALICE)
    assert v["state"] == "off" and v["removed_from_phone"] is True
    assert not (eng.root / ALICE).exists() and _kept(eng)[0]["reason"] == "disconnected"

    a = await _link(eng, jid=NEW_NUMBER)
    await a.sink("message", _row("M2"))
    await eng.unlink(ALICE, keep_history=False)
    assert not (eng.root / ALICE).exists()
    # Deleting this link's messages leaves what an earlier link kept alone.
    assert [k["number"] for k in _kept(eng)] == ["+385911110000"]


async def test_nothing_kept_when_there_was_nothing(eng):
    await _link(eng)
    eng.view(ALICE)
    eng._open_store(eng._links[ALICE])
    await eng.unlink(ALICE)
    assert _kept(eng) == [] and await eng.news(ALICE, agent="x") == {"connected": False}
    await eng.link(ALICE, consented_at="t")
    await eng.cancel(ALICE)  # never paired
    assert _kept(eng) == []


async def test_when_they_cant_be_moved_nothing_is_deleted(eng, monkeypatch):
    a = await _link(eng)
    await a.sink("message", _row("M1"))

    def refuse(*_a, **_k):
        raise OSError("disk full")

    with monkeypatch.context() as m:
        m.setattr(kept_history, "archive", refuse)
        with pytest.raises(LinkError) as e:
            await eng.unlink(ALICE)
        assert e.value.status == 500 and (eng.root / ALICE / "store.sqlite3").exists()
        with pytest.raises(LinkError):
            await eng.link(ALICE, consented_at="t2")
        assert (eng.root / ALICE / "store.sqlite3").exists()
    await eng.link(ALICE, consented_at="t3")
    assert _kept(eng)[0]["stats"]["messages"] == 1


async def test_a_ban_when_they_cant_be_moved_leaves_them_for_link_again(eng, monkeypatch):
    a = await _link(eng)
    await a.sink("message", _row("M1"))

    def refuse(*_a, **_k):
        raise OSError("disk full")

    with monkeypatch.context() as m:
        m.setattr(kept_history, "archive", refuse)
        await a.sink("logged_out", {})
        await _settle()
    assert eng.view(ALICE)["state"] == "logged_out" and not eng.view(ALICE)["has_session"]
    assert (eng.root / ALICE / "store.sqlite3").exists()
    await eng.link(ALICE, consented_at="t2")
    assert _kept(eng)[0]["stats"]["messages"] == 1


async def test_kept_files_stay_private(eng):
    a = await _link(eng)
    await a.sink("message", _row("M1", "unique-needle"))
    await eng.unlink(ALICE)
    entry = _kept(eng)[0]["id"]
    folder = eng.root / kept_history.KEPT_DIR / ALICE / entry
    for path in (eng.root / kept_history.KEPT_DIR, folder.parent, folder):
        assert stat.S_IMODE(path.stat().st_mode) == 0o700, path
    assert stat.S_IMODE((folder / "kept.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((folder / "store.sqlite3").stat().st_mode) == 0o600
    assert b"unique-needle" not in (folder / "store.sqlite3").read_bytes()


async def test_a_restart_finds_it_and_doesnt_take_it_for_a_user(eng, tmp_path):
    a = await _link(eng)
    await a.sink("message", _row("M1"))
    await eng.unlink(ALICE)
    FakeAdapter.instances.clear()
    again = Engine(tmp_path / "waw", adapter_factory=FakeAdapter)
    assert await again.resume_all() == 0 and not FakeAdapter.instances
    assert (await again.chats(ALICE, with_kept=True))["chats"][0]["jid"] == ANA


# ── reading ─────────────────────────────────────────────────────────


async def test_the_new_number_reads_both_and_a_shared_message_once(eng):
    a = await _link(eng)
    await a.sink("message", _row("OLD", "from the old number", ts=1_790_000_000))
    await a.sink("message", _row("G1", "the group", chat=GROUP, ts=1_790_000_100, push="Ivo"))
    await eng.unlink(ALICE)
    a = await _link(eng, jid=NEW_NUMBER)
    await a.sink("message", _row("NEW", "from the new number", ts=1_790_000_200))
    await a.sink("message", _row("G1", "the group", chat=GROUP, ts=1_790_000_100, push="Ivo"))

    # The linked account alone, as the sending rules see it.
    assert {c["jid"] for c in (await eng.chats(ALICE))["chats"]} == {ANA, GROUP}

    data = await eng.chats(ALICE, with_kept=True)
    assert data["state"] == "connected" and data["stats"]["messages"] == 2
    assert [k["number"] for k in data["kept"]] == ["+385911110000"]
    ana = next(c for c in data["chats"] if c["jid"] == ANA)
    assert ana["messages"] == 2 and ana["kept_only"] is False
    items = (await eng.messages(ALICE, chat=ANA, with_kept=True))["messages"]
    assert [m["id"] for m in items] == ["NEW", "OLD"]
    # Across chats, newest first, the group's message once.
    items = (await eng.messages(ALICE, with_kept=True))["messages"]
    assert [m["id"] for m in items] == ["NEW", "G1", "OLD"]
    assert [m["id"] for m in (await eng.messages(ALICE, with_kept=True, limit=2))["messages"]] == ["NEW", "G1"]
    # Search covers both.
    items = (await eng.messages(ALICE, query="old number", with_kept=True))["messages"]
    assert [m["id"] for m in items] == ["OLD"]
    # What a reply is written from stays with the linked account.
    assert [m["id"] for m in (await eng.messages(ALICE, chat=ANA))["messages"]] == ["NEW"]


async def test_a_chat_only_the_old_number_had_is_marked(eng):
    a = await _link(eng)
    await a.sink("message", _row("I1", "hello", chat=IVO, push="Ivo"))
    await eng.unlink(ALICE)
    a = await _link(eng, jid=NEW_NUMBER)
    await a.sink("message", _row("A1", "hi"))
    chats = {c["jid"]: c for c in (await eng.chats(ALICE, with_kept=True))["chats"]}
    assert chats[IVO]["kept_only"] is True and chats[ANA]["kept_only"] is False
    assert (await eng.messages(ALICE, chat="Ivo", with_kept=True))["chat"]["jid"] == IVO
    assert (await eng.contacts(ALICE, query="Ivo"))["contacts"][0]["jid"] == IVO


async def test_the_news_line_counts_only_the_linked_account(eng):
    a = await _link(eng)
    await a.sink("message", _row("OLD"))
    await eng.unlink(ALICE)
    # No link: nothing new, but the tool stays, for what's kept.
    assert await eng.news(ALICE, agent="bot") == {"connected": True, "chats": [], "total": 0, "first_look": False}
    a = await _link(eng, jid=NEW_NUMBER)
    await a.sink("message", _row("NEW"))
    news = await eng.news(ALICE, agent="bot")
    assert news["connected"] and news["total"] == 1


async def test_a_look_at_kept_messages_moves_only_the_linked_accounts_cursor(eng):
    a = await _link(eng)
    for i in range(5):
        await a.sink("message", _row(f"OLD{i}", chat=IVO))
    await eng.unlink(ALICE)
    a = await _link(eng, jid=NEW_NUMBER)
    await a.sink("message", _row("FIRST", chat=IVO))
    # The kept store's rowids run ahead of the live store's: they mustn't become
    # its cursor, or the next messages here would never count as new.
    await eng.messages(ALICE, chat=IVO, with_kept=True, agent="bot", mark_seen=True)
    await eng.news(ALICE, agent="bot")
    await a.sink("message", _row("NEXT", chat=IVO))
    assert (await eng.news(ALICE, agent="bot"))["total"] == 1


async def test_one_unreadable_entry_doesnt_stop_the_others(eng):
    a = await _link(eng)
    await a.sink("message", _row("M1"))
    await eng.unlink(ALICE)
    a = await _link(eng, jid=NEW_NUMBER)
    await a.sink("message", _row("M2", chat=IVO, push="Ivo"))
    await eng.unlink(ALICE)
    entry = next(k for k in _kept(eng) if k["number"] == "+385922220000")
    bad = kept_history.store_path(eng.root, ALICE, entry["id"])
    bad.write_bytes(b"not a database" * 100)
    eng._close_kept()
    assert [c["jid"] for c in (await eng.chats(ALICE, with_kept=True))["chats"]] == [ANA]


# ── sending stays with the linked number ────────────────────────────


async def test_nothing_goes_to_a_chat_only_the_old_number_had(eng):
    a = await _link(eng)
    await a.sink("message", _row("I1", "hello", chat=IVO, push="Ivo"))
    await eng.set_send_policy(ALICE, {"enabled": True, "allowed": [IVO]})
    await eng.unlink(ALICE)

    # With no link at all, nothing can be sent.
    with pytest.raises(LinkError) as e:
        await eng.send(ALICE, chat=IVO, text="hi")
    assert e.value.status == 409

    a = await _link(eng, jid=NEW_NUMBER)
    await a.sink("message", _row("A1", "hi"))
    # The whitelist stayed with the old number, and can't take its chats back.
    policy = await eng.send_policy(ALICE)
    assert policy["enabled"] is False and policy["allowed"] == []
    out = await eng.set_send_policy(ALICE, {"enabled": True, "allowed": [IVO, ANA]})
    assert out["allowed"] == [ANA] and out["dropped"] == [IVO]
    with pytest.raises(LinkError) as e:
        await eng.send(ALICE, chat=IVO, text="hi")
    assert e.value.status == 403 and "earlier number" in e.value.detail
    assert not a.sent
    assert (await eng.send(ALICE, chat=ANA, text="hi"))["status"] == "sent"
    assert a.sent == [(ANA, "hi\n\n— sent by my assistant")]


# ── deleting, retention, other users ────────────────────────────────


async def test_deleting_one_entry_or_refusing_a_bad_one(eng):
    a = await _link(eng)
    await a.sink("message", _row("M1"))
    await eng.unlink(ALICE)
    a = await _link(eng, jid=NEW_NUMBER)
    await a.sink("message", _row("M2"))
    await eng.unlink(ALICE)
    assert (await eng.chats(ALICE, with_kept=True))["chats"]
    newest, oldest = _kept(eng)
    for bad in ("nope", "../../etc", oldest["id"] + "x"):
        with pytest.raises(LinkError) as e:
            await eng.delete_kept(ALICE, bad)
        assert e.value.status == 404
    assert (await eng.delete_kept(ALICE, oldest["id"]))["removed"] == 1
    assert [k["id"] for k in _kept(eng)] == [newest["id"]]
    await eng.delete_kept(ALICE, newest["id"])
    assert _kept(eng) == [] and not (eng.root / kept_history.KEPT_DIR / ALICE).exists()
    with pytest.raises(LinkError):
        await eng.chats(ALICE, with_kept=True)


async def test_one_users_history_is_out_of_anothers_reach(eng):
    a = await _link(eng)
    await a.sink("message", _row("M1"))
    await eng.unlink(ALICE)
    entry = _kept(eng)[0]["id"]
    with pytest.raises(LinkError):
        await eng.chats(BOB, with_kept=True)
    with pytest.raises(LinkError):
        await eng.delete_kept(BOB, entry)
    assert await eng.news(BOB, agent="x") == {"connected": False}
    assert _kept(eng, BOB) == [] and len(_kept(eng)) == 1


async def test_the_retention_chosen_goes_on_counting(eng):
    now = int(time.time())
    a = await _link(eng)
    await a.sink("message", _row("OLD", ts=now - 100 * 86400))
    await a.sink("message", _row("NEW", ts=now))
    await eng.set_retention(ALICE, 90)  # takes OLD at once
    await a.sink("message", _row("MID", ts=now - 80 * 86400))
    await eng.unlink(ALICE)
    assert _kept(eng)[0]["retention_days"] == 90 and _kept(eng)[0]["stats"]["messages"] == 2
    # Eleven days later: MID is past 90 days, and the card's count follows.
    eng._clock = lambda: now + 11 * 86400
    await eng._purge_kept()
    assert _kept(eng)[0]["stats"]["messages"] == 1
    # When the last of it is past, the entry goes.
    eng._clock = lambda: now + 91 * 86400
    await eng._purge_kept()
    assert _kept(eng) == []


async def test_keep_everything_is_never_purged(eng):
    now = int(time.time())
    a = await _link(eng)
    await a.sink("message", _row("OLD", ts=now - 4000 * 86400))
    await eng.unlink(ALICE)
    await eng._purge_kept()
    assert _kept(eng)[0]["stats"]["messages"] == 1


async def test_the_auto_reply_log_doesnt_travel(eng):
    a = await _link(eng)
    await a.sink("message", _row("M1"))
    await eng.add_autoreply_log(ALICE, chat=ANA, outcome="sent", details={"steps": []})
    await eng.unlink(ALICE)
    _, store = eng._kept_stores(ALICE)[0]
    assert store.autoreply_log() == []


# ── the service's API ───────────────────────────────────────────────


async def test_the_api(eng):
    a = await _link(eng)
    await a.sink("message", _row("M1"))
    c = TestClient(build_app(eng, "s"))
    h = {"Authorization": "Bearer s"}
    assert c.post(f"/links/{ALICE}/disconnect", json={"keep_history": True}, headers=h).json()["state"] == "off"
    assert c.get(f"/links/{ALICE}/chats", headers=h).status_code == 409
    r = c.get(f"/links/{ALICE}/chats?kept=true", headers=h).json()
    assert r["chats"][0]["jid"] == ANA and r["kept"][0]["number"] == "+385911110000" and r["state"] == "off"
    assert c.post(f"/links/{ALICE}/messages", json={}, headers=h).status_code == 409
    assert c.post(f"/links/{ALICE}/messages", json={"kept": True}, headers=h).json()["messages"][0]["id"] == "M1"
    entry = r["kept"][0]["id"]
    assert c.delete(f"/links/{ALICE}/kept/{entry}").status_code == 401
    assert c.delete(f"/links/{ALICE}/kept/{entry}", headers=h).json() == {"removed": 1}
    assert c.delete(f"/links/{ALICE}/kept/{entry}", headers=h).status_code == 404
    # The old way in is gone, so it can't delete what someone meant to keep.
    assert c.delete(f"/links/{ALICE}", headers=h).status_code in (404, 405)


# ── Flight Deck's routes ────────────────────────────────────────────


class Svc:
    """A stand-in for the service, keeping real files: answers, records, and can
    be down or too old."""

    def __init__(self, root):
        self.root = root
        self.calls = []
        self.down = False
        self.too_old = False

    user_has_link = sup.Supervisor.user_has_link
    offline_view = sup.Supervisor.offline_view
    kept = sup.Supervisor.kept
    has_kept = sup.Supervisor.has_kept
    forget_local = sup.Supervisor.forget_local
    forget_kept_local = sup.Supervisor.forget_kept_local

    async def request(self, method, path, body=None, **_):
        self.calls.append((method, path, body))
        if self.down:
            raise sup.ServiceDown("down")
        if self.too_old:
            raise sup.ServiceError(404, "Not Found")
        if path.endswith("/disconnect"):
            return {"state": "off", "running": False, "removed_from_phone": True}
        if "/kept/" in path:
            return {"removed": 1}
        if "/chats" in path:
            return {"chats": [{"jid": ANA, "title": "Ana"}], "stats": None, "state": "off"}
        return {"state": "connected", "running": True, "messages": []}


def _linked_on_disk(root: Path, uid: str, *, state="connected", messages=1):
    from captain_claw.flight_deck.whatsapp_web.store import Store

    (root / uid).mkdir(parents=True, exist_ok=True)
    LinkState(state=state, jid="385976707736@s.whatsapp.net", linked_at="2026-10-26T09:00:00+00:00").save(
        root / uid / "state.json")
    store = Store(root / uid / "store.sqlite3")
    store.ingest([_row(f"M{i}") for i in range(messages)], live=True)
    store.close()


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


def _as(uid):
    return {"Authorization": f"Bearer {auth.create_access_token(uid, 'user')}"}


def test_disconnect_passes_the_answer_on_and_defaults_to_keeping(deck):
    _db, svc, c = deck
    _linked_on_disk(svc.root, BOB)
    c.delete("/fd/whatsapp-web", headers=_as(BOB))
    assert svc.calls[-1] == ("POST", f"/links/{BOB}/disconnect", {"keep_history": True})
    c.delete("/fd/whatsapp-web?keep_history=false", headers=_as(BOB))
    assert svc.calls[-1] == ("POST", f"/links/{BOB}/disconnect", {"keep_history": False})


def test_a_service_too_old_to_keep_history_changes_nothing(deck):
    _db, svc, c = deck
    _linked_on_disk(svc.root, BOB)
    svc.too_old = True
    r = c.delete("/fd/whatsapp-web", headers=_as(BOB))
    assert r.status_code == 409 and "older" in r.json()["detail"]
    assert (svc.root / BOB / "store.sqlite3").exists() and svc.kept(BOB) == []


def test_with_the_service_down_the_files_do_as_the_user_said(deck):
    _db, svc, c = deck
    svc.down = True
    _linked_on_disk(svc.root, BOB, messages=3)
    v = c.delete("/fd/whatsapp-web", headers=_as(BOB)).json()
    assert v["removed_from_phone"] is False and v["link"]["state"] == "off"
    assert not (svc.root / BOB).exists()
    assert v["kept"][0]["number"] == "+385976707736" and v["kept"][0]["stats"]["messages"] == 3
    _linked_on_disk(svc.root, BOB)
    v = c.delete("/fd/whatsapp-web?keep_history=false", headers=_as(BOB)).json()
    assert not (svc.root / BOB).exists() and len(v["kept"]) == 1


def test_with_the_deck_switched_off_too(deck, monkeypatch):
    _db, svc, c = deck
    _linked_on_disk(svc.root, BOB)
    monkeypatch.setenv("FD_WHATSAPP_WEB", "off")
    v = c.delete("/fd/whatsapp-web", headers=_as(BOB)).json()
    assert not svc.calls and len(v["kept"]) == 1 and not v["available"]
    # The user can still delete their history, and agents read nothing.
    assert c.get("/fd/whatsapp-web/agent/status", headers={"X-Agent-Auth": "tok-bob"}).json() == {"connected": False}
    v = c.delete(f"/fd/whatsapp-web/kept/{v['kept'][0]['id']}", headers=_as(BOB)).json()
    assert v["kept"] == [] and not svc.calls


def test_the_card_lists_what_is_kept_and_only_its_own(deck):
    _db, svc, c = deck
    _linked_on_disk(svc.root, BOB)
    svc.forget_local(BOB)
    v = c.get("/fd/whatsapp-web", headers=_as(BOB)).json()
    assert v["link"]["state"] == "off" and len(v["kept"]) == 1
    entry = v["kept"][0]["id"]
    assert c.get("/fd/whatsapp-web", headers=_as(ALICE)).json()["kept"] == []
    assert c.delete(f"/fd/whatsapp-web/kept/{entry}", headers=_as(ALICE)).status_code == 404
    assert len(svc.kept(BOB)) == 1 and not svc.calls
    c.delete(f"/fd/whatsapp-web/kept/{entry}", headers=_as(BOB))
    assert svc.calls[-1][:2] == ("DELETE", f"/links/{BOB}/kept/{entry}")


def test_with_only_history_kept_agents_still_read_it(deck):
    _db, svc, c = deck
    bob = {"X-Agent-Auth": "tok-bob"}
    _linked_on_disk(svc.root, BOB, state="logged_out")
    assert c.get("/fd/whatsapp-web/agent/status", headers=bob).json() == {"connected": False}
    svc.forget_local(BOB)
    assert c.get("/fd/whatsapp-web/agent/status", headers=bob).json() == {"connected": True}
    assert c.post("/fd/whatsapp-web/agent/chats", headers=bob).json()["chats"][0]["jid"] == ANA
    assert svc.calls[-1][1] == f"/links/{BOB}/chats?kept=true"
    c.post("/fd/whatsapp-web/agent/messages", json={"chat": "Ana"}, headers=bob)
    assert svc.calls[-1][2]["kept"] is True
    # Alice's agent gets none of it.
    assert c.get("/fd/whatsapp-web/agent/status", headers={"X-Agent-Auth": "tok-alice"}).json() == {"connected": False}


def test_the_whitelist_is_picked_from_the_linked_accounts_chats_only(deck):
    _db, svc, c = deck
    _linked_on_disk(svc.root, BOB)
    c.get("/fd/whatsapp-web/sending", headers=_as(BOB))
    assert ("GET", f"/links/{BOB}/chats", None) in svc.calls
