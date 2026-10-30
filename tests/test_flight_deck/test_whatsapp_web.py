"""WhatsApp Web, Phase 0 (docs/whatsapp-web-plan.md): linking and unlinking.

* link_state — phone numbers, JIDs, every event's effect and bell; the QR and
  pairing code never reach state.json;
* neonize_adapter — a handler that raises is logged, not lost; the runtime
  check refuses a missing / different neonize before it could fetch a library;
  the libmagic stand-in;
* engine — through a fake client: pairing, expiry, pairing code, cancel,
  unlink (logout, forget the link), the phone unlinking, "stream replaced", resume
  after a restart, one client per user however often Link is pressed, late
  events from a stopped client ignored;
* api — the bearer secret, errors as HTTP statuses;
* the service process itself — lock, service.json, adoption, a second copy
  exits with 3 (neonize need not be installed: it then serves its reason);
* the routes — the deck switch (admin only, off by default), the kill switch,
  consent, and a user only ever reaching their own link.

WhatsApp is never contacted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from captain_claw.flight_deck import auth
from captain_claw.flight_deck import whatsapp_web_routes as routes
from captain_claw.flight_deck import whatsapp_web_supervisor as sup
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.flight_deck.whatsapp_web import neonize_adapter as na
from captain_claw.flight_deck.whatsapp_web.api import build_app
from captain_claw.flight_deck.whatsapp_web.engine import Engine, LinkError
from captain_claw.flight_deck.whatsapp_web.link_state import (
    LinkState,
    apply_event,
    bare_jid,
    normalize_phone,
    number_of,
)

ALICE = "user-alice"
BOB = "user-bob"
JID = "385976707736:56@s.whatsapp.net"


# ── link_state ──────────────────────────────────────────────────────


def test_phone_numbers_take_the_usual_forms():
    assert normalize_phone("+385 97 670 7736") == "385976707736"
    assert normalize_phone("00385976707736") == "385976707736"
    assert normalize_phone("(385) 97-670-7736") == "385976707736"
    for bad in ("", "abc", "12345", "0976707736", "+1234567890123456"):
        with pytest.raises(ValueError):
            normalize_phone(bad)


def test_jids():
    assert bare_jid(JID) == "385976707736@s.whatsapp.net"
    assert number_of(bare_jid(JID)) == "+385976707736"
    assert number_of("200317291462663@lid") == ""
    assert number_of("") == ""


def test_pairing_to_connected_and_the_transient_fields(tmp_path):
    st = LinkState(state="pairing")
    assert apply_event(st, "qr", {"code": "2@abc"}, now=100) is None
    assert st.qr == "2@abc" and st.state == "pairing"
    apply_event(st, "pair_code", {"code": "ABCD-EFGH"}, now=101)
    assert st.view()["pair_code"] == "ABCD-EFGH"
    apply_event(st, "paired", {"jid": JID}, now=102)
    assert (st.state, st.jid, st.qr, st.pair_code) == ("connecting", "385976707736@s.whatsapp.net", "", "")
    apply_event(st, "connected", now=103)
    apply_event(st, "history", {"sync_type": "RECENT", "progress": 9, "messages": 5000}, now=104)
    apply_event(st, "history", {"sync_type": "FULL", "progress": 100, "messages": 2643}, now=105)
    assert st.state == "connected"
    assert st.sync["chunks"] == 2 and st.sync["messages"] == 7643 and st.sync["progress"] == 100

    st.qr, st.pair_code = "should-not-be-saved", "NOPE-NOPE"
    path = tmp_path / "state.json"
    st.save(path)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    saved = json.loads(path.read_text())
    assert "qr" not in saved and "pair_code" not in saved and "qr_at" not in saved
    again = LinkState.load(path)
    assert again.jid == st.jid and again.qr == "" and again.sync["messages"] == 7643


def test_problem_events_ring_bells_and_plain_ones_dont():
    for name in ("qr", "paired", "connected", "disconnected", "history", "expired"):
        assert apply_event(LinkState(state="connected"), name, {}, now=1) is None
    for name in ("logged_out", "stream_replaced", "temporary_ban", "outdated"):
        st = LinkState(state="connected", jid="385976707736@s.whatsapp.net")
        notice = apply_event(st, name, {"expires_at": 10_000.0}, now=1)
        assert notice is not None and notice.title, name
        assert st.state == name
    st = LinkState(state="connected")
    apply_event(st, "temporary_ban", {"expires_at": 86400.0}, now=1)
    assert st.ban_expires_at.startswith("1970-01-02")


def test_disconnect_only_marks_a_connected_link():
    st = LinkState(state="connecting")
    apply_event(st, "disconnected", now=1)
    assert st.state == "connecting"
    st.state = "connected"
    apply_event(st, "disconnected", now=2)
    assert st.state == "disconnected"


def test_a_corrupt_or_unknown_state_file(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("not json")
    assert LinkState.load(p) is None
    p.write_text(json.dumps({"state": "weird", "jid": "x", "unknown_field": 1}))
    st = LinkState.load(p)
    assert st.state == "failed" and st.jid == "x"


# ── neonize_adapter ─────────────────────────────────────────────────


async def test_a_handler_that_raises_is_logged_not_lost(caplog):
    async def boom(*_):
        raise AttributeError("PushName")

    with caplog.at_level(logging.ERROR, logger="whatsapp_web.neonize"):
        await na.guarded("message", boom)(object(), object())
    assert "AttributeError: PushName" in caplog.text


def test_runtime_check_refuses_before_neonize_could_download(monkeypatch, tmp_path):
    monkeypatch.setattr(na.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(na.RuntimeUnavailable, match="isn't installed"):
        na.check_runtime()

    pkg = tmp_path / "neonize"
    pkg.mkdir()
    (pkg / "download.py").write_text('__GONEONIZE_VERSION__ = "0.5.2"\n')
    spec = type("Spec", (), {"submodule_search_locations": [str(pkg)]})()
    monkeypatch.setattr(na.importlib.util, "find_spec", lambda name: spec)
    import importlib.metadata as md

    monkeypatch.setattr(md, "version", lambda name: "0.4.0")
    with pytest.raises(na.RuntimeUnavailable, match="needs neonize 0.5.2"):
        na.check_runtime()
    monkeypatch.setattr(md, "version", lambda name: na.NEONIZE_VERSION)
    with pytest.raises(na.RuntimeUnavailable, match="no native library"):
        na.check_runtime()  # rather than letting neonize fetch one from GitHub


def test_magic_stand_in(monkeypatch):
    monkeypatch.setitem(sys.modules, "magic", None)  # makes `import magic` fail
    na._ensure_magic()
    import magic

    assert magic.from_buffer(b"\xff\xd8\xff\xe0rest", mime=True) == "image/jpeg"
    assert magic.from_buffer(b"\x89PNG\r\n\x1a\n", mime=True) == "image/png"
    assert magic.from_buffer(b"OggS\x00", mime=True) == "audio/ogg"
    assert magic.from_buffer(b"\x00\x00\x00\x18ftypmp42", mime=True) == "video/mp4"
    assert magic.from_buffer(b"RIFF\x00\x00\x00\x00WEBPVP8", mime=True) == "image/webp"
    assert magic.from_buffer(b"plain", mime=True) == "application/octet-stream"


def test_ban_expiry_units():
    assert na._ban_expiry(3600, 1000.0) == 4600.0
    assert na._ban_expiry(3600 * 10**9, 1000.0) == 4600.0  # Go time.Duration, ns
    assert na._ban_expiry(0, 1000.0) is None
    assert na._ban_expiry("x", 1000.0) is None


# ── engine, through a fake client ───────────────────────────────────


class FakeAdapter:
    instances: list[FakeAdapter] = []

    def __init__(self, client_id, session_path, sink, *, history_days):
        self.client_id, self.session_path, self.sink = client_id, Path(session_path), sink
        self.history_days = history_days
        self.calls: list[str] = []
        self.fail_logout = False
        FakeAdapter.instances.append(self)

    async def connect(self):
        self.calls.append("connect")
        self.session_path.write_bytes(b"whatsmeow")  # whatsmeow creates its db

    async def pair_code(self, phone):
        self.calls.append(f"pair_code:{phone}")
        return "ABCD-EFGH"

    async def logout(self):
        self.calls.append("logout")
        if self.fail_logout:
            raise RuntimeError("not connected")

    async def stop(self):
        self.calls.append("stop")

    async def emit(self, name, data=None):
        await self.sink(name, data or {})


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture()
def fake():
    FakeAdapter.instances.clear()
    yield FakeAdapter
    FakeAdapter.instances.clear()


def _engine(tmp_path, clock=None, **kw):
    return Engine(tmp_path / "waw", adapter_factory=FakeAdapter, clock=clock or Clock(), **kw)


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0)


async def test_link_pair_connect(tmp_path, fake):
    eng = _engine(tmp_path)
    assert eng.view(ALICE)["state"] == "off"
    v = await eng.link(ALICE, consented_at="2026-10-26T09:00:00+00:00")
    assert v["state"] == "pairing" and v["running"]
    a = fake.instances[0]
    assert a.client_id == f"cc-{ALICE}" and a.history_days == 3650
    assert a.session_path == tmp_path / "waw" / ALICE / "session.sqlite3"
    await a.emit("qr", {"code": "2@qr-payload"})
    v = eng.view(ALICE)
    assert v["state"] == "pairing" and "pairing_expires_at" in v
    try:
        import segno  # noqa: F401
        assert v["qr_svg"].startswith("data:image/svg+xml")
    except ImportError:
        pass
    await a.emit("paired", {"jid": JID})
    await a.emit("connected")
    v = eng.view(ALICE)
    assert v["state"] == "connected" and v["number"] == "+385976707736" and v["has_session"]
    assert "qr_svg" not in v
    saved = json.loads((tmp_path / "waw" / ALICE / "state.json").read_text())
    assert saved["consented_at"] == "2026-10-26T09:00:00+00:00"


async def test_pressing_link_twice_starts_one_client(tmp_path, fake):
    eng = _engine(tmp_path)
    await asyncio.gather(*(eng.link(ALICE, consented_at="t") for _ in range(4)))
    assert len(fake.instances) == 1
    await eng.link(ALICE, consented_at="t")
    assert len(fake.instances) == 1


async def test_users_get_separate_links(tmp_path, fake):
    eng = _engine(tmp_path)
    await eng.link(ALICE, consented_at="t")
    await eng.link(BOB, consented_at="t")
    assert {a.client_id for a in fake.instances} == {f"cc-{ALICE}", f"cc-{BOB}"}
    await fake.instances[0].emit("paired", {"jid": JID})
    assert eng.view(ALICE)["state"] == "connecting"
    assert eng.view(BOB)["state"] == "pairing"


async def test_pairing_expires(tmp_path, fake, monkeypatch):
    clock = Clock()
    eng = _engine(tmp_path, clock, pairing_ttl=60)
    real_sleep = asyncio.sleep
    monkeypatch.setattr("captain_claw.flight_deck.whatsapp_web.engine.asyncio.sleep",
                        lambda s: real_sleep(0))
    await eng.link(ALICE, consented_at="t")
    a = fake.instances[0]
    clock.t += 30
    await _settle()
    assert eng.view(ALICE)["state"] == "pairing"
    clock.t += 31
    await _settle()
    v = eng.view(ALICE)
    assert v["state"] == "expired" and not v["running"] and not v["has_session"]
    assert "stop" in a.calls
    await eng.cancel(ALICE)
    assert eng.view(ALICE)["state"] == "off"


async def test_pair_code(tmp_path, fake):
    eng = _engine(tmp_path)
    with pytest.raises(LinkError) as e:
        await eng.pair_code(ALICE, "+385 97 670 7736")
    assert e.value.status == 409
    await eng.link(ALICE, consented_at="t")
    with pytest.raises(LinkError) as e:
        await eng.pair_code(ALICE, "12")
    assert e.value.status == 422
    v = await eng.pair_code(ALICE, "+385 97 670 7736")
    assert v["pair_code"] == "ABCD-EFGH"
    assert "pair_code:385976707736" in fake.instances[0].calls


async def test_cancel_only_unpaired(tmp_path, fake):
    eng = _engine(tmp_path)
    await eng.link(ALICE, consented_at="t")
    await eng.cancel(ALICE)
    assert eng.view(ALICE)["state"] == "off" and not (tmp_path / "waw" / ALICE).exists()
    await eng.link(ALICE, consented_at="t")
    await fake.instances[-1].emit("paired", {"jid": JID})
    with pytest.raises(LinkError):
        await eng.cancel(ALICE)


async def test_unlink_logs_out_and_deletes(tmp_path, fake):
    eng = _engine(tmp_path)
    await eng.link(ALICE, consented_at="t")
    a = fake.instances[0]
    await a.emit("paired", {"jid": JID})
    await a.emit("connected")
    v = await eng.unlink(ALICE)
    assert v["state"] == "off" and v["removed_from_phone"] is True
    assert a.calls[-2:] == ["logout", "stop"]
    assert not (tmp_path / "waw" / ALICE).exists()
    assert eng.events_after(0) == []  # our own logout rings no bell


async def test_unlink_when_logout_fails_still_deletes(tmp_path, fake):
    eng = _engine(tmp_path)
    await eng.link(ALICE, consented_at="t")
    a = fake.instances[0]
    await a.emit("paired", {"jid": JID})
    await a.emit("connected")
    a.fail_logout = True
    v = await eng.unlink(ALICE)
    assert v["removed_from_phone"] is False and not (tmp_path / "waw" / ALICE).exists()


async def test_unlinked_on_the_phone(tmp_path, fake):
    eng = _engine(tmp_path)
    await eng.link(ALICE, consented_at="t")
    a = fake.instances[0]
    await a.emit("paired", {"jid": JID})
    await a.emit("connected")
    await a.emit("logged_out", {"reason": "401"})
    await _settle()
    v = eng.view(ALICE)
    assert v["state"] == "logged_out" and not v["running"] and not v["has_session"]
    assert "stop" in a.calls
    bells = eng.events_after(0)
    assert len(bells) == 1 and bells[0]["user_id"] == ALICE and "unlinked" in bells[0]["title"]
    assert eng.events_after(bells[0]["seq"]) == []
    # Link again starts over with a fresh client.
    await eng.link(ALICE, consented_at="t2")
    assert len(fake.instances) == 2 and eng.view(ALICE)["state"] == "pairing"


async def test_stream_replaced_stops_and_stays_stopped(tmp_path, fake):
    eng = _engine(tmp_path)
    await eng.link(ALICE, consented_at="t")
    a = fake.instances[0]
    await a.emit("paired", {"jid": JID})
    await a.emit("connected")
    await a.emit("stream_replaced")
    await _settle()
    assert eng.view(ALICE)["state"] == "stream_replaced" and "stop" in a.calls

    again = _engine(tmp_path)  # a service restart
    assert await again.resume_all() == 0  # resuming would fight the other client
    v = await again.reconnect(ALICE)  # unless the user asks
    assert v["state"] == "connecting" and len(fake.instances) == 2


async def test_resume_after_a_restart(tmp_path, fake):
    eng = _engine(tmp_path)
    await eng.link(ALICE, consented_at="t")
    await fake.instances[0].emit("paired", {"jid": JID})
    await fake.instances[0].emit("connected")
    await eng.link(BOB, consented_at="t")  # still pairing when the service stops
    await eng.shutdown()
    assert {"stop"} <= set(fake.instances[0].calls) and {"stop"} <= set(fake.instances[1].calls)

    again = _engine(tmp_path)
    assert await again.resume_all() == 1
    assert again.view(ALICE)["state"] == "connecting" and again.view(ALICE)["running"]
    assert again.view(BOB)["state"] == "off"  # its QR is long gone
    assert not (tmp_path / "waw" / BOB).exists()


async def test_late_events_from_a_stopped_client_are_ignored(tmp_path, fake):
    eng = _engine(tmp_path)
    await eng.link(ALICE, consented_at="t")
    old = fake.instances[0]
    await eng.cancel(ALICE)
    await eng.link(ALICE, consented_at="t")
    await old.emit("paired", {"jid": "999@s.whatsapp.net"})
    assert eng.view(ALICE)["state"] == "pairing" and eng.view(ALICE)["jid"] == ""


async def test_refusals(tmp_path, fake):
    eng = _engine(tmp_path)
    for bad in ("../etc", "..", ".", ".hidden", "a/b", ""):
        with pytest.raises(LinkError) as e:
            await eng.link(bad, consented_at="t")
        assert e.value.status == 400, bad
    off = Engine(tmp_path / "x", adapter_factory=None, unavailable="no neonize here")
    with pytest.raises(LinkError) as e:
        await off.link(ALICE, consented_at="t")
    assert (e.value.status, e.value.detail) == (503, "no neonize here")
    with pytest.raises(LinkError) as e:
        await eng.reconnect(ALICE)
    assert e.value.status == 409


async def test_a_client_that_fails_to_start(tmp_path, fake):
    class Broken(FakeAdapter):
        async def connect(self):
            raise RuntimeError("websocket refused")

    eng = Engine(tmp_path / "waw", adapter_factory=Broken)
    with pytest.raises(LinkError) as e:
        await eng.link(ALICE, consented_at="t")
    assert e.value.status == 502
    v = eng.view(ALICE)
    assert v["state"] == "failed" and not v["running"] and "websocket refused" in v["detail"]


# ── api ─────────────────────────────────────────────────────────────


def test_api_needs_the_secret_and_maps_errors(tmp_path, fake):
    eng = _engine(tmp_path)
    client = TestClient(build_app(eng, "s3cret", version="0.5.2"))
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"Authorization": "Bearer nope"}).status_code == 401
    h = {"Authorization": "Bearer s3cret"}
    health = client.get("/health", headers=h).json()
    assert health["ok"] and health["available"] and health["neonize"] == "0.5.2"
    r = client.post(f"/links/{ALICE}/start", json={"consented_at": "t"}, headers=h)
    assert r.status_code == 200 and r.json()["state"] == "pairing"
    r = client.post(f"/links/{ALICE}/pair-code", json={"phone": "1"}, headers=h)
    assert r.status_code == 422
    for bad in ("..", ".hidden", "..bad", "a%2Fb"):
        assert client.get(f"/links/{bad}", headers=h).status_code in (400, 404), bad
    assert client.post(f"/links/{ALICE}/disconnect", json={}, headers=h).json()["state"] == "off"
    assert client.get("/events?after=0", headers=h).json() == {"events": []}


# ── the service process ─────────────────────────────────────────────


def _service_env(root: Path) -> dict[str, str]:
    code_root = str(Path(__file__).resolve().parents[2])
    return {**os.environ, "WAW_DATA_DIR": str(root),
            "PYTHONPATH": code_root + os.pathsep + os.environ.get("PYTHONPATH", "")}


async def test_service_process_lock_and_adoption(tmp_path):
    db = type("DB", (), {"add_notification": None})()
    s = sup.Supervisor(tmp_path, db)
    s._secret()
    root = tmp_path / "whatsapp_web"
    first = subprocess.Popen([sys.executable, "-m", "captain_claw.flight_deck.whatsapp_web"],
                             env=_service_env(root), cwd=str(root),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not (root / "service.json").exists():
            await asyncio.sleep(0.2)
        info = json.loads((root / "service.json").read_text())
        assert info["pid"] == first.pid
        assert stat.S_IMODE(os.stat(root / "service.secret").st_mode) == 0o600
        assert await s._adopt()  # what a restarted Flight Deck does
        health = await s.request("GET", "/health")
        assert health["pid"] == first.pid and isinstance(health["available"], bool)

        second = subprocess.run([sys.executable, "-m", "captain_claw.flight_deck.whatsapp_web"],
                                env=_service_env(root), cwd=str(root), timeout=30,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert second.returncode == 3  # the lock: never two clients on one session

        await s.stop()
        assert first.wait(timeout=15) is not None
        assert not (root / "service.json").exists()
    finally:
        if first.poll() is None:
            first.kill()


# ── routes ──────────────────────────────────────────────────────────


class FakeSupervisor:
    def __init__(self):
        self.calls: list[tuple] = []
        self.links: set[str] = set()
        self.stopped = False

    def user_has_link(self, uid):
        return uid in self.links

    def any_links(self):
        return bool(self.links)

    def running(self):
        return True

    def offline_view(self, uid):
        return {"state": "connecting", "running": False}

    def kept(self, uid):
        return []

    def has_kept(self, uid):
        return False

    def forget_local(self, uid, *, keep_history=True):
        self.calls.append(("forget", uid))
        self.links.discard(uid)

    async def stop(self):
        self.stopped = True

    async def ensure_running(self):
        self.calls.append(("ensure",))

    async def request(self, method, path, body=None, **_):
        self.calls.append((method, path, body))
        if path == "/health":
            return {"ok": True, "available": True, "reason": "", "neonize": "0.5.2", "links": {}}
        uid = path.split("/")[2]
        if method == "POST" and path.endswith("/start"):
            self.links.add(uid)
            return {"state": "pairing", "running": True}
        if path.endswith("/disconnect"):
            self.links.discard(uid)
            return {"state": "off", "running": False, "removed_from_phone": True}
        return {"state": "connected", "running": True}


@pytest.fixture()
async def deck(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.delenv("FD_WHATSAPP_WEB", raising=False)
    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    d = FlightDeckDB(str(tmp_path / "fd.db"))
    await d.init()
    prev = auth._db
    auth.set_auth_db(d)
    now = "2026-01-01T00:00:00Z"
    for uid, email, role in [(ALICE, "alice@x.co", "admin"), (BOB, "bob@x.co", "user")]:
        await d._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)", (uid, email, "h", uid.title(), role, now, now))
    await d._db.commit()
    fs = FakeSupervisor()
    monkeypatch.setattr(sup, "get", lambda: fs)
    monkeypatch.setattr(sup, "installed", lambda: True)
    app = FastAPI()
    app.include_router(routes.router)
    try:
        yield d, fs, TestClient(app, client=("127.0.0.1", 50123))
    finally:
        auth.set_auth_db(prev)
        await d.close()


def _as(uid, role="user"):
    return {"Authorization": f"Bearer {auth.create_access_token(uid, role)}"}


async def test_off_until_an_admin_allows_it(deck):
    _db, fs, c = deck
    v = c.get("/fd/whatsapp-web", headers=_as(BOB)).json()
    assert not v["available"] and "admin hasn't allowed" in v["reason"] and not v["can_admin"]
    assert c.post("/fd/whatsapp-web/link", json={"consent": True}, headers=_as(BOB)).status_code == 409
    assert c.put("/fd/admin/whatsapp-web", json={"allowed": True}, headers=_as(BOB)).status_code == 403
    r = c.put("/fd/admin/whatsapp-web", json={"allowed": True}, headers=_as(ALICE, "admin"))
    assert r.status_code == 200 and r.json()["allowed"] is True
    v = c.get("/fd/whatsapp-web", headers=_as(BOB)).json()
    assert v["available"] and v["link"]["state"] == "off"


async def test_link_needs_consent_and_reaches_only_your_own(deck):
    db, fs, c = deck
    await sup.save_config(db, allowed=True)
    assert c.post("/fd/whatsapp-web/link", json={}, headers=_as(BOB)).status_code == 422
    v = c.post("/fd/whatsapp-web/link", json={"consent": True}, headers=_as(BOB)).json()
    assert v["link"]["state"] == "pairing"
    method, path, body = fs.calls[-1]
    assert (method, path) == ("POST", f"/links/{BOB}/start") and body["consented_at"]
    c.post("/fd/whatsapp-web/pair-code", json={"phone": "+385 97 670 7736"}, headers=_as(BOB))
    assert fs.calls[-1][:2] == ("POST", f"/links/{BOB}/pair-code")
    assert c.get("/fd/whatsapp-web", headers=_as(ALICE)).json()["link"]["state"] == "off"
    v = c.delete("/fd/whatsapp-web", headers=_as(BOB)).json()
    assert v["removed_from_phone"] is True and fs.calls[-1][:2] == ("POST", f"/links/{BOB}/disconnect")


async def test_kill_switch_and_turning_it_off(deck, monkeypatch):
    db, fs, c = deck
    await sup.save_config(db, allowed=True)
    fs.links.add(BOB)
    monkeypatch.setenv("FD_WHATSAPP_WEB", "off")
    v = c.get("/fd/whatsapp-web", headers=_as(BOB)).json()
    assert not v["available"] and "FD_WHATSAPP_WEB" in v["reason"]
    assert v["link"]["running"] is False  # read from disk, the service isn't asked
    monkeypatch.delenv("FD_WHATSAPP_WEB")
    c.put("/fd/admin/whatsapp-web", json={"allowed": False}, headers=_as(ALICE, "admin"))
    assert fs.stopped
    # Disconnecting while it's off still forgets the link.
    v = c.delete("/fd/whatsapp-web", headers=_as(BOB)).json()
    assert v["removed_from_phone"] is False and ("forget", BOB) in fs.calls


async def test_not_installed(deck, monkeypatch):
    db, _fs, c = deck
    await sup.save_config(db, allowed=True)
    monkeypatch.setattr(sup, "installed", lambda: False)
    v = c.get("/fd/whatsapp-web", headers=_as(BOB)).json()
    assert not v["available"] and "pip install" in v["reason"]


async def test_config_defaults_off(deck):
    db, _fs, _c = deck
    assert (await sup.load_config(db)) == {"allowed": False}
    await db.set_system_setting(sup.SETTING, "garbage")
    assert (await sup.load_config(db)) == {"allowed": False}
