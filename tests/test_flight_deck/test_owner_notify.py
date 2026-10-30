"""Nudges and proposals waiting for approval reach the owner on every channel
they switched on (owner_notify.py): WhatsApp (``nudge_to_whatsapp``), their
Telegram chat with the deck's bot (``notify_telegram``) and their connected
devices (``notify_devices``), which beep and show how many proposals wait —
and always to the bell, where a proposal is approved or rejected (the kiosk
has nowhere else). A device asleep when something came beeps for it once, when it connects; the
count follows the ledger with silent updates; the switches live behind
/fd/autonomy/notify and survive a whole-config save."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from captain_claw.config import AutonomousWorkConfig
from captain_claw.flight_deck import (
    action_catalog,
    arbiter,
    autonomy,
    autonomy_routes,
    basna_routes,
    device_bridge,
    events,
    fd_dispatch,
    fd_scheduler,
    owner_notify,
    whatsapp_bridge,
)
from captain_claw.flight_deck.auth import get_optional_user

ALICE, BOB = "user-alice", "user-bob"


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = autonomy.AutonomyStore(tmp_path / "autonomy.db")
    monkeypatch.setattr(autonomy, "_STORE", s)
    base = AutonomousWorkConfig().model_dump()
    monkeypatch.setattr(autonomy, "global_defaults", lambda: dict(base))
    monkeypatch.setattr(autonomy_routes, "global_defaults", lambda: dict(base))
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("WHATSAPP_ALLOWED_WAIDS", "111")
    monkeypatch.delenv("FD_PUBLIC_URL", raising=False)
    for name in ("_last_event", "_device_heard", "_sent_count"):
        monkeypatch.setattr(owner_notify, name, {})
    monkeypatch.setattr(autonomy_routes, "_last_notify_test_at", {})
    # Never in quiet hours unless a test says so.
    s.set_overrides(ALICE, {"quiet_hours_start": 0, "quiet_hours_end": 0})
    return s


@pytest.fixture
def channels(monkeypatch):
    """Stub every channel: ``channels["whatsapp"|"telegram"|"devices"]`` list what went out."""
    out: dict[str, list] = {"whatsapp": [], "telegram": [], "devices": [], "bell": []}

    async def fake_push(waid, text):
        out["whatsapp"].append((waid, text))
        return True

    async def fake_telegram(owner, text):
        out["telegram"].append((owner, text))
        return {"via": "telegram", "ok": True, "note": "sent"}

    async def fake_devices(user_id, count, text="", beep=False):
        out["devices"].append({"user": user_id, "count": count, "text": text, "beep": beep})
        return ["paper-1"]

    monkeypatch.setattr(whatsapp_bridge, "push_to_waid", fake_push)
    monkeypatch.setattr(fd_scheduler, "_deliver_telegram_owner", fake_telegram)
    class _DB:
        async def add_notification(self, user_id, type, title, body="", ref_type="", ref_id=""):
            out["bell"].append({"user": user_id, "type": type, "title": title, "body": body,
                                "ref_type": ref_type, "ref_id": ref_id})
            return "n1"

    from captain_claw.flight_deck import auth
    monkeypatch.setattr(device_bridge, "push_waiting", fake_devices)
    monkeypatch.setattr(auth, "get_db", lambda: _DB())
    return out


def _run_coro(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _settings(store, uid, **kw):
    store.set_overrides(uid, {**store.get_overrides(uid), **kw})


# ── nudges ────────────────────────────────────────────────────────────

def _nudge(store, monkeypatch, uid=ALICE, output="Your 3pm call is in an hour."):
    async def fake_dispatch(port, auth, instruction, timeout, **kw):
        return {"ok": True, "output": output}

    monkeypatch.setattr(basna_routes, "_dispatch_one", fake_dispatch)
    row = store.add_action(uid, kind="nudge", title="Remind about the 3pm call",
                           risk="low", status="dispatched")
    _run_coro(fd_dispatch._execute_and_judge(uid, row, {"slug": "a", "port": 1, "auth": ""}))
    return store.get_action(row["id"])


def test_a_nudge_reaches_every_channel_switched_on(store, channels, monkeypatch):
    _settings(store, ALICE, notify_waid="111", notify_telegram=True)
    done = _nudge(store, monkeypatch)
    assert done["status"] == "done" and done["outcome"] == "success"
    assert channels["whatsapp"] == [("111", "Your 3pm call is in an hour.")]
    assert channels["telegram"] == [(ALICE, "Your 3pm call is in an hour.")]
    assert channels["devices"][-1] == {"user": ALICE, "count": 0, "beep": True,
                                       "text": "Your 3pm call is in an hour."}
    assert channels["bell"] == [{"user": ALICE, "type": "autonomy_nudge", "title": "Nudge from a",
                                 "body": "Your 3pm call is in an hour.",
                                 "ref_type": "autonomy_nudge", "ref_id": done["id"]}]
    events_ = [r["event"] for r in store.list_log(ALICE)]
    assert {"nudge → whatsapp", "nudge → telegram", "nudge → devices"} <= set(events_)


def test_shipped_defaults_are_whatsapp_and_devices_not_telegram():
    cfg = AutonomousWorkConfig()
    assert cfg.nudge_to_whatsapp is True and cfg.notify_devices is True
    assert cfg.notify_telegram is False


def test_channels_switched_off_get_nothing(store, channels, monkeypatch):
    _settings(store, ALICE, notify_waid="111", nudge_to_whatsapp=False,
              notify_telegram=False, notify_devices=False)
    _nudge(store, monkeypatch)
    assert (channels["whatsapp"], channels["telegram"], channels["devices"]) == ([], [], [])
    assert len(channels["bell"]) == 1  # the bell is Flight Deck itself: no switch


def test_quiet_hours_show_the_count_without_a_beep(store, channels, monkeypatch):
    _settings(store, ALICE, quiet_hours_start=0, quiet_hours_end=24)
    monkeypatch.setattr(arbiter, "_in_quiet_hours", lambda s, e: True)
    _nudge(store, monkeypatch)
    assert channels["devices"][-1]["beep"] is False


def test_device_line_is_one_plain_line():
    line = owner_notify.device_line("**Ana** 👋 is waiting for a reply\nmore below")
    assert line == "Ana is waiting for a reply"
    assert len(owner_notify.device_line("x " * 400)) <= owner_notify.DEVICE_LINE_MAX


# ── proposals ─────────────────────────────────────────────────────────

def test_a_proposal_says_how_many_wait(store, channels, monkeypatch):
    monkeypatch.setenv("FD_PUBLIC_URL", "https://deck.example")
    _settings(store, ALICE, notify_waid="111", notify_telegram=True)
    store.add_action(ALICE, kind="nudge", title="Older one", status="awaiting_approval")
    row = store.add_action(ALICE, kind="run_prompt", title="Summarise the Q3 deck",
                           rationale="You asked for it before the board call.", status="awaiting_approval")
    _run_coro(owner_notify.notify_proposal(ALICE, row))
    text = channels["telegram"][0][1]
    assert text == ("Waiting for your approval in Flight Deck (2 waiting): Summarise the Q3 deck"
                    "\nhttps://deck.example")
    assert channels["whatsapp"][0][1] == text
    assert channels["devices"][-1] == {"user": ALICE, "count": 2, "beep": True,
                                       "text": "Waiting for your approval: Summarise the Q3 deck"}
    assert channels["bell"] == [{"user": ALICE, "type": "autonomy_proposal", "title": "Waiting for your approval",
                                 "body": "Summarise the Q3 deck\nYou asked for it before the board call.",
                                 "ref_type": "autonomy_action", "ref_id": row["id"]}]


@pytest.fixture
def ranker(monkeypatch):
    """The ranking LLM proposes one nudge (as in test_autonomy_pending_proposals)."""
    import captain_claw.games.remote_provider as rp

    class _Resp:
        def __init__(self, content):
            self.content = content

    class _Provider:
        def __init__(self, **kw):
            pass

        async def complete(self, **kw):
            return _Resp(json.dumps([{"kind": "nudge", "title": "Check in about the Q3 report",
                                      "rationale": "r", "risk": "low", "domain": "general",
                                      "score": 0.8}]))

    def _no_events():
        raise RuntimeError("no events store in this test")

    async def _no_runs(uid):
        return []

    monkeypatch.setattr(rp, "RemoteLLMProvider", _Provider)
    monkeypatch.setattr(events, "get_store", _no_events)
    monkeypatch.setattr(arbiter, "_gather_active_runs", _no_runs)
    monkeypatch.setattr(action_catalog, "list_catalog", lambda **kw: [])


def test_the_arbiter_tells_the_owner_about_a_new_proposal(store, channels, ranker, monkeypatch):
    _settings(store, ALICE, enabled=True, autonomy_level="propose", notify_telegram=True)
    res = _run_coro(arbiter.maybe_run_arbiter(
        ALICE, {"intentions": ["Check in about the Q3 report"]},
        {"host": "localhost", "port": 1, "auth": "", "name": "a"}, [], trigger="pulse"))
    assert res.get("proposed") == 1 and not res.get("dispatched"), res
    assert channels["telegram"] == [
        (ALICE, "Waiting for your approval in Flight Deck: Check in about the Q3 report")]
    assert channels["devices"][-1]["count"] == 1 and channels["devices"][-1]["beep"] is True


# ── devices ───────────────────────────────────────────────────────────

def test_a_device_that_slept_beeps_once_when_it_connects(store, channels):
    _run_coro(owner_notify.notify_owner(ALICE, "nudge", "Ana is waiting for a reply"))
    store.add_action(ALICE, kind="nudge", title="t", status="awaiting_approval")
    first = owner_notify.device_connected(ALICE, "watch-1")
    assert first == {"type": "waiting", "count": 1, "text": "Ana is waiting for a reply", "beep": True}
    again = owner_notify.device_connected(ALICE, "watch-1")
    assert again == {"type": "waiting", "count": 1, "text": "", "beep": False}
    # The device that was connected when it came already beeped for it.
    assert owner_notify.device_connected(ALICE, "paper-1")["beep"] is False


def test_an_old_event_only_shows_the_count(store, channels):
    owner_notify._last_event[ALICE] = (time.time() - owner_notify.UNSEEN_KEEP_S - 5, "old news")
    assert owner_notify.device_connected(ALICE, "watch-1") == {
        "type": "waiting", "count": 0, "text": "", "beep": False}


def test_devices_switched_off_get_no_frame(store):
    _settings(store, ALICE, notify_devices=False)
    assert owner_notify.device_connected(ALICE, "watch-1") is None


@pytest.mark.asyncio
async def test_the_count_follows_the_ledger_silently(store, channels):
    row = store.add_action(ALICE, kind="nudge", title="t", status="awaiting_approval")
    await asyncio.sleep(0)
    assert channels["devices"][-1] == {"user": ALICE, "count": 1, "text": "", "beep": False}
    store.update_status(row["id"], "rejected")
    await asyncio.sleep(0)
    assert channels["devices"][-1]["count"] == 0
    n = len(channels["devices"])
    store.update_status(row["id"], "rejected", outcome_note="again")  # no change in the count
    await asyncio.sleep(0)
    assert len(channels["devices"]) == n


@pytest.mark.asyncio
async def test_push_waiting_reaches_only_the_owners_devices(monkeypatch):
    class _WS:
        def __init__(self):
            self.sent = []

        async def send_text(self, text):
            self.sent.append(json.loads(text))

    def conn(uid, device):
        return device_bridge._DeviceConn(_WS(), {"id": uid}, device, lang="", speak=False,
                                         screen=(240, 135))

    mine, theirs, local = conn(ALICE, "paper-1"), conn(BOB, "watch-2"), conn("local", "stick-3")
    monkeypatch.setattr(device_bridge, "_live", {mine, theirs, local})
    assert await device_bridge.push_waiting(ALICE, 2, "x", True) == ["paper-1"]
    assert mine.ws.sent == [{"type": "waiting", "count": 2, "text": "x", "beep": True}]
    assert theirs.ws.sent == []
    assert device_bridge.connected_devices(ALICE) == 1
    # Auth off: the deck's one user is "local" (or "" from a request).
    assert await device_bridge.push_waiting("", 0) == ["stick-3"]


# ── routes ────────────────────────────────────────────────────────────

@pytest.fixture
def client(store, channels):
    app = FastAPI()
    app.include_router(autonomy_routes.router)

    async def fake_user(request: Request):
        uid = request.headers.get("X-Test-User", "")
        if uid:
            request.state.user_id = uid
            return {"id": uid, "role": "user"}
        return None

    app.dependency_overrides[get_optional_user] = fake_user
    return TestClient(app)


def _as(uid):
    return {"X-Test-User": uid}


def test_notify_state_lists_each_channel(client, store):
    _settings(store, ALICE, notify_waid="111")
    body = client.get("/fd/autonomy/notify", headers=_as(ALICE)).json()
    ch = body["channels"]
    assert ch["whatsapp"]["on"] is True and ch["telegram"]["on"] is False and ch["devices"]["on"] is True
    assert "link it in Connections" in ch["telegram"]["detail"] and ch["telegram"]["ready"] is False
    assert ch["devices"]["connected"] == 0 and body["waiting"] == 0


def test_switches_save_and_survive_a_whole_config_save(client, store, channels):
    r = client.put("/fd/autonomy/notify", headers=_as(ALICE),
                   json={"telegram": True, "devices": False, "whatsapp": False})
    ch = r.json()["channels"]
    assert (ch["whatsapp"]["on"], ch["telegram"]["on"], ch["devices"]["on"]) == (False, True, False)
    assert channels["devices"][-1] == {"user": ALICE, "count": 0, "text": "", "beep": False}  # cleared
    # The Autonomous Work page's Reset (or a stale tab) sends every key back.
    client.put("/fd/autonomy/config", headers=_as(ALICE),
               json={"config": {"notify_telegram": False, "notify_devices": True}})
    cfg = autonomy.resolve_config(ALICE)
    assert cfg["notify_telegram"] is True and cfg["notify_devices"] is False


def test_a_switch_must_be_a_bool(client):
    assert client.put("/fd/autonomy/notify", headers=_as(ALICE),
                      json={"devices": "off"}).status_code == 400


def test_the_test_goes_out_once_a_minute(client, store, channels):
    _settings(store, ALICE, notify_waid="111")
    r = client.post("/fd/autonomy/notify/test", headers=_as(ALICE))
    assert r.status_code == 200 and r.json()["sent"]["devices"] == 1
    assert channels["bell"] == []  # a test isn't kept
    assert channels["devices"][-1]["text"] == "Test: notifications reach this device"
    assert client.post("/fd/autonomy/notify/test", headers=_as(ALICE)).status_code == 429
    # A test isn't news: a device connecting later doesn't beep for it.
    assert owner_notify.device_connected(ALICE, "watch-9")["beep"] is False


def test_with_sign_in_off_the_bell_is_the_local_users(store, channels):
    _run_coro(owner_notify.notify_owner("", "nudge", "Hi", bell=("autonomy_nudge", "Nudge", "Hi"),
                                        action_id="act_1"))
    assert channels["bell"][0]["user"] == "local"


def test_a_failing_bell_never_stops_the_other_channels(store, channels, monkeypatch):
    from captain_claw.flight_deck import auth

    def _no_db():
        raise RuntimeError("no database")

    monkeypatch.setattr(auth, "get_db", _no_db)
    res = _run_coro(owner_notify.notify_owner(ALICE, "nudge", "Hi", bell=("autonomy_nudge", "Nudge", "Hi")))
    assert res["bell"] is False and res["devices"] == 1


def test_the_bell_fills_in_waiting_proposals_it_lacks(store, monkeypatch):
    """Proposals made before the bell held them (or when it couldn't be
    written) get an entry the next time the bell is read — once, read or not."""
    from captain_claw.flight_deck import auth

    rows: list[dict] = []

    class _DB:
        async def add_notification(self, user_id, type, title, body="", ref_type="", ref_id=""):
            rows.append({"user": user_id, "type": type, "ref_type": ref_type, "ref_id": ref_id, "body": body})
            return "n"

        async def count_notifications_since(self, user_id, since, types=(), *, ref_type="", ref_id=""):
            return sum(1 for r in rows if r["user"] == user_id and r["ref_id"] == ref_id
                       and r["ref_type"] == ref_type and r["type"] in ([types] if isinstance(types, str) else types))

    monkeypatch.setattr(auth, "get_db", lambda: _DB())
    old = store.add_action(ALICE, kind="run_prompt", title="Re-run the stock search",
                           rationale="Prices moved.", status="awaiting_approval")
    store.add_action(ALICE, kind="nudge", title="Already answered", status="rejected")
    store.add_action(BOB, kind="nudge", title="Bob's", status="awaiting_approval")
    assert _run_coro(owner_notify.bell_backfill(ALICE)) == 1
    assert rows == [{"user": ALICE, "type": "autonomy_proposal", "ref_type": "autonomy_action",
                     "ref_id": old["id"], "body": "Re-run the stock search\nPrices moved."}]
    assert _run_coro(owner_notify.bell_backfill(ALICE)) == 0  # once


async def test_a_waiting_proposal_stays_in_the_bell_past_its_page(store, tmp_path, monkeypatch):
    """However many notices came after it, a proposal still waiting for an
    answer comes with the bell's list — once, and only while it waits."""
    import httpx

    from captain_claw.flight_deck import auth as fd_auth, notification_routes
    from captain_claw.flight_deck.auth import get_current_user, set_auth_db
    from captain_claw.flight_deck.db import FlightDeckDB

    async def no_devices(user_id, count, text="", beep=False):
        return []

    monkeypatch.setattr(device_bridge, "push_waiting", no_devices)
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    try:
        act = store.add_action(ALICE, kind="run_prompt", title="Re-run the stock search",
                               status="awaiting_approval")
        bell = await db.add_notification(ALICE, "autonomy_proposal", "Waiting for you", "Re-run",
                                         ref_type="autonomy_action", ref_id=act["id"])
        await db._db.execute("UPDATE notifications SET created_at = ? WHERE id = ?",
                             ("2026-01-01T00:00:00+00:00", bell))
        await db._db.commit()
        for i in range(5):
            await db.add_notification(ALICE, "info", f"Notice {i}")
        app = FastAPI()
        app.include_router(notification_routes.router)
        app.dependency_overrides[get_current_user] = lambda: {"id": ALICE}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fd.test") as c:
            async def ids(limit):
                r = await c.get(f"/fd/notifications?limit={limit}")
                return [n["id"] for n in r.json()["notifications"]]

            got = await ids(3)
            assert len(got) == 4 and got[-1] == bell
            assert (await ids(50)).count(bell) == 1  # on the page already: not twice
            store.update_status(act["id"], "rejected")
            assert bell not in await ids(3)
    finally:
        await db.close()
        fd_auth._db = prev


def test_quiet_hours_keep_a_proposal_off_whatsapp_and_telegram(store, channels, monkeypatch):
    """Triage runs at any hour, so a proposal can come at 2 am: the bell has it
    and the devices' count shows it, silently; no phone rings for it."""
    _settings(store, ALICE, notify_waid="111", notify_telegram=True)
    monkeypatch.setattr(arbiter, "_in_quiet_hours", lambda s, e: True)
    row = store.add_action(ALICE, kind="tool_action", title="Reply to Ana", status="awaiting_approval")
    out = _run_coro(owner_notify.notify_proposal(ALICE, row))
    assert channels["whatsapp"] == [] and channels["telegram"] == [] and out["whatsapp"] is None
    assert len(channels["bell"]) == 1 and channels["devices"][-1]["beep"] is False
    # What the agent says itself (a nudge) goes as it always did, and by day a proposal does too.
    _nudge(store, monkeypatch)
    assert channels["whatsapp"] and channels["telegram"]
    channels["whatsapp"].clear()
    monkeypatch.setattr(arbiter, "_in_quiet_hours", lambda s, e: False)
    _run_coro(owner_notify.notify_proposal(ALICE, row))
    assert channels["whatsapp"]


def test_the_devices_count_follows_a_ledger_change_made_off_the_loop(store, channels, monkeypatch):
    """The Taskboard writes the ledger from worker threads: a next step it
    made or expired moves the count on the devices all the same."""
    from captain_claw.flight_deck import taskboard_passive

    async def go():
        monkeypatch.setattr(taskboard_passive, "_LOOP", asyncio.get_running_loop())
        row = await asyncio.to_thread(store.add_action, ALICE, kind="tool_action", title="Reply to Ana",
                                      status="awaiting_approval")
        await asyncio.sleep(0.05)
        assert channels["devices"][-1]["count"] == 1
        await asyncio.to_thread(store.expire_actions, [row["id"]], "expired: its task closed")
        await asyncio.sleep(0.05)
        assert channels["devices"][-1]["count"] == 0

    _run_coro(go())
    # No loop anywhere (a script): nothing to do, nothing raised.
    monkeypatch.setattr(taskboard_passive, "_LOOP", None)
    store.add_action(ALICE, kind="tool_action", title="Later", status="awaiting_approval")
