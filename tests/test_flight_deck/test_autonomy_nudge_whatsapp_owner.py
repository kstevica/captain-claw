"""An autonomous nudge reaches only its OWNER's WhatsApp — never every number on
the bridge allowlist. On a multi-user deck the fan-out to ``_allowed_waids()``
leaked one user's nudges onto everyone's phone. The owner's own ``notify_waid``
(per-user autonomy config, still allowlist-gated) picks the recipient; a deck
without auth keeps the old "every allowlisted number" behaviour."""

from __future__ import annotations

import asyncio

import pytest

from captain_claw.config import AutonomousWorkConfig
from captain_claw.flight_deck import autonomy, basna_routes, fd_dispatch, whatsapp_bridge

ALICE = "user-alice"
BOB = "user-bob"


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = autonomy.AutonomyStore(tmp_path / "autonomy.db")
    monkeypatch.setattr(autonomy, "_STORE", s)
    # Pure shipped defaults — never the dev box's config.yaml.
    monkeypatch.setattr(autonomy, "global_defaults", lambda: AutonomousWorkConfig().model_dump())
    return s


@pytest.fixture
def pushed(monkeypatch):
    sent: list[tuple[str, str]] = []

    async def fake_push(waid, text):
        sent.append((waid, text))
        return True

    async def fake_dispatch(port, auth, instruction, timeout, **kw):
        return {"ok": True, "output": "Hey — your 3pm call is in an hour."}

    monkeypatch.setattr(whatsapp_bridge, "push_to_waid", fake_push)
    monkeypatch.setattr(basna_routes, "_dispatch_one", fake_dispatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_WAIDS", "111,222")
    return sent


def _run_coro(coro):
    """Run *coro* on a private loop. ``asyncio.run`` would clear the current
    event loop afterwards, breaking later tests that use ``get_event_loop()``."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _nudge(store, uid):
    row = store.add_action(uid, kind="nudge", title="Remind about the 3pm call",
                           risk="low", status="dispatched")
    _run_coro(fd_dispatch._execute_and_judge(uid, row, {"slug": "a", "port": 1, "auth": ""}))
    return store.get_action(row["id"])


def test_auth_on_nudge_goes_only_to_the_owners_number(store, pushed, monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    store.set_overrides(ALICE, {"notify_waid": "+111"})
    store.set_overrides(BOB, {"notify_waid": "222"})
    done = _nudge(store, ALICE)
    assert done["status"] == "done" and done["outcome"] == "success"
    assert [w for w, _ in pushed] == ["111"]  # Bob's phone never sees Alice's nudge


def test_auth_on_owner_without_binding_is_not_pushed_and_logged(store, pushed, monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    store.set_overrides(ALICE, {"notify_waid": "111"})
    done = _nudge(store, BOB)
    assert done["status"] == "done" and done["outcome"] == "success"  # still delivered in chat
    assert pushed == []
    events = [r["event"] for r in store.list_log(BOB)]
    assert "nudge → whatsapp skipped" in events


def test_binding_off_the_allowlist_is_never_pushed(store, pushed, monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    store.set_overrides(ALICE, {"notify_waid": "999"})
    _nudge(store, ALICE)
    assert pushed == []
    assert "nudge → whatsapp skipped" in [r["event"] for r in store.list_log(ALICE)]


def test_auth_off_single_user_still_reaches_every_allowlisted_number(store, pushed, monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    _nudge(store, "")
    assert sorted(w for w, _ in pushed) == ["111", "222"]


def test_auth_off_explicit_binding_narrows_the_recipients(store, pushed, monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    store.set_overrides("", {"notify_waid": "222"})
    _nudge(store, "")
    assert [w for w, _ in pushed] == ["222"]


def test_no_whatsapp_configured_logs_nothing(store, pushed, monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("WHATSAPP_ALLOWED_WAIDS", "")
    _nudge(store, ALICE)
    assert pushed == []
    assert not [r for r in store.list_log(ALICE) if r["event"].startswith("nudge → whatsapp")]


def test_nudge_to_whatsapp_off_still_wins(store, pushed, monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    store.set_overrides(ALICE, {"notify_waid": "111", "nudge_to_whatsapp": False})
    _nudge(store, ALICE)
    assert pushed == []
