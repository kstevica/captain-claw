"""The Connections card's WhatsApp nudge routes: GET/PUT /fd/autonomy/whatsapp
and POST /fd/autonomy/whatsapp/test.

A user binds their own number (``notify_waid``) without touching the console.
The number must be on the bridge allowlist and not already another user's, the
refusal is the same either way (no probing which numbers the deck knows), saving
merges into the user's other autonomy overrides, the response never reveals the
allowlist, and with auth on a deck-wide ``notify_waid`` in config.yaml applies to
nobody — nor can the Autonomous Work page's whole-config save (or its Reset)
overwrite or inject one. Test sends are rate-limited and report what WhatsApp
said."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from captain_claw.config import AutonomousWorkConfig
from captain_claw.flight_deck import autonomy, autonomy_routes, whatsapp_bridge
from captain_claw.flight_deck.auth import get_optional_user

ALICE, BOB = "user-alice", "user-bob"


@pytest.fixture
def defaults(monkeypatch):
    """The global defaults (shipped config, overridable per test)."""
    base = AutonomousWorkConfig().model_dump()
    monkeypatch.setattr(autonomy, "global_defaults", lambda: dict(base))
    monkeypatch.setattr(autonomy_routes, "global_defaults", lambda: dict(base))
    return base


@pytest.fixture
def store(tmp_path, monkeypatch, defaults):
    s = autonomy.AutonomyStore(tmp_path / "autonomy.db")
    monkeypatch.setattr(autonomy, "_STORE", s)
    monkeypatch.setattr(autonomy_routes, "_last_test_at", {})
    monkeypatch.setattr(autonomy_routes, "_number_changes", {})
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("WHATSAPP_ALLOWED_WAIDS", "111,222,333,444,555")
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "42")
    return s


@pytest.fixture
def sent(monkeypatch):
    """Stub the checked sender: ``sent.log`` lists delivered numbers and
    ``sent.fail`` maps a number to the refusal WhatsApp gives it."""
    state = SimpleNamespace(log=[], fail={})

    async def fake_send(waid, text):
        if waid in state.fail:
            return False, state.fail[waid]
        state.log.append(waid)
        return True, ""

    monkeypatch.setattr(whatsapp_bridge, "send_text_checked", fake_send)
    return state


@pytest.fixture
def users(monkeypatch):
    """The deck's user table: only these ids exist."""
    from captain_claw.flight_deck import auth

    known = {ALICE, BOB}

    class _DB:
        async def get_user_by_id(self, uid):
            return {"id": uid} if uid in known else None

    monkeypatch.setattr(auth, "get_db", lambda: _DB())
    return known


@pytest.fixture
def client(store, users):
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


def _bind(client, uid, number):
    return client.put("/fd/autonomy/whatsapp", headers=_as(uid), json={"notify_waid": number})


# ── reading and binding ─────────────────────────────────────────────


def test_unbound_user_gets_no_recipients_and_no_allowlist(client):
    r = client.get("/fd/autonomy/whatsapp", headers=_as(ALICE))
    assert r.status_code == 200
    body = r.json()
    assert body["bridge_configured"] is True
    assert body["recipients"] == [] and body["notify_waid"] == ""
    assert "notify_waid" in body["issue"]
    assert not any(n in r.text for n in ("111", "222", "333"))


def test_bind_normalises_and_merges_with_other_overrides(client, store):
    store.set_overrides(ALICE, {"enabled": True, "autonomy_level": "propose"})
    r = _bind(client, ALICE, " +1 11 , 111")
    assert r.status_code == 200
    assert r.json()["notify_waid"] == "111"
    assert r.json()["recipients"] == ["111"]
    assert r.json()["autonomy_active"] is True
    assert store.get_overrides(ALICE) == {"enabled": True, "autonomy_level": "propose",
                                          "notify_waid": "111"}


def test_refusal_is_generic_and_nothing_is_saved(client, store):
    r = _bind(client, ALICE, "111, 999")
    assert r.status_code == 400
    assert "999" not in r.json()["detail"] and "111" not in r.json()["detail"]
    assert store.get_overrides(ALICE) == {}


def test_a_number_already_bound_to_another_user_is_refused_the_same_way(client, store, caplog):
    assert _bind(client, ALICE, "111").status_code == 200
    with caplog.at_level("WARNING", logger=autonomy_routes.__name__):
        r = _bind(client, BOB, "222, +111")
    assert r.status_code == 400
    assert r.json()["detail"] == _bind(client, BOB, "999").json()["detail"]
    assert store.get_overrides(BOB) == {}
    assert ALICE in caplog.text  # the admin can see who holds it
    # Re-saving your own number is fine.
    assert _bind(client, ALICE, "111").status_code == 200


def test_the_local_bucket_and_deleted_users_hold_no_numbers(client, store):
    store.set_overrides("", {"notify_waid": "111"})  # left over from running without auth
    store.set_overrides("user-gone", {"notify_waid": "222"})  # deleted from the users table
    assert _bind(client, ALICE, "111,222").status_code == 200


def test_number_changes_are_rate_limited(client, monkeypatch):
    monkeypatch.setattr(autonomy_routes, "_MAX_NUMBER_CHANGES_PER_HOUR", 2)
    assert _bind(client, ALICE, "111").status_code == 200
    assert _bind(client, ALICE, "111").status_code == 200  # unchanged: not counted
    assert _bind(client, ALICE, "999").status_code == 400  # a refused attempt counts
    assert _bind(client, ALICE, "222").status_code == 429
    assert _bind(client, BOB, "222").status_code == 200


def test_list_is_capped_and_junk_is_an_error(client, store):
    assert _bind(client, ALICE, "111,222,333,444").status_code == 400
    r = _bind(client, ALICE, "my phone")
    assert r.status_code == 400 and "phone number" in r.json()["detail"]
    assert client.put("/fd/autonomy/whatsapp", headers=_as(ALICE),
                      json={"notify_waid": 111}).status_code == 400
    assert store.get_overrides(ALICE) == {}


def test_toggle_must_be_a_real_boolean(client, store):
    r = client.put("/fd/autonomy/whatsapp", headers=_as(ALICE), json={"nudge_to_whatsapp": "false"})
    assert r.status_code == 400
    assert store.get_overrides(ALICE) == {}


def test_bridge_not_configured_refuses_a_number(client, monkeypatch):
    monkeypatch.setenv("WHATSAPP_ALLOWED_WAIDS", "")
    assert _bind(client, ALICE, "111").status_code == 400
    assert client.get("/fd/autonomy/whatsapp", headers=_as(ALICE)).json()["bridge_configured"] is False


def test_toggle_and_clear_keep_the_rest(client, store):
    _bind(client, ALICE, "222")
    r = client.put("/fd/autonomy/whatsapp", headers=_as(ALICE), json={"nudge_to_whatsapp": False})
    assert r.json()["nudge_to_whatsapp"] is False and r.json()["notify_waid"] == "222"
    r = _bind(client, ALICE, "")
    assert r.json()["notify_waid"] == "" and r.json()["recipients"] == []
    assert store.get_overrides(ALICE) == {"nudge_to_whatsapp": False, "notify_waid": ""}


def test_toggle_alone_works_after_the_number_left_the_allowlist(client, monkeypatch):
    _bind(client, ALICE, "222")
    monkeypatch.setenv("WHATSAPP_ALLOWED_WAIDS", "111")
    r = client.put("/fd/autonomy/whatsapp", headers=_as(ALICE), json={"nudge_to_whatsapp": False})
    assert r.status_code == 200 and r.json()["recipients"] == []


def test_bindings_are_per_user(client):
    _bind(client, ALICE, "111")
    bob = client.get("/fd/autonomy/whatsapp", headers=_as(BOB)).json()
    assert bob["notify_waid"] == "" and bob["recipients"] == []


@pytest.mark.parametrize("method, path", [
    ("get", "/fd/autonomy/whatsapp"),
    ("put", "/fd/autonomy/whatsapp"),
    ("post", "/fd/autonomy/whatsapp/test"),
])
def test_requires_a_user_when_auth_is_on(client, method, path):
    kw = {"json": {"notify_waid": "111"}} if method == "put" else {}
    assert getattr(client, method)(path, **kw).status_code == 401


def test_autonomy_active_needs_the_arbiter_running(client, store):
    store.set_overrides(ALICE, {"enabled": True, "autonomy_level": "off"})
    assert client.get("/fd/autonomy/whatsapp", headers=_as(ALICE)).json()["autonomy_active"] is False
    store.set_overrides(ALICE, {"enabled": True, "autonomy_level": "propose", "arbiter_on_pulse": False})
    assert client.get("/fd/autonomy/whatsapp", headers=_as(ALICE)).json()["autonomy_active"] is False


# ── the Autonomous Work page's whole-config save ────────────────────


def test_config_save_and_reset_keep_the_bound_number(client, store, defaults):
    defaults["notify_waid"] = "333"  # a deck-wide number in config.yaml
    _bind(client, ALICE, "111")
    page = client.get("/fd/autonomy/config", headers=_as(ALICE)).json()
    assert page["defaults"]["notify_waid"] == ""  # not shown to users with auth on
    # Reset to defaults + Save sends every default key back.
    reset = {**page["config"], **page["defaults"], "notify_waid": "333"}
    r = client.put("/fd/autonomy/config", headers=_as(ALICE), json={"config": reset})
    assert r.status_code == 200
    assert client.get("/fd/autonomy/whatsapp", headers=_as(ALICE)).json()["recipients"] == ["111"]


def test_config_save_cannot_set_a_number(client, store):
    client.put("/fd/autonomy/config", headers=_as(ALICE), json={"notify_waid": "222"})
    assert "notify_waid" not in store.get_overrides(ALICE)


def test_config_save_keeps_the_whatsapp_switch(client, store):
    client.put("/fd/autonomy/whatsapp", headers=_as(ALICE), json={"nudge_to_whatsapp": False})
    # A stale Autonomous Work tab saves its whole (old) config.
    client.put("/fd/autonomy/config", headers=_as(ALICE),
               json={"config": {"nudge_to_whatsapp": True, "enabled": True}})
    assert store.get_overrides(ALICE)["nudge_to_whatsapp"] is False
    assert store.get_overrides(ALICE)["enabled"] is True


def test_deck_wide_notify_waid_is_ignored_with_auth(client, defaults, monkeypatch):
    defaults["notify_waid"] = "222"
    assert autonomy.resolve_config(ALICE)["notify_waid"] == ""
    assert client.get("/fd/autonomy/whatsapp", headers=_as(ALICE)).json()["recipients"] == []
    # Without auth there is one trusted user, so the deck-wide value applies.
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    assert autonomy.resolve_config("")["notify_waid"] == "222"


def test_auth_off_unbound_reaches_every_allowlisted_number(client, monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    body = client.get("/fd/autonomy/whatsapp").json()
    assert body["recipients"] == ["111", "222", "333", "444", "555"]
    assert body["auth_enabled"] is False


# ── the test send ───────────────────────────────────────────────────


def test_test_message_goes_only_to_the_callers_number(client, sent):
    _bind(client, ALICE, "222")
    r = client.post("/fd/autonomy/whatsapp/test", headers=_as(BOB))
    assert r.status_code == 400 and sent.log == []
    r = client.post("/fd/autonomy/whatsapp/test", headers=_as(ALICE))
    assert r.json()["sent"] == 1 and r.json()["total"] == 1
    assert sent.log == ["222"]


def test_test_send_is_rate_limited_per_user(client, sent):
    _bind(client, ALICE, "111")
    _bind(client, BOB, "222")
    assert client.post("/fd/autonomy/whatsapp/test", headers=_as(ALICE)).status_code == 200
    r = client.post("/fd/autonomy/whatsapp/test", headers=_as(ALICE))
    assert r.status_code == 429 and "Wait" in r.json()["detail"]
    assert client.post("/fd/autonomy/whatsapp/test", headers=_as(BOB)).status_code == 200
    assert sent.log == ["111", "222"]


def test_test_send_reports_what_whatsapp_said(client, sent):
    _bind(client, ALICE, "111")
    sent.fail["111"] = "WhatsApp refused it (400: Re-engagement message)"
    body = client.post("/fd/autonomy/whatsapp/test", headers=_as(ALICE)).json()
    assert body["sent"] == 0
    assert body["results"] == [{"to": "111", "ok": False,
                                "error": "WhatsApp refused it (400: Re-engagement message)"}]


def test_missing_send_credentials_mean_not_configured(client, sent, monkeypatch):
    """A push without a token silently no-ops, so the card must not say Active
    and the test button must not report a send that never happened."""
    _bind(client, ALICE, "111")
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "")
    assert client.get("/fd/autonomy/whatsapp", headers=_as(ALICE)).json()["bridge_configured"] is False
    assert client.post("/fd/autonomy/whatsapp/test", headers=_as(ALICE)).status_code == 400
    assert sent.log == []


# ── send_text_checked (the bridge side of the test send) ────────────


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def graph(monkeypatch):
    """Route the bridge's httpx client to a fake Graph API."""
    calls: list[dict] = []
    reply = {"status": 200, "json": {"messages": [{"id": "wamid.1"}]}}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"url": str(request.url), "body": request.content.decode()})
        if isinstance(reply.get("raise"), Exception):
            raise reply["raise"]
        return httpx.Response(reply["status"], json=reply["json"])

    real = httpx.AsyncClient
    monkeypatch.setattr(whatsapp_bridge.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setenv("WHATSAPP_ALLOWED_WAIDS", "111")
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "42")
    monkeypatch.setattr(whatsapp_bridge, "is_push_muted", lambda w: w == "muted")
    return {"calls": calls, "reply": reply}


def test_checked_send_ok(graph):
    assert _run(whatsapp_bridge.send_text_checked("+111", "hi")) == (True, "")
    assert "/42/messages" in graph["calls"][0]["url"]


def test_checked_send_surfaces_the_graph_error(graph):
    graph["reply"].update(status=400, json={"error": {
        "message": "(#131030) Recipient phone number not in allowed list",
        "error_data": {"details": "Recipient phone number not in allowed list"}}})
    ok, why = _run(whatsapp_bridge.send_text_checked("111", "hi"))
    assert not ok and "400" in why and "not in allowed list" in why


def test_checked_send_network_error_is_reported_not_raised(graph):
    graph["reply"]["raise"] = httpx.ConnectError("boom")
    ok, why = _run(whatsapp_bridge.send_text_checked("111", "hi"))
    assert not ok and "ConnectError" in why


def test_checked_send_refuses_off_allowlist_and_muted(graph, monkeypatch):
    assert _run(whatsapp_bridge.send_text_checked("999", "hi"))[0] is False
    monkeypatch.setenv("WHATSAPP_ALLOWED_WAIDS", "111,muted")
    ok, why = _run(whatsapp_bridge.send_text_checked("muted", "hi"))
    assert not ok and "/unmute" in why
    assert graph["calls"] == []
