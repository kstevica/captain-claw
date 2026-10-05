"""The Connections card's WhatsApp nudge routes: GET/PUT /fd/autonomy/whatsapp
and POST /fd/autonomy/whatsapp/test.

A user binds their own number (``notify_waid``) without touching the console.
The number must be on the bridge allowlist, saving merges into the user's other
autonomy overrides, the response never reveals the allowlist itself, and with
auth on a deck-wide ``notify_waid`` in config.yaml is ignored (a phone number is
personal)."""

from __future__ import annotations

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
    return base


@pytest.fixture
def store(tmp_path, monkeypatch, defaults):
    s = autonomy.AutonomyStore(tmp_path / "autonomy.db")
    monkeypatch.setattr(autonomy, "_STORE", s)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("WHATSAPP_ALLOWED_WAIDS", "111,222")
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "42")
    return s


@pytest.fixture
def pushed(monkeypatch):
    sent: list[str] = []

    async def fake_push(waid, text):
        sent.append(waid)
        return True

    monkeypatch.setattr(whatsapp_bridge, "push_to_waid", fake_push)
    return sent


@pytest.fixture
def client(store):
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


def test_unbound_user_gets_no_recipients_and_no_allowlist(client):
    r = client.get("/fd/autonomy/whatsapp", headers=_as(ALICE))
    assert r.status_code == 200
    body = r.json()
    assert body["bridge_configured"] is True
    assert body["recipients"] == [] and body["notify_waid"] == ""
    assert "notify_waid" in body["issue"]
    assert "111" not in r.text and "222" not in r.text


def test_bind_normalises_and_merges_with_other_overrides(client, store):
    store.set_overrides(ALICE, {"enabled": True, "autonomy_level": "propose"})
    r = client.put("/fd/autonomy/whatsapp", headers=_as(ALICE), json={"notify_waid": " +1 11 , 111"})
    assert r.status_code == 200
    assert r.json()["notify_waid"] == "111"
    assert r.json()["recipients"] == ["111"]
    assert r.json()["autonomy_enabled"] is True
    assert store.get_overrides(ALICE) == {"enabled": True, "autonomy_level": "propose",
                                          "notify_waid": "111"}


def test_number_off_the_allowlist_is_refused_and_not_saved(client, store):
    r = client.put("/fd/autonomy/whatsapp", headers=_as(ALICE), json={"notify_waid": "111, 999"})
    assert r.status_code == 400
    assert "999" in r.json()["detail"]
    assert store.get_overrides(ALICE) == {}


def test_bridge_not_configured_refuses_a_number(client, monkeypatch):
    monkeypatch.setenv("WHATSAPP_ALLOWED_WAIDS", "")
    r = client.put("/fd/autonomy/whatsapp", headers=_as(ALICE), json={"notify_waid": "111"})
    assert r.status_code == 400
    assert client.get("/fd/autonomy/whatsapp", headers=_as(ALICE)).json()["bridge_configured"] is False


def test_missing_send_credentials_mean_not_configured(client, pushed, monkeypatch):
    """A push without a token silently no-ops, so the card must not say Active
    and the test button must not report a send that never happened."""
    client.put("/fd/autonomy/whatsapp", headers=_as(ALICE), json={"notify_waid": "111"})
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "")
    assert client.get("/fd/autonomy/whatsapp", headers=_as(ALICE)).json()["bridge_configured"] is False
    assert client.post("/fd/autonomy/whatsapp/test", headers=_as(ALICE)).status_code == 400
    assert pushed == []


def test_toggle_and_clear_keep_the_rest(client, store):
    client.put("/fd/autonomy/whatsapp", headers=_as(ALICE), json={"notify_waid": "222"})
    r = client.put("/fd/autonomy/whatsapp", headers=_as(ALICE), json={"nudge_to_whatsapp": False})
    assert r.json()["nudge_to_whatsapp"] is False and r.json()["notify_waid"] == "222"
    r = client.put("/fd/autonomy/whatsapp", headers=_as(ALICE), json={"notify_waid": ""})
    assert r.json()["notify_waid"] == "" and r.json()["recipients"] == []
    assert store.get_overrides(ALICE) == {"nudge_to_whatsapp": False, "notify_waid": ""}


def test_bindings_are_per_user(client):
    client.put("/fd/autonomy/whatsapp", headers=_as(ALICE), json={"notify_waid": "111"})
    bob = client.get("/fd/autonomy/whatsapp", headers=_as(BOB)).json()
    assert bob["notify_waid"] == "" and bob["recipients"] == []


def test_test_message_goes_only_to_the_bound_number(client, pushed):
    r = client.post("/fd/autonomy/whatsapp/test", headers=_as(ALICE))
    assert r.status_code == 400 and pushed == []
    client.put("/fd/autonomy/whatsapp", headers=_as(ALICE), json={"notify_waid": "222"})
    r = client.post("/fd/autonomy/whatsapp/test", headers=_as(ALICE))
    assert r.json() == {"sent": 1, "total": 1}
    assert pushed == ["222"]


def test_requires_a_user_when_auth_is_on(client):
    assert client.get("/fd/autonomy/whatsapp").status_code == 401


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
    assert body["recipients"] == ["111", "222"]
    assert body["auth_enabled"] is False
