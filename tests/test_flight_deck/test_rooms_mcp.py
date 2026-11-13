"""The data-room connector (docs/data-room-plan.md P2): rooms-scope sign-in with
a room choice, tokens that work only where they belong, the five read tools,
paging, search, the read log, token history and the alarm for real values in
Claude's arguments. Made-up data."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import re
import secrets
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import (
    anonymize_vault,
    auth,
    mcp_oauth_routes,
    mcp_server_routes,
    rooms_mcp,
    rooms_routes,
    rooms_store,
)
from test_flight_deck.test_rooms import (  # noqa: F401 — the deck fixture
    ALICE,
    ALICE_H,
    BOB,
    EMAIL,
    IBAN,
    _as,
    _files,
    _wait,
    _zip,
    deck,
)

BASE = "http://deck.test"
CALLBACK = "https://claude.ai/api/mcp/auth_callback"
PASSWORD = "falcon-pass-123"


def _app() -> FastAPI:
    app = FastAPI()
    for r in (rooms_routes.router, rooms_routes.access_router, rooms_mcp.router, mcp_oauth_routes.router,
              mcp_server_routes.router):
        app.include_router(r)
    return app


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(), client=("127.0.0.1", 50123)), base_url=BASE)


async def _released_room(c: httpx.AsyncClient, codename: str = "Project Falcon", files: dict | None = None,
                         headers=None) -> dict:
    headers = headers or ALICE_H
    room = (await c.post("/fd/rooms", headers=headers, json={"codename": codename})).json()
    files = files or {
        "Pravno/ugovor.txt": f"Ime i prezime: Ana Kovač\nE-mail: {EMAIL}\nIBAN: {IBAN}\nCijena: 42 milijuna EUR.",
        "Financije/dugi.txt": "Prihod raste.\n" * 5000,  # ~70k characters: two pages
    }
    await c.put(f"/fd/rooms/{room['id']}/zip", headers=headers, content=_zip(files))
    await c.post(f"/fd/rooms/{room['id']}/run", headers=headers, json={})
    await _wait(c, room["id"], headers=headers)
    await c.post(f"/fd/rooms/{room['id']}/release", headers=headers)
    return room


def _rooms_token(uid: str = ALICE, client_id: str = "cc_oc_test") -> dict[str, str]:
    return {"Authorization": f"Bearer {auth.create_mcp_access_token(uid, client_id, 'rooms')}"}


async def _rpc(c: httpx.AsyncClient, headers: dict, method: str, params: dict | None = None):
    r = await c.post("/fd/mcp-rooms", headers=headers,
                     json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}})
    return r


async def _tool(c: httpx.AsyncClient, headers: dict, name: str, **args):
    r = await _rpc(c, headers, "tools/call", {"name": name, "arguments": args})
    assert r.status_code == 200, r.text
    result = r.json()["result"]
    text = result["content"][0]["text"]
    if result["isError"]:
        return {"error": text}
    try:
        return json.loads(text)
    except ValueError:
        return text


# ── sign-in ─────────────────────────────────────────────────────────


@pytest.mark.usefixtures("deck")
async def test_rooms_sign_in_asks_which_rooms_and_gives_a_rooms_only_token():
    await auth._db._db.execute("UPDATE users SET password_hash=? WHERE id=?", (auth.hash_password(PASSWORD), ALICE))
    await auth._db._db.commit()
    async with _client() as c:
        falcon = await _released_room(c)
        other = (await c.post("/fd/rooms", headers=ALICE_H, json={"codename": "Project Heron"})).json()

        prm = (await c.get("/.well-known/oauth-protected-resource/fd/mcp-rooms")).json()
        assert prm["resource"].endswith("/fd/mcp-rooms") and prm["scopes_supported"] == ["rooms"]
        unauth = await c.post("/fd/mcp-rooms", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert unauth.status_code == 401
        assert "oauth-protected-resource/fd/mcp-rooms" in unauth.headers["www-authenticate"]

        cid = (await c.post("/register", json={"redirect_uris": [CALLBACK], "client_name": "Claude"})).json()["client_id"]
        verifier = secrets.token_urlsafe(40)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        q = urlencode({"response_type": "code", "client_id": cid, "redirect_uri": CALLBACK, "state": "s",
                       "code_challenge": challenge, "code_challenge_method": "S256", "scope": "mcp",
                       "resource": f"{BASE}/fd/mcp-rooms"})
        page = (await c.get(f"/authorize?{q}")).text
        assert "data rooms you choose" in page  # rooms consent, even though the client asked "mcp"
        blob = html.unescape(re.search(r"name=request value='([^']+)'", page).group(1))
        step = await c.post("/authorize", content=urlencode({"request": blob, "decision": "approve",
                                                             "email": "alice@x.co", "password": PASSWORD}),
                            headers={"Content-Type": "application/x-www-form-urlencoded"})
        assert step.status_code == 200 and "Which data rooms?" in step.text
        assert "Project Falcon" in step.text and "Project Heron" in step.text
        blob2 = html.unescape(re.search(r"name=request value='([^']+)'", step.text).group(1))
        # The room step can't be skipped back into the first step, nor the other way round.
        assert (await c.post("/authorize", content=urlencode({"request": blob2, "decision": "approve",
                                                              "email": "alice@x.co", "password": PASSWORD}),
                             headers={"Content-Type": "application/x-www-form-urlencoded"})).status_code == 400
        none = await c.post("/authorize/rooms", content=urlencode({"request": blob2, "decision": "approve"}),
                            headers={"Content-Type": "application/x-www-form-urlencoded"})
        assert "Tick at least one room" in none.text
        done = await c.post("/authorize/rooms",
                            content=urlencode({"request": blob2, "decision": "approve", "room": falcon["id"]}),
                            headers={"Content-Type": "application/x-www-form-urlencoded"})
        assert done.status_code == 302
        code = parse_qs(urlsplit(done.headers["location"]).query)["code"][0]
        tok = (await c.post("/token", content=urlencode({"grant_type": "authorization_code", "code": code,
                                                          "redirect_uri": CALLBACK, "client_id": cid,
                                                          "code_verifier": verifier}),
                            headers={"Content-Type": "application/x-www-form-urlencoded"})).json()
        assert tok["scope"] == "rooms"
        bearer = {"Authorization": f"Bearer {tok['access_token']}"}

        rooms = await _tool(c, bearer, "room_list")
        assert [r["codename"] for r in rooms["rooms"]] == ["Project Falcon"]  # Heron wasn't ticked
        assert (await _tool(c, bearer, "room_overview", room=other["id"]))["error"]
        # A rooms token is no fleet token, and a fleet token is no rooms token.
        assert (await c.post("/fd/mcp-server", headers=bearer,
                             json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})).status_code == 401
        fleet = {"Authorization": f"Bearer {auth.create_mcp_access_token(ALICE, cid, 'mcp')}"}
        assert (await _rpc(c, fleet, "tools/list")).status_code == 401

        access = (await c.get("/fd/rooms-access", headers=ALICE_H)).json()
        assert access["connector_url"].endswith("/fd/mcp-rooms")
        assert access["grants"][0]["app"] == "Claude" and access["grants"][0]["rooms"][0]["id"] == falcon["id"]
        assert (await c.delete(f"/fd/rooms-access/{cid}", headers=ALICE_H)).json()["deleted"] is True
        assert (await _tool(c, bearer, "room_list"))["rooms"] == []  # at once, before the token expires


# ── the tools ───────────────────────────────────────────────────────


@pytest.mark.usefixtures("deck")
async def test_protocol_instructions_and_no_stream():
    async with _client() as c:
        r = await _rpc(c, _rooms_token(), "initialize", {"protocolVersion": "2025-06-18"})
        result = r.json()["result"]
        assert result["protocolVersion"] == "2025-06-18" and "brackets included" in result["instructions"]
        old = await _rpc(c, _rooms_token(), "initialize", {"protocolVersion": "2024-11-05"})
        assert old.json()["result"]["protocolVersion"] == "2025-03-26"
        assert (await _rpc(c, _rooms_token(), "server/discover")).json()["error"]["code"] == -32601
        assert (await c.get("/fd/mcp-rooms")).status_code == 405
        tools = (await _rpc(c, _rooms_token(), "tools/list")).json()["result"]["tools"]
        reads = {"room_list", "room_overview", "room_list_files", "room_read", "room_search", "room_get_file"}
        delivers = {"room_deliver", "room_deliver_workbook", "room_deliver_deck", "room_deliver_document",
                    "room_deliver_file"}
        assert {t["name"] for t in tools} == reads | delivers
        writes = {t["name"] for t in tools if not t["annotations"]["readOnlyHint"]}
        assert writes == delivers  # only the deliveries change anything
        assert "image" not in json.dumps(tools)  # no tool takes a picture


@pytest.mark.usefixtures("deck")
async def test_read_only_released_anonymized_text_in_pages():
    async with _client() as c:
        room = await _released_room(c)
        await rooms_store.store().set_grant(ALICE, "cc_oc_test", [room["id"]], False)
        files = await _files(c, room["id"])
        await c.patch(f"/fd/rooms/{room['id']}", headers=ALICE_H, json={"roles": {"⟦PERSON_1⟧": "CFO of the target"}})
        # A file that came out blocked is never readable through the connector.
        await rooms_store.store().set_file(room["id"], files["Financije/dugi.txt"]["id"], state="blocked",
                                           released=False)
        t = _rooms_token()
        overview = await _tool(c, t, "room_overview", room="project falcon")  # codename, any case
        assert overview["roles"] == {"⟦PERSON_1⟧": "CFO of the target"} and overview["released_files"] == 1
        listing = await _tool(c, t, "room_list_files", room=room["id"])
        assert [f["path"] for f in listing["files"]] == [files["Pravno/ugovor.txt"]["clean_path"]]
        text = await _tool(c, t, "room_read", room=room["id"], path=files["Pravno/ugovor.txt"]["clean_path"])
        assert text.startswith("File: ") and "page 1 of 1" in text
        assert EMAIL not in text and IBAN not in text and "Ana Kovač" not in text and "⟦EMAIL_1⟧" in text
        assert "error" in await _tool(c, t, "room_read", room=room["id"], path=files["Financije/dugi.txt"]["clean_path"])

        # Paging: unblock the long file and read its second page.
        await rooms_store.store().set_file(room["id"], files["Financije/dugi.txt"]["id"], state="done", released=True)
        long_path = files["Financije/dugi.txt"]["clean_path"]
        p1 = await _tool(c, t, "room_read", room=room["id"], path=long_path)
        assert "page 1 of 2" in p1 and p1.rstrip().endswith("[continues on page 2]")
        p2 = await _tool(c, t, "room_read", room=room["id"], path=long_path, page=2)
        assert "page 2 of 2" in p2
        assert "has 2 page(s)" in (await _tool(c, t, "room_read", room=room["id"], path=long_path, page=3))["error"]

        hits = await _tool(c, t, "room_search", room=room["id"], query="Cijena")
        assert hits["results"][0]["path"] == files["Pravno/ugovor.txt"]["clean_path"] and hits["results"][0]["page"] == 1

        reads = (await c.get(f"/fd/rooms/{room['id']}/reads", headers=ALICE_H)).json()
        tools_logged = [r["tool"] for r in reads["reads"]]
        assert "room_read" in tools_logged and "room_search" in tools_logged and not reads["alarms"]
        assert all("Cijena" not in json.dumps(r) for r in reads["reads"])  # never a query

        vault = await anonymize_vault.store().get(ALICE, room["vault_id"])
        stats = await anonymize_vault.store().token_stats(vault)
        assert stats["⟦EMAIL_1⟧"]["sent"] >= 1


@pytest.mark.usefixtures("deck")
async def test_a_real_value_from_claude_raises_an_alarm():
    async with _client() as c:
        room = await _released_room(c)
        await rooms_store.store().set_grant(ALICE, "cc_oc_test", [], True)  # all rooms
        out = await _tool(c, _rooms_token(), "room_search", room=room["id"], query="Ana Kovač ugovor")
        assert "results" in out  # still answers: the value is already out
        alarms = (await c.get(f"/fd/rooms/{room['id']}/reads", headers=ALICE_H)).json()["alarms"]
        assert alarms and alarms[0]["tokens"] == ["⟦PERSON_1⟧"] and alarms[0]["seen"] is False
        assert "Ana" not in json.dumps(alarms)
        await c.post(f"/fd/rooms/{room['id']}/alarms/seen", headers=ALICE_H)
        assert (await c.get(f"/fd/rooms/{room['id']}/reads", headers=ALICE_H)).json()["alarms"][0]["seen"] is True


@pytest.mark.usefixtures("deck")
async def test_nothing_of_someone_elses_rooms():
    async with _client() as c:
        room = await _released_room(c)
        # Bob's token, with a grant naming Alice's room: still nothing.
        await rooms_store.store().set_grant(BOB, "cc_oc_bob", [room["id"]], False)
        bob = _rooms_token(BOB, "cc_oc_bob")
        assert (await _tool(c, bob, "room_list"))["rooms"] == []
        assert "error" in await _tool(c, bob, "room_read", room=room["id"], path="x")
        assert (await c.get("/fd/rooms-access", headers=_as(BOB))).json()["grants"][0]["rooms"] == []


@pytest.mark.usefixtures("deck")
async def test_no_grant_no_rooms_and_nothing_released_says_so():
    async with _client() as c:
        assert (await _tool(c, _rooms_token(client_id="cc_oc_none"), "room_list"))["rooms"] == []
        room = (await c.post("/fd/rooms", headers=ALICE_H, json={"codename": "Empty"})).json()
        await rooms_store.store().set_grant(ALICE, "cc_oc_test", [room["id"]], False)
        out = await _tool(c, _rooms_token(), "room_list")
        assert out["rooms"][0]["released_files"] == 0 and "Nothing is released" in out["note"]
