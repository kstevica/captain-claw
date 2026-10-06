"""PR C: the member HTTP routes of a shared agent (``/api/speaker/*``).

Contract parts 0b §2.1-§2.2, 2c, 2d §6: Flight Deck reaches a member's Files
and Datastore panels with the owner token AND a fresh HTTP assertion bound to
the request (``aud="http"``, method, path, single-use nonce). The agent
honours ``X-FD-Speaker`` only on ``/api/speaker/*`` (and passes ``/ws``
through to its own handshake), acknowledges every verified request with
``X-FD-Speaker-Ack``, and decides ownership itself from the principal.
Payloads never carry a host path. Also pinned: the owner's REST creator
fields and the owner-side export neutralisation.

Every test runs with HOME, FD_DATA_DIR, the config DB paths, the workspace and
the global datastore in a tmp dir (nothing here may reach ~/.captain-claw or
a real FD data dir).
"""

from __future__ import annotations

import asyncio
import contextvars
import csv
import io
import json
import os
import re
import secrets
import time
import types
from pathlib import Path

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from captain_claw import datastore as ds
from captain_claw import saved_attribution as sa
from captain_claw import speaker
from captain_claw.config import get_config
from captain_claw.speaker import Principal
from captain_claw.web import speaker_http as sh
from captain_claw.web.auth import create_auth_middleware

TOKEN = "agent-web-token"
REF = "process:helper:0123456789abcdef"
ANA = Principal("u-ana", "Ana", "Olga", "A", REF)
BOB = Principal("u-bob", "Bob", "Olga", "A", REF)
ANA_SLUG = "spk-u-ana-A"       # the stub server's session id for Ana on lane A
BOB_SLUG = "spk-u-bob-A"
COLS = [{"name": "k", "type": "text"}, {"name": "v", "type": "text"}]

_DB_FIELDS = (
    ("memory", "path"), ("session", "path"), ("insights", "db_path"),
    ("conversation_topics", "db_path"), ("nervous_system", "db_path"),
    ("sister_session", "db_path"), ("cognitive_metrics", "db_path"),
    ("datastore", "path"), ("autonomous_work", "db_path"),
)


# ── isolation ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".captain-claw").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    (tmp_path / "fd-data" / "vfs").mkdir(parents=True)
    monkeypatch.setenv("FD_DATA_DIR", str(tmp_path / "fd-data"))
    for var in ("CLAW_VFS_ROOT", "CLAW_VFS_USER", "FD_OWNER_ID", "FD_URL", "CLAW_DATASTORE_VFS"):
        monkeypatch.delenv(var, raising=False)
    cfg = get_config()
    for section, attr in _DB_FIELDS:
        monkeypatch.setattr(getattr(cfg, section), attr, str(home / ".captain-claw" / f"{section}.db"))
    ws = (tmp_path / "workspace").resolve()
    (ws / "saved").mkdir(parents=True)
    (ws / "output").mkdir()
    monkeypatch.setattr(cfg.workspace, "path", str(ws))
    monkeypatch.setattr(cfg.web, "auth_token", TOKEN)
    monkeypatch.setattr(cfg.web, "public_run", False)
    from captain_claw import session as _session

    monkeypatch.setattr(_session, "_manager", _session.SessionManager(home / ".captain-claw" / "s.db"))
    import captain_claw.conversation_topics as _ct

    monkeypatch.setattr(_ct, "_MANAGER", None)
    monkeypatch.setattr(speaker, "_IN_FLIGHT", 0)
    monkeypatch.setattr(speaker, "_MAIN_LOOP", None)
    monkeypatch.setattr(sa, "_BACKFILLED", set())
    speaker._NONCES.clear()
    yield home
    speaker._NONCES.clear()
    with sa._LOCK:
        for conn in sa._CONNS.values():
            conn.close()
        sa._CONNS.clear()


@pytest.fixture
async def dm(tmp_path, monkeypatch):
    mgr = ds.DatastoreManager(db_path=tmp_path / "store" / "datastore.db")
    monkeypatch.setattr(ds, "_manager", mgr)
    yield mgr
    await mgr.close()


@pytest.fixture
def saved() -> Path:
    return Path(get_config().workspace.path) / "saved"


class StubServer:
    """What speaker_http needs from the WebServer: its config and the
    member's lane session (created on first use)."""

    def __init__(self):
        self.config = get_config()
        self.session_calls: list[Principal] = []
        self.broadcasts: list = []
        self.seen_principal: list = []

    async def _speaker_session(self, p):
        self.session_calls.append(p)
        return types.SimpleNamespace(id=f"spk-{p.speaker_id}-{p.lane}")

    def _broadcast(self, msg):          # must never be called by a member route
        self.broadcasts.append(msg)


@pytest.fixture
async def http(dm):
    stub = StubServer()
    owner_hits: list[str] = []

    async def _owner(request):
        owner_hits.append(request.path)
        stub.seen_principal.append(speaker.current())
        return web.json_response({"owner": True})

    async def _bound(request):
        return web.json_response({"bound": speaker.current() is not None})

    app = web.Application(middlewares=[
        create_auth_middleware(stub.config.web), sh.create_speaker_http_middleware(stub),
    ])
    sh.register_speaker_routes(app, stub)
    for path in ("/api/files", "/api/datastore/tables", "/ws", "/ws/stt", "/api/bound"):
        app.router.add_get(path, _bound if path == "/api/bound" else _owner)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        yield types.SimpleNamespace(client=client, stub=stub, owner_hits=owner_hits)
    finally:
        await client.close()
        await _close_sessions()


async def _close_sessions() -> None:
    """Close the tmp session DB the routes opened (its worker thread would
    otherwise keep a failed run's interpreter alive)."""
    from captain_claw.session import get_session_manager

    await get_session_manager().close()


def assertion(method: str, path: str, p: Principal = ANA, **over) -> str:
    now = int(time.time())
    payload = {
        "v": 1, "sub": p.speaker_id, "name": p.display_name, "owner": "u-owner",
        "owner_name": "Olga", "ref": p.agent_ref, "lane": p.lane,
        "conn": "http-" + secrets.token_hex(8), "iat": now, "exp": now + 60,
        "nonce": secrets.token_hex(8), "aud": "http", "m": method, "p": path,
    }
    for k, v in over.items():
        if v is None:
            payload.pop(k, None)
        else:
            payload[k] = v
    return speaker.sign_assertion(payload, TOKEN)


async def member(h, method: str, path: str, *, p: Principal = ANA, header: str | None = None,
                 params: dict | None = None, **kw):
    header = header or assertion(method, path, p)
    resp = await h.client.request(method, path, params={"token": TOKEN, **(params or {})},
                                  headers={"X-FD-Speaker": header}, **kw)
    body = await resp.read()
    return resp, body, header


def acked(resp, header) -> bool:
    return resp.headers.get("X-FD-Speaker-Ack") == speaker.speaker_ack_for(header)


def _stamp(p: Principal | None, path: Path) -> None:
    ctx = contextvars.copy_context()
    if p is not None:
        ctx.run(speaker.bind, p)
    ctx.run(sa.note_write, path, None)


def _put(path: Path, data: bytes | str, by="none") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode()
    path.write_bytes(data)
    if by != "none":
        _stamp(by, path)
    return path


def _no_host_paths(text: str, tmp_path: Path) -> None:
    assert str(tmp_path) not in text
    assert str(tmp_path.resolve()) not in text
    assert not re.search(r"(?<![\w.])/(tmp|private|Users|var)\b", text), text


# ── the middleware ───────────────────────────────────────────────────


async def test_the_owner_token_is_still_required_first(http):
    h = assertion("GET", "/api/speaker/files")
    resp = await http.client.get("/api/speaker/files", headers={"X-FD-Speaker": h})
    assert resp.status == 401
    assert "X-FD-Speaker-Ack" not in resp.headers


async def test_a_member_path_without_the_header_is_refused(http):
    resp = await http.client.get("/api/speaker/files", params={"token": TOKEN})
    assert resp.status == 403
    assert (await resp.json())["error"] == sh.ONLY_MEMBERS_MESSAGE
    assert "X-FD-Speaker-Ack" not in resp.headers


@pytest.mark.parametrize("path", ["/api/files", "/api/datastore/tables", "/ws/stt"])
async def test_the_header_never_reaches_an_owner_route(http, path):
    resp, body, _h = await member(http, "GET", path, header=assertion("GET", path))
    assert resp.status == 403
    assert json.loads(body)["error"] == sh.HEADER_ELSEWHERE_MESSAGE
    assert http.owner_hits == []


async def test_the_header_on_ws_passes_through_untouched(http):
    h = assertion("GET", "/ws")
    resp, _body, _ = await member(http, "GET", "/ws", header=h)
    assert resp.status == 200 and http.owner_hits == ["/ws"]
    assert "X-FD-Speaker-Ack" not in resp.headers
    assert h  # the nonce was not burned here: the socket handshake verifies it
    assert not any(n for n in speaker._NONCES)


async def test_owner_traffic_is_untouched(http):
    resp = await http.client.get("/api/files", params={"token": TOKEN})
    assert resp.status == 200 and (await resp.json()) == {"owner": True}
    assert http.stub.seen_principal == [None]


@pytest.mark.parametrize("bad", ["signature", "no_aud", "ws_aud", "expired", "garbage"])
async def test_a_refused_assertion_gets_401_without_ack(http, bad):
    path = "/api/speaker/files"
    if bad == "signature":
        header = speaker.sign_assertion(json.loads("{}") or {"v": 1}, "other-token")
    elif bad == "no_aud":
        header = assertion("GET", path, aud=None)
    elif bad == "ws_aud":
        header = assertion("GET", path, aud="ws")
    elif bad == "expired":
        now = int(time.time())
        header = assertion("GET", path, iat=now - 200, exp=now - 100)
    else:
        header = "v1.not.valid"
    resp, body, _ = await member(http, "GET", path, header=header)
    assert resp.status == 401 and json.loads(body) == {"error": "speaker assertion refused"}
    assert "X-FD-Speaker-Ack" not in resp.headers


async def test_a_replayed_assertion_is_refused(http):
    header = assertion("GET", "/api/speaker/files")
    resp, _, _ = await member(http, "GET", "/api/speaker/files", header=header)
    assert resp.status == 200 and acked(resp, header)
    resp, _, _ = await member(http, "GET", "/api/speaker/files", header=header)
    assert resp.status == 401 and "X-FD-Speaker-Ack" not in resp.headers


async def test_a_method_or_path_mismatch_does_not_burn_the_nonce(http):
    header = assertion("GET", "/api/speaker/files")
    resp, _, _ = await member(http, "GET", "/api/speaker/datastore/tables", header=header)
    assert resp.status == 401
    resp, _, _ = await member(http, "POST", "/api/speaker/files/delete", header=header,
                              json={"id": "x"})
    assert resp.status == 401
    resp, _, _ = await member(http, "GET", "/api/speaker/files", header=header)
    assert resp.status == 200 and acked(resp, header)


async def test_docker_and_unverified_members_get_403_with_ack(http):
    docker = Principal("u-ana", "Ana", "Olga", "A", "docker:helper:0123456789abcdef")
    resp, body, header = await member(http, "GET", "/api/speaker/files", p=docker)
    assert resp.status == 403 and acked(resp, header)
    assert json.loads(body)["error"] == sh.NOT_AVAILABLE_MESSAGE


async def test_a_public_agent_refuses_with_ack(http, monkeypatch):
    monkeypatch.setattr(get_config().web, "public_run", "computer")
    resp, body, header = await member(http, "GET", "/api/speaker/files")
    assert resp.status == 403 and acked(resp, header)
    assert json.loads(body)["error"] == sh.PUBLIC_AGENT_MESSAGE


async def test_an_unknown_member_path_is_404_with_ack(http):
    resp, body, header = await member(http, "GET", "/api/speaker/nope")
    assert resp.status == 404 and acked(resp, header)
    assert "error" in json.loads(body)


async def test_a_handler_crash_is_500_with_ack_and_no_detail(http, monkeypatch):
    async def _boom(*a, **k):
        raise RuntimeError("/Users/someone/secret/path exploded")

    monkeypatch.setattr(sa, "ensure_member_sessions", _boom)
    resp, body, header = await member(http, "GET", "/api/speaker/files")
    assert resp.status == 500 and acked(resp, header)
    assert json.loads(body) == {"error": "internal error"}


async def test_the_principal_is_unbound_after_each_request(http):
    resp, _, _ = await member(http, "GET", "/api/speaker/files")
    assert resp.status == 200
    assert speaker.current() is None
    resp = await http.client.get("/api/bound", params={"token": TOKEN})
    assert (await resp.json()) == {"bound": False}


# ── files: listing ───────────────────────────────────────────────────


async def test_the_listing_shape_creators_and_what_it_skips(http, saved, tmp_path):
    ws = saved.parent
    _put(saved / "output" / "run" / "owner.md", "o", None)
    time.sleep(0.01)
    _put(saved / "downloads" / BOB_SLUG / "bob.csv", "b", BOB)
    time.sleep(0.01)
    _put(saved / "tmp" / ANA_SLUG / "mine.txt", "m", ANA)
    _put(saved / "tmp" / "x" / ".hidden.md", "h", None)
    _put(saved / ".dot" / "in-dot-dir.md", "h", None)
    _put(saved / "tmp" / "node_modules" / "skipped.js", "s", None)
    _put(ws / "notes.md", "workspace", None)
    os.symlink(ws / "notes.md", saved / "tmp" / "x" / "link.md")
    os.symlink(ws, saved / "tmp" / "dirlink")
    resp, body, header = await member(http, "GET", "/api/speaker/files")
    assert resp.status == 200 and acked(resp, header)
    data = json.loads(body)
    assert data["truncated"] is False
    ids = [f["id"] for f in data["files"]]
    assert ids == [f"tmp/{ANA_SLUG}/mine.txt", f"downloads/{BOB_SLUG}/bob.csv", "output/run/owner.md"]
    first = data["files"][0]
    assert set(first) == {"id", "filename", "extension", "size", "modified", "mime_type",
                          "is_text", "created_by"}
    assert first["filename"] == "mine.txt" and first["extension"] == ".txt"
    assert first["size"] == 1 and first["is_text"] is True and first["mime_type"] == "text/plain"
    assert first["created_by"] == {"kind": "member", "user_id": "u-ana", "name": "Ana"}
    assert data["files"][1]["created_by"] == {"kind": "member", "user_id": "u-bob", "name": "Bob"}
    assert data["files"][2]["created_by"] == {"kind": "owner", "user_id": "", "name": ""}
    _no_host_paths(body.decode(), tmp_path)


async def test_the_listing_is_truncated_at_its_caps(http, saved, monkeypatch):
    for i in range(3):
        _put(saved / "tmp" / "x" / f"f{i}.md", "x", None)
    monkeypatch.setattr(sh, "MEMBER_FILES_LIST_MAX", 2)
    resp, body, _ = await member(http, "GET", "/api/speaker/files")
    data = json.loads(body)
    assert len(data["files"]) == 2 and data["truncated"] is True
    monkeypatch.setattr(sh, "MEMBER_FILES_LIST_MAX", 2000)
    monkeypatch.setattr(sh, "MEMBER_FILES_SCAN_MAX", 1)
    resp, body, _ = await member(http, "GET", "/api/speaker/files")
    data = json.loads(body)
    assert len(data["files"]) == 1 and data["truncated"] is True


async def test_another_members_legacy_file_is_not_listed_nor_served(http, saved, monkeypatch):
    sa.note_member_session(BOB_SLUG, "u-bob", "Bob")
    _put(saved / "tmp" / BOB_SLUG / "legacy.md", "old", "none")
    _put(saved / "tmp" / BOB_SLUG / "stamped.md", "new", BOB)
    resp, body, _ = await member(http, "GET", "/api/speaker/files")
    ids = [f["id"] for f in json.loads(body)["files"]]
    assert f"tmp/{BOB_SLUG}/stamped.md" in ids and f"tmp/{BOB_SLUG}/legacy.md" not in ids
    resp, _, _ = await member(http, "GET", "/api/speaker/files/raw",
                              params={"id": f"tmp/{BOB_SLUG}/legacy.md"})
    assert resp.status == 404
    # Bob sees his own.
    resp, body, _ = await member(http, "GET", "/api/speaker/files", p=BOB)
    assert f"tmp/{BOB_SLUG}/legacy.md" in [f["id"] for f in json.loads(body)["files"]]
    monkeypatch.setattr(sa, "LEGACY_MEMBER_FILES_SHARED", True)
    resp, body, _ = await member(http, "GET", "/api/speaker/files")
    assert f"tmp/{BOB_SLUG}/legacy.md" in [f["id"] for f in json.loads(body)["files"]]


async def test_a_raw_request_first_after_a_restart_still_hides_legacy_files(http, saved):
    # Bob's A2-era session is known only to the session store (no listing,
    # no socket since the restart): raw and delete backfill before deciding.
    from captain_claw.session import get_session_manager
    from captain_claw.tools.write import WriteTool

    old = await get_session_manager().create_session(
        name="spk-old", metadata={"speaker_id": "u-bob", "speaker_name": "Bob"})
    rel = f"tmp/{WriteTool._normalize_session_id(old.id)}/legacy.md"
    _put(saved / rel, "old")
    resp, _, _ = await member(http, "GET", "/api/speaker/files/raw", params={"id": rel})
    assert resp.status == 404
    resp, _, _ = await member(http, "POST", "/api/speaker/files/delete", json={"id": rel})
    assert resp.status == 404 and (saved / rel).exists()
    resp, body, _ = await member(http, "GET", "/api/speaker/files/raw", p=BOB, params={"id": rel})
    assert resp.status == 200 and body == b"old"


# ── files: raw ───────────────────────────────────────────────────────


async def test_raw_bytes_and_errors(http, saved, monkeypatch):
    _put(saved / "output" / "run" / "data.bin", b"\x00\x01payload", None)
    resp, body, header = await member(http, "GET", "/api/speaker/files/raw",
                                      params={"id": "output/run/data.bin"})
    assert resp.status == 200 and body == b"\x00\x01payload" and acked(resp, header)
    assert resp.headers["Content-Type"].startswith("application/octet-stream")
    for bad in ("", "/etc/passwd", "../x", "a//b", "a/./b", ".env", "a/.git/x", "a\\b",
                "a\x01b", "x" * 1025):
        resp, body, _ = await member(http, "GET", "/api/speaker/files/raw", params={"id": bad})
        assert resp.status == 400, bad
        assert json.loads(body) == {"error": "Invalid file"}
    resp, body, _ = await member(http, "GET", "/api/speaker/files/raw", params={"id": "output/none.md"})
    assert resp.status == 404 and json.loads(body) == {"error": "No such file"}
    os.symlink(saved.parent / "output", saved / "out-link")
    _put(saved.parent / "output" / "o.md", "owner output", None)
    resp, _, _ = await member(http, "GET", "/api/speaker/files/raw", params={"id": "out-link/o.md"})
    assert resp.status == 404
    monkeypatch.setattr(sh, "MEMBER_DOWNLOAD_MAX_BYTES", 4)
    resp, body, _ = await member(http, "GET", "/api/speaker/files/raw",
                                 params={"id": "output/run/data.bin"})
    assert resp.status == 413
    assert json.loads(body) == {"error": "That file is too large to open here (50 MB at most)"}


# ── files: upload ────────────────────────────────────────────────────


def _form(name: str, data: bytes) -> FormData:
    form = FormData(quote_fields=False)       # the filename exactly as FD sends it
    form.add_field("file", data, filename=name, content_type="application/octet-stream")
    return form


async def test_an_upload_lands_in_the_members_lane_folder(http, saved, tmp_path):
    resp, body, header = await member(http, "POST", "/api/speaker/files/upload",
                                      data=_form("..\\..\\Quarterly Report.csv", b"a,b\n1,2\n"))
    assert resp.status == 200, body
    assert acked(resp, header)
    item = json.loads(body)
    assert re.fullmatch(rf"downloads/{ANA_SLUG}/Quarterly_Report-\d{{8}}-\d{{6}}\.csv", item["id"])
    dest = saved / item["id"]
    assert dest.read_bytes() == b"a,b\n1,2\n"
    assert item["created_by"] == {"kind": "member", "user_id": "u-ana", "name": "Ana"}
    c = sa.creator_of(dest)
    assert (c.user_id, c.source) == ("u-ana", "stamp")
    assert http.stub.broadcasts == []
    assert [p.speaker_id for p in http.stub.session_calls] == ["u-ana"]
    _no_host_paths(body.decode(), tmp_path)
    # The lane's session is recorded as hers (folder fallback for later files).
    assert sa.creator_of(_put(saved / "tmp" / ANA_SLUG / "later.md", "x")).user_id == "u-ana"


@pytest.mark.parametrize("name", ["evil.html", "bundle.zip", "x.svg", "noext", ""])
async def test_uploads_of_other_types_are_refused(http, saved, name):
    resp, body, _ = await member(http, "POST", "/api/speaker/files/upload", data=_form(name, b"x"))
    assert resp.status == 400
    assert json.loads(body) == {"error": "That kind of file can't be uploaded here"}
    assert not (saved / "downloads").exists()


async def test_upload_size_empty_and_quota(http, saved, monkeypatch):
    monkeypatch.setattr(sh, "MEMBER_UPLOAD_MAX_BYTES", 10)
    resp, body, _ = await member(http, "POST", "/api/speaker/files/upload",
                                 data=_form("big.txt", b"x" * 11))
    assert resp.status == 413 and json.loads(body) == {"error": "That file is too large (25 MB at most)"}
    resp, body, _ = await member(http, "POST", "/api/speaker/files/upload", data=_form("e.txt", b""))
    assert resp.status == 400 and json.loads(body) == {"error": "That file is empty"}
    _put(saved / "downloads" / "old" / "mine.bin", b"y" * 8, ANA)
    monkeypatch.setattr(sh, "MEMBER_UPLOAD_QUOTA_BYTES", 10)
    resp, body, _ = await member(http, "POST", "/api/speaker/files/upload",
                                 data=_form("q.txt", b"zzz"))
    assert resp.status == 413
    assert json.loads(body) == {
        "error": "You've used your 200 MB for files on this agent — delete some first."}
    assert not (saved / "downloads" / ANA_SLUG).exists() or not any(
        (saved / "downloads" / ANA_SLUG).iterdir())


async def test_parallel_uploads_cannot_overrun_the_quota(http, saved, monkeypatch):
    # Each upload alone fits; together they don't — the quota check and the
    # write are serialised per member, so only the ones that fit land.
    monkeypatch.setattr(sh, "MEMBER_UPLOAD_QUOTA_BYTES", 10)
    results = await asyncio.gather(*(
        member(http, "POST", "/api/speaker/files/upload", data=_form(f"p{i}.txt", b"abcd"))
        for i in range(5)))
    statuses = sorted(resp.status for resp, _, _ in results)
    assert statuses == [200, 200, 413, 413, 413]
    assert sa.member_bytes("u-ana") == 8


async def test_an_upload_never_writes_through_a_symlink(http, saved, tmp_path, monkeypatch):
    target = tmp_path / "victim.txt"
    target.write_text("untouched")

    class _Frozen:
        @staticmethod
        def now(tz=None):
            from datetime import datetime as real

            return real(2026, 10, 6, 12, 0, 0, tzinfo=tz)

    monkeypatch.setattr(sh, "datetime", _Frozen)
    folder = saved / "downloads" / ANA_SLUG
    folder.mkdir(parents=True)
    os.symlink(target, folder / "note-20261006-120000.txt")
    resp, body, _ = await member(http, "POST", "/api/speaker/files/upload",
                                 data=_form("note.txt", b"member data"))
    assert resp.status == 200, body
    assert json.loads(body)["id"] == f"downloads/{ANA_SLUG}/note-20261006-120000-2.txt"
    assert target.read_text() == "untouched"
    for i in range(3, 10):
        (folder / f"note-20261006-120000-{i}.txt").write_text("taken")
    resp, body, _ = await member(http, "POST", "/api/speaker/files/upload",
                                 data=_form("note.txt", b"again"))
    assert resp.status == 409 and json.loads(body) == {"error": "Try again in a second"}


async def test_a_symlinked_upload_folder_is_refused(http, saved, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (saved / "downloads").mkdir(parents=True)
    os.symlink(elsewhere, saved / "downloads" / ANA_SLUG)
    resp, body, _ = await member(http, "POST", "/api/speaker/files/upload",
                                 data=_form("a.txt", b"x"))
    assert resp.status == 400 and json.loads(body) == {"error": "Invalid upload folder"}
    assert list(elsewhere.iterdir()) == []


# ── files: delete ────────────────────────────────────────────────────


async def test_delete_own_only(http, saved):
    mine = _put(saved / "downloads" / "old-session" / "mine.txt", "m", ANA)
    owners = _put(saved / "output" / "run" / "owner.md", "o", None)
    bobs = _put(saved / "downloads" / BOB_SLUG / "bob.txt", "b", BOB)
    for path in (owners, bobs):
        rel = sa.rel_key(path)
        resp, body, header = await member(http, "POST", "/api/speaker/files/delete", json={"id": rel})
        assert resp.status == 403 and acked(resp, header)
        assert json.loads(body) == {"error": speaker.FILE_DELETE_NOT_YOURS}
        assert path.exists()
    resp, body, _ = await member(http, "POST", "/api/speaker/files/delete",
                                 json={"id": "downloads/old-session/mine.txt"})
    assert resp.status == 200 and json.loads(body) == {"ok": True}
    assert not mine.exists()
    mine.write_text("recreated by someone else")
    assert sa.creator_of(mine).source == "none"            # the record is gone
    resp, _, _ = await member(http, "POST", "/api/speaker/files/delete", json={"id": "nope/x.md"})
    assert resp.status == 404
    resp, _, _ = await member(http, "POST", "/api/speaker/files/delete", json={"id": "../x"})
    assert resp.status == 400
    resp, _, _ = await member(http, "POST", "/api/speaker/files/delete", data=b"not json")
    assert resp.status == 400


async def test_delete_under_another_spelling_is_refused(http, saved, tmp_path):
    probe = tmp_path / "CaseProbe"
    probe.write_text("x")
    if not (tmp_path / "caseprobe").exists():
        pytest.skip("case-sensitive filesystem")
    sa.note_member_session(ANA_SLUG, "u-ana", "Ana")
    owner_report = _put(saved / "output" / ANA_SLUG / "report.md", "owner", None)
    resp, body, _ = await member(http, "POST", "/api/speaker/files/delete",
                                 json={"id": f"output/{ANA_SLUG}/REPORT.md"})
    assert resp.status == 403 and json.loads(body) == {"error": speaker.FILE_DELETE_NOT_YOURS}
    assert owner_report.exists()


# ── datastore ────────────────────────────────────────────────────────


async def _seed(mgr):
    await mgr.create_table("owners", COLS)
    await mgr.insert_rows("owners", [{"k": "o", "v": "=SUM(A1:A9)"}])
    with speaker_bound(BOB):
        await mgr.insert_rows("owners", [{"k": "b", "v": "=HYPERLINK(\"http://x\",\"y\")"}])
        await mgr.create_table("bobs", COLS)


class speaker_bound:  # noqa: N801 — reads like a with-statement
    def __init__(self, p):
        self.p = p

    def __enter__(self):
        self.tok = speaker.bind(self.p)

    def __exit__(self, *exc):
        speaker.reset(self.tok)


async def test_datastore_tables_shape(http, dm):
    await _seed(dm)
    resp, body, header = await member(http, "GET", "/api/speaker/datastore/tables")
    assert resp.status == 200 and acked(resp, header)
    tables = {t["name"]: t for t in json.loads(body)["tables"]}
    assert set(tables) == {"owners", "bobs"}
    assert set(tables["owners"]) == {"name", "columns", "row_count", "created_at", "updated_at",
                                     "created_by"}
    assert tables["owners"]["columns"] == [{"name": "k", "type": "text", "position": 0},
                                           {"name": "v", "type": "text", "position": 1}]
    assert tables["owners"]["created_by"] == {"kind": "owner", "user_id": "", "name": ""}
    assert tables["bobs"]["created_by"] == {"kind": "member", "user_id": "u-bob", "name": "Bob"}


async def test_datastore_rows_shape_and_paging(http, dm):
    await _seed(dm)
    path = "/api/speaker/datastore/tables/owners/rows"
    resp, body, _ = await member(http, "GET", path, params={"order_dir": "desc", "limit": "1"})
    assert resp.status == 200
    data = json.loads(body)
    assert data["columns"] == ["_id", "k", "v"] and data["total"] == 2
    assert data["limit"] == 1 and data["offset"] == 0
    assert data["rows"] == [{"_id": 2, "k": "b", "v": "=HYPERLINK(\"http://x\",\"y\")",
                             "_creator": {"kind": "member", "user_id": "u-bob", "name": "Bob"}}]
    resp, body, _ = await member(http, "GET", path, params={"order_by": "_created_by"})
    assert [r["k"] for r in json.loads(body)["rows"]] == ["o", "b"]
    resp, body, _ = await member(http, "GET", path, params={"order_by": "nope"})
    assert resp.status == 400 and json.loads(body) == {"error": "Unknown column"}
    resp, body, _ = await member(http, "GET", path, params={"limit": "9999"})
    assert json.loads(body)["limit"] == 500
    resp, _, _ = await member(http, "GET", path, params={"offset": "-1"})
    assert resp.status == 400
    resp, body, _ = await member(http, "GET", "/api/speaker/datastore/tables/ghost/rows")
    assert resp.status == 404 and json.loads(body) == {"error": "No such table"}
    resp, _, _ = await member(http, "GET", "/api/speaker/datastore/tables/Bad-Name/rows")
    assert resp.status == 400


async def test_member_export_defuses_every_formula(http, dm):
    await _seed(dm)
    path = "/api/speaker/datastore/tables/owners/export"
    resp, body, header = await member(http, "GET", path, params={"format": "csv"})
    assert resp.status == 200 and acked(resp, header)
    rows = list(csv.reader(io.StringIO(body.decode())))
    assert rows[0] == ["_id", "k", "v"]
    assert [r[2] for r in rows[1:]] == ["'=SUM(A1:A9)", "'=HYPERLINK(\"http://x\",\"y\")"]
    resp, body, _ = await member(http, "GET", path, params={"format": "json"})
    assert resp.status == 200 and json.loads(body)[0]["k"] == "o"
    resp, body, _ = await member(http, "GET", path, params={"format": "xlsx"})
    assert resp.status == 200 and body[:2] == b"PK"
    resp, body, _ = await member(http, "GET", path, params={"format": "html"})
    assert resp.status == 400 and json.loads(body) == {"error": "Unsupported format"}
    resp, _, _ = await member(http, "GET", "/api/speaker/datastore/tables/ghost/export")
    assert resp.status == 404


async def test_member_routes_never_write_the_datastore(http, dm):
    await _seed(dm)
    for method, path in (("POST", "/api/speaker/datastore/tables"),
                         ("DELETE", "/api/speaker/datastore/tables/owners/rows")):
        resp, _, header = await member(http, method, path)
        assert resp.status in (404, 405) and acked(resp, header)
    assert [t.name for t in await dm.list_tables()] == ["bobs", "owners"]


# ── lane sessions ────────────────────────────────────────────────────


async def test_a_first_upload_racing_the_first_socket_yields_one_session(tmp_path, monkeypatch):
    from captain_claw.session import get_session_manager
    from captain_claw.tools.registry import ToolRegistry
    from captain_claw.web_server import WebServer

    sm = get_session_manager()
    await sm._ensure_db()      # an agent's session DB is long open by its first member
    server = WebServer.__new__(WebServer)
    server.config = get_config()
    server.agent = None
    server._init_speaker_state()

    async def _build(session, send, **kw):
        return types.SimpleNamespace(
            session=session, tools=ToolRegistry(),
            _current_session_slug=lambda: session.id,
        )

    server._build_scoped_agent = _build
    app = web.Application(middlewares=[
        create_auth_middleware(server.config.web), sh.create_speaker_http_middleware(server),
    ])
    sh.register_speaker_routes(app, server)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        h = types.SimpleNamespace(client=client)
        upload, agent = await asyncio.wait_for(asyncio.gather(
            member(h, "POST", "/api/speaker/files/upload", data=_form("a.txt", b"x")),
            server._get_speaker_agent(ANA),
        ), 30)
        assert upload[0].status == 200, upload[1]
        ids = json.loads(await sm.get_app_state("speaker_sessions:u-ana") or "[]")
        assert len(ids) == 1 and agent.session.id == ids[0]
        assert json.loads(upload[1])["id"].startswith("downloads/")
        assert sa.creator_of(Path(get_config().workspace.path) / "saved" / "downloads"
                             / ids[0] / "x").source in ("folder", "none")
        # The lane session is a recorded member session.
        later = Path(get_config().workspace.path) / "saved" / "tmp" / ids[0] / "later.md"
        assert sa.creator_of(_put(later, "x")).user_id == "u-ana"
    finally:
        await asyncio.wait_for(client.close(), 10)
        await sm.close()


# ── the owner's REST routes ──────────────────────────────────────────


@pytest.fixture
async def owner_api(dm):
    from captain_claw.web import rest_datastore, rest_files

    stub = types.SimpleNamespace(agent=None, _orchestrator=None, _broadcast=lambda m: None)

    def _route(fn):
        async def handler(request):
            return await fn(stub, request)
        return handler

    app = web.Application(middlewares=[create_auth_middleware(get_config().web)])
    app.router.add_get("/api/files", _route(rest_files.list_files))
    app.router.add_get("/api/files/content", _route(rest_files.get_file_content))
    app.router.add_post("/api/files/content", _route(rest_files.save_file_content))
    app.router.add_post("/api/files/delete", _route(rest_files.delete_files))
    app.router.add_get("/api/datastore/tables", _route(rest_datastore.list_tables))
    app.router.add_get("/api/datastore/tables/{name}", _route(rest_datastore.describe_table))
    app.router.add_get("/api/datastore/tables/{name}/rows", _route(rest_datastore.query_rows))
    app.router.add_get("/api/datastore/tables/{name}/export", _route(rest_datastore.export_table))
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        yield client
    finally:
        await client.close()
        await _close_sessions()


async def test_owner_file_listing_carries_creators(owner_api, saved):
    ws = saved.parent
    member_file = _put(saved / "downloads" / BOB_SLUG / "bob.csv", "b", BOB)
    owner_file = _put(saved / "output" / "run" / "o.md", "o", None)
    outside = _put(ws / "output" / "run" / "out.md", "x", None)
    resp = await owner_api.get("/api/files", params={"token": TOKEN})
    by_phys = {f["physical"]: f for f in await resp.json()}
    assert by_phys[str(member_file.resolve())]["created_by"] == {
        "kind": "member", "user_id": "u-bob", "name": "Bob"}
    assert by_phys[str(owner_file.resolve())]["created_by"] == {
        "kind": "owner", "user_id": "", "name": ""}
    assert by_phys[str(outside.resolve())]["created_by"] is None
    resp = await owner_api.get("/api/files/content",
                               params={"token": TOKEN, "path": str(member_file)})
    assert (await resp.json())["created_by"]["kind"] == "member"
    resp = await owner_api.get("/api/files/content", params={"token": TOKEN, "path": str(outside)})
    assert (await resp.json())["created_by"] is None


async def test_owner_content_first_after_a_restart_still_names_a_legacy_member(owner_api, saved):
    # An A2-era member session known only to the session store, an unstamped
    # file in its folder, and /api/files/content as the first route hit: the
    # file must read as the member's (FD's /deck/view and the glasses viewer
    # key off this), not fall back to the owner.
    from captain_claw.session import get_session_manager
    from captain_claw.tools.write import WriteTool

    old = await get_session_manager().create_session(
        name="spk-old", metadata={"speaker_id": "u-bob", "speaker_name": "Bob"})
    legacy = _put(saved / "tmp" / WriteTool._normalize_session_id(old.id) / "deck.html",
                  "<script>x</script>")
    resp = await owner_api.get("/api/files/content", params={"token": TOKEN, "path": str(legacy)})
    assert resp.status == 200
    assert (await resp.json())["created_by"] == {"kind": "member", "user_id": "u-bob", "name": "Bob"}


async def test_owner_routes_name_a_member_for_their_own_vfs_files(owner_api, tmp_path):
    # A member's write to their own vfs:<project>/… lands under THEIR VFS
    # root and the file registry lists it to the owner. Outside saved/, yet it
    # must not read as the owner's (None) — FD's /deck/view and glasses viewer
    # would then render the member's markup at FD's origin.
    from captain_claw.session import get_session_manager

    sm = get_session_manager()
    vfs = tmp_path / "fd-data" / "vfs"
    member_md = _put(vfs / "u-bob" / "notes" / "evil.md", "<img src=x onerror=alert(1)>")
    owner_md = _put(vfs / "local" / "notes" / "mine.md", "# mine")
    await sm.register_file("vfs:notes/evil.md", str(member_md), session_id=BOB_SLUG)
    await sm.register_file("vfs:notes/mine.md", str(owner_md), session_id="owner-s")
    resp = await owner_api.get("/api/files", params={"token": TOKEN})
    by_phys = {f["physical"]: f for f in await resp.json()}
    assert by_phys[str(member_md)]["created_by"] == {"kind": "member", "user_id": "u-bob", "name": ""}
    assert by_phys[str(owner_md)]["created_by"] is None
    resp = await owner_api.get("/api/files/content", params={"token": TOKEN, "path": str(member_md)})
    assert resp.status == 200
    assert (await resp.json())["created_by"]["kind"] == "member"
    resp = await owner_api.get("/api/files/content", params={"token": TOKEN, "path": str(owner_md)})
    assert (await resp.json())["created_by"] is None


async def test_owner_save_keeps_and_delete_forgets_the_creator(owner_api, saved):
    member_file = _put(saved / "downloads" / BOB_SLUG / "bob.md", "b", BOB)
    resp = await owner_api.post("/api/files/content", params={"token": TOKEN},
                                json={"path": str(member_file), "content": "owner edit"})
    assert resp.status == 200
    assert sa.creator_of(member_file).user_id == "u-bob"
    resp = await owner_api.post("/api/files/delete", params={"token": TOKEN},
                                json={"paths": [str(member_file)]})
    assert resp.status == 200 and not member_file.exists()
    member_file.write_text("someone else")
    assert sa.creator_of(member_file).source == "none"


async def test_owner_datastore_routes_carry_creators(owner_api, dm):
    await _seed(dm)
    resp = await owner_api.get("/api/datastore/tables", params={"token": TOKEN})
    tables = {t["name"]: t for t in await resp.json()}
    assert tables["bobs"]["created_by"] == {"kind": "member", "user_id": "u-bob", "name": "Bob"}
    assert tables["owners"]["created_by"]["kind"] == "owner"
    resp = await owner_api.get("/api/datastore/tables/bobs", params={"token": TOKEN})
    assert (await resp.json())["created_by"]["user_id"] == "u-bob"
    resp = await owner_api.get("/api/datastore/tables/owners/rows", params={"token": TOKEN})
    data = await resp.json()
    assert "creators" not in data
    assert [r["_creator"]["kind"] for r in data["rows"]] == ["owner", "member"]
    assert data["columns"] == ["_id", "k", "v"]


async def test_owner_export_defuses_only_member_rows(owner_api, dm):
    await _seed(dm)
    resp = await owner_api.get("/api/datastore/tables/owners/export",
                               params={"token": TOKEN, "format": "csv"})
    rows = list(csv.reader(io.StringIO(await resp.text())))
    assert [r[2] for r in rows[1:]] == ["=SUM(A1:A9)", "'=HYPERLINK(\"http://x\",\"y\")"]
    resp = await owner_api.get("/api/datastore/tables/owners/export",
                               params={"token": TOKEN, "format": "json"})
    assert [r["v"] for r in json.loads(await resp.text())] == [
        "=SUM(A1:A9)", "=HYPERLINK(\"http://x\",\"y\")"]
