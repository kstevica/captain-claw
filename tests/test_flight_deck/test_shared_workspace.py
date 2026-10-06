"""PR C — a shared agent's saved files and datastore as a commons (Flight Deck side).

Pinned here (contract part 1b §4):

* pure helpers: file-id and table validation, view / download / export
  headers, speaker and creator display names, the rate limit;
* the member routes: the refusal ladder (sharing off, bad ref, unknown agent,
  the owner, admins without membership, non-members, docker, stopped, rate)
  with no agent call; the browser can't steer the target; only
  ``/api/speaker/*`` paths, a fresh signed assertion per request bound to its
  method and path, no grant header or member marker; the ack gate (old agents
  → 426, anything unacknowledged → 502, its body never relayed); relayed agent
  errors; files / datastore mapping (creators by current name, no ids or
  emails, no host paths); view / download headers; upload (body guard before
  membership, membership before the body, fresh membership, type / size /
  lane checks, usage row); delete (fresh membership, usage row);
* ``GET /fd/shared-agents``: ``member_workspace`` and the ``datastore``
  capability;
* the owner proxies show member creators by current name; the owner's file
  view gets nosniff and a sandbox CSP for active content; ``/deck/view``
  refuses member-created files;
* the one-time owner bell; no token, assertion or JWT in the logs.

Same deck as ``test_agent_sharing`` (real FlightDeckDB + process registry in
tmp dirs, Docker faked). The agent is faked with ``httpx.MockTransport``: it
verifies every request the way a PR C agent would, without importing agent
code. Nothing touches ``~/.captain-claw``, a real FD data dir or a port.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import hmac
import inspect
import json
import logging
import textwrap

import httpx
import pytest
from fastapi.responses import Response as FastAPIResponse
from starlette.requests import Request

from captain_claw.flight_deck import agent_secret, glasses_bridge, server
from captain_claw.flight_deck import agent_sharing as sharing
from captain_claw.flight_deck import shared_workspace as sw
from test_flight_deck import test_agent_sharing as base
from test_flight_deck.test_agent_sharing import (
    ADMIN,
    HELPER_TOK,
    MEMBER,
    OTHER,
    OWNER,
    REF,
    FakeContainer,
    _client,
    _hdr,
)

deck = base.deck          # the A1 deck fixture

M2 = "u-member2"
BLANK = "u-blank"          # a user with no display name
GONE = "u-gone"            # never a user of this deck (a deleted member)
HELPER_PORT = 24987        # the deck fixture's helper port
BOX_INST = "7" * 16
BOX_REF = f"docker:box:{BOX_INST}"
BOX_TOK = "box-tok"
SLEEPY_REF = "process:sleepy:4444444444444444"   # a stopped process agent of OWNER
SHARED_SECRET = "test-agent-shared-secret"

# Part 0c §1, verbatim (U+2019 in "agent’s", ASCII apostrophes elsewhere).
COMMONS_TEXT = (
    "Members can also open everything in this agent’s saved/ folder — including what is "
    "already there: files you uploaded in your own chats, screenshots and browser captures, "
    "script outputs, the scripts and tools it saved for you (check them for passwords or keys), "
    "and what its channels and automations saved, such as WhatsApp or email attachments — and "
    "see every table and row in its datastore. They can add their own files, tables and rows, "
    "shown with their name, and change or delete only what they added; you can change or "
    "delete all of it. Its other files (workspace, output/, workflows/) stay yours. Your agent "
    "reads what members add, so treat it as untrusted input, especially in automations.")

UPLOAD_BLOCK = {"max_bytes": 26214400, "extensions": [
    ".avi", ".bmp", ".csv", ".doc", ".docx", ".gif", ".jpeg", ".jpg", ".m4v", ".md", ".mkv",
    ".mov", ".mp4", ".pdf", ".png", ".ppt", ".pptx", ".txt", ".webm", ".webp", ".xls", ".xlsx"]}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """No member caches or rate windows from other tests; tmp homes; the
    agent secret from the env (never a file in a real home)."""
    monkeypatch.setattr(sharing, "_MEMBER_GEN", {})
    monkeypatch.setattr(sharing, "_MEMBER_GEN_REF", {})
    monkeypatch.setattr(sharing, "_MEMBER_CACHE", {})
    sw._reset_for_tests()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FD_AGENT_SHARED_SECRET", SHARED_SECRET)
    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path / "claw-vfs"))
    for var in ("FD_PUBLIC_URL", "FD_LOCKDOWN", "FD_GLASSES_BRIDGE_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    agent_secret.reset_cache_for_tests()
    yield
    sw._reset_for_tests()
    agent_secret.reset_cache_for_tests()


async def _add_user(db, uid: str, name: str) -> None:
    await db._db.execute(
        "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
        " VALUES (?, ?, 'h', ?, 'user', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
        (uid, f"{uid}@x.co", name))
    await db._db.commit()


@pytest.fixture
async def wdeck(deck):
    """OWNER's helper (process, running) shared with MEMBER and M2; OWNER's box
    (docker) and sleepy (process, stopped) shared with MEMBER; OTHER no member."""
    db = deck.db
    await _add_user(db, M2, "Max Member")
    await _add_user(db, BLANK, "")
    deck.containers.append(FakeContainer(deck.containers, "box", OWNER, BOX_TOK, 24991,
                                         instance=BOX_INST))
    for ref, uid in ((REF, MEMBER), (REF, M2), (BOX_REF, MEMBER), (SLEEPY_REF, MEMBER)):
        await db.create_share("agent", ref, OWNER, uid, "view")
    return deck


# ── The fake PR C agent ───────────────────────────────────────────────────


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _creator(kind: str, uid: str = "", name: str = "") -> dict:
    return {"kind": kind, "user_id": uid, "name": name}


def _afile(fid: str, creator: dict | None, **extra) -> dict:
    name = fid.rsplit("/", 1)[-1]
    item = {"id": fid, "filename": name, "extension": "." + name.rsplit(".", 1)[-1],
            "size": 12, "modified": 1760000000.5, "mime_type": "text/plain", "is_text": True}
    if creator is not None:
        item["created_by"] = creator
    item.update(extra)
    return item


AGENT_FILES = [
    _afile("downloads/olga-sess/report.pdf", _creator("owner"),
           physical="/Users/olga/.captain-claw/workspace/saved/downloads/olga-sess/report.pdf"),
    _afile("downloads/mia-sess/notes.md", _creator("member", MEMBER, "Mia Member")),
    _afile("downloads/max-sess/data.csv", _creator("member", M2, "old")),
    _afile("downloads/zed-sess/z.txt", _creator("member", GONE, "Zed")),
    _afile("downloads/blank-sess/b.txt", _creator("member", BLANK, "Snap")),
    _afile("media/shot.png", None),
    _afile("../x", _creator("owner")),
    _afile(".env", _creator("owner")),
    "not a dict",
]

AGENT_TABLES = [
    {"name": "people", "columns": [{"name": "name", "type": "text", "position": 0}],
     "row_count": 3, "created_at": "2026-10-01T00:00:00Z", "updated_at": "2026-10-02T00:00:00Z",
     "created_by": _creator("member", M2, "old")},
    {"name": "Bad Name", "columns": [], "row_count": 0, "created_at": "", "updated_at": "",
     "created_by": _creator("owner")},
    {"name": "mine", "columns": [], "row_count": 0, "created_at": "", "updated_at": "",
     "created_by": _creator("member", MEMBER, "Mia")},
    {"name": "legacy", "columns": [], "row_count": 0, "created_at": "", "updated_at": ""},
]

AGENT_ROWS = {
    "columns": ["_id", "name", "_created_by", "_created_by_name"],
    "rows": [
        {"_id": 1, "name": "a", "_created_by": M2, "_created_by_name": "old",
         "_creator": _creator("member", M2, "old")},
        {"_id": 2, "name": "b", "_created_by": "", "_created_by_name": "",
         "_creator": _creator("owner")},
        {"_id": 3, "name": "c", "_creator": _creator("member", MEMBER, "Mia")},
    ],
    "total": 3, "offset": 0, "limit": 500,
}

EXPORT_CSV = b"name\r\na\r\nb\r\n"


class FakeWorkspaceAgent:
    """The agent's ``/api/speaker/*`` routes as far as FD can tell. Every
    request is checked the way a PR C agent checks it (token, signature, aud,
    method, path, speaker, ref, lane, lifetime, single-use nonce); a request
    that fails gets 401 without an ack (and is recorded in ``errors``)."""

    def __init__(self, expect_sub: str = MEMBER):
        self.expect_sub = expect_sub
        self.requests: list[dict] = []
        self.errors: list[str] = []
        self.nonces: set[str] = set()
        self.ack = "ok"            # ok | none | wrong
        self.reply = None          # request -> httpx.Response, instead of the canned routes

    def _verify(self, request: httpx.Request, header: str) -> tuple[dict | None, str | None]:
        parts = header.split(".")
        if len(parts) != 3 or parts[0] != "v1":
            return None, "header shape"
        want = _b64(hmac.new(sharing._speaker_key(HELPER_TOK), ("v1." + parts[1]).encode("ascii"),
                             hashlib.sha256).digest())
        if not hmac.compare_digest(want, parts[2]):
            return None, "signature"
        payload = json.loads(_unb64(parts[1]))
        checks = {
            "v": payload.get("v") == 1,
            "aud": payload.get("aud") == "http",
            "m": payload.get("m") == request.method,
            "p": payload.get("p") == request.url.path,
            "sub": payload.get("sub") == self.expect_sub,
            "owner": payload.get("owner") == OWNER,
            "ref": payload.get("ref") == REF,
            "lane": payload.get("lane") in ("A", "B", "C"),
            "ttl": payload.get("exp", 0) - payload.get("iat", 0) == 60,
            "conn": str(payload.get("conn", "")).startswith("http-"),
            "nonce": payload.get("nonce") not in self.nonces,
            "token": request.url.params.get("token") == HELPER_TOK,
            "host": request.url.host == "localhost",
            "port": request.url.port == HELPER_PORT,
        }
        self.nonces.add(payload.get("nonce"))
        failed = [k for k, ok in checks.items() if not ok]
        return payload, (",".join(failed) or None)

    def handler(self, request: httpx.Request) -> httpx.Response:
        header = request.headers.get(sharing.SPEAKER_HEADER, "")
        payload, problem = self._verify(request, header)
        self.requests.append({
            "method": request.method, "path": request.url.path,
            "params": dict(request.url.params), "headers": dict(request.headers),
            "header": header, "payload": payload, "body": request.read()})
        if problem:
            self.errors.append(f"{request.method} {request.url.path}: {problem}")
            return httpx.Response(401, json={"error": "speaker assertion refused"})
        resp = self.reply(request) if self.reply is not None else self._canned(request, payload)
        if self.ack == "ok":
            resp.headers[sw.SPEAKER_ACK_HEADER] = sharing.speaker_ack_for(header)
        elif self.ack == "wrong":
            resp.headers[sw.SPEAKER_ACK_HEADER] = "0" * 16
        return resp

    def _canned(self, request: httpx.Request, payload: dict) -> httpx.Response:
        key = (request.method, request.url.path)
        if key == ("GET", "/api/speaker/files"):
            return httpx.Response(200, json={"files": AGENT_FILES, "truncated": True})
        if key == ("GET", "/api/speaker/files/raw"):
            fid = request.url.params.get("id", "")
            return httpx.Response(200, content=b"<script>x</script>" + fid.encode(),
                                  headers={"Content-Type": "text/html"})
        if key == ("POST", "/api/speaker/files/upload"):
            return httpx.Response(200, json=_afile(
                "downloads/mia-sess/a-20261006T120000Z.csv",
                _creator("member", payload["sub"], payload["name"]),
                physical="/Users/olga/x/saved/downloads/mia-sess/a.csv"))
        if key == ("POST", "/api/speaker/files/delete"):
            return httpx.Response(200, json={"ok": True})
        if key == ("GET", "/api/speaker/datastore/tables"):
            return httpx.Response(200, json={"tables": AGENT_TABLES})
        if key == ("GET", "/api/speaker/datastore/tables/people/rows"):
            return httpx.Response(200, json=AGENT_ROWS)
        if key == ("GET", "/api/speaker/datastore/tables/people/export"):
            return httpx.Response(200, content=EXPORT_CSV, headers={"Content-Type": "text/html"})
        return httpx.Response(404, json={"error": "Not found"})


@pytest.fixture
def agent(monkeypatch):
    fake = FakeWorkspaceAgent()

    def client(timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(fake.handler), timeout=timeout,
                                 trust_env=False, follow_redirects=False)

    monkeypatch.setattr(sw, "_agent_client", client)
    yield fake
    assert fake.errors == []


# ── Calling the routes ────────────────────────────────────────────────────

ROUTES = ("list", "view", "download", "upload", "delete", "tables", "rows", "export")
MIA_FILE = "downloads/mia-sess/notes.md"


async def _hit(c, kind: str, uid: str = MEMBER, ref: str = REF, extra: dict | None = None,
               headers: dict | None = None) -> httpx.Response:
    h = {**_hdr(uid), **(headers or {})}
    q = {"ref": ref, **(extra or {})}
    if kind == "list":
        return await c.get("/fd/shared-agents/files", params=q, headers=h)
    if kind in ("view", "download"):
        return await c.get(f"/fd/shared-agents/files/{kind}", params={"id": MIA_FILE, **q},
                           headers=h)
    if kind == "upload":
        return await c.post("/fd/shared-agents/files/upload", params={"lane": "A", **q},
                            files={"file": ("a.csv", b"a,b\n1,2\n", "text/csv")}, headers=h)
    if kind == "delete":
        return await c.post("/fd/shared-agents/files/delete", params=q, json={"id": MIA_FILE},
                            headers=h)
    if kind == "tables":
        return await c.get("/fd/shared-agents/datastore/tables", params=q, headers=h)
    if kind == "rows":
        return await c.get("/fd/shared-agents/datastore/tables/people/rows", params=q, headers=h)
    if kind == "export":
        return await c.get("/fd/shared-agents/datastore/tables/people/export",
                           params={"format": "csv", **q}, headers=h)
    raise AssertionError(kind)


def _detail(r: httpx.Response) -> str:
    return r.json().get("detail")


# ── Pure helpers ──────────────────────────────────────────────────────────


class TestPureHelpers:
    @pytest.mark.parametrize("fid,ok", [
        ("output/abc/x.md", True), ("a", True), ("downloads/s/ä ö.csv", True),
        ("x" * 1024, True),
        ("", False), ("/x", False), ("a//b", False), ("a/../b", False), ("../x", False),
        (".env", False), ("a/.git/x", False), ("a\\b", False), ("a\x00b", False),
        ("a\nb", False), ("a\x7fb", False), ("x" * 1025, False), ("a/", False), ("./a", False),
        (None, False), (5, False),
    ])
    def test_valid_file_id(self, fid, ok):
        assert sw.valid_file_id(fid) is ok

    @pytest.mark.parametrize("name,ok", [
        ("people", True), ("a_1", True), ("x" * 128, True),
        ("Bad Name", False), ("People", False), ("a-b", False), ("", False), ("x" * 129, False),
        ("a\n", False), ("a;drop", False), (None, False),
    ])
    def test_valid_table(self, name, ok):
        assert sw.valid_table(name) is ok

    @pytest.mark.parametrize("name,ok", [
        ("_id", True), ("name", True), ("_created_by", True), ("name;drop", False),
        ("__x", True), ("-x", False), ("Name", False), ("", False), ("a b", False),
        ("_" + "x" * 128, True), ("x" * 129, False),
    ])
    def test_valid_order_by(self, name, ok):
        assert sw.valid_order_by(name) is ok

    @pytest.mark.parametrize("filename,media", [
        ("a.png", "image/png"), ("a.JPG", "image/jpeg"), ("a.jpeg", "image/jpeg"),
        ("a.gif", "image/gif"), ("a.webp", "image/webp"), ("a.bmp", "image/bmp"),
        ("a.svg", "image/svg+xml"), ("a.pdf", "application/pdf"), ("a.mp3", "audio/mpeg"),
        ("a.wav", "audio/wav"), ("a.ogg", "audio/ogg"), ("a.m4a", "audio/mp4"),
        ("a.mp4", "video/mp4"), ("a.m4v", "video/mp4"), ("a.webm", "video/webm"),
        ("a.mov", "video/quicktime"),
        ("a.html", "text/plain; charset=utf-8"), ("a.htm", "text/plain; charset=utf-8"),
        ("a.js", "text/plain; charset=utf-8"), ("a.xml", "text/plain; charset=utf-8"),
        ("a.csv", "text/plain; charset=utf-8"), ("noext", "text/plain; charset=utf-8"),
    ])
    def test_view_headers(self, filename, media):
        got, headers = sw.view_headers(filename)
        assert got == media
        assert headers["Content-Disposition"] == "inline"
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["Cache-Control"] == "no-store"
        if media == "application/pdf":
            assert "Content-Security-Policy" not in headers
        else:
            assert headers["Content-Security-Policy"] == "sandbox"

    def test_download_and_export_headers(self):
        assert sw.download_headers("ä b;\".txt") == {
            "Content-Type": "application/octet-stream",
            "Content-Disposition": "attachment; filename*=UTF-8''%C3%A4%20b%3B%22.txt",
            "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"}
        for fmt, media in (("csv", "text/csv; charset=utf-8"), ("json", "application/json"),
                           ("xlsx", "application/vnd.openxmlformats-officedocument."
                                    "spreadsheetml.sheet")):
            assert sw.export_headers("people", fmt) == (media, {
                "Content-Disposition": f'attachment; filename="people.{fmt}"',
                "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"})

    def test_speaker_display_name(self):
        assert sw.speaker_display_name({"display_name": "Mia  Member", "email": "m@x.co"}) == (
            "Mia Member")
        assert sw.speaker_display_name({"display_name": "  ", "email": "mia.k@x.co"}) == "mia.k"
        assert sw.speaker_display_name({"email": "mia@x.co"}) == "mia"
        assert sw.speaker_display_name({"display_name": "a\n\tb"}) == "a b"
        assert sw.speaker_display_name({"display_name": "x" * 300}) == "x" * 120

    async def test_creator_display_name(self, wdeck):
        db = wdeck.db
        assert await sw.creator_display_name(db, M2, "old") == "Max Member"   # not the snapshot
        assert await sw.creator_display_name(db, BLANK, "Snap") == BLANK      # email local part
        assert await sw.creator_display_name(db, GONE, "Zed") == "Zed (former member)"
        assert await sw.creator_display_name(db, GONE, "") == "Former member"
        assert await sw.creator_display_name(db, "", None) == "Former member"
        assert await sw.creator_display_name(db, GONE, " Zed\n  Z " + "y" * 200) == (
            "Zed Z " + "y" * 74 + " (former member)")
        cache: dict = {}
        assert await sw.creator_display_name(db, M2, "old", cache) == "Max Member"
        await db._db.execute("UPDATE users SET display_name = 'Renamed' WHERE id = ?", (M2,))
        await db._db.commit()
        assert await sw.creator_display_name(db, M2, "old", cache) == "Max Member"  # per call
        assert await sw.creator_display_name(db, M2, "old") == "Renamed"
        # A deleted user's snapshot is used per item even with a shared cache.
        assert await sw.creator_display_name(db, GONE, "Zed", cache) == "Zed (former member)"
        assert await sw.creator_display_name(db, GONE, "Zoe", cache) == "Zoe (former member)"

    def test_rate_limit(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr(sw.time, "monotonic", lambda: now[0])
        for _ in range(sw.MEMBER_HTTP_RATE_PER_MIN):
            sw.check_rate(REF, MEMBER)
        with pytest.raises(sw.HTTPException) as exc:
            sw.check_rate(REF, MEMBER)
        assert exc.value.status_code == 429 and exc.value.detail == sw.RATE_DETAIL
        sw.check_rate(REF, M2)                      # per member
        sw.check_rate(BOX_REF, MEMBER)              # per agent
        now[0] += 60.0
        sw.check_rate(REF, MEMBER)                  # the window moved on

    def test_rate_limit_keys_bounded(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr(sw.time, "monotonic", lambda: now[0])
        for i in range(5000):
            sw.check_rate(REF, f"u{i}")
        assert len(sw._RATE) <= 4096

    async def test_agent_client_is_local_only(self):
        async with sw._agent_client(5.0) as client:
            assert client.trust_env is False and client.follow_redirects is False
            assert client.timeout.read == 5.0

    def test_constants(self):
        assert sw.MEMBER_UPLOAD_MAX_BYTES == 26214400
        assert sw.MEMBER_DOWNLOAD_MAX_BYTES == 50 * 1024 * 1024
        assert sw.MEMBER_HTTP_RATE_PER_MIN == 120
        assert sw.SPEAKER_ACK_HEADER == "X-FD-Speaker-Ack"
        assert sw.RELAYED_STATUSES == {400, 403, 404, 409, 413}
        assert ".zip" not in sw.MEMBER_UPLOAD_EXTENSIONS
        assert ".html" not in sw.MEMBER_UPLOAD_EXTENSIONS
        assert ".svg" not in sw.MEMBER_UPLOAD_EXTENSIONS
        assert sw.OWNER_COMMONS_BELL_BODY == COMMONS_TEXT
        assert "’" in sw.OWNER_COMMONS_BELL_BODY
        assert sw.OWNER_COMMONS_BELL_BODY.count("’") == 1


# ── The refusal ladder ────────────────────────────────────────────────────


class TestRefusals:
    async def test_ladder_on_the_list_route(self, wdeck, agent, monkeypatch):
        async with _client() as c:
            cases = [
                (await _hit(c, "list", ref="nope"), 400, sw.BAD_REF_DETAIL),
                (await _hit(c, "list", ref="process:helper:aaaaaaaaaaaaaaaa"), 404,
                 sw.NOT_FOUND_DETAIL),
                (await _hit(c, "list", uid=OWNER), 400, sw.OWNER_DETAIL),
                (await _hit(c, "list", uid=ADMIN), 404, sw.NOT_SHARED_DETAIL),
                (await _hit(c, "list", uid=OTHER), 404, sw.NOT_SHARED_DETAIL),
                (await _hit(c, "list", ref=BOX_REF), 400, sw.DOCKER_DETAIL),
                (await _hit(c, "list", ref=SLEEPY_REF), 409, sw.STOPPED_DETAIL),
            ]
            monkeypatch.setattr(sharing, "SHARING_ENABLED", False)
            cases.append((await _hit(c, "list"), 400, sw.SHARING_OFF_DETAIL))
        for r, status, detail in cases:
            assert (r.status_code, _detail(r)) == (status, detail), r.text
        assert agent.requests == []

    async def test_unshareable_agent_is_not_found(self, wdeck, agent):
        await wdeck.db.create_share("agent", "process:orphan:2222222222222222", OWNER, MEMBER,
                                    "view")
        await wdeck.db.create_share("agent", "process:basna-1a2b3c4d-worker:1111111111111111",
                                    OWNER, MEMBER, "view")
        async with _client() as c:
            for ref in ("process:orphan:2222222222222222",
                        "process:basna-1a2b3c4d-worker:1111111111111111"):
                r = await _hit(c, "list", ref=ref)
                assert (r.status_code, _detail(r)) == (404, sw.NOT_FOUND_DETAIL)
        assert agent.requests == []

    @pytest.mark.parametrize("kind", ROUTES)
    async def test_every_route_refuses(self, wdeck, agent, monkeypatch, kind):
        async with _client() as c:
            r1 = await _hit(c, kind, uid=OTHER)
            r2 = await _hit(c, kind, ref=BOX_REF)
            r3 = await _hit(c, kind, uid=OWNER)
            r4 = await _hit(c, kind, uid=ADMIN)
            r5 = await c.get("/fd/shared-agents/files", params={"ref": REF})   # no JWT
            monkeypatch.setattr(sharing, "SHARING_ENABLED", False)
            r6 = await _hit(c, kind)
        assert (r1.status_code, _detail(r1)) == (404, sw.NOT_SHARED_DETAIL)
        assert (r2.status_code, _detail(r2)) == (400, sw.DOCKER_DETAIL)
        assert (r3.status_code, _detail(r3)) == (400, sw.OWNER_DETAIL)
        assert (r4.status_code, _detail(r4)) == (404, sw.NOT_SHARED_DETAIL)
        assert r5.status_code == 401
        assert (r6.status_code, _detail(r6)) == (400, sw.SHARING_OFF_DETAIL)
        assert agent.requests == []

    async def test_rate_limit_121st_request(self, wdeck, agent):
        async with _client() as c:
            for _ in range(3):
                assert (await _hit(c, "list")).status_code == 200
            for _ in range(sw.MEMBER_HTTP_RATE_PER_MIN - 3):
                sw.check_rate(REF, MEMBER)          # requests 4..120 within the minute
            r = await _hit(c, "tables")
            assert (r.status_code, _detail(r)) == (429, sw.RATE_DETAIL)
            assert len(agent.requests) == 3
            agent.expect_sub = M2
            assert (await _hit(c, "list", uid=M2)).status_code == 200   # per member


# ── Target, paths, assertion ──────────────────────────────────────────────


class TestTargetAndPaths:
    async def test_browser_cannot_steer_and_only_member_paths(self, wdeck, agent):
        # (A browser's own fd_member marker is refused by the grant guard before
        # any route runs; FD never adds one — checked below.)
        evil = {"host": "evil.example", "port": "1", "token": "x"}
        async with _client() as c:
            for kind in ROUTES:
                r = await _hit(c, kind, extra=evil)
                assert r.status_code == 200, (kind, r.text)
        assert len(agent.requests) == len(ROUTES)
        expected = {
            ("GET", "/api/speaker/files"), ("GET", "/api/speaker/files/raw"),
            ("POST", "/api/speaker/files/upload"), ("POST", "/api/speaker/files/delete"),
            ("GET", "/api/speaker/datastore/tables"),
            ("GET", "/api/speaker/datastore/tables/people/rows"),
            ("GET", "/api/speaker/datastore/tables/people/export")}
        assert {(q["method"], q["path"]) for q in agent.requests} == expected
        for q in agent.requests:
            assert q["path"].startswith("/api/speaker/")
            assert not q["path"].startswith(("/api/files", "/api/datastore"))
            assert q["params"]["token"] == HELPER_TOK      # the verifier checked host/port too
            assert not {"host", "port", "fd_member"} & set(q["params"])
            assert "x-fd-speaker-grant" not in {k.lower() for k in q["headers"]}
            assert q["payload"]["p"] == q["path"] and q["payload"]["m"] == q["method"]
            assert q["payload"]["name"] == "Mia Member"
            assert q["payload"]["owner_name"] == "Olga Owner"
        assert len({q["payload"]["nonce"] for q in agent.requests}) == len(ROUTES)
        assert len({q["header"] for q in agent.requests}) == len(ROUTES)   # fresh per request

    async def test_call_agent_refuses_non_member_paths(self, wdeck, agent):
        rec = sharing.resolve_agent_record(REF)
        user = await wdeck.db.get_user_by_id(MEMBER)
        for path in ("/api/files", "/api/datastore/tables", "/api/speaker/../files",
                     "/api/speaker/files?x=1", "/ws", "/api/speaker/"):
            with pytest.raises(AssertionError):
                await sw.call_agent(wdeck.db, user, rec, "GET", path)
        with pytest.raises(AssertionError):
            await sw.call_agent(wdeck.db, user, rec, "DELETE", "/api/speaker/files")
        assert agent.requests == []


# ── The ack gate and relayed errors ───────────────────────────────────────


class TestAckGate:
    @pytest.mark.parametrize("status", [404, 405])
    async def test_old_agent_is_outdated(self, wdeck, agent, status):
        agent.ack = "none"
        agent.reply = lambda req: httpx.Response(status, json={"error": "OWNER-SECRET"})
        async with _client() as c:
            r = await _hit(c, "list")
        assert (r.status_code, _detail(r)) == (426, sw.OUTDATED_DETAIL)
        assert "OWNER-SECRET" not in r.text

    async def test_unacked_success_is_never_relayed(self, wdeck, agent):
        agent.ack = "none"
        agent.reply = lambda req: httpx.Response(
            200, json={"files": [_afile("OWNER-SECRET.md", _creator("owner"))]})
        async with _client() as c:
            for kind in ROUTES:
                r = await _hit(c, kind)
                assert (r.status_code, _detail(r)) == (502, sw.AGENT_ERROR_DETAIL), kind
                assert "OWNER-SECRET" not in r.text

    async def test_wrong_ack(self, wdeck, agent):
        agent.ack = "wrong"
        async with _client() as c:
            r = await _hit(c, "list")
            r2 = await _hit(c, "rows")
        assert (r.status_code, _detail(r)) == (502, sw.AGENT_ERROR_DETAIL)
        assert r2.status_code == 502 and "people" not in r2.text

    async def test_unacked_401_is_502(self, wdeck, agent):
        agent.ack = "none"
        agent.reply = lambda req: httpx.Response(401, json={"error": "speaker assertion refused"})
        async with _client() as c:
            r = await _hit(c, "list")
        assert (r.status_code, _detail(r)) == (502, sw.AGENT_ERROR_DETAIL)

    async def test_transport_error(self, wdeck, agent):
        def boom(req):
            raise httpx.ConnectError("refused", request=req)

        agent.reply = boom
        async with _client() as c:
            r = await _hit(c, "list")
            r2 = await _hit(c, "upload")
        assert (r.status_code, _detail(r)) == (502, sw.AGENT_ERROR_DETAIL)
        assert (r2.status_code, _detail(r2)) == (502, sw.AGENT_ERROR_DETAIL)

    @pytest.mark.parametrize("status,error", [
        (400, "Bad id"),
        (403, "Only the person who created that file, or the agent's owner, can delete it."),
        (404, "No such file"),
        (409, "Try again in a second"),
        (413, "You have used your 200 MB"),
    ])
    async def test_relayed_with_ack(self, wdeck, agent, status, error):
        agent.reply = lambda req: httpx.Response(status, json={"error": error})
        async with _client() as c:
            r = await _hit(c, "delete")
        assert (r.status_code, _detail(r)) == (status, error)

    async def test_relayed_detail_is_one_line_and_capped(self, wdeck, agent):
        agent.reply = lambda req: httpx.Response(403, json={"error": "a\n  b\t" + "c" * 400})
        async with _client() as c:
            r = await _hit(c, "list")
        assert r.status_code == 403 and _detail(r) == ("a b " + "c" * 400)[:300]

    @pytest.mark.parametrize("reply", [
        lambda req: httpx.Response(403, json={"error": 5}),
        lambda req: httpx.Response(404, text="<html>nope</html>"),
        lambda req: httpx.Response(400, json=["x"]),
    ])
    async def test_relayed_without_text_is_generic(self, wdeck, agent, reply):
        agent.reply = reply
        async with _client() as c:
            r = await _hit(c, "list")
        assert r.status_code in (400, 403, 404) and _detail(r) == sw.AGENT_ERROR_DETAIL

    @pytest.mark.parametrize("status", [500, 503, 401, 302, 201])
    async def test_other_statuses(self, wdeck, agent, status):
        agent.reply = lambda req: httpx.Response(
            status, json={"error": "internal error"}, headers={"Location": "http://evil/"})
        async with _client() as c:
            r = await _hit(c, "delete")
        if status == 201:
            assert r.status_code == 200          # any 2xx is a success
        else:
            assert (r.status_code, _detail(r)) == (502, sw.AGENT_ERROR_DETAIL)
        assert len(agent.requests) == 1           # a redirect is never followed

    @pytest.mark.parametrize("kind", ["list", "tables", "rows", "upload"])
    async def test_malformed_success_is_502(self, wdeck, agent, kind):
        agent.reply = lambda req: httpx.Response(200, text="not json")
        async with _client() as c:
            r = await _hit(c, kind)
        assert (r.status_code, _detail(r)) == (502, sw.AGENT_ERROR_DETAIL)


# ── Files ─────────────────────────────────────────────────────────────────


class TestFiles:
    async def test_list_mapping(self, wdeck, agent):
        async with _client() as c:
            r = await _hit(c, "list")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["truncated"] is True
        assert body["upload"] == UPLOAD_BLOCK
        files = {f["id"]: f for f in body["files"]}
        assert list(files) == ["downloads/olga-sess/report.pdf", "downloads/mia-sess/notes.md",
                               "downloads/max-sess/data.csv", "downloads/zed-sess/z.txt",
                               "downloads/blank-sess/b.txt", "media/shot.png"]
        creators = {fid: (f["created_by"], f["can_delete"]) for fid, f in files.items()}
        assert creators == {
            "downloads/olga-sess/report.pdf": ({"kind": "owner", "name": "Olga Owner"}, False),
            "downloads/mia-sess/notes.md": ({"kind": "me", "name": ""}, True),
            "downloads/max-sess/data.csv": ({"kind": "member", "name": "Max Member"}, False),
            "downloads/zed-sess/z.txt": ({"kind": "member", "name": "Zed (former member)"}, False),
            "downloads/blank-sess/b.txt": ({"kind": "member", "name": BLANK}, False),
            "media/shot.png": ({"kind": "owner", "name": "Olga Owner"}, False),
        }
        assert files["downloads/mia-sess/notes.md"] == {
            "id": "downloads/mia-sess/notes.md", "path": "saved/downloads/mia-sess/notes.md",
            "filename": "notes.md", "extension": ".md", "size": 12, "modified": 1760000000.5,
            "mime_type": "text/plain", "is_text": True, "created_by": {"kind": "me", "name": ""},
            "can_delete": True}
        for f in body["files"]:
            assert f["path"] == "saved/" + f["id"]
        for forbidden in ("user_id", "@", "physical", "/Users/olga", "u-member", GONE, "old\""):
            assert forbidden not in r.text, forbidden

    async def test_view_headers_and_body(self, wdeck, agent):
        async with _client() as c:
            html = await c.get("/fd/shared-agents/files/view",
                               params={"ref": REF, "id": "downloads/m/page.html"},
                               headers=_hdr(MEMBER))
            png = await c.get("/fd/shared-agents/files/view",
                              params={"ref": REF, "id": "media/shot.png"}, headers=_hdr(MEMBER))
            pdf = await c.get("/fd/shared-agents/files/view",
                              params={"ref": REF, "id": "downloads/olga-sess/report.pdf"},
                              headers=_hdr(MEMBER))
        assert html.status_code == 200
        assert html.headers["content-type"] == "text/plain; charset=utf-8"   # not the agent's
        assert html.headers["content-security-policy"] == "sandbox"
        assert html.headers["x-content-type-options"] == "nosniff"
        assert html.headers["cache-control"] == "no-store"
        assert html.headers["content-disposition"] == "inline"
        assert html.content == b"<script>x</script>downloads/m/page.html"
        assert png.headers["content-type"] == "image/png"
        assert png.headers["content-security-policy"] == "sandbox"
        assert pdf.headers["content-type"] == "application/pdf"
        assert "content-security-policy" not in pdf.headers
        assert pdf.headers["x-content-type-options"] == "nosniff"
        assert [q["params"]["id"] for q in agent.requests] == [
            "downloads/m/page.html", "media/shot.png", "downloads/olga-sess/report.pdf"]

    async def test_download_headers(self, wdeck, agent):
        async with _client() as c:
            r = await c.get("/fd/shared-agents/files/download",
                            params={"ref": REF, "id": "downloads/s/ä b.html"},
                            headers=_hdr(MEMBER))
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/octet-stream"
        assert r.headers["content-disposition"] == (
            "attachment; filename*=UTF-8''%C3%A4%20b.html")
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["cache-control"] == "no-store"

    @pytest.mark.parametrize("bad", ["a/../b", "../x", ".env", "/etc/passwd", "", "a\\b"])
    async def test_bad_id_never_reaches_the_agent(self, wdeck, agent, bad):
        async with _client() as c:
            for kind in ("view", "download"):
                r = await c.get(f"/fd/shared-agents/files/{kind}", params={"ref": REF, "id": bad},
                                headers=_hdr(MEMBER))
                assert (r.status_code, _detail(r)) == (400, sw.BAD_ID_DETAIL)
        assert agent.requests == []

    async def test_fd_token_query_auth(self, wdeck, agent):
        tok = _hdr(MEMBER)["Authorization"].split(" ", 1)[1]
        async with _client() as c:
            r = await c.get("/fd/shared-agents/files/view",
                            params={"ref": REF, "id": "media/shot.png", "fd_token": tok})
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert "fd_token" not in agent.requests[0]["params"]

    async def test_download_size_cap(self, wdeck, agent, monkeypatch):
        monkeypatch.setattr(sw, "MEMBER_DOWNLOAD_MAX_BYTES", 10)
        async with _client() as c:
            r = await _hit(c, "view")
        assert (r.status_code, _detail(r)) == (413, sw.DOWNLOAD_TOO_LARGE_DETAIL)


class TestUpload:
    async def _up(self, c, name="a.csv", data=b"a,b\n1,2\n", uid=MEMBER, lane="A", **kw):
        return await c.post("/fd/shared-agents/files/upload", params={"ref": REF, "lane": lane},
                            files={"file": (name, data, "text/csv")}, headers=_hdr(uid), **kw)

    async def test_success_and_usage_row(self, wdeck, agent):
        async with _client() as c:
            r = await self._up(c, name="..\\evil\\a.csv")
        assert r.status_code == 200, r.text
        assert r.json() == {
            "id": "downloads/mia-sess/a-20261006T120000Z.csv",
            "path": "saved/downloads/mia-sess/a-20261006T120000Z.csv",
            "filename": "a-20261006T120000Z.csv", "extension": ".csv", "size": 12,
            "modified": 1760000000.5, "mime_type": "text/plain", "is_text": True,
            "created_by": {"kind": "me", "name": ""}, "can_delete": True}
        assert "physical" not in r.text
        (q,) = agent.requests
        assert q["payload"]["lane"] == "A"
        assert b'filename="a.csv"' in q["body"] and b"a,b\n1,2\n" in q["body"]
        assert b"evil" not in q["body"]
        rows = await wdeck.db.get_usage_logs(user_id=MEMBER, event_type="shared_agent_file_upload")
        assert len(rows) == 1
        assert json.loads(rows[0]["detail"]) == {"agent_ref": REF, "filename": "a.csv", "size": 8}

    async def test_lane(self, wdeck, agent):
        async with _client() as c:
            r = await self._up(c, lane="b")
            bad = await self._up(c, lane="D")
            bad2 = await self._up(c, lane="")
        assert r.status_code == 200 and agent.requests[0]["payload"]["lane"] == "B"
        assert (bad.status_code, _detail(bad)) == (400, sw.BAD_LANE_DETAIL)
        assert (bad2.status_code, _detail(bad2)) == (400, sw.BAD_LANE_DETAIL)
        assert len(agent.requests) == 1

    @pytest.mark.parametrize("name", ["a.zip", "page.html", "x.svg", "x.HTM", "noext", ".csv",
                                      "a.csv.exe", "dir/", "x" * 197 + ".csv"])
    async def test_type_refused_before_the_agent(self, wdeck, agent, name):
        async with _client() as c:
            r = await self._up(c, name=name)
        assert (r.status_code, _detail(r)) == (400, sw.BAD_TYPE_DETAIL)
        assert agent.requests == []

    async def test_nameless_file_is_no_file(self, wdeck, agent):
        async with _client() as c:
            r = await self._up(c, name="")       # Starlette reads it as a plain field
        assert (r.status_code, _detail(r)) == (400, sw.EMPTY_DETAIL)
        assert agent.requests == []

    async def test_too_large_and_empty(self, wdeck, agent):
        async with _client() as c:
            big = await self._up(c, data=b"x" * (sw.MEMBER_UPLOAD_MAX_BYTES + 1))
            empty = await self._up(c, data=b"")
            no_file = await c.post("/fd/shared-agents/files/upload",
                                   params={"ref": REF, "lane": "A"},
                                   files={"other": ("a.csv", b"1", "text/csv")},
                                   headers=_hdr(MEMBER))
            text_field = await c.post("/fd/shared-agents/files/upload",
                                      params={"ref": REF, "lane": "A"},
                                      data={"file": "just text"}, headers=_hdr(MEMBER))
            two = await c.post("/fd/shared-agents/files/upload",
                               params={"ref": REF, "lane": "A"},
                               files=[("file", ("a.csv", b"1", "text/csv")),
                                      ("file", ("b.csv", b"2", "text/csv"))],
                               headers=_hdr(MEMBER))
        assert (big.status_code, _detail(big)) == (413, sw.TOO_LARGE_DETAIL)
        assert (empty.status_code, _detail(empty)) == (400, sw.EMPTY_DETAIL)
        assert (no_file.status_code, _detail(no_file)) == (400, sw.EMPTY_DETAIL)
        assert (text_field.status_code, _detail(text_field)) == (400, sw.EMPTY_DETAIL)
        assert two.status_code == 400
        assert agent.requests == []

    async def test_membership_rechecked_fresh(self, wdeck, agent):
        async with _client() as c:
            assert (await self._up(c)).status_code == 200      # warms the member cache
            assert await wdeck.db.delete_share("agent", REF, OWNER, MEMBER)  # no invalidation
            assert (await _hit(c, "list")).status_code == 200  # reads ride the warm cache
            r = await self._up(c)
        assert (r.status_code, _detail(r)) == (404, sw.NOT_SHARED_DETAIL)
        assert [q["path"] for q in agent.requests] == ["/api/speaker/files/upload",
                                                      "/api/speaker/files"]

    async def test_bad_agent_answer_is_502(self, wdeck, agent):
        agent.reply = lambda req: httpx.Response(200, json=_afile("../x.csv", None))
        async with _client() as c:
            r = await self._up(c)
        assert (r.status_code, _detail(r)) == (502, sw.AGENT_ERROR_DETAIL)
        assert await wdeck.db.get_usage_logs(event_type="shared_agent_file_upload") == []


class TestUploadBodyGuard:
    """(r2) The size and the membership are checked before the body is read."""

    @pytest.fixture
    def no_form(self, monkeypatch):
        calls: list = []

        def form(self, *a, **kw):
            calls.append(1)
            raise AssertionError("the upload body was parsed")

        monkeypatch.setattr(Request, "form", form)
        return calls

    async def test_declared_size_over_the_cap(self, wdeck, agent, no_form):
        async with _client() as c:
            r = await c.post("/fd/shared-agents/files/upload", params={"ref": REF},
                             content=b"x", headers={**_hdr(OTHER), "Content-Length": "30000000",
                                                    "Content-Type": "multipart/form-data; "
                                                                    "boundary=b"})
        assert (r.status_code, _detail(r)) == (413, sw.TOO_LARGE_DETAIL)
        assert no_form == [] and agent.requests == []

    async def test_cap_includes_only_the_slack(self, wdeck, agent, no_form):
        cap = sw.MEMBER_UPLOAD_MAX_BYTES + sw.UPLOAD_BODY_SLACK
        async with _client() as c:
            over = await c.post("/fd/shared-agents/files/upload", params={"ref": REF},
                                content=b"x", headers={**_hdr(MEMBER),
                                                       "Content-Length": str(cap + 1)})
            at = await c.post("/fd/shared-agents/files/upload", params={"ref": REF},
                              content=b"x", headers={**_hdr(OTHER), "Content-Length": str(cap)})
        assert (over.status_code, _detail(over)) == (413, sw.TOO_LARGE_DETAIL)
        assert (at.status_code, _detail(at)) == (404, sw.NOT_SHARED_DETAIL)   # then membership
        assert no_form == []

    async def test_missing_length(self, wdeck, agent, no_form):
        async def chunks():
            yield b"--b\r\n"

        async with _client() as c:
            r = await c.post("/fd/shared-agents/files/upload", params={"ref": REF},
                             content=chunks(), headers=_hdr(MEMBER))
            bad = await c.post("/fd/shared-agents/files/upload", params={"ref": REF},
                               content=b"x", headers={**_hdr(MEMBER), "Content-Length": "abc"})
        assert (r.status_code, _detail(r)) == (411, sw.LENGTH_DETAIL)
        assert (bad.status_code, _detail(bad)) == (411, sw.LENGTH_DETAIL)
        assert no_form == [] and agent.requests == []

    async def test_non_member_small_upload(self, wdeck, agent, no_form):
        async with _client() as c:
            r = await c.post("/fd/shared-agents/files/upload", params={"ref": REF, "lane": "A"},
                             files={"file": ("a.csv", b"1,2", "text/csv")}, headers=_hdr(OTHER))
            owner = await c.post("/fd/shared-agents/files/upload",
                                 params={"ref": REF, "lane": "A"},
                                 files={"file": ("a.csv", b"1,2", "text/csv")},
                                 headers=_hdr(OWNER))
        assert (r.status_code, _detail(r)) == (404, sw.NOT_SHARED_DETAIL)
        assert (owner.status_code, _detail(owner)) == (400, sw.OWNER_DETAIL)
        assert no_form == [] and agent.requests == []


class TestDelete:
    async def test_success_and_usage_row(self, wdeck, agent):
        async with _client() as c:
            r = await _hit(c, "delete")
        assert r.status_code == 200 and r.json() == {"ok": True}
        (q,) = agent.requests
        assert q["method"] == "POST" and json.loads(q["body"]) == {"id": MIA_FILE}
        rows = await wdeck.db.get_usage_logs(user_id=MEMBER, event_type="shared_agent_file_delete")
        assert [json.loads(row["detail"]) for row in rows] == [{"agent_ref": REF, "id": MIA_FILE}]

    @pytest.mark.parametrize("body", [{"id": "../x"}, {"id": ".env"}, {"id": ""}, {}])
    async def test_bad_id(self, wdeck, agent, body):
        async with _client() as c:
            r = await c.post("/fd/shared-agents/files/delete", params={"ref": REF}, json=body,
                             headers=_hdr(MEMBER))
        assert (r.status_code, _detail(r)) == (400, sw.BAD_ID_DETAIL)
        assert agent.requests == []

    async def test_membership_rechecked_fresh(self, wdeck, agent):
        async with _client() as c:
            assert (await _hit(c, "list")).status_code == 200   # warm cache
            await wdeck.db.delete_share("agent", REF, OWNER, MEMBER)
            r = await _hit(c, "delete")
        assert (r.status_code, _detail(r)) == (404, sw.NOT_SHARED_DETAIL)
        assert [q["path"] for q in agent.requests] == ["/api/speaker/files"]
        assert await wdeck.db.get_usage_logs(event_type="shared_agent_file_delete") == []

    async def test_not_theirs_is_relayed_and_not_logged(self, wdeck, agent):
        text = "Only the person who created that file, or the agent's owner, can delete it."
        agent.reply = lambda req: httpx.Response(403, json={"error": text})
        async with _client() as c:
            r = await _hit(c, "delete")
        assert (r.status_code, _detail(r)) == (403, text)
        assert await wdeck.db.get_usage_logs(event_type="shared_agent_file_delete") == []


# ── Datastore ─────────────────────────────────────────────────────────────


class TestDatastore:
    async def test_tables(self, wdeck, agent):
        async with _client() as c:
            r = await _hit(c, "tables")
        assert r.status_code == 200
        assert r.json() == [
            {"name": "people", "columns": [{"name": "name", "type": "text", "position": 0}],
             "row_count": 3, "created_at": "2026-10-01T00:00:00Z",
             "updated_at": "2026-10-02T00:00:00Z",
             "created_by": {"kind": "member", "name": "Max Member"}},
            {"name": "mine", "columns": [], "row_count": 0, "created_at": "", "updated_at": "",
             "created_by": {"kind": "me", "name": ""}},
            {"name": "legacy", "columns": [], "row_count": 0, "created_at": "", "updated_at": "",
             "created_by": {"kind": "owner", "name": "Olga Owner"}},
        ]
        assert "u-member" not in r.text and "@" not in r.text

    async def test_rows(self, wdeck, agent):
        async with _client() as c:
            r = await c.get("/fd/shared-agents/datastore/tables/people/rows",
                            params={"ref": REF, "limit": 10000, "offset": -5, "order_by": "name",
                                    "order_dir": "DESC"}, headers=_hdr(MEMBER))
        assert r.status_code == 200, r.text
        assert r.json() == {
            "columns": ["_id", "name"],
            "rows": [
                {"_id": 1, "name": "a", "_creator": {"kind": "member", "name": "Max Member"}},
                {"_id": 2, "name": "b", "_creator": {"kind": "owner", "name": "Olga Owner"}},
                {"_id": 3, "name": "c", "_creator": {"kind": "me", "name": ""}},
            ],
            "total": 3, "offset": 0, "limit": 500}
        assert "u-member" not in r.text and "_created_by" not in r.text
        (q,) = agent.requests
        assert q["path"] == "/api/speaker/datastore/tables/people/rows" == q["payload"]["p"]
        assert {k: q["params"][k] for k in ("limit", "offset", "order_by", "order_dir")} == {
            "limit": "500", "offset": "0", "order_by": "name", "order_dir": "desc"}

    async def test_rows_defaults_and_clamps(self, wdeck, agent):
        async with _client() as c:
            await _hit(c, "rows")
            await c.get("/fd/shared-agents/datastore/tables/people/rows",
                        params={"ref": REF, "limit": 0, "offset": 99_999_999},
                        headers=_hdr(MEMBER))
        p1, p2 = (q["params"] for q in agent.requests)
        assert (p1["limit"], p1["offset"], p1["order_by"], p1["order_dir"]) == (
            "100", "0", "_id", "asc")
        assert (p2["limit"], p2["offset"]) == ("1", "10000000")

    @pytest.mark.parametrize("params", [
        {"order_by": "name;drop"}, {"order_by": "Name"}, {"order_by": "a b"},
        {"order_dir": "sideways"}, {"order_dir": "asc;"},
    ])
    async def test_rows_bad_params(self, wdeck, agent, params):
        async with _client() as c:
            r = await c.get("/fd/shared-agents/datastore/tables/people/rows",
                            params={"ref": REF, **params}, headers=_hdr(MEMBER))
        assert (r.status_code, _detail(r)) == (400, sw.BAD_TABLE_DETAIL)
        assert agent.requests == []

    @pytest.mark.parametrize("name", ["Bad%20Name", "People", "a-b", "a%2Fb", "x" * 129])
    async def test_bad_table_name(self, wdeck, agent, name):
        async with _client() as c:
            r = await c.get(f"/fd/shared-agents/datastore/tables/{name}/rows",
                            params={"ref": REF}, headers=_hdr(MEMBER))
            r2 = await c.get(f"/fd/shared-agents/datastore/tables/{name}/export",
                             params={"ref": REF}, headers=_hdr(MEMBER))
        for resp in (r, r2):
            assert resp.status_code in (400, 404)
            if resp.status_code == 400:
                assert _detail(resp) == sw.BAD_TABLE_DETAIL
        assert agent.requests == []

    async def test_export(self, wdeck, agent):
        async with _client() as c:
            r = await _hit(c, "export")
            bad = await c.get("/fd/shared-agents/datastore/tables/people/export",
                              params={"ref": REF, "format": "sql"}, headers=_hdr(MEMBER))
        assert r.status_code == 200 and r.content == EXPORT_CSV
        assert r.headers["content-type"] == "text/csv; charset=utf-8"   # not the agent's
        assert r.headers["content-disposition"] == 'attachment; filename="people.csv"'
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["cache-control"] == "no-store"
        assert (bad.status_code, _detail(bad)) == (400, sw.BAD_FORMAT_DETAIL)
        (q,) = agent.requests
        assert q["params"]["format"] == "csv"
        assert q["path"] == "/api/speaker/datastore/tables/people/export" == q["payload"]["p"]


# ── GET /fd/shared-agents ─────────────────────────────────────────────────


class TestSharedAgentsFlag:
    async def test_member_workspace_and_capabilities(self, wdeck, monkeypatch):
        async with _client() as c:
            body = (await c.get("/fd/shared-agents", headers=_hdr(MEMBER))).json()
            monkeypatch.setattr(sharing, "SHARING_ENABLED", False)
            off = (await c.get("/fd/shared-agents", headers=_hdr(MEMBER))).json()
        assert body["member_workspace"] is True and body["context_packs"] is True
        rows = {a["agent_ref"]: a for a in body["agents"]}
        assert rows[REF]["capabilities"] == {"google": True, "deep_memory": True, "files": True,
                                             "datastore": True}
        assert rows[BOX_REF]["capabilities"]["datastore"] is False
        assert rows[SLEEPY_REF]["capabilities"]["datastore"] is True
        assert off == {"enabled": False, "host_warning": "", "agents": []}
        assert "member_workspace" not in off


# ── Owner proxies, owner file view, /deck/view ────────────────────────────


@pytest.fixture
def owner_agent(monkeypatch):
    """The owner's agent behind FD's port-addressed owner proxies."""
    seen: list[httpx.Request] = []
    replies: dict[str, httpx.Response] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        resp = replies.get(request.url.path)
        return resp if resp is not None else httpx.Response(404, text="nope")

    real = httpx.AsyncClient

    def factory(*args, **kw):
        kw.setdefault("transport", httpx.MockTransport(handler))
        return real(*args, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return replies, seen


class TestOwnerSide:
    @pytest.fixture(autouse=True)
    def _db_on_app(self, wdeck, monkeypatch):
        monkeypatch.setattr(server.app.state, "fd_db", wdeck.db, raising=False)

    async def test_agent_files_names(self, wdeck, owner_agent):
        replies, seen = owner_agent
        replies["/api/files"] = httpx.Response(200, json=[
            {"logical": "saved/downloads/max-sess/data.csv", "filename": "data.csv",
             "created_by": {"kind": "member", "user_id": M2, "name": "old"}},
            {"logical": "saved/downloads/zed/z.txt", "filename": "z.txt",
             "created_by": {"kind": "member", "user_id": GONE, "name": "Zed"}},
            {"logical": "saved/x.md", "filename": "x.md",
             "created_by": {"kind": "owner", "user_id": "", "name": ""}},
            {"logical": "output/y.md", "filename": "y.md", "created_by": None},
            {"logical": "output/z.md", "filename": "z.md"},
        ])
        async with _client() as c:
            r = await c.get(f"/fd/agent-files/localhost/{HELPER_PORT}", headers=_hdr(OWNER))
        assert r.status_code == 200, r.text
        assert r.json() == [
            {"logical": "saved/downloads/max-sess/data.csv", "filename": "data.csv",
             "created_by": {"kind": "member", "user_id": M2, "name": "Max Member"}},
            {"logical": "saved/downloads/zed/z.txt", "filename": "z.txt",
             "created_by": {"kind": "member", "user_id": GONE, "name": "Zed (former member)"}},
            {"logical": "saved/x.md", "filename": "x.md",
             "created_by": {"kind": "owner", "user_id": "", "name": ""}},
            {"logical": "output/y.md", "filename": "y.md", "created_by": None},
            {"logical": "output/z.md", "filename": "z.md"},
        ]
        assert seen[0].url.path == "/api/files"

    async def test_agent_datastore_names(self, wdeck, owner_agent):
        replies, _seen = owner_agent
        replies["/api/datastore/tables"] = httpx.Response(200, json=[
            {"name": "people", "created_by": {"kind": "member", "user_id": M2, "name": "old"}},
            {"name": "legacy", "created_by": {"kind": "owner", "user_id": "", "name": ""}},
            {"name": "plain"}])
        replies["/api/datastore/tables/people/rows"] = httpx.Response(200, json={
            "columns": ["_id", "name"], "total": 2, "offset": 0, "limit": 100,
            "rows": [{"_id": 1, "name": "a",
                      "_creator": {"kind": "member", "user_id": M2, "name": "old"}},
                     {"_id": 2, "name": "b", "_creator": {"kind": "owner", "user_id": "",
                                                          "name": ""}},
                     {"_id": 3, "name": "c"}]})
        async with _client() as c:
            t = await c.get(f"/fd/agent-datastore/localhost/{HELPER_PORT}/tables",
                            headers=_hdr(OWNER))
            rows = await c.get(f"/fd/agent-datastore/localhost/{HELPER_PORT}/tables/people/rows",
                               headers=_hdr(OWNER))
        assert t.json() == [
            {"name": "people", "created_by": {"kind": "member", "user_id": M2,
                                              "name": "Max Member"}},
            {"name": "legacy", "created_by": {"kind": "owner", "user_id": "", "name": ""}},
            {"name": "plain"}]
        assert rows.json() == {
            "columns": ["_id", "name"], "total": 2, "offset": 0, "limit": 100,
            "rows": [{"_id": 1, "name": "a",
                      "_creator": {"kind": "member", "user_id": M2, "name": "Max Member"}},
                     {"_id": 2, "name": "b", "_creator": {"kind": "owner", "user_id": "",
                                                          "name": ""}},
                     {"_id": 3, "name": "c"}]}

    async def test_owner_proxies_unchanged_for_old_agents(self, wdeck, owner_agent):
        replies, _seen = owner_agent
        replies["/api/datastore/tables"] = httpx.Response(200, json={"weird": True})
        async with _client() as c:
            t = await c.get(f"/fd/agent-datastore/localhost/{HELPER_PORT}/tables",
                            headers=_hdr(OWNER))
        assert t.json() == {"weird": True}

    async def test_members_cannot_use_owner_proxies(self, wdeck, owner_agent):
        _replies, seen = owner_agent
        async with _client() as c:
            r = await c.get(f"/fd/agent-files/localhost/{HELPER_PORT}", headers=_hdr(MEMBER))
        assert r.status_code == 403 and seen == []

    @pytest.mark.parametrize("ctype,csp", [
        ("text/html; charset=utf-8", True), ("TEXT/HTML", True), ("image/svg+xml", True),
        ("application/xhtml+xml", True), ("text/xml", True), ("application/xml", True),
        # any *+xml type renders as an XML document (XHTML-namespaced script runs)
        ("application/rss+xml", True), ("application/xslt+xml", True), ("text/xsl", True),
        ("image/png", False), ("text/plain", False), ("application/pdf", False),
    ])
    async def test_owner_file_view_headers(self, wdeck, owner_agent, ctype, csp):
        replies, _seen = owner_agent
        replies["/api/files/view"] = httpx.Response(200, content=b"<b>x</b>",
                                                   headers={"Content-Type": ctype})
        async with _client() as c:
            r = await c.get(f"/fd/agent-file-view/localhost/{HELPER_PORT}",
                            params={"path": "/x/y"}, headers=_hdr(OWNER))
        assert r.status_code == 200 and r.content == b"<b>x</b>"
        assert r.headers["content-type"] == ctype
        assert r.headers["content-disposition"] == "inline"
        assert r.headers["x-content-type-options"] == "nosniff"
        if csp:
            assert r.headers["content-security-policy"] == (
                "sandbox allow-scripts allow-popups allow-forms allow-modals allow-downloads")
        else:
            assert "content-security-policy" not in r.headers

    async def test_decorate_never_raises(self, wdeck, monkeypatch):
        class Broken:
            async def get_user_by_id(self, uid):
                raise RuntimeError("db down")

        payload = [{"created_by": {"kind": "member", "user_id": "u-x", "name": "x"}},
                   {"created_by": {"kind": "member", "user_id": M2, "name": "old"}}]
        rows = {"rows": [{"_creator": {"kind": "member", "user_id": M2, "name": "old"}}]}
        assert await sw.decorate_owner_payload(Broken(), payload) is payload
        assert await sw.decorate_owner_payload(Broken(), rows) is rows
        assert await sw.decorate_owner_payload(None, payload) is payload

        calls: list = []

        async def boom_on_second(db, uid):   # fails half-way through the list
            calls.append(uid)
            if len(calls) > 1:
                raise RuntimeError("x")
            return "Someone"

        monkeypatch.setattr(sw.tenant_profile, "owner_name", boom_on_second)
        await _add_user(wdeck.db, "u-x", "X")
        assert await sw.decorate_owner_payload(wdeck.db, payload) is payload
        assert payload == [{"created_by": {"kind": "member", "user_id": "u-x", "name": "x"}},
                           {"created_by": {"kind": "member", "user_id": M2, "name": "old"}}]
        for weird in ("text", 5, {"rows": "x"}, [1, None, {"created_by": "member"}]):
            assert await sw.decorate_owner_payload(wdeck.db, weird) == weird


class TestDeckView:
    @pytest.fixture
    def deck_file(self, monkeypatch):
        holder: dict = {}

        async def fake_get(ch, path, params=None):
            assert path == "/api/files/content"
            return FastAPIResponse(content=json.dumps(holder["payload"]).encode(),
                                   status_code=200, media_type="application/json")

        monkeypatch.setattr(glasses_bridge, "_agent_get", fake_get)
        yield holder
        glasses_bridge._channels.pop("pr-c-deck", None)

    @pytest.mark.parametrize("created_by,refused", [
        ({"kind": "member", "user_id": M2, "name": "Max"}, True),
        ({"kind": "owner", "user_id": "", "name": ""}, False),
        (None, False),
        ("member", False),
    ])
    async def test_member_files_refused(self, deck, deck_file, created_by, refused):
        payload = {"content": "<html><body><h1>DECK-BODY</h1></body></html>"}
        if created_by is not None:
            payload["created_by"] = created_by
        deck_file["payload"] = payload
        async with _client() as c:
            r = await c.get("/deck/view", params={"c": "pr-c-deck", "path": "/w/saved/d.html"})
        if refused:
            assert (r.status_code, _detail(r)) == (403, glasses_bridge.DECK_MEMBER_FILE_DETAIL)
            assert "DECK-BODY" not in r.text
        else:
            assert r.status_code == 200 and "DECK-BODY" in r.text
            assert "__DECK_CHANNEL__" in r.text

    def test_detail_text(self):
        assert glasses_bridge.DECK_MEMBER_FILE_DETAIL == (
            "A file added by a member of this shared agent can't be presented as a deck")


# ── The one-time owner bell ───────────────────────────────────────────────


class TestOwnerBell:
    async def test_once_per_process_agent(self, deck):
        db = deck.db
        await _add_user(db, M2, "Max Member")
        deck.containers.append(FakeContainer(deck.containers, "box", OWNER, BOX_TOK, 24991,
                                             instance=BOX_INST))
        await db.create_share("agent", REF, OWNER, MEMBER, "view")
        await db.create_share("agent", REF, OWNER, M2, "view")           # one bell per agent
        await db.create_share("agent", BOX_REF, OWNER, MEMBER, "view")   # docker: none
        await db.create_share("agent", REF, OTHER, MEMBER, "view")       # not OTHER's agent
        await db.create_share("agent", "garbage", OWNER, MEMBER, "view")
        await db.create_share("agent", "process:gone:9999999999999999", OWNER, MEMBER, "view")

        assert await sw.notify_owners_of_commons_once(db) == 1
        notes = await db.list_notifications(OWNER)
        assert len(notes) == 1
        (n,) = notes
        assert n["type"] == "share"
        assert n["title"] == "Members of “Helper” can now open its saved/ folder and datastore"
        assert n["body"] == COMMONS_TEXT
        assert (n["ref_type"], n["ref_id"]) == ("agent", REF)
        for uid in (OTHER, MEMBER, M2, ADMIN):
            assert await db.list_notifications(uid) == []
        assert await db.get_system_setting(sw.COMMONS_NOTICE_KEY)

        assert await sw.notify_owners_of_commons_once(db) == 0          # once
        assert len(await db.list_notifications(OWNER)) == 1

    async def test_never_raises(self):
        class Broken:
            async def get_system_setting(self, key):
                raise RuntimeError("db down")

        assert await sw.notify_owners_of_commons_once(Broken()) == 0

    def test_lifespan_schedules_it_only_with_sharing_active(self):
        src = inspect.getsource(server.lifespan)
        assert src.count("notify_owners_of_commons_once") == 1
        tree = ast.parse(textwrap.dedent(src))
        guarded = []
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and ast.unparse(node.test) == (
                    "agent_sharing.sharing_active()"):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and ast.unparse(sub.func) == "asyncio.create_task":
                        if "notify_owners_of_commons_once(_fd_db)" in ast.unparse(sub):
                            guarded.append(sub)
        assert len(guarded) == 1
        assert src.index("_fd_db = await _init_fd_db()") < src.index(
            "notify_owners_of_commons_once") < src.index("yield")


# ── Logs ──────────────────────────────────────────────────────────────────


class TestNoSecretsInLogs:
    async def test_list_and_upload(self, wdeck, agent, caplog, capsys):
        jwt = _hdr(MEMBER)["Authorization"].split(" ", 1)[1]
        with caplog.at_level(logging.DEBUG):
            async with _client() as c:
                assert (await _hit(c, "list")).status_code == 200
                assert (await _hit(c, "upload")).status_code == 200
        captured = capsys.readouterr()
        text = caplog.text + captured.out + captured.err
        assert any("/api/speaker/files" in r.getMessage() for r in caplog.records
                   if r.name == "httpx")                     # httpx did log the calls
        headers = [q["header"] for q in agent.requests]
        assert len(headers) == 2
        for secret in (HELPER_TOK, jwt, *headers, *(h.split(".")[2] for h in headers)):
            assert secret not in text

    async def test_structured_log_fields(self, wdeck, agent, monkeypatch):
        calls: list = []

        class Spy:
            def __getattr__(self, name):
                def record(*args, **kwargs):
                    calls.append((name, args, kwargs))
                return record

        monkeypatch.setattr(sw, "log", Spy())
        async with _client() as c:
            assert (await _hit(c, "rows")).status_code == 200
        info = [kw for name, args, kw in calls if args == ("Shared-agent workspace call",)]
        assert info == [{"agent": "helper", "member": MEMBER,
                         "path": "/api/speaker/datastore/tables/people/rows", "status": 200}]
        blob = repr(calls)
        assert HELPER_TOK not in blob and agent.requests[0]["header"] not in blob
        assert "limit=" not in blob
