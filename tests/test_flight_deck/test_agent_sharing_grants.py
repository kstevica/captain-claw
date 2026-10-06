"""A2 — shared-agent members act with their own credentials and data (FD side).

Pinned here (contract part 1b):

* ``speaker_grants``: mint / lookup / end_turn / TTL / conn_dropped / revoke /
  caps, token never stored; the Google opt-in helpers over a real DB;
* ``deep_memory_filter.validate_filter_by``: the accepted grammar and refusals;
* the member-cache generation guard (a True computed across an invalidation is
  returned but never cached);
* ``acting_member`` on every grant-aware route: no owner fallback once either
  marker is present, the agent must be the grant's, membership and the Google
  opt-in are re-checked, owner changes and removals revoke and clear opt-ins;
  Google access token / agent_status / Gmail send / deep memory act as the member;
* ``filter_by`` validation on the dashboard and agent routes;
* the grant guard middleware (HTTP 403 off the grant-aware routes, WS 4403);
* the read-time VFS link-target guard (``safe_link_target``);
* the opt-in route, the settings guard, listings and O1 ``mine``;
* lifecycle: share DELETE / leave, agent removal, watchdog owner change;
* the WS proxy: one grant per member chat turn on a process agent, none for
  docker agents / slash commands / empty messages, client values never
  forwarded, turn_end closes it, a closed socket cuts it to ORPHAN_GRACE_S,
  revocations (incl. a racing one) revoke it.

Same deck as ``test_agent_sharing`` (real FlightDeckDB + process registry in
tmp dirs, Docker and agents faked). Google and Typesense are never contacted.
No test touches ``~/.captain-claw`` or a real FD data dir: the deck lives in
tmp, the agent secret comes from the env, and ``CAPTAIN_CLAW_FD_HOME`` /
``CLAW_VFS_ROOT`` point into tmp.
"""

from __future__ import annotations

import asyncio
import json
import re
import time

import httpx
import pytest
from starlette.websockets import WebSocketDisconnect

import captain_claw.flight_deck.deep_memory_routes as dr
import captain_claw.flight_deck.gmail_send_routes as gs
import captain_claw.flight_deck.google_oauth_routes as gr
from captain_claw import gmail_compose
from captain_claw.flight_deck import agent_secret, server, share_routes
from captain_claw.flight_deck import agent_sharing as sharing
from captain_claw.flight_deck import deep_memory_filter as dmf
from captain_claw.flight_deck import speaker_grants as sg
from captain_claw.flight_deck.archetype_compose import recall_filter
from captain_claw.flight_deck.auth import create_access_token
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.google_oauth import GoogleOAuthTokens
from test_flight_deck import test_agent_sharing as base
from test_flight_deck.test_agent_sharing import (
    HELPER_TOK,
    HTTP_FD,
    MEMBER,
    OTHER,
    OWNER,
    REF,
    FakeAgent,
    FakeContainer,
    _client,
    _expect_close,
    _hdr,
    _open,
    _recv_json,
)

deck = base.deck          # the A1 deck fixtures
ws_deck = base.ws_deck

TOKEN_RE = re.compile(sg.GRANT_TOKEN_RE)
SECOND_TOK = "second-tok"
SECOND_REF = f"process:second:{'6' * 16}"
BOX_INST = "7" * 16
BOX_REF = f"docker:box:{BOX_INST}"
BOX_TOK = "box-tok"
NON_MEMBER = OTHER
OWNER2 = OTHER
SHARED_SECRET = "test-agent-shared-secret"

GOOGLE_ROUTES = [("GET", "/fd/google/access_token"), ("GET", "/fd/google/agent_status"),
                 ("POST", "/fd/google/gmail/send")]
DM_ROUTES = [("POST", "/fd/deep-memory/agent/search"), ("POST", "/fd/deep-memory/agent/index"),
             ("POST", "/fd/deep-memory/agent/delete")]
ALL_ROUTES = GOOGLE_ROUTES + DM_ROUTES
# PR B added the context-pack VFS route (pinned in test_context_packs.py).
assert {p for _, p in ALL_ROUTES} | {"/fd/context-packs/agent/vfs"} == sg.GRANT_AWARE_PATHS
BODIES = {
    "/fd/google/gmail/send": {"to": "Bob <bob@x.co>", "subject": "Hello", "body": "Hi Bob"},
    "/fd/deep-memory/agent/search": {"query": "q"},
    "/fd/deep-memory/agent/index": {"text": "a note", "reference": "r1"},
    "/fd/deep-memory/agent/delete": {"reference": "r1"},
}


@pytest.fixture(autouse=True)
def _fresh_grants(monkeypatch, tmp_path):
    """Every test starts with no grants, no member-cache generations, the agent
    secret from the env (never a file in a real home) and tmp FD/VFS homes."""
    sg._reset_for_tests()
    monkeypatch.setattr(sharing, "_MEMBER_GEN", {})
    monkeypatch.setattr(sharing, "_MEMBER_GEN_REF", {})
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # nothing reaches ~/.captain-claw*
    monkeypatch.setenv("FD_AGENT_SHARED_SECRET", SHARED_SECRET)
    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path / "claw-vfs"))
    for var in ("FD_PUBLIC_URL", "FD_GMAIL_SEND", "FD_ARCHETYPE_GRID"):
        monkeypatch.delenv(var, raising=False)
    agent_secret.reset_cache_for_tests()
    yield
    sg._reset_for_tests()
    agent_secret.reset_cache_for_tests()


def _tokens(access: str) -> GoogleOAuthTokens:
    # Far-future expiry: _refresh_if_needed never makes a network call.
    return GoogleOAuthTokens(access_token=access, refresh_token=f"{access}-refresh",
                             token_type="Bearer", expires_at=time.time() + 3600,
                             scope="openid email https://www.googleapis.com/auth/gmail.compose")


def _add_agents(env) -> None:
    """O's second process agent (web_auth W2) and O's docker agent ``box``."""
    reg = server._load_process_registry()
    reg["second"] = {"slug": "second", "name": "Second", "description": "", "web_port": 24902,
                     "web_auth": SECOND_TOK, "owner": OWNER, "pid": None,
                     "instance_id": "6" * 16}
    server._save_process_registry(reg)
    env.containers.append(FakeContainer(env.containers, "box", OWNER, BOX_TOK, 24991,
                                        instance=BOX_INST))


class _FakeIndex:
    def __init__(self):
        self.calls: list[tuple] = []

    def delete_by_reference(self, reference, owner_id=""):
        self.calls.append(("delete_by_reference", reference, owner_id))
        return 1

    def delete_by_filter(self, filter_by):
        self.calls.append(("delete_by_filter", filter_by))
        return 2

    def index_document(self, **kw):
        self.calls.append(("index_document", kw))
        return 3

    @staticmethod
    def escape_filter_value(value: str) -> str:
        return "`" + str(value).replace("`", "") + "`"


@pytest.fixture
def dm(monkeypatch):
    """Deep memory without Typesense: a spy index and a spy search."""
    index = _FakeIndex()
    searches: list[tuple] = []

    def fake_search(owner_id, query, *, max_results=10, filter_by=""):
        searches.append((owner_id, query, filter_by))
        return [{"owner": owner_id}]

    monkeypatch.setattr(dr, "_require_connection", lambda: None)
    monkeypatch.setattr(dr.svc, "get_index", lambda: index)
    monkeypatch.setattr(dr.svc, "search", fake_search)
    return type("DM", (), {"index": index, "searches": searches})


@pytest.fixture
async def gdeck(deck, dm):
    """O (helper W, second W2, docker box), M a member of helper and box, N a
    non-member; Google configured, O and M connected (tok-O / tok-M)."""
    db = deck.db
    _add_agents(deck)
    await db.create_share("agent", REF, OWNER, MEMBER, "view")
    await db.create_share("agent", BOX_REF, OWNER, MEMBER, "view")
    await db.set_system_setting(gr._K_CLIENT_ID, "cid")
    await db.set_system_setting(gr._K_CLIENT_SECRET, "csecret")
    await gr._store_tokens(db, OWNER, _tokens("tok-O"))
    await gr._store_tokens(db, MEMBER, _tokens("tok-M"))
    gr._primary_owner_cache.update(id=OWNER, at=time.time())
    gs._policy_locks.clear()
    gs._send_locks.clear()
    deck.dm = dm
    try:
        yield deck
    finally:
        gr._primary_owner_cache.update(id=None, at=0.0)
        gs._policy_locks.clear()
        gs._send_locks.clear()


_RealAsyncClient = httpx.AsyncClient  # the gmail fixture patches httpx.AsyncClient


def _agent_client() -> httpx.AsyncClient:
    """An agent's view of FD: loopback, no Origin / Sec-Fetch-*."""
    return _RealAsyncClient(
        transport=httpx.ASGITransport(app=server.app, client=("127.0.0.1", 50123)),
        base_url="http://fd.test")


def _mint(ref: str = REF, owner: str = OWNER, speaker: str = MEMBER, lane: str = "A",
          turn: str = "t1", conn: str = "c1", **kw) -> str:
    tok = sg.mint(agent_ref=ref, owner=owner, speaker=speaker, lane=lane, turn=turn,
                  conn_id=conn, **kw)
    assert tok, "mint refused"
    return tok


async def _call(c: httpx.AsyncClient, method: str, path: str, *, grant: str | None = None,
                marker: bool = True, auth: str | None = HELPER_TOK, headers: dict | None = None,
                json_body: dict | None = None, extra_params: dict | None = None):
    h: dict = {}
    if auth is not None:
        h["X-Agent-Auth"] = auth
    if grant is not None:
        h[sg.GRANT_HEADER] = grant
    h.update(headers or {})
    params = {sg.MEMBER_MARKER_PARAM: "1"} if marker else {}
    params.update(extra_params or {})
    body = json_body if json_body is not None else BODIES.get(path)
    if method == "GET":
        return await c.get(path, headers=h, params=params)
    return await c.post(path, headers=h, params=params, json=body)


async def _opt_in(db, uid: str = MEMBER, ref: str = REF, owner: str = OWNER) -> None:
    await sg.set_google_optin(db, uid, ref, owner, True)


def _no_tokens_in(r) -> None:
    assert "tok-O" not in r.text and "tok-M" not in r.text


# ── Unit: speaker_grants ──────────────────────────────────────────────────


class TestGrantStore:
    def test_mint_token_shape_and_hash_only(self):
        tok = _mint()
        assert len(tok) == 43 and TOKEN_RE.fullmatch(tok)
        assert list(sg._GRANTS) == [sg.grant_key(tok)]
        g = sg._GRANTS[sg.grant_key(tok)]
        assert tok not in repr(sg.open_grants())
        for field in ("key", "agent_ref", "owner", "speaker", "lane", "turn", "conn_id"):
            assert tok not in str(getattr(g, field))
        assert (g.agent_ref, g.owner, g.speaker, g.lane, g.turn, g.conn_id) == (
            REF, OWNER, MEMBER, "A", "t1", "c1")
        assert _mint(turn="t2") != tok

    def test_lookup_and_end_turn(self):
        tok = _mint()
        assert sg.lookup(tok) is sg._GRANTS[sg.grant_key(tok)]
        for args in ((SECOND_REF, MEMBER, "A", "t1"), (REF, OTHER, "A", "t1"),
                     (REF, MEMBER, "B", "t1"), (REF, MEMBER, "A", "t2")):
            assert sg.end_turn(*args) is False
            assert sg.lookup(tok) is not None
        assert sg.end_turn(REF, MEMBER, "A", "t1") is True
        assert sg.lookup(tok) is None
        assert sg.end_turn(REF, MEMBER, "A", "t1") is False  # idempotent

    def test_ttl(self):
        t0 = 1000.0
        tok = _mint(now=t0)
        assert sg.lookup(tok, now=t0 + sg.GRANT_TTL_S - 1) is not None
        assert sg.lookup(tok, now=t0 + sg.GRANT_TTL_S) is None

    def test_conn_dropped(self):
        t0 = 1000.0
        mine = _mint(conn="c1", turn="t1", now=t0)
        sibling = _mint(conn="c2", turn="t2", now=t0)            # same member and lane
        other = _mint(speaker=OTHER, conn="c3", turn="t3", now=t0)
        short = _mint(conn="c1", turn="t4", now=t0 - sg.GRANT_TTL_S + 50)  # expires at t0+50
        assert sg.conn_dropped("c1", now=t0) == 2
        # (lookups purge what has expired by their `now`, so go forward in time)
        assert sg.lookup(short, now=t0 + 49) is not None
        assert sg.lookup(short, now=t0 + 51) is None              # kept its shorter TTL
        assert sg.lookup(mine, now=t0 + 119) is not None
        assert sg.lookup(mine, now=t0 + 121) is None
        assert sg.lookup(sibling, now=t0 + 121) is not None
        assert sg.lookup(other, now=t0 + 121) is not None
        assert sg.conn_dropped("", now=t0) == 0

    def test_revoke(self):
        a = _mint()
        b = _mint(speaker=OTHER, turn="t2")
        c = _mint(ref=SECOND_REF, turn="t3")
        assert sg.revoke(REF, MEMBER) == 1
        assert sg.lookup(a) is None and sg.lookup(b) is not None
        assert sg.revoke(REF) == 1
        assert sg.lookup(b) is None and sg.lookup(c) is not None

    @pytest.mark.parametrize("bad", ["-", "", "a" * 42, "a" * 44, "a" * 42 + "=",
                                     "a" * 42 + "!", None, 43, b"a" * 43])
    def test_valid_token_format_rejects(self, bad):
        assert sg.valid_token_format(bad) is False
        assert sg.lookup(bad) is None

    def test_valid_token_format_accepts(self):
        assert sg.valid_token_format("A-_z" + "0" * 39)

    def test_member_cap(self, monkeypatch):
        monkeypatch.setattr(sg, "MAX_OPEN_GRANTS_PER_MEMBER", 2)
        t1, t2 = _mint(turn="t1"), _mint(turn="t2")
        assert sg.mint(agent_ref=REF, owner=OWNER, speaker=MEMBER, lane="A", turn="t3",
                       conn_id="c1") == ""
        assert _mint(speaker=OTHER, turn="t4")                     # another member still mints
        assert sg.lookup(t1) and sg.lookup(t2)                     # nothing evicted
        sg.end_turn(REF, MEMBER, "A", "t1")
        assert _mint(turn="t5")                                    # room again (purge on mint)

    def test_agent_cap(self, monkeypatch):
        monkeypatch.setattr(sg, "MAX_OPEN_GRANTS_PER_AGENT", 2)
        t1 = _mint(speaker=MEMBER, turn="t1")
        t2 = _mint(speaker=OTHER, turn="t2")
        assert sg.mint(agent_ref=REF, owner=OWNER, speaker="u-third", lane="A", turn="t3",
                       conn_id="c3") == ""
        assert _mint(ref=SECOND_REF, speaker="u-third", turn="t4")  # another ref still mints
        assert sg.lookup(t1) and sg.lookup(t2)
        sg.end_turn(REF, OTHER, "A", "t2")
        assert _mint(speaker="u-third", turn="t5")

    def test_deck_cap(self, monkeypatch):
        monkeypatch.setattr(sg, "MAX_OPEN_GRANTS", 2)
        t1, t2 = _mint(turn="t1"), _mint(ref=SECOND_REF, turn="t2")
        assert sg.mint(agent_ref=BOX_REF, owner=OWNER, speaker=MEMBER, lane="A", turn="t3",
                       conn_id="c1") == ""
        assert sg.lookup(t1) and sg.lookup(t2)
        sg.revoke(SECOND_REF)
        assert _mint(ref=BOX_REF, turn="t4")

    def test_mint_needs_identifiers(self):
        for kw in ({"agent_ref": ""}, {"owner": ""}, {"speaker": ""}, {"turn": ""}):
            args = dict(agent_ref=REF, owner=OWNER, speaker=MEMBER, lane="A", turn="t",
                        conn_id="c")
            args.update(kw)
            assert sg.mint(**args) == ""
        assert sg.open_grants() == []


class TestGoogleOptInStore:
    async def test_round_trip(self, tmp_path):
        db = FlightDeckDB(tmp_path / "optin.db")
        await db.init()
        try:
            await base._add_users(db)
            await sg.set_google_optin(db, MEMBER, REF, OWNER, True)
            assert await db.get_setting(MEMBER, sg.google_optin_key(REF)) == OWNER
            assert sg.google_optin_key(REF) == "fd:shared-agent-google:" + REF
            assert await sg.google_opted_in(db, MEMBER, REF, OWNER) is True
            assert await sg.google_opted_in(db, MEMBER, REF, OTHER) is False
            assert await sg.google_opted_in(db, MEMBER, REF, "") is False
            assert await sg.google_opted_in(db, OTHER, REF, OWNER) is False
            await sg.set_google_optin(db, MEMBER, REF, OWNER, False)
            assert await db.get_setting(MEMBER, sg.google_optin_key(REF)) is None

            for uid in (MEMBER, OTHER):
                await sg.set_google_optin(db, uid, REF, OWNER, True)
            await sg.set_google_optin(db, MEMBER, SECOND_REF, OWNER, True)
            assert await sg.clear_google_optins(db, REF, MEMBER) == 1
            assert await sg.google_opted_in(db, OTHER, REF, OWNER)
            assert not await sg.google_opted_in(db, MEMBER, REF, OWNER)
            await sg.set_google_optin(db, MEMBER, REF, OWNER, True)
            assert await sg.clear_google_optins(db, REF) == 2
            for uid in (MEMBER, OTHER):
                assert not await sg.google_opted_in(db, uid, REF, OWNER)
            assert await sg.google_opted_in(db, MEMBER, SECOND_REF, OWNER)  # other ref kept
        finally:
            await db.close()

    async def test_fails_closed(self):
        class Broken:
            async def get_setting(self, *a):
                raise RuntimeError("db down")

        assert await sg.google_opted_in(Broken(), MEMBER, REF, OWNER) is False


# ── Unit: deep_memory_filter ──────────────────────────────────────────────


class TestFilterBy:
    @pytest.mark.parametrize("raw", [
        "", "source:=web_fetch", "tags:=[finance]", "tags:=[a, b]",
        "reference:=`vfs:x/a b.md`", "reference:=`a) || (b`",
        "source:!=agent && updated_at:>1700000000", "chunk_index:[0..5]", "path:foo",
    ])
    def test_accepted(self, raw):
        assert dmf.validate_filter_by(raw) == raw.strip()

    def test_returns_stripped(self):
        assert dmf.validate_filter_by("  source:=a  ") == "source:=a"

    @pytest.mark.parametrize("raw", [
        "x:=1) || (id:!=0", "source:=a || source:=b", "(source:=a)", "owner_id:=u1",
        "source:=a && owner_id:=b", "foo:=1", "source:>a", "source:=`a", "source:=a & b",
        "!source:=a", " && ".join(["source:=a"] * 11), "path:" + "a" * 996,
        None, 123, "source:=a\x00",
        # A backslash anywhere: were Typesense to read \` as an escaped backtick,
        # this would close FD's "(…)" and OR in another tenant's documents.
        "source:=`a\\` && path:=`) || owner_id:!=x || (source:=b`",
        "reference:=`a\\b`", "source:a\\b",
    ])
    def test_refused(self, raw):
        if isinstance(raw, str) and raw.startswith("path:"):
            assert len(raw) == 1001
        with pytest.raises(dmf.FilterByError):
            dmf.validate_filter_by(raw)

    @pytest.mark.parametrize("raw,why", [
        ("(source:=a)", "parenthes"), ("source:=a)", "parenthes"), ("source:=a | b", "||"),
        ("source:=a &&& path:b", "&&"), ("source:!a", "!="), ("owner_id:=u1", "owner_id"),
        ("tags:=[a, `b`, c", "unclosed"), ("chunk_index:=[0..5]", "range"),
        ("source:[0..5]", "numeric"), ("tags:=[]", "empty"), ("source:", "no value"),
        ("reference:=`a\\` && path:=`b`", "backslash"),
    ])
    def test_reasons(self, raw, why):
        with pytest.raises(dmf.FilterByError) as exc:
            dmf.validate_filter_by(raw)
        assert why in str(exc.value)

    def test_ten_clauses_and_max_len_are_fine(self):
        assert dmf.validate_filter_by(" && ".join(["source:=a"] * 10))
        raw = "path:`" + "a" * 500 + "` && path:" + "b" * 200 + " && path:" + "c" * 200
        assert len(raw) <= dmf.FILTER_BY_MAX_LEN and dmf.validate_filter_by(raw) == raw

    def test_owner_id_is_not_a_field(self):
        assert "owner_id" not in dmf.FILTER_FIELDS


# ── Unit: member cache generations ────────────────────────────────────────


class _GatedDB:
    def __init__(self):
        self.gate = asyncio.Event()
        self.waiting = asyncio.Event()
        self.calls = 0

    async def get_user_by_id(self, uid):
        return {"id": uid}

    async def is_agent_member(self, ref, owner, uid):
        self.calls += 1
        self.waiting.set()
        await self.gate.wait()
        return True


class TestMemberCacheGeneration:
    @pytest.mark.parametrize("member_only", [True, False])
    async def test_invalidation_during_check_is_not_recached(self, monkeypatch, member_only):
        monkeypatch.setattr(sharing, "_MEMBER_CACHE", {})
        db = _GatedDB()
        task = asyncio.create_task(sharing.member_check(db, REF, OWNER, MEMBER))
        await db.waiting.wait()
        sharing.invalidate_member_cache(REF, MEMBER if member_only else None)
        db.gate.set()
        assert await task is True                     # the DB said yes
        assert (REF, OWNER, MEMBER) not in sharing._MEMBER_CACHE
        await sharing.member_check(db, REF, OWNER, MEMBER)
        assert db.calls == 2                          # next call hits the DB

    async def test_without_invalidation_caches_as_a1(self, monkeypatch):
        monkeypatch.setattr(sharing, "_MEMBER_CACHE", {})
        db = _GatedDB()
        db.gate.set()
        assert await sharing.member_check(db, REF, OWNER, MEMBER)
        assert (REF, OWNER, MEMBER) in sharing._MEMBER_CACHE
        assert await sharing.member_check(db, REF, OWNER, MEMBER)
        assert db.calls == 1

    async def test_other_member_invalidation_does_not_block(self, monkeypatch):
        monkeypatch.setattr(sharing, "_MEMBER_CACHE", {})
        db = _GatedDB()
        task = asyncio.create_task(sharing.member_check(db, REF, OWNER, MEMBER))
        await db.waiting.wait()
        sharing.invalidate_member_cache(REF, OTHER)
        sharing.invalidate_member_cache(SECOND_REF)
        db.gate.set()
        assert await task
        assert (REF, OWNER, MEMBER) in sharing._MEMBER_CACHE


# ── acting_member + grant-aware routes ────────────────────────────────────


class TestOwnerPathUnchanged:
    async def test_no_header_no_marker_is_the_owner(self, gdeck):
        async with _agent_client() as c:
            r = await c.get("/fd/google/access_token", headers={"X-Agent-Auth": HELPER_TOK})
            assert r.status_code == 200 and r.json()["access_token"] == "tok-O"
            r = await c.get("/fd/google/agent_status", headers={"X-Agent-Auth": HELPER_TOK})
            assert r.json() == {"connected": True, "enabled": True}


class TestNoTurn:
    @pytest.mark.parametrize("method,path", ALL_ROUTES)
    async def test_bad_grants(self, gdeck, method, path):
        closed = _mint(turn="closed")
        sg.end_turn(REF, MEMBER, "A", "closed")
        expired = _mint(turn="old", now=time.monotonic() - sg.GRANT_TTL_S - 1)
        await _opt_in(gdeck.db)
        async with _agent_client() as c:
            for grant in ("-", "", "x" * 43, closed, expired):
                r = await _call(c, method, path, grant=grant)
                assert r.status_code == 403, (grant, r.text)
                assert r.json()["detail"] == sg.NO_TURN_DETAIL
                _no_tokens_in(r)
        assert gdeck.dm.index.calls == [] and gdeck.dm.searches == []

    @pytest.mark.parametrize("method,path", ALL_ROUTES)
    async def test_marker_without_header_never_falls_back(self, gdeck, method, path):
        _mint()
        await _opt_in(gdeck.db)
        async with _agent_client() as c:
            r = await _call(c, method, path, grant=None, marker=True)
        assert r.status_code == 403 and r.json()["detail"] == sg.NO_TURN_DETAIL
        _no_tokens_in(r)
        assert gdeck.dm.index.calls == [] and gdeck.dm.searches == []

    async def test_header_without_marker_is_the_member_path(self, gdeck):
        tok = _mint()
        await _opt_in(gdeck.db)
        async with _agent_client() as c:
            r = await _call(c, "GET", "/fd/google/access_token", grant=tok, marker=False)
            assert r.status_code == 200 and r.json()["access_token"] == "tok-M"
            # an unknown token with no marker is still the member path, never the owner
            r = await _call(c, "GET", "/fd/google/access_token", grant="y" * 43, marker=False)
            assert r.status_code == 403 and r.json()["detail"] == sg.NO_TURN_DETAIL


class TestWrongCaller:
    @pytest.mark.parametrize("method,path", ALL_ROUTES)
    async def test_another_agent_of_the_owner(self, gdeck, method, path):
        tok = _mint()
        await _opt_in(gdeck.db)
        async with _agent_client() as c:
            r = await _call(c, method, path, grant=tok, auth=SECOND_TOK)
            assert r.status_code == 403 and r.json()["detail"] == sg.NO_TURN_DETAIL
            _no_tokens_in(r)
            r = await _call(c, method, path, grant=tok, auth=None)
            assert r.status_code == 403 and r.json()["detail"] == sg.NO_TURN_DETAIL
        assert sg.lookup(tok) is not None  # a wrong caller can't end the member's turn

    async def test_browser_and_lockdown(self, gdeck, monkeypatch):
        tok = _mint()
        await _opt_in(gdeck.db)
        async with _agent_client() as c:
            r = await _call(c, "GET", "/fd/google/access_token", grant=tok,
                            headers={"Origin": "http://evil.example"})
            assert r.status_code == 403
            _no_tokens_in(r)
            # same-origin fetch metadata gets past the browser guard; the route refuses it
            r = await _call(c, "GET", "/fd/google/access_token", grant=tok,
                            headers={"Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors"})
            assert r.status_code == 403
            assert r.json()["detail"] == "This endpoint is for Flight Deck agents, not browsers"
            monkeypatch.setenv("FD_LOCKDOWN", "1")
            for method, path in ALL_ROUTES:
                r = await _call(c, method, path, grant=tok)
                assert r.status_code == 401, (path, r.text)
                _no_tokens_in(r)
            r = await _call(c, "GET", "/fd/google/access_token", grant=tok,
                            headers={"X-Agent-Secret": SHARED_SECRET})
            assert r.status_code == 200 and r.json()["access_token"] == "tok-M"


class TestRecordChecks:
    async def test_owner_changed_revokes_and_clears(self, gdeck):
        tok = _mint()
        await _opt_in(gdeck.db)
        reg = server._load_process_registry()
        reg["helper"]["owner"] = OWNER2
        server._save_process_registry(reg)
        async with _agent_client() as c:
            r = await _call(c, "GET", "/fd/google/access_token", grant=tok)
        assert r.status_code == 403 and r.json()["detail"] == sg.NO_TURN_DETAIL
        assert sg.lookup(tok) is None
        assert await gdeck.db.get_setting(MEMBER, sg.google_optin_key(REF)) is None

    async def test_unreadable_registry_keeps_the_grant_and_the_opt_ins(self, gdeck):
        """A failed registry read isn't a removed agent: refuse this one call,
        but keep the member's grant and their (persistent) Google consent."""
        tok = _mint()
        await _opt_in(gdeck.db)
        good = server.PROCESS_REGISTRY_FILE.read_text()
        server.PROCESS_REGISTRY_FILE.write_text("{ not json")
        try:
            assert sharing.resolve_agent_record(REF) is None
            with pytest.raises(sharing.RecordUnavailable):
                sharing.resolve_agent_record(REF, strict=True)
            async with _agent_client() as c:
                r = await _call(c, "GET", "/fd/google/access_token", grant=tok)
            assert r.status_code == 403 and r.json()["detail"] == sg.NO_TURN_DETAIL
            assert sg.lookup(tok) is not None
            assert await gdeck.db.get_setting(MEMBER, sg.google_optin_key(REF)) == OWNER
        finally:
            server.PROCESS_REGISTRY_FILE.write_text(good)
        async with _agent_client() as c:
            r = await _call(c, "GET", "/fd/google/access_token", grant=tok)
        assert r.status_code == 200 and r.json()["access_token"] == "tok-M"

    async def test_agent_removed_revokes_and_clears(self, gdeck):
        tok = _mint()
        await _opt_in(gdeck.db)
        reg = server._load_process_registry()
        reg.pop("helper")
        server._save_process_registry(reg)
        async with _agent_client() as c:
            r = await _call(c, "POST", "/fd/deep-memory/agent/search", grant=tok)
        assert r.status_code == 403 and r.json()["detail"] == sg.NO_TURN_DETAIL
        assert sg.lookup(tok) is None
        assert await gdeck.db.get_setting(MEMBER, sg.google_optin_key(REF)) is None

    async def test_docker_ref_is_refused_and_revoked(self, gdeck):
        tok = _mint(ref=BOX_REF)
        async with _agent_client() as c:
            r = await _call(c, "POST", "/fd/deep-memory/agent/search", grant=tok, auth=BOX_TOK)
        assert r.status_code == 403 and r.json()["detail"] == sg.NO_TURN_DETAIL
        assert sg.lookup(tok) is None
        assert gdeck.dm.searches == []

    async def test_consent_does_not_carry_over_to_a_new_owner(self, gdeck):
        db = gdeck.db
        await _opt_in(db)                                 # given while O owned it
        reg = server._load_process_registry()
        reg["helper"]["owner"] = OWNER2
        server._save_process_registry(reg)
        await db.create_share("agent", REF, OWNER2, MEMBER, "view")
        tok = _mint(owner=OWNER2)
        async with _agent_client() as c:
            r = await _call(c, "GET", "/fd/google/access_token", grant=tok)
            assert r.status_code == 403 and r.json()["detail"] == sg.GOOGLE_OFF_DETAIL
            _no_tokens_in(r)
            r = await _call(c, "GET", "/fd/google/agent_status", grant=tok)
            assert r.status_code == 200 and r.json() == {"connected": False, "enabled": False}

    async def test_membership_lost(self, gdeck):
        tok = _mint()
        await _opt_in(gdeck.db)
        async with _agent_client() as c:
            r = await _call(c, "GET", "/fd/google/access_token", grant=tok)
            assert r.status_code == 200                     # membership now cached
            await gdeck.db.delete_share("agent", REF, OWNER, MEMBER)
            sharing.invalidate_member_cache(REF, MEMBER)
            r = await _call(c, "GET", "/fd/google/access_token", grant=tok)
        assert r.status_code == 403 and r.json()["detail"] == sg.NOT_MEMBER_DETAIL
        assert sg.lookup(tok) is None

    async def test_member_user_deleted(self, gdeck):
        tok = _mint()
        await gdeck.db.delete_user(MEMBER)
        async with _agent_client() as c:
            r = await _call(c, "POST", "/fd/deep-memory/agent/search", grant=tok)
        assert r.status_code == 403 and r.json()["detail"] == sg.NOT_MEMBER_DETAIL
        assert sg.lookup(tok) is None

    @pytest.mark.parametrize("closer", ["turn_end", "revoke"])
    async def test_grant_closed_while_validating(self, gdeck, monkeypatch, closer):
        """The turn ends (or access is revoked) while acting_member awaits the
        membership read: the request acts for nobody, never the member."""
        tok = _mint()
        await _opt_in(gdeck.db)
        real = sharing.member_check

        async def closing(*a, **kw):
            ok = await real(*a, **kw)
            if closer == "turn_end":
                sg.end_turn(REF, MEMBER, "A", "t1")
            else:
                sg.revoke(REF)
            return ok

        monkeypatch.setattr(sharing, "member_check", closing)
        async with _agent_client() as c:
            r = await _call(c, "GET", "/fd/google/access_token", grant=tok)
        assert r.status_code == 403 and r.json()["detail"] == sg.NO_TURN_DETAIL
        _no_tokens_in(r)


class TestGoogleAsMember:
    async def test_no_optin(self, gdeck):
        tok = _mint()
        async with _agent_client() as c:
            r = await _call(c, "GET", "/fd/google/access_token", grant=tok)
            assert r.status_code == 403 and r.json()["detail"] == sg.GOOGLE_OFF_DETAIL
            _no_tokens_in(r)
            r = await _call(c, "GET", "/fd/google/agent_status", grant=tok)
            assert r.status_code == 200 and r.json() == {"connected": False, "enabled": False}
        assert sg.lookup(tok) is not None  # Google off doesn't end the turn

    async def test_optin_and_connected(self, gdeck):
        tok = _mint()
        await _opt_in(gdeck.db)
        async with _agent_client() as c:
            r = await _call(c, "GET", "/fd/google/access_token", grant=tok)
            assert r.status_code == 200 and r.json()["access_token"] == "tok-M"
            r = await _call(c, "GET", "/fd/google/agent_status", grant=tok)
        assert r.json() == {"connected": True, "enabled": True}
        assert not {"access_token", "scope", "refresh_token"} & set(r.json())

    async def test_optin_not_connected(self, gdeck):
        tok = _mint()
        await _opt_in(gdeck.db)
        await gdeck.db.delete_setting(MEMBER, gr._K_TOKENS)
        async with _agent_client() as c:
            r = await _call(c, "GET", "/fd/google/access_token", grant=tok)
            assert r.status_code == 404
            _no_tokens_in(r)
            r = await _call(c, "GET", "/fd/google/agent_status", grant=tok)
        assert r.json() == {"connected": False, "enabled": True}

    async def test_agent_status_owner(self, gdeck):
        await gdeck.db.delete_setting(OWNER, gr._K_TOKENS)
        gr._primary_owner_cache.update(id=OTHER, at=time.time())  # no legacy fallback for O
        async with _agent_client() as c:
            r = await c.get("/fd/google/agent_status", headers={"X-Agent-Auth": HELPER_TOK})
        assert r.json() == {"connected": False, "enabled": True}

    async def test_credentials_refuses_the_markers(self, gdeck):
        tok = _mint()
        async with _agent_client() as c:
            r = await _call(c, "GET", "/fd/google/credentials", grant=tok)
        assert r.status_code == 403 and r.json()["detail"] == sg.OFF_PATH_DETAIL


class _Gmail:
    def __init__(self):
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "POST" and request.url.path.endswith("/messages/send"):
            return httpx.Response(200, json={"id": "sent-1", "threadId": "t-new"})
        return httpx.Response(404, json={"error": {"message": "Not Found"}})


@pytest.fixture
def gmail(monkeypatch):
    g = _Gmail()
    monkeypatch.setattr(gs.httpx, "AsyncClient",
                        lambda **kw: _RealAsyncClient(transport=httpx.MockTransport(g), **kw))
    return g


async def _gmail_policy(db, uid: str, enabled: bool) -> None:
    await db.set_settings(uid, {gs._K_GMAIL_SEND: json.dumps(
        {"enabled": enabled, "allowed_recipients": [], "daily_limit": 50})})


class TestGmailSendAsMember:
    async def test_member_policy_off_refuses(self, gdeck, gmail):
        db = gdeck.db
        await _opt_in(db)
        await _gmail_policy(db, OWNER, True)
        await _gmail_policy(db, MEMBER, False)
        tok = _mint()
        async with _agent_client() as c:
            r = await _call(c, "POST", "/fd/google/gmail/send", grant=tok)
        assert r.status_code == 403
        assert r.headers.get(gmail_compose.SEND_REFUSED_HEADER) == "off"
        assert gmail.requests == []

    async def test_member_send_is_the_members(self, gdeck, gmail):
        db = gdeck.db
        await _opt_in(db)
        await _gmail_policy(db, OWNER, True)
        await _gmail_policy(db, MEMBER, True)
        tok = _mint()
        since = "2000-01-01T00:00:00+00:00"
        async with _agent_client() as c:
            r = await _call(c, "POST", "/fd/google/gmail/send", grant=tok)
        assert r.status_code == 200, r.text
        assert [q.headers["Authorization"] for q in gmail.requests] == ["Bearer tok-M"]
        assert await db.count_gmail_sends_since(MEMBER, since) == 1
        assert await db.count_gmail_sends_since(OWNER, since) == 0
        rows = await db.list_gmail_sends(MEMBER)
        assert rows[0]["owner_id"] == MEMBER
        assert rows[0]["agent"] == "Helper (shared by Olga Owner)"
        assert "(shared by " in rows[0]["agent"]
        assert [n["type"] for n in await db.list_notifications(MEMBER)] == ["email_sent"]
        assert await db.list_notifications(OWNER) == []

    async def test_member_send_needs_the_optin(self, gdeck, gmail):
        db = gdeck.db
        await _gmail_policy(db, MEMBER, True)
        tok = _mint()
        async with _agent_client() as c:
            r = await _call(c, "POST", "/fd/google/gmail/send", grant=tok)
        assert r.status_code == 403 and r.json()["detail"] == sg.GOOGLE_OFF_DETAIL
        assert gmail.requests == []


class TestDeepMemoryAsMember:
    @pytest.fixture(autouse=True)
    def _no_grid(self, monkeypatch):
        def boom(request):
            raise AssertionError("_agent_grid consulted on a member call")

        monkeypatch.setattr(dr, "_agent_grid", boom)

    async def test_search_index_delete_use_the_member(self, gdeck, monkeypatch):
        monkeypatch.setenv("FD_ARCHETYPE_GRID", "1")
        tok = _mint()
        async with _agent_client() as c:
            r = await _call(c, "POST", "/fd/deep-memory/agent/search", grant=tok,
                            json_body={"query": "q", "filter_by": "source:=web_fetch"})
            assert r.status_code == 200, r.text
            r = await _call(c, "POST", "/fd/deep-memory/agent/index", grant=tok)
            assert r.status_code == 200, r.text
            r = await _call(c, "POST", "/fd/deep-memory/agent/delete", grant=tok)
            assert r.status_code == 200, r.text
        assert gdeck.dm.searches == [(MEMBER, "q", "source:=web_fetch")]
        calls = gdeck.dm.index.calls
        assert calls[0] == ("delete_by_reference", "r1", MEMBER)
        assert calls[1][0] == "index_document"
        assert calls[1][1]["owner_id"] == MEMBER and calls[1][1]["tags"] is None
        assert calls[2] == ("delete_by_reference", "r1", MEMBER)

    async def test_member_filter_delete_refused(self, gdeck):
        tok = _mint()
        async with _agent_client() as c:
            r = await _call(c, "POST", "/fd/deep-memory/agent/delete", grant=tok,
                            json_body={"filter_by": "source:=agent"})
        assert r.status_code == 400 and r.json()["detail"] == sg.MEMBER_DELETE_DETAIL
        assert gdeck.dm.index.calls == []


class TestDeepMemoryOwnerAndFilters:
    async def test_owner_path_grid_and_filter_delete(self, gdeck, monkeypatch):
        tags, recall = ["agent:reviewer", "domain:legal"], "domain"
        monkeypatch.setattr(dr, "_agent_grid", lambda request: (tags, recall))
        rf = recall_filter(recall, tags)
        assert rf
        async with _agent_client() as c:
            h = {"X-Agent-Auth": HELPER_TOK}
            r = await c.post("/fd/deep-memory/agent/search", headers=h,
                             json={"query": "q", "filter_by": " source:=vfs "})
            assert r.status_code == 200, r.text
            r = await c.post("/fd/deep-memory/agent/search", headers=h, json={"query": "q2"})
            assert r.status_code == 200
            r = await c.post("/fd/deep-memory/agent/delete", headers=h,
                             json={"filter_by": "tags:=[a, b]"})
            assert r.status_code == 200, r.text
        assert gdeck.dm.searches == [(OWNER, "q", f"(source:=vfs) && {rf}"), (OWNER, "q2", rf)]
        assert gdeck.dm.index.calls == [
            ("delete_by_filter", f"(tags:=[a, b]) && owner_id:=`{OWNER}`")]

    @pytest.mark.parametrize("bad", ["x:=1) || (id:!=0", "owner_id:=u-member",
                                     "source:=a || source:=b"])
    async def test_invalid_filter_by(self, gdeck, bad):
        async with _agent_client() as c:
            h = {"X-Agent-Auth": HELPER_TOK}
            r = await c.post("/fd/deep-memory/agent/search", headers=h,
                             json={"query": "q", "filter_by": bad})
            assert r.status_code == 400 and r.json()["detail"].startswith("Invalid filter_by: ")
            r = await c.post("/fd/deep-memory/agent/delete", headers=h, json={"filter_by": bad})
            assert r.status_code == 400 and r.json()["detail"].startswith("Invalid filter_by: ")
        async with _client() as c:
            r = await c.get("/fd/deep-memory/search", params={"q": "x", "filter_by": bad},
                            headers=_hdr(OWNER))
            assert r.status_code == 400 and r.json()["detail"].startswith("Invalid filter_by: ")
            r = await c.get("/fd/deep-memory/search",
                            params={"q": "x", "filter_by": "source:=vfs"}, headers=_hdr(OWNER))
            assert r.status_code == 200
        assert gdeck.dm.index.calls == []
        assert gdeck.dm.searches == [(OWNER, "x", "source:=vfs")]

    async def test_member_invalid_filter(self, gdeck, monkeypatch):
        def boom(request):
            raise AssertionError("_agent_grid consulted on a member call")

        monkeypatch.setattr(dr, "_agent_grid", boom)
        tok = _mint()
        async with _agent_client() as c:
            r = await _call(c, "POST", "/fd/deep-memory/agent/search", grant=tok,
                            json_body={"query": "q", "filter_by": "(source:=a)"})
        assert r.status_code == 400 and r.json()["detail"].startswith("Invalid filter_by: ")
        assert gdeck.dm.searches == []


# ── Middleware ────────────────────────────────────────────────────────────


OFF_PATHS = [("GET", "/fd/google/credentials"), ("POST", "/fd/mcp/agent/x"),
             ("POST", "/fd/consult-peer"), ("POST", "/fd/basna/agent/x"),
             ("GET", "/fd/processes"), ("GET", "/fd/settings")]


class TestGrantGuardMiddleware:
    @pytest.mark.parametrize("method,path", OFF_PATHS)
    @pytest.mark.parametrize("variant", ["header", "marker", "odd-case", "blank-marker"])
    async def test_refused_off_path(self, gdeck, method, path, variant):
        headers = {"X-Agent-Auth": HELPER_TOK}
        params: dict = {}
        if variant == "header":
            headers[sg.GRANT_HEADER] = _mint()
        elif variant == "odd-case":
            headers["x-Fd-Speaker-GRANT"] = _mint()
        elif variant == "marker":
            params[sg.MEMBER_MARKER_PARAM] = "1"
        else:
            params[sg.MEMBER_MARKER_PARAM] = ""
        headers.update(_hdr(OWNER))
        async with _agent_client() as c:
            r = await c.request(method, path, headers=headers, params=params, json={})
        assert r.status_code == 403 and r.json() == {"detail": sg.OFF_PATH_DETAIL}

    @pytest.mark.parametrize("method,path", ALL_ROUTES)
    async def test_grant_aware_paths_reach_the_route(self, gdeck, method, path):
        async with _agent_client() as c:
            r = await _call(c, method, path, grant="z" * 43)
            # the route ran its grant validation (an unknown grant)
            assert r.status_code == 403 and r.json()["detail"] == sg.NO_TURN_DETAIL, r.text
            # With a trailing "/" the guard lets it through too; FD's own router
            # then answers it (its SPA catch-all: 404 / 405) — never the guard's 403.
            r = await _call(c, method, path + "/", grant="z" * 43)
            assert r.status_code != 403, r.text
            assert r.json() != {"detail": sg.OFF_PATH_DETAIL}

    async def test_root_path_is_stripped(self):
        seen: list = []

        async def app(scope, receive, send):
            seen.append(scope.get("path", scope["type"]))
            if scope["type"] == "http":
                await send({"type": "http.response.start", "status": 204, "headers": []})
                await send({"type": "http.response.body", "body": b""})

        mw = sg.GrantGuardMiddleware(app)
        sent: list = []

        async def send(msg):
            sent.append(msg)

        scope = {"type": "http", "method": "GET", "path": "/deck/fd/google/access_token/",
                 "root_path": "/deck", "query_string": b"fd_member=1", "headers": []}
        await mw(scope, None, send)
        assert seen == ["/deck/fd/google/access_token/"] and sent[0]["status"] == 204
        sent.clear()
        await mw(dict(scope, path="/deck/fd/processes"), None, send)
        assert sent[0]["status"] == 403 and len(seen) == 1
        # lifespan and the like pass through untouched
        await mw({"type": "lifespan"}, None, send)
        assert len(seen) == 2

    async def test_nothing_changes_without_markers(self, gdeck):
        async with _agent_client() as c:
            assert (await c.get("/fd/processes")).status_code == 401
            assert (await c.get("/fd/processes", headers=_hdr(OWNER))).status_code == 200
            r = await c.get("/fd/google/credentials", headers={"X-Agent-Auth": HELPER_TOK})
            assert r.status_code != 403 or r.json()["detail"] != sg.OFF_PATH_DETAIL

    def test_websocket_handshake_with_markers(self, ws_deck):
        url = base._ws_url(token=create_access_token(MEMBER))
        for extra, kw in (("", {"headers": {sg.GRANT_HEADER: "x" * 43}}),
                          ("", {"headers": {"x-FD-speaker-grant": ""}}),
                          ("&fd_member=1", {}), ("&fd_member", {})):
            with pytest.raises(WebSocketDisconnect) as exc:
                with ws_deck.client.websocket_connect(url + extra, **kw):
                    pass
            assert exc.value.code == 4403
        assert ws_deck.agent.requests == []  # never reached the agent
        cm, s, _ = _open(ws_deck)            # the control: same socket without markers
        cm.__exit__(None, None, None)


# ── VFS link-target guard ─────────────────────────────────────────────────


@pytest.fixture
def vfs_links(deck, tmp_path):
    from captain_claw.flight_deck import vfs_routes

    data = deck.data
    (data / "flightdeck.db").write_bytes(b"SQLITE-SECRET-DB")
    (data / "agent_secret").write_text("AGENT-SECRET-VALUE")
    peer = data / "vfs" / "u-peer"
    peer.mkdir(parents=True)
    (peer / "private.md").write_text("PEER-PRIVATE")
    root = vfs_routes._user_root(OWNER)
    gd = root / ".drive" / "gd"
    gd.mkdir(parents=True)
    (gd / "notes.md").write_text("DRIVE NOTES")
    (gd / ".drive-manifest.json").write_text(json.dumps(
        {"folder_id": "f1", "dirs": {"": "f1"},
         "files": {"notes.md": {"id": "d1", "state": "cloned"}}}))
    other = tmp_path / "other"
    other.mkdir()
    (other / "readme.md").write_text("EXTERNAL OK")
    links = {
        "evil": {"path": str(data), "mode": "rw"},
        "up": {"path": str(data.parent), "mode": "rw"},
        "peer": {"path": str(peer), "mode": "rw"},
        "gd": {"path": str(gd), "kind": "gdrive", "mode": "ro", "drive": {}},
        "ext": {"path": str(other), "mode": "rw"},
    }
    (root / ".vfs-links.json").write_text(json.dumps(links))
    deck.vfs_root = root
    return deck


_SECRETS = ("SQLITE-SECRET-DB", "AGENT-SECRET-VALUE", "PEER-PRIVATE", HELPER_TOK)


class TestVfsLinkGuard:
    async def test_listing_marks_refused_links_missing(self, vfs_links):
        async with _client() as c:
            r = await c.get("/fd/vfs/projects", headers=_hdr(OWNER))
        assert r.status_code == 200, r.text
        rows = {p["name"]: p for p in r.json()["projects"]}
        for name in ("evil", "up", "peer"):
            assert rows[name].get("missing") is True, rows[name]
        assert not rows["gd"].get("missing") and rows["gd"]["files"] >= 1
        assert not rows["ext"].get("missing") and rows["ext"]["files"] == 1

    @pytest.mark.parametrize("name,path", [("evil", "flightdeck.db"), ("evil", "agent_secret"),
                                           ("evil", ".processes.json"), ("up", "fd-data/flightdeck.db"),
                                           ("peer", "private.md")])
    async def test_refused_links_serve_nothing(self, vfs_links, name, path):
        async with _client() as c:
            h = _hdr(OWNER)
            for r in (await c.get("/fd/vfs/list", params={"project": name}, headers=h),
                      await c.get("/fd/vfs/read", params={"project": name, "path": path}, headers=h),
                      await c.get("/fd/vfs/download", params={"project": name, "path": path},
                                  headers=h),
                      await c.get("/fd/vfs/download-zip", params={"project": name}, headers=h)):
                assert r.status_code == 404, (r.request.url, r.status_code, r.text[:200])
                for secret in _SECRETS:
                    assert secret.encode() not in r.content

    async def test_drive_mount_and_external_link_still_work(self, vfs_links):
        async with _client() as c:
            h = _hdr(OWNER)
            r = await c.get("/fd/vfs/read", params={"project": "gd", "path": "notes.md"}, headers=h)
            assert r.status_code == 200 and r.json()["text"] == "DRIVE NOTES"
            r = await c.get("/fd/vfs/list", params={"project": "gd"}, headers=h)
            assert r.status_code == 200
            r = await c.get("/fd/vfs/read", params={"project": "ext", "path": "readme.md"},
                            headers=h)
            assert r.status_code == 200 and r.json()["text"] == "EXTERNAL OK"

    async def test_safe_link_target_unit(self, vfs_links):
        from captain_claw.flight_deck.vfs_routes import safe_link_target

        root = vfs_links.vfs_root
        for name in ("evil", "up", "peer", "nope"):
            assert safe_link_target(root, name) is None
        assert safe_link_target(root, "gd") == (root / ".drive" / "gd").resolve()
        assert safe_link_target(root, "ext") is not None

    async def test_deep_memory_service_and_code_routes(self, vfs_links):
        from fastapi import HTTPException

        from captain_claw.flight_deck import code_routes
        from captain_claw.flight_deck import deep_memory_service as dms

        root = vfs_links.vfs_root
        got = dms.resolve(OWNER, "evil", "x")
        assert got == (root / "evil" / "x").resolve()
        assert dms.resolve(OWNER, "ext", "readme.md").read_text() == "EXTERNAL OK"
        code_routes._write_project(OWNER, "proj", {"folders": [
            {"name": "bad", "kind": "link", "link": "evil", "mode": "rw"},
            {"name": "good", "kind": "link", "link": "ext", "mode": "rw"}]})
        with pytest.raises(HTTPException) as exc:
            code_routes._folder_repo(OWNER, "proj", "bad")
        assert exc.value.status_code == 404
        assert code_routes._folder_repo(OWNER, "proj", "good").name == "other"

    async def test_other_spellings_of_the_data_dir_are_refused(self, vfs_links):
        """resolve() keeps the caller's spelling: on a case-insensitive volume
        (macOS' default) ``…/FD-DATA`` is the data dir without being equal to
        it, and so is the ``/System/Volumes/Data/…`` firmlink path."""
        from captain_claw.flight_deck.vfs_routes import safe_link_target

        data = vfs_links.data
        spellings = _other_spellings(data)
        if not spellings:
            pytest.skip("this filesystem has a single spelling per directory")
        root = vfs_links.vfs_root
        links = json.loads((root / ".vfs-links.json").read_text())
        for i, alias in enumerate(spellings):
            links[f"alias{i}"] = {"path": str(alias), "mode": "rw"}
            links[f"aliaspeer{i}"] = {"path": str(alias / "vfs" / "u-peer"), "mode": "rw"}
            links[f"aliasown{i}"] = {"path": str(alias / "vfs" / OWNER / ".drive" / "gd"),
                                     "mode": "ro"}
        (root / ".vfs-links.json").write_text(json.dumps(links))
        for i in range(len(spellings)):
            assert safe_link_target(root, f"alias{i}") is None
            assert safe_link_target(root, f"aliaspeer{i}") is None
            assert safe_link_target(root, f"aliasown{i}") is not None  # still the own root
        async with _client() as c:
            h = _hdr(OWNER)
            for i, alias in enumerate(spellings):
                for project, path in ((f"alias{i}", "flightdeck.db"),
                                      (f"alias{i}", ".processes.json"),
                                      (f"aliaspeer{i}", "private.md")):
                    r = await c.get("/fd/vfs/read", params={"project": project, "path": path},
                                    headers=h)
                    assert r.status_code == 404, (project, path, r.text[:200])
                    for secret in _SECRETS:
                        assert secret.encode() not in r.content
                r = await c.post("/fd/vfs/links", json={"name": f"new{i}", "path": str(alias),
                                                        "mode": "rw"}, headers=h)
                assert r.status_code == 400, r.text

    async def test_public_hosting_never_serves_deck_internals(self, vfs_links, monkeypatch):
        """Published folders resolve through ``vfs.resolve_under`` (public, no
        login): a hosted project that is a refused link serves nothing."""
        from captain_claw.flight_deck import vfs_hosting as vh

        data = vfs_links.data
        monkeypatch.setenv("CLAW_VFS_ROOT", str(data / "vfs"))  # resolve_under = vfs_routes root
        monkeypatch.setattr(vh, "_VISITS", {})
        monkeypatch.setattr(vh, "_VISIT_COUNTS", {})
        site = vfs_links.vfs_root / "site"
        site.mkdir()
        (site / "index.html").write_text("SITE OK")
        root = vfs_links.vfs_root
        links = json.loads((root / ".vfs-links.json").read_text())
        for i, alias in enumerate(_other_spellings(data)):
            links[f"alias{i}"] = {"path": str(alias), "mode": "rw"}
        (root / ".vfs-links.json").write_text(json.dumps(links))
        projects = ["evil", "up", "peer", "ext", "site", "gd"] + [
            f"alias{i}" for i in range(len(_other_spellings(data)))]
        reg = {f"pub-{p}": {"kind": "static", "owner": OWNER, "project": p, "subdir": ""}
               for p in projects}
        vh.save_registry(reg)
        refused = [(n, p) for n in reg if n not in ("pub-ext", "pub-site", "pub-gd")
                   for p in ("", "flightdeck.db", "agent_secret", ".processes.json",
                             "fd-data/flightdeck.db", "private.md")]
        async with _client() as c:
            for name, path in refused:
                r = await c.get(f"/vfs/{name}/{path}")
                assert r.status_code == 404, (name, path, r.text[:200])
                for secret in _SECRETS:
                    assert secret.encode() not in r.content
            r = await c.get("/vfs/pub-ext/readme.md")
            assert r.status_code == 200 and r.text == "EXTERNAL OK"
            r = await c.get("/vfs/pub-site/")
            assert r.status_code == 200 and r.text == "SITE OK"
            r = await c.get("/vfs/pub-gd/notes.md")
            assert r.status_code == 200 and r.text == "DRIVE NOTES"
        assert vh.entry_dir(reg["pub-evil"]) is None
        assert vh.entry_dir(reg["pub-ext"]) is not None


def _other_spellings(p) -> list:
    """Other names this host accepts for directory ``p`` (a case-insensitive
    volume; macOS' /System/Volumes/Data firmlink) — [] on a host with neither."""
    from pathlib import Path

    out = []
    upper = p.parent / p.name.upper()
    if upper.name != p.name and upper.exists() and upper.samefile(p):
        out.append(upper)
    firm = Path("/System/Volumes/Data" + str(p))
    if firm.exists() and firm.samefile(p):
        out.append(firm)
    return out


# ── Opt-in route, settings guard, listings ────────────────────────────────


class TestOptInRoutes:
    async def test_put_on_off(self, gdeck):
        db = gdeck.db
        key = sg.google_optin_key(REF)
        async with _client() as c:
            r = await c.put("/fd/shared-agents/google", json={"agent_ref": REF, "enabled": True},
                            headers=_hdr(MEMBER))
            assert r.status_code == 200 and r.json() == {"agent_ref": REF, "google_enabled": True}
            assert await db.get_setting(MEMBER, key) == OWNER
            r = await c.put("/fd/shared-agents/google", json={"agent_ref": REF, "enabled": False},
                            headers=_hdr(MEMBER))
            assert r.json() == {"agent_ref": REF, "google_enabled": False}
            assert await db.get_setting(MEMBER, key) is None

    async def test_a_revoke_racing_the_write_leaves_no_opt_in(self, gdeck, monkeypatch):
        """Membership is lost between the route's check and its write (a share
        DELETE cleared opt-ins in between): the route removes what it wrote."""
        db = gdeck.db
        real = sharing.member_check
        calls = {"n": 0}

        async def flaky(*a, **kw):
            calls["n"] += 1
            return await real(*a, **kw) if calls["n"] == 1 else False

        monkeypatch.setattr(sharing, "member_check", flaky)
        async with _client() as c:
            r = await c.put("/fd/shared-agents/google", json={"agent_ref": REF, "enabled": True},
                            headers=_hdr(MEMBER))
        assert r.status_code == 404
        assert await db.get_setting(MEMBER, sg.google_optin_key(REF)) is None

    async def test_refusals(self, gdeck, monkeypatch):
        db = gdeck.db
        async with _client() as c:
            def put(uid, ref=REF, enabled=True):
                return c.put("/fd/shared-agents/google",
                             json={"agent_ref": ref, "enabled": enabled}, headers=_hdr(uid))

            assert (await put(NON_MEMBER)).status_code == 404
            assert (await put(OWNER)).status_code == 404
            assert (await put(MEMBER, ref="garbage")).status_code == 400
            assert (await put(MEMBER, ref="process:helper:aaaaaaaaaaaaaaaa")).status_code == 404
            r = await put(MEMBER, ref=BOX_REF)
            assert r.status_code == 400
            assert r.json()["detail"] == "Google isn't available for this agent in shared chats"
            assert await db.get_setting(MEMBER, sg.google_optin_key(BOX_REF)) is None
            monkeypatch.setattr(sharing, "SHARING_ENABLED", False)
            r = await put(MEMBER)
            assert r.status_code == 400 and r.json()["detail"] == (
                "Agent sharing is off on this Flight Deck")
        for uid in (MEMBER, NON_MEMBER, OWNER):
            assert await db.get_setting(uid, sg.google_optin_key(REF)) is None

    async def test_settings_routes_hide_and_refuse(self, gdeck):
        await _opt_in(gdeck.db)
        key = sg.google_optin_key(REF)
        async with _client() as c:
            r = await c.get("/fd/settings", headers=_hdr(MEMBER))
            assert r.status_code == 200 and key not in r.json()
            r = await c.put("/fd/settings", json={"settings": {key: "1"}}, headers=_hdr(MEMBER))
            assert r.status_code == 400
            r = await c.delete(f"/fd/settings/{key}", headers=_hdr(MEMBER))
            assert r.status_code == 400
        assert await gdeck.db.get_setting(MEMBER, key) == OWNER


class TestListings:
    async def test_shared_agents_rows(self, gdeck, monkeypatch):
        db = gdeck.db
        calls: list = []
        real = gr.is_google_connected

        async def spy(uid):
            calls.append(uid)
            return await real(uid)

        monkeypatch.setattr(gr, "is_google_connected", spy)
        await db.delete_setting(MEMBER, gr._K_TOKENS)
        async with _client() as c:
            body = (await c.get("/fd/shared-agents", headers=_hdr(MEMBER))).json()
            rows = {a["agent_ref"]: a for a in body["agents"]}
            assert set(rows) == {REF, BOX_REF}
            assert rows[REF]["capabilities"] == {"google": True, "deep_memory": True, "files": True,
                                                 "datastore": True}
            assert rows[BOX_REF]["capabilities"] == {"google": False, "deep_memory": False,
                                                     "files": False, "datastore": False}
            assert rows[REF]["google_enabled"] is False and rows[BOX_REF]["google_enabled"] is False
            assert rows[REF]["google_connected"] is False
            assert calls == [MEMBER]                 # once per request
            assert body["mine"] == {}

            await c.put("/fd/shared-agents/google", json={"agent_ref": REF, "enabled": True},
                        headers=_hdr(MEMBER))
            await sg.set_google_optin(db, MEMBER, BOX_REF, OWNER, True)  # never via the route
            await gr._store_tokens(db, MEMBER, _tokens("tok-M"))
            body = (await c.get("/fd/shared-agents", headers=_hdr(MEMBER))).json()
            rows = {a["agent_ref"]: a for a in body["agents"]}
            assert rows[REF]["google_enabled"] is True
            assert rows[BOX_REF]["google_enabled"] is False  # docker: never
            assert rows[REF]["google_connected"] is True and rows[BOX_REF]["google_connected"]
            assert calls == [MEMBER, MEMBER]
            for row in body["agents"]:
                assert not {"web_auth", "port", "host", "access_token"} & set(row)

    async def test_mine_for_the_owner(self, gdeck):
        await _opt_in(gdeck.db)
        async with _client() as c:
            body = (await c.get("/fd/shared-agents", headers=_hdr(OWNER))).json()
        assert body["agents"] == []
        assert body["mine"] == {REF: {"members": 1, "google": 1},
                                BOX_REF: {"members": 1, "google": 0}}

    async def test_flag_off_unchanged(self, gdeck, monkeypatch):
        monkeypatch.setattr(sharing, "SHARING_ENABLED", False)
        async with _client() as c:
            r = await c.get("/fd/shared-agents", headers=_hdr(OWNER))
        assert r.json() == {"enabled": False, "host_warning": "", "agents": []}

    async def test_owner_share_rows(self, gdeck):
        async with _client() as c:
            q = {"resource_type": "agent", "resource_id": REF}
            r = await c.get("/fd/shares", params=q, headers=_hdr(OWNER))
            assert [s["google_enabled"] for s in r.json()["shares"]] == [False]
            await _opt_in(gdeck.db)
            r = await c.get("/fd/shares", params=q, headers=_hdr(OWNER))
            assert [(s["grantee_id"], s["google_enabled"]) for s in r.json()["shares"]] == [
                (MEMBER, True)]
            vfs_dir = server.DATA_DIR / "vfs" / OWNER / "proj"
            vfs_dir.mkdir(parents=True)
            await gdeck.db.create_share("vfs", "proj", OWNER, MEMBER, "view")
            r = await c.get("/fd/shares", params={"resource_type": "vfs", "resource_id": "proj"},
                            headers=_hdr(OWNER))
            assert r.status_code == 200 and r.json()["shares"]
            assert all("google_enabled" not in s for s in r.json()["shares"])

    def test_host_trust_warning_text(self):
        assert sharing.HOST_TRUST_WARNING == (
            "Anyone on this deck who runs their own shell-capable process agent can act as "
            "any agent on this host, including this one, and can read every user's Flight Deck "
            "files at any time. While a member's message is being answered (up to 20 minutes), "
            "they can also use that member's deep memory and, if the member turned it on, their "
            "Google account.")


# ── Lifecycle ─────────────────────────────────────────────────────────────


class TestLifecycle:
    @pytest.fixture
    def order(self, monkeypatch):
        calls: list = []
        real_inv, real_revoke = sharing.invalidate_member_cache, sg.revoke

        def inv(ref, user_id=None):
            calls.append(("invalidate", ref, user_id))
            return real_inv(ref, user_id)

        def rev(ref, speaker=None):
            calls.append(("revoke", ref, speaker))
            return real_revoke(ref, speaker)

        async def close(ref, user_id=None, *, code=4403, reason="Access removed"):
            calls.append(("close", ref, user_id))
            return 0

        monkeypatch.setattr(sharing, "invalidate_member_cache", inv)
        monkeypatch.setattr(sg, "revoke", rev)
        monkeypatch.setattr(sharing, "close_member_sockets", close)
        return calls

    async def _two_members(self, db):
        await db.create_share("agent", REF, OWNER, OTHER, "view")
        await _opt_in(db, MEMBER)
        await _opt_in(db, OTHER)
        return _mint(speaker=MEMBER, turn="m"), _mint(speaker=OTHER, turn="o")

    async def test_owner_delete(self, gdeck, order):
        db = gdeck.db
        m, o = await self._two_members(db)
        async with _client() as c:
            r = await c.delete("/fd/shares", params={"resource_type": "agent", "resource_id": REF,
                                                     "grantee_id": MEMBER}, headers=_hdr(OWNER))
        assert r.json() == {"ok": True}
        assert sg.lookup(m) is None and sg.lookup(o) is not None
        assert not await sg.google_opted_in(db, MEMBER, REF, OWNER)
        assert await sg.google_opted_in(db, OTHER, REF, OWNER)
        assert order[:3] == [("invalidate", REF, MEMBER), ("revoke", REF, MEMBER),
                             ("close", REF, MEMBER)]

    async def test_member_leaves(self, gdeck, order):
        db = gdeck.db
        m, o = await self._two_members(db)
        async with _client() as c:
            r = await c.delete("/fd/shares/leave", params={"resource_type": "agent",
                                                           "resource_id": REF, "owner_id": OWNER},
                               headers=_hdr(MEMBER))
        assert r.json() == {"ok": True}
        assert sg.lookup(m) is None and sg.lookup(o) is not None
        assert not await sg.google_opted_in(db, MEMBER, REF, OWNER)
        assert await sg.google_opted_in(db, OTHER, REF, OWNER)
        assert [c[0] for c in order[:3]] == ["invalidate", "revoke", "close"]

    async def test_reshare_starts_with_google_off(self, gdeck):
        db = gdeck.db
        await _opt_in(db)
        async with _client() as c:
            await c.delete("/fd/shares", params={"resource_type": "agent", "resource_id": REF,
                                                 "grantee_id": MEMBER}, headers=_hdr(OWNER))
            r = await c.post("/fd/shares", json={"resource_type": "agent", "resource_id": REF,
                                                 "grantee_id": MEMBER}, headers=_hdr(OWNER))
            assert r.status_code == 200
        assert not await sg.google_opted_in(db, MEMBER, REF, OWNER)

    async def test_process_removed(self, gdeck):
        db = gdeck.db
        m, o = await self._two_members(db)
        keep = _mint(ref=SECOND_REF, turn="s")
        await sg.set_google_optin(db, MEMBER, SECOND_REF, OWNER, True)
        async with _client() as c:
            r = await c.delete("/fd/processes/helper", headers=_hdr(OWNER))
        assert r.status_code == 200, r.text
        assert sg.lookup(m) is None and sg.lookup(o) is None and sg.lookup(keep) is not None
        for uid in (MEMBER, OTHER):
            assert await db.get_setting(uid, sg.google_optin_key(REF)) is None
        assert await sg.google_opted_in(db, MEMBER, SECOND_REF, OWNER)

    async def test_container_removed(self, gdeck):
        db = gdeck.db
        g = _mint(ref=BOX_REF)
        await sg.set_google_optin(db, MEMBER, BOX_REF, OWNER, True)
        async with _client() as c:
            r = await c.delete("/fd/containers/sid-box", headers=_hdr(OWNER))
        assert r.status_code == 200, r.text
        assert sg.lookup(g) is None
        assert await db.get_setting(MEMBER, sg.google_optin_key(BOX_REF)) is None


# ── WS proxy ──────────────────────────────────────────────────────────────


def _pcall(env, fn, *args):
    """Run a (sync or async) callable on FD's event loop."""
    return env.portal.call(fn, *args)


def _grants(env) -> list:
    return _pcall(env, sg.open_grants)


def _wait(cond, timeout: float = 5.0, step: float = 0.02) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(step)
    raise AssertionError("condition not met in time")


def _chats(env, n: int) -> list[dict]:
    return [f for f in env.agent.wait_frames(n) if f["type"] == "chat"]


def _conn_ids(env) -> list[str]:
    return _pcall(env, lambda: list(sharing._SOCKETS))


class TestWsProxyGrants:
    def test_member_chat_carries_a_grant(self, ws_deck):
        cm, s, _ = _open(ws_deck)
        try:
            (conn_id,) = _conn_ids(ws_deck)
            s.send_json({"type": "chat", "content": "hello"})
            (chat,) = _chats(ws_deck, 2)
            assert TOKEN_RE.fullmatch(chat["_fd_grant"]) and chat["_fd_turn"]
            (g,) = _grants(ws_deck)
            assert (g.agent_ref, g.owner, g.speaker, g.lane, g.turn, g.conn_id) == (
                REF, OWNER, MEMBER, "A", chat["_fd_turn"], conn_id)
            assert g.key == sg.grant_key(chat["_fd_grant"])
            # the agent ends the turn → the grant closes
            done = {"type": "status", "status": "ready", "turn_end": chat["_fd_turn"]}
            ws_deck.agent.push(0, done)
            assert _recv_json(s) == done
            assert _pcall(ws_deck, sg.lookup, chat["_fd_grant"]) is None
        finally:
            cm.__exit__(None, None, None)

    def test_docker_agent_gets_no_grant(self, ws_deck):
        box = FakeAgent(BOX_TOK)
        try:
            ws_deck.containers.append(FakeContainer(ws_deck.containers, "box", OWNER, BOX_TOK,
                                                    box.port, instance=BOX_INST))
            _pcall(ws_deck, ws_deck.db.create_share, "agent", BOX_REF, OWNER, MEMBER, "view")
            cm, s, _ = _open(ws_deck, ref=BOX_REF)
            try:
                s.send_json({"type": "chat", "content": "hello", "_fd_grant": "x" * 43})
                (chat,) = [f for f in box.wait_frames(2) if f["type"] == "chat"]
                assert "_fd_grant" not in chat and chat["_fd_turn"]
                assert _grants(ws_deck) == []
            finally:
                cm.__exit__(None, None, None)
        finally:
            box.stop()

    def test_no_grant_for_slash_or_empty(self, ws_deck):
        cm, s, _ = _open(ws_deck)
        try:
            for i, content in enumerate(("   ", "/help", "  /help")):
                s.send_json({"type": "chat", "content": content, "_fd_grant": "x" * 43})
                chat = _chats(ws_deck, 2 + i)[-1]
                assert chat["content"] == content and "_fd_grant" not in chat
                ws_deck.agent.push(0, {"type": "status", "status": "ready",
                                       "turn_end": chat["_fd_turn"]})
                _recv_json(s)
            assert _grants(ws_deck) == []
        finally:
            cm.__exit__(None, None, None)

    def test_client_grant_is_replaced(self, ws_deck):
        cm, s, _ = _open(ws_deck)
        try:
            forged = "x" * 43
            s.send_json({"type": "chat", "content": "hi", "_fd_grant": forged,
                         "_fd_turn": "aaaaaaaaaaaaaaaa"})
            (chat,) = _chats(ws_deck, 2)
            assert chat["_fd_grant"] != forged and TOKEN_RE.fullmatch(chat["_fd_grant"])
            assert _pcall(ws_deck, sg.lookup, forged) is None
        finally:
            cm.__exit__(None, None, None)

    def test_closed_socket_cuts_its_grants_only(self, ws_deck, monkeypatch):
        monkeypatch.setattr(sg, "ORPHAN_GRACE_S", 0.2)
        cm1, s1, _ = _open(ws_deck)
        cm2, s2, _ = _open(ws_deck)          # same member, same lane A
        try:
            s1.send_json({"type": "chat", "content": "one"})
            s2.send_json({"type": "chat", "content": "two"})
            chats = _chats(ws_deck, 4)
            g1 = next(c["_fd_grant"] for c in chats if c["content"] == "one")
            g2 = next(c["_fd_grant"] for c in chats if c["content"] == "two")
            cm1.__exit__(None, None, None)
            _wait(lambda: _pcall(ws_deck, sharing.live_conn_count, REF, MEMBER) == 1)
            g = _pcall(ws_deck, sg.lookup, g1)
            assert g is None or g.expires_at <= time.monotonic() + 0.2
            time.sleep(0.3)
            assert _pcall(ws_deck, sg.lookup, g1) is None
            assert _pcall(ws_deck, sg.lookup, g2) is not None
        finally:
            cm2.__exit__(None, None, None)

    def test_turn_end_on_a_sibling_socket_closes_it(self, ws_deck):
        cm1, s1, _ = _open(ws_deck)
        cm2, s2, _ = _open(ws_deck)
        try:
            s1.send_json({"type": "chat", "content": "one"})
            (chat,) = _chats(ws_deck, 3)
            cm1.__exit__(None, None, None)
            _wait(lambda: _pcall(ws_deck, sharing.live_conn_count, REF, MEMBER) == 1)
            g = _pcall(ws_deck, sg.lookup, chat["_fd_grant"])
            assert g is not None and g.expires_at <= time.monotonic() + sg.ORPHAN_GRACE_S + 1
            done = {"type": "status", "status": "ready", "turn_end": chat["_fd_turn"]}
            ws_deck.agent.push(1, done)      # the agent's connection for socket 2
            assert _recv_json(s2) == done
            assert _pcall(ws_deck, sg.lookup, chat["_fd_grant"]) is None
        finally:
            cm2.__exit__(None, None, None)

    def test_only_socket_on_lane_closes(self, ws_deck, monkeypatch):
        monkeypatch.setattr(sg, "ORPHAN_GRACE_S", 0.2)
        cm_a, sa, _ = _open(ws_deck)
        cm_b, sb, _ = _open(ws_deck, lane="B")
        try:
            sa.send_json({"type": "chat", "content": "on A"})
            sb.send_json({"type": "chat", "content": "on B"})
            chats = _chats(ws_deck, 4)
            ga = next(c["_fd_grant"] for c in chats if c["content"] == "on A")
            gb = next(c["_fd_grant"] for c in chats if c["content"] == "on B")
            lanes = {g.lane for g in _grants(ws_deck)}
            assert lanes == {"A", "B"}
            cm_a.__exit__(None, None, None)
            _wait(lambda: _pcall(ws_deck, sharing.live_conn_count, REF, MEMBER) == 1)
            time.sleep(0.3)
            assert _pcall(ws_deck, sg.lookup, ga) is None
            assert _pcall(ws_deck, sg.lookup, gb) is not None
        finally:
            cm_b.__exit__(None, None, None)

    def test_delete_share_mid_turn(self, ws_deck):
        cm, s, _ = _open(ws_deck)
        try:
            s.send_json({"type": "chat", "content": "hello"})
            (chat,) = _chats(ws_deck, 2)
            r = ws_deck.client.delete(f"{HTTP_FD}/fd/shares", params={
                "resource_type": "agent", "resource_id": REF, "grantee_id": MEMBER},
                headers=_hdr(OWNER))
            assert r.json() == {"ok": True}
            _expect_close(s, 4403, timeout=3)
            assert _pcall(ws_deck, sg.lookup, chat["_fd_grant"]) is None
        finally:
            cm.__exit__(None, None, None)

    @pytest.mark.parametrize("close_sockets", [True, False])
    def test_revocation_racing_a_chat(self, ws_deck, monkeypatch, close_sockets):
        """The share goes away while the chat's member_check awaits the DB."""
        db = ws_deck.db
        cm, s, _ = _open(ws_deck)
        real = db.is_agent_member
        armed = {"on": True}

        async def racing(ref, owner, uid):
            ok = await real(ref, owner, uid)
            if armed["on"]:
                armed["on"] = False
                await db.delete_share("agent", REF, OWNER, MEMBER)
                if close_sockets:
                    await share_routes._revoke_agent_member(REF, MEMBER, "Access removed")
                else:  # the window before the sockets close
                    sharing.invalidate_member_cache(REF, MEMBER)
            return ok

        _pcall(ws_deck, sharing._MEMBER_CACHE.clear)  # the chat's check must reach the DB
        monkeypatch.setattr(db, "is_agent_member", racing)
        try:
            s.send_json({"type": "chat", "content": "racing"})
            time.sleep(0.5)
            chats = [f for _, f in ws_deck.agent.frames if f["type"] == "chat"]
            grants = [c["_fd_grant"] for c in chats if "_fd_grant" in c]
            if close_sockets:
                assert all("_fd_grant" not in c for c in chats)
                assert _grants(ws_deck) == []
            for g in grants:  # any grant that got out is refused on first use
                async def use(token=g):
                    async with _agent_client() as c:
                        return await _call(c, "POST", "/fd/deep-memory/agent/search",
                                           grant=token)

                with monkeypatch.context() as m:
                    m.setattr(dr, "_require_connection", lambda: None)
                    m.setattr(dr.svc, "search", lambda *a, **k: [])
                    r = _pcall(ws_deck, use)
                assert r.status_code == 403 and r.json()["detail"] == sg.NOT_MEMBER_DETAIL
                assert _pcall(ws_deck, sg.lookup, g) is None
            assert (REF, OWNER, MEMBER) not in sharing._MEMBER_CACHE
        finally:
            cm.__exit__(None, None, None)

    def test_watchdog_membership_loss(self, ws_deck, monkeypatch):
        monkeypatch.setattr(sharing, "RECHECK_INTERVAL_S", 0.1)
        cm, s, _ = _open(ws_deck)
        try:
            s.send_json({"type": "chat", "content": "hello"})
            (chat,) = _chats(ws_deck, 2)
            _pcall(ws_deck, ws_deck.db.delete_share, "agent", REF, OWNER, MEMBER)
            _expect_close(s, 4403, timeout=2)
            assert _pcall(ws_deck, sg.lookup, chat["_fd_grant"]) is None
        finally:
            cm.__exit__(None, None, None)

    def test_watchdog_owner_change(self, ws_deck, monkeypatch):
        monkeypatch.setattr(sharing, "RECHECK_INTERVAL_S", 0.1)
        db = ws_deck.db
        _pcall(ws_deck, sg.set_google_optin, db, MEMBER, REF, OWNER, True)
        cm, s, _ = _open(ws_deck)
        try:
            s.send_json({"type": "chat", "content": "hello"})
            (chat,) = _chats(ws_deck, 2)
            reg = server._load_process_registry()
            reg["helper"]["owner"] = OTHER
            server._save_process_registry(reg)
            _expect_close(s, 4403, timeout=2)
            assert _pcall(ws_deck, sg.lookup, chat["_fd_grant"]) is None
            assert _pcall(ws_deck, db.get_setting, MEMBER, sg.google_optin_key(REF)) is None
        finally:
            cm.__exit__(None, None, None)

    def test_upstream_closed_when_forwarding(self, ws_deck, monkeypatch):
        """The agent connection goes away just as FD forwards the chat: the send
        fails and the grant minted for it is closed at once."""
        from websockets.asyncio.client import ClientConnection

        sent: list = []
        real_send = ClientConnection.send

        async def send(self, message, *a, **kw):
            if isinstance(message, str) and '"_fd_grant"' in message:
                sent.append(json.loads(message)["_fd_grant"])
                await self.close()
            return await real_send(self, message, *a, **kw)

        monkeypatch.setattr(ClientConnection, "send", send)
        cm, s, _ = _open(ws_deck)
        try:
            s.send_json({"type": "chat", "content": "hello"})
            _wait(lambda: bool(sent))
            _wait(lambda: _pcall(ws_deck, sg.lookup, sent[0]) is None, timeout=3)
            assert _grants(ws_deck) == []
        finally:
            cm.__exit__(None, None, None)

    def test_third_concurrent_chat_gets_no_grant(self, ws_deck):
        cm, s, _ = _open(ws_deck)
        try:
            for n in ("one", "two", "three"):
                s.send_json({"type": "chat", "content": n})
            assert _recv_json(s)["code"] == "busy"
            end = _recv_json(s)
            chats = _chats(ws_deck, 3)
            assert [c["content"] for c in chats] == ["one", "two"]
            grants = _grants(ws_deck)
            assert sorted(g.turn for g in grants) == sorted(c["_fd_turn"] for c in chats)
            assert end["turn_end"] not in {g.turn for g in grants}
        finally:
            cm.__exit__(None, None, None)
