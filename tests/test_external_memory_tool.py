"""The agent's side of the external memory (``external_memory_client``, the
``external_memory`` tool and the mirror hooks):

* the tool's actions, parameters and description (with the rules) come from
  the manifest Flight Deck sends — a provider offering ``profile`` gets it, a
  read-only key gets no ``remember``;
* the registry hides the tool until the owner has a memory connected, and from
  a shared-agent member;
* insights and active intentions are mirrored in the background; nothing is
  mirrored standalone, on a member's turn, in a public "computer" run, or for
  a kind the person turned off.

Flight Deck is never contacted: ``httpx.AsyncClient`` gets a mock transport.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from captain_claw import external_memory_client as emc
from captain_claw.config import Config
from captain_claw.tools.external_memory import ExternalMemoryTool
from captain_claw.tools.registry import ToolRegistry

FD = "http://fd.test"
_RealAsyncClient = httpx.AsyncClient

MANIFEST = {
    "connected": True,
    "provider": "membase",
    "label": "Membase",
    "reaches": ["Work"],
    "actions": [
        {"name": "remember", "description": "Save one fact.", "writes": True, "destructive": False,
         "params": [{"name": "content", "type": "string", "description": "The fact.", "required": True},
                    {"name": "title", "type": "string", "description": "A title.", "required": False}]},
        {"name": "search", "description": "Find things.", "writes": False, "destructive": False,
         "params": [{"name": "query", "type": "string", "description": "What.", "required": True},
                    {"name": "limit", "type": "integer", "description": "How many.", "required": False}]},
        {"name": "profile", "description": "Who the person is.", "writes": False, "destructive": False,
         "params": [{"name": "limit", "type": "integer", "description": "How many.", "required": False}]},
    ],
    "rules": ["Shared by ALL of their agents.", "Cite the memory_name."],
    "mirror": ["insight", "intention"],
}


@pytest.fixture(autouse=True)
def _config(monkeypatch, tmp_path):
    import captain_claw.config as config_mod
    from captain_claw.flight_deck import agent_secret
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    monkeypatch.delenv("FD_URL", raising=False)
    agent_secret.reset_cache_for_tests()
    cfg = Config()
    monkeypatch.setattr(config_mod, "_config", cfg)
    monkeypatch.setattr(GoogleOAuthManager, "_flight_deck_base", staticmethod(lambda: ""))
    monkeypatch.setattr(GoogleOAuthManager, "_member_call", staticmethod(lambda: False))
    emc._set_manifest({})
    emc._MANIFEST_AT = 0.0
    emc._LAST_CHECK_AT = 0.0
    yield cfg
    emc._set_manifest({})
    emc._MANIFEST_AT = 0.0
    emc._LAST_CHECK_AT = 0.0
    agent_secret.reset_cache_for_tests()


def _under_fd(monkeypatch):
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    monkeypatch.setattr(GoogleOAuthManager, "_flight_deck_base", staticmethod(lambda: FD))


def _member(monkeypatch):
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    monkeypatch.setattr(GoogleOAuthManager, "_member_call", staticmethod(lambda: True))


class _Peer:
    def __init__(self, respond):
        self.respond = respond
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.respond(request)

    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]

    def body(self, i: int = -1):
        return json.loads(self.requests[i].content or b"null")


def _serve(monkeypatch, respond) -> _Peer:
    peer = _Peer(respond)
    monkeypatch.setattr(emc.httpx, "AsyncClient",
                        lambda **kw: _RealAsyncClient(transport=httpx.MockTransport(peer), **kw))
    return peer


def _fd(manifest=None, run_text="ok", mirror=None):
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/fd/external-memory/agent/status":
            return httpx.Response(200, json=manifest if manifest is not None else MANIFEST)
        if request.url.path == "/fd/external-memory/agent/run":
            return httpx.Response(200, json={"text": run_text, "data": {}})
        if request.url.path == "/fd/external-memory/agent/mirror":
            return httpx.Response(200, json=mirror or {"queued": True, "reason": ""})
        return httpx.Response(404, json={"detail": "Not Found"})
    return respond


# ── the tool ────────────────────────────────────────────────────────


class TestTool:
    def test_schema_follows_the_manifest(self):
        emc._set_manifest(MANIFEST)
        d = ExternalMemoryTool().get_definition()
        props = d["parameters"]["properties"]
        assert props["action"]["enum"] == ["remember", "search", "profile"]
        assert set(props) == {"action", "content", "title", "query", "limit"}
        assert "remember:" in props["content"]["description"]
        assert "search:" in props["limit"]["description"] and "profile:" in props["limit"]["description"]
        assert d["parameters"]["required"] == ["action"]
        assert "Connected: Membase" in d["description"]
        assert "Cite the memory_name." in d["description"]
        assert "timeout" not in props
        for spec in props.values():
            assert not isinstance(spec.get("type"), list) and "additionalProperties" not in spec

    def test_a_read_only_manifest_has_no_remember(self):
        emc._set_manifest({**MANIFEST, "actions": [a for a in MANIFEST["actions"] if a["name"] != "remember"]})
        assert "remember" not in ExternalMemoryTool().parameters["properties"]["action"]["enum"]

    def test_unconnected_schema_is_the_minimum(self):
        props = ExternalMemoryTool().parameters["properties"]
        assert props["action"]["enum"] == ["remember", "search"]

    async def test_execute_sends_only_the_actions_own_params(self, monkeypatch):
        _under_fd(monkeypatch)
        peer = _serve(monkeypatch, _fd(run_text="Saved."))
        out = await ExternalMemoryTool().execute(
            action="remember", content="Ana is Bob's sister.", query="ignored", _session_id="s1",
        )
        assert out.success and out.content == "Saved."
        assert peer.paths() == ["/fd/external-memory/agent/status", "/fd/external-memory/agent/run"]
        assert peer.body() == {"action": "remember", "args": {"content": "Ana is Bob's sister."}}

    async def test_missing_required_and_unknown_actions(self, monkeypatch):
        _under_fd(monkeypatch)
        _serve(monkeypatch, _fd())
        out = await ExternalMemoryTool().execute(action="search")
        assert not out.success and "query" in out.error
        out = await ExternalMemoryTool().execute(action="forget", id="mem_1")
        assert not out.success and "offers: remember, search, profile" in out.error

    async def test_not_connected_says_where_to_connect(self, monkeypatch):
        _under_fd(monkeypatch)
        _serve(monkeypatch, _fd(manifest={"connected": False}))
        out = await ExternalMemoryTool().execute(action="search", query="x")
        assert not out.success and "Connections → External memory" in out.error

    async def test_member_and_standalone_are_refused(self, monkeypatch):
        out = await ExternalMemoryTool().execute(action="search", query="x")
        assert not out.success and out.error == emc.NOT_UNDER_FD
        _under_fd(monkeypatch)
        _member(monkeypatch)
        peer = _serve(monkeypatch, _fd())
        out = await ExternalMemoryTool().execute(action="search", query="x")
        assert not out.success and out.error == emc.MEMBER_REFUSED and peer.requests == []

    async def test_flight_deck_errors_reach_the_agent(self, monkeypatch):
        _under_fd(monkeypatch)

        def respond(request):
            if request.url.path.endswith("/status"):
                return httpx.Response(200, json=MANIFEST)
            return httpx.Response(409, json={"detail": "Ask the person; only if they agree, call again with confirm=true."})

        _serve(monkeypatch, respond)
        out = await ExternalMemoryTool().execute(action="search", query="x")
        assert not out.success and "confirm=true" in out.error


# ── registry gate ───────────────────────────────────────────────────


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ExternalMemoryTool(), metadata={"requires_external_memory": True})
    return reg


def _names(reg: ToolRegistry, **kw) -> list[str]:
    return [t.name for t in reg._resolve_tools(**kw)]


class TestRegistryGate:
    def test_hidden_until_connected_and_fresh(self, monkeypatch):
        reg = _registry()
        emc._set_manifest(MANIFEST)
        assert "external_memory" not in _names(reg)  # standalone: never
        _under_fd(monkeypatch)
        assert "external_memory" in _names(reg)
        emc._MANIFEST_AT = time.time() - 3600
        assert "external_memory" not in _names(reg)
        emc._set_manifest({"connected": False})
        assert "external_memory" not in _names(reg)

    def test_member_never_sees_it(self, monkeypatch):
        _under_fd(monkeypatch)
        emc._set_manifest(MANIFEST)
        reg = _registry()
        reg.register_speaker_session("member-session")
        assert "external_memory" not in _names(reg, session_id="member-session")
        assert "external_memory" in _names(reg, session_id="owner-session")

    def test_always_enabled(self):
        assert "external_memory" in Config().tools.enabled


# ── mirroring ───────────────────────────────────────────────────────


async def _settle():
    for _ in range(20):
        if not emc._pending:
            return
        await asyncio.sleep(0.01)


class TestMirror:
    async def test_mirror_posts_in_the_background(self, monkeypatch):
        _under_fd(monkeypatch)
        peer = _serve(monkeypatch, _fd())
        emc.mirror("insight", "ins-1", "Ana is Bob's sister.", tags="family, contact",
                   meta={"category": "contact", "why": None})
        await _settle()
        assert peer.paths()[-1] == "/fd/external-memory/agent/mirror"
        body = peer.body()
        assert body["kind"] == "insight" and body["source_id"] == "ins-1"
        assert body["tags"] == ["family", "contact"] and body["meta"] == {"category": "contact"}

    async def test_a_kind_turned_off_is_not_sent(self, monkeypatch):
        _under_fd(monkeypatch)
        peer = _serve(monkeypatch, _fd())
        emc.mirror("reflection", "r1", "Be brief.")
        await _settle()
        assert "/fd/external-memory/agent/mirror" not in peer.paths()

    async def test_nothing_standalone_for_members_or_public_runs(self, monkeypatch, _config):
        peer = _serve(monkeypatch, _fd())
        emc.mirror("insight", "i1", "fact")       # standalone
        await _settle()
        _under_fd(monkeypatch)
        _config.web.public_run = "computer"
        emc.mirror("insight", "i2", "fact")       # visitors' memory
        await _settle()
        _config.web.public_run = ""
        _member(monkeypatch)
        emc.mirror("insight", "i3", "fact")       # a member's turn
        await _settle()
        assert peer.requests == []

    async def test_insights_are_mirrored_when_stored(self, monkeypatch, tmp_path):
        from captain_claw.insights import InsightsManager

        sent = []
        monkeypatch.setattr(emc, "mirror", lambda *a, **kw: sent.append((a, kw)))
        mgr = InsightsManager(tmp_path / "insights.db")
        iid = await mgr.add("Bob's dentist is Dr. Horvat.", "contact", why="for appointments")
        assert await mgr.add("Bob's dentist is Dr. Horvat.", "contact") is None  # deduped: not again
        await mgr.close()
        assert len(sent) == 1
        (kind, source_id, content), kw = sent[0]
        assert (kind, source_id, content) == ("insight", iid, "Bob's dentist is Dr. Horvat.")
        assert kw["meta"]["category"] == "contact" and kw["meta"]["why"] == "for appointments"

    async def test_intentions_are_mirrored_once_active(self, monkeypatch, tmp_path):
        from captain_claw.intentions import IntentionsManager

        sent = []
        monkeypatch.setattr(emc, "mirror", lambda *a, **kw: sent.append((a, kw)))
        mgr = IntentionsManager(tmp_path / "intentions.db")
        proposed = await mgr.create(origin="agent", title="Follow up with Acme", risk="normal")
        assert proposed["status"] != "active" and sent == []
        await mgr.set_status(proposed["id"], "active")
        assert [s[0][:2] for s in sent] == [("intention", proposed["id"])]
        note = await mgr.create(origin="user", title="Call mum on Sunday", approval_mode="silent")
        assert note["status"] == "active" and sent[-1][0][:3] == ("intention", note["id"], "Call mum on Sunday")
        await mgr.close()
