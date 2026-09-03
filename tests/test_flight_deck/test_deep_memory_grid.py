"""Deep-memory grid wiring: the FD proxy stamps an agent's axis tags on write and
narrows its reads by recall_mode — without the agent asserting anything.

Exercises the `/agent/index` and `/agent/search` route functions directly with a
stub Request and stubbed identity/connection/index seams, plus the server-side
grid resolvers that read the process registry.
"""

from __future__ import annotations

import types

import pytest

import captain_claw.flight_deck.deep_memory_routes as dr
import captain_claw.flight_deck.server as server


def _req(port: int = 41000, token: str = ""):
    headers = {"X-Agent-Auth": token} if token else {}
    return types.SimpleNamespace(
        headers=headers,
        client=types.SimpleNamespace(host="127.0.0.1", port=port),
    )


class _FakeIndex:
    """Captures index_document kwargs and delete_by_reference calls."""

    def __init__(self):
        self.indexed: dict = {}
        self.deleted: list = []

    def delete_by_reference(self, reference, owner_id=""):
        self.deleted.append((reference, owner_id))
        return 0

    def index_document(self, **kwargs):
        self.indexed = kwargs
        return 3


@pytest.fixture
def stub_proxy(monkeypatch):
    """Stub owner, connection, and index/search so the routes need no Typesense.
    Returns a namespace whose `.grid` / `.search_filter` the test can set/read."""
    ns = types.SimpleNamespace(index=_FakeIndex(), grid=([], ""), search_filter=None)

    monkeypatch.setattr(dr, "_agent_owner", lambda request: "alice")
    monkeypatch.setattr(dr, "_agent_grid", lambda request: ns.grid)
    monkeypatch.setattr(dr, "_require_connection", lambda: None)
    monkeypatch.setattr(dr.svc, "get_index", lambda: ns.index)

    def fake_search(owner_id, query, *, max_results=10, filter_by=""):
        ns.search_filter = filter_by
        return [{"owner": owner_id, "q": query}]

    monkeypatch.setattr(dr.svc, "search", fake_search)
    return ns


# ── /agent/index — tag stamping ──────────────────────────────────────

async def test_index_stamps_grid_tags(stub_proxy):
    stub_proxy.grid = (["agent:reviewer", "domain:legal"], "domain")
    body = dr.AgentIndexBody(text="a legal finding", reference="ref1")
    out = await dr.agent_index(body, _req())
    assert out["ok"] is True
    assert stub_proxy.index.indexed["tags"] == ["agent:reviewer", "domain:legal"]
    assert stub_proxy.index.indexed["owner_id"] == "alice"


async def test_index_non_grid_agent_writes_no_tags(stub_proxy):
    stub_proxy.grid = ([], "")
    body = dr.AgentIndexBody(text="plain note")
    await dr.agent_index(body, _req())
    assert stub_proxy.index.indexed["tags"] is None   # index_document drops falsy tags


# ── /agent/search — recall narrowing ─────────────────────────────────

async def test_search_pool_leaves_filter_untouched(stub_proxy):
    stub_proxy.grid = (["agent:researcher", "domain:finance"], "pool")
    await dr.agent_search(dr.AgentSearchBody(query="q"), _req())
    assert stub_proxy.search_filter == ""


async def test_search_domain_ands_domain_tag(stub_proxy):
    stub_proxy.grid = (["agent:reviewer", "domain:legal"], "domain")
    await dr.agent_search(dr.AgentSearchBody(query="q"), _req())
    assert stub_proxy.search_filter == "tags:=`domain:legal`"


async def test_search_domain_ands_onto_caller_filter(stub_proxy):
    stub_proxy.grid = (["agent:reviewer", "domain:legal"], "domain")
    await dr.agent_search(dr.AgentSearchBody(query="q", filter_by="source:=agent"), _req())
    assert stub_proxy.search_filter == "(source:=agent) && tags:=`domain:legal`"


async def test_search_self_ands_agent_tag(stub_proxy):
    stub_proxy.grid = (["agent:analyst", "domain:finance"], "self")
    await dr.agent_search(dr.AgentSearchBody(query="q"), _req())
    assert stub_proxy.search_filter == "tags:=`agent:analyst`"


# ── server-side grid resolvers (registry path; Docker skipped) ───────

@pytest.fixture
def stub_registry(monkeypatch):
    """Force the Docker branch to no-op and drive the process-registry branch."""
    monkeypatch.setattr(server, "get_docker", lambda: (_ for _ in ()).throw(RuntimeError("no docker")))
    reg = {
        "cc-rev": {
            "web_auth": "tok-rev", "web_port": 41000,
            "grid_tags": ["agent:reviewer", "domain:legal"], "grid_recall": "domain",
        },
        "cc-plain": {"web_auth": "tok-plain", "web_port": 41001},  # non-grid
    }
    monkeypatch.setattr(server, "_load_process_registry", lambda: reg)
    monkeypatch.setattr(server, "_process_is_alive", lambda slug: True)
    return reg


def test_resolve_grid_by_auth_returns_tags_and_mode(stub_registry):
    tags, recall = server._resolve_agent_grid_by_auth("tok-rev")
    assert tags == ["agent:reviewer", "domain:legal"]
    assert recall == "domain"


def test_resolve_grid_by_auth_non_grid_is_empty(stub_registry):
    assert server._resolve_agent_grid_by_auth("tok-plain") == ([], "")


def test_resolve_grid_by_auth_unknown_token_is_empty(stub_registry):
    assert server._resolve_agent_grid_by_auth("nope") == ([], "")


def test_resolve_grid_by_port(stub_registry):
    assert server._resolve_agent_grid(41000) == (["agent:reviewer", "domain:legal"], "domain")


def test_parse_grid_labels_reads_docker_labels():
    labels = {"flight-deck.grid-tags": '["agent:writer","domain:finance"]',
              "flight-deck.grid-recall": "pool"}
    assert server._parse_grid_labels(labels) == (["agent:writer", "domain:finance"], "pool")


def test_parse_grid_labels_malformed_is_empty():
    assert server._parse_grid_labels({"flight-deck.grid-tags": "{not json"}) == ([], "")
    assert server._parse_grid_labels({}) == ([], "")
