"""Increment 5: producer-aware wait heartbeat, BLOCKED directive, tool relay."""

import time
import types

import pytest

from captain_claw.flight_deck import deliverable_manifest as dm
from captain_claw.flight_deck import vatra_routes as vr


def _manifest():
    subs = [{"id": "s4", "depends_on": []}, {"id": "s5", "depends_on": ["s4"]}]
    return dm.parse({"path": "book.md", "parts": [
        {"path": "part-one.md", "owner": "s4"},
        {"path": "part-two.md", "owner": "s5"}]}, subs, "proj")


def test_resolve_producer_by_path(monkeypatch):
    sid = "sid1"
    monkeypatch.setitem(vr._run_manifest, sid, _manifest())
    monkeypatch.setitem(vr._owner_activity, sid, {
        "s4": {"state": "running", "role": "Editor", "last_seen": time.monotonic()}})
    prod = vr._resolve_producer(sid, "vfs:proj/part-one.md", "")
    assert prod is not None and prod["role"] == "Editor"
    # a path with no matching part → None
    assert vr._resolve_producer(sid, "vfs:proj/unknown.md", "") is None


def test_resolve_producer_by_query_running(monkeypatch):
    sid = "sid1"
    monkeypatch.setitem(vr._owner_activity, sid, {
        "s4": {"state": "done"}, "s5": {"state": "running", "role": "Writer"}})
    prod = vr._resolve_producer(sid, "", "some topic")
    assert prod is not None and prod["state"] == "running"


def test_build_subtask_prompt_blocked_directive():
    st = {"id": "s5", "title": "Part Two", "brief": "write ch6-10", "owner_archetype_id": "editor-writer"}
    subs = [st]
    p = vr._build_subtask_prompt("Editor", "write the book", st, [], subs,
                                 blocked_inputs=["part-one.md"])
    assert "BLOCKED:" in p
    assert "part-one.md" in p
    assert "do NOT reconstruct" in p.lower() or "do not reconstruct" in p.lower()
    # default: the "never stop / produce your best version" autonomy sentence
    p2 = vr._build_subtask_prompt("Editor", "x", st, [], subs)
    assert "produce your best version" in p2
    assert "BLOCKED:" not in p2


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


@pytest.mark.asyncio
async def test_tool_wait_relays_producer_alive(monkeypatch):
    from captain_claw.tools.vatra import VatraTool
    monkeypatch.setenv("CLAW_VATRA_SESSION", "sid1")
    monkeypatch.setenv("CLAW_VATRA_SUBTASK", "s5")
    monkeypatch.setenv("CLAW_VATRA_OWNER", "editor-writer")
    tool = VatraTool()

    captured = {}

    async def fake_post(self, fd_url, path, payload, timeout=45.0):
        captured.update(payload)
        return _FakeResp({"ready": False, "producer_alive": True,
                          "producer": {"role": "Editor", "state": "running", "last_seen_s": 12}})

    monkeypatch.setattr(VatraTool, "_post", fake_post)
    monkeypatch.setattr(VatraTool, "_get_fd_url", lambda self, **k: "http://x")
    res = await tool.execute(action="wait", path="vfs:proj/part-one.md")
    assert res.success
    assert "STILL WORKING" in res.content
    assert "do NOT proceed" in res.content or "do not proceed" in res.content.lower()
    # the tool forwarded the subtask id (so the server keys the ledger by subtask)
    assert captured.get("subtask_id") == "s5"


@pytest.mark.asyncio
async def test_tool_wait_relays_exhausted_note(monkeypatch):
    from captain_claw.tools.vatra import VatraTool
    monkeypatch.setenv("CLAW_VATRA_SESSION", "sid1")
    monkeypatch.setenv("CLAW_VATRA_SUBTASK", "s5")
    monkeypatch.setenv("CLAW_VATRA_OWNER", "editor-writer")
    tool = VatraTool()

    async def fake_post(self, fd_url, path, payload, timeout=45.0):
        return _FakeResp({"ready": False, "exhausted": True,
                          "note": "Wait budget spent — produce now.", "board": []})

    monkeypatch.setattr(VatraTool, "_post", fake_post)
    monkeypatch.setattr(VatraTool, "_get_fd_url", lambda self, **k: "http://x")
    res = await tool.execute(action="wait", query="story bible")
    assert "Wait budget spent" in res.content
