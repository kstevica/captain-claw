"""Keep on course for Basna and Vatra workers (flight_deck/run_course.py).

Flight Deck sends every worker's chat frame the run's goal, the worker's part
and the run's switch and check share; a worker's ``jev_course`` monitor event
comes back as its stage — a ``course`` ledger row (the run map shows it), a
``course`` event in the run's feed, and a flag after two off-track checks.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from captain_claw.flight_deck import auth as auth_mod
from captain_claw.flight_deck import basna_routes as b
from captain_claw.flight_deck import run_course as rc
from captain_claw.flight_deck import vatra_ledger as vl
from captain_claw.flight_deck import vatra_routes as vr
from captain_claw.flight_deck.db import FlightDeckDB

CHECK = {"step": "jev_course", "at_step": 6, "stage": 3.0, "stage_name": "producing",
         "p_on_track": 0.82, "next": "continue", "trigger": "steps", "latency_ms": 90}


@pytest.fixture()
async def db(monkeypatch):
    tmp = tempfile.mkdtemp()
    d = FlightDeckDB(str(Path(tmp) / "fd.db"))
    await d.init()
    u = await d.create_user("t@x.co", "h")
    sess = await d.create_basna_session(u["id"], "research the peers")
    d._test_sid = sess["id"]
    monkeypatch.setattr(auth_mod, "get_db", lambda: d)
    monkeypatch.setattr(vr, "get_db", lambda: d)
    yield d
    await d.close()


async def _drain() -> None:
    while vl._pending:
        await asyncio.gather(*list(vl._pending))


def _in_fresh_context(fn):
    """Run *fn* in a copy of the context, so the run's contextvars don't leak."""
    return contextvars.copy_context().run(fn)


# ── shares, frame, part ──────────────────────────────────────────────


def test_each_worker_gets_a_share_of_the_run_cap(monkeypatch):
    assert rc.worker_share(4, cap=150) == 8
    assert rc.worker_share(40, cap=150) == 3
    assert rc.worker_share(500, cap=150) == 1
    assert rc.worker_share(3, cap=0) == 0
    monkeypatch.setenv("FD_COURSE_MAX_CHECKS_PER_RUN", "20")
    assert rc.worker_share(5) == 4


def test_frame_carries_goal_switch_share_and_the_part():
    def run():
        assert rc.frame() is None
        rc.start_run("Compare   flights\nto Lisbon", check=True, team_size=4)
        assert rc.frame() == {"goal": "Compare flights to Lisbon", "check": True, "max_checks": 8}
        token = rc.set_part(rc.subtask_part({"title": "Prices", "description": "collect fares"}))
        assert rc.frame()["part"] == "Prices: collect fares"
        rc.reset_part(token)
        assert "part" not in rc.frame()
    _in_fresh_context(run)


def test_subtask_part_falls_back_to_what_there_is():
    assert rc.subtask_part({"title": "Prices"}) == "Prices"
    assert rc.subtask_part({"desc": "collect fares"}) == "collect fares"
    assert rc.subtask_part({}) == ""


# ── the switch ───────────────────────────────────────────────────────


def test_switch_follows_the_starting_agent_then_the_session_then_on(monkeypatch):
    monkeypatch.setattr(rc, "caller_switch", lambda request: None)
    assert rc.resolve_switch(None, {}) is True
    assert rc.resolve_switch(None, {"course_check": False}) is False
    monkeypatch.setattr(rc, "caller_switch", lambda request: True)
    assert rc.resolve_switch(None, {"course_check": False}) is True


def test_caller_switch_reads_the_calling_agents_switch(tmp_path, monkeypatch):
    from captain_claw.flight_deck import server

    agent_dir = tmp_path / "bob-agent"
    agent_dir.mkdir()
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "_find_agent_by_auth",
                        lambda tok: (True, "bob", "bob-agent") if tok == "tok" else (False, "", ""))
    monkeypatch.setattr(server, "_load_process_registry", lambda: {"bob-agent": {}})
    seen = {}

    def read(path, kind):
        seen["args"] = (path, kind)
        return False

    monkeypatch.setattr(server, "_read_agent_course_check", read)
    req = SimpleNamespace(headers={"X-Agent-Auth": "tok"})
    assert rc.caller_switch(req) is False
    assert seen["args"] == (agent_dir, "process")
    assert rc.caller_switch(SimpleNamespace(headers={})) is None


# ── answers coming back ──────────────────────────────────────────────


def test_course_from_monitor_reads_only_course_checks():
    msg = {"type": "monitor", "tool_name": "completion_gate", "arguments": CHECK}
    assert rc.course_from_monitor(msg) == {"stage": 3.0, "stage_name": "producing", "p_on_track": 0.82,
                                           "next": "continue", "at_step": 6}
    assert rc.course_from_monitor({"tool_name": "completion_gate", "arguments": {"step": "iteration_budget"}}) is None
    assert rc.course_from_monitor({"tool_name": "web_fetch", "arguments": CHECK}) is None


def test_describe():
    assert rc.describe({"stage": 3.2, "stage_name": "producing", "p_on_track": 0.82, "next": "continue"}) == \
        "stage 3 of 5 · producing · on track 0.82 · next: continue"
    assert rc.describe({}) == "course checked"


def test_off_track_flags_once_per_streak():
    rc.forget_run("s")
    low = {"p_on_track": 0.1}
    assert rc.off_track("s", "a", low) is False
    assert rc.off_track("s", "a", low) is True
    assert rc.off_track("s", "a", low) is False     # once per streak
    assert rc.off_track("s", "a", {"p_on_track": 0.9}) is False
    assert rc.off_track("s", "a", low) is False
    assert rc.off_track("s", "a", low) is True
    rc.forget_run("s")
    assert ("s", "a") not in rc._streaks


async def test_the_frame_rides_on_the_chat_and_the_check_comes_back():
    from websockets.asyncio.server import serve

    frames: list[dict] = []

    async def handler(ws):
        await ws.send(json.dumps({"type": "welcome"}))
        await ws.send(json.dumps({"type": "replay_done"}))
        async for raw in ws:
            frame = json.loads(raw)
            if frame.get("type") != "chat":
                continue
            frames.append(frame)
            await ws.send(json.dumps({"type": "monitor", "tool_name": "completion_gate", "arguments": CHECK}))
            await ws.send(json.dumps({"type": "chat_message", "role": "assistant", "content": "done"}))
            await ws.send(json.dumps({"type": "status", "status": "idle"}))

    seen: list[dict] = []

    async def go():
        rc.start_run("Compare flights to Lisbon", check=False, team_size=2)
        rc.set_part("Prices: collect fares")
        async with serve(handler, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            return await b._send_chat_and_collect(port, "", "go", 5.0, on_action=seen.append)

    out, actions = await asyncio.create_task(go(), context=contextvars.copy_context())
    assert out == "done"
    assert frames[0]["course"] == {"goal": "Compare flights to Lisbon", "check": False, "max_checks": 8,
                                   "part": "Prices: collect fares"}
    assert seen[0]["tool"] == "completion_gate"
    assert seen[0]["args"]["course"]["stage_name"] == "producing"
    assert "args" not in actions[0]


async def test_outside_a_run_no_frame_is_sent():
    from websockets.asyncio.server import serve

    frames: list[dict] = []

    async def handler(ws):
        await ws.send(json.dumps({"type": "welcome"}))
        await ws.send(json.dumps({"type": "replay_done"}))
        async for raw in ws:
            frame = json.loads(raw)
            if frame.get("type") == "chat":
                frames.append(frame)
                await ws.send(json.dumps({"type": "chat_message", "role": "assistant", "content": "ok"}))
                await ws.send(json.dumps({"type": "status", "status": "idle"}))

    async def go():
        async with serve(handler, "127.0.0.1", 0) as srv:
            return await b._send_chat_and_collect(srv.sockets[0].getsockname()[1], "", "go", 5.0)

    await asyncio.create_task(go(), context=contextvars.Context())
    assert "course" not in frames[0]


# ── ledger, map, feed ────────────────────────────────────────────────


async def test_a_course_check_becomes_the_owners_course_row_and_shows_on_the_map(db):
    sid = db._test_sid
    course = rc.course_from_monitor({"tool_name": "completion_gate", "arguments": CHECK})
    vl.record_action(sid, "researcher", "s1", "Researcher",
                     {"tool": "completion_gate", "detail": "", "args": {"course": course}})
    vl.record_action(sid, "researcher", "s1", "Researcher",
                     {"tool": "completion_gate", "detail": "",
                      "args": {"course": {**course, "stage": 4.0, "stage_name": "checking"}}})
    await _drain()
    rows = [r for r in await db.list_vatra_ledger(sid) if r["kind"] == "course"]
    assert len(rows) == 1 and rows[0]["key"] == "s1"
    assert rows[0]["detail"].startswith("stage 4 of 5 · checking")
    text = await vl.render_map(sid, "s2", {"s1": {"role": "Researcher", "state": "running"}})
    assert "## Researcher · running (piece s1) · stage 4 of 5 · checking" in text


def test_the_feed_gets_the_stage_and_an_off_track_flag(monkeypatch):
    events: list[tuple] = []
    monkeypatch.setattr(b, "_progress", lambda sid, stage, msg, **extra: events.append((stage, msg, extra)))
    rc.forget_run("run1")
    low = {"stage": 2.0, "stage_name": "gathering", "p_on_track": 0.1, "next": "refocus"}
    b._progress_course("run1", "Researcher", "s1", low, group="A")
    b._progress_course("run1", "Researcher", "s1", low, group="A")
    stages = [e[0] for e in events]
    assert stages == ["course", "course", "course_off_track"]
    stage, msg, extra = events[0]
    assert msg == "Researcher · stage 2 of 5 · gathering · on track 0.10 · next: refocus"
    assert extra["course_stage"] == 2.0 and extra["p_on_track"] == 0.1 and extra["group"] == "A"
    assert events[2][2]["ok"] is False
