"""The chat proxy across an agent restart (Flight Deck restarted it).

The proxy follows a process agent to the port it came back on, holds the
client's socket while the agent isn't answering yet, and logs one line when an
agent goes down and one when it is back — not one per retry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time

import pytest
import uvicorn
import websockets

import captain_claw.flight_deck.server as srv
from captain_claw.flight_deck import agent_ws_quiet


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def deck(monkeypatch):
    registry: dict[str, dict] = {}
    monkeypatch.setattr(srv, "AUTH_ENABLED", False)
    monkeypatch.setattr(srv, "_load_process_registry", lambda: registry)
    monkeypatch.setattr(srv, "_AGENT_WS_WAIT_S", 2.0)
    agent_ws_quiet._down.clear()
    yield registry
    agent_ws_quiet._down.clear()


async def _fake_agent(ws):
    if "token=secret" not in (ws.request.path or ""):
        await ws.close(4001)
        return
    await ws.send(json.dumps({"type": "welcome"}))
    async for m in ws:
        await ws.send(json.dumps({"type": "echo", "got": json.loads(m)}))


async def _recv(c, timeout=6.0) -> dict:
    return json.loads(await asyncio.wait_for(c.recv(), timeout))


def _run(scenario):
    async def main():
        port = _free_port()
        server = uvicorn.Server(uvicorn.Config(srv.app, host="127.0.0.1", port=port,
                                               lifespan="off", log_config=None))
        task = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.05)
        try:
            return await scenario(port)
        finally:
            server.should_exit = True
            await task
    # A loop of its own: asyncio.run() would leave later tests in this package
    # (which call asyncio.get_event_loop()) without a current loop.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(main())
    finally:
        loop.close()


def test_waits_for_the_agent_and_follows_it_to_its_new_port(deck):
    old, new = _free_port(), _free_port()
    deck["helper"] = {"name": "helper", "web_port": old, "web_auth": "secret"}

    async def scenario(fd_port):
        url = f"ws://127.0.0.1:{fd_port}/fd/agent-ws/localhost/{old}?agent=proc-helper"
        async with websockets.connect(url) as c:
            waiting = await _recv(c)
            await c.send(json.dumps({"type": "chat", "content": "sent while waiting"}))
            await asyncio.sleep(0.5)
            # The agent comes back on another port and announces it.
            agent = await websockets.serve(_fake_agent, "127.0.0.1", new)
            deck["helper"]["web_port"] = new
            try:
                frames = [await _recv(c) for _ in range(3)]
            finally:
                agent.close()
                await agent.wait_closed()
        return waiting, frames

    waiting, frames = _run(scenario)
    assert waiting == {"type": "fd_agent_waiting", "port": old}
    assert frames[0] == {"type": "fd_agent_endpoint", "port": new}
    assert frames[1]["type"] == "welcome"
    # What the client sent while the agent was away reached it.
    assert frames[2] == {"type": "echo", "got": {"type": "chat", "content": "sent while waiting"}}


def test_a_stale_port_goes_straight_to_the_agent(deck):
    stale, live = _free_port(), _free_port()
    deck["helper"] = {"name": "helper", "web_port": live, "web_auth": "secret"}

    async def scenario(fd_port):
        agent = await websockets.serve(_fake_agent, "127.0.0.1", live)
        try:
            url = f"ws://127.0.0.1:{fd_port}/fd/agent-ws/localhost/{stale}?agent=proc-helper"
            async with websockets.connect(url) as c:
                return [(await _recv(c))["type"] for _ in range(2)]
        finally:
            agent.close()
            await agent.wait_closed()

    assert _run(scenario) == ["fd_agent_endpoint", "welcome"]


def test_a_ping_is_answered_by_flight_deck_never_the_agent(deck):
    """A chat back from the lock screen checks its socket with fd_ping:
    Flight Deck answers, whether the agent is up or not answering yet, and
    the agent never sees it (it would only echo it, or refuse it)."""
    dead, live = _free_port(), _free_port()
    deck["helper"] = {"name": "helper", "web_port": dead, "web_auth": "secret"}

    async def scenario(fd_port):
        got = []
        url = f"ws://127.0.0.1:{fd_port}/fd/agent-ws/localhost/{dead}?agent=proc-helper"
        async with websockets.connect(url) as c:
            got.append((await _recv(c))["type"])          # fd_agent_waiting
            await c.send(json.dumps({"type": "fd_ping"}))
            got.append(await _recv(c))                    # while it is away
        url = f"ws://127.0.0.1:{fd_port}/fd/agent-ws/localhost/{live}?agent=proc-helper"
        deck["helper"]["web_port"] = live
        agent = await websockets.serve(_fake_agent, "127.0.0.1", live)
        try:
            async with websockets.connect(url) as c:
                got.append((await _recv(c))["type"])      # welcome
                await c.send(json.dumps({"type": "fd_ping"}))
                await c.send(json.dumps({"type": "chat", "content": "hi"}))
                got.append(await _recv(c))
                got.append(await _recv(c))
        finally:
            agent.close()
            await agent.wait_closed()
        return got

    got = _run(scenario)
    assert got[:3] == ["fd_agent_waiting", {"type": "fd_pong"}, "welcome"]
    # The pong first, then the agent's echo of the chat frame — and no echo
    # of the ping: it never reached the agent.
    assert got[3:] == [{"type": "fd_pong"}, {"type": "echo", "got": {"type": "chat", "content": "hi"}}]


def test_an_agent_that_stays_away_is_logged_once(deck, caplog):
    dead = _free_port()
    deck["helper"] = {"name": "helper", "web_port": dead, "web_auth": "secret"}
    caplog.set_level(logging.INFO)

    async def scenario(fd_port):
        url = f"ws://127.0.0.1:{fd_port}/fd/agent-ws/localhost/{dead}?agent=proc-helper"
        got = []
        for _ in range(3):
            async with websockets.connect(url) as c:
                got.append(((await _recv(c))["type"], (await _recv(c)).get("code")))
        return got

    got = _run(scenario)
    assert got == [("fd_agent_waiting", "agent_unreachable")] * 3
    lines = [r.getMessage() for r in caplog.records]
    assert sum("can't reach agent" in m for m in lines) == 1
    assert sum("[accepted]" in m and "/fd/agent-ws/" in m for m in lines) == 1
    assert not any(m in ("connection open", "connection closed") for m in lines)


def test_a_stopped_agent_is_not_waited_for(deck):
    dead = _free_port()
    deck["helper"] = {"name": "helper", "web_port": dead, "web_auth": "secret", "stopped": True}

    async def scenario(fd_port):
        url = f"ws://127.0.0.1:{fd_port}/fd/agent-ws/localhost/{dead}?agent=proc-helper"
        t0 = time.monotonic()
        async with websockets.connect(url) as c:
            frame = await _recv(c)
        return frame, time.monotonic() - t0

    frame, took = _run(scenario)
    assert frame["code"] == "agent_unreachable"
    assert took < 1.0


def test_a_refused_token_is_a_close_code_not_a_failed_handshake(deck, monkeypatch):
    # After a restart without FD_JWT_SECRET the browser's token is signed with
    # the old key. It must learn that (4001) to refresh it — a rejected
    # handshake looks like any network failure and was retried forever.
    monkeypatch.setattr(srv, "AUTH_ENABLED", True)
    deck["helper"] = {"name": "helper", "web_port": _free_port(), "web_auth": "secret"}

    async def scenario(fd_port):
        url = (f"ws://127.0.0.1:{fd_port}/fd/agent-ws/localhost/24001"
               "?fd_token=signed-before-restart&agent=proc-helper")
        async with websockets.connect(url) as c:
            try:
                await asyncio.wait_for(c.recv(), 5)
            except websockets.ConnectionClosed as exc:
                return exc.rcvd.code if exc.rcvd else None

    assert _run(scenario) == 4001


def _accepted(path: str) -> logging.LogRecord:
    return logging.LogRecord("uvicorn.error", logging.INFO, "", 0,
                             '%s - "WebSocket %s" [accepted]', ("127.0.0.1:5000", path), None)


def test_log_filter_mutes_only_agents_known_to_be_down():
    f = agent_ws_quiet.AgentWsLogFilter()
    agent_ws_quiet._down.clear()
    try:
        down = "/fd/agent-ws/localhost/24001?token=…&agent=proc-a"
        up = "/fd/agent-ws/localhost/24002?token=…&agent=proc-b"
        no_id = "/fd/agent-ws/localhost/24003?token=…"
        assert f.filter(_accepted(down))
        assert agent_ws_quiet.mark_down(agent_ws_quiet.target_key("localhost", 24001, "proc-a"))
        assert agent_ws_quiet.mark_down(agent_ws_quiet.target_key("localhost", 24003))
        assert not f.filter(_accepted(down))
        assert f.filter(_accepted(up))
        assert not f.filter(_accepted(no_id))
        assert f.filter(_accepted("/fd/agent-ws-shared?ref=x"))
        assert agent_ws_quiet.mark_up(agent_ws_quiet.target_key("localhost", 24001, "proc-a")) is not None
        assert f.filter(_accepted(down))
        bare = logging.LogRecord("uvicorn.error", logging.INFO, "", 0, "connection open", None, None)
        assert not f.filter(bare)
    finally:
        agent_ws_quiet._down.clear()
