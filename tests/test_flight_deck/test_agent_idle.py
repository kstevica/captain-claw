"""Flight Deck's agent controller (agent_idle): idle process agents go to sleep,
wake 5 minutes before something scheduled for them, and wake when something
needs them — a chat held by the proxy wakes on the client's first word."""

from __future__ import annotations

import asyncio
import json
import socket
import sqlite3
import time
from datetime import UTC, datetime

import pytest
import uvicorn
import websockets
from fastapi import HTTPException

import captain_claw.flight_deck.server as srv
from captain_claw.flight_deck import agent_idle, agent_ws_quiet, glasses_bridge


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _loop_run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def deck(monkeypatch, tmp_path):
    """A registry in memory, agents "alive" by name, nothing of the real deck."""
    registry: dict[str, dict] = {}
    alive: set[str] = set()
    monkeypatch.setattr(srv, "AUTH_ENABLED", False)
    monkeypatch.setattr(srv, "DATA_DIR", tmp_path)
    monkeypatch.setattr(srv, "_load_process_registry", lambda: registry)
    monkeypatch.setattr(srv, "_save_process_registry", lambda r: None)
    monkeypatch.setattr(srv, "_process_is_alive", lambda slug: slug in alive)
    monkeypatch.setattr(srv, "_AGENT_WS_WAIT_S", 2.0)
    monkeypatch.setattr(agent_idle, "_scheduler_next_runs", lambda: {})
    monkeypatch.setenv("FD_AGENT_IDLE_STOP_S", "600")
    agent_idle._last_active.clear()
    agent_idle._port_touched.clear()
    agent_idle._own_bot_cache.clear()
    agent_ws_quiet._down.clear()
    yield registry, alive
    agent_idle._last_active.clear()
    agent_ws_quiet._down.clear()


@pytest.fixture
def actions(monkeypatch):
    """put_to_sleep / wake record their calls instead of acting."""
    done: dict[str, list[str]] = {"slept": [], "woke": []}

    async def fake_sleep(slug, idle_s=0.0):
        done["slept"].append(slug)
        return True

    async def fake_wake(slug, timeout=0):
        done["woke"].append(slug)
        return True

    async def no_activity(entry):
        return None

    monkeypatch.setattr(agent_idle, "put_to_sleep", fake_sleep)
    monkeypatch.setattr(agent_idle, "wake", fake_wake)
    monkeypatch.setattr(agent_idle, "_agent_activity", no_activity)
    return done


def _idle(slug: str, seconds: float = 3600) -> None:
    agent_idle._last_active[slug] = time.time() - seconds


# ── The tick ──────────────────────────────────────────────────────────


def test_an_idle_agent_is_put_to_sleep_and_an_active_one_is_not(deck, actions):
    registry, alive = deck
    registry["idle"] = {"name": "idle", "web_port": 24101}
    registry["busy"] = {"name": "busy", "web_port": 24102}
    alive.update({"idle", "busy"})
    _idle("idle")
    agent_idle.touch("busy")

    _loop_run(agent_idle.tick())
    assert actions["slept"] == ["idle"]


def test_off_by_default_and_turning_it_off_wakes_who_sleeps(deck, actions, monkeypatch):
    registry, alive = deck
    monkeypatch.setenv("FD_AGENT_IDLE_STOP_S", "0")
    registry["idle"] = {"name": "idle", "web_port": 24101}
    registry["napping"] = {"name": "napping", "web_port": 24102, "asleep": True}
    alive.add("idle")
    _idle("idle")

    _loop_run(agent_idle.tick())
    assert actions["slept"] == []
    assert actions["woke"] == ["napping"]


def test_an_agent_mid_turn_stays_up(deck, actions, monkeypatch):
    registry, alive = deck
    registry["worker"] = {"name": "worker", "web_port": 24101}
    alive.add("worker")
    _idle("worker")

    async def busy(entry):
        return {"busy": True, "last_active_at": time.time() - 3600}

    monkeypatch.setattr(agent_idle, "_agent_activity", busy)
    _loop_run(agent_idle.tick())
    assert actions["slept"] == []
    assert time.time() - agent_idle.last_active("worker") < 5


def test_the_agents_own_last_activity_counts(deck, actions, monkeypatch):
    registry, alive = deck
    registry["helper"] = {"name": "helper", "web_port": 24101}
    alive.add("helper")
    _idle("helper")

    async def recent(entry):  # its own cron ran a minute ago
        return {"busy": False, "last_active_at": time.time() - 60}

    monkeypatch.setattr(agent_idle, "_agent_activity", recent)
    _loop_run(agent_idle.tick())
    assert actions["slept"] == []


def test_a_scheduled_task_due_soon_keeps_it_up_and_wakes_it(deck, actions, monkeypatch):
    registry, alive = deck
    registry["up"] = {"name": "up", "web_port": 24101}
    registry["asleep"] = {"name": "asleep", "web_port": 24102, "asleep": True}
    registry["later"] = {"name": "later", "web_port": 24103, "asleep": True}
    alive.add("up")
    _idle("up")
    now = time.time()
    monkeypatch.setattr(agent_idle, "_scheduler_next_runs", lambda: {
        "up": now + 200, "asleep": now + 240, "later": now + 3 * 3600})

    _loop_run(agent_idle.tick())
    assert actions["slept"] == []
    assert actions["woke"] == ["asleep"]


def test_the_agents_own_cron_wakes_it(deck, actions, tmp_path):
    registry, _alive = deck
    registry["cronny"] = {"name": "cronny", "web_port": 24101, "asleep": True}
    db = tmp_path / "cronny" / "data" / "sessions" / "sessions.db"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE cron_jobs (id TEXT, next_run_at TEXT, enabled INTEGER)")
    soon = datetime.fromtimestamp(time.time() + 120, UTC).isoformat()
    later = datetime.fromtimestamp(time.time() + 7200, UTC).isoformat()
    conn.executemany("INSERT INTO cron_jobs VALUES (?, ?, ?)",
                     [("a", later, 1), ("b", soon, 0), ("c", soon, 1)])
    conn.commit()
    conn.close()

    assert abs(agent_idle._agent_cron_next("cronny") - (time.time() + 120)) < 5
    _loop_run(agent_idle.tick())
    assert actions["woke"] == ["cronny"]


def test_workers_and_agents_with_their_own_bot_never_sleep(deck, actions, tmp_path):
    registry, alive = deck
    registry["vatra-0123abcd-writer"] = {"name": "w", "web_port": 24101}
    registry["botty"] = {"name": "botty", "web_port": 24102}
    registry["relayed"] = {"name": "relayed", "web_port": 24103}
    for slug in registry:
        alive.add(slug)
        _idle(slug)
    (tmp_path / "botty").mkdir()
    (tmp_path / "botty" / "config.yaml").write_text(
        "telegram:\n  enabled: true\n  bot_token: '123:abc'\n")
    # The team bot comes through Flight Deck's relay, which wakes it.
    (tmp_path / "relayed").mkdir()
    (tmp_path / "relayed" / ".env").write_text(
        "CLAW_TELEGRAM__ENABLED=true\nCLAW_TELEGRAM__BOT_TOKEN=x\n"
        "CLAW_TELEGRAM__API_BASE_URL=http://localhost:25080/fd/telegram-relay\n")
    # The bare variable the agent's config also reads.
    registry["discordy"] = {"name": "discordy", "web_port": 24104}
    alive.add("discordy")
    _idle("discordy")
    (tmp_path / "discordy").mkdir()
    (tmp_path / "discordy" / "config.yaml").write_text("discord:\n  enabled: true\n")
    (tmp_path / "discordy" / ".env").write_text("DISCORD_BOT_TOKEN=abc\n")

    _loop_run(agent_idle.tick())
    assert actions["slept"] == ["relayed"]


def test_stopped_by_its_owner_is_left_alone(deck, actions):
    registry, _alive = deck
    registry["off"] = {"name": "off", "web_port": 24101, "stopped": True, "asleep": True}
    _loop_run(agent_idle.tick())
    assert actions == {"slept": [], "woke": []}


def test_a_scheduled_run_in_flight_keeps_it_up(deck, actions):
    from captain_claw.flight_deck import fd_scheduler

    registry, alive = deck
    registry["sched"] = {"name": "sched", "web_port": 24101}
    alive.add("sched")
    _idle("sched")

    async def scenario():
        lock = fd_scheduler._AGENT_LOCKS.setdefault("sched", asyncio.Lock())
        async with lock:
            await agent_idle.tick()

    try:
        _loop_run(scenario())
    finally:
        fd_scheduler._AGENT_LOCKS.pop("sched", None)
    assert actions["slept"] == []


# ── Sleeping and waking the process ───────────────────────────────────


def test_sleep_marks_it_asleep_before_the_kill(deck, monkeypatch):
    registry, alive = deck
    registry["helper"] = {"name": "helper", "web_port": 24101, "pid": 4242}
    alive.add("helper")
    seen = []

    def kill(pid, timeout=5.0):
        seen.append((pid, dict(registry["helper"])))
        alive.discard("helper")

    monkeypatch.setattr(srv, "_kill_pid", kill)
    assert agent_idle._sleep_process("helper")
    assert seen[0][0] == 4242 and seen[0][1]["asleep"] is True
    assert registry["helper"]["pid"] is None and registry["helper"]["asleep"] is True
    assert agent_idle.is_asleep("helper")


def test_start_wakes_and_stop_switches_off(deck, monkeypatch, tmp_path):
    registry, alive = deck
    registry["helper"] = {"name": "helper", "web_port": 24101, "asleep": True, "asleep_at": 1.0}
    (tmp_path / "helper").mkdir()

    def start(slug, entry):
        entry["pid"] = 777
        alive.add(slug)
        return True

    monkeypatch.setattr(srv, "_start_registered_process", start)
    srv._do_start_process("helper")
    assert "asleep" not in registry["helper"] and registry["helper"]["pid"] == 777

    # Powered off while asleep: off for good, nothing wakes it.
    alive.discard("helper")
    registry["helper"]["asleep"] = True
    srv._do_stop_process("helper")
    assert registry["helper"]["stopped"] is True and "asleep" not in registry["helper"]


def test_a_restart_leaves_sleeping_agents_asleep(deck, monkeypatch):
    registry, _alive = deck
    registry["napping"] = {"name": "napping", "web_port": 24101, "asleep": True}
    started = []
    monkeypatch.setattr(srv, "_start_registered_process", lambda slug, e: started.append(slug) or True)
    srv._reattach_processes()
    assert started == []


def _http_agent(token: str):
    """A stand-in agent web server: /api/activity answers its own token only."""
    async def handle(reader, writer):
        head = (await reader.readuntil(b"\r\n\r\n")).decode()
        ok = f"token={token}" in head.split("\r\n", 1)[0]
        body = b'{"busy": false, "last_active_at": 0}' if ok else b'{"error": "unauthorized"}'
        status = b"200 OK" if ok else b"401 Unauthorized"
        writer.write(b"HTTP/1.1 " + status + b"\r\nContent-Type: application/json\r\nContent-Length: "
                     + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
        await writer.drain()
        writer.close()
    return handle


def test_wake_starts_it_once_and_waits_for_it_to_answer(deck, monkeypatch, tmp_path):
    registry, alive = deck
    port = _free_port()
    registry["helper"] = {"name": "helper", "web_port": port, "web_auth": "mine", "asleep": True}
    (tmp_path / "helper").mkdir()
    starts = []

    def start(slug, entry):
        starts.append(slug)
        entry["pid"] = 1
        alive.add(slug)
        return True

    monkeypatch.setattr(srv, "_start_registered_process", start)

    async def scenario():
        server = await asyncio.start_server(_http_agent("mine"), "127.0.0.1", port)
        async with server:
            return await asyncio.gather(agent_idle.wake("helper"), agent_idle.wake("helper"))

    assert _loop_run(scenario()) == [True, True]
    assert starts == ["helper"]
    assert _loop_run(agent_idle.wake("helper")) is False  # not asleep any more


def test_another_agent_on_its_port_is_not_taken_for_it(deck, monkeypatch, tmp_path):
    # Woken while its port was still held, it comes back on the next free one
    # and announces it; whoever answers on the old port with a 401 is not it.
    registry, alive = deck
    old, new = _free_port(), _free_port()
    registry["helper"] = {"name": "helper", "web_port": old, "web_auth": "mine", "asleep": True}
    (tmp_path / "helper").mkdir()

    def start(slug, entry):
        alive.add(slug)
        return True

    monkeypatch.setattr(srv, "_start_registered_process", start)

    async def scenario():
        stranger = await asyncio.start_server(_http_agent("theirs"), "127.0.0.1", old)
        async with stranger:
            task = asyncio.ensure_future(agent_idle.wake("helper", timeout=5))
            await asyncio.sleep(1.5)
            early = task.done()
            ours = await asyncio.start_server(_http_agent("mine"), "127.0.0.1", new)
            async with ours:
                registry["helper"]["web_port"] = new  # the announce
                return early, await task

    early, woke = _loop_run(scenario())
    assert early is False and woke is True


def test_a_running_agent_owns_a_shared_port(deck):
    registry, alive = deck
    registry["napping"] = {"name": "napping", "web_port": 24600, "asleep": True}
    registry["moved-in"] = {"name": "moved-in", "web_port": 24600}
    alive.add("moved-in")
    assert agent_idle.slug_for_port(24600) == "moved-in"
    alive.discard("moved-in")
    assert agent_idle.slug_for_port(24600) == "napping"


def test_sleep_is_called_off_when_something_reached_it(deck, monkeypatch):
    registry, alive = deck
    registry["helper"] = {"name": "helper", "web_port": 24101, "pid": 1}
    alive.add("helper")
    monkeypatch.setattr(srv, "_kill_pid", lambda pid, timeout=5.0: alive.discard("helper"))
    agent_idle.touch("helper")  # a message just now
    assert _loop_run(agent_idle.put_to_sleep("helper", 600)) is False
    assert "asleep" not in registry["helper"] and "helper" in alive
    _idle("helper")
    assert _loop_run(agent_idle.put_to_sleep("helper", 600)) is True


def test_peers_and_flows_see_a_sleeping_agent_as_reachable(deck):
    from captain_claw.flight_deck.flow_runner import FlowRunner

    registry, alive = deck
    registry["up"] = {"name": "up", "web_port": 24101}
    registry["napping"] = {"name": "napping", "web_port": 24102, "asleep": True}
    registry["off"] = {"name": "off", "web_port": 24103, "stopped": True}
    alive.add("up")
    status = {a["slug"]: a["status"] for a in srv._fleet_snapshot() if a["kind"] == "process"}
    assert status == {"up": "running", "napping": "asleep", "off": "stopped"}
    runner = FlowRunner.__new__(FlowRunner)
    runner.get_agents = lambda: list(reversed(srv._running_agents()))
    assert [a["name"] for a in runner._pool()] == ["up", "napping"]


def test_a_new_agent_does_not_take_a_sleeping_agents_port(deck):
    registry, _alive = deck
    registry["napping"] = {"name": "napping", "web_port": 24555, "asleep": True}
    assert srv._port_reserved(24555, "newcomer")
    assert not srv._port_reserved(24555, "napping")
    assert not srv._port_reserved(24556, "newcomer")


def test_the_bridges_still_find_a_sleeping_agent(deck, monkeypatch):
    from captain_claw.flight_deck.meta_webhook_bridge import resolve_agent_target

    registry, alive = deck
    registry["napping"] = {"name": "napping", "web_port": 24555, "asleep": True}
    registry["other"] = {"name": "other", "web_port": 24556}
    alive.add("other")
    monkeypatch.setenv("TEST_AGENT_SLUG", "napping")
    assert resolve_agent_target(slug_env="TEST_AGENT_SLUG", port_env="TEST_AGENT_PORT")[1] == 24555
    monkeypatch.delenv("TEST_AGENT_SLUG")
    # No slug set: the first agent stays the pick while it sleeps.
    assert resolve_agent_target(slug_env="TEST_AGENT_SLUG", port_env="TEST_AGENT_PORT")[1] == 24555


def test_the_admin_setting_is_bounded():
    from captain_claw.flight_deck import admin_routes

    assert admin_routes.SYSTEM_CONFIG_DEFAULTS["agent_idle_stop_minutes"] == 0

    class _DB:
        stored = None

        async def get_system_setting(self, key):
            return self.stored

        async def set_system_setting(self, key, value):
            self.stored = value

    db = _DB()
    orig = admin_routes.get_db
    admin_routes.get_db = lambda: db
    try:
        req = admin_routes.UpdateConfigRequest
        for bad in (5, 7 * 24 * 60 + 1, -1):
            with pytest.raises(HTTPException):
                _loop_run(admin_routes.update_config(req(agent_idle_stop_minutes=bad), admin={}))
        out = _loop_run(admin_routes.update_config(req(agent_idle_stop_minutes=30), admin={}))
        assert out["agent_idle_stop_minutes"] == 30
        assert _loop_run(admin_routes.update_config(req(agent_idle_stop_minutes=0), admin={}))[
            "agent_idle_stop_minutes"] == 0
    finally:
        admin_routes.get_db = orig


# ── Waking on demand: the chat proxy and the channel bus ─────────────


async def _fake_agent(ws):
    await ws.send(json.dumps({"type": "welcome"}))
    async for m in ws:
        await ws.send(json.dumps({"type": "echo", "got": json.loads(m)}))


async def _recv(c, timeout=6.0) -> dict:
    return json.loads(await asyncio.wait_for(c.recv(), timeout))


def _with_deck(scenario):
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
    return _loop_run(main())


def _waking_agent(deck, monkeypatch, port):
    """``agent_idle.wake`` brings a fake agent up on ``port``; returns its calls."""
    registry, alive = deck
    calls: list[str] = []
    servers = []

    async def wake(slug, timeout=0):
        if not agent_idle.is_asleep(slug):
            return False
        calls.append(slug)
        servers.append(await websockets.serve(_fake_agent, "127.0.0.1", port))
        registry[slug].pop("asleep", None)
        alive.add(slug)
        return True

    monkeypatch.setattr(agent_idle, "wake", wake)
    return calls, servers


def test_a_chat_holds_a_sleeping_agent_until_the_client_speaks(deck, monkeypatch):
    registry, _alive = deck
    port = _free_port()
    registry["helper"] = {"name": "helper", "web_port": port, "web_auth": "secret", "asleep": True}
    calls, servers = _waking_agent(deck, monkeypatch, port)

    async def scenario(fd_port):
        url = f"ws://127.0.0.1:{fd_port}/fd/agent-ws/localhost/{port}?agent=proc-helper"
        try:
            async with websockets.connect(url) as c:
                asleep = await _recv(c)
                await asyncio.sleep(0.5)
                woken_early = list(calls)  # a reconnect alone doesn't wake it
                await c.send(json.dumps({"type": "chat", "content": "good morning"}))
                frames = [await _recv(c) for _ in range(2)]
        finally:
            for s in servers:
                s.close()
                await s.wait_closed()
        return asleep, woken_early, frames

    asleep, woken_early, frames = _with_deck(scenario)
    assert asleep == {"type": "fd_agent_asleep"}
    assert woken_early == []
    assert calls == ["helper"]
    assert frames[0]["type"] == "welcome"
    assert frames[1] == {"type": "echo", "got": {"type": "chat", "content": "good morning"}}


def test_fd_wake_wakes_it_and_never_reaches_the_agent(deck, monkeypatch):
    registry, _alive = deck
    port = _free_port()
    registry["helper"] = {"name": "helper", "web_port": port, "web_auth": "secret", "asleep": True}
    calls, servers = _waking_agent(deck, monkeypatch, port)

    async def scenario(fd_port):
        url = f"ws://127.0.0.1:{fd_port}/fd/agent-ws/localhost/{port}?agent=proc-helper"
        try:
            async with websockets.connect(url) as c:
                await _recv(c)  # fd_agent_asleep
                await c.send(json.dumps({"type": "fd_wake"}))
                welcome = await _recv(c)
                await c.send(json.dumps({"type": "fd_wake"}))  # late: dropped too
                await c.send(json.dumps({"type": "chat", "content": "hi"}))
                echo = await _recv(c)
        finally:
            for s in servers:
                s.close()
                await s.wait_closed()
        return welcome, echo

    welcome, echo = _with_deck(scenario)
    assert welcome["type"] == "welcome"
    assert echo == {"type": "echo", "got": {"type": "chat", "content": "hi"}}
    assert calls == ["helper"]


def test_a_word_held_before_it_fell_asleep_wakes_it(deck, monkeypatch):
    # The client wrote while the proxy was still trying the dying agent; once
    # the agent reads as asleep, that word is the wake-up — not a second one.
    registry, alive = deck
    port = _free_port()
    registry["helper"] = {"name": "helper", "web_port": port, "web_auth": "secret", "pid": 9}
    alive.add("helper")  # still going down
    calls, servers = _waking_agent(deck, monkeypatch, port)

    async def scenario(fd_port):
        url = f"ws://127.0.0.1:{fd_port}/fd/agent-ws/localhost/{port}?agent=proc-helper"
        try:
            async with websockets.connect(url) as c:
                assert (await _recv(c))["type"] == "fd_agent_waiting"
                await c.send(json.dumps({"type": "chat", "content": "still there?"}))
                await asyncio.sleep(0.3)
                registry["helper"]["asleep"] = True
                alive.discard("helper")
                return [await _recv(c) for _ in range(2)]
        finally:
            for s in servers:
                s.close()
                await s.wait_closed()

    frames = _with_deck(scenario)
    assert calls == ["helper"]
    assert frames[0]["type"] == "welcome"
    assert frames[1] == {"type": "echo", "got": {"type": "chat", "content": "still there?"}}


def test_a_channel_send_wakes_a_sleeping_agent(deck, monkeypatch):
    registry, _alive = deck
    port = _free_port()
    registry["helper"] = {"name": "helper", "web_port": port, "asleep": True}
    calls, servers = _waking_agent(deck, monkeypatch, port)
    monkeypatch.setattr(srv, "_resolve_agent_auth", lambda port, host=None: "")

    async def scenario():
        ch = glasses_bridge._ChannelState(channel_id="t-wake")
        try:
            await glasses_bridge._ensure_agent_binding(ch, "localhost", port)
            await asyncio.sleep(0.3)  # the pump fails and backs off: asleep
            linked = await glasses_bridge.await_agent_link(ch)
            return linked
        finally:
            if ch.agent_ws_task:
                ch.bound_port = None
                ch.agent_ws_task.cancel()
            for s in servers:
                s.close()
                await s.wait_closed()

    assert _loop_run(scenario()) is True
    assert calls == ["helper"]


# ── On for a scheduled task ───────────────────────────────────────────


def _starter(monkeypatch, alive, starts):
    def start(slug, entry):
        starts.append(slug)
        entry["pid"] = 1
        alive.add(slug)
        return True

    monkeypatch.setattr(srv, "_start_registered_process", start)


def test_a_task_starts_an_agent_its_owner_turned_off_and_turns_it_off_after(deck, monkeypatch, tmp_path):
    registry, alive = deck
    port = _free_port()
    registry["glasses"] = {"name": "glasses", "web_port": port, "web_auth": "mine", "stopped": True}
    (tmp_path / "glasses").mkdir()
    starts: list[str] = []
    _starter(monkeypatch, alive, starts)
    kills: list[int] = []

    def kill(pid, timeout=5.0):
        kills.append(pid)
        alive.discard("glasses")

    monkeypatch.setattr(srv, "_kill_pid", kill)

    async def scenario():
        server = await asyncio.start_server(_http_agent("mine"), "127.0.0.1", port)
        async with server:
            up = await agent_idle.ready_for_task("glasses")
            during = dict(registry["glasses"])
            off = await agent_idle.release_after_task("glasses")
            return up, during, off

    up, during, off = _loop_run(scenario())
    assert up and starts == ["glasses"]
    assert during.get("on_for_task") and not during.get("stopped")
    assert off and kills == [1]
    # Off again, as its owner left it: nothing but the next task starts it.
    assert registry["glasses"]["stopped"] is True and "on_for_task" not in registry["glasses"]


def test_a_crashed_agent_is_started_for_its_task_and_stays_on(deck, monkeypatch, tmp_path):
    registry, alive = deck
    port = _free_port()
    registry["helper"] = {"name": "helper", "web_port": port, "web_auth": "mine", "pid": 99}
    (tmp_path / "helper").mkdir()
    starts: list[str] = []
    _starter(monkeypatch, alive, starts)

    async def scenario():
        server = await asyncio.start_server(_http_agent("mine"), "127.0.0.1", port)
        async with server:
            return await agent_idle.ready_for_task("helper"), await agent_idle.release_after_task("helper")

    assert _loop_run(scenario()) == (True, False)
    assert starts == ["helper"] and "on_for_task" not in registry["helper"]


def test_a_running_agent_is_ready_as_it_is(deck, monkeypatch):
    registry, alive = deck
    port = _free_port()
    registry["up"] = {"name": "up", "web_port": port, "web_auth": "mine"}
    alive.add("up")
    starts: list[str] = []
    _starter(monkeypatch, alive, starts)

    async def scenario():
        server = await asyncio.start_server(_http_agent("mine"), "127.0.0.1", port)
        async with server:
            return await agent_idle.ready_for_task("up")

    assert _loop_run(scenario()) is True and starts == []


def test_it_stays_on_while_needed_or_once_its_owner_turns_it_on(deck, actions, monkeypatch):
    from captain_claw.flight_deck import fd_scheduler

    registry, alive = deck
    registry["glasses"] = {"name": "glasses", "web_port": 24101, "on_for_task": 1.0}
    alive.add("glasses")
    turned_off: list[str] = []
    monkeypatch.setattr(srv, "_do_stop_process", lambda slug: turned_off.append(slug))

    # Another of its tasks is due in two minutes.
    now = time.time()
    monkeypatch.setattr(agent_idle, "_scheduler_next_runs", lambda: {"glasses": now + 120})
    assert not _loop_run(agent_idle.release_after_task("glasses"))
    monkeypatch.setattr(agent_idle, "_scheduler_next_runs", lambda: {})

    # Another run has it, or is waiting for it.
    async def held():
        lock = fd_scheduler._AGENT_LOCKS.setdefault("glasses", asyncio.Lock())
        async with lock:
            return await agent_idle.release_after_task("glasses")

    try:
        assert not _loop_run(held())
    finally:
        fd_scheduler._AGENT_LOCKS.pop("glasses", None)

    # Someone is in a turn with it.
    async def busy(entry):
        return {"busy": True, "last_active_at": time.time()}

    monkeypatch.setattr(agent_idle, "_agent_activity", busy)
    assert not _loop_run(agent_idle.release_after_task("glasses"))

    async def idle(entry):
        return None

    monkeypatch.setattr(agent_idle, "_agent_activity", idle)
    # Its owner turned it on meanwhile: on it stays.
    srv._do_start_process("glasses")
    assert "on_for_task" not in registry["glasses"]
    assert not _loop_run(agent_idle.release_after_task("glasses"))
    assert turned_off == []


def test_powering_it_off_during_the_task_is_off_for_good(deck, monkeypatch):
    registry, alive = deck
    registry["glasses"] = {"name": "glasses", "web_port": 24101, "on_for_task": 1.0, "pid": 5}
    alive.add("glasses")
    monkeypatch.setattr(srv, "_kill_pid", lambda pid, timeout=5.0: alive.discard("glasses"))
    srv._do_stop_process("glasses")
    assert registry["glasses"]["stopped"] is True and "on_for_task" not in registry["glasses"]
    assert not agent_idle.is_on_for_task("glasses")


def test_the_tick_starts_an_off_agent_before_its_task_and_turns_it_off_after(deck, actions, monkeypatch):
    registry, alive = deck
    now = time.time()
    registry["soon"] = {"name": "soon", "web_port": 24101, "stopped": True}
    registry["later"] = {"name": "later", "web_port": 24102, "stopped": True}
    registry["cron-only"] = {"name": "cron-only", "web_port": 24103, "stopped": True}
    registry["done"] = {"name": "done", "web_port": 24104, "on_for_task": now - 900}
    registry["fresh"] = {"name": "fresh", "web_port": 24105, "on_for_task": now - 30}
    registry["crashed"] = {"name": "crashed", "web_port": 24106}
    alive.update({"done", "fresh"})
    _idle("done", 900)
    agent_idle.touch("fresh")
    monkeypatch.setattr(agent_idle, "_scheduler_next_runs", lambda: {
        "soon": now + 200, "later": now + 3 * 3600, "crashed": now + 60})
    monkeypatch.setattr(agent_idle, "_agent_cron_next", lambda slug: now + 60 if slug == "cron-only" else None)
    readied: list[str] = []
    released: list[str] = []

    async def ready(slug, timeout=0):
        readied.append(slug)
        return True

    monkeypatch.setattr(agent_idle, "ready_for_task", ready)
    monkeypatch.setattr(agent_idle, "_turn_off_after_task", lambda slug: released.append(slug) or True)

    done = _loop_run(agent_idle.tick(now))
    # Off by its owner: only a scheduled task of Flight Deck's turns it on,
    # not the agent's own cron.
    assert readied == ["soon", "crashed"] and done["started"] == ["soon", "crashed"]
    assert released == ["done"] and done["off"] == ["done"]
    # One a task has on is turned off, never put to sleep.
    assert actions["slept"] == [] and actions["woke"] == []


def test_a_restart_turns_off_what_a_task_had_on(deck, monkeypatch):
    registry, _alive = deck
    registry["glasses"] = {"name": "glasses", "web_port": 24101, "on_for_task": 1.0, "pid": None}
    started: list[str] = []
    monkeypatch.setattr(srv, "_start_registered_process", lambda slug, e: started.append(slug) or True)
    srv._reattach_processes()
    assert started == []
    assert registry["glasses"]["stopped"] is True and "on_for_task" not in registry["glasses"]


def test_a_scheduled_run_readies_its_agent_and_lets_it_go_after(deck, monkeypatch, tmp_path):
    from captain_claw.flight_deck import fd_scheduler, usage_quota

    registry, alive = deck
    registry["glasses"] = {"name": "glasses", "web_port": 24101, "web_auth": "t"}
    calls: list[tuple] = []

    async def ready(slug, timeout=0):
        calls.append(("ready", slug))
        alive.add(slug)
        return True

    def release(slug):
        calls.append(("release", slug, fd_scheduler._AGENT_LOCKS[slug].locked()))

    async def capture(**kw):
        calls.append(("prompt", kw["port"]))
        return "done"

    async def not_blocked(owner):
        return False

    async def no_line(*a, **k):
        return None

    monkeypatch.setattr(agent_idle, "ready_for_task", ready)
    monkeypatch.setattr(agent_idle, "release_after_task_soon", release)
    monkeypatch.setattr(fd_scheduler, "run_prompt_and_capture", capture)
    monkeypatch.setattr(fd_scheduler, "_to_line", no_line)
    monkeypatch.setattr(fd_scheduler, "_in_quiet_hours", lambda *a: False)
    monkeypatch.setattr(usage_quota, "automation_blocked", not_blocked)
    store = fd_scheduler.SchedulerStore(db_path=tmp_path / "scheduler.db")
    monkeypatch.setattr(fd_scheduler, "_STORE", store)
    job = store.create(schedule="daily 08:00", agent_slug="glasses", prompt="hi",
                       delivery_kind="agent", delivery={"chat": True})

    status, result, _ = _loop_run(fd_scheduler.run_job(job, trigger="manual"))
    assert (status, result) == ("ok", "done")
    # Readied before the prompt; let go once the run no longer holds it.
    assert calls == [("ready", "glasses"), ("prompt", 24101), ("release", "glasses", False)]


def test_a_run_whose_agent_cannot_start_says_so(deck, monkeypatch):
    from captain_claw.flight_deck import fd_scheduler, usage_quota

    async def ready(slug, timeout=0):
        return False

    async def not_blocked(owner):
        return False

    monkeypatch.setattr(agent_idle, "ready_for_task", ready)
    monkeypatch.setattr(fd_scheduler, "_in_quiet_hours", lambda *a: False)
    monkeypatch.setattr(usage_quota, "automation_blocked", not_blocked)
    status, text = _loop_run(fd_scheduler.execute_job({"agent_slug": "gone", "prompt": "hi"}))
    assert status == "error:agent-not-running" and "could not be started" in text
