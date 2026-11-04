"""Removing an agent for good, and the kiosk's agent settings.

* ``DELETE /fd/processes/{slug}?purge=true`` takes the agent off the registry
  AND deletes its folder, Flight Deck's chat log with it (every lane), the
  scheduled tasks that run it and its File/Data library rows — another
  agent's are untouched; without ``purge`` the folder stays, as before;
* the folder delete never leaves DATA_DIR (a symlink goes, not its target);
* a kept folder stays its owner's: nobody else spawns or clones onto it (the
  owner is recorded in it); one from before owners were recorded goes only to
  the owner still on record, or to anyone when it holds no files;
* ``GET /fd/agent-mode`` reads the mode as the agent resolves it, and ``PUT``
  writes Neutra too (an archetype's default mode would otherwise come back);
* ``PUT /fd/agent-model`` with only ``vision`` leaves the model alone.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import yaml

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import data_library, file_library, rate_limiter, server
from captain_claw.flight_deck import fd_scheduler as sched
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

BOB = "user-bob"
CAROL = "user-carol"


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Auth-enabled deck: Bob and Carol own one process agent each."""
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    for uid in (BOB, CAROL):
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, 'user', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", uid))
    await db._db.commit()

    data = tmp_path / "fd-data"
    data.mkdir()
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", data / ".processes.json")
    monkeypatch.setattr(server, "_processes", {})
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    monkeypatch.setattr(server, "_schedule_fleet_notify", lambda *a, **k: None)
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())
    monkeypatch.setattr(sched, "_STORE", sched.SchedulerStore(db_path=tmp_path / "scheduler.db"))

    server._save_process_registry({
        "bob-agent": {"slug": "bob-agent", "name": "Bob's agent", "web_port": 24101,
                      "web_auth": "bob-tok", "owner": BOB, "pid": None},
        "carol-agent": {"slug": "carol-agent", "name": "Carol's agent", "web_port": 24102,
                        "web_auth": "carol-tok", "owner": CAROL, "pid": None},
    })
    for slug in ("bob-agent", "carol-agent"):
        cc = data / slug / "data" / "home-config-parent" / ".captain-claw"
        cc.mkdir(parents=True)
        (data / slug / "data" / "workspace").mkdir(parents=True)
        (data / slug / "data" / "workspace" / "notes.md").write_text("hi\n")
        (data / slug / "config.yaml").write_text(yaml.dump({"model": {"provider": "openai", "model": f"{slug}-m"}}))
        (cc / "config.yaml").write_text((data / slug / "config.yaml").read_text())
    try:
        yield db
    finally:
        await db.close()
        fd_auth._db = prev


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app, client=("203.0.113.9", 40001)), base_url="http://fd.test")


def _hdr(uid: str) -> dict:
    return {"Authorization": f"Bearer {create_access_token(uid)}"}


async def _chat(db, sid: str, uid: str, agent_id: str) -> None:
    await db.upsert_chat_session(sid, uid, agent_id, agent_id)
    await db._db.execute(
        "INSERT INTO chat_messages (session_id, role, content, created_at) VALUES (?, 'user', 'x', '2026')", (sid,))
    await db._db.commit()


async def _rows(db, sql: str) -> list:
    cur = await db._db.execute(sql)
    return [tuple(r) for r in await cur.fetchall()]


def _task(slug: str, owner: str) -> dict:
    return sched.get_store().create(owner_id=owner, name=slug, schedule="daily 08:00", agent_slug=slug,
                                    prompt="brief", delivery_kind="agent", delivery={"chat": True})


# ── remove for good ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_purge_deletes_the_folder_chat_log_and_tasks_of_that_agent_only(deck):
    await _chat(deck, "proc-bob-agent", BOB, "proc-bob-agent")
    await _chat(deck, "proc-bob-agent::side", BOB, "proc-bob-agent")
    await _chat(deck, "proc-carol-agent", CAROL, "proc-carol-agent")
    bob_task, carol_task = _task("bob-agent", BOB), _task("carol-agent", CAROL)
    store = sched.get_store()
    store.start_run(bob_task, "manual")

    async with _client() as c:
        r = await c.delete("/fd/processes/bob-agent?purge=true", headers=_hdr(BOB))
    assert r.status_code == 200 and r.json()["ok"], r.text

    assert set(server._load_process_registry()) == {"carol-agent"}
    assert not (server.DATA_DIR / "bob-agent").exists()
    assert (server.DATA_DIR / "carol-agent" / "data" / "workspace" / "notes.md").is_file()
    assert await _rows(deck, "SELECT id FROM chat_sessions") == [("proc-carol-agent",)]
    assert await _rows(deck, "SELECT DISTINCT session_id FROM chat_messages") == [("proc-carol-agent",)]
    assert store.get(bob_task["id"]) is None and store.get(carol_task["id"]) is not None
    assert store.list_runs(job_id=bob_task["id"]) == []


@pytest.mark.asyncio
async def test_without_purge_the_folder_and_log_stay(deck):
    await _chat(deck, "proc-bob-agent", BOB, "proc-bob-agent")
    async with _client() as c:
        r = await c.delete("/fd/processes/bob-agent", headers=_hdr(BOB))
    assert r.status_code == 200
    assert (server.DATA_DIR / "bob-agent" / "config.yaml").is_file()
    assert await _rows(deck, "SELECT id FROM chat_sessions") == [("proc-bob-agent",)]


@pytest.mark.asyncio
async def test_nobody_purges_someone_elses_agent(deck):
    async with _client() as c:
        r = await c.delete("/fd/processes/bob-agent?purge=true", headers=_hdr(CAROL))
    assert r.status_code in (403, 404)
    assert (server.DATA_DIR / "bob-agent" / "config.yaml").is_file()
    assert "bob-agent" in server._load_process_registry()


@pytest.mark.parametrize("slug", ["", ".", "..", "a/b", "..\\x"])
def test_the_folder_delete_refuses_anything_but_one_folder_name(deck, slug):
    assert server._remove_agent_dir(slug)
    assert (server.DATA_DIR / "bob-agent").is_dir()


def test_a_symlinked_folder_loses_the_link_not_its_target(deck, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("mine")
    (server.DATA_DIR / "linked").symlink_to(outside)
    assert server._remove_agent_dir("linked") == ""
    assert not (server.DATA_DIR / "linked").exists() and (outside / "keep.txt").is_file()


def test_read_only_entries_go_too(deck):
    ro = server.DATA_DIR / "bob-agent" / "data" / "workspace" / "repo"
    ro.mkdir()
    (ro / "obj").write_text("x")
    (ro / "obj").chmod(0o444)
    ro.chmod(0o555)
    assert server._remove_agent_dir("bob-agent") == ""
    assert not (server.DATA_DIR / "bob-agent").exists()


@pytest.mark.asyncio
async def test_purge_drops_that_agents_library_rows(deck, tmp_path, monkeypatch):
    files = file_library.LibraryStore(tmp_path / "file_library.db")
    tables = data_library.DataStore(tmp_path / "data_library.db")
    monkeypatch.setattr(file_library, "_STORE", files)
    monkeypatch.setattr(data_library, "_STORE", tables)
    for slug in ("bob-agent", "carol-agent"):
        files.sync_agent(slug, file_library.walk_workspace(slug))
        tables.sync_agent(slug, [{"table": "leads", "columns": [{"name": "email"}], "row_count": 1,
                                  "updated_at": "2026", "fingerprint": "f", "sample": [{"email": "a@x.co"}]}])
    async with _client() as c:
        r = await c.delete("/fd/processes/bob-agent?purge=true", headers=_hdr(BOB))
    assert r.status_code == 200 and r.json()["ok"], r.text
    assert {row["agent_slug"] for row in files.search(["bob-agent", "carol-agent"], [])} == {"carol-agent"}
    assert {row["agent_slug"] for row in tables.all_for(["bob-agent", "carol-agent"])} == {"carol-agent"}


# ── a kept folder stays its owner's ─────────────────────────────────────────


class _FakePopen:
    spawned: list[str] = []

    def __init__(self, args, cwd=None, env=None, stdout=None, stderr=None, start_new_session=None,
                 creationflags=0):
        _FakePopen.spawned.append(Path(cwd).name)
        self.pid = 434343
        if stdout is not None:
            stdout.close()

    def poll(self):
        return None


@pytest.fixture
def spawner(deck, monkeypatch):
    """Spawns go all the way through the route without starting anything;
    returns the folders a process was started in."""
    _FakePopen.spawned = []
    monkeypatch.setattr(server.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(server, "_is_port_available", lambda port: True)
    monkeypatch.setattr(server, "_deck_containers", lambda all=False, client=None: [])

    async def _no_cap(user, count):
        return None

    monkeypatch.setattr(server, "check_agent_count_limit", _no_cap)
    monkeypatch.setattr(rate_limiter, "_spawn_limiter", rate_limiter._SlidingWindow())
    monkeypatch.setenv("FD_SPAWN_SETTLE_S", "0")
    monkeypatch.setenv("FD_PORT", "25999")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fd-env-key")
    return _FakePopen.spawned


def _folder_owner(slug: str) -> str:
    return (server.DATA_DIR / slug / server.FOLDER_OWNER_FILE).read_text().strip()


@pytest.mark.asyncio
async def test_a_kept_folder_goes_to_no_one_else(deck, spawner):
    async with _client() as c:
        assert (await c.delete("/fd/processes/bob-agent", headers=_hdr(BOB))).status_code == 200
        spawn = await c.post("/fd/spawn-process", headers=_hdr(CAROL),
                             json={"name": "bob-agent", "web_port": 24310})
        clone = await c.post("/fd/processes/carol-agent/clone", headers=_hdr(CAROL),
                             json={"new_name": "Bob Agent"})
    for r in (spawn, clone):
        assert r.status_code == 409, r.text
        assert r.json()["detail"] == "An agent named 'bob-agent' already exists. Choose a different name."
    assert spawner == [] and "bob-agent" not in server._load_process_registry()
    assert _folder_owner("bob-agent") == BOB
    assert (server.DATA_DIR / "bob-agent" / "data" / "workspace" / "notes.md").read_text() == "hi\n"
    assert yaml.safe_load((server.DATA_DIR / "bob-agent" / "config.yaml").read_text())["model"]["model"] == "bob-agent-m"


@pytest.mark.asyncio
async def test_its_owner_gets_a_kept_folder_back(deck, spawner):
    async with _client() as c:
        await c.delete("/fd/processes/bob-agent", headers=_hdr(BOB))
        r = await c.post("/fd/spawn-process", headers=_hdr(BOB), json={"name": "bob-agent", "web_port": 24310})
    assert r.status_code == 200, r.text
    assert spawner == ["bob-agent"] and server._load_process_registry()["bob-agent"]["owner"] == BOB
    assert (server.DATA_DIR / "bob-agent" / "data" / "workspace" / "notes.md").is_file()
    assert _folder_owner("bob-agent") == BOB


@pytest.mark.asyncio
async def test_a_folder_from_before_owners_were_recorded(deck, spawner):
    """Nothing in it says whose it is: with its entry gone nobody gets it,
    unless it holds no files; with its entry there, that entry's owner does."""
    registry = server._load_process_registry()
    del registry["bob-agent"]
    server._save_process_registry(registry)
    (server.DATA_DIR / "blank" / "data" / "workspace").mkdir(parents=True)
    async with _client() as c:
        carol = await c.post("/fd/spawn-process", headers=_hdr(CAROL), json={"name": "bob-agent", "web_port": 24310})
        bob = await c.post("/fd/spawn-process", headers=_hdr(BOB), json={"name": "bob-agent", "web_port": 24311})
        blank = await c.post("/fd/spawn-process", headers=_hdr(CAROL), json={"name": "blank", "web_port": 24312})
        own = await c.post("/fd/spawn-process", headers=_hdr(CAROL),
                           json={"name": "carol-agent", "web_port": 24313})
    assert (carol.status_code, bob.status_code) == (409, 409), (carol.text, bob.text)
    assert (blank.status_code, own.status_code) == (200, 200), (blank.text, own.text)
    assert spawner == ["blank", "carol-agent"]
    assert not (server.DATA_DIR / "bob-agent" / server.FOLDER_OWNER_FILE).exists()
    assert _folder_owner("blank") == CAROL and _folder_owner("carol-agent") == CAROL


# ── cognitive mode ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mode_reads_the_file_then_the_config_default_then_neutra(deck):
    cc = server.DATA_DIR / "bob-agent" / "data" / "home-config-parent" / ".captain-claw"
    async with _client() as c:
        get = lambda: c.get("/fd/agent-mode/process/bob-agent", headers=_hdr(BOB))  # noqa: E731
        assert (await get()).json() == {"mode": "neutra"}

        (cc / "config.yaml").write_text(yaml.dump({"cognitive_mode": {"default_mode": "aeolian"}}))
        assert (await get()).json() == {"mode": "aeolian"}

        (cc / "cognitive_mode.txt").write_text("lydian\n")
        assert (await get()).json() == {"mode": "lydian"}

        # Neutra is written, not left out — or the archetype's aeolian comes back.
        r = await c.put("/fd/agent-mode/process/bob-agent", headers=_hdr(BOB), json={"mode": "neutra"})
        assert r.status_code == 200
        assert (cc / "cognitive_mode.txt").read_text() == "neutra"
        assert (await get()).json() == {"mode": "neutra"}

        assert (await c.get("/fd/agent-mode/process/bob-agent", headers=_hdr(CAROL))).status_code in (403, 404)


# ── vision only ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_vision_only_update_keeps_the_model(deck):
    async with _client() as c:
        r = await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(BOB), json={"vision": "on"})
        assert r.status_code == 200, r.text
        model = yaml.safe_load((server.DATA_DIR / "bob-agent" / "config.yaml").read_text())["model"]
        assert model == {"provider": "openai", "model": "bob-agent-m", "vision": True}

        r = await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(BOB), json={})
        assert r.status_code == 400


# ── model, key and endpoint reach every copy ────────────────────────────────


def _copies(slug: str) -> list[Path]:
    agent_dir = server.DATA_DIR / slug
    return server._agent_config_paths(agent_dir)


def _with_all_copies(slug: str, model: dict) -> None:
    for path in _copies(slug):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump({"model": dict(model), "context": {"max_tokens": 32768}}))


@pytest.mark.asyncio
async def test_a_new_provider_drops_the_old_key_and_endpoint_everywhere(deck):
    _with_all_copies("bob-agent", {"provider": "ollama", "model": "ornith:9b",
                                   "api_key": "old-key", "base_url": "http://127.0.0.1:11434"})
    async with _client() as c:
        r = await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(BOB),
                        json={"provider": "anthropic", "model": "claude-sonnet-5-5"})
    assert r.status_code == 200 and r.json()["updated"] == 3, r.text
    for path in _copies("bob-agent"):
        data = yaml.safe_load(path.read_text())
        assert data["model"] == {"provider": "anthropic", "model": "claude-sonnet-5-5", "api_key": "", "base_url": ""}
        assert data["context"] == {"max_tokens": 32768}
    entry = server._load_process_registry()["bob-agent"]
    assert (entry["provider"], entry["model"]) == ("anthropic", "claude-sonnet-5-5")
    async with _client() as c:
        listed = {p["slug"]: p for p in (await c.get("/fd/processes", headers=_hdr(BOB))).json()}
    assert listed["bob-agent"]["model"] == "claude-sonnet-5-5"
    # Carol's agent is untouched.
    assert server._load_process_registry()["carol-agent"].get("model") is None


@pytest.mark.asyncio
async def test_the_same_provider_keeps_its_endpoint_and_takes_a_new_key(deck):
    _with_all_copies("bob-agent", {"provider": "openai", "model": "a", "api_key": "k1",
                                   "base_url": "http://127.0.0.1:8000/v1"})
    async with _client() as c:
        await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(BOB),
                    json={"provider": "openai", "model": "b", "api_key": " k2 "})
        for path in _copies("bob-agent"):
            assert yaml.safe_load(path.read_text())["model"] == {
                "provider": "openai", "model": "b", "api_key": "k2", "base_url": "http://127.0.0.1:8000/v1"}

        # No key named: the one it has stays. A named endpoint is written.
        await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(BOB),
                    json={"provider": "openai", "model": "b", "base_url": "http://box:9000/v1"})
        for path in _copies("bob-agent"):
            m = yaml.safe_load(path.read_text())["model"]
            assert (m["api_key"], m["base_url"]) == ("k2", "http://box:9000/v1")

        # Back to ollama: its own endpoint.
        await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(BOB),
                    json={"provider": "ollama", "model": "ornith:9b"})
        assert yaml.safe_load(_copies("bob-agent")[2].read_text())["model"]["base_url"] == "http://127.0.0.1:11434"

        r = await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(BOB), json={"provider": "openai"})
        assert r.status_code == 400
        r = await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(CAROL),
                        json={"provider": "openai", "model": "x"})
        assert r.status_code in (403, 404)


@pytest.mark.asyncio
async def test_the_mode_reaches_every_config_copy_too(deck):
    _with_all_copies("bob-agent", {"provider": "openai", "model": "a"})
    async with _client() as c:
        r = await c.put("/fd/agent-mode/process/bob-agent", headers=_hdr(BOB), json={"mode": "dorian"})
    assert r.status_code == 200
    for path in _copies("bob-agent"):
        data = yaml.safe_load(path.read_text())
        assert data["cognitive_mode"]["default_mode"] == "dorian"
        assert data["model"]["model"] == "a"
    cc = server.DATA_DIR / "bob-agent" / "data" / "home-config-parent" / ".captain-claw"
    assert (cc / "cognitive_mode.txt").read_text() == "dorian"


# ── only providers the agent can run on ─────────────────────────────────────


@pytest.mark.asyncio
async def test_an_unsupported_provider_is_refused_and_nothing_written(deck):
    _with_all_copies("bob-agent", {"provider": "ollama", "model": "ornith:9b"})
    before = [path.read_text() for path in _copies("bob-agent")]
    async with _client() as c:
        r = await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(BOB),
                        json={"provider": "groq", "model": "llama"})
        assert r.status_code == 400 and "groq" in r.text
        assert [path.read_text() for path in _copies("bob-agent")] == before
        # An alias of a supported one is fine.
        r = await c.put("/fd/agent-model/process/bob-agent", headers=_hdr(BOB),
                        json={"provider": "claude", "model": "claude-sonnet-5-5"})
        assert r.status_code == 200


def test_every_provider_flight_deck_offers_is_one_an_agent_runs_on():
    import re

    from captain_claw.llm import is_supported_provider

    src = (Path(__file__).resolve().parents[2] / "flight-deck" / "src" / "services" / "tierConfig.ts").read_text()
    block = src[src.index("export const LLM_PROVIDERS"):]
    block = block[:block.index("] as const")]
    offered = re.findall(r"value: '([^']+)'", block)
    assert len(offered) >= 8
    assert [p for p in offered if not is_supported_provider(p)] == []
