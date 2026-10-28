"""A non-admin's agent .env can't set what Flight Deck pins — and is told so.

#255 pins, at every start, the variables that would have a non-admin's agent
load code or config from a path, run a command, or route its calls elsewhere
(`server._pin_confinement`, `_PINNED_ENV_KEYS` / `_PINNED_ENV_PREFIXES`): Flight Deck's
value, or none. A line setting one in the agent's .env never applied — and
the owner, saving it in the agent editor, had no way to know.

Pinned here: ``PUT /fd/agent-config`` refuses a .env that adds or changes a
pinned variable on an agent whose OWNER isn't an admin (it's the owner the pin
follows — also when an admin edits the agent for them), naming the variables,
with nothing written. A line already there round-trips, removing one is fine,
other variables are untouched; an admin's agent, and a deck that confines
nothing, save as before.

Real FlightDeckDB + process registry in tmp dirs; ``DATA_DIR`` is never the
real ``./fd-data``.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import rate_limiter, server
from captain_claw.flight_deck.auth import ACT_AS_HEADER, create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

ADMIN = "user-admin"
BOB = "user-bob"
REMOTE = ("203.0.113.9", 40001)
ENV = "BRAVE_API_KEY=brave-own\nOLLAMA_BASE_URL=http://127.0.0.1:11434\n"


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    for uid, role in ((ADMIN, "admin"), (BOB, "user")):
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, ?, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", uid, role))
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
    monkeypatch.delenv("FD_AGENT_CONFINE", raising=False)
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())
    # The owner's role as the pin reads it (sync, at start): Bob is a user.
    monkeypatch.setattr(server, "_owner_is_admin", lambda owner: owner == ADMIN)
    server._save_process_registry({"bob-agent": {"slug": "bob-agent", "name": "Bob's agent",
                                                 "web_port": 24101, "owner": BOB, "pid": None}})
    (data / "bob-agent").mkdir()
    (data / "bob-agent" / "config.yaml").write_text("model:\n  provider: ollama\n  model: m\n")
    (data / "bob-agent" / ".env").write_text(ENV)
    try:
        yield db
    finally:
        await db.close()
        fd_auth._db = prev


def _env() -> str:
    return (server.DATA_DIR / "bob-agent" / ".env").read_text()


async def _save(env: str, caller: str = BOB, act_as: str | None = None) -> httpx.Response:
    hdr = {"Authorization": f"Bearer {create_access_token(caller)}"}
    if act_as:
        hdr[ACT_AS_HEADER] = act_as
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, client=REMOTE),
                                 base_url="http://fd.test") as c:
        return await c.put("/fd/agent-config/process/bob-agent", headers=hdr, json={"env": env})


@pytest.mark.parametrize("line", [
    "PYTHONPATH=/home/bob/ws", "export LD_PRELOAD=x.so", "XDG_CONFIG_HOME=/home/bob/ws",
    "https_proxy=http://collector.example:8080", "OPENSSL_CONF=x.cnf", "PLAYWRIGHT_DRIVER_PATH=x",
    "SSL_CERT_FILE=ca.pem", "GIT_SSH_COMMAND=sh x", "PATH=/home/bob/ws",
])
async def test_setting_a_pinned_variable_is_refused(deck, line):
    r = await _save(ENV + line + "\n")
    assert r.status_code == 400
    assert line.removeprefix("export ").split("=")[0] in r.json()["detail"]
    assert _env() == ENV


async def test_every_pinned_one_is_named(deck):
    r = await _save(ENV + "PYTHONPATH=x\nNODE_OPTIONS=--require x\n")
    assert "NODE_OPTIONS, PYTHONPATH" in r.json()["detail"]


async def test_other_variables_save_as_before(deck):
    env = ENV + "MY_SERVICE_TOKEN=t-1\nTAVILY_API_KEY=tv\nMY_PATH_PREFIX=/x\n"
    assert (await _save(env)).status_code == 200
    assert _env() == env


async def test_a_line_already_there_round_trips_and_may_go(deck):
    with_pin = ENV + "PYTHONPATH=/set/by/an/admin\n"
    (server.DATA_DIR / "bob-agent" / ".env").write_text(with_pin)
    assert (await _save(with_pin + "X=1\n")).status_code == 200
    assert (await _save(with_pin.replace("/set/by/an/admin", "/mine"))).status_code == 400
    assert (await _save(ENV)).status_code == 200 and _env() == ENV


async def test_it_follows_the_owner_not_the_editor(deck):
    """An admin editing Bob's agent is refused too: the pin is Bob's agent's."""
    assert (await _save(ENV + "PYTHONPATH=x\n", caller=ADMIN, act_as=BOB)).status_code == 400


async def test_an_admins_agent_saves_as_before(deck, monkeypatch):
    monkeypatch.setattr(server, "_owner_is_admin", lambda owner: True)
    assert (await _save(ENV + "PYTHONPATH=/opt/lib\n")).status_code == 200


async def test_a_deck_that_confines_nothing_saves_as_before(deck, monkeypatch):
    monkeypatch.setenv("FD_AGENT_CONFINE", "off")
    assert (await _save(ENV + "PYTHONPATH=/opt/lib\n")).status_code == 200


def test_pinned_names():
    for name in ("PYTHONPATH", "pythonstartup", "XDG_CONFIG_HOME", "OPENSSL_CONF", "_JAVA_OPTIONS",
                 "PIP_INDEX_URL", "SSL_CERT_FILE", "https_proxy", "BROWSER", "HOSTALIASES"):
        assert server._pinned_env_name(name), name
    for name in ("BRAVE_API_KEY", "OPENAI_BASE_URL", "MY_PATH", "XDGX", "BROWSERLESS_TOKEN", "PIPELINE_ID"):
        assert not server._pinned_env_name(name), name
