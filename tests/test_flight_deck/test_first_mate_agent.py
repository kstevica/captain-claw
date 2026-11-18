"""The agent that is the owner's First Mate (P5, piece 3).

Pinned here:

* the owner picks one of their own agents, never someone else's; a picked
  agent that is gone shows as gone;
* "Make one" spawns an agent called First Mate from the concierge archetype
  with what it's for, and points First Mate at it;
* drafting asks that agent's model: through its REST while it runs, its
  fallback when it stops answering, the owner's tier set when none is picked.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import course as c
from captain_claw.flight_deck import events, line, rate_limiter, server
from captain_claw.flight_deck import first_mate_agent as mate
from captain_claw.flight_deck import first_mate_drafting as drafting
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.flight_deck.first_mate_store import FirstMateError

BOB, CAROL = "user-bob", "user-carol"
ANSWER = {"title": "Invoice replies", "charter": "Draft replies to invoice questions. Never send anything.",
          "schedule": "weekdays 08:00", "done_when": ["every question has a draft"], "cases": []}


@pytest.fixture(autouse=True)
def _no_register(monkeypatch):
    from captain_claw.flight_deck import tool_audit_routes
    monkeypatch.setattr(tool_audit_routes, "_data_dir", None)
    monkeypatch.setattr(tool_audit_routes, "_store", None)


@pytest.fixture(autouse=True)
def registry(tmp_path, monkeypatch):
    """Making an agent labels it in the process registry: a test's own, never the deck's."""
    path = tmp_path / ".processes.json"
    path.write_text(json.dumps({"first-mate": {"name": "First Mate", "owner": BOB}}))
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", path)
    return path


@pytest.fixture
async def db(tmp_path: Path):
    prev = fd_auth._db
    d = FlightDeckDB(tmp_path / "fd.db")
    await d.init()
    set_auth_db(d)
    for uid in (BOB, CAROL):
        await d._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, 'user', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", uid))
    await d._db.commit()
    try:
        yield d
    finally:
        await d.close()
        fd_auth._db = prev


@pytest.fixture
def store(tmp_path, monkeypatch, db):
    monkeypatch.setattr(events, "_STORE", events.EventsStore(tmp_path / "events.db"))
    monkeypatch.setattr(c, "_STORE", c.CourseStore(tmp_path / "course.db"))
    s = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", s)
    s.set_settings({"enabled": True, "admins_only": False})
    return s


@pytest.fixture(autouse=True)
def agents(monkeypatch):
    """Bob has two agents, Carol one; which of them run."""
    owned = {BOB: [{"slug": "bob-helper", "name": "Helper"}, {"slug": "bob-mate", "name": "Mate"}],
             CAROL: [{"slug": "carol-bot", "name": "Carol's bot"}]}
    alive = {"bob-mate"}
    monkeypatch.setattr(line, "user_agents", lambda uid: list(owned.get(uid, [])))
    monkeypatch.setattr(server, "_process_is_alive", lambda slug: slug in alive)
    async def fake_fan_out(user_id, item, to_phones=True, rung=False):
        return {}
    monkeypatch.setattr(line, "fan_out", fake_fan_out)
    return {"owned": owned, "alive": alive}


# ── picking ────────────────────────────────────────────────────────────

def test_current_and_picking(store, agents):
    assert mate.current(store, BOB) == {"slug": "", "name": "", "exists": False, "running": False}
    store.set_prefs(BOB, {"agent": "bob-mate"})
    assert mate.current(store, BOB) == {"slug": "bob-mate", "name": "Mate", "exists": True, "running": True}
    store.set_prefs(BOB, {"agent": "bob-helper"})
    assert mate.current(store, BOB)["running"] is False
    agents["owned"][BOB] = agents["owned"][BOB][1:]  # Helper was deleted
    assert mate.current(store, BOB) == {"slug": "bob-helper", "name": "bob-helper", "exists": False,
                                        "running": False}
    mate.check_pick(BOB, "bob-mate")
    with pytest.raises(FirstMateError, match="isn't one of yours"):
        mate.check_pick(BOB, "carol-bot")


async def test_making_one(store, monkeypatch, registry):
    registry_file = registry
    from captain_claw.flight_deck import agent_forge
    asked: list[dict] = []

    async def spawn(user, *, name, description, archetype, instructions, env):
        asked.append({"user": user["id"], "name": name, "archetype": archetype, "instructions": instructions})
        return SimpleNamespace(ok=True, slug="first-mate", message=""), name

    monkeypatch.setattr(agent_forge, "_spawn_agent", spawn)
    made = await mate.create(store, {"id": BOB, "role": "user"})
    assert made == {"slug": "first-mate", "name": "First Mate"}
    assert store.prefs(BOB)["agent"] == "first-mate"
    [a] = asked
    assert a["archetype"] == "concierge@balanced" and a["name"] == "First Mate"
    assert "first_mate tool" in a["instructions"] and "never approve" in a["instructions"]
    assert json.loads(registry_file.read_text())["first-mate"]["group"] == "First Mate"  # beside its workers

    async def broken(user, **kw):
        return SimpleNamespace(ok=False, slug="", message="no free port"), kw["name"]

    monkeypatch.setattr(agent_forge, "_spawn_agent", broken)
    with pytest.raises(FirstMateError, match="no free port"):
        await mate.create(store, {"id": CAROL, "role": "user"})
    assert store.prefs(CAROL)["agent"] == ""


# ── asking its model ───────────────────────────────────────────────────

async def test_asking_the_agents_model(store, monkeypatch):
    from captain_claw.flight_deck import agent_preset, file_library, taskboard_triage
    assert await mate.ask(store, BOB, "system", "user") is None  # none picked: the tier set answers
    store.set_prefs(BOB, {"agent": "bob-mate"})
    seen: list[str] = []

    async def reader_for(task, user_id):
        seen.append(task["agent_slug"])
        return {"agent": "bob-mate", "model": "its-model", "fallback": {"model": "tier-model"}}

    async def agent_complete(reader, system, prompt, owner_id, max_tokens, **kw):
        return SimpleNamespace(text=f"agent says ({max_tokens})")

    async def model_reply(reader, system, prompt, owner_id, **kw):
        return SimpleNamespace(text=f"{reader['model']} says")

    monkeypatch.setattr(taskboard_triage, "reader_for", reader_for)
    monkeypatch.setattr(agent_preset, "_agent_complete", agent_complete)
    monkeypatch.setattr(file_library, "model_reply", model_reply)
    assert await mate.ask(store, BOB, "system", "user") == f"agent says ({mate.MAX_TOKENS})" and seen == ["bob-mate"]

    async def gone(*a, **kw):
        raise agent_preset._AgentGoneError("stopped")

    monkeypatch.setattr(agent_preset, "_agent_complete", gone)
    assert await mate.ask(store, BOB, "system", "user") == "tier-model says"


async def test_drafting_uses_the_agent(store, monkeypatch):
    store.set_prefs(BOB, {"agent": "bob-mate"})
    asked: list[str] = []

    async def ask(st, owner, system, user, max_tokens=0):
        asked.append(owner)
        return json.dumps(ANSWER)

    monkeypatch.setattr(mate, "ask", ask)
    made = await drafting.draft(store, BOB, "Draft replies to invoice questions every weekday.")
    assert made["title"] == "Invoice replies" and asked == [BOB]


# ── routes ─────────────────────────────────────────────────────────────

def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, client=("203.0.113.9", 40001)),
                             base_url="http://fd.test")


def _hdr(who):
    return {"Authorization": f"Bearer {create_access_token(who)}"}


@pytest.fixture
def deck(monkeypatch, store):
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())


async def test_the_routes(deck, store, monkeypatch):
    from captain_claw.flight_deck import agent_forge

    async def spawn(user, *, name, description, archetype, instructions, env):
        return SimpleNamespace(ok=True, slug="bob-first-mate", message=""), name

    monkeypatch.setattr(agent_forge, "_spawn_agent", spawn)
    async with _client() as cl:
        got = (await cl.get("/fd/first-mate/my-agent", headers=_hdr(BOB))).json()
        assert got["current"]["slug"] == "" and [a["slug"] for a in got["agents"]] == ["bob-helper", "bob-mate"]
        r = await cl.put("/fd/first-mate/prefs", headers=_hdr(BOB), json={"agent": "carol-bot"})
        assert r.status_code == 400 and "isn't one of yours" in r.json()["detail"]
        r = await cl.put("/fd/first-mate/prefs", headers=_hdr(BOB), json={"agent": "bob-mate"})
        assert r.status_code == 200 and r.json()["agent"] == "bob-mate"
        r = await cl.put("/fd/first-mate/prefs", headers=_hdr(BOB), json={"agent": ""})
        assert r.status_code == 200 and r.json()["agent"] == ""
        r = await cl.post("/fd/first-mate/my-agent", headers=_hdr(BOB))
        assert r.status_code == 200 and r.json()["slug"] == "bob-first-mate"
    assert store.prefs(BOB)["agent"] == "bob-first-mate"
