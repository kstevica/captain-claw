"""Flight Deck carries an agent's vision switch (``model.vision``) end to end.

* spawn: ``AgentConfig.vision`` (auto | on | off) is written by both config
  writers as a YAML boolean — and left out for auto, so existing configs stay
  byte-for-byte what they were;
* the agent reads what Flight Deck wrote;
* ``PUT /fd/agent-model`` sets / clears it in all three config copies and
  leaves it alone when the body doesn't name it;
* a Library tier marked "Sees images" turns it on for an archetype spawn,
  unless the caller set it.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
import yaml

import captain_claw.flight_deck.archetypes as arch_mod
import captain_claw.flight_deck.auth as auth_mod
import captain_claw.flight_deck.basna_routes as basna_mod
import captain_claw.flight_deck.server as server
from captain_claw.config import Config
from captain_claw.flight_deck.db import FlightDeckDB


@pytest.mark.parametrize("value,expected", [
    ("on", True), ("ON", True), (True, True), ("yes", True),
    ("off", False), (False, False), ("0", False),
    ("auto", None), ("", None), (None, None), ("maybe", None),
])
def test_vision_switch(value, expected):
    assert server._vision_switch(value) is expected


@pytest.mark.parametrize("writer", ["docker", "process"])
@pytest.mark.parametrize("vision,expected", [("on", True), ("off", False)])
def test_both_writers_put_the_switch_in_the_model_block(tmp_path: Path, writer, vision, expected):
    c = server.AgentConfig(name="x", provider="openai", model="nvidia/Qwen3.8-27B-NVFP4",
                           base_url="http://127.0.0.1:8000", vision=vision)
    out = (server._build_config_yaml(c) if writer == "docker"
           else server._build_process_config_yaml(c, tmp_path))
    data = yaml.safe_load(out)
    assert data["model"]["vision"] is expected
    # The agent reads it back as written.
    assert Config(**data).model.vision is expected


@pytest.mark.parametrize("writer", ["docker", "process"])
def test_auto_leaves_the_config_as_it_was(tmp_path: Path, writer):
    def build(c):
        return (server._build_config_yaml(c) if writer == "docker"
                else server._build_process_config_yaml(c, tmp_path))
    assert build(server.AgentConfig(name="x", vision="auto")) == build(server.AgentConfig(name="x"))
    assert "vision" not in yaml.safe_load(build(server.AgentConfig(name="x")))["model"]


# ── PUT /fd/agent-model ──────────────────────────────────────────────────────


def _config_copies(agent_dir: Path) -> list[Path]:
    return [
        agent_dir / "config.yaml",
        agent_dir / "data" / "home-config" / "config.yaml",
        agent_dir / "data" / "home-config-parent" / ".captain-claw" / "config.yaml",
    ]


@pytest.fixture()
def agent_dir(tmp_path, monkeypatch) -> Path:
    for path in _config_copies(tmp_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump({"model": {"provider": "openai", "model": "m", "api_key": ""},
                                   "context": {"max_tokens": 32768}}))
    monkeypatch.setattr(server, "_resolve_agent_dir", lambda identifier, kind, user_id: tmp_path)
    # The route is called with no user: a deck without accounts (with them, a
    # non-admin's model change is checked against the team's keys).
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    return tmp_path


def _request():
    return types.SimpleNamespace(state=types.SimpleNamespace(user_id="u1"))


async def _update(**body):
    return await server.update_agent_model(
        "process", "agent-1", server.AgentModelUpdate(provider="openai", model="m", **body),
        _request(), None)


@pytest.mark.asyncio
async def test_model_update_sets_clears_and_keeps_the_switch(agent_dir):
    result = await _update(vision="on")
    assert result["updated"] == 3
    for path in _config_copies(agent_dir):
        assert yaml.safe_load(path.read_text())["model"]["vision"] is True

    await _update()  # a body without vision leaves it alone
    for path in _config_copies(agent_dir):
        assert yaml.safe_load(path.read_text())["model"]["vision"] is True

    await _update(vision="off")
    assert yaml.safe_load(_config_copies(agent_dir)[0].read_text())["model"]["vision"] is False

    await _update(vision="auto")
    for path in _config_copies(agent_dir):
        data = yaml.safe_load(path.read_text())
        assert "vision" not in data["model"] and data["context"] == {"max_tokens": 32768}


# ── Library tier "Sees images" ───────────────────────────────────────────────

_ARCH = {"id": "analyst", "role": "Analyst", "tier": "balanced"}


@pytest.fixture()
async def db(monkeypatch, tmp_path):
    fd_db = FlightDeckDB(tmp_path / "flight-deck.db")
    await fd_db.init()
    monkeypatch.setattr(auth_mod, "_db", fd_db)
    monkeypatch.setattr(basna_mod, "_SYSTEM_PROVIDER_KEYS", {})
    monkeypatch.setattr(basna_mod, "_SYSTEM_ENDPOINT_KEYS", {})
    monkeypatch.setattr(basna_mod, "_SYSTEM_NO_FALLBACK", frozenset())
    monkeypatch.setattr(basna_mod, "_SYSTEM_KEYS_TS", 0.0)

    async def fake_merged(_db, _uid):
        return [dict(_ARCH)]

    monkeypatch.setattr(arch_mod, "merged_archetypes", fake_merged)
    yield fd_db
    await fd_db.close()


async def _owner_with_tier(db, uid: str, **tier) -> None:
    await db._db.execute(
        "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
        " VALUES (?, ?, 'h', ?, 'user', '2026-01-01', '2026-01-01')", (uid, f"{uid}@x.co", uid))
    await db._db.commit()
    tiers = {"balanced": {"provider": "openai", "model": "nvidia/Qwen3.8-27B-NVFP4",
                          "api_key": "", "base_url": "http://127.0.0.1:8000", **tier}}
    await db.set_settings(uid, {"fd:forge-tiers": json.dumps(
        {"sets": [{"id": "s1", "name": "MTPLX", "tiers": tiers}], "activeSetId": "s1"})})


@pytest.mark.asyncio
@pytest.mark.parametrize("tier,explicit,expected", [
    ({"vision": True}, None, "on"),
    ({}, None, "auto"),
    ({"vision": True}, "off", "off"),  # the caller's choice wins
])
async def test_a_tier_that_sees_images_turns_the_switch_on(db, tier, explicit, expected):
    await _owner_with_tier(db, "u1", **tier)
    cfg = (server.AgentConfig(name="x", archetype="analyst") if explicit is None
           else server.AgentConfig(name="x", archetype="analyst", vision=explicit))
    await server._resolve_archetype(cfg, _request(), None)
    assert cfg.model == "nvidia/Qwen3.8-27B-NVFP4"
    assert cfg.vision == expected
