"""An agent on its owner's own API keys marks its LLM usage as theirs.

cohort001 (2026-11-19): an owner moved an agent onto their own Anthropic key.
Its 174 calls were stored with ``byok=0``, so the deck's spend ledger counted
them as the deck's money. Flight Deck now starts such an agent with
``FD_KEY_SOURCE=byok`` and every usage row it writes says so.
"""

from __future__ import annotations

import pytest

from captain_claw.flight_deck import server
from captain_claw.session import SessionManager


async def _byok_flags(mgr: SessionManager) -> list[bool]:
    return [bool(r["byok"]) for r in await mgr.query_llm_usage(limit=10)]


@pytest.mark.asyncio
async def test_rows_say_own_key_when_the_deck_started_the_agent_on_one(tmp_path, monkeypatch):
    mgr = SessionManager(tmp_path / "s.db")
    monkeypatch.delenv("FD_KEY_SOURCE", raising=False)
    await mgr.record_llm_usage(session_id="s1", interaction="turn_1", provider="anthropic", model="m")
    await mgr.record_llm_usage(session_id="s1", interaction="turn_1", provider="openai", model="m", byok=True)
    monkeypatch.setenv("FD_KEY_SOURCE", "byok")
    await mgr.record_llm_usage(session_id="s1", interaction="reflection", provider="anthropic", model="m")
    assert sorted(await _byok_flags(mgr)) == [False, True, True]


def test_the_pin_is_set_and_never_inherited():
    env = {"FD_KEY_SOURCE": "byok", "PATH": "/bin"}
    server._pin_key_source(env, False)
    assert "FD_KEY_SOURCE" not in env
    server._pin_key_source(env, True)
    assert env["FD_KEY_SOURCE"] == "byok"


@pytest.fixture
def started(tmp_path, monkeypatch):
    """_start_registered_process with Popen recording the child's environment."""
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "_processes", {})
    monkeypatch.setenv("FD_PORT", "25999")
    seen: list[dict] = []

    class _Popen:
        def __init__(self, args, env=None, **kw):
            seen.append(dict(env or {}))
            self.pid = 4242

        def poll(self):
            return None

    monkeypatch.setattr(server.subprocess, "Popen", _Popen)
    (tmp_path / "ana").mkdir()
    # The agent's own .env can't claim it either way.
    (tmp_path / "ana" / ".env").write_text("FD_KEY_SOURCE=byok\n", encoding="utf-8")
    return seen


def test_an_agent_on_own_keys_starts_with_the_flag(started):
    assert server._start_registered_process("ana", {"web_port": 24990, "owner": "u1", "key_source": "byok"})
    assert started[-1]["FD_KEY_SOURCE"] == "byok"


def test_an_agent_on_the_decks_keys_starts_without_it(started):
    assert server._start_registered_process("ana", {"web_port": 24990, "owner": "u1"})
    assert "FD_KEY_SOURCE" not in started[-1]
