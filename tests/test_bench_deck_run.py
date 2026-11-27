"""captain-claw-once on a Flight Deck: the agent it asks for, and the run
itself against a fake deck (spawn → MCP task → transcript → clean-up)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from captain_claw.bench import deck_run as dr
from captain_claw.bench import run_once as ro


def _args(**kw):
    args = ro.build_parser().parse_args(["-p", "anthropic", "-m", "claude-haiku-5-5"])
    for key, value in kw.items():
        setattr(args, key, value)
    return args


def test_the_agent_sees_pictures_when_told_and_captain_decides_otherwise(tmp_path):
    assert dr.agent_config(_args(vision="on"), tmp_path, "b")["vision"] == "on"
    assert dr.agent_config(_args(vision="off"), tmp_path, "b")["vision"] == "off"
    assert "vision" not in dr.agent_config(_args(), tmp_path, "b")
    assert "vision" not in dr.agent_config(_args(vision="auto"), tmp_path, "b")


def test_provider_key_follows_the_provider():
    env = {"ANTHROPIC_API_KEY": "a", "GOOGLE_API_KEY": "g", "OPENAI_API_KEY": "o"}
    assert dr.provider_key("anthropic", env) == "a"
    assert dr.provider_key("Gemini", env) == "g"          # GEMINI_API_KEY missing → GOOGLE_API_KEY
    assert dr.provider_key("openrouter", env) == ""
    assert dr.provider_key("ollama", env) == ""


def test_agent_env_writes_in_place_unless_sandboxed_and_passes_search_keys():
    env = {"BRAVE_API_KEY": "b", "ANTHROPIC_API_KEY": "secret"}
    got = dr.agent_env_vars(_args(max_iterations=30), env)
    assert {"key": "CLAW_WRITE_DIRECT", "value": "1"} in got
    assert {"key": "CLAW_MAX_ITERATIONS", "value": "30"} in got
    assert {"key": "BRAVE_API_KEY", "value": "b"} in got
    assert all(e["key"] != "ANTHROPIC_API_KEY" for e in got)   # the model key goes in the config
    sandboxed = dr.agent_env_vars(_args(sandboxed_writes=True), env)
    assert all(e["key"] != "CLAW_WRITE_DIRECT" for e in sandboxed)


def test_agent_config_points_the_agent_at_the_task_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    body = dr.agent_config(_args(), tmp_path, "bench-x")
    assert body["workspace_path"] == str(tmp_path)
    assert body["provider"] == "anthropic" and body["model"] == "claude-haiku-5-5"
    assert body["provider_api_key"] == "k" and body["web_port"] == 0
    assert "base_url" not in body and "archetype" not in body
    assert dr.agent_config(_args(archetype="computer-user"), tmp_path, "b")["archetype"] == "computer-user"


def test_compact_event_and_usage():
    mon = dr.compact_event({"type": "monitor", "at": "t",
                            "data": {"tool_name": "read", "arguments": {"path": "a"}, "output": "x" * 5000}})
    assert mon["tool"] == "read" and len(mon["output"]) <= 2000
    assert dr.compact_event({"type": "status", "data": {"status": "sent"}})["status"] == "sent"
    assert dr.usage_of({"type": "usage", "total": {"prompt_tokens": 5, "note": "x"}, "last": {"prompt_tokens": 1}}) == {"prompt_tokens": 5}
    assert dr.usage_of({"last": {"completion_tokens": 2}}) == {"completion_tokens": 2}
    assert dr.usage_of(None) == {}


class FakeDeck:
    """A deck that answers on the second poll and writes the agent's file."""

    def __init__(self, workspace: Path, *, fail_spawn: bool = False, usage_fails: bool = False,
                 spawned_on: tuple[str, str] = ("anthropic", "claude-haiku-5-5")):
        self.workspace = workspace
        self.fail_spawn = fail_spawn
        self.usage_fails = usage_fails
        self.running_on = spawned_on   # what a store agent's tier may have picked
        self.calls: list[str] = []
        self.polls = 0

    def client(self, url, token=""):
        self.calls.append(f"connect {url} {token}")
        return self

    async def spawn_agent(self, body):
        self.calls.append("spawn")
        if self.fail_spawn:
            raise RuntimeError("spawn failed: no port")
        assert body["workspace_path"] == str(self.workspace)
        return {"ok": True, "slug": "bench-1"}

    async def wait_agent(self, slug):
        provider, model = self.running_on
        return {"slug": slug, "web_port": 24999, "web_auth": "t", "status": "running",
                "provider": provider, "model": model}

    async def set_effort(self, slug, level):
        self.calls.append(f"effort {level}")

    async def set_drive(self, slug, level):
        self.calls.append(f"drive {level}")

    async def set_model(self, slug, provider, model, api_key, base_url):
        self.calls.append(f"model {provider}/{model}")
        self.next_model = (provider, model)

    async def restart_agent(self, slug):
        self.calls.append(f"restart {slug}")
        self.running_on = self.next_model

    async def mcp(self, tool, arguments):
        self.calls.append(tool)
        if tool == "send_task":
            return {"task_id": "t1"}
        if tool == "get_result":
            self.polls += 1
            if self.polls == 1:
                return {"done": False, "next_cursor": 1, "events": [
                    {"type": "monitor", "data": {"tool_name": "read", "arguments": {"path": "Inbox/a.eml"}}}]}
            (self.workspace / "output").mkdir(exist_ok=True)
            (self.workspace / "output" / "answer.md").write_text("42")
            return {"done": True, "status": "done", "result": "The answer is 42.", "next_cursor": 1,
                    "usage": {"type": "usage", "total": {"prompt_tokens": 100, "completion_tokens": 7}}}
        return {}

    async def usage(self, info):
        if self.usage_fails:
            raise RuntimeError("no /api/usage")
        provider, model = self.running_on
        return {"totals": {"prompt_tokens": 300, "completion_tokens": 20, "total_tokens": 320,
                           "cache_read_input_tokens": 200, "total_calls": 3},
                "records": [{"provider": provider, "model": model, "prompt_tokens": 300}]}

    async def transcript(self, info):
        return [{"role": "user", "content": "q"}, {"role": "tool", "tool_name": "read"},
                {"role": "assistant", "content": "The answer is 42."}]

    async def stop_agent(self, slug):
        self.calls.append(f"stop {slug}")

    async def forget_agent(self, slug):
        self.calls.append(f"forget {slug}")

    async def close(self):
        self.calls.append("close")


@pytest.fixture
def no_poll_wait(monkeypatch):
    monkeypatch.setattr(dr, "_POLL_SECONDS", 0)


async def test_run_on_a_deck_collects_answer_usage_files_and_cleans_up(monkeypatch, tmp_path, no_poll_wait, capsys):
    ws = tmp_path / "app"
    ws.mkdir()
    deck = FakeDeck(ws)
    monkeypatch.setattr(dr, "DeckClient", deck.client)
    args = _args(deck_url="http://127.0.0.1:1", deck_token="pat", workspace=str(ws), effort="high",
                 drive="persistent", quiet=True)

    code = await dr.run_on_deck(args, "What is the answer?", tmp_path)

    assert code == 0
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["ok"] and result["answer"] == "The answer is 42."
    assert result["mode"] == "deck" and result["deck"]["agent"] == "bench-1"
    # Usage is what the agent recorded (turn + side calls), not the MCP frame.
    assert result["usage"]["prompt_tokens"] == 300 and result["usage"]["llm_calls"] == 3
    assert result["files"]["created"] == ["output/answer.md"]
    assert result["tools_used"] == {"read": 1}
    assert [e["tool"] for e in result["events"] if e["type"] == "monitor"] == ["read"]
    assert deck.calls[0] == "connect http://127.0.0.1:1 pat"
    assert "effort high" in deck.calls and "drive persistent" in deck.calls
    assert result["effort"] == "high" and result["drive"] == "persistent"
    assert result["models_used"] == ["anthropic/claude-haiku-5-5"]
    assert not any(c.startswith(("model ", "restart ")) for c in deck.calls)   # already on the asked model
    # A --deck-url deck keeps no benchmark agents on its list; nothing is purged.
    assert deck.calls[-3:] == ["stop bench-1", "forget bench-1", "close"]
    assert capsys.readouterr().out.strip() == "The answer is 42."


async def test_a_store_agent_runs_the_asked_model_not_its_tiers(monkeypatch, tmp_path, no_poll_wait):
    deck = FakeDeck(tmp_path, spawned_on=("anthropic", "claude-sonnet-4-6"))
    monkeypatch.setattr(dr, "DeckClient", deck.client)
    args = _args(workspace=str(tmp_path), archetype="inbox-manager", deck_url="http://x", quiet=True)

    assert await dr.run_on_deck(args, "q", tmp_path) == 0

    result = json.loads((tmp_path / "result.json").read_text())
    # The swap is undone before the task goes in.
    calls = deck.calls
    assert calls.index("model anthropic/claude-haiku-5-5") < calls.index("restart bench-1") < calls.index("send_task")
    assert result["store_agent"] == "inbox-manager"
    assert result["models_used"] == ["anthropic/claude-haiku-5-5"]
    assert any("claude-sonnet-4-6" in e.get("text", "") for e in result["events"])


async def test_usage_falls_back_to_the_mcp_frame(monkeypatch, tmp_path, no_poll_wait):
    deck = FakeDeck(tmp_path, usage_fails=True)
    monkeypatch.setattr(dr, "DeckClient", deck.client)
    await dr.run_on_deck(_args(deck_url="http://x", workspace=str(tmp_path), quiet=True), "q", tmp_path)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["usage"] == {"prompt_tokens": 100, "completion_tokens": 7}
    assert any("usage not read" in e.get("text", "") for e in result["events"])


def test_recorded_usage_prices_each_call_at_its_model(monkeypatch):
    seen = []

    def fake_cost(provider, model, usage):
        seen.append((provider, model, usage["prompt_tokens"]))
        return 0.5 if model == "big" else None

    monkeypatch.setattr("captain_claw.bench.run_once.estimate_cost", fake_cost)
    usage, cost, models = dr.recorded_usage({
        "totals": {"prompt_tokens": 30, "completion_tokens": 3, "total_calls": 2},
        "records": [{"provider": "anthropic", "model": "big", "prompt_tokens": 20},
                    {"model": "", "prompt_tokens": 10}],
    }, "anthropic", "small")
    assert usage["prompt_tokens"] == 30 and usage["llm_calls"] == 2
    assert seen == [("anthropic", "big", 20), ("anthropic", "small", 10)]
    assert cost == 0.5
    assert models == ["anthropic/big", "anthropic/small"]
    assert dr.recorded_usage({"totals": {}, "records": []}, "p", "m")[1] is None
    # A call that recorded "provider/model" is the same model, priced once.
    seen.clear()
    _, _, models = dr.recorded_usage({"totals": {}, "records": [
        {"provider": "anthropic", "model": "anthropic/big", "prompt_tokens": 5},
        {"provider": "anthropic", "model": "big", "prompt_tokens": 5}]}, "anthropic", "big")
    assert models == ["anthropic/big"] and seen == [("anthropic", "big", 5), ("anthropic", "big", 5)]


def test_store_lines_put_featured_first():
    lines = dr.store_lines([
        {"id": "zz-other", "family": "A", "role": "Other"},
        {"id": "inbox-manager", "family": "Ops", "role": "Inbox"},
        {"id": "deep-researcher", "family": "Research", "role": "Research"},
    ])
    assert [line[2:].split()[0] for line in lines] == ["deep-researcher", "inbox-manager", "zz-other"]
    assert lines[0].startswith("*") and lines[2].startswith(" ")


async def test_a_failed_spawn_still_writes_the_result_and_closes(monkeypatch, tmp_path, no_poll_wait):
    deck = FakeDeck(tmp_path, fail_spawn=True)
    monkeypatch.setattr(dr, "DeckClient", deck.client)
    args = _args(deck_url="http://127.0.0.1:1", workspace=str(tmp_path), quiet=True)

    code = await dr.run_on_deck(args, "q", tmp_path)

    assert code == 1
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["ok"] is False and "spawn failed" in result["error"]
    assert deck.calls[-1] == "close" and not any(c.startswith("stop") for c in deck.calls)


async def test_private_deck_is_started_and_stopped(monkeypatch, tmp_path, no_poll_wait):
    deck = FakeDeck(tmp_path)
    events: list[str] = []

    class FakePrivate:
        def __init__(self, run_dir, port=0, task_cap=0):
            self.url = "http://127.0.0.1:2"
            events.append(f"cap {task_cap:g}")

        async def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

    monkeypatch.setattr(dr, "PrivateDeck", FakePrivate)
    monkeypatch.setattr(dr, "DeckClient", deck.client)
    code = await dr.run_on_deck(_args(workspace=str(tmp_path), quiet=True), "q", tmp_path)

    # No --timeout: the harness sets the real limit, the deck allows hours.
    assert code == 0 and events == [f"cap {dr.DEFAULT_TASK_SECONDS:g}", "start", "stop"]
    assert "forget bench-1" not in deck.calls   # a private deck is thrown away whole


async def test_provider_and_model_are_required(tmp_path):
    args = _args(provider="", quiet=True)
    assert await dr.run_on_deck(args, "q", tmp_path) == 2


def test_stopping_the_private_deck_removes_the_keys_it_wrote(tmp_path):
    deck = dr.PrivateDeck(tmp_path, port=1234)
    agent = deck.root / "data" / "bench-1"
    agent.mkdir(parents=True)
    (agent / ".env").write_text("ANTHROPIC_API_KEY=x\n")
    (agent / "process.log").write_text("log")
    deck.stop()   # never started: only the clean-up runs
    assert not (agent / ".env").exists() and (agent / "process.log").exists()


def test_private_deck_env_is_its_own(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAW_WRITE_DIRECT", "1")
    env = dr.PrivateDeck(tmp_path, port=1234).env()
    assert env["FD_AUTH_ENABLED"] == "false"
    assert env["FD_DATA_DIR"] == str(tmp_path / "deck" / "data")
    assert env["HOME"] == str(tmp_path / "deck" / "home")
    assert "CLAW_WRITE_DIRECT" not in env


def test_lock_passes_the_forced_levels_to_the_agent():
    from captain_claw import drive
    from captain_claw.llm import reasoning

    # The names the agent reads (kept as literals in deck_run to stay light).
    assert dr.EFFORT_LOCK_ENV == reasoning.LOCK_ENV and dr.DRIVE_LOCK_ENV == drive.LOCK_ENV
    locked = dr.agent_env_vars(_args(effort="max", drive="ambitious", lock=True), {})
    assert {"key": "CLAW_LOCK_REASONING_EFFORT", "value": "max"} in locked
    assert {"key": "CLAW_LOCK_DRIVE", "value": "ambitious"} in locked
    # Without --lock the levels are only defaults the agent may change.
    loose = dr.agent_env_vars(_args(effort="max", drive="ambitious"), {})
    assert not any(e["key"].startswith("CLAW_LOCK_") for e in loose)
    assert dr.lock_env(_args(lock=True)) == {}       # nothing given, nothing forced


async def test_a_locked_run_says_so_in_the_result(monkeypatch, tmp_path, no_poll_wait):
    deck = FakeDeck(tmp_path)
    monkeypatch.setattr(dr, "DeckClient", deck.client)
    args = _args(deck_url="http://x", workspace=str(tmp_path), effort="max", drive="ambitious",
                 lock=True, quiet=True)
    assert await dr.run_on_deck(args, "q", tmp_path) == 0
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["locked"] is True and result["effort"] == "max" and result["drive"] == "ambitious"


def test_skill_archives_zip_each_skill_folder(tmp_path):
    import io
    import zipfile

    (tmp_path / "agent-browser" / "scripts").mkdir(parents=True)
    (tmp_path / "agent-browser" / "SKILL.md").write_text("---\nname: agent-browser\n---\nUse it.")
    (tmp_path / "agent-browser" / "scripts" / "run.sh").write_text("echo hi")
    (tmp_path / "notes").mkdir()                       # no SKILL.md: not a skill
    archives = dr.skill_archives(tmp_path)
    assert [name for name, _ in archives] == ["agent-browser"]
    names = zipfile.ZipFile(io.BytesIO(archives[0][1])).namelist()
    assert sorted(names) == ["agent-browser/SKILL.md", "agent-browser/scripts/run.sh"]
    assert dr.skill_archives(tmp_path / "missing") == []


def test_a_private_deck_lets_one_task_run_as_long_as_the_run_allows(tmp_path):
    env = dr.PrivateDeck(tmp_path, port=1, task_cap=5400).env()
    assert env["FD_MCP_TASK_MAX_SECONDS"] == "5460"


def test_the_deck_mcp_task_cap_follows_its_environment(monkeypatch):
    from captain_claw.flight_deck import mcp_server_routes as mcp

    monkeypatch.delenv("FD_MCP_TASK_MAX_SECONDS", raising=False)
    assert mcp._task_cap_seconds() == 1800.0
    monkeypatch.setenv("FD_MCP_TASK_MAX_SECONDS", "7200")
    assert mcp._task_cap_seconds() == 7200.0
    for bad in ("nonsense", "3"):
        monkeypatch.setenv("FD_MCP_TASK_MAX_SECONDS", bad)
        assert mcp._task_cap_seconds() == 1800.0


def test_a_local_model_endpoint_and_context_reach_the_agent(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "local-key")
    args = _args(provider="openai", model="qwen3.8-flash-next-iq3_s",
                 base_url="http://192.168.0.2:8080/v1", max_context=200_000, max_output=16_000)
    body = dr.agent_config(args, tmp_path, "bench-q")
    assert body["base_url"] == "http://192.168.0.2:8080/v1" and body["provider_api_key"] == "local-key"
    assert body["max_context"] == 200_000 and body["max_tokens"] == 16_000
    plain = dr.agent_config(_args(), tmp_path, "b")
    assert "max_context" not in plain and "max_tokens" not in plain
