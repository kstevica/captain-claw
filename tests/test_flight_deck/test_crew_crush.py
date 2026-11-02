"""Crush as a Crew worker: its turn script, read against a stand-in with Crush's own command line and
session JSON (checked against the real 0.99.0), and what it's given to run on."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

from captain_claw.flight_deck import (
    crew_crush_turn,
    crew_harness,
    crew_logins,
    crew_model,
    crew_runner,
    crew_store,
)
from captain_claw.flight_deck.crew_access import Grant
from captain_claw.flight_deck.crew_harness import HARNESSES, Route, RunSpec, TurnState, parse_line

from .test_crew_runner import _dispatch, _settle, deck  # noqa: F401
from .test_crew_runner import _run as _git

SCRIPT = Path(crew_crush_turn.__file__)
FAKE = r"""#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("crush version v0.99.0"); sys.exit(0)
assert args[0] == "-D", args
data = args[1]
path = os.path.join(data, "session.json")
rest = args[2:]
if rest[:1] == ["-c"]:
    rest = rest[2:]
if rest[0] == "session":
    if not os.path.exists(path):
        print("no sessions found", file=sys.stderr); sys.exit(1)
    found = json.load(open(path))
    print(json.dumps([{"uuid": found["meta"]["uuid"]}] if rest[1] == "list" else found))
    sys.exit(0)
assert rest[0] == "run", rest
with open(os.path.join(data, "ran.jsonl"), "a") as log:
    log.write(json.dumps({"args": rest, "env": os.environ.get("CREW_CRUSH_CONFIG", "none")}) + "\n")
script = json.loads(os.environ["FAKE_CRUSH"])
if "session" in script:
    json.dump(script["session"], open(path, "w"))
sys.stdout.write(script.get("stdout", ""))
sys.stderr.write(script.get("stderr", ""))
sys.exit(script.get("code", 0))
"""


def _message(mid: str, role: str, *parts: dict, model: str = "") -> dict:
    return {"id": mid, "role": role, "model": model, "parts": list(parts)}


def _session(*messages: dict, uuid: str = "u-1", cost: float = 0.0, tokens: tuple[int, int] = (0, 0)) -> dict:
    return {"meta": {"uuid": uuid, "cost": cost, "prompt_tokens": tokens[0], "completion_tokens": tokens[1]}, "messages": list(messages)}


CALL = {"type": "tool_call", "tool_call_id": "c1", "name": "bash", "input": '{"command": "ls"}'}
FINISH = {"type": "finish", "reason": "stop"}


@pytest.fixture
def crush(tmp_path, monkeypatch):
    """A stand-in on its own, in a folder of its own, like the real one's data folder."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "bin" / "crush"
    path.parent.mkdir()
    path.write_text(FAKE)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _run(binary: Path, data: Path, script: dict, *run: str, session: str = "", env: dict | None = None) -> tuple[list[dict], int]:
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-I", str(SCRIPT), "crew-turn", str(binary), str(data), session, "run", *run],
        capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL,
        env={"PATH": os.environ["PATH"], "FAKE_CRUSH": json.dumps(script), **(env or {})},
    )
    return [json.loads(line) for line in done.stdout.splitlines()], done.returncode


# ── What a read of the session tells ────────────────────────────────────


def test_each_thing_is_told_once():
    told: dict = {}
    said: list = []
    session = _session(_message("a", "assistant", CALL), _message("t", "tool", {"type": "tool_result", "name": "bash", "content": "x"}))
    assert crew_crush_turn._tell(session, told, say=said.append) == 2
    assert crew_crush_turn._tell(session, told, say=said.append) == 0
    assert [line["type"] for line in said] == ["tool_call", "tool_result"]


def test_what_is_still_being_written_waits():
    told: dict = {}
    said: list = []
    writing = _session(_message("a", "assistant", {"type": "text", "text": "Hel"}))
    half = _session(_message("a", "assistant", {"type": "tool_call", "name": "bash", "input": '{"comm'}))
    assert crew_crush_turn._tell(writing, told, say=said.append) == 0
    assert crew_crush_turn._tell(half, {}, say=said.append) == 0
    whole = _session(_message("a", "assistant", {"type": "text", "text": "Hello"}, FINISH))
    assert crew_crush_turn._tell(whole, told, say=said.append) == 1
    assert said == [{"type": "text", "text": "Hello"}]


def test_a_call_shows_while_it_runs():
    """A long command has no result yet, and its message isn't finished until it has."""
    said: list = []
    crew_crush_turn._tell(_session(_message("a", "assistant", CALL)), {}, say=said.append)
    assert said[0]["type"] == "tool_call"


def test_the_end_tells_what_never_finished():
    said: list = []
    crew_crush_turn._tell(_session(_message("a", "assistant", {"type": "text", "text": "Cut sh"})), {}, ended=True, say=said.append)
    assert said == [{"type": "text", "text": "Cut sh"}]


def test_nothing_of_the_persons_or_its_thinking_is_told():
    said: list = []
    crew_crush_turn._tell(_session(_message("u", "user", {"type": "text", "text": "the brief"}, FINISH),
                                   _message("a", "assistant", {"type": "reasoning", "thinking": "hm"}, FINISH)), {}, say=said.append)
    assert said == []


def test_a_huge_result_is_cut():
    said: list = []
    crew_crush_turn._tell(_session(_message("t", "tool", {"type": "tool_result", "name": "bash", "content": "x" * 50_000})), {}, say=said.append)
    assert len(said[0]["content"]) == crew_crush_turn._CONTENT_MAX


def test_its_error_comes_out_of_its_box():
    boxed = '\n          \n   ERROR  \n          \n  Agent processing failed: Post                                 \n  "http://x/v1": connection refused.  \n\n  Try --help for usage.\n'
    assert crew_crush_turn.why_it_failed(boxed) == 'Agent processing failed: Post "http://x/v1": connection refused.'
    assert crew_crush_turn.why_it_failed("\x1b[31mplain\x1b[0m") == "plain"


def test_a_turn_moving_with_nothing_to_show_says_so(monkeypatch):
    """How the watcher knows a long reply isn't a hung worker."""
    answers = [(_session(_message("a", "assistant", {"type": "text", "text": "Hel"})), 10),
               (_session(_message("a", "assistant", {"type": "text", "text": "Hello"})), 12),
               (_session(_message("a", "assistant", {"type": "text", "text": "Hello"})), 12)]
    said: list = []
    monkeypatch.setattr(crew_crush_turn, "_ask", lambda *_a: answers.pop(0))
    monkeypatch.setattr(crew_crush_turn, "_say", said.append)
    turn = crew_crush_turn._Turn("crush", "/d", "")
    turn.read(), turn.read(), turn.read()
    assert [line["type"] for line in said] == ["session", "alive", "alive"]


# ── A whole turn ────────────────────────────────────────────────────────


def test_a_turn(crush, tmp_path):
    session = _session(_message("a", "assistant", CALL, FINISH, model="m-1"),
                       _message("t", "tool", {"type": "tool_result", "name": "bash", "content": "ok"}),
                       _message("b", "assistant", {"type": "text", "text": "Done.\n\nCREW-STATUS: done it"}, FINISH, model="m-1"), tokens=(900, 40), cost=0.25)
    lines, code = _run(crush, tmp_path / "data", {"session": session, "stdout": "Done."}, "--quiet", "go")
    assert code == 0
    assert [line["type"] for line in lines] == ["session", "tool_call", "tool_result", "text"]
    assert lines[0] == {"type": "session", "uuid": "u-1", "model": "m-1", "cost": 0.25, "prompt_tokens": 900, "completion_tokens": 40}


def test_a_later_turn_tells_only_its_own(crush, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    old = _message("a", "assistant", {"type": "text", "text": "Earlier"}, FINISH)
    (data / "session.json").write_text(json.dumps(_session(old, cost=1.0)))
    new = _session(old, _message("b", "assistant", {"type": "text", "text": "Now"}, FINISH), cost=1.5)
    lines, _ = _run(crush, data, {"session": new}, "--session", "u-1", "go", session="u-1")
    assert [line.get("text") for line in lines if line["type"] == "text"] == ["Now"]
    assert [line["cost"] for line in lines if line["type"] == "session"] == [0.5], "its cost adds up over the session"


def test_a_new_turn_doesnt_take_an_old_session_for_its_own(crush, tmp_path):
    """One that failed before anyone heard its name is still there, and still the last."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "session.json").write_text(json.dumps(_session(_message("a", "assistant", {"type": "text", "text": "Old"}, FINISH))))
    lines, code = _run(crush, data, {"stderr": "\n   ERROR  \n\n  nope\n", "code": 1}, "go")
    assert code == 1
    assert [line["type"] for line in lines] == ["error"]


def test_its_words_are_kept_if_its_session_cant_be_read(crush, tmp_path):
    lines, code = _run(crush, tmp_path / "data", {"stdout": "Done.\n\nCREW-STATUS: done it\n"}, "go")
    assert code == 0
    assert lines == [{"type": "text", "text": "Done.\n\nCREW-STATUS: done it"}]


def test_a_failure_says_why_with_its_code(crush, tmp_path):
    lines, code = _run(crush, tmp_path / "data", {"stderr": '\n   ERROR  \n\n  Failed to override models: large model "x/y" not found.\n', "code": 1}, "go")
    assert code == 1
    assert lines == [{"type": "error", "text": 'Failed to override models: large model "x/y" not found.'}]


def test_a_failure_with_nothing_said_still_says_something(crush, tmp_path):
    lines, code = _run(crush, tmp_path / "data", {"code": 3}, "go")
    assert code == 3
    assert lines == [{"type": "error", "text": "Crush exited with code 3."}]


def test_providers_go_in_its_data_folder_and_stay_out_of_what_it_runs(crush, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "crush.json").write_text(json.dumps({"models": {"large": {"model": "m"}}, "providers": {"old": {"name": "gone"}}}))
    wanted = {"own": {"name": "Yours", "api_key": "$CREW_OWN_MODEL_KEY"}}
    _run(crush, data, {}, "go", env={"CREW_CRUSH_CONFIG": json.dumps(wanted)})
    saved = json.loads((data / "crush.json").read_text())
    assert saved == {"models": {"large": {"model": "m"}}, "providers": wanted}, "what Crush keeps there is Crush's"
    assert json.loads((data / "ran.jsonl").read_text())["env"] == "none", "the worker's commands don't get it"


def test_providers_dont_outlast_the_choice(crush, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "crush.json").write_text(json.dumps({"providers": {"crew": {"name": "last turn"}}}))
    _run(crush, data, {}, "go")
    assert json.loads((data / "crush.json").read_text())["providers"] == {}


# ── The adapter ─────────────────────────────────────────────────────────


def _argv(**fields) -> list[str]:
    return crew_harness._crush_argv(RunSpec(binary="/b/crush", prompt="go", cwd="/w", state_dir="/d", **fields))


def test_argv_of_a_first_turn():
    argv = _argv()
    assert argv[:3] == [sys.executable, "-I", str(SCRIPT)]
    assert argv[3:] == ["crew-turn", "/b/crush", "/d", "", "run", "--quiet", "go"]


def test_argv_goes_on_with_its_session_and_model():
    argv = _argv(session="u-1", model="anthropic/claude", effort="high")
    assert argv[3:] == ["crew-turn", "/b/crush", "/d", "u-1", "run", "--quiet", "--session", "u-1", "--model", "anthropic/claude", "--reasoning-effort", "high", "go"]


@pytest.mark.parametrize("fields", [{"model": "own/llama"}, {"model": "crew/m", "route": Route("http://d/v1", "m", "openai")}])
def test_a_model_it_was_told_about_gets_no_effort(fields):
    """Crush refuses to start on an effort a model doesn't list."""
    assert "--reasoning-effort" not in _argv(effort="high", **fields)


def test_a_prompt_cant_be_taken_for_a_flag():
    argv = crew_harness._crush_argv(RunSpec(binary="/b/crush", prompt="--yolo", cwd="/w", state_dir="/d"))
    assert argv[-1] == " --yolo"


def test_nothing_skips_crushs_own_list_of_refused_commands():
    assert not {"-y", "-yy", "--yolo"} & set(_argv())


def test_argv_in_the_desktop_app(monkeypatch):
    monkeypatch.setattr(sys, "_MEIPASS", "/app", raising=False)
    monkeypatch.setattr(crew_harness.shutil, "which", lambda name, path=None: "/usr/bin/python3" if name == "python3" else None)
    assert _argv()[0] == "/usr/bin/python3"
    monkeypatch.setattr(crew_harness.shutil, "which", lambda name, path=None: None)
    with pytest.raises(OSError, match="Python 3"):
        _argv()


def test_real_run_as_read():
    state = TurnState()
    events: list[dict] = []
    for line in (Path(__file__).parent / "crew_fixtures" / "crush.jsonl").read_text().splitlines():
        events.extend(parse_line("crush", line, state))
    assert state.session and state.model == "fake-1"
    assert state.tokens_in == 1002 and state.tokens_out == 50
    assert "CREW-STATUS: done" in state.final
    assert [e["kind"] for e in events] == ["tool", "result", "tool", "result", "text"]
    assert events[0]["tool"] == "bash" and events[0]["text"].startswith("echo hello")
    assert events[2]["tool"] == "view" and events[2]["text"] == "made.txt"


def test_a_failed_call_reads_as_one():
    events = parse_line("crush", json.dumps({"type": "tool_result", "name": "bash", "content": "boom", "is_error": True}), TurnState())
    assert events[0]["kind"] == "error" and "about" not in events[0]


def test_a_failed_turn_is_the_models_to_try_again():
    state = TurnState()
    events = parse_line("crush", json.dumps({"type": "error", "text": "connection refused"}), state)
    assert state.error == "connection refused" and events[0]["about"] == "model"
    assert crew_harness.passing(state.error)


def test_nothing_new_is_no_event():
    assert parse_line("crush", '{"type": "alive"}', TurnState()) == []


def test_its_turn_is_not_a_command_it_is_running(tmp_path, monkeypatch):
    """Otherwise a silent worker would never be taken for hung."""
    from captain_claw.flight_deck import crew_runner

    monkeypatch.setattr(crew_runner, "_descendants", lambda _pid: {41, 42})
    listing = f"41 /v/bin/python -I {SCRIPT} crew-turn /b/crush /d  run go\n42 /b/crush -D /d run go\n"
    monkeypatch.setattr(crew_runner.subprocess, "run", lambda *_a, **_k: subprocess.CompletedProcess([], 0, stdout=listing))
    assert not crew_runner._running_a_command({"pid": 40, "harness": "crush"})
    listing += "43 pytest -q\n"
    monkeypatch.setattr(crew_runner, "_descendants", lambda _pid: {41, 42, 43})
    assert crew_runner._running_a_command({"pid": 40, "harness": "crush"})


def test_its_telemetry_is_off():
    assert HARNESSES["crush"].env["CRUSH_DISABLE_METRICS"] == "1"
    assert HARNESSES["crush"].env["DO_NOT_TRACK"] == "1"


# ── Through the deck ────────────────────────────────────────────────────


WORKING = r"""#!/usr/bin/env python3
import json, os, subprocess, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("crush version v0.99.0"); sys.exit(0)
data, rest = args[1], args[2:]
if rest[:1] == ["-c"]:
    rest = rest[2:]
path = os.path.join(data, "session.json")
found = json.load(open(path)) if os.path.exists(path) else {"meta": {"uuid": "u-1", "cost": 0.0}, "messages": []}
if rest[0] == "session":
    if not found["messages"]:
        sys.exit(1)
    print(json.dumps([found["meta"]] if rest[1] == "list" else found))
    sys.exit(0)
open(os.path.join(data, "ran.log"), "a").write(json.dumps({"args": rest, "cwd": os.getcwd()}) + "\n")
open("work.txt", "a").write("x\n")
subprocess.run(["git", "add", "work.txt"], check=True)
subprocess.run(["git", "-c", "user.name=T", "-c", "user.email=t@t", "commit", "-qm", "work"], check=True)
n = len(found["messages"])
found["meta"].update(cost=found["meta"]["cost"] + 0.5, prompt_tokens=800, completion_tokens=30)
found["messages"] += [
    {"id": f"a{n}", "role": "assistant", "model": "m-1", "parts": [{"type": "tool_call", "name": "bash", "input": json.dumps({"command": "git commit"})}, {"type": "finish"}]},
    {"id": f"t{n}", "role": "tool", "parts": [{"type": "tool_result", "name": "bash", "content": "done"}]},
    {"id": f"b{n}", "role": "assistant", "model": "m-1", "parts": [{"type": "text", "text": f"Turn {n // 3 + 1}.\n\nCREW-STATUS: done made it"}, {"type": "finish"}]},
]
json.dump(found, open(path, "w"))
print("Turn.")
"""


async def test_a_task_through_the_deck(deck, monkeypatch, tmp_path):  # noqa: F811
    fake = tmp_path / "bin" / "crush"
    fake.write_text(WORKING)
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("FD_CREW_CRUSH_BIN", str(fake))
    grant = Grant(True, "admin", ("crush",), 4)
    task = await _dispatch(deck, "Do the work.", grant=grant, harness="crush", model="anthropic/claude-x")
    done = await _settle(task)
    assert done["status"] == "ready" and done["note"] == "made it"
    assert done["session"] == "u-1" and done["model_used"] == "m-1"
    assert done["tokens_in"] == 800 and done["tokens_out"] == 30 and done["cost_usd"] == pytest.approx(0.5)
    assert done["evidence"]["commits"] == 1

    data = crew_store.task_dir(done["user_id"], done["id"]) / "cli"
    ran = json.loads((data / "ran.log").read_text())
    assert Path(ran["cwd"]).resolve() == Path(done["worktree"]).resolve()
    assert ran["args"][:2] == ["run", "--quiet"] and ran["args"][ran["args"].index("--model") + 1] == "anthropic/claude-x"
    assert "--session" not in ran["args"]
    assert not (Path(done["worktree"]) / ".crush").exists() and not _git(Path(done["worktree"]), "status", "--short")
    kinds = [e["kind"] for e in crew_store.read_events(done["user_id"], done["id"]) if e["kind"] in ("tool", "result", "text")]
    assert kinds == ["tool", "result", "text"]

    await crew_runner.steer(done["user_id"], done["id"], "One more.")
    again = await _settle(crew_store.get_store().get(done["user_id"], done["id"]))
    assert again["status"] == "ready" and again["evidence"]["commits"] == 2
    ran = [json.loads(line) for line in (data / "ran.log").read_text().splitlines()]
    assert ran[1]["args"][ran[1]["args"].index("--session") + 1] == "u-1"
    kinds = [e["kind"] for e in crew_store.read_events(done["user_id"], done["id"]) if e["kind"] in ("tool", "result", "text")]
    assert kinds == ["tool", "result", "text"] * 2, "the first turn isn't told again"
    assert again["cost_usd"] == pytest.approx(1.0), "its cost is counted once"


# ── What it runs on ─────────────────────────────────────────────────────


def test_own_model_config_names_only_the_variable():
    own = {"url": "http://h:8080/v1", "key": "s3cret", "models": ["qwen", "org/big"], "context": 40000}
    config = crew_logins.crush_own_config(own)
    assert "s3cret" not in json.dumps(config)
    assert config["own"]["api_key"] == "$CREW_OWN_MODEL_KEY"
    assert config["own"]["type"] == "openai-compat" and config["own"]["base_url"] == own["url"]
    assert config["own"]["models"][1] == {"id": "org/big", "name": "org/big", "context_window": 40000, "default_max_tokens": 10000}


def test_own_model_with_no_key_or_size():
    config = crew_logins.crush_own_config({"url": "http://h/v1", "key": "", "models": ["m"], "context": 0})
    assert "api_key" not in config["own"]
    assert config["own"]["models"] == [{"id": "m", "name": "m"}], "with no size it never compacts, and sets no limit"


def test_route_config_on_openais_api():
    config = crew_logins.crush_route_config(Route("http://d/fd/crew/llm/v1", "m", "openai", context=100000, output=9000))["crew"]
    assert config["type"] == "openai-compat" and config["base_url"] == "http://d/fd/crew/llm/v1"
    assert config["api_key"] == "$CREW_MODEL_TOKEN"
    assert config["models"] == [{"id": "m", "name": "m", "context_window": 100000, "default_max_tokens": 9000}]


def test_route_config_on_anthropics_api():
    """Crush adds /v1/messages itself, as seen on the real one, and Anthropic's API requires a limit."""
    config = crew_logins.crush_route_config(Route("http://d/fd/crew/llm/v1", "m", "anthropic"))["crew"]
    assert config["type"] == "anthropic" and config["base_url"] == "http://d/fd/crew/llm"
    assert config["models"] == [{"id": "m", "name": "m", "default_max_tokens": 16384}]


def test_route_name():
    assert crew_harness.route_model("crush", Route("http://d/v1", "m", "openai")) == "crew/m"


def test_crush_talks_to_both_apis():
    assert crew_model._CLI_APIS["crush"] == (crew_model.OPENAI, crew_model.ANTHROPIC)
    assert "crush" in crew_logins.OWN_HARNESSES


@pytest.mark.parametrize("url", ["http://h/$(whoami)/v1", "http://h/$HOME", "http://h/`id`"])
def test_an_address_cant_run_anything(url):
    with pytest.raises(HTTPException):
        crew_logins.clean_url(url)


class _Db:
    def __init__(self, settings: dict | None = None) -> None:
        self.settings = settings or {}

    async def get_setting(self, _uid, name):
        return self.settings.get(name)


@pytest.fixture
def worker(tmp_path, monkeypatch):
    from captain_claw.flight_deck import byok, crew_store

    crew_store.set_root(tmp_path)
    monkeypatch.chdir(tmp_path)

    async def no_keys(_db, _uid):
        return {}

    monkeypatch.setattr(byok, "read_keys", no_keys)
    yield {"id": "u1", "email": "u@x"}
    crew_store.set_root(None)


NOT_ADMIN = Grant(True, "approved", ("crush",), 2)


@pytest.mark.asyncio
async def test_no_login_no_turn(worker):
    state = await crew_logins.login_state(_Db(), worker, NOT_ADMIN, "crush")
    assert state == {"ok": False, "how": ""}, "Crush has no free models to fall back on"
    with pytest.raises(HTTPException, match="Crush"):
        await crew_logins.worker_env(_Db(), worker, NOT_ADMIN, "crush", "/b/crush")


@pytest.mark.asyncio
async def test_a_routed_turn_is_given_the_address_and_the_key_apart(worker):
    route = Route("http://127.0.0.1:1/fd/crew/llm/v1", "m", "openai")
    env = await crew_logins.worker_env(_Db(), worker, NOT_ADMIN, "crush", "/b/crush", route=route, token="tok-1")
    assert env["CREW_MODEL_TOKEN"] == "tok-1"
    assert "tok-1" not in env[crew_logins.CRUSH_CONFIG_ENV]
    assert json.loads(env[crew_logins.CRUSH_CONFIG_ENV])["crew"]["base_url"] == route.url
    assert env["CRUSH_DISABLE_METRICS"] == "1"
    assert env["XDG_CONFIG_HOME"].startswith(env["HOME"]), "its own home, not the deck user's"


@pytest.mark.asyncio
async def test_a_provider_key_reaches_it(worker, monkeypatch):
    async def logins(_db, _uid):
        return {"anthropic": "sk-a"}

    monkeypatch.setattr(crew_logins, "read_logins", logins)
    assert (await crew_logins.login_state(_Db(), worker, NOT_ADMIN, "crush"))["ok"]
    env = await crew_logins.worker_env(_Db(), worker, NOT_ADMIN, "crush", "/b/crush")
    assert env["ANTHROPIC_API_KEY"] == "sk-a"
    assert crew_logins.CRUSH_CONFIG_ENV not in env


@pytest.mark.asyncio
async def test_their_own_model_reaches_it(worker, monkeypatch):
    own = {"url": "http://h/v1", "key": "s3cret", "models": ["m"], "context": 0}

    async def read_own(_db, _uid):
        return own

    monkeypatch.setattr(crew_logins, "read_own_model", read_own)
    assert (await crew_logins.login_state(_Db(), worker, NOT_ADMIN, "crush"))["how"] == "your own model"
    env = await crew_logins.worker_env(_Db(), worker, NOT_ADMIN, "crush", "/b/crush")
    assert env["CREW_OWN_MODEL_KEY"] == "s3cret"
    assert "s3cret" not in env[crew_logins.CRUSH_CONFIG_ENV]
    assert crew_logins.effective_model({}, own, "crush") == "own/m"
