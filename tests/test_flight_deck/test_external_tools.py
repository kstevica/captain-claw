"""Custom tools, Phase 0 (docs/external-tools-plan.md): Flight Deck keeps a tool's
repository, publishes a commit only when its tests and examples pass, and runs
one process per call with an environment built from nothing."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import admin_routes, external_tools_routes, fd_crypto, settings_routes
from captain_claw.flight_deck import auth as auth_mod
from captain_claw.flight_deck import external_tools as xt
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.flight_deck.db import FlightDeckDB

GIT = ["git", "-c", "user.name=T", "-c", "user.email=t@example.com"]
LOOPBACK = ("127.0.0.1", 40001)


@pytest.fixture
def data(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr("captain_claw.flight_deck.server.DATA_DIR", tmp_path / "fd-data")
    monkeypatch.setenv("FD_DATA_DIR", str(tmp_path / "fd-data"))
    monkeypatch.setattr(xt, "_last_sweep", 0.0)
    xt.reset_for_tests()
    yield tmp_path
    xt.reset_for_tests()


def _commit(owner: str, slug: str, files: dict[str, str | None], message: str = "change") -> str:
    repo = xt.repo_dir(owner, slug)
    for rel, text in files.items():
        p = repo / rel
        if text is None:
            p.unlink(missing_ok=True)
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    subprocess.run([*GIT, "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run([*GIT, "-C", str(repo), "commit", "-q", "-m", message], check=True, capture_output=True)
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


def _manifest(slug: str, **over) -> str:
    m = {
        "name": slug, "title": slug.title(), "description": "A test tool.", "entry": "main.py",
        "input": {"type": "object"}, "output": {}, "examples": [{"input": {}}],
        "secrets": [], "network": False, "timeout_s": 20, "memory_mb": 1024,
    }
    m.update(over)
    return json.dumps(m)


async def _tool(owner: str, slug: str, main_py: str, *, publish: bool = True, **manifest) -> dict:
    """A tool whose main.py is *main_py*, published unless told not to."""
    await xt.create_tool(owner, slug, title=slug.title(), description="A test tool.")
    _commit(owner, slug, {"main.py": main_py, "tool.json": _manifest(slug, **manifest), "tests/test_main.py": None})
    if not publish:
        return {}
    state = await xt.publish(owner, slug, by=owner)
    assert state["state"] == "passed", state
    return state


async def _as_published(owner: str, slug: str) -> dict:
    """The starter tool, made and published."""
    await xt.create_tool(owner, slug, title=slug.title(), description="A test tool.")
    state = await xt.publish(owner, slug, by=owner)
    assert state["state"] == "passed", state
    return state


# ── Creating and publishing ──────────────────────────────────────────────


async def test_the_starter_publishes_and_runs(data):
    made = await xt.create_tool("u1", "echo", title="Echo", description="Says it back.")
    assert made["live"] is None and made["title"] == "Echo"
    state = await xt.publish("u1", "echo", by="u1")
    assert state["state"] == "passed", state
    assert [s["step"] for s in state["steps"]] == ["commit", "copy", "check tool.json", "install", "tests", "examples", "live"]
    res = await xt.run_tool("u1", "echo", "u1", {"text": "zdravo"})
    assert res["ok"] and res["output"] == {"text": "zdravo"} and res["version"] == state["version"]
    detail = xt.detail("u1", "echo")
    assert detail["live"]["version"] == state["version"]
    assert detail["manifest"]["title"] == "Echo"
    # The live version's files can't be written by accident.
    vdir = xt.versions_dir("u1", "echo") / state["version"]
    assert not os.access(vdir / "main.py", os.W_OK) or os.geteuid() == 0
    assert not (vdir / ".git").exists()


async def test_a_failing_example_keeps_the_old_version_live(data):
    first = await _tool("u1", "add", "import json,sys\nd=json.load(sys.stdin)\nprint(json.dumps({'sum': d['a']+d['b']}))\n",
                        examples=[{"input": {"a": 1, "b": 2}, "expect": {"sum": 3}}])
    _commit("u1", "add", {"main.py": "import json,sys\nd=json.load(sys.stdin)\nprint(json.dumps({'sum': d['a']-d['b']}))\n"})
    second = await xt.publish("u1", "add", by="u1")
    assert second["state"] == "failed" and "example" in second["error"]
    assert "doesn't contain what 'expect' says" in second["steps"][-1]["detail"]
    assert xt.read_live("u1", "add")["version"] == first["version"]
    # The failed build isn't kept as something to go back to.
    assert not (xt.versions_dir("u1", "add") / second["version"]).exists()
    res = await xt.run_tool("u1", "add", "u1", {"a": 5, "b": 1})
    assert res["output"] == {"sum": 6}


async def test_failing_tests_stop_a_publish(data):
    await xt.create_tool("u1", "t", title="T", description="d")
    _commit("u1", "t", {"tests/test_main.py": "import unittest\nclass T(unittest.TestCase):\n    def test_no(self):\n        self.fail('nope')\n"})
    state = await xt.publish("u1", "t", by="u1")
    assert state["state"] == "failed" and state["error"] == "Its tests failed."
    assert "nope" in state["steps"][-1]["detail"]
    assert xt.read_live("u1", "t") is None


@pytest.mark.parametrize("change, words", [
    ({"examples": []}, "'examples' must list"),
    ({"runtime": "node"}, "custom tools are Python programs"),
    ({"entry": "../main.py"}, "must be a path inside the tool"),
    ({"entry": "missing.py"}, "isn't a file in the tool"),
    ({"secrets": [{"name": "PATH", "why": ""}]}, "not one Flight Deck sets"),
    ({"secrets": [{"name": "FD_JWT_SECRET", "why": ""}]}, "not one Flight Deck sets"),
    ({"input": {"type": "array"}}, "must describe a JSON object"),
    ({"timeout_s": 9999}, "'timeout_s' must be a whole number"),
    ({"name": "other"}, "but the tool is"),
])
async def test_a_bad_tool_json_is_refused(data, change, words):
    await xt.create_tool("u1", "bad", title="Bad", description="d")
    _commit("u1", "bad", {"tool.json": _manifest("bad", **change), "tests/test_main.py": None,
                          "main.py": "print('{}')\n"})
    state = await xt.publish("u1", "bad", by="u1")
    assert state["state"] == "failed"
    assert words in state["error"], state["error"]


async def test_dependencies_come_from_requirements(data):
    # A local package, so the test needs no network: pip/uv install it into the venv.
    pkg = data / "pkg"
    (pkg / "xtdemo").mkdir(parents=True)
    (pkg / "xtdemo" / "__init__.py").write_text("VALUE = 42\n")
    (pkg / "pyproject.toml").write_text('[project]\nname = "xtdemo"\nversion = "0.1"\n'
                                        '[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n')
    await xt.create_tool("u1", "deps", title="Deps", description="d")
    _commit("u1", "deps", {
        "requirements.txt": f"{pkg}\n",
        "main.py": "import json, xtdemo\nprint(json.dumps({'v': xtdemo.VALUE}))\n",
        "tool.json": _manifest("deps", examples=[{"input": {}, "expect": {"v": 42}}]),
        "tests/test_main.py": None,
    })
    state = await xt.publish("u1", "deps", by="u1")
    if state["state"] != "passed" and "setuptools" in json.dumps(state["steps"]):
        pytest.skip("no setuptools reachable to build the local package")
    assert state["state"] == "passed", state
    assert (await xt.run_tool("u1", "deps", "u1", {}))["output"] == {"v": 42}


async def test_going_back_to_a_kept_version(data):
    one = await _tool("u1", "ver", "print('{\"v\": 1}')\n", examples=[{"input": {}, "expect": {"v": 1}}])
    _commit("u1", "ver", {"main.py": "print('{\"v\": 2}')\n",
                          "tool.json": _manifest("ver", examples=[{"input": {}, "expect": {"v": 2}}])})
    two = await xt.publish("u1", "ver", by="u1")
    assert two["state"] == "passed"
    assert (await xt.run_tool("u1", "ver", "u1", {}))["output"] == {"v": 2}
    assert [v["version"] for v in xt.kept_versions("u1", "ver")] == [two["version"], one["version"]]
    await xt.make_live("u1", "ver", one["version"])
    assert (await xt.run_tool("u1", "ver", "u1", {}))["output"] == {"v": 1}
    assert xt.read_live("u1", "ver")["previous"] == [two["version"]]
    with pytest.raises(xt.ToolError) as exc:
        await xt.make_live("u1", "ver", "0" * 12)
    assert exc.value.status == 404


async def test_names_are_checked(data):
    for bad in ("../x", "A", "a" * 49, "-a", "a-", "a/b", ""):
        with pytest.raises(xt.ToolError):
            await xt.create_tool("u1", bad, title="x", description="y")
    with pytest.raises(xt.ToolError):
        xt.tool_dir("../other", "ok")
    await xt.create_tool("u1", "dup", title="x", description="y")
    with pytest.raises(xt.ToolError) as exc:
        await xt.create_tool("u1", "dup", title="x", description="y")
    assert exc.value.status == 409


async def test_deleting_removes_read_only_folders(data):
    await _tool("u1", "gone", "print('{}')\n")
    await xt.delete_tool("u1", "gone")
    assert not xt.tool_dir("u1", "gone").exists()
    assert xt.list_tools("u1") == []


# ── Running ──────────────────────────────────────────────────────────────


async def test_the_environment_carries_nothing_of_flight_decks(data, monkeypatch):
    monkeypatch.setenv("FD_AGENT_SECRET", "agent-secret")
    monkeypatch.setenv("FD_JWT_SECRET", "jwt-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-deck")
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    await _tool("u1", "env", "import json, os, sys\nprint(json.dumps({'env': dict(os.environ), 'cwd': os.getcwd(), 'exe': sys.executable}))\n")
    res = await xt.run_tool("u1", "env", "u1", {})
    env = res["output"]["env"]
    assert not {"FD_AGENT_SECRET", "FD_JWT_SECRET", "OPENAI_API_KEY", "PYTHONPATH", "FD_DATA_DIR"} & set(env)
    vdir = (xt.versions_dir("u1", "env") / res["version"]).resolve()
    assert Path(env["PATH"].split(os.pathsep)[0]).resolve() == (vdir / ".deps" / "bin").resolve()
    assert Path(res["output"]["cwd"]).resolve() == vdir
    assert env["XT_CALL"] == res["call_id"]
    assert Path(env["HOME"]).name == "home" and res["call_id"] in env["HOME"]
    # The call's folder is gone afterwards.
    assert not Path(env["XT_OUT"]).exists()


async def test_secrets_are_the_callers_and_required(data):
    await _tool("u1", "sec", "import json, os\nprint(json.dumps({'t': os.environ.get('API_TOKEN')}))\n",
                secrets=[{"name": "API_TOKEN", "why": "calls the API"}], publish=False)
    state = await xt.publish("u1", "sec", by="u1", secret_values={"API_TOKEN": "owner"})
    assert state["state"] == "passed", state
    with pytest.raises(xt.ToolError) as exc:
        await xt.run_tool("u1", "sec", "u2", {})
    assert exc.value.kind == "missing_secret" and exc.value.message.startswith("API_TOKEN isn't set: Custom tools → ")
    # It names the section the tool's sheet shows.
    assert exc.value.message.endswith("→ Your secrets for it.")
    res = await xt.run_tool("u1", "sec", "u2", {}, secret_values={"API_TOKEN": "runner", "OTHER": "x"})
    assert res["output"] == {"t": "runner"}


async def test_a_publish_without_a_needed_secret_fails_on_the_example(data):
    await _tool("u1", "sec2", "import json, os\nprint(json.dumps({'t': os.environ['API_TOKEN']}))\n",
                secrets=[{"name": "API_TOKEN", "why": ""}], publish=False)
    state = await xt.publish("u1", "sec2", by="u1")
    assert state["state"] == "failed" and "API_TOKEN isn't set" in state["steps"][-1]["detail"]
    await _tool("u1", "sec3", "import json, os\nprint(json.dumps({'t': os.environ['API_TOKEN']}))\n",
                secrets=[{"name": "API_TOKEN", "why": ""}, {"name": "OTHER", "why": ""}], publish=False)
    state = await xt.publish("u1", "sec3", by="u1")
    assert state["state"] == "failed" and "API_TOKEN, OTHER aren't set" in state["steps"][-1]["detail"]


async def test_input_must_fit_the_schema(data):
    await xt.create_tool("u1", "echo", title="Echo", description="d")
    await xt.publish("u1", "echo", by="u1")
    with pytest.raises(xt.ToolError) as exc:
        await xt.run_tool("u1", "echo", "u1", {"text": 5})
    assert exc.value.status == 422 and "text" in exc.value.message


@pytest.mark.parametrize("main_py, kind", [
    ("import sys\nprint('oops', file=sys.stderr)\nsys.exit(3)\n", "crashed"),
    ("print('not json')\n", "bad_output"),
    ("import sys\nsys.stdout.write('x' * 1_200_000)\n", "too_large"),
])
async def test_a_tool_that_fails_is_a_result(data, main_py, kind):
    await _tool("u1", "f", "import json,sys\nd=json.load(sys.stdin)\nif d.get('go'):\n" +
                "".join(f"    {line}\n" for line in main_py.splitlines()) + "print('{}')\n")
    res = await xt.run_tool("u1", "f", "u1", {"go": True})
    assert res["ok"] is False and res["error"]["kind"] == kind, res
    if kind == "crashed":
        assert "oops" in res["stderr"] and res["exit_code"] == 3


async def test_output_must_fit_what_it_promised(data):
    await _tool("u1", "o", "import json,sys\nd=json.load(sys.stdin)\nprint(json.dumps({'n': 'x' if d.get('bad') else 1}))\n",
                output={"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]})
    res = await xt.run_tool("u1", "o", "u1", {"bad": True})
    assert res["ok"] is False and res["error"]["kind"] == "bad_output" and "(n)" in res["error"]["message"]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _gone(pid: int) -> bool:
    end = time.monotonic() + 5
    while time.monotonic() < end:
        if not _alive(pid):
            return True
        await asyncio.sleep(0.05)
    return False


async def test_a_timeout_stops_the_tool_and_what_it_started(data):
    main_py = ("import json, subprocess, sys, time\n"
               "d = json.load(sys.stdin)\n"
               "if d.get('slow'):\n"
               "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
               "    print(child.pid, file=sys.stderr, flush=True)\n"
               "    time.sleep(60)\n"
               "print('{}')\n")
    await _tool("u1", "slow", main_py, timeout_s=1)
    started = time.monotonic()
    res = await xt.run_tool("u1", "slow", "u1", {"slow": True})
    assert time.monotonic() - started < 10
    assert res["ok"] is False and res["error"]["kind"] == "timeout"
    assert await _gone(int(res["stderr"].strip().splitlines()[-1]))


async def test_a_child_left_holding_stdout_doesnt_hold_the_call(data):
    main_py = ("import json, subprocess, sys\n"
               "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
               "print(json.dumps({'child': child.pid}))\n")
    await _tool("u1", "left", main_py)
    started = time.monotonic()
    res = await xt.run_tool("u1", "left", "u1", {})
    assert res["ok"], res
    assert time.monotonic() - started < 8
    assert await _gone(res["output"]["child"])


async def test_the_tool_cant_write_its_own_folder(data):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores file modes")
    main_py = ("import json, sys\nd = json.load(sys.stdin)\n"
               "if d.get('w'):\n"
               "    try:\n        open('planted.py', 'w').write('x')\n        print(json.dumps({'wrote': True}))\n"
               "    except OSError as e:\n        print(json.dumps({'wrote': False}))\n"
               "else:\n    print('{}')\n")
    await _tool("u1", "ro", main_py)
    res = await xt.run_tool("u1", "ro", "u1", {"w": True})
    assert res["output"] == {"wrote": False}


async def test_files_in_and_out(data):
    main_py = ("import json, os, sys\nfrom pathlib import Path\n"
               "d = json.load(sys.stdin)\n"
               "if 'doc' not in d:\n    print('{}')\n    raise SystemExit\n"
               "text = Path(d['doc']).read_text()\n"
               "Path(os.environ['XT_OUT'], 'upper.txt').write_text(text.upper())\n"
               "Path(os.environ['XT_OUT'], 'sub').mkdir()\n"
               "Path(os.environ['XT_OUT'], 'sub', 'n.txt').write_text(str(len(text)))\n"
               "print(json.dumps({'chars': len(text), 'inside': d['doc'].startswith(os.environ['XT_IN'])}))\n")
    await _tool("u1", "file", main_py, input={"type": "object", "properties": {"doc": {"type": "string", "format": "file"}}})
    src = data / "hello.txt"
    src.write_text("hello")
    got: list = []

    async def deliver(call_id, files):
        got.append((call_id, [(rel, Path(p).read_text()) for rel, p in files]))
        return [{"name": rel} for rel, _ in files]

    res = await xt.run_tool("u1", "file", "u1", {"doc": "hello.txt"}, resolve_file=lambda ref: data / ref, deliver=deliver)
    assert res["output"] == {"chars": 5, "inside": True}
    assert [(c, sorted(f)) for c, f in got] == [(res["call_id"], [("sub/n.txt", "5"), ("upper.txt", "HELLO")])]
    with pytest.raises(xt.ToolError):
        await xt.run_tool("u1", "file", "u1", {"doc": "x"})  # nobody to resolve files


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS refuses RLIMIT_AS")
async def test_memory_is_limited(data):
    main_py = ("import json, sys\nd = json.load(sys.stdin)\n"
               "if d.get('big'):\n    x = bytearray(1024 * 1024 * 1024)\n"
               "print('{}')\n")
    await _tool("u1", "mem", main_py, memory_mb=256)
    res = await xt.run_tool("u1", "mem", "u1", {"big": True})
    assert res["ok"] is False and res["error"]["kind"] == "crashed" and "MemoryError" in res["stderr"]


async def test_an_unpublished_tool_doesnt_run(data):
    await xt.create_tool("u1", "new", title="New", description="d")
    with pytest.raises(xt.ToolError) as exc:
        await xt.run_tool("u1", "new", "u1", {"text": "x"})
    assert exc.value.kind == "not_live"


# ── Over HTTP ────────────────────────────────────────────────────────────


@pytest.fixture
async def deck(data, monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_DATA_KEY_FILE", str(data / "data_key"))
    monkeypatch.delenv("FD_CREW", raising=False)
    fd_crypto.reset_for_tests()
    db = FlightDeckDB(data / "fd.db")
    await db.init()
    monkeypatch.setattr(auth_mod, "_db", db)
    admin = await db.create_user("admin@example.com", "x", role="admin")
    user = await db.create_user("u@example.com", "x", role="user")
    other = await db.create_user("o@example.com", "x", role="user")
    yield types.SimpleNamespace(db=db, admin=admin, user=user, other=other)
    await db.close()


def _client(user: dict) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(external_tools_routes.router)
    app.include_router(admin_routes.router)
    app.include_router(settings_routes.router)
    app.dependency_overrides[get_current_user] = lambda: user
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=LOOPBACK), base_url="http://deck.test")


async def _switch_on(deck) -> None:
    async with _client(deck.admin) as c:
        resp = await c.put("/fd/admin/config", json={"external_tools_enabled": True})
    assert resp.status_code == 200 and resp.json()["external_tools_enabled"] is True


async def _published(c: httpx.AsyncClient, slug: str) -> dict:
    resp = await c.post(f"/fd/external-tools/{slug}/publish", json={})
    assert resp.status_code == 202, resp.text
    end = time.monotonic() + 30
    while time.monotonic() < end:
        state = (await c.get(f"/fd/external-tools/{slug}/publish")).json()["publish"]
        if state["state"] != "running":
            return state
        await asyncio.sleep(0.05)
    raise AssertionError("publish never finished")


async def test_off_until_an_admin_turns_it_on(deck):
    async with _client(deck.admin) as c:
        status = (await c.get("/fd/external-tools/status")).json()
        assert {k: status[k] for k in ("enabled", "everybody_needs_approval", "can_build", "require_sandbox")} == {
            "enabled": False, "everybody_needs_approval": True, "can_build": False, "require_sandbox": False}
        resp = await c.get("/fd/external-tools")
        assert resp.status_code == 403 and "Admin → Settings" in resp.json()["detail"]
        config = (await c.get("/fd/admin/config")).json()
        assert config["external_tools_enabled"] is False and config["external_tools_everybody_approval"] is True
    async with _client(deck.user) as c:
        assert (await c.put("/fd/admin/config", json={"external_tools_enabled": True})).status_code == 403
    await _switch_on(deck)
    async with _client(deck.admin) as c:
        assert (await c.get("/fd/external-tools/status")).json()["can_build"] is True
        assert (await c.get("/fd/external-tools")).json()["tools"] == []
        resp = await c.put("/fd/admin/config", json={"external_tools_everybody_approval": False})
        assert resp.json()["external_tools_everybody_approval"] is False


async def test_making_a_tool_needs_crew(deck):
    await _switch_on(deck)
    async with _client(deck.user) as c:
        assert (await c.get("/fd/external-tools/status")).json()["can_build"] is False
        resp = await c.post("/fd/external-tools", json={"slug": "mine", "title": "Mine", "description": "d"})
        assert resp.status_code == 403 and "Crew" in resp.json()["detail"]
    async with _client(deck.admin) as c:
        resp = await c.post("/fd/external-tools", json={"slug": "mine", "title": "Mine", "description": "d"})
        assert resp.status_code == 200, resp.text


async def test_a_whole_tool_over_http(deck):
    await _switch_on(deck)
    uid = deck.admin["id"]
    async with _client(deck.admin) as c:
        assert (await c.post("/fd/external-tools", json={"slug": "up", "title": "Up", "description": "Upper-cases a file."})).status_code == 200
        _commit(uid, "up", {
            "main.py": ("import json, os, sys\nfrom pathlib import Path\nd = json.load(sys.stdin)\n"
                        "t = Path(d['doc']).read_text() if d.get('doc') else 'x'\n"
                        "Path(os.environ['XT_OUT'], 'up.txt').write_text(t.upper())\n"
                        "print(json.dumps({'n': len(t), 'key': os.environ['KEY']}))\n"),
            "tool.json": _manifest("up", input={"type": "object", "properties": {"doc": {"type": "string", "format": "file"}}},
                                   secrets=[{"name": "KEY", "why": "for the test"}]),
            "tests/test_main.py": None,
        })
        resp = await c.put("/fd/external-tools/up/secrets", json={"values": {"KEY": "k-123"}})
        assert resp.status_code == 200 and resp.json()["secrets"] == [{"name": "KEY", "why": "for the test", "set": True}]
        assert "k-123" not in resp.text
        assert (await c.put("/fd/external-tools/up/secrets", json={"values": {"NOPE": "x"}})).status_code == 400
        # The settings API neither shows nor takes a tool's secrets.
        assert not [k for k in (await c.get("/fd/settings")).json() if k.startswith("xtool:")]
        assert (await c.put("/fd/settings", json={"settings": {f"xtool:{uid}/up:KEY": "x"}})).status_code == 400
        state = await _published(c, "up")
        assert state["state"] == "passed", state
        notes = Path(xt.root()).parent / "vfs" / uid / "notes"
        notes.mkdir(parents=True)
        (notes / "a.txt").write_text("hello")
        resp = await c.post("/fd/external-tools/up/run", json={"input": {"doc": "vfs:notes/a.txt"}})
        body = resp.json()
        assert resp.status_code == 200 and body["ok"], body
        assert body["output"] == {"n": 5, "key": "k-123"}
        [f] = body["files"]
        assert f["path"].startswith("vfs:custom-tools/up/") and f["path"].endswith("/up.txt")
        delivered = Path(xt.root()).parent / "vfs" / uid / f["path"].removeprefix("vfs:")
        assert delivered.read_text() == "HELLO"
        resp = await c.post("/fd/external-tools/up/run", json={"input": {"doc": "vfs:notes/../../x"}})
        assert resp.status_code == 400
        resp = await c.post("/fd/external-tools/up/run", json={"input": {"doc": "/etc/passwd"}})
        assert resp.status_code == 400 and "vfs:" in resp.json()["detail"]
        detail = (await c.get("/fd/external-tools/up")).json()
        assert detail["live"]["version"] == state["version"] and detail["secrets"][0]["set"] is True
        assert (await c.delete("/fd/external-tools/up")).status_code == 200
        assert await deck.db.get_setting(uid, f"xtool:{uid}/up:KEY") is None
        assert (await c.get("/fd/external-tools/up")).status_code == 404


async def test_someone_elses_tool_is_out_of_reach(deck):
    await _switch_on(deck)
    async with _client(deck.admin) as c:
        assert (await c.post("/fd/external-tools", json={"slug": "priv", "title": "P", "description": "d"})).status_code == 200
        assert (await _published(c, "priv"))["state"] == "passed"
    async with _client(deck.other) as c:
        assert (await c.get("/fd/external-tools")).json()["tools"] == []
        for method, path in (("get", "/fd/external-tools/priv"), ("post", "/fd/external-tools/priv/run"),
                             ("delete", "/fd/external-tools/priv"), ("get", "/fd/external-tools/priv/secrets")):
            resp = await getattr(c, method)(path, **({"json": {"input": {"text": "x"}}} if method == "post" else {}))
            assert resp.status_code == 404, (path, resp.status_code)
        assert (await c.post("/fd/external-tools/priv/publish", json={})).status_code == 404
