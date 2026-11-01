"""A deck updating itself: what stops it before anything is touched, the order, and getting back
from a version that doesn't come up. Real git repositories and real processes throughout."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import deck_update, deck_update_guard, deck_update_routes
from captain_claw.flight_deck.auth import get_current_user

GIT = ["git", "-c", "user.name=T", "-c", "user.email=t@example.com"]
GUARD = Path(deck_update_guard.__file__)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run([*GIT, "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


def commit(cwd: Path, name: str, text: str, title: str = "") -> str:
    (cwd / name).parent.mkdir(parents=True, exist_ok=True)
    (cwd / name).write_text(text)
    git(cwd, "add", name)
    git(cwd, "commit", "-qm", title or f"change {name}", "--allow-empty")
    return git(cwd, "rev-parse", "HEAD")


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A bare remote, the deck's checkout on main, and someone else's clone pushing to it."""
    remote, deck, dev = tmp_path / "remote.git", tmp_path / "deck", tmp_path / "dev"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    subprocess.run(["git", "clone", "-q", str(remote), str(dev)], check=True, capture_output=True)
    git(dev, "checkout", "-q", "-b", "main")
    commit(dev, "captain_claw/flight_deck/server.py", "# the deck\n", "first")
    commit(dev, "pyproject.toml", '[project]\nname = "x"\ndependencies = ["a"]\n', "the project")
    git(dev, "push", "-q", "-u", "origin", "main")
    subprocess.run(["git", "clone", "-q", str(remote), str(deck)], check=True, capture_output=True)

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(deck_update, "find_repo", lambda: (deck, True))
    monkeypatch.setattr(deck_update, "_srv", lambda: types.SimpleNamespace(
        DATA_DIR=data, _load_process_registry=lambda: {}, _process_is_alive=lambda slug: False))
    monkeypatch.setattr(deck_update, "_exit_server", lambda: None)
    monkeypatch.setattr(deck_update, "_task", None)
    monkeypatch.setattr(deck_update, "_checked_at", 0.0)
    monkeypatch.setattr(deck_update, "_crew_running", lambda: 0)
    return types.SimpleNamespace(remote=remote, deck=deck, dev=dev, data=data, tmp=tmp_path)


def push(world, name="new.txt", text="new", title="") -> str:
    sha = commit(world.dev, name, text, title)
    git(world.dev, "push", "-q", "origin", "main")
    return sha


# ── What's new, and what stops it ───────────────────────────────────


async def test_it_shows_what_is_new_only_after_asking_the_remote(world):
    target = push(world, title="the new thing")
    before = await deck_update.status()
    assert before["available"] and before["behind"] == 0
    seen = await deck_update.status(fetch=True)
    assert seen["behind"] == 1 and seen["target"] == target
    assert [c["title"] for c in seen["commits"]] == ["the new thing"]
    assert seen["branch"] == "main" and seen["upstream"] == "origin/main"
    assert seen["blockers"] == [] and seen["checked_at"] > 0


async def test_a_fork_is_followed_because_the_branch_says_so(world, tmp_path):
    fork = tmp_path / "fork.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(world.remote), str(fork)], check=True, capture_output=True)
    git(world.deck, "remote", "add", "myfork", str(fork))
    git(world.deck, "fetch", "-q", "myfork")
    git(world.deck, "branch", "-q", "--set-upstream-to=myfork/main", "main")
    seen = await deck_update.status(fetch=True)
    assert seen["upstream"] == "myfork/main" and seen["remote_url"] == str(fork)
    assert seen["behind"] == 0


def test_it_finds_the_checkout_and_knows_a_copy_from_the_real_thing(tmp_path, monkeypatch):
    import captain_claw

    monkeypatch.delenv("FD_UPDATE_REPO", raising=False)
    package = Path(captain_claw.__file__).resolve().parent.parent
    # Editable: the code it runs is the checkout.
    assert deck_update.find_repo() == (package, True) or not (package / ".git").exists()
    # Named, or where it was started: the venv holds a copy of it.
    other = tmp_path / "other"
    (other / "captain_claw/flight_deck").mkdir(parents=True)
    (other / "captain_claw/flight_deck/server.py").write_text("")
    (other / ".git").mkdir()
    monkeypatch.setenv("FD_UPDATE_REPO", str(other))
    assert deck_update.find_repo() == (other.resolve(), False)
    # A folder that isn't one of these is not taken for it.
    (other / "captain_claw/flight_deck/server.py").unlink()
    assert deck_update.find_repo()[0] != other.resolve()
    # A frozen build has none.
    monkeypatch.setattr(sys, "_MEIPASS", "/somewhere", raising=False)
    assert deck_update.find_repo() == (None, False)


def test_a_token_in_a_remote_address_is_never_shown():
    assert deck_update._clean_url("https://x:ghp_secret@github.com/a/b.git") == "https://github.com/a/b.git"
    assert deck_update._clean_url("https://ghp_secret@host:8443/a") == "https://host:8443/a"
    assert deck_update._clean_url("git@github.com:a/b.git") == "git@github.com:a/b.git"


async def test_a_deck_with_no_checkout_says_so(world, monkeypatch):
    monkeypatch.setattr(deck_update, "find_repo", lambda: (None, False))
    seen = await deck_update.status(fetch=True)
    assert not seen["available"] and "git checkout" in seen["why"]
    with pytest.raises(deck_update.RefusedError, match="git checkout"):
        await deck_update.start(by="a", target="abc")


async def test_a_detached_checkout_or_one_with_no_upstream_says_so(world):
    git(world.deck, "checkout", "-q", "--detach")
    assert "isn't on a branch" in (await deck_update.status())["why"]
    git(world.deck, "checkout", "-q", "-b", "mine")
    assert "no upstream" in (await deck_update.status())["why"]


async def test_a_local_commit_blocks_it(world):
    push(world)
    commit(world.deck, "mine.txt", "mine")
    seen = await deck_update.status(fetch=True)
    assert "can't simply move forward" in seen["blockers"][0]


async def test_a_local_change_or_file_in_the_way_blocks_it(world):
    (world.deck / "pyproject.toml").write_text("hand edited\n")
    push(world, "pyproject.toml", '[project]\nname = "x"\ndependencies = ["a"]\nversion = "2"\n')
    assert "would be overwritten" in (await deck_update.status(fetch=True))["blockers"][0]
    git(world.deck, "checkout", "-q", "pyproject.toml")
    assert (await deck_update.status(fetch=True))["blockers"] == []
    # An untracked one in the way counts too.
    (world.deck / "new.txt").write_text("not committed")
    push(world, "new.txt", "from the remote")
    assert "would be overwritten" in (await deck_update.status(fetch=True))["blockers"][0]


async def test_a_change_the_update_doesnt_touch_is_only_mentioned(world):
    (world.deck / "captain_claw/flight_deck/server.py").write_text("# hand patched\n")
    push(world)
    seen = await deck_update.status(fetch=True)
    assert seen["blockers"] == []
    assert any("aren't committed" in n for n in seen["notes"])


async def test_a_deck_not_started_by_its_command_cant_restart(world, monkeypatch):
    monkeypatch.setattr(deck_update, "_exit_server", None)
    push(world)
    assert "can't restart itself" in (await deck_update.status(fetch=True))["blockers"][0]


async def test_it_mentions_source_without_a_bundle(world):
    push(world, "flight-deck/src/App.tsx", "x")
    assert any("bundle" in n for n in (await deck_update.status(fetch=True))["notes"])
    push(world, "captain_claw/flight_deck/static/index.html", "x")
    assert not any("bundle" in n for n in (await deck_update.status(fetch=True))["notes"])


# ── An update ───────────────────────────────────────────────────────


@pytest.fixture
def steps(world, monkeypatch):
    """The rehearsal, the guard and the exec stand aside; the checkout and the order are real."""
    done: list[str] = []

    async def rehearse(repo, target):
        done.append("rehearse")

    def head():
        return git(world.deck, "rev-parse", "HEAD")[:7]

    monkeypatch.setattr(deck_update, "rehearse", rehearse)
    monkeypatch.setattr(deck_update, "_spawn_guard", lambda: done.append("guard"))
    monkeypatch.setattr(deck_update, "stop_agents", lambda: done.append(f"stop on {head()}"))
    monkeypatch.setattr(deck_update, "start_agents", lambda: done.append("start"))
    monkeypatch.setattr(deck_update, "_request_relaunch", lambda: done.append(f"relaunch on {head()}"))

    async def bell(user_ids, title, body):
        done.append(f"bell {title}")

    async def admins():
        return ["admin"]

    monkeypatch.setattr(deck_update, "_bell", bell)
    monkeypatch.setattr(deck_update, "_admins", admins)
    return done


async def test_the_order_is_rehearse_stop_pull_restart(world, steps):
    target = push(world)
    old = git(world.deck, "rev-parse", "HEAD")
    await deck_update.status(fetch=True)
    job = await deck_update.start(by="admin@x", target=target)
    assert job["phase"] == "rehearsing" and job["from"] == old and job["to"] == target
    await deck_update._task
    assert steps == ["rehearse", f"stop on {old[:7]}", "guard", f"relaunch on {target[:7]}"]
    state = deck_update.read_state()
    assert state["phase"] == "restarting" and git(world.deck, "rev-parse", "HEAD") == target
    # What the guard needs, and what the page doesn't get.
    assert state["from"] == old and state["pid"] == os.getpid() and state["argv"]
    assert not {"env", "argv", "repo", "cwd"} & set(deck_update._view(state))
    assert (deck_update._state_path().stat().st_mode & 0o777) == 0o600


async def test_it_goes_to_the_commit_the_admin_saw_not_a_later_one(world, steps):
    saw = push(world, "one.txt")
    push(world, "two.txt")
    await deck_update.status(fetch=True)
    await deck_update.start(by="a", target=saw)
    await deck_update._task
    assert git(world.deck, "rev-parse", "HEAD") == saw


async def test_it_refuses_what_isnt_on_the_branch(world, steps):
    target = push(world)
    await deck_update.status(fetch=True)
    head = git(world.deck, "rev-parse", "HEAD")
    with pytest.raises(deck_update.RefusedError, match="isn't here"):
        await deck_update.start(by="a", target="deadbeef" * 5)
    with pytest.raises(deck_update.RefusedError, match="already on"):
        await deck_update.start(by="a", target=head)
    git(world.deck, "branch", "-q", "elsewhere", target)
    aside = commit(world.deck, "aside.txt", "x")
    git(world.deck, "reset", "-q", "--hard", head)
    with pytest.raises(deck_update.RefusedError, match="isn't on origin/main"):
        await deck_update.start(by="a", target=aside)
    assert steps == []


async def test_one_at_a_time(world, steps):
    target = push(world)
    await deck_update.status(fetch=True)
    deck_update._write_state(phase="pulling")
    with pytest.raises(deck_update.RefusedError, match="already under way"):
        await deck_update.start(by="a", target=target)
    assert (await deck_update.status())["active"] is True


async def test_a_refused_rehearsal_changes_nothing(world, steps, monkeypatch):
    target = push(world)
    old = git(world.deck, "rev-parse", "HEAD")
    await deck_update.status(fetch=True)

    async def refuse(repo, sha):
        raise deck_update.RefusedError("The new version doesn't start: boom")

    monkeypatch.setattr(deck_update, "rehearse", refuse)
    await deck_update.start(by="a", target=target)
    await deck_update._task
    state = deck_update.read_state()
    assert state["phase"] == "failed" and "boom" in state["error"] and state["refused"] == target
    assert git(world.deck, "rev-parse", "HEAD") == old
    assert steps == ["bell Deck update failed"]  # no agent was stopped, so none is started


async def test_a_pull_that_fails_after_the_stop_brings_the_agents_back(world, steps, monkeypatch):
    target = push(world)
    old = git(world.deck, "rev-parse", "HEAD")
    await deck_update.status(fetch=True)

    def in_the_way():
        steps.append("guard")
        (world.deck / "new.txt").write_text("appeared after the check")

    monkeypatch.setattr(deck_update, "_spawn_guard", in_the_way)
    await deck_update.start(by="a", target=target)
    await deck_update._task
    state = deck_update.read_state()
    assert state["phase"] == "failed" and "wouldn't move forward" in state["error"]
    assert git(world.deck, "rev-parse", "HEAD") == old
    assert steps == ["rehearse", f"stop on {old[:7]}", "guard", "start", "bell Deck update failed"]


async def test_a_copy_in_the_venv_is_installed_again_and_undone_if_that_fails(world, steps, monkeypatch):
    target = push(world)
    old = git(world.deck, "rev-parse", "HEAD")
    await deck_update.status(fetch=True)
    monkeypatch.setattr(deck_update, "find_repo", lambda: (world.deck, False))

    async def pip(*args):
        steps.append(f"pip {args[0]}")
        return 1, "no build backend"

    monkeypatch.setattr(deck_update, "_pip", pip)
    await deck_update.start(by="a", target=target)
    await deck_update._task
    assert "no build backend" in deck_update.read_state()["error"]
    assert "pip --no-deps" in steps
    assert git(world.deck, "rev-parse", "HEAD") == old


async def test_new_requirements_go_in_before_the_rehearsal_and_can_stop_it(world, steps, monkeypatch):
    target = push(world, "pyproject.toml", '[project]\nname = "x"\ndependencies = ["a", "b>=2"]\n')
    await deck_update.status(fetch=True)
    assert any("New packages" in n for n in (await deck_update.status())["notes"])
    calls: list[tuple] = []

    async def pip(*args):
        calls.append(args)
        return 0, ""

    monkeypatch.setattr(deck_update, "_pip", pip)
    await deck_update.start(by="a", target=target)
    await deck_update._task
    assert calls == [("b>=2",)]
    assert steps[0] == "rehearse"

    # And with an install that fails.
    deck_update._state_path().unlink()
    git(world.deck, "reset", "-q", "--hard", "HEAD~1")
    steps.clear()

    async def fails(*args):
        return 1, "no such package"

    monkeypatch.setattr(deck_update, "_pip", fails)
    await deck_update.start(by="a", target=target)
    await deck_update._task
    assert "no such package" in deck_update.read_state()["error"]
    assert "rehearse" not in steps


async def test_waiting_gives_turns_their_time_then_stops_anyway(world, steps, monkeypatch):
    target = push(world)
    await deck_update.status(fetch=True)
    answers = iter([["Ana"], ["Ana"], []])

    async def busy():
        return next(answers)

    real_sleep = asyncio.sleep
    monkeypatch.setattr(deck_update, "busy_agents", busy)
    monkeypatch.setattr(deck_update.asyncio, "sleep", lambda s: real_sleep(0))
    await deck_update.start(by="a", target=target, wait=True)
    await deck_update._task
    texts = [s["text"] for s in deck_update.read_state()["steps"]]
    assert any("Waiting for Ana" in t for t in texts) and not any("stopped anyway" in t for t in texts)

    deck_update._state_path().unlink()
    git(world.deck, "reset", "-q", "--hard", "HEAD~1")

    async def always():
        return ["Ana"]

    monkeypatch.setattr(deck_update, "busy_agents", always)
    monkeypatch.setattr(deck_update, "_WAIT_FOR_TURNS_S", 0.0)
    await deck_update.start(by="a", target=target, wait=True)
    await deck_update._task
    assert any("stopped anyway" in s["text"] for s in deck_update.read_state()["steps"])


# ── The page's files ────────────────────────────────────────────────


def test_a_page_missing_its_files_is_found(tmp_path):
    static = tmp_path / "captain_claw/flight_deck/static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text('<script src="/assets/index-AAAAAAAA.js"></script><link href="/assets/index-BBBBBBBB.css">')
    (static / "hud.html").write_text("<p>no assets</p>")
    assert deck_update.bundle_problems(tmp_path) == ["index-AAAAAAAA.js", "index-BBBBBBBB.css"]
    (static / "assets/index-BBBBBBBB.css").write_text("")
    (static / "assets/index-AAAAAAAA.js").write_text('import("./chunk-CCCCCCCC.js"); const u = "not-an-asset.js"')
    assert deck_update.bundle_problems(tmp_path) == ["chunk-CCCCCCCC.js"]
    (static / "assets/chunk-CCCCCCCC.js").write_text("")
    assert deck_update.bundle_problems(tmp_path) == []


# ── The rehearsal, for real ─────────────────────────────────────────


GOOD_SERVER = textwrap.dedent("""\
    import argparse, http.server, os, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--host"); parser.add_argument("--port", type=int)
    args = parser.parse_args()
    leaked = any(k in os.environ for k in ("FD_SECRET_FOR_TEST", "TELEGRAM_BOT_TOKEN"))
    data = os.environ.get("FD_DATA_DIR", "")
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(500 if leaked or not data or not os.path.isdir(data) else 200)
            self.end_headers()
        def log_message(self, *a): pass
    print("up", flush=True)
    http.server.HTTPServer((args.host, args.port), H).serve_forever()
""")


def _rehearsal_repo(world, server=GOOD_SERVER, agents="") -> str:
    commit(world.dev, "captain_claw/__init__.py", "")
    commit(world.dev, "captain_claw/flight_deck/__init__.py", "")
    commit(world.dev, "captain_claw/flight_deck/server.py", server)
    return push(world, "captain_claw/web_server.py", agents or "# agents\n")


async def test_the_rehearsal_starts_it_with_nothing_of_the_decks(world, monkeypatch):
    monkeypatch.setenv("FD_SECRET_FOR_TEST", "x")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    target = _rehearsal_repo(world)
    git(world.deck, "fetch", "-q")
    before = set(Path(os.environ.get("TMPDIR", "/tmp")).glob("deck-rehearsal-*"))
    await deck_update.rehearse(world.deck, target)
    assert set(Path(os.environ.get("TMPDIR", "/tmp")).glob("deck-rehearsal-*")) == before
    # Neither the checkout nor the deck's data saw any of it.
    assert not (world.deck / "captain_claw/web_server.py").exists()
    assert list(world.data.iterdir()) == []


async def test_the_rehearsal_refuses_a_deck_that_cant_start(world):
    target = _rehearsal_repo(world, server="raise RuntimeError('the new code is broken')\n")
    git(world.deck, "fetch", "-q")
    with pytest.raises(deck_update.RefusedError, match="the new code is broken"):
        await deck_update.rehearse(world.deck, target)


async def test_the_rehearsal_refuses_a_deck_that_answers_badly(world):
    target = _rehearsal_repo(world, server=GOOD_SERVER.replace("500 if leaked or not data or not os.path.isdir(data) else 200", "500"))
    git(world.deck, "fetch", "-q")
    with pytest.raises(deck_update.RefusedError, match="doesn't answer properly"):
        await deck_update.rehearse(world.deck, target)


async def test_the_rehearsal_refuses_agents_that_cant_start(world):
    target = _rehearsal_repo(world, agents="import a_package_nobody_has\n")
    git(world.deck, "fetch", "-q")
    with pytest.raises(deck_update.RefusedError, match="(?s)agents wouldn't start.*a_package_nobody_has"):
        await deck_update.rehearse(world.deck, target)


async def test_the_rehearsal_refuses_a_page_missing_its_files(world):
    commit(world.dev, "captain_claw/flight_deck/static/index.html", '<script src="/assets/index-AAAAAAAA.js"></script>')
    target = _rehearsal_repo(world)
    git(world.deck, "fetch", "-q")
    with pytest.raises(deck_update.RefusedError, match="would open blank"):
        await deck_update.rehearse(world.deck, target)


# ── The guard, with a deck that really crashes or hangs ─────────────


FAKE_DECK = textwrap.dedent("""\
    # What the checkout says it is decides what this does; the state file is the real one's protocol.
    import http.server, json, os, sys, time
    state_path, port = sys.argv[1], int(sys.argv[2])
    what = open(os.path.join(sys.argv[3], "VERSION")).read().strip()
    if what == "broken": sys.exit(1)
    if what == "hangs": time.sleep(600)
    state = json.load(open(state_path))
    if state["phase"] == "restarting":
        state["phase"] = "booted"
        json.dump(state, open(state_path, "w"))
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers()
        def log_message(self, *a): pass
    http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
""")


@pytest.fixture
def guarded(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    old = commit(repo, "VERSION", "good\n", "old")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    state_dir = tmp_path / "update"
    state_dir.mkdir()
    state_path = state_dir / "state.json"
    script = tmp_path / "fake_deck.py"
    script.write_text(FAKE_DECK)
    started: list[subprocess.Popen] = []

    def deck_cmd() -> list[str]:
        return [sys.executable, str(script), str(state_path), str(port), str(repo)]

    def begin(new_version: str, **over) -> subprocess.Popen:
        new = commit(repo, "VERSION", f"{new_version}\n", "new")
        state = {"phase": "restarting", "from": old, "to": new, "repo": str(repo), "editable": True, "cwd": str(tmp_path),
                 "python": sys.executable, "argv": deck_cmd(), "env": dict(os.environ), "boot_timeout": 15,
                 "url": f"http://127.0.0.1:{port}", "registry": str(tmp_path / "registry.json"), "steps": [], **over}
        state_path.write_text(json.dumps(state))
        proc = subprocess.Popen(deck_cmd(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        started.append(proc)
        state["pid"] = proc.pid
        state_path.write_text(json.dumps(state))
        return proc

    def run_guard() -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-I", str(GUARD), str(state_path)], capture_output=True, text=True, timeout=60)

    def state() -> dict:
        return json.loads(state_path.read_text())

    yield types.SimpleNamespace(repo=repo, old=old, begin=begin, run_guard=run_guard, state=state, state_path=state_path,
                                state_dir=state_dir, tmp=tmp_path, started=started)
    for proc in started:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    pid = json.loads(state_path.read_text()).get("pid") if state_path.exists() else None
    if pid:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except OSError:
            pass


def test_the_guard_passes_a_version_that_comes_up(guarded):
    proc = guarded.begin("good")
    done = guarded.run_guard()
    assert done.returncode == 0, done.stdout + done.stderr
    assert guarded.state()["phase"] == "done"
    assert proc.poll() is None  # left alone
    assert git(guarded.repo, "rev-parse", "HEAD") == guarded.state()["to"]


def test_the_guard_takes_back_a_version_that_crashes(guarded):
    proc = guarded.begin("broken")
    new = guarded.state()["to"]
    done = guarded.run_guard()
    proc.wait()
    state = guarded.state()
    assert done.returncode == 1 and state["phase"] == "rolled_back", done.stdout + done.stderr
    assert git(guarded.repo, "rev-parse", "HEAD") == guarded.old
    assert state["refused"] == new and "ended before it was up" in state["error"]
    assert state["pid"] != proc.pid and state["detached_log"] == str(guarded.state_dir / "deck.log")
    # It's really serving, on the old version.
    assert httpx.get(f"{state['url']}/fd/health", timeout=5, trust_env=False).status_code == 200


def test_the_guard_takes_back_a_version_that_hangs(guarded):
    proc = guarded.begin("hangs", boot_timeout=2)
    done = guarded.run_guard()
    assert guarded.state()["phase"] in ("rolled_back", "failed")
    assert done.returncode == 1
    proc.wait(timeout=15)
    assert git(guarded.repo, "rev-parse", "HEAD") == guarded.old


def test_the_guard_ends_what_the_failed_version_started(guarded):
    agent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    guarded.started.append(agent)
    (guarded.tmp / "registry.json").write_text(json.dumps({"ana": {"pid": agent.pid, "web_port": 1}, "bo": {"pid": None}}))
    guarded.begin("broken")
    guarded.run_guard()
    assert agent.wait(timeout=10) is not None
    # The registry stays: a dead pid in it means "start this again".
    assert json.loads((guarded.tmp / "registry.json").read_text())["ana"]["pid"] == agent.pid


def test_the_guard_says_so_when_the_old_version_wont_start_either(guarded):
    (guarded.repo / "VERSION").write_text("good\n")
    guarded.begin("broken", boot_timeout=2)
    state = guarded.state()
    state["argv"] = [sys.executable, "-c", "raise SystemExit(1)"]
    guarded.state_path.write_text(json.dumps(state))
    guarded.run_guard()
    state = guarded.state()
    assert state["phase"] == "failed" and "deck is down" in state["error"]


def test_the_guard_leaves_when_the_deck_gave_up(guarded):
    proc = guarded.begin("good", phase="failed")
    head = git(guarded.repo, "rev-parse", "HEAD")
    done = guarded.run_guard()
    assert done.returncode == 0 and guarded.state()["phase"] == "failed"
    assert proc.poll() is None and git(guarded.repo, "rev-parse", "HEAD") == head


def test_the_guard_keeps_a_change_of_someones_through_a_rollback(guarded):
    commit(guarded.repo, "other.txt", "tracked\n", "other")
    guarded.old = git(guarded.repo, "rev-parse", "HEAD")
    (guarded.repo / "other.txt").write_text("hand patched\n")
    proc = guarded.begin("broken")
    state = guarded.state()
    state["from"] = guarded.old
    guarded.state_path.write_text(json.dumps(state))
    guarded.run_guard()
    proc.wait()
    assert guarded.state()["phase"] == "rolled_back"
    assert (guarded.repo / "other.txt").read_text() == "hand patched\n"


# ── Coming up ───────────────────────────────────────────────────────


def test_a_restart_in_the_middle_of_one_ends_it(world):
    for phase in ("rehearsing", "waiting", "stopping", "pulling"):
        deck_update._write_state(phase=phase)
        deck_update.on_boot()
        state = deck_update.read_state()
        assert state["phase"] == "failed" and "was restarted" in state["error"]


def test_booted_is_only_said_after_a_restart(world):
    deck_update._write_state(phase="restarting")
    deck_update.on_boot()
    assert deck_update.read_state()["phase"] == "restarting"
    deck_update.say_booted()
    assert deck_update.read_state()["phase"] == "booted"
    # The previous version coming back isn't the new one coming up.
    deck_update._write_state(phase="rolling_back")
    deck_update.on_boot()
    deck_update.say_booted()
    assert deck_update.read_state()["phase"] == "rolling_back"
    # And a deck with no update going on writes nothing.
    deck_update._state_path().unlink()
    deck_update.on_boot()
    deck_update.say_booted()
    assert not deck_update._state_path().exists()


def test_a_worker_from_before_the_restart_doesnt_stay_a_zombie():
    """An exec keeps the pid, so a child from before it is still ours, and when it ended it would
    look alive to Crew forever. These are found by pid and nothing else is waited on."""
    theirs = subprocess.Popen(["sleep", "0"])
    pid = theirs.pid
    theirs._child_created = False  # like a Popen object that went with the old image
    mine = subprocess.Popen(["sleep", "0"])
    time.sleep(0.3)
    os.kill(pid, 0)  # a zombie still "exists"
    children = deck_update._children_of_this_pid()
    assert {pid, mine.pid} <= children
    deck_update._reap({pid})
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    # It didn't touch the one something here is waiting for.
    assert mine.wait(timeout=5) == 0


# ── Ending: with the uvicorn that's installed, and a worker thread that's busy ──


BUSY_SERVER = textwrap.dedent("""\
    import asyncio, os, sys, time, uvicorn
    from captain_claw.flight_deck import deck_update

    FAIL = sys.argv[1] == "fail"

    async def app(scope, receive, send):
        assert scope["type"] == "lifespan"
        await receive()
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, time.sleep, 120)  # asyncio.run would wait for this
        if FAIL:
            await send({"type": "lifespan.startup.failed", "message": "it can't start"})
            return
        await send({"type": "lifespan.startup.complete"})
        await asyncio.sleep(0.5)
        deck_update._relaunch_asked.set()
        server.should_exit = True
        await receive()
        await send({"type": "lifespan.shutdown.complete"})

    def relaunched():
        print("relaunched", flush=True)
        os._exit(7)

    deck_update.relaunch = relaunched
    server = deck_update.make_server(uvicorn.Config(app, host="127.0.0.1", port=0, log_config=None))
    server.run()
    print("uvicorn returned", flush=True)
""")


def _run_busy(tmp_path, how: str) -> tuple[subprocess.CompletedProcess, float]:
    script = tmp_path / "busy.py"
    script.write_text(BUSY_SERVER)
    begun = time.monotonic()
    done = subprocess.run([sys.executable, str(script), how], capture_output=True, text=True, timeout=60)
    return done, time.monotonic() - begun


def test_a_failed_start_ends_the_process_with_a_thread_still_busy(tmp_path):
    done, took = _run_busy(tmp_path, "fail")
    assert done.returncode == deck_update.STARTUP_FAILURE, done.stdout + done.stderr
    assert took < 30


def test_the_restart_doesnt_wait_for_a_busy_thread(tmp_path):
    done, took = _run_busy(tmp_path, "restart")
    assert done.returncode == 7 and "relaunched" in done.stdout, done.stdout + done.stderr
    assert took < 30


# ── Telling people ──────────────────────────────────────────────────


async def test_the_result_is_told_once_and_the_secrets_leave_the_disk(world, monkeypatch):
    told: list[tuple] = []

    async def bell(user_ids, title, body):
        told.append((user_ids, title))

    monkeypatch.setattr(deck_update, "_bell", bell)
    deck_update._write_state(phase="done", user_id="u1", env={"KEY": "secret"}, argv=["x"], reported=False)
    await deck_update._report()
    await deck_update._report()
    assert told == [(["u1"], "Deck updated")]
    assert "secret" not in deck_update._state_path().read_text()


async def test_the_switch_does_nothing_when_off(world, monkeypatch):
    push(world)

    async def must_not(**kw):
        raise AssertionError("it looked")

    monkeypatch.setattr(deck_update, "status", must_not)
    await deck_update.tick(_FakeDb())


class _FakeDb:
    def __init__(self, mode: str = "off") -> None:
        self.saved = json.dumps({"mode": mode})

    async def get_system_setting(self, key: str) -> str | None:
        assert key == "fd:deck-update"
        return self.saved

    async def set_system_setting(self, key: str, value: str) -> None:
        self.saved = value


async def test_tell_me_tells_once_per_version(world, steps):
    db = _FakeDb("notify")
    await deck_update.tick(db)
    assert steps == []  # nothing new
    push(world)
    await deck_update.tick(db)
    await deck_update.tick(db)
    assert steps == ["bell New version for this deck"]
    push(world, "another.txt")
    await deck_update.tick(db)
    assert steps == ["bell New version for this deck"] * 2
    assert git(world.deck, "rev-parse", "HEAD") != git(world.dev, "rev-parse", "HEAD")


async def test_install_by_itself_waits_for_quiet_and_skips_what_failed(world, steps, monkeypatch):
    db = _FakeDb("install")
    target = push(world)

    async def busy():
        return ["Ana"]

    monkeypatch.setattr(deck_update, "busy_agents", busy)
    await deck_update.tick(db)
    assert deck_update.read_state().get("phase") in (None, "idle")  # told, not started
    assert steps == ["bell New version for this deck"]
    steps.clear()

    async def quiet():
        return []

    monkeypatch.setattr(deck_update, "busy_agents", quiet)

    # The one it tried and couldn't start isn't tried again, every 15 minutes.
    async def refuse(repo, sha):
        raise deck_update.RefusedError("doesn't start")

    monkeypatch.setattr(deck_update, "rehearse", refuse)
    await deck_update.tick(db)
    await deck_update._task
    assert deck_update.read_state()["refused"] == target and steps == ["bell Deck update failed"]
    steps.clear()
    deck_update._write_state(phase="failed")
    await deck_update.tick(db)
    assert steps == [] and deck_update.read_state()["phase"] == "failed"

    # A newer one is.
    async def fine(repo, sha):
        steps.append("rehearse")

    monkeypatch.setattr(deck_update, "rehearse", fine)
    newer = push(world, "fix.txt")
    await deck_update.tick(db)
    await deck_update._task
    assert git(world.deck, "rev-parse", "HEAD") == newer
    assert deck_update.read_state()["by"] == "this deck, by itself"


# ── Who may ─────────────────────────────────────────────────────────


def _routes(user: dict) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(deck_update_routes.router)
    app.dependency_overrides[get_current_user] = lambda: user
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://deck.test")


async def test_only_an_admin_gets_near_it(world, monkeypatch):
    monkeypatch.setattr(deck_update_routes, "get_db", lambda: _FakeDb())
    async with _routes({"id": "u", "role": "user"}) as client:
        for method, path, body in (("GET", "", None), ("POST", "/check", None), ("POST", "/start", {"target": "x"}),
                                   ("PUT", "/mode", {"mode": "install"})):
            resp = await client.request(method, f"/fd/system/update{path}", json=body)
            assert resp.status_code == 403, path


async def test_an_admin_works_it_from_the_page(world, steps, monkeypatch):
    db = _FakeDb()
    monkeypatch.setattr(deck_update_routes, "get_db", lambda: db)
    target = push(world, title="the new thing")
    async with _routes({"id": "a", "role": "admin", "email": "a@x"}) as client:
        seen = (await client.post("/fd/system/update/check")).json()
        assert seen["behind"] == 1 and seen["mode"] == "off" and seen["target"] == target
        assert (await client.put("/fd/system/update/mode", json={"mode": "install"})).json() == {"mode": "install"}
        assert (await client.get("/fd/system/update")).json()["mode"] == "install"
        assert (await client.put("/fd/system/update/mode", json={"mode": "yes"})).status_code == 400
        assert (await client.post("/fd/system/update/start", json={"target": "nope"})).status_code == 409
        started = await client.post("/fd/system/update/start", json={"target": target})
        assert started.status_code == 202 and started.json()["by"] == "a@x"
        await deck_update._task
        while_on = (await client.get("/fd/system/update")).json()
        assert while_on["active"] is True and while_on["job"]["phase"] == "restarting"
        assert "env" not in json.dumps(while_on["job"])


# ── Stopping ────────────────────────────────────────────────────────


def test_agents_get_time_to_stop_and_it_doesnt_wait_longer(tmp_path, monkeypatch):
    from captain_claw.flight_deck import server

    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", tmp_path / ".processes.json")
    monkeypatch.setattr(server, "_processes", {})
    finished, ready = tmp_path / "finished", tmp_path / "ready"
    child = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""\
        import pathlib, signal, sys, time
        def stop(*_):
            time.sleep(1)
            pathlib.Path({str(finished)!r}).write_text("saved")
            sys.exit(0)
        signal.signal(signal.SIGTERM, stop)
        pathlib.Path({str(ready)!r}).write_text("")
        time.sleep(60)
    """)])
    while not ready.exists():
        time.sleep(0.05)
    server._save_process_registry({"ana": {"pid": child.pid, "web_port": 1}, "bo": {"pid": None, "stopped": True}})
    begun = time.monotonic()
    server._stop_all_processes()
    took = time.monotonic() - begun
    assert finished.read_text() == "saved"  # it wasn't killed 0.2 seconds in
    assert took < 4.5  # and a child that's gone isn't waited on as a zombie
    registry = server._load_process_registry()
    assert registry["ana"]["pid"] is None and "stopped" not in registry["ana"]  # so the next start brings it back
    assert registry["bo"]["stopped"] is True


def test_an_agent_that_wont_stop_is_killed(tmp_path, monkeypatch):
    from captain_claw.flight_deck import server

    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", tmp_path / ".processes.json")
    monkeypatch.setattr(server, "_processes", {})
    ready = tmp_path / "ready"
    child = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""\
        import pathlib, signal, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        pathlib.Path({str(ready)!r}).write_text("")
        time.sleep(60)
    """)])
    while not ready.exists():
        time.sleep(0.05)
    server._save_process_registry({"ana": {"pid": child.pid, "web_port": 1}})
    server._stop_all_processes()
    time.sleep(0.3)
    with pytest.raises((ProcessLookupError, ChildProcessError)):
        os.waitpid(child.pid, os.WNOHANG) if child.poll() is None else os.kill(child.pid, 0)
