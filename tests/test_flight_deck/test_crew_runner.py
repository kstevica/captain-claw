"""Crew's runner, for real: git worktrees, detached processes, a fake CLI that
speaks Claude's JSON. Nothing is mocked but the CLI itself."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException

from captain_claw.flight_deck import auth as auth_mod
from captain_claw.flight_deck import crew_access, crew_logins, crew_runner, crew_store, fd_crypto
from captain_claw.flight_deck.crew_access import Grant
from captain_claw.flight_deck.db import FlightDeckDB

# The scenario comes from the prompt; the worker's environment is built from nothing.
FAKE_CLI = r"""#!/bin/sh
all="$*"
say() { printf '%s\n' "$1"; }
printf '%s' "$all" > "$(git rev-parse --git-dir)/seen"
case "$all" in
  *SCENARIO:earlyfail*) echo "no such model" >&2; exit 4 ;;
esac
sid=$(printf '%s' "$all" | sed -n 's/.*--resume \([^ ]*\).*/\1/p')
[ -n "$sid" ] || sid="sess-$$"
say "{\"type\":\"system\",\"subtype\":\"init\",\"session_id\":\"$sid\"}"
case "$all" in
  *SCENARIO:slow*) sleep 60 ;;
  *SCENARIO:quiet*) sleep 2; echo q > work.txt; git add work.txt; git commit -qm "quiet work" ;;
  *SCENARIO:fail*) echo "the API said no" >&2; exit 3 ;;
  *SCENARIO:envdump*) env > "$PWD/env.txt" ; git add env.txt; git commit -qm "env" ;;
  *SCENARIO:ask*) ;;
  *SCENARIO:dirty*) echo x > left-over.txt ;;
  *SCENARIO:detach*) git checkout -q --detach; echo x > lost.txt; git add lost.txt; git commit -qm "off the branch" ;;
  *SCENARIO:nothing*) ;;
  *SCENARIO:page*) mkdir -p site/css; echo '<html></html>' > site/index.html; echo 'a{}' > site/css/a.css; git add site; git commit -qm "a page" ;;
  *SCENARIO:clash*) echo mine > shared.txt; git add shared.txt; git commit -qm "mine" ;;
  *) echo "$all" | head -c 40 > work.txt; git add work.txt; git commit -qm "the work" ;;
esac
case "$all" in
  *SCENARIO:ask*) text='Which colour?\n\nCREW-STATUS: needs-decision red or blue?' ;;
  *--resume*) echo more >> work.txt; git add work.txt; git commit -qm "more" ; text='Went on.\n\nCREW-STATUS: done went on' ;;
  *) text='All done.\n\nCREW-STATUS: done made the change' ;;
esac
say "{\"type\":\"assistant\",\"session_id\":\"$sid\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"$text\"}]}}"
say "{\"type\":\"result\",\"subtype\":\"success\",\"is_error\":false,\"session_id\":\"$sid\",\"total_cost_usd\":0.01,\"usage\":{\"input_tokens\":10,\"output_tokens\":5}}"
"""
ADMIN_GRANT = Grant(True, "admin", ("claude", "codex", "opencode", "pi"), 4)
GIT = ["git", "-c", "user.name=T", "-c", "user.email=t@example.com"]


def _run(cwd: Path, *args: str) -> str:
    return subprocess.run([*GIT, *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
async def deck(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_DATA_KEY_FILE", str(tmp_path / "data_key"))
    fd_crypto.reset_for_tests()
    crew_store.set_root(tmp_path / "crew")
    fake = tmp_path / "bin" / "claude"
    fake.parent.mkdir()
    fake.write_text(FAKE_CLI)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FD_CREW_CLAUDE_BIN", str(fake))
    monkeypatch.setattr("captain_claw.flight_deck.server.DATA_DIR", tmp_path / "fd-data")
    told: list[dict] = []

    async def notify(task, *, stalled=False):
        told.append({**task, "stalled": stalled})

    monkeypatch.setattr(crew_runner, "_notify", notify)
    monkeypatch.setattr(crew_runner, "_sync_board", lambda task: None)
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    monkeypatch.setattr(auth_mod, "_db", db)
    admin = await db.create_user("admin@example.com", "x", role="admin")
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "init", "-q", "-b", "main", ".")
    (repo / "shared.txt").write_text("base\n")
    _run(repo, "add", ".")
    _run(repo, "commit", "-qm", "base")
    yield {"db": db, "admin": admin, "repo": repo, "told": told, "tmp": tmp_path}
    for task in crew_store.get_store().with_status(*crew_store.ACTIVE):
        await crew_runner._terminate(task)
    crew_store.get_store().close()
    crew_store.set_root(None)
    crew_runner._PROCS.clear()
    crew_runner._LOCKS.clear()
    await db.close()


async def _dispatch(deck, brief="SCENARIO:work", user=None, grant=ADMIN_GRANT, **over) -> dict:
    body = {"harness": "claude", "brief": brief, "repo_path": str(deck["repo"]), **over}
    return await crew_runner.dispatch(user or deck["admin"], grant, body)


async def _settle(task: dict, deadline: float = 20.0) -> dict:
    uid, tid = task["user_id"], task["id"]
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        await crew_runner.tick()
        fresh = crew_store.get_store().get(uid, tid)
        if fresh["status"] not in crew_store.ACTIVE:
            await asyncio.sleep(0.01)  # what it was told goes out in the background
            return fresh
        await asyncio.sleep(0.05)
    raise AssertionError(f"still {fresh['status']}: {fresh['note']}")


async def test_a_task_runs_in_its_own_worktree_and_git_is_the_evidence(deck):
    task = await _dispatch(deck)
    assert task["status"] == "running"
    assert task["branch"].startswith("crew/")
    wt = Path(task["worktree"])
    assert wt.is_dir()
    assert wt.resolve() != deck["repo"].resolve()
    done = await _settle(task)
    assert done["status"] == "ready"
    assert done["note"] == "made the change"
    assert done["final"] == "All done."
    assert done["evidence"]["commits"] == 1
    assert done["evidence"]["dirty"] == 0
    assert done["evidence"]["log"][0].endswith("the work")
    assert done["tokens_in"] == 10 and done["tokens_out"] == 5
    assert done["session"].startswith("sess-")
    # The repo itself: nothing yet.
    assert _run(deck["repo"], "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert not (deck["repo"] / "work.txt").exists()
    assert len(deck["told"]) == 1 and deck["told"][0]["status"] == "ready"
    kinds = [e["kind"] for e in crew_store.read_events(task["user_id"], task["id"])]
    assert kinds[0] == "turn" and kinds[-1] == "status" and "text" in kinds


async def test_a_question_waits_and_the_answer_resumes_the_same_session(deck):
    task = await _dispatch(deck, "SCENARIO:ask")
    asked = await _settle(task)
    assert asked["status"] == "waiting"
    assert asked["note"] == "red or blue?"
    session = asked["session"]
    await crew_runner.steer(task["user_id"], task["id"], "red")
    answered = await _settle(task)
    assert answered["status"] == "ready"
    assert answered["session"] == session
    assert answered["turn"] == 2
    assert answered["note"] == "went on"
    argv_prompt = crew_store.read_events(task["user_id"], task["id"])
    assert any(e["kind"] == "turn" and e["text"] == "red" for e in argv_prompt)


async def test_steering_a_running_worker_waits_for_its_turn_to_end(deck):
    task = await _dispatch(deck, "SCENARIO:quiet")
    steered = await crew_runner.steer(task["user_id"], task["id"], "and one more thing")
    assert steered["status"] == "running"
    assert steered["pending"] == ["and one more thing"]
    done = await _settle(task)
    assert done["turn"] == 2
    assert done["pending"] == []
    assert not deck["told"] or deck["told"][-1]["turn"] == 2


async def test_a_failed_turn_says_why_and_keeps_its_session(deck):
    done = await _settle(await _dispatch(deck, "SCENARIO:fail"))
    assert done["status"] == "failed"
    assert "the API said no" in done["note"]
    assert done["session"].startswith("sess-")
    resumed = await crew_runner.steer(done["user_id"], done["id"], "try again")
    assert resumed["turn"] == 2


async def test_stop_ends_the_process_group_and_keeps_the_work(deck):
    task = await _dispatch(deck, "SCENARIO:slow")
    pid = task["pid"]
    assert crew_runner._alive(task)
    stopped = await crew_runner.stop(task["user_id"], task["id"])
    assert stopped["status"] == "stopped"
    assert Path(stopped["worktree"]).is_dir()
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)


async def test_a_flight_deck_restart_loses_nothing(deck):
    task = await _dispatch(deck, "SCENARIO:quiet")
    # What a restart forgets: the process handle, and every lock.
    crew_runner._PROCS.clear()
    crew_runner._LOCKS.clear()
    crew_store.get_store().close()
    crew_store.set_root(deck["tmp"] / "crew")
    fresh = crew_store.get_store().get(task["user_id"], task["id"])
    assert fresh["status"] == "running"
    done = await _settle(fresh)
    assert done["status"] == "ready"
    assert done["evidence"]["commits"] == 1


async def test_a_reused_pid_is_not_our_worker(deck):
    task = await _dispatch(deck, "SCENARIO:slow")
    crew_runner._PROCS.clear()
    assert crew_runner._alive(task)
    assert not crew_runner._alive({**task, "pid_start": "Mon Jan  1 00:00:00 2001"})
    await crew_runner._terminate({**task, "pid_start": "Mon Jan  1 00:00:00 2001"})
    assert crew_runner._alive(task), "a pid that isn't ours must not be signalled"


async def test_a_process_that_vanishes_is_a_failure(deck):
    task = await _dispatch(deck, "SCENARIO:slow")
    os.killpg(task["pid"], 9)
    await asyncio.sleep(0.2)
    done = await _settle(task)
    assert done["status"] == "failed"
    assert "without saying how" in done["note"]


async def test_merge_lands_on_the_checked_out_base_and_tears_down(deck):
    done = await _settle(await _dispatch(deck))
    merged = await crew_runner.merge(deck["admin"], ADMIN_GRANT, done["id"])
    assert merged["status"] == "delivered"
    assert (deck["repo"] / "work.txt").exists()
    assert merged["delivery"]["kind"] == "merge"
    assert merged["delivery"]["sha"] == _run(deck["repo"], "rev-parse", "HEAD")
    assert not Path(done["worktree"]).exists()
    assert done["branch"] not in _run(deck["repo"], "branch", "--list")
    with pytest.raises(HTTPException) as err:
        await crew_runner.steer(done["user_id"], done["id"], "more")
    assert err.value.status_code == 409


async def test_merge_when_the_base_is_not_checked_out_moves_only_the_branch(deck):
    done = await _settle(await _dispatch(deck))
    _run(deck["repo"], "checkout", "-q", "-b", "elsewhere")
    await crew_runner.merge(deck["admin"], ADMIN_GRANT, done["id"])
    assert not (deck["repo"] / "work.txt").exists()
    assert _run(deck["repo"], "rev-parse", "--abbrev-ref", "HEAD") == "elsewhere"
    assert "work.txt" in _run(deck["repo"], "show", "--stat", "--format=", "main^2")
    listing = _run(deck["repo"], "worktree", "list")
    assert "merge-" not in listing


async def test_a_conflict_changes_nothing_and_keeps_the_work(deck):
    done = await _settle(await _dispatch(deck, "SCENARIO:clash"))
    (deck["repo"] / "shared.txt").write_text("theirs\n")
    _run(deck["repo"], "commit", "-qam", "theirs")
    head = _run(deck["repo"], "rev-parse", "HEAD")
    with pytest.raises(HTTPException) as err:
        await crew_runner.merge(deck["admin"], ADMIN_GRANT, done["id"])
    assert err.value.status_code == 409
    assert _run(deck["repo"], "rev-parse", "HEAD") == head
    assert _run(deck["repo"], "status", "--porcelain") == ""
    assert Path(done["worktree"]).is_dir()
    assert crew_store.get_store().get(done["user_id"], done["id"])["status"] == "ready"


async def test_merge_refuses_what_is_not_landable(deck):
    dirty = await _settle(await _dispatch(deck, "SCENARIO:dirty"))
    with pytest.raises(HTTPException, match="uncommitted|committed nothing") as err:
        await crew_runner.merge(deck["admin"], ADMIN_GRANT, dirty["id"])
    assert err.value.status_code == 409
    nothing = await _settle(await _dispatch(deck, "SCENARIO:nothing"))
    with pytest.raises(HTTPException, match="committed nothing"):
        await crew_runner.merge(deck["admin"], ADMIN_GRANT, nothing["id"])
    scout = await _settle(await _dispatch(deck, kind="scout"))
    with pytest.raises(HTTPException, match="report"):
        await crew_runner.merge(deck["admin"], ADMIN_GRANT, scout["id"])
    running = await _dispatch(deck, "SCENARIO:slow")
    with pytest.raises(HTTPException, match="still working"):
        await crew_runner.merge(deck["admin"], ADMIN_GRANT, running["id"])


async def test_close_refuses_work_that_has_not_landed_until_a_person_says_so(deck):
    done = await _settle(await _dispatch(deck))
    with pytest.raises(HTTPException) as err:
        await crew_runner.close(done["user_id"], done["id"])
    assert err.value.status_code == 409
    assert Path(done["worktree"]).is_dir()
    closed = await crew_runner.close(done["user_id"], done["id"], discard=True)
    assert closed["status"] == "closed"
    assert not Path(done["worktree"]).exists()
    assert done["branch"] not in _run(deck["repo"], "branch", "--list")


async def test_a_scout_leaves_a_report_and_closes_clean(deck):
    done = await _settle(await _dispatch(deck, "SCENARIO:nothing", kind="scout"))
    assert crew_runner.report(done["user_id"], done["id"]) == "All done."
    closed = await crew_runner.close(done["user_id"], done["id"])
    assert closed["status"] == "closed"


async def test_a_user_runs_only_as_many_as_approved(deck):
    db = deck["db"]
    user = await db.create_user("u@example.com", "x", role="user", metadata={
        "crew": {"approved": True, "harnesses": ["claude"], "max_parallel": 1}})
    user = await db.get_user_by_id(user["id"])
    await crew_runner_login(db, user)
    grant = crew_access.grant_of(user)
    assert grant.max_parallel == 1
    url_task = deck["repo"]
    first = await _project_task(deck, user, grant, "SCENARIO:quiet", url_task)
    second = await _project_task(deck, user, grant, "SCENARIO:work", url_task)
    assert first["status"] == "running"
    assert second["status"] == "queued"
    second_done = await _settle(second)
    assert second_done["status"] == "ready"
    assert second_done["turn"] == 1


async def crew_runner_login(db, user: dict) -> None:
    from captain_claw.flight_deck import crew_logins

    await crew_logins.save_login(db, user["id"], "claude", "sk-ant-oat01-test-token")


async def _project_task(deck, user, grant, brief, _repo) -> dict:
    return await crew_runner.dispatch(user, grant, {"harness": "claude", "brief": brief, "project": "demo"})


async def test_withdrawing_the_approval_stops_what_is_running(deck):
    db = deck["db"]
    made = await db.create_user("u@example.com", "x", role="user", metadata={
        "crew": {"approved": True, "harnesses": ["claude"]}})
    user = await db.get_user_by_id(made["id"])
    await crew_runner_login(db, user)
    task = await _project_task(deck, user, crew_access.grant_of(user), "SCENARIO:slow", None)
    assert task["status"] == "running"
    await db.update_user(user["id"], metadata=json.dumps({}))
    assert await crew_runner.enforce_grant(user["id"]) == 1
    fresh = crew_store.get_store().get(user["id"], task["id"])
    assert fresh["status"] == "stopped"
    assert "withdrawn" in fresh["note"]
    assert not crew_runner._alive(task)


async def test_a_non_admins_worker_gets_their_login_and_nothing_of_the_decks(deck, monkeypatch):
    monkeypatch.setenv("FD_JWT_SECRET", "deck-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "deck-owner-key")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "deck-owner-token")
    monkeypatch.setenv("GH_TOKEN", "deck-owner-gh")
    db = deck["db"]
    made = await db.create_user("u@example.com", "x", display_name="Una", role="user", metadata={
        "crew": {"approved": True, "harnesses": ["claude"]}})
    user = await db.get_user_by_id(made["id"])
    await crew_runner_login(db, user)
    done = await _settle(await _project_task(deck, user, crew_access.grant_of(user), "SCENARIO:envdump", None))
    assert done["status"] == "ready", done["note"]
    env = dict(line.split("=", 1) for line in (Path(done["worktree"]) / "env.txt").read_text().splitlines()
               if "=" in line)
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-test-token"
    assert "ANTHROPIC_API_KEY" not in env
    for name in ("FD_JWT_SECRET", "GH_TOKEN"):
        assert name not in env
    assert env["HOME"].startswith(str(deck["tmp"] / "crew"))
    assert env["GIT_AUTHOR_NAME"] == "Una"
    assert env["CAPTAIN_CREW"] == "1"
    # OpenSSH ignores $HOME: the deck's keys are out of its reach only through this.
    assert "IdentityFile=/dev/null" in env["GIT_SSH_COMMAND"]
    assert "SSH_AUTH_SOCK" not in env


async def test_a_non_admin_without_a_login_never_runs_on_the_decks(deck, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "deck-owner-key")
    db = deck["db"]
    made = await db.create_user("u@example.com", "x", role="user", metadata={
        "crew": {"approved": True, "harnesses": ["claude"]}})
    user = await db.get_user_by_id(made["id"])
    with pytest.raises(HTTPException, match="login"):
        await _project_task(deck, user, crew_access.grant_of(user), "SCENARIO:work", None)


async def test_a_non_admin_cannot_name_a_folder_on_this_computer(deck):
    db = deck["db"]
    made = await db.create_user("u@example.com", "x", role="user", metadata={
        "crew": {"approved": True, "harnesses": ["claude"]}})
    user = await db.get_user_by_id(made["id"])
    await crew_runner_login(db, user)
    with pytest.raises(HTTPException) as err:
        await _dispatch(deck, user=user, grant=crew_access.grant_of(user))
    assert err.value.status_code == 403


@pytest.mark.parametrize("over", [{"model": "--dangerously"}, {"effort": "-x"}, {"model": "a b"}])
async def test_model_and_effort_cannot_become_flags(deck, over):
    with pytest.raises(HTTPException) as err:
        await _dispatch(deck, **over)
    assert err.value.status_code == 400


async def test_a_harness_outside_the_approval_is_refused(deck):
    grant = Grant(True, "approved", ("codex",), 1)
    with pytest.raises(HTTPException) as err:
        await _dispatch(deck, grant=grant)
    assert err.value.status_code == 403


async def test_a_quiet_worker_is_flagged_once_and_left_running(deck, monkeypatch):
    task = await _dispatch(deck, "SCENARIO:slow")
    await asyncio.sleep(0.5)
    await crew_runner.tick()  # its first line, read
    monkeypatch.setattr(crew_runner, "_stall_seconds", lambda: 0.0)
    await asyncio.sleep(0.05)
    await crew_runner.tick()
    await crew_runner.tick()
    await asyncio.sleep(0.01)
    fresh = crew_store.get_store().get(task["user_id"], task["id"])
    assert fresh["status"] == "running"
    assert fresh["stalled_at"] > 0
    assert sum(1 for t in deck["told"] if t["stalled"]) == 1


async def test_someone_elses_task_does_not_exist(deck):
    task = await _dispatch(deck, "SCENARIO:nothing")
    for call in (crew_runner.steer("other", task["id"], "x"), crew_runner.stop("other", task["id"]),
                 crew_runner.close("other", task["id"]), crew_runner.diff("other", task["id"])):
        with pytest.raises(HTTPException) as err:
            await call
        assert err.value.status_code == 404


async def test_too_many_waiting_is_refused(deck):
    grant = Grant(True, "admin", ("claude",), 1)
    for _ in range(crew_runner._ACTIVE_PER_SLOT):
        await _dispatch(deck, "SCENARIO:slow", grant=grant)
    with pytest.raises(HTTPException) as err:
        await _dispatch(deck, "SCENARIO:slow", grant=grant)
    assert err.value.status_code == 429


async def test_a_pull_request_pushes_opens_and_merges_on_github(deck, monkeypatch):
    """GitHub stands in; the token goes in the environment, never on a command line."""
    _run(deck["repo"], "remote", "add", "origin", "git@github.com:acme/widgets.git")
    seen: dict = {"git": [], "http": []}
    real_git = crew_runner._git

    async def git(cwd, *args, env=None, timeout=300.0):
        if args[:1] == ("push",):
            seen["git"].append((args, env or {}))
            return 0, ""
        return await real_git(cwd, *args, env=env, timeout=timeout)

    monkeypatch.setattr(crew_runner, "_git", git)

    def github(request: httpx.Request) -> httpx.Response:
        seen["http"].append((request.method, request.url.path, json.loads(request.content), request.headers["authorization"]))
        if request.method == "POST":
            return httpx.Response(201, json={"number": 7, "html_url": "https://github.com/acme/widgets/pull/7"})
        return httpx.Response(200, json={"sha": "abc123"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(github), **kw))

    async def token(db, user, grant):
        return "gh-secret-token"

    monkeypatch.setattr(crew_logins, "github_token", token)
    done = await _settle(await _dispatch(deck, deliver="pr"))
    assert done["status"] == "pr_open"
    assert done["delivery"] == {"kind": "pr", "pr": 7, "url": "https://github.com/acme/widgets/pull/7", "repo": "acme/widgets"}
    args, env = seen["git"][0]
    assert "https://github.com/acme/widgets.git" in args
    assert "gh-secret-token" not in " ".join(args)
    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    method, path, body, auth = seen["http"][0]
    assert (method, path) == ("POST", "/repos/acme/widgets/pulls")
    assert body["head"] == done["branch"] and body["base"] == "main"
    assert auth == "Bearer gh-secret-token"
    # Told more: the same pull request gets what's new.
    await crew_runner.steer(done["user_id"], done["id"], "one more thing")
    again = await _settle(done)
    assert again["status"] == "pr_open" and again["delivery"]["pr"] == 7
    assert len(seen["git"]) == 2 and len([h for h in seen["http"] if h[0] == "POST"]) == 1
    merged = await crew_runner.merge(deck["admin"], ADMIN_GRANT, done["id"])
    assert merged["status"] == "delivered"
    assert seen["http"][-1][:2] == ("PUT", "/repos/acme/widgets/pulls/7/merge")
    assert seen["http"][-1][2]["sha"] == again["evidence"]["head"]
    assert not Path(done["worktree"]).exists()


async def test_a_pull_request_that_fails_leaves_the_task_ready_and_says_why(deck, monkeypatch):
    async def token(db, user, grant):
        return ""

    monkeypatch.setattr(crew_logins, "github_token", token)
    done = await _settle(await _dispatch(deck, deliver="pr"))
    assert done["status"] == "ready"
    assert "GitHub token" in done["note"]


def _seen(task: dict) -> str:
    wt = Path(task["worktree"])
    return (Path(_run(wt, "rev-parse", "--absolute-git-dir")) / "seen").read_text()


async def test_what_a_failed_ps_says_is_not_that_the_worker_died(deck, monkeypatch):
    task = await _dispatch(deck, "SCENARIO:slow")
    crew_runner._PROCS.clear()
    monkeypatch.setattr(crew_runner, "_proc_start", lambda pid: "")
    assert crew_runner._alive(task)
    await crew_runner.tick()
    assert crew_store.get_store().get(task["user_id"], task["id"])["status"] == "running"
    # And it can still be stopped.
    await crew_runner.stop(task["user_id"], task["id"])
    monkeypatch.undo()
    with pytest.raises(ProcessLookupError):
        os.killpg(task["pid"], 0)


async def test_commits_on_a_detached_head_are_not_lost(deck):
    done = await _settle(await _dispatch(deck, "SCENARIO:detach"))
    assert done["evidence"]["stray"] == 1
    assert done["evidence"]["off_branch"] == "a detached HEAD"
    with pytest.raises(HTTPException, match="off its branch"):
        await crew_runner.merge(deck["admin"], ADMIN_GRANT, done["id"])
    with pytest.raises(HTTPException, match="hasn't landed"):
        await crew_runner.close(done["user_id"], done["id"])
    assert Path(done["worktree"]).is_dir()


async def test_merge_leaves_a_base_being_rebased_alone(deck):
    done = await _settle(await _dispatch(deck))
    repo = deck["repo"]
    (repo / "shared.txt").write_text("on main\n")
    _run(repo, "commit", "-qam", "main side")
    _run(repo, "checkout", "-q", "-b", "side", "HEAD~1")
    (repo / "shared.txt").write_text("on side\n")
    _run(repo, "commit", "-qam", "side side")
    _run(repo, "checkout", "-q", "main")
    subprocess.run([*GIT, "rebase", "side"], cwd=repo, capture_output=True)
    before = _run(repo, "rev-parse", "refs/heads/main")
    with pytest.raises(HTTPException) as err:
        await crew_runner.merge(deck["admin"], ADMIN_GRANT, done["id"])
    assert err.value.status_code == 409
    assert _run(repo, "rev-parse", "refs/heads/main") == before
    subprocess.run([*GIT, "rebase", "--abort"], cwd=repo, check=True, capture_output=True)
    assert _run(repo, "for-each-ref", "refs/crew-merge") == ""
    assert Path(done["worktree"]).is_dir()


async def test_a_stop_after_the_queue_read_its_list_wins(deck, monkeypatch):
    grant = Grant(True, "admin", ("claude",), 1)
    first = await _dispatch(deck, "SCENARIO:slow", grant=grant)
    queued = await _dispatch(deck, "SCENARIO:work")
    assert queued["status"] in ("queued", "running")
    await crew_runner.stop(first["user_id"], first["id"])
    await crew_runner.stop(queued["user_id"], queued["id"])
    stale = {**crew_store.get_store().get(queued["user_id"], queued["id"]), "status": "queued"}
    store = crew_store.get_store()
    monkeypatch.setattr(store, "with_status", lambda *s: [stale])
    await crew_runner._start_queued()
    fresh = store.get(queued["user_id"], queued["id"])
    assert fresh["status"] == "stopped"
    assert fresh["pid"] == 0


async def test_a_steer_before_it_starts_goes_with_the_brief(deck):
    db = deck["db"]
    made = await db.create_user("u@example.com", "x", role="user", metadata={
        "crew": {"approved": True, "harnesses": ["claude"], "max_parallel": 1}})
    user = await db.get_user_by_id(made["id"])
    await crew_runner_login(db, user)
    grant = crew_access.grant_of(user)
    first = await _project_task(deck, user, grant, "SCENARIO:quiet", None)
    second = await _project_task(deck, user, grant, "SCENARIO:nothing", None)
    assert second["status"] == "queued"
    await crew_runner.steer(user["id"], second["id"], "and mind the tests")
    done = await _settle(second)
    assert first["id"] != second["id"]
    seen = _seen(done)
    assert "SCENARIO:nothing" in seen and "and mind the tests" in seen and "CREW-STATUS" in seen
    assert done["pending"] == []


async def test_a_turn_that_failed_before_it_had_a_session_starts_over_whole(deck):
    failed = await _settle(await _dispatch(deck, "SCENARIO:earlyfail"))
    assert failed["status"] == "failed" and not failed["session"]
    assert "no such model" in failed["note"]
    await crew_runner.steer(failed["user_id"], failed["id"], "try another way")
    again = await _settle(failed)
    seen = _seen(again)
    assert "SCENARIO:earlyfail" in seen and "try another way" in seen and "CREW-STATUS: done" in seen
    assert "--resume" not in seen


async def test_a_network_error_opening_a_pull_request_still_tells_them(deck, monkeypatch):
    _run(deck["repo"], "remote", "add", "origin", "https://github.com/acme/widgets.git")
    real_git = crew_runner._git

    async def git(cwd, *args, env=None, timeout=300.0):
        return (0, "") if args[:1] == ("push",) else await real_git(cwd, *args, env=env, timeout=timeout)

    monkeypatch.setattr(crew_runner, "_git", git)

    def down(request):
        raise httpx.ConnectError("no route to host")

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(down), **kw))

    async def token(db, user, grant):
        return "t"

    monkeypatch.setattr(crew_logins, "github_token", token)
    done = await _settle(await _dispatch(deck, deliver="pr"))
    assert done["status"] == "ready"
    assert "no route to host" in done["note"]
    assert deck["told"] and deck["told"][-1]["id"] == done["id"]


async def test_merging_a_pull_request_sends_new_commits_and_names_the_exact_one(deck, monkeypatch):
    """Commits made after the pull request opened, on a task that then stopped: they go
    up before the merge, and GitHub is told to merge that commit or nothing."""
    _run(deck["repo"], "remote", "add", "origin", "https://github.com/acme/widgets.git")
    real_git = crew_runner._git
    pushed = []

    async def git(cwd, *args, env=None, timeout=300.0):
        if args[:1] == ("push",):
            pushed.append(_run(Path(cwd), "rev-parse", "HEAD"))
            return 0, ""
        return await real_git(cwd, *args, env=env, timeout=timeout)

    monkeypatch.setattr(crew_runner, "_git", git)
    calls = []

    def github(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, json.loads(request.content)))
        if request.method == "POST":
            return httpx.Response(201, json={"number": 7, "html_url": "https://github.com/acme/widgets/pull/7"})
        return httpx.Response(200, json={"sha": "abc"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(github), **kw))

    async def token(db, user, grant):
        return "t"

    monkeypatch.setattr(crew_logins, "github_token", token)
    done = await _settle(await _dispatch(deck, deliver="pr"))
    assert done["status"] == "pr_open"
    wt = Path(done["worktree"])
    (wt / "late.txt").write_text("late")
    _run(wt, "add", "late.txt")
    _run(wt, "commit", "-qm", "late work")
    head = _run(wt, "rev-parse", "HEAD")
    crew_store.get_store().update(done["id"], status="stopped")
    merged = await crew_runner.merge(deck["admin"], ADMIN_GRANT, done["id"])
    assert merged["status"] == "delivered"
    assert pushed[-1] == head
    method, body = calls[-1]
    assert method == "PUT" and body["sha"] == head


async def test_a_merge_is_only_for_the_commit_it_was_asked_for(deck):
    done = await _settle(await _dispatch(deck))
    head = done["evidence"]["head"]
    with pytest.raises(HTTPException, match="changed since"):
        await crew_runner.merge(deck["admin"], ADMIN_GRANT, done["id"], expect_head="0" * 40)
    merged = await crew_runner.merge(deck["admin"], ADMIN_GRANT, done["id"], expect_head=head)
    assert merged["status"] == "delivered"


async def test_a_replaced_codex_login_is_the_one_used(deck):
    db = deck["db"]
    made = await db.create_user("u@example.com", "x", role="user", metadata={
        "crew": {"approved": True, "harnesses": ["codex"]}})
    user = await db.get_user_by_id(made["id"])
    grant = crew_access.grant_of(user)
    await crew_logins.save_login(db, user["id"], "codex", '{"auth_mode":"chatgpt","who":"first"}')
    env = await crew_logins.worker_env(db, user, grant, "codex", "/bin/codex")
    auth = Path(env["CODEX_HOME"]) / "auth.json"
    assert "first" in auth.read_text()
    assert stat.S_IMODE(auth.stat().st_mode) == 0o600
    await crew_logins.save_login(db, user["id"], "codex", '{"auth_mode":"chatgpt","who":"second"}')
    await crew_logins.worker_env(db, user, grant, "codex", "/bin/codex")
    assert "second" in auth.read_text()


async def test_two_tasks_on_one_new_repository_clone_it_once(deck, monkeypatch):
    clones = []

    async def git(cwd, *args, env=None, timeout=300.0):
        if "clone" in args:
            clones.append(args)
            await asyncio.sleep(0.05)
            Path(args[-1]).mkdir(parents=True)
            return 0, ""
        if args[:2] == ("remote", "get-url"):
            return 0, "https://example.com/a/b.git"
        return 0, ""

    monkeypatch.setattr(crew_runner, "_git", git)
    body = {"repo_url": "https://example.com/a/b.git"}
    await asyncio.gather(*(crew_runner._resolve_repo(deck["admin"], ADMIN_GRANT, body) for _ in range(3)))
    assert len(clones) == 1
    assert clones[0][clones[0].index("--") + 1] == body["repo_url"]


async def test_linked_folders_follow_the_rules_of_folders(deck, monkeypatch):
    from captain_claw.flight_deck import code_routes

    async def folder(uid, project, name):
        return name

    monkeypatch.setattr(code_routes, "_ensure_project", lambda uid, name: name)
    monkeypatch.setattr(code_routes, "_ensure_folder", folder)
    monkeypatch.setattr(code_routes, "_folder_repo", lambda uid, p, f: deck["repo"])
    plain = Grant(True, "approved", ("claude",), 1)
    monkeypatch.setattr(code_routes, "_folder_meta", lambda uid, p, f: {"kind": "link", "mode": "rw"})
    with pytest.raises(HTTPException) as err:
        await crew_runner._resolve_repo(deck["admin"], plain, {"project": "linked", "folder": "x"})
    assert err.value.status_code == 403
    where = await crew_runner._resolve_repo(deck["admin"], ADMIN_GRANT, {"project": "linked", "folder": "x"})
    assert where["repo"] == str(deck["repo"])
    monkeypatch.setattr(code_routes, "_folder_meta", lambda uid, p, f: {"kind": "link", "mode": "ro"})
    with pytest.raises(HTTPException, match="read-only"):
        await crew_runner._resolve_repo(deck["admin"], ADMIN_GRANT, {"project": "linked", "folder": "x"})


# ── What a task made, in the VFS ─────────────────────────────────────────


def _results_folder(task: dict) -> Path:
    return crew_runner._vfs_root(task["user_id"]) / "crew" / "repo" / task["branch"].removeprefix("crew/")


async def test_what_a_task_made_comes_to_the_vfs(deck):
    done = await _settle(await _dispatch(deck))
    folder = _results_folder(done)
    results = done["results"]
    assert results["files_path"] == f"vfs:crew/repo/{folder.name}/files/"
    assert results["report"] == f"vfs:crew/repo/{folder.name}/report.md"
    assert results["changed"] == ["work.txt"]
    assert (folder / "files" / "work.txt").exists()
    # A small project comes whole: a page needs its other files too.
    assert results["whole"] and (folder / "files" / "shared.txt").read_text() == "base\n"
    assert not (folder / "files" / ".git").exists()
    assert (folder / "report.md").read_text() == "All done."
    told = crew_runner.notice_text(done)
    assert results["files_path"] in told and "don't start a task to fetch them" in told
    assert "crew action=request_merge" in told


async def test_a_big_project_brings_only_what_changed_until_asked(deck, monkeypatch):
    monkeypatch.setattr(crew_runner, "_SMALL_FILES", 0)
    done = await _settle(await _dispatch(deck))
    folder = _results_folder(done)
    assert not done["results"]["whole"] and done["results"]["files"] == ["work.txt"]
    assert not (folder / "files" / "shared.txt").exists()
    results = await crew_runner.files(done["user_id"], done["id"], whole=True)
    assert results["whole"] and (folder / "files" / "shared.txt").exists()
    monkeypatch.setattr(crew_runner, "_MAX_FILES", 1)
    fresh = crew_store.get_store().get(done["user_id"], done["id"])
    crew_store.get_store().update(done["id"], results={})
    with pytest.raises(HTTPException) as err:
        await crew_runner.files(fresh["user_id"], fresh["id"], whole=True)
    assert err.value.status_code == 413


async def test_a_page_says_how_to_publish_it_from_the_clean_copy(deck):
    done = await _settle(await _dispatch(deck, "SCENARIO:page"))
    folder = _results_folder(done)
    assert done["results"]["publish"] == {"project": "crew", "subdir": f"repo/{folder.name}/files/site"}
    assert (folder / "files" / "site" / "css" / "a.css").exists()
    assert "hosting action=publish kind=static project=crew" in crew_runner.notice_text(done)


async def test_copying_runs_nothing_from_the_repository_and_follows_no_link(deck):
    ran = deck["tmp"] / "ran"
    repo = deck["repo"]
    (repo / ".gitattributes").write_text("*.txt filter=evil\n")
    (repo / "out").symlink_to("/etc/hosts")
    _run(repo, "config", "filter.evil.smudge", f"touch {ran}; cat")
    _run(repo, "add", ".")
    _run(repo, "commit", "-qm", "a trap")
    done = await _settle(await _dispatch(deck))
    ran.unlink(missing_ok=True)  # making the worktree is a checkout: that runs filters
    await crew_runner._export_results(done, ref=await crew_runner._result_ref(done), whole=True)
    files = _results_folder(done) / "files"
    assert not ran.exists()
    assert (files / "shared.txt").exists() and (files / ".gitattributes").exists()
    assert not (files / "out").exists() and not (files / "out").is_symlink()


async def test_a_scout_brings_its_report_and_the_project_only_on_request(deck):
    done = await _settle(await _dispatch(deck, "SCENARIO:nothing", kind="scout"))
    assert done["results"]["report"] and not done["results"]["files_path"]
    assert (_results_folder(done) / "report.md").read_text() == "All done."
    results = await crew_runner.files(done["user_id"], done["id"], whole=True)
    assert results["files"] == ["shared.txt"]


async def test_an_approve_tells_the_agent_where_it_landed_and_a_click_does_not(deck):
    done = await _settle(await _dispatch(deck))
    deck["told"].clear()
    head = done["evidence"]["head"]
    merged = await crew_runner.merge(deck["admin"], ADMIN_GRANT, done["id"], expect_head=head)
    await asyncio.sleep(0.01)
    assert [t["status"] for t in deck["told"]] == ["delivered"]
    assert merged["results"]["head"] == merged["delivery"]["sha"]
    assert merged["results"]["changed"] == ["work.txt"]  # what it changed, not all that's on the base since
    other = await _settle(await _dispatch(deck, "SCENARIO:clash"))
    deck["told"].clear()
    await crew_runner.merge(deck["admin"], ADMIN_GRANT, other["id"])
    await asyncio.sleep(0.01)
    assert deck["told"] == []


async def test_a_task_merged_before_results_existed_still_answers(deck):
    done = await _settle(await _dispatch(deck))
    merged = await crew_runner.merge(deck["admin"], ADMIN_GRANT, done["id"])
    crew_store.get_store().update(merged["id"], results={})
    shutil.rmtree(_results_folder(done))
    results = await crew_runner.files(done["user_id"], done["id"])
    assert results["landed"] and "work.txt" in results["files"]


async def test_discarding_takes_the_copy_with_it(deck):
    done = await _settle(await _dispatch(deck))
    folder = _results_folder(done)
    assert folder.exists()
    await crew_runner.close(done["user_id"], done["id"], discard=True)
    assert not folder.exists()
    assert not crew_store.get_store().get(done["user_id"], done["id"])["results"]


async def test_files_wait_for_the_worker_and_stay_in_their_folder(deck):
    running = await _dispatch(deck, "SCENARIO:slow")
    with pytest.raises(HTTPException) as err:
        await crew_runner.files(running["user_id"], running["id"])
    assert err.value.status_code == 409
    await crew_runner.stop(running["user_id"], running["id"])
    done = await _settle(await _dispatch(deck))
    assert crew_runner.read_result(done["user_id"], done["id"], "shared.txt")["text"] == "base\n"
    for path in ("../report.md", "/etc/hosts", "../../../../../../etc/hosts", "nothing.txt"):
        with pytest.raises(HTTPException) as err:
            crew_runner.read_result(done["user_id"], done["id"], path)
        assert err.value.status_code == 404, path


def test_the_whole_message_is_where_a_cut_callback_says():
    task = {"id": "t", "user_id": "u", "kind": "scout", "status": "ready", "final": "x" * 6000, "repo": "/r",
            "results": {"report": "vfs:crew/p/t/report.md"}}
    text = crew_runner.notice_text(task)
    assert "x" * 5000 in text and "x" * 5001 not in text
    assert "All of it: vfs:crew/p/t/report.md" in text
    assert "All of it" not in crew_runner.notice_text({**task, "final": "short"})


# ── Projects ─────────────────────────────────────────────────────────────


def test_a_project_is_a_code_project_with_its_folders_or_the_repository():
    of = crew_runner.project_of
    assert of({"repo_kind": "project", "repo_label": "Game/client"}) == ("project:Game", "Game")
    assert of({"repo_kind": "project", "repo_label": "Game/server"}) == ("project:Game", "Game")
    assert of({"repo_kind": "url", "repo_label": "lib"}) == ("url:lib", "lib")
    assert of({"repo_kind": "path", "repo_label": "x", "repo": "/a/x"}) == ("path:/a/x", "x")
    assert of({"repo_kind": "path", "repo_label": "x", "repo": "/b/x"}) == ("path:/b/x", "x")


async def test_tidying_clears_what_is_finished_and_never_what_has_not_landed(deck):
    uid = deck["admin"]["id"]
    store = crew_store.get_store()
    scout = await _settle(await _dispatch(deck, "SCENARIO:nothing", kind="scout"))
    landed = await _settle(await _dispatch(deck))
    await crew_runner.merge(deck["admin"], ADMIN_GRANT, landed["id"])
    unlanded = await _settle(await _dispatch(deck, "SCENARIO:clash"))
    question = await _settle(await _dispatch(deck, "SCENARIO:ask"))
    key = crew_runner.project_of(scout)[0]
    assert await crew_runner.tidy(uid, key) == {"cleared": 2, "kept": 2}
    left = {t["id"] for t in store.list_for(uid)}
    assert left == {unlanded["id"], question["id"]}
    assert not Path(scout["worktree"]).exists()
    assert Path(unlanded["worktree"]).exists()
    # Off the page, not gone: nor is its report.
    assert store.get(uid, scout["id"])["status"] == "closed"
    assert (_results_folder(scout) / "report.md").exists()
    assert await crew_runner.tidy(uid, "project:elsewhere") == {"cleared": 0, "kept": 0}


async def test_a_new_task_is_warned_of_work_that_has_not_landed(deck):
    uid = deck["admin"]["id"]
    done = await _settle(await _dispatch(deck))
    key = crew_runner.project_of(done)[0]
    assert crew_runner.unlanded_in(uid, key) == [done["title"]]
    assert crew_runner.unlanded_in(uid, key, but=done["id"]) == []
    await crew_runner.merge(deck["admin"], ADMIN_GRANT, done["id"])
    assert crew_runner.unlanded_in(uid, key) == []


def test_a_store_from_before_gets_the_new_columns(tmp_path):
    import sqlite3

    path = tmp_path / "crew.db"
    conn = sqlite3.connect(path)
    old = ("id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL, kind TEXT NOT NULL, harness TEXT NOT NULL,"
           " brief TEXT NOT NULL, status TEXT NOT NULL, repo TEXT NOT NULL, created_at REAL NOT NULL,"
           " updated_at REAL NOT NULL")
    conn.execute(f"CREATE TABLE tasks ({old})")
    conn.execute("INSERT INTO tasks VALUES ('t', 'u', 'old', 'ship', 'claude', 'b', 'ready', '/r', 1, 1)")
    conn.commit()
    conn.close()
    store = crew_store.CrewStore(path)
    task = store.list_for("u")[0]
    assert task["id"] == "t" and task["results"] == {} and task["archived_at"] == 0
    store.update("t", archived_at=5.0)
    assert store.list_for("u") == []
    store.close()


def test_reading_the_status_line():
    read = crew_runner.read_status
    assert read("All good.\n\nCREW-STATUS: done shipped it") == ("done", "shipped it", "All good.")
    assert read("**CREW-STATUS:** needs-decision  red or blue?")[:2] == ("needs-decision", "red or blue?")
    assert read("CREW-STATUS: blocked - no network")[:2] == ("blocked", "no network")
    assert read("Text without one.") == ("", "", "Text without one.")
    assert read("CREW-STATUS: done early\nmore\nCREW-STATUS: blocked late")[:2] == ("blocked", "late")
    assert read("") == ("", "", "")


@pytest.mark.parametrize(("remote", "expected"), [
    ("https://github.com/a/b.git", "a/b"), ("https://github.com/a/b", "a/b"), ("git@github.com:a/b.git", "a/b"),
    ("ssh://git@github.com/a/b.git", "a/b"), ("https://token@github.com/a/b.git", "a/b"),
    ("https://gitlab.com/a/b.git", ""), ("https://github.com.evil.example/a/b.git", ""),
])
def test_which_remotes_are_github(remote, expected):
    assert crew_runner._github_repo(remote) == expected


def test_repo_addresses(monkeypatch):
    plain = Grant(True, "approved", ("claude",), 1)
    crew_runner._check_url("https://github.com/a/b.git", plain)
    for bad in ("git@github.com:a/b.git", "file:///etc", "ext::sh -c id", "http://x/y", "/tmp/repo", "ssh://h/r",
                "--upload-pack=x"):
        with pytest.raises(HTTPException):
            crew_runner._check_url(bad, plain)
    crew_runner._check_url("git@github.com:a/b.git", ADMIN_GRANT)
    for bad in ("file:///etc", "ext::sh -c id", "--upload-pack=x@host:y", "-x@host:y"):
        with pytest.raises(HTTPException):
            crew_runner._check_url(bad, ADMIN_GRANT)


def test_another_accounts_git_gets_none_of_the_decks_config():
    env = crew_runner._git_env(home="/h", isolated=True)
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert "IdentityFile=/dev/null" in env["GIT_SSH_COMMAND"]
    assert env["HOME"] == "/h"
    tokened = crew_runner._git_env(token="abc")
    assert "abc" not in tokened["GIT_CONFIG_VALUE_0"]
