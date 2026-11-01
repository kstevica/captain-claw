"""Crew: folders of a person's VFS given to a worker. Its own copy to read, the real folder to change."""

# ruff: noqa: F811
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi import HTTPException

from captain_claw.flight_deck import crew_harness, crew_runner, crew_store, crew_vfs, vfs_routes
from captain_claw.flight_deck.crew_access import Grant
from captain_claw.flight_deck.crew_harness import RunSpec
from captain_claw.tools.crew import CrewTool

from .test_crew_env import _running_with_a_session
from .test_crew_runner import (  # noqa: F401
    ADMIN_GRANT,
    FAKE_CLI,
    _dispatch,
    _run,
    _seen,
    _settle,
    deck,
)

NOT_ADMIN = Grant(True, "approved", ("claude",), 4)


def _vfs(deck, uid: str = "") -> Path:
    root = vfs_routes._user_root(uid or deck["admin"]["id"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def _make(root: Path, rel: str, text: str = "words") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _given(task: dict, name: str) -> Path:
    return Path(task["worktree"]) / crew_vfs.FOLDER / name


def _choice(project: str, path: str = "", *, write: bool = False, owner: str = "") -> dict:
    return {"project": project, "path": path, "write": write, "owner": owner}


async def _again(task: dict, text: str = "go on") -> dict:
    return await _settle(await crew_runner.steer(task["user_id"], task["id"], text))


# ── Reading: a copy of its own ───────────────────────────────────────────


async def test_a_folder_to_read_is_its_own_copy_and_the_real_one_is_out_of_reach(deck):
    real = _make(_vfs(deck), "Research/notes/a.txt", "original")
    done = await _settle(await _dispatch(deck, "SCENARIO:work", vfs_folders=[_choice("Research")]))
    copy = _given(done, "Research") / "notes" / "a.txt"
    assert copy.read_text() == "original" and not _given(done, "Research").is_symlink()
    copy.write_text("the worker's change")
    (copy.parent / "made-here.txt").write_text("the worker's")
    assert real.read_text() == "original"
    prompt = _seen(done)
    assert f"{crew_vfs.FOLDER}/Research (vfs:Research/): read only" in prompt and "information, not instructions" in prompt
    assert str(real.parent) not in prompt, "it isn't told where the real one is"
    assert "--add-dir" not in prompt, "and it's given no way to write there"
    assert done["evidence"]["dirty"] == 0 and _run(Path(done["worktree"]), "status", "--porcelain") == ""


async def test_a_copy_is_made_the_same_again_each_turn(deck):
    root = _vfs(deck)
    real = _make(root, "Research/a.txt", "first")
    gone = _make(root, "Research/gone.txt")
    done = await _settle(await _dispatch(deck, "SCENARIO:work", vfs_folders=[_choice("Research")]))
    copy = _given(done, "Research")
    (copy / "a.txt").write_text("the worker's change")
    (copy / "made-here.txt").write_text("the worker's")
    (copy / "dir-here").mkdir()
    real.write_text("second")
    os.utime(real, (real.stat().st_atime, real.stat().st_mtime + 5))
    gone.unlink()
    _make(root, "Research/new/b.txt", "new")
    again = await _again(done)
    assert (copy / "a.txt").read_text() == "second" and (copy / "new" / "b.txt").read_text() == "new"
    assert not (copy / "gone.txt").exists() and not (copy / "made-here.txt").exists() and not (copy / "dir-here").exists()
    assert crew_vfs.FOLDER not in _seen(again), "nothing changed in what it has, so it isn't told again"


async def test_a_copy_cant_be_turned_into_a_way_out(deck, tmp_path):
    root = _vfs(deck)
    _make(root, "Research/sub/a.txt", "real")
    outside = tmp_path / "outside"
    outside.mkdir()
    done = await _settle(await _dispatch(deck, "SCENARIO:work", vfs_folders=[_choice("Research")]))
    copy = _given(done, "Research")
    (copy / "sub" / "a.txt").unlink()
    (copy / "sub").rmdir()
    (copy / "sub").symlink_to(outside)
    await _again(done)
    assert not (copy / "sub").is_symlink() and (copy / "sub" / "a.txt").read_text() == "real"
    assert list(outside.iterdir()) == [], "the copy went to its own folder, not along the worker's link"


async def test_a_copy_leaves_out_links_and_what_can_be_made_again(deck, tmp_path):
    root = _vfs(deck)
    _make(root, "Research/keep.txt")
    _make(root, "Research/.git/config")
    _make(root, "Research/node_modules/x/y.js")
    _make(root, "Research/.vfs-meta.jsonl")
    secret = _make(tmp_path, "secret.txt", "not for it")
    (root / "Research" / "link.txt").symlink_to(secret)
    (root / "Research" / "linkdir").symlink_to(secret.parent)
    done = await _settle(await _dispatch(deck, "SCENARIO:work", vfs_folders=[_choice("Research")]))
    assert sorted(p.name for p in _given(done, "Research").iterdir()) == ["keep.txt"]


async def test_copies_have_a_limit_and_say_so(deck, monkeypatch):
    root = _vfs(deck)
    big = _make(root, "Big/a.bin", "x" * 100)
    _make(root, "Small/a.txt", "y")
    monkeypatch.setattr(crew_vfs, "MAX_COPY_BYTES", 50)
    with pytest.raises(HTTPException) as err:
        await _dispatch(deck, "SCENARIO:work", vfs_folders=[_choice("Big")])
    assert err.value.status_code == 413 and "read and write" in err.value.detail and "vfs:Big/" in err.value.detail
    # To change it's no copy, so it's no limit.
    done = await _settle(await _dispatch(deck, "SCENARIO:work", vfs_folders=[_choice("Big", write=True), _choice("Small")]))
    assert _given(done, "Big").is_symlink()
    # A folder that grew past it is left out of the next turn, and that's said.
    _make(root, "Small/b.bin", "z" * 100)
    told = []
    original = crew_runner._say
    monkeypatch.setattr(crew_runner, "_say", lambda task, kind, text, **extra: (told.append(text), original(task, kind, text, **extra))[1])
    await _again(done)
    assert not _given(done, "Small").exists() and any("too much to copy" in t for t in told)
    assert big.exists()


# ── Writing: the real folder ─────────────────────────────────────────────


async def test_a_folder_to_change_is_the_real_one_and_closing_leaves_it(deck):
    real = _make(_vfs(deck), "Reports/a.txt", "theirs")
    done = await _settle(await _dispatch(deck, "SCENARIO:work", vfs_folders=[_choice("Reports", write=True)]))
    here = _given(done, "Reports")
    assert here.is_symlink() and here.resolve() == real.parent
    prompt = _seen(done)
    assert f"--add-dir {real.parent}" in prompt, "Claude refuses to write where it isn't given"
    assert f"{crew_vfs.FOLDER}/Reports (vfs:Reports/): read and write" in prompt and "no undo" in prompt
    (here / "new.txt").write_text("the worker's")
    assert (real.parent / "new.txt").read_text() == "the worker's"
    assert _run(Path(done["worktree"]), "status", "--porcelain") == "", "it isn't the project's work"
    await crew_runner.close(done["user_id"], done["id"], discard=True)
    assert not Path(done["worktree"]).exists()
    assert real.read_text() == "theirs" and (real.parent / "new.txt").read_text() == "the worker's", "closing takes the link, not the folder"


async def test_codex_can_write_only_where_it_may(deck, monkeypatch):
    root = _vfs(deck)
    live = _make(root, "Reports/a.txt").parent
    copy = _make(root, "Research/a.txt").parent
    fake = deck["tmp"] / "bin" / "codex"
    fake.write_text(FAKE_CLI)
    fake.chmod(0o755)
    monkeypatch.setenv("FD_CREW_CODEX_BIN", str(fake))
    task = await _settle(await _dispatch(deck, "SCENARIO:work", harness="codex", vfs_folders=[_choice("Reports", write=True), _choice("Research")]))
    seen = _seen(task)
    roots = seen.split("sandbox_workspace_write.writable_roots=", 1)[1].split("] ", 1)[0]
    assert str(live) in roots and str(copy) not in roots


def test_claude_is_given_each_folder_it_can_change():
    argv = crew_harness._claude_argv(RunSpec(binary="claude", prompt="p", cwd="/w", add_dirs=("/a", "/b")))
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--add-dir"] == ["/a", "/b"]
    assert "--add-dir" not in crew_harness._claude_argv(RunSpec(binary="claude", prompt="p", cwd="/w"))


async def test_a_setup_task_can_have_folders_and_leaves_the_main_folder_as_it_was(deck):
    repo: Path = deck["repo"]
    real = _make(_vfs(deck), "Reports/a.txt", "theirs")
    done = await _settle(await _dispatch(deck, "SCENARIO:setup", kind="setup", vfs_folders=[_choice("Reports", write=True), _choice("Reports", "")]))
    assert (repo / crew_vfs.FOLDER / "Reports").is_symlink()
    assert crew_vfs.FOLDER not in _run(repo, "status", "--porcelain")
    await crew_runner.close(done["user_id"], done["id"])
    assert not (repo / crew_vfs.FOLDER).exists() and real.read_text() == "theirs"


# ── Which folders ────────────────────────────────────────────────────────


async def test_many_folders_at_once_each_its_own_way_and_one_inside_another(deck):
    root = _vfs(deck)
    _make(root, "A/docs/a.txt", "a")
    _make(root, "B/docs/b.txt", "b")
    _make(root, "C/c.txt", "c")
    done = await _settle(await _dispatch(deck, "SCENARIO:work", vfs_folders=[
        _choice("A", "docs"), _choice("B", "docs/", write=True), _choice("C"), _choice("C")]))
    assert (_given(done, "docs") / "a.txt").read_text() == "a"
    assert _given(done, "docs-2").is_symlink() and (_given(done, "docs-2") / "b.txt").read_text() == "b"
    assert (_given(done, "C") / "c.txt").read_text() == "c"
    assert [(f["project"], f["path"], f["write"], f["as"]) for f in done["vfs"]] == [
        ("A", "docs", False, "docs"), ("B", "docs", True, "docs-2"), ("C", "", False, "C")]


@pytest.mark.parametrize(("project", "path"), [
    ("Nope", ""), ("Research", "nope"), ("Research", "../.."), ("Research", "../Other"), ("..", ""), ("", ""),
    (".drive", ""), ("Research", ".hidden"), ("Research", "a.txt"), ("Research/../Other", ""),
])
async def test_only_a_real_folder_of_their_own_can_be_named(deck, project, path):
    root = _vfs(deck)
    _make(root, "Research/a.txt")
    _make(root, "Research/.hidden/a.txt")
    _make(root, "Other/a.txt")
    _make(root, ".drive/m/a.txt")
    with pytest.raises(HTTPException) as err:
        await crew_vfs.choose(deck["admin"]["id"], ADMIN_GRANT, [_choice(project, path)])
    assert err.value.status_code == 404, err.value.detail


async def test_a_link_inside_a_folder_is_no_way_out_of_it(deck, tmp_path):
    root = _vfs(deck)
    (root / "Research").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "Research" / "out").symlink_to(outside)
    with pytest.raises(HTTPException):
        await crew_vfs.choose(deck["admin"]["id"], ADMIN_GRANT, [_choice("Research", "out")])


async def test_nobody_elses_folder_without_a_share(deck):
    other = await deck["db"].create_user("o@example.com", "x", role="user")
    _make(_vfs(deck, other["id"]), "Theirs/a.txt")
    uid = deck["admin"]["id"]
    for choice in (_choice("Theirs"), _choice("Theirs", owner=other["id"])):
        with pytest.raises(HTTPException) as err:
            await crew_vfs.choose(uid, ADMIN_GRANT, [choice])
        assert err.value.status_code == 404
    assert await crew_vfs.choices(uid, ADMIN_GRANT) == []


def _write_links(root: Path, links: dict) -> None:
    (root / ".vfs-links.json").write_text(json.dumps(links))


async def test_drive_and_onedrive_folders_are_not_offered_or_accepted(deck):
    root = _vfs(deck)
    uid = deck["admin"]["id"]
    _make(root, ".drive/Board/a.txt")
    _make(root, "Mine/a.txt")
    _write_links(root, {"Board": {"path": str(root / ".drive" / "Board"), "kind": "gdrive", "mode": "ro"},
                        "Sales": {"path": str(root / ".drive" / "Board"), "kind": "onedrive", "mode": "ro"}})
    assert [f["project"] for f in await crew_vfs.choices(uid, ADMIN_GRANT)] == ["Mine"]
    for name in ("Board", "Sales"):
        with pytest.raises(HTTPException) as err:
            await crew_vfs.choose(uid, ADMIN_GRANT, [_choice(name)])
        assert err.value.status_code == 400 and "Google Drive or OneDrive" in err.value.detail


async def test_a_folder_linked_from_the_computer_is_an_admins_and_keeps_its_mode(deck, tmp_path):
    root = _vfs(deck)
    uid = deck["admin"]["id"]
    rw = tmp_path / "ext-rw"
    ro = tmp_path / "ext-ro"
    _make(rw, "a.txt")
    _make(ro, "a.txt")
    _write_links(root, {"Rw": {"path": str(rw), "mode": "rw"}, "Ro": {"path": str(ro), "mode": "ro"}})
    assert await crew_vfs.choices(uid, NOT_ADMIN) == []
    with pytest.raises(HTTPException) as err:
        await crew_vfs.choose(uid, NOT_ADMIN, [_choice("Rw")])
    assert err.value.status_code == 403
    found = {f["project"]: f for f in await crew_vfs.choices(uid, ADMIN_GRANT)}
    assert found["Rw"]["can_write"] and found["Rw"]["kind"] == "link" and not found["Ro"]["can_write"]
    assert (await crew_vfs.choose(uid, ADMIN_GRANT, [_choice("Rw", write=True)]))[0]["write"]
    with pytest.raises(HTTPException) as err:
        await crew_vfs.choose(uid, ADMIN_GRANT, [_choice("Ro", write=True)])
    assert err.value.status_code == 403
    assert not (await crew_vfs.choose(uid, ADMIN_GRANT, [_choice("Ro")]))[0]["write"]


async def test_a_shared_folder_is_as_shared_and_gone_when_it_stops_being(deck, monkeypatch):
    db = deck["db"]
    owner = await db.create_user("o@example.com", "x", role="user")
    real = _make(_vfs(deck, owner["id"]), "Theirs/a.txt", "theirs")
    uid = deck["admin"]["id"]
    await db.create_share("vfs", "Theirs", owner["id"], uid, "view")
    found = await crew_vfs.choices(uid, ADMIN_GRANT)
    assert [(f["project"], f["owner"], f["kind"], f["can_write"]) for f in found] == [("Theirs", owner["id"], "shared", False)]
    with pytest.raises(HTTPException) as err:
        await crew_vfs.choose(uid, ADMIN_GRANT, [_choice("Theirs", write=True, owner=owner["id"])])
    assert err.value.status_code == 403
    done = await _settle(await _dispatch(deck, "SCENARIO:work", vfs_folders=[_choice("Theirs", owner=owner["id"])]))
    assert not _given(done, "Theirs").is_symlink() and (_given(done, "Theirs") / "a.txt").read_text() == "theirs"
    # Named without its owner, as the tool names it.
    assert (await crew_vfs.choose(uid, ADMIN_GRANT, [_choice("Theirs")]))[0]["owner"] == owner["id"]
    # To change: a share that says so. Then taken down to view, the next turn gets a copy, not the folder.
    await db.create_share("vfs", "Theirs", owner["id"], uid, "edit")
    live = await _settle(await _dispatch(deck, "SCENARIO:work with write", vfs_folders=[_choice("Theirs", write=True, owner=owner["id"])]))
    assert _given(live, "Theirs").resolve() == real.parent
    await db.create_share("vfs", "Theirs", owner["id"], uid, "view")
    again = await _again(live)
    assert not _given(live, "Theirs").is_symlink() and "read only" in _seen(again)
    assert f"--add-dir {real.parent}" not in _seen(again)
    # Stopped: gone from the worker's folder by its next turn, copy and all.
    await db.delete_share("vfs", "Theirs", owner["id"], uid)
    gone = await _again(live)
    assert not _given(live, "Theirs").exists()
    assert crew_vfs.NONE_LEFT in _seen(gone)
    told = [e["text"] for e in crew_store.read_events(uid, live["id"]) if e["kind"] == "note"]
    assert any("vfs:Theirs/" in t and "doesn't have this folder" in t for t in told)


# ── Changing them, and asking again ──────────────────────────────────────


async def test_folders_can_change_for_a_tasks_next_turn(deck):
    root = _vfs(deck)
    _make(root, "A/a.txt")
    _make(root, "B/b.txt")
    uid = deck["admin"]["id"]
    done = await _settle(await _dispatch(deck, "SCENARIO:work", vfs_folders=[_choice("A")]))
    made = await crew_runner.set_folders(uid, ADMIN_GRANT, done["id"], [_choice("B", write=True)])
    assert made["status"] == done["status"] and _given(done, "A").exists(), "it takes effect on its next turn, not before"
    again = await _again(done)
    assert not _given(done, "A").exists() and _given(done, "B").is_symlink()
    prompt = _seen(again)
    assert f"{crew_vfs.FOLDER}/B (vfs:B/): read and write" in prompt and "vfs:A/" not in prompt
    await crew_runner.set_folders(uid, ADMIN_GRANT, done["id"], [])
    last = await _again(done)
    assert crew_vfs.NONE_LEFT in _seen(last) and list((Path(done["worktree"]) / crew_vfs.FOLDER).iterdir()) == []
    with pytest.raises(HTTPException):
        await crew_runner.set_folders(uid, ADMIN_GRANT, done["id"], [_choice("Nope")])
    await crew_runner.close(uid, done["id"], discard=True)
    with pytest.raises(HTTPException) as err:
        await crew_runner.set_folders(uid, ADMIN_GRANT, done["id"], [_choice("A")])
    assert err.value.status_code == 409


async def test_a_retry_with_the_same_folders_is_the_same_task(deck):
    _make(_vfs(deck), "A/a.txt")
    _make(_vfs(deck), "B/b.txt")
    first = await _dispatch(deck, "SCENARIO:slow", vfs_folders=[_choice("A")])
    assert (await _dispatch(deck, "SCENARIO:slow", vfs_folders=[_choice("A")]))["id"] == first["id"]
    assert (await _dispatch(deck, "SCENARIO:slow", vfs_folders=[_choice("B")]))["id"] != first["id"]


async def test_folders_for_a_setup_under_way_say_so(deck):
    _make(_vfs(deck), "A/a.txt")
    first = await _dispatch(deck, "SCENARIO:lagsetup", kind="setup")
    with pytest.raises(HTTPException) as err:
        await _dispatch(deck, "SCENARIO:lagsetup", kind="setup", vfs_folders=[_choice("A")])
    assert err.value.status_code == 409 and first["id"] in err.value.detail


async def test_a_running_worker_gets_new_folders_when_its_sent_something_now(deck):
    _make(_vfs(deck), "A/a.txt")
    task = await _running_with_a_session(deck)
    uid = task["user_id"]
    await crew_runner.set_folders(uid, ADMIN_GRANT, task["id"], [_choice("A")])
    assert not _given(task, "A").exists()
    done = await _settle(await crew_runner.steer(uid, task["id"], "use it", now=True))
    assert (_given(done, "A") / "a.txt").exists() and f"{crew_vfs.FOLDER}/A (vfs:A/)" in _seen(done)


# ── The tool ─────────────────────────────────────────────────────────────


class _Asked:
    def __init__(self, **answers) -> None:
        self.answers = answers
        self.asked: list[tuple[str, dict | None]] = []

    async def call(self, method, path, body=None):
        self.asked.append((path, body))
        return 200, self.answers[path]


MADE = {"id": "t", "status": "running", "harness_label": "Claude Code", "repo_label": "p", "branch": "crew/x"}


async def test_the_tool_names_folders_as_flight_deck_does():
    client = _Asked(dispatch=MADE)
    out = await CrewTool()._do_dispatch(client, harness="claude", brief="x", vfs_read=["vfs:Research/2026/", "Notes"], vfs_write="vfs:Reports")
    assert client.asked[0][1]["vfs_folders"] == [
        {"project": "Research", "path": "2026", "write": False}, {"project": "Notes", "path": "", "write": False},
        {"project": "Reports", "path": "", "write": True}]
    assert "vfs:Research/2026/ (read)" in out.content and "vfs:Reports/ (read and write)" in out.content
    plain = _Asked(dispatch=MADE)
    await CrewTool()._do_dispatch(plain, harness="claude", brief="x")
    assert "vfs_folders" not in plain.asked[0][1]


async def test_the_tool_changes_a_tasks_folders_and_says_when():
    client = _Asked(folders={"id": "t", "vfs": [{"project": "A", "path": "", "write": False}]})
    out = await CrewTool()._do_folders(client, task_id="t", vfs_read=["vfs:A"])
    assert client.asked == [("folders", {"task_id": "t", "vfs_folders": [{"project": "A", "path": "", "write": False}]})]
    assert out.success and "next turn" in out.content and "vfs:A/ (read)" in out.content
    none = await CrewTool()._do_folders(_Asked(folders={"id": "t", "vfs": []}), task_id="t")
    assert "none of the user's folders" in none.content


async def test_an_agent_walled_to_part_of_the_vfs_stays_walled(monkeypatch):
    monkeypatch.setenv("CLAW_VFS_SCOPE", "commons")
    client = _Asked(dispatch=MADE)
    refused = await CrewTool()._do_dispatch(client, harness="claude", brief="x", vfs_read=["vfs:Secret"])
    assert not refused.success and "can't reach Secret" in refused.error and client.asked == []
    assert (await CrewTool()._do_dispatch(client, harness="claude", brief="x", vfs_read=["vfs:commons/a"])).success


def test_the_page_offers_it_on_a_new_task_and_a_tasks_page():
    """Reading the source: the frontend has no test runner of its own for this."""
    source = Path(__file__).resolve().parents[2] / "flight-deck" / "src" / "components" / "crew"
    board = (source / "CrewBoard.tsx").read_text(encoding="utf-8")
    assert "vfs_folders: vfsFolders" in board and "<FolderPicker value={vfsFolders} onChange={setVfsFolders} />" in board
    assert "<TaskFolders task={task} canChange={canTell}" in board
    # A fieldset won't shrink below its content by itself, and a long folder row runs a phone's form off the screen.
    assert '<fieldset className="min-w-0">\n            <legend className={labelCls}>Folders from your files' in board
    picker = (source / "CrewFolders.tsx").read_text(encoding="utf-8")
    assert "const needed = open || value.length > 0" in picker, "what can be written is known before the tree is opened"
