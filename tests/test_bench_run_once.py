"""captain-claw-once helpers: paths, instruction, workspace diff, result."""

from __future__ import annotations

import argparse
import time
from collections import Counter

import pytest

from captain_claw.bench import run_once as ro


def test_user_path_keeps_the_callers_home_after_restart(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "run-home"))
    monkeypatch.setenv(ro._REAL_HOME_ENV, str(tmp_path / "me"))
    assert ro.user_path("~/task.md") == (tmp_path / "me" / "task.md").resolve()
    assert ro.user_path("~") == (tmp_path / "me").resolve()


def test_user_path_plain_expanduser_without_restart(monkeypatch, tmp_path):
    monkeypatch.delenv(ro._REAL_HOME_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert ro.user_path("~/x") == (tmp_path / "x").resolve()


def _args(**kw) -> argparse.Namespace:
    args = ro.build_parser().parse_args([])
    for key, value in kw.items():
        setattr(args, key, value)
    return args


def test_instruction_file_wins_over_argument_and_env(monkeypatch, tmp_path):
    f = tmp_path / "task.md"
    f.write_text("  from the file \n", encoding="utf-8")
    monkeypatch.setenv(ro.INSTRUCTION_ENV, "from env")
    assert ro.read_instruction(_args(instruction="from arg", instruction_file=str(f))) == "from the file"
    assert ro.read_instruction(_args(instruction="from arg")) == "from arg"
    assert ro.read_instruction(_args()) == "from env"


def test_snapshot_diff_sees_created_changed_deleted_and_skips_git(tmp_path):
    (tmp_path / "keep.txt").write_text("a")
    (tmp_path / "edit.txt").write_text("a")
    (tmp_path / "gone.txt").write_text("a")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref")
    before = ro.snapshot(tmp_path)
    assert ".git/HEAD" not in before

    time.sleep(0.01)
    (tmp_path / "edit.txt").write_text("changed")
    (tmp_path / "gone.txt").unlink()
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "answer.md").write_text("x")
    diff = ro.diff_snapshots(before, ro.snapshot(tmp_path))
    assert diff == {"created": ["output/answer.md"], "changed": ["edit.txt"], "deleted": ["gone.txt"]}


def test_snapshot_gives_up_on_huge_folders(monkeypatch, tmp_path):
    monkeypatch.setattr(ro, "_SNAPSHOT_LIMIT", 2)
    for i in range(4):
        (tmp_path / f"f{i}").write_text("x")
    assert ro.snapshot(tmp_path) is None
    assert ro.diff_snapshots(None, {}) is None


def test_tool_counts_counts_tool_messages_only():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "tool_name": "read"},
        {"role": "tool", "tool_name": "read"},
        {"role": "tool"},
        {"role": "assistant", "content": "ok"},
    ]
    assert ro.tool_counts(msgs) == Counter({"read": 2, "unknown": 1})


def test_bench_profile_and_isolation_never_keep_the_real_databases(tmp_path):
    from captain_claw.config import Config

    cfg = Config()
    cfg.session.path = "/home/someone/.captain-claw/sessions.db"
    ro.isolate_paths(cfg, tmp_path)
    ro.apply_bench_profile(cfg)
    assert cfg.session.path == str(tmp_path / "sessions.db")
    assert cfg.memory.path == str(tmp_path / "memory.db")
    assert cfg.memory.enabled is False and cfg.deep_memory.enabled is False
    assert str(cfg.tools.shell.default_policy) == "allow"


def test_result_payload_puts_messages_last_and_merges_extra(tmp_path):
    result = ro.build_result_payload(
        ok=True, error=None, answer="done", provider="anthropic", model="m", effort="",
        workspace=tmp_path, run_dir=tmp_path, started=time.monotonic(), usage={},
        messages=[{"role": "user"}], counts=Counter({"read": 2}), files=None,
        extra={"mode": "deck"},
    )
    assert list(result)[-1] == "messages"
    assert result["mode"] == "deck" and result["tool_calls"] == 2
    assert result["effort"] is None and result["cost_usd_estimate"] is None


def test_no_instruction_exits_2_before_starting_anything(monkeypatch, tmp_path):
    monkeypatch.delenv(ro.INSTRUCTION_ENV, raising=False)
    started = []
    monkeypatch.setattr("captain_claw.bench.deck_run.PrivateDeck.start",
                        lambda self: started.append(1))
    with pytest.raises(SystemExit) as exc:
        ro.main(["--run-dir", str(tmp_path), "-p", "anthropic", "-m", "m"])
    assert exc.value.code == 2 and not started
