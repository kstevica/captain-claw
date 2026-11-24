"""Drive (captain_claw/drive.py), phase 1: the levels, the turn scope, the
agent's default, and the level on usage rows. Nothing changes behaviour yet."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from captain_claw import drive


@pytest.fixture()
def cfg(monkeypatch):
    c = SimpleNamespace(drive=SimpleNamespace(default=""))
    monkeypatch.setattr("captain_claw.config.get_config", lambda: c)
    return c


def test_levels_run_from_relaxed_to_ambitious_and_diligent_is_the_default():
    assert [lvl.name for lvl in drive.list_levels()] == ["relaxed", "diligent", "persistent", "ambitious"]
    assert drive.DEFAULT == "diligent"
    for lvl in drive.list_levels():
        assert drive.level_to_dict(lvl) == {"name": lvl.name, "label": lvl.label, "summary": lvl.summary}
        assert lvl.label and lvl.summary


@pytest.mark.parametrize("raw, want", [
    ("Persistent", "persistent"), (" relaxed ", "relaxed"), ("stubborn", "persistent"),
    ("", None), (None, None), ("auto", None), ("default", None), ("turbo", None), (3, None),
])
def test_normalize(raw, want):
    assert drive.normalize(raw) == want


def test_nothing_picked_means_the_agents_default_then_diligent(cfg):
    assert drive.current_name() == "diligent" and drive.current_turn_drive() is None
    cfg.drive.default = "relaxed"
    assert drive.current_name() == "relaxed" and drive.current().label == "Relaxed"
    cfg.drive.default = "nonsense"
    assert drive.current_name() == "diligent"


def test_a_turns_pick_wins_and_ends_with_the_turn(cfg):
    cfg.drive.default = "relaxed"
    with drive.turn_drive("ambitious") as picked:
        assert picked == "ambitious" == drive.current_name()
        with drive.turn_drive("auto"):            # an inner turn that picked nothing
            assert drive.current_name() == "relaxed"
        assert drive.current_name() == "ambitious"
    assert drive.current_name() == "relaxed"


async def test_tasks_started_in_the_turn_inherit_it_and_others_do_not(cfg):
    async def seen() -> str:
        await asyncio.sleep(0)
        return drive.current_name()

    with drive.turn_drive("persistent"):
        inside = asyncio.create_task(seen())
    outside = asyncio.create_task(seen())          # a scheduled run started elsewhere
    assert await inside == "persistent"
    assert await outside == "diligent"


def test_a_broken_config_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("no config")

    monkeypatch.setattr("captain_claw.config.get_config", boom)
    assert drive.agent_default() is None and drive.current_name() == "diligent"


async def test_usage_rows_carry_the_level(tmp_path, monkeypatch, cfg):
    from captain_claw.session import SessionManager

    monkeypatch.setattr("captain_claw.fd_usage_report.report_tokens", lambda *a, **k: None)
    monkeypatch.setattr("captain_claw.cognitive_mode.effective_mode_name", lambda agent=None: "")
    sm = SessionManager(db_path=tmp_path / "sessions.db")
    try:
        await sm.record_llm_usage(model="m", total_tokens=1)
        with drive.turn_drive("relaxed"):
            await sm.record_llm_usage(model="m", total_tokens=2)                       # the turn's level
            await sm.record_llm_usage(model="m", total_tokens=3, drive="persistent")   # the caller's word
        rows = {r["total_tokens"]: r["drive"] for r in await sm.query_llm_usage()}
    finally:
        await sm.close()
    assert rows == {1: "diligent", 2: "relaxed", 3: "persistent"}


def test_the_config_has_the_setting():
    from captain_claw.config import Config

    assert Config().drive.default == ""
