"""Drive (captain_claw/drive.py): the levels and their limits, the turn scope,
the agent's default, the mode's suggestion, and the level on usage rows."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from captain_claw import drive


@pytest.fixture()
def cfg(monkeypatch):
    c = SimpleNamespace(drive=SimpleNamespace(default=""), cognitive_mode=SimpleNamespace(enabled=False))
    monkeypatch.setattr("captain_claw.config.get_config", lambda: c)
    return c


def test_levels_run_from_relaxed_to_ambitious_and_diligent_is_the_default():
    assert [lvl.name for lvl in drive.list_levels()] == ["relaxed", "diligent", "persistent", "ambitious"]
    assert drive.DEFAULT == "diligent"
    for lvl in drive.list_levels():
        d = drive.level_to_dict(lvl)
        assert (d["name"], d["label"], d["summary"]) == (lvl.name, lvl.label, lvl.summary)
        assert d["validation_retries"] == lvl.validation_retries and lvl.label and lvl.summary


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


# ── The limits a level sets ───────────────────────────────────────────


def test_every_limit_is_read_by_the_harness():
    """A limit nobody reads is a promise the level does not keep."""
    import dataclasses
    import re
    from pathlib import Path

    root = Path(drive.__file__).resolve().parent
    sources = "\n".join(p.read_text(encoding="utf-8") for p in root.rglob("*.py") if p.name != "drive.py")
    for f in dataclasses.fields(drive.DriveLevel):
        if f.name in ("name", "label", "summary"):
            continue
        assert re.search(rf"\.{f.name}\b", sources), f"{f.name} is never read"


def test_diligent_is_the_harness_as_it_was():
    from captain_claw.agent_orchestration_mixin import MAX_STALL_RETRIES, _turn_limits

    d = drive.LEVELS["diligent"]
    assert (d.step_ceiling_multiplier, d.step_ceiling_cap, d.free_extension_multiplier) == (4, 300, 2)
    assert (d.validation_retries, d.validation_retries_after_list, d.stall_retries) == (2, 1, MAX_STALL_RETRIES)
    assert _turn_limits(40, d) == (160, 80) and _turn_limits(300, d) == (300, 600)


def test_limits_only_grow_with_the_level_and_no_level_drops_a_stall_retry():
    from captain_claw.agent_orchestration_mixin import MAX_STALL_RETRIES, _turn_limits

    levels = drive.list_levels()
    for low, high in zip(levels, levels[1:]):
        for field in ("step_ceiling_multiplier", "step_ceiling_cap", "free_extension_multiplier",
                      "validation_retries", "validation_retries_after_list", "stall_retries"):
            assert getattr(low, field) <= getattr(high, field), (low.name, high.name, field)
    assert all(lvl.stall_retries >= MAX_STALL_RETRIES for lvl in levels)
    assert [_turn_limits(40, lvl) for lvl in levels] == [(80, 40), (160, 80), (240, 120), (240, 120)]
    assert _turn_limits(200, drive.LEVELS["relaxed"]) == (200, 200)       # never below the planned budget
    assert _turn_limits(200, drive.LEVELS["persistent"]) == (450, 600)


def test_the_step_budget_follows_the_turns_level(cfg, monkeypatch):
    from captain_claw.agent_orchestration_mixin import _turn_limits

    monkeypatch.delenv("CLAW_FLOATING_BUDGET", raising=False)
    assert _turn_limits(40) == (160, 80)
    with drive.turn_drive("relaxed"):
        assert _turn_limits(40) == (80, 40)
    cfg.drive.default = "persistent"
    assert _turn_limits(40) == (240, 120)
    monkeypatch.setenv("CLAW_FLOATING_BUDGET", "5000")                     # Bat's workers: unchanged
    with drive.turn_drive("relaxed"):
        assert _turn_limits(40) == (5000, 5000)


# ── What a cognitive mode suggests ────────────────────────────────────


def test_ionian_and_phrygian_suggest_persistent_and_any_pick_wins(cfg):
    from captain_claw import cognitive_mode as cm

    assert {m.name for m in cm.list_modes() if m.suggested_drive} == {"ionian", "phrygian"}
    assert all(drive.normalize(m.suggested_drive) for m in cm.list_modes() if m.suggested_drive)
    ionian = SimpleNamespace(_cognitive_mode=cm.get_mode("ionian"))
    dorian = SimpleNamespace(_cognitive_mode=cm.get_mode("dorian"))
    no_mode = SimpleNamespace(_cognitive_mode=None)                        # modes off, or a micro prompt
    assert drive.current_name(ionian) == "persistent"
    assert drive.current_name(dorian) == drive.current_name(no_mode) == "diligent"
    cfg.drive.default = "diligent"                                         # the agent's own default wins
    assert drive.current_name(ionian) == "diligent"
    cfg.drive.default = ""
    with drive.turn_drive("relaxed"):                                      # and a chat pick over both
        assert drive.current_name(ionian) == "relaxed"
