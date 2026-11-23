"""Cognitive modes: every parameter is read somewhere, every mode has its
prompt text, the shared frame and a review lens."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from captain_claw import cognitive_mode as cm
from captain_claw.instructions import InstructionLoader

ROOT = Path(cm.__file__).resolve().parent
REAL_MODES = [m for m in cm.list_modes() if m.name != "neutra"]


@pytest.fixture(scope="module")
def loader() -> InstructionLoader:
    return InstructionLoader(base_dir=ROOT / "instructions", use_micro=False)


def test_every_param_is_read_outside_the_registry():
    """A parameter nobody reads is a promise the mode does not keep."""
    sources = "\n".join(
        p.read_text(encoding="utf-8")
        for p in ROOT.rglob("*.py")
        if p.name != "cognitive_mode.py"
    )
    for f in dataclasses.fields(cm.CognitiveModeParams):
        assert re.search(rf"\.{f.name}\b", sources), f"{f.name} is never read"


def test_neutra_is_a_no_op(loader):
    neutra = cm.get_mode("neutra")
    assert cm.cognitive_mode_to_prompt_block(neutra, loader) == ""
    assert cm.cognitive_mode_review_block(neutra) == ""
    assert neutra.params == cm.CognitiveModeParams()


@pytest.mark.parametrize("mode", REAL_MODES, ids=lambda m: m.name)
def test_mode_block_has_body_and_frame(mode, loader):
    block = cm.cognitive_mode_to_prompt_block(mode, loader)
    assert block.startswith(f"COGNITIVE MODE: {mode.label}")
    assert "How to hold this mode:" in block
    assert "never changes what counts as done or true" in block
    # Format braces would break the system prompt template.
    assert "{" not in block and "}" not in block


@pytest.mark.parametrize("mode", REAL_MODES, ids=lambda m: m.name)
def test_mode_text_does_not_invite_known_failures(mode, loader):
    text = cm.cognitive_mode_to_prompt_block(mode, loader).lower()
    for phrase in (
        "more questions than",
        "never \"finalize\"",
        "completion standard is loose",
        "do not second-guess",
        "longer, more detailed",
        "cognitive tempo",
    ):
        assert phrase not in text


@pytest.mark.parametrize("mode", REAL_MODES, ids=lambda m: m.name)
def test_review_lens(mode):
    block = cm.cognitive_mode_review_block(mode)
    assert block.startswith(f"Review lens: {mode.label}.")
    assert len(block) < 400


def test_no_loader_or_missing_file_injects_nothing(tmp_path):
    mode = cm.get_mode("phrygian")
    assert cm.cognitive_mode_to_prompt_block(mode, None) == ""
    empty = InstructionLoader(base_dir=tmp_path, use_micro=False)
    assert cm.cognitive_mode_to_prompt_block(mode, empty) == ""


def test_mode_to_dict_matches_params():
    for mode in cm.list_modes():
        d = cm.mode_to_dict(mode)
        assert set(d["params"]) == {f.name for f in dataclasses.fields(cm.CognitiveModeParams)}
        assert d["review_lens"] == mode.review_lens


def test_game_temperatures_follow_the_modes():
    t = {m.name: m.params.game_temperature for m in cm.list_modes()}
    assert all(0.4 <= v <= 1.0 for v in t.values())
    assert t["lydian"] > t["ionian"]
    assert t["lydian"] > t["aeolian"]


# ── The mode in logs ─────────────────────────────────────────────────


def test_effective_mode_name(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(cm, "load_agent_mode", lambda: "dorian")
    cfg = SimpleNamespace(cognitive_mode=SimpleNamespace(enabled=True))
    monkeypatch.setattr("captain_claw.config.get_config", lambda: cfg)
    assert cm.effective_mode_name() == "dorian"
    # An agent whose prompt was built says what that prompt carried.
    assert cm.effective_mode_name(SimpleNamespace(_cognitive_mode=cm.get_mode("lydian"))) == "lydian"
    assert cm.effective_mode_name(SimpleNamespace(_cognitive_mode=None)) == ""
    # Not built yet: what the process is set to.
    assert cm.effective_mode_name(SimpleNamespace()) == "dorian"
    cfg.cognitive_mode.enabled = False
    assert cm.effective_mode_name() == ""


async def test_llm_usage_rows_carry_the_mode(tmp_path, monkeypatch):
    import sqlite3

    from captain_claw.session import SessionManager

    monkeypatch.setattr("captain_claw.fd_usage_report.report_tokens", lambda *a, **k: None)
    monkeypatch.setattr(cm, "effective_mode_name", lambda agent=None: "aeolian")
    # A database made before the column existed gains it.
    db = tmp_path / "sessions.db"
    old = sqlite3.connect(db)
    old.execute("CREATE TABLE llm_usage (id TEXT PRIMARY KEY, session_id TEXT, interaction TEXT NOT NULL DEFAULT '',"
                " provider TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT '', prompt_tokens INTEGER NOT NULL"
                " DEFAULT 0, completion_tokens INTEGER NOT NULL DEFAULT 0, total_tokens INTEGER NOT NULL DEFAULT 0,"
                " cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0, cache_read_input_tokens INTEGER NOT NULL"
                " DEFAULT 0, input_bytes INTEGER NOT NULL DEFAULT 0, output_bytes INTEGER NOT NULL DEFAULT 0,"
                " streaming INTEGER NOT NULL DEFAULT 0, tools_enabled INTEGER NOT NULL DEFAULT 0, max_tokens INTEGER,"
                " finish_reason TEXT NOT NULL DEFAULT '', error INTEGER NOT NULL DEFAULT 0, latency_ms INTEGER NOT NULL"
                " DEFAULT 0, created_at TEXT NOT NULL)")
    old.execute("INSERT INTO llm_usage (id, created_at) VALUES ('old', '2026-01-01T00:00:00+00:00')")
    old.commit()
    old.close()
    sm = SessionManager(db_path=db)
    try:
        await sm.record_llm_usage(model="m", total_tokens=5)                          # side call: process mode
        await sm.record_llm_usage(model="m", total_tokens=7, cognitive_mode="phrygian")
        await sm.record_llm_usage(model="m", total_tokens=9, cognitive_mode="")       # no mode in effect
        rows = {r["total_tokens"]: r["cognitive_mode"] for r in await sm.query_llm_usage()}
    finally:
        await sm.close()
    assert rows == {0: "", 5: "aeolian", 7: "phrygian", 9: ""}
