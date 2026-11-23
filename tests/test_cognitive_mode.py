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
