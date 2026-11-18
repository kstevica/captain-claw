"""Save as you go: a Bat step's worker keeps its results and a progress note in
the run's folder while it works, so an interrupted step loses little.

Pinned here:

* every step's prompt says where results go as they're made, and where its
  progress note is (progress/<step>.md, the step's key made safe for a file);
* the deck reads a step's progress note back, capped, and finds none cleanly;
* the workers' notes (progress/, notes/) are left out of what the reviewers see
  as the run's files and out of a check that scans the whole folder; the same
  names deeper down, and every other file, are not;
* First Mate's progress panel carries an unfinished step's note.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from captain_claw.flight_deck import bat_routes, code_contract

RUN = {"id": "bat_0123456789ab", "owner_id": "u1", "task": "Collect 200 leads.", "config": {}}


@pytest.fixture
def folder(tmp_path, monkeypatch):
    from captain_claw.flight_deck import vfs_routes
    monkeypatch.setattr(vfs_routes, "_user_root", lambda owner: tmp_path / owner)
    f = tmp_path / "u1" / "bat-bat_0123"
    f.mkdir(parents=True)
    return f


def test_every_step_is_told_to_save_as_it_goes():
    step = {"step_key": "step-3", "title": "Collect candidates"}
    prompt = bat_routes._build_step_prompt(RUN, step, [])
    assert "# Save as you go" in prompt and "vfs:bat-bat_0123/progress/step-3.md" in prompt
    assert "every 10-25 items" in prompt
    factory = bat_routes._build_step_prompt({**RUN, "config": {"factory_id": "fac_1"}}, step, [])
    assert "# Save as you go" in factory
    assert bat_routes.progress_path(RUN, "fix 2: the list / again") == "vfs:bat-bat_0123/progress/fix-2-the-list-again.md"


def test_the_deck_reads_a_steps_note(folder):
    assert bat_routes.read_progress(RUN, "step-3") == ""
    (folder / "progress").mkdir()
    (folder / "progress" / "step-3.md").write_text("Done: 47 of 200 (leads.csv).\nNext: UK fintech.\n" + "x" * 9000)
    note = bat_routes.read_progress(RUN, "step-3")
    assert note.startswith("Done: 47 of 200") and len(note) == bat_routes.PROGRESS_MAX
    assert bat_routes.read_progress(RUN, "step-3", 20) == "Done: 47 of 200 (lea"


async def test_notes_are_not_the_result(folder):
    (folder / "progress").mkdir()
    (folder / "notes").mkdir()
    (folder / "data" / "progress").mkdir(parents=True)
    (folder / "progress" / "step-3.md").write_text("TODO: finish UK list")
    (folder / "notes" / "bat_0123-step-3-notes.md").write_text("TODO: check source")
    (folder / "leads.csv").write_text("name,email,reason\nAna,ana@x.example,uses agents\n")
    (folder / "data" / "progress" / "log.md").write_text("kept: a deliverable of the same name, deeper down")
    evidence = bat_routes._evidence_block(folder)
    assert "leads.csv" in evidence and "data/progress/log.md" in evidence
    assert "progress/step-3.md" not in evidence.replace("data/progress", "") and "notes/" not in evidence
    rule = [{"id": "c1", "text": "no TODO left", "severity": "major",
             "check": {"type": "no_pattern", "pattern": "TODO", "glob": "**/*"}}]
    assert (await code_contract.validate(folder, rule, skip=bat_routes.WORK_DIRS))["passed"]
    assert (await code_contract.validate(folder, rule))["failed"]  # Code mode's scan is as it was
