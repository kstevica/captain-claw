"""A step that runs again picks up where it stopped (piece 3).

Pinned here:

* a step's first turn gets no resume section;
* a step worked on before is told not to start over and gets its progress
  note (fenced as what the last attempt wrote), a pointer to its research
  notes when it has them, and the run folder's files as they are now, newest
  first, without the workers' notes; with no note, it's told to work out from
  the files what is done;
* an interrupted step (a stop, a cap, a restart) gives its cut-off attempt
  back: only failures use up its tries.
"""

from __future__ import annotations

import os
import time

import pytest

from captain_claw.flight_deck import bat_routes
from captain_claw.flight_deck.bat_store import BatStore

RUN = {"id": "bat_0123456789ab", "owner_id": "u1", "task": "Collect 200 leads.", "config": {}}
STEP = {"step_key": "step-3", "title": "Collect candidates", "attempt": 0}


@pytest.fixture
def folder(tmp_path, monkeypatch):
    from captain_claw.flight_deck import vfs_routes
    monkeypatch.setattr(vfs_routes, "_user_root", lambda owner: tmp_path / owner)
    f = tmp_path / "u1" / "bat-bat_0123"
    f.mkdir(parents=True)
    return f


def test_a_first_turn_starts_fresh(folder):
    (folder / "leads.csv").write_text("name,email,reason\n")
    assert bat_routes._resume_section(RUN, STEP) == ""
    assert "# Pick up where" not in bat_routes._build_step_prompt(RUN, STEP, [])


def test_a_step_worked_on_before_picks_up_from_its_note_and_files(folder):
    (folder / "progress").mkdir()
    (folder / "notes").mkdir()
    (folder / "progress" / "step-3.md").write_text("Done: 47 of 200 in leads.csv.\nNext: UK fintech.")
    (folder / "notes" / "bat_0123-step-3-notes.md").write_text("Sources: a.example, b.example")
    (folder / "lead_criteria.md").write_text("B2B, 10-500 people")
    (folder / "leads.csv").write_text("name,email,reason\n" + "x,y,z\n" * 47)
    old = time.time() - 3600
    os.utime(folder / "lead_criteria.md", (old, old))
    prompt = bat_routes._build_step_prompt(RUN, STEP, [])  # interrupted: its try was given back
    assert "# Pick up where the last attempt stopped" in prompt and "Don't start over" in prompt
    assert "----- note -----\nDone: 47 of 200 in leads.csv.\nNext: UK fintech.\n----- end of note -----" in prompt
    assert "vfs:bat-bat_0123/notes/bat_0123-step-3-notes.md: read them before searching again" in prompt
    listing = prompt.split("The run's folder now, newest first:\n", 1)[1]
    assert listing.index("leads.csv") < listing.index("lead_criteria.md")  # newest first
    assert "progress/" not in listing and "notes/" not in listing
    assert "changed 60 min ago" in listing


def test_a_failed_step_with_no_note_works_it_out_from_the_files(folder):
    tried = {**STEP, "attempt": 1, "error": "the model call timed out"}
    section = bat_routes._resume_section(RUN, tried)
    assert "It left no progress note" in section and "holds no results yet" in section
    prompt = bat_routes._build_step_prompt(RUN, tried, [])
    assert "# Your last attempt at this step failed" in prompt and "# Pick up where" in prompt


async def test_an_interruption_gives_its_try_back(tmp_path):
    store = BatStore(tmp_path / "bat.db")
    await store.init()
    try:
        await store.create_run(run_id="bat_x", owner_id="u1", title="t", task="t", config={}, status="running")
        await store.upsert_step("bat_x", "a", title="A", status="running", attempt=2, seq=0)
        await store.upsert_step("bat_x", "b", title="B", status="running", attempt=0, seq=1)
        await store.upsert_step("bat_x", "c", title="C", status="failed", attempt=2, seq=2)
        assert await store.demote_running_steps("bat_x") == 2
        got = {s["step_key"]: (s["status"], s["attempt"]) for s in await store.list_steps("bat_x")}
        assert got == {"a": ("pending", 1), "b": ("pending", 0), "c": ("failed", 2)}  # a failure keeps its count
    finally:
        await store.close()
