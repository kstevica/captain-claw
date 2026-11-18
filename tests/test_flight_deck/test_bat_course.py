"""Checks between Bat steps: is what a step made still on course for the goal?

Pinned here:

* a step's part is checked by script against the owner's rules that hold for a
  part (counts aren't), the run's other lists' columns (when it has conventions)
  and items other parts already have; a broken rule sends it round again with
  why, and its last try is kept with the finding;
* what it broke becomes a correction every later step's prompt shows, above
  the conventions, and the final reviewers see too;
* a model looks only on a signal, after the first step writes the conventions
  (with the columns the rules ask for), and before the last step; its
  correction reaches the next steps;
* runs of fewer than three steps, coding steps and runs that turn it off aren't
  checked;
* Bat's loop retries a step the check sends back and tells it on its last try;
* First Mate says the run corrected its course.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from captain_claw.flight_deck import bat_course, bat_loop, bat_routes
from captain_claw.flight_deck import first_mate_results as results
from captain_claw.flight_deck.bat_loop import BatDriver
from captain_claw.flight_deck.bat_store import BatStore

RULES = [
    {"id": "fm1", "text": "At least 500 leads", "severity": "major",
     "check": {"type": "table", "path": "", "min_rows": 500, "columns": ["Name", "Email", "Source"]}},
    {"id": "fm2", "text": "No university staff", "severity": "major",
     "check": {"type": "table", "path": "", "exclude_pattern": r"\.ac\.uk"}},
    {"id": "fm3", "text": "Each reason is specific", "severity": "major", "check": {"type": "judge"}},
]


@pytest.fixture
async def bat(tmp_path, monkeypatch):
    b = BatStore(tmp_path / "bat.db")
    await b.init()

    async def kick():
        return 0

    monkeypatch.setattr(bat_loop, "_STORE", b)
    monkeypatch.setattr(bat_loop, "kick", kick)
    monkeypatch.setattr(bat_loop, "_STOP_CHILDREN", None)
    from captain_claw.flight_deck import vfs_routes
    monkeypatch.setattr(vfs_routes, "_user_root", lambda owner: tmp_path / "vfs" / owner)
    yield b
    await b.close()


class _Reply:
    def __init__(self, content):
        self.content = content


@pytest.fixture
def model(monkeypatch):
    """A model for the reviews: it answers what ``state["answer"]`` says, and keeps what it saw."""
    state = {"answer": {"state": "on_course", "why": "fine", "correction": ""}, "seen": []}

    class Prov:
        async def complete(self, messages, **kw):
            state["seen"].append(messages[1].content)
            return _Reply(json.dumps(state["answer"]))

    from captain_claw.flight_deck import auth, basna_routes

    async def tiers(_db, _owner):
        return {"reason": {"provider": "x", "model": "m"}}, None

    monkeypatch.setattr(auth, "get_db", lambda: None)
    monkeypatch.setattr(basna_routes, "_load_owner_tiers", tiers)
    monkeypatch.setattr(basna_routes, "_provider_call", lambda creds, **kw: (Prov(), 1000))
    return state


async def _run(bat, steps: int = 4, **cfg) -> dict:
    await bat.create_run(run_id="bat_course0001", owner_id="u1", title="t", task="Find 500 UK leads, no academics.",
                         status="running", config={"factory_id": "fac_1", "contract": RULES, **cfg})
    for i in range(steps):
        await bat.upsert_step("bat_course0001", f"step-{i + 1}", title=f"Find 50 leads, segment {i}" if i else "Write conventions",
                              status="done" if i == 0 else "pending", seq=i)
    return await bat.get_run("bat_course0001")


def _folder(run):
    f = bat_routes._run_folder(run)
    (f / "shared").mkdir(parents=True, exist_ok=True)
    return f


def _old(path):
    os.utime(path, (time.time() - 3600, time.time() - 3600))


# ── scripts on the part ─────────────────────────────────────────────────

def test_rules_for_a_part():
    got = bat_course.step_rules(RULES + [{"id": "x", "text": "1500 words", "check": {"type": "document", "min_words": 1500}}])
    assert [r["id"] for r in got] == ["fm1", "fm2"]
    assert "min_rows" not in got[0]["check"] and got[0]["check"]["columns"] == ["Name", "Email", "Source"]


async def test_a_part_that_breaks_the_rules_goes_round_again(bat):
    run = await _run(bat)
    f = _folder(run)
    (f / "shared" / "conventions.md").write_text("Columns: Name,Email,Source")
    (f / "batch_01.csv").write_text("Name,Email,Source\nAna,ana@a.example,https://a.example\nBo,bo@b.example,https://b.example\n"
                                    "Cy,cy@c.example,https://c.example\n")
    _old(f / "batch_01.csv")
    _old(f / "shared" / "conventions.md")
    started = time.time()
    (f / "batch_02.csv").write_text("name,email\nAna,ana@a.example\nBo,bo@b.example\nCy,cy@c.example\nDr X,x@ed.ac.uk\n")
    step = {"step_key": "step-2", "title": "Find 50 leads, segment 1", "seq": 1, "output": "Found 4."}
    got = await bat_course.check_step(run, step, started, last_try=False)
    assert got["redo"] is True
    why = got["why"]
    assert "breaks \"No university staff\"" in why and "missing columns: Source" in why
    assert "unlike the run's other lists" in why and "3 of the 4 items in batch_02.csv are already in other parts" in why
    fixes = bat_course.read_corrections(f)
    assert fixes and fixes[0].startswith("(found in step-2) Don't repeat:")
    prompt = bat_routes._build_step_prompt(run, {"step_key": "step-3", "title": "Find 50 leads, segment 2"}, [])
    assert "# Corrections from Captain's checks of earlier steps" in prompt
    assert prompt.index("Corrections from Captain") < prompt.index("The run's shared conventions")
    notes = [e["message"] for e in await bat.list_events("bat_course0001") if e["stage"] == "note"]
    assert any(n.startswith("step-2 goes round again:") for n in notes)
    last = await bat_course.check_step(run, step, started, last_try=True)
    assert last["redo"] is False
    notes = [e["message"] for e in await bat.list_events("bat_course0001") if e["stage"] == "note"]
    assert any(n.startswith("step-2 kept on its last try, with:") for n in notes)


async def test_a_clean_part_passes_without_a_model(bat):
    run = await _run(bat)
    f = _folder(run)
    started = time.time()
    (f / "batch_02.csv").write_text("Name,Email,Source\n" + "".join(
        f"P{i},p{i}@org{i}.example,https://org{i}.example/team\n" for i in range(50)))
    got = await bat_course.check_step(run, {"step_key": "step-2", "title": "Find 50 leads", "seq": 1, "output": ""},
                                      started, last_try=False)
    assert got == {"redo": False, "why": "", "state": ""}
    assert bat_course.read_corrections(f) == []


async def test_what_isnt_checked(bat):
    run = await _run(bat, steps=2)
    assert await bat_course.check_step(run, {"step_key": "step-2", "seq": 1}, time.time(), False) is None
    await bat.set_config("bat_course0001", {**run["config"], "course_check": False})
    for i in range(2, 5):
        await bat.upsert_step("bat_course0001", f"step-{i + 1}", title="x", status="pending", seq=i)
    off = await bat.get_run("bat_course0001")
    assert await bat_course.check_step(off, {"step_key": "step-2", "seq": 1}, time.time(), False) is None
    coding = {**off, "config": {**off["config"], "course_check": True, "step_hows": {"step-2": "code"}}}
    assert await bat_course.check_step(coding, {"step_key": "step-2", "seq": 1}, time.time(), False) is None


# ── the model's look ────────────────────────────────────────────────────

async def test_a_signal_gets_a_look_and_its_correction_reaches_the_next_steps(bat, model):
    run = await _run(bat, steps=5)
    f = _folder(run)
    model["answer"] = {"state": "drifting", "why": "half are role inboxes",
                       "correction": "Collect named people with personal emails; skip info@ and hello@ inboxes."}
    started = time.time()
    (f / "batch_02.csv").write_text("Name,Email,Source\n" + "".join(
        f"Org {i},info@org{i}.example,https://org{i}.example\n" for i in range(12)))
    got = await bat_course.check_step(run, {"step_key": "step-2", "title": "Find 50 leads", "seq": 1, "output": "12 leads"},
                                      started, last_try=False)
    assert got["redo"] is False and got["state"] == "drifting"
    seen = model["seen"][0]
    assert "# The goal\nFind 500 UK leads" in seen and "role inboxes" in seen and "came back short: 12 of the 50" in seen
    assert bat_course.read_corrections(f) == ["(after step-2) Collect named people with personal emails; skip info@ and hello@ inboxes."]
    prompt = bat_routes._build_step_prompt(run, {"step_key": "step-3", "title": "next"}, [])
    assert "skip info@ and hello@ inboxes" in prompt
    notes = [e["message"] for e in await bat.list_events("bat_course0001") if e["stage"] == "note"]
    assert any("course check after step-2: drifting (half are role inboxes)" in n for n in notes)
    # the same correction isn't given twice
    assert bat_course.add_correction(f, "again", "collect named people with personal emails; skip info@ and hello@ inboxes.") is False


async def test_the_conventions_and_the_last_step_get_a_look(bat, model):
    run = await _run(bat)
    f = _folder(run)
    started = time.time()
    (f / "shared" / "conventions.md").write_text("Columns: Name, Email")
    model["answer"] = {"state": "drifting", "why": "no Source column", "correction": "Add a Source column with the page's link."}
    await bat.upsert_step("bat_course0001", "step-1", status="running")
    got = await bat_course.check_step(run, {"step_key": "step-1", "title": "Write conventions", "seq": 0, "output": ""},
                                      started, last_try=False)
    assert got["state"] == "drifting"
    assert "just wrote the run's conventions" in model["seen"][-1]
    assert "don't name the column the rules ask for: Source" in model["seen"][-1]
    # every step but the last done: a look before the merge, with what the parts hold
    for i in (0, 1, 2):
        await bat.upsert_step("bat_course0001", f"step-{i + 1}", status="done")
    (f / "batch_02.csv").write_text("Name,Email,Source\nAna,ana@a.example,https://a.example\n")
    _old(f / "batch_02.csv")
    _old(f / "shared" / "conventions.md")  # written by step 1, a while ago
    model["answer"] = {"state": "on_course", "why": "fine", "correction": ""}
    await bat_course.check_step(await bat.get_run("bat_course0001"),
                                {"step_key": "step-3", "title": "Find 50 leads, segment 2", "seq": 2, "output": ""},
                                time.time(), last_try=False)
    assert "Every step but the last is done" in model["seen"][-1] and "VERIFIED" in model["seen"][-1]
    notes = [e["message"] for e in await bat.list_events("bat_course0001") if e["stage"] == "note"]
    assert any(n == "course check after step-3: on course" for n in notes)


async def test_a_short_step_with_more_to_try_goes_round_once(bat, model):
    """A real run: 4 of 25 rows after 15 minutes, called an "honest shortfall", with
    fifteen candidates still listed for later."""
    run = await _run(bat, steps=5)
    f = _folder(run)
    started = time.time()
    (f / "batch_02.csv").write_text("Name,Email,Source\n" + "".join(
        f"P{i},p{i}@org{i}.example,https://org{i}.example/team\n" for i in range(12)))
    model["answer"] = {"state": "drifting", "why": "12 of 50", "correction": "", "stopped_early": True}
    step = {"step_key": "step-2", "title": "Find 50 leads, segment 1", "seq": 1, "attempt": 1,
            "output": "12 leads; honest shortfall. Not yet tried: 15 firms on the list in progress/step-2.md"}
    got = await bat_course.check_step(run, step, started, last_try=False)
    assert got["redo"] is True and "came back short: 12 of the 50" in got["why"]
    assert "Keep going: pick up from your progress note" in got["error"]
    notes = [e["message"] for e in await bat.list_events("bat_course0001") if e["stage"] == "note"]
    assert any(n.startswith("step-2 goes round again: it stopped short with more to try") for n in notes)
    # the goal's own amounts go to the review, so a correction can't invent another
    assert "# The goal's amounts (quote these exactly, never another number)\n- At least 500 leads" in model["seen"][-1]
    assert '"stopped_early"' in bat_course._REVIEW_SYSTEM
    # once only: the second try is kept, short or not
    again = await bat_course.check_step(run, {**step, "attempt": 2}, started, last_try=False)
    assert again["redo"] is False
    # a real wall isn't stopping early
    model["answer"] = {"state": "drifting", "why": "sources ran dry", "correction": "", "stopped_early": False}
    assert (await bat_course.check_step(run, step, started, last_try=False))["redo"] is False
    # nor is a step on its last try
    model["answer"]["stopped_early"] = True
    assert (await bat_course.check_step(run, step, started, last_try=True))["redo"] is False


async def test_a_step_that_got_its_amount_never_goes_round_for_it(bat, model):
    run = await _run(bat, steps=5)
    f = _folder(run)
    started = time.time()
    (f / "batch_02.csv").write_text("Name,Email,Source\n" + "".join(
        f"P{i},p{i}@org{i}.example,https://org{i}.example/team\n" for i in range(45)))
    model["answer"] = {"state": "on_course", "why": "fine", "correction": "", "stopped_early": True}
    got = await bat_course.check_step(run, {"step_key": "step-2", "title": "Find 50 leads, segment 1", "seq": 1,
                                            "attempt": 1, "output": "45 leads"}, started, last_try=False)
    assert got["redo"] is False


# ── in Bat's loop, and on First Mate ────────────────────────────────────

async def test_the_loop_retries_a_step_the_check_sends_back(bat, monkeypatch):
    await bat.create_run(run_id="bat_loopchk01", owner_id="u1", title="t", task="t", status="running", config={"steps": []})
    await bat.upsert_step("bat_loopchk01", "s1", title="S1", status="pending", seq=0)
    calls = []

    async def check(run, step, started_at, last_try):
        calls.append((step["attempt"], last_try))
        return {"redo": step["attempt"] == 1, "why": "university staff in it"}

    async def runner(_run, step):
        return {"ok": True, "output": f"try {step.get('attempt')}"}

    async def judge(_run, _steps):
        return {"done": True, "reason": "good"}

    monkeypatch.setattr(bat_loop, "_STEP_CHECK", check)
    assert await BatDriver(bat, attempt_runner=runner, judge=judge, max_step_attempts=2).drive("bat_loopchk01") == "done"
    assert calls == [(1, False), (2, True)]
    [step] = await bat.list_steps("bat_loopchk01")
    assert step["status"] == "done" and step["attempt"] == 2
    fails = [json.loads(e["data"]) if isinstance(e["data"], str) else e["data"]
             for e in await bat.list_events("bat_loopchk01") if e["stage"] == "step"]
    assert any(d.get("ok") is False and "isn't on course: university staff" in d.get("detail", "") for d in fails)


async def test_the_loop_tells_a_retry_what_the_check_said(bat, monkeypatch):
    await bat.create_run(run_id="bat_loopchk02", owner_id="u1", title="t", task="t", status="running", config={"steps": []})
    await bat.upsert_step("bat_loopchk02", "s1", title="S1", status="pending", seq=0)

    async def check(run, step, started_at, last_try):
        return {"redo": step["attempt"] == 1, "why": "short", "error": "Keep going: 4 of 25, 15 candidates left"}

    async def runner(_run, step):
        return {"ok": True, "output": "done"}

    async def judge(_run, _steps):
        return {"done": True, "reason": "good"}

    monkeypatch.setattr(bat_loop, "_STEP_CHECK", check)
    assert await BatDriver(bat, attempt_runner=runner, judge=judge, max_step_attempts=2).drive("bat_loopchk02") == "done"
    fails = [json.loads(e["data"]) if isinstance(e["data"], str) else e["data"]
             for e in await bat.list_events("bat_loopchk02") if e["stage"] == "step"]
    assert any(d.get("ok") is False and d.get("detail", "").startswith("Keep going: 4 of 25") for d in fails)


def test_first_mate_says_it_corrected_course():
    line = results.worth_a_look({"checked": {"flags": []}, "corrections": ["(after step-2) skip info@", "(after step-5) new segments"]})
    assert line == "Worth a look: Captain corrected its course 2 times during the run"
