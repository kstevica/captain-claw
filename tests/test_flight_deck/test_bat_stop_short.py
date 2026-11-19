"""A run that stopped short: what its fix rounds saw, and what its owner sees.

What happened: a factory's run of a 60-row list ended with 54 rows, one group 6
short, after its fix rounds spent hours on the groups that were already full.
Their planner saw each step's first 300 characters and a broken check, never the
counts; the list loop ran over the owner's reference links and the run's own files,
and its "Processed 4/4" replaced the worker's report; the duplicate flag counted
the merged list as one more part; and the owner got "failed" with the last
worker's "reply continue".

Pinned here:

* Bat's workers run without the list loop;
* a step's report is kept without the step-limit notice (said in Bat's words) and
  the rating hint;
* Captain counts the result against the goal's amounts, from a table check or
  from the rules' own words, group by group;
* the fix round's planner and its workers get those counts and the newest
  finished steps' reports and progress notes;
* a list made from the others is no "other part", and isn't checked against them;
  a real repeat across parts still is;
* a stopped run's result starts with what it made, and its note is short.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from captain_claw import agent_stuck
from captain_claw.flight_deck import bat_course, bat_loop, bat_routes, code_contract
from captain_claw.flight_deck import first_mate_runs as fmr

SEGS = ["boutique consultancies", "research shops", "marketing agencies", "legal boutiques"]


def _leads(counts: dict[str, int], start: int = 0) -> str:
    rows, i = ["Firm,Email,Segment"], start
    for seg, n in counts.items():
        for _ in range(n):
            rows.append(f"Firm {i},f{i}@firm{i}.example,{seg}")
            i += 1
    return "\n".join(rows) + "\n"


# ── the list loop ──────────────────────────────────────────────────────

def test_bats_workers_run_without_the_list_loop():
    env = {e["key"]: e["value"] for e in bat_routes._bat_env("bat_x", "step-1", "u1")}
    assert env["CLAW_NO_SCALE"] == "1"


# ── a step's report ────────────────────────────────────────────────────

def test_a_reports_chat_manners_are_left_out():
    partial = agent_stuck.budget_partial("leads.csv now has 54 rows; boutique is 6 short.", 41)
    got = bat_loop.step_report(partial)
    assert "continue" not in got.lower() and got.startswith(bat_loop.RAN_OUT)
    assert got.endswith("leads.csv now has 54 rows; boutique is 6 short.")
    assert bat_loop.step_report(agent_stuck.MSG_BUDGET_EXHAUSTED) == bat_loop.RAN_OUT
    rated = ("All done.\n\n---\n\U0001f4a1 *If this worked well or could be improved, say \"rate good\" or "
             "\"rate bad\" to save this pattern for future tasks.*")
    assert bat_loop.step_report(rated) == "All done."
    assert bat_loop.step_report("8 firms added.\n\n## Counts\n| a | 1 |") == "8 firms added.\n\n## Counts\n| a | 1 |"
    assert bat_loop.step_report(None) == ""


# ── where the result stands ────────────────────────────────────────────

def test_the_counts_from_a_table_check(tmp_path):
    (tmp_path / "leads.csv").write_text(_leads({SEGS[0]: 9, SEGS[1]: 15, SEGS[2]: 15, SEGS[3]: 15}))
    rules = [{"id": "fm1", "text": "60 rows, 15 per segment", "check": {
        "type": "table", "path": "leads.csv", "min_rows": 60, "min_per": {"Segment": 15},
        "allowed": {"Segment": SEGS}}}]
    got = code_contract.standing(tmp_path, rules)
    assert got[0] == "`leads.csv`: 54 rows of 60 (6 more)"
    assert got[1] == ("`leads.csv` Segment: boutique consultancies 9 of 15 (6 more); research shops 15 of 15 (full); "
                      "marketing agencies 15 of 15 (full); legal boutiques 15 of 15 (full)")
    # a group with no rows yet still shows, when the rule names the groups
    (tmp_path / "leads.csv").write_text(_leads({SEGS[1]: 15, SEGS[2]: 15, SEGS[3]: 15}))
    assert "boutique consultancies 0 of 15 (15 more)" in code_contract.standing(tmp_path, rules)[1]


def test_the_counts_from_the_rules_own_words(tmp_path):
    """The night's check was a script cut short: the amounts were only in its words."""
    (tmp_path / "leads.csv").write_text(_leads({SEGS[0]: 9, SEGS[1]: 15}))
    rules = [{"id": "fm1", "severity": "critical",
              "text": "leads.csv has exactly 16 columns in the specified order, 60 unique rows, Segment exactly one "
                      "of the four named segments, and 15 per segment",
              "check": {"type": "command", "cmd": "python3 -c 'import csv; rows=list(csv.DictReader(open(\"leads.c"}}]
    got = code_contract.standing(tmp_path, rules)
    assert got == ["`leads.csv`: 24 rows of 60 (36 more)",
                   "`leads.csv` Segment: boutique consultancies 9 of 15 (6 more); research shops 15 of 15 (full)"]


def test_nothing_to_count(tmp_path):
    rules = [{"id": "fm1", "text": "Each reason is specific", "check": {"type": "judge"}}]
    assert code_contract.standing(tmp_path, rules) == []
    assert code_contract.standing(tmp_path, [{"id": "x", "text": "t", "check": {
        "type": "table", "path": "leads.csv", "min_rows": 10}}]) == ["`leads.csv` isn't there yet"]
    # its working folders aren't the result
    (tmp_path / "progress").mkdir()
    (tmp_path / "progress" / "leads.csv").write_text(_leads({SEGS[0]: 3}))
    words = [{"id": "fm1", "text": "progress/leads.csv has 60 rows", "check": {"type": "judge"}}]
    assert code_contract.standing(tmp_path, words, ("progress",)) == []


# ── the fix round ──────────────────────────────────────────────────────

@pytest.fixture
def planner(monkeypatch, tmp_path):
    """The fix round's model: it keeps what it was shown and plans one step."""
    from captain_claw.flight_deck import auth, basna_routes
    seen: list[str] = []

    class Prov:
        async def complete(self, messages, **_kw):
            seen.append(messages[-1].content)
            return type("R", (), {"content": json.dumps([{"step": "Add 6 boutique consultancies", "after": []}]),
                                  "usage": None, "model": "m"})()

    async def tiers(_db, _owner):
        return {"reason": {"model": "big"}}, []

    folder = tmp_path / "run"
    (folder / "progress").mkdir(parents=True)
    monkeypatch.setattr(auth, "get_db", lambda: None)
    monkeypatch.setattr(basna_routes, "_load_owner_tiers", tiers)
    monkeypatch.setattr(basna_routes, "_effective_key", lambda _p, k, _b=None: k)
    monkeypatch.setattr(basna_routes, "_provider_call", lambda *_a, **_k: (Prov(), 256))
    monkeypatch.setattr(bat_routes, "_run_folder", lambda _run: folder)
    return {"seen": seen, "folder": folder}


async def test_the_fix_round_plans_from_the_counts_and_the_newest_reports(planner):
    f = planner["folder"]
    (f / "leads.csv").write_text(_leads({SEGS[0]: 9, SEGS[1]: 15, SEGS[2]: 15, SEGS[3]: 15}))
    (f / "progress" / "step-10.md").write_text("Merged. Left: 6 boutique consultancies.")
    (f / "progress" / "step-2.md").write_text("an early note")
    run = {"id": "bat_short0001", "owner_id": "u1", "task": "Find 60 firms, 15 per segment.", "config": {
        "contract": [{"id": "fm1", "text": "60 unique rows and 15 per segment", "severity": "critical",
                      "check": {"type": "table", "path": "leads.csv", "min_rows": 60, "min_per": {"Segment": 15},
                                "allowed": {"Segment": SEGS}}}]}}
    table = "Merge done.\n\n" + "x" * 400 + "\n| boutique consultancies | 15 | 9 | 6 |"
    steps = [
        {"step_key": "step-2", "title": "Add 8 boutique consultancies", "status": "done", "seq": 1,
         "output": "Processed members: 4 / 4"},
        {"step_key": "step-9", "title": "Add 7 legal boutiques", "status": "done", "seq": 8,
         "output": agent_stuck.budget_partial("7 legal boutiques added.", 41)},
        {"step_key": "step-10", "title": "Merge into leads.csv", "status": "done", "seq": 9, "output": table},
    ]
    verdict = {"det_criticals": ["acceptance: 60 unique rows and 15 per segment (`leads.csv`: 54 rows)"]}
    plan = await bat_routes._bat_replan(run, steps, verdict, 1)
    shown = planner["seen"][0]
    assert "# Where the result stands" in shown and "boutique consultancies 9 of 15 (6 more)" in shown
    assert "marketing agencies 15 of 15 (full)" in shown
    assert shown.index("# Where the result stands") < shown.index("# The work so far")
    assert "| boutique consultancies | 15 | 9 | 6 |" in shown  # the newest step's whole report
    assert "Merged. Left: 6 boutique consultancies." in shown  # and its progress note
    assert "an early note" not in shown  # an older finished step's note isn't needed
    assert "reply **continue**" not in shown.lower() and bat_loop.RAN_OUT in shown
    assert "nothing for a group that is full" in bat_routes._REPLAN_SYSTEM
    # its workers get the counts too
    assert plan[0]["title"] == "Add 6 boutique consultancies"
    assert "boutique consultancies 9 of 15 (6 more)" in plan[0]["brief"]


async def test_no_counts_no_section(planner):
    run = {"id": "bat_short0002", "owner_id": "u1", "task": "Write a report.",
           "config": {"contract": [{"id": "a", "text": "it is clear", "check": {"type": "judge"}}]}}
    plan = await bat_routes._bat_replan(run, [], {"reason": "not clear yet"}, 1)
    assert "# Where the result stands" not in planner["seen"][0]
    assert plan[0]["brief"] == "- not clear yet"


# ── the duplicate flag ─────────────────────────────────────────────────

def _old(path):
    os.utime(path, (time.time() - 3600, time.time() - 3600))


def _parts(f, *, merged: bool = True):
    (f / "shared").mkdir(parents=True, exist_ok=True)
    (f / "shared" / "conventions.md").write_text("Columns: Firm,Email,Segment")
    (f / "part_a.csv").write_text(_leads({SEGS[0]: 4}, 0))
    (f / "part_b.csv").write_text(_leads({SEGS[1]: 4}, 10))
    (f / "part_c.csv").write_text(_leads({SEGS[2]: 4}, 20))
    if merged:
        rows = [ln for p in ("part_a", "part_b", "part_c") for ln in (f / f"{p}.csv").read_text().splitlines()[1:]]
        (f / "leads.csv").write_text("Firm,Email,Segment\n" + "\n".join(rows) + "\n")
    for p in f.glob("*.csv"):
        _old(p)


def _findings(f, step_key="fix-2-1"):
    run = {"id": "bat_dup00001", "owner_id": "u1", "config": {"contract": []}}
    step = {"step_key": step_key, "title": "Add 4 marketing agencies", "seq": 5, "round": 1}
    hard, _soft, _ = bat_course.script_findings(run, step, f, bat_course.changed_since(f, time.time() - 5), False)
    return hard


def test_the_merged_list_isnt_another_part(tmp_path):
    keys = {"a.csv": {"1", "2", "3"}, "b.csv": {"4", "5", "6"}, "c.csv": {"7", "8", "9"},
            "leads.csv": {str(i) for i in range(1, 10)}, "dup.csv": {"1", "2", "3", "x"}}
    assert bat_course._merged_lists(keys) == {"leads.csv"}
    # a part rewritten with its own rows, which the merged list already holds
    _parts(tmp_path)
    (tmp_path / "part_c.csv").write_text((tmp_path / "part_c.csv").read_text())
    assert not any("already in other parts" in h for h in _findings(tmp_path))
    # the merged list itself, rebuilt or added to
    (tmp_path / "leads.csv").write_text((tmp_path / "leads.csv").read_text() + "New,new@new.example,research shops\n")
    assert not any("already in other parts" in h for h in _findings(tmp_path, "fix-1-2"))


def test_a_real_repeat_across_parts_is_still_found(tmp_path):
    _parts(tmp_path)
    (tmp_path / "part_d.csv").write_text(_leads({SEGS[0]: 4}, 0))  # part_a's firms again
    assert any("4 of the 4 items in part_d.csv are already in other parts" in h for h in _findings(tmp_path))
    # with no merged list too
    other = tmp_path / "other"
    other.mkdir()
    _parts(other, merged=False)
    (other / "part_d.csv").write_text(_leads({SEGS[0]: 4}, 0))
    assert any("already in other parts" in h for h in _findings(other))


# ── what the owner sees ────────────────────────────────────────────────

def test_a_stopped_runs_note_is_short():
    long = ("no progress: the same checks failed after 2 fix rounds: deterministic critical check(s) failed: "
            "acceptance: leads.csv has 60 unique rows and 15 per segment (`python3 -c 'import csv,sys; rows=…` "
            "exited non-zero:\nCommand blocked: Command is not parseable)")
    got = fmr.stop_words(long)
    assert got == f"{fmr.NO_PROGRESS} Still not met: leads.csv has 60 unique rows and 15 per segment"
    assert fmr.stop_words("no progress") == fmr.NO_PROGRESS
    assert fmr.stop_words("llm_token_cap_reached") == fmr.STOP_WORDS["llm_token_cap_reached"]
    assert fmr.stop_words("") == "Bat stopped it"
    assert len(fmr.stop_words("x" * 900)) == fmr.STOP_NOTE_MAX


async def test_a_stopped_runs_result_starts_with_what_it_made(monkeypatch, tmp_path):
    (tmp_path / "leads.csv").write_text(_leads({SEGS[0]: 9, SEGS[1]: 15}))
    monkeypatch.setattr(bat_routes, "_run_folder", lambda _run: tmp_path)
    contract = [{"id": "fm1", "text": "30 rows, 15 per segment", "check": {
        "type": "table", "path": "leads.csv", "min_rows": 30, "min_per": {"Segment": 15},
        "allowed": {"Segment": SEGS[:2]}}}]
    steps = [
        {"step_key": "step-1", "title": "Find firms", "status": "done", "seq": 0, "output": "15 research shops."},
        {"step_key": "step-2", "title": "Merge", "status": "done", "seq": 1,
         "output": agent_stuck.budget_partial("Merged 24 rows.", 41)},
    ]

    class Store:
        async def get_run(self, _id):
            return {"id": "bat_stop0001", "owner_id": "u1", "status": "error", "cumulative_tokens": 5,
                    "stopped_reason": "no progress: the same checks failed after 2 fix rounds: acceptance: 30 rows",
                    "truth": "", "config": {"contract": contract}}

        async def list_steps(self, _id):
            return steps

    runner = fmr.BatRunner()
    monkeypatch.setattr(runner, "_store", lambda: Store())
    got = await runner.poll("bat_stop0001")
    assert got["status"] == "failed" and got["note"].startswith(fmr.NO_PROGRESS)
    result = got["result"]
    assert result.startswith("**What it made before it stopped**")
    assert "`leads.csv`: 24 rows of 30 (6 more)" in result and "boutique consultancies 9 of 15 (6 more)" in result
    assert "continue" not in result.lower() and "Merged 24 rows." in result
    assert result.index("What it made") < result.index("## Merge")


def test_nothing_counted_nothing_added(monkeypatch, tmp_path):
    monkeypatch.setattr(bat_routes, "_run_folder", lambda _run: tmp_path)
    import asyncio
    assert asyncio.run(fmr.made_before_stop({"id": "b", "owner_id": "u", "config": {}})) == ""
