"""Tightening long runs, from what a real lead-generation run got wrong.

Pinned here:

* Captain reads what a run delivered, whatever its kind: a table (rows, broken
  rows, duplicates, role inboxes, university addresses, domains it leans on,
  links), a document (words, sections, placeholders, links), code (syntax errors,
  tests, TODOs), JSON; a code project is one result; the reviewers get it as
  VERIFIED, and a finished run keeps a summary without sample rows;
* done-checks can test a list (``table``), a document (``document``) and Python
  (``python_syntax``) by what they hold; a bad value is dropped, not trusted;
* a factory's own rules get those checks once per run (their words kept), and
  the judge hands back what the files hold;
* the planner splits by segment, has the first step write shared conventions
  and the last only merge, and can hand a step to a team or a coding run;
  the step's prompt says how, and shows the conventions and the seen list;
* one model (or one self-hosted machine) behind every tier: steps one at a time,
  and the driver keeps to a run's own limit;
* a run passed by a majority keeps the reviewer who said no on top of its result;
* a run waiting for its plan's approval counts as waiting for its owner;
* the finish message and the Results route say what's worth a look.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from captain_claw.flight_deck import bat_loop, bat_routes, code_contract
from captain_claw.flight_deck import deliverable_profile as dp
from captain_claw.flight_deck import first_mate_results as results
from captain_claw.flight_deck import first_mate_runs as runs
from captain_claw.flight_deck.bat_loop import BatDriver
from captain_claw.flight_deck.bat_store import BatStore


def _found(folder: Path) -> list:
    return bat_routes._run_files(folder)


def _leads(folder: Path, name: str = "final_leads.csv") -> Path:
    rows = ["Name,Email,Country,Source,Reason"]
    for i in range(30):
        rows.append(f"Org {i},info@org{i}.example,UK,org{i}.example homepage,fits")
    for i in range(20):
        rows.append(f"Prof {i},p{i}@ed.ac.uk,UK,https://www.ed.ac.uk/people/{i},fits")
    rows.append("Broken,row,with,too,many,fields")
    rows.append("Org 1,info@org1.example,UK,org1.example homepage,fits")
    p = folder / name
    p.write_text("\n".join(rows) + "\n")
    return p


# ── reading what was delivered ──────────────────────────────────────────

def test_a_table_is_read_for_what_it_holds(tmp_path):
    prof = dp.profile_table(_leads(tmp_path))
    assert prof["rows"] == 52 and prof["broken_rows"] == 1 and prof["duplicate_rows"] == 1
    assert prof["email"]["role"] == 31 and prof["email"]["academic"] == 20
    assert prof["links"] == {"column": "Source", "count": 52, "links": 20, "email_on_source_site": 20}
    flags = " | ".join(dp.flags(prof))
    for said in ("1 rows have a different number of fields", "1 duplicate rows", "role inboxes",
                 "university addresses", "of Source are links", "on the site their source points to"):
        assert said in flags, said


def test_a_document_and_code_are_read_too(tmp_path):
    doc = tmp_path / "report.md"
    doc.write_text("Intro. " + "word " * 1600 + "\n\nTODO: add the numbers [insert chart]\n")
    prof = dp.profile_document(doc)
    flags = dp.flags(prof)
    assert any("placeholders left" in f for f in flags) and any("no headings" in f for f in flags)
    assert any("no links or sources" in f for f in flags)
    good = tmp_path / "ok.md"
    good.write_text("# Report\n\n## Findings\n\n" + "word " * 400 + "\n\n## Sources\n\n- https://example.com/a\n")
    assert dp.flags(dp.profile_document(good)) == []
    code = tmp_path / "app"
    (code / "src").mkdir(parents=True)
    (code / "pyproject.toml").write_text("[project]\nname='x'\n")
    (code / "src" / "a.py").write_text("def f(:\n    pass\n")
    (code / "src" / "b.py").write_text("# TODO one\n# TODO two\nx = 1\n")
    found = _found(code)
    assert dp.is_code_project(found)
    prof = dp.profile(found)
    assert prof["kind"] == "code" and prof["main"][0]["path"] == "."
    flags = prof["main"][0]["flags"]
    assert any("syntax errors: src/a.py:1" in f for f in flags) and "no tests" in flags and "no README" in flags
    assert dp.sort_files(found, set())["main"] == found  # one result, all of it


def test_the_reviewers_block_and_what_a_run_keeps(tmp_path):
    _leads(tmp_path)
    (tmp_path / "leads_batch_01.csv").write_text("Name,Email\nA,a@x.example\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "fix.py").write_text("print(1)\n")
    prof = dp.profile(_found(tmp_path), {"final_leads.csv", "missing.csv"})
    assert [m["path"] for m in prof["main"]] == ["final_leads.csv"] and prof["missing"] == ["missing.csv"]
    block = dp.render(prof)
    assert block.startswith("## VERIFIED") and "WORTH A LOOK" in block and "MISSING: missing.csv" in block
    assert "first rows:" in block and "beside these: 1 other files and 1 working files" in block
    kept = dp.summary(prof)
    assert "sample" not in kept["main"][0]["profile"] and kept["flags"][0].startswith("a file its checks name is missing")


# ── checks a script decides ─────────────────────────────────────────────

def _rules(*checks: dict) -> list[dict]:
    return code_contract.parse_contract(json.dumps({"constraints": [
        {"id": f"a{i}", "text": f"rule {i}", "check": c} for i, c in enumerate(checks, 1)]}))


def test_table_and_document_checks(tmp_path):
    _leads(tmp_path)
    (tmp_path / "report.md").write_text("# Report\n\n## Method\n\n" + "word " * 300 + "\nTBD\n")
    rules = _rules(
        {"type": "table", "min_rows": 50, "columns": ["Name", "Email"], "unique": ["Email"]},
        {"type": "table", "path": "final_leads.csv", "max_role_share": 0.2, "link_columns": ["Source"],
         "exclude_pattern": r"\.ac\.uk", "allowed": {"Country": ["UK", "USA"]}},
        {"type": "document", "min_words": 200, "headings": ["Method", "Sources"], "no_placeholders": True},
        {"type": "python_syntax"},
        {"type": "table", "min_rows": "lots", "max_role_share": 7},  # nothing usable: a reviewer decides
    )
    assert rules[-1]["check"] == {"type": "judge"}
    res = asyncio.run(code_contract.validate(tmp_path, rules, skip=bat_routes.WORK_DIRS))
    failed = {r["id"]: r["note"] for r in res["failed"]}
    assert "a1" in failed and "repeated values in Email" in failed["a1"] and "different number of fields" in failed["a1"]
    assert "role inboxes" in failed["a2"] and "are links" in failed["a2"] and "match the excluded" in failed["a2"]
    assert "1 Country values outside UK, USA (e.g. with)" in failed["a2"]  # the broken row's shifted fields
    assert "no section on: Sources" in failed["a3"] and "placeholders left: TBD" in failed["a3"]
    assert [r["id"] for r in res["passed"]] == ["a4"] and [r["id"] for r in res["unresolved"]] == ["a5"]
    (tmp_path / "bad.py").write_text("def (:\n")
    res = asyncio.run(code_contract.validate(tmp_path, rules[3:4]))
    assert "bad.py:1" in res["failed"][0]["note"]
    empty = tmp_path / "empty"
    empty.mkdir()
    res = asyncio.run(code_contract.validate(empty, rules[:1]))
    assert "no table" in res["failed"][0]["note"]


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


async def test_a_factorys_rules_get_checks_once(bat, monkeypatch):
    rules = runs.bat_contract({"checks": {"contract": ["At least 500 leads in final_leads.csv",
                                                       "Each reason is specific to Captain"]}})
    await bat.create_run(run_id="bat_shape0001", owner_id="u1", title="t", task="Find 500 leads.", status="planning",
                         config={"factory_id": "fac_1", "contract": rules})
    seen = []

    class Prov:
        async def complete(self, messages, **kw):
            seen.append(messages[1].content)
            return _Reply(json.dumps({"constraints": [
                {"id": rules[0]["id"], "text": "changed words", "check": {"type": "table", "path": "final_leads.csv", "min_rows": 500}},
                {"id": rules[1]["id"], "text": rules[1]["text"], "check": {"type": "judge"}}]}))

    from captain_claw.flight_deck import basna_routes
    monkeypatch.setattr(basna_routes, "_provider_call", lambda creds, **kw: (Prov(), 1000))
    run = await bat.get_run("bat_shape0001")
    await bat_routes._shape_contract(run, {"provider": "x"}, False)
    cfg = (await bat.get_run("bat_shape0001"))["config"]
    assert cfg["contract_shaped"] is True
    assert cfg["contract"][0]["text"] == "At least 500 leads in final_leads.csv"  # the owner's words stay
    assert cfg["contract"][0]["check"] == {"type": "table", "path": "final_leads.csv", "min_rows": 500}
    assert cfg["contract"][1]["check"] == {"type": "judge"}
    assert "Each reason is specific" in seen[0]
    await bat_routes._shape_contract(await bat.get_run("bat_shape0001"), {"provider": "x"}, False)
    assert len(seen) == 1  # once per run
    notes = [e["message"] for e in await bat.list_events("bat_shape0001") if e["stage"] == "note"]
    assert any("1 of its 2 done-checks are checked by Captain itself" in n for n in notes)


async def test_the_judge_hands_back_what_the_files_hold(bat, tmp_path, monkeypatch):
    from captain_claw.flight_deck import auth as fd_auth
    from captain_claw.flight_deck import basna_routes
    from captain_claw.flight_deck.auth import set_auth_db
    from captain_claw.flight_deck.db import FlightDeckDB
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)

    async def no_tiers(*_a, **_kw):
        raise RuntimeError("no tiers here")

    monkeypatch.setattr(basna_routes, "_load_owner_tiers", no_tiers)
    try:
        await bat.create_run(run_id="bat_judge0001", owner_id="u1", title="t", task="Find leads.", status="running",
                             config={"contract": []})
        await bat.upsert_step("bat_judge0001", "s1", title="Find", status="done", seq=0, output="done")
        folder = bat_routes._run_folder({"id": "bat_judge0001", "owner_id": "u1"})
        folder.mkdir(parents=True)
        _leads(folder)
        verdict = await bat_routes._bat_judge(await bat.get_run("bat_judge0001"), await bat.list_steps("bat_judge0001"))
        assert verdict["profile"]["kind"] == "table"
        assert any("role inboxes" in f for f in verdict["profile"]["flags"])
    finally:
        await db.close()
        fd_auth._db = prev


# ── planning: segments, conventions, who does a step ────────────────────

def test_the_planner_can_hand_a_step_to_a_team():
    steps = bat_routes._parse_plan(json.dumps([
        {"step": "Write conventions", "after": []},
        {"step": "Research the market", "after": [1], "how": "vatra"},
        {"step": "Build the tool", "after": [1], "how": "CODE"},
        {"step": "Merge", "after": [2, 3], "how": "helicopter"},
    ]), "task")
    assert [s.get("how") for s in steps] == [None, "vatra", "code", None]
    assert "split by segment" in bat_routes._STEP_SIZE and "shared/conventions.md" in bat_routes._STEP_SIZE
    assert "does no new research" in bat_routes._STEP_SIZE and "Never lower the bar" in bat_routes._FACTORY_STEP_SYSTEM


def test_a_steps_prompt_says_how_and_shows_the_conventions(tmp_path, monkeypatch):
    from captain_claw.flight_deck import vfs_routes
    monkeypatch.setattr(vfs_routes, "_user_root", lambda owner: tmp_path / owner)
    run = {"id": "bat_0123456789ab", "owner_id": "u1", "task": "Collect leads.",
           "config": {"step_hows": {"step-2": "vatra", "step-3": "crew"}}}
    prompt = bat_routes._build_step_prompt(run, {"step_key": "step-2", "title": "Research"}, [])
    assert "# How to do this step: a Vatra team" in prompt and 'mode "vatra"' in prompt
    assert "vfs:bat-bat_0123/" in prompt and "do the step yourself" in prompt
    assert "a coding agent (Crew)" in bat_routes._build_step_prompt(run, {"step_key": "step-3", "title": "Build"}, [])
    assert "# How to do this step" not in bat_routes._build_step_prompt(run, {"step_key": "step-1", "title": "x"}, [])
    assert "shared conventions" not in prompt
    shared = tmp_path / "u1" / "bat-bat_0123" / "shared"
    shared.mkdir(parents=True)
    (shared / "conventions.md").write_text("Columns: Name,Email,Source,Reason")
    (shared / "seen.csv").write_text("email\n")
    prompt = bat_routes._build_step_prompt(run, {"step_key": "step-1", "title": "x"}, [])
    assert "Columns: Name,Email,Source,Reason" in prompt and "check vfs:bat-bat_0123/shared/seen.csv" in prompt
    assert "shared" in bat_routes.WORK_DIRS


# ── one model, one step at a time ───────────────────────────────────────

def test_one_machine():
    one = {"fast": {"provider": "ollama", "model": "m", "base_url": "http://x.example"},
           "reason": {"provider": "ollama", "model": "m", "base_url": "http://x.example"}}
    assert bat_routes.one_machine(one)
    local = {"fast": {"provider": "ollama", "model": "a", "base_url": "http://192.168.1.5:11434"},
             "reason": {"provider": "ollama", "model": "b", "base_url": "http://192.168.1.5:11434"}}
    assert bat_routes.one_machine(local)
    cloud = {"fast": {"provider": "anthropic", "model": "a"}, "reason": {"provider": "openai", "model": "b"}}
    assert not bat_routes.one_machine(cloud) and not bat_routes.one_machine({})


async def test_a_runs_own_limit_keeps_steps_one_at_a_time(bat):
    await bat.create_run(run_id="bat_par00001", owner_id="u1", title="t", task="t", status="running",
                         config={"steps": [], "max_parallel": 1})
    for i in range(3):
        await bat.upsert_step("bat_par00001", f"s{i}", title=f"S{i}", status="pending", seq=i)
    live = {"now": 0, "most": 0}

    async def runner(_run, _step):
        live["now"] += 1
        live["most"] = max(live["most"], live["now"])
        await asyncio.sleep(0.05)
        live["now"] -= 1
        return {"ok": True, "output": "done"}

    async def judge(_run, _steps):
        return {"done": True, "reason": "good", "panel": {"votes": [
            {"vote": "agree", "reason": "fine"}, {"vote": "disagree", "reason": "half the emails are role inboxes"},
            {"vote": "agree", "reason": "fine"}]}}

    assert await BatDriver(bat, attempt_runner=runner, judge=judge).drive("bat_par00001") == "done"
    assert live["most"] == 1
    truth = (await bat.get_run("bat_par00001"))["truth"]
    assert truth.startswith("> 1 of 3 reviewers objected: half the emails are role inboxes")
    assert bat_loop.run_parallel({"config": {"max_parallel": "2"}}) == 2 and bat_loop.run_parallel(None) == 0


async def test_one_model_sets_the_runs_limit(bat):
    await bat.create_run(run_id="bat_one00001", owner_id="u1", title="t", task="t", status="planning", config={})
    tiers = {"fast": {"provider": "ollama", "model": "m", "base_url": "http://x.example"}}
    await bat_routes._one_at_a_time(await bat.get_run("bat_one00001"), tiers)
    assert (await bat.get_run("bat_one00001"))["config"]["max_parallel"] == 1


# ── First Mate: waiting, and what's worth a look ────────────────────────

def _ev(ts, stage, message="", **data):
    return {"ts": ts, "stage": stage, "message": message, "data": data}


def test_waiting_for_a_plans_approval_is_waiting():
    events = [_ev(0, "plan", "6 step(s) planned"), _ev(1, "awaiting_plan", "needs approval"),
              _ev(4700, "note", "First Mate answered for its factory"), _ev(4701, "phase", "running"),
              _ev(4710, "step", "A", agent="a", attempt=1), _ev(5010, "step", "A", agent="a", ok=True)]
    got = results.timing_of(events, 0, 5100, now=9999)
    assert got["waiting_s"] == 4700 and got["working_s"] == 300


def test_worth_a_look():
    got = {"checked": {"flags": ["final_leads.csv: 43% role inboxes", "final_leads.csv: 39% university"]},
           "checks": [{"text": "at least 500 rows (480 rows)", "passed": False}],
           "dissent": "1 of 3 reviewers objected: third-party emails"}
    line = results.worth_a_look(got)
    assert line.startswith("Worth a look: failed checks: at least 500 rows (480 rows) · final_leads.csv: 43% role inboxes")
    assert line.endswith("1 of 3 reviewers objected: third-party emails")
    assert results.worth_a_look({"checked": {"flags": []}}) == ""
    assert results._check_results({"det_criticals": ["acceptance: at least 500 rows (480)", "all planned steps completed"]}) \
        == [{"text": "at least 500 rows (480)", "passed": False}]


def test_the_final_step_first_keeps_the_dissent_on_top():
    steps = [{"step_key": "s1", "seq": 0, "title": "Batch", "status": "done", "output": "50 leads"},
             {"step_key": "s2", "seq": 1, "title": "Merge", "status": "done", "output": "500 leads"}]
    text = runs.final_first(steps, "> 1 of 3 reviewers objected: thin\n\n## Batch\n\n50 leads")
    assert text.startswith("> 1 of 3 reviewers objected: thin\n\n## Merge")


def test_drafting_asks_for_checkable_done_checks():
    from captain_claw.flight_deck import first_mate_drafting as d
    assert d.DONE_WHEN_MAX == 6
    assert "Make the countable ones countable" in d.SYSTEM and "who is out" in d.SYSTEM
