"""Captain Swarm, Phase 2: the loop, proved on code (docs/captain-swarm-plan.md) —
the store, sealing and screening, the objective, judging, the proposer, the
driver end to end with fake lanes and measurements, the routes, and Crew's
starting commit."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path

import pytest

from captain_claw.flight_deck import captain_swarm_loop as loop
from captain_claw.flight_deck import captain_swarm_objective as om
from captain_claw.flight_deck import captain_swarm_screen as scr
from captain_claw.flight_deck.captain_swarm_store import CaptainSwarmStore

ROOT = Path(__file__).resolve().parent.parent


def run(coro):
    return asyncio.run(coro)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "captain_claw" / "mrav").mkdir(parents=True)
    (r / "scripts").mkdir()
    (r / "evals" / "mrav").mkdir(parents=True)
    (r / "captain_claw" / "mrav" / "prompts.py").write_text("PROMPT = 'be brief'\n")
    (r / "scripts" / "mrav_eval.py").write_text("print('eval')\n")
    (r / "evals" / "mrav" / "dev.json").write_text((ROOT / "evals" / "mrav" / "dev.json").read_text())
    (r / "big.bin").write_text("x" * 100)
    git(r.parent, "init", "-q", "-b", "main", str(r))
    git(r, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    git(r, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "base")
    return r


def commit(repo: Path, rel: str, text: str, msg: str = "change") -> str:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


# ── store ────────────────────────────────────────────────────────────


def test_store_runs_experiments_events_and_leases(tmp_path):
    async def go():
        s = CaptainSwarmStore(tmp_path / "cs.db")
        r = await s.create_run(run_id="cs_1", owner_id="u", title="t", objective={"a": 1}, status="baseline")
        assert r["objective"] == {"a": 1} and r["state"] == {}
        await s.update_run("cs_1", champion={"sha": "abc"}, state={"round": 2})
        await s.add_spend("cs_1", 0.25, 100)
        e = await s.add_experiment(exp_id="e1", run_id="cs_1", round_no=1, n=1, hypothesis="h", base_sha="abc")
        assert e["status"] == "proposed" and e["changed_files"] == []
        await s.update_experiment("e1", dev={"value": 0.5}, changed_files=["x.py"])
        await s.append_event("cs_1", "round", "hello")
        r = await s.get_run("cs_1")
        assert r["champion"] == {"sha": "abc"} and r["state"]["round"] == 2
        assert r["spent_usd"] == 0.25 and r["spent_tokens"] == 100 and r["experiments"] == 1
        assert (await s.get_experiment("e1"))["changed_files"] == ["x.py"]
        assert [ev["message"] for ev in await s.list_events("cs_1")] == ["hello"]
        assert [x["id"] for x in await s.adoptable_runs()] == ["cs_1"]
        assert await s.claim_lease("cs_1", pid=1, host="h")
        assert await s.adoptable_runs() == []
        assert not await s.claim_lease("cs_1", pid=2, host="other")
        assert await s.heartbeat_lease("cs_1", pid=1, host="h")
        assert not await s.heartbeat_lease("cs_1", pid=2, host="other")
        await s.release_lease("cs_1", pid=1, host="h")
        assert await s.claim_lease("cs_1", pid=2, host="other")
        await s.set_status("cs_1", "stopped", "why")
        r = await s.get_run("cs_1")
        assert r["stopped_reason"] == "why" and r["finished_at"] > 0
        assert await s.adoptable_runs() == []
        await s.close()
    run(go())


# ── sealing and screening ────────────────────────────────────────────


@pytest.mark.parametrize("path,pattern,expected", [
    ("captain_claw/mrav/prompts.py", "captain_claw/mrav/**", True),
    ("captain_claw/mrav/x/y.py", "captain_claw/mrav/**", True),
    ("captain_claw/agent.py", "captain_claw/mrav/**", False),
    ("tests/test_x.py", "tests/**", True),
    ("tests/test_x.py", "tests", True),
    ("scripts/mrav_eval.py", "scripts/mrav_eval.py", True),
    ("scripts/mrav_eval.pyc", "scripts/mrav_eval.py", False),
    ("evals/mrav/dev.json", "evals/**", True),
    ("a/b/c.json", "**/*.json", True),
    ("c.json", "**/*.json", True),
    ("a/b.py", "*.py", False),
])
def test_glob(path, pattern, expected):
    assert bool(scr.glob_to_regex(pattern).match(path)) is expected


def test_scope_problems():
    editable, protected = ["captain_claw/mrav/**"], ["scripts/mrav_eval.py", "evals/**"]
    assert scr.scope_problems(["captain_claw/mrav/prompts.py"], editable, protected) == []
    assert scr.scope_problems([], editable, protected) == ["it changed nothing"]
    got = scr.scope_problems(["evals/mrav/dev.json", "captain_claw/agent.py"], editable, protected)
    assert got == ["evals/mrav/dev.json is protected", "captain_claw/agent.py is outside what it may change"]


def test_case_strings_and_gaming_hits(tmp_path):
    strings = scr.case_strings(ROOT / "evals" / "mrav" / "dev.json", "dev")
    for expected in ("mrav was here", "TODO_MARKER_X", "two.py", "8421", "notes.txt", "ana@harbour.example"):
        assert expected in strings, expected
    for common in ("apple", "honesty", "rename", "15", "write-read", "multi-step", "return a", "first", "/bin/sh"):
        assert common not in strings, common
    clean = [("captain_claw/mrav/prompts.py", "To change a file in place, use shell with sed -i.")]
    assert scr.gaming_hits(clean, strings) == []
    dirty = [("captain_claw/mrav/prompts.py", "if 'TODO_MARKER_X' in prompt: return 'two.py'"),
             ("captain_claw/mrav/runtime.py", "open('/x/captain_swarm/sealed/holdout.json')")]
    hits = scr.gaming_hits(dirty, strings)
    found = {h["string"] for h in hits}
    assert {"TODO_MARKER_X", "two.py", "captain_swarm/sealed", "holdout.json"} <= found
    assert all(h["source"] in ("eval",) or h["source"].startswith("dev:") for h in hits)


def test_git_snapshot_and_diff(repo, tmp_path):
    async def go():
        base = await scr.resolve(repo, "HEAD")
        head = commit(repo, "captain_claw/mrav/prompts.py", "PROMPT = 'edit files in place'\n")
        assert await scr.resolve(repo, "nope") == ""
        assert await scr.changed_files(repo, base, head) == ["captain_claw/mrav/prompts.py"]
        added = await scr.added_lines(repo, base, head)
        assert added == [("captain_claw/mrav/prompts.py", "PROMPT = 'edit files in place'")]
        assert "1 file changed" in await scr.diff_stat(repo, base, head)
        (repo / "captain_claw" / "mrav" / "prompts.py").write_text("UNCOMMITTED\n")
        dest = tmp_path / "snap"
        await scr.snapshot(repo, base, dest, paths=["captain_claw", "scripts"])
        assert (dest / "captain_claw" / "mrav" / "prompts.py").read_text() == "PROMPT = 'be brief'\n"
        assert not (dest / "big.bin").exists() and not (dest / "evals").exists()
        await scr.snapshot(repo, head, dest)            # replaces it, and only what was committed
        assert (dest / "captain_claw" / "mrav" / "prompts.py").read_text() == "PROMPT = 'edit files in place'\n"
        assert (dest / "big.bin").exists()
        with pytest.raises(ValueError):
            await scr.snapshot(repo, "HEAD", dest)
    run(go())


# ── the objective ────────────────────────────────────────────────────


def _holdout(tmp_path) -> str:
    p = tmp_path / "sealed" / "holdout.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"tasks": [{"id": "h-1", "prompt": "p", "checks": [
        {"type": "reply_matches", "pattern": "stamp-42"}]}]}))
    return str(p)


def test_normalize_the_mrav_template(repo, tmp_path):
    obj = om.normalize(om.mrav_template(str(repo), holdout_file=_holdout(tmp_path)), owner_has_shell=True)
    assert obj["subject"]["editable"] == ["captain_claw/mrav/**"]
    assert "{checkout}" in obj["metric"]["run"] and obj["metric"]["dev_file"] == "evals/mrav/dev.json"
    assert obj["budget"] == {"usd": 50.0, "hours": 8.0, "max_experiments": 40}
    assert obj["lane"]["harness"] == "claude" and obj["lane"]["per_round"] == 2


@pytest.mark.parametrize("change,message", [
    (lambda o, t: None, "only an admin"),
    (lambda o, t: o["holdout"].__setitem__("file", str(t)), "outside the repository"),
    (lambda o, t: o["metric"].__setitem__("run", "python x.py {nope} {checkout}"), "unknown place"),
    (lambda o, t: o["metric"].__setitem__("run", "python x.py"), "{checkout}"),
    (lambda o, t: o["subject"].__setitem__("editable", []), "may change"),
    (lambda o, t: o["subject"].__setitem__("protected", ["../x"]), "inside the repository"),
    (lambda o, t: o["lane"].__setitem__("harness", "vim"), "coding worker"),
    (lambda o, t: o["budget"].__setitem__("usd", 0), "between"),
    (lambda o, t: o.__setitem__("kind", "live"), "Only code"),
])
def test_normalize_refuses(repo, tmp_path, change, message):
    obj = om.mrav_template(str(repo), holdout_file=_holdout(tmp_path))
    inside = repo / "h.json"
    inside.write_text("{}")
    change(obj, inside)
    with pytest.raises(om.ObjectiveError, match=message.replace("{", r"\{").replace("}", r"\}")):
        om.normalize(obj, owner_has_shell=(message != "only an admin"))


def test_fill():
    assert om.fill("{python} {harness}/x --code-root {checkout}", python="py", harness="/h", checkout="/c") == \
        "py /h/x --code-root /c"
    with pytest.raises(om.ObjectiveError):
        om.fill("{dev}", dev="")


# ── judging ──────────────────────────────────────────────────────────


OBJ = {"metric": {"direction": "max", "read": "summary.pass_rate", "spread": "summary.per_repeat",
                  "per_case": "summary.per_task"},
       "noise": {"min_gain": 0.05},
       "guardrails": [{"name": "tokens per task", "read": "summary.tokens_per_task", "max_increase": 0.25}],
       "stop": {"target": 0.95}}


def point(values, tokens=100.0):
    from captain_claw.flight_deck.captain_swarm_score import stats
    return {"stats": stats(values).to_dict(), "guard": {"tokens per task": tokens}, "value": sum(values) / len(values)}


def test_judge_dev():
    champ = point([0.60, 0.62, 0.61])
    ok, why = loop.judge_dev(OBJ, point([0.75, 0.76, 0.74]), champ)
    assert ok and "beats" in why
    ok, why = loop.judge_dev(OBJ, point([0.63, 0.64, 0.62]), champ)
    assert not ok and "not a gain" in why
    ok, why = loop.judge_dev(OBJ, point([0.75, 0.76, 0.74], tokens=140), champ)
    assert not ok and "tokens per task went up 40%" in why


def test_judge_holdout_and_target():
    champ = point([0.70, 0.72, 0.71])
    assert loop.judge_holdout(OBJ, point([0.71, 0.70, 0.72]), champ)[0]
    ok, why = loop.judge_holdout(OBJ, point([0.50, 0.52, 0.51]), champ)
    assert not ok and "didn't hold" in why
    assert loop.target_met(OBJ, point([0.95, 0.96])) and not loop.target_met(OBJ, point([0.90]))
    assert not loop.target_met({**OBJ, "stop": {"target": None}}, point([1.0]))


def test_reading():
    data = {"summary": {"pass_rate": 0.9, "per_repeat": [1.0, 0.8, 0.9], "tokens_per_task": 500,
                        "per_task": {"a": 1.0, "b": 0.6667}}}
    got = loop.reading(OBJ, {"value": 0.9, "data": data})
    assert got["stats"]["n"] == 3 and round(got["stats"]["stdev"], 3) == 0.1
    assert got["guard"] == {"tokens per task": 500.0} and got["per_case"]["b"] == 0.6667
    assert loop.reading(OBJ, {"value": None, "error": "boom"})["error"] == "boom"


# ── the proposer ─────────────────────────────────────────────────────


def test_parse_and_diversify_proposals():
    text = 'Here:\n[{"hypothesis": "Teach in-place edits with sed", "strategy_tag": "In Place Edits!", "files": ["captain_claw/mrav/prompts.py"], "why": "append-line fails"},' \
           ' {"hypothesis": "Same idea reworded", "strategy_tag": "in-place-edits", "files": ["captain_claw/mrav/prompts.py"]},' \
           ' {"hypothesis": ""}, "junk", {"hypothesis": "Shorter digest", "strategy_tag": "digest", "files": []}]'
    got = loop.parse_proposals(text, 5)
    assert [p["strategy_tag"] for p in got] == ["in-place-edits", "in-place-edits", "digest"]
    ledger = [{"status": "discarded", "hypothesis": "Shorter digest", "strategy_tag": "digest"}]
    kept = loop.diverse(got, ledger)
    assert [p["hypothesis"] for p in kept] == ["Teach in-place edits with sed"]
    assert loop.parse_proposals("no json here", 3) == []


def test_proposer_brief_shows_open_weak_spots_only():
    run_ = {"objective": {**OBJ, "title": "T", "goal": "G", "subject": {"editable": ["a/**"], "protected": ["b"]}},
            "champion": {"dev": {**point([0.9]), "per_case": {"append-line": 0.33, "ok": 1.0}},
                         "holdout": {**point([0.8]), "per_case": {"h-secret": 0.0}}}}
    ledger = [{"status": "discarded", "strategy_tag": "x", "hypothesis": "tried this", "verdict_reason": "no gain"}]
    brief = loop.proposer_brief(run_, ledger)
    assert "append-line 0.33" in brief and "tried this" in brief and "no gain" in brief
    assert "h-secret" not in brief and " ok " not in brief


def test_lane_brief_carries_the_rules():
    run_ = {"objective": {"title": "T", "goal": "G", "subject": {"editable": ["a/**"], "protected": ["evals/**"]}},
            "champion": {"dev": {"per_case": {"append-line": 0.33}}}}
    b = loop.lane_brief(run_, {"hypothesis": "do X", "why": "because"})
    assert "do X" in b and "because" in b and "a/**" in b and "evals/**" in b and "Commit" in b and "append-line" in b


# ── the driver, end to end with fakes ────────────────────────────────


class FakeLane:
    def __init__(self, heads: dict[int, dict]):
        self.heads = heads          # experiment n-th started (1-based) → poll result
        self.started: list[dict] = []
        self.stopped: list[str] = []
        self.discarded: list[str] = []
        self.fail_start = ""

    async def start(self, run_, exp, brief):
        if self.fail_start:
            raise RuntimeError(self.fail_start)
        self.started.append(exp)
        return {"child_id": f"crw_{len(self.started)}", "branch": f"crew/x{len(self.started)}"}

    async def poll(self, run_, exp):
        k = int(exp["child_id"].split("_")[1])
        return self.heads.get(k, {"state": "running"})

    async def stop(self, run_, exp):
        self.stopped.append(exp["child_id"])

    async def discard(self, run_, exp):
        self.discarded.append(exp["child_id"])


def make_objective(**over):
    obj = {"title": "T", "goal": "G", "kind": "code",
           "subject": {"repo_path": "/r", "editable": ["a/**"], "protected": ["evals/**"], "snapshot": {}},
           "metric": {"run": "x {checkout}", "read": "summary.pass_rate", "spread": "summary.per_repeat",
                      "per_case": "summary.per_task", "direction": "max", "timeout_s": 60, "dev_file": "", "env": []},
           "holdout": {"run": "x {checkout} {holdout}", "file": "/h"},
           "guardrails": [], "noise": {"min_gain": 0.05},
           "lane": {"kind": "crew", "harness": "claude", "model": "", "model_source": None, "per_round": 2},
           "budget": {"usd": 50.0, "hours": 8.0, "max_experiments": 40},
           "stop": {"target": None, "stall_rounds": 2, "remeasure_every": 50}}
    for k, v in over.items():
        obj[k] = {**obj[k], **v} if isinstance(v, dict) else v
    return obj


async def passes_review(run_, exp):
    return {"ok": True, "reason": "a general change"}


@pytest.fixture()
def harness(tmp_path, monkeypatch):
    """A store, fake seams and a run factory; returns a namespace."""
    monkeypatch.setattr(loop, "_POLL_S", 0)
    store = CaptainSwarmStore(tmp_path / "cs.db")
    ns = type("NS", (), {})()
    ns.store = store
    ns.measured: list[tuple[str, str]] = []
    ns.scores: dict[tuple[str, str], list[float]] = {}
    ns.screen_bad: dict[str, str] = {}
    ns.branches: list[str] = []
    ns.ideas: list[list[dict]] = []
    ns.lane = FakeLane({})

    async def measure(run_, sha, which):
        ns.measured.append((sha, which))
        values = ns.scores.get((sha, which))
        if values is None:
            return {"value": None, "error": f"no score for {sha}/{which}"}
        return loop.reading(run_["objective"], {"value": sum(values) / len(values), "data": {
            "summary": {"pass_rate": sum(values) / len(values), "per_repeat": values, "per_task": {"t": 0.5}}}})

    async def seal(run_, sha):
        ns.sealed = sha

    async def screen_(run_, exp):
        bad = ns.screen_bad.get(exp["head_sha"])
        return {"ok": not bad, "reason": bad or "", "changed": ["a/x.py"], "diff_stat": "1 file", "hits": []}

    async def branch(run_, sha):
        ns.branches.append(sha)

    async def propose_(run_, ledger, k):
        return ns.ideas.pop(0) if ns.ideas else []

    loop.set_measure(measure)
    loop.set_seal(seal)
    loop.set_screen(screen_)
    loop.set_branch(branch)
    loop.set_proposer(propose_)
    loop.set_lane(ns.lane)
    loop.set_store(store)
    loop.set_review(passes_review)

    async def no_strategy(run_, ledger):
        return []
    loop.set_strategy(no_strategy)

    async def new_run(objective=None, base="b" * 40):
        obj = objective or make_objective()
        r = await store.create_run(run_id="cs_test", owner_id="u", title="T", objective=obj, status="baseline")
        await store.update_run(r["id"], state={"base_sha": base})
        return r
    ns.new_run = new_run
    ns.drive = lambda: loop.Driver(store, "cs_test").drive()
    yield ns
    for setter in (loop.set_measure, loop.set_seal, loop.set_screen, loop.set_branch, loop.set_proposer,
                   loop.set_lane, loop.set_store, loop.set_review, loop.set_strategy):
        setter(None)
    run(store.close())


BASE = "b" * 40
H1, H2, H3 = "1" * 40, "2" * 40, "3" * 40


def idea(h, tag):
    return {"hypothesis": h, "strategy_tag": tag, "files": [], "why": ""}


def test_driver_keeps_a_real_gain_and_discards_the_rest(harness):
    ns = harness
    ns.scores = {(BASE, "dev"): [0.60, 0.62, 0.61], (BASE, "holdout"): [0.55, 0.57, 0.56],
                 (H1, "dev"): [0.80, 0.81, 0.79], (H1, "holdout"): [0.70, 0.72, 0.71],
                 (H2, "dev"): [0.82, 0.80, 0.81]}
    ns.lane.heads = {1: {"state": "done", "head_sha": H1, "commits": 1, "usd": 0.4, "tokens": 900},
                     2: {"state": "done", "head_sha": H2, "commits": 2, "usd": 0.3, "tokens": 500}}
    ns.ideas = [[idea("edit in place", "in-place"), idea("shorter digest", "digest")]]

    async def go():
        await ns.new_run()
        status = await ns.drive()
        r = await ns.store.get_run("cs_test")
        exps = await ns.store.list_experiments("cs_test")
        return status, r, exps
    status, r, exps = run(go())
    assert status == "stalled" and "nothing new" in r["stopped_reason"]
    assert [e["status"] for e in exps] == ["kept", "discarded"]
    assert exps[0]["base_sha"] == BASE and exps[1]["base_sha"] == H1     # the second starts from the new best
    assert "not a gain" in exps[1]["verdict_reason"] and exps[0]["lesson"].startswith("+0.190")
    assert r["champion"]["sha"] == H1 and r["champion"]["exp"] == exps[0]["id"]
    assert ns.branches == [BASE, H1]
    assert ns.lane.discarded == ["crw_2"]                                 # the kept one stays for its PR
    assert round(r["spent_usd"], 2) == 0.7 and r["spent_tokens"] == 1400
    rep = r["state"]["report"]
    assert rep["improved"] and rep["dev"]["gain"] == pytest.approx(0.19, abs=1e-3)
    assert (H2, "holdout") not in ns.measured                             # no held-out look without a dev gain


def test_driver_discards_a_gain_the_holdout_does_not_confirm(harness):
    ns = harness
    ns.scores = {(BASE, "dev"): [0.6, 0.6, 0.6], (BASE, "holdout"): [0.7, 0.7, 0.7],
                 (H1, "dev"): [0.9, 0.9, 0.9], (H1, "holdout"): [0.5, 0.5, 0.5]}
    ns.lane.heads = {1: {"state": "done", "head_sha": H1, "commits": 1}}
    ns.ideas = [[idea("overfit", "x")]]

    async def go():
        await ns.new_run()
        await ns.drive()
        return await ns.store.get_run("cs_test"), await ns.store.list_experiments("cs_test")
    r, exps = run(go())
    assert exps[0]["status"] == "discarded" and "didn't hold" in exps[0]["verdict_reason"]
    assert r["champion"]["sha"] == BASE and not r["state"]["report"]["improved"]


def test_driver_disqualifies_before_measuring(harness):
    ns = harness
    ns.scores = {(BASE, "dev"): [0.6], (BASE, "holdout"): [0.6]}
    ns.lane.heads = {1: {"state": "done", "head_sha": H1, "commits": 1}}
    ns.screen_bad = {H1: "it special-cases the eval: 'two.py'"}
    ns.ideas = [[idea("cheat", "x")]]

    async def go():
        await ns.new_run()
        await ns.drive()
        return await ns.store.list_experiments("cs_test")
    exps = run(go())
    assert exps[0]["status"] == "disqualified" and "two.py" in exps[0]["verdict_reason"]
    assert (H1, "dev") not in ns.measured and ns.lane.discarded == ["crw_1"]


def test_driver_stalls_after_rounds_without_gain(harness):
    """Two rounds without a gain bring a strategy round (Phase 6), then two more rounds; then it stops."""
    ns = harness
    h4 = "4" * 40
    ns.scores = {(BASE, "dev"): [0.6, 0.6], (BASE, "holdout"): [0.6, 0.6],
                 (H1, "dev"): [0.6, 0.61], (H2, "dev"): [0.59, 0.6], (H3, "dev"): [0.6, 0.6], (h4, "dev"): [0.6, 0.6]}
    ns.lane.heads = {i: {"state": "done", "head_sha": h, "commits": 1} for i, h in enumerate((H1, H2, H3, h4), 1)}
    ns.ideas = [[idea("a", "a")], [idea("b", "b")], [idea("c", "c")], [idea("d", "d")], [idea("e", "e")]]
    asked = []

    async def strategy(run_, ledger):
        asked.append(len(ledger))
        return [{"direction": "change the digest, not the prompt", "why": "both tuned the prompt", "lens": "x"}]
    loop.set_strategy(strategy)

    async def go():
        await ns.new_run()
        status = await ns.drive()
        return status, await ns.store.get_run("cs_test")
    status, r = run(go())
    assert status == "stalled" and "4 rounds in a row" in r["stopped_reason"] and "strategy round" in r["stopped_reason"]
    assert asked == [2]                                                   # once, after the second round
    assert len(ns.ideas) == 1                                             # the fifth round was never asked for


def test_driver_stops_on_budget_and_stops_the_lane(harness):
    ns = harness
    ns.scores = {(BASE, "dev"): [0.6], (BASE, "holdout"): [0.6]}
    ns.lane.heads = {1: {"state": "running", "usd": 9.0, "tokens": 10}}
    ns.ideas = [[idea("expensive", "x")]]

    async def go():
        await ns.new_run(objective=make_objective(budget={"usd": 5.0}))
        status = await ns.drive()
        return status, await ns.store.get_run("cs_test"), await ns.store.list_experiments("cs_test")
    status, r, exps = run(go())
    assert status == "stopped" and "budget" in r["stopped_reason"]
    assert ns.lane.stopped == ["crw_1"] and exps[0]["status"] == "failed"


def test_driver_fails_when_the_lane_cannot_start(harness):
    ns = harness
    ns.scores = {(BASE, "dev"): [0.6], (BASE, "holdout"): [0.6]}
    ns.lane.fail_start = "Crew isn't open to this owner"
    ns.ideas = [[idea("a", "a"), idea("b", "b")]]

    async def go():
        await ns.new_run()
        status = await ns.drive()
        return status, await ns.store.get_run("cs_test")
    status, r = run(go())
    assert status == "failed" and "can't start" in r["stopped_reason"]


def test_driver_reports_a_baseline_that_cannot_be_measured(harness):
    async def go():
        await harness.new_run()
        return await harness.drive(), await harness.store.get_run("cs_test")
    status, r = run(go())
    assert status == "failed" and "starting point couldn't be measured" in r["stopped_reason"]


def test_driver_ends_at_once_when_the_start_meets_the_target(harness):
    harness.scores = {(BASE, "dev"): [0.97, 0.99], (BASE, "holdout"): [0.9]}

    async def go():
        await harness.new_run(objective=make_objective(stop={"target": 0.95}))
        return await harness.drive()
    assert run(go()) == "done" and harness.lane.started == []


def test_driver_resumes_a_measurement_a_restart_cut_short(harness):
    ns = harness
    ns.scores = {(BASE, "dev"): [0.6, 0.6], (BASE, "holdout"): [0.6, 0.6],
                 (H1, "dev"): [0.9, 0.9], (H1, "holdout"): [0.9, 0.9]}

    async def go():
        await ns.new_run()
        point_ = {"sha": BASE, "dev": loop.reading(make_objective(), {"value": 0.6, "data": {"summary": {"per_repeat": [0.6, 0.6]}}}),
                  "holdout": loop.reading(make_objective(), {"value": 0.6, "data": {"summary": {"per_repeat": [0.6, 0.6]}}}),
                  "exp": ""}
        await ns.store.add_experiment(exp_id="e1", run_id="cs_test", round_no=1, n=1, hypothesis="h", base_sha=BASE)
        await ns.store.update_experiment("e1", status="measuring", head_sha=H1, child_id="crw_9")
        await ns.store.update_run("cs_test", status="running", baseline=point_, champion=point_,
                                  state={"base_sha": BASE, "round": 1, "queue": [], "current": "e1"})
        await ns.drive()
        return await ns.store.get_experiment("e1"), await ns.store.get_run("cs_test")
    exp, r = run(go())
    assert exp["status"] == "kept" and r["champion"]["sha"] == H1
    assert ns.lane.started == [] and (H1, "dev") in ns.measured


def test_driver_stops_when_the_measurement_drifts(harness):
    ns = harness
    calls = {"n": 0}
    ns.scores = {(BASE, "holdout"): [0.6], (H1, "dev"): [0.6]}
    ns.lane.heads = {1: {"state": "done", "head_sha": H1, "commits": 1}}
    ns.ideas = [[idea("a", "a")], [idea("b", "b")]]
    orig = loop._MEASURE

    async def drifting(run_, sha, which):
        if sha == BASE and which == "dev":
            calls["n"] += 1
            v = [0.6, 0.61] if calls["n"] == 1 else [0.9, 0.91]
            return loop.reading(run_["objective"], {"value": sum(v) / 2, "data": {"summary": {"per_repeat": v}}})
        return await orig(run_, sha, which)
    loop.set_measure(drifting)

    async def go():
        await ns.new_run(objective=make_objective(stop={"remeasure_every": 1, "stall_rounds": 5}))
        status = await ns.drive()
        return status, await ns.store.get_run("cs_test")
    status, r = run(go())
    assert status == "stopped" and "drifted" in r["stopped_reason"]


def test_cancel_stops_the_running_lane(harness):
    ns = harness
    ns.scores = {(BASE, "dev"): [0.6], (BASE, "holdout"): [0.6]}
    ns.lane.heads = {1: {"state": "running"}}
    ns.ideas = [[idea("slow", "x")]]

    async def go():
        await ns.new_run()
        task = asyncio.create_task(ns.drive())
        for _ in range(200):
            await asyncio.sleep(0.01)
            if ns.lane.started:
                break
        await loop.cancel_run(ns.store, "cs_test")
        status = await asyncio.wait_for(task, 5)
        return status, await ns.store.get_run("cs_test"), await ns.store.list_experiments("cs_test")
    status, r, exps = run(go())
    assert status == "cancelled" and r["state"]["report"]["reason"] == "stopped by the owner"
    assert "crw_1" in ns.lane.stopped and exps[0]["status"] == "failed"


def test_supervisor_adopts_and_drives(harness):
    ns = harness
    ns.scores = {(BASE, "dev"): [0.6], (BASE, "holdout"): [0.6]}

    async def go():
        await ns.new_run()
        sup = loop.Supervisor(ns.store)
        assert await sup.tick() == 1
        await asyncio.wait_for(asyncio.gather(*sup.tasks.values()), 5)
        r = await ns.store.get_run("cs_test")
        return r
    r = run(go())
    assert r["status"] == "stalled" and r["lease_pid"] == 0


# ── routes ───────────────────────────────────────────────────────────


@pytest.fixture()
def client(harness, monkeypatch, repo, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from captain_claw.flight_deck import captain_swarm_routes as routes
    from captain_claw.flight_deck.auth import get_current_user

    app = FastAPI()
    app.include_router(routes.router)
    user = {"id": "u", "display_name": "Ana", "role": "admin"}
    app.dependency_overrides[get_current_user] = lambda: user

    async def can_run(owner, objective=None, user=None):
        return None
    monkeypatch.setattr(routes, "_can_run", can_run)
    monkeypatch.setattr(routes, "_source_repo", lambda: str(repo))
    monkeypatch.setattr(routes, "_default_holdout", lambda: _holdout(tmp_path))
    monkeypatch.setattr("captain_claw.flight_deck.bat_routes._owner_has_shell", lambda u: True)
    kicked = []

    async def kick():
        kicked.append(1)
        return 0
    monkeypatch.setattr(loop, "kick", kick)
    with TestClient(app) as c:
        yield c, routes, kicked


def test_routes_start_list_get_cancel(client):
    c, routes, kicked = client
    r = c.post("/fd/captain-swarm/runs", json={"template": "mrav", "options": {"model": "ornith:9b", "budget_usd": 5}})
    assert r.status_code == 200, r.text
    rid = r.json()["id"]
    assert r.json()["status"] == "baseline" and kicked
    assert [x["id"] for x in c.get("/fd/captain-swarm/runs").json()["runs"]] == [rid]
    got = c.get(f"/fd/captain-swarm/runs/{rid}").json()
    assert got["objective"]["budget"]["usd"] == 5 and got["objective"]["holdout"] == {"file": "(sealed)"}
    assert "ornith:9b" in got["objective"]["metric"]["run"]
    assert c.post(f"/fd/captain-swarm/runs/{rid}/pr").status_code == 409
    assert c.post(f"/fd/captain-swarm/runs/{rid}/cancel").json()["status"] == "cancelled"
    assert c.post(f"/fd/captain-swarm/runs/{rid}/cancel").status_code == 409
    assert c.post(f"/fd/captain-swarm/runs/{rid}/pr").status_code == 409     # nothing beat the start
    assert c.get("/fd/captain-swarm/runs/cs_nope").status_code == 404


def test_routes_refuse_a_bad_objective(client):
    c, _, _ = client
    r = c.post("/fd/captain-swarm/runs", json={"template": "mrav", "options": {"harness": "vim"}})
    assert r.status_code == 400 and "coding worker" in r.json()["detail"]
    assert c.post("/fd/captain-swarm/runs", json={"template": "nope"}).status_code == 400


def test_agent_start_waits_for_the_owners_go(client, monkeypatch):
    c, routes, kicked = client
    bells = []

    class DB:
        async def add_notification(self, *a):
            bells.append(a)
    monkeypatch.setattr("captain_claw.flight_deck.auth.get_db", lambda: DB())
    monkeypatch.setattr(routes, "_agent_owner", lambda body: "u")

    async def user_of(uid):
        return {"id": uid, "role": "admin"}
    monkeypatch.setattr("captain_claw.flight_deck.crew_runner._user_of", user_of)
    got = c.post("/fd/captain-swarm/agent/start", json={"template": "mrav", "agent_name": "Helper"}).json()
    assert got["status"] == "awaiting_go" and bells and bells[0][1] == "captain_swarm"
    rid = got["run_id"]
    assert not kicked
    assert c.post("/fd/captain-swarm/agent/status", json={"run_id": rid}).json()["status"] == "awaiting_go"
    r = c.post(f"/fd/captain-swarm/runs/{rid}/go")
    assert r.status_code == 200 and r.json()["status"] == "baseline" and kicked
    assert c.post(f"/fd/captain-swarm/runs/{rid}/go").status_code == 409
    got2 = c.post("/fd/captain-swarm/agent/start", json={"template": "mrav"}).json()
    assert c.post(f"/fd/captain-swarm/runs/{got2['run_id']}/decline").json()["status"] == "declined"
    bad = c.post("/fd/captain-swarm/agent/start", json={"template": "mrav", "options": {"harness": "vim"}}).json()
    assert bad["status"] == "rejected"


# ── Crew starts a worktree where it's told ───────────────────────────


def test_crew_worktree_starts_at_a_given_commit(repo, tmp_path, monkeypatch):
    from captain_claw.flight_deck import crew_runner, crew_store
    from captain_claw.flight_deck.crew_access import Grant

    base = git(repo, "rev-parse", "HEAD")
    later = commit(repo, "captain_claw/mrav/prompts.py", "PROMPT = 'later'\n")
    monkeypatch.setattr(crew_store, "user_dir", lambda uid: tmp_path / "crew" / uid)
    (tmp_path / "crew" / "u").mkdir(parents=True)
    monkeypatch.setattr(crew_runner, "_repo_env", lambda user, grant, token="": None)
    grant = Grant(True, "local", ("claude",), 2, False)

    async def go():
        tree = await crew_runner._make_worktree({"id": "u"}, grant, {"repo": str(repo)}, "cs-1", start_at=base)
        assert tree["base_sha"] == base and tree["base_branch"] == "main"
        assert (Path(tree["worktree"]) / "captain_claw" / "mrav" / "prompts.py").read_text() == "PROMPT = 'be brief'\n"
        plain = await crew_runner._make_worktree({"id": "u"}, grant, {"repo": str(repo)}, "cs-2")
        assert plain["base_sha"] == later
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            await crew_runner._make_worktree({"id": "u"}, grant, {"repo": str(repo)}, "cs-3", start_at="deadbeef")
        with pytest.raises(HTTPException):
            await crew_runner._make_worktree({"id": "u"}, grant, {"repo": str(repo)}, "cs-4", start_at="main; rm -rf /")
    run(go())
