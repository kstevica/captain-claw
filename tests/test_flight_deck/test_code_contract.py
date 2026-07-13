"""Tests for A1 — the acceptance contract for Code's build/fix loop.

Parsing/persistence/rendering are pure; validation uses the real filesystem
(tmp_path) with an injected command runner, so no subprocess or model is needed.
"""

from __future__ import annotations

import json

from captain_claw.flight_deck import code_contract


# ── parsing ──────────────────────────────────────────────────────────

def test_parse_empty_and_garbage():
    assert code_contract.parse_contract("") == []
    assert code_contract.parse_contract("not json at all") == []
    assert code_contract.parse_contract('{"constraints": []}') == []


def test_parse_each_check_type():
    raw = json.dumps({"constraints": [
        {"id": "a1", "text": "build passes", "severity": "critical",
         "check": {"type": "command", "cmd": "npm run build"}},
        {"id": "a2", "text": "entry exists", "severity": "major",
         "check": {"type": "file_exists", "path": "src/index.ts"}},
        {"id": "a3", "text": "exports foo", "severity": "major",
         "check": {"type": "file_contains", "path": "src/x.py", "pattern": "def foo"}},
        {"id": "a4", "text": "no TODO left", "severity": "minor",
         "check": {"type": "no_pattern", "pattern": "TODO", "glob": "**/*.py"}},
        {"id": "a5", "text": "reads nicely", "severity": "major",
         "check": {"type": "judge"}},
    ]})
    out = code_contract.parse_contract(raw)
    assert [c["check"]["type"] for c in out] == \
        ["command", "file_exists", "file_contains", "no_pattern", "judge"]
    assert out[0]["severity"] == "critical"


def test_parse_from_fenced_block():
    text = "Here you go:\n```json\n" + json.dumps(
        {"constraints": [{"id": "a1", "text": "x", "severity": "critical",
                          "check": {"type": "file_exists", "path": "a.py"}}]}) + "\n```"
    out = code_contract.parse_contract(text)
    assert len(out) == 1 and out[0]["check"]["path"] == "a.py"


def test_malformed_check_falls_back_to_judge():
    raw = json.dumps({"constraints": [
        {"id": "a1", "text": "cmd missing", "severity": "critical",
         "check": {"type": "command"}},                       # no cmd
        {"id": "a2", "text": "bad severity", "severity": "wat",
         "check": {"type": "file_exists"}},                   # no path
    ]})
    out = code_contract.parse_contract(raw)
    assert all(c["check"]["type"] == "judge" for c in out)
    assert out[1]["severity"] == "major"   # unknown severity normalized


def test_pathless_and_textless_dropped():
    raw = json.dumps({"constraints": [
        {"id": "a1", "text": "", "check": {"type": "judge"}},  # no text → dropped
    ]})
    assert code_contract.parse_contract(raw) == []


def test_relpath_rejects_escapes():
    raw = json.dumps({"constraints": [
        {"id": "a1", "text": "abs path", "severity": "major",
         "check": {"type": "file_exists", "path": "/etc/passwd"}},
        {"id": "a2", "text": "dotdot", "severity": "major",
         "check": {"type": "file_exists", "path": "../../secret"}},
    ]})
    out = code_contract.parse_contract(raw)
    # Both unsafe paths make the check fall back to judge (path rejected).
    assert all(c["check"]["type"] == "judge" for c in out)


# ── persistence ──────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    cons = code_contract.parse_contract(json.dumps({"constraints": [
        {"id": "a1", "text": "x", "severity": "critical",
         "check": {"type": "file_exists", "path": "a.py"}}]}))
    code_contract.save(tmp_path, cons, "the task")
    assert (tmp_path / code_contract.CONTRACT_FILE).is_file()
    loaded = code_contract.load(tmp_path)
    assert loaded == cons


def test_load_absent_is_none(tmp_path):
    assert code_contract.load(tmp_path) is None


def test_load_renormalizes_hand_edit(tmp_path):
    # A hand-edited file with a malformed check must not smuggle it into the loader.
    (tmp_path / code_contract.CONTRACT_FILE).write_text(json.dumps({"constraints": [
        {"id": "a1", "text": "x", "severity": "critical",
         "check": {"type": "command"}}]}))  # cmd missing
    loaded = code_contract.load(tmp_path)
    assert loaded[0]["check"]["type"] == "judge"


# ── directive ────────────────────────────────────────────────────────

def test_directive_empty_when_no_constraints():
    assert code_contract.contract_directive([]) == ""


def test_directive_lists_rules():
    cons = code_contract.parse_contract(json.dumps({"constraints": [
        {"id": "a1", "text": "build passes", "severity": "critical",
         "check": {"type": "command", "cmd": "make"}}]}))
    d = code_contract.contract_directive(cons)
    assert "ACCEPTANCE CONTRACT" in d and "build passes" in d


# ── deterministic validation ─────────────────────────────────────────

async def test_validate_file_exists(tmp_path):
    (tmp_path / "there.py").write_text("x = 1\n")
    cons = code_contract.parse_contract(json.dumps({"constraints": [
        {"id": "a1", "text": "there", "severity": "critical",
         "check": {"type": "file_exists", "path": "there.py"}},
        {"id": "a2", "text": "missing", "severity": "critical",
         "check": {"type": "file_exists", "path": "gone.py"}}]}))
    res = await code_contract.validate(tmp_path, cons)
    assert [f["id"] for f in res["passed"]] == ["a1"]
    assert [f["id"] for f in res["failed"]] == ["a2"]


async def test_validate_file_contains(tmp_path):
    (tmp_path / "x.py").write_text("def foo():\n    return 1\n")
    cons = code_contract.parse_contract(json.dumps({"constraints": [
        {"id": "a1", "text": "has foo", "severity": "major",
         "check": {"type": "file_contains", "path": "x.py", "pattern": r"def foo"}},
        {"id": "a2", "text": "has bar", "severity": "major",
         "check": {"type": "file_contains", "path": "x.py", "pattern": r"def bar"}}]}))
    res = await code_contract.validate(tmp_path, cons)
    assert {f["id"] for f in res["passed"]} == {"a1"}
    assert {f["id"] for f in res["failed"]} == {"a2"}


async def test_validate_no_pattern(tmp_path):
    (tmp_path / "a.py").write_text("# TODO: fix me\nx = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    cons = code_contract.parse_contract(json.dumps({"constraints": [
        {"id": "a1", "text": "no todo", "severity": "major",
         "check": {"type": "no_pattern", "pattern": "TODO", "glob": "**/*.py"}}]}))
    res = await code_contract.validate(tmp_path, cons)
    assert res["failed"][0]["id"] == "a1"
    assert "a.py" in res["failed"][0]["note"]


async def test_validate_no_pattern_clean_passes(tmp_path):
    (tmp_path / "b.py").write_text("y = 2\n")
    cons = code_contract.parse_contract(json.dumps({"constraints": [
        {"id": "a1", "text": "no todo", "severity": "major",
         "check": {"type": "no_pattern", "pattern": "TODO"}}]}))
    res = await code_contract.validate(tmp_path, cons)
    assert res["passed"][0]["id"] == "a1"


async def test_validate_command_uses_injected_runner(tmp_path):
    calls = []

    async def runner(cmd, cwd):
        calls.append((cmd, cwd))
        return (cmd == "make ok"), "output here"

    cons = code_contract.parse_contract(json.dumps({"constraints": [
        {"id": "a1", "text": "ok", "severity": "critical",
         "check": {"type": "command", "cmd": "make ok"}},
        {"id": "a2", "text": "bad", "severity": "critical",
         "check": {"type": "command", "cmd": "make bad"}}]}))
    res = await code_contract.validate(tmp_path, cons, runner=runner)
    assert {f["id"] for f in res["passed"]} == {"a1"}
    assert res["failed"][0]["id"] == "a2"
    assert "make bad" in res["failed"][0]["note"]
    assert len(calls) == 2


async def test_validate_judge_type_is_unresolved(tmp_path):
    cons = code_contract.parse_contract(json.dumps({"constraints": [
        {"id": "a1", "text": "subjective", "severity": "major",
         "check": {"type": "judge"}}]}))
    res = await code_contract.validate(tmp_path, cons)
    assert res["unresolved"][0]["id"] == "a1"
    assert not res["passed"] and not res["failed"]


async def test_validate_unparseable_pattern_skipped_not_failed(tmp_path):
    (tmp_path / "x.py").write_text("hi\n")
    cons = code_contract.parse_contract(json.dumps({"constraints": [
        {"id": "a1", "text": "bad regex", "severity": "major",
         "check": {"type": "file_contains", "path": "x.py", "pattern": "("}}]}))
    res = await code_contract.validate(tmp_path, cons)
    # A broken pattern must not manufacture a failure — it passes (skipped).
    assert res["passed"] and not res["failed"]


# ── judge fold ───────────────────────────────────────────────────────

def test_parse_and_apply_judgement():
    result = {"passed": [], "failed": [],
              "unresolved": [{"id": "a1", "text": "t1", "severity": "major"},
                             {"id": "a2", "text": "t2", "severity": "critical"},
                             {"id": "a3", "text": "t3", "severity": "major"}]}
    judgements = code_contract.parse_judgement(json.dumps([
        {"id": "a1", "verdict": "pass"},
        {"id": "a2", "verdict": "fail", "note": "nope"},
        {"id": "a3", "verdict": "unclear"}]))
    out = code_contract.apply_judgement(result, judgements)
    assert [f["id"] for f in out["passed"]] == ["a1"]
    assert [f["id"] for f in out["failed"]] == ["a2"]
    assert [f["id"] for f in out["unresolved"]] == ["a3"]   # unclear stays unresolved


def test_apply_judgement_unmentioned_stays_unresolved():
    result = {"passed": [], "failed": [],
              "unresolved": [{"id": "a1", "text": "t1", "severity": "major"}]}
    out = code_contract.apply_judgement(result, [])   # judge said nothing
    assert out["unresolved"][0]["id"] == "a1"


# ── triage bridge + summary ──────────────────────────────────────────

def test_as_review_entry_none_when_clean():
    assert code_contract.as_review_entry({"passed": [], "failed": [], "unresolved": []}) is None


def test_as_review_entry_only_on_critical_or_major():
    # A minor-only failure shouldn't manufacture a blocking triage entry.
    minor = {"failed": [{"id": "a1", "text": "x", "severity": "minor"}]}
    assert code_contract.as_review_entry(minor) is None
    major = {"failed": [{"id": "a2", "text": "big", "severity": "critical", "note": "why"}]}
    entry = code_contract.as_review_entry(major)
    assert entry["id"] == "acceptance-contract"
    assert "GROUND TRUTH" in entry["output"] and "big" in entry["output"]


def test_summarize_counts():
    result = {
        "passed": [{"id": "a1", "text": "x", "severity": "major"}],
        "failed": [{"id": "a2", "text": "y", "severity": "critical", "how": "deterministic:command"},
                   {"id": "a3", "text": "z", "severity": "major", "how": "judged"}],
        "unresolved": [{"id": "a4", "text": "w", "severity": "major"}],
    }
    s = code_contract.summarize(result)
    assert s["checked"] == 4 and s["passed"] == 1
    assert s["failed_critical"] == 1 and s["failed_major"] == 1
    assert s["unresolved"] == 1
    assert len(s["failed"]) == 2
