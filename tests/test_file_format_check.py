"""The format check (file_format_check, agent_completion_mixin): data files a
turn changed on disk are parsed by their format, and a file that fails sends
the turn back once with the place and the reason."""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from captain_claw import drive
from captain_claw import file_format_check as ffc

_spec = importlib.util.spec_from_file_location(
    "_delivery_check_tests_format", Path(__file__).with_name("test_drive_delivery_check.py"))
_dc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dc)
_quiet = _dc._quiet          # the same config/drive isolation (autouse fixture)

PASSES = json.dumps({"complete": True, "gaps": []})
WORK = {"role": "tool", "tool_name": "shell", "tool_arguments": {"command": "python make.py"}, "content": "ok"}


def _ics(*events: str) -> str:
    return "\r\n".join(["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//t//t//EN", *events, "END:VCALENDAR"]) + "\r\n"


def _event(start: str, end: str) -> str:
    return "\r\n".join(["BEGIN:VEVENT", "UID:a@example.com", "DTSTAMP:20260317T090000Z",
                        start, end, "SUMMARY:Sync", "END:VEVENT"])


# ── iCalendar ────────────────────────────────────────────────────────

@pytest.mark.parametrize("start,end", [
    ("DTSTART:20260414T093000Z", "DTEND:20260414T100000Z"),                       # UTC
    ("DTSTART;TZID=Asia/Shanghai:20260414T173000", "DTEND;TZID=Asia/Shanghai:20260414T180000"),
    ("DTSTART:20260414T173000", "DTEND:20260414T180000"),                         # floating local time
    ("DTSTART;VALUE=DATE:20260414", "DTEND;VALUE=DATE:20260415"),                 # all-day
])
def test_calendar_times_the_standard_allows(tmp_path, start, end):
    path = tmp_path / "plan.ics"
    path.write_text(_ics(_event(start, end)))
    assert ffc.problems(path) == []


def test_an_offset_in_a_calendar_time_is_found_with_its_line(tmp_path):
    path = tmp_path / "plan.ics"
    path.write_text(_ics(_event("DTSTART:20260414T173000+08:00", "DTEND:20260414T180000+08:00")))
    found = ffc.problems(path)
    assert len(found) == 2
    assert found[0].startswith("line 7: DTSTART value '20260414T173000+08:00' is not an iCalendar date-time")
    assert "+08:00 is not allowed" in found[0]


def test_other_calendar_mistakes(tmp_path):
    path = tmp_path / "plan.ics"
    path.write_text(_ics(_event("DTSTART;TZID=Europe/Zagreb:20260414T093000Z", "DTEND:20260414T100000Z"))
                    .replace("END:VEVENT", "END:VTODO"))
    found = ffc.problems(path)
    assert any("TZID and a UTC 'Z'" in p for p in found)
    assert any("END:VTODO does not close BEGIN:VEVENT" in p for p in found)
    path.write_text("BEGIN:VEVENT\r\nDTSTART:20260414T093000Z\r\n")
    assert ffc.problems(path)[:2] == ["it does not start with BEGIN:VCALENDAR", "it does not end with END:VCALENDAR"]


def test_folded_calendar_lines_are_joined(tmp_path):
    path = tmp_path / "plan.ics"
    path.write_text(_ics(_event("DTSTART:20260414T09\r\n 3000Z", "DTEND:20260414T100000Z")))
    assert ffc.problems(path) == []


# ── Other formats ────────────────────────────────────────────────────

@pytest.mark.parametrize("name,text,expected", [
    ("a.json", '{"a": 1}', None),
    ("a.json", '{"a": 1,}', "not valid JSON at line 1"),
    ("a.jsonl", '{"a": 1}\n\n{"b": 2}\n', None),
    ("a.jsonl", '{"a": 1}\n{"b": \n', "line 2 is not JSON"),
    ("a.xml", "<r><a/></r>", None),
    ("a.xml", "<r><a></r>", "not well-formed XML"),
    ("a.yaml", "a: 1\nb: [1, 2]\n", None),
    ("a.yml", "a: [1, 2\n", "not valid YAML"),
    ("a.toml", 'a = 1\n[b]\nc = "x"\n', None),
    ("a.toml", "a = \n", "not valid TOML"),
])
def test_data_formats(tmp_path, name, text, expected):
    path = tmp_path / name
    path.write_text(text)
    found = ffc.problems(path)
    if expected is None:
        assert found == []
    else:
        assert len(found) == 1 and found[0].startswith(expected)


# ── Which files ──────────────────────────────────────────────────────

def test_only_files_changed_since_the_turn_began(tmp_path):
    old = tmp_path / "input.json"
    old.write_text("{broken")
    past = time.time() - 600
    os.utime(old, (past, past))
    since = time.time() - 1
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "out.json").write_text("{broken")
    (tmp_path / "results" / "notes.txt").write_text("{not checked}")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.json").write_text("{broken")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "y.json").write_text("{broken")
    (tmp_path / "logs").mkdir()                                   # Captain's own LLM logs
    (tmp_path / "logs" / "llm.jsonl").write_text('{"a": \n')
    (tmp_path / "saved" / "tmp" / "s1").mkdir(parents=True)       # the write tool's scratch space
    (tmp_path / "saved" / "tmp" / "s1" / "draft.json").write_text("{broken")
    (tmp_path / "saved" / "report").mkdir(parents=True)
    (tmp_path / "saved" / "report" / "logs").mkdir()              # a deeper logs/ is the agent's
    (tmp_path / "saved" / "report" / "logs" / "a.json").write_text("{}")
    assert set(ffc.changed_files(tmp_path, since)) == {
        tmp_path / "results" / "out.json", tmp_path / "saved" / "report" / "logs" / "a.json"}
    report = ffc.check(tmp_path, since)
    assert report == ["results/out.json: not valid JSON at line 1, column 2 (Expecting property name enclosed in double quotes)"]
    assert ffc.check(None, since) == [] and ffc.check(tmp_path, 0) == []


# ── The completion gate ──────────────────────────────────────────────

def _stub(root: Path, since: float):
    stub = _dc._Stub([WORK], PASSES)
    stub.tools = SimpleNamespace(runtime_base_path=root, has_tool=lambda _name: False)
    stub._turn_started_wall = since
    return stub


async def test_a_file_that_does_not_read_sends_the_turn_back_once(tmp_path):
    since = time.time() - 1
    (tmp_path / "scheduled.ics").write_text(_ics(_event("DTSTART:20260414T173000+08:00", "DTEND:20260414T180000Z")))
    stub = _stub(tmp_path, since)
    with drive.turn_drive("ambitious"):
        finalized, text, _ok, feedback, _pw = await _dc._finalize(stub, "Done. scheduled.ics is valid.")
        assert (finalized, text) == (False, "")
        assert "scheduled.ics: line 7: DTSTART value '20260414T173000+08:00'" in feedback
        assert stub.emitted[-1] == {"step": "format_check", "passed": False, "files": 1}
        finalized, text, *_ = await _dc._finalize(stub, "Done.", iteration=5)
    assert finalized is True and text == "Done."


async def test_files_that_read_fine_pass(tmp_path):
    since = time.time() - 1
    (tmp_path / "scheduled.ics").write_text(_ics(_event("DTSTART:20260414T093000Z", "DTEND:20260414T100000Z")))
    (tmp_path / "unscheduled.json").write_text("[]")
    with drive.turn_drive("ambitious"):
        finalized, *_ = await _dc._finalize(_stub(tmp_path, since), "Done.")
    assert finalized is True


@pytest.mark.parametrize("level", ["relaxed", "diligent", None])
async def test_lower_levels_never_check(tmp_path, level):
    since = time.time() - 1
    (tmp_path / "out.json").write_text("{broken")
    with drive.turn_drive(level):
        finalized, *_ = await _dc._finalize(_stub(tmp_path, since), "Done.")
    assert finalized is True
