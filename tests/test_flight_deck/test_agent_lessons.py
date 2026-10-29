"""An agent's lessons as Flight Deck keeps and writes them (flight_deck/agent_lessons.py).

* a setup record's rules, when-to-ask and its taught lessons are the agent's
  lessons, each once, with where it came from and who taught it; asking
  first is always in mind unless it says otherwise;
* they are written where the agent reads them, again when the record is
  newer — every agent's at the deck's start;
* a lesson taught again is not added twice.
"""

from __future__ import annotations

import json
import os
import time

from captain_claw.flight_deck import agent_lessons as al


def _record(**kw) -> dict:
    return {"rules": [{"text": "Answer in Croatian.", "always": True, "source": {"doc": "policy.pdf", "section": "Tone"}},
                      {"text": "Refunds take 14 days.", "source": {"doc": "policy.pdf", "section": "Refunds"}}],
            "escalation": [{"text": "Ask before refunds over 200 EUR.", "source": {"doc": "instructions"}}],
            "lessons": [{"kind": "fact", "text": "Support is help@acme.test", "taught_by": "setup"}], **kw}


def test_a_record_is_its_lessons():
    got = al.from_record({**_record(), "rules": [*_record()["rules"], {"text": "answer in  croatian."}]})
    assert [(x["kind"], x["text"], x["always"], x["taught_by"]) for x in got] == [
        ("rule", "Answer in Croatian.", True, "setup"), ("rule", "Refunds take 14 days.", False, "setup"),
        ("ask_first", "Ask before refunds over 200 EUR.", True, "setup"),
        ("fact", "Support is help@acme.test", False, "setup")]           # the same rule twice: once
    assert got[0]["source"] == {"doc": "policy.pdf", "section": "Tone"} and got[2]["source"] == {"doc": "", "section": ""}
    assert len({x["id"] for x in got}) == 4 and got[0]["id"] == al.lesson_id("rule", "answer in   Croatian.")
    assert al.counts(_record()) == {"rule": 2, "ask_first": 1, "fact": 1, "always": 2}


def test_they_are_written_for_the_agent_and_again_when_the_record_changes(tmp_path):
    agent = tmp_path / "helpdesk"
    agent.mkdir()
    assert al.ensure(agent) is False                                    # no setup: nothing to write
    (agent / ".owner-setup.json").write_text(json.dumps(_record()))
    assert al.ensure(agent) is True and al.ensure(agent) is False       # written once
    data = json.loads(al.path(agent).read_text())
    assert data["version"] == 1 and len(data["lessons"]) == 4
    assert al.path(agent).parent.name == ".captain-claw"                 # its HOME's, beside its skills
    earlier = time.time() - 60
    os.utime(al.path(agent), (earlier, earlier))                       # the record changed since they were written
    (agent / ".owner-setup.json").write_text(json.dumps(_record(rules=[])))
    assert al.ensure(agent) is True and len(json.loads(al.path(agent).read_text())["lessons"]) == 2
    other = tmp_path / "other"
    other.mkdir()
    (other / ".owner-setup.json").write_text(json.dumps(_record()))
    assert al.ensure_all(tmp_path) == 1                                  # only the one not written yet


def test_a_lesson_taught_again_is_not_added_twice():
    record, added = al.taught(_record(), "fact", "The refund window is 14 days.",
                              {"doc": "handbook.md", "section": "Refunds"}, taught_by="exam")
    assert added and record["lessons"][-1]["taught_by"] == "exam" and record["lessons"][-1]["added"] > 0
    again, added = al.taught(record, "fact", "the refund window is  14 days.", taught_by="exam")
    assert not added and again is record
    _, added = al.taught(_record(), "rule", "Refunds take 14 days.", taught_by="exam")
    assert not added                                                    # the record's own rule
    assert al.facts_as_lessons([{"content": "x", "importance": 9}, {"content": " "}, "junk"])[0]["importance"] == 9
