"""The agent's lessons, brought in with the messages they bear on (lessons.py).

* the lessons a message is about are found by its words — two of them, its
  only word, or one rare one — and by meaning, when the agent has a local
  embedding model, fused by rank;
* none already in its instructions comes again; a turn gets a few, within a
  character budget; the file is read again when it changes, and nothing is
  found without one;
* the turn's note says what each is and where it came from.
"""

from __future__ import annotations

import json
import os
import time
import types

from captain_claw import lessons
from captain_claw.agent_context_mixin import AgentContextMixin


def _write(path, items) -> None:
    path.write_text(json.dumps({"version": 1, "lessons": [
        {"id": f"l{n}", "kind": k, "text": t, "source": s, "always": False} for n, (k, t, s) in enumerate(items)]}))


_ITEMS = [
    ("rule", "Refunds are paid within 14 days of the returned parcel arriving.", {"doc": "policy.pdf", "section": "Refunds"}),
    ("ask_first", "Ask the owner before any refund over 200 EUR.", {"doc": "", "section": ""}),
    ("fact", "The warehouse closes at 15:00 on Fridays.", {"doc": "delivery.md", "section": "12. Warehouse"}),
    ("rule", "Always answer in Croatian unless the customer writes in English.", {"doc": "", "section": ""}),
    ("fact", "Express shipping to Zone B costs 11.40 EUR.", {"doc": "rates.md", "section": "Zone B"}),
]


def test_a_message_brings_the_lessons_its_words_are_about(tmp_path):
    _write(tmp_path / "lessons.json", _ITEMS)
    book = lessons.LessonBook(tmp_path / "lessons.json")
    got = [x["text"] for x in book.relevant("A customer wants a refund for a returned parcel")]
    assert got[0].startswith("Refunds are paid within 14 days")
    assert not any("Warehouse" in t or "warehouse" in t for t in got)
    assert [x["kind"] for x in book.relevant("warehouse")] == ["fact"]          # its only word
    assert book.relevant("hello there") == [] and book.relevant("   ") == []


def test_meaning_finds_what_the_words_miss(tmp_path):
    _write(tmp_path / "lessons.json", _ITEMS)
    book = lessons.LessonBook(tmp_path / "lessons.json")
    topics = {"money": ("refund", "money back", "eur"), "time": ("friday", "weekend", "closes")}

    def embed(texts):
        return [[float(any(w in t.lower() for w in words)) for words in topics.values()] + [0.1] for t in texts]

    got = [x["text"] for x in book.relevant("Can I have my money back?", embed=embed)]
    assert any(t.startswith("Refunds are paid") for t in got)
    assert book.relevant("Can I have my money back?") == []                     # words alone: nothing


def test_none_already_in_its_instructions_comes_again_and_a_turn_gets_a_few(tmp_path):
    _write(tmp_path / "lessons.json", _ITEMS)
    book = lessons.LessonBook(tmp_path / "lessons.json")
    instructions = "## Your owner's rules\n- Refunds are paid within 14 days of the returned   parcel arriving.\n"
    got = [x["text"] for x in book.relevant("refund for a returned parcel over 200 EUR", instructions=instructions)]
    assert got and not any(t.startswith("Refunds are paid") for t in got)
    assert len(book.relevant("refund parcel EUR shipping zone warehouse", limit=2)) <= 2
    one = book.relevant("refund parcel EUR shipping zone", budget=10)
    assert len(one) == 1                                                        # the first always fits


def test_the_file_is_read_again_when_it_changes(tmp_path):
    path = tmp_path / "lessons.json"
    book = lessons.LessonBook(path)
    assert book.relevant("warehouse") == []                                     # no file: nothing
    _write(path, _ITEMS[:2])
    assert book.relevant("warehouse") == []
    _write(path, _ITEMS)
    later = time.time() + 2
    os.utime(path, (later, later))
    assert [x["kind"] for x in book.relevant("warehouse")] == ["fact"]
    path.write_text("{not json")
    os.utime(path, (later + 2, later + 2))
    assert book.relevant("warehouse") == []


def test_the_turns_note_says_what_each_is_and_where_it_came_from():
    note = lessons.note([{"kind": "rule", "text": "Refunds take 14 days.", "source": {"doc": "policy.pdf", "section": "Refunds"}},
                         {"kind": "ask_first", "text": "Ask before refunds.", "source": {"doc": "", "section": ""}},
                         {"kind": "fact", "text": "Zone B is 11.40 EUR.", "source": {"doc": "rates.md", "section": ""}}])
    assert note.startswith("Your owner's lessons that bear on this message")
    assert "- [rule] Refunds take 14 days. (from policy.pdf — Refunds)" in note
    assert "- [ask your owner first] Ask before refunds." in note
    assert "- [fact] Zone B is 11.40 EUR. (from rates.md)" in note
    assert lessons.note([]) == ""


async def test_a_turn_gets_its_lessons_in_its_context(tmp_path, monkeypatch):
    _write(tmp_path / "lessons.json", _ITEMS)
    monkeypatch.setattr(lessons, "_BOOK", lessons.LessonBook(tmp_path / "lessons.json"))
    agent = types.SimpleNamespace(
        session=types.SimpleNamespace(metadata={"fleet_instructions": "- Ask the owner before any refund over 200 EUR."}),
        memory=None, _suppress_memory_context=False)
    await AgentContextMixin._refresh_lessons_context_cache(agent, "refund over 200 EUR for a returned parcel")
    texts = [x["text"] for x in agent._lessons_context_cache]
    assert texts and not any(t.startswith("Ask the owner") for t in texts)     # already in its instructions
    note = AgentContextMixin._build_lessons_context_note(agent)
    assert "[rule] Refunds are paid within 14 days" in note
    agent._suppress_memory_context = True
    assert AgentContextMixin._build_lessons_context_note(agent) == ""
    await AgentContextMixin._refresh_lessons_context_cache(agent, "")
    assert agent._lessons_context_cache == []
