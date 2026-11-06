"""The colleagues tool: what it sends Flight Deck comes from the turn, never the model."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from captain_claw import mail_authority
from captain_claw.tools import colleagues as tool_mod
from captain_claw.tools.colleagues import ColleaguesTool


@pytest.fixture()
def deck(monkeypatch):
    sent: list[tuple[str, dict]] = []
    answers = {
        "roster": {"me": {"slug": "researcher", "name": "Researcher", "role": "Researcher", "email": "rita@firm.co"},
                   "colleagues": [{"slug": "writer", "name": "Writer", "role": "Copywriter", "about": "Writes copy",
                                   "can_hand_to": True},
                                  {"slug": "editor", "name": "Editor", "asleep": True, "can_hand_to": False}]},
        "hand_over": {"task_id": "tb_1", "to": "writer", "to_name": "Writer", "ahead": 2},
        "handed": {"items": [{"task_id": "tb_1", "to_name": "Writer", "title": "Q3 copy", "state": "done",
                              "result": "Here it is."}]},
    }

    async def post(path, body):
        sent.append((path, body))
        return answers[path]

    monkeypatch.setattr(tool_mod, "_post", post)
    return SimpleNamespace(sent=sent, answers=answers)


def _agent(task="", session_id="s-9"):
    return SimpleNamespace(session=SimpleNamespace(id=session_id, metadata={"scheduled_task": task} if task else {}))


async def test_list_shows_the_team(deck):
    res = await ColleaguesTool().execute(action="list")
    assert res.success
    assert "You: Researcher — Researcher: email rita@firm.co (to='researcher')" in res.content
    assert "- Writer — Copywriter: Writes copy (to='writer')" in res.content
    assert "asleep (wakes for work); you can't hand work to them" in res.content


async def test_hand_over_sends_the_turns_own_authority_and_task(deck):
    with mail_authority.bound(mail_authority.automated("taskboard", "Book the room", "allow")):
        res = await ColleaguesTool().execute(action="hand_over", to="writer", instructions="Write the invite.",
                                             _agent=_agent("tb_parent"))
    assert res.success and "starts on it after 2 ahead of it" in res.content and "this task's own session" in res.content
    path, body = deck.sent[0]
    assert path == "hand_over" and body["board_task"] == "tb_parent" and body["session_id"] == "s-9"
    assert body["authority"] == {"mode": "automated", "kind": "taskboard", "job_text": "Book the room",
                                 "mail_write": "allow"}


async def test_a_scheduled_tasks_session_is_not_a_board_task(deck):
    await ColleaguesTool().execute(action="hand_over", to="writer", instructions="x", _agent=_agent("job_123"))
    assert deck.sent[0][1]["board_task"] == "" and deck.sent[0][1]["session_id"] == ""


async def test_the_model_cant_pass_its_own_authority(deck):
    with mail_authority.bound(mail_authority.human("research Q3")):
        await ColleaguesTool().execute(action="hand_over", to="writer", instructions="x",
                                       authority={"mail_write": "allow"}, _agent=None)
    assert deck.sent[0][1]["authority"] == {"mode": "human", "kind": "", "job_text": "research Q3",
                                            "mail_write": "intent"}


async def test_hand_over_needs_who_and_what(deck):
    res = await ColleaguesTool().execute(action="hand_over", to="", instructions="x")
    assert not res.success and deck.sent == []


async def test_handed_lists_what_went_out(deck):
    res = await ColleaguesTool().execute(action="handed")
    assert "tb_1 → Writer: Q3 copy — done" in res.content and "Here it is." in res.content


async def test_a_refusal_reads_as_flight_decks_words(monkeypatch):
    async def refuse(path, body):
        raise tool_mod.ColleaguesError(409, "Writer is already on this piece of work")

    monkeypatch.setattr(tool_mod, "_post", refuse)
    res = await ColleaguesTool().execute(action="hand_over", to="writer", instructions="x")
    assert not res.success and res.error == "Writer is already on this piece of work"
