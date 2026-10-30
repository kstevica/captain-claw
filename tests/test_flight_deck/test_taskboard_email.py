"""The whole email behind a task: read from the user's own Gmail when they open
the task bigger, and kept nowhere."""

from __future__ import annotations

import base64
import json

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import auth, gmail_autoreply, google_oauth_routes, taskboard_email

from .test_taskboard_nextsteps import ALICE, _task, board  # noqa: F401

LONG = "Hi Stevica,\n\n" + "I am following up on the AI Hackathon in December. " * 40 + "\n\nTHE VERY LAST LINE"


def _gmail(mid, *, body=LONG, frm="Paul Suchar <paul.suchar@bbcc.hr>", to="me@x.co", at=1000, labels=("INBOX",)):
    data = base64.urlsafe_b64encode(body.encode()).decode()
    return {"id": mid, "threadId": "t-1", "labelIds": list(labels), "internalDate": str(at),
            "payload": {"mimeType": "text/plain", "body": {"data": data},
                        "headers": [{"name": "From", "value": frm}, {"name": "To", "value": to},
                                    {"name": "Subject", "value": "BBCC - FRC AI Hackathon"}]}}


@pytest.fixture()
def gmail(monkeypatch):
    """Gmail: what each thread holds, and who asked for what."""
    state = {"threads": {"t-1": [_gmail("m-1")]}, "asked": [], "token": "tok"}

    async def thread(client, token, thread_id):
        state["asked"].append((token, thread_id))
        messages = state["threads"].get(thread_id)
        return {"messages": messages} if messages is not None else None

    async def token(uid):
        return state["token"]

    monkeypatch.setattr(gmail_autoreply, "_thread", thread)
    monkeypatch.setattr(google_oauth_routes, "get_valid_google_access_token", token)
    return state


def _with_thread(board, thread="t-1"):  # noqa: F811
    t = _task(board, flags={}, pending=False)
    board.add_link(ALICE, f"gmail:thread:{thread}", t)
    return t


async def test_it_gives_the_whole_email_not_what_the_board_keeps(board, gmail):  # noqa: F811
    got = await taskboard_email.whole_email(ALICE, _with_thread(board))
    message = got["messages"][0]
    assert message["text"].endswith("THE VERY LAST LINE") and len(message["text"]) > 500
    assert "\n\n" in message["text"], "its line breaks are still there"
    assert message["from_name"] == "Paul Suchar" and message["from_address"] == "paul.suchar@bbcc.hr"
    assert message["to"] == ["me@x.co"] and got["reason"] == ""


async def test_it_is_kept_nowhere(board, gmail):  # noqa: F811
    await taskboard_email.whole_email(ALICE, _with_thread(board))
    dump = "\n".join(str(tuple(r)) for table in ("tasks", "task_events", "task_decisions")
                     for r in board._conn.execute(f"SELECT * FROM {table}").fetchall())
    assert "THE VERY LAST LINE" not in dump


async def test_a_thread_reads_in_order_without_what_each_quotes(board, gmail):  # noqa: F811
    gmail["threads"]["t-1"] = [
        _gmail("m-2", body="Thanks!\n\nOn Mon, 13 Oct 2026 at 10:15, Paul wrote:\n> the first one", at=2000),
        _gmail("m-1", body="the first one", at=1000),
    ]
    got = await taskboard_email.whole_email(ALICE, _with_thread(board))
    assert [m["id"] for m in got["messages"]] == ["m-1", "m-2"]
    assert got["messages"][1]["text"] == "Thanks!"


async def test_an_email_that_only_quotes_is_still_shown(board, gmail):  # noqa: F811
    gmail["threads"]["t-1"] = [_gmail("m-1", body="> only this")]
    got = await taskboard_email.whole_email(ALICE, _with_thread(board))
    assert got["messages"][0]["text"] == "> only this"


async def test_a_draft_in_the_thread_isnt_shown(board, gmail):  # noqa: F811
    gmail["threads"]["t-1"] = [_gmail("m-1"), _gmail("d-1", body="my unsent words", at=2000, labels=("DRAFT",))]
    got = await taskboard_email.whole_email(ALICE, _with_thread(board))
    assert [m["id"] for m in got["messages"]] == ["m-1"]


async def test_a_long_thread_keeps_its_latest(board, gmail):  # noqa: F811
    total = taskboard_email.MAX_MESSAGES + 5
    gmail["threads"]["t-1"] = [_gmail(f"m-{i}", body=f"number {i}", at=1000 + i) for i in range(total)]
    got = await taskboard_email.whole_email(ALICE, _with_thread(board))
    assert len(got["messages"]) == taskboard_email.MAX_MESSAGES and got["more"] == 5
    assert got["messages"][-1]["text"] == f"number {total - 1}"


async def test_no_google_says_so_and_asks_gmail_nothing(board, gmail):  # noqa: F811
    gmail["token"] = None
    got = await taskboard_email.whole_email(ALICE, _with_thread(board))
    assert got["messages"] == [] and "Connect Google" in got["reason"] and gmail["asked"] == []


async def test_an_email_gmail_wont_give_says_so(board, gmail):  # noqa: F811
    got = await taskboard_email.whole_email(ALICE, _with_thread(board, "gone"))
    assert got["messages"] == [] and "deleted" in got["reason"]


async def test_a_task_with_no_thread_has_nothing_to_say(board, gmail):  # noqa: F811
    got = await taskboard_email.whole_email(ALICE, _task(board, flags={}, pending=False))
    assert got == {"messages": [], "more": 0, "reason": ""} and gmail["asked"] == []


async def test_a_key_that_isnt_a_thread_id_never_reaches_gmail(board, gmail):  # noqa: F811
    t = _task(board, flags={}, pending=False)
    board.add_link(ALICE, "gmail:thread:../../profile", t)
    got = await taskboard_email.whole_email(ALICE, t)
    assert gmail["asked"] == [] and got["messages"] == []


async def test_only_the_users_own_token_and_task(board, gmail, monkeypatch):  # noqa: F811
    t = _with_thread(board)
    with pytest.raises(LookupError):
        await taskboard_email.whole_email("someone-else", t)
    assert gmail["asked"] == []
    asked_for = []

    async def token(uid):
        asked_for.append(uid)
        return "alices"

    monkeypatch.setattr(google_oauth_routes, "get_valid_google_access_token", token)
    await taskboard_email.whole_email(ALICE, t)
    assert asked_for == [ALICE] and gmail["asked"] == [("alices", "t-1")]


async def test_the_route_is_the_callers_own(board, gmail):  # noqa: F811
    t = _with_thread(board)
    app = FastAPI()
    app.include_router(taskboard_email.router)
    current = {"id": ALICE, "email": "a@x.co", "display_name": "A", "role": "user"}
    app.dependency_overrides[auth.get_current_user] = lambda: current
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://deck") as client:
        ok = await client.get(f"/fd/taskboard/tasks/{t}/email")
        assert ok.status_code == 200 and ok.json()["messages"][0]["id"] == "m-1"
        current = {"id": "someone-else", "email": "b@x.co", "display_name": "B", "role": "admin"}
        app.dependency_overrides[auth.get_current_user] = lambda: current
        assert (await client.get(f"/fd/taskboard/tasks/{t}/email")).status_code == 404
    assert json.loads(ok.text)["reason"] == ""
