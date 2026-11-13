"""Tasks the person leaves for Claude from the Inbox: anonymized with the
room's vault first (the person sees what Claude reads), queued, picked up by
Claude through room_tasks, answered with task=<n>. Made-up data."""

from __future__ import annotations

import json

import pytest

from captain_claw.flight_deck import rooms_store
from test_flight_deck.test_rooms import (  # noqa: F401 — the deck fixture
    ALICE,
    ALICE_H,
    deck,
)
from test_flight_deck.test_rooms_mcp import _client, _released_room, _rooms_token, _tool


async def _granted_room(c):
    room = await _released_room(c)
    await rooms_store.store().set_grant(ALICE, "cc_oc_test", [room["id"]], False)
    return room


@pytest.mark.usefixtures("deck")
async def test_a_task_is_anonymized_queued_picked_up_and_answered():
    async with _client() as c:
        room = await _granted_room(c)
        t = _rooms_token()
        first = await _tool(c, t, "room_deliver", room=room["id"], title="Sažetak", content="⟦PERSON_1⟧ je CFO.")
        assert first["delivered"] == "#1"

        # The person writes with the real name; Claude would read the token.
        prompt = "Dodaj koliko dionica ima Ana Kovač i usporedi s prošlom godinom."
        preview = (await c.post(f"/fd/rooms/{room['id']}/tasks/preview", headers=ALICE_H,
                                json={"prompt": prompt})).json()
        assert "Ana" not in preview["prompt_tokens"] and "⟦PERSON_1⟧" in preview["prompt_tokens"]
        assert preview["still_real"] == []
        task = (await c.post(f"/fd/rooms/{room['id']}/tasks", headers=ALICE_H,
                             json={"prompt": prompt, "about": 1})).json()
        assert task["seq"] == 1 and task["state"] == "waiting" and task["prompt"] == prompt
        assert "Kovač".encode() not in rooms_store.store().path.read_bytes()  # encrypted at rest
        summary = (await c.get(f"/fd/rooms/{room['id']}", headers=ALICE_H)).json()
        assert summary["tasks"] == {"total": 1, "open": 1}

        # Claude sees there's a task, and reads it in tokens only, with what it's about.
        listing = await _tool(c, t, "room_list")
        assert listing["rooms"][0]["tasks_waiting"] == 1 and "room_tasks" in listing["note"]
        got = await _tool(c, t, "room_tasks")
        item = got["rooms"][0]["tasks"][0]
        assert item["task"] == 1 and "⟦PERSON_1⟧" in item["prompt"] and "Ana" not in json.dumps(got)
        assert item["about"] == {"delivery": 1, "title": "Sažetak", "content": "⟦PERSON_1⟧ je CFO."}
        tasks = (await c.get(f"/fd/rooms/{room['id']}/tasks", headers=ALICE_H)).json()["tasks"]
        assert tasks[0]["state"] == "taken"
        page = await _tool(c, t, "room_get_delivery", room=room["id"], delivery=1)
        assert "⟦PERSON_1⟧ je CFO." in page and "Ana" not in page

        answer = await _tool(c, t, "room_deliver", room=room["id"], title="Dionice ⟦PERSON_1⟧",
                             content="⟦PERSON_1⟧ ima 12% dionica.", task=1)
        assert answer["delivered"] == "#2" and "answered" in answer["task"]
        tasks = (await c.get(f"/fd/rooms/{room['id']}/tasks", headers=ALICE_H)).json()["tasks"]
        assert tasks[0]["state"] == "done" and tasks[0]["delivery"] == 2
        assert (await _tool(c, t, "room_tasks"))["tasks"] == []
        again = await _tool(c, t, "room_deliver", room=room["id"], title="Opet", content="x", task=1)
        assert "already answered" in again["task"]


@pytest.mark.usefixtures("deck")
async def test_cancel_and_bad_input():
    async with _client() as c:
        room = await _granted_room(c)
        url = f"/fd/rooms/{room['id']}/tasks"
        assert (await c.post(url, headers=ALICE_H, json={"prompt": "  "})).status_code == 422
        assert (await c.post(url, headers=ALICE_H, json={"prompt": "x" * 9000})).status_code == 422
        assert (await c.post(url, headers=ALICE_H, json={"prompt": "Sažmi.", "about": 99})).status_code == 422
        task = (await c.post(url, headers=ALICE_H, json={"prompt": "Sažmi ugovor."})).json()
        assert (await c.delete(f"{url}/{task['seq']}", headers=ALICE_H)).json()["cancelled"] is True
        assert (await c.delete(f"{url}/{task['seq']}", headers=ALICE_H)).status_code == 409
        assert (await _tool(c, _rooms_token(), "room_tasks"))["tasks"] == []
