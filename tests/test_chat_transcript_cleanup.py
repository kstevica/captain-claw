"""What the web chat shows of a turn, as real users met it (cohort001).

* Messages the runtime writes into the agent's session as "user" — a guard's
  "[system] STOP — you have already written this file", a relayed result
  ("This is the ANSWER you were waiting for. Relay it…"), a fleet notice —
  reached the chat as the user's own bubbles. Replays and notification frames
  now say where they came from, and the Flight Deck log tags such rows that
  an earlier client stored as the user's.
* A file a script or the docx/xlsx/pdf tools made got no card: the reply only
  named a server path. Every new file the turn made under saved/ now gets a
  card frame after the reply.
"""

from __future__ import annotations

import os
import time
import types
from pathlib import Path

import pytest

from captain_claw import msg_origin
from captain_claw import turn_deliverables as td
from captain_claw.session import Session

WRITE_LOOP = (
    "[system] STOP — you have already written this file. The file is saved and complete. "
    "Do NOT rewrite it again."
)
RELAY = (
    "[Delegated result from Researcher] This is the ANSWER you were waiting for. Relay it "
    "to the user now, concisely, in their language. Do NOT delegate again, do NOT call any "
    "tool, and do NOT say you are still waiting — you already have the result below:\n\n"
    "Three tenders match."
)
FLEET = "[Flight Deck] Agent 'Ana' has joined the fleet."


# ── Where a transcript's user row came from ─────────────────────────────


@pytest.mark.parametrize("text, origin", [
    (WRITE_LOOP, "corrective"),
    ("[system] Scale advisory: 40 items discovered via glob — process them in batches.", "corrective"),
    (RELAY, "delegated_result"),
    ("[Basna 'Market scan' finished successfully] This is the RESULT …", "delegated_result"),
    (FLEET, "fleet_notice"),
    ("[SCHEDULED TASK — cron job 'tenders'] Check EOJN", "cron"),
    # A person's words, even ones that look like a prompt.
    ("STOP doing that, it's wrong", ""),
    ("nisi mi dao finalni proizvod, di je?", ""),
    ("## Project: Q3 report\nWrite it up.", ""),
    ("You are an autonomous research assistant running in the background. Find X.", ""),
])
def test_transcript_origin(text, origin):
    assert msg_origin.transcript_origin(text) == origin


def test_the_log_tags_runtime_rows_stored_as_the_users():
    from captain_claw.flight_deck.chat_routes import _with_origin

    row = {"id": 1, "role": "user", "content": WRITE_LOOP, "metadata": "{}"}
    assert _with_origin(row)["origin"] == "corrective"
    mine = {"id": 2, "role": "user", "content": "where is the file?", "metadata": "{}"}
    assert _with_origin(mine) is mine
    reply = {"id": 3, "role": "assistant", "content": WRITE_LOOP, "metadata": "{}"}
    assert "origin" not in _with_origin(reply)


def test_notification_frames_name_their_origin():
    from captain_claw.web.ws_handler import notification_origin

    assert notification_origin(RELAY) == "delegated_result"
    assert notification_origin(FLEET) == "fleet_notice"
    assert notification_origin("Something happened on the deck.") == "notification"


async def test_topics_skip_runtime_rows(tmp_path):
    from captain_claw.flight_deck.db import FlightDeckDB

    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    user = await db.create_user("a@example.com", "x")
    await db.upsert_chat_session("proc-ana", user["id"], agent_id="ana")
    topic = '{"topic": {"action": "new", "agentSessionId": "s1", "source": "web"}}'
    await db.add_chat_messages("proc-ana", user["id"], [
        {"role": "system", "content": "New topic", "metadata": topic},
        {"role": "user", "content": WRITE_LOOP},
        {"role": "user", "content": "make the tender report"},
        {"role": "assistant", "content": "Done."},
    ])
    try:
        [t] = await db.list_chat_topics("proc-ana", user["id"])
    finally:
        await db.close()
    assert t["first_user_message"] == "make the tender report"
    assert t["message_count"] == 2


# ── New files of a turn ─────────────────────────────────────────────────


@pytest.fixture
def world(tmp_path):
    saved = tmp_path / "workspace" / "saved"
    for sub in ("showcase/s1", "showcase/s2", "output", "media/s1", "tmp/s1", "scripts/s1",
                "downloads/s1", "video_frames/x", "showcase/s1/.cache"):
        (saved / sub).mkdir(parents=True, exist_ok=True)
    session = Session(id="s1", name="t")
    session.add_message("user", "make the report", origin="human", channel="web")
    started = time.time() - 1
    agent = types.SimpleNamespace(
        session=session,
        tools=types.SimpleNamespace(get_saved_base_path=lambda create=False: saved),
        _current_session_slug=lambda: "s1",
    )
    return types.SimpleNamespace(saved=saved.resolve(), session=session, agent=agent,
                                 started=started, tmp=tmp_path)


def _write(path: Path, data: bytes = b"x" * 10, *, age: float = 0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if age:
        stamp = time.time() - age
        os.utime(path, (stamp, stamp))
    return path.resolve()


def _tool(world, tool: str, content: str) -> None:
    world.session.add_message("tool", content, tool_name=tool, tool_call_id=tool)


def test_every_file_the_turn_made_is_found(world):
    page = _write(world.saved / "showcase" / "s1" / "x.html")
    book = _write(world.saved / "output" / "renewals.xlsx")
    other = _write(world.saved / "showcase" / "s2" / "notes.md")
    _write(world.saved / "showcase" / "s1" / "old.pdf", age=3600)
    for skipped in ("media/s1/upload.png", "tmp/s1/scratch.json", "scripts/s1/make.py",
                    "downloads/s1/drive.pdf", "video_frames/x/f1.jpg", "showcase/s1/.cache/c.bin",
                    "showcase/s1/.hidden.txt"):
        _write(world.saved / skipped)
    got = td.new_files_this_turn(world.agent, world.started)
    assert set(got) == {page, book, other}


def test_a_members_turn_sees_only_its_session_folders(world):
    page = _write(world.saved / "showcase" / "s1" / "x.html")
    _write(world.saved / "showcase" / "s2" / "theirs.md")
    _write(world.saved / "output" / "shared.xlsx")
    assert td.new_files_this_turn(world.agent, world.started, own_session_only=True) == [page]


def test_a_link_out_of_saved_is_no_file_of_the_turn(world):
    outside = _write(world.tmp / "secret.txt")
    (world.saved / "showcase" / "s1" / "secret.txt").symlink_to(outside)
    assert td.new_files_this_turn(world.agent, world.started) == []


def test_many_files_keep_the_newest(world):
    paths = []
    for i in range(5):
        p = _write(world.saved / "output" / f"f{i}.csv")
        stamp = time.time() - 0.5 + i * 0.01
        os.utime(p, (stamp, stamp))
        paths.append(p)
    assert td.new_files_this_turn(world.agent, world.started, limit=3) == paths[-3:]


def test_no_turn_start_means_no_scan(world):
    _write(world.saved / "output" / "r.xlsx")
    assert td.new_files_this_turn(world.agent, 0.0) == []


async def test_the_web_chat_gets_a_card_for_each_file_a_script_made(world):
    from captain_claw.web.chat_handler import _send_reply_images

    _tool(world, "shell", "python saved/scripts/s1/make.py\nok")
    page = _write(world.saved / "showcase" / "s1" / "x.html")
    book = _write(world.saved / "output" / "renewals.xlsx", b"PK" + b"x" * 98)
    frames: list[dict] = []
    await _send_reply_images(frames.append, world.agent, 0,
                             "Done: saved/showcase/s1/x.html and renewals.xlsx", world.started)
    by_role = {(f["role"], f["content"]) for f in frames}
    assert by_role == {("html_file", str(page)), ("file", str(book))}
    [card] = [f for f in frames if f["role"] == "file"]
    assert card["name"] == "renewals.xlsx" and card["size"] == 100


async def test_files_a_tool_already_showed_get_no_second_card(world):
    from captain_claw.web.chat_handler import _send_reply_images

    page = _write(world.saved / "showcase" / "s1" / "page.html")
    chart = _write(world.saved / "showcase" / "s1" / "chart.png")
    viz = _write(world.saved / "showcase" / "s1" / "viz.html")
    doc = _write(world.saved / "output" / "memo.docx")
    _tool(world, "write", f"Written 120 chars (4 lines) to {page}")
    _tool(world, "image_gen", f"Image generated.\nPath: {chart}")
    _tool(world, "visualize", f"Path: {chart}\nHTML: {viz}")
    frames: list[dict] = []
    await _send_reply_images(frames.append, world.agent, 0, "Here it is.", world.started)
    assert [(f["role"], f["content"]) for f in frames] == [("file", str(doc))]


async def test_a_members_turn_gets_cards_for_its_own_files_only(world):
    from captain_claw.web.chat_handler import _send_reply_images

    _tool(world, "shell", "ok")
    mine = _write(world.saved / "showcase" / "s1" / "mine.pdf")
    _write(world.saved / "showcase" / "s2" / "theirs.pdf")
    frames: list[dict] = []
    await _send_reply_images(frames.append, world.agent, 0, "Here.", world.started,
                             own_session_only=True)
    assert [(f["role"], f["content"]) for f in frames] == [("file", str(mine))]


# ── Progress lines in the user's language ───────────────────────────────


@pytest.mark.parametrize("line, user, hidden", [
    ("Starting to collect the renewals from the CRM export…",
     "Napravi mi tablicu obnova polica za studeni.", True),
    ("Now let me check the EOJN tenders for this week.",
     "Pogledaj mi natječaje na EOJN-u za ovaj tjedan i javi što je novo.", True),
    # The model listened: a Croatian line stays.
    ("Prikupljam obnove iz izvoza CRM-a…", "Napravi mi tablicu obnova polica za studeni.", False),
    # The person writes English.
    ("Starting to collect the renewals now.", "Make me the renewals table for November please.", False),
    # Too short to judge, either side.
    ("Checking…", "Napravi mi tablicu obnova polica za studeni.", False),
    ("Starting to collect the renewals from the CRM export…", "Hvala!", False),
    # Croatian words that are English-looking ("i", "a", "to", "on", "data") don't make it English.
    ("Starting to collect the renewals from the CRM export…",
     "I to je on rekao, a data je i ponuda za studeni", True),
])
def test_out_of_language(line, user, hidden):
    from captain_claw import narration_language as nl

    assert nl.out_of_language(line, user) is hidden


def test_the_reference_is_the_persons_last_message():
    from captain_claw import narration_language as nl

    session = Session(id="s1", name="t")
    session.add_message("user", "Napravi mi tablicu obnova polica za studeni.", origin="human")
    session.add_message("assistant", "Gotovo.")
    session.add_message("user", "[SCHEDULED TASK — cron job 'tenders'] Check the EOJN tenders now",
                        origin="cron")
    assert nl.last_human_text(session) == "Napravi mi tablicu obnova polica za studeni."
    assert nl.last_human_text(Session(id="s2", name="t")) == ""


def _narrating_agent(user_text: str):
    from captain_claw.agent import Agent

    agent = Agent.__new__(Agent)
    agent.session = Session(id="s1", name="t")
    agent.session.add_message("user", user_text, origin="human")
    shown: list[str] = []
    agent.narration_callback = lambda text, iteration=0: shown.append(text)
    return agent, shown


def test_an_english_line_in_a_croatian_chat_is_handled_but_not_shown():
    agent, shown = _narrating_agent("Napravi mi tablicu obnova polica za studeni.")
    # Handled (True): the caller must not fold it into the answer instead.
    assert agent._emit_narration("Starting to collect the renewals from the CRM export…") is True
    assert shown == []
    assert agent._emit_narration("Prikupljam obnove iz izvoza CRM-a…") is True
    assert shown == ["Prikupljam obnove iz izvoza CRM-a…"]


def test_an_english_chat_keeps_its_progress_lines():
    agent, shown = _narrating_agent("Make me the renewals table for November please.")
    assert agent._emit_narration("Starting to collect the renewals from the CRM export…") is True
    assert shown == ["Starting to collect the renewals from the CRM export…"]
