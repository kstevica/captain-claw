"""Needs you means blocked on you: a question, an approval, a Go, a decision —
not news. An item that waits for an answer given somewhere else (a Bat
question, a Swarm's Go, a purchase) leaves Needs you when answered there, not
when read; the client knows the same list."""

from __future__ import annotations

import re
from pathlib import Path

from captain_claw.flight_deck import line

from .test_line import ALICE, db  # noqa: F401

_LINE_TS = Path(__file__).resolve().parents[2] / "flight-deck" / "src" / "services" / "line.ts"


def test_reports_are_not_needs_you():
    for kind in ("deck_update", "system", "data_room_delivery", "custom_tool",
                 "never_heard_of_it", "run", "bat_done", "share", "email_sent"):
        assert line.classify(kind)[1] is False, kind


def test_what_blocks_on_you_is():
    for kind in ("person_ask", "bat_ask", "captain_swarm", "payment", "autonomy_proposal", "run_error",
                 "crew_task", "course_review", "email_autoreply_held"):
        assert line.classify(kind)[1] is True, kind
    # Waiting by what it's about, whatever the type.
    assert line.classify("custom_tool", "custom_tool_approval")[1] is True
    assert line.classify("custom_tool", "custom_tool_proposal")[1] is True
    assert line.classify("info", "purchase")[1] is True


def test_the_client_knows_the_same_answered_elsewhere_list():
    text = _LINE_TS.read_text(encoding="utf-8")
    block = re.search(r"ANSWERED_ELSEWHERE = new Set\(\[(.*?)\]\)", text, re.S).group(1)
    assert set(re.findall(r"'([a-z_]+)'", block)) == set(line.ANSWERED_REFS)


async def test_answered_elsewhere_stays_until_settled(db):  # noqa: F811
    ask = await db.add_notification(ALICE, "bat_ask", "Bat needs you", "Which account?", "bat_ask", "run-1")
    go = await db.add_notification(ALICE, "captain_swarm", "A Captain Swarm run waits for your Go",
                                   "Sort the inbox", "captain_swarm_run", "sw-1")
    buy = await db.add_notification(ALICE, "payment", "Approve a purchase", ref_type="purchase", ref_id="pu-1")
    err = await db.add_notification(ALICE, "run_error", "Bat run failed", "", "bat", "run-1")
    waiting = {r["id"] for r in await db.list_unresolved_line_items(ALICE)}
    assert {ask, go, buy, err} <= waiting
    await db.mark_all_line_items_read(ALICE)
    waiting = {r["id"] for r in await db.list_unresolved_line_items(ALICE)}
    # Reading a failed run's report answers it; the questions still wait.
    assert {ask, go, buy} <= waiting and err not in waiting
    assert await line.settle(ALICE, "bat_ask", "run-1") == 1
    assert await line.settle(ALICE, "captain_swarm_run", "sw-1") == 1
    assert await line.settle(ALICE, "purchase", "pu-1") == 1
    assert not {ask, go, buy} & {r["id"] for r in await db.list_unresolved_line_items(ALICE)}


async def test_settle_never_raises(db):  # noqa: F811
    assert await line.settle("", "bat_ask", "x") == 0
    assert await line.settle(ALICE, "bat_ask", "nothing-there") == 0


async def test_the_bat_question_rings_with_its_own_ref(db, monkeypatch):  # noqa: F811
    from captain_claw.flight_deck import bat_routes

    await bat_routes._notify_ask({"owner_id": ALICE, "question": "Which account?", "run_id": "run-9", "secret": True})
    (item,) = [i for i in await db.list_line_items(ALICE) if i["title"] == "Bat needs you"]
    assert item["ref_type"] == "bat_ask" and item["ref_id"] == "run-9" and item["needs_you"]


def test_telegram_results_are_reports():
    src = (Path(line.__file__).with_name("telegram_shared_routes.py")).read_text(encoding="utf-8")
    assert '"success", "Telegram access approved"' in src and '"info", "Telegram access not approved"' in src
    assert src.count('ref_type="telegram_access"') == 2
