"""Bat Phase 4 — the ask store + human_ask registry (pause-and-ask).

Durable asks, compare-and-set answers, and the secret rule: a secret value is
kept in memory only, never written to the durable row (which gets a redacted
placeholder), and is never answerable over a chat channel.
"""

from __future__ import annotations

import pytest

from captain_claw.flight_deck import human_ask
from captain_claw.flight_deck.bat_store import BatStore


@pytest.fixture
async def store(tmp_path):
    s = BatStore(tmp_path / "bat.db")
    await s.init()
    await s.create_run(run_id="r1", owner_id="u1", title="t", task="task")
    yield s
    await s.close()


@pytest.fixture(autouse=True)
def _notifier():
    notes: list[dict] = []

    async def rec(ask):
        notes.append(ask)

    saved = human_ask._NOTIFY
    human_ask.set_notifier(rec)
    human_ask._SECRETS.clear()
    yield notes
    human_ask.set_notifier(saved)
    human_ask._SECRETS.clear()


# ── store asks ─────────────────────────────────────────────────────────

async def test_store_ask_crud_and_cas(store):
    await store.create_ask(ask_id="a1", run_id="r1", owner_id="u1", kind="input",
                           question="the code?")
    assert (await store.get_ask("a1"))["status"] == "open"
    assert (await store.latest_open_ask("r1"))["id"] == "a1"
    assert [a["id"] for a in await store.open_asks_for_owner("u1")] == ["a1"]

    assert await store.answer_ask("a1", "42", via="ui") is True
    assert await store.answer_ask("a1", "again", via="ui") is False  # already answered (CAS)
    a = await store.get_ask("a1")
    assert a["status"] == "answered" and a["answer"] == "42" and a["answered_via"] == "ui"
    assert await store.open_asks_for_owner("u1") == []


async def test_store_expire_returns_run_ids(store):
    await store.create_ask(ask_id="a1", run_id="r1", owner_id="u1", kind="input",
                           question="q", expires_at=100.0)
    assert await store.expire_asks(now=50.0) == []          # not yet
    assert await store.expire_asks(now=200.0) == ["r1"]     # now expired
    assert (await store.get_ask("a1"))["status"] == "expired"


# ── human_ask registry ─────────────────────────────────────────────────

async def test_raise_ask_persists_and_notifies(store, _notifier):
    ask_id = await human_ask.raise_ask(store, run_id="r1", owner="u1", kind="input",
                                       question="what now?")
    row = await store.get_ask(ask_id)
    assert row and row["question"] == "what now?"
    assert _notifier and _notifier[0]["id"] == ask_id and _notifier[0]["question"] == "what now?"


async def test_secret_ask_redacts_row_and_notification_keeps_raw_in_memory(store, _notifier):
    ask_id = await human_ask.raise_ask(store, run_id="r1", owner="u1", kind="secret",
                                       question="enter your 2FA code", secret=True)
    # the fan-out must not carry the question's secret framing as a value, and
    # the notifier is told it's secret so it won't go to a chat channel.
    assert _notifier[0]["secret"] is True
    assert _notifier[0]["question"] == human_ask._REDACTED

    res = await human_ask.answer(store, ask_id, "123456", via="ui")
    assert res["ok"]
    row = await store.get_ask(ask_id)
    assert row["answer"] == human_ask._REDACTED          # raw code never persisted
    assert "123456" not in row["answer"]
    assert human_ask.take_secret(ask_id) == "123456"     # raw available in memory, once
    assert human_ask.take_secret(ask_id) is None         # consumed


async def test_answer_non_secret_stores_text(store):
    ask_id = await human_ask.raise_ask(store, run_id="r1", owner="u1", kind="input", question="q")
    assert (await human_ask.answer(store, ask_id, "the value"))["ok"]
    assert (await store.get_ask(ask_id))["answer"] == "the value"


async def test_answer_already_answered_is_rejected(store):
    ask_id = await human_ask.raise_ask(store, run_id="r1", owner="u1", kind="input", question="q")
    assert (await human_ask.answer(store, ask_id, "first"))["ok"]
    assert not (await human_ask.answer(store, ask_id, "second"))["ok"]


async def test_answer_for_owner_single_multiple_and_secret(store):
    # single open ask → resolves
    a1 = await human_ask.raise_ask(store, run_id="r1", owner="u1", kind="input", question="q1")
    assert (await human_ask.answer_for_owner(store, "u1", "x"))["ok"]
    # multiple open → ambiguous, refuse (answer by id in UI)
    await human_ask.raise_ask(store, run_id="r1", owner="u1", kind="input", question="q2")
    await human_ask.raise_ask(store, run_id="r1", owner="u1", kind="input", question="q3")
    assert not (await human_ask.answer_for_owner(store, "u1", "y"))["ok"]
    # a lone secret ask is never answerable over a channel
    await store.answer_ask((await store.latest_open_ask("r1"))["id"], "x")  # clear one
    # leave exactly one, make it secret
    for a in await store.open_asks_for_owner("u1"):
        await store.cancel_ask(a["id"])
    sid = await human_ask.raise_ask(store, run_id="r1", owner="u1", kind="secret",
                                    question="code", secret=True)
    assert not (await human_ask.answer_for_owner(store, "u1", "999"))["ok"]
    assert (await store.get_ask(sid))["status"] == "open"  # untouched
