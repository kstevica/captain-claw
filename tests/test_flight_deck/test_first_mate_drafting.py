"""First Mate drafts a factory from words or a Course goal (P5, piece 2).

Pinned here:

* the model's answer becomes a factory through the store's own checks: a
  name, the charter, starts (a usable schedule, new mail), done-when lines,
  grants only for mail sending and organizing, never spending or accounts;
  a schedule the scheduler refuses is dropped; up to three sealed cases;
* an answer that isn't JSON is asked again once, then refused; no model
  answer is no draft;
* the draft waits for approval (submitted), with a card on the Line keyed by
  its hash; approving, declining or changing it settles the card;
* the route drafts from words or one of the owner's goals; the tool's draft
  action does the same; a full deck refuses before calling the model.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import course as c
from captain_claw.flight_deck import events, line, rate_limiter, server
from captain_claw.flight_deck import first_mate_cases as cases
from captain_claw.flight_deck import first_mate_drafting as drafting
from captain_claw.flight_deck import first_mate_routes as routes
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.flight_deck.first_mate_store import FirstMateError

BOB, CAROL = "user-bob", "user-carol"

ANSWER = {
    "title": "Invoice replies",
    "charter": "Every weekday, go through my inbox for client mail about invoices and draft a short reply to each. "
               "Never send anything. Finish with who wrote and whether a draft waits.",
    "schedule": "weekdays 08:00",
    "on_new_mail": False,
    "done_when": ["every invoice question has a draft", "nothing was sent"],
    "may_send_email": False,
    "may_organize_mail": False,
    "cases": [{"title": "Total question", "input": "From: Ana <ana@client.example>\nIs the total right?",
               "good_result": ["drafts a reply to Ana", "doesn't send it"]}],
}


@pytest.fixture(autouse=True)
def _no_register(monkeypatch):
    from captain_claw.flight_deck import tool_audit_routes
    monkeypatch.setattr(tool_audit_routes, "_data_dir", None)
    monkeypatch.setattr(tool_audit_routes, "_store", None)


@pytest.fixture
async def db(tmp_path: Path):
    prev = fd_auth._db
    d = FlightDeckDB(tmp_path / "fd.db")
    await d.init()
    set_auth_db(d)
    for uid in (BOB, CAROL):
        await d._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, 'user', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", uid))
    await d._db.commit()
    try:
        yield d
    finally:
        await d.close()
        fd_auth._db = prev


@pytest.fixture
def store(tmp_path, monkeypatch, db):
    monkeypatch.setattr(events, "_STORE", events.EventsStore(tmp_path / "events.db"))
    monkeypatch.setattr(c, "_STORE", c.CourseStore(tmp_path / "course.db"))
    s = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", s)
    s.set_settings({"enabled": True, "admins_only": False})
    return s


@pytest.fixture(autouse=True)
def quiet_line(monkeypatch):
    async def fake_fan_out(user_id, item, to_phones=True, rung=False):
        return {}
    monkeypatch.setattr(line, "fan_out", fake_fan_out)


class Model:
    """Stands in for the owner's model: answers in turn from ``answers``."""

    def __init__(self, monkeypatch, *answers):
        self.answers, self.asked, self.budgets, self.systems = list(answers), [], [], []

        async def ask(owner, system, user, max_tokens=drafting.DRAFT_TOKENS):
            self.asked.append(user)
            self.systems.append(system)
            self.budgets.append(max_tokens)
            got = self.answers.pop(0) if self.answers else ""
            if isinstance(got, Exception):
                raise got
            return got if isinstance(got, str) else json.dumps(got)

        monkeypatch.setattr(drafting, "_ask_model", ask)


async def _items(db, who=BOB):
    await asyncio.sleep(0)
    return await db.list_line_items(who, limit=50)


# ── the answer as a factory ────────────────────────────────────────────

def test_an_answer_becomes_checked_fields():
    f = drafting.fields_of({**ANSWER, "may_send_email": True, "may_organize_mail": True, "on_new_mail": True})
    assert f["title"] == "Invoice replies" and f["checks"]["contract"] == ANSWER["done_when"]
    assert f["triggers"][0] == {"kind": "schedule", "schedule": "weekdays 08:00", "tz": ""}
    assert f["triggers"][1]["kind"] == "events" and f["triggers"][1]["source"] == "gmail"
    assert f["grants"]["email"] and f["grants"]["organize"] and f["grants"]["sends_per_run"] == 3
    assert not f["grants"]["spend"] and not f["grants"]["accounts"]
    sneaky = drafting.fields_of({**ANSWER, "spend": True, "accounts": True, "grants": {"spend": True}})
    assert not sneaky["grants"]["spend"] and not sneaky["grants"]["accounts"] and not sneaky["grants"]["email"]
    assert drafting.fields_of({**ANSWER, "schedule": "whenever it feels right"})["triggers"] == []
    with pytest.raises(FirstMateError, match="no charter"):
        drafting.fields_of({**ANSWER, "charter": ""})
    many = drafting.fields_of({**ANSWER, "cases": [ANSWER["cases"][0]] * 6, "done_when": ["a"] * 9})
    assert len(many["cases"]) == 3 and len(many["checks"]["contract"]) == drafting.DONE_WHEN_MAX == 6
    assert drafting._parse('Here it is:\n```json\n{"title": "x"}\n```') == {"title": "x"}


async def test_a_draft_waits_with_its_card(store, db, monkeypatch):
    model = Model(monkeypatch, ANSWER)
    made = await drafting.draft(store, BOB, "Draft replies to clients asking about invoices, every weekday.")
    assert made["state"] == "awaiting_charter" and made["version"] == 0 and made["cases"] == 1
    d = made["draft"]
    assert d["status"] == "awaiting" and d["created_by"] == "first_mate" and d["note"] == "Drafted from your words, rewritten as a charter"
    assert "invoices" in model.asked[0]
    [case] = cases.cases(store, made["id"])
    assert case["sealed"] and case["origin"] == "first_mate"
    [item] = await _items(db)
    assert item["title"] == "A new factory to approve: Invoice replies" and item["needs_you"]
    assert item["ref_type"] == "first_mate_charter" and item["ref_id"] == f"{made['id']}:{d['sha'][:12]}"
    assert "**Done when**" in item["body"] and "**Starts:** weekdays 08:00" in item["body"]
    assert "**Test cases:** 1, sealed" in item["body"] and "You asked: Draft replies" in item["body"]


async def test_an_unusable_answer_is_asked_again_then_refused(store, db, monkeypatch):
    model = Model(monkeypatch, "Sure! I'd make an inbox job.", ANSWER)
    made = await drafting.draft(store, BOB, "Draft replies to invoice questions.")
    assert made["title"] == "Invoice replies" and "couldn't be used" in model.asked[1]
    Model(monkeypatch, "nope", "still nope")
    with pytest.raises(FirstMateError, match="couldn't draft it"):
        await drafting.draft(store, BOB, "Draft replies to invoice questions.")
    Model(monkeypatch, RuntimeError("connection refused"))
    with pytest.raises(FirstMateError, match="didn't answer"):
        await drafting.draft(store, BOB, "Draft replies to invoice questions.")
    none = Model(monkeypatch, FirstMateError("no model to draft with: set up a tier set first"), ANSWER)
    with pytest.raises(FirstMateError, match="^no model to draft with"):
        await drafting.draft(store, BOB, "Draft replies to invoice questions.")
    assert len(none.asked) == 1  # not asked again
    with pytest.raises(FirstMateError, match="a sentence or two"):
        await drafting.draft(store, BOB, "hi")
    assert len(store.list_factories(BOB)) == 1


async def test_a_full_deck_refuses_before_the_model(store, db, monkeypatch):
    store.set_settings({"max_factories": 1})
    model = Model(monkeypatch, ANSWER, ANSWER)
    await drafting.draft(store, BOB, "Draft replies to invoice questions.")
    with pytest.raises(FirstMateError, match="already have 1"):
        await drafting.draft(store, BOB, "Another one, about follow-ups.")
    assert len(model.asked) == 1


# ── routes and the card ────────────────────────────────────────────────

def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, client=("203.0.113.9", 40001)),
                             base_url="http://fd.test")


def _hdr(who):
    return {"Authorization": f"Bearer {create_access_token(who)}"}


@pytest.fixture
def deck(monkeypatch, store):
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())


async def test_the_route_and_settling_the_card(deck, store, db, monkeypatch):
    Model(monkeypatch, ANSWER, ANSWER)
    async with _client() as cl:
        r = await cl.post("/fd/first-mate/draft", headers=_hdr(BOB),
                          json={"words": "Draft replies to invoice questions every weekday."})
        assert r.status_code == 200
        made = r.json()
        sha = made["draft"]["sha"]
        # Changing the draft settles the card that showed the old one.
        r = await cl.patch(f"/fd/first-mate/factories/{made['id']}/draft", headers=_hdr(BOB),
                           json={"charter": "Draft short replies to invoice questions. Never send."})
        assert r.status_code == 200
        [item] = await _items(db)
        assert item["resolved_at"]
        new_sha = r.json()["draft"]["sha"]
        assert new_sha != sha
        r = await cl.post(f"/fd/first-mate/factories/{made['id']}/approve", headers=_hdr(BOB),
                          json={"version": 1, "sha": new_sha})
        assert r.status_code == 200 and r.json()["state"] == "running"
        # A second draft, declined: its card is settled too.
        r = await cl.post("/fd/first-mate/draft", headers=_hdr(BOB), json={"words": "Draft replies to follow-ups."})
        second = r.json()
        r = await cl.post(f"/fd/first-mate/factories/{second['id']}/decline", headers=_hdr(BOB), json={"version": 1})
        assert r.status_code == 200
        r = await cl.post("/fd/first-mate/draft", headers=_hdr(BOB), json={"words": 7})
        assert r.status_code == 400
    cards = [i for i in await _items(db) if i["ref_type"] == "first_mate_charter"]
    assert len(cards) == 2 and all(i["resolved_at"] for i in cards)


async def test_a_draft_from_a_goal(deck, store, db, monkeypatch):
    model = Model(monkeypatch, ANSWER)
    goal = c.get_store().create(scope="user", owner_id=BOB, horizon="medium", title="Get paid on time",
                                     detail="Clients pay late.", success="Invoices paid within 30 days")
    theirs = c.get_store().create(scope="user", owner_id=CAROL, horizon="medium", title="Carol goal", success="done")
    async with _client() as cl:
        r = await cl.post("/fd/first-mate/draft", headers=_hdr(BOB), json={"goal_id": goal["id"]})
        assert r.status_code == 200
        made = r.json()
        r = await cl.post("/fd/first-mate/draft", headers=_hdr(BOB), json={"goal_id": theirs["id"]})
        assert r.status_code == 400
    assert made["goal_id"] == goal["id"] and made["draft"]["note"] == "Drafted from the goal: Get paid on time"
    assert "Get paid on time" in model.asked[0] and "Invoices paid within 30 days" in model.asked[0]


async def test_the_tools_draft_action(store, db, monkeypatch):
    from captain_claw.flight_deck import basna_routes
    monkeypatch.setattr(basna_routes, "_resolve_owner", lambda body: BOB)
    Model(monkeypatch, ANSWER)
    got = await routes.agent_tool(routes.AgentToolReq(action="draft", words="Draft replies to invoice questions."))
    assert got["ok"] and got["text"].startswith('Drafted "Invoice replies". It waits for your approval')
    assert "It came with 1 sealed test cases." in got["text"]
    assert store.get(got["id"])["state"] == "awaiting_charter"


async def test_a_cut_off_answer_gets_more_room(store, db, monkeypatch):
    """A thinking model spent its budget thinking: the first answer is cut off (or
    empty), and the second try says why and has twice the room."""
    cut = json.dumps(ANSWER)[:300]
    model = Model(monkeypatch, cut, ANSWER)
    got = await drafting.draft(store, BOB, "Reply to invoice questions from clients, as drafts for me to send.")
    assert got["title"] == "Invoice replies"
    assert model.budgets == [drafting.DRAFT_TOKENS, drafting.RETRY_TOKENS]
    assert "the draft was cut off before its end" in model.asked[1] and "keep your thinking short" in model.asked[1]
    model = Model(monkeypatch, "[]", "")
    with pytest.raises(FirstMateError, match="came back empty"):
        await drafting.draft(store, BOB, "Reply to invoice questions from clients, as drafts for me to send.")
    assert drafting.DRAFT_TOKENS >= 12_000 and "Keep your thinking short" in drafting.SYSTEM


async def test_a_task_is_rewritten_as_a_charter_unless_switched_off(store, db, monkeypatch):
    """A task as its owner wrote it ("find about 500 leads in two countries") left the
    runs to guess its groups, its columns and where a fact must come from. The draft
    now rewrites it as a full charter; switched off, their words are the charter."""
    words = "Find about 500 companies and people in two countries who would use my product. Name, email, reason."
    full = "Find about 500 leads, in four groups of 125. Columns: Name, Email, Reason. An email comes from its own site."
    model = Model(monkeypatch, {**ANSWER, "charter": full, "is_list": True})
    made = await drafting.draft(store, BOB, words)
    assert made["draft"]["charter"] == full and made["draft"]["runner"] == "funnel"  # a list runs as a list
    asked = model.systems[0]
    for said in ("groups with an amount for each", "The columns, named, in order", "where it must come from",
                 "when a fact isn't found", "do not relax a rule to fill it", "keep the link"):
        assert said in asked
    assert "exactly as they wrote them" not in asked

    model = Model(monkeypatch, {**ANSWER, "charter": "Something the model made up instead.", "is_list": False})
    made = await drafting.draft(store, BOB, words, rephrase=False)
    assert made["draft"]["charter"] == words and made["draft"]["runner"] == "bat"  # theirs, whatever came back
    assert made["draft"]["note"] == "Drafted from your words, the charter as written"
    assert "exactly as they wrote them" in model.systems[0] and "groups with an amount" not in model.systems[0]
    assert made["draft"]["checks"]["contract"] == ANSWER["done_when"]  # the rest is still drafted


def test_the_rewrite_keeps_what_a_why_it_fits_column_is_for():
    """A rewritten charter turned "a reason why they would use our product" into "a
    sentence from the lead's own content", and every reason left the product out."""
    from captain_claw.flight_deck import first_mate_drafting as d
    assert "keeps that purpose in their words" in d.system_for(True)
    assert "keeps that purpose" not in d.system_for(False)
