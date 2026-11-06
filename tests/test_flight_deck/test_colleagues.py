"""Colleagues: an owner's agents hand each other work on the Taskboard.

A hand-over is a task of its own on the owner's board, part of the task the
first agent is on; the colleague gets it through its line with what the first
agent's turn could do and never more; its answer closes its task and goes back
to the agent that handed it over, which carries on in its own task session."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from captain_claw.flight_deck import agent_notify, colleagues, fd_scheduler, imap_routes, taskboard_triage
from captain_claw.flight_deck import line as line_mod
from captain_claw.flight_deck import taskboard_handover as handover
from captain_claw.flight_deck.auth import get_current_user

from .test_taskboard_nextsteps import ALICE, _task, board, jev, ledger, model, told  # noqa: F401

RESEARCHER = {"slug": "researcher", "name": "Researcher", "asleep": False}
WRITER = {"slug": "writer", "name": "Writer", "asleep": False}
EDITOR = {"slug": "editor", "name": "Editor", "asleep": True}
CHECKER = {"slug": "checker", "name": "Checker", "asleep": False}
TEAM = [RESEARCHER, WRITER, EDITOR, CHECKER]
HUMAN = {"mode": "human", "kind": "", "job_text": "Research Q3 and have the copy written", "mail_write": "intent"}


@pytest.fixture()
def team(tmp_path, monkeypatch):
    store = colleagues.ColleaguesStore(tmp_path / "colleagues.db")
    colleagues.set_store(store)

    async def no_inboxes(owner):
        return {}

    monkeypatch.setattr(colleagues, "_inboxes", no_inboxes)
    yield store
    colleagues.set_store(None)
    store.close()


@pytest.fixture()
def relay(monkeypatch):
    """The agents: each answers what ``replies`` says for it (None: no answer)."""
    state = SimpleNamespace(calls=[], replies={"writer": "Here is the copy: Q3 was strong.",
                                               "researcher": "Done: the report has Writer's copy in it.",
                                               "editor": "Edited.", "checker": "Checked."})

    async def run(agent, prompt, automation):
        state.calls.append({"agent": agent["slug"], "prompt": prompt, "automation": automation})
        await asyncio.sleep(0)
        reply = state.replies.get(agent["slug"])
        return reply, ({"task_session": {"id": f"s-{agent['slug']}", "name": "x"}} if reply is not None
                       else {"error": "it isn't running"})

    monkeypatch.setattr(handover, "_relay", run)
    monkeypatch.setattr(handover, "agents_for", lambda uid: list(TEAM))
    monkeypatch.setattr(handover, "_RUNNING", {})
    monkeypatch.setattr(handover, "_LINES", {})
    monkeypatch.setattr(handover, "_LINE_AGAIN", set())
    monkeypatch.setattr(handover, "_CHECK_WAIT_S", 3600.0)

    async def spent(uid):
        return False

    monkeypatch.setattr(taskboard_triage, "_quota_spent", spent)
    return state


@pytest.fixture()
def chat(monkeypatch):
    """What reached an agent's chat as a notification."""
    out: list[dict] = []

    async def notify(**kw):
        out.append(kw)

    monkeypatch.setattr(agent_notify, "notify_source_agent", notify)
    monkeypatch.setattr(fd_scheduler, "resolve_agent_by_slug", lambda slug, owner: ("localhost", 24001, "tok"))
    return out


async def _settle():
    for _ in range(400):
        lines = [t for t in handover._LINES.values() if not t.done()]
        running = [t for t in handover._RUNNING.values() if not t.done()]
        if not lines and not running:
            await asyncio.sleep(0)
            if not [t for t in handover._LINES.values() if not t.done()]:
                return
        await asyncio.wait(lines + running or [asyncio.ensure_future(asyncio.sleep(0.01))], timeout=2)
    raise AssertionError("the lines never ended")


def _said(board, t):  # noqa: F811
    return [e["summary"] for e in board.events(ALICE, t)]


def _hand(frm="researcher", to="writer", words="Write two lines of copy on Q3.", **kw):
    body = {"to": to, "instructions": words, "authority": HUMAN, "origin": {"platform": "web"}, **kw}
    return colleagues.hand_over(ALICE, frm, body)


# ── The hand-over ─────────────────────────────────────────────────────


async def test_it_is_a_task_of_its_own_part_of_the_one_the_agent_is_on(board, ledger, relay, team, chat):  # noqa: F811
    parent = _task(board, flags={}, pending=False)
    res = await _hand(board_task=parent, session_id="s-parent", title="Q3 copy")
    child = board.get_task(ALICE, res["task_id"])
    assert res["to"] == "writer" and res["to_name"] == "Writer"
    assert child["kind"] == "request" and child["direction"] == "internal" and child["channel"] == "colleague"
    assert child["agent_slug"] == "writer" and child["owes_next"] == "agent" and child["title"] == "Q3 copy"
    col = json.loads(child["flags"])["colleague"]
    assert col["from"] == "researcher" and col["to"] == "writer" and col["parent"] == parent and col["via"] == []
    assert "Researcher asked Writer: Write two lines of copy on Q3." in _said(board, res["task_id"])
    assert "Researcher handed part of it to Writer: Q3 copy" in _said(board, parent)
    await _settle()


async def test_the_colleague_gets_the_request_and_no_more_than_the_turn_could(board, ledger, relay, team, chat):  # noqa: F811
    parent = _task(board, flags={}, pending=False)
    await _hand(board_task=parent, session_id="s-parent")
    await _settle()
    first = relay.calls[0]
    assert first["agent"] == "writer"
    auto = first["automation"]
    assert auto["kind"] == "colleague" and auto["mail_write"] == "intent"
    assert auto["job_text"] == HUMAN["job_text"], "the person's own words, never the agent's"
    assert auto["task"]["board"] is True and auto["ref"].startswith("act_")
    assert "Your colleague Researcher" in first["prompt"] and "Write two lines of copy on Q3." in first["prompt"]
    assert "isn't your owner's own words" in first["prompt"]


async def test_the_answer_closes_its_task_and_goes_back_into_the_parents_session(board, ledger, relay, team, chat):  # noqa: F811
    parent = _task(board, flags={}, pending=False)
    res = await _hand(board_task=parent, session_id="s-parent")
    await _settle()
    child = board.get_task(ALICE, res["task_id"])
    assert child["status"] == "done" and child["closed_by_party"] == "Writer"
    assert child["result_text"] == "Here is the copy: Q3 was strong."
    back = relay.calls[1]
    assert back["agent"] == "researcher"
    assert back["automation"]["task"]["id"] == parent and back["automation"]["task"]["session_id"] == "s-parent"
    assert back["automation"]["kind"] == "colleague" and back["automation"]["job_text"] == HUMAN["job_text"]
    head, _, rest = back["prompt"].partition("<<<DATA")
    assert "Writer finished what you handed over" in head and "Q3 was strong" in rest
    row = board.get_task(ALICE, parent)
    assert row["owes_next"] == "me" and row["result_text"] == "Done: the report has Writer's copy in it."
    said = _said(board, parent)
    assert any(s.startswith("Writer finished what Researcher handed over") for s in said)
    assert "Writer's answer went back to Researcher" in said
    assert team.handoffs_from(ALICE, "researcher")[0]["state"] == "done"
    assert chat == [], "it was on a board task: no chat notification"


async def test_without_a_board_task_the_answer_comes_to_its_chat(board, ledger, relay, team, chat):  # noqa: F811
    res = await _hand(title="Copy")
    await _settle()
    assert [c["agent"] for c in relay.calls] == ["writer"]
    assert len(chat) == 1 and chat[0]["ok"] is True and "Q3 was strong" in chat[0]["summary"]
    assert chat[0]["source_port"] == 24001 and chat[0]["origin"] == {"platform": "web", "user_id": None, "chat_id": None}
    assert board.get_task(ALICE, res["task_id"])["status"] == "done"


async def test_when_the_colleague_cant_it_waits_for_the_owner_and_the_first_agent_hears(board, ledger, relay, team, chat):  # noqa: F811
    relay.replies["writer"] = None
    parent = _task(board, flags={}, pending=False)
    res = await _hand(board_task=parent, session_id="s-parent")
    await _settle()
    child = board.get_task(ALICE, res["task_id"])
    assert child["status"] == "open" and child["owes_next"] == "me"
    back = relay.calls[1]
    assert back["agent"] == "researcher" and "couldn't finish" in back["prompt"]
    assert team.handoffs_from(ALICE, "researcher")[0]["state"] == "failed"


async def test_work_still_out_keeps_the_task_with_the_agents(board, ledger, relay, team, chat):  # noqa: F811
    """The first agent's own run ends before its colleague is through: it isn't the owner's move yet."""
    parent = _task(board, flags={}, pending=False)
    gates = {"researcher": asyncio.Event(), "writer": asyncio.Event()}
    real = handover._relay
    held = {"researcher": True}

    async def slow(agent, prompt, automation):
        if agent["slug"] == "writer" or (agent["slug"] == "researcher" and held.pop("researcher", False)):
            await gates[agent["slug"]].wait()
        return await real(agent, prompt, automation)

    handover._relay = slow
    try:
        await handover.hand_over(ALICE, parent, "researcher", "Write the report.")
        await _hand(board_task=parent, session_id="s-parent")
        gates["researcher"].set()
        for _ in range(200):
            if "Still with Writer" in _said(board, parent):
                break
            await asyncio.sleep(0.01)
        row = board.get_task(ALICE, parent)
        assert row["owes_next"] == "agent" and "Still with Writer" in _said(board, parent)
        gates["writer"].set()
        await _settle()
        assert board.get_task(ALICE, parent)["owes_next"] == "me"
        assert [c["agent"] for c in relay.calls] == ["researcher", "writer", "researcher"]
    finally:
        for g in gates.values():
            g.set()
        await _settle()
        handover._relay = real


# ── Never more than it could ──────────────────────────────────────────


def test_authority_passes_on_and_never_grows():
    allow = colleagues.authority_in({"mode": "automated", "kind": "taskboard", "job_text": "Book it", "mail_write": "allow"})
    assert colleagues.colleague_marker(allow) == {"kind": "colleague", "job_text": "Book it", "mail_write": "allow"}
    human = colleagues.authority_in({"mode": "human", "job_text": "email Ana the report", "mail_write": "allow"})
    assert colleagues.colleague_marker(human)["mail_write"] == "intent", "a person's turn is judged on their words"
    draft = colleagues.authority_in({"mode": "automated", "kind": "inbox", "mail_write": "draft", "send_from": "rita"})
    assert "send_from" not in colleagues.colleague_marker(draft), "an inbox never passes to a colleague"
    assert colleagues.back_marker(draft) == {"kind": "inbox", "job_text": "", "mail_write": "draft", "send_from": "rita"}
    bad = colleagues.authority_in({"mode": "automated", "kind": "made-up", "mail_write": "allow"})
    assert colleagues.colleague_marker(bad)["mail_write"] == "deny"
    assert colleagues.authority_in(None)["mail_write"] == "deny"


# ── Guards ────────────────────────────────────────────────────────────


async def test_not_to_itself_nor_to_someone_unknown(board, ledger, relay, team):  # noqa: F811
    with pytest.raises(ValueError, match="That's you"):
        await _hand(to="Researcher")
    with pytest.raises(LookupError, match="Your colleagues: Writer"):
        await _hand(to="nobody")
    with pytest.raises(ValueError, match="instructions"):
        await _hand(words="  ")


async def test_by_name_answered_to_or_role(board, ledger, relay, team, chat):  # noqa: F811
    team.save_identity(ALICE, "writer", colleagues.clean_identity({"role": "Copywriter", "names": ["Wendy"]}))
    assert (await _hand(to="wendy"))["to"] == "writer"
    assert (await _hand(to="copywriter"))["to"] == "writer"
    await _settle()


async def test_never_back_along_the_chain(board, ledger, relay, team, chat):  # noqa: F811
    parent = _task(board, flags={}, pending=False)
    child = (await _hand(board_task=parent))["task_id"]
    with pytest.raises(PermissionError, match="already on this piece of work"):
        await colleagues.hand_over(ALICE, "writer", {"to": "researcher", "instructions": "x", "board_task": child,
                                                      "authority": HUMAN})
    await _settle()


async def test_a_chain_stops_at_four_agents(board, ledger, relay, team, chat, monkeypatch):  # noqa: F811
    monkeypatch.setattr(colleagues, "CHAIN_MAX", 3)
    parent = _task(board, flags={}, pending=False)
    c1 = (await _hand(board_task=parent))["task_id"]
    c2 = (await colleagues.hand_over(ALICE, "writer", {"to": "editor", "instructions": "edit", "board_task": c1,
                                                         "authority": HUMAN}))["task_id"]
    assert json.loads(board.get_task(ALICE, c2)["flags"])["colleague"]["via"] == ["researcher"]
    with pytest.raises(PermissionError, match="passed through 3 agents"):
        await colleagues.hand_over(ALICE, "editor", {"to": "checker", "instructions": "check", "board_task": c2,
                                                      "authority": HUMAN})
    await _settle()


async def test_a_daily_cap(board, ledger, relay, team, chat, monkeypatch):  # noqa: F811
    monkeypatch.setattr(colleagues, "DAILY_MAX", 1)
    await _hand()
    with pytest.raises(PermissionError, match="in the last day"):
        await _hand()
    await _settle()


async def test_only_to_the_colleagues_it_may_hand_to(board, ledger, relay, team):  # noqa: F811
    team.save_identity(ALICE, "researcher", colleagues.clean_identity({"hands_to": ["editor"]}))
    with pytest.raises(PermissionError, match="hasn't let you hand work to Writer"):
        await _hand()


async def test_an_agent_that_isnt_the_owners_has_no_colleagues(board, ledger, relay, team):  # noqa: F811
    with pytest.raises(PermissionError):
        await colleagues.hand_over(ALICE, "someone-elses", {"to": "writer", "instructions": "x"})


async def test_agent_batches_stay_off_the_users_banner(board, ledger, relay, team, chat):  # noqa: F811
    await _hand()
    assert board.line_batches(ALICE, ended_since=0) == []
    await _settle()


# ── Identity ──────────────────────────────────────────────────────────


def test_identity_keeps_what_it_was_not_sent():
    base = colleagues.clean_identity({"role": " Researcher  of   things ", "names": ["Rita", "rita", "R", "Researcher"],
                                      "addresses": ["Me+Rita@Gmail.com"], "pick_up_mail": True})
    assert base["role"] == "Researcher of things" and base["names"] == ["Rita", "Researcher"]
    assert base["addresses"] == ["me+rita@gmail.com"] and base["pick_up_mail"] is True
    more = colleagues.clean_identity({"about": "Finds numbers"}, base)
    assert more["role"] == "Researcher of things" and more["about"] == "Finds numbers"
    with pytest.raises(ValueError, match="isn't an email address"):
        colleagues.clean_identity({"addresses": ["nope"]})
    assert colleagues.clean_identity({"hands_to": "all"})["hands_to"] == "all"


def test_picking_up_email_is_on_unless_switched_off():
    assert colleagues.picks_up_mail(None) is True and colleagues.picks_up_mail({"role": "x"}) is True
    assert colleagues.picks_up_mail({"pick_up_mail": False}) is False


def test_the_roster_view_defaults_to_the_agents_name():
    v = colleagues.view({"slug": "w", "name": "Writer"}, None)
    assert v["names"] == ["Writer"] and v["names_set"] is False and v["email"] == "" and v["pick_up_mail"] is True, "every agent picks up its email unless switched off"
    v = colleagues.view({"slug": "w", "name": "Writer"}, {"inbox": "w-box", "addresses": ["a@b.co"]}, {"w-box": "w@firm.co"})
    assert v["email"] == "w@firm.co" and v["inbox_email"] == "w@firm.co"


# ── Routes ────────────────────────────────────────────────────────────


@pytest.fixture()
def client(team, monkeypatch, board, ledger, relay):  # noqa: F811
    app = FastAPI()
    app.include_router(colleagues.router)
    app.dependency_overrides[get_current_user] = lambda: {"id": ALICE, "role": "user"}
    monkeypatch.setattr(colleagues, "_uid", lambda user: ALICE)
    monkeypatch.setattr(line_mod, "user_agents", lambda uid: [{"slug": a["slug"], "name": a["name"]} for a in TEAM])
    state = {"recs": [{"id": "rita-box", "email": "rita@firm.co", "agents": [], "sending": {"enabled": False}},
                      {"id": "main", "email": "alice@firm.co", "agents": [], "sending": {"enabled": True},
                       "smtp_host": "smtp.firm.co"}]}

    async def load(db, uid):
        return [dict(r) for r in state["recs"]]

    async def save(db, uid, recs):
        state["recs"] = recs

    monkeypatch.setattr(imap_routes, "_load", load)
    monkeypatch.setattr(imap_routes, "_save", save)
    import captain_claw.flight_deck.auth as auth_mod

    monkeypatch.setattr(auth_mod, "get_db", lambda: None)

    async def ready(uid):
        return {"gmail": True, "outlook": False, "imap": True}

    monkeypatch.setattr(colleagues, "_mail_sources", ready)

    async def caller(request):
        return ALICE, request.headers.get("X-Test-Agent", "researcher")

    monkeypatch.setattr(colleagues, "_agent", caller)
    return SimpleNamespace(http=TestClient(app), state=state)


def test_the_editor_sets_an_identity_and_an_inbox_becomes_the_agents_own(client, team):
    r = client.http.put("/fd/colleagues/researcher", json={"role": "Researcher", "inbox": "rita-box",
                                                           "pick_up_mail": True, "names": ["Rita"]})
    assert r.status_code == 200, r.text
    agent = r.json()["agent"]
    assert agent["inbox_email"] == "rita@firm.co" and agent["email"] == "rita@firm.co" and agent["names"] == ["Rita"]
    assert next(x for x in client.state["recs"] if x["id"] == "rita-box")["agents"] == ["researcher"]
    r = client.http.put("/fd/colleagues/writer", json={"inbox": "rita-box"})
    assert r.status_code == 200
    assert team.identity(ALICE, "researcher")["inbox"] == "", "an inbox is one agent's own"
    assert client.http.put("/fd/colleagues/writer", json={"inbox": "nope"}).status_code == 400
    assert client.http.put("/fd/colleagues/stranger", json={"role": "x"}).status_code == 404
    listed = client.http.get("/fd/colleagues").json()
    assert [a["slug"] for a in listed["agents"]] == ["researcher", "writer", "editor", "checker"]
    assert {i["id"] for i in listed["inboxes"]} == {"rita-box", "main"} and listed["mail"]["gmail"] is True


def test_the_agent_sees_itself_and_its_colleagues(client, team):
    team.save_identity(ALICE, "researcher", colleagues.clean_identity({"role": "Researcher", "hands_to": ["writer"]}))
    data = client.http.post("/fd/colleagues/agent/roster").json()
    assert data["me"]["slug"] == "researcher" and data["me"]["role"] == "Researcher"
    by = {c["slug"]: c for c in data["colleagues"]}
    assert set(by) == {"writer", "editor", "checker"}
    assert by["writer"]["can_hand_to"] is True and by["editor"]["can_hand_to"] is False


def test_hand_over_route_answers_with_what_to_say(client, team):
    r = client.http.post("/fd/colleagues/agent/hand_over", json={"to": "nobody", "instructions": "x"})
    assert r.status_code == 404 and "Your colleagues" in r.json()["detail"]
    r = client.http.post("/fd/colleagues/agent/hand_over", json={"to": "writer", "instructions": ""})
    assert r.status_code == 400
