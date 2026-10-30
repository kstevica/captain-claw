"""The Taskboard v1 end to end, in one process: an agent's turn register and
tool-call register (the agent side's own record builders and shippers, over
the agent's own auth) into the deck's routes, the sweep, then triage — rules
alone first (no Jev key, no model), then a Jev stub and a fake model.

What it shows: the owner's chat question is a task on the owner's board, done
as "answered in chat" by rules; a shared agent member's turn is on the
member's board only (the owner gets a 404); a corrective and a nudge make
nothing; the tool calls of a turn are on its task's timeline, an ok send as
recorded evidence (and its Gmail thread linked); with Jev, its answers move
the tasks by the plan's thresholds, ``/why`` explains them, the badge leaves
out a task admitted at low confidence, the member's task is never asked with
the owner's key, and the user's confirm / not-done win."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from captain_claw import fd_client, jev_client, speaker, tool_audit, turn_register
from captain_claw.flight_deck import (
    auth,
    file_library,
    jev_routes,
    taskboard,
    taskboard_ingest,
    taskboard_sources,
    taskboard_triage,
)
from captain_claw.flight_deck import tool_audit_routes as tr
from captain_claw.flight_deck.tool_audit_store import ToolAuditStore
from captain_claw.tools.registry import ToolResult

from .test_taskboard_sources import HELPER_REF, env  # noqa: F401

TOKEN = "tok-helper"
THREAD = "18f0aa77c1d2e3f4"
CORRECTIVE = ("STOP: All your tool calls were blocked as duplicates. You already have the data you need "
              "to answer.")


class _Session:
    def __init__(self, sid: str, name: str) -> None:
        self.id, self.name, self.metadata = sid, name, {}


class _Tools:
    def __init__(self, saved: Path) -> None:
        self.saved = saved

    def get_saved_base_path(self, create: bool = False) -> Path:
        return self.saved


class AgentTurn:
    """What the turn register reads off an Agent once ``complete()`` has set
    the turn's origin (``_set_turn_origin``)."""

    def __init__(self, saved: Path, text: str, *, origin=("human", "", "web"), source=("web", "Web chat"),
                 session=("s-web", "Chat")) -> None:
        self._turn_origin, self._turn_source, self._turn_user_text = origin, source, text
        self.session = _Session(*session)
        self._last_complete_success = True
        self.tools = _Tools(saved)

    def _current_session_slug(self) -> str:
        return self.session.id


@pytest.fixture()
async def deck(env, tmp_path, monkeypatch):  # noqa: F811
    """The deck with sign-in (Alice owns the agent "helper", Bob is a member
    of it) serving the board, the turn register and the tool-call register;
    and the agent side, spawned by it, shipping to it over its own token."""
    from captain_claw.flight_deck import (
        agent_preset,
        agent_sharing,
        context_packs,
        server,
        taskboard_store,
    )

    # The first import of server points the board at the deck's real data
    # dir (taskboard.configure): back to this deck's, where its ledgers are.
    monkeypatch.setattr(taskboard_store, "_DATA_DIR", env.tmp)
    auth.set_auth_db(env.db)
    audit = ToolAuditStore(tmp_path / "tool_audit.db")
    tr.set_store(audit)
    tr._identities.clear()
    monkeypatch.setattr(taskboard, "_SYNC_GAP_S", 0.0)
    # The deck's own records of its agents: the token is helper's, Alice's.
    monkeypatch.setattr(server, "_agent_caller_ok", lambda request: True)
    monkeypatch.setattr(server, "_find_agent_by_auth",
                        lambda t: (True, env.alice, "helper") if t == TOKEN else (False, "", ""))
    monkeypatch.setattr(server, "_find_agent_endpoint_by_auth", lambda t: {
        "port": 1, "agent_ref": HELPER_REF, "name": "Helper"} if t == TOKEN else None)
    monkeypatch.setattr(server, "_load_process_registry", lambda: {"helper": {"owner": env.alice, "name": "Helper"}})
    monkeypatch.setattr(server, "_process_is_alive", lambda slug: False)   # asleep: never woken
    monkeypatch.setattr(server, "_agent_config_model", lambda slug: None)
    monkeypatch.setattr(agent_preset, "_agent_config", lambda slug: {})
    monkeypatch.setattr(context_packs, "agent_ref_for_auth", lambda t: HELPER_REF if t == TOKEN else "")

    async def member_check(db_, ref, owner, uid, **kw):
        return ref == HELPER_REF and owner == env.alice and uid == env.bob

    monkeypatch.setattr(agent_sharing, "member_check", member_check)
    # No model and no Jev until a test sets them up.
    model = SimpleNamespace(tier=None, calls=[], titles={})

    async def resolve_tier(user_id, prefer=("fast", "balanced")):
        return model.tier

    async def model_reply(tier, system, user, owner_id, max_tokens=0, request_timeout=None, images=None):
        model.calls.append(SimpleNamespace(system=system, prompt=user, owner=owner_id))
        title = next((t for k, t in model.titles.items() if k in user), "")
        out = {"title": title, "ask": "", "requester": "", "parties": [], "due_at": None,
               "expected_deliverable": "", "result_text": "", "next_steps": []}
        return file_library.ModelReply(text="```json\n" + json.dumps(out) + "\n```", tokens=42)

    monkeypatch.setattr(file_library, "_resolve_tier", resolve_tier)
    monkeypatch.setattr(file_library, "model_reply", model_reply)
    jev = SimpleNamespace(asks=[], answers={})

    class FakeJevAPI:
        def __init__(self, key: str) -> None:
            self.key = key

        async def ask(self, state, questions, model_name=None):
            body = jev_client.build_request(state, questions, model_name)   # what the real client checks
            jev.asks.append(SimpleNamespace(key=self.key, body=body))
            out = {}
            for name in body["questions"]:
                i, q = re.fullmatch(r"t(\d+)_(\w+)", name).groups()
                message = body["state"]["items"][int(i)]["quoted"]["message"]
                mine = next((a for k, a in jev.answers.items() if k in message), {})
                out[name] = mine.get(q) or _jev("noul", 0.5)
            return {"model": "jev-test", "answers": out, "usage": {}}

    monkeypatch.setattr(jev_routes, "_api", FakeJevAPI)
    monkeypatch.setattr(taskboard_triage, "_KICKED", set())
    monkeypatch.setattr(taskboard_triage, "_RUNNING", set())

    app = FastAPI()
    app.include_router(taskboard.router)
    app.include_router(taskboard_ingest.router)
    app.include_router(tr.router)
    users = {env.alice: {"id": env.alice, "email": "alice@x.co", "display_name": "Alice", "role": "user"},
             env.bob: {"id": env.bob, "email": "bob@x.co", "display_name": "Bob", "role": "user"}}
    current = {"user": users[env.alice]}
    app.dependency_overrides[auth.get_current_user] = lambda: current["user"]

    # The agent: spawned by this deck (FD_URL, its slug, its own token), its
    # registers spooled in tmp_path and shipped through FDClient to the app.
    monkeypatch.setenv("FD_URL", "http://deck")
    monkeypatch.setenv("FD_AGENT_SLUG", "helper")
    monkeypatch.delenv("FD_AGENT_SHARED_SECRET", raising=False)
    monkeypatch.setenv("CLAW_TOOL_AUDIT_SPOOL", str(tmp_path / "agent" / "tool_audit_spool.db"))
    monkeypatch.setenv("CLAW_TURN_REGISTER_SPOOL", str(tmp_path / "agent" / "turn_register_spool.db"))
    monkeypatch.delenv("CLAW_TOOL_AUDIT", raising=False)
    monkeypatch.delenv("CLAW_TURN_REGISTER", raising=False)
    monkeypatch.setattr(fd_client, "_agent_web_auth_token", lambda: TOKEN)
    for module in (tool_audit, turn_register):
        monkeypatch.setattr(module, "_SPOOL", None)
        monkeypatch.setattr(module, "_SPOOL_FAILED", False)
        monkeypatch.setattr(module, "_ensure_shipper", lambda: None)   # shipped by flush_now() here

    async def _http(self):
        if self._client is None:
            self._client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), timeout=self._timeout)
        return self._client

    monkeypatch.setattr(fd_client.FDClient, "_http", _http)
    saved = tmp_path / "agent" / "saved"
    saved.mkdir(parents=True)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://deck") as client:
        yield SimpleNamespace(client=client, env=env, board=env.board, alice=env.alice, bob=env.bob, audit=audit,
                              saved=saved, model=model, jev=jev, be=lambda uid: current.update(user=users[uid]))
    for module in (tool_audit, turn_register):
        if module._SPOOL is not None and module._SPOOL._conn is not None:
            module._SPOOL._conn.close()
    await audit.close()
    tr.set_store(None)
    auth.set_auth_db(None)  # type: ignore[arg-type]


def _jev(kind: str, value, p: float = 0.0) -> dict:
    if kind == "noul":
        return {"type": "noul", "noul": value}
    if kind == "choice":
        return {"type": "choice", "choice": value, "confidence": p, "probabilities": {value: p}}
    return {"type": "score", "score": value + 1, "legend": {"1": "", "2": "", "3": "", "4": ""}}


def _answers(is_task: float, owes: tuple[str, float], fulfilled: float, urgency: int = 0,
             result: tuple[str, float] = ("not_attempted", 0.6)) -> dict:
    return {"is_task": _jev("noul", is_task), "owes_next": _jev("choice", *owes),
            "fulfilled": _jev("noul", fulfilled), "result": _jev("choice", *result),
            "urgency": _jev("score", urgency)}


async def _turn(deck, text: str, reply: str, *, calls=(), member: bool = False, saves: str = "", **kw) -> str:
    """One turn as the agent runs it: opened, its tool calls made in it (each
    ``(tool, params, outcome, result text)``), closed — then both registers
    shipped to the deck. The turn's id."""
    agent = AgentTurn(deck.saved, text, **kw)
    bound = speaker.bind(speaker.Principal(speaker_id=deck.bob, display_name="Bob", owner_name="Alice", lane="A",
                                           agent_ref=HELPER_REF)) if member else None
    try:
        turn = turn_register.open_turn(agent)
        assert turn is not None, "the turn register is off"
        for tool, params, outcome, content in calls:
            handle = tool_audit.start(tool, params, session_id=agent.session.id, member=member)
            assert handle is not None
            tool_audit.finish(handle, outcome, "" if outcome == "ok" else content,
                              result=ToolResult(success=outcome == "ok", content=content))
        if saves:
            path = deck.saved / "output" / agent.session.id / saves
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Weekly report")
        turn_register.close_turn(turn, turn_register.outcome_of(agent, reply), reply, agent=agent)
    finally:
        if bound is not None:
            speaker.reset(bound)
    assert turn_register.current_turn_id.get() == ""
    await turn_register.flush_now()
    await tool_audit.flush_now()
    assert not turn_register._spool().pending() and not tool_audit._spool().pending(), "the deck refused a batch"
    return turn.turn_id


async def _get(deck, path: str, status: int = 200, **params):
    r = await deck.client.get(path, params=params)
    assert r.status_code == status, r.text
    return r.json()


async def _act(deck, task_id: str, action: str) -> dict:
    r = await deck.client.post(f"/fd/taskboard/tasks/{task_id}/action", json={"action": action})
    assert r.status_code == 200, r.text
    return r.json()["task"]


def _task_of(deck, uid: str, turn_id: str) -> dict | None:
    task_id = deck.board.task_for_key(uid, f"turn:{turn_id}")
    return deck.board.get_task(uid, task_id) if task_id else None


async def _sweep_and_triage(deck) -> dict:
    import asyncio

    await asyncio.to_thread(taskboard_sources.run_sweep, deck.board)
    return {uid: await taskboard_triage.triage_user(uid) for uid in (deck.alice, deck.bob)}


async def test_v1_end_to_end(deck):
    # ── Rules alone: no Jev key, no model ─────────────────────────────
    chat = await _turn(deck, "What's on my calendar tomorrow? Also mail Ana the Q3 invoice.",
                       "Two meetings: 10:00 stand-up, 14:00 Ana. I mailed her the invoice.", calls=[
                           ("google_calendar", {"action": "list_events"}, "ok", "2 events"),
                           ("google_mail", {"action": "send", "to": "ana@x.co", "subject": "Q3 invoice"}, "ok",
                            f"Email sent.\n  To: ana@x.co\n  Message ID: 18f0aa77c1d2e3f5\n  Thread ID: {THREAD}"),
                       ])
    shared = await _turn(deck, "Draft my weekly report", "Here is your weekly report.", member=True,
                         source=("member", "Shared-agent members"), session=("s-bob", "Bob's chat"),
                         calls=[("web_search", {"query": "weekly report template"}, "ok", "3 results")],
                         saves="weekly.md")
    # The deck's own record of that send (/fd/google/gmail/send): the same
    # email, not a second task.
    await deck.env.db.add_gmail_send(deck.alice, agent="helper", to_addrs="ana@x.co", subject="Q3 invoice",
                                     gmail_message_id="18f0aa77c1d2e3f5", thread_id=THREAD)
    corrective = await _turn(deck, CORRECTIVE, "Here is the answer.")
    nudge = await _turn(deck, "[Autonomous nudge] Check whether Ana replied.", "She hasn't yet.",
                        origin=("autonomy", "nudge", "web"))

    # The deck kept every turn under the agent its token names, and the tool
    # calls under the turn they were made in.
    for tid in (chat, shared, corrective, nudge):
        turn = deck.board.get_turn(tid)
        assert (turn["agent_slug"], turn["agent_owner_id"]) == ("helper", deck.alice)
        assert turn["outcome"] == "ok" and turn["ended_at"] is not None
    assert deck.board.get_turn(chat)["gate"] == "task" and deck.board.get_turn(chat)["tool_calls"] == 2
    assert deck.board.get_turn(corrective)["gate"] == "skipped:literal:corrective"
    assert deck.board.get_turn(nudge)["gate"].startswith("skipped:")
    assert deck.board.get_turn(shared)["user_id"] == deck.bob and deck.board.get_turn(shared)["private"]
    assert deck.board.get_turn(shared)["deliverables"] == ["output/s-bob/weekly.md"]
    calls = {r["tool"]: r for r in await deck.audit.list({}, limit=10)}
    assert calls["google_mail"]["turn_id"] == chat and calls["web_search"]["turn_id"] == shared
    assert json.loads(calls["google_mail"]["refs"]) == {"gmail_message_id": "18f0aa77c1d2e3f5",
                                                        "gmail_thread_id": THREAD}
    assert calls["web_search"]["member"] == 1 and calls["web_search"]["speaker_id"] == deck.bob
    assert (await deck.audit.verify())["ok"]
    assert {deck.alice, deck.bob} <= taskboard_triage._KICKED

    triaged = await _sweep_and_triage(deck)
    assert triaged[deck.alice]["rules"] == 1 and triaged[deck.alice]["jev"] == 0
    assert "No model is set up" in triaged[deck.alice]["message"]

    # The owner's chat question: done, answered in chat, the send recorded.
    task = _task_of(deck, deck.alice, chat)
    assert (task["kind"], task["channel"], task["status"], task["closed_by"]) == ("request", "web", "done", "rule")
    assert task["result_text"] == "Answered in chat" and task["evidence"] == "verified"
    flags = json.loads(task["flags"])
    assert flags["answered_in_chat"] is True and "contains_instructions" not in flags
    detail = await _get(deck, f"/fd/taskboard/tasks/{task['id']}")
    timeline = [(e["kind"], e["summary"], e["evidence"]) for e in detail["events"]]
    assert timeline[0][0] == "received" and [k for k, _, _ in timeline].count("reply") == 1
    assert ("tool_call", "google_calendar list_events: ok", "") in timeline
    assert ("sent", "google_mail send: ok", "recorded") in timeline
    assert ("sent", "Email sent to ana@x.co: Q3 invoice", "recorded") in timeline
    tool_event = next(e for e in detail["events"] if e["kind"] == "tool_call")
    assert tool_event["refs"]["tool"] == "google_calendar" and "params" not in tool_event["refs"]
    assert {"turn:" + chat, "gmail:thread:" + THREAD} <= {link["key"] for link in detail["links"]}
    why = await _get(deck, f"/fd/taskboard/tasks/{task['id']}/why")
    assert why["items"][0]["judge"] == "rule" and "answered" in why["summary"].lower()
    assert task["id"] not in {i["id"] for i in (await _get(deck, "/fd/taskboard/tasks", view="needs_me"))["items"]}

    # The member's turn: on Bob's board only, with its tool call.
    assert _task_of(deck, deck.alice, shared) is None
    bobs = _task_of(deck, deck.bob, shared)
    assert bobs is not None and bobs["status"] == "done" and bobs["private"]
    await _get(deck, f"/fd/taskboard/tasks/{bobs['id']}", status=404)
    await _get(deck, f"/fd/taskboard/tasks/{bobs['id']}/why", status=404)
    assert bobs["id"] not in {i["id"] for i in (await _get(deck, "/fd/taskboard/tasks", view="all"))["items"]}
    deck.be(deck.bob)
    mine = await _get(deck, f"/fd/taskboard/tasks/{bobs['id']}")
    assert [e["summary"] for e in mine["events"] if e["kind"] == "tool_call"] == ["web_search: ok"]
    reply = next(e for e in mine["events"] if e["kind"] == "reply")
    assert reply["refs"]["deliverables"] == ["output/s-bob/weekly.md"]
    assert {i["id"] for i in (await _get(deck, "/fd/taskboard/tasks", view="all"))["items"]} == {bobs["id"]}
    deck.be(deck.alice)

    # Code-made openings made nothing, on either board.
    for tid in (corrective, nudge):
        assert _task_of(deck, deck.alice, tid) is None and _task_of(deck, deck.bob, tid) is None
    assert (await _get(deck, "/fd/taskboard/tasks", view="all"))["total"] == 1

    # ── Jev (Alice's own key) and the agent's model (a tier, the agent asleep) ──
    await jev_routes._write(deck.alice, {jev_routes._K_KEY: "jk-alice"})
    deck.model.tier = {"provider": "fake", "model": "tier-model"}
    deck.model.titles = {"Split": "Book a flight to Split", "report to Ana": "Send Ana the report"}
    deck.jev.answers = {
        "Split": _answers(0.92, ("me", 0.8), 0.3, urgency=2),
        "thanks": _answers(0.2, ("nobody", 0.9), 0.9),
        "garage": _answers(0.5, ("agent", 0.6), 0.1),
        "report to Ana": _answers(0.72, ("agent", 0.7), 0.7),
    }
    flight = await _turn(deck, "Find me a flight to Split next Friday", "Three options — which one do you want?")
    thanks = await _turn(deck, "thanks!", "You're welcome.")
    garage = await _turn(deck, "maybe check the garage door sensor sometime", "Noted.")
    report = await _turn(deck, "Send the report to Ana", "I've drafted it.")
    bob_again = await _turn(deck, "Summarise my notes", "Summary: …", member=True,
                            source=("member", "Shared-agent members"), session=("s-bob", "Bob's chat"))
    triaged = await _sweep_and_triage(deck)
    assert triaged[deck.alice]["jev"] == 4 and triaged[deck.alice]["extracted"] == 4
    assert triaged[deck.bob]["jev"] == 0 and triaged[deck.bob]["rules"] == 1   # Bob has no Jev key

    # One batched ask, with Alice's own key; the member's task never reached it.
    assert len(deck.jev.asks) == 1 and {a.key for a in deck.jev.asks} == {"jk-alice"}
    assert "Summarise my notes" not in json.dumps([a.body for a in deck.jev.asks])
    assert "Find me a flight to Split" in json.dumps(deck.jev.asks[0].body["state"]["items"])
    # Extraction: the board user's allowance pays for their own tasks; the text fenced.
    assert {c.owner for c in deck.model.calls if "Summarise my notes" in c.prompt} == {deck.bob}
    assert all(re.search(r"<<<DATA (\w+)>>>.*<<<END \1>>>", c.prompt, re.S) for c in deck.model.calls)

    task = _task_of(deck, deck.alice, flight)
    assert (task["title"], task["status"], task["owes_next"], task["urgency"]) == \
        ("Book a flight to Split", "open", "me", 2)
    why = await _get(deck, f"/fd/taskboard/tasks/{task['id']}/why")
    latest = why["items"][0]
    assert (latest["judge"], latest["qset"]) == ("jev", "taskboard.v1")
    assert latest["answers"]["is_task"] == {"p": 0.92}
    assert latest["answers"]["owes_next"] == {"choice": "me", "p": 0.8}
    assert latest["answers"]["urgency"] == {"score": 2.0}
    assert "next move is yours (80%)" in latest["why"] and "It waits on you." in why["summary"]
    assert "model" in {i["judge"] for i in why["items"]}
    assert _task_of(deck, deck.alice, thanks)["status"] == "not_task"
    maybe = _task_of(deck, deck.alice, garage)
    assert maybe["status"] == "maybe"
    low = _task_of(deck, deck.alice, report)
    low_flags = json.loads(low["flags"])
    assert (low["status"], low_flags["low_confidence"], low_flags["probably_done"]) == ("open", True, True)
    assert _task_of(deck, deck.bob, bob_again)["status"] == "done"

    # Needs me holds the flight and the probably-done report; the badge leaves
    # out the one admitted at low confidence.
    needs = await _get(deck, "/fd/taskboard/tasks", view="needs_me")
    assert {i["id"] for i in needs["items"]} == {task["id"], low["id"]}
    assert (await _get(deck, "/fd/taskboard/summary")) == {"needs_me": 1}
    facets = (await _get(deck, "/fd/taskboard/tasks", view="all"))["facets"]["status"]
    assert {f["id"]: f["count"] for f in facets}.get("maybe") == 1

    settings = await _get(deck, "/fd/taskboard/settings")
    assert settings["jev"] == {"connected": True, "source": "own"}
    assert (settings["counters"]["jev"], settings["counters"]["extract"]) == (4, 4)
    deck.be(deck.bob)
    assert (await _get(deck, "/fd/taskboard/settings"))["jev"] == {"connected": False, "source": ""}
    deck.be(deck.alice)

    # The user's word: a maybe task confirmed, the report not done after all.
    confirmed = await _act(deck, maybe["id"], "confirm_task")
    assert (confirmed["status"], confirmed["owes_next"]) == ("open", "agent")
    assert "status" in confirmed["user_locks"]
    not_done = await _act(deck, low["id"], "not_done")
    assert not_done["status"] == "open" and not not_done["flags"].get("probably_done")
    assert (await _get(deck, f"/fd/taskboard/tasks/{low['id']}/why"))["items"][0]["judge"] == "user"
    # A later Jev pass that thinks it is done again leaves the user's word standing.
    deck.board.update_fields(deck.alice, low["id"], {"triage_state": "pending", "triage_after": time.time() - 1})
    deck.board.ingest(deck.alice, keys=[f"turn:{report}"], patch=None, event={
        "source": "test", "source_ref": "nudge-1", "kind": "note", "summary": "A new note on it"})
    await taskboard_triage.triage_user(deck.alice)
    assert not json.loads(deck.board.get_task(deck.alice, low["id"])["flags"]).get("probably_done")
