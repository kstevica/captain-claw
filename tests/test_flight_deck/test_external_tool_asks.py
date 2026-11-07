# ruff: noqa: F811 — `data`, `deck` and `as_agent` are fixtures imported from the other Custom tools tests
"""Custom tools, Phase 7d: a tool asks the person running it and waits. The
question opens in their Flight Deck (and the bell); the tool's time stops while
it waits; "Don't answer", no answer in time and nobody to ask each end it; the
agent follows a long run instead of holding one request open."""

from __future__ import annotations

import asyncio
import json
import sys

import httpx
import pytest
from fastapi import FastAPI

from captain_claw import custom_tools_client
from captain_claw.flight_deck import ask_routes
from captain_claw.flight_deck import external_tools as xt
from captain_claw.flight_deck import external_tools_routes as xr
from captain_claw.flight_deck import person_asks as asks
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.tools.custom_tools import CustomToolsTool

from .test_external_tools import LOOPBACK, _commit, _manifest, _switch_on, data, deck  # noqa: F401
from .test_external_tools_agent import _agent_client, as_agent  # noqa: F401

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="the line is POSIX only")

ASKS = """import json, sys, time, captain
d = json.load(sys.stdin)
try:
    got = captain.ask("Which account?", choices=["Business", "Personal"], wait_s=d.get("wait", 600))
except captain.Declined:
    got = "declined"
except captain.NoAnswer as exc:
    got = "no answer: " + str(exc)
time.sleep(d.get("after", 0))
print(json.dumps({"got": got}))
"""
RECORDED = [{"input": {}, "expect": {"got": "Business"},
             "calls": [{"ask": ["answer"], "reply": {"answers": {"answer": "Business"}}}]}]


def _client(user: dict) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(ask_routes.agent_router)
    app.include_router(ask_routes.router)
    app.include_router(xr.catalog_router)
    app.include_router(xr.router)
    app.dependency_overrides[get_current_user] = lambda: user
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=LOOPBACK), base_url="http://deck.test")


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    asks.reset_for_tests()
    xr._FOLLOWING.clear()
    monkeypatch.setattr(asks, "MIN_WAIT_S", 0.3)
    yield
    asks.reset_for_tests()


async def _publish(uid: str, slug: str = "pick", main_py: str = ASKS, **manifest) -> None:
    await xt.create_tool(uid, slug, title="Pick", description="A test tool.")
    _commit(uid, slug, {"main.py": main_py, "tool.json": _manifest(slug, **{"asks": True, "examples": RECORDED, **manifest}),
                        "tests/test_main.py": None})
    state = await xt.publish(uid, slug, by=uid)
    assert state["state"] == "passed", state["steps"]


async def _next_question(c: httpx.AsyncClient, known: str = "") -> dict:
    for _ in range(50):
        found = (await c.get("/fd/asks", params={"wait": 1, "known": known})).json()["asks"]
        if found:
            return found[0]
    raise AssertionError("no question opened")


def test_questions_and_answers_are_checked():
    q = asks.check_questions([
        {"id": "account", "question": "Which account?", "choices": ["Business", "Personal", "Business"]},
        {"id": "months", "question": "Which months?", "choices": ["Jan", "Feb", "Mar"], "multiple": True, "max": 2},
        {"id": "send", "question": "Send it?", "confirm": True},
        {"id": "note", "question": "Anything else?", "lines": 3, "optional": True},
    ])
    assert [x["kind"] for x in q] == ["choice", "multiple", "confirm", "text"] and q[0]["choices"] == ["Business", "Personal"]
    ok = asks.check_answers(q, {"account": "Business", "months": ["Feb", "Jan"], "send": False})
    assert ok == {"account": "Business", "months": ["Jan", "Feb"], "send": False, "note": ""}
    for bad, words in [({"account": "Other", "months": ["Jan"], "send": True}, "choices"),
                       ({"account": "Business", "months": ["Jan", "Feb", "Mar"], "send": True}, "1–2"),
                       ({"account": "Business", "months": ["Jan"], "send": "yes"}, "yes or no"),
                       ({"months": ["Jan"], "send": True}, "needs an answer")]:
        with pytest.raises(asks.AskError, match=words):
            asks.check_answers(q, bad)
    for bad in ([], [{"question": "x"}] * 2, [{"id": "a", "question": ""}], [{"id": "a", "question": "x", "choices": []}],
                [{"id": "a", "question": "x", "lines": 99}], [{"id": "a", "question": "x", "choices": ["y"], "default": "z"}]):
        with pytest.raises(asks.AskError):
            asks.check_questions(bad)
    assert asks.attended("web") and asks.attended("telegram") and not asks.attended("scheduled")
    assert not asks.attended("autonomy:plan")


async def test_the_person_answers_in_flight_deck_and_the_tools_time_stops(deck):
    await _switch_on(deck)
    uid = deck.user["id"]
    await _publish(uid, timeout_s=2)
    async with _client(deck.user) as c:
        run = asyncio.create_task(c.post("/fd/external-tools/pick/run", json={"input": {}}))
        q = await _next_question(c)
        assert q["title"] == "Pick" and q["questions"][0]["choices"] == ["Business", "Personal"]
        bells = [n for n in await deck.db.list_notifications(uid) if n.get("ref_type") == asks.BELL_REF]
        assert bells and "Pick asks you something" in bells[0]["title"] and q["source"] == "tool"
        # Longer than the tool's 2 s: its time doesn't run while it waits.
        await asyncio.sleep(2.5)
        async with _client(deck.other) as other:
            assert (await other.post(f"/fd/asks/{q['id']}", json={"answers": {"answer": "x"}})).status_code == 404
        bad = await c.post(f"/fd/asks/{q['id']}", json={"answers": {"answer": "Neither"}})
        assert bad.status_code == 422
        assert (await c.post(f"/fd/asks/{q['id']}", json={"answers": {"answer": "Personal"}})).status_code == 200
        res = await run
    assert res.status_code == 200, res.text
    assert res.json()["output"] == {"got": "Personal"}
    assert asks.open_for(uid) == []
    settled = [n for n in await deck.db.list_notifications(uid) if n.get("ref_type") == asks.BELL_REF]
    assert all(n.get("resolved") for n in settled)


async def test_dont_answer_and_no_answer_in_time(deck):
    await _switch_on(deck)
    await _publish(deck.user["id"])
    async with _client(deck.user) as c:
        run = asyncio.create_task(c.post("/fd/external-tools/pick/run", json={"input": {}}))
        q = await _next_question(c)
        await c.post(f"/fd/asks/{q['id']}", json={"declined": True})
        assert (await run).json()["output"] == {"got": "declined"}
        timed = await c.post("/fd/external-tools/pick/run", json={"input": {"wait": 0.3}})
        assert timed.json()["output"] == {"got": "no answer: No answer in time."}


async def test_the_wait_returns_when_the_open_questions_change(deck):
    await _switch_on(deck)
    await _publish(deck.user["id"])
    async with _client(deck.user) as c:
        idle = await c.get("/fd/asks", params={"wait": 0.2})
        assert idle.json() == {"asks": []}
        waiting = asyncio.create_task(c.get("/fd/asks", params={"wait": 20}))
        run = asyncio.create_task(c.post("/fd/external-tools/pick/run", json={"input": {}}))
        opened = (await asyncio.wait_for(waiting, 15)).json()["asks"]
        assert len(opened) == 1
        # A tab that already shows it waits until it changes: here, it's answered elsewhere.
        again = asyncio.create_task(c.get("/fd/asks", params={"wait": 20, "known": opened[0]["id"]}))
        await asyncio.sleep(0.2)
        assert not again.done()
        await c.post(f"/fd/asks/{opened[0]['id']}", json={"answers": {"answer": "Business"}})
        assert (await asyncio.wait_for(again, 5)).json() == {"asks": []}
        await run


async def test_only_a_tool_that_says_so_asks_and_only_five_times(data):
    quiet = """import json, captain
try:
    captain.ask("Which?", confirm=True)
    print(json.dumps({"r": "asked"}))
except captain.Refused as exc:
    print(json.dumps({"r": str(exc)}))
"""
    refused = 'Asking the person needs "asks": true in tool.json.'
    await _publish("u1", "quiet", quiet, asks=False, uses=[{"tool": "write", "why": "x"}],
                   examples=[{"input": {}, "expect": {"r": refused}}])
    many = """import json, captain
n = 0
for _ in range(7):
    try:
        captain.ask("Again?", confirm=True)
        n += 1
    except captain.Refused:
        break
print(json.dumps({"n": n}))
"""
    recorded = [{"input": {}, "expect": {"n": 5}, "calls": [{"ask": ["answer"], "reply": {"answers": {"answer": True}}}] * 5}]
    await _publish("u1", "many", many, examples=recorded)


async def test_nobody_to_ask_on_a_scheduled_run(deck, as_agent):
    await _switch_on(deck)
    uid = deck.user["id"]
    as_agent["owner"] = uid
    await _publish(uid)
    async with _agent_client() as c:
        res = await c.post("/fd/external-tools/agent/run", json={"tool": "pick", "input": {}, "origin": "scheduled"})
    assert res.json()["output"]["got"].startswith("no answer: No one to ask")
    assert asks.open_for(uid) == []


async def test_the_agent_follows_a_run_that_waits_and_can_stop_it(deck, as_agent, monkeypatch):
    await _switch_on(deck)
    uid = deck.user["id"]
    as_agent["owner"] = uid
    await _publish(uid)

    async def call(method, path, body=None):
        async with _agent_client() as c:
            resp = await c.request(method, f"/fd/external-tools/agent/{path}", json=body)
        return resp.status_code, resp.json()

    monkeypatch.setattr(custom_tools_client, "_fd_base", lambda: "http://deck.test")
    monkeypatch.setattr(custom_tools_client, "call", call)
    said: list[str] = []
    tool = asyncio.create_task(CustomToolsTool()._run("pick", {}, stream=said.append))
    async with _client(deck.user) as c:
        q = await _next_question(c)
        await asyncio.sleep(0.5)
        await c.post(f"/fd/asks/{q['id']}", json={"answers": {"answer": "Business"}})
    result = await asyncio.wait_for(tool, 30)
    assert result.success and '"got": "Business"' in result.content
    # Stopping the turn stops the run and closes its question.
    tool = asyncio.create_task(CustomToolsTool()._run("pick", {}, stream=said.append))
    async with _client(deck.user) as c:
        await _next_question(c)
    await asyncio.sleep(0.3)
    tool.cancel()
    with pytest.raises(asyncio.CancelledError):
        await tool
    for _ in range(50):
        if not asks.open_for(uid):
            break
        await asyncio.sleep(0.1)
    assert asks.open_for(uid) == [] and xr._FOLLOWING == {}


async def test_publishing_answers_from_recorded_answers(data):
    await _publish("u1")
    declined = [{"input": {}, "expect": {"got": "declined"}, "calls": [{"ask": ["answer"], "reply": {"declined": True}}]}]
    _commit("u1", "pick", {"tool.json": _manifest("pick", asks=True, examples=declined)})
    assert (await xt.publish("u1", "pick", by="u1"))["state"] == "passed"
    declares = xr.declares_of(xt.manifest_of("u1", "pick"))
    assert declares["asks"] is True
    assert "to ask you questions while it runs" in xr.declares_more({**declares, "asks": False}, declares)
    assert "captain.ask" in xt.WORKER_CONTRACT and json.dumps(xt._HELPER.read_text()).count("def ask(") == 1


GREET = """import json, captain
try:
    name = captain.ask("What is your name?")
except (captain.Declined, captain.NoAnswer):
    name = None
print(json.dumps({"greeting": f"Hello, {(name or '').strip() or 'stranger'}!"}))
"""


@pytest.mark.parametrize("recorded, words", [
    # Keyed by the question's text (what a Crew worker wrote on the Pi), not its id.
    ({"ask": ["answer"], "reply": {"answers": {"What is your name?": "Alice"}}}, "aren't question ids"),
    ({"ask": ["What is your name?"], "reply": {"answers": {"answer": "Alice"}}}, "list the questions' ids"),
    ({"ask": ["answer"], "reply": {"answers": {"answer": None}}}, 'record {"no_answer": true}'),
    ({"ask": ["answer"], "reply": {"name": "Alice"}}, 'needs "answers"'),
])
async def test_a_recorded_answer_the_dialog_couldnt_give_fails_and_says_why(data, recorded, words):
    await xt.create_tool("u1", "greet", title="Greet", description="A test tool.")
    example = {"input": {}, "expect": {"greeting": "Hello, Alice!"}, "calls": [recorded]}
    _commit("u1", "greet", {"main.py": GREET, "tool.json": _manifest("greet", asks=True, examples=[example]),
                            "tests/test_main.py": None})
    state = await xt.publish("u1", "greet", by="u1")
    assert state["state"] == "failed"
    assert words in state["steps"][-1]["detail"], state["steps"][-1]["detail"]


async def test_no_answer_is_recorded_as_such(data):
    examples = [{"input": {}, "expect": {"greeting": "Hello, Alice!"},
                 "calls": [{"ask": ["answer"], "reply": {"answers": {"answer": "Alice"}}}]},
                {"input": {}, "expect": {"greeting": "Hello, stranger!"},
                 "calls": [{"ask": ["answer"], "reply": {"no_answer": True}}]}]
    await _publish("u1", "greet", GREET, examples=examples)


def test_the_helper_checks_recorded_answers_in_tests_too(tmp_path):
    def run(calls):
        fake = tmp_path / "calls.json"
        fake.write_text(json.dumps(calls))
        tool = tmp_path / "tool"
        tool.mkdir(exist_ok=True)
        (tool / "captain.py").write_text(xt._HELPER.read_text())
        (tool / "main.py").write_text(GREET.replace("except (captain.Declined, captain.NoAnswer):",
                                                    "except (captain.Declined, captain.NoAnswer):  # noqa"))
        import subprocess
        return subprocess.run([sys.executable, str(tool / "main.py")], input="{}", capture_output=True, text=True,
                              env={"PATH": "/usr/bin:/bin", "XT_FAKE_CALLS": str(fake)}, cwd=tool)

    good = run([{"ask": ["answer"], "reply": {"answers": {"answer": "Alice"}}}])
    assert json.loads(good.stdout) == {"greeting": "Hello, Alice!"}
    for calls, words in [([{"ask": ["answer"], "reply": {"answers": {"What is your name?": "Alice"}}}], "aren't question ids"),
                         ([{"ask": ["What is your name?"], "reply": {"answers": {"answer": "Alice"}}}], "list the questions' ids"),
                         ([{"ask": ["answer"], "reply": {"answers": {"answer": None}}}], "no_answer")]:
        bad = run(calls)
        assert bad.returncode != 0 and words in bad.stderr, bad.stderr


def test_a_tools_tests_dont_pick_up_the_decks_pytest_settings(tmp_path):
    assert xt._pin_pytest_root(tmp_path) is True
    assert "[pytest]" in (tmp_path / "pytest.ini").read_text()
    own = tmp_path / "own"
    own.mkdir()
    (own / "pyproject.toml").write_text("[tool.pytest.ini_options]\naddopts = '-q'\n")
    assert xt._pin_pytest_root(own) is False and not (own / "pytest.ini").exists()
    other = tmp_path / "other"
    other.mkdir()
    (other / "pyproject.toml").write_text("[project]\nname = 'x'\n")  # no pytest table: still pinned
    assert xt._pin_pytest_root(other) is True


async def test_a_published_version_has_its_own_pytest_root(data):
    state = await _publish("u1", "greet", GREET, examples=[{"input": {}, "expect": {"greeting": "Hello, Alice!"},
                                                             "calls": [{"ask": ["answer"],
                                                                        "reply": {"answers": {"answer": "Alice"}}}]}])
    assert state is None
    live = xt.read_live("u1", "greet")["version"]
    assert (xt.versions_dir("u1", "greet") / live / "pytest.ini").is_file()
