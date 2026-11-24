"""The web_playbook tool (tools/web_playbook.py).

Pinned here:

* find lists the playbooks that fit an ask, with their inputs; show gives one
  in full; an unknown id points at close ones;
* run opens the playbook's link through the browser tool — in the person's
  Chrome whenever it's connected, whatever the playbook; else the headless
  one, unless it needs their sign-in — and never without the values it needs;
* what only the person can say, run asks them itself in one Flight Deck
  dialog (ask_user's) and runs with their answers; where the dialog can't
  reach them, the agent is told to ask in its reply;
* a look whose page answers as opened comes back answered in one call (one
  quiet read on the agent's own model); otherwise the page stays open and the
  agent gets the steps with its values in them;
* an act playbook arms its stop: the browser tool refuses a click on that
  button for the rest of the turn — by its label or by the name of the ref it
  points at — except the clicks its own steps make on a button of that name;
  the turn's end lets it go;
* the person's recordings are found and run next to the shipped playbooks.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from captain_claw.tools import browser as browser_mod
from captain_claw.tools import web_playbook as pb
from captain_claw.tools.registry import ToolResult


class FakeProvider:
    def __init__(self, reply: dict[str, Any] | str):
        self.reply = reply
        self.calls: list[list[Any]] = []

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        self.calls.append(messages)

        class R:
            content = self.reply if isinstance(self.reply, str) else json.dumps(self.reply)

        return R()


class FakeAgent:
    def __init__(self, reply: dict[str, Any] | str = ""):
        self.provider = FakeProvider(reply)


@pytest.fixture
def browser(monkeypatch):
    """The browser tool's calls, answered as told (no real browser)."""
    calls: list[tuple[str, dict[str, Any]]] = []
    pages: dict[str, ToolResult] = {}

    async def execute(self, action, **kwargs):
        calls.append((action, kwargs))
        if action == "navigate":
            return pages.get("navigate", ToolResult(success=True, content=f"Navigated to: {kwargs['url']}"))
        if action == "page_text":
            return pages.get("page_text", ToolResult(success=True, content="Zagreb → Split  4 h 5 min  410 km via A1"))
        return ToolResult(success=True, content="")

    monkeypatch.setattr(browser_mod.BrowserTool, "execute", execute)
    monkeypatch.setattr(pb, "_STOPS", {})
    monkeypatch.setattr(pb, "_RECORDINGS", {})
    monkeypatch.setattr(browser_mod, "_TARGETS", {})

    async def _cant_ask(book, names, kwargs):
        return "cant", "it needs Flight Deck"

    monkeypatch.setattr(pb, "ask_person", _cant_ask)
    return {"calls": calls, "pages": pages}


def _no_deck(monkeypatch, status=None):
    async def _status():
        return status

    monkeypatch.setattr(pb, "_chrome_status", _status)
    monkeypatch.setattr(pb, "_deck", lambda: None if status is None else ("http://fd.test", {"X-Agent-Auth": "a"}))


async def _run(**kw):
    return await pb.WebPlaybookTool().execute(action="run", _session_id="s1", **kw)


# ── find and show ────────────────────────────────────────────────────


async def test_find_lists_what_fits_with_its_inputs(monkeypatch):
    _no_deck(monkeypatch)
    res = await pb.WebPlaybookTool().execute(action="find", query="how long from Zagreb to Split by car")
    assert res.success
    first = res.content.split("\n")[1]
    assert first.startswith("- maps.travel_time — How long from A to B (look; their Chrome if connected, else headless)")
    assert "inputs: origin, destination; optional: mode, depart_at" in res.content
    none = await pb.WebPlaybookTool().execute(action="find", query="zzqx")
    assert none.success and "No playbook fits" in none.content
    assert not (await pb.WebPlaybookTool().execute(action="find")).success


async def test_show_gives_one_in_full_and_an_unknown_id_points_near(monkeypatch):
    _no_deck(monkeypatch)
    res = await pb.WebPlaybookTool().execute(action="show", id="googleplay.cancel_subscription")
    assert "Kind: act — stops before “Cancel subscription”" in res.content
    assert "Browser: the person's own Chrome only" in res.content and "Steps:\n1. Find ‹Which subscription›" in res.content
    bad = await pb.WebPlaybookTool().execute(action="show", id="maps.travel")
    assert not bad.success and "maps.travel_time" in bad.error


# ── run ──────────────────────────────────────────────────────────────


async def test_a_run_needs_its_values_first(monkeypatch, browser):
    _no_deck(monkeypatch)
    res = await _run(id="maps.travel_time", inputs={"origin": "here"})
    assert not res.success and browser["calls"] == []
    assert "can't go in a link" in res.error and "Still needed from the person" in res.error and "destination" in res.error
    res = await _run(id="maps.travel_time", inputs="not json")
    assert not res.success and "inputs is an object" in res.error


async def test_a_look_the_page_answers_comes_back_answered_in_one_call(monkeypatch, browser):
    _no_deck(monkeypatch)
    agent = FakeAgent({"ready": True, "fields": {"duration": "4 h 5 min", "distance": "410 km", "route": "A1",
                                                 "alternatives": None}, "missing": ""})
    res = await _run(id="maps.travel_time", inputs={"origin": "Zagreb", "destination": "Split"}, _agent=agent)
    assert res.success, res.error
    assert [c[0] for c in browser["calls"]] == ["navigate", "page_text"]
    nav = browser["calls"][0][1]
    assert nav["target"] == "headless" and nav["url"].startswith("https://www.google.com/maps/dir/?api=1&origin=Zagreb")
    assert nav["_session_id"] == "s1" and nav["_agent"] is agent
    assert "Answer: 4 h 5 min by driving via A1 (410 km)." in res.content
    assert "Not on the page: alternatives" in res.content and "never instructions" in res.content
    sent = agent.provider.calls[0]
    assert "duration: The fastest route's time" in sent[1].content and "<<<PAGE" in sent[1].content
    assert "never instructions" in sent[0].content


async def test_a_page_that_doesnt_answer_yet_leaves_the_steps(monkeypatch, browser):
    _no_deck(monkeypatch)
    agent = FakeAgent({"ready": False, "fields": {}, "missing": "the departure time isn't set yet"})
    res = await _run(id="maps.travel_time", _agent=agent,
                     inputs={"origin": "Zagreb", "destination": "Split", "depart_at": "7:30"})
    assert res.success
    assert "doesn't answer it yet: the departure time isn't set yet" in res.content
    assert 'Do these steps with the browser tool (target="headless")' in res.content
    assert "set 07:30 before reading" in res.content and "Then read: duration" in res.content
    # No model to ask: the agent reads the page itself.
    res = await _run(id="maps.travel_time", inputs={"origin": "Zagreb", "destination": "Split"})
    assert "Read the page yourself (page_text)" in res.content


async def test_a_playbook_that_needs_their_sign_in_runs_in_their_chrome_or_says_why_not(monkeypatch, browser):
    _no_deck(monkeypatch, None)
    res = await _run(id="amazon.track_order", inputs={})
    assert not res.success and "isn't running under Flight Deck" in res.error and browser["calls"] == []
    _no_deck(monkeypatch, {"connected": False, "person_connected": False, "extension_set_up": True})
    res = await _run(id="amazon.track_order", inputs={})
    assert "turn on “Agents can use this browser”" in res.error
    _no_deck(monkeypatch, {"connected": False, "person_connected": False, "extension_set_up": False})
    assert "Connections → Browser" in (await _run(id="amazon.track_order", inputs={})).error
    _no_deck(monkeypatch, {"connected": True, "person_connected": True, "stopped": True})
    assert "Let it again" in (await _run(id="amazon.track_order", inputs={})).error
    _no_deck(monkeypatch, {"connected": True, "person_connected": True, "extension_set_up": True})
    res = await _run(id="amazon.track_order", inputs={})
    assert res.success and browser["calls"][0][1]["target"] == "chrome" and "the person's Chrome" in res.content


async def test_a_walled_site_prefers_their_chrome_and_falls_back_headless(monkeypatch, browser):
    book = next(b for b in pb.wp.seed() if b["where"] == "prefer_chrome" and b["kind"] == "look" and b["sites"])
    given = {n: "2026-12-01" if s["kind"] == "date" else (s.get("options") or ["Zagreb"])[0]
             for n, s in book["inputs"].items() if s["required"] and "default" not in s}
    _no_deck(monkeypatch, {"person_connected": False, "extension_set_up": True})
    res = await _run(id=book["id"], inputs=given)
    assert res.success and browser["calls"][0][1]["target"] == "headless"
    assert "the person's Chrome isn't connected" in res.content
    _no_deck(monkeypatch, {"person_connected": True, "extension_set_up": True})
    browser["calls"].clear()
    await _run(id=book["id"], inputs=given)
    assert browser["calls"][0][1]["target"] == "chrome"


async def test_a_headless_run_keeps_off_the_servers_network(monkeypatch, browser):
    from captain_claw import deck_confine

    monkeypatch.setenv(deck_confine.CONFINE_ENV, "user")
    _no_deck(monkeypatch, {"person_connected": False})
    book = next(b for b in pb.wp.seed() if b["where"] == "prefer_chrome" and not b["sites"])
    slot = pb.wp._SLOT.fullmatch(book["open"]).group(1)
    given = {n: "x" for n, s in book["inputs"].items() if s["required"] and "default" not in s}
    given[slot] = "https://127.0.0.1:8080/"
    res = await _run(id=book["id"], inputs=given)
    assert not res.success and browser["calls"] == []


async def test_a_page_that_wont_open_says_so(monkeypatch, browser):
    _no_deck(monkeypatch)
    browser["pages"]["navigate"] = ToolResult(success=False, error="Browser navigate failed: net::ERR_NAME_NOT_RESOLVED")
    res = await _run(id="maps.travel_time", inputs={"origin": "Zagreb", "destination": "Split"})
    assert not res.success and "ERR_NAME_NOT_RESOLVED" in res.error and "show(id='maps.travel_time')" in res.error


# ── the stop ─────────────────────────────────────────────────────────


async def test_an_act_playbook_hands_over_its_steps_and_stops_before_its_button(monkeypatch, browser):
    _no_deck(monkeypatch, {"person_connected": True, "extension_set_up": True})
    agent = FakeAgent()
    res = await _run(id="googleplay.cancel_subscription", inputs={"subscription": "Duolingo"}, _agent=agent)
    assert res.success and [c[0] for c in browser["calls"]] == ["navigate"] and not agent.provider.calls
    assert "1. Find Duolingo in the list" in res.content
    assert "Stop before “Cancel subscription”: never press it" in res.content
    armed = pb.armed_stop({"_session_id": "s1"})
    assert armed["stop"] == "Cancel subscription" and armed["left"] == 1 and armed["target"] == "chrome"

    tool = browser_mod.BrowserTool()
    ctx = {"_session_id": "s1", "_agent": agent, "target": "chrome"}
    # The first one only opens the reason dialog (its step says so): it goes through.
    assert await pb.stop_check(tool, "click", "chrome", {**ctx, "text": "Cancel subscription"}) is None
    refused = await tool._execute("click", **ctx, text="Cancel  subscription")
    assert not refused.success and "never presses “Cancel subscription”" in refused.error
    assert "in their Chrome" in refused.error
    # Other buttons, other sessions: not this playbook's business.
    assert await pb.stop_check(tool, "click", "chrome", {**ctx, "text": "Continue"}) is None
    assert pb.armed_stop({"_session_id": "s2"}) is None
    # The turn is over: the person may ask for it in their next message.
    pb.turn_over(agent)
    assert pb.armed_stop({"_session_id": "s1"}) is None


async def test_a_click_by_ref_is_known_by_the_elements_name(monkeypatch):
    monkeypatch.setattr(pb, "_STOPS", {})
    book = pb.wp.by_id("amazon.start_return")
    pb.arm_stop({"_session_id": "s9"}, book, "headless")

    class Tree:
        asked: list[dict[str, Any]] = []

        async def _execute(self, action, **kw):
            self.asked.append({"action": action, **kw})
            return ToolResult(success=True, content=(
                "Page: Returns — https://www.amazon.example/returns\nViewport 1280×720\nInside ref_7:\n"
                f"- button {json.dumps(book['stop'])} [ref_7]\n  - text \"and print the label\""))

    tree = Tree()
    refused = await pb.stop_check(tree, "click", "headless", {"_session_id": "s9", "ref": "ref_7"})
    assert refused is not None and "headless browser, so they can't press it there" in refused.error
    assert tree.asked[0]["action"] == "read_page" and tree.asked[0]["ref"] == "ref_7"
    assert tree.asked[0]["target"] == "headless"
    assert await pb.stop_check(tree, "click", "headless", {"_session_id": "s9", "ref": "ref_99"}) is None
    assert await pb.stop_check(tree, "hover", "headless", {"_session_id": "s9", "ref": "ref_7"}) is None


def test_the_tool_comes_with_the_browser_tool():
    from captain_claw.agent_context_mixin import AgentContextMixin

    assert AgentContextMixin._BUILTIN_TOOL_MAP["browser"] == ["browser", "web_playbook"]


# ── the person's recordings ──────────────────────────────────────────

REC = {"id": "rec_0a1b2c3d4e5f", "title": "Monthly report", "category": "mine", "says": ["VAT report from the portal"],
       "kind": "recording", "where": "recording", "region": "global", "sites": ["portal.example"],
       "inputs": {"s3": {"label": "Period", "kind": "text", "required": True, "ask": "before"}},
       "open": "https://portal.example/reports", "steps": [], "read": {}, "answer": "", "stop": "", "needs": [],
       "tags": [], "about": "Last month's VAT report", "step_count": 4}


async def test_the_persons_recordings_are_found_and_run_in_their_chrome(monkeypatch, browser):
    _no_deck(monkeypatch, {"person_connected": True, "extension_set_up": True})

    async def _recs():
        return [REC]

    async def _one(pid):
        return REC if pid == REC["id"] else None

    async def _words(pid, values):
        return [f"1. Type “{values.get('s3')}” into the field “Period”"]

    monkeypatch.setattr(pb, "_recordings", _recs)
    monkeypatch.setattr(pb, "_recording", _one)
    monkeypatch.setattr(pb, "_recording_words", _words)
    found = await pb.WebPlaybookTool().execute(action="find", query="monthly VAT report")
    assert found.content.startswith("The person's own recordings:\n- rec_0a1b2c3d4e5f — Monthly report (recording; their Chrome only)")
    res = await _run(id=REC["id"], inputs={"s3": "2026-11"})
    assert res.success and browser["calls"][0][1] == {**browser["calls"][0][1], "target": "chrome",
                                                      "url": "https://portal.example/reports"}
    assert res.content.startswith("Recording rec_0a1b2c3d4e5f — Monthly report")
    assert "The person recorded these steps" in res.content and "1. Type “2026-11” into the field “Period”" in res.content
    assert not (await _run(id=REC["id"], inputs={})).success


async def test_a_page_with_a_step_to_do_first_skips_the_read(monkeypatch, browser):
    _no_deck(monkeypatch)
    agent = FakeAgent({"ready": True, "fields": {"tomatometer": "92%"}})
    res = await _run(id="rottentomatoes.scores", inputs={"title": "Dune Part Two"}, _agent=agent)
    assert res.success and [c[0] for c in browser["calls"]] == ["navigate"] and not agent.provider.calls
    assert "1. Open the matching result" in res.content and "Reject all" in res.content


async def test_the_read_and_the_steps_carry_the_values(monkeypatch, browser):
    _no_deck(monkeypatch)
    agent = FakeAgent({"ready": False, "fields": {}, "missing": "consent page"})
    res = await _run(id="xe.convert_amount", inputs={"amount": "100", "from": "EUR", "to": "USD"}, _agent=agent)
    sent = agent.provider.calls[0][1].content
    assert "How many USD one EUR buys" in sent and "{to}" not in sent
    assert "Then read: converted" in res.content and "How many USD one EUR buys" in res.content


async def test_a_link_that_wont_open_repeats_what_it_carried(monkeypatch, browser):
    _no_deck(monkeypatch)
    browser["pages"]["navigate"] = ToolResult(success=False, error="Page title: Sorry, this page does not exist.\nHTTP status: 404")
    res = await _run(id="ecb.reference_rate", inputs={"currency": "USD"})
    assert not res.success and "currency = 'USD' — Lowercase three-letter code" in res.error
    assert "run it again with it fixed" in res.error


# ── the person's Chrome first ────────────────────────────────────────


async def test_any_playbook_runs_in_their_chrome_when_its_connected(monkeypatch, browser):
    _no_deck(monkeypatch, {"person_connected": True, "extension_set_up": True})
    res = await _run(id="maps.travel_time", inputs={"origin": "Zagreb", "destination": "Split"})
    assert res.success and browser["calls"][0][1]["target"] == "chrome"
    assert "in the person's Chrome" in res.content
    # Stopped in the panel, or not there: the headless browser does for a playbook that may.
    for status in ({"person_connected": True, "stopped": True}, {"person_connected": False}, {}):
        _no_deck(monkeypatch, status)
        browser["calls"].clear()
        await _run(id="maps.travel_time", inputs={"origin": "Zagreb", "destination": "Split"})
        assert browser["calls"][0][1]["target"] == "headless"


async def test_no_fit_points_at_their_chrome_when_its_connected(monkeypatch):
    _no_deck(monkeypatch, {"person_connected": True, "extension_set_up": True})
    res = await pb.WebPlaybookTool().execute(action="find", query="zzqx")
    assert "No playbook fits" in res.content and pb.NO_FIT_CHROME in res.content
    assert "ask_user, every question in one dialog" in res.content
    _no_deck(monkeypatch, {"person_connected": False})
    res = await pb.WebPlaybookTool().execute(action="find", query="zzqx")
    assert pb.NO_FIT_CHROME not in res.content and "ask_user" in res.content


# ── asking the person ────────────────────────────────────────────────


def test_the_questions_follow_the_inputs():
    book = pb.wp.by_id("asfinag.buy_vignette")
    names = list(book["inputs"])
    questions, ids = pb.ask_questions(book, names)
    by_id = {q["id"]: q for q in questions}
    assert ids == {n: n for n in names}
    assert by_id["plate"]["help"].startswith("For “Buy the Austrian digital vignette”.")
    assert by_id["vehicle"]["choices"] == ["car", "motorcycle"]
    assert by_id["period"]["choices"] == ["1 day", "10 days", "2 months", "1 year"]
    assert by_id["start"]["format"] == "date" and by_id["use_at_once"]["confirm"] is True
    assert "18 days" in by_id["use_at_once"]["help"] and "help" not in by_id["email"]
    odd = {"id": "x.y", "title": "T", "inputs": {"a" * 41: {"label": "Long", "kind": "place"}}}
    questions, ids = pb.ask_questions(odd, ["a" * 41])
    assert ids == {"input_1": "a" * 41} and "A town or an address." in questions[0]["help"]


@pytest.fixture
def asked(monkeypatch):
    """ask_user's dialog, answered as told."""
    from captain_claw.tools import ask_user

    seen: dict[str, Any] = {"reply": (200, {"ok": True, "answers": {}})}

    async def follow(questions, wait_s, agent, stream=None, waiting=""):
        seen.update(questions=questions, wait_s=wait_s, agent=agent, waiting=waiting)
        return seen["reply"]

    monkeypatch.setattr(ask_user, "_fd_base", lambda: "http://fd.test")
    monkeypatch.setattr(ask_user, "ask_and_follow", follow)
    return seen


async def test_a_run_asks_the_person_for_what_it_still_needs_and_goes_on(monkeypatch, browser, asked):
    monkeypatch.setattr(pb, "ask_person", _real_ask_person)
    _no_deck(monkeypatch)
    asked["reply"] = (200, {"ok": True, "answers": {"destination": "Split"}})
    agent = FakeAgent()
    res = await _run(id="maps.travel_time", inputs={"origin": "Zagreb"}, _agent=agent)
    assert res.success, res.error
    assert [q["id"] for q in asked["questions"]] == ["destination"] and asked["agent"] is agent
    assert "How long from A to B" in asked["waiting"] and asked["wait_s"] == pb.ASK_WAIT_S
    assert "destination=Split" in browser["calls"][0][1]["url"]


async def test_an_act_playbook_asks_once_then_fills_in_their_chrome(monkeypatch, browser, asked):
    monkeypatch.setattr(pb, "ask_person", _real_ask_person)
    _no_deck(monkeypatch, {"person_connected": True, "extension_set_up": True})
    asked["reply"] = (200, {"ok": True, "answers": {
        "plate_country": "Croatia", "vehicle": "car", "period": "10 days", "start": "2026-12-01",
        "email": "driver@example.com", "use_at_once": True}})
    res = await _run(id="asfinag.buy_vignette", inputs={"plate": "AB123CD"})
    assert res.success, res.error
    assert len(asked["questions"]) == 6 and "plate" not in [q["id"] for q in asked["questions"]]
    nav = browser["calls"][0][1]
    assert nav["target"] == "chrome" and nav["url"] == "https://shop.asfinag.at/de/fahrzeugtypauswahl/?source=Vignette"
    assert 'Press "Kaufen" on the 10 days vignette' in res.content and "(AB123CD)" in res.content
    assert 'Do these steps with the browser tool (target="chrome")' in res.content
    assert "ask_user too, every question in one dialog" in res.content
    armed = pb.armed_stop({"_session_id": "s1"})
    assert armed["stop"] == "Weiter" and armed["left"] == 1 and armed["target"] == "chrome"


async def test_a_run_the_person_wont_answer_doesnt_open(monkeypatch, browser, asked):
    monkeypatch.setattr(pb, "ask_person", _real_ask_person)
    _no_deck(monkeypatch)
    asked["reply"] = (200, {"ok": False, "declined": True})
    res = await _run(id="maps.travel_time", inputs={"origin": "Zagreb"})
    assert not res.success and browser["calls"] == [] and "chose not to answer “To”" in res.error
    asked["reply"] = (200, {"ok": False, "no_answer": True, "error": "no answer in time"})
    res = await _run(id="maps.travel_time", inputs={"origin": "Zagreb"})
    assert not res.success and browser["calls"] == [] and "no answer in time" in res.error
    asked["reply"] = (200, {"ok": True, "answers": {"destination": "here"}})
    res = await _run(id="maps.travel_time", inputs={"origin": "Zagreb"})
    assert not res.success and browser["calls"] == []
    assert "the person answered in the dialog" in res.error and "don't ask them again" in res.error


async def test_where_the_dialog_cant_reach_them_the_agent_asks_in_its_reply(monkeypatch, browser, asked):
    monkeypatch.setattr(pb, "ask_person", _real_ask_person)
    _no_deck(monkeypatch)
    asked["reply"] = (200, {"ok": False, "no_answer": True, "elsewhere": True,
                            "error": "They're talking to you in whatsapp, where this dialog doesn't open"})
    res = await _run(id="maps.travel_time", inputs={"origin": "Zagreb"})
    assert not res.success and browser["calls"] == []
    assert "ask them in your reply (the dialog can't reach them: They're talking to you in whatsapp" in res.error
    from captain_claw.tools import ask_user

    monkeypatch.setattr(ask_user, "_fd_base", lambda: "")
    res = await _run(id="maps.travel_time", inputs={"origin": "Zagreb"})
    assert "ask them in your reply (the dialog can't reach them: it needs Flight Deck)" in res.error


_real_ask_person = pb.ask_person


def test_every_shipped_playbooks_questions_are_a_dialog_flight_deck_takes():
    from captain_claw.flight_deck import person_asks

    for book in pb.wp.seed():
        names = [n for n, s in book["inputs"].items() if s["required"] and "default" not in s and s["ask"] == "before"]
        if names:
            person_asks.check_questions(pb.ask_questions(book, names)[0])


# ── booking forms ask up front ───────────────────────────────────────


def test_booking_forms_ask_for_the_details_up_front():
    import re

    table = pb.wp.by_id("opentable.book_table")
    checked = pb.wp.check_values(table, {"restaurant": "Bistro", "date": "2026-12-05", "time": "19:30",
                                         "guest_name": "Alex Doe"})
    assert checked["ask_before"] == ["phone", "email"]
    questions, _ = pb.ask_questions(table, checked["ask_before"])
    assert questions[1]["format"] == "email" and "country code" in questions[0]["help"]
    assert "phone 555 0100, email ‹Email for the confirmation›" in "\n".join(
        pb.wp.step_lines(table, {**checked["values"], "phone": "555 0100"}))
    assert pb.wp.by_id("doctolib.book_appointment")["inputs"]["who"]["label"] == "Doctor, speciality or practice"
    # A form that may or may not ask (a job application): named, asked when it gets there.
    job = pb.wp.check_values(pb.wp.by_id("linkedin.job_application"), {"job": "Data analyst at Example"})
    assert job["ask_before"] == [] and job["ask_during"] == ["salary", "notice", "work_permit", "start"]
    # No shipped step leaves "ask for anything missing" to a reply.
    for book in pb.wp.seed():
        for step in book["steps"]:
            text = " ".join(str(step.get(k) or "") for k in ("text", "note"))
            if re.search(r"\bask (?:the person )?for anything\b", text, re.I):
                assert "ask_user" in text, (book["id"], text)


async def test_a_job_application_names_what_to_ask_when_the_form_gets_there(monkeypatch, browser):
    _no_deck(monkeypatch, {"person_connected": True, "extension_set_up": True})
    res = await _run(id="linkedin.job_application", inputs={"job": "Data analyst at Example"})
    assert res.success, res.error
    assert ("When a step needs “Salary expectation”, “Notice period”, “Work permit for the job's country”, "
            "“Earliest start date”, ask the person with ask_user (not in your reply)") in res.content
