"""Crew's tool: an agent running a worker on its own model (the default), its CLI's own login, another agent's or a deck tier."""

# ruff: noqa: F811
from __future__ import annotations

import json

import pytest

from captain_claw.flight_deck import crew_logins, crew_store, server
from captain_claw.tools.crew import CrewTool, _choice_lines, _model_source_of

from .test_crew_routes import _approve, _client, as_agent, deck  # noqa: F401

TEAM_KEY = "sk-team-key-that-nobody-may-read"
TIER = {"provider": "anthropic", "model": "claude-test", "base_url": "", "api_key": "@system", "input_ctx": 200000}
GEMINI = {"provider": "gemini", "model": "gemini-test", "base_url": "", "api_key": "@system"}


class _Deck:
    """The tool's client, over the real routes, as the agent."""

    async def call(self, method: str, path: str, body=None):
        async with _client(None) as agent:
            reply = await agent.request(method, f"/fd/crew/agent/{path}", **({"json": body} if body is not None else {}))
        try:
            return reply.status_code, reply.json()
        except ValueError:
            return reply.status_code, reply.text


@pytest.fixture
async def ready(deck, as_agent, monkeypatch):
    """An approved person, a team tier, and two agents of theirs. The one asking is "helper"."""
    await _approve(deck, deck.user)
    as_agent["owner"] = deck.user["id"]
    await deck.db.set_system_setting("fd:shared-tier-sets", json.dumps({"sets": [{"id": "team", "name": "Team", "tiers": {"balanced": TIER, "vision": GEMINI}}], "defaultSetId": "team"}))
    await deck.db.set_system_setting("fd:provider-keys", json.dumps({"anthropic": TEAM_KEY, "gemini": TEAM_KEY}))
    home = deck.tmp / "fd-data"
    home.mkdir(exist_ok=True)
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", home / ".processes.json")
    registry = {}
    for slug, model, owner in (("helper", "its-own-model", deck.user["id"]), ("sokol", "sokols-model", deck.user["id"]), ("elsewhere", "theirs", deck.other["id"])):
        (home / slug).mkdir()
        (home / slug / "config.yaml").write_text(json.dumps({"model": {"provider": "anthropic", "model": model}}))
        (home / slug / ".env").write_text(f"ANTHROPIC_API_KEY=sk-of-{slug}\n")
        registry[slug] = {"name": slug.title(), "owner": owner, "slug": slug}
    (home / ".processes.json").write_text(json.dumps(registry))
    return deck


# ── Reading what it's given ──────────────────────────────────────────────


@pytest.mark.parametrize("said, source", [
    ("me", {"kind": "agent", "agent": "me"}),
    (" ME ", {"kind": "agent", "agent": "me"}),
    ("agent:sokol", {"kind": "agent", "agent": "sokol"}),
    ("tier:Balanced", {"kind": "tier", "tier": "balanced"}),
    ("cli", {"kind": "cli"}),
    (" CLI", {"kind": "cli"}),
    ("", None),
    (None, None),
])
def test_what_runs_on_names(said, source):
    assert _model_source_of(said) == (source, "")


@pytest.mark.parametrize("said", ["balanced", "tier:", "agent: ", "typed:http://x", "gpt-5"])
def test_what_it_cant_read_is_said_plainly(said):
    source, why = _model_source_of(said)
    assert source is None and "'me', 'cli', 'agent:<name>' or 'tier:<name>'" in why


class _Asked:
    def __init__(self, **replies):
        self.replies = replies
        self.asked = []

    async def call(self, method, path, body=None):
        self.asked.append((path, body))
        return 200, self.replies.get(path, {})


MADE = {"id": "t", "status": "running", "harness_label": "OpenCode", "repo_label": "p", "branch": "b"}


async def test_it_isnt_sent_when_it_cant_be_read_or_contradicts_itself():
    client = _Asked(dispatch=MADE)
    bad = await CrewTool()._do_dispatch(client, harness="pi", brief="x", runs_on="balanced")
    assert not bad.success and "tier:<name>" in bad.error
    both = await CrewTool()._do_dispatch(client, harness="pi", brief="x", runs_on="me", model="some/model")
    assert not both.success and "not both" in both.error
    setup = await CrewTool()._do_prepare_env(client, harness="pi", runs_on="balanced", project="p")
    assert not setup.success and client.asked == []


async def test_nothing_named_is_left_to_the_deck_to_fill_in():
    """Its own model is the default, and Flight Deck knows who's asking: the tool sends nothing."""
    client = _Asked(dispatch=MADE)
    out = await CrewTool()._do_dispatch(client, harness="pi", brief="x")
    assert out.success and "model_source" not in client.asked[0][1]


async def test_its_cli_own_login_is_named_and_can_have_a_model():
    client = _Asked(dispatch=MADE)
    out = await CrewTool()._do_dispatch(client, harness="opencode", brief="x", runs_on="cli", model="opencode/some-free")
    assert out.success, out.error
    assert client.asked[0][1]["model_source"] == {"kind": "cli"} and client.asked[0][1]["model"] == "opencode/some-free"
    assert "It runs on OpenCode's own login." in out.content


async def test_a_fallback_is_relayed():
    client = _Asked(dispatch={**MADE, "model_note": "It can't run on your own model: no. So it runs on OpenCode's own login (x)."})
    out = await CrewTool()._do_dispatch(client, harness="opencode", brief="x")
    assert "It can't run on your own model: no. So it runs on OpenCode's own login (x)." in out.content


def test_the_options_put_itself_first_and_say_what_cant_be_used():
    lines = _choice_lines({
        "agents": [{"id": "sokol", "label": "Sokol", "model": "m1", "works_with": ["pi"]}, {"id": "me-agent", "label": "Me", "model": "m2", "works_with": ["claude", "pi"]},
                   {"id": "broken", "label": "Broken", "model": "", "works_with": [], "why_not": "It has no model set."}],
        "tiers": [{"id": "balanced", "label": "Balanced", "model": "m3", "works_with": ["codex"]}],
    }, "me-agent")
    assert lines[1] == "  me: Me's model, m2, for claude, pi"
    assert "  agent:sokol: Sokol's model, m1, for pi" in lines and "  tier:balanced: Balanced tier, m3, for codex" in lines
    assert "  agent:broken: Broken's model, can't be used: It has no model set." in lines
    assert _choice_lines({}, "x") == []


# ── Over the routes ──────────────────────────────────────────────────────


async def test_options_list_them_with_no_key_in_sight(ready):
    out = await CrewTool()._do_options(_Deck())
    assert out.success
    assert "me: Helper's model, its-own-model, for claude" in out.content
    assert "agent:sokol: Sokol's model, sokols-model" in out.content
    assert "tier:balanced: Balanced tier, claude-test, for claude" in out.content
    assert "tier:vision: Vision tier, gemini-test, for opencode, crush, pi" in out.content
    assert "elsewhere" not in out.content, "only the person's own agents"
    assert TEAM_KEY not in out.content and "sk-of-" not in out.content
    assert "no login of its own, which only runs_on=cli needs" in out.content
    assert "Yours, 'me', is what it runs on when runs_on is left out" in out.content


async def test_me_is_the_agent_asking_and_needs_no_login(ready):
    out = await CrewTool()._do_dispatch(_Deck(), harness="claude", brief="SCENARIO:work", project="p", runs_on="me")
    assert out.success, out.error
    assert "It runs on Helper's model: its-own-model." in out.content and "sk-of-" not in out.content
    task = crew_store.get_store().list_for(ready.user["id"], limit=1)[0]
    assert task["model_source"]["agent"] == "helper" and task["model"] == "its-own-model" and task["agent_slug"] == "helper"


async def test_another_of_their_agents(ready):
    out = await CrewTool()._do_dispatch(_Deck(), harness="claude", brief="SCENARIO:work", project="p", runs_on="agent:sokol")
    assert out.success and "It runs on Sokol's model: sokols-model." in out.content


async def test_but_not_someone_elses(ready):
    out = await CrewTool()._do_dispatch(_Deck(), harness="claude", brief="SCENARIO:work", project="p", runs_on="agent:elsewhere")
    assert not out.success and "no agent called elsewhere" in out.error


async def test_a_tier_of_theirs(ready):
    out = await CrewTool()._do_dispatch(_Deck(), harness="claude", brief="SCENARIO:work", project="p", runs_on="tier:balanced")
    assert out.success and "It runs on Balanced tier: claude-test." in out.content
    assert TEAM_KEY not in out.content


async def test_a_tier_its_cli_cant_talk_to_says_who_can(ready):
    out = await CrewTool()._do_dispatch(_Deck(), harness="claude", brief="SCENARIO:work", project="p", runs_on="tier:vision")
    assert not out.success and "Claude Code can't talk to" in out.error and "OpenCode, Crush or Pi can" in out.error


async def test_a_tier_that_isnt_there(ready):
    out = await CrewTool()._do_dispatch(_Deck(), harness="claude", brief="SCENARIO:work", project="p", runs_on="tier:nope")
    assert not out.success and "no nope tier" in out.error


async def test_setting_a_project_up_can_run_on_one_too(ready):
    out = await CrewTool()._do_prepare_env(_Deck(), harness="claude", project="p", runs_on="tier:balanced")
    assert out.success, out.error
    assert "It runs on Balanced tier: claude-test." in out.content


async def test_status_says_what_it_runs_on(ready):
    made = await CrewTool()._do_dispatch(_Deck(), harness="claude", brief="SCENARIO:work", project="p", runs_on="tier:balanced")
    task = crew_store.get_store().list_for(ready.user["id"], limit=1)[0]
    assert made.success
    out = await CrewTool()._do_status(_Deck(), task_id=task["id"])
    assert "(Balanced tier)" in out.content and TEAM_KEY not in out.content


async def test_nothing_named_runs_on_the_agents_own_model_and_key(ready):
    """The agent doesn't have to ask which model or login: it's its own, as it would do the work itself."""
    out = await CrewTool()._do_dispatch(_Deck(), harness="claude", brief="SCENARIO:work", project="p")
    assert out.success, out.error
    assert "It runs on Helper's model: its-own-model." in out.content and "sk-of-" not in out.content
    task = crew_store.get_store().list_for(ready.user["id"], limit=1)[0]
    assert task["model_source"] == {"kind": "agent", "agent": "helper", "label": "Helper's model"} and task["model"] == "its-own-model"


async def test_setting_a_project_up_runs_on_it_too(ready):
    out = await CrewTool()._do_prepare_env(_Deck(), harness="claude", project="p")
    assert out.success, out.error
    assert "It runs on Helper's model: its-own-model." in out.content


async def test_its_cli_own_login_when_asked(ready):
    out = await CrewTool()._do_dispatch(_Deck(), harness="claude", brief="SCENARIO:work", project="p", runs_on="cli")
    assert not out.success and "Add your Claude Code login" in out.error and "an agent's model or a deck tier" in out.error


async def test_a_model_of_the_clis_own_is_its_own_login(ready):
    """An agent from before runs_on=cli names only the model: that was, and is, the CLI's own."""
    out = await CrewTool()._do_dispatch(_Deck(), harness="claude", brief="SCENARIO:work", project="p", model="opus")
    assert not out.success and "Add your Claude Code login" in out.error


def _on_gemini(deck):
    """The agent asking is on a model Claude Code can't talk to."""
    home = deck.tmp / "fd-data" / "helper"
    (home / "config.yaml").write_text(json.dumps({"model": {"provider": "gemini", "model": "gemini-own"}}))
    (home / ".env").write_text("GEMINI_API_KEY=sk-of-helper\n")


async def test_its_model_cant_drive_the_cli_and_theres_no_login_says_what_would(ready):
    _on_gemini(ready)
    out = await CrewTool()._do_dispatch(_Deck(), harness="claude", brief="SCENARIO:work", project="p")
    assert not out.success
    assert "It can't run on your own model: Claude Code can't talk to that model's server. OpenCode, Crush or Pi can." in out.error
    assert "no login of its own here either" in out.error and "Crew → Logins" in out.error and "runs_on" in out.error
    assert crew_store.get_store().list_for(ready.user["id"], limit=1) == []


async def test_its_model_cant_drive_the_cli_so_its_own_login_does_and_it_says_so(ready):
    _on_gemini(ready)
    await crew_logins.save_login(ready.db, ready.user["id"], "claude", "sk-ant-oat-saved-login")
    out = await CrewTool()._do_dispatch(_Deck(), harness="claude", brief="SCENARIO:work", project="p")
    assert out.success, out.error
    assert "It can't run on your own model" in out.content and "So it runs on Claude Code's own login (your saved login)." in out.content
    assert "sk-ant-oat" not in out.content
    task = crew_store.get_store().list_for(ready.user["id"], limit=1)[0]
    assert not task.get("model_source")


async def test_an_agent_cant_hand_a_key_through_the_route(ready):
    async with _client(None) as agent:
        got = await agent.post("/fd/crew/agent/dispatch", json={"harness": "claude", "brief": "x", "project": "p", "model_source": {
            "kind": "typed", "url": "http://10.0.0.9/v1", "model": "m", "key": "sk-from-a-chat"}})
    assert got.status_code == 403 and "New task page" in got.json()["detail"]


async def test_the_person_can_still_enter_one_on_the_page(ready):
    async with _client(ready.user) as person:
        got = await person.post("/fd/crew/tasks", json={"harness": "claude", "brief": "SCENARIO:work", "project": "p", "model_source": {
            "kind": "typed", "url": "http://10.0.0.9/v1", "model": "m", "key": "sk-theirs"}})
    assert got.status_code in (200, 201), got.text


def test_the_tool_says_how_and_when():
    text = CrewTool.description
    assert "runs_on" in text and "yours, with your key, unless the user asks for another" in text
    assert "don't ask them which model or login to use" in text and "never ask for a key in a chat" in text
    assert "runs_on" in CrewTool.parameters["properties"]
