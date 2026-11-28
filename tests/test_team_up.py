"""Team up (captain_claw/team_up.py): settings, parts, the helper guard, the
live notes, the budget, and a whole run on a fake web server."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from captain_claw import team_up


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(team_up, "_HOME", tmp_path)
    monkeypatch.setattr(team_up, "_TEAMS", {})
    monkeypatch.setattr(team_up, "_SERVER", None)
    yield


# ── Fakes ────────────────────────────────────────────────────────────────

class FakeSession:
    _n = 0

    def __init__(self, name: str = "chat", metadata: dict | None = None):
        FakeSession._n += 1
        self.id = f"s{FakeSession._n}"
        self.name = name
        self.metadata = metadata or {}
        self.messages: list[dict] = []


class FakeSM:
    def __init__(self):
        self.created: list[FakeSession] = []

    async def create_session(self, name: str = "default", metadata: dict | None = None):
        s = FakeSession(name, metadata)
        self.created.append(s)
        return s

    async def save_session(self, session):
        return None


class FakeProvider:
    provider = "anthropic"
    model = "claude-haiku-5-5"
    base_url = ""


class FakeAgent:
    """Drives one 'turn': a narration, a tool line, usage, then the reply."""

    def __init__(self, session=None, send=None, reply: str = "", tokens: int = 1000,
                 delay: float = 0.0, hang: bool = False):
        self.session = session or FakeSession()
        self.provider = FakeProvider()
        self.ws_broadcast = send
        self._send = send
        self.cancel_event = asyncio.Event()
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                           "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        self._reply = reply
        self._tokens = tokens
        self._delay = delay
        self._hang = hang
        self._last_complete_cancelled = False
        self.briefs: list[str] = []

    def _turn_channel(self) -> str:
        return "web"

    async def complete(self, text: str) -> str:
        self.briefs.append(text)
        send = self._send or (lambda m: None)
        send({"type": "narration", "text": "Looking at the sources"})
        send({"type": "tool_output_inline", "tool": "web_search", "summary": "Searching the web: prices"})
        send({"type": "turn_usage", "prompt_tokens": self._tokens, "completion_tokens": 100,
              "total_tokens": self._tokens + 100})
        if self._hang:
            await self.cancel_event.wait()
            self._last_complete_cancelled = True
            return "partial"
        if self._delay:
            await asyncio.sleep(self._delay)
        self.last_usage = {**self.last_usage, "prompt_tokens": self._tokens, "completion_tokens": 100,
                           "total_tokens": self._tokens + 100}
        return self._reply or f"Result for: {text.splitlines()[0][:40]}"


class FakeServer:
    LANE_MAIN = "A"
    REMOTE_LANE_PREFIX = "BP"

    def __init__(self, helper_kwargs: dict | None = None):
        self.agent = FakeAgent(session=FakeSession("main chat"))
        self.agent.session.messages = [
            {"role": "user", "content": "compare three suppliers", "origin": "human"},
            {"role": "assistant", "content": "I can team up on this."},
        ]
        self._lane_agents: dict[str, Any] = {}
        self._lane_sockets: dict[str, set] = {}
        self.clients: set = set()
        self._busy = False
        self.frames: list[dict] = []
        self.helpers: list[FakeAgent] = []
        self.helper_kwargs = helper_kwargs or {}

    @staticmethod
    def normalize_lane(raw):
        return str(raw or "").upper()[:8] or "A"

    def _broadcast(self, msg, exclude=None):
        self.frames.append(msg)

    def _lane_send(self, lane):
        return lambda msg: self.frames.append({**msg, "_lane": lane})

    async def _build_scoped_agent(self, session, send, **kwargs):
        name = (session.metadata.get("team_helper") or {}).get("name", "")
        per = self.helper_kwargs.get(name, {})
        agent = FakeAgent(session=session, send=send, **per)
        self.helpers.append(agent)
        return agent


@pytest.fixture
def server(monkeypatch):
    srv = FakeServer()
    sm = FakeSM()
    import captain_claw.session as session_mod

    monkeypatch.setattr(session_mod, "get_session_manager", lambda: sm)
    team_up.bind_server(srv)
    srv.sm = sm
    return srv


def _parts(n=3):
    names = ["Prices", "Reviews", "Delivery", "Support"]
    return [{"name": names[i], "goal": f"Find the {names[i].lower()} of A, B and C"} for i in range(n)]


def _events(srv, event):
    return [f for f in srv.frames if f.get("type") == "team" and f.get("event") == event]


# ── Settings and the normal turn ─────────────────────────────────────────

def test_settings_default_and_clamped(tmp_path):
    assert team_up.read_settings() == {"mode": "ask", "max_helpers": 3, "budget_x": 6}
    (tmp_path / team_up.SETTINGS_FILENAME).write_text('{"mode": "ON", "max_helpers": 9, "budget_x": 1}')
    assert team_up.read_settings() == {"mode": "on", "max_helpers": 4, "budget_x": 3}
    (tmp_path / team_up.SETTINGS_FILENAME).write_text("not json")
    assert team_up.read_settings()["mode"] == "ask"
    assert team_up.normalize_settings({"mode": "sometimes"})["mode"] == "ask"


def test_normal_turn_median_with_floor_and_default():
    assert team_up.normal_turn() == team_up.TURN_DEFAULT
    for n in (10_000, 20_000):
        team_up.note_turn({"prompt_tokens": n})
    assert team_up.normal_turn() == team_up.TURN_DEFAULT  # fewer than MIN_TURNS_KNOWN
    for n in (80_000, 90_000, 100_000):
        team_up.note_turn({"prompt_tokens": n - 1000, "completion_tokens": 1000})
    assert team_up.normal_turn() == 80_000
    assert team_up.team_budget({"budget_x": 6}) == 480_000


def test_turn_tokens_leaves_cache_reads_out():
    usage = {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200,
             "cache_read_input_tokens": 50_000, "cache_creation_input_tokens": 300}
    assert team_up.turn_tokens(usage) == 1500
    assert team_up.turn_tokens(None) == 0


def test_note_turn_keeps_the_last_fifty():
    for i in range(60):
        team_up.note_turn({"prompt_tokens": 40_000 + i})
    assert len(team_up._read_turns()) == team_up.TURNS_KEPT


# ── Parts ────────────────────────────────────────────────────────────────

def test_parts_clean_names_and_limits():
    parts, err = team_up.parse_parts(
        [{"name": "Prices!!<>", "goal": "  look   up prices "}, {"name": "Prices", "goal": "again"},
         {"goal": "no name"}, {"name": "x", "goal": ""}], 4)
    assert err == ""
    assert [p["name"] for p in parts] == ["Prices", "Prices 2", "Helper 3"]
    assert parts[0]["goal"] == "look up prices"
    _, err = team_up.parse_parts([{"goal": "only one"}], 3)
    assert "at least 2" in err
    _, err = team_up.parse_parts(_parts(4), 3)
    assert "At most 3" in err
    parts, err = team_up.parse_parts('[{"goal": "a"}, {"goal": "b"}]', 3)
    assert err == "" and len(parts) == 2


# ── Availability ─────────────────────────────────────────────────────────

def test_unavailable_without_server_off_helper_and_local():
    agent = FakeAgent()
    assert "web chat" in team_up.why_unavailable(agent)
    srv = FakeServer()
    team_up.bind_server(srv)
    assert team_up.why_unavailable(srv.agent) == ""
    assert team_up.listed()
    (team_up._HOME / team_up.SETTINGS_FILENAME).write_text('{"mode": "off"}')
    assert "off" in team_up.why_unavailable(srv.agent)
    assert not team_up.listed()
    (team_up._HOME / team_up.SETTINGS_FILENAME).write_text('{"mode": "ask"}')
    assert "web chat" in team_up.why_unavailable(FakeAgent())  # not one of the server's lanes
    srv.agent.provider.base_url = "http://localhost:11434/v1"
    assert "this machine" in team_up.why_unavailable(srv.agent)
    srv.agent.provider.base_url = ""
    srv.agent._team_helper = ("t", "h1")
    assert "helpers don't team up" in team_up.why_unavailable(srv.agent)


def test_unavailable_on_other_channels_and_remote_lanes():
    srv = FakeServer()
    team_up.bind_server(srv)
    srv.agent._turn_channel = lambda: "whatsapp"
    assert "not on whatsapp" in team_up.why_unavailable(srv.agent)
    remote = FakeAgent()
    srv._lane_agents["BPabc"] = remote
    assert "owner's web chat" in team_up.why_unavailable(remote)
    lane_b = FakeAgent()
    srv._lane_agents["B"] = lane_b
    assert team_up.why_unavailable(lane_b) == ""
    assert team_up.lane_of(lane_b) == "B"


# ── The guard ────────────────────────────────────────────────────────────

def _helper_agent(team_id="t1"):
    team = team_up.Team(id=team_id, title="x", lead=FakeAgent(), lane="A", lead_session_id="",
                        helpers=[team_up.Helper(id="h1", name="Prices", goal="g", state="working"),
                                 team_up.Helper(id="h2", name="Reviews", goal="g", state="working")],
                        state="running")
    team_up._TEAMS[team_id] = team
    a1, a2 = FakeAgent(), FakeAgent()
    a1._team_helper = (team_id, "h1")
    a2._team_helper = (team_id, "h2")
    return team, a1, a2


@pytest.mark.parametrize("name,args", [
    ("send_mail", {}), ("whatsapp_send_file", {}), ("payments", {"action": "pay"}),
    ("google_mail", {"action": "send"}), ("google_mail", {"action": "create_draft"}),
    ("google_mail", {"action": "archive"}), ("google_calendar", {"action": "create_event"}),
    ("cron", {"action": "add"}), ("facts", {"action": "add"}), ("model_switch", {"action": "set"}),
    ("basna", {}), ("vatra", {}), ("flight_deck", {"action": "delegate"}), ("terminal", {}),
    ("colleagues", {"action": "hand_over"}), ("public_link", {}), ("team", {"action": "propose"}),
    ("team", {"action": "stop"}),
])
def test_helper_refused(name, args):
    _team, a1, _a2 = _helper_agent()
    assert team_up.tool_block(name, args, a1)


@pytest.mark.parametrize("name,args", [
    ("web_search", {}), ("web_fetch", {}), ("read", {}), ("write", {}), ("shell", {}),
    ("google_mail", {"action": "search"}), ("google_calendar", {"action": "list_events"}),
    ("datastore", {"action": "insert"}), ("ask_user", {}), ("team", {"action": "status"}),
    ("facts", {"action": "search"}), ("pdf_extract", {}),
])
def test_helper_allowed(name, args):
    _team, a1, _a2 = _helper_agent()
    assert team_up.tool_block(name, args, a1) == ""


def test_not_a_helper_is_never_guarded():
    assert team_up.tool_block("send_mail", {}, FakeAgent()) == ""
    assert team_up.tool_block("send_mail", {}, None) == ""


def test_browser_has_one_holder_lead_included():
    team, a1, a2 = _helper_agent()
    assert team_up.tool_block("browser", {"action": "open"}, a1) == ""
    assert team.browser_holder == "h1"
    assert "in use by Prices" in team_up.tool_block("browser", {"action": "open"}, a2)
    assert "in use by your helper Prices" in team_up.tool_block("browser", {}, team.lead)
    team.helpers[0].state = "done"
    assert team_up.tool_block("browser", {"action": "open"}, a2) == ""
    assert team.browser_holder == "h2"


def test_ask_user_question_names_the_helper():
    _team, a1, _a2 = _helper_agent()
    out = team_up.labelled_question(a1, {"question": "Which region?", "_agent": a1})
    assert out["question"] == "Prices: Which region?"
    assert out["_agent"] is a1
    plain = FakeAgent()
    assert team_up.labelled_question(plain, {"question": "Q"}) == {"question": "Q"}


# ── Notes ────────────────────────────────────────────────────────────────

def test_lead_note_live_and_proposed():
    team, _a1, _a2 = _helper_agent()
    team.budget = 100_000
    team.helpers[0].tokens = 30_000
    team.helpers[0].steps.append({"t": 0, "kind": "tool", "text": "Searching the web: prices"})
    team.helpers[0].step_count = 1
    note = team_up.lead_note(team)
    assert note.startswith("[Team now")
    assert "Prices: working · 1 steps · last: Searching the web: prices · 30k tokens" in note
    assert "Budget: 30k of 100k tokens." in note
    team.state = "proposed"
    assert "haven't answered the card" in team_up.lead_note(team)
    team.state = "done"
    assert team_up.lead_note(team) == ""


def test_helper_note_carries_tells_and_wrap_up():
    team, _a1, _a2 = _helper_agent()
    h = team.helpers[0]
    assert team_up.helper_note(team, h) == ""
    assert team_up.tell(team, "prices", "Only EU suppliers") == ""
    assert "Only EU suppliers" in team_up.helper_note(team, h)
    assert team_up.tell(team, "all", "Hurry") == ""
    assert "Hurry" in team_up.helper_note(team, team.helpers[1])
    assert "No helper called" in team_up.tell(team, "nobody", "x")
    h.wrap_up = True
    assert "budget reached" in team_up.helper_note(team, h).lower()


# ── Budget ───────────────────────────────────────────────────────────────

def test_budget_wraps_up_then_stops():
    team, a1, a2 = _helper_agent()
    team.helpers[0].agent, team.helpers[1].agent = a1, a2
    team.budget = 10_000
    team.helpers[0].tokens = 4_000
    team.helpers[1].tokens = 4_500
    team_up._check_budget(team)
    assert team.warned and not team.helpers[0].wrap_up
    team.helpers[1].tokens = 6_100   # over 10k total
    team_up._check_budget(team)
    assert team.helpers[0].wrap_up and team.helpers[1].wrap_up
    assert team.helpers[0].state == "working"
    team.helpers[0].tokens = 6_000   # 12.1k > 1.15x
    team_up._check_budget(team)
    assert team.helpers[0].state == "stopping" and a1.cancel_event.is_set()


# ── A whole run ──────────────────────────────────────────────────────────

async def _until(cond, timeout=3.0):
    end = asyncio.get_event_loop().time() + timeout
    while not cond():
        if asyncio.get_event_loop().time() > end:
            raise AssertionError("timed out")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_propose_ask_then_start_run_and_deliver(server, monkeypatch):
    delivered: list[dict] = []

    async def fake_handle_chat(srv, ws, content, **kw):
        delivered.append({"content": content, "lane": getattr(ws, "_lane", None), **kw})
        return True

    import captain_claw.web.chat_handler as ch

    monkeypatch.setattr(ch, "handle_chat", fake_handle_chat)
    team, err = await team_up.propose(server.agent, "Compare suppliers", _parts(3))
    assert err == "" and team.state == "proposed"
    assert _events(server, "proposed")
    assert "haven't answered" in server.agent._team_note_fn()

    assert await team_up.start(team, by="person") == ""
    assert team.state == "running"
    assert len(server.helpers) == 3
    # Each helper forks the lead's conversation into its own child session.
    for s in server.sm.created:
        assert [m["content"] for m in s.messages] == ["compare three suppliers", "I can team up on this."]
        assert s.metadata["team_helper"]["team"] == team.id
        assert s.metadata["subagent"]["spawned_by"] == server.agent.session.id
        # Files (and the prompt's folder line) stay the lead's.
        assert s.metadata["files_session_id"] == server.agent.session.id
    await _until(lambda: team.state in ("done", "failed"))
    assert "Your part: Find the prices of A, B and C" in server.helpers[0].briefs[0]
    assert "The other helpers: Reviews" in server.helpers[0].briefs[0]
    assert team.state == "done" and team.delivered
    assert [h.state for h in team.helpers] == ["done"] * 3
    assert delivered and delivered[0]["content"].startswith("[Team 'Compare suppliers' finished]")
    assert "## Prices — done" in delivered[0]["content"]
    assert delivered[0]["automation"].kind == "peer_relay"
    steps = _events(server, "step")
    assert {s["step"]["kind"] for s in steps} >= {"narration", "tool", "note"}
    assert server.agent.session.metadata["team_runs"][-1]["state"] == "done"
    assert getattr(server.agent, "_team_note_fn", None) is None


@pytest.mark.asyncio
async def test_on_mode_starts_at_once_and_cuts_the_open_turn(server, monkeypatch, tmp_path):
    (tmp_path / team_up.SETTINGS_FILENAME).write_text('{"mode": "on"}')

    async def fake_handle_chat(srv, ws, content, **kw):
        return True

    import captain_claw.web.chat_handler as ch

    monkeypatch.setattr(ch, "handle_chat", fake_handle_chat)
    server._busy = True  # the lead is inside the turn that proposed the team
    server.agent.session.messages.append({"role": "user", "content": "do it", "origin": "human"})
    server.agent.session.messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]})
    team, err = await team_up.propose(server.agent, "Compare", _parts(2))
    assert err == "" and team.state == "running"
    assert server.sm.created[0].messages[-1]["content"] == "do it"
    server._busy = False
    await _until(lambda: team.state == "done")


@pytest.mark.asyncio
async def test_stop_all_by_person_no_finishing_turn(monkeypatch):
    srv = FakeServer(helper_kwargs={"Prices": {"hang": True}, "Reviews": {"hang": True}})
    sm = FakeSM()
    import captain_claw.session as session_mod
    import captain_claw.web.chat_handler as ch

    monkeypatch.setattr(session_mod, "get_session_manager", lambda: sm)
    called: list[str] = []

    async def fake_handle_chat(srv_, ws, content, **kw):
        called.append(content)
        return True

    monkeypatch.setattr(ch, "handle_chat", fake_handle_chat)
    team_up.bind_server(srv)
    team, _ = await team_up.propose(srv.agent, "Compare", _parts(2))
    await team_up.start(team)
    await _until(lambda: all(h.step_count >= 2 for h in team.helpers))
    assert team_up.stop(team, "", by="you") == ""
    await _until(lambda: team.state == "stopped")
    assert called == []
    assert [h.state for h in team.helpers] == ["stopped", "stopped"]


@pytest.mark.asyncio
async def test_one_helper_stopped_the_rest_deliver(monkeypatch):
    srv = FakeServer(helper_kwargs={"Prices": {"hang": True}, "Reviews": {"delay": 0.05}})
    sm = FakeSM()
    import captain_claw.session as session_mod
    import captain_claw.web.chat_handler as ch

    monkeypatch.setattr(session_mod, "get_session_manager", lambda: sm)
    got: list[str] = []

    async def fake_handle_chat(srv_, ws, content, **kw):
        got.append(content)
        return True

    monkeypatch.setattr(ch, "handle_chat", fake_handle_chat)
    team_up.bind_server(srv)
    team, _ = await team_up.propose(srv.agent, "Compare", _parts(2))
    await team_up.start(team)
    await _until(lambda: team.helpers[0].step_count >= 2)
    assert team_up.stop(team, "Prices") == ""
    await _until(lambda: team.state in ("done", "failed"))
    assert "## Prices — stopped by you" in got[0]
    assert "## Reviews — done" in got[0]


@pytest.mark.asyncio
async def test_finishing_turn_waits_for_a_free_lead(server, monkeypatch):
    got: list[str] = []

    async def fake_handle_chat(srv, ws, content, **kw):
        got.append(content)
        return True

    import captain_claw.web.chat_handler as ch

    monkeypatch.setattr(ch, "handle_chat", fake_handle_chat)
    team, _ = await team_up.propose(server.agent, "Compare", _parts(2))
    await team_up.start(team)
    server._busy = True
    await _until(lambda: team.state == "finishing")
    await asyncio.sleep(0.2)
    assert got == []
    server._busy = False
    await _until(lambda: team.state == "done", timeout=4)
    assert got


@pytest.mark.asyncio
async def test_one_live_team_per_lead_and_proposal_replaced(server):
    first, _ = await team_up.propose(server.agent, "First", _parts(2))
    second, _ = await team_up.propose(server.agent, "Second", _parts(2))
    assert first.state == "dismissed" and second.state == "proposed"
    assert team_up.dismiss(second) == ""
    assert team_up.active_for(server.agent) is None
    assert server.agent._team_note_fn is None


def test_sync_frames_and_snapshot_shape():
    team, _a1, _a2 = _helper_agent()
    team.helpers[0].steps.append({"t": 1, "kind": "tool", "text": "x"})
    frames = team_up.sync_frames("A", steps=10)
    assert frames[0]["team"]["helpers"][0]["steps"][0]["text"] == "x"
    snap = team.snapshot()
    assert set(snap) >= {"id", "title", "state", "budget", "used", "helpers", "browser_holder"}


def test_status_text_for_lead_and_helper():
    team, _a1, _a2 = _helper_agent()
    team.helpers[1].state = "done"
    team.helpers[1].result = "B is cheapest"
    text = team_up.status_text(team)
    assert "Reviews — done" in text and "B is cheapest" in text
    mine = team_up.status_text(team, for_helper=team.helpers[0])
    assert "Prices (you)" in mine and "B is cheapest" not in mine


def test_result_message_shape_and_literal():
    team, _a1, _a2 = _helper_agent()
    team.title = "Suppliers'"
    team.started_at = 100.0
    for h in team.helpers:
        h.state = "done"
        h.result = "R" * 100
        h.ended_at = 220.0
    text = team_up.result_message(team)
    assert team_up.is_result_message(text)
    from captain_claw import msg_origin

    assert msg_origin.detect_literal(text) == ("delegated_result", "team_result")
    assert "2 helpers · 2m 00s" in text


def test_team_is_never_deferred_by_tool_search():
    from captain_claw import tool_search

    assert "team" in tool_search.ESSENTIAL_TOOLS


# ── P1: the team board ───────────────────────────────────────────────────

class _Resp:
    def __init__(self, content, usage=None):
        self.content = content
        self.usage = usage or {"prompt_tokens": 900, "completion_tokens": 60}


class LeadProvider:
    """The lead's model for quiet answers: records what it was asked."""

    provider = "anthropic"
    model = "claude-haiku-5-5"
    base_url = ""

    def __init__(self, answer="Use EU suppliers only."):
        self.answer = answer
        self.calls: list[list] = []

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        self.calls.append(messages)
        return _Resp(self.answer)


def _board_team(lead_answer="Use EU suppliers only."):
    srv = FakeServer()
    team_up.bind_server(srv)
    team, a1, a2 = _helper_agent()
    team.lead = srv.agent
    team.lead.provider = LeadProvider(lead_answer)
    team.helpers.append(team_up.Helper(id="h3", name="Delivery", goal="g", state="working"))
    return srv, team


def _board_events(srv):
    return [f["entry"] for f in srv.frames if f.get("type") == "team" and f.get("event") == "board"]


def test_post_and_remember_go_on_the_board():
    srv, team = _board_team()
    prices, reviews = team.helpers[0], team.helpers[1]
    assert team_up.board_post(team, prices, "Acme is €389 list", "Acme price").startswith("Posted #1")
    out = team_up.board_post(team, prices, "remember: Acme quotes need a VAT id")
    assert out.startswith("Noted for memory")
    kinds = [e["kind"] for e in team.board.entries]
    assert kinds == ["post", "remember"]
    assert team.board.entries[1]["text"] == "Acme quotes need a VAT id"
    assert [e["id"] for e in _board_events(srv)] == [1, 2]
    # The others see a pointer once, at their next step.
    note = team_up.helper_note(team, reviews)
    assert "#1 Prices: Acme price" in note
    assert "New on the team board" not in team_up.helper_note(team, reviews)
    assert team_up.board_post(team, prices, "") == "Give the text to post."
    team.state = "done"
    assert "board is closed" in team_up.board_post(team, prices, "late")


@pytest.mark.asyncio
async def test_ask_a_helper_then_answer_shows_to_the_asker():
    srv, team = _board_team()
    prices, reviews = team.helpers[0], team.helpers[1]
    out = await team_up.board_ask(team, prices, "Reviews", "Which supplier has the slowest returns?")
    assert out.startswith("Asked Reviews (#1)")
    assert "#1 Prices asks you: Which supplier" in team_up.helper_note(team, reviews)
    assert "no answer yet" in team_up.helper_note(team, prices)
    assert team_up.board_answer(team, team.helpers[2], 1, "x").startswith("Ask #1 is for Reviews")
    assert team_up.board_answer(team, reviews, 1, "Corvid, 3 weeks").startswith("Answered #1")
    assert team.board.get(1)["answered"] is True
    assert "→ Reviews: Corvid, 3 weeks" in team_up.helper_note(team, prices)
    assert "asks you" not in team_up.helper_note(team, reviews)
    assert any(s["kind"] == "answer" and "Reviews answered" in s["text"] for s in prices.steps)
    assert srv.agent.provider.calls == []  # a working helper answers itself


@pytest.mark.asyncio
async def test_ask_the_lead_is_answered_quietly_from_what_it_knows():
    srv, team = _board_team()
    prices = team.helpers[0]
    team.fork_len = 2
    srv.agent.session.messages.append({"role": "user", "content": "only EU suppliers please", "origin": "human"})
    await team_up.board_ask(team, prices, "lead", "Should I include US suppliers?")
    await _until(lambda: team.board.get(1)["answered"])
    answer = team.board.answers_to(1)[0]
    assert answer["who"] == "lead" and answer["text"] == "Use EU suppliers only."
    sent = srv.agent.provider.calls[0]
    ctx = sent[1].content
    assert "only EU suppliers please" in ctx and "Should I include US suppliers?" in ctx
    assert team.lead_tokens == 960 and team.used() >= 960
    assert "Prices asked you: Should I include US" in team_up.lead_note(team)
    # Nothing went to the chat as a message — only board/step frames.
    assert not [f for f in srv.frames if f.get("type") == "chat_message"]


@pytest.mark.asyncio
async def test_ask_to_a_finished_helper_or_anyone_alone_goes_to_the_lead():
    srv, team = _board_team(lead_answer="Reviews found Borealis best.")
    prices, reviews, delivery = team.helpers
    reviews.state = "done"
    reviews.result = "Borealis 4.6, Acme 4.4"
    await team_up.board_ask(team, prices, "Reviews", "Who had the best reviews?")
    await _until(lambda: team.board.get(1)["answered"])
    assert "who has finished" in srv.agent.provider.calls[0][1].content
    delivery.state = "done"
    await team_up.board_ask(team, prices, "anyone", "Any data on Corvid support?")
    await _until(lambda: team.board.get(3)["answered"])


@pytest.mark.asyncio
async def test_open_asks_go_to_the_lead_when_their_helper_finishes():
    srv, team = _board_team()
    prices, reviews = team.helpers[0], team.helpers[1]
    await team_up.board_ask(team, prices, "Reviews", "Return times?")
    assert srv.agent.provider.calls == []
    reviews.state = "done"
    team_up._route_open_asks(team, reviews)
    team_up._route_open_asks(team, reviews)  # once only
    await _until(lambda: team.board.get(1)["answered"])
    await asyncio.sleep(0.05)
    assert len(srv.agent.provider.calls) == 1


@pytest.mark.asyncio
async def test_ask_limits_and_mistakes():
    srv, team = _board_team()
    prices = team.helpers[0]
    assert "That's you" in await team_up.board_ask(team, prices, "Prices", "x")
    assert "No helper called" in await team_up.board_ask(team, prices, "Nobody", "x")
    await team_up.board_ask(team, prices, "Reviews", "Return times?")
    assert "already asked" in await team_up.board_ask(team, prices, "reviews", "return times")
    team.board.asks = team_board_max = __import__("captain_claw.team_board", fromlist=["x"]).MAX_ASKS
    assert team_board_max == 12
    assert "used all its asks" in await team_up.board_ask(team, prices, "Reviews", "Another?")


@pytest.mark.asyncio
async def test_inbox_waits_for_an_answer():
    srv, team = _board_team()
    prices, reviews = team.helpers[0], team.helpers[1]
    await team_up.board_ask(team, prices, "Reviews", "Return times?")

    async def answer_soon():
        await asyncio.sleep(0.1)
        team_up.board_answer(team, reviews, 1, "Two weeks")

    asyncio.create_task(answer_soon())
    t0 = asyncio.get_event_loop().time()
    out = await team_up.board_inbox(team, prices, wait=10)
    assert "→ Reviews: Two weeks" in out
    assert asyncio.get_event_loop().time() - t0 < 2


@pytest.mark.asyncio
async def test_wait_for_a_helper_or_a_post():
    srv, team = _board_team()
    prices, reviews = team.helpers[0], team.helpers[1]

    async def finish_soon():
        await asyncio.sleep(0.1)
        reviews.state = "done"
        reviews.result = "Borealis wins"
        team.board._pulse()

    asyncio.create_task(finish_soon())
    out = await team_up.board_wait(team, prices, helper_ref="Reviews", wait=10)
    assert "Reviews is done" in out and "Borealis wins" in out

    async def post_soon():
        await asyncio.sleep(0.1)
        team_up.board_post(team, team.helpers[2], "Corvid delivers to Milan in 2 days", "Milan delivery")

    asyncio.create_task(post_soon())
    out = await team_up.board_wait(team, prices, query="Milan delivery", wait=10)
    assert "Found on the board" in out and "Corvid delivers" in out
    assert "can't wait for yourself" in await team_up.board_wait(team, prices, helper_ref="Prices")


def test_read_search_map():
    srv, team = _board_team()
    prices, reviews = team.helpers[0], team.helpers[1]
    team_up.board_post(team, prices, "Acme €389, net 30", "Acme price")
    team_up.board_post(team, reviews, "Borealis 4.6 stars", "Borealis reviews")
    assert "Borealis 4.6" in team_up.board_read(team, "Reviews")
    assert "Acme" not in team_up.board_read(team, "Reviews")
    assert "Acme price" in team_up.board_search(team, "acme")
    assert team_up.board_search(team, "acme", for_helper=prices).startswith("Nothing")
    m = team_up.board_map(team, for_helper=prices)
    assert "Prices (you)" in m and "Latest posts" in m and "Borealis reviews" in m
    assert "kind is one of" in team_up.board_read(team, kind="gossip")


def test_result_message_lists_remember_notes_and_unanswered_asks():
    srv, team = _board_team()
    prices, reviews = team.helpers[0], team.helpers[1]
    team_up.board_post(team, prices, "remember: our Milan office prefers Corvid")
    team.board.add("ask", "h2", "Reviews", "Is Acme's warranty 2 or 3 years?", to="h1", to_name="Prices")
    for h in team.helpers:
        h.state, h.result, h.ended_at = "done", "ok", 10.0
    text = team_up.result_message(team)
    assert "## Worth remembering" in text and "our Milan office prefers Corvid" in text
    assert "Unanswered: Reviews asked Prices: Is Acme" in text


def test_guard_lets_helpers_use_the_board_only():
    _team, a1, _a2 = _helper_agent()
    for act in ("post", "ask", "answer", "inbox", "wait", "read", "search", "map", "status"):
        assert team_up.tool_block("team", {"action": act}, a1) == "", act
    for act in ("propose", "start", "tell", "stop"):
        assert team_up.tool_block("team", {"action": act}, a1), act
    assert "remember:" in team_up.tool_block("facts", {"action": "add"}, a1)


@pytest.mark.asyncio
async def test_team_tool_routes_helper_and_lead_verbs():
    from captain_claw.tools.team import TeamTool

    srv, team = _board_team()
    prices = team.helpers[0]
    agent = FakeAgent()
    agent._team_helper = (team.id, prices.id)
    tool = TeamTool()
    r = await tool.execute(action="post", text="Acme €389", title="Acme", _agent=agent)
    assert r.success and "Posted #1" in r.content
    r = await tool.execute(action="ask", to="Reviews", text="Returns?", _agent=agent)
    assert r.success and "Asked Reviews" in r.content
    r = await tool.execute(action="propose", _agent=agent)
    assert not r.success
    team.lead = srv.agent
    r = await tool.execute(action="read", _agent=srv.agent)
    assert r.success and "Acme" in r.content
    r = await tool.execute(action="post", text="x", _agent=srv.agent)
    assert not r.success and "for your helpers" in r.error
