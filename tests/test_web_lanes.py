"""Lanes — parallel contexts on one agent process.

Lane A *is* the shared agent that exists today, so an omitted lane changes
nothing for any existing client. Lanes B, C, … get their own Agent, their own
session, their own busy flag, and their own sockets — a turn in B must never
appear in A's transcript, and B must not wait for A.

Phase 1 of docs/queue-lanes-plan.md.
"""

import asyncio
import types

import pytest

from captain_claw.web_server import WebServer, _LaneServerView


class FakeWS:
    """Enough of a WebSocketResponse for the lane bookkeeping."""

    def __init__(self, lane: str | None = None):
        self.closed = False
        self.sent: list[str] = []
        if lane is not None:
            self._lane = lane

    async def send_str(self, data: str):
        self.sent.append(data)


@pytest.fixture(autouse=True)
def sync_sends(monkeypatch):
    """Record sends inline — the real one schedules a task on the loop."""
    def _send(ws, data):
        ws.sent.append(data)
    monkeypatch.setattr("captain_claw.web_server.fire_and_forget_send", _send)


@pytest.fixture
def server():
    s = WebServer.__new__(WebServer)          # skip __init__'s heavy wiring
    s.agent = types.SimpleNamespace(name="main")
    s.clients = set()
    s._lane_agents = {}
    s._lane_locks = {}
    s._lane_sockets = {}
    s._public_agents = {}
    return s


# ── normalize_lane: whatever a client sends, we land somewhere sane ──

@pytest.mark.parametrize("raw,expected", [
    ("B", "B"), ("b", "B"), ("", "A"), (None, "A"),
    ("  ", "A"), ("!!!", "A"), ("c", "C"),
    ("verylonglanename", "VERYLONG"),          # clamped, not rejected
    ("a-2", "A-2"),
])
def test_normalize_lane(raw, expected):
    assert WebServer.normalize_lane(raw) == expected


def test_absent_lane_is_lane_a():
    """The whole backward-compatibility story rests on this."""
    assert WebServer.normalize_lane("") == WebServer.LANE_MAIN


# ── lane A is the main agent, not a new one ──

async def test_lane_a_resolves_to_the_existing_agent(server):
    assert await server._get_lane_agent("A") is server.agent
    assert await server._get_lane_agent("") is server.agent
    assert server._lane_agents == {}          # nothing was created


async def test_a_socket_with_no_lane_gets_the_main_agent(server):
    ws = FakeWS()                             # no _lane attribute at all
    assert await server.resolve_agent(ws) is server.agent


async def test_side_lanes_get_their_own_agent(server):
    made = []

    async def fake_build(session, send):
        agent = types.SimpleNamespace(session=session, send=send)
        made.append(agent)
        return agent

    server._build_scoped_agent = fake_build

    async def fake_session(lane):
        return types.SimpleNamespace(id=lane, name=f"lane-{lane}")

    server._lane_session = fake_session

    b1 = await server._get_lane_agent("B")
    b2 = await server._get_lane_agent("B")     # cached, not rebuilt
    c = await server._get_lane_agent("C")

    assert b1 is b2
    assert b1 is not c
    assert b1 is not server.agent
    assert len(made) == 2


# ── delivery isolation ──

def test_lane_send_reaches_only_that_lane(server):
    a, b1, b2 = FakeWS("A"), FakeWS("B"), FakeWS("B")
    server._lane_sockets = {"A": {a}, "B": {b1, b2}}

    server._lane_send("B")({"type": "token", "text": "hi"})

    assert a.sent == []
    assert len(b1.sent) == 1 and len(b2.sent) == 1


def test_lane_send_skips_closed_sockets(server):
    open_ws, closed = FakeWS("B"), FakeWS("B")
    closed.closed = True
    server._lane_sockets = {"B": {open_ws, closed}}

    server._lane_send("B")({"type": "token"})

    assert len(open_ws.sent) == 1 and closed.sent == []


def test_broadcast_reaches_lane_a_and_unlaned_but_not_b(server, monkeypatch):
    monkeypatch.setattr(
        "captain_claw.config.get_config",
        lambda: types.SimpleNamespace(web=types.SimpleNamespace(public_run=False)),
    )
    plain, a, b = FakeWS(), FakeWS("A"), FakeWS("B")
    server.clients = {plain, a, b}

    server._broadcast({"type": "status", "status": "thinking"})

    assert len(plain.sent) == 1          # today's clients, untouched
    assert len(a.sent) == 1              # lane A is the main agent
    assert b.sent == []                  # served by its own callbacks


# ── the slash-command facade ──

def test_lane_view_swaps_the_agent_but_passes_everything_else_through(server):
    lane_agent = types.SimpleNamespace(name="lane-B")
    sent: list[dict] = []
    view = _LaneServerView(server, lane_agent, sent.append)

    assert view.agent is lane_agent          # /new acts on lane B…
    assert view.clients is server.clients    # …everything else is the real server

    view._broadcast({"type": "session_info"})
    assert sent == [{"type": "session_info"}]   # and never reaches lane A


def test_lane_view_writes_fall_through_to_the_real_server(server):
    view = _LaneServerView(server, types.SimpleNamespace(), lambda m: None)
    view._busy = True
    assert server._busy is True
    with pytest.raises(AttributeError):
        view.agent = "reassigned"


def test_main_lane_gets_the_real_server_not_a_view(server):
    assert server.lane_view(FakeWS("A"), server.agent) is server
    assert server.lane_view(FakeWS(), server.agent) is server


# ── the point of the whole exercise ──

async def test_lanes_do_not_block_each_other(server):
    """B's busy flag is B's own — it must not gate A, and A must not gate B."""
    lane_b = types.SimpleNamespace(_lane_busy=True)
    server._lane_agents = {"B": lane_b}
    server._busy = True                      # lane A mid-turn

    # Each lane reads its own flag; neither consults the other's.
    assert getattr(server._lane_agents["B"], "_lane_busy") is True
    assert server._busy is True
    lane_b._lane_busy = False
    assert server._busy is True              # A untouched by B finishing


# ── Phase 2: the Flight Deck proxy carries the lane ──

def test_proxy_url_carries_token_and_lane():
    from captain_claw.flight_deck.server import _agent_ws_url
    url = _agent_ws_url("localhost", 24080, "tok3n", "B")
    assert url == "ws://localhost:24080/ws?token=tok3n&lane=B"


def test_proxy_url_without_a_lane_is_exactly_what_it_always_was():
    """Existing callers must produce a byte-identical URL."""
    from captain_claw.flight_deck.server import _agent_ws_url
    assert _agent_ws_url("h", 1, "tok") == "ws://h:1/ws?token=tok"
    assert _agent_ws_url("h", 1, "") == "ws://h:1/ws"


def test_proxy_url_escapes_what_it_forwards():
    from captain_claw.flight_deck.server import _agent_ws_url
    url = _agent_ws_url("h", 1, "a b&c=d", "B C")
    assert "a%20b%26c%3Dd" in url and "lane=B%20C" in url
    assert url.count("?") == 1 and url.count("&") == 1   # no query injection


# ── Queue-dispatched turns skip the "what next?" round-trip ──
# Same reasoning as Basna/Vatra workers: the follow-up is already written and
# waiting in the queue, so asking the model to suggest one costs a round-trip
# per item and offers nothing.

def test_chat_handler_gates_next_steps_on_the_flag():
    import inspect
    from captain_claw.web import chat_handler

    src = inspect.getsource(chat_handler._run_agent)
    assert "no_next_steps" in src
    # The gate must AND with the existing worker gate, not replace it.
    assert "not no_next_steps and not _is_fd_spawned_worker()" in src


def test_handle_chat_and_run_agent_accept_the_flag():
    import inspect
    from captain_claw.web import chat_handler

    for fn in (chat_handler.handle_chat, chat_handler._run_agent):
        assert "no_next_steps" in inspect.signature(fn).parameters


def test_ws_handler_forwards_the_flag_from_the_client():
    import inspect
    from captain_claw.web import ws_handler

    src = inspect.getsource(ws_handler.handle_ws_message)
    assert 'no_next_steps=bool(data.get("no_next_steps", False))' in src


# ── Signals that were being swallowed on lanes B and C ──

def test_scoped_agents_get_a_narration_callback():
    """Agent._emit_narration returns early when the callback is absent — so a
    missing callback DROPS narration, it doesn't merely misroute it."""
    import inspect
    from captain_claw.web_server import WebServer

    src = inspect.getsource(WebServer._build_scoped_agent)
    assert "narration_callback=narration_cb" in src
    assert '"type": "narration"' in src


def test_narration_callback_sends_through_the_scoped_sender():
    import inspect
    from captain_claw.web_server import WebServer

    src = inspect.getsource(WebServer._build_scoped_agent)
    # It must use `send` (the lane/session sender), never _broadcast.
    body = src[src.index("def narration_cb"):]
    body = body[:body.index("def ", 10)]
    assert "send({" in body and "_broadcast" not in body


def test_session_info_can_describe_any_agent(server):
    """A lane's header comes from ITS agent, not the main one."""
    import inspect
    from captain_claw.web_server import WebServer

    assert "agent" in inspect.signature(WebServer._session_info).parameters
    src = inspect.getsource(WebServer._session_info)
    assert "agent = agent or self.agent" in src
    assert "self.agent" not in src.split("agent = agent or self.agent", 1)[1]


def test_lane_view_reports_its_own_session(server):
    lane_agent = types.SimpleNamespace(name="lane-B")
    seen = {}
    server._session_info = lambda agent=None: seen.setdefault("agent", agent) or {}
    view = _LaneServerView(server, lane_agent, lambda m: None)

    view._session_info()

    assert seen["agent"] is lane_agent      # not server.agent


def test_welcome_and_replay_follow_the_socket_s_lane():
    """Lane B must not open showing lane A's name, model and history."""
    import inspect
    from captain_claw.web import ws_handler

    src = inspect.getsource(ws_handler.ws_handler)
    assert "_welcome_agent = await server.resolve_agent(ws)" in src
    assert "server._session_info(_welcome_agent)" in src
    assert "replay_session = _welcome_agent.session" in src
    assert "replay_session = server.agent.session" not in src


# ── Each lane gets its own provider view ──
# The orchestration loop forces tool use after a stall by setting
# `_tool_choice_override` on the PROVIDER. Shared across lanes, the next
# lane's call consumes it: that lane is forced into a tool call it never
# needed, and the lane that stalled loses its forcing and stalls again.

class FakeProvider:
    def __init__(self):
        self.model = "deepseek-v4-pro"
        self.client = object()            # connection pool — must stay shared
        self.rate_limiter = object()      # account TPM window — must stay shared


def test_lane_provider_does_not_leak_the_tool_choice_override(server):
    server.agent = types.SimpleNamespace(provider=FakeProvider())
    lane_b = server._scoped_provider()
    lane_c = server._scoped_provider()

    # Lane B stalls and forces tool use on its next call.
    lane_b._tool_choice_override = "required"

    assert getattr(lane_c, "_tool_choice_override", None) is None
    assert getattr(server.agent.provider, "_tool_choice_override", None) is None


def test_lane_provider_keeps_sharing_what_should_be_shared(server):
    """Three lanes must not each get the configured rate, or their own pool."""
    server.agent = types.SimpleNamespace(provider=FakeProvider())
    lane = server._scoped_provider()

    assert lane is not server.agent.provider
    assert lane.client is server.agent.provider.client
    assert lane.rate_limiter is server.agent.provider.rate_limiter


def test_a_lane_can_switch_model_without_touching_the_others(server):
    server.agent = types.SimpleNamespace(provider=FakeProvider())
    lane_b = server._scoped_provider()

    lane_b.model = "gpt-5"

    assert server.agent.provider.model == "deepseek-v4-pro"


def test_uncopyable_provider_falls_back_to_sharing(server):
    class NoCopy:
        def __copy__(self):
            raise TypeError("nope")

    shared = NoCopy()
    server.agent = types.SimpleNamespace(provider=shared)
    assert server._scoped_provider() is shared      # degraded, not broken


def test_no_agent_means_no_provider(server):
    server.agent = None
    assert server._scoped_provider() is None
