"""Insert now — a waiting queue task handed to the RUNNING turn.

The chat sends it as a `btw` note. The reply says whether a turn was on to
take it (`busy`). A note marked `running_only` is for that turn only: with
none in flight it is refused, not kept for the next turn, so the client runs
the task on its own instead. A plain note (the council's context, sent right
before its first turn) still waits for the next turn, as it always has.
"""

import types

from captain_claw.web import ws_handler
from captain_claw.web_server import WebServer


class _Server:
    LANE_MAIN = WebServer.LANE_MAIN
    normalize_lane = staticmethod(WebServer.normalize_lane)

    def __init__(self, agent, busy=False):
        self.agent = agent
        self._busy = busy
        self.sent: list[dict] = []

    async def resolve_agent(self, ws):
        return self.agent

    async def _send(self, ws, msg):
        self.sent.append(msg)


def _notes(agent) -> list[str]:
    return list(getattr(agent, "_btw_instructions", None) or [])


async def _btw(server, ws, content="also check the footer", **extra):
    await ws_handler.handle_ws_message(server, ws, {"type": "btw", "content": content, **extra})
    return server.sent


# ── the reply says whether a turn took the note ──

async def test_a_running_turn_takes_the_note():
    agent = types.SimpleNamespace()
    server = _Server(agent, busy=True)
    sent = await _btw(server, types.SimpleNamespace(), running_only=True)
    assert _notes(agent) == ["also check the footer"]
    assert len(sent) == 1
    assert sent[0]["type"] == "command_result" and sent[0]["command"] == "/btw"
    assert sent[0]["busy"] is True
    assert "also check the footer" in sent[0]["content"]


async def test_with_no_turn_on_a_running_only_note_is_refused_not_kept():
    agent = types.SimpleNamespace()
    server = _Server(agent, busy=False)
    sent = await _btw(server, types.SimpleNamespace(), running_only=True)
    assert _notes(agent) == []                      # nothing for the next turn
    assert len(sent) == 1
    assert sent[0]["command"] == "/btw" and sent[0]["busy"] is False
    assert "Nothing is running" in sent[0]["content"]


async def test_a_plain_note_still_waits_for_the_next_turn():
    """The council sends its context this way, right before its first turn."""
    agent = types.SimpleNamespace()
    server = _Server(agent, busy=False)
    sent = await _btw(server, types.SimpleNamespace())
    assert _notes(agent) == ["also check the footer"]
    assert sent[0]["busy"] is False                 # kept, but no turn took it yet


async def test_notes_accumulate_on_the_running_turn():
    agent = types.SimpleNamespace()
    server = _Server(agent, busy=True)
    await _btw(server, types.SimpleNamespace(), "one", running_only=True)
    await _btw(server, types.SimpleNamespace(), "two", running_only=True)
    assert _notes(agent) == ["one", "two"]


async def test_an_empty_note_is_ignored():
    agent = types.SimpleNamespace()
    server = _Server(agent, busy=True)
    assert await _btw(server, types.SimpleNamespace(), "   ", running_only=True) == []
    assert _notes(agent) == []


# ── whose turn: the lane's own busy flag, the server's on the main lane ──

def test_turn_in_flight_reads_the_right_flag():
    main = types.SimpleNamespace()
    assert ws_handler.turn_in_flight(_Server(main, busy=True), types.SimpleNamespace(), main) is True
    assert ws_handler.turn_in_flight(_Server(main, busy=False), types.SimpleNamespace(), main) is False
    # Lane A is the main agent: the server's flag, whatever the agent carries.
    main_flagged = types.SimpleNamespace(_lane_busy=True)
    assert ws_handler.turn_in_flight(_Server(main_flagged, busy=False), types.SimpleNamespace(_lane="A"), main_flagged) is False

    lane_b = types.SimpleNamespace(_lane_busy=True)
    assert ws_handler.turn_in_flight(_Server(lane_b, busy=False), types.SimpleNamespace(_lane="B"), lane_b) is True
    lane_b_idle = types.SimpleNamespace(_lane_busy=False)
    assert ws_handler.turn_in_flight(_Server(lane_b_idle, busy=True), types.SimpleNamespace(_lane="b"), lane_b_idle) is False

    pub = types.SimpleNamespace(_public_busy=True)
    assert ws_handler.turn_in_flight(_Server(pub, busy=False), types.SimpleNamespace(_public_session_id="s1"), pub) is True
    pub_idle = types.SimpleNamespace()
    assert ws_handler.turn_in_flight(_Server(pub_idle, busy=True), types.SimpleNamespace(_public_session_id="s1"), pub_idle) is False


async def test_a_side_lane_goes_by_its_own_turn():
    lane_b = types.SimpleNamespace(_lane_busy=True)
    server = _Server(lane_b, busy=False)             # lane A idle, B running
    sent = await _btw(server, types.SimpleNamespace(_lane="B"), running_only=True)
    assert _notes(lane_b) == ["also check the footer"]
    assert sent[0]["busy"] is True
