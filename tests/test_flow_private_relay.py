"""PR D: a flow's member-private level on the AGENT side of a flow relay.

Flight Deck keeps the agent-facing privacy header out of flow output a person
reads and reports the level instead (``member_private``). When a flow is
triggered by a message the agent received — a consult or delegate from
another agent among them — the agent relays the flow output on that socket:
inline (``/fd/flows/evaluate``) as its reply frame, deferred through
``/api/chat/push``. Either frame must carry the level, so FD's consult /
delegate relay can put the header back in front for the receiving agent; the
text itself stays plain.
"""

from __future__ import annotations

import json
import types

import httpx
import pytest

from captain_claw.web import chat_handler


class _Resp:
    status_code = 200

    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def _fd_answers(monkeypatch, data: dict, posts: list):
    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, **kw):
            posts.append((url, json))
            return _Resp(data)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setenv("FD_URL", "http://fd.test")


def _agent():
    return types.SimpleNamespace(session=types.SimpleNamespace(metadata={}))


@pytest.mark.parametrize("level", ["content", "data"])
async def test_evaluate_level_is_passed_on(monkeypatch, level):
    posts: list = []
    _fd_answers(monkeypatch, {"matched": True, "output": "Mia: invoices", "member_private": level}, posts)
    got = await chat_handler._maybe_run_flow(_agent(), "digest", is_public=False)
    assert posts and posts[0][0] == "http://fd.test/fd/flows/evaluate"
    assert got == {"output": "Mia: invoices", "member_private": level}


@pytest.mark.parametrize("extra", [{}, {"member_private": "yes"}, {"member_private": None}])
async def test_unmarked_evaluate_is_unchanged(monkeypatch, extra):
    _fd_answers(monkeypatch, {"matched": True, "output": "plain", **extra}, [])
    assert await chat_handler._maybe_run_flow(_agent(), "digest", is_public=False) == {"output": "plain"}


@pytest.mark.parametrize("level", ["content", "data", ""])
async def test_inline_flow_reply_frame_carries_the_level(monkeypatch, level):
    """The relayed flow output is the turn's reply frame (what a consult or
    delegate waits for): plain text, ``member_private`` when FD said so."""
    frames: list = []
    monkeypatch.setattr(chat_handler, "fire_and_forget_send", lambda ws, data: frames.append(json.loads(data)))

    async def _flow(agent, text, **kw):
        return {"output": "Mia: invoices", **({"member_private": level} if level else {})}

    monkeypatch.setattr(chat_handler, "_maybe_run_flow", _flow)
    server = types.SimpleNamespace(LANE_MAIN="A", _busy=True, _active_task=None)
    agent = types.SimpleNamespace(get_runtime_model_details=lambda: {})
    await chat_handler._run_agent(server, object(), agent, "digest", no_broadcast=True)
    replies = [f for f in frames if f.get("type") == "chat_message"]
    assert len(replies) == 1
    assert replies[0]["content"] == "Mia: invoices"
    assert replies[0].get("member_private") == (level or None)
    if not level:
        assert "member_private" not in replies[0]


@pytest.mark.parametrize("sent,expected", [("content", "content"), ("data", "data"),
                                           ("yes", None), (None, None)])
async def test_chat_push_frame_carries_the_level(monkeypatch, sent, expected):
    """A deferred flow's delivery (``/api/chat/push``) — broadcast to every
    socket, a waiting consult's included — keeps the level on the frame."""
    from captain_claw.web_server import WebServer

    frames: list = []
    fake = types.SimpleNamespace(_broadcast=frames.append)

    class _Req:
        async def json(self):
            body = {"text": "Mia: invoices"}
            if sent is not None:
                body["member_private"] = sent
            return body

    monkeypatch.setattr("captain_claw.config.get_config",
                        lambda: types.SimpleNamespace(web=types.SimpleNamespace(auth_token="")))
    resp = await WebServer._api_chat_push(fake, _Req())
    assert resp.status == 200
    assert len(frames) == 1 and frames[0]["content"] == "Mia: invoices"
    assert frames[0].get("member_private") == expected
    if expected is None:
        assert "member_private" not in frames[0]
