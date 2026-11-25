"""Pictures on a borrowed call: ``POST /api/llm/complete`` with ``images``.

WhatsApp Web auto-replies borrow the agent's model, and a group asked it about
a photo it never got: the endpoint took text only. A message's pictures now go
inline to a model that sees, are described by the agent's vision fallback for
one that doesn't, and say so when nothing here can look at them.
"""

from __future__ import annotations

import base64
import json
import types
from pathlib import Path

import pytest

from captain_claw.llm import LLMResponse
from captain_claw.web import borrowed_pictures as bp

# A 1×1 PNG.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
B64 = base64.b64encode(PNG).decode("ascii")


def _pic(label="picture 1", mime="image/png", data=B64):
    return {"label": label, "mime": mime, "data": data}


# ── taking them ─────────────────────────────────────────────────────


def test_take_reads_each_messages_pictures():
    found = bp.take([{"role": "system", "content": "x"},
                     {"role": "user", "content": "y", "images": [_pic(), _pic("picture 2", "image/jpeg")]}])
    assert list(found) == [1]
    assert [(p.label, p.ext, p.data) for p in found[1]] == [("picture 1", ".png", PNG), ("picture 2", ".jpg", PNG)]
    assert bp.take([{"role": "user", "content": "y"}]) == {}


@pytest.mark.parametrize("images, why", [
    ([_pic()] * (bp.MAX_PICTURES + 1), "at most"),
    ([_pic(mime="application/pdf")], "not a picture"),
    ([_pic(mime="")], "not a picture"),
    ([_pic(data="not base64!")], "base64"),
    ([_pic(data="")], "empty"),
    ("x", "must be a list"),
    (["x"], "a picture is"),
])
def test_bad_pictures_are_refused(images, why):
    with pytest.raises(bp.BadPictures, match=why):
        bp.take([{"role": "user", "content": "y", "images": images}])


def test_a_picture_too_large_is_refused(monkeypatch):
    monkeypatch.setattr(bp, "MAX_PICTURE_BYTES", 10)
    with pytest.raises(bp.BadPictures, match="too large"):
        bp.take([{"role": "user", "content": "y", "images": [_pic(data=base64.b64encode(b"x" * 64).decode())]}])


# ── the endpoint ────────────────────────────────────────────────────


class _Req:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def _server(monkeypatch, *, sees: bool):
    from captain_claw import anonymize_gate_client
    from captain_claw.web_server import WebServer

    seen: list = []

    class Provider:
        provider, model, base_url = "openai", "m", ""

        async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
            # What the model gets, and whether its files are there while it looks.
            seen.append([(m.role, m.content, list(m.images or []),
                          all(Path(p).is_file() for p in m.images or [])) for m in messages])
            return LLMResponse(content="ok", model="m")

    async def no_gate(agent):
        return None

    async def refused_fallback(paths):
        return "a description"

    provider = Provider()
    server = WebServer.__new__(WebServer)
    server.agent = types.SimpleNamespace(provider=provider, session=None, sees_images=lambda: sees,
                                         _describe_refused_images=refused_fallback)
    monkeypatch.setattr(WebServer, "_scoped_provider", lambda self: self.agent.provider, raising=False)
    monkeypatch.setattr(WebServer, "_at_work", lambda self: False)
    monkeypatch.setattr(anonymize_gate_client, "decision", no_gate)
    return server, seen, provider


_CALL = [{"role": "system", "content": "rules"},
         {"role": "user", "content": "[picture 1] what lighthouse is this?", "images": [_pic()]}]


@pytest.mark.asyncio
async def test_a_model_that_sees_gets_the_pictures_inline(monkeypatch):
    server, seen, provider = _server(monkeypatch, sees=True)
    resp = await server._llm_complete(_Req({"messages": _CALL}))
    body = json.loads(resp.text)
    assert body["ok"] and body["pictures"] == {"count": 1, "seen": bp.INLINE}
    (call,) = seen
    assert call[0] == ("system", "rules", [], True)
    role, content, images, there = call[1]
    assert role == "user" and len(images) == 1 and there and images[0].endswith(".png")
    # The label leads its marker, so a refusal can still have it described.
    assert content.startswith("[picture 1] what lighthouse is this?") and f"picture 1: [Attached image: {images[0]}]" in content
    assert provider.sees_images is True and callable(provider.describe_refused_images)
    # Gone once the call is over.
    assert not Path(images[0]).exists() and not Path(images[0]).parent.exists()


@pytest.mark.asyncio
async def test_a_model_that_doesnt_gets_a_description(monkeypatch):
    from captain_claw.web import chat_handler

    asked = []

    async def describe(agent, paths, prompt, send=None):
        asked.append((len(paths), prompt))
        return [(p, "A red-and-white lighthouse on a rocky cape, likely a coastal light.") for p in paths]

    monkeypatch.setattr(chat_handler, "describe_images_without_vision", describe)
    server, seen, _ = _server(monkeypatch, sees=False)
    body = json.loads((await server._llm_complete(_Req({"messages": _CALL}))).text)
    assert body["pictures"] == {"count": 1, "seen": bp.DESCRIBED}
    _role, content, images, _there = seen[0][1]
    assert images == [] and "[Attached image:" not in content  # no marker for Ollama's legacy scan
    assert "[picture 1 — described by a vision model, you don't see it yourself]" in content
    assert "likely a coastal light" in content and asked == [(1, bp.DESCRIBE_PROMPT)]


@pytest.mark.asyncio
async def test_nothing_to_see_with_says_so(monkeypatch):
    from captain_claw.web import chat_handler

    async def describe(agent, paths, prompt, send=None):
        return [(p, None) for p in paths]

    monkeypatch.setattr(chat_handler, "describe_images_without_vision", describe)
    server, seen, _ = _server(monkeypatch, sees=False)
    body = json.loads((await server._llm_complete(_Req({"messages": _CALL}))).text)
    assert body["pictures"] == {"count": 1, "seen": bp.UNSEEN}
    assert "[picture 1] (you can't see this picture" in seen[0][1][1]
    # A failed analysis is no description either.
    async def failed(agent, paths, prompt, send=None):
        return [(p, "(automatic analysis failed: timeout)") for p in paths]

    monkeypatch.setattr(chat_handler, "describe_images_without_vision", failed)
    body = json.loads((await server._llm_complete(_Req({"messages": _CALL}))).text)
    assert body["pictures"]["seen"] == bp.UNSEEN


@pytest.mark.asyncio
async def test_a_call_without_pictures_is_unchanged(monkeypatch):
    server, seen, provider = _server(monkeypatch, sees=True)
    body = json.loads((await server._llm_complete(_Req({"messages": [{"role": "user", "content": "hi"}]}))).text)
    assert body["ok"] and "pictures" not in body
    assert seen == [[("user", "hi", [], True)]] and not hasattr(provider, "sees_images")


@pytest.mark.asyncio
async def test_bad_pictures_are_a_400_and_the_model_isnt_called(monkeypatch):
    server, seen, _ = _server(monkeypatch, sees=True)
    resp = await server._llm_complete(_Req({"messages": [{"role": "user", "content": "x",
                                                          "images": [_pic(mime="text/html")]}]}))
    assert resp.status == 400 and "not a picture" in json.loads(resp.text)["error"] and seen == []


@pytest.mark.asyncio
async def test_the_files_go_when_the_model_fails(monkeypatch):
    server, _seen, provider = _server(monkeypatch, sees=True)
    made = []

    async def boom(messages, tools=None, temperature=None, max_tokens=None):
        made.extend(messages[1].images)
        raise RuntimeError("server down")

    provider.complete = boom
    resp = await server._llm_complete(_Req({"messages": _CALL}))
    assert resp.status == 500 and made and not Path(made[0]).exists()
