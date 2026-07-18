"""Browser inference broker + BrowserProvider (mrav Phase 2)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from captain_claw.exceptions import LLMAPIError
from captain_claw.flight_deck.infer_broker import (
    InferBroker,
    InferJobError,
    NoWorkerError,
)
from captain_claw.llm import BrowserProvider, Message


class FakeSocket:
    def __init__(self):
        self.sent: list[dict] = []

    async def send(self, msg: dict) -> None:
        self.sent.append(msg)


def _register(broker: InferBroker, owner: str = "u1", **caps):
    sock = FakeSocket()
    defaults = {"engine": "webllm", "model": "qwen3-4b", "ctx_max": 8192, "schema": True}
    worker_id = broker.register(owner, {**defaults, **caps}, sock.send)
    return worker_id, sock


async def _wait_for_job(sock: FakeSocket) -> dict:
    for _ in range(100):
        if sock.sent:
            return sock.sent[-1]
        await asyncio.sleep(0.001)
    raise AssertionError("worker never received a job")


# ── broker ───────────────────────────────────────────────────────────


def test_register_status_and_owner_isolation():
    broker = InferBroker()
    wid, _ = _register(broker, "alice")
    _register(broker, "bob", model="tiny")

    alice = broker.status("alice")
    assert len(alice) == 1 and alice[0]["worker_id"] == wid
    assert alice[0]["model"] == "qwen3-4b" and alice[0]["schema"] is True
    assert len(broker.status("bob")) == 1
    assert broker.status("carol") == []
    assert broker.owner_ids() == ["alice", "bob"]


@pytest.mark.asyncio
async def test_submit_roundtrip_with_schema_and_pin():
    broker = InferBroker()
    wid, sock = _register(broker, "u1")

    task = asyncio.create_task(broker.submit(
        "u1",
        [{"role": "user", "content": "hi"}],
        response_schema={"type": "object"},
        max_tokens=99,
        session_key="sess-a",
    ))
    job = await _wait_for_job(sock)
    assert job["type"] == "job" and job["max_tokens"] == 99
    assert job["response_schema"] == {"type": "object"}

    broker.handle_message(wid, {
        "type": "result", "job_id": job["job_id"],
        "content": "hello", "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    })
    result = await task
    assert result["content"] == "hello"
    assert result["usage"]["prompt_tokens"] == 7
    assert result["model"] == "qwen3-4b"  # falls back to worker model
    assert broker._pins[("u1", "sess-a")] == wid  # pinned for KV affinity


@pytest.mark.asyncio
async def test_submit_without_worker_and_owner_isolation():
    broker = InferBroker()
    _register(broker, "someone-else")
    with pytest.raises(NoWorkerError):
        await broker.submit("u1", [{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_worker_error_propagates():
    broker = InferBroker()
    wid, sock = _register(broker, "u1")
    task = asyncio.create_task(broker.submit("u1", [{"role": "user", "content": "x"}]))
    job = await _wait_for_job(sock)
    broker.handle_message(wid, {"type": "error", "job_id": job["job_id"], "message": "OOM"})
    with pytest.raises(InferJobError, match="OOM"):
        await task


@pytest.mark.asyncio
async def test_unregister_fails_inflight_jobs():
    broker = InferBroker()
    wid, sock = _register(broker, "u1")
    task = asyncio.create_task(broker.submit("u1", [{"role": "user", "content": "x"}]))
    await _wait_for_job(sock)
    broker.unregister(wid)
    with pytest.raises(InferJobError, match="disconnected"):
        await task
    assert broker.status("u1") == []


@pytest.mark.asyncio
async def test_pin_falls_over_when_worker_dies():
    broker = InferBroker()
    wid1, sock1 = _register(broker, "u1")
    wid2, sock2 = _register(broker, "u1", model="second")

    task = asyncio.create_task(broker.submit(
        "u1", [{"role": "user", "content": "a"}], session_key="s"))
    for _ in range(100):
        if sock1.sent or sock2.sent:
            break
        await asyncio.sleep(0.001)
    pinned = broker._pins[("u1", "s")]
    job = (sock1.sent or sock2.sent)[-1]
    broker.handle_message(pinned, {"type": "result", "job_id": job["job_id"], "content": "one"})
    assert (await task)["content"] == "one"

    broker.unregister(pinned)
    survivor_id = wid2 if pinned == wid1 else wid1
    survivor_sock = sock2 if pinned == wid1 else sock1
    task2 = asyncio.create_task(broker.submit(
        "u1", [{"role": "user", "content": "b"}], session_key="s"))
    job2 = await _wait_for_job(survivor_sock)
    broker.handle_message(survivor_id, {"type": "result", "job_id": job2["job_id"], "content": "two"})
    assert (await task2)["content"] == "two"
    assert broker._pins[("u1", "s")] == survivor_id


@pytest.mark.asyncio
async def test_submit_timeout():
    broker = InferBroker()
    _register(broker, "u1")
    with pytest.raises(asyncio.TimeoutError):
        await broker.submit("u1", [{"role": "user", "content": "x"}], timeout=0.05)


def test_stray_and_late_messages_are_ignored():
    broker = InferBroker()
    wid, _ = _register(broker, "u1")
    broker.handle_message(wid, {"type": "result", "job_id": "nope", "content": "x"})
    broker.handle_message("ghost-worker", {"type": "pong"})
    broker.handle_message(wid, {"type": "pong"})  # refreshes last_seen, no crash


# ── BrowserProvider ──────────────────────────────────────────────────


def _provider_with_transport(handler) -> BrowserProvider:
    provider = BrowserProvider(base_url="http://fd.test")
    provider.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


@pytest.mark.asyncio
async def test_browser_provider_complete_and_structured():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        seen.append(_json.loads(request.content))
        return httpx.Response(200, json={
            "content": '{"action":"final","text":"ok"}',
            "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            "model": "Qwen3-4B-q4f16_1-MLC",
        })

    provider = _provider_with_transport(handler)
    messages = [Message(role="system", content="s"), Message(role="user", content="u")]
    response = await provider.complete_structured(messages, {"type": "object"}, max_tokens=64)

    assert response.content.startswith('{"action"')
    assert response.usage["prompt_tokens"] == 11
    assert response.model == "Qwen3-4B-q4f16_1-MLC"
    body = seen[0]
    assert body["response_schema"] == {"type": "object"}
    assert body["max_tokens"] == 64
    assert body["session_key"] == provider.session_key  # stable → KV pinning
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


@pytest.mark.asyncio
async def test_browser_provider_no_worker_maps_to_clear_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "no worker"})

    provider = _provider_with_transport(handler)
    with pytest.raises(LLMAPIError, match="Local inference"):
        await provider.complete([Message(role="user", content="x")])


@pytest.mark.asyncio
async def test_browser_provider_broker_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    provider = _provider_with_transport(handler)
    with pytest.raises(LLMAPIError, match="unreachable"):
        await provider.complete([Message(role="user", content="x")])
