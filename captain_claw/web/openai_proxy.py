"""OpenAI-compatible API proxy handlers."""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


def extract_api_session_id(request: web.Request) -> str | None:
    """Extract session ID from ``Authorization: Bearer <session_id>``."""
    auth = request.headers.get("Authorization", "").strip()
    if not auth.lower().startswith("bearer "):
        return None
    token = auth[7:].strip()
    return token or None


def build_chat_completion_response(
    content: str,
    model: str,
    usage: dict[str, int],
    completion_id: str | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-compatible chat completion response."""
    return {
        "id": completion_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


async def write_sse_streaming_response(
    server: WebServer,
    request: web.Request,
    content: str,
    model: str,
    usage: dict[str, int],
    completion_id: str,
) -> web.StreamResponse:
    """Stream a completed response as Server-Sent Events."""
    created = int(time.time())
    resp = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await resp.prepare(request)

    async def _sse(data: str) -> None:
        await resp.write(f"data: {data}\n\n".encode("utf-8"))

    await _sse(json.dumps({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }))

    words = content.split(" ")
    chunk_size = 5
    for i in range(0, len(words), chunk_size):
        text_chunk = " ".join(words[i : i + chunk_size])
        if i > 0:
            text_chunk = " " + text_chunk
        await _sse(json.dumps({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": text_chunk}, "finish_reason": None}],
        }))

    await _sse(json.dumps({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }))

    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


async def api_chat_completions(server: WebServer, request: web.Request) -> web.Response | web.StreamResponse:
    """``POST /v1/chat/completions`` — OpenAI-compatible endpoint."""
    if not server._api_pool:
        return web.json_response(
            {"error": {"message": "API proxy is disabled", "type": "server_error", "code": "api_disabled"}},
            status=503,
        )

    session_id = extract_api_session_id(request)
    if not session_id:
        return web.json_response(
            {"error": {"message": "Missing or invalid Authorization header. Expected: Bearer <session_id>", "type": "invalid_request_error", "code": "missing_api_key"}},
            status=401,
        )

    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": {"message": "Invalid JSON in request body", "type": "invalid_request_error", "code": "invalid_json"}},
            status=400,
        )

    messages = body.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return web.json_response(
            {"error": {"message": "messages array is required and must not be empty", "type": "invalid_request_error", "code": "invalid_messages"}},
            status=400,
        )

    user_message: str | None = None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_message = content.strip()
            elif isinstance(content, list):
                parts = [
                    str(p.get("text", ""))
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                user_message = " ".join(parts).strip()
            break

    if not user_message:
        return web.json_response(
            {"error": {"message": "No user message found in messages array", "type": "invalid_request_error", "code": "no_user_message"}},
            status=400,
        )

    stream = bool(body.get("stream", False))

    try:
        agent = await server._api_pool.get_or_create(session_id)
    except Exception as exc:
        log.error("API agent creation failed", session_id=session_id, error=str(exc))
        return web.json_response(
            {"error": {"message": f"Failed to initialize session: {exc}", "type": "server_error", "code": "agent_init_failed"}},
            status=500,
        )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    try:
        response_text = await agent.complete(user_message)
    except Exception as exc:
        log.error("API completion failed", session_id=session_id, error=str(exc))
        return web.json_response(
            {"error": {"message": f"Completion failed: {exc}", "type": "server_error", "code": "completion_failed"}},
            status=500,
        )
    finally:
        await server._api_pool.release(session_id)

    model_name = "captain-claw"
    try:
        details = agent.get_runtime_model_details()
        model_name = details.get("model", "captain-claw")
    except Exception:
        pass

    usage = getattr(agent, "last_usage", None) or {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
    }

    if not stream:
        return web.json_response(
            build_chat_completion_response(
                content=response_text,
                model=model_name,
                usage=usage,
                completion_id=completion_id,
            )
        )

    return await write_sse_streaming_response(
        server=server,
        request=request,
        content=response_text,
        model=model_name,
        usage=usage,
        completion_id=completion_id,
    )


async def api_list_models(server: WebServer, request: web.Request) -> web.Response:
    """``GET /v1/models`` — list available models."""
    models_data: list[dict[str, Any]] = []
    created = int(time.time())

    if server.agent:
        for entry in server.agent.get_allowed_models():
            model_id = entry.get("id", "unknown")
            models_data.append({
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": "captain-claw",
                "permission": [],
                "root": model_id,
                "parent": None,
            })

    if not any(m["id"] == "captain-claw" for m in models_data):
        models_data.insert(0, {
            "id": "captain-claw",
            "object": "model",
            "created": created,
            "owned_by": "captain-claw",
            "permission": [],
            "root": "captain-claw",
            "parent": None,
        })

    return web.json_response({"object": "list", "data": models_data})
