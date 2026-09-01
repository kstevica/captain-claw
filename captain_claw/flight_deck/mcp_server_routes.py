"""Inbound MCP server — expose the user's Flight Deck agent fleet over MCP.

An external MCP client (Claude Code / Claude Desktop) connects to
``POST /fd/mcp-server`` (Streamable-HTTP JSON-RPC 2.0) authenticated with a
personal access token, and can:

  * ``list_agents``  — the caller's running/known agents
  * ``send_task``    — dispatch a task to one agent (async; returns a task id)
  * ``get_result``   — poll a task for live progress + the final answer
  * ``cancel_task``  — abort an in-flight task

Everything is owner-scoped: an MCP caller only ever sees and drives their own
agents. Task execution reuses the same agent ``/ws`` consult mechanism the
peer-consult route uses. PAT management (``/fd/mcp-tokens``) is browser-auth'd
so users self-serve their tokens from the Flight Deck UI.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from captain_claw.flight_deck.auth import (
    get_current_user, get_db, get_mcp_user, hash_token, new_pat,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/fd", tags=["mcp-inbound"])

PROTOCOL_VERSION = "2025-03-26"
SERVER_NAME = "captain-claw-fleet"
SERVER_VERSION = "1.0.0"


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


# ── JSON-RPC envelope helpers (mirrors captain_claw/mcp_serve.py) ─────

def _jsonrpc_result(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _jsonrpc_error(req_id: Any, code: int, message: str, data: Any = None) -> dict:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


# ── Per-user async task store ─────────────────────────────────────────

@dataclass
class _MCPTask:
    id: str
    user_id: str
    port: int
    agent_name: str
    task: str
    status: str = "running"  # running | done | error | cancelled
    events: list[dict] = field(default_factory=list)
    result: str = ""
    usage: dict | None = None
    error: str = ""
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)
    aio: Any = None  # the backing asyncio.Task (for cancel)


_TASKS: dict[str, _MCPTask] = {}
_TASK_TTL_SECONDS = 3600.0  # prune terminal tasks after an hour
_FORWARD_EVENT_TYPES = {"status", "thinking", "monitor", "tool_stream"}


def _prune_tasks() -> None:
    """Drop terminal tasks older than the TTL so the store can't grow forever."""
    now = datetime.now(UTC)
    stale = []
    for tid, t in _TASKS.items():
        if t.status in ("running",):
            continue
        try:
            age = (now - datetime.fromisoformat(t.updated_at)).total_seconds()
        except Exception:
            age = 0.0
        if age > _TASK_TTL_SECONDS:
            stale.append(tid)
    for tid in stale:
        _TASKS.pop(tid, None)


# ── Fleet enumeration (owner-scoped) ─────────────────────────────────

def _user_agents(user_id: str) -> list[dict]:
    """The caller's running/known agents (docker + process), owner-scoped.

    Mirrors GET /fd/fleet but callable without a Request. Server helpers are
    imported lazily to avoid a circular import (server.py includes this router).
    """
    from captain_claw.flight_deck import server as S

    out: list[dict] = []
    try:
        client = S.get_docker()
        for c in client.containers.list(all=True, filters={"label": S.CONTAINER_LABEL}):
            labels = c.labels or {}
            if S.AUTH_ENABLED and user_id and labels.get(S.OWNER_LABEL, "") != user_id:
                continue
            wp = labels.get("flight-deck.web-port", "")
            name = labels.get("flight-deck.agent-name", c.name)
            out.append({
                "name": name, "slug": S._slug(name), "kind": "docker",
                "host": "localhost", "port": int(wp) if wp else 0,
                "status": c.status, "description": labels.get("flight-deck.description", ""),
            })
    except Exception:
        pass
    try:
        for slug, entry in S._load_process_registry().items():
            if S.AUTH_ENABLED and user_id and entry.get("owner", "") != user_id:
                continue
            alive = S._process_is_alive(slug)
            out.append({
                "name": entry.get("name", slug), "slug": slug, "kind": "process",
                "host": "localhost", "port": entry.get("web_port", 0),
                "status": "running" if alive else "stopped",
                "description": entry.get("description", ""),
            })
    except Exception:
        pass
    return out


def _resolve_agent(user_id: str, selector: str) -> dict | None:
    """Find one of the caller's agents by slug, name, or port."""
    sel = str(selector or "").strip()
    if not sel:
        return None
    agents = _user_agents(user_id)
    for a in agents:
        if str(a.get("port")) == sel or a.get("slug") == sel or a.get("name") == sel:
            return a
    low = sel.lower()
    for a in agents:
        if low in str(a.get("name", "")).lower() or low in str(a.get("slug", "")).lower():
            return a
    return None


# ── Task runner: consult the agent over its /ws and collect the reply ─

async def _run_agent_task(t: _MCPTask, host: str, auth: str, timeout: float) -> None:
    import websockets

    def _emit(ev_type: str, data: dict) -> None:
        t.events.append({"type": ev_type, "data": data, "at": _utcnow_iso()})
        t.updated_at = _utcnow_iso()

    url = f"ws://{host}:{t.port}/ws" + (f"?token={auth}" if auth else "")
    loop = asyncio.get_event_loop()
    try:
        async with websockets.connect(url, max_size=4 * 1024 * 1024) as ws:
            welcome = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if welcome.get("type") != "welcome":
                raise RuntimeError("unexpected handshake from agent")
            # Skip the replay backlog.
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if msg.get("type") == "replay_done":
                    break
                if msg.get("type") not in ("chat_message",) or not msg.get("replay"):
                    break
            payload = {"type": "chat", "content": t.task, "no_broadcast": True}
            await ws.send(json.dumps(payload))
            _emit("status", {"status": "sent"})

            parts: list[str] = []
            deadline = loop.time() + timeout
            busy = 0
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    if not parts:
                        raise TimeoutError("timed out waiting for the agent")
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 15.0))
                except asyncio.TimeoutError:
                    _emit("heartbeat", {"elapsed": int(timeout - remaining), "timeout": int(timeout)})
                    continue
                msg = json.loads(raw)
                mt = msg.get("type", "")
                if mt in _FORWARD_EVENT_TYPES:
                    _emit(mt, msg)
                if mt == "chat_message" and msg.get("role") == "assistant" and not msg.get("replay"):
                    content = msg.get("content", "")
                    if content:
                        parts.append(content)
                    # The agent emits a `usage` summary right after the reply.
                    try:
                        for _ in range(4):
                            m2 = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))
                            if m2.get("type") == "usage":
                                t.usage = m2
                                break
                            if m2.get("type") in _FORWARD_EVENT_TYPES:
                                _emit(m2.get("type"), m2)
                    except Exception:
                        pass
                    break
                if mt == "error":
                    err = str(msg.get("message", "agent error"))
                    if busy < 8 and ("busy processing" in err.lower() or "session is busy" in err.lower()):
                        busy += 1
                        _emit("status", {"status": f"peer busy, retrying ({busy})…"})
                        await asyncio.sleep(min(2 + busy * 2, 12))
                        try:
                            await ws.send(json.dumps(payload))
                        except Exception:
                            raise RuntimeError(err)
                        continue
                    raise RuntimeError(err)
            t.result = "\n".join(parts) if parts else "(no response)"
            t.status = "done"
    except asyncio.CancelledError:
        t.status = "cancelled"
        raise
    except Exception as exc:  # noqa: BLE001 — surface as task error, never crash
        t.status = "error"
        t.error = str(exc)
    finally:
        t.updated_at = _utcnow_iso()


# ── MCP tool definitions ─────────────────────────────────────────────

MCP_TOOLS = [
    {
        "name": "list_agents",
        "description": "List your running/known Captain Claw agents. Returns each agent's name, slug, port, kind (docker/process), status and description. Use a slug or name as the `agent` argument to send_task.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "send_task",
        "description": "Send a task/message to one of your agents and start it running. Returns a task_id immediately (the run is async — agent turns can take minutes). Poll get_result with that id for live progress and the final answer.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent": {"type": "string", "description": "Which agent: its slug, name, or port (from list_agents)."},
                "task": {"type": "string", "description": "The task or message to send the agent."},
                "timeout_seconds": {"type": "number", "description": "Max seconds to wait for the agent (default 300, max 1800)."},
            },
            "required": ["agent", "task"],
        },
    },
    {
        "name": "get_result",
        "description": "Poll a task started by send_task. Returns status (running/done/error/cancelled), any new progress events since `since` (pass the returned `next_cursor` to get only new ones), the final result text, and token usage when done.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "since": {"type": "integer", "description": "Event cursor — only events at/after this index are returned (default 0)."},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "cancel_task",
        "description": "Cancel an in-flight task started by send_task (aborts the agent turn).",
        "inputSchema": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]},
    },
]


async def _tool_list_agents(user: dict, args: dict) -> dict:
    return {"agents": _user_agents(user["id"])}


async def _tool_send_task(user: dict, args: dict) -> dict:
    from captain_claw.flight_deck import server as S

    agent_sel = str(args.get("agent") or "").strip()
    task = str(args.get("task") or "").strip()
    if not agent_sel or not task:
        raise ValueError("`agent` and `task` are required")
    agent = _resolve_agent(user["id"], agent_sel)
    if not agent:
        raise ValueError(f"No agent of yours matches '{agent_sel}'. Call list_agents to see options.")
    if not agent.get("port"):
        raise ValueError(f"Agent '{agent['name']}' has no web port (is it running?).")
    if str(agent.get("status")) not in ("running", "created"):
        raise ValueError(f"Agent '{agent['name']}' is not running (status: {agent.get('status')}).")

    timeout = float(args.get("timeout_seconds") or 300.0)
    timeout = max(10.0, min(timeout, 1800.0))
    auth = S._resolve_agent_auth(int(agent["port"]))

    _prune_tasks()
    import uuid as _uuid
    tid = _uuid.uuid4().hex
    t = _MCPTask(id=tid, user_id=user["id"], port=int(agent["port"]),
                 agent_name=agent["name"], task=task)
    _TASKS[tid] = t
    t.aio = asyncio.create_task(_run_agent_task(t, agent["host"], auth, timeout))
    return {"task_id": tid, "agent": agent["name"], "status": "running"}


def _owned_task(user: dict, task_id: str) -> _MCPTask:
    t = _TASKS.get(str(task_id or ""))
    if not t or t.user_id != user["id"]:
        raise ValueError("Unknown task_id")
    return t


async def _tool_get_result(user: dict, args: dict) -> dict:
    t = _owned_task(user, args.get("task_id"))
    since = int(args.get("since") or 0)
    since = max(0, min(since, len(t.events)))
    new_events = t.events[since:]
    out: dict[str, Any] = {
        "task_id": t.id,
        "agent": t.agent_name,
        "status": t.status,
        "events": new_events,
        "next_cursor": len(t.events),
        "done": t.status != "running",
    }
    if t.status == "done":
        out["result"] = t.result
        if t.usage is not None:
            out["usage"] = t.usage
    elif t.status == "error":
        out["error"] = t.error
    return out


async def _tool_cancel_task(user: dict, args: dict) -> dict:
    t = _owned_task(user, args.get("task_id"))
    if t.status == "running" and t.aio is not None:
        t.aio.cancel()
    return {"task_id": t.id, "status": t.status if t.status != "running" else "cancelling"}


_TOOL_HANDLERS = {
    "list_agents": _tool_list_agents,
    "send_task": _tool_send_task,
    "get_result": _tool_get_result,
    "cancel_task": _tool_cancel_task,
}


# ── JSON-RPC method handlers ─────────────────────────────────────────

async def _handle_initialize(req_id: Any, params: dict, user: dict) -> dict:
    return _jsonrpc_result(req_id, {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
    })


async def _handle_tools_list(req_id: Any, params: dict, user: dict) -> dict:
    return _jsonrpc_result(req_id, {"tools": MCP_TOOLS})


async def _handle_tools_call(req_id: Any, params: dict, user: dict) -> dict:
    name = params.get("name")
    args = params.get("arguments") or {}
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        return _jsonrpc_error(req_id, -32602, f"Unknown tool: {name}")
    try:
        result = await handler(user, args)
        text = json.dumps(result, indent=2, default=str)
        return _jsonrpc_result(req_id, {"content": [{"type": "text", "text": text}], "isError": False})
    except Exception as exc:  # noqa: BLE001 — tool errors are in-band, not JSON-RPC errors
        return _jsonrpc_result(req_id, {
            "content": [{"type": "text", "text": f"Error: {exc}"}],
            "isError": True,
        })


async def _handle_ping(req_id: Any, params: dict, user: dict) -> dict:
    return _jsonrpc_result(req_id, {})


_METHOD_HANDLERS = {
    "initialize": _handle_initialize,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
    "ping": _handle_ping,
}
_NOTIFICATIONS = {"notifications/initialized", "notifications/cancelled"}


async def _dispatch_one(msg: dict, user: dict) -> dict | None:
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return _jsonrpc_error(msg.get("id") if isinstance(msg, dict) else None, -32600, "Invalid Request")
    method = msg.get("method", "")
    req_id = msg.get("id")
    params = msg.get("params") or {}
    if method in _NOTIFICATIONS or (req_id is None and method.startswith("notifications/")):
        return None  # notifications get no response
    handler = _METHOD_HANDLERS.get(method)
    if handler is None:
        return _jsonrpc_error(req_id, -32601, f"Method not found: {method}")
    return await handler(req_id, params, user)


@router.post("/mcp-server")
async def mcp_server(request: Request, user: dict = Depends(get_mcp_user)):
    """Inbound MCP JSON-RPC endpoint (Streamable HTTP, JSON responses)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(_jsonrpc_error(None, -32700, "Parse error"), status_code=400)

    if isinstance(body, list):  # JSON-RPC batch
        responses = []
        for m in body:
            r = await _dispatch_one(m, user)
            if r is not None:
                responses.append(r)
        if not responses:
            return Response(status_code=202)
        return JSONResponse(responses)

    resp = await _dispatch_one(body, user)
    if resp is None:
        return Response(status_code=202)
    return JSONResponse(resp)


# ── Personal access token self-serve (browser auth) ──────────────────

class CreatePATRequest(BaseModel):
    name: str = ""


@router.get("/mcp-tokens")
async def list_mcp_tokens(user: dict = Depends(get_current_user)):
    """List the caller's personal access tokens (never returns the secret)."""
    db = get_db()
    return {"tokens": await db.list_pats(user["id"])}


@router.post("/mcp-tokens")
async def create_mcp_token(body: CreatePATRequest = Body(default=CreatePATRequest()),
                           user: dict = Depends(get_current_user)):
    """Mint a new personal access token. The raw token is returned ONCE — it is
    stored only as a hash and cannot be retrieved again."""
    db = get_db()
    raw = new_pat()
    pid = await db.create_pat(user["id"], hash_token(raw), (body.name or "").strip()[:80])
    return {"id": pid, "name": (body.name or "").strip()[:80], "token": raw,
            "created_at": _utcnow_iso()}


@router.delete("/mcp-tokens/{token_id}")
async def revoke_mcp_token(token_id: str, user: dict = Depends(get_current_user)):
    db = get_db()
    ok = await db.revoke_pat(token_id, user["id"])
    if not ok:
        raise HTTPException(404, "token not found")
    return {"ok": True}
