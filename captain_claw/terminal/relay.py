"""Flight Deck PTY relay — bridges the agent's terminal tool to a PTY daemon
that dialed out from behind NAT.

Topology: the daemon (on the user's Mac) opens a persistent WebSocket to
``/fd/pty/connect`` and registers as a named *worker*. The agent's
``terminal`` tool then talks plain HTTP to ``/fd/pty/{worker}/{op}`` — the
SAME open/input/read/… API the daemon serves locally — and this relay
forwards each call down the worker's socket and returns its reply. So the
tool is identical whether the daemon is local or remote; only the URL the
tool points at changes.

Auth: a shared token (``CLAW_PTY_TOKEN`` in FD's environment). The daemon
presents it in the register frame; the agent presents it in the
``X-Claw-Token`` header. A PTY is remote code execution — keep the relay on
TLS/HTTPS and the token secret.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect

log = logging.getLogger("claw.pty.relay")

router = APIRouter()

# Forwarded ops the tool may invoke. ``health`` is allowed so a caller can
# probe a worker without opening a session.
_OPS = {"open", "input", "read", "resize", "close", "list", "health"}

# Generous ceiling: longer than any `read` wait the tool issues, so a slow
# interactive program doesn't trip a false timeout.
_REQUEST_TIMEOUT = 150.0


def _token() -> str | None:
    return os.environ.get("CLAW_PTY_TOKEN") or None


class _Worker:
    """One connected PTY daemon and its in-flight request futures."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.pending: dict[str, asyncio.Future] = {}
        self.send_lock = asyncio.Lock()


# Single FD process → module-level registry keyed by worker name.
_workers: dict[str, _Worker] = {}


@router.websocket("/fd/pty/connect")
async def pty_connect(ws: WebSocket) -> None:
    """A PTY daemon dials in here, registers, then answers forwarded requests."""
    await ws.accept()
    try:
        reg = await ws.receive_json()
    except Exception:
        await ws.close(code=4000)
        return

    if not isinstance(reg, dict) or reg.get("type") != "register":
        await ws.close(code=4001)
        return
    token = _token()
    if token and reg.get("token") != token:
        log.warning("pty worker rejected: bad token")
        await ws.close(code=4003)
        return

    worker = str(reg.get("worker") or "default")
    conn = _Worker(ws)
    # If a worker reconnects, drop the stale registration.
    _workers[worker] = conn
    log.info("pty worker connected: %r", worker)

    try:
        while True:
            data = await ws.receive_json()
            if not isinstance(data, dict):
                continue
            if data.get("type") == "response":
                fut = conn.pending.pop(str(data.get("id")), None)
                if fut is not None and not fut.done():
                    fut.set_result(data)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("pty worker %r socket error: %s", worker, exc)
    finally:
        if _workers.get(worker) is conn:
            del _workers[worker]
        for fut in conn.pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("worker disconnected"))
        log.info("pty worker disconnected: %r", worker)


async def _forward(worker: str, op: str, payload: dict) -> dict:
    conn = _workers.get(worker)
    if conn is None:
        raise HTTPException(status_code=503, detail=f"pty worker {worker!r} not connected")

    rid = uuid.uuid4().hex
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    conn.pending[rid] = fut
    try:
        async with conn.send_lock:
            await conn.ws.send_json({"type": "request", "id": rid, "op": op, "payload": payload})
    except Exception as exc:
        conn.pending.pop(rid, None)
        raise HTTPException(status_code=502, detail=f"failed to reach worker: {exc}")

    try:
        resp = await asyncio.wait_for(fut, timeout=_REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        conn.pending.pop(rid, None)
        raise HTTPException(status_code=504, detail="pty worker timed out")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    status = int(resp.get("status", 200))
    body = resp.get("body")
    body = body if isinstance(body, dict) else {}
    if status >= 400:
        raise HTTPException(status_code=status, detail=body.get("error") or "worker error")
    return body


@router.get("/fd/pty/workers")
async def pty_workers(request: Request) -> dict:
    """List currently-connected PTY workers (token-gated)."""
    token = _token()
    if token and request.headers.get("X-Claw-Token") != token:
        raise HTTPException(status_code=401, detail="bad or missing X-Claw-Token")
    return {"workers": sorted(_workers.keys())}


@router.post("/fd/pty/{worker}/{op}")
async def pty_op(worker: str, op: str, request: Request) -> dict:
    """Forward one terminal op to *worker*. Mirrors the daemon's HTTP API."""
    token = _token()
    if token and request.headers.get("X-Claw-Token") != token:
        raise HTTPException(status_code=401, detail="bad or missing X-Claw-Token")
    if op not in _OPS:
        raise HTTPException(status_code=404, detail=f"unknown op {op!r}")
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return await _forward(worker, op, payload)
