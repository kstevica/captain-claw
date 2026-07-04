"""Deliver a background-run completion back to the originating agent.

Shared by agent-initiated Basna/Vatra runs and agent-initiated Code sessions:
the run executes server-side (fire-and-forget), and when it finishes Flight
Deck opens a WebSocket to the agent that started it and sends a `notification`
with `trigger_response`. The payload carries the origin channel (whatsapp /
telegram / web / …) captured at start time, so the agent's relay reply lands
wherever the user originally asked. Best-effort: failures are logged, never
raised.
"""

from __future__ import annotations

import asyncio
import json
import logging


class _KwargLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = {k: v for k, v in kwargs.items() if k not in ("exc_info", "stack_info", "stacklevel")}
        for k in list(kwargs.keys()):
            if k not in ("exc_info", "stack_info", "stacklevel"):
                kwargs.pop(k)
        if extra:
            msg = f"{msg} " + " ".join(f"{k}={v}" for k, v in extra.items())
        return msg, kwargs


log = _KwargLogger(logging.getLogger("flight_deck"), {})


async def notify_source_agent(
    *, source_host: str, source_port: int, origin: dict,
    kind: str, title: str, run_ref: str, ok: bool, summary: str,
    no_restart_hint: str = "",
) -> None:
    """Send a completion callback to the agent that started a background run.

    ``kind``: human label for the run ("Basna", "Vatra", "coding session", …).
    ``run_ref``: identifier the agent can use in follow-up tool calls.
    ``no_restart_hint``: kind-specific "do NOT start another …" directive.
    """
    import websockets

    from captain_claw.flight_deck.server import _resolve_agent_auth
    if not source_port:
        return
    auth = _resolve_agent_auth(int(source_port))
    params = f"?token={auth}" if auth else ""
    url = f"ws://{source_host or 'localhost'}:{source_port}/ws{params}"
    verb = "finished successfully" if ok else "ran into an error"
    callback_msg = (
        f"[{kind} '{title}' {verb}] This is the RESULT of the autonomous {kind} "
        f"you started ({run_ref}). Relay it to the user now, concisely, in their "
        f"language. {no_restart_hint}Do NOT say you are still waiting — you "
        f"already have the outcome below:\n\n{summary}"
    )
    payload: dict = {"type": "notification", "content": callback_msg, "trigger_response": True}
    if origin.get("platform") and origin["platform"] != "web":
        payload["origin_platform"] = origin["platform"]
        payload["origin_user_id"] = origin.get("user_id", "")
        payload["origin_chat_id"] = origin.get("chat_id", 0)
    try:
        async with websockets.connect(url, open_timeout=10, close_timeout=5) as ws:
            welcome = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if welcome.get("type") != "welcome":
                return
            while True:  # skip the session replay before our message
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if msg.get("type") == "replay_done":
                    break
            await ws.send(json.dumps(payload))
            try:
                await asyncio.wait_for(ws.recv(), timeout=30)
            except TimeoutError:
                pass
    except Exception as exc:  # noqa: BLE001 — best-effort delivery
        log.warning(f"{kind} completion delivery failed",
                    run_ref=run_ref, port=source_port, error=str(exc))
