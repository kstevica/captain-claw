"""A tiny stdio MCP server that asks the person before it changes anything.

Its ``deploy`` tool works the way servers like OpsCrew do: with a client that
declared form elicitation it sends ``elicitation/create`` (a string id, a
yes/no field and a region to pick) and deploys only on an accepted "yes";
without the capability it refuses and changes nothing. ``echo`` answers at
once, ``wait`` sleeps a little, ``ping_me`` pings the client (with the
call's own id) and answers once it hears back. A tool call
the client cancels takes its question back (``notifications/cancelled``).

Plain JSON-RPC over stdin/stdout, standard library only; used by
tests/test_flight_deck/test_mcp_elicitation.py and for trying the card by hand.
"""

from __future__ import annotations

import json
import sys
import threading
import time

_out_lock = threading.Lock()
_client_caps: dict = {}
_waiting: dict[str, object] = {}     # our request id → the tools/call id it belongs to
_pinging: dict[str, object] = {}     # a ping we sent → the tools/call it belongs to
_asked = 0


def send(msg: dict) -> None:
    with _out_lock:
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()


def result(call_id, text: str, *, error: bool = False) -> None:
    send({"jsonrpc": "2.0", "id": call_id,
          "result": {"content": [{"type": "text", "text": text}], "isError": error}})


TOOLS = [
    {"name": "deploy", "description": "Deploy the app (asks the person first).",
     "inputSchema": {"type": "object", "properties": {"app": {"type": "string"}}}},
    {"name": "echo", "description": "Say it back.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
    {"name": "wait", "description": "Sleep a moment.",
     "inputSchema": {"type": "object", "properties": {"seconds": {"type": "number"}}}},
    {"name": "ping_me", "description": "Ping the client, then answer.",
     "inputSchema": {"type": "object", "properties": {}}},
]


def ask(call_id, app: str) -> None:
    global _asked
    _asked += 1
    rid = f"ask-{_asked}"
    _waiting[rid] = call_id
    send({"jsonrpc": "2.0", "id": rid, "method": "elicitation/create", "params": {
        "mode": "form",
        "message": f"Deploy {app} to a new server?\n\n+ server  cx22  €4.51/month\n+ firewall\n"
                   "Your coding agent cannot approve this for you.",
        "requestedSchema": {
            "type": "object",
            "properties": {
                "approve": {"type": "boolean", "title": "Approve", "description": f"Deploy {app}",
                            "default": False},
                "region": {"type": "string", "title": "Region",
                           "oneOf": [{"const": "fsn1", "title": "Falkenstein"},
                                     {"const": "nbg1", "title": "Nuremberg"}],
                           "default": "nbg1"},
            },
            "required": ["approve", "region"],
        },
    }})


def on_answer(msg: dict) -> None:
    pinged = _pinging.pop(json.dumps(msg.get("id")), None)
    if pinged is not None:
        result(pinged, "pong received" if "result" in msg else "ping failed", error="result" not in msg)
        return
    call_id = _waiting.pop(str(msg.get("id")), None)
    if call_id is None:
        return
    if "error" in msg:
        result(call_id, f"no dialog: {msg['error'].get('message')}", error=True)
        return
    res = msg.get("result") or {}
    action = res.get("action")
    content = res.get("content") or {}
    if action == "accept" and content.get("approve") is True:
        result(call_id, f"deployed to {content.get('region')}")
    elif action == "accept":
        result(call_id, "not approved: nothing changed")
    else:
        result(call_id, f"{action}: nothing changed")


def handle(msg: dict) -> None:
    method = msg.get("method")
    mid = msg.get("id")
    if method is None:
        on_answer(msg)  # a response to one of our requests
        return
    if method == "initialize":
        _client_caps.clear()
        _client_caps.update((msg.get("params") or {}).get("capabilities") or {})
        send({"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": (msg.get("params") or {}).get("protocolVersion", "2025-06-18"),
            "capabilities": {"tools": {}}, "serverInfo": {"name": "elicit-fake", "version": "1"}}})
    elif method == "notifications/cancelled":
        call_id = (msg.get("params") or {}).get("requestId")
        for rid, cid in list(_waiting.items()):
            if cid == call_id:
                _waiting.pop(rid, None)
                send({"jsonrpc": "2.0", "method": "notifications/cancelled",
                      "params": {"requestId": rid, "reason": "the tool call was cancelled"}})
    elif mid is None:
        return  # other notifications
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
    elif method == "tools/call":
        params = msg.get("params") or {}
        name, args = params.get("name"), params.get("arguments") or {}
        if name == "echo":
            result(mid, str(args.get("text", "")))
        elif name == "wait":
            seconds = float(args.get("seconds") or 0.5)
            threading.Thread(target=lambda: (time.sleep(seconds), result(mid, "waited")), daemon=True).start()
        elif name == "ping_me":
            # The ping reuses the tool call's own id: a client that took any
            # message with an id for a response would end the call with it.
            _pinging[json.dumps(mid)] = mid
            send({"jsonrpc": "2.0", "id": mid, "method": "ping"})
        elif name == "deploy":
            if "elicitation" not in _client_caps:
                result(mid, "This client can't show the approval dialog: nothing changed.", error=True)
            else:
                ask(mid, str(args.get("app") or "the app"))
        else:
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no tool {name}"}})
    else:
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no method {method}"}})


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if isinstance(msg, dict):
            handle(msg)


if __name__ == "__main__":
    main()
