"""Mock captain-claw agent for HUD end-to-end tests.

Implements just enough of the agent web API that Flight Deck proxies:
  WS  /ws?token=            welcome / replay_batch / chat turn events
  GET /api/files            file list
  GET /api/files/view       file bytes
  GET /api/datastore/tables
  GET /api/datastore/tables/{name}/rows
Every inbound WS frame is appended to frames.jsonl for assertions.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from aiohttp import WSMsgType, web

TOKEN = "mocktoken"
HERE = Path(__file__).resolve().parent
RUN = Path(os.environ.get("E2E_RUN_DIR", HERE / ".run"))
RUN.mkdir(parents=True, exist_ok=True)
FRAMES = RUN / "frames.jsonl"
history: list[dict] = []

REPORT_MD = """# Quarterly report

Revenue grew **18%** quarter over quarter, driven by the *glasses* rollout.

## Highlights

- Shipped the HUD for Meta Ray-Ban Display
- Pairing codes replace passwords on the glasses
- Datastore browser with record cards

## Small table

| Region | Revenue |
|---|---|
| EU | 1.2M |
| US | 2.4M |

## Wide table

| Product | Owner | Status | Revenue | Margin | Notes |
|---|---|---|---|---|---|
| HUD | Ana | Shipped | 120k | 42% | First release on **glasses** |
| Pairing | Ben | Beta | 0 | — | Device-code flow |
| Datastore | Cleo | Shipped | 80k | 38% | Read-only browser with `cards` |

```python
def hello():
    return "a fairly long line of code that should wrap on a six hundred pixel display without scrolling"
```

> Quote: keep it short on the glasses.

See [the docs](https://example.com/docs) and ![a chart](https://example.com/chart.png).

""" + "\n".join(f"Paragraph {i}: " + "lorem ipsum dolor sit amet " * 8 for i in range(1, 9))

NOTES_MD = "# Notes from a member\n\nThis file was **created by a member**.\n"


def _check(request: web.Request) -> None:
    if request.query.get("token") != TOKEN:
        raise web.HTTPUnauthorized(text=json.dumps({"error": "unauthorized"}), content_type="application/json")


def _files() -> list[dict]:
    now = time.time()
    return [
        {"logical": "saved/report.md", "physical": "/srv/mock/workspace/saved/report.md", "filename": "report.md",
         "extension": ".md", "exists": True, "size": len(REPORT_MD.encode()), "modified": now - 120,
         "mime_type": "text/markdown", "is_text": True, "source": "registry", "created_by": None},
        {"logical": "saved/notes.md", "physical": "/srv/mock/workspace/saved/notes.md", "filename": "notes.md",
         "extension": ".md", "exists": True, "size": len(NOTES_MD.encode()), "modified": now - 7200,
         "mime_type": "text/markdown", "is_text": True, "source": "registry",
         "created_by": {"kind": "member", "user_id": "u2", "name": "Dana"}},
        {"logical": "output/data.csv", "physical": "/srv/mock/workspace/output/data.csv", "filename": "data.csv",
         "extension": ".csv", "exists": True, "size": 10, "modified": now - 60,
         "mime_type": "text/csv", "is_text": True, "source": "registry", "created_by": None},
    ]


async def files(request: web.Request) -> web.Response:
    _check(request)
    return web.json_response(_files())


async def file_view(request: web.Request) -> web.Response:
    _check(request)
    path = request.query.get("path", "")
    if path.endswith("report.md"):
        return web.Response(text=REPORT_MD, content_type="text/markdown")
    if path.endswith("notes.md"):
        return web.Response(text=NOTES_MD, content_type="text/markdown")
    return web.json_response({"error": "File not found"}, status=404)


COLUMNS = [
    {"name": "name", "type": "text", "position": 0},
    {"name": "email", "type": "text", "position": 1},
    {"name": "age", "type": "integer", "position": 2},
    {"name": "active", "type": "boolean", "position": 3},
    {"name": "joined", "type": "date", "position": 4},
    {"name": "score", "type": "real", "position": 5},
    {"name": "meta", "type": "json", "position": 6},
]
ROWS = [
    {"_id": i, "name": f"Person {i}", "email": f"person{i}@example.com", "age": 20 + i, "active": i % 2,
     "joined": f"2026-0{1 + i % 9}-1{i % 9}", "score": i * 1.123456789,
     "meta": json.dumps({"tags": ["a", "b"], "n": i}),
     "_creator": {"kind": "member" if i == 3 else "owner", "user_id": "u2" if i == 3 else "", "name": "Dana" if i == 3 else ""}}
    for i in range(1, 24)
]


async def tables(request: web.Request) -> web.Response:
    _check(request)
    return web.json_response([
        {"name": "contacts", "columns": COLUMNS, "row_count": len(ROWS), "created_at": "2026-10-01T10:00:00",
         "updated_at": "2026-10-09T10:00:00", "created_by": {"kind": "owner", "user_id": "", "name": ""}},
        {"name": "empty_table", "columns": [{"name": "x", "type": "text", "position": 0}], "row_count": 0,
         "created_at": "2026-10-01T10:00:00", "updated_at": None, "created_by": {"kind": "owner", "user_id": "", "name": ""}},
    ])


async def rows(request: web.Request) -> web.Response:
    _check(request)
    name = request.match_info["name"]
    if name == "empty_table":
        return web.json_response({"columns": ["_id", "x"], "rows": [], "total": 0, "offset": 0, "limit": 100})
    if name != "contacts":
        return web.json_response({"error": f"Table not found: {name}"}, status=400)
    limit = int(request.query.get("limit", "100"))
    offset = int(request.query.get("offset", "0"))
    order_by = request.query.get("order_by", "_id") or "_id"
    desc = request.query.get("order_dir", "asc").lower() == "desc"
    data = sorted(ROWS, key=lambda r: r.get(order_by) or 0, reverse=desc)
    return web.json_response({"columns": ["_id"] + [c["name"] for c in COLUMNS],
                              "rows": data[offset:offset + limit], "total": len(ROWS),
                              "offset": offset, "limit": limit})


REPLY = """**Done.** Here is the summary:

| Item | Owner | State | Due |
|---|---|---|---|
| HUD | Ana | shipped | Oct 10 |
| Docs | Ben | review | Oct 12 |

Full write-up in `saved/report.md`.
""" + "\n\n".join(f"Detail {i}: " + "the agent explains things at some length here " * 5 for i in range(1, 7))


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    _check(request)
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    await ws.send_json({"type": "welcome", "session": {"id": "s1", "name": "default", "model": "mock-1",
                                                        "provider": "mock", "message_count": len(history)},
                        "models": [], "commands": [], "personalities": []})
    if history:
        await ws.send_json({"type": "replay_batch", "messages": [
            {"type": "chat_message", "role": m["role"], "content": m["content"], "replay": True,
             "timestamp": m["ts"], "model": "mock-1"} for m in history]})
        await ws.send_json({"type": "replay_done"})
    async for msg in ws:
        if msg.type != WSMsgType.TEXT:
            continue
        data = json.loads(msg.data)
        with FRAMES.open("a") as f:
            f.write(json.dumps(data) + "\n")
        kind = data.get("type")
        if kind == "chat":
            content = data.get("content", "")
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            await ws.send_json({"type": "chat_message", "role": "user", "content": content, "timestamp": ts})
            history.append({"role": "user", "content": content, "ts": ts})
            await ws.send_json({"type": "status", "status": "thinking", "client_msg_id": data.get("client_msg_id")})
            await asyncio.sleep(0.3)
            await ws.send_json({"type": "narration", "text": "Looking things up…", "iteration": 1})
            await ws.send_json({"type": "status", "status": "Using web_search…"})
            if "approve" in content.lower():
                await ws.send_json({"type": "approval_request", "id": "ap1", "message": "Run shell command `ls`?",
                                    "category": "shell"})
                continue
            await asyncio.sleep(0.6)
            await ws.send_json({"type": "chat_message", "role": "assistant", "content": REPLY, "timestamp": ts,
                                "model": "mock-1"})
            history.append({"role": "assistant", "content": REPLY, "ts": ts})
            await ws.send_json({"type": "next_steps", "options": [
                {"label": "Summarize", "action": "Summarize the report in 3 bullets"},
                {"label": "Open report", "action": "Show saved/report.md"}]})
            await ws.send_json({"type": "status", "status": "ready"})
        elif kind == "approval_response":
            await ws.send_json({"type": "chat_message", "role": "assistant",
                                "content": "Approved: ran `ls`." if data.get("approved") else "Denied.",
                                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": "mock-1"})
            await ws.send_json({"type": "status", "status": "ready"})
        elif kind == "command" and data.get("command") == "/new":
            history.clear()
            await ws.send_json({"type": "command_result", "command": "/new", "content": "New session created"})
        elif kind == "cancel":
            await ws.send_json({"type": "status", "status": "ready"})
    return ws


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 24001
    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/api/files", files)
    app.router.add_get("/api/files/view", file_view)
    app.router.add_get("/api/datastore/tables", tables)
    app.router.add_get("/api/datastore/tables/{name}/rows", rows)
    web.run_app(app, host="127.0.0.1", port=port, print=None)


if __name__ == "__main__":
    main()
