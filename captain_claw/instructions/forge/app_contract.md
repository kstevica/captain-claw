<!-- part:intro -->
## How agents connect to this app

Captain's agents use this app as well as people, so this is part of the job, not an extra. Build it from the first commit and test it. People's screens may use whatever suits them.

Read the port from the `PORT` environment variable and listen on 127.0.0.1 when it is set. Keep every link and asset path relative, because the app may be served under a path such as `/vfs-apps/<name>/`.

The key agents use comes from the `AGENT_API_KEY` environment variable. If it is unset, generate one on first start, keep it in a file the repository ignores, and print it once. Never write a key into the repository, a log line or a response.
<!-- part:api -->
## Agent API (HTTP)

- Everything for agents lives under `/api/agent/`. JSON in, JSON out.
- Every request needs `Authorization: Bearer <AGENT_API_KEY>`. Anything else gets 401, with no detail about why.
- `GET /api/agent/health` answers `{"ok": true}` with no key, for a check that the app is up.
- `GET /api/agent/manifest` says what the app offers, so an agent never has to guess:
  `{"name": "...", "version": "...", "description": "...", "ws": "/agent/ws" or null, "endpoints": [{"method": "...", "path": "/api/agent/...", "purpose": "...", "input": {...}, "output": {...}}], "events": [{"name": "...", "purpose": "..."}]}`
  Its `path` values are relative to the app's address. Generate this from the routes if the framework allows, so it can't go stale.
- Agents can't send DELETE. Use POST for removing something, such as `POST /api/agent/funds/<id>/archive`.
- An error is `{"error": {"code": "...", "message": "..."}}` with the right status. The message says what to do differently, because a model reads it.
- A write that can be retried takes an `Idempotency-Key` header, and the same key twice does the work once.
- A list takes `limit` and `cursor`, and answers with `next_cursor`.
<!-- part:wss -->
## Agent socket (WebSocket)

Captain connects out to the app and stays connected, so an agent needs no public address. Serve a WebSocket at `/agent/ws`, using exactly these frames. They are Captain's agent link protocol.

The connection opens with `Authorization: Bearer <AGENT_API_KEY>`. A wrong key closes it with code 4401. Each agent holds its own connection, and it speaks first, every time it connects:

`{"type": "hello", "link": "<link name>", "agent": "<agent name>", "framework": "captain", "version": "...", "model": "...", "runs_local": true}`

To hand an agent work, the app sends:

`{"type": "message", "id": "<your id, up to 64 of A-Z a-z 0-9 _ - . :>", "text": "what to do, in plain words", "data": {...}, "files": [{"name": "report.pdf", "data": "<base64>"}]}`

`data` and `files` are optional. Files are at most 10 and 10 MB in all. Messages run one at a time per agent. An id seen before doesn't run again.

The agent answers with these frames, and `re` is your message id:

- `{"type": "accepted", "id": "...", "queued": 0}`
- `{"type": "state", "re": "...", "state": "thinking"}`, and `"ready"` when it is done
- `{"type": "status", "re": "...", "text": "..."}`, progress you can show
- `{"type": "reply", "re": "...", "text": "..."}`, the answer
- `{"type": "error", "re": "...", "text": "..."}`, that message failed
- `{"type": "file", "re": "...", "file": "<id>", "name": "...", "mime": "...", "size": 1, "parts": 2}`, then that many `{"type": "file_part", "re": "...", "file": "<id>", "part": 0, "data": "<base64>"}`
- `{"type": "push", "seq": 1, "text": "...", "files": 0, "created_at": "..."}`, something the agent sent on its own

Send `{"type": "ack", "re": "..."}` once you have stored a reply, or it comes again after a reconnect. The same reply can still arrive twice, so tell them apart by `re`. Ignore any frame type you don't know. Captain sends WebSocket pings every 30 seconds, so answer them, which most libraries do by themselves.

Keep what is waiting for an agent that isn't connected, and send it when that agent says `hello`. Track who is connected by the `agent` in `hello`, and show that to people.
<!-- part:handover -->
## What to hand over

- `AGENT_API.md` at the top of the repository: what the app is for, how to start it, and each endpoint or event with one real example.
- `agent-connection.json` at the top of the repository, which Captain reads:
  `{"name": "...", "start": "<one command that starts the app>", "port_env": "PORT", "health": "/api/agent/health", "manifest": "/api/agent/manifest", "api": "/api/agent", "ws": "/agent/ws" or null, "key_env": "AGENT_API_KEY"}`
- Tests that prove it, run before you finish: a request with no key is refused, the manifest matches the real routes, and (if it has a socket) a fake agent connects, says `hello`, gets a `message`, sends `accepted`, `state` and `reply`, and the app stores the reply.

Finish by saying how to start the app and where the key is.
