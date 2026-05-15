"""Demo code-app backend: a tiny notes service.

Shows the full contract the agent's authoring loop will follow:

* Define ``handle(method, path, headers, body)`` returning
  ``{status, headers, body}``.
* Use ``app_datastore_client.datastore("notes")`` to persist records
  in the shared FD entity store. The slug is picked up from the
  subprocess environment automatically.

This file is shipped in the package only as a reference example. The
agent-authored apps live under ``~/.captain-claw-fd/apps/<slug>/``.
"""

from __future__ import annotations

import json

from captain_claw.flight_deck.app_datastore_client import datastore


def _json_response(payload, status: int = 200):
    return {
        "status": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload, default=str),
    }


def _parse_json(body: bytes) -> dict:
    if not body:
        return {}
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def handle(method: str, path: str, headers: dict, body: bytes) -> dict:
    notes = datastore("notes")

    # Strip query string for routing.
    route = path.split("?", 1)[0].rstrip("/") or "/"

    if route == "/" and method == "GET":
        # Tiny health/hello endpoint so the smoke test has something
        # that doesn't depend on the datastore being writable.
        return _json_response({"ok": True, "service": "notes_demo"})

    if route == "/items" and method == "GET":
        return _json_response({"items": notes.list()})

    if route == "/items" and method == "POST":
        data = _parse_json(body)
        title = str(data.get("title") or "").strip()
        text = str(data.get("body") or "").strip()
        if not title:
            return _json_response({"error": "title is required"}, status=400)
        rec = notes.create({"title": title, "body": text})
        return _json_response({"item": rec}, status=201)

    if route.startswith("/items/") and method == "DELETE":
        item_id = route.split("/", 2)[2]
        ok = notes.delete(item_id)
        return _json_response({"ok": ok}, status=200 if ok else 404)

    return _json_response({"error": "not found", "path": route, "method": method}, status=404)
