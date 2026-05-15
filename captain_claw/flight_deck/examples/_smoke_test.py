"""Manual smoke test for the code-app runtime.

Spawns a fresh ``notes_demo`` app, proxies a few requests through the
runtime, and prints the results. Not a pytest — just a quick
end-to-end check during development.

Run from project root::

    .venv/bin/python -m captain_claw.flight_deck.examples._smoke_test
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


async def run() -> int:
    # Use an isolated FD home so we don't pollute the user's real data.
    tmp_home = Path(tempfile.mkdtemp(prefix="fd-app-smoke-"))
    os.environ["CAPTAIN_CLAW_FD_HOME"] = str(tmp_home)
    print(f"[smoke] FD_HOME = {tmp_home}")

    # Import after env var is set so module-level path resolution sees it.
    from captain_claw.flight_deck import app_runtime

    slug = "notes_demo"
    example = Path(__file__).parent / "notes_demo"
    target = app_runtime.app_dir(slug)
    # Copy demo backend + frontend into the app dir.
    shutil.copy(example / "backend.py", target / "backend.py")
    shutil.copy(example / "frontend.html", target / "frontend.html")
    app_runtime.write_app_manifest(slug, {
        "name": "Notes demo",
        "version": "0.1.0",
        "slug": slug,
    })

    runtime = app_runtime.get_runtime()
    await runtime.start()
    failures = 0
    try:
        # 1. Hello endpoint
        status, _h, body = await runtime.proxy(slug, "GET", "/", {}, b"")
        print(f"[smoke] GET / -> {status}: {body.decode()}")
        if status != 200 or b'"ok": true' not in body:
            failures += 1
            print("[smoke] FAIL: hello endpoint")

        # 2. List (empty)
        status, _h, body = await runtime.proxy(slug, "GET", "/items", {}, b"")
        print(f"[smoke] GET /items -> {status}: {body.decode()}")
        if status != 200 or b'"items": []' not in body:
            failures += 1
            print("[smoke] FAIL: initial empty list")

        # 3. Create
        payload = json.dumps({"title": "Hello", "body": "World"}).encode()
        status, _h, body = await runtime.proxy(
            slug, "POST", "/items",
            {"Content-Type": "application/json"},
            payload,
        )
        print(f"[smoke] POST /items -> {status}: {body.decode()[:120]}")
        if status != 201:
            failures += 1
            print("[smoke] FAIL: create")
        created = json.loads(body.decode()).get("item") or {}
        rec_id = created.get("id", "")

        # 4. List (one record)
        status, _h, body = await runtime.proxy(slug, "GET", "/items", {}, b"")
        items = json.loads(body.decode()).get("items") or []
        if status != 200 or len(items) != 1:
            failures += 1
            print(f"[smoke] FAIL: list after create returned {len(items)} items")
        else:
            print(f"[smoke] GET /items -> 1 record: {items[0]['title']}")

        # 5. Delete
        if rec_id:
            status, _h, body = await runtime.proxy(
                slug, "DELETE", f"/items/{rec_id}", {}, b"",
            )
            print(f"[smoke] DELETE /items/{rec_id} -> {status}: {body.decode()}")
            if status != 200:
                failures += 1
                print("[smoke] FAIL: delete")

        # 6. 404 path
        status, _h, body = await runtime.proxy(slug, "GET", "/missing", {}, b"")
        print(f"[smoke] GET /missing -> {status}: {body.decode()}")
        if status != 404:
            failures += 1
            print("[smoke] FAIL: 404 path")

        # 7. Log tail surface
        logs = runtime.tail_logs(slug, n=20)
        print(f"[smoke] tail_logs -> stderr={len(logs['stderr'])} stdout={len(logs['stdout'])} lines")

        # 8. List running
        live = runtime.list_running()
        print(f"[smoke] list_running -> {live}")
        if not live:
            failures += 1
            print("[smoke] FAIL: list_running empty")

    finally:
        await runtime.shutdown()
        shutil.rmtree(tmp_home, ignore_errors=True)

    print(f"[smoke] done, failures = {failures}")
    return failures


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
