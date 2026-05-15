"""Smoke test for the self-repair loop.

Exercises the same path an agent would walk after a broken
deploy: scaffold a bad backend → proxy → 500 → tail logs (must
contain the traceback) → rewrite backend → restart → proxy
succeeds. This is the regression guard for Module 8 — the
"backend errors feed back into the agent's tool loop" contract.

Run from project root::

    .venv/bin/python -m captain_claw.flight_deck.examples._smoke_test_repair
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path


_BROKEN_BACKEND = '''\
"""A backend that always raises — to exercise the self-repair loop."""

async def handle(method, path, headers, body):
    raise RuntimeError("intentional smoke-test failure")
'''


_GOOD_BACKEND = '''\
"""A backend that always returns 200."""

import json

async def handle(method, path, headers, body):
    return {
        "status": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"ok": True}),
    }
'''


_FRONTEND = "<!doctype html><html><body>OK</body></html>"


async def run() -> int:
    tmp_home = Path(tempfile.mkdtemp(prefix="fd-app-repair-"))
    os.environ["CAPTAIN_CLAW_FD_HOME"] = str(tmp_home)
    print(f"[repair] FD_HOME = {tmp_home}")

    from captain_claw.flight_deck import app_runtime

    slug = "repair_demo"
    target = app_runtime.app_dir(slug)
    (target / "backend.py").write_text(_BROKEN_BACKEND, encoding="utf-8")
    (target / "frontend.html").write_text(_FRONTEND, encoding="utf-8")
    app_runtime.write_app_manifest(
        slug, {"name": "Repair demo", "version": "0.1.0", "slug": slug},
    )

    runtime = app_runtime.get_runtime()
    await runtime.start()
    failures = 0
    try:
        # 1. Hit the broken backend — should produce a 500.
        status, _h, body = await runtime.proxy(slug, "GET", "/", {}, b"")
        body_text = body.decode("utf-8", errors="replace")
        print(f"[repair] broken GET / -> {status}: {body_text[:120]}")
        if status != 500:
            failures += 1
            print(f"[repair] FAIL: expected 500, got {status}")

        # 2. tail_logs MUST surface the traceback — that's the contract
        # the agent's self-repair loop depends on.
        logs = runtime.tail_logs(slug, n=200)
        stderr_blob = "\n".join(logs["stderr"])
        last_error = logs.get("last_error") or ""
        print(f"[repair] tail_logs stderr={len(logs['stderr'])} stdout={len(logs['stdout'])}")
        # The traceback shows up in either ``stderr`` (printed by the
        # subprocess) or ``last_error`` (captured by the runtime when
        # the proxied call raised). Either is acceptable for the
        # repair loop.
        repair_signal = "intentional smoke-test failure" in stderr_blob or \
                        "intentional smoke-test failure" in last_error
        if not repair_signal:
            failures += 1
            print(f"[repair] FAIL: stderr/last_error doesn't contain the original error")
            print(f"[repair]   last_error: {last_error[:200]}")
            print(f"[repair]   stderr tail: {stderr_blob[-200:]}")

        # 3. Rewrite the backend with the good version.
        (target / "backend.py").write_text(_GOOD_BACKEND, encoding="utf-8")

        # 4. Restart so the new backend.py is picked up.
        await runtime.restart(slug)

        # 5. Re-issue the request — should now succeed.
        status, _h, body = await runtime.proxy(slug, "GET", "/", {}, b"")
        body_text = body.decode("utf-8", errors="replace")
        print(f"[repair] repaired GET / -> {status}: {body_text}")
        if status != 200 or b'"ok": true' not in body:
            failures += 1
            print("[repair] FAIL: repaired backend didn't return 200/ok")

    finally:
        await runtime.shutdown()
        shutil.rmtree(tmp_home, ignore_errors=True)

    print(f"[repair] done, failures = {failures}")
    return failures


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
