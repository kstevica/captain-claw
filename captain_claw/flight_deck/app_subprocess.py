"""Worker entry point for an agent-coded app subprocess.

Spawned by :mod:`captain_claw.flight_deck.app_runtime` as::

    python -m captain_claw.flight_deck.app_subprocess \\
        --slug <slug> \\
        --socket <path> \\
        --code-dir <path> \\
        --mem-limit-mb <int>

Responsibilities (kept deliberately small):

* Apply ``RLIMIT_AS`` (address space) and ``RLIMIT_CPU`` so a runaway
  backend can't take down the host.
* Add ``--code-dir`` to ``sys.path`` and import the user's ``backend``
  module from there.
* Stand up an aiohttp app on the unix socket at ``--socket`` and route
  every incoming request through the user's ``handle()`` function.
* Honor SIGTERM by cleanly stopping the runner so the socket file is
  released.

The user's ``backend.py`` only needs to expose ``handle`` — either as
``async def handle(method, path, headers, body)`` or as a sync function
with the same signature. The return shape is::

    {
      "status":  int,            # default 200
      "headers": dict[str, str], # default {}
      "body":    bytes | str,    # str is utf-8 encoded; default b""
    }

Anything raised inside ``handle()`` is caught, logged to stderr (which
FD pipes into the self-repair channel), and returned to the client as a
500. The traceback travels in the response body so the agent can see it
without scraping logs.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import inspect
import json
import logging
import os
import resource
import signal
import sys
import traceback
from pathlib import Path
from typing import Any

from aiohttp import web


log = logging.getLogger("app_subprocess")


# ── rlimit setup ──────────────────────────────────────────────────────


def _apply_rlimits(mem_limit_mb: int) -> None:
    """Cap address space + CPU time so a bad app can't nuke the host.

    Caps are best-effort: on systems where the soft limit cannot be
    raised, we lower it instead of failing. The CPU limit is generous
    (long enough for legit batch work) but bounded — if an app loops
    forever, the kernel terminates it with SIGXCPU.
    """
    # Address space: hard cap based on caller's request.
    try:
        soft = max(64 * 1024 * 1024, mem_limit_mb * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
    except (ValueError, OSError) as exc:
        # macOS often refuses RLIMIT_AS — that's known, log and move on.
        log.warning("RLIMIT_AS not applied: %s", exc)

    # CPU seconds: kill runaway loops after a while. Generous so legit
    # work (image processing, parsing) isn't disrupted.
    try:
        cpu_seconds = int(os.environ.get("FD_APP_CPU_LIMIT_SECONDS", "600"))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except (ValueError, OSError) as exc:
        log.warning("RLIMIT_CPU not applied: %s", exc)

    # Core dumps: off. We don't want agent-generated apps littering the
    # filesystem with multi-gig cores on crash.
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass


# ── backend module loading ────────────────────────────────────────────


def _load_backend(code_dir: Path) -> Any:
    """Import ``backend.py`` from ``code_dir`` and return the module.

    We load via ``importlib.util`` rather than ``importlib.import_module``
    so the user's file name never collides with a real top-level
    ``backend`` package somewhere on the path.
    """
    backend_path = code_dir / "backend.py"
    if not backend_path.exists():
        raise FileNotFoundError(f"no backend.py at {backend_path}")

    spec = importlib.util.spec_from_file_location(
        "_user_backend",
        backend_path,
        submodule_search_locations=[str(code_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load backend.py at {backend_path}")
    module = importlib.util.module_from_spec(spec)
    # Make the code-dir importable so the user's backend can split
    # across multiple files (helpers.py, models.py, etc.).
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    spec.loader.exec_module(module)

    if not hasattr(module, "handle"):
        raise AttributeError(
            "backend.py must define `handle(method, path, headers, body)`"
        )
    return module


# ── dispatch ──────────────────────────────────────────────────────────


async def _call_handle(backend: Any, method: str, path: str, headers: dict[str, str], body: bytes) -> dict[str, Any]:
    """Invoke the user's ``handle()`` regardless of sync/async."""
    fn = backend.handle
    if inspect.iscoroutinefunction(fn):
        result = await fn(method, path, headers, body)
    else:
        # Run sync handlers in the default thread pool so a slow
        # blocking call doesn't freeze the event loop for other
        # requests to the same subprocess.
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, fn, method, path, headers, body,
        )
    if not isinstance(result, dict):
        raise TypeError(
            f"handle() must return a dict, got {type(result).__name__}"
        )
    return result


def _normalize_response(result: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
    """Coerce a user-shaped handler return into ``(status, headers, body)``."""
    status_raw = result.get("status", 200)
    try:
        status = int(status_raw)
    except (TypeError, ValueError):
        status = 200
    raw_headers = result.get("headers") or {}
    if not isinstance(raw_headers, dict):
        raw_headers = {}
    headers = {str(k): str(v) for k, v in raw_headers.items()}
    body_raw = result.get("body", b"")
    if isinstance(body_raw, str):
        body = body_raw.encode("utf-8")
        headers.setdefault("Content-Type", "text/plain; charset=utf-8")
    elif isinstance(body_raw, (bytes, bytearray)):
        body = bytes(body_raw)
    elif body_raw is None:
        body = b""
    else:
        # Fall back to JSON for dicts/lists/scalars — the most common
        # shape for tiny apps.
        try:
            body = json.dumps(body_raw, default=str).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        except (TypeError, ValueError):
            body = repr(body_raw).encode("utf-8")
    return status, headers, body


def _make_handler(backend: Any):
    """Build the aiohttp request handler bound to a backend module."""

    async def _handle_request(request: web.Request) -> web.Response:
        body = await request.read()
        headers = {k: v for k, v in request.headers.items()}
        method = request.method
        # Include the query string in the path the user sees — most
        # ergonomic for tiny apps that just do ``if path == "/api/x":``.
        path_with_qs = request.path_qs
        try:
            result = await _call_handle(backend, method, path_with_qs, headers, body)
            status, out_headers, out_body = _normalize_response(result)
            return web.Response(status=status, headers=out_headers, body=out_body)
        except Exception:
            tb = traceback.format_exc()
            # Stream to stderr so FD's log pump captures it for the
            # self-repair channel.
            print(tb, file=sys.stderr, flush=True)
            # Also return it in the body so the agent's HTTP-level
            # tooling sees the traceback even without log access.
            body_text = (
                "Internal error in app backend:\n\n"
                + tb
                + "\n\nRequest: "
                + f"{method} {path_with_qs}"
            )
            return web.Response(
                status=500,
                headers={"Content-Type": "text/plain; charset=utf-8"},
                body=body_text.encode("utf-8"),
            )

    return _handle_request


# ── server boot ───────────────────────────────────────────────────────


async def _serve(slug: str, socket_path: Path, code_dir: Path) -> None:
    backend = _load_backend(code_dir)
    print(f"app_subprocess[{slug}] backend loaded from {code_dir}", flush=True)

    aio_app = web.Application(
        # Bigger client max for image uploads & multi-MB JSON payloads.
        client_max_size=32 * 1024 * 1024,
    )
    handler = _make_handler(backend)
    # Catch-all route: every method, every path, dispatched through
    # the user's handle().
    aio_app.router.add_route("*", "/{tail:.*}", handler)

    runner = web.AppRunner(aio_app, access_log=None)
    await runner.setup()

    # Clear any stale socket file. The parent does this too, but
    # double-up — a half-restart can leave a stale entry behind.
    try:
        socket_path.unlink(missing_ok=True)
    except OSError:
        pass

    site = web.UnixSite(runner, str(socket_path))
    await site.start()
    # Loosen socket perms so the parent (running as the same user) can
    # connect even if umask is strict. 0o600 keeps it user-only.
    try:
        os.chmod(socket_path, 0o600)
    except OSError:
        pass
    print(f"app_subprocess[{slug}] listening on {socket_path}", flush=True)

    # Block forever, until SIGTERM closes our event loop.
    stop_event = asyncio.Event()

    def _on_term() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _on_term)
        except NotImplementedError:
            # Windows: not supported. We won't run there in production
            # but don't crash if a developer tries.
            pass

    try:
        await stop_event.wait()
    finally:
        print(f"app_subprocess[{slug}] shutting down", flush=True)
        await runner.cleanup()
        try:
            socket_path.unlink(missing_ok=True)
        except OSError:
            pass


# ── entry point ───────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Flight Deck app subprocess worker")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--socket", required=True, help="Unix socket path to bind")
    parser.add_argument("--code-dir", required=True, help="App directory containing backend.py")
    parser.add_argument("--mem-limit-mb", type=int, default=512)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("FD_APP_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    _apply_rlimits(args.mem_limit_mb)

    socket_path = Path(args.socket)
    code_dir = Path(args.code_dir)

    try:
        asyncio.run(_serve(args.slug, socket_path, code_dir))
    except KeyboardInterrupt:
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
