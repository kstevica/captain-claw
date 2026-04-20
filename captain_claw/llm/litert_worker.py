"""Subprocess worker for litert-lm.

litert-lm's C++ engine has two failure modes that take down the host
process when it runs inline:

1. **KV-cache overflow** on long conversations — the ``.litertlm`` file's
   baked-in max-seq-length is exceeded mid-decode and the C++ side
   either calls ``LOG(FATAL)`` or hangs forever holding the GIL. The
   parent agent then loses its WebSocket clients and looks "disconnected"
   to Flight Deck even though the process is still bound on its port.
2. **GPU context exhaustion** when multiple ``Engine`` objects in the
   same process try to grab the same Metal device.

Running the engine in a dedicated child process isolates both failure
modes: the parent can SIGKILL a wedged child and respawn a fresh one
without losing its WebSocket clients or its session state.

Architecture
------------
* :func:`worker_main` — child-process entry point. Owns the
  ``litert_lm.Engine`` and answers ``send_message`` requests over a pair
  of multiprocessing queues. Uses stdlib logging only so that it does
  not pull captain_claw config initialization into the spawned child.
* :class:`LiteRTWorkerClient` — parent-side RPC client. Wraps the child
  process, enforces a wall-clock timeout on every call, detects crashes
  (timeout, queue-EOF, child died), and transparently respawns a fresh
  child on the next call.
* :func:`get_or_create_litert_worker` — process-wide registry keyed by
  ``(abs_model_path, backend, max_num_tokens)`` so that multiple
  ``LiteRTProvider`` instances in the same parent share one child
  process. The model is mmap'd once and the GPU context is grabbed
  once.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from typing import Any

# Parent-side structured logger. The worker (child process) deliberately
# uses stdlib logging instead so that ``spawn`` does not have to re-init
# captain_claw.config inside the subprocess.
try:
    from captain_claw.logging import get_logger

    log = get_logger(__name__)
except Exception:  # pragma: no cover - fallback for unusual import paths
    log = logging.getLogger(__name__)


# How long to wait for a single send_message call before declaring the
# child wedged. Configurable via env var; default 300s. Small prompts
# land in ~13s on Apple M5 Pro / Gemma-4 E4B, but when the engine is
# built for 32k context a full prefill (say ~20k tokens) takes
# significantly longer — so 300s (5 min) is the safe default that also
# matches the agent-side chat watchdog.
_DEFAULT_TIMEOUT = float(os.getenv("LITERT_TIMEOUT_SECONDS", "300") or 300)

# How long to give the worker to finish booting (load the model file
# and build the engine). Cold loads of multi-GB Gemma weights take a
# while on first run, so be generous.
_DEFAULT_BOOT_TIMEOUT = float(os.getenv("LITERT_BOOT_TIMEOUT_SECONDS", "600") or 600)

# When set to a truthy value, the worker rebuilds the ``Engine`` after
# every successful ``send_message``. This is an experimental switch for
# diagnosing state-leak vs preface-overflow theories: if the crash at
# round 4–5 disappears with recycling on, the leak is engine-internal;
# if it persists, the prompt itself is too long and we need to prune
# history. Off by default because the rebuild costs ~3–10s per turn
# (file is in OS page cache after first load, but the GPU context and
# VRAM upload are still rebuilt).
_RECYCLE_AFTER_EACH = str(
    os.getenv("LITERT_RECYCLE_AFTER_EACH", "") or ""
).strip().lower() in {"1", "true", "yes", "on"}


# ── Worker side (runs in the spawned child process) ──────────────────────


def _worker_log(msg: str, **fields: Any) -> None:
    """Tiny structured-ish logger for the child. Writes to stderr."""
    extras = " ".join(f"{k}={v!r}" for k, v in fields.items())
    line = f"[litert-worker pid={os.getpid()}] {msg}"
    if extras:
        line = f"{line} {extras}"
    try:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except Exception:  # pylint: disable=broad-exception-caught
        pass


def _redirect_child_fds_to_log() -> None:
    """Redirect fds 1/2 so native library writes don't reach the TUI.

    litert_lm's C++ engine writes initialization/progress messages
    straight to the inherited stderr, and our own ``_worker_log`` helper
    does the same. In the CLI those bytes land on top of the locked
    status and prompt rows. Swapping the child's fds for a log file
    stops the garbage at the source.
    """
    try:
        log_dir = os.path.expanduser("~/.captain-claw/logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"litert-worker-{os.getpid()}.log")
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            sys.stdout.flush()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        try:
            sys.stderr.flush()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)
    except Exception:  # pylint: disable=broad-exception-caught
        pass


def worker_main(
    req_q: "mp.Queue",
    resp_q: "mp.Queue",
    init_kwargs: dict[str, Any],
) -> None:
    """Child-process entry point. Loads the engine and serves requests.

    The child speaks a tiny request/response protocol over two queues.
    Every response is a ``dict`` with at least an ``op`` and an ``ok``
    key; on success the payload is keyed by op (``text`` for
    ``send_message``). On failure ``ok=False`` and the dict carries
    ``error`` + ``error_type`` strings.
    """
    _redirect_child_fds_to_log()

    try:
        import litert_lm  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        resp_q.put({"op": "boot", "ok": False, "error": f"import litert_lm failed: {e}"})
        return

    try:
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    model_path = str(init_kwargs.get("model_path", "")).strip()
    backend_name = str(init_kwargs.get("backend", "gpu") or "gpu").lower()
    max_num_tokens = int(init_kwargs.get("max_num_tokens", 8192) or 8192)
    recycle_after_each = bool(
        init_kwargs.get("recycle_after_each", _RECYCLE_AFTER_EACH)
    )

    backend_enum = (
        litert_lm.Backend.GPU if backend_name == "gpu" else litert_lm.Backend.CPU
    )

    def _build_engine() -> Any:
        """Construct a fresh ``litert_lm.Engine`` and time the call."""
        t0 = time.monotonic()
        eng = litert_lm.Engine(
            model_path,
            backend=backend_enum,
            max_num_tokens=max_num_tokens,
        )
        _worker_log(
            "engine built",
            elapsed_s=round(time.monotonic() - t0, 2),
        )
        return eng

    def _teardown_engine(eng: Any) -> None:
        """Best-effort engine shutdown. Swallows errors."""
        if eng is None:
            return
        try:
            close = getattr(eng, "close", None)
            if callable(close):
                close()
                return
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        try:
            exit_fn = getattr(eng, "__exit__", None)
            if callable(exit_fn):
                exit_fn(None, None, None)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    _worker_log(
        "booting engine",
        model_path=model_path,
        backend=backend_name,
        max_num_tokens=max_num_tokens,
        recycle_after_each=recycle_after_each,
    )

    try:
        engine = _build_engine()
    except Exception as e:  # noqa: BLE001
        resp_q.put({"op": "boot", "ok": False, "error": f"engine init failed: {e}"})
        return

    resp_q.put({"op": "boot", "ok": True})
    _worker_log("engine ready, entering serve loop")

    while True:
        try:
            req = req_q.get()
        except (EOFError, KeyboardInterrupt):
            break
        if not isinstance(req, dict):
            continue
        op = req.get("op")
        if op == "shutdown":
            _worker_log("shutdown requested, exiting")
            _teardown_engine(engine)
            break
        if op == "ping":
            resp_q.put({"op": "ping", "ok": True})
            continue
        if op == "send_message":
            preface = req.get("preface") or []
            last_user = req.get("last_user", "") or ""
            send_failed = False
            try:
                with engine.create_conversation(messages=preface) as conv:
                    response = conv.send_message(last_user)
                content = response.get("content") or []
                if content and isinstance(content[0], dict):
                    text = content[0].get("text", "") or ""
                else:
                    parts: list[str] = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", "") or "")
                    text = "".join(parts)
                resp_q.put({"op": "send_message", "ok": True, "text": text})
            except Exception as e:  # noqa: BLE001
                send_failed = True
                _worker_log(
                    "send_message failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                resp_q.put(
                    {
                        "op": "send_message",
                        "ok": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )

            # Optional engine recycle. We do this AFTER answering the
            # parent so the user-visible latency is hidden behind the
            # next request. The next call will block briefly on the
            # already-rebuilt engine instead of waiting now.
            #
            # If the rebuild itself fails, exit the worker entirely so
            # the parent's next call triggers a clean respawn.
            if recycle_after_each and not send_failed:
                try:
                    t0 = time.monotonic()
                    _teardown_engine(engine)
                    engine = None
                    # Give Metal / the GPU driver a moment to actually
                    # release the previous context before we ask for a
                    # fresh one. Without this pause the rebuild can land
                    # on top of a half-released context and either hang
                    # or crash the child.
                    _worker_log("engine torn down; pausing 2s before rebuild")
                    time.sleep(2.0)
                    engine = _build_engine()
                    _worker_log(
                        "engine recycled",
                        total_elapsed_s=round(time.monotonic() - t0, 2),
                    )
                except Exception as e:  # noqa: BLE001
                    _worker_log(
                        "engine recycle failed; exiting worker",
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    return
            continue
        # Unknown op — surface it instead of silently dropping.
        resp_q.put({"op": str(op or "unknown"), "ok": False, "error": "unknown op"})


# ── Parent side ──────────────────────────────────────────────────────────


class LiteRTWorkerCrashed(RuntimeError):
    """Raised when the worker process died, hung, or returned an error."""


class LiteRTWorkerClient:
    """Parent-side RPC client for a litert-lm worker child process.

    A single client owns one child and serializes calls into it via an
    ``asyncio.Lock`` so that multiple ``LiteRTProvider`` instances in
    the same parent can safely share the same engine. On timeout or
    crash the child is killed and the client marks itself dead; the
    next call transparently respawns a fresh child.
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "gpu",
        max_num_tokens: int = 32768,
        timeout: float = _DEFAULT_TIMEOUT,
        boot_timeout: float = _DEFAULT_BOOT_TIMEOUT,
        recycle_after_each: bool | None = None,
    ):
        self.model_path = os.path.abspath(model_path)
        self.backend = (backend or "gpu").lower()
        self.max_num_tokens = int(max_num_tokens)
        self.timeout = float(timeout)
        self.boot_timeout = float(boot_timeout)
        # ``None`` means "follow the env var inside the worker".
        self.recycle_after_each = recycle_after_each

        # ``spawn`` is required on macOS — fork-after-Metal is undefined
        # behaviour and will deadlock the GPU driver.
        self._ctx = mp.get_context("spawn")
        # Async lock guards in-flight RPCs against concurrent providers
        # sharing this client. Threading lock guards (re)spawn against
        # racing callers.
        self._lock = asyncio.Lock()
        self._spawn_lock = threading.Lock()

        self._proc: Any = None  # mp.process.BaseProcess | None
        self._req_q: Any = None  # mp.Queue | None
        self._resp_q: Any = None  # mp.Queue | None

    # -- lifecycle --------------------------------------------------------

    def _spawn_locked(self) -> None:
        """Start a fresh worker process and wait for the boot ack.

        Caller must hold ``_spawn_lock``. Existing worker (if any) is
        torn down first.
        """
        if self._proc is not None and self._proc.is_alive():
            return
        self._cleanup_locked()

        req_q = self._ctx.Queue()
        resp_q = self._ctx.Queue()
        init_kwargs: dict[str, Any] = {
            "model_path": self.model_path,
            "backend": self.backend,
            "max_num_tokens": self.max_num_tokens,
        }
        if self.recycle_after_each is not None:
            init_kwargs["recycle_after_each"] = bool(self.recycle_after_each)
        proc = self._ctx.Process(
            target=worker_main,
            args=(req_q, resp_q, init_kwargs),
            name=f"litert-worker[{os.path.basename(self.model_path)}]",
            daemon=True,
        )
        try:
            log.info(
                "Spawning litert worker subprocess",
                model_path=self.model_path,
                backend=self.backend,
                max_num_tokens=self.max_num_tokens,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        proc.start()

        # Wait for the boot ack so we know the engine is ready (or
        # report a clean failure if it isn't).
        try:
            ack = resp_q.get(timeout=self.boot_timeout)
        except Exception as e:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            try:
                proc.join(timeout=5)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            raise LiteRTWorkerCrashed(
                f"litert worker did not boot within {self.boot_timeout}s: {e}"
            ) from e

        if not (isinstance(ack, dict) and ack.get("op") == "boot" and ack.get("ok")):
            err = (
                ack.get("error", "unknown")
                if isinstance(ack, dict)
                else f"unexpected boot ack: {ack!r}"
            )
            try:
                proc.kill()
                proc.join(timeout=5)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            raise LiteRTWorkerCrashed(f"litert worker boot failed: {err}")

        self._proc = proc
        self._req_q = req_q
        self._resp_q = resp_q
        try:
            log.info("Litert worker subprocess ready", pid=proc.pid)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    def _cleanup_locked(self) -> None:
        """Tear down the current worker. Caller must hold ``_spawn_lock``.

        Also explicitly closes the multiprocessing queues so that the
        underlying semaphores are released back to the OS instead of
        being reported by ``resource_tracker`` as leaks at process
        exit.
        """
        proc = self._proc
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.kill()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            try:
                proc.join(timeout=5)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        # Explicitly close + join queue background threads. Without
        # this, ``mp.Queue``'s feeder thread keeps the semaphore alive
        # past parent shutdown and ``resource_tracker`` warns about
        # leaked semaphores.
        for q in (self._req_q, self._resp_q):
            if q is None:
                continue
            try:
                q.close()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            try:
                q.join_thread()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        self._proc = None
        self._req_q = None
        self._resp_q = None

    def _mark_dead(self) -> None:
        """Kill the worker so the next call respawns a fresh one."""
        with self._spawn_lock:
            self._cleanup_locked()

    def shutdown(self) -> None:
        """Politely stop the worker.

        Used on full provider shutdown. Provider.close() should NOT call
        this when the client is shared via the registry — let the child
        die when the parent exits (it's marked daemon).
        """
        with self._spawn_lock:
            if self._proc is None:
                self._cleanup_locked()
                return
            try:
                if self._req_q is not None and self._proc.is_alive():
                    self._req_q.put({"op": "shutdown"})
                    self._proc.join(timeout=5)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            self._cleanup_locked()

    # -- RPC --------------------------------------------------------------

    async def send_message(
        self,
        preface: list[dict[str, Any]],
        last_user: str,
    ) -> str:
        """Run a single ``send_message`` in the worker. Returns the answer text.

        Raises :class:`LiteRTWorkerCrashed` on timeout, crash, EOF, or
        worker-side exception. The worker is killed on timeout/crash so
        the next call will respawn a fresh process.
        """
        async with self._lock:
            # Ensure we have a live worker.
            need_spawn = self._proc is None or not getattr(self._proc, "is_alive", lambda: False)()
            if need_spawn:
                try:
                    await asyncio.to_thread(self._spawn_under_lock)
                except LiteRTWorkerCrashed:
                    raise
                except Exception as e:  # noqa: BLE001
                    raise LiteRTWorkerCrashed(f"failed to spawn worker: {e}") from e

            req_q = self._req_q
            resp_q = self._resp_q
            if req_q is None or resp_q is None:
                raise LiteRTWorkerCrashed("worker queues missing after spawn")

            req = {
                "op": "send_message",
                "preface": preface,
                "last_user": last_user,
            }
            try:
                await asyncio.to_thread(req_q.put, req)
            except Exception as e:  # noqa: BLE001
                self._mark_dead()
                raise LiteRTWorkerCrashed(f"failed to enqueue request: {e}") from e

            t0 = time.monotonic()
            try:
                resp = await asyncio.to_thread(self._blocking_get, resp_q, self.timeout)
            except Exception as e:  # noqa: BLE001
                elapsed = time.monotonic() - t0
                try:
                    log.error(
                        "Litert worker call timed out / failed",
                        elapsed_s=round(elapsed, 1),
                        timeout_s=self.timeout,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
                self._mark_dead()
                raise LiteRTWorkerCrashed(
                    f"litert worker did not respond within {self.timeout}s: {e}"
                ) from e

            if not isinstance(resp, dict) or resp.get("op") != "send_message":
                self._mark_dead()
                raise LiteRTWorkerCrashed(f"unexpected response from worker: {resp!r}")

            if not resp.get("ok"):
                # Worker raised but is still alive — surface the error
                # without killing the process so subsequent calls can
                # still use it.
                err = resp.get("error", "unknown error")
                err_type = resp.get("error_type", "Exception")
                raise LiteRTWorkerCrashed(f"{err_type}: {err}")

            return str(resp.get("text", "") or "")

    def _spawn_under_lock(self) -> None:
        """Acquire the threading spawn lock and (re)spawn the worker."""
        with self._spawn_lock:
            self._spawn_locked()

    @staticmethod
    def _blocking_get(q: Any, timeout: float) -> dict[str, Any]:
        """Blocking ``Queue.get`` with a hard wall-clock timeout."""
        return q.get(timeout=timeout)


# ── Process-wide registry ────────────────────────────────────────────────

_REGISTRY: dict[tuple[str, str, int], LiteRTWorkerClient] = {}
_REGISTRY_LOCK = threading.Lock()


def _shutdown_all_workers() -> None:
    """``atexit`` hook: cleanly stop every registered worker.

    Without this, ``multiprocessing.resource_tracker`` reports leaked
    semaphores at parent shutdown because the worker's ``mp.Queue``
    objects are still holding open fds.
    """
    with _REGISTRY_LOCK:
        clients = list(_REGISTRY.values())
        _REGISTRY.clear()
    for client in clients:
        try:
            client.shutdown()
        except Exception:  # pylint: disable=broad-exception-caught
            pass


atexit.register(_shutdown_all_workers)


def get_or_create_litert_worker(
    model_path: str,
    backend: str = "gpu",
    max_num_tokens: int = 32768,
    recycle_after_each: bool | None = None,
) -> LiteRTWorkerClient:
    """Return a shared worker client for ``(model_path, backend, max_num_tokens)``.

    Multiple ``LiteRTProvider`` instances in the same parent process
    will share the same child via this registry, so the model is mmap'd
    once and the GPU context is grabbed once. Different settings get
    different children.
    """
    key = (
        os.path.abspath(model_path),
        (backend or "gpu").lower(),
        int(max_num_tokens),
    )
    with _REGISTRY_LOCK:
        client = _REGISTRY.get(key)
        if client is None:
            client = LiteRTWorkerClient(
                model_path=key[0],
                backend=key[1],
                max_num_tokens=key[2],
                recycle_after_each=recycle_after_each,
            )
            _REGISTRY[key] = client
        return client
