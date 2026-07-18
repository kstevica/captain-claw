"""Browser inference broker — tabs serve tokens, the server runs the loop.

A Flight Deck browser tab can register as an *inference worker* over
``/fd/infer-ws`` (WebLLM running in a WebWorker, WebGPU). Backend callers —
today the Mrav micro runtime via ``BrowserProvider`` — submit completion
jobs here; the broker routes each job to one of the owner's live workers
and awaits the result. The tab never executes tools; it only turns
(messages, schema) into tokens.

Design notes (docs/mrav-micro-agent-plan.md, Phase 2):
- Owner-scoped: a worker only ever serves its own user's jobs.
- Session pinning: jobs carrying the same ``session_key`` go to the same
  worker while it lives — WebLLM's KV delta-prefill turns a 10-40s cold
  8k prefill into ~1-3s per step, but only if the conversation keeps
  hitting the same engine.
- One job at a time per worker (the WebLLM engine is serial anyway);
  concurrent submits queue on the worker's lock.
- Framework-free: the FastAPI layer passes ``send`` as an async callable,
  so this module is fully unit-testable without sockets.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

SendFn = Callable[[dict], Awaitable[None]]

# Cold path = model may still be prefilling a full 8k prompt on an iGPU.
DEFAULT_JOB_TIMEOUT = 240.0


class NoWorkerError(Exception):
    """No live inference worker for this owner."""


class InferJobError(Exception):
    """The worker reported a failure for this job."""


@dataclass
class InferWorker:
    worker_id: str
    owner_id: str
    engine: str = "webllm"
    model: str = ""
    ctx_max: int = 8192
    schema_ok: bool = False
    vram_mb: int = 0
    send: SendFn | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    connected_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    jobs_done: int = 0
    jobs_failed: int = 0

    def to_status(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "engine": self.engine,
            "model": self.model,
            "ctx_max": self.ctx_max,
            "schema": self.schema_ok,
            "vram_mb": self.vram_mb,
            "connected_at": self.connected_at,
            "last_seen": self.last_seen,
            "jobs_done": self.jobs_done,
            "jobs_failed": self.jobs_failed,
            "busy": self.lock.locked(),
        }


@dataclass
class _Job:
    job_id: str
    owner_id: str
    worker_id: str
    future: asyncio.Future


class InferBroker:
    """Registry + job router for browser inference workers."""

    def __init__(self, job_timeout: float = DEFAULT_JOB_TIMEOUT):
        self.job_timeout = job_timeout
        self._workers: dict[str, InferWorker] = {}
        self._jobs: dict[str, _Job] = {}
        # (owner_id, session_key) → worker_id, for KV-cache affinity.
        self._pins: dict[tuple[str, str], str] = {}

    # ── worker lifecycle ──

    def register(self, owner_id: str, caps: dict[str, Any], send: SendFn) -> str:
        worker_id = uuid.uuid4().hex[:12]
        self._workers[worker_id] = InferWorker(
            worker_id=worker_id,
            owner_id=str(owner_id or ""),
            engine=str(caps.get("engine") or "webllm"),
            model=str(caps.get("model") or ""),
            ctx_max=int(caps.get("ctx_max") or 8192),
            schema_ok=bool(caps.get("schema")),
            vram_mb=int(caps.get("vram_mb") or 0),
            send=send,
        )
        log.info("infer worker registered", worker_id=worker_id,
                 owner=owner_id, model=caps.get("model"))
        return worker_id

    def unregister(self, worker_id: str) -> None:
        worker = self._workers.pop(worker_id, None)
        if worker is None:
            return
        self._pins = {k: v for k, v in self._pins.items() if v != worker_id}
        # Fail everything still waiting on this worker — a vanished tab
        # must surface as an error, not a silent hang until timeout.
        for job in list(self._jobs.values()):
            if job.worker_id == worker_id and not job.future.done():
                job.future.set_exception(
                    InferJobError("inference worker disconnected mid-job")
                )
        log.info("infer worker unregistered", worker_id=worker_id, owner=worker.owner_id)

    def workers_for(self, owner_id: str) -> list[InferWorker]:
        return [w for w in self._workers.values() if w.owner_id == str(owner_id or "")]

    def owner_ids(self) -> list[str]:
        return sorted({w.owner_id for w in self._workers.values()})

    def status(self, owner_id: str) -> list[dict[str, Any]]:
        return [w.to_status() for w in self.workers_for(owner_id)]

    # ── job routing ──

    def _pick(self, owner_id: str, session_key: str) -> InferWorker | None:
        candidates = self.workers_for(owner_id)
        if not candidates:
            return None
        key = (str(owner_id or ""), str(session_key or ""))
        pinned_id = self._pins.get(key)
        if pinned_id and pinned_id in self._workers:
            return self._workers[pinned_id]
        # Prefer an idle worker; fall back to the least loaded.
        chosen = next((w for w in candidates if not w.lock.locked()), candidates[0])
        if session_key:
            self._pins[key] = chosen.worker_id
        return chosen

    async def submit(
        self,
        owner_id: str,
        messages: list[dict[str, Any]],
        *,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        session_key: str = "",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Route one completion to a worker and await its result."""
        worker = self._pick(owner_id, session_key)
        if worker is None or worker.send is None:
            raise NoWorkerError(f"no inference worker online for owner {owner_id or '(unknown)'}")

        job_id = uuid.uuid4().hex[:16]
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._jobs[job_id] = _Job(job_id=job_id, owner_id=owner_id,
                                  worker_id=worker.worker_id, future=future)
        message = {
            "type": "job",
            "job_id": job_id,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if response_schema:
            message["response_schema"] = response_schema

        try:
            async with worker.lock:
                await worker.send(message)
                result = await asyncio.wait_for(future, timeout or self.job_timeout)
            worker.jobs_done += 1
            worker.last_seen = time.time()
            return result
        except (TimeoutError, InferJobError):
            worker.jobs_failed += 1
            raise
        finally:
            self._jobs.pop(job_id, None)

    def handle_message(self, worker_id: str, msg: dict[str, Any]) -> None:
        """Process one message from a worker socket."""
        worker = self._workers.get(worker_id)
        if worker is not None:
            worker.last_seen = time.time()
        mtype = str(msg.get("type") or "")
        if mtype == "pong" or mtype == "ping":
            return
        job = self._jobs.get(str(msg.get("job_id") or ""))
        if job is None or job.future.done():
            return
        if mtype == "result":
            job.future.set_result({
                "content": str(msg.get("content") or ""),
                "usage": msg.get("usage") or {},
                "model": str(msg.get("model") or (worker.model if worker else "")),
                "finish_reason": str(msg.get("finish_reason") or ""),
            })
        elif mtype == "error":
            job.future.set_exception(
                InferJobError(str(msg.get("message") or "worker error"))
            )


_broker: InferBroker | None = None


def get_infer_broker() -> InferBroker:
    """Process-wide broker singleton (one FD server → one broker)."""
    global _broker
    if _broker is None:
        _broker = InferBroker()
    return _broker
