"""Agents' own schedules in Flight Deck's Scheduled tasks.

An agent's ``cron`` tool (``tools/cron_tool.py``) keeps its jobs in the
agent's own ``data/sessions/sessions.db`` and runs them itself, in the chat
it set them up from. Scheduled tasks lists Flight Deck's own scheduler; these
routes put the agents' jobs next to it, so a job an agent set up in chat is
not invisible:

* the list reads each of the caller's agents' file read-only, so an asleep or
  turned-off agent's jobs show too (``agent_idle`` reads the same table to
  wake an agent before one is due);
* pause / resume / remove go through the agent's REST while its process runs,
  and straight into its file while it is down (nothing else writes it then);
* run now wakes the agent, or starts it as a scheduled task would, then asks it.

Only people use these routes, and only for their own agents (an admin, any).
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from captain_claw.cron import compute_next_run, now_utc, schedule_to_text, to_utc_iso
from captain_claw.flight_deck.fd_scheduler import (
    _NO_CACHE,
    _require_scheduler_caller,
    _SchedulerCaller,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter()

# A run's last monitor step that means it ended (cron_dispatch.execute_cron_job).
_END_STEPS = frozenset({"job_done", "job_queued", "job_failed"})
# A run still "started" after this long is not shown as running: the agent
# went down mid-run and never wrote its end.
_RUNNING_STALE_S = 3 * 3600.0
_RUNS_MAX = 30
_RESULT_MAX = 20_000
_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_ACTIONS = ("pause", "resume", "run")


def _srv():
    from captain_claw.flight_deck import server

    return server


# ── Whose agents ──────────────────────────────────────────────────────


def _require_person(request: Request) -> _SchedulerCaller:
    caller = _require_scheduler_caller(request)
    if not caller.is_human:
        # An agent manages its own jobs with its cron tool, never another's here.
        raise HTTPException(status_code=403, detail="only a person manages agents' schedules here")
    return caller


def _caller_agents(caller: _SchedulerCaller) -> dict[str, dict]:
    """slug → registry entry of every process agent the caller may see."""
    registry = _srv()._load_process_registry()
    if caller.sees_all:
        return dict(registry)
    if not caller.owner:
        return {}
    return {s: e for s, e in registry.items() if str(e.get("owner") or "") == caller.owner}


def _agent_for(caller: _SchedulerCaller, slug: str) -> dict:
    """The caller's agent ``slug``, or 404 (another person's too, so slugs don't leak)."""
    entry = _caller_agents(caller).get(slug)
    if entry is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return entry


def _agent_state(slug: str, entry: dict) -> str:
    """``on`` (its process runs), ``asleep`` (woken before its jobs) or ``off``
    (its own jobs don't run until it is on again)."""
    if _srv()._process_is_alive(slug):
        return "on"
    try:
        from captain_claw.flight_deck import agent_idle

        if agent_idle.is_asleep(slug, entry):
            return "asleep"
    except Exception:
        pass
    return "off"


# ── The agent's file ──────────────────────────────────────────────────


_SESSION_PATHS: dict[Path, tuple[float, str | None]] = {}


def _config_session_path(cfg: Path) -> str | None:
    """``session.path`` of one config file (cached by its mtime)."""
    try:
        mtime = cfg.stat().st_mtime
    except OSError:
        return None
    hit = _SESSION_PATHS.get(cfg)
    if hit and hit[0] == mtime:
        return hit[1]
    value: str | None = None
    try:
        import yaml

        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        section = data.get("session") if isinstance(data, dict) else None
        if isinstance(section, dict) and isinstance(section.get("path"), str) and section["path"].strip():
            value = section["path"].strip()
    except Exception:
        value = None
    _SESSION_PATHS[cfg] = (mtime, value)
    return value


def _db_path(slug: str) -> Path:
    """The agent's sessions.db as the agent finds it: ``session.path`` from its
    config (the home one wins, as at start-up; FD writes
    ``<agent>/data/sessions/sessions.db`` there), else the agent's default
    under its HOME (``data/home-config-parent``)."""
    agent_dir = _srv().DATA_DIR / slug
    home = agent_dir / "data" / "home-config-parent"
    for cfg in (home / ".captain-claw" / "config.yaml", agent_dir / "config.yaml"):
        raw = _config_session_path(cfg)
        if not raw:
            continue
        if raw.startswith("~"):
            path = home / raw[1:].lstrip("/\\")
        else:
            path = Path(raw)
            if not path.is_absolute():
                path = agent_dir / path  # the agent runs in its folder
        return path
    # No path set: the agent's own default, under its HOME. Where FD would
    # have put it only when that is all there is.
    home_default = home / ".captain-claw" / "sessions.db"
    fd_default = agent_dir / "data" / "sessions" / "sessions.db"
    if not home_default.is_file() and fd_default.is_file():
        return fd_default
    return home_default


def _connect(slug: str, *, write: bool = False) -> sqlite3.Connection | None:
    db = _db_path(slug)
    if not db.is_file():
        return None
    try:
        conn = sqlite3.connect(
            f"{db.resolve().as_uri()}?mode={'rw' if write else 'ro'}", uri=True, timeout=5.0,
        )
    except sqlite3.Error:
        return None
    conn.row_factory = sqlite3.Row
    return conn


def _loads(text: Any, default: Any) -> Any:
    if not isinstance(text, str) or not text:
        return default
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return default
    return value if isinstance(value, type(default)) else default


def _epoch(value: Any) -> float | None:
    if not value:
        return None
    from datetime import UTC, datetime

    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


_COLUMNS = (
    "id, kind, payload, schedule, session_id, enabled, created_at, updated_at, "
    "last_run_at, next_run_at, last_status, last_error"
)
# The last monitor step only, not the whole history (it can hold 400 events).
_LAST_STEP = (
    ", CASE WHEN json_valid(monitor_history) THEN json_extract(monitor_history, '$[#-1].step') END"
    " AS last_step"
    ", CASE WHEN json_valid(monitor_history) THEN json_extract(monitor_history, '$[#-1].timestamp') END"
    " AS last_step_at"
)


def _read_rows(slug: str) -> list[sqlite3.Row]:
    conn = _connect(slug)
    if conn is None:
        return []
    try:
        try:
            return conn.execute(f"SELECT {_COLUMNS}{_LAST_STEP} FROM cron_jobs").fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc):
                return []
            # A SQLite without JSON functions: no "running" then.
            return conn.execute(f"SELECT {_COLUMNS} FROM cron_jobs").fetchall()
    except sqlite3.Error as exc:
        log.debug("agent cron: read failed", slug=slug, error=str(exc)[:200])
        return []
    finally:
        conn.close()


def _task_text(kind: str, payload: dict) -> str:
    return str(payload.get("text") or payload.get("workflow") or payload.get("path") or kind)


def _public_job(slug: str, name: str, row: sqlite3.Row, state: str, now: float) -> dict[str, Any]:
    payload = _loads(row["payload"], {})
    schedule = _loads(row["schedule"], {})
    kind = str(row["kind"] or "prompt")
    try:
        sched_text = schedule_to_text(schedule)
    except Exception:
        sched_text = ""
    keys = row.keys()
    last_step = row["last_step"] if "last_step" in keys else None
    step_at = _epoch(row["last_step_at"]) if "last_step_at" in keys else None
    running = bool(
        state == "on" and last_step and last_step not in _END_STEPS
        and step_at is not None and now - step_at < _RUNNING_STALE_S
    )
    return {
        "id": str(row["id"]),
        "agent_slug": slug,
        "agent_name": name,
        "agent_state": state,
        "kind": kind,
        "task": _task_text(kind, payload),
        # Who wrote it: "agent" for the cron tool, else a person (/cron, the panel).
        "author": str(payload.get("author") or ""),
        "schedule": sched_text,
        "schedule_type": str(schedule.get("type") or ""),
        "session_id": str(row["session_id"] or ""),
        "enabled": bool(row["enabled"]),
        "created_at": _epoch(row["created_at"]),
        "updated_at": _epoch(row["updated_at"]),
        "last_run_at": _epoch(row["last_run_at"]),
        "next_run_at": _epoch(row["next_run_at"]),
        "last_status": str(row["last_status"] or ""),
        "last_error": str(row["last_error"] or ""),
        "running": running,
    }


def _edit_file(slug: str, job_id: str, action: str) -> bool:
    """Pause / resume / remove in the file of an agent that isn't running.
    The same changes its REST makes (``web/rest_cron.py``). False: no such job."""
    conn = _connect(slug, write=True)
    if conn is None:
        return False
    now_iso = to_utc_iso(now_utc())
    try:
        with conn:
            if action == "remove":
                cur = conn.execute("DELETE FROM cron_jobs WHERE id = ?", (job_id,))
            elif action == "pause":
                cur = conn.execute(
                    "UPDATE cron_jobs SET enabled = 0, last_status = 'paused', updated_at = ? WHERE id = ?",
                    (now_iso, job_id),
                )
            elif action == "resume":
                row = conn.execute("SELECT schedule FROM cron_jobs WHERE id = ?", (job_id,)).fetchone()
                if row is None:
                    return False
                # From now, so a long-paused job doesn't fire the moment it's back.
                nxt = to_utc_iso(compute_next_run(_loads(row["schedule"], {})))
                cur = conn.execute(
                    "UPDATE cron_jobs SET enabled = 1, last_status = 'pending', next_run_at = ?,"
                    " updated_at = ? WHERE id = ?",
                    (nxt, now_iso, job_id),
                )
            else:
                return False
            return cur.rowcount > 0
    except sqlite3.Error as exc:
        raise HTTPException(status_code=500, detail=f"could not change the agent's schedule: {exc}") from exc
    finally:
        conn.close()


def _job_in_file(slug: str, job_id: str) -> sqlite3.Row | None:
    conn = _connect(slug)
    if conn is None:
        return None
    try:
        return conn.execute(
            f"SELECT {_COLUMNS}, chat_history FROM cron_jobs WHERE id = ?", (job_id,)
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()


# ── Its runs, from the job's own chat history ─────────────────────────


def runs_from_history(chat_history: list, *, alive: bool, now: float | None = None) -> list[dict[str, Any]]:
    """The job's runs, newest first, from the events ``cron_dispatch`` and
    ``prompt_execution`` append to it: ``[CRON] job start``, the prompt, the
    agent's answer, ``[CRON] job done|queued|failed``."""
    now = time.time() if now is None else now
    runs: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for ev in chat_history:
        if not isinstance(ev, dict):
            continue
        role = str(ev.get("role") or "")
        content = str(ev.get("content") or "")
        at = _epoch(ev.get("timestamp"))
        if role == "system" and content.startswith("[CRON] job start"):
            if cur is not None and cur["status"] == "running":
                cur["status"] = "interrupted"
            cur = {"started_at": at, "finished_at": None, "trigger": str(ev.get("trigger") or ""),
                   "status": "running", "result": "", "error": ""}
            runs.append(cur)
            continue
        if cur is None:
            continue
        if role == "assistant":
            if content.strip():
                cur["result"] = content
            if cur["status"] == "queued":
                # The chat was busy; the answer came when it was free.
                cur["status"], cur["finished_at"] = "ok", at
        elif role == "system":
            if content.startswith("[CRON] job done"):
                if cur["status"] == "running":
                    cur["status"] = "ok"
                cur["finished_at"] = at
            elif content.startswith("[CRON] job queued"):
                if cur["status"] == "running":
                    cur["status"] = "queued"
            elif content.startswith(("[CRON] job failed", "[CRON] prompt failed")):
                cur["status"], cur["finished_at"] = "error", at
                cur["error"] = content.split(":", 1)[1].strip() if ":" in content else content
            elif content.startswith("[CRON] prompt cancelled"):
                cur["status"], cur["finished_at"] = "cancelled", at
    if cur is not None and cur["status"] == "running":
        started = cur["started_at"]
        if not alive or started is None or now - started >= _RUNNING_STALE_S:
            cur["status"] = "interrupted"
    out = []
    for i, run in enumerate(reversed(runs[-_RUNS_MAX:])):
        result = run["result"]
        if len(result) > _RESULT_MAX:
            result = result[:_RESULT_MAX] + "\n\n…"
        out.append({**run, "n": len(runs) - i, "result": result})
    return out


# ── The agent's REST ──────────────────────────────────────────────────


async def _agent_rest(entry: dict, method: str, path: str, name: str) -> dict[str, Any]:
    port = int(entry.get("web_port") or 0)
    if not port:
        raise HTTPException(status_code=503, detail=f"{name} has no port to reach it on")
    auth = str(entry.get("web_auth") or "")
    params = {"token": auth} if auth else {}
    try:
        async with httpx.AsyncClient(timeout=15.0, trust_env=False) as client:
            resp = await client.request(method, f"http://127.0.0.1:{port}{path}", params=params)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"could not reach {name}: {exc}") from exc
    try:
        data = resp.json()
    except ValueError:
        data = {"error": resp.text[:300]}
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="job not found")
    if resp.status_code >= 400:
        detail = data.get("error") if isinstance(data, dict) else None
        raise HTTPException(status_code=502, detail=str(detail or f"{name} answered {resp.status_code}"))
    return data if isinstance(data, dict) else {}


def _check_job_id(job_id: str) -> str:
    if not _JOB_ID.match(job_id or ""):
        raise HTTPException(status_code=404, detail="job not found")
    return job_id


# ── Routes ────────────────────────────────────────────────────────────


@router.get("/scheduler/agent-jobs")
async def list_agent_jobs(request: Request) -> JSONResponse:
    """Every job the caller's agents scheduled for themselves (their cron)."""
    caller = _require_person(request)
    now = time.time()
    out: list[dict[str, Any]] = []
    for slug, entry in _caller_agents(caller).items():
        rows = _read_rows(slug)
        if not rows:
            continue
        state = _agent_state(slug, entry)
        name = str(entry.get("name") or slug)
        out.extend(_public_job(slug, name, row, state, now) for row in rows)
    out.sort(key=lambda j: (not j["enabled"], j["next_run_at"] or float("inf"), j["agent_slug"]))
    return JSONResponse(out, headers=_NO_CACHE)


@router.get("/scheduler/agent-jobs/{slug}/{job_id}/runs")
async def agent_job_runs(slug: str, job_id: str, request: Request) -> JSONResponse:
    caller = _require_person(request)
    _agent_for(caller, slug)
    row = _job_in_file(slug, _check_job_id(job_id))
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    runs = runs_from_history(
        _loads(row["chat_history"], []), alive=_srv()._process_is_alive(slug),
    )
    return JSONResponse(runs, headers=_NO_CACHE)


@router.post("/scheduler/agent-jobs/{slug}/{job_id}/{action}")
async def agent_job_action(slug: str, job_id: str, action: str, request: Request) -> JSONResponse:
    """``pause`` / ``resume`` / ``run`` one of an agent's own jobs."""
    if action not in _ACTIONS:
        raise HTTPException(status_code=404, detail="unknown action")
    caller = _require_person(request)
    entry = _agent_for(caller, slug)
    _check_job_id(job_id)
    name = str(entry.get("name") or slug)
    if _job_in_file(slug, job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    if action == "run":
        from captain_claw.flight_deck import agent_idle

        # Woken if asleep, started if down — off again after, if its owner had it off.
        if not await agent_idle.ready_for_task(slug):
            raise HTTPException(status_code=503, detail=f"{name} could not be started for the run")
        entry = _srv()._load_process_registry().get(slug) or entry  # its port may have moved
        await _agent_rest(entry, "POST", f"/api/cron/jobs/{job_id}/run", name)
        agent_idle.touch(slug)
        return JSONResponse({"ok": True, "status": "started"}, headers=_NO_CACHE)
    if _srv()._process_is_alive(slug):
        await _agent_rest(entry, "POST", f"/api/cron/jobs/{job_id}/{action}", name)
    elif not _edit_file(slug, job_id, action):
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse({"ok": True}, headers=_NO_CACHE)


@router.delete("/scheduler/agent-jobs/{slug}/{job_id}")
async def delete_agent_job(slug: str, job_id: str, request: Request) -> JSONResponse:
    caller = _require_person(request)
    entry = _agent_for(caller, slug)
    _check_job_id(job_id)
    name = str(entry.get("name") or slug)
    if _srv()._process_is_alive(slug):
        await _agent_rest(entry, "DELETE", f"/api/cron/jobs/{job_id}", name)
    elif not _edit_file(slug, job_id, "remove"):
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse({"ok": True}, headers=_NO_CACHE)
