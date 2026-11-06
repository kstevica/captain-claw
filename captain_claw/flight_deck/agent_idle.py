"""Flight Deck's agent controller: idle process agents go to sleep, and wake
when something needs them — memory on the server goes to agents at work.

An admin sets how long an agent may sit idle (Settings → Agents; system config
``agent_idle_stop_minutes``, 0 = never). About once a minute the controller
stops every process agent idle for longer. It marks it ``asleep`` in the
process registry, which is not its owner's "stopped": a sleeping agent is
still on, it just isn't holding memory, and Flight Deck's restart leaves it
asleep instead of starting it again.

Idle means nothing went through it for that long: no chat frame through Flight
Deck's proxy, no traffic on a channel bound to it (WhatsApp, Telegram, the
devices, the glasses), no scheduled run — and the agent itself says it isn't
in a turn (``GET /api/activity``; an agent too old to answer is judged on
Flight Deck's own view).

It wakes:

* ``WAKE_LEAD_S`` (5 minutes) before something scheduled for it is due: a
  scheduled task of Flight Deck's that prompts it, or a job of its own cron
  (read from its sessions.db). An agent with one due that soon is not put to
  sleep either. A scheduled task never fails for its agent being down: one
  its owner turned off is started for the run and turned off again after
  (``ready_for_task`` / ``release_after_task``), one that crashed is started;
* when something needs it (``wake`` / ``wake_port``): its chat opened or
  written to, a message on a channel bound to it, a member of a shared agent,
  a peer agent, a scheduled run, a team-bot Telegram message.

Never put to sleep: agents Flight Deck runs itself (a run's workers, a being's
body), an agent with its own Telegram / Slack / Discord bot (it polls them
itself, so nothing could wake it), and one with a scheduled run in flight.
Docker agents are left alone.

``FD_AGENT_IDLE_STOP_S`` (seconds) overrides the admin's setting for a deck
(operators, tests); ``FD_AGENT_IDLE_TICK_S`` sets how often it looks.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from captain_claw.logging import get_logger

log = get_logger(__name__)

# How far ahead of a scheduled run its agent is woken.
WAKE_LEAD_S = float(os.environ.get("FD_AGENT_IDLE_WAKE_LEAD_S", "300") or 300)
# How long a woken agent gets to answer on its port.
WAKE_TIMEOUT_S = 120.0
# One a scheduled task turned on goes off again once nothing reached it this
# long (the run's own end turns it off at once, see release_after_task).
_RELEASE_QUIET_S = 120.0
# The admin's setting, bounds in minutes (0 = off).
MIN_IDLE_MINUTES = 10
MAX_IDLE_MINUTES = 7 * 24 * 60

_STARTED_AT = time.time()
# slug → when Flight Deck last saw it at work (epoch seconds). Unknown agents
# count from this process's start: a restart gives every agent a full window.
_last_active: dict[str, float] = {}
# port → when it was last touched, so traffic on a busy socket doesn't read
# the registry for every frame.
_port_touched: dict[int, float] = {}
_PORT_TOUCH_EVERY_S = 10.0
# slug → the wake in progress (callers arriving meanwhile wait for the same one).
_waking: dict[str, asyncio.Task] = {}
# slug → held while it is being put to sleep or woken, so the two never overlap.
_locks: dict[str, asyncio.Lock] = {}
# Being put to sleep right now (a wake waits for it to finish).
_sleeping: set[str] = set()
# slug → (config mtimes, has its own bot), see _has_own_bot.
_own_bot_cache: dict[str, tuple[tuple[float, float], bool]] = {}


def _srv():
    # Lazy: server.py imports this module.
    from captain_claw.flight_deck import server

    return server


def _lock(slug: str) -> asyncio.Lock:
    return _locks.setdefault(slug, asyncio.Lock())


# ── Settings ──────────────────────────────────────────────────────────


def _env_idle_seconds() -> float | None:
    raw = os.environ.get("FD_AGENT_IDLE_STOP_S", "").strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


async def idle_seconds() -> float | None:
    """How long a process agent may sit idle before it is put to sleep (0 =
    never; None = the setting couldn't be read, so nothing changes now)."""
    env = _env_idle_seconds()
    if env is not None:
        return env
    try:
        from captain_claw.flight_deck.admin_routes import _load_system_config
        from captain_claw.flight_deck.auth import get_db

        cfg = _load_system_config(await get_db().get_system_setting("fd:system-config"))
        minutes = int(cfg.get("agent_idle_stop_minutes") or 0)
    except Exception:
        return None
    return float(minutes * 60) if minutes > 0 else 0.0


# ── Activity ──────────────────────────────────────────────────────────


def touch(slug: str) -> None:
    """``slug`` is at work now."""
    if slug:
        _last_active[slug] = time.time()


def touch_port(port: int | None) -> None:
    """The process agent on ``port`` is at work now (channel traffic)."""
    if not port:
        return
    now = time.time()
    if now - _port_touched.get(port, 0.0) < _PORT_TOUCH_EVERY_S:
        return
    _port_touched[port] = now
    slug = slug_for_port(port)
    if slug:
        _last_active[slug] = now


def last_active(slug: str) -> float:
    return _last_active.get(slug, _STARTED_AT)


def slug_for_port(port: int) -> str:
    """The process agent registered on ``port`` ("" if none). A running one
    wins: an agent woken while its own port was still held comes back on the
    next free one, which may be a sleeping agent's (until that one wakes and
    moves on in turn)."""
    srv = _srv()
    found = ""
    try:
        port = int(port)
        for slug, entry in srv._load_process_registry().items():
            if int(entry.get("web_port") or 0) == port:
                if srv._process_is_alive(slug):
                    return slug
                found = found or slug
    except Exception:
        pass
    return found


def is_asleep(slug: str, entry: dict | None = None) -> bool:
    """Put to sleep by this controller (and not running since)."""
    srv = _srv()
    if entry is None:
        entry = srv._load_process_registry().get(slug)
    return bool(entry and entry.get("asleep") and not entry.get("stopped")
                and not srv._process_is_alive(slug))


def is_available(slug: str) -> bool:
    """Running, or asleep — something that needs it can have it."""
    srv = _srv()
    return srv._process_is_alive(slug) or is_asleep(slug)


# ── Waking ────────────────────────────────────────────────────────────


async def _answers(port: int, token: str) -> bool:
    """The agent holding ``token`` answers on ``port`` — not whatever else
    listens there (another agent's 401)."""
    try:
        async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
            resp = await client.get(f"http://127.0.0.1:{int(port)}/api/activity",
                                    params={"token": token} if token else {})
    except Exception:
        return False
    # 404: an agent from before /api/activity — the token got it past auth.
    return resp.status_code not in (401, 403)


async def _wake(slug: str, timeout: float) -> bool:
    srv = _srv()
    loop = asyncio.get_running_loop()
    async with _lock(slug):
        if not is_asleep(slug):
            # Woken (or stopped by its owner) while this waited for the lock.
            return srv._process_is_alive(slug)
        log.info("agent-idle: waking", slug=slug)
        try:
            await loop.run_in_executor(None, srv._do_start_process, slug)
        except Exception as exc:
            log.warning("agent-idle: wake failed", slug=slug, error=str(exc)[:200])
            return False
        touch(slug)
    return await _until_it_answers(slug, timeout)


async def _until_it_answers(slug: str, timeout: float) -> bool:
    """Wait for a just-started ``slug`` to answer on its port."""
    srv = _srv()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        # Re-read: one whose port was taken comes back on another and announces it.
        entry = srv._load_process_registry().get(slug) or {}
        port = int(entry.get("web_port") or 0)
        if port and await _answers(port, str(entry.get("web_auth") or "")):
            touch(slug)
            return True
        if not srv._process_is_alive(slug):
            log.warning("agent-idle: woken agent exited", slug=slug)
            return False
        await asyncio.sleep(0.5)
    log.warning("agent-idle: woken agent not answering", slug=slug, timeout_s=timeout)
    return False


async def wake(slug: str, timeout: float = WAKE_TIMEOUT_S) -> bool:
    """Wake ``slug`` if this controller put it to sleep. True when it was asleep
    (or being woken) and now answers on its port; False when it wasn't asleep
    — running, stopped by its owner, unknown — or didn't come back."""
    if not slug:
        return False
    if slug in _sleeping:
        # Going down right now: let it finish, then wake it.
        async with _lock(slug):
            pass
    task = _waking.get(slug)
    if task is None:
        if not is_asleep(slug):
            return False
        task = asyncio.ensure_future(_wake(slug, timeout))
        _waking[slug] = task
        task.add_done_callback(lambda _t, s=slug: _waking.pop(s, None))
    try:
        return await asyncio.shield(task)
    except Exception:
        return False


async def wake_port(port: int | None, host: str | None = "localhost") -> bool:
    """``wake`` for the process agent on ``port`` of this machine."""
    if not port or (host or "localhost").strip().lower() not in ("localhost", "127.0.0.1", "::1", "[::1]"):
        return False
    slug = slug_for_port(int(port))
    return await wake(slug) if slug else False


_BACKGROUND: set[asyncio.Task] = set()


def wake_soon(slug: str) -> None:
    """Start waking ``slug`` without waiting for it (it's needed shortly)."""
    if not is_asleep(slug):
        return
    try:
        task = asyncio.ensure_future(wake(slug))
    except RuntimeError:
        return
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)


# ── On for a scheduled task ───────────────────────────────────────────
#
# A scheduled task never fails for its agent being down. Asleep, it is woken;
# down without being turned off (it crashed, or didn't come back), it is
# started; turned off by its owner, it is started for the run — marked
# ``on_for_task`` in the registry — and turned off again once the run is over
# and nothing else is due soon. Its owner turning it on meanwhile keeps it on.


def is_on_for_task(slug: str, entry: dict | None = None) -> bool:
    """Running only because a scheduled task needs it (its owner left it off)."""
    if entry is None:
        entry = _srv()._load_process_registry().get(slug)
    return bool(entry and entry.get("on_for_task") and not entry.get("stopped"))


async def _start_for_task(slug: str, timeout: float) -> bool:
    srv = _srv()
    loop = asyncio.get_running_loop()
    async with _lock(slug):
        entry = srv._load_process_registry().get(slug)
        if not entry:
            return False
        if not srv._process_is_alive(slug):
            off = bool(entry.get("stopped"))
            log.info("agent-idle: starting for a scheduled task", slug=slug,
                     was="off" if off else ("asleep" if entry.get("asleep") else "down"))
            try:
                await loop.run_in_executor(
                    None, lambda: srv._do_start_process(slug, for_task=off))
            except Exception as exc:
                log.warning("agent-idle: start for a task failed", slug=slug, error=str(exc)[:200])
                return False
        touch(slug)
    return await _until_it_answers(slug, timeout)


async def ready_for_task(slug: str, timeout: float = WAKE_TIMEOUT_S) -> bool:
    """Make sure ``slug`` answers before a scheduled task prompts it: woken if
    asleep, started if down — for the run only if its owner turned it off.
    True when it answers on its port."""
    if not slug:
        return False
    if slug in _sleeping:
        async with _lock(slug):
            pass
    srv = _srv()
    if slug in _waking or is_asleep(slug):
        if await wake(slug, timeout):
            return True
    if srv._process_is_alive(slug):
        entry = srv._load_process_registry().get(slug) or {}
        port = int(entry.get("web_port") or 0)
        if port and await _answers(port, str(entry.get("web_auth") or "")):
            touch(slug)
            return True
        # Up but not answering yet (just started by someone else): wait for it.
        return await _until_it_answers(slug, timeout)
    task = _waking.get(slug)
    if task is None:
        task = asyncio.ensure_future(_start_for_task(slug, timeout))
        _waking[slug] = task
        task.add_done_callback(lambda _t, s=slug: _waking.pop(s, None))
    try:
        return await asyncio.shield(task)
    except Exception:
        return False


def _turn_off_after_task(slug: str) -> bool:
    """Back to off, as its owner left it (blocking: the kill waits)."""
    srv = _srv()
    if not is_on_for_task(slug):
        return False
    srv._do_stop_process(slug)
    return True


async def release_after_task(slug: str, quiet_s: float = 0.0, now: float | None = None) -> bool:
    """A scheduled task started ``slug`` while its owner had it off; turn it
    off again — unless another scheduled run is in flight or due within
    ``WAKE_LEAD_S``, it is mid-turn, or (``quiet_s``) something reached it
    that recently. True when it went off."""
    if not slug or not is_on_for_task(slug):
        return False
    now = time.time() if now is None else now
    if quiet_s and now - last_active(slug) < quiet_s:
        return False
    if _scheduler_run_in_flight(slug):
        return False
    due = _scheduler_next_runs().get(slug)
    if due is not None and due - now <= WAKE_LEAD_S + _tick_seconds():
        return False
    entry = _srv()._load_process_registry().get(slug) or {}
    activity = await _agent_activity(entry)
    if activity is not None and activity.get("busy"):
        touch(slug)
        return False
    async with _lock(slug):
        if _scheduler_run_in_flight(slug):
            return False
        # A run arriving now waits for it to be down, then starts it again.
        _sleeping.add(slug)
        loop = asyncio.get_running_loop()
        try:
            done = await loop.run_in_executor(None, _turn_off_after_task, slug)
        except Exception as exc:
            log.warning("agent-idle: turning off after a task failed", slug=slug, error=str(exc)[:200])
            return False
        finally:
            _sleeping.discard(slug)
    if done:
        log.info("agent-idle: off again after its scheduled task", slug=slug)
    return done


def release_after_task_soon(slug: str) -> None:
    """``release_after_task`` without waiting for it (a run just ended)."""
    if not is_on_for_task(slug):
        return
    try:
        task = asyncio.ensure_future(release_after_task(slug))
    except RuntimeError:
        return
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)


# ── Sleeping ──────────────────────────────────────────────────────────


def _sleep_process(slug: str) -> bool:
    """Stop ``slug`` and mark it asleep (blocking: the kill waits for it)."""
    srv = _srv()
    registry = srv._load_process_registry()
    entry = registry.get(slug)
    if not entry or entry.get("stopped") or not srv._process_is_alive(slug):
        return False
    # Asleep first: a chat that reconnects while it goes down must find it
    # asleep (and wait for a word to wake it), not dead.
    entry["asleep"] = True
    entry["asleep_at"] = time.time()
    srv._save_process_registry(registry)
    proc = srv._processes.get(slug)
    pid = proc.pid if proc and proc.poll() is None else entry.get("pid")
    if pid:
        srv._kill_pid(pid)
    # Re-read: the kill can take seconds, and a port announce meanwhile must
    # not be overwritten with the snapshot from before it.
    registry = srv._load_process_registry()
    fresh = registry.get(slug)
    if fresh is not None and fresh.get("asleep"):
        fresh["pid"] = None
        srv._save_process_registry(registry)
    srv._processes.pop(slug, None)
    return True


async def put_to_sleep(slug: str, idle_s: float = 0.0) -> bool:
    """Stop ``slug`` for being idle — unless something reached it meanwhile
    (``idle_s``: how long it must have been idle)."""
    async with _lock(slug):
        if idle_s and time.time() - last_active(slug) < idle_s:
            return False
        _sleeping.add(slug)
        loop = asyncio.get_running_loop()
        try:
            done = await loop.run_in_executor(None, _sleep_process, slug)
        except Exception as exc:
            log.warning("agent-idle: sleep failed", slug=slug, error=str(exc)[:200])
            return False
        finally:
            _sleeping.discard(slug)
    if done:
        log.info("agent-idle: asleep", slug=slug, idle_s=int(time.time() - last_active(slug)))
    return done


# ── What it is due to do ──────────────────────────────────────────────


def _scheduler_next_runs() -> dict[str, float]:
    """slug → its soonest scheduled task of Flight Deck's (enabled, prompts it)."""
    out: dict[str, float] = {}
    try:
        from captain_claw.flight_deck.fd_scheduler import get_store

        jobs = get_store().list()
    except Exception:
        return out
    for job in jobs:
        slug = str(job.get("agent_slug") or "")
        nxt = job.get("next_run_at")
        if not slug or not job.get("enabled") or job.get("flow_id") or nxt is None:
            continue
        try:
            at = float(nxt)
        except (TypeError, ValueError):
            continue
        out[slug] = min(out.get(slug, at), at)
    return out


def _parse_iso(value: Any) -> float | None:
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _agent_cron_next(slug: str) -> float | None:
    """The soonest enabled job of the agent's own cron, from its sessions.db."""
    db = _srv().DATA_DIR / slug / "data" / "sessions" / "sessions.db"
    if not db.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2.0)
    except sqlite3.Error:
        return None
    try:
        rows = conn.execute("SELECT next_run_at FROM cron_jobs WHERE enabled = 1").fetchall()
    except sqlite3.Error:
        return None  # no cron table yet
    finally:
        conn.close()
    times = [t for t in (_parse_iso(r[0]) for r in rows) if t is not None]
    return min(times) if times else None


def next_due(slug: str, scheduled: dict[str, float] | None = None) -> float | None:
    """When something scheduled for ``slug`` is next due (epoch), if anything is."""
    if scheduled is None:
        scheduled = _scheduler_next_runs()
    times = [t for t in (scheduled.get(slug), _agent_cron_next(slug)) if t is not None]
    return min(times) if times else None


def _scheduler_run_in_flight(slug: str) -> bool:
    try:
        from captain_claw.flight_deck.fd_scheduler import _AGENT_LOCKS

        lock = _AGENT_LOCKS.get(slug)
        # Waiters too: another run is next in line for it.
        return bool(lock and (lock.locked() or getattr(lock, "_waiters", None)))
    except Exception:
        return False


# ── Who is never put to sleep ─────────────────────────────────────────


def _bot_configured(cfg: dict, env: dict[str, str], name: str) -> bool:
    section = cfg.get(name) if isinstance(cfg.get(name), dict) else {}
    prefix = f"CLAW_{name.upper()}__"
    enabled = env.get(prefix + "ENABLED", str(section.get("enabled", "")))
    # The agent's config also takes a bare TELEGRAM_BOT_TOKEN (etc.), from its
    # .env or the environment it inherits from Flight Deck.
    token = (env.get(prefix + "BOT_TOKEN") or str(section.get("bot_token") or "")
             or env.get(f"{name.upper()}_BOT_TOKEN") or os.environ.get(f"{name.upper()}_BOT_TOKEN", ""))
    if str(enabled).strip().lower() not in ("true", "1", "yes", "on") or not token.strip():
        return False
    if name == "telegram":
        # The deck's shared team bot reaches the agent through Flight Deck's
        # relay, which wakes it (telegram_shared.enqueue) — not its own bot.
        base = env.get(prefix + "API_BASE_URL", str(section.get("api_base_url") or ""))
        if "/fd/telegram-relay" in base:
            return False
    return True


def _has_own_bot(slug: str) -> bool:
    """Telegram, Slack or Discord with a bot of its own: the agent polls it
    itself, and asleep it would miss every message."""
    agent_dir = _srv().DATA_DIR / slug
    cfg_path, env_path = agent_dir / "config.yaml", agent_dir / ".env"
    try:
        mtimes = (cfg_path.stat().st_mtime if cfg_path.is_file() else 0.0,
                  env_path.stat().st_mtime if env_path.is_file() else 0.0)
    except OSError:
        return False
    cached = _own_bot_cache.get(slug)
    if cached and cached[0] == mtimes:
        return cached[1]
    cfg: dict = {}
    env: dict[str, str] = {}
    try:
        import yaml

        if cfg_path.is_file():
            loaded = yaml.safe_load(cfg_path.read_text()) or {}
            cfg = loaded if isinstance(loaded, dict) else {}
        if env_path.is_file():
            for line in env_path.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    except Exception:
        # Can't tell: keep it awake rather than cut off a bot.
        _own_bot_cache[slug] = (mtimes, True)
        return True
    own = any(_bot_configured(cfg, env, n) for n in ("telegram", "slack", "discord"))
    _own_bot_cache[slug] = (mtimes, own)
    return own


def exempt(slug: str, entry: dict) -> bool:
    from captain_claw.flight_deck.agent_sharing import is_managed_agent

    return is_managed_agent(slug, str(entry.get("description") or "")) or _has_own_bot(slug)


# ── The agent's own view ──────────────────────────────────────────────


async def _agent_activity(entry: dict) -> dict | None:
    """``{busy, last_active_at}`` from the agent, or None (an older agent
    without the endpoint, or no answer)."""
    port = int(entry.get("web_port") or 0)
    if not port:
        return None
    params = {"token": str(entry.get("web_auth") or "")} if entry.get("web_auth") else {}
    try:
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            resp = await client.get(f"http://127.0.0.1:{port}/api/activity", params=params)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except (ValueError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


# ── The loop ──────────────────────────────────────────────────────────


async def tick(now: float | None = None) -> dict[str, list[str]]:
    """One pass: wake or start who is due, put the idle to sleep, turn off
    again who a scheduled task had on. Returns what it did."""
    srv = _srv()
    now = time.time() if now is None else now
    done: dict[str, list[str]] = {"woke": [], "slept": [], "started": [], "off": []}
    idle_s = await idle_seconds()
    registry = srv._load_process_registry()
    scheduled = _scheduler_next_runs()
    for slug, entry in list(registry.items()):
        if entry.get("stopped"):
            # Off by its owner, with a scheduled task due: on for the run.
            due = scheduled.get(slug)
            if due is not None and due - now <= WAKE_LEAD_S and not srv._process_is_alive(slug):
                if await ready_for_task(slug):
                    done["started"].append(slug)
            continue
        if is_on_for_task(slug, entry):
            if srv._process_is_alive(slug) and await release_after_task(
                    slug, quiet_s=_RELEASE_QUIET_S, now=now):
                done["off"].append(slug)
            continue
        if is_asleep(slug, entry):
            if idle_s is None:
                continue  # the setting couldn't be read: nothing changes now
            due = next_due(slug, scheduled)
            # Turned off: what it put to sleep runs again.
            if not idle_s or (due is not None and due - now <= WAKE_LEAD_S):
                if await wake(slug):
                    done["woke"].append(slug)
            continue
        if not srv._process_is_alive(slug):
            # Down without being turned off (it crashed, or didn't come back
            # after a restart), with a scheduled task due: up again for it.
            due = scheduled.get(slug)
            if due is not None and due - now <= WAKE_LEAD_S and entry.get("web_port"):
                if await ready_for_task(slug):
                    done["started"].append(slug)
            continue
        if not idle_s or exempt(slug, entry):
            continue
        if now - last_active(slug) < idle_s or slug in _waking:
            continue
        due = next_due(slug, scheduled)
        if due is not None and due - now <= WAKE_LEAD_S + _tick_seconds():
            continue
        if _scheduler_run_in_flight(slug):
            touch(slug)
            continue
        activity = await _agent_activity(entry)
        if activity is not None:
            if activity.get("busy"):
                touch(slug)
                continue
            try:
                theirs = float(activity.get("last_active_at") or 0)
            except (TypeError, ValueError):
                theirs = 0.0
            if theirs > last_active(slug):
                _last_active[slug] = min(theirs, now)
                if now - theirs < idle_s:
                    continue
        if await put_to_sleep(slug, idle_s):
            done["slept"].append(slug)
    return done


def _tick_seconds() -> float:
    try:
        return max(5.0, float(os.environ.get("FD_AGENT_IDLE_TICK_S", "60") or 60))
    except ValueError:
        return 60.0


async def controller_loop(stop_event: asyncio.Event) -> None:
    """Background loop, started from Flight Deck's lifespan."""
    every = _tick_seconds()
    log.info("agent-idle: controller started", tick_s=every)
    while not stop_event.is_set():
        try:
            await tick()
        except Exception as exc:
            log.warning("agent-idle: tick failed", error=str(exc)[:200])
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=every)
        except TimeoutError:
            pass
    log.info("agent-idle: controller stopped")
