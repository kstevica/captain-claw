"""Flight Deck's side of the WhatsApp Web service (:mod:`.whatsapp_web`).

* **Start or adopt.** The service is started on demand — when someone links,
  or at Flight Deck start when links exist — and it outlives a Flight Deck
  restart, as agents do. :meth:`Supervisor.ensure_running` first looks for a
  running one through ``service.json`` (pid, port) and the shared
  ``service.secret``, and only spawns when there is none. The service's own
  lock makes a second copy exit (code 3) rather than fight the first over every
  WhatsApp session.
* **Restart.** A service this Flight Deck started that dies is started again
  with a growing pause (1 s … 60 s) while links exist and the deck allows it.
* **Bells.** Every few seconds the service's notices (unlinked on the phone,
  stream replaced, temporary ban, outdated) become the owner's notifications.
  Its ``incoming`` events — messages in chats the agent answers by itself —
  go to :mod:`.whatsapp_web_autoreply` instead.
* **Switches.** The admin's ``fd:whatsapp-web`` system setting (off by
  default) and the kill switch ``FD_WHATSAPP_WEB=off``. Turning the setting off
  stops the service; links stay on disk and come back when it's on again.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from captain_claw.flight_deck.whatsapp_web import DIR_NAME, kept_history
from captain_claw.flight_deck.whatsapp_web.link_state import LinkState
from captain_claw.logging import get_logger

log = get_logger(__name__)

SETTING = "fd:whatsapp-web"
BELL_TYPE = "whatsapp_web"
_DRAIN_EVERY = 3.0
_START_TIMEOUT = 25.0
_STOP_TIMEOUT = 10.0
_REQUEST_TIMEOUT = 30.0
_BACKOFF = (1.0, 2.0, 5.0, 10.0, 30.0, 60.0)


class ServiceDown(RuntimeError):
    """The service isn't running and couldn't be started."""


class ServiceError(RuntimeError):
    """The service answered with an error (``status``, ``detail``)."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def kill_switch() -> bool:
    return os.environ.get("FD_WHATSAPP_WEB", "").strip().lower() in ("off", "0", "false", "no")


def send_kill_switch() -> bool:
    """``FD_WHATSAPP_WEB_SEND=off``: no agent send, whatever the users chose."""
    return os.environ.get("FD_WHATSAPP_WEB_SEND", "").strip().lower() in ("off", "0", "false", "no")


def installed() -> bool:
    return importlib.util.find_spec("neonize") is not None


async def load_config(db: Any) -> dict[str, Any]:
    try:
        raw = json.loads(await db.get_system_setting(SETTING) or "{}")
    except ValueError:
        raw = {}
    return {"allowed": bool(raw.get("allowed")) if isinstance(raw, dict) else False}


async def save_config(db: Any, *, allowed: bool) -> dict[str, Any]:
    await db.set_system_setting(SETTING, json.dumps({"allowed": bool(allowed)}))
    return await load_config(db)


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False  # 0 / -1 would address a whole process group
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Supervisor:
    def __init__(self, data_dir: Path, db: Any) -> None:
        self.root = Path(data_dir) / DIR_NAME
        self.db = db
        self._proc: subprocess.Popen | None = None
        self._port = 0
        self._pid = 0
        self._seq = 0
        self._lock = asyncio.Lock()
        self._drain: asyncio.Task | None = None
        self._wanted = True
        self._failures = 0
        self._retry_at = 0.0

    # ── files ───────────────────────────────────────────────────────

    def _secret(self) -> str:
        path = self.root / "service.secret"
        try:
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
        except OSError:
            pass
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        value = secrets.token_urlsafe(32)
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(value)
        return value

    def _info(self) -> dict[str, Any]:
        try:
            raw = json.loads((self.root / "service.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def user_has_link(self, user_id: str) -> bool:
        return (self.root / user_id / "state.json").is_file()

    def any_links(self) -> bool:
        try:
            return any((d / "state.json").is_file() for d in self.root.iterdir() if d.is_dir())
        except OSError:
            return False

    def offline_view(self, user_id: str) -> dict[str, Any]:
        """A link's last known state, read from disk while the service is off."""
        st = LinkState.load(self.root / user_id / "state.json")
        if st is None:
            return {"state": "off", "running": False}
        out = st.view()
        out["running"] = False
        out["has_session"] = (self.root / user_id / "session.sqlite3").exists()
        return out

    def kept(self, user_id: str) -> list[dict[str, Any]]:
        """What's kept from the user's earlier links (read from disk)."""
        return kept_history.entries(self.root, user_id)

    def has_kept(self, user_id: str) -> bool:
        return bool(self.kept(user_id))

    def forget_local(self, user_id: str, *, keep_history: bool = True) -> None:
        """Forget a user's link without the service (it's off): their messages
        move aside first, unless they said to delete them. If they can't be
        moved, the OSError leaves everything in place."""
        import shutil

        if keep_history:
            st = LinkState.load(self.root / user_id / "state.json")
            kept_history.archive(self.root, user_id, self.root / user_id, jid=st.jid if st else "",
                                 linked_at=st.linked_at if st else "", reason="disconnected", now=time.time())
        shutil.rmtree(self.root / user_id, ignore_errors=True)

    def forget_kept_local(self, user_id: str, entry_id: str) -> int:
        """Delete what's kept from one earlier link without the service."""
        return kept_history.discard(self.root, user_id, entry_id)

    # ── the process ─────────────────────────────────────────────────

    async def _health(self, port: int) -> dict[str, Any] | None:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"http://127.0.0.1:{port}/health",
                                     headers={"Authorization": f"Bearer {self._secret()}"})
            return r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None

    async def _adopt(self) -> bool:
        info = self._info()
        pid, port = int(info.get("pid") or 0), int(info.get("port") or 0)
        if not (pid and port and _alive(pid)):
            return False
        health = await self._health(port)
        if not health or int(health.get("pid") or 0) != pid:
            return False
        if pid != self._pid:
            # A service this Flight Deck didn't see yet: its old notices were
            # rung already (or belong to a previous run); start from now.
            self._seq = await self._latest_seq(port)
        self._port, self._pid = port, pid
        return True

    async def _latest_seq(self, port: int) -> int:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"http://127.0.0.1:{port}/events",
                                     headers={"Authorization": f"Bearer {self._secret()}"})
            events = r.json().get("events") or []
            return max((int(e.get("seq") or 0) for e in events), default=0)
        except (httpx.HTTPError, ValueError, AttributeError):
            return 0

    def _child_env(self) -> dict[str, str]:
        env = {**os.environ, "WAW_DATA_DIR": str(self.root)}
        if not env.get("FD_DATA_KEY_FILE", "").strip():
            # The service seals with Flight Deck's data key. Resolved here: from
            # its own working directory a relative FD_DATA_DIR would point
            # somewhere else, and messages sealed there FD couldn't open.
            from captain_claw.flight_deck import fd_crypto

            env["FD_DATA_KEY_FILE"] = str(fd_crypto.key_path().resolve())
        return env

    async def _spawn(self) -> None:
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        self._secret()
        out = open(self.root / "service.out", "ab")  # noqa: SIM115 - handed to the child
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "captain_claw.flight_deck.whatsapp_web"],
                cwd=str(self.root),
                env=self._child_env(),
                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                # Its own session: Ctrl-C on Flight Deck's terminal doesn't take
                # every WhatsApp link down with it.
                start_new_session=True,
            )
        finally:
            out.close()
        deadline = time.monotonic() + _START_TIMEOUT
        while time.monotonic() < deadline:
            await asyncio.sleep(0.25)
            code = self._proc.poll()
            if code is not None:
                self._proc = None
                if code == 3 and await self._adopt():  # another copy holds the lock: use it
                    return
                raise ServiceDown(f"The WhatsApp Web service exited (code {code}); see {self.root / 'service.log'}")
            info = self._info()
            if int(info.get("pid") or 0) == self._proc.pid and await self._adopt():
                self._seq = 0  # a fresh service: every notice is new
                return
        raise ServiceDown("The WhatsApp Web service didn't come up in time")

    async def ensure_running(self) -> None:
        async with self._lock:
            self._wanted = True
            if self._port and await self._health(self._port):
                return
            self._port = 0
            if not await self._adopt():
                await self._spawn()
            self._failures = 0
            if self._drain is None or self._drain.done():
                self._drain = asyncio.get_running_loop().create_task(self._drain_loop())

    def running(self) -> bool:
        return bool(self._port)

    async def stop(self) -> None:
        """Stop the service (the admin turned WhatsApp Web off)."""
        async with self._lock:
            self._wanted = False
            await self._cancel_drain()
            # Signal only our own child or a pid the service itself confirms
            # (secret-authenticated /health): service.json may be stale and its
            # pid reused by some unrelated process.
            pid = self._proc.pid if self._proc is not None and self._proc.poll() is None else 0
            if not pid:
                info = self._info()
                port = int(info.get("port") or 0)
                health = await self._health(port) if port else None
                if health and int(health.get("pid") or 0) == int(info.get("pid") or 0):
                    pid = int(health["pid"])
            self._port = 0
            if pid and _alive(pid):
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pid = 0
                deadline = time.monotonic() + _STOP_TIMEOUT
                while pid and _alive(pid) and time.monotonic() < deadline:
                    if self._proc is not None:
                        self._proc.poll()  # reap our own child so it can't linger as a zombie
                    await asyncio.sleep(0.2)
                if pid and _alive(pid):
                    os.kill(pid, signal.SIGKILL)
            if self._proc is not None:
                try:
                    self._proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            self._proc, self._pid = None, 0

    async def detach(self) -> None:
        """Flight Deck is shutting down: stop watching, leave the service running."""
        await self._cancel_drain()

    async def _cancel_drain(self) -> None:
        task, self._drain = self._drain, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # ── calls ───────────────────────────────────────────────────────

    async def request(self, method: str, path: str, body: dict[str, Any] | None = None, *,
                      timeout: float = _REQUEST_TIMEOUT) -> dict[str, Any]:
        for attempt in (1, 2):
            await self.ensure_running()
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    r = await client.request(
                        method, f"http://127.0.0.1:{self._port}{path}", json=body,
                        headers={"Authorization": f"Bearer {self._secret()}"})
            except httpx.TransportError:
                self._port = 0
                if attempt == 2:
                    raise ServiceDown("The WhatsApp Web service isn't answering") from None
                continue
            try:
                data = r.json()
            except ValueError:
                data = {}
            if r.status_code >= 400:
                detail = data.get("detail") if isinstance(data, dict) else ""
                raise ServiceError(r.status_code, str(detail or f"HTTP {r.status_code}"))
            return data if isinstance(data, dict) else {}
        raise ServiceDown("The WhatsApp Web service isn't answering")

    async def request_raw(self, method: str, path: str, body: dict[str, Any] | None = None, *,
                          timeout: float = 180.0) -> tuple[bytes, dict[str, str]]:
        """Like :meth:`request`, for a route that answers with bytes (a file):
        returns (content, lower-cased headers)."""
        for attempt in (1, 2):
            await self.ensure_running()
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    r = await client.request(
                        method, f"http://127.0.0.1:{self._port}{path}", json=body,
                        headers={"Authorization": f"Bearer {self._secret()}"})
            except httpx.TransportError:
                self._port = 0
                if attempt == 2:
                    raise ServiceDown("The WhatsApp Web service isn't answering") from None
                continue
            if r.status_code >= 400:
                try:
                    detail = (r.json() or {}).get("detail")
                except ValueError:
                    detail = ""
                raise ServiceError(r.status_code, str(detail or f"HTTP {r.status_code}"))
            return r.content, {k.lower(): v for k, v in r.headers.items()}
        raise ServiceDown("The WhatsApp Web service isn't answering")

    # ── bells + restarts ────────────────────────────────────────────

    async def _drain_loop(self) -> None:
        while True:
            await asyncio.sleep(_DRAIN_EVERY)
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("WhatsApp Web supervisor tick failed: %s", exc)

    async def _tick(self) -> None:
        if self._proc is not None and self._proc.poll() is not None:
            log.warning("WhatsApp Web service exited (code %s)", self._proc.returncode)
            self._proc, self._port, self._pid = None, 0, 0
        if not self._port:
            if not (self._wanted and self.any_links()) or time.monotonic() < self._retry_at:
                return
            try:
                async with self._lock:
                    if not await self._adopt():
                        await self._spawn()
                self._failures = 0
            except ServiceDown as exc:
                pause = _BACKOFF[min(self._failures, len(_BACKOFF) - 1)]
                self._failures += 1
                self._retry_at = time.monotonic() + pause
                log.warning("WhatsApp Web service restart failed (%s); next try in %.0f s", exc, pause)
                return
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"http://127.0.0.1:{self._port}/events", params={"after": self._seq},
                                     headers={"Authorization": f"Bearer {self._secret()}"})
            events = r.json().get("events") or [] if r.status_code == 200 else []
        except (httpx.HTTPError, ValueError, AttributeError):
            self._port = 0
            return
        for e in events:
            self._seq = max(self._seq, int(e.get("seq") or 0))
            uid = str(e.get("user_id") or "")
            if not uid:
                continue
            if e.get("kind") == "incoming":
                from captain_claw.flight_deck import whatsapp_web_autoreply

                try:
                    whatsapp_web_autoreply.on_incoming(self, e)
                except Exception as exc:
                    log.warning("WhatsApp Web auto-reply for %s not queued: %s", uid, type(exc).__name__)
                continue
            try:
                await self.db.add_notification(uid, BELL_TYPE, str(e.get("title") or "")[:120],
                                               str(e.get("body") or "")[:300])
            except Exception as exc:
                log.warning("WhatsApp Web bell for %s failed: %s", uid, type(exc).__name__)


# ── the deck's one supervisor ───────────────────────────────────────

_SUPERVISOR: Supervisor | None = None


def get() -> Supervisor | None:
    return _SUPERVISOR


async def init(data_dir: Path, db: Any) -> Supervisor:
    """Called from Flight Deck's lifespan. Starts (or adopts) the service in the
    background when links exist and the deck allows WhatsApp Web."""
    global _SUPERVISOR
    s = _SUPERVISOR = Supervisor(data_dir, db)
    wanted = not kill_switch() and installed() and (await load_config(db))["allowed"]

    async def start() -> None:
        try:
            if wanted and s.any_links():
                await s.ensure_running()
            elif not wanted and _alive(int(s._info().get("pid") or 0)):
                # Left running from before the switch went off: let go of every link.
                log.info("WhatsApp Web is off on this deck; stopping the service left running")
                await s.stop()
        except Exception as exc:
            log.warning("WhatsApp Web service not started: %s", exc)

    asyncio.get_running_loop().create_task(start())
    return s


async def shutdown() -> None:
    if _SUPERVISOR is not None:
        await _SUPERVISOR.detach()
