"""The only module that imports neonize.

Three things neonize does on its own that the service must not allow:

* **It downloads a native library from GitHub releases** when the bundled one's
  version doesn't match the package (``neonize/_binder.py``). The library is
  loaded as soon as neonize is imported, so :func:`check_runtime` verifies the
  bundled library BEFORE the import (by reading the pinned version from
  ``neonize/download.py`` and asking the library itself), and turns the
  downloader into an error after it.
* **It needs libmagic to import**: ``python-magic`` loads it at import, though
  neonize only calls ``magic.from_buffer(…, mime=True)`` when it builds
  outgoing media. Without libmagic, :func:`_ensure_magic` installs a small
  signature sniffer in its place.
* **It drops exceptions from event handlers**: they run through
  ``asyncio.run_coroutine_threadsafe`` and nobody reads the future, so a
  handler that raises leaves no trace. Every handler here goes through
  :func:`guarded`.

:class:`NeonizeAdapter` turns neonize's events into the engine's plain ones
(``qr``, ``paired``, ``connected``, ``history`` …, see :mod:`.link_state`).
It never sends presence, never marks anything read and sends no messages.
"""

from __future__ import annotations

import asyncio
import ctypes
import importlib.util
import logging
import platform
import re
import sys
import time
import traceback
import types
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from captain_claw.flight_deck.whatsapp_web import NEONIZE_VERSION, parse
from captain_claw.flight_deck.whatsapp_web.link_state import DEVICE_NAME

log = logging.getLogger("whatsapp_web.neonize")

EventSink = Callable[[str, dict[str, Any]], Awaitable[None]]

# The address book arrives by app-state sync a little after connecting; read
# it (and the group names) then, and again now and then.
_DIRECTORY_FIRST = 20.0
_DIRECTORY_EVERY = 900.0

_VERSION_RE = re.compile(r'__GONEONIZE_VERSION__\s*=\s*["\']([^"\']+)["\']')


class RuntimeUnavailable(RuntimeError):
    """neonize can't run here; the message says why, in the admin's terms."""


# ── runtime checks ──────────────────────────────────────────────────


def _library_name() -> str:
    """The native library's file name, as neonize/utils/platform.py builds it."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = {"aarch64": "arm64", "x86_64": "amd64"}.get(machine, machine)
    ext = {"windows": "dll", "darwin": "dylib"}.get(system, "so")
    return f"neonize-{system}-{arch}.{ext}"


def check_runtime() -> str:
    """Check neonize can load without fetching anything; return its version.

    Raises RuntimeUnavailable naming what's wrong: not installed, a different
    version than the pinned one, or a bundled library that's missing or
    reports another version."""
    spec = importlib.util.find_spec("neonize")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeUnavailable(
            "WhatsApp Web isn't installed on this deck: pip install 'captain-claw[whatsapp-web]'")
    pkg = Path(next(iter(spec.submodule_search_locations)))
    try:
        from importlib.metadata import version

        installed = version("neonize")
    except Exception:
        installed = ""
    if installed != NEONIZE_VERSION:
        raise RuntimeUnavailable(
            f"WhatsApp Web needs neonize {NEONIZE_VERSION}, this deck has "
            f"{installed or 'an unknown version'}: pip install 'neonize=={NEONIZE_VERSION}'")
    try:
        expected = _VERSION_RE.search((pkg / "download.py").read_text(encoding="utf-8"))[1]
    except (OSError, TypeError):
        raise RuntimeUnavailable("neonize's install is incomplete (no download.py): reinstall it") from None
    lib = pkg / _library_name()
    if not lib.is_file():
        raise RuntimeUnavailable(
            f"neonize has no native library for this platform ({lib.name}); "
            "WhatsApp Web can't run here")
    try:
        gocode = ctypes.CDLL(str(lib))
        gocode.GetVersion.restype = ctypes.c_char_p
        loaded = gocode.GetVersion().decode()
    except (OSError, AttributeError) as exc:
        raise RuntimeUnavailable(f"neonize's native library won't load: {exc}") from None
    if loaded != expected:
        # neonize would now download a replacement from GitHub. Don't let it.
        raise RuntimeUnavailable(
            f"neonize's native library is {loaded}, the package expects {expected}: "
            f"reinstall neonize=={NEONIZE_VERSION}")
    return installed


def _ensure_magic() -> None:
    """Make ``import magic`` work without libmagic (see the module docstring)."""
    try:
        import magic  # noqa: F401
        return
    except ImportError:
        pass
    stub = types.ModuleType("magic")
    stub.from_buffer = _sniff_mime  # type: ignore[attr-defined]
    sys.modules["magic"] = stub
    log.warning("libmagic not found: using the built-in file-type sniffer for outgoing media")


_SIGNATURES: tuple[tuple[bytes, int, str], ...] = (
    (b"\xff\xd8\xff", 0, "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", 0, "image/png"),
    (b"GIF8", 0, "image/gif"),
    (b"%PDF", 0, "application/pdf"),
    (b"OggS", 0, "audio/ogg"),
    (b"ID3", 0, "audio/mpeg"),
    (b"PK\x03\x04", 0, "application/zip"),
)


def _sniff_mime(buffer: bytes, mime: bool = False) -> str:
    data = bytes(buffer[:64]) if buffer else b""
    for sig, offset, kind in _SIGNATURES:
        if data[offset:offset + len(sig)] == sig:
            return kind
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[4:8] == b"ftyp":
        return "audio/mp4" if data[8:12] in (b"M4A ", b"M4B ") else "video/mp4"
    if len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0:
        return "audio/mpeg"
    return "application/octet-stream"


def load() -> str:
    """:func:`check_runtime`, then import neonize with its downloader disabled."""
    version = check_runtime()
    _ensure_magic()
    import neonize._binder as binder
    import neonize.download as download

    def _refuse(*_a: Any, **_k: Any) -> None:
        raise RuntimeError(
            f"refusing to download a neonize library at runtime; reinstall neonize=={NEONIZE_VERSION}")

    download.download = _refuse  # type: ignore[assignment]
    binder.download = _refuse  # type: ignore[attr-defined]
    return version


# ── handlers ────────────────────────────────────────────────────────


def guarded(name: str, fn: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
    """*fn*, but an exception is logged with its traceback instead of vanishing."""

    async def wrapper(*args: Any) -> None:
        try:
            await fn(*args)
        except Exception:
            log.error("whatsapp_web handler %s failed:\n%s", name, traceback.format_exc())

    wrapper.__name__ = getattr(fn, "__name__", name)
    return wrapper


def _jid(j: Any) -> str:
    user = getattr(j, "User", "")
    return f"{user}@{getattr(j, 'Server', '')}" if user else ""


def _ban_expiry(expire: Any, now: float) -> float | None:
    """TemporaryBan.Expire is a duration (Go's time.Duration in nanoseconds,
    or seconds); return when the ban ends, as a unix time."""
    try:
        value = float(expire)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    seconds = value / 1e9 if value > 1e10 else value
    return now + seconds


class NeonizeAdapter:
    """One user's neonize client. ``connect`` starts it (QR events follow when
    the session has no device yet); ``stop`` disconnects without unlinking."""

    def __init__(self, client_id: str, session_path: Path, sink: EventSink, *,
                 history_days: int) -> None:
        from neonize.aioze import events as ev
        from neonize.aioze.client import NewAClient
        from neonize.proto.waCompanionReg.WAWebProtobufsCompanionReg_pb2 import DeviceProps

        props = DeviceProps(
            os=DEVICE_NAME,
            platformType=DeviceProps.DESKTOP,
            requireFullSync=True,
            historySyncConfig=DeviceProps.HistorySyncConfig(fullSyncDaysLimit=history_days),
        )
        self._client = NewAClient(str(session_path), props=props, uuid=client_id)
        self._sink = sink
        self._stopping = False
        self._directory: asyncio.Task | None = None
        c = self._client

        async def on_qr(_: Any, payload: bytes) -> None:
            await sink("qr", {"code": bytes(payload).decode("utf-8", "replace")})

        c.event.qr(guarded("qr", on_qr))

        def on(ev_type: Any, name: str, build: Callable[[Any], dict[str, Any]] | None = None) -> None:
            async def handler(_: Any, event: Any) -> None:
                await sink(name, build(event) if build else {})

            c.event(ev_type)(guarded(name, handler))

        def paired(e: Any) -> dict[str, Any]:
            return {"jid": _jid(e.ID), "error": e.Error, "ok": e.Status == e.SUCCESS}

        async def on_pair(_: Any, e: Any) -> None:
            data = paired(e)
            if data["ok"]:
                await sink("paired", data)
            else:
                await sink("connect_failure", {"reason": data["error"] or "pairing failed"})

        c.event(ev.PairStatusEv)(guarded("paired", on_pair))
        async def on_connected(_: Any, __: Any) -> None:
            await sink("connected", {})
            self._start_directory()

        c.event(ev.ConnectedEv)(guarded("connected", on_connected))
        on(ev.DisconnectedEv, "disconnected")
        on(ev.LoggedOutEv, "logged_out", lambda e: {"reason": str(e.Reason)})
        on(ev.StreamReplacedEv, "stream_replaced")
        on(ev.ClientOutdatedEv, "outdated")
        on(ev.ConnectFailureEv, "connect_failure", lambda e: {"reason": e.Message or str(e.Reason)})
        on(ev.TemporaryBanEv, "temporary_ban",
           lambda e: {"code": str(e.Code), "expires_at": _ban_expiry(e.Expire, time.time())})

        on(ev.HistorySyncEv, "history", lambda e: parse.history_chunk(e.Data))
        on(ev.MessageEv, "message", parse.live_row)
        # Whether what an agent sent reached the other phone (delivered / read),
        # or WhatsApp turned it away: WhatsApp accepting a send isn't delivery.
        on(ev.ReceiptEv, "receipt", parse.receipt)

    async def connect(self) -> None:
        self._stopping = False
        # connect() returns at once; the Go client runs in the task it hands
        # back, and a failure there would go unseen unless someone looks.
        task = await self._client.connect()
        if isinstance(task, asyncio.Future):
            task.add_done_callback(self._connect_done)

    def _connect_done(self, task: asyncio.Future) -> None:
        if task.cancelled() or self._stopping:
            return
        exc = task.exception()
        if exc is not None:
            log.error("whatsapp_web client stopped with an error: %s", exc)
            task.get_loop().create_task(self._sink("connect_failure", {"reason": str(exc)}))

    async def pair_code(self, phone: str) -> str:
        return str(await self._client.PairPhone(phone, True))

    async def download(self, field: str, raw_hex: str) -> bytes:
        """A message's media, from WhatsApp's servers (decrypted). *field* and
        *raw_hex* are what :func:`parse.content` kept: the media sub-message."""
        from neonize.proto.waE2E.WAWebProtobufsE2E_pb2 import Message

        msg = Message()
        getattr(msg, field).MergeFromString(bytes.fromhex(raw_hex))
        return bytes(await self._client.download_any(msg) or b"")

    async def send_text(self, jid: str, text: str) -> str:
        """Send one text message to an existing chat; WhatsApp's message id.
        Plain text only: no link preview, no mentions, no presence."""
        from neonize.utils import build_jid

        user, _, server = jid.partition("@")
        resp = await self._client.send_message(build_jid(user, server or "s.whatsapp.net"), text)
        return str(getattr(resp, "ID", "") or "")

    async def logout(self) -> None:
        await self._client.logout()

    def _start_directory(self) -> None:
        if self._directory is None or self._directory.done():
            self._directory = asyncio.get_running_loop().create_task(self._directory_loop())

    async def _directory_loop(self) -> None:
        await asyncio.sleep(_DIRECTORY_FIRST)
        while not self._stopping:
            try:
                await self._sink("directory", await self.directory())
            except Exception as exc:
                log.warning("whatsapp_web directory refresh failed: %s", exc)
            await asyncio.sleep(_DIRECTORY_EVERY)

    async def directory(self) -> dict[str, Any]:
        """The address book and the joined groups' names (reads only)."""
        contacts = []
        for ct in await self._client.contact.get_all_contacts():
            info = ct.Info
            contacts.append({
                "jid": parse.jid_str(ct.JID), "full_name": info.FullName, "first_name": info.FirstName,
                "push_name": info.PushName, "business_name": info.BusinessName,
            })
        groups = [{"jid": parse.jid_str(g.JID), "name": g.GroupName.Name}
                  for g in await self._client.get_joined_groups()]
        return {"contacts": contacts, "groups": groups}

    async def stop(self) -> None:
        self._stopping = True
        if self._directory is not None:
            self._directory.cancel()
        await self._client.stop()


def make_adapter(client_id: str, session_path: Path, sink: EventSink, *,
                 history_days: int) -> NeonizeAdapter:
    return NeonizeAdapter(client_id, session_path, sink, history_days=history_days)
