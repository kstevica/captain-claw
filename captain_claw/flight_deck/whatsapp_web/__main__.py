"""``python -m captain_claw.flight_deck.whatsapp_web`` — the WhatsApp Web service.

Flight Deck starts it (:mod:`captain_claw.flight_deck.whatsapp_web_supervisor`)
with ``WAW_DATA_DIR`` set; nobody else needs to. On start it:

1. sets umask 077, so the session files are the owner's alone;
2. takes ``service.lock`` — a second copy on the same data exits with code 3
   instead of fighting the first over every session ("stream replaced");
3. reads ``service.secret`` (Flight Deck writes it);
4. loads neonize, or keeps serving with the reason it can't (the card shows it);
5. binds 127.0.0.1 on a free port, writes ``service.json`` ({pid, port}) and
   resumes every paired link.

It outlives a Flight Deck restart, as agents do: the next Flight Deck finds it
through ``service.json`` and adopts it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

LOCK_HELD = 3
# Each neonize client parks one worker thread in its Go call, and every other
# neonize call borrows one too: the default executor (cpu+4) would run dry.
_WORKERS = 128


def _lock(path: Path):
    import fcntl

    fh = path.open("a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def main() -> int:
    raw_root = os.environ.get("WAW_DATA_DIR", "").strip()
    if not raw_root:
        print("WAW_DATA_DIR is not set; Flight Deck starts this service", file=sys.stderr)
        return 2
    root = Path(raw_root).resolve()
    os.umask(0o077)
    root.mkdir(parents=True, mode=0o700, exist_ok=True)

    logging.basicConfig(
        filename=str(root / "service.log"), level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("whatsapp_web")

    lock = _lock(root / "service.lock")
    if lock is None:
        log.warning("another WhatsApp Web service holds %s; exiting", root)
        return LOCK_HELD

    try:
        secret = (root / "service.secret").read_text(encoding="utf-8").strip()
    except OSError:
        secret = ""
    if not secret:
        log.error("no service.secret in %s; Flight Deck writes it before starting the service", root)
        return 2

    from captain_claw.flight_deck.whatsapp_web import neonize_adapter
    from captain_claw.flight_deck.whatsapp_web.api import build_app
    from captain_claw.flight_deck.whatsapp_web.engine import Engine

    try:
        version = neonize_adapter.load()
        engine = Engine(root, adapter_factory=neonize_adapter.make_adapter)
    except neonize_adapter.RuntimeUnavailable as exc:
        log.warning("WhatsApp Web unavailable: %s", exc)
        version = ""
        engine = Engine(root, adapter_factory=None, unavailable=str(exc))

    import uvicorn

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

    info = root / "service.json"

    @asynccontextmanager
    async def lifespan(_app):
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=_WORKERS))
        tmp = info.with_suffix(".tmp")
        tmp.write_text(json.dumps({"pid": os.getpid(), "port": port, "neonize": version}), encoding="utf-8")
        tmp.replace(info)
        resumed = await engine.resume_all()
        maintenance = asyncio.get_running_loop().create_task(engine.maintain())
        delivery = asyncio.get_running_loop().create_task(engine.watch_delivery())
        log.info("WhatsApp Web service on 127.0.0.1:%d (neonize %s), %d link(s) resumed",
                 port, version or "unavailable", resumed)
        yield
        maintenance.cancel()
        delivery.cancel()
        await engine.shutdown()
        try:
            if json.loads(info.read_text(encoding="utf-8")).get("pid") == os.getpid():
                info.unlink()
        except (OSError, ValueError):
            pass
        log.info("WhatsApp Web service stopped")

    app = build_app(engine, secret, version=version, lifespan=lifespan)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    asyncio.run(server.serve(sockets=[sock]))
    lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
