"""Flight Deck's log on disk, and the polling noise kept out of it.

On a deck run in tmux the log lived only in the pane's scrollback: 2,000 lines,
about eleven minutes, because every open tab polls a handful of routes every
few seconds (``/fd/asks``, ``/fd/system/update/notice``, ``/fd/crew/tasks/…``,
``/fd/line/unread`` …) and each poll was an access line. Nothing from earlier
than that survived to look at.

* :class:`QuietPollingFilter` drops the access lines of a *successful* poll
  (2xx/304) on those routes; a failed one still shows. ``FD_ACCESS_LOG=all``
  keeps them all.
* :func:`install` adds a rotating file under ``<FD_DATA_DIR>/logs/``
  (``FD_LOG_FILE`` names another path; ``off`` turns it off). The deck's
  structlog lines (agent mail, the spend ledger …), which go straight to
  stderr, are copied into it too.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_NAME = "flight_deck.log"
MAX_BYTES = 20 * 1024 * 1024
BACKUPS = 5
_STRUCTLOG_LOGGER = "fd.structlog"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Routes a browser tab or an agent polls every few seconds.
_POLL_PREFIXES = (
    "/fd/asks", "/fd/system/update/notice", "/fd/line/unread", "/fd/notifications",
    "/fd/crew/tasks", "/fd/crew/usage", "/fd/processes", "/fd/containers",
    "/fd/shared-agents", "/fd/hosting", "/fd/taskboard/summary", "/scheduler/jobs",
    "/scheduler/agent-jobs", "/fd/quota/agent/usage", "/fd/tool-audit/agent/calls",
    "/fd/telegram-relay/", "/fd/google/access_token", "/fd/microsoft/access_token",
)
_POLL_RE = re.compile(r"^/fd/[\w-]+/agent[/_]status\b|^/fd/imap/agent_status\b")


def _is_poll(path: str) -> bool:
    bare = path.split("?", 1)[0]
    return bare.startswith(_POLL_PREFIXES) or bool(_POLL_RE.match(bare))


class QuietPollingFilter(logging.Filter):
    """Drop uvicorn access lines for successful polls (see module doc)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if (os.environ.get("FD_ACCESS_LOG") or "").strip().lower() == "all":
            return True
        args = record.args
        if not isinstance(args, tuple) or len(args) < 5:
            return True
        path, status = str(args[2] or ""), args[4]
        try:
            code = int(status)
        except (TypeError, ValueError):
            return True
        if (200 <= code < 300 or code == 304) and _is_poll(path):
            return False
        return True


class _FileFormatter(logging.Formatter):
    """Plain lines for the file; a copied structlog line is kept as it was
    rendered (it carries its own time and level), minus the colours."""

    def __init__(self) -> None:
        super().__init__("[%(asctime)s] %(levelname)-8s %(name)s  %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        if record.name == _STRUCTLOG_LOGGER:
            return _ANSI.sub("", record.getMessage())
        return _ANSI.sub("", super().format(record))


def log_path(data_dir: Path) -> Path | None:
    """Where the log goes, or None when ``FD_LOG_FILE=off``."""
    raw = (os.environ.get("FD_LOG_FILE") or "").strip()
    if raw.lower() in ("off", "0", "false", "no", "none"):
        return None
    return Path(raw).expanduser() if raw else Path(data_dir) / "logs" / LOG_NAME


def install(data_dir: Path) -> Path | None:
    """Add the rotating file to the root logger and copy structlog lines into
    it. Once per process; returns the file, or None when off or unwritable."""
    path = log_path(data_dir)
    if path is None:
        return None
    root = logging.getLogger()
    for h in root.handlers:
        if getattr(h, "_fd_logfile", False):
            return Path(h.baseFilename)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8")
    except OSError as exc:
        print(f"Flight Deck: log file {path} not opened ({exc}); logging to the terminal only",
              file=sys.stderr)
        return None
    handler._fd_logfile = True  # type: ignore[attr-defined]
    handler.setFormatter(_FileFormatter())
    handler.setLevel(logging.INFO)
    root.addHandler(handler)

    copy = logging.getLogger(_STRUCTLOG_LOGGER)
    copy.handlers = [handler]
    copy.propagate = False
    copy.setLevel(logging.INFO)

    from captain_claw.logging import set_system_log_sink

    def _tee(line: str) -> None:
        sys.stderr.write(line + "\n")
        try:
            copy.info(line)
        except Exception:  # noqa: BLE001 — the terminal still has it
            pass

    set_system_log_sink(_tee)
    return path
