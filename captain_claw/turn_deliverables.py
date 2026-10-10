"""Which files a chat turn may send back to the user: its deliverables.

A turn's tool outputs are untrusted text. A Drive download, a web page, an
email body or a vision model's description of a page can carry a line like
``Path: /Users/me/secret.pdf``, and a bridge that sends every file named on a
``Path:`` line sends that file into the chat. Channel bridges (Telegram, the
CLI bridges, WhatsApp) therefore send only files that are:

* inside the agent's own ``saved/`` area: never ``scripts/``, ``tools/`` or
  ``skills/`` (working code), never ``downloads/`` (inputs the agent fetched),
  never a hidden folder, never reached through a link leaving the area;
* created or changed during this turn;
* and, unless a file-making tool reported them (image_gen, termux,
  pocket_tts, a browser screenshot), named by the reply ("here's report.docx")
  — :func:`named_new_files` walks the folder itself, so no text can name a
  file into it.
"""

from __future__ import annotations

import os
import re
import stat
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# saved/ folders that never hold deliverables: working code, and inputs the
# agent fetched (a Drive file it summarises isn't sent back unasked).
WORKING_DIRS = ("scripts", "tools", "skills", "downloads")


def saved_root_of(agent: Any) -> Path | None:
    """The agent's ``saved/`` folder, or None when it can't be told."""
    try:
        return Path(agent.tools.get_saved_base_path(create=False)).resolve()
    except Exception:
        return None


def turn_started_at(session: Any, turn_start_idx: int) -> float:
    """Epoch seconds the turn began: the timestamp of its first message (the
    user's), or 0.0 when unknown."""
    try:
        msg = session.messages[max(0, int(turn_start_idx))]
        stamp = datetime.fromisoformat(str(msg.get("timestamp") or "").replace("Z", "+00:00"))
    except Exception:
        return 0.0
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp.timestamp()


def _is_working_dir(root: Path, name: str) -> bool:
    """Is ``root/name`` one of the WORKING_DIRS? By spelling (NFC, casefold)
    and by identity: a case-insensitive file system opens "Downloads/" or
    "downloadſ/" (long s) as downloads/, and realpath keeps the spelling."""
    if unicodedata.normalize("NFC", name).casefold() in WORKING_DIRS:
        return True
    try:
        st = (root / name).stat()
    except OSError:
        return True  # can't tell: not a deliverable
    for wd in WORKING_DIRS:
        try:
            wst = (root / wd).stat()
        except OSError:
            continue
        if (st.st_dev, st.st_ino) == (wst.st_dev, wst.st_ino):
            return True
    return False


def is_deliverable(path: Any, saved_root: Any, *, since: float = 0.0) -> bool:
    """A regular file inside ``saved_root`` (links resolved), in a folder that
    holds deliverables (not a working dir, nothing hidden), modified at or
    after ``since`` when given."""
    try:
        root = Path(os.path.realpath(os.fspath(saved_root)))
        real = Path(os.path.realpath(os.fspath(path)))
        rel = real.relative_to(root)
    except (ValueError, OSError, TypeError):
        return False
    parts = rel.parts
    if len(parts) < 2 or any(p.startswith(".") for p in parts) or _is_working_dir(root, parts[0]):
        return False
    try:
        st = real.stat()
    except OSError:
        return False
    if not stat.S_ISREG(st.st_mode):
        return False
    return not since or st.st_mtime >= since


def keep_deliverables(paths: list[Path], saved_root: Path | None, *, since: float = 0.0) -> list[Path]:
    """The paths that are deliverables (all of them dropped without a root)."""
    if saved_root is None:
        return []
    return [p for p in paths if is_deliverable(p, saved_root, since=since)]


def named_new_files(agent: Any, reply: str, since: float, extensions: set[str]) -> list[Path]:
    """Files this session wrote during the turn (under
    ``saved/<category>/<session>/``, modified at or after *since*) that the
    reply names by their whole file name — "here's report.docx". Not
    scripts, tools, skills or downloads; only the given extensions."""
    text = str(reply or "").lower()
    if not text or since <= 0:
        return []
    ext_re = re.compile(
        r"\.(?:" + "|".join(sorted(re.escape(e.lstrip(".")) for e in extensions)) + r")\b", re.IGNORECASE)
    if not ext_re.search(text):
        return []
    try:
        base = Path(agent.tools.get_saved_base_path(create=False))
        slug = agent._current_session_slug()
    except Exception:
        return []
    if not base.is_dir() or not slug:
        return []
    found: list[tuple[float, Path]] = []
    for category in base.iterdir():
        folder = category / slug
        if category.name.startswith(".") or _is_working_dir(base, category.name) or not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            try:
                if path.suffix.lower() not in extensions or not path.is_file():
                    continue
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime < since or not is_deliverable(path, base, since=since):
                continue
            if re.search(r"(?<![\w.-])" + re.escape(path.name.lower()) + r"(?![\w-])", text):
                found.append((mtime, path))
    found.sort()
    return [p for _m, p in found]
