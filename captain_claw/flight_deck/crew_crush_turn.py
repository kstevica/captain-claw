"""One turn of Crush, as JSON lines for Crew to read.

``crush run`` prints only what the model says: no tool calls, no session, no usage.
All of that is in its session, which ``crush session show --json`` gives out while the
turn is running. So this starts the turn, reads the session every few seconds, and
prints each thing the first time it's there, in Crush's own words:

    {"type": "session", "uuid": …, "model": …, "cost": …, "prompt_tokens": …, "completion_tokens": …}
    {"type": "tool_call", "name": …, "input": "<json>"}
    {"type": "tool_result", "name": …, "content": …, "is_error": …}
    {"type": "text", "text": …}
    {"type": "alive"}                 it's moving, with nothing whole to show yet
    {"type": "error", "text": …}      why the turn failed

It exits with Crush's code. Flight Deck runs it with ``python -I`` and it's on its own
from then on, like any CLI of a turn, so it imports nothing from Captain.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

READ_EVERY_S = 3.0
_READ_SHARE = 10  # however slow a read is here, reading takes a tenth of the time at most
_READ_TIMEOUT_S = 30
_CONTENT_MAX = 20_000
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_CONFIG_ENV = "CREW_CRUSH_CONFIG"
_CONFIG_NAME = "crush.json"


def _say(line: dict[str, Any]) -> None:
    print(json.dumps(line, ensure_ascii=False), flush=True)


def _ask(binary: str, data_dir: str, *what: str) -> tuple[Any, int]:
    """Its session commands, and the size of the answer. Asked from the data folder, so a project's
    own crushrc isn't run again on every read, and with no fetching of its model catalog, which
    would make each one nearly a second."""
    env = {**os.environ, "CRUSH_DISABLE_PROVIDER_AUTO_UPDATE": "1"}
    try:
        done = subprocess.run([binary, "-D", data_dir, "-c", data_dir, "session", *what, "--json"], cwd=data_dir, env=env,  # noqa: S603
                              capture_output=True, text=True, timeout=_READ_TIMEOUT_S, stdin=subprocess.DEVNULL)
        return (json.loads(done.stdout), len(done.stdout)) if done.returncode == 0 else (None, 0)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None, 0


def _place_config(data_dir: str) -> None:
    """Crush reads the crush.json in its data folder last, over everything else. Its providers are what
    this turn was given, and anything else in there is what Crush keeps for itself."""
    wanted = json.loads(os.environ.pop(_CONFIG_ENV, "") or "{}")
    path = Path(data_dir) / _CONFIG_NAME
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        current = {}
    if not isinstance(current, dict):
        current = {}
    if current.get("providers", {}) == wanted:
        return
    current["providers"] = wanted
    with tempfile.NamedTemporaryFile("w", dir=data_dir, suffix=".tmp", delete=False, encoding="utf-8") as tmp:
        json.dump(current, tmp, indent=2)
    os.replace(tmp.name, path)


def _is_whole_call(part: dict[str, Any]) -> bool:
    try:
        return isinstance(json.loads(part.get("input") or ""), dict)
    except ValueError:
        return False


def _tell(session: dict[str, Any], told: dict[str, int], *, ended: bool = False, say=_say) -> int:
    """Print what hasn't been yet, and say how much that was. While the turn runs, the last part of a
    message may still be being written, and only a call's whole input is JSON."""
    count = 0
    for message in session.get("messages") or []:
        parts = message.get("parts") or []
        role, place = message.get("role"), told.get(message.get("id"), 0)
        while place < len(parts):
            part = parts[place]
            kind = part.get("type")
            if not ended and ((kind in ("text", "reasoning") and place == len(parts) - 1)
                              or (kind == "tool_call" and not _is_whole_call(part))):
                break
            if role == "assistant" and kind == "text" and str(part.get("text") or "").strip():
                say({"type": "text", "text": part["text"]})
                count += 1
            elif role == "assistant" and kind == "tool_call":
                say({"type": "tool_call", "name": part.get("name") or "", "input": part.get("input") or ""})
                count += 1
            elif role == "tool" and kind == "tool_result":
                say({"type": "tool_result", "name": part.get("name") or "", "content": str(part.get("content") or "")[:_CONTENT_MAX],
                     "is_error": bool(part.get("is_error"))})
                count += 1
            place += 1
        told[message.get("id")] = place
    return count


def _usage(session: dict[str, Any], cost_before: float) -> dict[str, Any]:
    """Crush's cost adds up over a session, and its token counts are its last call's."""
    meta = session.get("meta") or {}
    models = [m.get("model") for m in session.get("messages") or [] if m.get("role") == "assistant" and m.get("model")]
    return {"type": "session", "uuid": meta.get("uuid") or "", "model": models[-1] if models else "",
            "cost": max(0.0, float(meta.get("cost") or 0.0) - cost_before),
            "prompt_tokens": meta.get("prompt_tokens") or 0, "completion_tokens": meta.get("completion_tokens") or 0}


def why_it_failed(stderr: str) -> str:
    """Crush draws an error in a box, wrapped to fit a screen."""
    lines = [line.strip() for line in _ANSI.sub("", stderr).splitlines() if line.strip()]
    if lines and lines[0].upper() == "ERROR":
        lines = lines[1:]
    return " ".join(line for line in lines if not line.startswith("Try --help"))


class _Turn:
    def __init__(self, binary: str, data_dir: str, session: str) -> None:
        self.binary, self.data_dir, self.session = binary, data_dir, session
        self.told: dict[str, int] = {}
        self.others: set[str] = set()
        self.cost_before = 0.0
        self.size = 0
        self.usage: dict[str, Any] = {}
        self.spoke = False

    def _say(self, line: dict[str, Any]) -> None:
        self.spoke = self.spoke or line["type"] == "text"
        _say(line)

    def begin(self) -> None:
        """What was there before this turn isn't this turn's."""
        if self.session:
            found, self.size = _ask(self.binary, self.data_dir, "show", self.session)
            if found:
                _tell(found, self.told, ended=True, say=lambda _line: None)
                self.cost_before = float((found.get("meta") or {}).get("cost") or 0.0)
        else:
            found, _ = _ask(self.binary, self.data_dir, "list")
            self.others = {str(s.get("uuid")) for s in found or [] if isinstance(s, dict)}

    def read(self, *, ended: bool = False) -> None:
        found, size = _ask(self.binary, self.data_dir, *(("show", self.session) if self.session else ("last",)))
        if not found or (found.get("meta") or {}).get("uuid") in self.others:
            return
        self.session = str((found.get("meta") or {}).get("uuid") or "")
        usage = _usage(found, self.cost_before)
        if usage != self.usage:
            self.usage = usage
            self._say(usage)
        if not _tell(found, self.told, ended=ended, say=self._say) and size != self.size:
            self._say({"type": "alive"})
        self.size = size


def main(args: list[str]) -> int:
    """``crew-turn <crush> <data folder> <session, or nothing> <its arguments>``. The first word is how
    Flight Deck tells this from a command the worker is running."""
    _, binary, data_dir, session, *arguments = args
    # Stopping a turn signals its whole group: this stays to read how it ended. A handler,
    # not SIG_IGN, which Crush would inherit.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: None)
    os.makedirs(data_dir, mode=0o700, exist_ok=True)
    _place_config(data_dir)
    turn = _Turn(binary, data_dir, session)
    turn.begin()
    with tempfile.TemporaryFile("w+", dir=data_dir, encoding="utf-8") as out, tempfile.TemporaryFile("w+", dir=data_dir, encoding="utf-8") as err:
        proc = subprocess.Popen([binary, "-D", data_dir, *arguments], stdin=subprocess.DEVNULL, stdout=out, stderr=err)  # noqa: S603
        wait = READ_EVERY_S
        while True:
            try:
                code = proc.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                began = time.monotonic()
                turn.read()
                wait = max(READ_EVERY_S, (time.monotonic() - began) * _READ_SHARE)
        turn.read(ended=True)
        out.seek(0)
        err.seek(0)
        said, stderr = out.read(), err.read()
    # Its session is the better account, but if it can't be read, these words are still its final message.
    if not turn.spoke and said.strip():
        _say({"type": "text", "text": said.strip()})
    sys.stderr.write(stderr)
    if code != 0:
        _say({"type": "error", "text": why_it_failed(stderr) or f"Crush exited with code {code}."})
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
