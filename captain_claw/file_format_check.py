"""Files a turn produced that their own format can't read.

An agent often writes data files through a script rather than the write tool
(``python make_calendar.py``), so the check looks at what changed on disk in
the workspace during the turn, not at the write tool's calls. A file whose
format has a parser here (iCalendar, JSON, JSON Lines, XML, YAML, TOML) is
parsed; one that fails is reported with the place and the reason, so the agent
can fix it before it says the work is done. A file that reads fine, a format
without a parser here, and anything the turn didn't change are left alone.

A calendar file is the common case: a model writes ``DTSTART:20260414T173000+08:00``,
which no calendar program reads (RFC 5545 takes ``…T173000Z`` in UTC, or a
``TZID=`` parameter with a local time), and says the file is valid.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

# Folders never searched: tool caches, environments, version control.
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env",
    "site-packages", ".cache", ".mypy_cache", ".pytest_cache", ".tox", ".idea", ".vscode",
})
# Captain's own folders at the top of a workspace (its LLM logs, Mrav's
# state) and the write tool's disposable scratch space: written every turn,
# never a deliverable.
_OWN_TOP_DIRS = frozenset({"logs", "mrav"})
_SCRATCH = ("saved", "tmp")
MAX_ENTRIES = 20000      # directory entries looked at, at most
MAX_FILES = 25           # changed files parsed, at most
MAX_BYTES = 5_000_000    # larger files are not parsed
MAX_PROBLEMS_PER_FILE = 3

SUFFIXES = frozenset({".ics", ".ical", ".json", ".jsonl", ".ndjson", ".xml", ".yaml", ".yml", ".toml"})

# iCalendar properties whose value is a DATE or DATE-TIME (RFC 5545 3.8.x).
_ICS_DATE_PROPS = frozenset({
    "DTSTART", "DTEND", "DTSTAMP", "DUE", "RECURRENCE-ID", "CREATED", "LAST-MODIFIED",
    "EXDATE", "RDATE", "COMPLETED",
})
_ICS_DATE = re.compile(r"^\d{8}$")
_ICS_DATE_TIME = re.compile(r"^\d{8}T\d{6}Z?$")


def changed_files(root: str | os.PathLike | None, since: float) -> list[Path]:
    """Files under *root* with a checkable suffix, changed at or after *since*
    (a ``time.time()``), newest first, at most ``MAX_FILES``."""
    if not root or not since:
        return []
    base = Path(root)
    if not base.is_dir():
        return []
    found: list[tuple[float, Path]] = []
    seen = 0
    for dirpath, dirnames, filenames in os.walk(base):
        rel = Path(dirpath).relative_to(base).parts
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
            and not (not rel and d in _OWN_TOP_DIRS)
            and (*rel, d)[:2] != _SCRATCH
        ]
        for name in filenames:
            seen += 1
            if seen > MAX_ENTRIES:
                break
            path = Path(dirpath) / name
            if path.suffix.lower() not in SUFFIXES:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_mtime >= since and 0 < stat.st_size <= MAX_BYTES:
                found.append((stat.st_mtime, path))
        if seen > MAX_ENTRIES:
            break
    found.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in found[:MAX_FILES]]


def problems(path: Path) -> list[str]:
    """Why *path* doesn't read as its format: short reasons, empty when it does."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ["it is not UTF-8 text"]
    except OSError:
        return []
    suffix = path.suffix.lower()
    try:
        if suffix in (".ics", ".ical"):
            return _ics_problems(text)
        if suffix == ".json":
            json.loads(text)
        elif suffix in (".jsonl", ".ndjson"):
            for number, line in enumerate(text.splitlines(), start=1):
                if line.strip():
                    try:
                        json.loads(line)
                    except ValueError as exc:
                        return [f"line {number} is not JSON ({exc.msg})"]
        elif suffix == ".xml":
            ET.fromstring(text)
        elif suffix in (".yaml", ".yml"):
            import yaml

            list(yaml.safe_load_all(text))
        elif suffix == ".toml":
            tomllib.loads(text)
    except json.JSONDecodeError as exc:
        return [f"not valid JSON at line {exc.lineno}, column {exc.colno} ({exc.msg})"]
    except ET.ParseError as exc:
        return [f"not well-formed XML ({exc})"]
    except tomllib.TOMLDecodeError as exc:
        return [f"not valid TOML ({exc})"]
    except Exception as exc:  # noqa: BLE001 — yaml's errors, and any parser surprise
        name = type(exc).__name__
        if "yaml" in type(exc).__module__.lower():
            return [f"not valid YAML ({' '.join(str(exc).split())[:200]})"]
        return [f"could not be read as {suffix[1:].upper()} ({name})"]
    return []


def _ics_lines(text: str) -> list[tuple[int, str]]:
    """Content lines with folded continuations joined, each with the line
    number where it starts."""
    out: list[tuple[int, str]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if raw[:1] in (" ", "\t") and out:
            out[-1] = (out[-1][0], out[-1][1] + raw[1:])
        elif raw.strip():
            out.append((number, raw))
    return out


def _ics_problems(text: str) -> list[str]:
    found: list[str] = []
    lines = _ics_lines(text)
    if not lines or lines[0][1].strip().upper() != "BEGIN:VCALENDAR":
        found.append("it does not start with BEGIN:VCALENDAR")
    if not lines or lines[-1][1].strip().upper() != "END:VCALENDAR":
        found.append("it does not end with END:VCALENDAR")
    stack: list[str] = []
    for number, line in lines:
        if len(found) >= MAX_PROBLEMS_PER_FILE:
            break
        name_part, sep, value = line.partition(":")
        if not sep:
            found.append(f"line {number} has no ':' between name and value")
            continue
        name, *params = name_part.split(";")
        name = name.strip().upper()
        if name == "BEGIN":
            stack.append(value.strip().upper())
            continue
        if name == "END":
            component = value.strip().upper()
            if not stack or stack[-1] != component:
                found.append(f"line {number}: END:{component} does not close "
                             f"{('BEGIN:' + stack[-1]) if stack else 'an open component'}")
            else:
                stack.pop()
            continue
        if name not in _ICS_DATE_PROPS:
            continue
        param_map = {}
        for param in params:
            key, _, val = param.partition("=")
            param_map[key.strip().upper()] = val.strip()
        for item in value.strip().split(","):
            if param_map.get("VALUE", "").upper() == "DATE":
                ok = bool(_ICS_DATE.match(item))
                form = "YYYYMMDD"
            elif param_map.get("VALUE", "").upper() == "PERIOD":
                ok = True
            else:
                ok = bool(_ICS_DATE_TIME.match(item)) or bool(_ICS_DATE.match(item))
                form = "YYYYMMDDTHHMMSS, followed by Z for UTC or with a TZID= parameter"
            if ok and "TZID" in param_map and item.endswith("Z"):
                found.append(f"line {number}: {name} has a TZID and a UTC 'Z' time ({item}); use one")
                break
            if not ok:
                found.append(f"line {number}: {name} value {item!r} is not an iCalendar date-time ({form}; "
                             "an offset like +08:00 is not allowed)")
                break
    if stack and len(found) < MAX_PROBLEMS_PER_FILE:
        found.append(f"BEGIN:{stack[-1]} is never closed")
    return found[:MAX_PROBLEMS_PER_FILE]


def check(root: str | os.PathLike | None, since: float) -> list[str]:
    """One line per changed file that doesn't read as its format: the path
    (from *root*) and why."""
    report: list[str] = []
    for path in changed_files(root, since):
        found = problems(path)
        if found:
            try:
                shown = path.relative_to(Path(root)).as_posix()
            except ValueError:
                shown = str(path)
            report.append(f"{shown}: " + "; ".join(found))
    return report


def feedback(report: list[str]) -> str:
    return (
        "[Check before delivering] These files this turn wrote don't read as their own format, "
        "so a program that opens them will fail:\n"
        + "\n".join(f"- {line}" for line in report)
        + "\n\nFix each file (or the script that writes it), check it reads, then give your "
        "complete answer again. Do not mention this check."
    )
