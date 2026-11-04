"""What the per-turn loop guards count as a repeat.

Two guards keep a weak model from spinning inside one turn:

* the write-loop STOP (``agent_orchestration_mixin``): the same whole file
  written again and again;
* the duplicate-call guard (``agent_tool_loop_mixin``): the same call, with the
  same answer, again and again.

Both used to refuse ordinary iterative work, and the agent then stopped early
and claimed work it never finished: two targeted edits to a script fired the
STOP ("PDF built" before the script ran), a re-run after the script was fixed
was a "duplicate", and a second line range of the same file was a "duplicate"
too (a 496-page report read only as far as its front matter).

The rules here keep the loops caught and let the work through:

* only a whole-file write counts toward the STOP — an edit or an append is
  progress, not a rewrite;
* a read's range is part of its signature, so the next section is a new call;
* a call is new again once a file it touches has changed — an edit or a write
  through the tools zeroes the matching counts, and a change made any other
  way (a script, ``sed -i``) shows in the file's mtime and size.
"""

from __future__ import annotations

import json
import os
import shlex
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Tools that write a file whole (unless asked to append).
FULL_WRITE_TOOLS = frozenset({"write", "file_write"})
# Tools that change a file on disk.
FILE_CHANGE_TOOLS = frozenset({"write", "file_write", "edit", "file_edit"})
# Tools that run a command whose answer depends on the files it runs.
RUN_TOOLS = frozenset({"shell", "termux"})
# Arguments that pick a section of a file. One counts in a signature only when
# the tool declares it: an argument the tool ignores returns the same text, so
# it must not make a repeat look like a new call.
RANGE_ARGS = ("offset", "limit", "pages", "page", "start_page", "end_page")
# How many words of a command are checked for file names.
_MAX_COMMAND_FILES = 32


def _args_dict(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


# ── Write-loop STOP ───────────────────────────────────────────────────────


def full_rewrite_path(tool_name: str, arguments: Any) -> str:
    """The path a call writes whole, or "" when it doesn't.

    An ``edit`` changes part of a file and an append adds to it: neither is a
    rewrite, and a few of them in a row is how a script gets debugged.
    """
    if str(tool_name or "").strip().lower() not in FULL_WRITE_TOOLS:
        return ""
    args = _args_dict(arguments)
    if _truthy(args.get("append")):
        return ""
    return str(args.get("path", "") or args.get("file_path", "") or "").strip()


def write_loop_paths(tool_calls: Iterable[Any], counts: dict[str, int], loop_at: int) -> list[str]:
    """Count this iteration's whole-file rewrites into ``counts``.

    Returns the rewritten paths when the STOP should fire: every call in the
    iteration rewrote a whole file that has now been written ``loop_at`` times
    or more this turn. Otherwise returns [].
    """
    calls = list(tool_calls)
    paths: list[str] = []
    for tc in calls:
        path = full_rewrite_path(getattr(tc, "name", ""), getattr(tc, "arguments", None))
        if path:
            paths.append(path)
    for path in paths:
        counts[path] = counts.get(path, 0) + 1
    if paths and len(paths) == len(calls) and all(counts.get(p, 0) >= loop_at for p in paths):
        return paths
    return []


# ── Duplicate-call guard ──────────────────────────────────────────────────


def _norm_range_value(key: str, value: Any) -> str:
    """A range argument as it appears in a signature; "" when it is the default."""
    if value is None or value == "":
        return ""
    if key in ("offset", "limit", "page", "start_page", "end_page"):
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return str(value).strip()
        # offset 0 and 1 both start at the first line.
        if key == "offset" and number <= 1:
            return ""
        return str(number)
    return str(value).strip()


def range_signature(arguments: Any, declared: Iterable[str]) -> str:
    """The section part of a path tool's duplicate signature, e.g. ``|offset=200|limit=100``.

    Only range arguments in ``declared`` (the tool's own parameters) count.
    A whole-file read has no range part, so its signature is unchanged.
    """
    args = _args_dict(arguments)
    declared_set = set(declared)
    parts = []
    for key in RANGE_ARGS:
        if key not in declared_set:
            continue
        norm = _norm_range_value(key, args.get(key))
        if norm:
            parts.append(f"|{key}={norm}")
    return "".join(parts)


def declared_params(registry: Any, tool_name: str) -> set[str]:
    """The parameter names a registered tool declares (empty when unknown)."""
    try:
        tool = registry.get(tool_name)
        props = (getattr(tool, "parameters", None) or {}).get("properties") or {}
        return set(props)
    except Exception:  # noqa: BLE001 — a signature must never break a call
        return set()


def _resolve(raw: str, base: Path | None) -> Path | None:
    """An existing file a path argument names (workspace root first, then CWD)."""
    raw = str(raw or "").strip().strip("'\"")
    if not raw or raw.startswith("-") or "://" in raw or raw.startswith("vfs:"):
        return None
    try:
        p = Path(raw).expanduser()
        candidates = [p] if p.is_absolute() else [
            *([base / p] if base is not None else []), Path.cwd() / p,
        ]
        for cand in candidates:
            if cand.is_file():
                return cand.resolve()
    except (OSError, ValueError):
        return None
    return None


def _command_words(command: str) -> list[str]:
    try:
        words = shlex.split(command, posix=True)
    except ValueError:
        words = command.split()
    out: list[str] = []
    for word in words:
        # `a;b`, `x&&y`, `--input=data.csv`: look at each piece.
        for sep in (";", "&&", "||", "|", ">", "<", "="):
            word = word.replace(sep, " ")
        out.extend(w for w in word.split() if "." in w or "/" in w)
    return out


def touched_files(tool_lower: str, arguments: Any, base: Path | None, *, path_tools: Iterable[str]) -> list[Path]:
    """The existing files whose content decides a call's answer.

    A path tool: the file it reads. A shell command: the files it names
    (``python build_pdf.py``). Anything else: none.
    """
    args = _args_dict(arguments)
    if tool_lower in set(path_tools):
        found = _resolve(str(args.get("path", "") or ""), base)
        return [found] if found is not None else []
    if tool_lower in RUN_TOOLS:
        command = str(args.get("command", "") or "")
        files: list[Path] = []
        for word in _command_words(command)[:_MAX_COMMAND_FILES]:
            found = _resolve(word, base)
            if found is not None and found not in files:
                files.append(found)
        return files
    return []


def fingerprint(files: Iterable[Path]) -> tuple:
    """The state of ``files``: mtime and size, or None for one that is gone."""
    state = []
    for f in files:
        try:
            st = f.stat()
            state.append((str(f), st.st_mtime_ns, st.st_size))
        except OSError:
            state.append((str(f), None, None))
    return tuple(sorted(state, key=lambda s: s[0]))


def forget_repeats_after_change(counts: dict[str, int], changed_path: str, *, path_tools: Iterable[str]) -> list[str]:
    """A file just changed: re-running a command, or re-reading that file, is a new call.

    Zeroes the counts of every shell call (it may run the changed file, as
    ``make`` or ``pytest`` do without naming it) and of every path-tool call on
    a file with the changed file's name (the write may have landed under
    ``saved/`` while the read named the bare path). Counts are zeroed, not
    dropped: the keys also record which tools ran this turn. The count of the
    edit or write itself stays, so the same edit twice is still a repeat.
    """
    name = os.path.basename(str(changed_path or "").strip().rstrip("/\\"))
    tools = {t.lower() for t in path_tools}
    zeroed: list[str] = []
    for sig in list(counts):
        tool, _, rest = sig.partition("|")
        tool = tool.lower()
        hit = tool in RUN_TOOLS
        if not hit and name and tool in tools and rest.startswith("path="):
            hit = os.path.basename(rest[len("path="):].split("|", 1)[0]) == name
        if hit and counts[sig]:
            counts[sig] = 0
            zeroed.append(sig)
    return zeroed
