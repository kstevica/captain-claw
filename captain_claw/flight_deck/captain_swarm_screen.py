"""Captain Swarm — sealing the scorer, and screening an attempt before it is measured.

docs/captain-swarm-plan.md §5. A swarm that can raise its score by cheating
will, so before an attempt's number counts:

* **Sealed scorer** (:func:`snapshot`): at start the scorer is copied out of
  the base commit into the run's sealed folder, and every measurement runs
  that copy against a clean copy of the attempt's commit — never anything the
  attempt left lying around, and never the attempt's own copy of the scorer.
* **Scope** (:func:`scope_problems`): every changed file must be one the
  objective lets attempts change, and none may be protected (the scorer, its
  cases, the tests). A miss disqualifies the attempt before it is measured.
* **Gaming scan** (:func:`gaming_hits`): the lines an attempt added must not
  carry what the cases contain — their ids, file names, expected answers — or
  point at the eval itself. Held-out strings in a diff mean the attempt saw
  cases it was never shown.

Pure, apart from git and file reads.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import shutil
import tarfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_GIT_TIMEOUT_S = 300.0
_MIN_STRING = 5
# What every eval leaves in a diff that games it, whatever its cases are.
EVAL_MARKERS = ("captain_swarm/sealed", "holdout.json", "mrav_eval", "evals/mrav", "MRAV_RESULT", "CLAW_CAPTAIN_SWARM")
_WORDS = frozenset("""
about after again because before being between could every first found given
their there these thing those three under until where which while would write
should other reply return string number value total files lines answer""".split())


# ── git ──────────────────────────────────────────────────────────────


async def git(repo: str | Path, *args: str, timeout: float = _GIT_TIMEOUT_S) -> tuple[int, str]:
    """``git -C repo args…`` → ``(exit code, stdout)``; stderr folded in on failure."""
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", str(repo), *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "git timed out"
    text = out.decode("utf-8", errors="replace")
    if proc.returncode:
        text = (text + err.decode("utf-8", errors="replace")).strip()
    return proc.returncode or 0, text.strip()


async def resolve(repo: str | Path, ref: str) -> str:
    """The full commit id *ref* names in *repo*, or "" when it names none."""
    rc, out = await git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    return out if rc == 0 and _SHA_RE.match(out or "") else ""


async def snapshot(repo: str | Path, sha: str, dest: Path, *, paths: Iterable[str] = (),
                   exclude: Iterable[str] = ()) -> None:
    """A clean copy of commit *sha* (only *paths*, minus *exclude*, when given) at
    *dest*: what was committed, nothing else. Replaces whatever *dest* held."""
    if not _SHA_RE.match(sha or ""):
        raise ValueError("a snapshot is of a commit id")
    spec = [*paths, *(f":(exclude){x}" for x in exclude)]
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", str(repo), "archive", "--format=tar", sha, *(["--", *spec] if spec else []),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await asyncio.wait_for(proc.communicate(), timeout=_GIT_TIMEOUT_S)
    if proc.returncode:
        raise RuntimeError(f"git archive failed: {err.decode('utf-8', errors='replace')[-300:]}")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    def extract() -> None:
        with tarfile.open(fileobj=io.BytesIO(out)) as tar:
            tar.extractall(dest, filter="data")
    await asyncio.to_thread(extract)


async def changed_files(repo: str | Path, base: str, head: str) -> list[str]:
    rc, out = await git(repo, "diff", "--name-only", "--no-renames", f"{base}..{head}")
    if rc:
        raise RuntimeError(f"git diff failed: {out[-300:]}")
    return [ln for ln in out.splitlines() if ln.strip()]


async def added_lines(repo: str | Path, base: str, head: str) -> list[tuple[str, str]]:
    """``(file, line)`` for every line *head* added on top of *base*."""
    rc, out = await git(repo, "diff", "--no-color", "--unified=0", "--no-renames", f"{base}..{head}")
    if rc:
        raise RuntimeError(f"git diff failed: {out[-300:]}")
    lines, current = [], ""
    for ln in out.splitlines():
        if ln.startswith("+++ "):
            current = ln[6:] if ln.startswith("+++ b/") else ""
        elif ln.startswith("+") and not ln.startswith("+++"):
            lines.append((current, ln[1:]))
    return lines


async def diff_stat(repo: str | Path, base: str, head: str) -> str:
    rc, out = await git(repo, "diff", "--shortstat", f"{base}..{head}")
    return out if rc == 0 else ""


# ── scope ────────────────────────────────────────────────────────────


def glob_to_regex(pattern: str) -> re.Pattern:
    """A path glob as a regex: ``**`` crosses folders, ``*`` and ``?`` don't;
    a pattern naming a folder (``tests`` or ``tests/``) covers what's in it."""
    p = pattern.strip().lstrip("./").rstrip("/")
    out, i = "", 0
    while i < len(p):
        if p.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif p.startswith("**", i):
            out += ".*"
            i += 2
        elif p[i] == "*":
            out += "[^/]*"
            i += 1
        elif p[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(p[i])
            i += 1
    return re.compile(f"^{out}(?:/.*)?$")


def matches(path: str, patterns: Iterable[str]) -> bool:
    return any(glob_to_regex(p).match(path) for p in patterns if p.strip())


def scope_problems(changed: list[str], editable: list[str], protected: list[str]) -> list[str]:
    """Why these changes are out of bounds, one line per file; empty when they
    are in bounds. No changes at all is a problem too: there's nothing to measure."""
    if not changed:
        return ["it changed nothing"]
    out = []
    for f in changed:
        if matches(f, protected):
            out.append(f"{f} is protected")
        elif editable and not matches(f, editable):
            out.append(f"{f} is outside what it may change")
    return out


# ── gaming scan ──────────────────────────────────────────────────────


def _literal_runs(pattern: str) -> list[str]:
    """The literal text inside a regex: what it insists on seeing."""
    text = re.sub(r"\{\d*,?\d*\}", "  ", pattern)
    text = re.sub(r"\\[bBdDsSwW]", " ", text)
    text = re.sub(r"\\(.)", r"\1", text)
    text = re.sub(r"[\^\$\(\)\[\]\{\}\|\*\+\?]", "  ", text)
    return [t.strip() for t in re.split(r"\s{2,}", text) if t.strip()]


def _tidy(s: str) -> str:
    return s.strip().rstrip(".,;:!?").strip()


def _keep(s: str) -> bool:
    """Is *s* distinctive enough that a fair change has no reason to carry it?
    A plain word, a short phrase, a bit of code shape ("return a") or a system
    path isn't; a name with a dot or digits, an address or a long phrase is."""
    s = _tidy(s)
    if not s or s.startswith("/") or s.lower() in _WORDS:
        return False
    if " " in s:
        return len(s) >= 10
    has_digit = any(c.isdigit() for c in s)
    if len(s) < (4 if has_digit else _MIN_STRING):
        return False
    return bool(has_digit or re.search(r"[._@+-]", s) or s != s.lower())


def case_strings(task_file: Path, label: str) -> dict[str, str]:
    """What a case file's cases contain that a fair change has no reason to
    mention: setup file names and contents, expected answers. Each
    string maps to where it came from (``dev:<id>`` / ``holdout:<id>``)."""
    try:
        data = json.loads(task_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, str] = {}
    for task in data.get("tasks") or []:
        where = f"{label}:{task.get('id', '?')}"
        # Not the id: the agent under test never sees it, and ids are everyday words
        # ("multi-step") an honest prompt may well use.
        found: list[str] = []
        setup = task.get("setup") or {}
        for name, content in (setup.get("files") or {}).items():
            found.append(str(name))
            found += re.findall(r"[A-Za-z0-9_.@+/-]+", str(content))
        for check in task.get("checks") or []:
            if check.get("path"):
                found.append(str(check["path"]))
            if check.get("pattern"):
                found += _literal_runs(str(check["pattern"]))
        for s in found:
            if _keep(s):
                out.setdefault(_tidy(s), where)
    return out


def gaming_hits(lines: list[tuple[str, str]], strings: dict[str, str],
                markers: Iterable[str] = EVAL_MARKERS) -> list[dict[str, str]]:
    """Every added line that carries a case's contents or points at the eval:
    ``[{"file", "string", "source", "line"}]``. The first hit per string is enough."""
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    needles = [(s, src) for s, src in strings.items()] + [(m, "eval") for m in markers]
    for file, line in lines:
        low = line.lower()
        for s, src in needles:
            if s in seen:
                continue
            if s.lower() in low:
                seen.add(s)
                hits.append({"file": file, "string": s, "source": src, "line": line.strip()[:200]})
    return hits
