"""Per-project git history for Code mode.

Each Code project is a VFS folder on disk (``<fd-data>/vfs/<user>/<project>/``).
We make that folder a git repo so every phase of a build — plan, build, each
fix round — lands as its own commit. That gives the UI real diffs ("what did
the Reviewer change?"), a phase timeline, and rollback to any prior phase.

All operations run ``git`` as a subprocess anchored at the project dir with an
explicit identity (``-c user.*``) so they never depend on the host's global git
config. Access to a single project's index is serialized by an asyncio lock so
concurrent phases/agents can't corrupt it mid-commit.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from pathlib import Path

from captain_claw.logging import get_logger

log = get_logger(__name__)

# One lock per project dir — git's index is not safe under concurrent writers.
_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

_AUTHOR = ("-c", "user.name=Captain Claw", "-c", "user.email=code@captain-claw.local")

# Things a coding agent's runtime litters into the workspace that should never
# be versioned (shell pre-creates ``saved/`` under the workspace root).
_GITIGNORE = """\
.code/
.captain-claw/
saved/
node_modules/
__pycache__/
*.pyc
.venv/
venv/
.env
.DS_Store
dist/
build/
.pytest_cache/
"""

# Runtime artifacts an agent litters into ANY workspace (incl. linked real
# repos). Written to .git/info/exclude — a LOCAL ignore that never modifies the
# repo's own tracked .gitignore, so linking a real project stays non-invasive.
_LOCAL_EXCLUDES = ["saved/", ".code/", ".captain-claw/", ".DS_Store"]


async def _ensure_excludes(project_dir: Path | str) -> None:
    info = Path(project_dir) / ".git" / "info"
    if not info.is_dir():
        return
    ex = info / "exclude"
    try:
        cur = ex.read_text() if ex.exists() else ""
        if "captain-claw runtime" in cur:
            return
        block = "\n# captain-claw runtime artifacts\n" + "\n".join(_LOCAL_EXCLUDES) + "\n"
        ex.write_text((cur.rstrip() + "\n" if cur.strip() else "") + block)
    except OSError:
        pass


def _lock(project_dir: Path | str) -> asyncio.Lock:
    return _locks[str(Path(project_dir).resolve())]


async def _git(project_dir: Path | str, *args: str, check: bool = False) -> tuple[int, str]:
    """Run ``git -C <project_dir> <args>``; return ``(returncode, combined_output)``."""
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", str(project_dir), *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    text = (out or b"").decode("utf-8", "replace")
    rc = proc.returncode or 0
    if check and rc != 0:
        raise RuntimeError(f"git {' '.join(args)} failed ({rc}): {text.strip()}")
    return rc, text


async def is_repo(project_dir: Path | str) -> bool:
    """Whether *project_dir* is its OWN git repo root.

    NOT merely "inside a work tree" — the VFS tree can live inside another repo
    (e.g. captain-claw's own checkout during local dev), and a plain
    ``--is-inside-work-tree`` would then resolve to that ancestor repo, leaking
    its history and committing project files into the wrong index. We require the
    top-level to be exactly this folder so every project stays isolated.
    """
    d = Path(project_dir).resolve()
    rc, out = await _git(d, "rev-parse", "--show-toplevel")
    if rc != 0:
        return False
    try:
        return Path(out.strip()).resolve() == d
    except OSError:
        return False


async def git_init(project_dir: Path | str) -> bool:
    """Initialise the project as its own git repo (idempotent). Returns True if newly created.

    Safe to nest inside another repo — ``git init`` creates a ``.git`` in this
    folder, and :func:`is_repo` then confirms this folder is the top-level.
    """
    d = Path(project_dir)
    d.mkdir(parents=True, exist_ok=True)
    async with _lock(d):
        if await is_repo(d):
            await _ensure_excludes(d)   # linked real repos: local excludes only
            return False
        await _git(d, "init", check=True)
        await _ensure_excludes(d)
        gi = d / ".gitignore"
        if not gi.exists():
            gi.write_text(_GITIGNORE)
        # Seed an empty root commit so diffs/rollback have a base even before
        # any agent has written code.
        await _git(d, "add", "-A")
        await _git(d, *_AUTHOR, "commit", "--allow-empty", "-m", "[init] code project")
        log.info("code_git: initialised repo", project=str(d))
        return True


async def git_commit(project_dir: Path | str, message: str) -> str | None:
    """Stage everything and commit. Returns the commit sha, or None if nothing changed."""
    d = Path(project_dir)
    if not await is_repo(d):
        await git_init(d)
    await _ensure_excludes(d)
    async with _lock(d):
        await _git(d, "add", "-A")
        rc, _ = await _git(d, "diff", "--cached", "--quiet")
        if rc == 0:
            return None  # nothing staged
        await _git(d, *_AUTHOR, "commit", "-m", message, check=True)
        _, sha = await _git(d, "rev-parse", "HEAD")
        return sha.strip()


async def git_log(project_dir: Path | str, limit: int = 50) -> list[dict]:
    """Return recent commits newest-first as ``[{sha, short, message, ts}]``."""
    d = Path(project_dir)
    if not await is_repo(d):
        return []
    fmt = "%H%x1f%h%x1f%s%x1f%ct"
    rc, out = await _git(d, "log", f"-{int(limit)}", f"--pretty=format:{fmt}")
    if rc != 0 or not out.strip():
        return []
    rows: list[dict] = []
    for line in out.splitlines():
        parts = line.split("\x1f")
        if len(parts) != 4:
            continue
        sha, short, msg, ts = parts
        rows.append({"sha": sha, "short": short, "message": msg,
                     "ts": int(ts) if ts.isdigit() else 0})
    return rows


async def git_diff(project_dir: Path | str, ref_a: str = "", ref_b: str = "") -> str:
    """Return a unified diff. No refs → working-tree vs HEAD; one ref → ref vs HEAD."""
    d = Path(project_dir)
    if not await is_repo(d):
        return ""
    if ref_a and ref_b:
        args = ["diff", ref_a, ref_b]
    elif ref_a:
        args = ["diff", ref_a]
    else:
        args = ["diff", "HEAD"]
    _, out = await _git(d, *args)
    return out


async def git_show(project_dir: Path | str, sha: str) -> str:
    """Return a single commit's patch (``git show <sha>``), no color."""
    d = Path(project_dir)
    if not await is_repo(d):
        return ""
    _, out = await _git(d, "show", "--no-color", "--stat", "--patch", sha)
    return out


async def git_reset(project_dir: Path | str, ref: str) -> None:
    """Hard-reset the working tree to ``ref`` (rollback)."""
    d = Path(project_dir)
    async with _lock(d):
        await _git(d, "reset", "--hard", ref, check=True)
        log.info("code_git: rolled back", project=str(d), ref=ref)
