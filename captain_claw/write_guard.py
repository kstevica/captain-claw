"""Deterministic guards for the `write` tool boundary.

Pure logic, no Flight Deck imports, so it is safe to import from the core tool
layer. Three jobs:

1. **Reject placeholder content.** After a successful write the agent runtime
   compacts the model's own history so the write's ``content`` argument reads as
   the acknowledgement/compaction marker ``[written to disk: … — use read tool
   to view]`` (``agent_session_mixin.py``). A weak model that later re-issues
   that write from its compacted history sends the marker AS the file body — the
   88-byte stub in the Captain Spark "story run 3" incident. Nothing downstream
   noticed because the write tool accepts any string. :func:`is_placeholder_content`
   catches the marker (and the shell-compaction and the tool's own result string)
   at the boundary so the write is refused loudly instead of persisting junk.

2. **Verify the write landed.** :func:`verify_readback` re-opens the file after
   ``fsync`` and compares the on-disk byte count (and, for small overwrites, the
   sha256) with what was sent, so a silently-truncated or dropped write becomes an
   error the caller must act on rather than a reported success.

3. **Repair a declared deliverable name.** In a Vatra run each subtask can be
   assigned an exact artifact filename; :func:`repair_declared_name` maps a
   model-shortened basename (``eppo-authenticity-pac``) back to the declared name
   (``eppo-authenticity-pack.md``) when the match is unambiguous, and
   :func:`requires_extension` flags an extensionless deliverable path.

Everything is gated by :func:`guard_config`, whose master kill-switch is the
``CLAW_WRITE_GUARD`` env var (unset / ``1`` = on).
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

# Anchored: matches only when the marker is what the content STARTS with, so a
# document that merely mentions "written to disk" in prose is not refused. Covers
# the write compaction marker (agent_session_mixin.py:997-1000), the shell
# compaction tail (_compact_shell_tool_call), the "use read tool to view" phrase
# on its own, and the write tool's own result string echoed back as content.
PLACEHOLDER_RE = re.compile(
    r"^[\s\[]*"
    r"(?:"
    r"written to disk"
    r"|use read tool to view"
    r"|\.\.\. shell command truncated"
    r"|Written \d+ chars \(\d+ lines\) to\b"
    # The in-turn context governor's pointers (context_spill): a spilled
    # result's preview line, and a de-duplicated page's notice.
    r"|[a-z_]+ .{1,600}? · \d+ chars · full text saved to\b"
    r"|already fetched in this run\b"
    r")",
    re.IGNORECASE,
)

# Files that are conventionally empty or extensionless — an empty body or a
# missing suffix is legitimate for these and must not be refused.
EMPTY_ALLOW = frozenset({
    "__init__.py", ".gitkeep", ".keep", "py.typed",
})
EXTENSIONLESS_ALLOW = frozenset({
    "Makefile", "Dockerfile", "LICENSE", "LICENCE", "README", "CHANGELOG",
    "Procfile", "CNAME", "NOTICE", "AUTHORS", "COPYING", "VERSION", "Rakefile",
    "Gemfile", "Vagrantfile", "Jenkinsfile", "CODEOWNERS",
})


def is_placeholder_content(text: str | None) -> bool:
    """True when *text* IS a tool acknowledgement/compaction marker, not file text."""
    if not text:
        return False
    # The first line: a governed pointer line names a URL that can run long.
    return bool(PLACEHOLDER_RE.match(text[:200]) or PLACEHOLDER_RE.match(text[:800].split("\n", 1)[0]))


def is_empty_content(content: str | None) -> bool:
    """True when *content* is empty or whitespace-only."""
    return not (content or "").strip()


def basename_of(path: str) -> str:
    """Last path segment of a plain or ``vfs:<project>/…`` path."""
    p = (path or "").strip()
    # Drop a vfs: scheme + project so we look only at the file part.
    if "/" in p:
        p = p.rsplit("/", 1)[-1]
    return p


def requires_extension(path: str) -> bool:
    """True when a vfs deliverable path lacks a usable extension.

    A dot-file (``.env``), an allow-listed extensionless name (``Makefile``), or
    any name with a real suffix is fine.
    """
    name = basename_of(path)
    if not name:
        return False
    if name in EXTENSIONLESS_ALLOW:
        return False
    if name.startswith("."):
        return False  # dot-file, e.g. .env / .gitignore
    return "." not in name


def repair_declared_name(basename: str, declared: list[str] | None) -> str | None:
    """Map a (possibly model-shortened) *basename* to a declared filename.

    Returns the unique declared name that *basename* is a prefix of (comparing
    both the full declared name and its stem), or ``None`` when there is no match
    or the match is ambiguous. An exact hit returns ``None`` — nothing to repair.
    """
    name = basename_of(basename)
    if not name or not declared:
        return None
    decl = [basename_of(d) for d in declared if (d or "").strip()]
    if name in decl:
        return None  # already correct
    cands = []
    for d in decl:
        stem = d.rsplit(".", 1)[0] if "." in d else d
        if d.startswith(name) or stem.startswith(name) or stem == name:
            cands.append(d)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq = [c for c in cands if not (c in seen or seen.add(c))]
    return uniq[0] if len(uniq) == 1 else None


_HASH_MAX_BYTES = 4 * 1024 * 1024  # sha256 only for files this small

# ── Line endings ──────────────────────────────────────────────────────────
# A file keeps its own line endings, on every OS. Python's text mode writes
# os.linesep for every "\n" — on Windows a 35-line file sent as 1648 bytes lands
# as 1683 — and reads CRLF back as LF, so `edit` used to turn a Windows file into
# an LF one on Linux and an LF file into CRLF on Windows. Writers here open
# with newline="" (bytes as sent) and pick the ending by these rules:
#   1. an existing file keeps the ending most of its lines already have;
#   2. a new .bat/.cmd gets CRLF (cmd.exe misreads labels and GOTO in LF files);
#   3. anything else is written exactly as the model sent it.
CRLF_SUFFIXES = frozenset({".bat", ".cmd"})
_EOL_SNIFF_BYTES = 64 * 1024


def eol_of(data: bytes | str) -> str | None:
    """``"\\r\\n"`` when most of *data*'s line breaks are CRLF, ``"\\n"`` when
    they are LF, ``None`` when it has none."""
    lf, crlf = ("\n", "\r\n") if isinstance(data, str) else (b"\n", b"\r\n")
    n = data.count(lf)
    if not n:
        return None
    return "\r\n" if data.count(crlf) * 2 > n else "\n"


def file_eol(path: str | Path) -> str | None:
    """The line ending an existing file uses (its first 64 KB), else ``None``."""
    try:
        with open(path, "rb") as f:
            return eol_of(f.read(_EOL_SNIFF_BYTES))
    except OSError:
        return None


def to_eol(text: str, eol: str | None) -> str:
    """*text* with every line break made *eol*; unchanged when *eol* is None."""
    if not eol or not text:
        return text
    flat = text.replace("\r\n", "\n")
    return flat if eol == "\n" else flat.replace("\n", "\r\n")


def eol_for_write(path: str | Path, *, exists: bool) -> str | None:
    """The line ending a write to *path* must use (rules above), or ``None`` to
    keep the content exactly as sent."""
    if exists:
        return file_eol(path)
    return "\r\n" if Path(path).suffix.lower() in CRLF_SUFFIXES else None


def _sha8(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:8]


def _flat(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def verify_readback(
    file_path: str | Path,
    content: str,
    append: bool = False,
    prev_size: int = 0,
) -> dict:
    """Re-read *file_path* and confirm it holds what was written.

    Returns ``{ok, bytes, sha8, reason, note}``. ``bytes`` is the observed
    on-disk size. For an overwrite the whole file is compared (length always;
    sha256 when ≤ 4 MB). For an append only the length is checked against
    ``prev_size + len(content)`` — the prior bytes are not re-hashed.

    A file whose only difference is its line endings (a writer that went through
    text mode on Windows) holds everything that was sent: that is ``ok`` with a
    ``note``, never a failure — calling it truncated made the model re-issue the
    same write, which landed the same way again.
    """
    fp = Path(file_path)
    want = content.encode("utf-8", errors="replace")
    expected_new = len(want)
    try:
        observed = fp.stat().st_size
    except Exception as e:  # noqa: BLE001 — a missing file after write IS the failure
        return {"ok": False, "bytes": 0, "sha8": "", "reason": f"stat_failed: {e}", "note": ""}

    if append:
        expected_total = prev_size + expected_new
        if observed == expected_total:
            return {"ok": True, "bytes": observed, "sha8": "", "reason": "", "note": ""}
        # Text mode adds one byte per "\n" that was not already "\r\n".
        crlf_total = prev_size + expected_new + content.count("\n") - content.count("\r\n")
        if observed == crlf_total:
            return {"ok": True, "bytes": observed, "sha8": "", "reason": "",
                    "note": "line endings became CRLF on disk"}
        return {"ok": False, "bytes": observed, "sha8": "", "note": "",
                "reason": f"append_size_mismatch: on_disk={observed} expected={expected_total}"}

    if observed != expected_new:
        if expected_new <= _HASH_MAX_BYTES and observed <= 2 * expected_new + 2:
            try:
                disk = fp.read_bytes()
            except Exception:  # noqa: BLE001
                disk = b""
            if disk and _flat(disk) == _flat(want):
                return {"ok": True, "bytes": observed, "sha8": _sha8(disk), "reason": "",
                        "note": "only the line endings differ from what was sent"}
        return {"ok": False, "bytes": observed, "sha8": "", "note": "",
                "reason": f"size_mismatch: on_disk={observed} expected={expected_new}"}
    # Length matches — for small files also confirm the bytes are identical, so a
    # same-length corruption (control chars, encoding surprise) is caught too.
    if expected_new <= _HASH_MAX_BYTES:
        try:
            disk = fp.read_bytes()
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "bytes": observed, "sha8": "", "reason": f"read_failed: {e}",
                    "note": ""}
        if disk != want:
            return {"ok": False, "bytes": observed, "sha8": _sha8(disk),
                    "reason": "content_mismatch", "note": ""}
        return {"ok": True, "bytes": observed, "sha8": _sha8(disk), "reason": "", "note": ""}
    return {"ok": True, "bytes": observed, "sha8": "", "reason": "", "note": ""}


@dataclass(frozen=True)
class GuardConfig:
    reject_placeholder: bool
    verify_readback: bool
    verify_retries: int
    require_extension_vfs: bool


_TRUE = {"1", "true", "yes", "on", ""}
_FALSE = {"0", "false", "no", "off"}


def _env_kill() -> bool:
    """True when CLAW_WRITE_GUARD explicitly disables every guard."""
    return os.environ.get("CLAW_WRITE_GUARD", "").strip().lower() in _FALSE


def guard_config() -> GuardConfig:
    """Resolve the effective guard config: env kill-switch AND per-feature config.

    Defensive: if the app config cannot be read (bare tool tests) the safe
    defaults (all guards on, 2 retries) apply. ``CLAW_WRITE_GUARD`` set to a
    falsy value turns every guard off.
    """
    if _env_kill():
        return GuardConfig(False, False, 0, False)
    reject = verify = req_ext = True
    retries = 2
    try:
        from captain_claw.config import get_config
        w = get_config().tools.write
        reject = bool(getattr(w, "reject_placeholder", True))
        verify = bool(getattr(w, "verify_readback", True))
        retries = int(getattr(w, "verify_retries", 2))
        req_ext = bool(getattr(w, "require_extension_vfs", True))
    except Exception:  # noqa: BLE001 — config unavailable → safe defaults
        pass
    return GuardConfig(reject, verify, max(0, retries), req_ext)


def strict_worker() -> bool:
    """True when this process is a Vatra worker under the strict deliverable tier."""
    return os.environ.get("CLAW_WRITE_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}


def declared_files() -> list[str]:
    """Deliverable basenames the worker was told to produce (CLAW_DECLARED_FILES JSON)."""
    import json
    raw = os.environ.get("CLAW_DECLARED_FILES", "").strip()
    if not raw:
        return []
    try:
        val = json.loads(raw)
        return [str(x) for x in val] if isinstance(val, list) else []
    except Exception:  # noqa: BLE001
        return []


def min_bytes_floor() -> int:
    """Byte floor for a declared deliverable part (CLAW_WRITE_MIN_BYTES), 0 = off."""
    try:
        return max(0, int(os.environ.get("CLAW_WRITE_MIN_BYTES", "0") or 0))
    except Exception:  # noqa: BLE001
        return 0
