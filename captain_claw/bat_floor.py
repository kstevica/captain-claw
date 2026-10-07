"""Bat hard floor (Invariant A) — the un-widenable system-harm denylist.

Bat is the "stubborn finisher" mode: inside a Bat run, workers are allowed to be
aggressive — browser, computer use, logins, long loops — and almost every soft
guard (iteration caps, duplicate-call blocks, deliverable-strictness) may be
relaxed per run. This module is the ONE thing that can never be relaxed: a
deterministic, token-free deny of operations that would irreversibly harm the
host or its drives.

Design contract:
  * Pure and deterministic. No LLM, no I/O, no config lookup. ``screen()`` always
    classifies; ``active()`` only decides whether the caller enforces it.
  * UN-WIDENABLE. There is no config knob and no Bat relax flag that reaches this.
    A new Bat capability is added by relaxing a *soft* guard elsewhere, never by
    editing this deny set downward.
  * Enforced as a HARD refusal at the universal tool chokepoint
    (``ToolRegistry.execute``), so it covers every exec-capable tool — ``shell``,
    ``terminal`` (the user's real Mac over the PTY bridge), ``desktop_action`` —
    not just ``shell``.
  * Narrow on purpose. The floor is ONLY system/drive destruction ("rm, umount
    drives and similar", in the owner's words). It is NOT the money / outward
    gate (that is Bat's spend pre-authorisation) and NOT the opt-in blast-radius
    approval guard (``flight_deck/blast_radius.py``). Ordinary destructive dev
    work — ``rm -rf build``, ``rm -rf node_modules``, dropping a dev table — is
    deliberately allowed; a stubborn builder needs it. Only catastrophic targets
    (root, home, system dirs, whole drives, raw devices) are refused.
  * Bias on the narrow set: when a target is ambiguous between "a subpath of a
    project" and "a system/home/drive root", refuse. A false refusal costs the
    Bat run one human-ask; a false allow can wipe the machine.

This module imports nothing from ``captain_claw`` so the core tool registry can
import it without a dependency cycle (same posture as ``write_guard``).
"""

from __future__ import annotations

import os
import re
import shlex
from typing import Any

#: Env marker a spawned Bat worker carries. The floor enforces only when this is
#: set, so existing (non-Bat) agents are unaffected. NOTE: this marker is
#: deliberately NOT added to the Basna/Vatra recursion guard
#: (``tools/basna.py``), so a Bat worker may start a Vatra/Basna sub-run — Bat is
#: meant to reach every tool, including the orchestration launchers.
MARKER = "CLAW_BAT_WORKER"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Tools that actually execute commands against the OS. The floor screens these;
#: for anything else (write/edit/read/browser/http/...) it is a no-op, so a file
#: whose *content* mentions ``rm -rf /`` is never mistaken for an execution.
EXEC_TOOLS: frozenset[str] = frozenset({
    "shell", "bash", "sh", "zsh", "run_shell", "exec", "command",
    "terminal", "desktop_action",
})

#: Fields on an exec tool call that may carry a command / typed keystrokes. We
#: scan these only for EXEC_TOOLS, never for content-bearing tools.
_EXEC_FIELDS: tuple[str, ...] = (
    "command", "cmd", "script", "statement", "input", "data", "text",
    "keys", "args", "code",
)

# Shell wrappers to skip when finding a segment's real base command.
_WRAPPERS: frozenset[str] = frozenset({
    "sudo", "doas", "env", "nice", "nohup", "time", "command", "builtin",
    "exec", "xargs", "stdbuf", "setsid", "ionice",
})

# Base commands whose non-flag targets we path-check for catastrophic roots.
_DELETE_CMDS: frozenset[str] = frozenset({"rm", "rmdir", "shred", "unlink"})
_PERM_CMDS: frozenset[str] = frozenset({"chmod", "chown", "chgrp"})

# First path component of an absolute target that is never legitimate to delete
# or chmod recursively, at ANY depth.
_SYSTEM_DIRS: frozenset[str] = frozenset({
    "/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64", "/var", "/boot",
    "/dev", "/private", "/System", "/Library", "/opt", "/proc", "/sys",
    "/root", "/cores",
})
# First component that is dangerous only near the top (a whole user home, the
# users root, or a mounted drive root) — depth > 2 under these is allowed.
_SHALLOW_DIRS: frozenset[str] = frozenset({
    "/Users", "/home", "/Volumes", "/mnt", "/media", "/srv",
})
# Bare tokens that mean "everything here / the root / my home".
_BARE_DANGER: frozenset[str] = frozenset({
    "/", "/*", "*", ".", "..", "./", "../", "~", "~/", "~/*",
    "$HOME", "${HOME}", "$HOME/", "${HOME}/", "$HOME/*", "${HOME}/*",
})

# Always-block command signatures (whole-command scan, case-insensitive). These
# are operations with no legitimate place in a task-completion run.
_ALWAYS_BLOCK: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(rx, re.IGNORECASE), reason) for rx, reason in (
        # filesystem create / format
        (r"\bmkfs(\.\w+)?\b", "filesystem format (mkfs)"),
        (r"\bnewfs(_\w+)?\b", "filesystem format (newfs)"),
        (r"\bmke2fs\b", "filesystem format (mke2fs)"),
        (r"\bmkswap\b", "swap format (mkswap)"),
        (r"\bwipefs\b", "filesystem signature wipe (wipefs)"),
        (r"\bblkdiscard\b", "block device discard (blkdiscard)"),
        # partition tables / disk tooling
        (r"\b(fdisk|gdisk|sgdisk|cfdisk|parted)\b", "partition table edit"),
        # macOS diskutil destructive verbs
        (r"\bdiskutil\s+(erase\w*|reformat|partitiondisk|zerodisk|"
         r"securedisk|secureerase|apfs\s+delete\w*)\b", "diskutil disk erase/reformat"),
        # unmount / eject a drive
        (r"\bdiskutil\s+(unmount\w*|eject)\b", "diskutil unmount/eject"),
        (r"\bumount\b", "unmount a filesystem (umount)"),
        # raw device write
        (r"\bdd\b[^\n;|&]*\bof=\s*/dev/", "raw device write (dd of=/dev/...)"),
        (r">\s*/dev/(disk|r?disk|sd[a-z]|nvme\d|hd[a-z])", "redirect onto a raw device"),
        # power control
        (r"\b(shutdown|reboot|halt|poweroff)\b", "host power control"),
        (r"\b(init|telinit)\s+[06]\b", "runlevel power control"),
        (r"\bsystemctl\s+(poweroff|reboot|halt|kexec)\b", "systemctl power control"),
        (r"\blaunchctl\s+reboot\b", "launchctl reboot"),
        # broadcast kill (every process the user owns)
        (r"\bkill\s+-(9|KILL)?\s*-1\b", "broadcast kill (kill -1)"),
        (r"\bkill\s+-1\b", "broadcast kill (kill -1)"),
        # fork bomb (with or without whitespace)
        (r":\s*\(\s*\)\s*\{.*\|.*&.*\}", "fork bomb"),
        (r":\(\)\{:\|:&\};:", "fork bomb"),
        # recursive find-delete at a catastrophic root
        (r"\bfind\s+(/|~|\$HOME|\$\{HOME\}|/etc|/usr|/bin|/var|/System|/Library|"
         r"/Users|/home)[^\n]*-delete\b", "find -delete at a system/home root"),
        (r"\bfind\s+(/|~|\$HOME|/etc|/usr|/bin|/var|/System|/Library|/Users|/home)"
         r"[^\n]*-exec\s+rm\b", "find -exec rm at a system/home root"),
        # inline-interpreter equivalents of a root wipe
        (r"rmtree\(\s*['\"]/['\"]", "shutil.rmtree('/')"),
        (r"rmtree\(\s*os\.path\.expanduser", "rmtree of the home directory"),
        (r"removedirs\(\s*['\"]/", "os.removedirs at root"),
        (r"\b(python3?|perl|ruby|node|php)\b[^\n]{0,80}\brm\s+-[a-z]*[rf][a-z]*\s+/(\s|$)",
         "inline interpreter running rm -rf /"),
    )
)


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in _TRUTHY


_ACTIVE: bool | None = None


def active() -> bool:
    """True when the current process is a Bat worker and must enforce the floor.

    Cached: the marker is set once at spawn and never changes within a process.
    """
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = _env_truthy(MARKER)
    return _ACTIVE


def _norm(tok: str) -> str:
    return tok.strip().strip('"').strip("'")


def _catastrophic_target(tok: str) -> bool:
    """True if *tok* names the root, a home root, a system dir (any depth), or a
    whole drive/user-home near the top. Ordinary relative/deep paths are safe."""
    t = _norm(tok)
    if not t:
        return False
    if t in _BARE_DANGER:
        return True
    # Normalise a trailing "/*" or "/" for the structural checks below.
    core = t
    if core.endswith("/*"):
        core = core[:-2]
    core = core.rstrip("/")
    if core in ("", "~", "$HOME", "${HOME}", ".", ".."):
        return True
    # A home SUBPATH (~/proj, $HOME/x) is fine; only a bare home root is caught
    # above.
    if core.startswith(("~/", "$HOME/", "${HOME}/", "~", "$HOME", "${HOME}")) and \
            not core.startswith("/"):
        return False
    if core.startswith("/"):
        parts = [p for p in core.split("/") if p]
        if not parts:
            return True  # "/"
        first = "/" + parts[0]
        if first in _SYSTEM_DIRS:
            return True  # never recurse-delete inside a system dir, any depth
        if first in _SHALLOW_DIRS:
            # "/Users", "/Users/alice", "/Volumes/Data" → whole home / drive.
            # Deeper ("/Users/alice/Dev/x") is a project path → allowed.
            return len(parts) <= 2
    return False


def _segments(command: str) -> list[list[str]]:
    """Split a command line into operator-separated segments, each tokenised.

    Best-effort: shlex where possible, naive split otherwise. The always-block
    regexes run on the raw text too, so a tokenisation miss cannot by itself
    open a hole for the format/device/power classes."""
    raw = re.split(r"\|\||&&|[;\n|&]", command or "")
    out: list[list[str]] = []
    for seg in raw:
        seg = seg.strip()
        if not seg:
            continue
        try:
            toks = shlex.split(seg, comments=False, posix=True)
        except ValueError:
            toks = seg.split()
        if toks:
            out.append(toks)
    return out


def _base_and_targets(tokens: list[str]) -> tuple[str, list[str]]:
    """Return (base command, non-flag argument tokens) skipping env-assignments
    and wrappers (sudo/env/nice/...)."""
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if "=" in t and not t.startswith("-") and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t):
            i += 1  # VAR=value prefix
            continue
        if t in _WRAPPERS or t.startswith("/") and os.path.basename(t) in _WRAPPERS:
            i += 1
            continue
        break
    if i >= len(tokens):
        return "", []
    base = os.path.basename(tokens[i])
    targets = [t for t in tokens[i + 1:] if not t.startswith("-")]
    return base, targets


def _scan_command(text: str) -> tuple[bool, str]:
    """Classify one command-bearing blob. Returns (blocked, reason)."""
    t = text or ""
    if not t.strip():
        return False, ""
    for rx, reason in _ALWAYS_BLOCK:
        if rx.search(t):
            return True, reason
    for tokens in _segments(t):
        base, targets = _base_and_targets(tokens)
        if base in _DELETE_CMDS:
            for tgt in targets:
                if _catastrophic_target(tgt):
                    return True, f"recursive delete of a protected path ({_norm(tgt)})"
        elif base in _PERM_CMDS:
            # Only recursive perm changes on a protected root are catastrophic.
            recursive = any(
                f == "-R" or f == "--recursive" or (f.startswith("-") and "R" in f)
                for f in tokens
            )
            if recursive:
                for tgt in targets:
                    if _catastrophic_target(tgt):
                        return True, f"recursive permission change on a protected path ({_norm(tgt)})"
    return False, ""


def _blobs(tool_name: str, arguments: dict[str, Any] | None) -> list[str]:
    args = arguments if isinstance(arguments, dict) else {}
    blobs: list[str] = []
    for key in _EXEC_FIELDS:
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            blobs.append(v)
        elif isinstance(v, (list, tuple)):
            blobs.append(" ".join(str(x) for x in v))
    return blobs


def screen(tool_name: str, arguments: dict[str, Any] | None) -> tuple[bool, str]:
    """Classify a concrete exec-tool call. Returns (blocked, reason).

    Pure — safe to call anywhere and in tests. For non-exec tools, always
    (False, "")."""
    name = (tool_name or "").strip().lower()
    if name not in EXEC_TOOLS:
        return False, ""
    for blob in _blobs(name, arguments):
        blocked, reason = _scan_command(blob)
        if blocked:
            return True, reason
    return False, ""


def refusal(tool_name: str, reason: str) -> str:
    """Human-readable refusal for the ToolBlockedError raised at the chokepoint."""
    return (
        f"Bat hard floor: refused `{tool_name}` — {reason}. "
        "System/drive destruction is the one thing a Bat run may never do; "
        "everything else is negotiable. If this is genuinely required, ask the "
        "human to run it."
    )
