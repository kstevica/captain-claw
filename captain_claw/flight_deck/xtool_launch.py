"""Start one custom-tool program under limits and, on Linux, in a sandbox
(see ``external_tools.py``).

Run by Flight Deck as a script, never imported::

    python -E -s xtool_launch.py --mem <mb> --cpu <seconds>
        [--ro <path>]... [--rw <path>]... [--no-net] [--require-sandbox]
        -- <program> [args...]
    python -E -s xtool_launch.py --probe     # prints what this host offers, as JSON

It sets the limits and the sandbox on itself and then becomes the program
(``execv``), so the program inherits both and Flight Deck's pipes. Standard
library only: it runs with an environment built from nothing, and must not
import Captain.

Limits:

- ``RLIMIT_AS``: address space, the nearest thing to a memory cap. macOS often
  refuses it; the tool then runs without one.
- ``RLIMIT_CPU``: CPU seconds; Flight Deck's own wall-clock timeout covers the rest.
- ``RLIMIT_CORE`` = 0: a crash leaves no core file with the caller's data in it.
- Not ``RLIMIT_NPROC``: it counts every process of this OS user, Flight Deck's
  included, so any useful value would stop the tool from starting.

The sandbox is Landlock (Linux 5.13+, no root needed; docs/agent-os-sandbox-plan.md):

- Files: read and run only what ``--ro`` names (the tool's version folder, the
  Python it runs on, the system's own folders), read and write only what
  ``--rw`` names (the call's folder) and a few devices. Everything else, other
  users' folders and Flight Deck's database included, is refused by the kernel.
  ``/proc/self`` is the tool's own: /proc of other processes (Flight Deck's
  environment) is out of reach.
- Network (Landlock ABI 4+): with ``--no-net``, no TCP connect or bind at all.
- Signals and abstract sockets (ABI 6+): the tool can't signal or reach
  processes outside its sandbox.

Without Landlock (macOS, Windows, an old or locked-down kernel) the tool runs
without a sandbox, unless ``--require-sandbox``: then it doesn't run (exit 126).

Windows has no limits either: there it runs the program and passes its exit code on.
"""

from __future__ import annotations

import json
import os
import platform
import stat
import subprocess
import sys

# ── Landlock (include/uapi/linux/landlock.h) ─────────────────────────────

_SYS_CREATE, _SYS_ADD_RULE, _SYS_RESTRICT = 444, 445, 446  # the same number on every architecture
_CREATE_VERSION = 1
_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

FS_EXECUTE = 1 << 0
FS_WRITE_FILE = 1 << 1
FS_READ_FILE = 1 << 2
FS_READ_DIR = 1 << 3
FS_REMOVE_DIR = 1 << 4
FS_REMOVE_FILE = 1 << 5
FS_MAKE_CHAR = 1 << 6
FS_MAKE_DIR = 1 << 7
FS_MAKE_REG = 1 << 8
FS_MAKE_SOCK = 1 << 9
FS_MAKE_FIFO = 1 << 10
FS_MAKE_BLOCK = 1 << 11
FS_MAKE_SYM = 1 << 12
FS_REFER = 1 << 13        # ABI 2
FS_TRUNCATE = 1 << 14     # ABI 3
FS_IOCTL_DEV = 1 << 15    # ABI 5
NET_BIND_TCP = 1 << 0     # ABI 4
NET_CONNECT_TCP = 1 << 1  # ABI 4
SCOPE_ABSTRACT_UNIX = 1 << 0  # ABI 6
SCOPE_SIGNAL = 1 << 1         # ABI 6

READ_ONLY = FS_EXECUTE | FS_READ_FILE | FS_READ_DIR
# What a rule on a file (not a folder) may grant.
FILE_RIGHTS = FS_EXECUTE | FS_WRITE_FILE | FS_READ_FILE | FS_TRUNCATE | FS_IOCTL_DEV
# Devices a program reasonably needs, read and write.
DEVICES = ("/dev/null", "/dev/zero", "/dev/urandom", "/dev/random", "/dev/full")


class SandboxError(Exception):
    pass


def _libc():
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    libc.prctl.restype = ctypes.c_int
    return ctypes, libc


def landlock_abi() -> int:
    """The Landlock ABI this kernel offers, 0 when none (or not Linux)."""
    if not sys.platform.startswith("linux"):
        return 0
    try:
        ctypes, libc = _libc()
        abi = libc.syscall(_SYS_CREATE, None, ctypes.c_size_t(0), ctypes.c_uint32(_CREATE_VERSION))
    except Exception:  # noqa: BLE001
        return 0
    return int(abi) if abi > 0 else 0


def probe() -> dict:
    abi = landlock_abi()
    return {"os": platform.system().lower(), "landlock": abi, "files": abi >= 1, "network": abi >= 4,
            "signals": abi >= 6}


def sandbox(ro: list[str], rw: list[str], *, no_net: bool) -> int:
    """Restrict this process (and what it becomes) to *ro* and *rw*. Returns the
    ABI used; raises :class:`SandboxError` when Landlock is there but fails."""
    abi = landlock_abi()
    if abi < 1:
        return 0
    ctypes, libc = _libc()

    class RulesetAttr(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64),
                    ("scoped", ctypes.c_uint64)]

    class PathBeneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    fs_all = (1 << 13) - 1
    if abi >= 2:
        fs_all |= FS_REFER
    if abi >= 3:
        fs_all |= FS_TRUNCATE
    if abi >= 5:
        fs_all |= FS_IOCTL_DEV
    net = (NET_BIND_TCP | NET_CONNECT_TCP) if (no_net and abi >= 4) else 0
    scoped = (SCOPE_ABSTRACT_UNIX | SCOPE_SIGNAL) if abi >= 6 else 0
    attr = RulesetAttr(fs_all, net, scoped)
    # Only as much of the struct as this kernel knows: a newer field it doesn't is refused.
    size = 24 if abi >= 6 else 16 if abi >= 4 else 8
    ruleset = libc.syscall(_SYS_CREATE, ctypes.byref(attr), ctypes.c_size_t(size), ctypes.c_uint32(0))
    if ruleset < 0:
        raise SandboxError(f"landlock_create_ruleset: {os.strerror(ctypes.get_errno())}")
    try:
        def allow(path: str, access: int) -> None:
            try:
                fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            except OSError:
                return  # not on this host: nothing to allow
            try:
                if not stat.S_ISDIR(os.fstat(fd).st_mode):
                    access &= FILE_RIGHTS
                rule = PathBeneath(access & fs_all, fd)
                if libc.syscall(_SYS_ADD_RULE, ctypes.c_int(ruleset), ctypes.c_int(_RULE_PATH_BENEATH),
                                ctypes.byref(rule), ctypes.c_uint32(0)) < 0:
                    raise SandboxError(f"landlock_add_rule {path}: {os.strerror(ctypes.get_errno())}")
            finally:
                os.close(fd)

        for path in ro:
            allow(path, READ_ONLY)
        for path in DEVICES:
            allow(path, FS_READ_FILE | FS_WRITE_FILE)
        for path in rw:
            allow(path, fs_all)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise SandboxError(f"prctl(NO_NEW_PRIVS): {os.strerror(ctypes.get_errno())}")
        if libc.syscall(_SYS_RESTRICT, ctypes.c_int(ruleset), ctypes.c_uint32(0)) < 0:
            raise SandboxError(f"landlock_restrict_self: {os.strerror(ctypes.get_errno())}")
    finally:
        os.close(ruleset)
    return abi


# ── Limits ───────────────────────────────────────────────────────────────


def _limit(name: str, value: int) -> None:
    try:
        import resource
    except ImportError:  # Windows
        return
    which = getattr(resource, name, None)
    if which is None:
        return
    try:
        _, hard = resource.getrlimit(which)
        soft = value if hard == resource.RLIM_INFINITY else min(value, hard)
        # Hard = soft too, so the program can't raise its own limit again.
        resource.setrlimit(which, (soft, soft))
    except (ValueError, OSError):
        pass  # refused (macOS and RLIMIT_AS): run without it


def main(argv: list[str]) -> int:
    if argv[:1] == ["--probe"]:
        print(json.dumps(probe()))
        return 0
    mem_mb, cpu_s = 0, 0
    ro: list[str] = []
    rw: list[str] = []
    no_net = require = False
    args = list(argv)
    while args and args[0] != "--":
        flag = args.pop(0)
        if flag == "--mem" and args:
            mem_mb = int(args.pop(0))
        elif flag == "--cpu" and args:
            cpu_s = int(args.pop(0))
        elif flag == "--ro" and args:
            ro.append(args.pop(0))
        elif flag == "--rw" and args:
            rw.append(args.pop(0))
        elif flag == "--no-net":
            no_net = True
        elif flag == "--require-sandbox":
            require = True
        else:
            print(f"xtool_launch: unknown argument {flag!r}", file=sys.stderr)
            return 2
    if not args or len(args) < 2:
        print("xtool_launch: nothing to run", file=sys.stderr)
        return 2
    program = args[1:]
    if mem_mb > 0:
        _limit("RLIMIT_AS", mem_mb * 1024 * 1024)
    if cpu_s > 0:
        _limit("RLIMIT_CPU", cpu_s)
    _limit("RLIMIT_CORE", 0)
    if ro or rw or require:
        try:
            used = sandbox(ro, rw, no_net=no_net)
        except SandboxError as exc:
            print(f"xtool_launch: the sandbox couldn't be set up ({exc}), so the tool didn't run.", file=sys.stderr)
            return 126
        if not used and require:
            print("xtool_launch: this deck runs custom tools only in a sandbox, and this host has none, so the "
                  "tool didn't run.", file=sys.stderr)
            return 126
    if os.name == "nt":
        return subprocess.call(program)  # noqa: S603 — Flight Deck chose the program
    os.execv(program[0], program)  # noqa: S606
    return 127  # not reached


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
