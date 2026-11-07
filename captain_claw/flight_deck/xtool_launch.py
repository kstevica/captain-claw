"""Start one custom-tool program under limits (see ``external_tools.py``).

Run by Flight Deck as a script, never imported::

    python -E -s xtool_launch.py --mem <mb> --cpu <seconds> -- <program> [args...]

It sets the limits on itself and then becomes the program (``execv``), so the
program inherits them and Flight Deck's pipes. Standard library only: it runs
with an environment built from nothing, and must not import Captain.

- ``RLIMIT_AS``: address space, the nearest thing to a memory cap. macOS often
  refuses it; the tool then runs without one.
- ``RLIMIT_CPU``: CPU seconds; Flight Deck's own wall-clock timeout covers the rest.
- ``RLIMIT_CORE`` = 0: a crash leaves no core file with the caller's data in it.

Not ``RLIMIT_NPROC``: it counts every process of this OS user, Flight Deck's
included, so any useful value would stop the tool from starting.

Windows has none of these: there it runs the program and passes its exit code on.
"""

from __future__ import annotations

import os
import subprocess
import sys


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
    mem_mb, cpu_s = 0, 0
    args = list(argv)
    while args and args[0] != "--":
        flag = args.pop(0)
        if flag == "--mem" and args:
            mem_mb = int(args.pop(0))
        elif flag == "--cpu" and args:
            cpu_s = int(args.pop(0))
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
    if os.name == "nt":
        return subprocess.call(program)  # noqa: S603 — Flight Deck chose the program
    os.execv(program[0], program)  # noqa: S606
    return 127  # not reached


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
