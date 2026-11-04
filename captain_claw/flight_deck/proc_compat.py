"""Process and file-lock calls that mean the same thing on Windows.

Flight Deck grew up on Unix. There ``os.kill(pid, 0)`` only asks whether a
process exists; on Windows the same call *terminates* it (every signal but the
two console CTRL events goes to TerminateProcess). ``fcntl``, ``SIGKILL`` and
``os.WNOHANG`` don't exist there at all. Each helper keeps the Unix behaviour
exactly as it was and gives Windows its own way to do the same.
"""

from __future__ import annotations

import os
import signal
import subprocess
from typing import IO, Any

IS_WINDOWS = os.name == "nt"

try:
    import fcntl as _fcntl
except ImportError:  # Windows
    _fcntl = None  # type: ignore[assignment]

# On Windows os.kill with SIGTERM already is TerminateProcess: nothing harder exists.
SIGKILL: int = getattr(signal, "SIGKILL", signal.SIGTERM)

# A console child of a windowless deck would otherwise open a console window of its own.
NO_WINDOW: int = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def lock_shared(f: IO[Any]) -> None:
    """Advisory shared lock on an open file; nothing where flock doesn't exist."""
    if _fcntl is None:
        return
    try:
        _fcntl.flock(f.fileno(), _fcntl.LOCK_SH)
    except OSError:
        pass  # unsupported on some file systems (NFS…): best effort


def lock_exclusive(f: IO[Any]) -> None:
    """Advisory exclusive lock on an open file; nothing where flock doesn't exist."""
    if _fcntl is None:
        return
    try:
        _fcntl.flock(f.fileno(), _fcntl.LOCK_EX)
    except OSError:
        pass


def pid_alive(pid: int | None) -> bool:
    """Whether a process of ours with this pid is running.

    Never sends anything on Windows: there it opens the process for a query
    and reads its exit code. One that exists but can't be opened (another
    user's) counts as not ours, as PermissionError does on Unix.
    """
    if not pid or pid <= 0:
        return False  # 0 / -1 would address a whole process group
    if IS_WINDOWS:
        return _win_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except OSError:  # ProcessLookupError, PermissionError
        return False
    return True


def _win_pid_alive(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def terminate(pid: int) -> None:
    """Ask a process to end: SIGTERM on Unix. Windows has no asking, so there the
    process and everything it started (an agent's browser, say) end at once."""
    if not pid or pid <= 0:
        return
    if IS_WINDOWS:
        try:
            subprocess.run(["taskkill", "/pid", str(pid), "/t", "/f"],  # noqa: S603, S607
                           capture_output=True, timeout=15, check=False,
                           stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
            return
        except (OSError, subprocess.SubprocessError):
            pass  # no taskkill: the process alone, below
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass


def kill(pid: int) -> None:
    """End a process now: SIGKILL on Unix, the process tree on Windows."""
    if not pid or pid <= 0:
        return
    if IS_WINDOWS:
        terminate(pid)
        return
    try:
        os.kill(pid, SIGKILL)
    except OSError:
        pass


def reaped(pid: int) -> bool:
    """Reap our own ended child (Unix: it stays a zombie, and "alive", until then).
    True once it is gone. Windows keeps no zombies, so there is nothing to reap."""
    if IS_WINDOWS:
        return False
    try:
        return bool(os.waitpid(pid, os.WNOHANG)[0])
    except ChildProcessError:
        return False  # not ours: an agent from before a deck restart
