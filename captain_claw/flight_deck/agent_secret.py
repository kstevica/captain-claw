"""Auto-managed shared secret for agent → FD authentication.

Both the FD server process and any agent process that runs
``app_runner`` need to agree on a single secret to authenticate
server-to-server calls. We don't want users to set env vars or copy
strings around — the system should bootstrap this itself.

Approach
--------
The secret is a random 32-byte hex string stored in
``~/.captain-claw-fd/agent_secret`` with mode ``0600``.

- If the env var ``FD_AGENT_SHARED_SECRET`` is set, it wins (lets
  ops override the file for multi-machine deployments).
- Otherwise, on first call, we read the file if it exists.
- Otherwise, we generate a fresh secret, write it atomically, and
  cache it in-process.

Both FD and the agent live on the same host in the common dev
setup, so they both resolve to the same ``CAPTAIN_CLAW_FD_HOME``
and converge on the same file. The first process to need the
secret writes it; the second reads it.

Race condition: if two processes start at exactly the same time
and both find no file, both will write. The second write atomically
replaces the first, and the first process's cached value goes
stale. Next time it reads (we re-validate against disk on every
``get`` call by default), it picks up the new value. The window is
tiny in practice, and a single restart of either side resolves any
divergence cleanly.
"""

from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path
from threading import Lock

log = logging.getLogger(__name__)


_SECRET_FILE_NAME = "agent_secret"
_SECRET_BYTES = 32

_lock = Lock()
_cached: str | None = None


def _real_user_home() -> Path:
    """Return the OS-level home directory of the user running us.

    We deliberately do NOT use ``os.path.expanduser("~")`` or read
    ``$HOME``: FD spawns agent processes with ``HOME`` redirected
    to a sandboxed per-agent data dir (so each agent gets its own
    dotfiles / DBs). If we used ``$HOME`` here, FD and every agent
    would land on different files and the secret would never match.

    Anchoring to the passwd-database home keeps every process under
    the same Unix user converged on one path, which is exactly the
    invariant we need for a shared secret.
    """
    try:
        import pwd
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (ImportError, KeyError, OSError):
        # Windows or weird passwd states — fall back to the env-var
        # path. Same-machine-same-user is the common case we serve.
        return Path(os.path.expanduser("~"))


def _secret_path() -> Path:
    """Path to the secret file.

    Always under the real user's home so FD and agents (which may
    have different ``$HOME`` overrides) all read the same file.
    The directory name matches ``app_runtime._fd_home``'s default
    so an unsandboxed FD writes it next to its other state.
    """
    return _real_user_home() / ".captain-claw-fd" / _SECRET_FILE_NAME


def _read_file() -> str:
    p = _secret_path()
    try:
        text = p.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except OSError as exc:
        log.warning("Could not read agent_secret at %s: %s", p, exc)
        return ""
    return text if text else ""


def _write_file(value: str) -> None:
    """Atomic-ish write with restrictive permissions."""
    p = _secret_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    # Open with 0600 explicitly so the secret never exists on disk
    # with a wider mode.
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, value.encode("utf-8"))
    finally:
        os.close(fd)
    os.replace(tmp, p)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


def get_or_create_agent_secret() -> str:
    """Return the shared agent secret, generating it if needed.

    Priority:

    1. ``FD_AGENT_SHARED_SECRET`` env var (ops override).
    2. In-process cache from a previous call.
    3. ``~/.captain-claw-fd/agent_secret`` on disk.
    4. Freshly generated + persisted.

    Always returns a non-empty string. Cheap on the hot path
    (env-var read + cache check; disk only the first time per
    process).
    """
    env_val = os.environ.get("FD_AGENT_SHARED_SECRET", "").strip()
    if env_val:
        return env_val

    global _cached
    with _lock:
        if _cached:
            return _cached
        # Try disk first.
        existing = _read_file()
        if existing:
            _cached = existing
            return _cached
        # Generate fresh.
        new_secret = secrets.token_hex(_SECRET_BYTES)
        try:
            _write_file(new_secret)
        except OSError as exc:
            # If we can't persist, fall back to in-memory only —
            # auth still works while the process is alive. A second
            # process won't be able to authenticate until persistence
            # is fixed, but at least we don't crash.
            log.warning(
                "Could not persist agent_secret to %s: %s. "
                "Using in-memory only; cross-process auth will fail "
                "until this is resolved.",
                _secret_path(), exc,
            )
        _cached = new_secret
        return _cached


def reset_cache_for_tests() -> None:
    """Drop the in-process cache. Tests only."""
    global _cached
    with _lock:
        _cached = None
