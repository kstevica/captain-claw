"""Custom tools ("external tools"): small Python programs Crew writes and agents
call through Flight Deck. Plan: docs/external-tools-plan.md.

Flight Deck keeps them, runs them and (from Phase 3) shares them; an agent never
runs one itself. One call is one process: the input goes in as JSON on stdin, one
JSON value comes back on stdout.

Layout under ``<DATA_DIR>/external_tools/``::

    <owner>/<slug>/repo/           git repository; main = what the owner approved
    <owner>/<slug>/versions/<v>/   clean copy of commit <v> + its venv (.deps),
                                   read-only once built
    <owner>/<slug>/live.json       which version runs, and the ones kept before it
    <owner>/<slug>/publish.json    the last publish: state, steps, log tails
    .runs/<caller>/<call_id>/      one call's in/ out/ home/, removed after it
    .cache/                        uv / pip download caches shared by publishes

Nothing a tool can write reaches another call: its working folder is the call's
own, the version folder is made read-only, and the environment is built from
nothing (no Flight Deck variable, no agent secret). Until the OS sandbox exists
(docs/agent-os-sandbox-plan.md) that read-only folder is a guard against
accidents, not a wall: the process still runs as Flight Deck's OS user.
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import re
import secrets as _secrets
import shutil
import signal
import stat
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from captain_claw.flight_deck.proc_compat import IS_WINDOWS, NO_WINDOW
from captain_claw.logging import get_logger

log = get_logger(__name__)

# ── Limits ───────────────────────────────────────────────────────────────

DEFAULT_TIMEOUT_S = 60
MAX_TIMEOUT_S = 300
DEFAULT_MEMORY_MB = 1024
MIN_MEMORY_MB = 128
MAX_MEMORY_MB = 4096
MAX_STDOUT = 1_000_000            # bytes of JSON a call may print
STDERR_TAIL = 8_192               # bytes of stderr kept for the caller
MAX_IN_BYTES = 200 * 1024 * 1024  # files given to one call
MAX_OUT_FILES = 50
MAX_OUT_BYTES = 100 * 1024 * 1024
MAX_EXAMPLES = 10
MAX_SECRETS = 10
KEEP_PREVIOUS = 3                 # versions kept for "Make live" besides the live one
INSTALL_TIMEOUT_S = 600
TEST_TIMEOUT_S = 300
PER_TOOL_PARALLEL = 2             # calls at once per person per tool
MAX_PARALLEL = 8                  # calls at once on this deck
SLOT_WAIT_S = 30
MANIFEST = "tool.json"
READY = ".xtool-ready"            # written into a version folder once it's built
LOG_TAIL = 4_000
HELPER_NAME = "captain.py"        # Flight Deck's helper, put into every version (Phase 7)
MAX_RECORDED = 50                 # recorded replies per example
RECORDED_KINDS = ("tool", "custom_tool", "file", "ask")
MAX_LINE = 4 * 1024 * 1024        # one request or answer on a tool's line to the agent
MAX_ANSWER = 1_000_000            # characters of an agent tool's answer a tool gets

_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,46}[a-z0-9])?$")
_OWNER_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_VERSION_RE = re.compile(r"^[0-9a-f]{12}$")
_SECRET_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
# A secret can't stand in for what the tool's environment is made of.
_RESERVED_ENV = frozenset({"PATH", "HOME", "TMPDIR", "TEMP", "TMP", "USERPROFILE", "LANG", "LC_ALL", "TZ",
                           "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "VIRTUAL_ENV", "SSL_CERT_FILE",
                           "SSL_CERT_DIR"})
_RESERVED_PREFIXES = ("XT_", "PYTHON", "LD_", "DYLD_", "FD_", "CLAW_", "CAPTAIN_CLAW", "UV_", "PIP_")
_PROXY_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")
_LAUNCHER = Path(__file__).with_name("xtool_launch.py")
_HELPER = Path(__file__).with_name("xtool_captain.py")


class ToolError(Exception):
    """Something the caller has to change (a 4xx), as opposed to a tool that ran
    and failed, which :func:`run_tool` returns as a result."""

    def __init__(self, kind: str, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status = status


# ── Where things are ─────────────────────────────────────────────────────


def root() -> Path:
    from captain_claw.flight_deck.server import DATA_DIR

    return Path(DATA_DIR) / "external_tools"


# Names a route already uses under /fd/external-tools/ (or will): a tool can't take them.
RESERVED_SLUGS = frozenset({"agent", "status", "admin", "catalog", "shared", "proposals", "approvals", "build",
                            "added", "review", "agents", "asks"})


def check_slug(slug: str) -> str:
    slug = str(slug or "").strip()
    if not _SLUG_RE.match(slug):
        raise ToolError("bad_name", "A tool's name is 1–48 lowercase letters, digits and dashes, "
                                    "starting and ending with a letter or digit (like invoice-parse).")
    return slug


def _check_owner(owner: str) -> str:
    owner = str(owner or "")
    if not _OWNER_RE.match(owner):
        raise ToolError("bad_owner", "Unknown owner.", status=404)
    return owner


def tool_dir(owner: str, slug: str) -> Path:
    return root() / _check_owner(owner) / check_slug(slug)


def repo_dir(owner: str, slug: str) -> Path:
    return tool_dir(owner, slug) / "repo"


def versions_dir(owner: str, slug: str) -> Path:
    return tool_dir(owner, slug) / "versions"


def _live_file(owner: str, slug: str) -> Path:
    return tool_dir(owner, slug) / "live.json"


def _publish_file(owner: str, slug: str) -> Path:
    return tool_dir(owner, slug) / "publish.json"


def venv_python(deps: Path) -> Path:
    return deps / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def _venv_bin(deps: Path) -> Path:
    return deps / ("Scripts" if IS_WINDOWS else "bin")


def exists(owner: str, slug: str) -> bool:
    return (repo_dir(owner, slug) / ".git").exists()


def _require(owner: str, slug: str) -> None:
    if not exists(owner, slug):
        raise ToolError("not_found", f"There's no tool called {slug}.", status=404)


# ── Small file helpers ───────────────────────────────────────────────────


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data: Any) -> None:
    """Written aside and renamed in, so a reader never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{_secrets.token_hex(3)}")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _make_readonly(folder: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(folder):
        for name in filenames:
            p = Path(dirpath) / name
            try:
                if not p.is_symlink():
                    p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
            except OSError:
                pass
    # Folders last, or their files couldn't be reached to change.
    for dirpath, dirnames, _ in os.walk(folder, topdown=False):
        for name in dirnames:
            p = Path(dirpath) / name
            try:
                if not p.is_symlink():
                    p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
            except OSError:
                pass
    try:
        folder.chmod(folder.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    except OSError:
        pass


def _rmtree(path: Path) -> None:
    """Remove a folder, read-only ones too."""
    if not path.exists() and not path.is_symlink():
        return

    def again(func, p, _exc):  # noqa: ANN001
        try:
            os.chmod(os.path.dirname(p), 0o700)
            os.chmod(p, 0o700)
            func(p)
        except OSError:
            pass

    if path.is_symlink() or path.is_file():
        try:
            path.unlink()
        except OSError:
            pass
        return
    try:
        path.chmod(0o700)
    except OSError:
        pass
    shutil.rmtree(path, onerror=again)  # onexc is 3.12+ only


def _tail(text: str | bytes, limit: int = LOG_TAIL) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    text = text.strip()
    return text if len(text) <= limit else "…" + text[-limit:]


# ── tool.json ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Manifest:
    name: str
    title: str
    description: str
    entry: str
    test: tuple[str, ...]
    input: dict
    output: dict
    examples: tuple[dict, ...]
    secrets: tuple[dict, ...]
    network: bool
    timeout_s: int
    memory_mb: int
    raw: dict = field(repr=False, compare=False)
    # Phase 7: what it may ask the agent that ran it for (external_tool_uses).
    uses: tuple[dict, ...] = ()
    # Phase 7d: it may stop and ask the person running it (person_asks).
    asks: bool = False

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "title": self.title, "description": self.description,
                "input": self.input, "output": self.output, "examples": list(self.examples),
                "secrets": [dict(s) for s in self.secrets], "network": self.network,
                "timeout_s": self.timeout_s, "memory_mb": self.memory_mb,
                "uses": [dict(u) for u in self.uses], "asks": self.asks}

    @property
    def calls_back(self) -> bool:
        """Does a run need its line to Flight Deck (Phase 7)?"""
        return bool(self.uses) or self.asks


def _bad(message: str) -> ToolError:
    return ToolError("bad_manifest", f"{MANIFEST}: {message}")


def _check_schema(value: Any, what: str) -> dict:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError

    if not isinstance(value, dict):
        raise _bad(f"'{what}' must be a JSON Schema object.")
    try:
        Draft202012Validator.check_schema(value)
    except SchemaError as exc:
        raise _bad(f"'{what}' isn't a valid JSON Schema: {exc.message}") from None
    return value


def _int(raw: dict, key: str, default: int, low: int, high: int) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise _bad(f"'{key}' must be a whole number from {low} to {high}.")
    return value


def _inside(folder: Path, rel: str) -> Path | None:
    """*rel* under *folder*, or None when it would leave it (``..``, absolute, a link out)."""
    if not rel or "\0" in rel or rel.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", rel):
        return None
    if ".." in PurePosixPath(rel.replace("\\", "/")).parts:
        return None
    try:
        base = folder.resolve()
        target = (folder / rel).resolve()
        target.relative_to(base)
    except (OSError, ValueError):
        return None
    return target


def parse_manifest(raw: Any, folder: Path | None = None, *, slug: str | None = None) -> Manifest:
    """Check a tool.json. *folder* (the tool's files) also checks that the entry exists."""
    if not isinstance(raw, dict):
        raise _bad("must be a JSON object.")
    name = raw.get("name")
    if not isinstance(name, str) or not _SLUG_RE.match(name):
        raise _bad("'name' must be the tool's name (lowercase letters, digits, dashes).")
    if slug is not None and name != slug:
        raise _bad(f"'name' is {name!r} but the tool is {slug!r}.")
    if raw.get("runtime", "python") != "python":
        raise _bad("'runtime' must be \"python\": custom tools are Python programs.")
    title = raw.get("title")
    if not isinstance(title, str) or not 1 <= len(title.strip()) <= 80:
        raise _bad("'title' must be 1–80 characters.")
    description = raw.get("description")
    if not isinstance(description, str) or not 1 <= len(description.strip()) <= 600:
        raise _bad("'description' must be 1–600 characters.")
    entry = raw.get("entry")
    if not isinstance(entry, str) or not entry.endswith(".py"):
        raise _bad("'entry' must be the Python file that runs the tool, like \"main.py\".")
    if entry.startswith(("/", "\\")) or ".." in PurePosixPath(entry.replace("\\", "/")).parts:
        raise _bad(f"'entry' {entry!r} must be a path inside the tool.")
    if folder is not None:
        target = _inside(folder, entry)
        if target is None or not target.is_file():
            raise _bad(f"'entry' {entry!r} isn't a file in the tool.")
    test = raw.get("test", [])
    if not isinstance(test, list) or len(test) > 20 or not all(isinstance(a, str) and len(a) <= 200 for a in test):
        raise _bad("'test' must be a list of arguments for Python, like [\"-m\", \"unittest\", \"discover\", \"-s\", \"tests\"].")
    input_schema = _check_schema(raw.get("input"), "input")
    if input_schema.get("type") != "object":
        raise _bad("'input' must describe a JSON object (\"type\": \"object\").")
    output_schema = _check_schema(raw.get("output", {}), "output")
    examples = raw.get("examples")
    if not isinstance(examples, list) or not 1 <= len(examples) <= MAX_EXAMPLES:
        raise _bad(f"'examples' must list 1–{MAX_EXAMPLES} examples: publishing runs each one.")
    from jsonschema import Draft202012Validator

    checker = Draft202012Validator(input_schema)
    for i, ex in enumerate(examples, 1):
        if not isinstance(ex, dict) or not isinstance(ex.get("input"), dict):
            raise _bad(f"example {i} needs an 'input' object.")
        problem = next(iter(checker.iter_errors(ex["input"])), None)
        if problem is not None:
            raise _bad(f"example {i}'s input doesn't fit 'input': {problem.message}")
        calls = ex.get("calls")
        if calls is not None:
            if not isinstance(calls, list) or len(calls) > MAX_RECORDED:
                raise _bad(f"example {i}: 'calls' lists at most {MAX_RECORDED} recorded replies.")
            for c in calls:
                if not isinstance(c, dict) or sum(k in c for k in RECORDED_KINDS) != 1:
                    raise _bad(f"example {i}: each recorded call names one of {', '.join(RECORDED_KINDS)}, "
                               "with its 'reply' (or a 'path' for a file).")
    secret_list = raw.get("secrets", [])
    if not isinstance(secret_list, list) or len(secret_list) > MAX_SECRETS:
        raise _bad(f"'secrets' must be a list of at most {MAX_SECRETS}.")
    seen: set[str] = set()
    secrets_out = []
    for s in secret_list:
        if not isinstance(s, dict) or not isinstance(s.get("name"), str):
            raise _bad("each secret needs a 'name' (and a 'why').")
        sname = s["name"]
        if not _SECRET_RE.match(sname) or sname in _RESERVED_ENV or sname.startswith(_RESERVED_PREFIXES):
            raise _bad(f"secret name {sname!r} must be UPPER_CASE and not one Flight Deck sets "
                       f"(PATH, HOME, XT_*, PYTHON*, FD_*, …).")
        if sname in seen:
            raise _bad(f"secret {sname!r} is listed twice.")
        seen.add(sname)
        why = s.get("why", "")
        if not isinstance(why, str) or len(why) > 200:
            raise _bad(f"secret {sname!r}: 'why' is at most 200 characters.")
        secrets_out.append({"name": sname, "why": why.strip()})
    network = raw.get("network", False)
    if not isinstance(network, bool):
        raise _bad("'network' must be true or false.")
    from captain_claw.flight_deck import external_tool_uses as uses_policy

    def custom_use(entry: dict) -> dict:
        other = entry.get("custom_tool")
        if not isinstance(other, str) or not _SLUG_RE.match(other.strip()) or set(entry) - {"custom_tool", "why"}:
            raise uses_policy.UsesError("each use names a 'tool' (one of the agent's tools) or a 'custom_tool' "
                                        "(another of your custom tools, by its name).")
        if other.strip() == name:
            raise uses_policy.UsesError("a tool can't use itself.")
        why = entry.get("why")
        if not isinstance(why, str) or not 1 <= len(why.strip()) <= 200:
            raise uses_policy.UsesError(f"use of {other!r}: 'why' must say in 1–200 characters why the tool needs it.")
        return {"custom_tool": other.strip(), "why": why.strip()}

    try:
        uses = uses_policy.parse_uses(raw.get("uses"), parse_other=custom_use)
    except uses_policy.UsesError as exc:
        raise _bad(f"'uses': {exc}") from None
    asks = raw.get("asks", False)
    if not isinstance(asks, bool):
        raise _bad("'asks' must be true or false.")
    return Manifest(
        name=name, title=title.strip(), description=description.strip(), entry=entry,
        test=tuple(test), input=input_schema, output=output_schema, examples=tuple(examples),
        secrets=tuple(secrets_out), network=network,
        timeout_s=_int(raw, "timeout_s", DEFAULT_TIMEOUT_S, 1, MAX_TIMEOUT_S),
        memory_mb=_int(raw, "memory_mb", DEFAULT_MEMORY_MB, MIN_MEMORY_MB, MAX_MEMORY_MB),
        raw=raw, uses=uses, asks=asks,
    )


def load_manifest(folder: Path, *, slug: str | None = None) -> Manifest:
    path = folder / MANIFEST
    try:
        if path.stat().st_size > 64_000:
            raise _bad("is larger than 64 KB.")
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise _bad("is missing.") from None
    except (OSError, ValueError) as exc:
        raise _bad(f"can't be read as JSON ({exc}).") from None
    return parse_manifest(raw, folder, slug=slug)


def file_fields(schema: dict) -> dict[str, bool]:
    """Top-level input fields that name files: ``{field: is_list}``. A string with
    ``"format": "file"``, or an array whose items are."""
    out: dict[str, bool] = {}
    for key, prop in (schema.get("properties") or {}).items():
        if not isinstance(prop, dict):
            continue
        if prop.get("format") == "file":
            out[key] = False
        elif prop.get("type") == "array" and isinstance(prop.get("items"), dict) and prop["items"].get("format") == "file":
            out[key] = True
    return out


# ── Processes ────────────────────────────────────────────────────────────


@dataclass
class ProcResult:
    code: int | None
    stdout: bytes
    stderr: str
    seconds: float
    timed_out: bool = False
    too_large: bool = False


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    """The process and everything it started (it leads its own process group)."""
    try:
        if IS_WINDOWS:
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


# Answers one request a tool made on its line (Phase 7); never raises.
Bridge = Callable[[dict], Awaitable[dict]]


class RunClock:
    """A run's time limit, which stops while the run waits for a person (7d)."""

    def __init__(self, seconds: float) -> None:
        self.deadline = time.monotonic() + seconds
        self._paused = 0
        self._since: float | None = None

    def pause(self) -> None:
        self._paused += 1
        if self._paused == 1:
            self._since = time.monotonic()

    def resume(self) -> None:
        self._paused = max(0, self._paused - 1)
        if self._paused == 0 and self._since is not None:
            self.deadline += time.monotonic() - self._since
            self._since = None

    def left(self) -> float:
        now = self._since if self._since is not None else time.monotonic()
        return self.deadline - now

    def expired(self) -> bool:
        return self._paused == 0 and self.left() <= 0


async def _serve_line(sock: Any, bridge: Bridge) -> None:
    """A tool's line to the agent: one JSON request per line, one answer back,
    in order, until the tool closes it or ends."""
    reader, writer = await asyncio.open_unix_connection(sock=sock, limit=MAX_LINE)
    try:
        while True:
            try:
                line = await reader.readline()
            except (ValueError, asyncio.LimitOverrunError):
                return  # a request longer than the line allows: the tool gets EOF
            if not line:
                return
            try:
                req = json.loads(line)
                if not isinstance(req, dict):
                    raise ValueError("not an object")
            except ValueError:
                reply: dict[str, Any] = {"ok": False, "refused": True, "error": "A request must be one JSON object."}
                rid = None
            else:
                rid = req.get("id")
                try:
                    reply = await bridge(req)
                except Exception as exc:  # noqa: BLE001 — the tool hears it, the run goes on
                    log.warning("A custom tool's request failed", error=str(exc))
                    reply = {"ok": False, "error": f"Flight Deck couldn't answer: {type(exc).__name__}"}
            data = (json.dumps({**reply, "id": rid}, ensure_ascii=False, default=str) + "\n").encode("utf-8")
            if len(data) > MAX_LINE:
                data = (json.dumps({"id": rid, "ok": False, "error": "The answer was too large to hand over."})
                        + "\n").encode("utf-8")
            writer.write(data)
            await writer.drain()
    except (ConnectionError, BrokenPipeError):
        return
    finally:
        writer.close()


async def _run_proc(cmd: list[str], *, cwd: Path, env: dict[str, str], timeout: float,
                    stdin: bytes | None = None, stdout_cap: int = MAX_STDOUT,
                    bridge: Bridge | None = None, clock: RunClock | None = None) -> ProcResult:
    """Run *cmd* under *timeout*. With *bridge*, the program also gets a line
    to Flight Deck (Phase 7): one end of a socket pair, its descriptor in
    ``$XT_CAPTAIN_FD``; *bridge* answers what comes on it."""
    started = time.monotonic()
    kwargs: dict[str, Any] = {"creationflags": NO_WINDOW} if IS_WINDOWS else {"start_new_session": True}
    ours = theirs = None
    if bridge is not None:
        import socket

        ours, theirs = socket.socketpair()
        env = {**env, "XT_CAPTAIN_FD": str(theirs.fileno())}
        kwargs["pass_fds"] = (theirs.fileno(),)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(cwd), env=env,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **kwargs,
        )
    except OSError as exc:
        if ours is not None:
            ours.close()
        return ProcResult(None, b"", f"couldn't start: {exc}", 0.0)
    finally:
        if theirs is not None:
            theirs.close()  # the tool has its own copy
    out = bytearray()
    err: collections.deque[bytes] = collections.deque()
    err_len = 0
    state = {"too_large": False}

    async def feed() -> None:
        if stdin is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(stdin)
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass  # it didn't read all of it: its business
        finally:
            try:
                proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass

    async def read_out() -> None:
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(65_536)
            if not chunk:
                return
            if len(out) + len(chunk) > stdout_cap:
                state["too_large"] = True
                _kill_group(proc)
                return
            out.extend(chunk)

    async def read_err() -> None:
        nonlocal err_len
        assert proc.stderr is not None
        while True:
            chunk = await proc.stderr.read(16_384)
            if not chunk:
                return
            err.append(chunk)
            err_len += len(chunk)
            while err_len - len(err[0]) >= STDERR_TAIL and len(err) > 1:
                err_len -= len(err.popleft())

    clock = clock or RunClock(timeout)

    async def exited() -> None:
        # Not proc.wait(): on 3.11 that returns only once every pipe is closed, so
        # a child the tool left holding stdout would make a finished call look
        # like a timeout. The return code is set the moment the process ends.
        # The clock, not a fixed timeout: it stops while the tool waits for a person.
        while proc.returncode is None:
            if clock.expired():
                raise TimeoutError
            await asyncio.sleep(0.02)

    tasks = [asyncio.create_task(feed()), asyncio.create_task(read_out()), asyncio.create_task(read_err())]
    if ours is not None and bridge is not None:
        tasks.append(asyncio.create_task(_serve_line(ours, bridge)))
    timed_out = False
    try:
        await exited()
    except TimeoutError:
        timed_out = True
    except asyncio.CancelledError:
        _kill_group(proc)
        for t in tasks:
            t.cancel()
        raise
    # Whatever it left running goes too; that also ends the pipes it held open.
    _kill_group(proc)
    _, pending = await asyncio.wait(tasks, timeout=5)
    for t in pending:
        t.cancel()
    try:
        await asyncio.wait_for(proc.wait(), 5)
    except TimeoutError:
        pass
    stderr = b"".join(err)[-STDERR_TAIL:].decode("utf-8", "replace")
    return ProcResult(proc.returncode, bytes(out), stderr, time.monotonic() - started,
                      timed_out=timed_out, too_large=state["too_large"])


def _system_path() -> list[str]:
    if IS_WINDOWS:
        sysroot = os.environ.get("SYSTEMROOT", r"C:\Windows")
        return [os.path.join(sysroot, "System32"), sysroot]
    return ["/usr/local/bin", "/usr/bin", "/bin"]


def _base_env(home: Path, *, network: bool) -> dict[str, str]:
    env = {
        "HOME": str(home), "TMPDIR": str(home),
        "LANG": os.environ.get("LANG") or "C.UTF-8",
    }
    for key in ("TZ", "SSL_CERT_FILE", "SSL_CERT_DIR"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    if network:
        env.update({k: os.environ[k] for k in _PROXY_ENV if os.environ.get(k)})
    if IS_WINDOWS:
        env.update(TEMP=str(home), TMP=str(home), USERPROFILE=str(home))
        env.update({k: os.environ[k] for k in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT") if os.environ.get(k)})
    return env


def _tool_env(vdir: Path, call_dir: Path, call_id: str, manifest: Manifest,
              secret_values: dict[str, str]) -> dict[str, str]:
    """Built from nothing: the tool sees its own venv, its call's folders and the
    caller's secrets for it. Never Flight Deck's environment: code apps get
    ``FD_AGENT_SECRET`` (``app_runtime._build_subprocess_env``), which here
    would let a tool call Flight Deck as any agent."""
    deps = vdir / ".deps"
    env = _base_env(call_dir / "home", network=manifest.network)
    env.update({
        "PATH": os.pathsep.join([str(_venv_bin(deps)), *_system_path()]),
        "VIRTUAL_ENV": str(deps),
        "XT_IN": str(call_dir / "in"), "XT_OUT": str(call_dir / "out"), "XT_CALL": call_id,
    })
    for s in manifest.secrets:
        if s["name"] in secret_values:
            env[s["name"]] = secret_values[s["name"]]
    return env


# ── The sandbox (Phase 6) ────────────────────────────────────────────────
#
# On Linux the launcher puts the tool in a Landlock sandbox: it reads and runs
# only its version folder, the Python it runs on and the system's folders, and
# writes only its call folder; no TCP unless tool.json says "network": true;
# no signals to anything outside. Elsewhere (macOS, Windows, an old kernel) it
# runs without one, unless the admin asked for tools to run only in a sandbox.

_SANDBOX: dict[str, Any] | None = None
_SYSTEM_RO = ("/usr", "/lib", "/lib32", "/lib64", "/bin", "/sbin", "/etc")


def sandbox_info() -> dict[str, Any]:
    """What this host offers (the launcher's ``--probe``), asked once."""
    global _SANDBOX
    if _SANDBOX is None:
        import subprocess

        try:
            done = subprocess.run([sys.executable, "-E", "-s", str(_LAUNCHER), "--probe"], capture_output=True,
                                  text=True, timeout=15)
            info = json.loads(done.stdout or "{}")
        except (OSError, ValueError, subprocess.SubprocessError):
            info = {}
        _SANDBOX = {"os": str(info.get("os") or sys.platform), "landlock": int(info.get("landlock") or 0),
                    "files": bool(info.get("files")), "network": bool(info.get("network")),
                    "signals": bool(info.get("signals"))}
    return dict(_SANDBOX)


def _system_read_only() -> list[str]:
    """Where a tool's Python and its libraries live: read and run, never write."""
    paths = [*_SYSTEM_RO, sys.base_prefix, os.path.dirname(os.path.realpath(sys.executable)),
             os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))]
    return [p for p in dict.fromkeys(paths) if p and p != "/" and os.path.exists(p)] + ["/proc/self"]


async def _require_sandbox() -> bool:
    """The admin's "only run tools in a sandbox" (system config); off when it can't be read."""
    try:
        from captain_claw.flight_deck.admin_routes import _load_system_config
        from captain_claw.flight_deck.auth import get_db

        cfg = _load_system_config(await get_db().get_system_setting("fd:system-config"))
        return bool(cfg.get("external_tools_require_sandbox"))
    except Exception:  # noqa: BLE001 — no deck database (tests, tools): the default
        return False


async def _sandbox_gate() -> bool:
    """Whether this run must be sandboxed; refuses when it must and can't be."""
    required = await _require_sandbox()
    if required and not (await asyncio.to_thread(sandbox_info))["files"]:
        raise ToolError("no_sandbox", "This deck runs custom tools only in a sandbox, and this host has none "
                                      "(it needs Linux with Landlock).", status=503)
    return required


def _launch(manifest: Manifest, python: Path, args: list[str], *, cpu_s: int, read_only: list[Path],
            writable: list[Path], require: bool = False) -> list[str]:
    """Through the launcher, which sets the limits and the sandbox and then becomes
    the program."""
    cmd = [sys.executable, "-E", "-s", str(_LAUNCHER), "--mem", str(manifest.memory_mb), "--cpu", str(cpu_s)]
    for path in [*(str(p) for p in read_only), *_system_read_only()]:
        cmd += ["--ro", path]
    for path in writable:
        cmd += ["--rw", str(path)]
    if not manifest.network:
        cmd.append("--no-net")
    if require:
        cmd.append("--require-sandbox")
    return [*cmd, "--", str(python), "-E", "-s", "-B", *args]


# ── Running ──────────────────────────────────────────────────────────────

FileResolver = Callable[[str], Path]
# (call_id, [(relative name, file)]) -> what the caller gets told about each file
Deliver = Callable[[str, list[tuple[str, Path]]], Awaitable[list[dict[str, Any]]]]

_IN_USE: collections.Counter[tuple[str, str, str]] = collections.Counter()
_SLOTS: dict[Any, asyncio.Semaphore] = {}
_last_sweep = 0.0


def reset_for_tests() -> None:
    _RUNS.clear()
    _IN_USE.clear()
    _SLOTS.clear()
    _PUBLISHING.clear()
    _LOCKS.clear()


def _slot(key: Any, size: int) -> asyncio.Semaphore:
    sem = _SLOTS.get(key)
    if sem is None:
        sem = _SLOTS[key] = asyncio.Semaphore(size)
    return sem


def _sweep_runs() -> None:
    """Call folders a crash or restart left behind, after an hour."""
    global _last_sweep
    now = time.time()
    if now - _last_sweep < 600:
        return
    _last_sweep = now
    runs = root() / ".runs"
    if not runs.is_dir():
        return
    for caller in runs.iterdir():
        for call in caller.iterdir() if caller.is_dir() else ():
            try:
                if now - call.stat().st_mtime > 3600:
                    _rmtree(call)
            except OSError:
                pass


def _subset(expect: Any, got: Any) -> bool:
    """Does *got* hold everything *expect* says (objects may have more keys)?"""
    if isinstance(expect, dict):
        return isinstance(got, dict) and all(k in got and _subset(v, got[k]) for k, v in expect.items())
    if isinstance(expect, list):
        return isinstance(got, list) and len(expect) == len(got) and all(_subset(e, g) for e, g in zip(expect, got))
    if isinstance(expect, bool) or isinstance(got, bool):
        return expect is got
    return expect == got


def _copy_inputs(manifest: Manifest, data: dict, in_dir: Path, resolve_file: FileResolver | None) -> dict:
    fields = file_fields(manifest.input)
    if not fields:
        return data
    data = dict(data)
    total = 0
    n = 0
    for key, is_list in fields.items():
        if key not in data:
            continue
        refs = data[key] if is_list else [data[key]]
        if not isinstance(refs, list) or not all(isinstance(r, str) for r in refs):
            raise ToolError("bad_input", f"'{key}' must name {'files' if is_list else 'a file'}.", status=422)
        placed = []
        for ref in refs:
            if resolve_file is None:
                raise ToolError("files", "Files can't be passed here.", status=400)
            src = resolve_file(ref)
            try:
                if not src.is_file() or src.is_symlink():
                    raise ToolError("files", f"{ref} isn't a file you can read.", status=400)
                total += src.stat().st_size
            except OSError:
                raise ToolError("files", f"{ref} can't be read.", status=400) from None
            if total > MAX_IN_BYTES:
                raise ToolError("too_large", f"The files given come to more than {MAX_IN_BYTES // 1_048_576} MB.", status=413)
            n += 1
            name = re.sub(r"[^A-Za-z0-9._-]+", "_", src.name).strip("._") or "file"
            dest = in_dir / f"{n}-{name[:100]}"
            shutil.copyfile(src, dest)
            placed.append(str(dest))
        data[key] = placed if is_list else placed[0]
    return data


def _collect_outputs(out_dir: Path) -> tuple[list[tuple[str, Path]], int]:
    files: list[tuple[str, Path]] = []
    skipped = 0
    used = 0
    for dirpath, dirnames, filenames in os.walk(out_dir):
        dirnames[:] = sorted(d for d in dirnames if not (Path(dirpath) / d).is_symlink())
        for name in sorted(filenames):
            p = Path(dirpath) / name
            try:
                if p.is_symlink() or not p.is_file():
                    skipped += 1
                    continue
                size = p.stat().st_size
            except OSError:
                skipped += 1
                continue
            if len(files) >= MAX_OUT_FILES or used + size > MAX_OUT_BYTES:
                skipped += 1
                continue
            used += size
            files.append((p.relative_to(out_dir).as_posix(), p))
    return files, skipped


# ── A tool's line to the agent (Phase 7) ─────────────────────────────────
#
# While it runs, a tool that declares "uses" can ask the agent that ran it to
# use one of the agent's tools, or bring in a file from the person's folders.
# Each request is checked here (external_tool_uses) and then handed to a Link:
# the routes' AgentLink passes it to that agent; when publishing, RecordedLink
# answers from the example's recorded replies.


@dataclass
class CallContext:
    """One run, as its requests see it."""

    owner: str
    slug: str
    title: str
    version: str
    caller: str          # the runner: whose tools, secrets and files
    call_id: str
    call_dir: Path
    manifest: Manifest
    deadline: float      # time.monotonic() when the run's time is up
    requests: list[int] = field(default_factory=lambda: [0])  # shared with nested runs
    depth: int = 1       # 1 for a run an agent started, 2 for a tool it ran, …
    chain: tuple[tuple[str, str], ...] = ()  # (owner, slug) of the runs above this one
    clock: RunClock | None = None
    outer: CallContext | None = None   # the run that asked for this one (7b)
    attended: bool = True              # someone is there to answer a question (7d)
    agent_name: str = ""
    asks: int = 0

    @property
    def owner_runs(self) -> bool:
        return self.caller == self.owner

    def remaining(self) -> float:
        return self.clock.left() if self.clock is not None else self.deadline - time.monotonic()

    def lineage(self) -> list[CallContext]:
        out, c = [], self
        while c is not None:
            out.append(c)
            c = c.outer
        return out


@dataclass(frozen=True)
class HandedFile:
    """A file the tool wrote, marked with captain.file() to hand to a tool."""

    path: Path


class Link:
    """Where a run's requests go. Answers ``{"ok", "content", "error"}`` and the like."""

    async def tool(self, ctx: CallContext, name: str, params: dict) -> dict[str, Any]:
        raise ToolError("no_agent", "This run has no agent to ask.")

    async def file(self, ctx: CallContext, ref: str) -> Path:
        raise ToolError("no_agent", "This run has no agent to ask.")

    async def custom_tool(self, ctx: CallContext, slug: str, params: dict) -> dict[str, Any]:
        raise ToolError("no_agent", "This run has no agent to ask.")

    async def ask(self, ctx: CallContext, questions: list[dict], wait_s: float) -> dict[str, Any]:
        raise ToolError("no_agent", "This run has nobody to ask.")


MAX_DEPTH = 3                      # an agent's run, a tool it runs, and one more
_RUNS: dict[str, CallContext] = {}  # running calls that can ask for more, by call id


def running(call_id: str) -> CallContext | None:
    """A call that is still running and may run other tools (Phase 7b)."""
    return _RUNS.get(str(call_id or ""))


def _refuse(message: str) -> dict[str, Any]:
    return {"ok": False, "refused": True, "error": message}


def _mark_files(ctx: CallContext, value: Any) -> Any:
    """captain.file() markers in a request, checked: only a file inside this
    run's own folder can be handed on."""
    if isinstance(value, dict):
        if set(value) == {"$file"}:
            raw = value["$file"]
            try:
                target = Path(str(raw)).resolve()
                target.relative_to(ctx.call_dir.resolve())
            except (OSError, ValueError):
                raise ToolError("files", f"{raw!r}: only a file this run wrote can be handed on.") from None
            if not target.is_file() or target.is_symlink():
                raise ToolError("files", f"{raw!r} isn't a file.")
            return HandedFile(target)
        return {k: _mark_files(ctx, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_mark_files(ctx, v) for v in value]
    return value


async def _answer(ctx: CallContext, link: Link | None, req: dict) -> dict[str, Any]:
    """One request from a running tool, checked and handed to *link*."""
    from captain_claw.flight_deck import external_tool_uses as uses_policy

    op = req.get("op")
    if op not in ("tool", "file", "custom_tool", "ask"):
        return _refuse(f"Flight Deck doesn't know the request {op!r}.")
    if link is None:
        return _refuse("This run has no agent to ask.")
    ctx.requests[0] += 1
    if ctx.requests[0] > uses_policy.MAX_REQUESTS:
        return _refuse(f"A run may make at most {uses_policy.MAX_REQUESTS} requests.")
    left = ctx.remaining()
    if left < 1:
        return _refuse("The run has no time left for another request.")
    if op == "ask":
        return await _ask(ctx, link, req)
    try:
        if op == "tool":
            name = str(req.get("name") or "").strip().lower()
            params = req.get("input")
            why = uses_policy.check_request(ctx.manifest.uses, name, params, owner_runs=ctx.owner_runs)
            if why:
                return _refuse(why)
            params = _mark_files(ctx, params)
            reply = await asyncio.wait_for(link.tool(ctx, name, params), left)
            content = str(reply.get("content") or "")
            if len(content) > MAX_ANSWER:
                content = content[:MAX_ANSWER] + f"\n… (cut: {len(content)} characters in all)"
            return {"ok": bool(reply.get("ok")), "content": content, "error": reply.get("error") or None}
        if op == "custom_tool":
            name = str(req.get("name") or "").strip()
            if not any(u.get("custom_tool") == name for u in ctx.manifest.uses):
                return _refuse(f"{name} isn't in this tool's uses (tool.json): it can only run the custom tools it "
                               "declared.")
            params = req.get("input")
            if not isinstance(params, dict):
                return _refuse(f"The input for {name} must be an object.")
            reply = await asyncio.wait_for(link.custom_tool(ctx, name, _mark_files(ctx, params)), left)
            return {"ok": bool(reply.get("ok")), "output": reply.get("output"),
                    "files": [str(f) for f in reply.get("files") or []], "error": reply.get("error") or None}
        # op == "file": a file from the person's folders, brought into this run.
        if not any(u.get("tool") == "read" for u in ctx.manifest.uses):
            return _refuse('Bringing in your files needs {"tool": "read"} in this tool\'s uses.')
        path = await asyncio.wait_for(link.file(ctx, str(req.get("ref") or "")), left)
        return {"ok": True, "path": str(path)}
    except ToolError as exc:
        return {"ok": False, "error": exc.message}
    except TimeoutError:
        return {"ok": False, "error": "The run ran out of time waiting for the answer."}


async def _ask(ctx: CallContext, link: Link, req: dict) -> dict[str, Any]:
    """The tool asks the person running it (7d). Its time stops while it waits,
    and so does the time of every run above it."""
    from captain_claw.flight_deck import person_asks as asks

    if not ctx.manifest.asks:
        return _refuse('Asking the person needs "asks": true in tool.json.')
    ctx.asks += 1
    if ctx.asks > asks.MAX_ASKS_PER_RUN:
        return _refuse(f"A run may ask at most {asks.MAX_ASKS_PER_RUN} times.")
    try:
        questions = asks.check_questions(req.get("questions"))
    except asks.AskError as exc:
        return _refuse(str(exc))
    if not ctx.attended:
        return {"ok": False, "no_answer": True,
                "error": "No one to ask: this ran on its own (a schedule or another automatic start)."}
    try:
        wait_s = float(req.get("wait_s") or asks.DEFAULT_WAIT_S)
    except (TypeError, ValueError):
        wait_s = asks.DEFAULT_WAIT_S
    clocks = [c.clock for c in ctx.lineage() if c.clock is not None]
    for clock in clocks:
        clock.pause()
    try:
        reply = await link.ask(ctx, questions, wait_s)
    except ToolError as exc:
        return {"ok": False, "no_answer": True, "error": exc.message}
    finally:
        for clock in clocks:
            clock.resume()
    out = {"ok": bool(reply.get("ok"))}
    for key in ("answers", "declined", "no_answer", "error"):
        if key in reply:
            out[key] = reply[key]
    return out


def _bring_in(ctx: CallContext, src: Path, name: str = "") -> Path:
    """Copy *src* into the run's in/ folder; the path the tool reads."""
    in_dir = ctx.call_dir / "in"
    in_dir.mkdir(parents=True, exist_ok=True)
    used = sum(p.stat().st_size for p in in_dir.iterdir() if p.is_file())
    if used + src.stat().st_size > MAX_IN_BYTES:
        raise ToolError("too_large", f"The run's files would come to more than {MAX_IN_BYTES // 1_048_576} MB.")
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", name or src.name).strip("._") or "file"
    dest = in_dir / f"x{_secrets.token_hex(3)}-{clean[:100]}"
    shutil.copyfile(src, dest)
    return dest


class RecordedLink(Link):
    """Publishing: no agent, so an example's recorded replies answer, in order.
    A request that doesn't match the next one is a problem the example fails on."""

    def __init__(self, calls: Any, vdir: Path) -> None:
        self.calls: list[dict] = [c for c in calls if isinstance(c, dict)] if isinstance(calls, list) else []
        self.vdir = vdir
        self.problems: list[str] = []

    def _next(self, kind: str, what: str) -> dict:
        if not self.calls:
            msg = f"it asked for {kind} {what} and there's no recorded reply left for it"
            self.problems.append(msg)
            raise ToolError("recorded", msg)
        nxt = self.calls.pop(0)
        want = nxt.get(kind)
        if want is None or (kind != "ask" and str(want) != what):
            shown = json.dumps({k: v for k, v in nxt.items() if k not in ("reply",)}, ensure_ascii=False)
            msg = f"it asked for {kind} {what}, but the next recorded call is {shown}"
            self.problems.append(msg)
            raise ToolError("recorded", msg)
        return nxt

    async def tool(self, ctx: CallContext, name: str, params: dict) -> dict[str, Any]:
        reply = self._next("tool", name).get("reply")
        reply = reply if isinstance(reply, dict) else {"ok": True, "content": "" if reply is None else str(reply)}
        return {"ok": bool(reply.get("ok", True)), "content": str(reply.get("content") or ""),
                "error": reply.get("error")}

    async def file(self, ctx: CallContext, ref: str) -> Path:
        nxt = self._next("file", ref)
        src = _inside(self.vdir, str(nxt.get("path") or ""))
        if src is None or not src.is_file():
            msg = f"the recorded file for {ref} must be a 'path' to a file in the tool"
            self.problems.append(msg)
            raise ToolError("recorded", msg)
        return _bring_in(ctx, src, Path(ref).name)

    async def custom_tool(self, ctx: CallContext, slug: str, params: dict) -> dict[str, Any]:
        reply = self._next("custom_tool", slug).get("reply")
        reply = reply if isinstance(reply, dict) else {"ok": True, "output": reply}
        return {"ok": bool(reply.get("ok", True)), "output": reply.get("output"), "files": [],
                "error": reply.get("error")}

    async def ask(self, ctx: CallContext, questions: list[dict], wait_s: float) -> dict[str, Any]:
        """A recorded answer, checked the way a real one is: keyed by the
        questions' ids (a question asked as plain text has the id "answer"),
        each answer one the question takes. A recording that couldn't come
        from the dialog fails the example and says how to record it."""
        from captain_claw.flight_deck import person_asks

        ids = [q["id"] for q in questions]
        recorded = self._next("ask", ",".join(ids))
        listed = recorded.get("ask")
        if isinstance(listed, list) and [str(x) for x in listed] != ids:
            return self._bad_recording(f"it asked {ids}, but the recorded call lists {listed}: list the questions' "
                                       f"ids (a question asked as plain text has the id \"answer\")")
        reply = recorded.get("reply")
        reply = reply if isinstance(reply, dict) else {}
        if reply.get("declined"):
            return {"ok": False, "declined": True, "error": "They chose not to answer."}
        if reply.get("no_answer"):
            return {"ok": False, "no_answer": True, "error": "No answer in time."}
        answers = reply.get("answers")
        if not isinstance(answers, dict):
            return self._bad_recording('the recorded reply to ask needs "answers" (by question id), '
                                       '{"declined": true} or {"no_answer": true}')
        unknown = [k for k in answers if k not in ids]
        if unknown:
            return self._bad_recording(f"the recorded answers name {unknown}, which aren't question ids: key them "
                                       f"by {ids} (a question asked as plain text has the id \"answer\")")
        try:
            return {"ok": True, "answers": person_asks.check_answers(questions, answers)}
        except person_asks.AskError as exc:
            return self._bad_recording(f"the recorded answers couldn't come from the dialog ({exc}); for no answer "
                                       'at all, record {"no_answer": true} or {"declined": true}')

    def _bad_recording(self, why: str) -> dict[str, Any]:
        self.problems.append(why)
        return {"ok": False, "no_answer": True, "error": f"Bad recording: {why}"}

    def left_over(self) -> str:
        return f"{len(self.calls)} recorded call(s) weren't asked for" if self.calls else ""


async def _run_version(owner: str, slug: str, version: str, vdir: Path, manifest: Manifest, caller: str,
                       data: Any, *, secret_values: dict[str, str], resolve_file: FileResolver | None,
                       deliver: Deliver | None, link: Link | None = None,
                       parent: CallContext | None = None, attended: bool = True,
                       agent_name: str = "") -> dict[str, Any]:
    from jsonschema import Draft202012Validator

    timeout_s = manifest.timeout_s
    if parent is not None:
        # Run by another tool: inside its time, below it in the chain.
        timeout_s = min(timeout_s, int(parent.remaining() - 5))
        if timeout_s < 1:
            raise ToolError("no_time", f"The tool that asked for {manifest.title} has no time left to run it.",
                            status=409)

    if manifest.calls_back:
        if IS_WINDOWS:
            raise ToolError("windows", f"{manifest.title} calls back into the agent, which needs Linux or macOS.",
                            status=501)
        if link is None:
            raise ToolError("needs_agent", f"{manifest.title} uses your agent's tools, so it runs only from an "
                                           "agent (ask one in chat).", status=409)

    if not isinstance(data, dict):
        raise ToolError("bad_input", "The input must be a JSON object.", status=422)
    problem = next(iter(Draft202012Validator(manifest.input).iter_errors(data)), None)
    if problem is not None:
        where = "/".join(str(p) for p in problem.absolute_path) or "input"
        raise ToolError("bad_input", f"The input doesn't fit this tool ({where}): {problem.message}", status=422)
    missing = [s["name"] for s in manifest.secrets if not secret_values.get(s["name"])]
    if missing:
        raise ToolError("missing_secret", f"{', '.join(missing)} isn't set: Custom tools → {manifest.title} → Secrets.",
                        status=400)
    _check_owner(caller)
    require = await _sandbox_gate()
    _sweep_runs()
    call_id = "xc_" + _secrets.token_hex(6)
    call_dir = root() / ".runs" / caller / call_id
    key = (owner, slug, version)
    tool_slot = _slot(("tool", caller, owner, slug), PER_TOOL_PARALLEL)
    deck_slot = _slot("deck", MAX_PARALLEL)
    try:
        await asyncio.wait_for(tool_slot.acquire(), SLOT_WAIT_S)
    except TimeoutError:
        raise ToolError("busy", f"{manifest.title} is busy with your other calls. Try again in a moment.", status=429) from None
    try:
        try:
            await asyncio.wait_for(deck_slot.acquire(), SLOT_WAIT_S)
        except TimeoutError:
            raise ToolError("busy", "Too many custom tools are running on this deck. Try again in a moment.",
                            status=429) from None
        _IN_USE[key] += 1
        try:
            for sub in ("in", "out", "home"):
                (call_dir / sub).mkdir(parents=True, exist_ok=True)
            data = _copy_inputs(manifest, data, call_dir / "in", resolve_file)
            env = _tool_env(vdir, call_dir, call_id, manifest, secret_values)
            cmd = _launch(manifest, venv_python(vdir / ".deps"), [manifest.entry], cpu_s=timeout_s + 5,
                          read_only=[vdir], writable=[call_dir], require=require)
            bridge = None
            clock = RunClock(timeout_s)
            if manifest.calls_back:
                ctx = CallContext(owner=owner, slug=slug, title=manifest.title, version=version, caller=caller,
                                  call_id=call_id, call_dir=call_dir, manifest=manifest,
                                  deadline=clock.deadline, clock=clock, attended=attended, agent_name=agent_name)
                if parent is not None:
                    ctx.requests = parent.requests
                    ctx.depth = parent.depth + 1
                    ctx.chain = (*parent.chain, (parent.owner, parent.slug))
                    ctx.outer = parent
                    ctx.attended = parent.attended
                    ctx.agent_name = parent.agent_name
                _RUNS[call_id] = ctx

                async def bridge(req: dict) -> dict:
                    return await _answer(ctx, link, req)

            try:
                res = await _run_proc(cmd, cwd=vdir, env=env, timeout=timeout_s,
                                      stdin=json.dumps(data, ensure_ascii=False).encode("utf-8"), bridge=bridge,
                                      clock=clock)
            finally:
                _RUNS.pop(call_id, None)
                if manifest.asks:
                    from captain_claw.flight_deck import person_asks

                    person_asks.close_call(call_id)
            base = {"call_id": call_id, "version": version, "seconds": round(res.seconds, 3), "exit_code": res.code}

            def failed(kind: str, message: str) -> dict[str, Any]:
                return {**base, "ok": False, "error": {"kind": kind, "message": message}, "stderr": _tail(res.stderr)}

            if res.timed_out:
                return failed("timeout", f"It ran longer than {timeout_s} s and was stopped.")
            if res.too_large:
                return failed("too_large", f"It printed more than {MAX_STDOUT // 1000} KB; put big results in a file in $XT_OUT.")
            if res.code != 0:
                return failed("crashed", f"It stopped with exit code {res.code}.")
            text = res.stdout.decode("utf-8", "replace").strip()
            try:
                output = json.loads(text)
            except ValueError:
                head = text[:300] or "(nothing)"
                return failed("bad_output", f"It didn't print one JSON value. It printed: {head}")
            problem = next(iter(Draft202012Validator(manifest.output).iter_errors(output)), None)
            if problem is not None:
                where = "/".join(str(p) for p in problem.absolute_path) or "output"
                return failed("bad_output", f"Its output doesn't fit what it promised ({where}): {problem.message}")
            files, skipped = _collect_outputs(call_dir / "out")
            delivered = await deliver(call_id, files) if (deliver and files) else []
            result = {**base, "ok": True, "output": output, "files": delivered}
            if skipped:
                result["files_skipped"] = skipped
            if res.stderr.strip():
                result["stderr"] = _tail(res.stderr, 2_000)
            return result
        finally:
            _IN_USE[key] -= 1
            if _IN_USE[key] <= 0:
                del _IN_USE[key]
            deck_slot.release()
            await asyncio.to_thread(_rmtree, call_dir)
    finally:
        tool_slot.release()


def read_live(owner: str, slug: str) -> dict | None:
    live = _read_json(_live_file(owner, slug))
    if not isinstance(live, dict) or not _VERSION_RE.match(str(live.get("version") or "")):
        return None
    return live


def _ready(owner: str, slug: str, version: str) -> Path | None:
    if not _VERSION_RE.match(version or ""):
        return None
    vdir = versions_dir(owner, slug) / version
    return vdir if (vdir / READY).is_file() else None


def manifest_of(owner: str, slug: str) -> Manifest | None:
    """The live version's tool.json, or None."""
    live = read_live(owner, slug)
    vdir = _ready(owner, slug, live["version"]) if live else None
    if vdir is None:
        return None
    try:
        return load_manifest(vdir, slug=slug)
    except ToolError:
        return None


def depends_on(owner: str, slug: str) -> list[str]:
    """The owner's other tools the live version runs (its custom_tool uses)."""
    m = manifest_of(owner, slug)
    return [u["custom_tool"] for u in (m.uses if m else ()) if "custom_tool" in u]


async def run_tool(owner: str, slug: str, caller: str, data: Any, *, secret_values: dict[str, str] | None = None,
                   resolve_file: FileResolver | None = None, deliver: Deliver | None = None,
                   link: Link | None = None, parent: CallContext | None = None, attended: bool = True,
                   agent_name: str = "") -> dict[str, Any]:
    """Run the live version for *caller*. A tool that ran and failed is a result
    (``ok: False``); something the caller must change raises :class:`ToolError`.
    Who may run it is the caller's business (the routes check it)."""
    _require(owner, slug)
    live = read_live(owner, slug)
    vdir = _ready(owner, slug, live["version"]) if live else None
    if vdir is None:
        raise ToolError("not_live", f"{slug} has no published version yet.", status=409)
    manifest = load_manifest(vdir, slug=slug)
    result = await _run_version(owner, slug, live["version"], vdir, manifest, caller, data,
                                secret_values=secret_values or {}, resolve_file=resolve_file, deliver=deliver,
                                link=link, parent=parent, attended=attended, agent_name=agent_name)
    record_call(owner, slug, caller, result)
    return result


# ── Usage: what the owner sees of the calls (counts only) ────────────────

USAGE_DAYS = 30


def _usage_file(owner: str, slug: str) -> Path:
    return tool_dir(owner, slug) / "usage.json"


def record_call(owner: str, slug: str, caller: str, result: dict[str, Any]) -> None:
    """Count one call. The caller is kept as a short hash, only to count people:
    never their input, output, files or error text."""
    import hashlib

    try:
        path = _usage_file(owner, slug)
        data = _read_json(path) or {}
        days = data.get("days") if isinstance(data.get("days"), dict) else {}
        today = time.strftime("%Y-%m-%d", time.gmtime())
        day = days.setdefault(today, {"calls": 0, "failed": 0, "kinds": {}, "people": [], "versions": {}})
        day["calls"] += 1
        if not result.get("ok"):
            day["failed"] += 1
            kind = str((result.get("error") or {}).get("kind") or "error")
            day["kinds"][kind] = day["kinds"].get(kind, 0) + 1
        who = hashlib.sha256(f"{owner}/{slug}/{caller}".encode()).hexdigest()[:12]
        if who not in day["people"]:
            day["people"].append(who)
        version = str(result.get("version") or "")
        if version:
            day["versions"][version] = day["versions"].get(version, 0) + 1
        for old in sorted(days)[:-USAGE_DAYS]:
            del days[old]
        _write_json(path, {"days": days})
    except (OSError, ToolError) as exc:
        log.debug("Couldn't count a custom tool call: %s", exc)


def usage_summary(owner: str, slug: str) -> dict[str, Any]:
    data = _read_json(_usage_file(owner, slug)) or {}
    days = data.get("days") if isinstance(data.get("days"), dict) else {}
    rows, kinds, versions, people = [], {}, {}, set()
    for name in sorted(days):
        d = days[name]
        rows.append({"day": name, "calls": d.get("calls", 0), "failed": d.get("failed", 0),
                     "people": len(d.get("people") or [])})
        for k, n in (d.get("kinds") or {}).items():
            kinds[k] = kinds.get(k, 0) + n
        for v, n in (d.get("versions") or {}).items():
            versions[v] = versions.get(v, 0) + n
        people.update(d.get("people") or [])
    return {"days": rows, "failures": kinds, "versions": versions, "people": len(people),
            "calls": sum(r["calls"] for r in rows), "failed": sum(r["failed"] for r in rows)}


# ── Creating a tool ──────────────────────────────────────────────────────

# What a Crew worker building or changing a tool is told (crew_runner._first_prompt).
WORKER_CONTRACT = """This repository is a Captain custom tool: a small Python program Flight Deck runs for agents, one process per call.
- tool.json describes it and must stay true: name (never change it), title, description, entry (the Python file it runs), test (the arguments after `python` that run the tests), input and output (JSON Schema; input is an object), examples (1 to 10, each an `input` and the `expect` part the output must contain), secrets (environment variables it reads, each with a `why`), network (true only if it uses the internet), timeout_s (at most 300) and memory_mb (at most 4096).
- The program reads ONE JSON object on stdin and prints ONE JSON value on stdout, and nothing else there; logs go to stderr. A failure is a non-zero exit with the reason on stderr.
- An input field with "format": "file" arrives as the path of a file to read. Files for the caller go in the folder named by $XT_OUT. Secrets come only from the environment variables listed in `secrets`: never put a key in the repository. When it runs, its own folder is read-only: write only to $XT_OUT or $TMPDIR.
- Python only. Dependencies go in requirements.txt with pinned versions, test-only ones in requirements-test.txt. If you need them while you work, make a venv in .deps/ (git ignores it).
- Before you finish, run the tests (what `test` says) and every example (pipe its input JSON into the entry file) and check each output contains its `expect`. Publishing runs exactly these, and a version goes live only if all of them pass. Commit your work.
- Keep it deterministic and small: no network unless the task needs it, nothing left running when it exits.
- Using the agent (only when the task needs it): the tool can ask the agent that ran it to use one of the agent's tools, through `captain.py` (Flight Deck's helper, already in the repository; don't change it): `captain.tool("web_fetch", url=...)` returns {"ok", "content", "error"}; `captain.vfs_file("vfs:folder/file")` brings in one of the person's files; `captain.file(path)` hands a file the tool wrote to a tool. Declare each in tool.json "uses", with a "why": {"tool": "web_fetch", "hosts": ["api.example.com"], "why": "..."} (web fetching must name its hosts), {"tool": "google_mail", "actions": ["search", "read"], "why": "..."}, {"tool": "read", "why": "..."} for vfs_file. Allowed: the agent's read and write tools; sending (mail, messages, calendar invitations) only works when the tool's owner runs it; never shell, code, payments, scheduling or changing the agent. Anything not declared is refused.
- Asking the person (only when the task needs a choice or a decision only they can make): set "asks": true in tool.json and call `captain.ask("Which account?", choices=[...])` (or confirm=True, multiple=True, or a list of question objects with ids for a set shown in one dialog). It waits for the answer in Flight Deck (the tool's time stops meanwhile) and raises captain.Declined or captain.NoAnswer when there is none (always so when nobody is there, like a scheduled run): handle both. Never ask for passwords or keys: those are secrets.
- A tool with "uses" or "asks" can't reach an agent or a person while publishing, so each example records the replies it gets, in order, under "calls": [{"tool": "web_fetch", "reply": {"ok": true, "content": "..."}}] ({"file": "vfs:...", "path": "fixtures/x.csv"} for vfs_file, {"ask": ["answer"], "reply": {"answers": {"answer": "Business"}}} for ask). Answers are keyed by question id, never by the question's text: a question asked as plain text (`captain.ask("What is your name?")`) has the id "answer"; for a list, use the ids you gave. A recorded answer must be one the dialog could give (a choice from the list, true/false for confirm, text for a required question); for no answer record {"ask": [...], "reply": {"no_answer": true}} or {"declined": true}. Publishing checks recordings this way. Tests do the same: write such a list to a JSON file and set XT_FAKE_CALLS to its path before running the entry file. Results from the web or a model-backed tool change from run to run: say so in the description."""

_IDENTITY = ("-c", "user.name=Captain", "-c", "user.email=tools@captain-claw.local")


def _starter(slug: str, title: str, description: str) -> dict[str, str]:
    manifest = {
        "name": slug, "title": title, "description": description, "runtime": "python",
        "entry": "main.py",
        "test": ["-m", "unittest", "discover", "-s", "tests"],
        "input": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        "output": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        "examples": [{"input": {"text": "hello"}, "expect": {"text": "hello"}}],
        "secrets": [], "network": False, "timeout_s": DEFAULT_TIMEOUT_S, "memory_mb": DEFAULT_MEMORY_MB,
    }
    main_py = f'''"""{title}

{description}

Reads one JSON object on stdin and prints one JSON value on stdout.
Logs and progress go to stderr. A file named in the input is a path you can
read; files for the caller go in the folder named by $XT_OUT.
"""

import json
import sys


def run(data: dict) -> dict:
    return {{"text": data["text"]}}


def main() -> None:
    data = json.load(sys.stdin)
    json.dump(run(data), sys.stdout)


if __name__ == "__main__":
    main()
'''
    test_py = '''import json
import subprocess
import sys
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parent.parent


class MainTest(unittest.TestCase):
    def test_echo(self):
        done = subprocess.run([sys.executable, str(TOOL / "main.py")], input=json.dumps({"text": "hi"}),
                              capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(done.stdout), {"text": "hi"})


if __name__ == "__main__":
    unittest.main()
'''
    readme = f"""# {title}

{description}

A Captain custom tool. Flight Deck runs `main.py` once per call: the input arrives
as one JSON object on stdin, and the tool prints one JSON value on stdout.

- `tool.json` describes it: input and output (JSON Schema), examples, secrets,
  network, time and memory limits. Keep it true.
- Every example runs when the tool is published; a version goes live only when
  the tests (`test` in tool.json) and every example pass. `expect` lists what the
  output must contain.
- Dependencies: `requirements.txt` (and `requirements-test.txt` for tests only).
  Python only.
- Secrets arrive as environment variables named in `secrets`; never put them in
  files here.
- An input field with `"format": "file"` arrives as a path you can read. Write
  files for the caller to the folder in `$XT_OUT`.
- Nothing else is guaranteed: no network unless `network` is true, and the
  tool's own folder is read-only when it runs.
- `captain.py` is Flight Deck's helper for asking the agent that ran the tool
  to use one of its tools (declared under `uses` in tool.json). Don't change
  it: publishing puts in Flight Deck's own copy.
"""
    return {
        MANIFEST: json.dumps(manifest, indent=2) + "\n",
        "main.py": main_py,
        "tests/test_main.py": test_py,
        "README.md": readme,
        ".gitignore": ".deps/\n__pycache__/\n*.pyc\n.pytest_cache/\n",
        HELPER_NAME: _HELPER.read_text(encoding="utf-8"),
    }


def place_helper(folder: Path) -> bool:
    """Put Flight Deck's current helper into a Crew worktree of a tool, so its
    worker can use and test it; True when it changed something. Publishing puts
    in its own copy either way."""
    target = Path(folder) / HELPER_NAME
    current = _HELPER.read_text(encoding="utf-8")
    try:
        if target.is_file() and not target.is_symlink() and target.read_text(encoding="utf-8") == current:
            return False
        if target.is_symlink():
            target.unlink()
        target.write_text(current, encoding="utf-8")
        return True
    except OSError as exc:
        log.warning("Couldn't put the custom tool helper in place", folder=str(folder), error=str(exc))
        return False


async def create_tool(owner: str, slug: str, *, title: str, description: str) -> dict[str, Any]:
    """A new tool's folder and repository, with a starter commit that already
    passes its own test and example. Crew builds on it from there."""
    from captain_claw.flight_deck import crew_runner

    slug = check_slug(slug)
    if slug in RESERVED_SLUGS:
        raise ToolError("bad_name", f"“{slug}” is a name Flight Deck uses itself: pick another.")
    title = " ".join(str(title or "").split()) or slug
    description = " ".join(str(description or "").split())
    if not 1 <= len(title) <= 80:
        raise ToolError("bad_input", "The title is 1–80 characters.", status=422)
    if not 1 <= len(description) <= 600:
        raise ToolError("bad_input", "Say in 1–600 characters what the tool does.", status=422)
    folder = tool_dir(owner, slug)
    async with _lock(owner, slug):
        if folder.exists():
            raise ToolError("exists", f"You already have a tool called {slug}.", status=409)
        repo = folder / "repo"
        repo.mkdir(parents=True)
        try:
            for rel, text in _starter(slug, title, description).items():
                p = repo / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text, encoding="utf-8")
            for args in (("init", "--quiet"), ("symbolic-ref", "HEAD", "refs/heads/main"), ("add", "-A"),
                         (*_IDENTITY, "commit", "--quiet", "-m", f"Start {slug}")):
                rc, out = await crew_runner._git(repo, *args)
                if rc != 0:
                    raise ToolError("git", f"Couldn't set up the repository: {_tail(out, 300)}", status=500)
        except BaseException:
            _rmtree(folder)
            raise
    return summary(owner, slug)


async def delete_tool(owner: str, slug: str) -> None:
    _require(owner, slug)
    async with _lock(owner, slug):
        if (owner, slug) in _PUBLISHING:
            raise ToolError("publishing", "It's being published. Wait for that to finish.", status=409)
        busy = sum(n for (o, s, _), n in _IN_USE.items() if (o, s) == (owner, slug))
        if busy:
            raise ToolError("busy", f"{busy} call(s) are running. Try again when they finish.", status=409)
        await asyncio.to_thread(_rmtree, tool_dir(owner, slug))


# ── Publishing ───────────────────────────────────────────────────────────

_PUBLISHING: dict[tuple[str, str], asyncio.Task] = {}
_LOCKS: dict[tuple[str, str], asyncio.Lock] = {}
# Called after a version goes live: (owner, slug, old version or None, new version).
# external_tools_routes tells the people who added the tool what it now asks for.
AFTER_PUBLISH: list[Callable[[str, str, str | None, str], Awaitable[None]]] = []


def _lock(owner: str, slug: str) -> asyncio.Lock:
    key = (owner, slug)
    lock = _LOCKS.get(key)
    if lock is None:
        lock = _LOCKS[key] = asyncio.Lock()
    return lock


_PYTEST_OWN = (("pytest.ini", ""), (".pytest.ini", ""), ("pyproject.toml", "[tool.pytest"),
               ("tox.ini", "[pytest]"), ("setup.cfg", "[tool:pytest]"))


def _pin_pytest_root(folder: Path) -> bool:
    """A tool without a pytest configuration of its own gets an empty pytest.ini,
    so its tests never pick up one from the folders above: on a deck whose data
    sits inside a Captain checkout that would be Captain's own (its asyncio
    mode, and testpaths = tests: a bare `pytest` would run Captain's suite).
    True when it put one in."""
    for name, marker in _PYTEST_OWN:
        path = folder / name
        try:
            if path.is_file() and (not marker or marker in path.read_text(encoding="utf-8", errors="replace")):
                return False
        except OSError:
            return False
    (folder / "pytest.ini").write_text("# Put in by Flight Deck: this tool's tests use pytest's defaults, not a\n"
                                       "# configuration found in the folders above it.\n[pytest]\n", encoding="utf-8")
    return True


def _base_python() -> str:
    """The Python venvs are made from: Flight Deck's own (the desktop app's is a
    real, uv-installed Python), unless this is a frozen build."""
    if not getattr(sys, "frozen", False):
        return sys.executable
    found = shutil.which("python3") or shutil.which("python")
    if not found:
        raise ToolError("no_python", "This Flight Deck has no Python to build custom tools with.", status=500)
    return found


def _install_env(home: Path) -> dict[str, str]:
    env = _base_env(home, network=True)
    uv = shutil.which("uv")
    extra = [str(Path(uv).parent)] if uv else []
    env["PATH"] = os.pathsep.join([*extra, *_system_path()])
    cache = root() / ".cache"
    env["UV_CACHE_DIR"] = str(cache / "uv")
    env["PIP_CACHE_DIR"] = str(cache / "pip")
    return env


class _StepFailedError(Exception):
    pass


async def _install(vdir: Path, home: Path, step: Callable[[str, bool, str], None]) -> None:
    deps = vdir / ".deps"
    reqs = [name for name in ("requirements.txt", "requirements-test.txt") if (vdir / name).is_file()]
    base = _base_python()
    env = _install_env(home)
    uv = shutil.which("uv")
    if uv:
        cmds = [[uv, "venv", "--quiet", "--python", base, str(deps)]]
        for r in reqs:
            cmds.append([uv, "pip", "install", "--quiet", "--python", str(venv_python(deps)), "-r", r])
    else:
        cmds = [[base, "-m", "venv", *([] if reqs else ["--without-pip"]), str(deps)]]
        for r in reqs:
            cmds.append([str(venv_python(deps)), "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                         "--no-input", "-r", r])
    logs = []
    for cmd in cmds:
        res = await _run_proc(cmd, cwd=vdir, env=env, timeout=INSTALL_TIMEOUT_S)
        logs.append(f"$ {' '.join(Path(cmd[0]).name if i == 0 else a for i, a in enumerate(cmd))}\n"
                    f"{res.stdout.decode('utf-8', 'replace')}{res.stderr}")
        if res.timed_out or res.code != 0:
            why = "took too long" if res.timed_out else f"exit code {res.code}"
            step("install", False, _tail("\n".join(logs)) + f"\n({why})")
            raise _StepFailedError("Installing its dependencies failed.")
    step("install", True, _tail("\n".join(logs)) or ("no requirements" if not reqs else ""))


async def _run_tests(vdir: Path, manifest: Manifest, home: Path, secret_values: dict[str, str],
                     step: Callable[[str, bool, str], None]) -> None:
    if not manifest.test:
        step("tests", True, "no tests listed in tool.json")
        return
    call_dir = home.parent
    for sub in ("in", "out"):
        (call_dir / sub).mkdir(exist_ok=True)
    env = _tool_env(vdir, call_dir, "xc_tests", manifest, secret_values)
    # Not live yet: its tests may write in their own folder (a cache, a fixture).
    cmd = _launch(manifest, venv_python(vdir / ".deps"), list(manifest.test), cpu_s=TEST_TIMEOUT_S + 5,
                  read_only=[], writable=[vdir, call_dir], require=await _sandbox_gate())
    res = await _run_proc(cmd, cwd=vdir, env=env, timeout=TEST_TIMEOUT_S, stdout_cap=MAX_STDOUT * 4)
    log_text = _tail(res.stdout.decode("utf-8", "replace") + "\n" + res.stderr)
    if res.timed_out or res.code != 0:
        why = f"took longer than {TEST_TIMEOUT_S} s" if res.timed_out else f"exit code {res.code}"
        step("tests", False, f"{log_text}\n({why})")
        raise _StepFailedError("Its tests failed.")
    step("tests", True, log_text)
    # What the tests left in the tool's own files (never inside its venv).
    for dirpath, dirnames, _ in os.walk(vdir):
        if Path(dirpath) == vdir and ".deps" in dirnames:
            dirnames.remove(".deps")
        for d in list(dirnames):
            if d in ("__pycache__", ".pytest_cache"):
                _rmtree(Path(dirpath) / d)
                dirnames.remove(d)


def read_publish(owner: str, slug: str) -> dict | None:
    state = _read_json(_publish_file(owner, slug))
    if not isinstance(state, dict):
        return None
    if state.get("state") == "running" and (owner, slug) not in _PUBLISHING:
        # Flight Deck stopped partway through.
        state = {**state, "state": "failed", "error": "Flight Deck restarted while this was being published."}
    return state


def start_publish(owner: str, slug: str, *, sha: str = "", by: str = "",
                  secret_values: dict[str, str] | None = None) -> dict[str, Any]:
    """Publish in the background; :func:`read_publish` follows it."""
    _require(owner, slug)
    key = (owner, slug)
    if key in _PUBLISHING:
        raise ToolError("publishing", "It's already being published.", status=409)
    state = {"state": "running", "sha": sha, "by": by, "started_at": time.time(), "steps": []}
    _write_json(_publish_file(owner, slug), state)
    task = asyncio.create_task(publish(owner, slug, sha=sha, by=by, secret_values=secret_values, _claimed=True))
    _PUBLISHING[key] = task
    task.add_done_callback(lambda _t: _PUBLISHING.pop(key, None) if _PUBLISHING.get(key) is _t else None)
    return state


async def publish(owner: str, slug: str, *, sha: str = "", by: str = "",
                  secret_values: dict[str, str] | None = None, _claimed: bool = False) -> dict[str, Any]:
    """Build commit *sha* (main when empty), test it, run every example through
    the real run path as *by* would, and only then make it live. On any failure
    the version that was live stays live."""
    from captain_claw.flight_deck import crew_runner

    _require(owner, slug)
    key = (owner, slug)
    if not _claimed:
        if key in _PUBLISHING:
            raise ToolError("publishing", "It's already being published.", status=409)
        _PUBLISHING[key] = asyncio.current_task()  # type: ignore[assignment]
    state: dict[str, Any] = {"state": "running", "sha": sha, "by": by, "started_at": time.time(), "steps": []}

    def step(name: str, ok: bool, detail: str = "") -> None:
        state["steps"].append({"step": name, "ok": ok, "detail": detail, "at": time.time()})
        _write_json(_publish_file(owner, slug), state)

    repo = repo_dir(owner, slug)
    building: Path | None = None
    fresh: Path | None = None
    version = ""
    try:
        _write_json(_publish_file(owner, slug), state)
        async with _lock(owner, slug):
            rev = sha or "refs/heads/main"
            rc, full = await crew_runner._git(repo, "rev-parse", "--verify", "-q", f"{rev}^{{commit}}")
            if rc != 0 or not re.fullmatch(r"[0-9a-f]{40,64}", full.strip()):
                step("commit", False, f"No commit {rev!r} in the tool's repository.")
                raise _StepFailedError("That commit isn't in the tool's repository.")
            full = full.strip()
            version = full[:12]
            state["sha"] = full
            state["version"] = version
            step("commit", True, full)
            vdir = versions_dir(owner, slug) / version
            with tempfile.TemporaryDirectory(prefix="xtool-publish-") as tmp:
                home = Path(tmp) / "home"
                home.mkdir()
                if (vdir / READY).is_file():
                    manifest = load_manifest(vdir, slug=slug)
                    step("build", True, "already built; reusing it")
                else:
                    _rmtree(vdir)  # a half-built leftover
                    building = versions_dir(owner, slug) / f".{version}.building-{_secrets.token_hex(3)}"
                    building.mkdir(parents=True)
                    tree = await crew_runner._tree_of(str(repo), full)
                    if not tree:
                        step("copy", False, "The commit has no files.")
                        raise _StepFailedError("The commit has no files.")
                    await asyncio.to_thread(crew_runner._write_blobs, str(repo), tree, building)
                    # Flight Deck's own helper, whatever copy the repository has.
                    shutil.copyfile(_HELPER, building / HELPER_NAME)
                    _pin_pytest_root(building)
                    step("copy", True, f"{len(tree)} files")
                    try:
                        manifest = load_manifest(building, slug=slug)
                    except ToolError as exc:
                        step("check tool.json", False, exc.message)
                        raise _StepFailedError(exc.message) from None
                    step("check tool.json", True, "")
                    await _install(building, home, step)
                    await _run_tests(building, manifest, home, secret_values or {}, step)
                    (building / READY).write_text(json.dumps({"sha": full, "built_at": time.time()}), encoding="utf-8")
                    _make_readonly(building)
                    os.replace(building, vdir)
                    building = None
                    fresh = vdir
            # Every example, the way a real call runs, as the person publishing.
            failures = []
            for i, ex in enumerate(manifest.examples, 1):
                def from_tool(ref: str, _vdir: Path = vdir) -> Path:
                    target = _inside(_vdir, ref)
                    if target is None:
                        raise ToolError("files", f"Example file {ref!r} isn't inside the tool.")
                    return target

                recorded = RecordedLink(ex.get("calls"), vdir) if manifest.calls_back else None
                try:
                    res = await _run_version(owner, slug, version, vdir, manifest, by or owner, ex["input"],
                                             secret_values=secret_values or {}, resolve_file=from_tool, deliver=None,
                                             link=recorded)
                except ToolError as exc:
                    failures.append(f"example {i}: {exc.message}")
                    continue
                if recorded is not None and (recorded.problems or recorded.left_over()):
                    failures.append(f"example {i}: " + "; ".join([*recorded.problems, recorded.left_over()]).strip("; "))
                elif not res["ok"]:
                    failures.append(f"example {i}: {res['error']['message']}\n{res.get('stderr', '')}".strip())
                elif "expect" in ex and not _subset(ex["expect"], res["output"]):
                    got = json.dumps(res["output"], ensure_ascii=False)[:500]
                    failures.append(f"example {i}: the output doesn't contain what 'expect' says. Got: {got}")
            if failures:
                step("examples", False, _tail("\n\n".join(failures)))
                raise _StepFailedError("Not every example passed.")
            step("examples", True, f"{len(manifest.examples)} passed")
            old = read_live(owner, slug)
            previous = []
            if old and old["version"] != version:
                previous = [old["version"], *[v for v in old.get("previous", []) if v != version]]
            elif old:
                previous = list(old.get("previous", []))
            previous = [v for v in dict.fromkeys(previous) if _ready(owner, slug, v)][:KEEP_PREVIOUS]
            _write_json(_live_file(owner, slug), {
                "version": version, "sha": full, "published_at": time.time(), "by": by, "previous": previous,
            })
            fresh = None
            _prune(owner, slug, keep={version, *previous})
            step("live", True, version)
            state.update(state="passed", finished_at=time.time())
            for hook in AFTER_PUBLISH:
                try:
                    await hook(owner, slug, (old or {}).get("version"), version)
                except Exception as exc:  # noqa: BLE001 — a bell must not undo a publish
                    log.warning("A custom tool's after-publish step failed", owner=owner, slug=slug, error=str(exc))
    except _StepFailedError as exc:
        state.update(state="failed", error=str(exc), finished_at=time.time())
    except ToolError as exc:
        state.update(state="failed", error=exc.message, finished_at=time.time())
    except Exception as exc:  # noqa: BLE001 — the owner hears about it, the deck carries on
        log.exception("Publishing a custom tool failed", owner=owner, slug=slug)
        state.update(state="failed", error=f"Something went wrong: {type(exc).__name__}: {exc}", finished_at=time.time())
    finally:
        if building is not None:
            await asyncio.to_thread(_rmtree, building)
        if fresh is not None and state.get("state") != "passed":
            # Built here and failed its examples: never a version to go back to.
            await asyncio.to_thread(_rmtree, fresh)
        if not _claimed and _PUBLISHING.get(key) is asyncio.current_task():
            _PUBLISHING.pop(key, None)
        _write_json(_publish_file(owner, slug), state)
    return state


def _prune(owner: str, slug: str, *, keep: set[str]) -> None:
    folder = versions_dir(owner, slug)
    if not folder.is_dir():
        return
    for child in folder.iterdir():
        name = child.name
        if name in keep or name.startswith("."):
            continue
        if _IN_USE.get((owner, slug, name)):
            continue  # a call is using it; the next publish takes it
        _rmtree(child)


async def make_live(owner: str, slug: str, version: str) -> dict[str, Any]:
    """Go back (or forward) to a version that's still kept."""
    _require(owner, slug)
    async with _lock(owner, slug):
        if _ready(owner, slug, version) is None:
            raise ToolError("not_found", f"Version {version} isn't kept any more.", status=404)
        old = read_live(owner, slug)
        if old and old["version"] == version:
            return old
        marker = _read_json(versions_dir(owner, slug) / version / READY) or {}
        before = ([old["version"]] + list(old.get("previous", []))) if old else []
        previous = [v for v in dict.fromkeys(before) if v != version and _ready(owner, slug, v)][:KEEP_PREVIOUS]
        live = {"version": version, "sha": marker.get("sha", ""), "published_at": time.time(),
                "by": (old or {}).get("by", ""), "previous": previous, "made_live": True}
        _write_json(_live_file(owner, slug), live)
        return live


# ── What the page and the agent see ──────────────────────────────────────


def kept_versions(owner: str, slug: str) -> list[dict[str, Any]]:
    folder = versions_dir(owner, slug)
    out = []
    if folder.is_dir():
        for child in sorted(folder.iterdir()):
            marker = _read_json(child / READY) if _VERSION_RE.match(child.name) else None
            if isinstance(marker, dict):
                out.append({"version": child.name, "sha": marker.get("sha", ""), "built_at": marker.get("built_at")})
    out.sort(key=lambda v: v.get("built_at") or 0, reverse=True)
    return out


def summary(owner: str, slug: str) -> dict[str, Any]:
    live = read_live(owner, slug)
    manifest = None
    if live and (vdir := _ready(owner, slug, live["version"])):
        try:
            manifest = load_manifest(vdir, slug=slug)
        except ToolError:
            manifest = None
    if manifest is None:
        try:
            manifest = load_manifest(repo_dir(owner, slug), slug=slug)
        except ToolError:
            manifest = None
    pub = read_publish(owner, slug)
    return {
        "owner": owner, "slug": slug,
        "title": manifest.title if manifest else slug,
        "description": manifest.description if manifest else "",
        "live": {"version": live["version"], "published_at": live.get("published_at")} if live else None,
        "publish": {k: pub.get(k) for k in ("state", "version", "error", "started_at", "finished_at")} if pub else None,
    }


def detail(owner: str, slug: str) -> dict[str, Any]:
    _require(owner, slug)
    out = summary(owner, slug)
    live = read_live(owner, slug)
    manifest = None
    if live and (vdir := _ready(owner, slug, live["version"])):
        try:
            manifest = load_manifest(vdir, slug=slug)
        except ToolError:
            pass
    out["manifest"] = manifest.public() if manifest else None
    out["versions"] = [{**v, "live": bool(live and v["version"] == live["version"])} for v in kept_versions(owner, slug)]
    out["publish"] = read_publish(owner, slug)
    return out


def list_tools(owner: str) -> list[dict[str, Any]]:
    folder = root() / _check_owner(owner)
    if not folder.is_dir():
        return []
    return [summary(owner, child.name) for child in sorted(folder.iterdir())
            if _SLUG_RE.match(child.name) and (child / "repo" / ".git").exists()]


def secret_names(owner: str, slug: str) -> list[dict[str, str]]:
    """The secrets the live version asks for, else what the repository's tool.json lists."""
    live = read_live(owner, slug)
    for folder in ([_ready(owner, slug, live["version"])] if live else []) + [repo_dir(owner, slug)]:
        if folder is None:
            continue
        try:
            return [dict(s) for s in load_manifest(folder, slug=slug).secrets]
        except ToolError:
            continue
    return []


def secret_key(owner: str, slug: str, name: str) -> str:
    """Where a person's value for one of a tool's secrets is kept (their settings, sealed)."""
    return f"xtool:{_check_owner(owner)}/{check_slug(slug)}:{name}"
