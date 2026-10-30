"""Every tool call this agent makes, reported to Flight Deck's tool-call register.

Flight Deck keeps a register of agents' tool calls (agent, time, tool,
parameters, what started the call and how it ended) that the owner and the
deck's admins read (``flight_deck/tool_audit_routes.py``). Only the agent sees
every call: a Telegram turn, a scheduled job or a Bat worker never passes
through Flight Deck. So :class:`~captain_claw.tools.registry.ToolRegistry`
records each call here — at its start, before any policy check, so a refused
call is recorded too — and this module ships the records.

* **Masked first.** Parameters are masked before they touch disk
  (:func:`mask_params`): secret-named fields, key- and token-shaped values,
  card numbers and the password in a URL are replaced; a long value keeps its
  start, its length and its sha256, never the rest.
* **Spooled.** Each record goes to a small SQLite file next to the agent's
  sessions (``tool_audit_spool.db``) before it is sent, so a Flight Deck
  restart or a crash loses nothing; a call that never finished is reported as
  ``interrupted`` by the next process.
* **Numbered.** Records carry this process's ``boot`` id and a ``seq`` counted
  from 1, so Flight Deck can tell a missing record from a quiet agent.
* **Never in the way.** Recording and shipping never raise into a tool call;
  a record that can't be written is logged and dropped (and shows up as a gap).
* **Tied to its turn.** A call carries the ``turn_id`` of the turn it was made
  in (``turn_register.py``), and an ok send, draft or calendar call the ids
  its result named (:func:`result_refs`: a Gmail message, a WhatsApp Web
  message, a calendar event …) — ids only, never a body.

Active only in an agent Flight Deck spawned (``FD_URL``, ``FD_AGENT_SLUG`` and
the agent's own auth token): Flight Deck identifies the caller by that token,
never by anything in the records. The spool and the shipper here carry the
turn register's records too (:func:`open_spool_db`, :func:`ship_loop`).

The agent reads its own calls back (:func:`read_own`, the ``activity`` tool and
:func:`reflection_digest`) the way ``tool_audit_digest`` shapes them: its
owner's turns only, read-only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

INGEST_PATH = "/fd/tool-audit/agent/calls"

OUTCOMES = ("ok", "error", "blocked", "timeout", "aborted", "interrupted")
MASK = "[masked]"
INTERRUPTED_REASON = "The agent stopped before this call finished"

_MAX_STR = 2000           # longer strings keep their start, length and hash
_HEAD = 300
_MAX_ITEMS = 50           # list items / dict keys kept per level
_MAX_DEPTH = 6
_MAX_PARAMS_BYTES = 16_000
_MAX_REASON = 500

_FLUSH_S = 2.0
_START_GRACE_S = 15.0     # a call still running this long is reported as started
_BATCH = 200
_SPOOL_CAP = 50_000
_BACKOFF_MAX_S = 300.0

# ── Masking ───────────────────────────────────────────────────────────

# A key whose name says it holds a secret. Matched on the key's words
# (snake_case, kebab-case and camelCase split), so "author" or "tokens_used"
# don't count but "api_key", "accessToken" and "client-secret" do.
_SECRET_WORDS = frozenset({
    "password", "passwd", "pwd", "passphrase", "secret", "token", "apikey",
    "authorization", "auth", "cookie", "cookies", "credential", "credentials",
    "otp", "totp", "pin", "cvv", "cvc", "privatekey", "pan",
})
_SECRET_PAIRS = frozenset({("api", "key"), ("private", "key"), ("access", "key"),
                           ("card", "number"), ("session", "key"), ("client", "secret")})
_NOT_SECRET_KEYS = frozenset({"max_tokens", "tokens", "token_count", "page_token", "next_page_token"})

_SECRET_VALUE_RES = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)", re.S),
    re.compile(r"\b(?:sk|pk|rk)-(?:ant-|proj-|live-|test-)?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"\bya29\.[0-9A-Za-z_\-]{20,}"),
    re.compile(r"\b(?:cc_pat|cc_link)_[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=\-]{12,}"),
]
_URL_PASSWORD_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://[^/\s:@]+):([^@\s/]+)@")
_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:access_token|token|api_key|apikey|key|secret|password|pwd|sig|signature|auth)=)[^&#\s]+")


def _key_words(key: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return [w for w in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if w]


def secret_key(key: str) -> bool:
    """Does a parameter's name say it holds a secret?"""
    if str(key).lower() in _NOT_SECRET_KEYS:
        return False
    words = _key_words(key)
    if any(w in _SECRET_WORDS for w in words):
        return True
    if "".join(words) in _SECRET_WORDS:
        return True
    return any((a, b) in _SECRET_PAIRS for a, b in zip(words, words[1:]))


def mask_text(text: str) -> str:
    """*text* with key- and token-shaped values, URL passwords, secret query
    parameters and card numbers replaced."""
    out = str(text)
    for rx in _SECRET_VALUE_RES:
        out = rx.sub(MASK, out)
    out = _URL_PASSWORD_RE.sub(lambda m: f"{m.group(1)}:{MASK}@", out)
    out = _QUERY_SECRET_RE.sub(lambda m: f"{m.group(1)}{MASK}", out)
    if any(ch.isdigit() for ch in out):
        try:
            from captain_claw import card_scrub

            out, _found = card_scrub.scrub(out)
        except Exception:
            pass
    return out


def _mask_value(value: Any, depth: int) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return {"_bytes": len(value), "_sha256": hashlib.sha256(bytes(value)).hexdigest()}
    if isinstance(value, str):
        if len(value) > _MAX_STR:
            return {
                "_head": mask_text(value[:_HEAD]),
                "_len": len(value),
                "_sha256": hashlib.sha256(value.encode("utf-8", "replace")).hexdigest(),
            }
        return mask_text(value)
    if depth >= _MAX_DEPTH:
        return "[…]"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= _MAX_ITEMS:
                out["_more_keys"] = len(value) - _MAX_ITEMS
                break
            key = str(k)
            out[key] = MASK if secret_key(key) and v not in (None, "") else _mask_value(v, depth + 1)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        out_list = [_mask_value(v, depth + 1) for v in items[:_MAX_ITEMS]]
        if len(items) > _MAX_ITEMS:
            out_list.append(f"[… {len(items) - _MAX_ITEMS} more]")
        return out_list
    return mask_text(repr(value)[:_HEAD])


def mask_params(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """A tool call's own arguments, safe to store: the runtime's ``_``-keys
    (``_agent``, ``_session`` …) dropped, then :func:`mask_tree`."""
    if not isinstance(arguments, dict):
        return {}
    return mask_tree({k: v for k, v in arguments.items() if not str(k).startswith("_")})


def mask_tree(params: dict[str, Any] | None) -> dict[str, Any]:
    """*params* with secrets masked, long values cut and the whole capped at
    16 KB. Idempotent: Flight Deck runs it again on what an agent sends."""
    if not isinstance(params, dict):
        return {}
    out = _mask_value(params, 0)
    if not isinstance(out, dict):
        return {}
    raw = json.dumps(out, ensure_ascii=False, default=str)
    if len(raw.encode("utf-8")) > _MAX_PARAMS_BYTES:
        return {"_truncated": True, "_len": len(raw),
                "_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "_head": raw[:_HEAD * 4]}
    return out


# ── What an ok call made ──────────────────────────────────────────────

# The ids a send, draft or calendar result may name, as Flight Deck stores them.
REF_KEYS = ("gmail_message_id", "gmail_thread_id", "gmail_draft_id", "wa_id", "waweb_message_id",
            "calendar_event_id", "telegram_message_id")
_REF_SCAN = 4000          # a result's head is where its ids are printed
_REF_ID_RE = re.compile(r"^[A-Za-z0-9_\-.=+@:/<>]{1,256}$")

# Per tool: the actions whose result names what was made ("" when the call
# gave none), and the lines that name it.
_REF_ACTIONS: dict[str, frozenset[str]] = {
    "google_mail": frozenset({"create_draft", "update_draft", "send", "send_draft", "reply", "forward"}),
    "whatsapp_web": frozenset({"send"}),
    "whatsapp_send_file": frozenset({"send", ""}),
    "google_calendar": frozenset({"create_event", "update_event"}),
    "microsoft_calendar": frozenset({"create_event", "update_event"}),
}
_GMAIL_REFS = (
    ("gmail_draft_id", re.compile(r"^[ \t]*Draft ID:[ \t]*(\S+)", re.M)),
    ("gmail_message_id", re.compile(r"^[ \t]*Message ID:[ \t]*(\S+)", re.M)),
    ("gmail_thread_id", re.compile(r"^[ \t]*(?:Thread ID|Threaded under):[ \t]*(\S+)", re.M)),
)
_REF_RES: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    "google_mail": _GMAIL_REFS,
    "whatsapp_web": (("waweb_message_id", re.compile(r"\A[^\n]*\(message id ([^)\s]+)\)")),),
    "whatsapp_send_file": (("wa_id", re.compile(r"\ASent '[^\n]*' to WhatsApp \+?(\d{6,20}) as a")),),
    "google_calendar": (("calendar_event_id", re.compile(r"^[ \t]*ID:[ \t]*(\S+)", re.M)),),
    "microsoft_calendar": (("calendar_event_id", re.compile(r"^[ \t]*ID:[ \t]*(\S+)", re.M)),),
}


def clean_refs(refs: Any) -> dict[str, str]:
    """*refs* with only the known keys and id-shaped values kept."""
    if not isinstance(refs, dict):
        return {}
    out: dict[str, str] = {}
    for key in REF_KEYS:
        value = refs.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            text = str(value).strip()
            if text and text != "?" and _REF_ID_RE.match(text):
                out[key] = text
    return out


def result_refs(tool: str, action: str, result: Any) -> dict[str, str]:
    """The ids an ok *tool* call's result names ({} for any other call).
    Never raises."""
    try:
        if str(action or "") not in _REF_ACTIONS.get(str(tool or ""), frozenset()):
            return {}
        content = getattr(result, "content", "")
        if not isinstance(content, str) or not content:
            return {}
        head = content[:_REF_SCAN]
        found = {key: m.group(1) for key, rx in _REF_RES.get(str(tool), ()) if (m := rx.search(head))}
        return clean_refs(found)
    except Exception:
        return {}


# ── What started the call ─────────────────────────────────────────────


def _call_context(arguments: dict[str, Any], member: bool) -> dict[str, str]:
    """Where the turn came from (its source) and who was speaking.

    An automation bound to this task (a scheduled job, a flow, Autonomous
    Work's ``run_tool`` …) says it first: the agent's ``_turn_source`` is the
    last *turn's*, and a call made outside a turn would otherwise carry it."""
    agent = arguments.get("_agent") if isinstance(arguments, dict) else None
    source, label = "", ""
    try:
        from captain_claw import mail_authority, turn_source

        bound = mail_authority.bound_authority()
        if bound is not None and bound.mode == "automated":
            source = turn_source.from_automation(bound.kind) or "agent"
            label = turn_source.label_for(source)
    except Exception:
        pass
    turn = getattr(agent, "_turn_source", None) if agent is not None else None
    if not source and isinstance(turn, tuple) and len(turn) == 2:
        source, label = str(turn[0] or ""), str(turn[1] or "")
    if not source:
        try:
            from captain_claw import mail_authority, turn_source

            auth = mail_authority.current()
            if auth.mode == "automated":
                source = turn_source.from_automation(auth.kind) or "agent"
                label = turn_source.label_for(source)
        except Exception:
            pass
    ctx = {"source": source[:64], "source_label": label[:80],
           "speaker_id": "", "speaker_name": "", "speaker_ref": ""}
    try:
        from captain_claw import speaker

        p = speaker.principal_for(agent)
    except Exception:
        p = None
    if p is not None:
        ctx.update(speaker_id=str(p.speaker_id or "")[:128],
                   speaker_name=str(p.display_name or "")[:80],
                   speaker_ref=str(p.agent_ref or "")[:160])
    elif member:
        ctx["speaker_name"] = "Member"
    if (p is not None or member) and not ctx["source"]:
        ctx["source"], ctx["source_label"] = "member", "Shared-agent members"
    # Made while the turn holds members' private data (it read it, or the
    # conversation still carries it): its parameters may restate that data,
    # so the agent never reads this call back (tool_audit_digest).
    try:
        from captain_claw import member_privacy

        ctx["private"] = 1 if agent is not None and member_privacy.output_level(agent) else 0
    except Exception:
        ctx["private"] = 1
    return ctx


# ── Spool ─────────────────────────────────────────────────────────────

_SPOOL_DDL = """
CREATE TABLE IF NOT EXISTS calls (
    boot        TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    started_at  REAL NOT NULL,
    body        TEXT NOT NULL,
    ended       INTEGER NOT NULL DEFAULT 0,
    outcome     TEXT NOT NULL DEFAULT '',
    reason      TEXT NOT NULL DEFAULT '',
    duration_ms INTEGER,
    ended_at    REAL,
    sent_start  INTEGER NOT NULL DEFAULT 0,
    refs        TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (boot, seq)
);
"""


def open_spool_db(path: Path, ddl: str) -> sqlite3.Connection:
    """A spool's SQLite file (0600, WAL), its tables made from *ddl*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(ddl)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return conn


class Spool:
    """The records not yet accepted by Flight Deck, in a SQLite file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._inserts = 0

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = open_spool_db(self.path, _SPOOL_DDL)
            # A spool an older agent left (records still waiting in it).
            have = {str(r[1]) for r in conn.execute("PRAGMA table_info(calls)").fetchall()}
            if "refs" not in have:
                conn.execute("ALTER TABLE calls ADD COLUMN refs TEXT NOT NULL DEFAULT ''")
                conn.commit()
            self._conn = conn
        return self._conn

    def add(self, boot: str, seq: int, started_at: float, body: dict[str, Any]) -> None:
        with self._lock:
            db = self._db()
            db.execute("INSERT OR REPLACE INTO calls (boot, seq, started_at, body) VALUES (?, ?, ?, ?)",
                       (boot, seq, started_at, json.dumps(body, ensure_ascii=False, default=str)))
            db.commit()
            self._inserts += 1
            if self._inserts % 500 == 0:
                self._cap(db)

    def _cap(self, db: sqlite3.Connection) -> None:
        (n,) = db.execute("SELECT count(*) FROM calls").fetchone()
        if n > _SPOOL_CAP:
            db.execute("DELETE FROM calls WHERE rowid IN (SELECT rowid FROM calls "
                       "ORDER BY started_at LIMIT ?)", (n - _SPOOL_CAP,))
            db.commit()
            log.warning("Tool-call register spool full: dropped the oldest records",
                        dropped=n - _SPOOL_CAP)

    def end(self, boot: str, seq: int, outcome: str, reason: str, duration_ms: int, ended_at: float,
            refs: dict[str, str] | None = None) -> None:
        with self._lock:
            db = self._db()
            db.execute("UPDATE calls SET ended=1, outcome=?, reason=?, duration_ms=?, ended_at=?, refs=? "
                       "WHERE boot=? AND seq=?",
                       (outcome, reason, duration_ms, ended_at, json.dumps(refs) if refs else "", boot, seq))
            db.commit()

    def pending(self) -> bool:
        with self._lock:
            return self._db().execute("SELECT 1 FROM calls LIMIT 1").fetchone() is not None

    def batch(self, boot: str, now: float) -> tuple[list[dict[str, Any]], list[tuple[str, int]], list[tuple[str, int]]]:
        """``(records to send, keys to delete once sent, keys sent as started)``."""
        with self._lock:
            rows = self._db().execute(
                "SELECT boot, seq, started_at, body, ended, outcome, reason, duration_ms, ended_at, sent_start, "
                "refs FROM calls WHERE ended=1 OR boot != ? OR (started_at < ? AND sent_start=0) "
                "ORDER BY started_at, seq LIMIT ?",
                (boot, now - _START_GRACE_S, _BATCH)).fetchall()
        records: list[dict[str, Any]] = []
        done: list[tuple[str, int]] = []
        started: list[tuple[str, int]] = []
        for r_boot, seq, started_at, body, ended, outcome, reason, duration_ms, ended_at, sent_start, refs in rows:
            key = (r_boot, int(seq))
            rec: dict[str, Any] = {"boot": r_boot, "seq": int(seq), "started_at": float(started_at)}
            if not ended and r_boot != boot:
                ended, outcome, reason, duration_ms, ended_at = 1, "interrupted", INTERRUPTED_REASON, None, None
            if ended:
                rec.update(phase="end" if sent_start else "full", outcome=outcome or "error",
                           reason=reason or "", duration_ms=duration_ms, ended_at=ended_at)
                if refs:
                    try:
                        rec["refs"] = json.loads(refs)
                    except ValueError:
                        pass
                done.append(key)
            else:
                rec["phase"] = "start"
                started.append(key)
            if rec["phase"] != "end":
                try:
                    rec.update(json.loads(body))
                except ValueError:
                    rec.update(tool="?", params={})
            records.append(rec)
        return records, done, started

    def settle(self, done: list[tuple[str, int]], started: list[tuple[str, int]]) -> None:
        with self._lock:
            db = self._db()
            db.executemany("DELETE FROM calls WHERE boot=? AND seq=?", done)
            db.executemany("UPDATE calls SET sent_start=1 WHERE boot=? AND seq=?", started)
            db.commit()


# ── Recording ─────────────────────────────────────────────────────────


@dataclass
class Handle:
    boot: str
    seq: int
    t0: float
    finished: bool = False
    tool: str = ""
    action: str = ""
    turn: Any = None          # the turn_register.Turn it was made in, if any


# Outcomes that count as a turn's failed tool call (a refusal is the guards
# doing their job, not the tool failing).
_FAILED = frozenset({"error", "timeout", "aborted"})


_BOOT = secrets.token_hex(8)
_SEQ = 0
_SEQ_LOCK = threading.Lock()
_SPOOL: Spool | None = None
_SPOOL_FAILED = False
_SHIPPERS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Task] = weakref.WeakKeyDictionary()


def enabled() -> bool:
    """Only an agent Flight Deck spawned, that can prove which one it is."""
    if (os.environ.get("CLAW_TOOL_AUDIT") or "").strip().lower() in ("0", "off", "false", "no"):
        return False
    from captain_claw.fd_client import _agent_web_auth_token, flight_deck_base, flight_deck_slug

    return bool(flight_deck_base() and flight_deck_slug() and _agent_web_auth_token())


def _spool_path() -> Path:
    override = (os.environ.get("CLAW_TOOL_AUDIT_SPOOL") or "").strip()
    if override:
        return Path(override).expanduser()
    from captain_claw.config import get_config

    return Path(get_config().session.path).expanduser().parent / "tool_audit_spool.db"


def _spool() -> Spool | None:
    global _SPOOL, _SPOOL_FAILED
    if _SPOOL is None and not _SPOOL_FAILED:
        try:
            _SPOOL = Spool(_spool_path())
        except Exception as exc:
            _SPOOL_FAILED = True
            log.warning("Tool-call register disabled: no spool", error=str(exc))
    return _SPOOL


def _next_seq() -> int:
    global _SEQ
    with _SEQ_LOCK:
        _SEQ += 1
        return _SEQ


def start(name: str, arguments: dict[str, Any] | None, *, session_id: str | None = None,
          member: bool = False) -> Handle | None:
    """Record that *name* is being called; None when recording is off.
    Never raises."""
    try:
        if not enabled():
            return None
        from captain_claw import turn_register

        turn = turn_register.current()
        if turn is not None:
            turn.tool_calls += 1
        spool = _spool()
        if spool is None:
            return None
        args = arguments if isinstance(arguments, dict) else {}
        now = time.time()
        action = args.get("action")
        handle = Handle(_BOOT, _next_seq(), time.monotonic(), tool=str(name or ""),
                        action=action.strip().lower() if isinstance(action, str) else "", turn=turn)
        body = {"tool": str(name or "")[:128], "params": mask_params(args),
                "session_id": str(session_id or "")[:128], **_call_context(args, member),
                "turn_id": turn.turn_id if turn is not None else ""}
        spool.add(handle.boot, handle.seq, now, body)
        _ensure_shipper()
        return handle
    except Exception as exc:
        log.warning("Tool call not recorded", tool=name, error=str(exc))
        return None


def finish(handle: Handle | None, outcome: str, reason: str = "", result: Any = None) -> None:
    """Record how a started call ended (an ok call's *result* gives its
    :func:`result_refs`). Never raises; a second finish is ignored."""
    if handle is None or handle.finished:
        return
    handle.finished = True
    try:
        spool = _spool()
        if spool is None:
            return
        outcome = outcome if outcome in OUTCOMES else "error"
        if handle.turn is not None and outcome in _FAILED:
            handle.turn.failed_tool_calls += 1
        refs = result_refs(handle.tool, handle.action, result) if outcome == "ok" and result is not None else {}
        duration_ms = int(max(0.0, time.monotonic() - handle.t0) * 1000)
        spool.end(handle.boot, handle.seq, outcome, mask_text(str(reason or ""))[:_MAX_REASON],
                  duration_ms, time.time(), refs)
        _ensure_shipper()
    except Exception as exc:
        log.warning("Tool call end not recorded", error=str(exc))


def refused(name: str, arguments: dict[str, Any] | None, reason: str, *,
            session_id: str | None = None, member: bool = False) -> None:
    """A call stopped before it reached the registry (the agent's guards)."""
    finish(start(name, arguments, session_id=session_id, member=member), "blocked", reason)


# ── Shipping ──────────────────────────────────────────────────────────


Poster = Callable[[Any, list[dict[str, Any]]], Awaitable[int]]


def ensure_shipper(shippers: weakref.WeakKeyDictionary, run: Callable[[], Any]) -> None:
    """Start ``run()`` on the running loop unless it is already shipping
    there (one shipper per loop, kept in *shippers*)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # no loop here: the next call made on one ships these
    task = shippers.get(loop)
    if task is None or task.done():
        shippers[loop] = loop.create_task(run())


def _ensure_shipper() -> None:
    ensure_shipper(_SHIPPERS, _ship_loop)


async def _post(client: Any, records: list[dict[str, Any]]) -> int:
    resp = await client.post(INGEST_PATH, json={"records": records})
    return int(resp.status_code)


async def ship_loop(name: str, get_spool: Callable[[], Any], post: Poster) -> None:
    """Send a spool's ready records with *post* until none is left: every
    ``_FLUSH_S``, backing off while Flight Deck is away or refuses, dropping
    a batch it can't read. *get_spool()* gives an object with ``batch(boot,
    now) -> (records, done, started)``, ``settle(done, started)`` and
    ``pending()``; *name* is what the log lines call the register."""
    from captain_claw.fd_client import FDClient

    client = FDClient(timeout=10.0)
    backoff = _FLUSH_S
    try:
        while True:
            await asyncio.sleep(backoff)
            spool = get_spool()
            if spool is None:
                return
            try:
                records, done, started = spool.batch(_BOOT, time.time())
            except Exception as exc:
                log.warning(f"{name} spool unreadable", error=str(exc))
                return
            if not records:
                if not spool.pending():
                    return
                backoff = _FLUSH_S
                continue
            try:
                status = await post(client, records)
            except Exception as exc:
                log.debug(f"{name}: Flight Deck unreachable", error=type(exc).__name__)
                backoff = min(_BACKOFF_MAX_S, backoff * 2)
                continue
            if status == 200:
                spool.settle(done, started)
                backoff = _FLUSH_S if len(records) < _BATCH else 0.05
            elif status in (400, 413, 422):
                # Flight Deck can't read these: drop them (they show as a gap there).
                log.warning(f"{name} refused a batch", status=status, records=len(records))
                spool.settle(done + started, [])
                backoff = _FLUSH_S
            else:
                log.debug(f"{name}: report refused", status=status)
                backoff = min(_BACKOFF_MAX_S, max(backoff * 2, 30.0))
    finally:
        try:
            await client.close()
        except Exception:
            pass


async def _ship_loop() -> None:
    await ship_loop("Tool-call register", _spool, lambda client, records: _post(client, records))


async def flush_spool(spool: Any, post: Poster) -> None:
    """Ship one batch of what *spool* has ready, now (tests, shutdown)."""
    from captain_claw.fd_client import FDClient

    if spool is None:
        return
    records, done, started = spool.batch(_BOOT, time.time())
    if not records:
        return
    client = FDClient(timeout=10.0)
    try:
        if await post(client, records) == 200:
            spool.settle(done, started)
    finally:
        await client.close()


async def flush_now() -> None:
    """Ship whatever is ready (tests, shutdown)."""
    await flush_spool(_spool(), _post)


# ── Reading back ──────────────────────────────────────────────────────

MINE_PATH = "/fd/tool-audit/agent/mine"
DIGEST_PATH = "/fd/tool-audit/agent/digest"


class ReadError(RuntimeError):
    """Flight Deck's register couldn't be read."""


async def read_own(path: str, params: dict[str, Any]) -> dict[str, Any]:
    """GET one of Flight Deck's routes for this agent's own calls. What is
    ready to ship goes first, so a call that just ended is in the answer."""
    from captain_claw.fd_client import FDClient

    if not enabled():
        raise ReadError("This agent's tool calls are recorded only when Flight Deck runs it.")
    try:
        await flush_now()
    except Exception:
        pass
    client = FDClient(timeout=15.0)
    try:
        resp = await client.get(path, params={k: v for k, v in params.items() if v not in (None, "")})
    except Exception as exc:
        raise ReadError(f"Flight Deck is unreachable ({type(exc).__name__}).") from exc
    finally:
        await client.close()
    if resp.status_code != 200:
        raise ReadError(f"Flight Deck refused the request (HTTP {resp.status_code}).")
    try:
        return resp.json()
    except ValueError as exc:
        raise ReadError("Flight Deck sent something unreadable.") from exc


async def reflection_digest(since: float | None) -> str:
    """The digest of this agent's calls since *since* as text, for a
    reflection; empty when there is none to give."""
    from captain_claw import tool_audit_digest

    if not enabled():
        return ""
    try:
        d = await read_own(DIGEST_PATH, {"since": since})
    except ReadError as exc:
        log.debug("Tool-call digest unavailable", error=str(exc))
        return ""
    return tool_audit_digest.format_digest(d, patterns_only=True) if d.get("total") else ""
