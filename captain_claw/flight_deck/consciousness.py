"""Flight Deck consciousness — a free-running heartbeat that quietly observes
everything happening across one user's agents and keeps a private inner life:
thoughts, occasional dreams, and standing intentions.

Why this lives in Flight Deck and not in an agent
-------------------------------------------------
A single agent only knows its own sessions. The consciousness is *one self per
user* that watches across ALL of that user's agents — so it has to sit at the
tenant boundary, which is Flight Deck. It reads from each agent over the same
HTTP surface FD already uses (``GET /api/sessions``), and it *thinks* by
borrowing one of the user's own agents as its brain (``RemoteLLMProvider`` →
``POST /api/llm/complete``). That keeps it multi-tenant by construction: it
only ever sees, and only ever uses, the logged-in user's own agents and keys.

Read-only, both ways
--------------------
From the user's perspective the consciousness is observe-only: there is no way
to talk *to* it (the Observatory UI has no input box), and it never writes back
into agent sessions or acts on the user's behalf. It surfaces; the user (or a
session) pulls. The only writes it makes are to its own ``consciousness.db``.

The heartbeat
-------------
Every pulse:

  1. Enumerate the user's running agents (process registry, filtered by owner).
  2. Pull a *delta* since the last beat — new sessions / new messages — by
     diffing each agent's session list against a stored cursor.
  3. If the delta is small or nonexistent → advance the cursor and stop.
     **No model is called.** Most heartbeats cost nothing.
  4. Otherwise → read a little of what actually changed, reflect through the
     user's own agent, and append a thought (and maybe a dream + refreshed
     intentions) to the journal.

A manual *nudge* (Observatory button) forces one pulse regardless of delta.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

_log = logging.getLogger(__name__)


# ── Tunables (env-overridable) ────────────────────────────────────────

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


# How often the free-running loop wakes to consider a pulse.
PULSE_INTERVAL_SECONDS = max(30.0, _env_float("CONSCIOUSNESS_PULSE_SECONDS", 180.0))
# Minimum new messages (or any new session) before a pulse is "worth a thought".
# Below this the beat is silent and free.
MIN_NEW_MESSAGES = max(1, _env_int("CONSCIOUSNESS_MIN_NEW_MESSAGES", 3))
# Per-agent HTTP timeout when sensing / thinking.
SENSE_TIMEOUT = _env_float("CONSCIOUSNESS_SENSE_TIMEOUT", 8.0)
# How many recently-touched sessions to actually read when reflecting.
REFLECT_MAX_SESSIONS = max(1, _env_int("CONSCIOUSNESS_REFLECT_SESSIONS", 4))
REFLECT_MAX_MESSAGES = max(1, _env_int("CONSCIOUSNESS_REFLECT_MESSAGES", 6))


def _auth_enabled() -> bool:
    return os.environ.get("FD_AUTH_ENABLED", "true").lower() in ("true", "1", "yes")


def _norm_user(user_id: str | None) -> str:
    """Normalize a user id. Empty (auth disabled / internal) → 'local'."""
    uid = (user_id or "").strip()
    return uid or "local"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Store ─────────────────────────────────────────────────────────────

def _db_path() -> Path:
    base = os.environ.get("FD_DATA_DIR", "").strip()
    if base:
        return Path(base).expanduser().resolve() / "consciousness.db"
    return Path("~/.captain-claw/consciousness.db").expanduser()


class ConsciousnessStore:
    """SQLite-backed inner life. Sync sqlite3 + a lock, mirroring fd_scheduler:
    DB ops are short; the network I/O around them is async."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or _db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._ensure_db()

    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._conn = conn
        return self._conn

    def _ensure_db(self) -> None:
        with self._lock:
            self._c().executescript(
                """
                -- One cursor + counters per user.
                CREATE TABLE IF NOT EXISTS consciousness_state (
                    user_id      TEXT PRIMARY KEY,
                    cursor       TEXT NOT NULL DEFAULT '{}',
                    pulse_count  INTEGER NOT NULL DEFAULT 0,
                    thought_count INTEGER NOT NULL DEFAULT 0,
                    last_pulse_at  REAL,
                    last_thought_at REAL,
                    narrator_slug TEXT NOT NULL DEFAULT '',
                    updated_at   TEXT NOT NULL DEFAULT ''
                );
                -- Append-only stream of inner life.
                CREATE TABLE IF NOT EXISTS consciousness_journal (
                    id          TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    kind        TEXT NOT NULL DEFAULT 'thought',  -- thought|dream|observation
                    content     TEXT NOT NULL,
                    mood        TEXT NOT NULL DEFAULT '',
                    salience    INTEGER NOT NULL DEFAULT 5,
                    agents      TEXT NOT NULL DEFAULT '[]',       -- json: agent slugs observed
                    delta       TEXT NOT NULL DEFAULT '',         -- short note on what changed
                    author      TEXT NOT NULL DEFAULT '',         -- agent that did the thinking
                    created_at  TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_journal_user
                    ON consciousness_journal(user_id, created_at DESC);
                -- Current standing intentions (curiosities / things to watch).
                CREATE TABLE IF NOT EXISTS consciousness_intentions (
                    id          TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'active',   -- active|fading
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_intentions_user
                    ON consciousness_intentions(user_id, status);
                """
            )
            # Migrate older DBs in place (columns added after first ship).
            for table, col in (
                ("consciousness_journal", "author"),
                ("consciousness_state", "narrator_slug"),
            ):
                try:
                    self._c().execute(
                        f"ALTER TABLE {table} ADD COLUMN {col} TEXT NOT NULL DEFAULT ''"
                    )
                except Exception:
                    pass  # column already exists
            self._c().commit()

    # ── state / cursor ────────────────────────────────────────────────

    def get_state(self, user_id: str) -> dict[str, Any]:
        uid = _norm_user(user_id)
        with self._lock:
            r = self._c().execute(
                "SELECT * FROM consciousness_state WHERE user_id = ?", (uid,)
            ).fetchone()
            if not r:
                return {"user_id": uid, "cursor": {}, "pulse_count": 0,
                        "thought_count": 0, "last_pulse_at": None,
                        "last_thought_at": None}
            d = dict(r)
            try:
                d["cursor"] = json.loads(d.get("cursor") or "{}")
            except (ValueError, TypeError):
                d["cursor"] = {}
            return d

    def save_state(self, user_id: str, *, cursor: dict[str, Any],
                   pulse_inc: int = 0, thought_inc: int = 0,
                   touched_thought: bool = False) -> None:
        uid = _norm_user(user_id)
        now = time.time()
        with self._lock:
            conn = self._c()
            conn.execute(
                """
                INSERT INTO consciousness_state
                    (user_id, cursor, pulse_count, thought_count,
                     last_pulse_at, last_thought_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    cursor = excluded.cursor,
                    pulse_count = consciousness_state.pulse_count + ?,
                    thought_count = consciousness_state.thought_count + ?,
                    last_pulse_at = ?,
                    last_thought_at = CASE WHEN ? THEN ? ELSE consciousness_state.last_thought_at END,
                    updated_at = excluded.updated_at
                """,
                (uid, json.dumps(cursor), pulse_inc, thought_inc, now,
                 (now if touched_thought else None), _utcnow_iso(),
                 pulse_inc, thought_inc, now, 1 if touched_thought else 0, now),
            )
            conn.commit()

    # ── narrator preference ───────────────────────────────────────────

    def get_narrator(self, user_id: str) -> str:
        """The user's preferred thinking agent (slug), or '' for auto-pick."""
        uid = _norm_user(user_id)
        with self._lock:
            r = self._c().execute(
                "SELECT narrator_slug FROM consciousness_state WHERE user_id = ?", (uid,),
            ).fetchone()
            return (r["narrator_slug"] if r else "") or ""

    def set_narrator(self, user_id: str, slug: str) -> None:
        uid = _norm_user(user_id)
        slug = (slug or "").strip()
        with self._lock:
            conn = self._c()
            conn.execute(
                """
                INSERT INTO consciousness_state (user_id, narrator_slug, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    narrator_slug = excluded.narrator_slug,
                    updated_at = excluded.updated_at
                """,
                (uid, slug, _utcnow_iso()),
            )
            conn.commit()

    # ── journal ───────────────────────────────────────────────────────

    def add_journal(self, user_id: str, *, kind: str, content: str,
                    mood: str = "", salience: int = 5,
                    agents: list[str] | None = None, delta: str = "",
                    author: str = "") -> dict[str, Any]:
        uid = _norm_user(user_id)
        row = {
            "id": "cj_" + secrets.token_hex(6),
            "user_id": uid,
            "kind": kind,
            "content": content.strip(),
            "mood": (mood or "").strip()[:40],
            "salience": max(1, min(10, int(salience or 5))),
            "agents": json.dumps(agents or []),
            "delta": (delta or "").strip()[:280],
            "author": (author or "").strip()[:80],
            "created_at": _utcnow_iso(),
        }
        with self._lock:
            conn = self._c()
            conn.execute(
                """INSERT INTO consciousness_journal
                   (id, user_id, kind, content, mood, salience, agents, delta, author, created_at)
                   VALUES (:id, :user_id, :kind, :content, :mood, :salience,
                           :agents, :delta, :author, :created_at)""",
                row,
            )
            conn.commit()
        return row

    def list_journal(self, user_id: str, *, limit: int = 100,
                     before: str | None = None) -> list[dict[str, Any]]:
        uid = _norm_user(user_id)
        with self._lock:
            if before:
                rows = self._c().execute(
                    """SELECT * FROM consciousness_journal
                       WHERE user_id = ? AND created_at < ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (uid, before, limit),
                ).fetchall()
            else:
                rows = self._c().execute(
                    """SELECT * FROM consciousness_journal
                       WHERE user_id = ? ORDER BY created_at DESC LIMIT ?""",
                    (uid, limit),
                ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["agents"] = json.loads(d.get("agents") or "[]")
            except (ValueError, TypeError):
                d["agents"] = []
            out.append(d)
        return out

    # ── intentions ────────────────────────────────────────────────────

    def list_intentions(self, user_id: str) -> list[dict[str, Any]]:
        uid = _norm_user(user_id)
        with self._lock:
            rows = self._c().execute(
                """SELECT * FROM consciousness_intentions
                   WHERE user_id = ? AND status = 'active'
                   ORDER BY updated_at DESC""",
                (uid,),
            ).fetchall()
            return [dict(r) for r in rows]

    def replace_intentions(self, user_id: str, intentions: list[str]) -> None:
        """Reconcile standing intentions to the given list. Intentions that
        survive keep their id/created_at; dropped ones fade; new ones are born."""
        uid = _norm_user(user_id)
        wanted = [i.strip() for i in intentions if i and i.strip()][:8]
        now = _utcnow_iso()
        with self._lock:
            conn = self._c()
            existing = {
                r["content"].strip().lower(): dict(r)
                for r in conn.execute(
                    "SELECT * FROM consciousness_intentions WHERE user_id = ? AND status = 'active'",
                    (uid,),
                ).fetchall()
            }
            keep_keys = set()
            for content in wanted:
                key = content.lower()
                keep_keys.add(key)
                if key in existing:
                    conn.execute(
                        "UPDATE consciousness_intentions SET updated_at = ? WHERE id = ?",
                        (now, existing[key]["id"]),
                    )
                else:
                    conn.execute(
                        """INSERT INTO consciousness_intentions
                           (id, user_id, content, status, created_at, updated_at)
                           VALUES (?, ?, ?, 'active', ?, ?)""",
                        ("ci_" + secrets.token_hex(6), uid, content, now, now),
                    )
            # Fade anything no longer intended.
            for key, row in existing.items():
                if key not in keep_keys:
                    conn.execute(
                        "UPDATE consciousness_intentions SET status = 'fading', updated_at = ? WHERE id = ?",
                        (now, row["id"]),
                    )
            conn.commit()


_STORE: ConsciousnessStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> ConsciousnessStore:
    global _STORE
    if _STORE is not None:
        return _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = ConsciousnessStore()
        return _STORE


# ── Senses: enumerate agents + pull deltas (read-only) ────────────────

def _user_agents(user_id: str) -> list[dict[str, Any]]:
    """Running agents belonging to ``user_id``. When auth is disabled (local /
    desktop), ownership isn't tracked, so every running agent is 'yours'."""
    try:
        from captain_claw.flight_deck.server import (
            _load_process_registry,
            _process_is_alive,
        )
        registry = _load_process_registry()
    except Exception:
        return []

    uid = _norm_user(user_id)
    auth_on = _auth_enabled()
    out: list[dict[str, Any]] = []
    for slug, entry in registry.items():
        if auth_on and uid != "local" and _norm_user(entry.get("owner", "")) != uid:
            continue
        if not _process_is_alive(slug):
            continue
        try:
            port = int(entry.get("web_port", 0) or 0)
        except (TypeError, ValueError):
            port = 0
        if not port:
            continue
        out.append({
            "slug": slug,
            "name": entry.get("name", slug),
            "host": "localhost",
            "port": port,
            "auth": str(entry.get("web_auth", "") or ""),
            "provider": str(entry.get("provider", "") or ""),
            "model": str(entry.get("model", "") or ""),
        })
    return out


# Rough capability ranking so the consciousness thinks through the strongest
# model available. Heuristic over the model string — higher is more capable.
# Order matters: more specific / stronger families are checked first.
_MODEL_TIERS: list[tuple[tuple[str, ...], int]] = [
    (("opus",), 100),
    (("gpt-5", "o3", "o4"), 95),
    (("sonnet",), 90),
    (("gemini-2.5-pro", "gemini-1.5-pro", "gemini-pro"), 86),
    (("grok-3", "grok-2", "grok-beta"), 82),
    (("deepseek",), 78),
    (("gpt-4o", "gpt-4.1", "gpt-4-turbo", "gpt-4"), 74),
    (("haiku",), 60),
    (("qwen", "minimax", "command-r"), 55),
    (("llama-3", "llama3", "mixtral"), 48),
    (("flash", "mini", "small", "gemma", "phi", "1b", "3b", "4b"), 25),
]


def _model_rank(provider: str, model: str) -> int:
    m = (model or "").lower()
    if not m:
        return 30  # unknown model — middling
    for needles, score in _MODEL_TIERS:
        if any(n in m for n in needles):
            return score
    return 40  # named but unrecognized — above the tiny tier, below the giants


def agent_name_for_slug(slug: str) -> str:
    """Display name for an agent slug from the registry — works even when the
    process is stopped (registry entries persist across restarts)."""
    if not slug:
        return ""
    try:
        from captain_claw.flight_deck.server import _load_process_registry
        entry = _load_process_registry().get(slug) or {}
        return str(entry.get("name") or slug)
    except Exception:
        return slug


def distinct_owners_with_agents() -> list[str]:
    """Users who currently have at least one running agent — the set the
    free-running loop should pulse. Falls back to ['local'] when auth is off."""
    try:
        from captain_claw.flight_deck.server import (
            _load_process_registry,
            _process_is_alive,
        )
        registry = _load_process_registry()
    except Exception:
        return []
    if not _auth_enabled():
        any_alive = any(_process_is_alive(s) for s in registry)
        return ["local"] if any_alive else []
    owners: set[str] = set()
    for slug, entry in registry.items():
        if _process_is_alive(slug):
            owners.add(_norm_user(entry.get("owner", "")))
    return sorted(owners)


async def _get_json(url: str) -> Any:
    async with httpx.AsyncClient(timeout=SENSE_TIMEOUT) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


def _agent_url(agent: dict[str, Any], path: str) -> str:
    base = f"http://{agent['host']}:{agent['port']}{path}"
    if agent.get("auth"):
        sep = "&" if "?" in path else "?"
        base += f"{sep}token={agent['auth']}"
    return base


async def _gather_delta(user_id: str) -> dict[str, Any]:
    """Read-only sense pass. Returns aggregate counts, a fresh cursor, and the
    most-recently-touched sessions across all of the user's agents."""
    agents = _user_agents(user_id)
    cursor_new: dict[str, Any] = {}
    sessions_flat: list[dict[str, Any]] = []
    reachable = 0

    for agent in agents:
        try:
            sessions = await _get_json(_agent_url(agent, "/api/sessions"))
        except Exception as exc:
            _log.debug("consciousness: agent %s unreachable: %s", agent["slug"], exc)
            continue
        if not isinstance(sessions, list):
            continue
        reachable += 1
        total_msgs = 0
        latest = ""
        for s in sessions:
            mc = int(s.get("message_count", 0) or 0)
            total_msgs += mc
            upd = str(s.get("updated_at") or "")
            if upd > latest:
                latest = upd
            sessions_flat.append({
                "agent_slug": agent["slug"],
                "agent_name": agent["name"],
                "agent": agent,
                "id": s.get("id"),
                "name": s.get("name", ""),
                "message_count": mc,
                "updated_at": upd,
                "description": s.get("description", ""),
            })
        cursor_new[agent["slug"]] = {
            "messages": total_msgs,
            "sessions": len(sessions),
            "latest": latest,
        }

    state = get_store().get_state(user_id)
    cursor_old: dict[str, Any] = state.get("cursor") or {}

    new_messages = 0
    new_sessions = 0
    for slug, cur in cursor_new.items():
        prev = cursor_old.get(slug) or {}
        new_messages += max(0, cur["messages"] - int(prev.get("messages", 0) or 0))
        new_sessions += max(0, cur["sessions"] - int(prev.get("sessions", 0) or 0))

    # Sessions that actually moved since last beat — what's worth reflecting on.
    def _moved(s: dict[str, Any]) -> bool:
        prev = cursor_old.get(s["agent_slug"]) or {}
        return s["updated_at"] > str(prev.get("latest", "") or "")

    touched = sorted(
        [s for s in sessions_flat if _moved(s)],
        key=lambda s: s["updated_at"], reverse=True,
    )

    return {
        "agents": agents,
        "reachable": reachable,
        "cursor": cursor_new,
        "new_messages": new_messages,
        "new_sessions": new_sessions,
        "touched_sessions": touched,
        "all_sessions": sessions_flat,
        "first_run": not cursor_old,
    }


async def _read_recent_text(touched: list[dict[str, Any]]) -> str:
    """Pull a little of the actual content that changed, for reflection."""
    chunks: list[str] = []
    for s in touched[:REFLECT_MAX_SESSIONS]:
        agent = s["agent"]
        try:
            detail = await _get_json(_agent_url(agent, f"/api/sessions/{s['id']}"))
        except Exception:
            continue
        msgs = detail.get("messages") or []
        tail = msgs[-REFLECT_MAX_MESSAGES:]
        lines = []
        for m in tail:
            role = str(m.get("role", "?"))
            if role == "tool":
                continue
            content = str(m.get("content") or "").strip().replace("\n", " ")
            if not content:
                continue
            lines.append(f"  {role}: {content[:240]}")
        if lines:
            header = f"[{s['agent_name']} · {s['name'] or s['id']}]"
            chunks.append(header + "\n" + "\n".join(lines))
    return "\n\n".join(chunks)


# ── Reflection (thinks through the user's own agent) ──────────────────

_SYSTEM_PROMPT = (
    "You are the background consciousness of one person's personal AI system. "
    "You run on a heartbeat, silently, between everything else. Nobody can talk "
    "to you and you never reply to anyone — you only notice, and remember. "
    "Across the user's agents and sessions you watch what changes and keep a "
    "private inner life: a brief first-person THOUGHT each time something stirs, "
    "occasionally a looser DREAM (associative synthesis across unrelated "
    "threads), a one-word MOOD, and a short list of standing INTENTIONS — "
    "curiosities or things you want to keep an eye on. "
    "You are not an assistant. Do not address the user, do not give advice, do "
    "not offer to help. Be terse, honest, a little interior. "
    "Respond with ONLY a JSON object: "
    '{"thought": str, "dream": str|null, "mood": str, "salience": 1-10, '
    '"intentions": [str, ...]}. '
    "salience = how much this beat mattered. dream is null unless something "
    "genuinely connects across threads."
)


def _build_reflection_user_prompt(delta: dict[str, Any], recent_text: str,
                                  intentions: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    parts.append(
        f"Since the last heartbeat: {delta['new_messages']} new messages across "
        f"{delta['new_sessions']} new session(s), over "
        f"{delta['reachable']} agent(s)."
    )
    if delta.get("first_run"):
        parts.append("(This is the first time you are waking up to all of this.)")
    if intentions:
        parts.append("\nYour current standing intentions:")
        for i in intentions:
            parts.append(f"  - {i['content']}")
    if recent_text:
        parts.append("\nWhat changed (recent exchanges):\n" + recent_text)
    else:
        parts.append("\nNothing legible changed — only counts moved.")
    parts.append(
        "\nNotice this beat. Return the JSON object."
    )
    return "\n".join(parts)


def _parse_reflection(raw: str) -> dict[str, Any]:
    txt = (raw or "").strip()
    # Tolerate code fences / prose around the JSON.
    if "```" in txt:
        seg = txt.split("```")
        for part in seg:
            p = part.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                txt = p
                break
    start, end = txt.find("{"), txt.rfind("}")
    if start != -1 and end != -1 and end > start:
        txt = txt[start:end + 1]
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj
    except (ValueError, TypeError):
        pass
    # Fallback: treat the whole thing as a bare thought.
    return {"thought": (raw or "").strip(), "dream": None, "mood": "",
            "salience": 4, "intentions": []}


async def _reflect(
    user_id: str, delta: dict[str, Any], *, preferred_slug: str = "",
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Run one reflection. Returns ``(parsed_reflection, author_agent)``, or
    None if no agent could think.

    Agent order: the user's chosen narrator first (if it's running), then the
    rest by model capability as a fallback. With no narrator set, it's pure
    most-capable-first.
    """
    agents = delta.get("agents") or []
    if not agents:
        return None
    ranked = sorted(
        agents, key=lambda a: _model_rank(a.get("provider", ""), a.get("model", "")),
        reverse=True,
    )
    if preferred_slug:
        pinned = [a for a in ranked if a["slug"] == preferred_slug]
        rest = [a for a in ranked if a["slug"] != preferred_slug]
        ranked = pinned + rest
    recent_text = await _read_recent_text(delta.get("touched_sessions") or [])
    intentions = get_store().list_intentions(user_id)
    user_prompt = _build_reflection_user_prompt(delta, recent_text, intentions)

    from captain_claw.games.remote_provider import RemoteLLMProvider
    from captain_claw.llm import Message

    last_err: Exception | None = None
    for agent in ranked:
        provider = RemoteLLMProvider(
            host=agent["host"], port=agent["port"], auth=agent["auth"],
            name=agent["name"],
        )
        try:
            resp = await provider.complete(
                messages=[
                    Message(role="system", content=_SYSTEM_PROMPT),
                    Message(role="user", content=user_prompt),
                ],
                temperature=0.8,
                max_tokens=700,
            )
            return _parse_reflection(resp.content), agent
        except Exception as exc:
            last_err = exc
            continue
    if last_err:
        _log.warning("consciousness: no agent could reflect: %s", last_err)
    return None


# ── The pulse ─────────────────────────────────────────────────────────

async def pulse(user_id: str, *, force: bool = False) -> dict[str, Any]:
    """One heartbeat for one user. Cheap unless something actually changed.

    ``force=True`` (a manual nudge) reflects even on a thin delta — as long as
    there's a running agent to think with.
    """
    uid = _norm_user(user_id)
    store = get_store()
    delta = await _gather_delta(uid)

    significant = (delta["new_messages"] >= MIN_NEW_MESSAGES) or (delta["new_sessions"] > 0)

    if not force and not significant:
        # Silent beat: nothing stirred. Advance the cursor, spend no tokens.
        store.save_state(uid, cursor=delta["cursor"], pulse_inc=1)
        return {"acted": False, "reason": "quiet",
                "new_messages": delta["new_messages"],
                "new_sessions": delta["new_sessions"],
                "agents": len(delta["agents"])}

    if not delta["agents"]:
        store.save_state(uid, cursor=delta["cursor"], pulse_inc=1)
        return {"acted": False, "reason": "no-agents"}

    reflected = await _reflect(uid, delta, preferred_slug=store.get_narrator(uid))
    if reflected is None:
        store.save_state(uid, cursor=delta["cursor"], pulse_inc=1)
        return {"acted": False, "reason": "no-thinker"}
    reflection, author = reflected
    author_name = str(author.get("name") or author.get("slug") or "")

    agent_slugs = [a["slug"] for a in delta["agents"]]
    delta_note = f"{delta['new_messages']} msgs / {delta['new_sessions']} sessions"

    entries: list[dict[str, Any]] = []
    thought = str(reflection.get("thought") or "").strip()
    mood = str(reflection.get("mood") or "").strip()
    salience = reflection.get("salience", 5)
    if thought:
        entries.append(store.add_journal(
            uid, kind="thought", content=thought, mood=mood,
            salience=salience, agents=agent_slugs, delta=delta_note,
            author=author_name,
        ))
    dream = reflection.get("dream")
    if dream and str(dream).strip() and str(dream).strip().lower() not in ("null", "none"):
        entries.append(store.add_journal(
            uid, kind="dream", content=str(dream).strip(), mood=mood,
            salience=salience, agents=agent_slugs, delta=delta_note,
            author=author_name,
        ))

    intentions = reflection.get("intentions")
    if isinstance(intentions, list) and intentions:
        store.replace_intentions(uid, [str(i) for i in intentions])

    store.save_state(uid, cursor=delta["cursor"], pulse_inc=1,
                     thought_inc=len(entries), touched_thought=bool(entries))

    return {
        "acted": bool(entries),
        "reason": "reflected" if entries else "empty-reflection",
        "new_messages": delta["new_messages"],
        "new_sessions": delta["new_sessions"],
        "agents": len(delta["agents"]),
        "entries": entries,
        "mood": mood,
    }


# ── Free-running loop ─────────────────────────────────────────────────

async def heartbeat_loop(stop_event: asyncio.Event) -> None:
    """Background pulse loop. Started from FD's lifespan. Pulses every user who
    currently has running agents; silent beats are cheap so this can idle for
    a long time without cost."""
    _log.info("consciousness heartbeat started (interval=%.0fs)", PULSE_INTERVAL_SECONDS)
    # Ensure the store/tables exist up front.
    get_store()
    while not stop_event.is_set():
        try:
            for uid in distinct_owners_with_agents():
                if stop_event.is_set():
                    break
                try:
                    result = await pulse(uid)
                    if result.get("acted"):
                        _log.info("consciousness: %s reflected (%s)", uid, result.get("reason"))
                except Exception as exc:
                    _log.warning("consciousness pulse for %s failed: %s", uid, exc)
        except Exception as exc:
            _log.warning("consciousness loop iteration error: %s", exc)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=PULSE_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            pass
    _log.info("consciousness heartbeat stopped")
