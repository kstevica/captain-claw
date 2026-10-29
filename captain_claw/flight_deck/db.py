"""Flight Deck SQLite database — users, settings, chat persistence."""

from __future__ import annotations

import json
import re
import secrets
import uuid
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiosqlite


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return uuid.uuid4().hex


def _with_run_labels(row: dict) -> dict:
    """Shape a basna_runs row for readers: the raw column `success` is the judge's
    automatic label, kept as `judge_success`; `success` becomes the effective label
    (the human thumbs vote when there is one, else the judge's), so every consumer
    keeps seeing the human override while the judge verdict survives for calibration.
    """
    judge = row.get("success")
    human = row.get("human_success")
    row["judge_success"] = judge
    row["success"] = human if human is not None else judge
    return row


# ── Gmail topic auto-replies: the ledger's vocabulary ─────────────────
#
# new → matching → [waiting_agent] → writing → checking → approved → sending
# → sent | unknown; terminal: skipped · held · blocked · failed · dismissed;
# the owner path is held → approved (approved_by) → sending → sent.

AUTOREPLY_STATES: tuple[str, ...] = (
    "new", "matching", "waiting_agent", "writing", "checking", "approved",
    "sending", "sent", "unknown", "skipped", "held", "blocked", "failed", "dismissed",
)
# What the caps count: a reply that went out, may have gone out, or is going.
AUTOREPLY_SEND_STATES: tuple[str, ...] = ("sending", "sent", "unknown")
# Mid-pipeline states a restart interrupts; nothing was sent from any of them.
AUTOREPLY_WORK_STATES: tuple[str, ...] = ("matching", "writing", "checking", "approved")
# Skip reasons the activity list hides unless "Show all".
AUTOREPLY_QUIET_REASONS: tuple[str, ...] = (
    "labels", "auto_or_list", "automated_sender", "no_match", "unauthenticated", "burst",
    "classified_automated",
)
AUTOREPLY_RECOVER_ATTEMPTS = 2   # a restart requeues an interrupted row while attempts < this
AUTOREPLY_HELD_DAYS = 7          # held cards expire (→ dismissed) after this
AUTOREPLY_TEXT_DAYS = 14         # reply_text is cleared after this
AUTOREPLY_QUIET_DAYS = 30        # skipped / failed rows are deleted after this
AUTOREPLY_KEEP_DAYS = 90         # every row is deleted after this

# Ledger columns update_autoreply / cas_autoreply may write, with a clip length
# (0 = unclipped). id, owner_id, inbound_message_id, state and the timestamps
# are not here: a state only ever changes through cas_autoreply's to_state.
_AUTOREPLY_TEXT_COLUMNS: dict[str, int] = {
    "thread_id": 0, "received_at": 0, "from_addr": 320, "from_domain": 255,
    "subject": 200, "reason": 200, "detail": 300, "topic_id": 0, "topic_name": 120,
    "match_judge": 16, "reply_text": 0, "agent_ref": 0, "agent_name": 120,
    "approved_by": 0, "sent_message_id": 0,
}
_AUTOREPLY_JSON_COLUMNS: dict[str, type] = {"findings": list, "judge": dict, "sources_used": list}
_AUTOREPLY_STATE_TEXT_COLUMNS: dict[str, int] = {
    "history_id": 0, "google_sub": 0, "account_email": 0, "enabled_at": 0,
    "own_addresses_at": 0, "last_poll_at": 0, "last_ok_at": 0, "last_error": 300,
    "classify_day": 0,
}


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def _text(value: Any, clip: int = 0) -> str:
    s = "" if value is None else str(value)
    return s[:clip] if clip else s


def _autoreply_states(states: Iterable[str] | str | None) -> list[str]:
    """Validated state names; a typo raises rather than silently matching
    nothing (an under-counted cap would fail open)."""
    if states is None:
        return []
    if isinstance(states, str):
        states = (states,)
    out = list(dict.fromkeys(states))
    bad = [s for s in out if s not in AUTOREPLY_STATES]
    if bad:
        raise ValueError(f"unknown auto-reply state(s): {bad}")
    return out


def _autoreply_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Coerce whitelisted ledger columns for a write; unknown names raise."""
    out: dict[str, Any] = {}
    for key, value in fields.items():
        if key in _AUTOREPLY_TEXT_COLUMNS:
            v = _text(value, _AUTOREPLY_TEXT_COLUMNS[key])
            out[key] = v.strip().lower() if key in ("from_addr", "from_domain") else v
        elif key in _AUTOREPLY_JSON_COLUMNS:
            out[key] = value if isinstance(value, str) else json.dumps(
                value, ensure_ascii=False, default=_json_default)
        elif key == "confidence":
            out[key] = None if value is None else float(value)
        elif key == "attempts":
            out[key] = int(value)
        else:
            raise ValueError(f"gmail_autoreplies column not writable: {key!r}")
    return out


def _autoreply_row(row: Any) -> dict:
    """A ledger row for readers, with its JSON columns parsed (a corrupt value
    reads as empty)."""
    d = dict(row)
    for col, kind in _AUTOREPLY_JSON_COLUMNS.items():
        try:
            val = json.loads(d.get(col) or "")
        except (ValueError, TypeError):
            val = None
        d[col] = val if isinstance(val, kind) else kind()
    return d


def _own_addresses_json(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value or "[]")
        except ValueError:
            raise ValueError("own_addresses must be a JSON list") from None
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError("own_addresses must be a list of addresses")
    addrs = [str(a).strip().lower() for a in value if str(a or "").strip()]
    return json.dumps(list(dict.fromkeys(addrs)))


def _autoreply_state_fields(fields: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in fields.items():
        if key in _AUTOREPLY_STATE_TEXT_COLUMNS:
            out[key] = _text(value, _AUTOREPLY_STATE_TEXT_COLUMNS[key])
        elif key == "own_addresses":
            out[key] = _own_addresses_json(value)
        elif key == "classify_count":
            out[key] = int(value)
        else:
            raise ValueError(f"gmail_autoreply_state column not writable: {key!r}")
    return out


def _autoreply_state_row(row: Any) -> dict:
    d = dict(row)
    try:
        addrs = json.loads(d.get("own_addresses") or "[]")
    except (ValueError, TypeError):
        addrs = []
    d["own_addresses"] = [str(a) for a in addrs] if isinstance(addrs, list) else []
    return d


class FlightDeckDB:
    """Async SQLite store for Flight Deck multi-tenant data."""

    def __init__(self, db_path: Path | str):
        self._db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._create_tables()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def _create_tables(self) -> None:
        assert self._db is not None
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id           TEXT PRIMARY KEY,
                email        TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL DEFAULT '',
                role         TEXT NOT NULL DEFAULT 'user',
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                metadata     TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS user_sessions (
                id                 TEXT PRIMARY KEY,
                user_id            TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                refresh_token_hash TEXT NOT NULL,
                expires_at         TEXT NOT NULL,
                created_at         TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_user_sessions_user
                ON user_sessions(user_id);

            CREATE TABLE IF NOT EXISTS user_settings (
                user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                key        TEXT NOT NULL,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, key)
            );

            -- Account applications while self-registration is closed. The
            -- applicant names who recommended them; an admin approves (the
            -- account is created with the password they chose — the hash is
            -- cleared from here once decided) or rejects. One pending
            -- application per email.
            CREATE TABLE IF NOT EXISTS waitlist (
                id             TEXT PRIMARY KEY,
                email          TEXT NOT NULL,
                display_name   TEXT NOT NULL DEFAULT '',
                password_hash  TEXT NOT NULL DEFAULT '',
                referrer_name  TEXT NOT NULL DEFAULT '',
                referrer_email TEXT NOT NULL DEFAULT '',
                cohort         TEXT NOT NULL DEFAULT '',
                status         TEXT NOT NULL DEFAULT 'pending',
                created_at     TEXT NOT NULL,
                decided_at     TEXT NOT NULL DEFAULT '',
                decided_by     TEXT NOT NULL DEFAULT '',
                user_id        TEXT NOT NULL DEFAULT ''
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_waitlist_pending_email
                ON waitlist(email) WHERE status = 'pending';
            CREATE INDEX IF NOT EXISTS idx_waitlist_status
                ON waitlist(status, created_at);

            -- Invitations an admin sent (invite_routes). The invitee opens
            -- the link — only its sha256 is kept here — picks a password,
            -- and the account is created then. kiosk NULL = the deck's
            -- "new users start in kiosk" at acceptance. One pending
            -- invitation per email.
            CREATE TABLE IF NOT EXISTS invites (
                id           TEXT PRIMARY KEY,
                email        TEXT NOT NULL,
                display_name TEXT NOT NULL DEFAULT '',
                role         TEXT NOT NULL DEFAULT 'user',
                kiosk        INTEGER,
                note         TEXT NOT NULL DEFAULT '',
                token_hash   TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'pending',
                invited_by   TEXT NOT NULL DEFAULT '',
                created_at   TEXT NOT NULL,
                expires_at   TEXT NOT NULL,
                sent_via     TEXT NOT NULL DEFAULT '',
                sent_from    TEXT NOT NULL DEFAULT '',
                sent_at      TEXT NOT NULL DEFAULT '',
                accepted_at  TEXT NOT NULL DEFAULT '',
                user_id      TEXT NOT NULL DEFAULT ''
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_invites_pending_email
                ON invites(email) WHERE status = 'pending';
            CREATE UNIQUE INDEX IF NOT EXISTS idx_invites_token
                ON invites(token_hash);

            CREATE TABLE IF NOT EXISTS chat_sessions (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                agent_id   TEXT NOT NULL DEFAULT '',
                agent_name TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chat_sessions_user
                ON chat_sessions(user_id);

            CREATE TABLE IF NOT EXISTS chat_messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL DEFAULT '',
                metadata   TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                ON chat_messages(session_id);

            CREATE TABLE IF NOT EXISTS system_settings (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS usage_logs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                event_type TEXT NOT NULL,
                detail     TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_usage_logs_user
                ON usage_logs(user_id);
            CREATE INDEX IF NOT EXISTS idx_usage_logs_type
                ON usage_logs(event_type);
            CREATE INDEX IF NOT EXISTS idx_usage_logs_created
                ON usage_logs(created_at);

            -- What each user spent per UTC day against their plan's daily
            -- allowances (usage_quota): human messages and LLM tokens.
            CREATE TABLE IF NOT EXISTS daily_usage (
                user_id    TEXT NOT NULL,
                day        TEXT NOT NULL,
                messages   INTEGER NOT NULL DEFAULT 0,
                tokens     INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, day)
            );
            CREATE INDEX IF NOT EXISTS idx_daily_usage_day
                ON daily_usage(day);

            -- Free resets an admin gave a user (usage_quota): spending one
            -- zeroes the user's daily_usage for today.
            CREATE TABLE IF NOT EXISTS usage_reset_credits (
                user_id    TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                balance    INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS council_sessions (
                id              TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title           TEXT NOT NULL DEFAULT '',
                topic           TEXT NOT NULL DEFAULT '',
                session_type    TEXT NOT NULL DEFAULT 'brainstorm',
                verbosity       TEXT NOT NULL DEFAULT 'message',
                max_rounds      INTEGER NOT NULL DEFAULT 5,
                current_round   INTEGER NOT NULL DEFAULT 0,
                status          TEXT NOT NULL DEFAULT 'setup',
                moderator_mode  TEXT NOT NULL DEFAULT 'round-robin',
                moderator_agent TEXT NOT NULL DEFAULT '',
                agents          TEXT NOT NULL DEFAULT '[]',
                pinned_ids      TEXT NOT NULL DEFAULT '[]',
                config          TEXT NOT NULL DEFAULT '{}',
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_council_sessions_user
                ON council_sessions(user_id);

            CREATE TABLE IF NOT EXISTS council_messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL REFERENCES council_sessions(id) ON DELETE CASCADE,
                round           INTEGER NOT NULL DEFAULT 1,
                agent_id        TEXT NOT NULL DEFAULT '',
                agent_name      TEXT NOT NULL DEFAULT '',
                role            TEXT NOT NULL,
                action          TEXT NOT NULL DEFAULT '',
                suitability     REAL NOT NULL DEFAULT 0.0,
                target_agent_id TEXT NOT NULL DEFAULT '',
                content         TEXT NOT NULL DEFAULT '',
                pinned          INTEGER NOT NULL DEFAULT 0,
                metadata        TEXT NOT NULL DEFAULT '{}',
                created_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_council_messages_session
                ON council_messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_council_messages_round
                ON council_messages(session_id, round);

            CREATE TABLE IF NOT EXISTS council_votes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL REFERENCES council_sessions(id) ON DELETE CASCADE,
                round           INTEGER NOT NULL,
                agent_id        TEXT NOT NULL,
                agent_name      TEXT NOT NULL DEFAULT '',
                vote            TEXT NOT NULL,
                reason          TEXT NOT NULL DEFAULT '',
                created_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_council_votes_session
                ON council_votes(session_id);

            CREATE TABLE IF NOT EXISTS council_artifacts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL REFERENCES council_sessions(id) ON DELETE CASCADE,
                kind            TEXT NOT NULL,
                agent_id        TEXT NOT NULL DEFAULT '',
                agent_name      TEXT NOT NULL DEFAULT '',
                content         TEXT NOT NULL DEFAULT '',
                created_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_council_artifacts_session
                ON council_artifacts(session_id);

            -- ── Basna: router → selective spawn → weighted merge → learning ──
            CREATE TABLE IF NOT EXISTS basna_sessions (
                id           TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title        TEXT NOT NULL DEFAULT '',
                intent       TEXT NOT NULL DEFAULT '',
                domain       TEXT NOT NULL DEFAULT '',
                difficulty   TEXT NOT NULL DEFAULT '',
                merge_kind   TEXT NOT NULL DEFAULT 'converge',
                status       TEXT NOT NULL DEFAULT 'routing',
                route        TEXT NOT NULL DEFAULT '{}',
                truth        TEXT NOT NULL DEFAULT '',
                confidence   REAL NOT NULL DEFAULT 0.0,
                config       TEXT NOT NULL DEFAULT '{}',
                progress     TEXT NOT NULL DEFAULT '[]',  -- JSON: execution progress log
                files        TEXT NOT NULL DEFAULT '[]',  -- JSON: attached files [{name,mime,size}]
                analysis     TEXT NOT NULL DEFAULT '{}',  -- JSON: cross-agent analysis (agreement/diffs/blind spots)
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_basna_sessions_user
                ON basna_sessions(user_id);

            CREATE TABLE IF NOT EXISTS basna_runs (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id     TEXT NOT NULL REFERENCES basna_sessions(id) ON DELETE CASCADE,
                archetype_id   TEXT NOT NULL DEFAULT '',
                role           TEXT NOT NULL DEFAULT '',
                provider       TEXT NOT NULL DEFAULT '',
                model          TEXT NOT NULL DEFAULT '',
                tier           TEXT NOT NULL DEFAULT '',
                weight_at_run  REAL NOT NULL DEFAULT 0.0,
                output         TEXT NOT NULL DEFAULT '',
                actions        TEXT NOT NULL DEFAULT '[]',  -- JSON: per-agent tool actions
                success        INTEGER,            -- judge/auto label: NULL until scored; 1 = success, 0 = fail
                human_success  INTEGER,            -- human thumbs label: NULL until voted; 1 / 0
                human_feedback_at TEXT,            -- when the human last voted
                latency_ms     INTEGER NOT NULL DEFAULT 0,
                created_at     TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_basna_runs_session
                ON basna_runs(session_id);

            -- Learned, per-user reliability of each archetype within a domain.
            CREATE TABLE IF NOT EXISTS archetype_reliability (
                user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                archetype_id TEXT NOT NULL,
                domain       TEXT NOT NULL DEFAULT '',
                successes    INTEGER NOT NULL DEFAULT 0,
                fails        INTEGER NOT NULL DEFAULT 0,
                runs         INTEGER NOT NULL DEFAULT 0,
                weight       REAL NOT NULL DEFAULT 0.7,
                updated_at   TEXT NOT NULL,
                PRIMARY KEY (user_id, archetype_id, domain)
            );

            -- R2 automatic CONSTRAINT edge (opt-in `constraint_learning`): short,
            -- reusable rules distilled from accepted/fixed runs, injected into the
            -- splitter/planner/contract briefs of later runs. Additive — created on
            -- demand, IF NOT EXISTS, no migration of existing tables. No users FK
            -- (mirrors cost_ledger) so it is engine- and test-friendly.
            CREATE TABLE IF NOT EXISTS learned_constraints (
                id              TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                domain          TEXT NOT NULL DEFAULT '',
                engine          TEXT NOT NULL DEFAULT '',   -- 'vatra' | 'code'
                trigger_text    TEXT NOT NULL DEFAULT '',
                constraint_text TEXT NOT NULL DEFAULT '',
                severity        TEXT NOT NULL DEFAULT 'major',
                source_id       TEXT NOT NULL DEFAULT '',    -- session/project id
                hits            INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_learned_constraints_user
                ON learned_constraints(user_id, domain);

            -- Vatra blackboard: cross-agent "asks" a specialist posts when it needs
            -- something outside its slice. The coordinator routes each to a helper
            -- and writes the answer back; the reporter folds answered asks in.
            CREATE TABLE IF NOT EXISTS basna_asks (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT NOT NULL REFERENCES basna_sessions(id) ON DELETE CASCADE,
                from_owner   TEXT NOT NULL DEFAULT '',   -- asker archetype id
                from_subtask TEXT NOT NULL DEFAULT '',
                text         TEXT NOT NULL DEFAULT '',
                status       TEXT NOT NULL DEFAULT 'open',  -- open|claimed|answered|dropped
                answer       TEXT NOT NULL DEFAULT '',
                answered_by  TEXT NOT NULL DEFAULT '',
                depth        INTEGER NOT NULL DEFAULT 0,
                note         TEXT NOT NULL DEFAULT '',      -- e.g. drop reason
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_basna_asks_session
                ON basna_asks(session_id);

            -- Vatra shared board: every agent's notes, outputs, and files stream
            -- here as they work, so teammates can read/search each other's work in
            -- real time (a shared memory / clipboard for the run).
            CREATE TABLE IF NOT EXISTS vatra_board (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT NOT NULL REFERENCES basna_sessions(id) ON DELETE CASCADE,
                from_owner   TEXT NOT NULL DEFAULT '',   -- archetype id of the author
                from_subtask TEXT NOT NULL DEFAULT '',
                kind         TEXT NOT NULL DEFAULT 'note',  -- note|narration|output|file
                title        TEXT NOT NULL DEFAULT '',
                content      TEXT NOT NULL DEFAULT '',
                created_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_vatra_board_session
                ON vatra_board(session_id);

            -- Vatra run ledger: a rule-based map of who is covering what — each
            -- owner's current focus, the sources read, the files written, the
            -- findings posted, the asks answered — built from events Flight Deck
            -- already sees, and read by workers through `vatra map`. One row per
            -- (session, kind, key); a repeat bumps `count` and records who.
            CREATE TABLE IF NOT EXISTS vatra_ledger (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL REFERENCES basna_sessions(id) ON DELETE CASCADE,
                kind        TEXT NOT NULL,              -- focus|source|file|finding|ask_answer
                key         TEXT NOT NULL,
                owner       TEXT NOT NULL DEFAULT '',   -- archetype id of the first actor
                subtask_id  TEXT NOT NULL DEFAULT '',
                label       TEXT NOT NULL DEFAULT '',   -- the actor's display name
                detail      TEXT NOT NULL DEFAULT '',
                pointer     TEXT NOT NULL DEFAULT '',
                count       INTEGER NOT NULL DEFAULT 1,
                others      TEXT NOT NULL DEFAULT '',   -- other actors' labels, comma-separated
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                UNIQUE(session_id, kind, key)
            );
            CREATE INDEX IF NOT EXISTS idx_vatra_ledger_session
                ON vatra_ledger(session_id, kind);

            -- Vatra resume checkpoint: one row per owner (subtask) capturing its
            -- finished slice, so a stalled/cancelled run can be resumed — completed
            -- owners are restored from here (no re-run, no re-spend) and only the
            -- missing ones are re-dispatched. UPSERT-keyed on (session_id, subtask_id)
            -- so a re-dispatched owner overwrites its own checkpoint idempotently.
            CREATE TABLE IF NOT EXISTS vatra_runs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT NOT NULL REFERENCES basna_sessions(id) ON DELETE CASCADE,
                subtask_id    TEXT NOT NULL DEFAULT '',
                archetype_id  TEXT NOT NULL DEFAULT '',
                role          TEXT NOT NULL DEFAULT '',
                weight        REAL NOT NULL DEFAULT 0.0,
                output        TEXT NOT NULL DEFAULT '',
                produced_file INTEGER NOT NULL DEFAULT 0,
                status        TEXT NOT NULL DEFAULT 'done',  -- done|failed|skipped
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL,
                UNIQUE (session_id, subtask_id)
            );
            CREATE INDEX IF NOT EXISTS idx_vatra_runs_session
                ON vatra_runs(session_id);

            -- Basna/Vatra projects: a bundle of runs sharing one theme (description +
            -- instructions injected into each run) and one read-only VFS folder
            -- (uploads, auto-added as a reference folder). Runs stay independent —
            -- the project just groups them and seeds their context.
            CREATE TABLE IF NOT EXISTS basna_projects (
                id           TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name         TEXT NOT NULL DEFAULT '',
                description  TEXT NOT NULL DEFAULT '',
                instructions TEXT NOT NULL DEFAULT '',
                vfs_folder   TEXT NOT NULL DEFAULT '',   -- the project's VFS folder
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_basna_projects_user
                ON basna_projects(user_id);

            CREATE TABLE IF NOT EXISTS prompts (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title      TEXT NOT NULL DEFAULT '',
                content    TEXT NOT NULL DEFAULT '',
                files      TEXT NOT NULL DEFAULT '[]',
                tags       TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_prompts_user
                ON prompts(user_id);

            -- Per-user (per-tenant) agent archetypes. The base set lives in
            -- instructions/archetypes.json; rows here are added on top and, when
            -- archetype_id matches a base one, shadow it for that user. `data`
            -- holds the full archetype JSON (role, family, keywords,
            -- cognitive_mode, tier, tools, description, fleet_instructions,
            -- lead, reliability_seed).
            CREATE TABLE IF NOT EXISTS user_archetypes (
                id           TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                archetype_id TEXT NOT NULL,
                data         TEXT NOT NULL DEFAULT '{}',
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                UNIQUE(user_id, archetype_id)
            );
            CREATE INDEX IF NOT EXISTS idx_user_archetypes_user
                ON user_archetypes(user_id);

            -- Cross-user resource sharing. An owner grants another FD user
            -- access to one of their resources. `resource_type` is one of
            -- archetype | code | basna | council | vfs; `resource_id` is the
            -- natural key within the owner's namespace (archetype slug, Basna
            -- or Council session id, or a VFS/Code project folder name).
            -- `permission` is 'view' (read-only) or 'edit' (collaborate);
            -- archetypes are always use-only regardless of permission.
            CREATE TABLE IF NOT EXISTS resource_shares (
                id            TEXT PRIMARY KEY,
                resource_type TEXT NOT NULL,
                resource_id   TEXT NOT NULL,
                owner_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                grantee_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                permission    TEXT NOT NULL DEFAULT 'view',
                created_at    TEXT NOT NULL,
                UNIQUE(resource_type, resource_id, owner_id, grantee_id)
            );
            CREATE INDEX IF NOT EXISTS idx_resource_shares_grantee
                ON resource_shares(grantee_id, resource_type);
            CREATE INDEX IF NOT EXISTS idx_resource_shares_owner
                ON resource_shares(owner_id, resource_type, resource_id);

            -- Agents that ever had members (a share row is deleted when the
            -- member is removed; their learnings stay in the agent's commons).
            CREATE TABLE IF NOT EXISTS agent_share_history (
                agent_ref        TEXT NOT NULL,
                owner_id         TEXT NOT NULL,
                first_shared_at  TEXT NOT NULL,
                PRIMARY KEY (agent_ref, owner_id)
            );
            INSERT OR IGNORE INTO agent_share_history (agent_ref, owner_id, first_shared_at)
                SELECT resource_id, owner_id, MIN(created_at) FROM resource_shares
                WHERE resource_type = 'agent' GROUP BY resource_id, owner_id;

            -- Public file links (public_file_shares.py): one agent file,
            -- snapshotted into <data dir>/public_shares/<token>, readable by
            -- anyone holding the token. No FK on owner_id: an auth-disabled
            -- deck's owner is the synthetic "local" user (delete_user drops
            -- a removed user's links instead).
            CREATE TABLE IF NOT EXISTS public_file_shares (
                token          TEXT PRIMARY KEY,
                owner_id       TEXT NOT NULL,
                agent_ref      TEXT NOT NULL DEFAULT '',
                agent_name     TEXT NOT NULL DEFAULT '',
                source_path    TEXT NOT NULL,
                filename       TEXT NOT NULL,
                kind           TEXT NOT NULL,
                size           INTEGER NOT NULL DEFAULT 0,
                created_at     TEXT NOT NULL,
                updated_at     TEXT NOT NULL,
                expires_at     TEXT,
                view_count     INTEGER NOT NULL DEFAULT 0,
                last_viewed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_public_file_shares_owner
                ON public_file_shares(owner_id, agent_ref, source_path);
            -- The address each user's browser reaches this deck at (its
            -- Origin), so a link an AGENT makes points where people can open
            -- it — the agent itself only knows the deck's internal URL.
            CREATE TABLE IF NOT EXISTS public_link_origins (
                owner_id  TEXT PRIMARY KEY,
                origin    TEXT NOT NULL,
                seen_at   TEXT NOT NULL
            );

            -- The one Flight Deck that runs the auto-reply engine on this
            -- database (two decks can share one fd-data): pid + host, and a
            -- heartbeat; a cold claim is taken over.
            CREATE TABLE IF NOT EXISTS gmail_autoreply_runner (
                id            INTEGER PRIMARY KEY CHECK (id = 1),
                pid           INTEGER NOT NULL,
                host          TEXT NOT NULL,
                claimed_at    TEXT NOT NULL,
                heartbeat_at  TEXT NOT NULL
            );

            -- Persisted run costs (Iskra Phase 0, docs/living-beings-plan.md
            -- §10). One row per finished run: the aggregate token usage +
            -- dollar cost that pricing.summarize computed, which previously
            -- lived only in the in-memory progress log. owner_type 'being'
            -- rows carry the being slug in owner_ref; wallet debits stay in
            -- beings.db — this table is the dollar/reporting side.
            CREATE TABLE IF NOT EXISTS cost_ledger (
                id              TEXT PRIMARY KEY,
                owner_user_id   TEXT NOT NULL,
                owner_type      TEXT NOT NULL DEFAULT 'user',
                owner_ref       TEXT NOT NULL DEFAULT '',
                run_kind        TEXT NOT NULL,
                run_id          TEXT NOT NULL DEFAULT '',
                usage           TEXT NOT NULL DEFAULT '{}',
                usd             REAL,
                elapsed_seconds REAL,
                at              TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cost_ledger_owner
                ON cost_ledger(owner_user_id, at);
            CREATE TABLE IF NOT EXISTS notifications (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                type        TEXT NOT NULL DEFAULT 'info',
                title       TEXT NOT NULL DEFAULT '',
                body        TEXT NOT NULL DEFAULT '',
                ref_type    TEXT NOT NULL DEFAULT '',
                ref_id      TEXT NOT NULL DEFAULT '',
                read        INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_notifications_user
                ON notifications(user_id, read, created_at);
            -- Gmail sends made through Flight Deck (POST /fd/google/gmail/send):
            -- the audit trail the owner sees, and what the daily limit and the
            -- duplicate check count. One row per message Gmail accepted
            -- (status 'sent') or may have sent: the send call got a 5xx or no
            -- answer (status 'unknown').
            CREATE TABLE IF NOT EXISTS gmail_sends (
                id               TEXT PRIMARY KEY,
                owner_id         TEXT NOT NULL,
                agent            TEXT NOT NULL DEFAULT '',
                to_addrs         TEXT NOT NULL DEFAULT '',
                cc_addrs         TEXT NOT NULL DEFAULT '',
                bcc_addrs        TEXT NOT NULL DEFAULT '',
                subject          TEXT NOT NULL DEFAULT '',
                gmail_message_id TEXT NOT NULL DEFAULT '',
                thread_id        TEXT NOT NULL DEFAULT '',
                draft_id         TEXT NOT NULL DEFAULT '',
                content_hash     TEXT NOT NULL DEFAULT '',
                status           TEXT NOT NULL DEFAULT 'sent',
                created_at       TEXT NOT NULL,
                -- JSON list of the attached files' names ('' = none)
                attachments      TEXT NOT NULL DEFAULT '',
                -- who decided the send: 'agent' (POST /gmail/send), 'autoreply'
                -- (the topic auto-reply engine) or 'autoreply_approved' (the
                -- owner sent a held auto-reply); limits and listings filter on it
                origin           TEXT NOT NULL DEFAULT 'agent',
                topic_id         TEXT NOT NULL DEFAULT '',
                in_reply_to_message_id TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_gmail_sends_owner
                ON gmail_sends(owner_id, created_at);
            -- Gmail topic auto-replies (gmail_autoreply.py). Per-owner poll
            -- state: a row exists ONLY while the owner has auto-replies on —
            -- the enabling PUT creates it (the history baseline), turning off
            -- deletes it, and the engine polls exactly the owners listed here.
            -- `own_addresses` is a JSON list; `classify_day` / `classify_count`
            -- are the per-UTC-day classification budget.
            CREATE TABLE IF NOT EXISTS gmail_autoreply_state (
                owner_id         TEXT PRIMARY KEY,
                history_id       TEXT NOT NULL DEFAULT '',
                google_sub       TEXT NOT NULL DEFAULT '',
                account_email    TEXT NOT NULL DEFAULT '',
                enabled_at       TEXT NOT NULL DEFAULT '',
                own_addresses    TEXT NOT NULL DEFAULT '[]',
                own_addresses_at TEXT NOT NULL DEFAULT '',
                last_poll_at     TEXT NOT NULL DEFAULT '',
                last_ok_at       TEXT NOT NULL DEFAULT '',
                last_error       TEXT NOT NULL DEFAULT '',
                classify_day     TEXT NOT NULL DEFAULT '',
                classify_count   INTEGER NOT NULL DEFAULT 0,
                updated_at       TEXT NOT NULL
            );
            -- The auto-reply ledger: one row per INBOX message the engine
            -- claimed (INSERT OR IGNORE, before any fetch or LLM call), moved
            -- through its state machine by compare-and-set only. The inbound
            -- body is never stored. `findings` (masked hints), `judge` (scores
            -- and a note) and `sources_used` are JSON.
            CREATE TABLE IF NOT EXISTS gmail_autoreplies (
                id                 TEXT PRIMARY KEY,
                owner_id           TEXT NOT NULL,
                inbound_message_id TEXT NOT NULL,
                thread_id          TEXT NOT NULL DEFAULT '',
                received_at        TEXT NOT NULL DEFAULT '',
                from_addr          TEXT NOT NULL DEFAULT '',
                from_domain        TEXT NOT NULL DEFAULT '',
                subject            TEXT NOT NULL DEFAULT '',
                state              TEXT NOT NULL DEFAULT 'new',
                reason             TEXT NOT NULL DEFAULT '',
                detail             TEXT NOT NULL DEFAULT '',
                topic_id           TEXT NOT NULL DEFAULT '',
                topic_name         TEXT NOT NULL DEFAULT '',
                confidence         REAL,
                match_judge        TEXT NOT NULL DEFAULT '',
                reply_text         TEXT NOT NULL DEFAULT '',
                findings           TEXT NOT NULL DEFAULT '[]',
                judge              TEXT NOT NULL DEFAULT '{}',
                sources_used       TEXT NOT NULL DEFAULT '[]',
                agent_ref          TEXT NOT NULL DEFAULT '',
                agent_name         TEXT NOT NULL DEFAULT '',
                attempts           INTEGER NOT NULL DEFAULT 0,
                approved_by        TEXT NOT NULL DEFAULT '',
                sent_message_id    TEXT NOT NULL DEFAULT '',
                created_at         TEXT NOT NULL,
                updated_at         TEXT NOT NULL,
                UNIQUE(owner_id, inbound_message_id)
            );
            CREATE INDEX IF NOT EXISTS idx_gmail_autoreplies_owner
                ON gmail_autoreplies(owner_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_gmail_autoreplies_thread
                ON gmail_autoreplies(owner_id, thread_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_gmail_autoreplies_sender
                ON gmail_autoreplies(owner_id, from_addr, created_at);
            CREATE INDEX IF NOT EXISTS idx_gmail_autoreplies_state
                ON gmail_autoreplies(owner_id, state);
            CREATE TABLE IF NOT EXISTS personal_access_tokens (
                id           TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash   TEXT NOT NULL,
                name         TEXT NOT NULL DEFAULT '',
                created_at   TEXT NOT NULL,
                last_used_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_pat_hash
                ON personal_access_tokens(token_hash);
            CREATE INDEX IF NOT EXISTS idx_pat_user
                ON personal_access_tokens(user_id);
            -- Agent links (agent_link.py): a system someone connected to one of
            -- their agents over /link/ws, with a key of its own. Only the key's
            -- sha256 is stored; key_hint is its last 4 characters.
            CREATE TABLE IF NOT EXISTS agent_links (
                id               TEXT PRIMARY KEY,
                user_id          TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name             TEXT NOT NULL DEFAULT '',
                agent_slug       TEXT NOT NULL DEFAULT '',
                instructions     TEXT NOT NULL DEFAULT '',
                key_hash         TEXT NOT NULL,
                key_hint         TEXT NOT NULL DEFAULT '',
                agent_session_id TEXT NOT NULL DEFAULT '',
                agent_may_send   INTEGER NOT NULL DEFAULT 1,
                webhook_url      TEXT NOT NULL DEFAULT '',
                webhook_secret   TEXT NOT NULL DEFAULT '',
                -- 'outbound': Flight Deck connects out to remote_url with the
                -- system's key (remote_key, sealed) — agent_link_out.py.
                direction        TEXT NOT NULL DEFAULT 'inbound',
                remote_url       TEXT NOT NULL DEFAULT '',
                remote_key       TEXT NOT NULL DEFAULT '',
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL,
                last_used_at     TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_links_key
                ON agent_links(key_hash);
            CREATE INDEX IF NOT EXISTS idx_agent_links_user
                ON agent_links(user_id);
            -- What Flight Deck has for a linked system (agent_link_http.py): answers
            -- to its HTTP messages and the agent's own messages, kept a week.
            CREATE TABLE IF NOT EXISTS agent_link_events (
                seq        INTEGER PRIMARY KEY AUTOINCREMENT,
                link_id    TEXT NOT NULL REFERENCES agent_links(id) ON DELETE CASCADE,
                kind       TEXT NOT NULL,
                re         TEXT NOT NULL DEFAULT '',
                body       TEXT NOT NULL DEFAULT '{}',
                delivered  TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_agent_link_events
                ON agent_link_events(link_id, seq);
            -- The files those events name: where they are on the agent.
            CREATE TABLE IF NOT EXISTS agent_link_files (
                id         TEXT PRIMARY KEY,
                link_id    TEXT NOT NULL REFERENCES agent_links(id) ON DELETE CASCADE,
                agent_slug TEXT NOT NULL DEFAULT '',
                path       TEXT NOT NULL,
                name       TEXT NOT NULL,
                kind       TEXT NOT NULL DEFAULT 'file',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS oauth_clients (
                id            TEXT PRIMARY KEY,
                client_name   TEXT NOT NULL DEFAULT '',
                redirect_uris TEXT NOT NULL DEFAULT '[]',
                created_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS oauth_codes (
                id             TEXT PRIMARY KEY,
                code_hash      TEXT NOT NULL UNIQUE,
                client_id      TEXT NOT NULL,
                user_id        TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                redirect_uri   TEXT NOT NULL,
                code_challenge TEXT NOT NULL,
                scope          TEXT NOT NULL DEFAULT 'mcp',
                resource       TEXT,
                created_at     TEXT NOT NULL,
                expires_at     TEXT NOT NULL,
                used_at        TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_oauth_codes_hash
                ON oauth_codes(code_hash);
            CREATE TABLE IF NOT EXISTS oauth_refresh_tokens (
                id           TEXT PRIMARY KEY,
                token_hash   TEXT NOT NULL UNIQUE,
                client_id    TEXT NOT NULL,
                client_name  TEXT NOT NULL DEFAULT '',
                user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                scope        TEXT NOT NULL DEFAULT 'mcp',
                created_at   TEXT NOT NULL,
                last_used_at TEXT,
                revoked_at   TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_oauth_refresh_hash
                ON oauth_refresh_tokens(token_hash);
            CREATE INDEX IF NOT EXISTS idx_oauth_refresh_user
                ON oauth_refresh_tokens(user_id, revoked_at);

            -- Shared-agent context packs (PR B, see context_packs.py): a user
            -- publishes one of their own resources (profile / VFS folder /
            -- deep-memory pool) to an agent they own or are a member of.
            -- Whether a row is in effect is computed per request (current
            -- owner, live membership, the folder still being the same one).
            CREATE TABLE IF NOT EXISTS context_packs (
                id           TEXT PRIMARY KEY,
                agent_ref    TEXT NOT NULL,
                agent_owner  TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                pack_owner   TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                kind         TEXT NOT NULL,
                resource_id  TEXT NOT NULL DEFAULT '',
                resource_key TEXT NOT NULL DEFAULT '',
                alias        TEXT NOT NULL DEFAULT '',
                slice        TEXT NOT NULL DEFAULT '{}',
                created_at   TEXT NOT NULL,
                UNIQUE(agent_ref, pack_owner, kind, resource_id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_context_packs_alias
                ON context_packs(agent_ref, alias) WHERE alias != '';
            CREATE INDEX IF NOT EXISTS idx_context_packs_owner
                ON context_packs(pack_owner);
            -- Alias tombstones: an alias, once used on an agent, stays that
            -- publisher's there for good. No FK on pack_owner on purpose: the
            -- reservation outlives the user.
            CREATE TABLE IF NOT EXISTS context_pack_aliases (
                agent_ref  TEXT NOT NULL,
                alias      TEXT NOT NULL,
                pack_owner TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (agent_ref, alias)
            );
            -- The deck's shared Telegram bot (telegram_shared.py): one row per
            -- Flight Deck user who asked for it — which of their agents gets
            -- their Telegram messages, whether an admin approved it, and the
            -- Telegram account the one-time link_code bound to it.
            CREATE TABLE IF NOT EXISTS telegram_links (
                user_id       TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                agent_slug    TEXT NOT NULL DEFAULT '',
                agent_name    TEXT NOT NULL DEFAULT '',
                status        TEXT NOT NULL DEFAULT 'pending',
                link_code     TEXT NOT NULL DEFAULT '',
                tg_user_id    TEXT NOT NULL DEFAULT '',
                tg_username   TEXT NOT NULL DEFAULT '',
                tg_first_name TEXT NOT NULL DEFAULT '',
                requested_at  TEXT NOT NULL,
                decided_at    TEXT,
                decided_by    TEXT NOT NULL DEFAULT '',
                linked_at     TEXT,
                updated_at    TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_telegram_links_tg
                ON telegram_links(tg_user_id) WHERE tg_user_id != '';
            CREATE UNIQUE INDEX IF NOT EXISTS idx_telegram_links_code
                ON telegram_links(link_code) WHERE link_code != '';
            -- Telegram groups the shared bot answers in for a user's agent
            -- (telegram_agent_groups.py): who brought it in; their link names
            -- the agent. The title is sealed (fd_crypto) when it can be.
            CREATE TABLE IF NOT EXISTS telegram_agent_groups (
                chat_id  TEXT PRIMARY KEY,
                user_id  TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title    TEXT NOT NULL DEFAULT '',
                added_at TEXT NOT NULL
            );
            -- Telegram inbox (telegram_inbox.py): a read-only copy of a user's
            -- Telegram chats that their agents read — private chats through a
            -- Chat Automation (business) connection to the shared bot, groups
            -- the bot was added to. Text and names are sealed (fd_crypto);
            -- only ids and times are plain.
            CREATE TABLE IF NOT EXISTS telegram_inbox_settings (
                user_id        TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                enabled        INTEGER NOT NULL DEFAULT 0,
                consented_at   TEXT,
                retention_days INTEGER NOT NULL DEFAULT 30,
                updated_at     TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS telegram_inbox_connections (
                connection_id TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                tg_user_id    TEXT NOT NULL,
                enabled       INTEGER NOT NULL DEFAULT 1,
                rights        TEXT NOT NULL DEFAULT '{}',
                connected_at  TEXT NOT NULL,
                updated_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS telegram_inbox_chats (
                user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                chat_id    TEXT NOT NULL,
                kind       TEXT NOT NULL,
                title      TEXT NOT NULL DEFAULT '',
                active     INTEGER NOT NULL DEFAULT 1,
                last_at    INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                PRIMARY KEY (user_id, chat_id)
            );
            CREATE INDEX IF NOT EXISTS idx_tg_inbox_chats_chat
                ON telegram_inbox_chats(chat_id);
            CREATE TABLE IF NOT EXISTS telegram_inbox_messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                chat_id    TEXT NOT NULL,
                message_id INTEGER NOT NULL,
                sender_id  TEXT NOT NULL DEFAULT '',
                outgoing   INTEGER NOT NULL DEFAULT 0,
                sent_at    INTEGER NOT NULL,
                body       TEXT NOT NULL DEFAULT '',
                reply_to   INTEGER,
                edited_at  INTEGER,
                deleted    INTEGER NOT NULL DEFAULT 0,
                UNIQUE (user_id, chat_id, message_id)
            );
            CREATE INDEX IF NOT EXISTS idx_tg_inbox_msgs_time
                ON telegram_inbox_messages(user_id, sent_at);
            CREATE INDEX IF NOT EXISTS idx_tg_inbox_msgs_chat
                ON telegram_inbox_messages(user_id, chat_id, sent_at);
            -- How far each agent has looked (telegram_inbox.news): the newest
            -- message id it was told about — chat_id '' for every chat, a chat's
            -- own row when the agent read that chat further.
            CREATE TABLE IF NOT EXISTS telegram_inbox_cursors (
                user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                agent_slug TEXT NOT NULL,
                chat_id    TEXT NOT NULL DEFAULT '',
                last_id    INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, agent_slug, chat_id)
            );
        """)
        # Lightweight migrations: add columns introduced after a table first shipped.
        for table, col, ddl in [
            ("basna_runs", "actions", "TEXT NOT NULL DEFAULT '[]'"),
            ("basna_runs", "human_success", "INTEGER"),
            ("basna_runs", "human_feedback_at", "TEXT"),
            ("basna_sessions", "progress", "TEXT NOT NULL DEFAULT '[]'"),
            ("basna_sessions", "files", "TEXT NOT NULL DEFAULT '[]'"),
            ("basna_sessions", "analysis", "TEXT NOT NULL DEFAULT '{}'"),
            ("basna_sessions", "title", "TEXT NOT NULL DEFAULT ''"),
            ("gmail_sends", "status", "TEXT NOT NULL DEFAULT 'sent'"),
            ("gmail_sends", "attachments", "TEXT NOT NULL DEFAULT ''"),
            ("gmail_sends", "origin", "TEXT NOT NULL DEFAULT 'agent'"),
            ("gmail_sends", "topic_id", "TEXT NOT NULL DEFAULT ''"),
            ("gmail_sends", "in_reply_to_message_id", "TEXT NOT NULL DEFAULT ''"),
            ("agent_links", "agent_may_send", "INTEGER NOT NULL DEFAULT 1"),
            ("agent_links", "webhook_url", "TEXT NOT NULL DEFAULT ''"),
            ("agent_links", "webhook_secret", "TEXT NOT NULL DEFAULT ''"),
            ("agent_links", "direction", "TEXT NOT NULL DEFAULT 'inbound'"),
            ("agent_links", "remote_url", "TEXT NOT NULL DEFAULT ''"),
            ("agent_links", "remote_key", "TEXT NOT NULL DEFAULT ''"),
            ("daily_usage", "byok_tokens", "INTEGER NOT NULL DEFAULT 0"),
        ]:
            try:
                await self._db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
            except Exception:
                pass  # column already exists
        # The "pro" plan is "premium" now.
        await self._db.execute(
            "UPDATE users SET metadata = json_set(metadata, '$.plan', 'premium')"
            " WHERE json_valid(metadata) AND json_extract(metadata, '$.plan') = 'pro'")
        await self._db.commit()

    # ── Users ────────────────────────────────────────────────────────

    async def create_user(
        self, email: str, password_hash: str, display_name: str = "",
        role: str = "user", metadata: dict | None = None,
    ) -> dict:
        now = _utcnow()
        uid = _uuid()
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at, metadata)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (uid, email.lower().strip(), password_hash, display_name, role, now, now,
             json.dumps(metadata or {})),
        )
        await self._db.commit()
        return {"id": uid, "email": email.lower().strip(), "display_name": display_name,
                "role": role, "created_at": now}

    async def ensure_local_user(self, user_id: str, password_hash: str) -> None:
        """A row for the synthetic local user of a deck without sign-in, so
        what it keeps per user (settings, archetypes, shares…) can be stored:
        those tables reference users(id). Nothing reads its role or password
        while sign-in is off; it is a "user" with a random password, so turning
        sign-in on later leaves no dormant admin and no way to sign in as it."""
        now = _utcnow()
        assert self._db is not None
        await self._db.execute(
            "INSERT OR IGNORE INTO users (id, email, password_hash, display_name, role, created_at, updated_at, metadata)"
            " VALUES (?, ?, ?, ?, 'user', ?, ?, ?)",
            (user_id, user_id, password_hash, "Local", now, now, json.dumps({"local": True})),
        )
        await self._db.commit()

    async def get_user_by_email(self, email: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_user_by_id(self, user_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT id, email, display_name, role, created_at, updated_at, metadata"
            " FROM users WHERE id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def update_user(self, user_id: str, **fields) -> bool:
        assert self._db is not None
        allowed = {"email", "password_hash", "display_name", "role", "metadata"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        updates["updated_at"] = _utcnow()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [user_id]
        await self._db.execute(f"UPDATE users SET {set_clause} WHERE id = ?", vals)
        await self._db.commit()
        return True

    async def count_users(self) -> int:
        assert self._db is not None
        async with self._db.execute("SELECT COUNT(*) FROM users") as cur:
            row = await cur.fetchone()
            return row[0] if row else 0

    # ── Waitlist ─────────────────────────────────────────────────────

    async def create_waitlist_entry(
        self, *, email: str, display_name: str, password_hash: str,
        referrer_name: str, referrer_email: str, cohort: str,
    ) -> dict:
        """A pending application. Raises ``sqlite3.IntegrityError`` when this
        email already has one pending."""
        eid = _uuid()
        now = _utcnow()
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO waitlist (id, email, display_name, password_hash, referrer_name,"
            " referrer_email, cohort, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
            (eid, email.lower().strip(), display_name, password_hash, referrer_name,
             referrer_email.lower().strip(), cohort, now),
        )
        await self._db.commit()
        return {"id": eid, "email": email.lower().strip(), "status": "pending", "created_at": now}

    async def get_waitlist_entry(self, entry_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute("SELECT * FROM waitlist WHERE id = ?", (entry_id,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_pending_waitlist_entry(self, email: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM waitlist WHERE email = ? AND status = 'pending'",
            (email.lower().strip(),),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def list_waitlist(self, status: str | None = None, limit: int = 500) -> list[dict]:
        """Newest first; *status* None = every application. Never the hash."""
        assert self._db is not None
        cols = ("id, email, display_name, referrer_name, referrer_email, cohort, status,"
                " created_at, decided_at, decided_by, user_id")
        if status:
            sql, args = (f"SELECT {cols} FROM waitlist WHERE status = ?"
                         " ORDER BY created_at DESC LIMIT ?", (status, limit))
        else:
            sql, args = f"SELECT {cols} FROM waitlist ORDER BY created_at DESC LIMIT ?", (limit,)
        async with self._db.execute(sql, args) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def count_waitlist(self, status: str = "pending") -> int:
        assert self._db is not None
        async with self._db.execute(
            "SELECT COUNT(*) FROM waitlist WHERE status = ?", (status,)
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else 0

    async def decide_waitlist_entry(
        self, entry_id: str, status: str, decided_by: str, user_id: str = "",
    ) -> bool:
        """Move a pending application to *status* (approved | rejected) and drop
        its password hash. False when it was not pending (already decided)."""
        assert self._db is not None
        cur = await self._db.execute(
            "UPDATE waitlist SET status = ?, decided_at = ?, decided_by = ?, user_id = ?,"
            " password_hash = '' WHERE id = ? AND status = 'pending'",
            (status, _utcnow(), decided_by, user_id, entry_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def delete_waitlist_entry(self, entry_id: str) -> bool:
        assert self._db is not None
        cur = await self._db.execute("DELETE FROM waitlist WHERE id = ?", (entry_id,))
        await self._db.commit()
        return cur.rowcount > 0

    # ── Invitations ──────────────────────────────────────────────────

    _INVITE_COLS = ("id, email, display_name, role, kiosk, note, status, invited_by, created_at,"
                    " expires_at, sent_via, sent_from, sent_at, accepted_at, user_id")

    async def create_invite(
        self, *, email: str, display_name: str, role: str, kiosk: bool | None,
        note: str, token_hash: str, invited_by: str, expires_at: str,
    ) -> dict:
        """A pending invitation. Raises ``sqlite3.IntegrityError`` when this
        email already has one pending."""
        iid = _uuid()
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO invites (id, email, display_name, role, kiosk, note, token_hash,"
            " status, invited_by, created_at, expires_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)",
            (iid, email.lower().strip(), display_name, role,
             None if kiosk is None else int(kiosk), note, token_hash, invited_by,
             _utcnow(), expires_at),
        )
        await self._db.commit()
        return await self.get_invite(iid) or {}

    async def get_invite(self, invite_id: str) -> dict | None:
        """One invitation, never its token hash."""
        assert self._db is not None
        async with self._db.execute(
            f"SELECT {self._INVITE_COLS} FROM invites WHERE id = ?", (invite_id,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_invite_by_token_hash(self, token_hash: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            f"SELECT {self._INVITE_COLS} FROM invites WHERE token_hash = ?", (token_hash,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_pending_invite(self, email: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            f"SELECT {self._INVITE_COLS} FROM invites WHERE email = ? AND status = 'pending'",
            (email.lower().strip(),),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def list_invites(self, status: str | None = None, limit: int = 500) -> list[dict]:
        """Newest first; *status* None = every invitation."""
        assert self._db is not None
        if status:
            sql, args = (f"SELECT {self._INVITE_COLS} FROM invites WHERE status = ?"
                         " ORDER BY created_at DESC LIMIT ?", (status, limit))
        else:
            sql, args = (f"SELECT {self._INVITE_COLS} FROM invites"
                         " ORDER BY created_at DESC LIMIT ?", (limit,))
        async with self._db.execute(sql, args) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def renew_invite(self, invite_id: str, token_hash: str, expires_at: str) -> bool:
        """A new link (the old one stops working) and expiry for a pending
        invitation. False when it isn't pending."""
        assert self._db is not None
        cur = await self._db.execute(
            "UPDATE invites SET token_hash = ?, expires_at = ? WHERE id = ? AND status = 'pending'",
            (token_hash, expires_at, invite_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def mark_invite_sent(self, invite_id: str, via: str, sent_from: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE invites SET sent_via = ?, sent_from = ?, sent_at = ? WHERE id = ?",
            (via, sent_from, _utcnow(), invite_id),
        )
        await self._db.commit()

    async def claim_invite(self, invite_id: str) -> bool:
        """Mark a pending invitation accepted — before its account is made, so
        two tabs can't both use one link. False when it was not pending (used
        a moment ago, or removed)."""
        assert self._db is not None
        cur = await self._db.execute(
            "UPDATE invites SET status = 'accepted', accepted_at = ?"
            " WHERE id = ? AND status = 'pending'",
            (_utcnow(), invite_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def finish_invite(self, invite_id: str, user_id: str) -> None:
        """The account a claimed invitation made."""
        assert self._db is not None
        await self._db.execute("UPDATE invites SET user_id = ? WHERE id = ?", (user_id, invite_id))
        await self._db.commit()

    async def release_invite(self, invite_id: str) -> None:
        """Back to pending: the account could not be made after the claim."""
        assert self._db is not None
        await self._db.execute(
            "UPDATE invites SET status = 'pending', accepted_at = ''"
            " WHERE id = ? AND status = 'accepted' AND user_id = ''",
            (invite_id,),
        )
        await self._db.commit()

    async def delete_invite(self, invite_id: str) -> bool:
        assert self._db is not None
        cur = await self._db.execute("DELETE FROM invites WHERE id = ?", (invite_id,))
        await self._db.commit()
        return cur.rowcount > 0

    # ── Refresh sessions ─────────────────────────────────────────────

    async def create_refresh_session(
        self, user_id: str, refresh_token_hash: str, expires_at: str,
    ) -> str:
        sid = _uuid()
        now = _utcnow()
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO user_sessions (id, user_id, refresh_token_hash, expires_at, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (sid, user_id, refresh_token_hash, expires_at, now),
        )
        await self._db.commit()
        return sid

    async def get_refresh_session(self, session_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM user_sessions WHERE id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def delete_refresh_session(self, session_id: str) -> None:
        assert self._db is not None
        await self._db.execute("DELETE FROM user_sessions WHERE id = ?", (session_id,))
        await self._db.commit()

    async def delete_user_refresh_sessions(
        self, user_id: str, keep_token_hash: str | None = None,
    ) -> int:
        """Sign *user_id* out everywhere — except the session whose refresh
        token hashes to *keep_token_hash* (the caller's own). Returns how many
        sessions went."""
        assert self._db is not None
        if keep_token_hash:
            cur = await self._db.execute(
                "DELETE FROM user_sessions WHERE user_id = ? AND refresh_token_hash != ?",
                (user_id, keep_token_hash),
            )
        else:
            cur = await self._db.execute("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
        await self._db.commit()
        return cur.rowcount

    async def cleanup_expired_sessions(self) -> None:
        assert self._db is not None
        now = _utcnow()
        await self._db.execute("DELETE FROM user_sessions WHERE expires_at < ?", (now,))
        await self._db.commit()

    # ── Personal access tokens (long-lived; for MCP clients) ──────────
    # Only the sha256 hash is stored; the raw token is shown to the user once
    # at creation. A remote MCP client presents it as a Bearer credential.

    async def create_pat(self, user_id: str, token_hash: str, name: str = "") -> str:
        assert self._db is not None
        pid = _uuid()
        await self._db.execute(
            "INSERT INTO personal_access_tokens (id, user_id, token_hash, name, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (pid, user_id, token_hash, name, _utcnow()),
        )
        await self._db.commit()
        return pid

    async def get_pat_by_hash(self, token_hash: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM personal_access_tokens WHERE token_hash = ?", (token_hash,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def list_pats(self, user_id: str) -> list[dict]:
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT id, name, created_at, last_used_at FROM personal_access_tokens"
            " WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
        return [dict(r) for r in rows]

    async def touch_pat(self, pat_id: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE personal_access_tokens SET last_used_at = ? WHERE id = ?",
            (_utcnow(), pat_id),
        )
        await self._db.commit()

    async def revoke_pat(self, pat_id: str, user_id: str) -> bool:
        assert self._db is not None
        async with self._db.execute(
            "DELETE FROM personal_access_tokens WHERE id = ? AND user_id = ?",
            (pat_id, user_id),
        ) as cur:
            await self._db.commit()
            return (cur.rowcount or 0) > 0

    # ── Agent links (agent_link.py) ───────────────────────────────────
    # A connected system's key is shown once, at creation or rotation; only
    # its hash is kept.

    _AGENT_LINK_FIELDS = ("name", "agent_slug", "instructions", "agent_session_id",
                          "agent_may_send", "webhook_url", "webhook_secret",
                          "direction", "remote_url", "remote_key")
    LINK_EVENTS_KEPT_DAYS = 7
    LINK_EVENTS_KEPT = 500  # per link

    async def create_agent_link(self, user_id: str, *, name: str, agent_slug: str,
                                instructions: str, key_hash: str, key_hint: str,
                                agent_may_send: bool = True, webhook_url: str = "",
                                webhook_secret: str = "", direction: str = "inbound",
                                remote_url: str = "", remote_key: str = "") -> dict:
        assert self._db is not None
        now = _utcnow()
        row = {"id": _uuid(), "user_id": user_id, "name": name, "agent_slug": agent_slug,
               "instructions": instructions, "key_hash": key_hash, "key_hint": key_hint,
               "agent_session_id": "", "agent_may_send": int(agent_may_send), "webhook_url": webhook_url,
               "webhook_secret": webhook_secret, "direction": direction, "remote_url": remote_url,
               "remote_key": remote_key, "created_at": now, "updated_at": now, "last_used_at": None}
        await self._db.execute(
            "INSERT INTO agent_links (id, user_id, name, agent_slug, instructions, key_hash, key_hint,"
            " agent_session_id, agent_may_send, webhook_url, webhook_secret, direction, remote_url,"
            " remote_key, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?)",
            (row["id"], user_id, name, agent_slug, instructions, key_hash, key_hint,
             int(agent_may_send), webhook_url, webhook_secret, direction, remote_url, remote_key, now, now),
        )
        await self._db.commit()
        return row

    async def get_agent_link(self, link_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute("SELECT * FROM agent_links WHERE id = ?", (link_id,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_agent_link_by_key(self, key_hash: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute("SELECT * FROM agent_links WHERE key_hash = ?", (key_hash,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def list_agent_links(self, user_id: str) -> list[dict]:
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT * FROM agent_links WHERE user_id = ? ORDER BY created_at DESC", (user_id,),
        )
        return [dict(r) for r in rows]

    async def list_outbound_links(self) -> list[dict]:
        """Every link Flight Deck connects out from (all owners), for start-up."""
        assert self._db is not None
        rows = await self._db.execute_fetchall("SELECT * FROM agent_links WHERE direction = 'outbound'")
        return [dict(r) for r in rows]

    async def update_agent_link(self, link_id: str, user_id: str, **fields: str | int) -> bool:
        """Change a link's name / agent / instructions / session / sending and
        webhook settings (other fields are ignored)."""
        assert self._db is not None
        cols = [k for k in fields if k in self._AGENT_LINK_FIELDS]
        if not cols:
            return False
        sets = ", ".join(f"{c} = ?" for c in cols)
        async with self._db.execute(
            f"UPDATE agent_links SET {sets}, updated_at = ? WHERE id = ? AND user_id = ?",
            (*[int(fields[c]) if isinstance(fields[c], bool) else fields[c] for c in cols],
             _utcnow(), link_id, user_id),
        ) as cur:
            await self._db.commit()
            return (cur.rowcount or 0) > 0

    async def set_agent_link_key(self, link_id: str, user_id: str, key_hash: str, key_hint: str) -> bool:
        assert self._db is not None
        async with self._db.execute(
            "UPDATE agent_links SET key_hash = ?, key_hint = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (key_hash, key_hint, _utcnow(), link_id, user_id),
        ) as cur:
            await self._db.commit()
            return (cur.rowcount or 0) > 0

    async def touch_agent_link(self, link_id: str) -> None:
        assert self._db is not None
        await self._db.execute("UPDATE agent_links SET last_used_at = ? WHERE id = ?", (_utcnow(), link_id))
        await self._db.commit()

    async def delete_agent_link(self, link_id: str, user_id: str) -> bool:
        assert self._db is not None
        async with self._db.execute(
            "DELETE FROM agent_links WHERE id = ? AND user_id = ?", (link_id, user_id),
        ) as cur:
            await self._db.commit()
            return (cur.rowcount or 0) > 0

    # What Flight Deck keeps for a linked system: events (a week, the newest
    # 500 per link) and the files they name.

    @staticmethod
    def _link_event(row: Any) -> dict:
        event = dict(row)
        try:
            event["body"] = json.loads(event.get("body") or "{}")
        except ValueError:
            event["body"] = {}
        return event

    async def add_link_event(self, link_id: str, kind: str, re: str, body: dict) -> dict:
        assert self._db is not None
        now = _utcnow()
        async with self._db.execute(
            "INSERT INTO agent_link_events (link_id, kind, re, body, created_at) VALUES (?, ?, ?, ?, ?)",
            (link_id, kind, re, json.dumps(body, ensure_ascii=False), now),
        ) as cur:
            seq = cur.lastrowid
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.LINK_EVENTS_KEPT_DAYS)).isoformat()
        await self._db.execute(
            "DELETE FROM agent_link_events WHERE link_id = ? AND (created_at < ? OR seq <= ?)",
            (link_id, cutoff, (seq or 0) - self.LINK_EVENTS_KEPT),
        )
        await self._db.execute("DELETE FROM agent_link_files WHERE link_id = ? AND created_at < ?", (link_id, cutoff))
        await self._db.commit()
        return {"seq": seq, "link_id": link_id, "kind": kind, "re": re, "body": body,
                "delivered": "", "created_at": now}

    async def link_events_after(self, link_id: str, after: int, limit: int = 50) -> list[dict]:
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT * FROM agent_link_events WHERE link_id = ? AND seq > ? ORDER BY seq LIMIT ?",
            (link_id, after, limit),
        )
        return [self._link_event(r) for r in rows]

    async def link_event_for(self, link_id: str, re: str) -> dict | None:
        """The newest event answering message *re*."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM agent_link_events WHERE link_id = ? AND re = ? ORDER BY seq DESC LIMIT 1",
            (link_id, re),
        ) as cur:
            row = await cur.fetchone()
            return self._link_event(row) if row else None

    async def undelivered_link_events(self, link_id: str, kind: str) -> list[dict]:
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT * FROM agent_link_events WHERE link_id = ? AND kind = ? AND delivered = ''"
            " ORDER BY seq", (link_id, kind),
        )
        return [self._link_event(r) for r in rows]

    async def mark_link_event(self, seq: int, via: str) -> None:
        """How an event reached the system (the first way only)."""
        assert self._db is not None
        await self._db.execute(
            "UPDATE agent_link_events SET delivered = ? WHERE seq = ? AND delivered = ''", (via, seq),
        )
        await self._db.commit()

    async def add_link_file(self, link_id: str, *, agent_slug: str, path: str, name: str, kind: str) -> str:
        assert self._db is not None
        file_id = "f_" + secrets.token_urlsafe(18)
        await self._db.execute(
            "INSERT INTO agent_link_files (id, link_id, agent_slug, path, name, kind, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)", (file_id, link_id, agent_slug, path, name, kind, _utcnow()),
        )
        await self._db.commit()
        return file_id

    async def get_link_file(self, file_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute("SELECT * FROM agent_link_files WHERE id = ?", (file_id,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    # ── OAuth 2.1 (inbound MCP; claude.ai custom connectors) ──────────

    async def create_oauth_client(self, client_id: str, client_name: str, redirect_uris: list[str]) -> None:
        assert self._db is not None
        import json as _json
        await self._db.execute(
            "INSERT INTO oauth_clients (id, client_name, redirect_uris, created_at)"
            " VALUES (?, ?, ?, ?)",
            (client_id, client_name, _json.dumps(redirect_uris), _utcnow()),
        )
        await self._db.commit()

    async def get_oauth_client(self, client_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM oauth_clients WHERE id = ?", (client_id,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        import json as _json
        d = dict(row)
        try:
            d["redirect_uris"] = _json.loads(d.get("redirect_uris") or "[]")
        except (_json.JSONDecodeError, TypeError):
            d["redirect_uris"] = []
        return d

    async def create_oauth_code(
        self, code_hash: str, client_id: str, user_id: str, redirect_uri: str,
        code_challenge: str, scope: str, resource: str | None, expires_at: str,
    ) -> None:
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO oauth_codes (id, code_hash, client_id, user_id, redirect_uri,"
            " code_challenge, scope, resource, created_at, expires_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_uuid(), code_hash, client_id, user_id, redirect_uri, code_challenge,
             scope, resource, _utcnow(), expires_at),
        )
        await self._db.commit()

    async def get_oauth_code_by_hash(self, code_hash: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM oauth_codes WHERE code_hash = ?", (code_hash,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def burn_oauth_code(self, code_id: str) -> bool:
        """Atomically mark a code used; returns True only for the first caller."""
        assert self._db is not None
        async with self._db.execute(
            "UPDATE oauth_codes SET used_at = ? WHERE id = ? AND used_at IS NULL",
            (_utcnow(), code_id),
        ) as cur:
            await self._db.commit()
            return (cur.rowcount or 0) == 1

    async def create_oauth_refresh(
        self, token_hash: str, client_id: str, client_name: str, user_id: str, scope: str,
    ) -> str:
        assert self._db is not None
        rid = _uuid()
        await self._db.execute(
            "INSERT INTO oauth_refresh_tokens (id, token_hash, client_id, client_name,"
            " user_id, scope, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rid, token_hash, client_id, client_name, user_id, scope, _utcnow()),
        )
        await self._db.commit()
        return rid

    async def get_oauth_refresh_by_hash(self, token_hash: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM oauth_refresh_tokens WHERE token_hash = ? AND revoked_at IS NULL",
            (token_hash,),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def rotate_oauth_refresh(self, refresh_id: str) -> bool:
        """Atomically revoke a refresh token (single-use rotation)."""
        assert self._db is not None
        async with self._db.execute(
            "UPDATE oauth_refresh_tokens SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
            (_utcnow(), refresh_id),
        ) as cur:
            await self._db.commit()
            return (cur.rowcount or 0) == 1

    async def list_oauth_grants(self, user_id: str) -> list[dict]:
        """Distinct connected apps for a user (non-revoked refresh tokens)."""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT client_id, MAX(client_name) AS client_name,"
            " MIN(created_at) AS connected_at, MAX(last_used_at) AS last_used_at,"
            " COUNT(*) AS tokens"
            " FROM oauth_refresh_tokens WHERE user_id = ? AND revoked_at IS NULL"
            " GROUP BY client_id ORDER BY connected_at DESC",
            (user_id,),
        )
        return [dict(r) for r in rows]

    async def revoke_oauth_grant(self, user_id: str, client_id: str) -> int:
        """Revoke every refresh token for one connected app (disconnect it)."""
        assert self._db is not None
        async with self._db.execute(
            "UPDATE oauth_refresh_tokens SET revoked_at = ?"
            " WHERE user_id = ? AND client_id = ? AND revoked_at IS NULL",
            (_utcnow(), user_id, client_id),
        ) as cur:
            await self._db.commit()
            return cur.rowcount or 0

    async def prune_oauth(self) -> None:
        """Best-effort GC: drop expired/used codes."""
        assert self._db is not None
        now = _utcnow()
        try:
            await self._db.execute(
                "DELETE FROM oauth_codes WHERE expires_at < ? OR used_at IS NOT NULL", (now,))
            await self._db.commit()
        except Exception:
            pass

    # ── User settings ────────────────────────────────────────────────

    async def get_all_settings(self, user_id: str) -> dict[str, str]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT key, value FROM user_settings WHERE user_id = ?", (user_id,)
        ) as cur:
            rows = await cur.fetchall()
            return {r["key"]: r["value"] for r in rows}

    async def get_setting(self, user_id: str, key: str) -> str | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT value FROM user_settings WHERE user_id = ? AND key = ?",
            (user_id, key),
        ) as cur:
            row = await cur.fetchone()
            return row["value"] if row else None

    async def set_settings(self, user_id: str, settings: dict[str, str]) -> None:
        assert self._db is not None
        now = _utcnow()
        for key, value in settings.items():
            await self._db.execute(
                "INSERT INTO user_settings (user_id, key, value, updated_at)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value,"
                " updated_at = excluded.updated_at",
                (user_id, key, value, now),
            )
        await self._db.commit()

    # ── System settings (no FK, for global config) ────────────────

    async def get_system_setting(self, key: str) -> str | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT value FROM system_settings WHERE key = ?", (key,)
        ) as cur:
            row = await cur.fetchone()
            return row["value"] if row else None

    async def set_system_setting(self, key: str, value: str) -> None:
        assert self._db is not None
        now = _utcnow()
        await self._db.execute(
            "INSERT INTO system_settings (key, value, updated_at)"
            " VALUES (?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value,"
            " updated_at = excluded.updated_at",
            (key, value, now),
        )
        await self._db.commit()

    async def get_all_system_settings(self) -> dict[str, str]:
        assert self._db is not None
        async with self._db.execute("SELECT key, value FROM system_settings") as cur:
            rows = await cur.fetchall()
            return {r["key"]: r["value"] for r in rows}

    async def delete_setting(self, user_id: str, key: str) -> bool:
        assert self._db is not None
        cur = await self._db.execute(
            "DELETE FROM user_settings WHERE user_id = ? AND key = ?", (user_id, key)
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def delete_setting_for_all_users(self, key: str) -> int:
        """Drop one setting key from every user (e.g. a removed shared agent's
        per-member opt-in). Returns how many rows went."""
        assert self._db is not None
        cur = await self._db.execute("DELETE FROM user_settings WHERE key = ?", (key,))
        await self._db.commit()
        return int(cur.rowcount or 0)

    async def users_with_setting(self, key: str) -> dict[str, str]:
        """Every user holding setting *key*: ``{user_id: value}``."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT user_id, value FROM user_settings WHERE key = ?", (key,),
        ) as cur:
            return {r["user_id"]: r["value"] for r in await cur.fetchall()}

    # ── Chat sessions ────────────────────────────────────────────────

    async def list_chat_sessions(self, user_id: str) -> list[dict]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def get_chat_session(self, session_id: str, user_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM chat_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def upsert_chat_session(
        self, session_id: str, user_id: str, agent_id: str = "",
        agent_name: str = "",
    ) -> dict:
        assert self._db is not None
        now = _utcnow()
        await self._db.execute(
            "INSERT INTO chat_sessions (id, user_id, agent_id, agent_name, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET agent_name = excluded.agent_name,"
            " updated_at = excluded.updated_at",
            (session_id, user_id, agent_id, agent_name, now, now),
        )
        await self._db.commit()
        return {"id": session_id, "user_id": user_id, "agent_id": agent_id,
                "agent_name": agent_name, "updated_at": now}

    async def delete_chat_session(self, session_id: str, user_id: str) -> bool:
        assert self._db is not None
        cur = await self._db.execute(
            "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def delete_chat_sessions_for_agent(self, agent_id: str) -> int:
        """Drop every chat log with one agent (all its lanes, every user) —
        for an agent removed for good. Messages go with them (cascade)."""
        assert self._db is not None
        if not agent_id:
            return 0
        lanes = f"{agent_id}::"
        cur = await self._db.execute(
            "DELETE FROM chat_sessions WHERE agent_id = ? OR id = ? OR substr(id, 1, ?) = ?",
            (agent_id, agent_id, len(lanes), lanes),
        )
        await self._db.commit()
        return int(cur.rowcount or 0)

    async def get_chat_messages(
        self, session_id: str, user_id: str,
        limit: int = 100, before_id: int | None = None, since_id: int | None = None,
    ) -> list[dict]:
        """Rows of one chat log, oldest first: the newest ``limit`` (those
        before ``before_id`` when given), or with ``since_id`` the log from
        that row on (a topic divider the reader jumped to)."""
        assert self._db is not None
        # Verify ownership
        sess = await self.get_chat_session(session_id, user_id)
        if not sess:
            return []
        if since_id is not None:
            async with self._db.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? AND id >= ?"
                " ORDER BY id ASC LIMIT ?",
                (session_id, int(since_id), max(1, min(int(limit), 2000))),
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]
        query = "SELECT * FROM chat_messages WHERE session_id = ?"
        params: list = [session_id]
        if before_id is not None:
            query += " AND id < ?"
            params.append(before_id)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        async with self._db.execute(query, params) as cur:
            rows = [dict(r) for r in await cur.fetchall()]
        rows.reverse()  # Return chronological order
        return rows

    async def list_chat_topics(self, session_id: str, user_id: str) -> list[dict]:
        """The topic dividers of one chat log, oldest first — ``system`` rows
        whose metadata carries a ``topic`` object (the chat timeline's "New
        topic" / "Back on …" / "Context cleared" marks) — each with the first
        user message after it and the number of user+assistant turns before
        the next divider. Unknown or foreign session: ``[]``."""
        assert self._db is not None
        sess = await self.get_chat_session(session_id, user_id)
        if not sess:
            return []
        async with self._db.execute(
            "SELECT id, metadata, created_at FROM chat_messages"
            " WHERE session_id = ? AND role = 'system' AND metadata LIKE '%\"topic\"%'"
            " ORDER BY id ASC",
            (session_id,),
        ) as cur:
            divider_rows = [dict(r) for r in await cur.fetchall()]
        dividers: list[tuple[int, dict, str]] = []
        for r in divider_rows:
            try:
                meta = json.loads(r.get("metadata") or "{}")
            except (TypeError, ValueError):
                continue
            topic = meta.get("topic") if isinstance(meta, dict) else None
            if isinstance(topic, dict):
                dividers.append((int(r["id"]), topic, str(r.get("created_at") or "")))
        if not dividers:
            return []
        async with self._db.execute(
            "SELECT id, role, content FROM chat_messages"
            " WHERE session_id = ? AND role IN ('user', 'assistant') AND id > ?"
            " ORDER BY id ASC",
            (session_id, dividers[0][0]),
        ) as cur:
            turns = [dict(r) for r in await cur.fetchall()]
        out: list[dict] = []
        ti = 0
        for i, (row_id, topic, started_at) in enumerate(dividers):
            end = dividers[i + 1][0] if i + 1 < len(dividers) else None
            count = 0
            first_user = ""
            while ti < len(turns) and (end is None or int(turns[ti]["id"]) < end):
                t = turns[ti]
                ti += 1
                if int(t["id"]) <= row_id:
                    continue
                content = str(t.get("content") or "")
                # A typed slash command (/clear, /session switch …) is a
                # control line, not a turn: it names no topic and counts for none.
                if t.get("role") == "user" and content.lstrip().startswith("/"):
                    continue
                count += 1
                if not first_user and t.get("role") == "user":
                    first_user = content
            out.append({
                "row_id": row_id,
                "agent_session_id": str(topic.get("agentSessionId") or ""),
                "action": str(topic.get("action") or "new"),
                "title": str(topic.get("title") or ""),
                "started_at": started_at,
                "first_user_message": first_user,
                "message_count": count,
            })
        return out

    async def delete_chat_message(self, session_id: str, user_id: str, row_id: int) -> bool:
        """Remove one USER row of the user's chat log — a composer echo the
        client withdrew (a "new topic:" line re-sent trimmed). Never a reply
        or a divider."""
        assert self._db is not None
        sess = await self.get_chat_session(session_id, user_id)
        if not sess:
            return False
        cur = await self._db.execute(
            "DELETE FROM chat_messages WHERE id = ? AND session_id = ? AND role = 'user'",
            (int(row_id), session_id),
        )
        await self._db.commit()
        return (cur.rowcount or 0) > 0

    async def set_chat_topic_title(
        self, session_id: str, user_id: str, row_id: int, title: str,
    ) -> bool:
        """Rename one topic divider of the user's chat log (its
        ``metadata.topic.title``). False when the session isn't the user's or
        the row isn't a divider."""
        assert self._db is not None
        sess = await self.get_chat_session(session_id, user_id)
        if not sess:
            return False
        async with self._db.execute(
            "SELECT id, role, metadata FROM chat_messages WHERE id = ? AND session_id = ?",
            (int(row_id), session_id),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return False
        row = dict(row)
        if row.get("role") != "system":
            return False
        try:
            meta = json.loads(row.get("metadata") or "{}")
        except (TypeError, ValueError):
            return False
        topic = meta.get("topic") if isinstance(meta, dict) else None
        if not isinstance(topic, dict):
            return False
        topic["title"] = " ".join(str(title or "").split())[:120]
        meta["topic"] = topic
        await self._db.execute(
            "UPDATE chat_messages SET metadata = ? WHERE id = ?",
            (json.dumps(meta), int(row_id)),
        )
        await self._db.commit()
        return True

    async def add_chat_messages(
        self, session_id: str, user_id: str, messages: list[dict],
    ) -> list[int]:
        assert self._db is not None
        sess = await self.get_chat_session(session_id, user_id)
        if not sess:
            return []
        now = _utcnow()
        ids = []
        for msg in messages:
            cur = await self._db.execute(
                "INSERT INTO chat_messages (session_id, role, content, metadata, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (session_id, msg.get("role", ""), msg.get("content", ""),
                 msg.get("metadata", "{}"), now),
            )
            ids.append(cur.lastrowid)
        # Touch session
        await self._db.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (now, session_id)
        )
        await self._db.commit()
        return ids

    # ── Usage logs ───────────────────────────────────────────────────

    async def log_usage(
        self, user_id: str, event_type: str, detail: str = "{}",
    ) -> int:
        assert self._db is not None
        now = _utcnow()
        cur = await self._db.execute(
            "INSERT INTO usage_logs (user_id, event_type, detail, created_at)"
            " VALUES (?, ?, ?, ?)",
            (user_id, event_type, detail, now),
        )
        await self._db.commit()
        return cur.lastrowid or 0

    async def get_usage_logs(
        self, user_id: str | None = None, event_type: str | None = None,
        since: str | None = None, limit: int = 200,
    ) -> list[dict]:
        assert self._db is not None
        query = "SELECT * FROM usage_logs WHERE 1=1"
        params: list = []
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if since:
            query += " AND created_at >= ?"
            params.append(since)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        async with self._db.execute(query, params) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def add_daily_usage(
        self, user_id: str, day: str, *, messages: int = 0, tokens: int = 0,
        byok_tokens: int = 0,
    ) -> dict:
        """Add to *user_id*'s counters for *day*; returns the new totals.
        ``byok_tokens`` were spent on the user's own keys (not the plan's)."""
        assert self._db is not None
        now = _utcnow()
        await self._db.execute(
            "INSERT INTO daily_usage (user_id, day, messages, tokens, byok_tokens, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(user_id, day) DO UPDATE SET"
            " messages = messages + excluded.messages,"
            " tokens = tokens + excluded.tokens,"
            " byok_tokens = byok_tokens + excluded.byok_tokens,"
            " updated_at = excluded.updated_at",
            (user_id, day, max(0, int(messages)), max(0, int(tokens)),
             max(0, int(byok_tokens)), now),
        )
        await self._db.commit()
        return await self.get_daily_usage(user_id, day)

    async def get_daily_usage(self, user_id: str, day: str) -> dict:
        """*user_id*'s counters for *day* (zeros when nothing was spent)."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT messages, tokens, byok_tokens FROM daily_usage WHERE user_id = ? AND day = ?",
            (user_id, day),
        ) as cur:
            row = await cur.fetchone()
        return {"messages": int(row["messages"]), "tokens": int(row["tokens"]),
                "byok_tokens": int(row["byok_tokens"])} if row \
            else {"messages": 0, "tokens": 0, "byok_tokens": 0}

    async def reset_daily_usage(self, user_id: str, day: str) -> None:
        """Zero *user_id*'s counters for *day*."""
        assert self._db is not None
        await self._db.execute(
            "DELETE FROM daily_usage WHERE user_id = ? AND day = ?", (user_id, day))
        await self._db.commit()

    async def list_daily_usage(self, day: str) -> dict[str, dict]:
        """Every user's counters for *day*, by user id."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT user_id, messages, tokens, byok_tokens FROM daily_usage WHERE day = ?", (day,),
        ) as cur:
            return {r["user_id"]: {"messages": int(r["messages"]), "tokens": int(r["tokens"]),
                                   "byok_tokens": int(r["byok_tokens"])}
                    for r in await cur.fetchall()}

    async def get_usage_resets(self, user_id: str) -> int:
        """How many free resets of the day's allowances *user_id* holds."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT balance FROM usage_reset_credits WHERE user_id = ?", (user_id,),
        ) as cur:
            row = await cur.fetchone()
        return int(row["balance"]) if row else 0

    async def list_usage_resets(self) -> dict[str, int]:
        """Every user holding free resets, by user id."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT user_id, balance FROM usage_reset_credits WHERE balance > 0",
        ) as cur:
            return {r["user_id"]: int(r["balance"]) for r in await cur.fetchall()}

    async def grant_usage_resets(
        self, user_ids: list[str] | None, count: int, cap: int,
    ) -> dict[str, int]:
        """Give each non-admin user in *user_ids* (None: every one) *count*
        more free resets, up to *cap* held — a grant never takes resets away
        from someone already over it. Returns each reached user's balance."""
        assert self._db is not None
        now = _utcnow()
        count, cap = max(0, int(count)), max(0, int(cap))
        chunks: list[list[str] | None] = [None] if user_ids is None else [
            user_ids[i:i + 500] for i in range(0, len(user_ids), 500)]
        reached: list[str] = []
        for chunk in chunks:
            where, params = "role != 'admin'", []
            if chunk is not None:
                if not chunk:
                    continue
                where += f" AND id IN ({','.join('?' * len(chunk))})"
                params = list(chunk)
            async with self._db.execute(f"SELECT id FROM users WHERE {where}", params) as cur:
                ids = [r["id"] for r in await cur.fetchall()]
            await self._db.executemany(
                "INSERT INTO usage_reset_credits (user_id, balance, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(user_id) DO UPDATE SET"
                " balance = MAX(balance, MIN(balance + ?, ?)), updated_at = excluded.updated_at",
                [(uid, min(count, cap), now, count, cap) for uid in ids],
            )
            reached += ids
        await self._db.commit()
        balances: dict[str, int] = {}
        for i in range(0, len(reached), 500):
            chunk = reached[i:i + 500]
            async with self._db.execute(
                "SELECT user_id, balance FROM usage_reset_credits"
                f" WHERE user_id IN ({','.join('?' * len(chunk))})", chunk,
            ) as cur:
                balances.update({r["user_id"]: int(r["balance"]) for r in await cur.fetchall()})
        return balances

    async def set_usage_resets(self, user_id: str, balance: int) -> int:
        """Set *user_id*'s free resets to *balance* (an admin taking some back)."""
        assert self._db is not None
        balance = max(0, int(balance))
        await self._db.execute(
            "INSERT INTO usage_reset_credits (user_id, balance, updated_at) VALUES (?, ?, ?)"
            " ON CONFLICT(user_id) DO UPDATE SET"
            " balance = excluded.balance, updated_at = excluded.updated_at",
            (user_id, balance, _utcnow()),
        )
        await self._db.commit()
        return balance

    async def use_usage_reset(self, user_id: str, day: str) -> bool:
        """Spend one of *user_id*'s free resets on zeroing their counters for
        *day* — both or neither. False when they hold none."""
        assert self._db is not None
        cur = await self._db.execute(
            "UPDATE usage_reset_credits SET balance = balance - 1, updated_at = ?"
            " WHERE user_id = ? AND balance > 0",
            (_utcnow(), user_id),
        )
        if cur.rowcount != 1:  # nothing changed (no rollback: the connection is shared)
            return False
        await self._db.execute(
            "DELETE FROM daily_usage WHERE user_id = ? AND day = ?", (user_id, day))
        await self._db.commit()
        return True

    async def get_usage_summary(
        self, user_id: str | None = None, since: str | None = None,
    ) -> dict[str, int]:
        """Return event counts grouped by event_type."""
        assert self._db is not None
        query = "SELECT event_type, COUNT(*) as cnt FROM usage_logs WHERE 1=1"
        params: list = []
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if since:
            query += " AND created_at >= ?"
            params.append(since)
        query += " GROUP BY event_type"
        async with self._db.execute(query, params) as cur:
            rows = await cur.fetchall()
            return {r["event_type"]: r["cnt"] for r in rows}

    # ── Admin helpers ────────────────────────────────────────────────

    async def list_users(self, limit: int = 100, offset: int = 0) -> list[dict]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT id, email, display_name, role, created_at, updated_at, metadata"
            " FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def delete_user(self, user_id: str) -> bool:
        assert self._db is not None
        cur = await self._db.execute("DELETE FROM users WHERE id = ?", (user_id,))
        # A removed user's public links stop working (their snapshots are
        # swept by public_file_shares once no row names them).
        await self._db.execute("DELETE FROM public_file_shares WHERE owner_id = ?", (user_id,))
        await self._db.execute("DELETE FROM public_link_origins WHERE owner_id = ?", (user_id,))
        await self._db.commit()
        return cur.rowcount > 0

    # ── Council sessions ─────────────────────────────────────────────

    async def create_council_session(
        self, user_id: str, title: str, topic: str,
        session_type: str = "brainstorm", verbosity: str = "message",
        max_rounds: int = 5, moderator_mode: str = "round-robin",
        moderator_agent: str = "", agents: str = "[]", config: str = "{}",
    ) -> dict:
        assert self._db is not None
        now = _utcnow()
        sid = _uuid()
        await self._db.execute(
            "INSERT INTO council_sessions"
            " (id, user_id, title, topic, session_type, verbosity, max_rounds,"
            "  current_round, status, moderator_mode, moderator_agent, agents,"
            "  pinned_ids, config, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 0, 'setup', ?, ?, ?, '[]', ?, ?, ?)",
            (sid, user_id, title, topic, session_type, verbosity, max_rounds,
             moderator_mode, moderator_agent, agents, config, now, now),
        )
        await self._db.commit()
        return {"id": sid, "user_id": user_id, "title": title, "topic": topic,
                "session_type": session_type, "verbosity": verbosity,
                "max_rounds": max_rounds, "current_round": 0, "status": "setup",
                "moderator_mode": moderator_mode, "moderator_agent": moderator_agent,
                "agents": agents, "pinned_ids": "[]", "config": config,
                "created_at": now, "updated_at": now}

    async def list_council_sessions(self, user_id: str) -> list[dict]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM council_sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def get_council_session(self, session_id: str, user_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM council_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def council_access(
        self, session_id: str, caller_id: str,
    ) -> tuple[dict | None, str | None]:
        """Resolve a caller's access to a council session (owner or shared).

        Returns ``(row, access)`` with access ∈ {'owner','edit','view'}, or
        ``(None, None)`` if invisible. The row's ``user_id`` is always the true
        owner — use it for child lookups and VFS/file resolution.
        """
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM council_sessions WHERE id = ?", (session_id,),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None, None
        row = dict(row)
        if row["user_id"] == caller_id:
            return row, "owner"
        share = await self.get_share_for_grantee(
            "council", session_id, caller_id, row["user_id"],
        )
        if share:
            return row, share.get("permission") or "view"
        return None, None

    async def update_council_session(
        self, session_id: str, user_id: str, **fields,
    ) -> bool:
        assert self._db is not None
        allowed = {"title", "topic", "status", "current_round", "moderator_mode",
                   "moderator_agent", "agents", "pinned_ids", "config"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        updates["updated_at"] = _utcnow()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [session_id, user_id]
        cur = await self._db.execute(
            f"UPDATE council_sessions SET {set_clause} WHERE id = ? AND user_id = ?", vals,
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def delete_council_session(self, session_id: str, user_id: str) -> bool:
        assert self._db is not None
        cur = await self._db.execute(
            "DELETE FROM council_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    # ── Council messages ─────────────────────────────────────────────

    async def get_council_messages(
        self, session_id: str, user_id: str,
        round_num: int | None = None, limit: int = 500,
    ) -> list[dict]:
        assert self._db is not None
        sess = await self.get_council_session(session_id, user_id)
        if not sess:
            return []
        query = "SELECT * FROM council_messages WHERE session_id = ?"
        params: list = [session_id]
        if round_num is not None:
            query += " AND round = ?"
            params.append(round_num)
        query += " ORDER BY id ASC LIMIT ?"
        params.append(limit)
        async with self._db.execute(query, params) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def add_council_messages(
        self, session_id: str, user_id: str, messages: list[dict],
    ) -> list[int]:
        assert self._db is not None
        sess = await self.get_council_session(session_id, user_id)
        if not sess:
            return []
        now = _utcnow()
        ids = []
        for msg in messages:
            cur = await self._db.execute(
                "INSERT INTO council_messages"
                " (session_id, round, agent_id, agent_name, role, action,"
                "  suitability, target_agent_id, content, pinned, metadata, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, msg.get("round", 1), msg.get("agent_id", ""),
                 msg.get("agent_name", ""), msg.get("role", "agent"),
                 msg.get("action", ""), msg.get("suitability", 0.0),
                 msg.get("target_agent_id", ""), msg.get("content", ""),
                 msg.get("pinned", 0), msg.get("metadata", "{}"), now),
            )
            ids.append(cur.lastrowid)
        await self._db.execute(
            "UPDATE council_sessions SET updated_at = ? WHERE id = ?", (now, session_id),
        )
        await self._db.commit()
        return ids

    async def toggle_council_pin(
        self, session_id: str, user_id: str, message_id: int,
    ) -> bool:
        assert self._db is not None
        sess = await self.get_council_session(session_id, user_id)
        if not sess:
            return False
        await self._db.execute(
            "UPDATE council_messages SET pinned = CASE WHEN pinned = 0 THEN 1 ELSE 0 END"
            " WHERE id = ? AND session_id = ?",
            (message_id, session_id),
        )
        await self._db.commit()
        return True

    async def update_council_message(
        self, session_id: str, user_id: str, message_id: int, fields: dict,
    ) -> bool:
        """Patch a single message (used to checkpoint/finalize a streaming turn)."""
        assert self._db is not None
        sess = await self.get_council_session(session_id, user_id)
        if not sess:
            return False
        allowed = ("content", "action", "suitability", "target_agent_id", "metadata")
        sets, params = [], []
        for k in allowed:
            if k in fields:
                sets.append(f"{k} = ?")
                params.append(fields[k])
        if not sets:
            return False
        params.extend([message_id, session_id])
        await self._db.execute(
            f"UPDATE council_messages SET {', '.join(sets)}"
            " WHERE id = ? AND session_id = ?",
            params,
        )
        await self._db.execute(
            "UPDATE council_sessions SET updated_at = ? WHERE id = ?",
            (_utcnow(), session_id),
        )
        await self._db.commit()
        return True

    async def delete_council_messages(
        self, session_id: str, user_id: str, round_num: int,
    ) -> int:
        """Delete all messages for a given round (used to restart a round).

        Returns the number of rows removed, or -1 if the session isn't owned by
        the user / doesn't exist.
        """
        assert self._db is not None
        sess = await self.get_council_session(session_id, user_id)
        if not sess:
            return -1
        cur = await self._db.execute(
            "DELETE FROM council_messages WHERE session_id = ? AND round = ?",
            (session_id, round_num),
        )
        await self._db.execute(
            "UPDATE council_sessions SET updated_at = ? WHERE id = ?",
            (_utcnow(), session_id),
        )
        await self._db.commit()
        return cur.rowcount

    # ── Council votes ────────────────────────────────────────────────

    async def add_council_votes(
        self, session_id: str, user_id: str, votes: list[dict],
    ) -> list[int]:
        assert self._db is not None
        sess = await self.get_council_session(session_id, user_id)
        if not sess:
            return []
        now = _utcnow()
        ids = []
        for v in votes:
            cur = await self._db.execute(
                "INSERT INTO council_votes"
                " (session_id, round, agent_id, agent_name, vote, reason, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, v.get("round", 1), v.get("agent_id", ""),
                 v.get("agent_name", ""), v.get("vote", "abstain"),
                 v.get("reason", ""), now),
            )
            ids.append(cur.lastrowid)
        await self._db.commit()
        return ids

    async def get_council_votes(
        self, session_id: str, user_id: str, round_num: int | None = None,
    ) -> list[dict]:
        assert self._db is not None
        sess = await self.get_council_session(session_id, user_id)
        if not sess:
            return []
        query = "SELECT * FROM council_votes WHERE session_id = ?"
        params: list = [session_id]
        if round_num is not None:
            query += " AND round = ?"
            params.append(round_num)
        query += " ORDER BY id ASC"
        async with self._db.execute(query, params) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # ── Council artifacts ───────────────────────────────────────────

    async def get_council_artifacts(
        self, session_id: str, user_id: str, kind: str | None = None,
    ) -> list[dict]:
        assert self._db is not None
        sess = await self.get_council_session(session_id, user_id)
        if not sess:
            return []
        query = "SELECT * FROM council_artifacts WHERE session_id = ?"
        params: list = [session_id]
        if kind is not None:
            query += " AND kind = ?"
            params.append(kind)
        query += " ORDER BY id ASC"
        async with self._db.execute(query, params) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def upsert_council_artifact(
        self, session_id: str, user_id: str,
        kind: str, agent_id: str, agent_name: str, content: str,
    ) -> int:
        assert self._db is not None
        sess = await self.get_council_session(session_id, user_id)
        if not sess:
            return 0
        now = _utcnow()
        # Delete existing artifact with same key, then insert
        await self._db.execute(
            "DELETE FROM council_artifacts WHERE session_id = ? AND kind = ? AND agent_id = ?",
            (session_id, kind, agent_id),
        )
        async with self._db.execute(
            "INSERT INTO council_artifacts"
            " (session_id, kind, agent_id, agent_name, content, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, kind, agent_id, agent_name, content, now),
        ) as cur:
            art_id = cur.lastrowid or 0
        await self._db.commit()
        return art_id

    async def delete_council_artifacts(
        self, session_id: str, user_id: str, kind: str | None = None,
    ) -> bool:
        assert self._db is not None
        sess = await self.get_council_session(session_id, user_id)
        if not sess:
            return False
        if kind:
            await self._db.execute(
                "DELETE FROM council_artifacts WHERE session_id = ? AND kind = ?",
                (session_id, kind),
            )
        else:
            await self._db.execute(
                "DELETE FROM council_artifacts WHERE session_id = ?", (session_id,),
            )
        await self._db.commit()
        return True

    # ── Basna sessions ───────────────────────────────────────────────

    async def create_basna_session(
        self, user_id: str, intent: str, config: str = "{}", title: str = "",
    ) -> dict:
        assert self._db is not None
        now = _utcnow()
        sid = _uuid()
        await self._db.execute(
            "INSERT INTO basna_sessions"
            " (id, user_id, title, intent, domain, difficulty, merge_kind, status,"
            "  route, truth, confidence, config, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, '', '', 'converge', 'routing', '{}', '', 0.0, ?, ?, ?)",
            (sid, user_id, title, intent, config, now, now),
        )
        await self._db.commit()
        return {"id": sid, "user_id": user_id, "title": title, "intent": intent, "domain": "",
                "difficulty": "", "merge_kind": "converge", "status": "routing",
                "route": "{}", "truth": "", "confidence": 0.0, "config": config,
                "created_at": now, "updated_at": now}

    async def list_basna_sessions(self, user_id: str) -> list[dict]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM basna_sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def get_basna_session(self, session_id: str, user_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM basna_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def basna_access(
        self, session_id: str, caller_id: str,
    ) -> tuple[dict | None, str | None]:
        """Resolve access to a Basna session (owner or shared). See council_access."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM basna_sessions WHERE id = ?", (session_id,),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None, None
        row = dict(row)
        if row["user_id"] == caller_id:
            return row, "owner"
        share = await self.get_share_for_grantee(
            "basna", session_id, caller_id, row["user_id"],
        )
        if share:
            return row, share.get("permission") or "view"
        return None, None

    async def update_basna_session(
        self, session_id: str, user_id: str, **fields,
    ) -> bool:
        assert self._db is not None
        allowed = {"title", "intent", "domain", "difficulty", "merge_kind", "status",
                   "route", "truth", "confidence", "config", "progress", "files", "analysis"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        updates["updated_at"] = _utcnow()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [session_id, user_id]
        cur = await self._db.execute(
            f"UPDATE basna_sessions SET {set_clause} WHERE id = ? AND user_id = ?", vals,
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def delete_basna_session(self, session_id: str, user_id: str) -> bool:
        assert self._db is not None
        cur = await self._db.execute(
            "DELETE FROM basna_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    # ── Basna projects (run bundles) ─────────────────────────────────

    async def create_basna_project(
        self, user_id: str, name: str, vfs_folder: str,
        description: str = "", instructions: str = "",
    ) -> dict:
        assert self._db is not None
        now = _utcnow()
        pid = _uuid()
        await self._db.execute(
            "INSERT INTO basna_projects"
            " (id, user_id, name, description, instructions, vfs_folder, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, user_id, name, description, instructions, vfs_folder, now, now),
        )
        await self._db.commit()
        return {"id": pid, "user_id": user_id, "name": name, "description": description,
                "instructions": instructions, "vfs_folder": vfs_folder,
                "created_at": now, "updated_at": now}

    async def list_basna_projects(self, user_id: str) -> list[dict]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM basna_projects WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def get_basna_project(self, project_id: str, user_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM basna_projects WHERE id = ? AND user_id = ?",
            (project_id, user_id),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def update_basna_project(
        self, project_id: str, user_id: str, **fields,
    ) -> bool:
        assert self._db is not None
        allowed = {"name", "description", "instructions", "vfs_folder"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        updates["updated_at"] = _utcnow()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [project_id, user_id]
        cur = await self._db.execute(
            f"UPDATE basna_projects SET {set_clause} WHERE id = ? AND user_id = ?", vals,
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def delete_basna_project(self, project_id: str, user_id: str) -> bool:
        """Delete the project. Its runs are NOT deleted — the caller decides
        whether to cascade (they carry project_id in their config JSON)."""
        assert self._db is not None
        cur = await self._db.execute(
            "DELETE FROM basna_projects WHERE id = ? AND user_id = ?",
            (project_id, user_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    # ── Basna runs ───────────────────────────────────────────────────

    async def add_basna_runs(
        self, session_id: str, user_id: str, runs: list[dict],
    ) -> list[int]:
        assert self._db is not None
        sess = await self.get_basna_session(session_id, user_id)
        if not sess:
            return []
        now = _utcnow()
        ids: list[int] = []
        for r in runs:
            cur = await self._db.execute(
                "INSERT INTO basna_runs"
                " (session_id, archetype_id, role, provider, model, tier,"
                "  weight_at_run, output, actions, success, latency_ms, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, r.get("archetype_id", ""), r.get("role", ""),
                 r.get("provider", ""), r.get("model", ""), r.get("tier", ""),
                 float(r.get("weight_at_run", 0.0)), r.get("output", ""),
                 r.get("actions", "[]"),
                 r.get("success"), int(r.get("latency_ms", 0)), now),
            )
            ids.append(cur.lastrowid or 0)
        await self._db.execute(
            "UPDATE basna_sessions SET updated_at = ? WHERE id = ?", (now, session_id),
        )
        await self._db.commit()
        return ids

    async def list_basna_runs(self, session_id: str, user_id: str) -> list[dict]:
        assert self._db is not None
        sess = await self.get_basna_session(session_id, user_id)
        if not sess:
            return []
        async with self._db.execute(
            "SELECT * FROM basna_runs WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ) as cur:
            return [_with_run_labels(dict(r)) for r in await cur.fetchall()]

    async def score_basna_run(
        self, run_id: int, user_id: str, success: bool,
    ) -> bool:
        """Record the judge's automatic success/fail label, ownership-checked via
        the parent session. Never touches the human label."""
        assert self._db is not None
        cur = await self._db.execute(
            "UPDATE basna_runs SET success = ?"
            " WHERE id = ? AND session_id IN"
            " (SELECT id FROM basna_sessions WHERE user_id = ?)",
            (1 if success else 0, run_id, user_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def set_basna_run_human_label(
        self, run_id: int, user_id: str, success: bool,
    ) -> bool:
        """Record a human thumbs vote on a run, ownership-checked via the parent
        session. Leaves the judge's `success` intact so the pair stays comparable."""
        assert self._db is not None
        cur = await self._db.execute(
            "UPDATE basna_runs SET human_success = ?, human_feedback_at = ?"
            " WHERE id = ? AND session_id IN"
            " (SELECT id FROM basna_sessions WHERE user_id = ?)",
            (1 if success else 0, _utcnow(), run_id, user_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def list_labeled_basna_runs(
        self, user_id: str, since: str | None = None, include_text: bool = False,
    ) -> list[dict]:
        """Runs in this user's sessions that a human has voted on, with the judge's
        label alongside (`judge_success`, NULL when the judge left it unscored) —
        the raw material for judge-vs-human agreement. `since` keeps only runs
        created at/after that ISO date or timestamp."""
        assert self._db is not None
        cols = ("r.id, r.session_id, r.archetype_id, r.role, r.tier, r.model,"
                " r.weight_at_run, r.latency_ms, r.created_at,"
                " r.success AS judge_success, r.human_success, r.human_feedback_at,"
                " s.domain, s.merge_kind, s.config")
        if include_text:
            cols += ", s.intent, s.truth, r.output"
        query = (f"SELECT {cols} FROM basna_runs r JOIN basna_sessions s ON r.session_id = s.id"
                 " WHERE s.user_id = ? AND r.human_success IS NOT NULL")
        params: list = [user_id]
        if since:
            query += " AND r.created_at >= ?"
            params.append(since)
        query += " ORDER BY r.id ASC"
        async with self._db.execute(query, params) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # ── Vatra blackboard (cross-agent asks) ──────────────────────────

    async def create_vatra_ask(
        self, session_id: str, from_owner: str, from_subtask: str, text: str,
        depth: int = 0,
    ) -> dict:
        assert self._db is not None
        now = _utcnow()
        cur = await self._db.execute(
            "INSERT INTO basna_asks"
            " (session_id, from_owner, from_subtask, text, status, depth, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, 'open', ?, ?, ?)",
            (session_id, from_owner, from_subtask, text, depth, now, now),
        )
        await self._db.commit()
        return {"id": cur.lastrowid, "session_id": session_id, "from_owner": from_owner,
                "from_subtask": from_subtask, "text": text, "status": "open",
                "answer": "", "answered_by": "", "depth": depth, "created_at": now}

    async def list_vatra_asks(
        self, session_id: str, status: str | None = None, from_owner: str | None = None,
    ) -> list[dict]:
        assert self._db is not None
        query = "SELECT * FROM basna_asks WHERE session_id = ?"
        params: list = [session_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        if from_owner:
            query += " AND from_owner = ?"
            params.append(from_owner)
        query += " ORDER BY id ASC"
        async with self._db.execute(query, params) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def count_vatra_asks(self, session_id: str) -> int:
        """Total asks ever created for this session — the budget counter."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT COUNT(*) AS n FROM basna_asks WHERE session_id = ?", (session_id,),
        ) as cur:
            row = await cur.fetchone()
            return int(row["n"]) if row else 0

    async def claim_vatra_ask(self, ask_id: int) -> bool:
        """Atomically move an ask open → claimed. Returns False if already taken."""
        assert self._db is not None
        cur = await self._db.execute(
            "UPDATE basna_asks SET status = 'claimed', updated_at = ?"
            " WHERE id = ? AND status = 'open'",
            (_utcnow(), ask_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def answer_vatra_ask(self, ask_id: int, answer: str, answered_by: str) -> bool:
        assert self._db is not None
        cur = await self._db.execute(
            "UPDATE basna_asks SET status = 'answered', answer = ?, answered_by = ?,"
            " updated_at = ? WHERE id = ?",
            (answer, answered_by, _utcnow(), ask_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def drop_vatra_ask(self, ask_id: int, note: str = "") -> bool:
        assert self._db is not None
        cur = await self._db.execute(
            "UPDATE basna_asks SET status = 'dropped', note = ?, updated_at = ?"
            " WHERE id = ? AND status IN ('open', 'claimed')",
            (note, _utcnow(), ask_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    # ── Vatra shared board (real-time shared memory) ─────────────────

    async def add_vatra_board(
        self, session_id: str, from_owner: str, from_subtask: str,
        kind: str, title: str, content: str,
    ) -> dict:
        assert self._db is not None
        now = _utcnow()
        cur = await self._db.execute(
            "INSERT INTO vatra_board"
            " (session_id, from_owner, from_subtask, kind, title, content, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, from_owner, from_subtask, kind, title, content, now),
        )
        await self._db.commit()
        return {"id": cur.lastrowid, "session_id": session_id, "from_owner": from_owner,
                "from_subtask": from_subtask, "kind": kind, "title": title,
                "content": content, "created_at": now}

    async def list_vatra_board(
        self, session_id: str, kinds: list[str] | None = None, limit: int = 100,
        exclude_owner: str | None = None, exclude_subtask: str | None = None,
    ) -> list[dict]:
        assert self._db is not None
        query = "SELECT * FROM vatra_board WHERE session_id = ?"
        params: list = [session_id]
        if kinds:
            query += " AND kind IN (%s)" % ",".join("?" * len(kinds))
            params.extend(kinds)
        # Prefer per-subtask exclusion (two owners can share an archetype id, so
        # excluding by owner hides a same-archetype teammate's posts too).
        if exclude_subtask:
            query += " AND from_subtask != ?"
            params.append(exclude_subtask)
        elif exclude_owner:
            query += " AND from_owner != ?"
            params.append(exclude_owner)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        async with self._db.execute(query, params) as cur:
            rows = [dict(r) for r in await cur.fetchall()]
        rows.reverse()  # oldest-first for readability
        return rows

    _BOARD_STOPWORDS = frozenset({
        "the", "and", "for", "with", "from", "that", "this", "into", "your",
        "their", "are", "was", "has", "have", "will", "not", "but", "its",
    })

    async def search_vatra_board(
        self, session_id: str, query: str, limit: int = 20,
        exclude_owner: str | None = None, exclude_subtask: str | None = None,
    ) -> list[dict]:
        """Board search that a multi-word 'keyword' wait can actually satisfy.

        Matches the exact whole phrase in any row (legacy behaviour, first attempt),
        OR — for a produced artifact — every content token (≥ 3 chars, minus
        stopwords) appearing in an ``output``/``note``/``file`` row (so a narration
        row can't spuriously satisfy a wait). ``exclude_subtask`` hides only the
        caller's own posts, keeping a same-archetype teammate's visible.
        """
        assert self._db is not None
        q = (query or "").strip()
        like = f"%{q}%"
        tokens = [t for t in re.findall(r"[a-z0-9]{3,}", q.lower())
                  if t not in self._BOARD_STOPWORDS]
        base = "SELECT * FROM vatra_board WHERE session_id = ?"
        params: list = [session_id]
        if tokens:
            tok_parts = []
            tok_params: list = []
            for t in tokens:
                tok_parts.append("(content LIKE ? OR title LIKE ?)")
                tok_params.extend([f"%{t}%", f"%{t}%"])
            tok_clause = ("(kind IN ('output','note','file') AND "
                          + " AND ".join(tok_parts) + ")")
            sql = base + " AND ((content LIKE ? OR title LIKE ?) OR " + tok_clause + ")"
            params += [like, like] + tok_params
        else:
            sql = base + " AND (content LIKE ? OR title LIKE ?)"
            params += [like, like]
        if exclude_subtask:
            sql += " AND from_subtask != ?"
            params.append(exclude_subtask)
        elif exclude_owner:
            sql += " AND from_owner != ?"
            params.append(exclude_owner)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        async with self._db.execute(sql, params) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # ── Vatra run ledger (`vatra map`) ──────────────────────────────

    async def upsert_vatra_ledger(
        self, session_id: str, kind: str, key: str, *, owner: str = "",
        subtask_id: str = "", label: str = "", detail: str = "", pointer: str = "",
        replace: bool = False,
    ) -> None:
        """Record one ledger fact. A new key inserts; a repeat bumps ``count`` and
        adds a different actor to ``others`` — or, with ``replace`` (an owner's
        current focus), overwrites the row. Atomic: teammates fetching the same
        page at the same moment both land."""
        assert self._db is not None
        now = _utcnow()
        key = str(key or "")[:500]
        detail = str(detail or "")[:400]
        label = str(label or "")[:120]
        cur = await self._db.execute(
            "INSERT OR IGNORE INTO vatra_ledger (session_id, kind, key, owner, subtask_id, label,"
            " detail, pointer, count, others, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, '', ?, ?)",
            (session_id, kind, key, owner, subtask_id, label, detail, pointer, now, now),
        )
        if cur.rowcount == 0 and replace:
            await self._db.execute(
                "UPDATE vatra_ledger SET owner = ?, subtask_id = ?, label = ?, detail = ?,"
                " pointer = ?, updated_at = ? WHERE session_id = ? AND kind = ? AND key = ?",
                (owner, subtask_id, label, detail, pointer, now, session_id, kind, key),
            )
        elif cur.rowcount == 0:
            other = label.replace(",", " ").strip()   # `others` is comma-separated
            await self._db.execute(
                "UPDATE vatra_ledger SET count = count + 1, updated_at = ?,"
                " pointer = CASE WHEN pointer = '' THEN ? ELSE pointer END,"
                " others = CASE"
                "   WHEN ? = '' OR ? = label OR length(others) > 600"
                "        OR instr(',' || others || ',', ',' || ? || ',') > 0 THEN others"
                "   WHEN others = '' THEN ?"
                "   ELSE others || ',' || ? END"
                " WHERE session_id = ? AND kind = ? AND key = ?",
                (now, pointer, other, label, other, other, other, session_id, kind, key),
            )
        await self._db.commit()

    async def list_vatra_ledger(self, session_id: str, kinds: list[str] | None = None,
                                limit: int = 500) -> list[dict]:
        """Ledger rows for a run, newest first."""
        assert self._db is not None
        query = "SELECT * FROM vatra_ledger WHERE session_id = ?"
        params: list = [session_id]
        if kinds:
            query += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        query += " ORDER BY updated_at DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        async with self._db.execute(query, params) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # ── Vatra resume checkpoints (per-owner slice, restored on resume) ─

    async def save_vatra_run(
        self, session_id: str, subtask_id: str, archetype_id: str, role: str,
        weight: float, output: str, produced_file: bool, status: str = "done",
    ) -> None:
        """Persist (UPSERT) one owner's finished slice as a resume checkpoint.

        Keyed on (session_id, subtask_id) so re-dispatching the same owner (on a
        resume, or across the main/finalize passes) overwrites its checkpoint in
        place rather than accumulating rows. Called from the run coroutine, which
        has already ownership-checked the session — no user_id gate here."""
        assert self._db is not None
        now = _utcnow()
        await self._db.execute(
            "INSERT INTO vatra_runs"
            " (session_id, subtask_id, archetype_id, role, weight, output,"
            "  produced_file, status, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(session_id, subtask_id) DO UPDATE SET"
            "  archetype_id = excluded.archetype_id, role = excluded.role,"
            "  weight = excluded.weight, output = excluded.output,"
            "  produced_file = excluded.produced_file, status = excluded.status,"
            "  updated_at = excluded.updated_at",
            (session_id, subtask_id, archetype_id, role, float(weight or 0.0),
             output or "", 1 if produced_file else 0, status, now, now),
        )
        await self._db.commit()

    async def list_vatra_runs(self, session_id: str) -> list[dict]:
        """All resume checkpoints for a session, oldest-first."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM vatra_runs WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # ── Archetype reliability (learned routing weights) ──────────────

    @staticmethod
    def _reliability_weight(
        successes: int, fails: int, seed: float = 0.7, alpha: float = 4.0,
    ) -> float:
        """Bayesian-shrunk success rate toward `seed`, penalizing fails 2×.

        No data → returns `seed`; as runs accrue it approaches the empirical
        rate. Fails count double so an unreliable archetype decays quickly.
        """
        numerator = successes + alpha * seed
        denominator = successes + 2.0 * fails + alpha
        w = numerator / denominator if denominator > 0 else seed
        return max(0.05, min(0.99, w))

    async def get_archetype_reliability(
        self, user_id: str, domain: str | None = None,
    ) -> list[dict]:
        assert self._db is not None
        query = "SELECT * FROM archetype_reliability WHERE user_id = ?"
        params: list = [user_id]
        if domain is not None:
            query += " AND domain = ?"
            params.append(domain)
        async with self._db.execute(query, params) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def get_archetype_weight(
        self, user_id: str, archetype_id: str, domain: str, seed: float = 0.7,
    ) -> float:
        """Current learned weight for an archetype in a domain; `seed` if unseen."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT weight FROM archetype_reliability"
            " WHERE user_id = ? AND archetype_id = ? AND domain = ?",
            (user_id, archetype_id, domain),
        ) as cur:
            row = await cur.fetchone()
            return float(row["weight"]) if row else seed

    async def record_archetype_outcome(
        self, user_id: str, archetype_id: str, domain: str,
        success: bool, seed: float = 0.7,
    ) -> dict:
        """Upsert one outcome and recompute the learned weight."""
        assert self._db is not None
        now = _utcnow()
        async with self._db.execute(
            "SELECT successes, fails FROM archetype_reliability"
            " WHERE user_id = ? AND archetype_id = ? AND domain = ?",
            (user_id, archetype_id, domain),
        ) as cur:
            row = await cur.fetchone()
        successes = (row["successes"] if row else 0) + (1 if success else 0)
        fails = (row["fails"] if row else 0) + (0 if success else 1)
        runs = successes + fails
        weight = self._reliability_weight(successes, fails, seed)
        await self._db.execute(
            "INSERT INTO archetype_reliability"
            " (user_id, archetype_id, domain, successes, fails, runs, weight, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(user_id, archetype_id, domain) DO UPDATE SET"
            " successes = excluded.successes, fails = excluded.fails,"
            " runs = excluded.runs, weight = excluded.weight,"
            " updated_at = excluded.updated_at",
            (user_id, archetype_id, domain, successes, fails, runs, weight, now),
        )
        await self._db.commit()
        return {"user_id": user_id, "archetype_id": archetype_id, "domain": domain,
                "successes": successes, "fails": fails, "runs": runs,
                "weight": weight, "updated_at": now}

    async def get_basna_run(self, run_id: int, user_id: str) -> dict | None:
        """One run, ownership-checked via its parent session."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT r.* FROM basna_runs r JOIN basna_sessions s ON r.session_id = s.id"
            " WHERE r.id = ? AND s.user_id = ?",
            (run_id, user_id),
        ) as cur:
            row = await cur.fetchone()
            return _with_run_labels(dict(row)) if row else None

    async def adjust_archetype_reliability(
        self, user_id: str, archetype_id: str, domain: str,
        d_success: int, d_fail: int, seed: float = 0.7,
    ) -> dict:
        """Apply signed deltas to the success/fail counters and recompute weight.

        Used to *revise* a learned outcome (e.g. a human thumbs flipping an
        auto-scored run) without double-counting: move one from one bucket to the
        other rather than appending a fresh outcome. Counters never go negative.
        """
        assert self._db is not None
        now = _utcnow()
        async with self._db.execute(
            "SELECT successes, fails FROM archetype_reliability"
            " WHERE user_id = ? AND archetype_id = ? AND domain = ?",
            (user_id, archetype_id, domain),
        ) as cur:
            row = await cur.fetchone()
        successes = max(0, (row["successes"] if row else 0) + d_success)
        fails = max(0, (row["fails"] if row else 0) + d_fail)
        runs = successes + fails
        weight = self._reliability_weight(successes, fails, seed)
        await self._db.execute(
            "INSERT INTO archetype_reliability"
            " (user_id, archetype_id, domain, successes, fails, runs, weight, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(user_id, archetype_id, domain) DO UPDATE SET"
            " successes = excluded.successes, fails = excluded.fails,"
            " runs = excluded.runs, weight = excluded.weight,"
            " updated_at = excluded.updated_at",
            (user_id, archetype_id, domain, successes, fails, runs, weight, now),
        )
        await self._db.commit()
        return {"user_id": user_id, "archetype_id": archetype_id, "domain": domain,
                "successes": successes, "fails": fails, "runs": runs,
                "weight": weight, "updated_at": now}

    # ── Learned constraints (R2 automatic CONSTRAINT edge, opt-in) ────

    async def add_learned_constraint(
        self, user_id: str, domain: str, engine: str,
        trigger: str, constraint: str, severity: str = "major",
        source_id: str = "", *, cap_per_domain: int = 50,
    ) -> dict:
        """Persist one distilled constraint. Dedups on (user_id, domain, text):
        an identical rule bumps `hits`/updated_at instead of duplicating. Prunes
        the oldest rows beyond `cap_per_domain` so storage stays bounded. Additive;
        never migrates existing tables. Returns {"id", "deduped"}; {} for empty."""
        assert self._db is not None
        now = _utcnow()
        domain = (domain or "").strip()
        constraint = (constraint or "").strip()
        if not constraint:
            return {}
        if severity not in ("critical", "major", "minor"):
            severity = "major"
        async with self._db.execute(
            "SELECT id, hits FROM learned_constraints"
            " WHERE user_id = ? AND domain = ? AND lower(constraint_text) = lower(?)",
            (user_id, domain, constraint),
        ) as cur:
            row = await cur.fetchone()
        if row:
            await self._db.execute(
                "UPDATE learned_constraints SET hits = ?, severity = ?,"
                " trigger_text = ?, updated_at = ? WHERE id = ?",
                (int(row["hits"]) + 1, severity, (trigger or "").strip(), now, row["id"]))
            await self._db.commit()
            return {"id": row["id"], "deduped": True}
        cid = _uuid()
        await self._db.execute(
            "INSERT INTO learned_constraints"
            " (id, user_id, domain, engine, trigger_text, constraint_text,"
            "  severity, source_id, hits, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
            (cid, user_id, domain, (engine or "").strip(), (trigger or "").strip(),
             constraint, severity, (source_id or "").strip(), now, now))
        # Bounded storage: keep only the newest `cap_per_domain` per (user, domain).
        # LIMIT/OFFSET on the SELECT subquery (not the DELETE) is always supported.
        await self._db.execute(
            "DELETE FROM learned_constraints WHERE id IN ("
            " SELECT id FROM learned_constraints WHERE user_id = ? AND domain = ?"
            " ORDER BY updated_at DESC LIMIT -1 OFFSET ?)",
            (user_id, domain, max(1, int(cap_per_domain))))
        await self._db.commit()
        return {"id": cid, "deduped": False}

    async def get_learned_constraints(
        self, user_id: str, domain: str | None = None, limit: int = 5,
    ) -> list[dict]:
        """Top learned constraints for a user (optionally one domain), most severe
        then most recent first. `domain=None` returns across all domains (used at
        the Vatra lead decompose, where the domain is still an unknown output)."""
        assert self._db is not None
        query = "SELECT * FROM learned_constraints WHERE user_id = ?"
        params: list = [user_id]
        if domain is not None:
            query += " AND domain = ?"
            params.append((domain or "").strip())
        query += (" ORDER BY CASE severity WHEN 'critical' THEN 0"
                  " WHEN 'major' THEN 1 ELSE 2 END, updated_at DESC LIMIT ?")
        params.append(max(1, int(limit)))
        async with self._db.execute(query, params) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # ── Prompts ─────────────────────────────────────────────────────

    async def list_prompts(self, user_id: str) -> list[dict]:
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT * FROM prompts WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        )
        return [dict(r) for r in rows]

    async def get_prompt(self, prompt_id: str, user_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM prompts WHERE id = ? AND user_id = ?",
            (prompt_id, user_id),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def create_prompt(
        self, user_id: str, title: str, content: str,
        files: str = "[]", tags: str = "[]",
    ) -> dict:
        assert self._db is not None
        now = _utcnow()
        pid = _uuid()
        await self._db.execute(
            "INSERT INTO prompts (id, user_id, title, content, files, tags, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, user_id, title, content, files, tags, now, now),
        )
        await self._db.commit()
        return {"id": pid, "user_id": user_id, "title": title, "content": content,
                "files": files, "tags": tags, "created_at": now, "updated_at": now}

    async def update_prompt(
        self, prompt_id: str, user_id: str, **fields: str,
    ) -> dict | None:
        assert self._db is not None
        existing = await self.get_prompt(prompt_id, user_id)
        if not existing:
            return None
        allowed = {"title", "content", "files", "tags"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return existing
        updates["updated_at"] = _utcnow()
        sets = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [prompt_id, user_id]
        await self._db.execute(
            f"UPDATE prompts SET {sets} WHERE id = ? AND user_id = ?", vals,
        )
        await self._db.commit()
        return await self.get_prompt(prompt_id, user_id)

    async def delete_prompt(self, prompt_id: str, user_id: str) -> bool:
        assert self._db is not None
        async with self._db.execute(
            "DELETE FROM prompts WHERE id = ? AND user_id = ?",
            (prompt_id, user_id),
        ) as cur:
            await self._db.commit()
            return (cur.rowcount or 0) > 0

    # ── User archetypes ──────────────────────────────────────────────

    async def list_user_archetypes(self, user_id: str) -> list[dict]:
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT * FROM user_archetypes WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        )
        return [dict(r) for r in rows]

    async def get_user_archetype(self, user_id: str, archetype_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM user_archetypes WHERE user_id = ? AND archetype_id = ?",
            (user_id, archetype_id),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def create_user_archetype(
        self, user_id: str, archetype_id: str, data: str,
    ) -> dict:
        assert self._db is not None
        now = _utcnow()
        aid = _uuid()
        await self._db.execute(
            "INSERT INTO user_archetypes (id, user_id, archetype_id, data, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (aid, user_id, archetype_id, data, now, now),
        )
        await self._db.commit()
        return {"id": aid, "user_id": user_id, "archetype_id": archetype_id,
                "data": data, "created_at": now, "updated_at": now}

    async def update_user_archetype(
        self, user_id: str, archetype_id: str, data: str,
    ) -> dict | None:
        assert self._db is not None
        existing = await self.get_user_archetype(user_id, archetype_id)
        if not existing:
            return None
        now = _utcnow()
        await self._db.execute(
            "UPDATE user_archetypes SET data = ?, updated_at = ?"
            " WHERE user_id = ? AND archetype_id = ?",
            (data, now, user_id, archetype_id),
        )
        await self._db.commit()
        return await self.get_user_archetype(user_id, archetype_id)

    async def delete_user_archetype(self, user_id: str, archetype_id: str) -> bool:
        assert self._db is not None
        async with self._db.execute(
            "DELETE FROM user_archetypes WHERE user_id = ? AND archetype_id = ?",
            (user_id, archetype_id),
        ) as cur:
            await self._db.commit()
            return (cur.rowcount or 0) > 0

    # ── Public file links (anyone with the token) ────────────────────

    async def save_public_file_share(self, row: dict) -> dict | None:
        """Insert a public link, or update one (same token) after a re-snapshot."""
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO public_file_shares"
            " (token, owner_id, agent_ref, agent_name, source_path, filename, kind, size,"
            "  created_at, updated_at, expires_at)"
            " VALUES (:token, :owner_id, :agent_ref, :agent_name, :source_path, :filename, :kind,"
            "  :size, :created_at, :updated_at, :expires_at)"
            " ON CONFLICT(token) DO UPDATE SET"
            "  agent_name = excluded.agent_name, filename = excluded.filename,"
            "  kind = excluded.kind, size = excluded.size,"
            "  updated_at = excluded.updated_at, expires_at = excluded.expires_at",
            row,
        )
        await self._db.commit()
        return await self.get_public_file_share(row["token"])

    async def get_public_file_share(self, token: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM public_file_shares WHERE token = ?", (token,),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def find_public_file_share(
        self, owner_id: str, agent_ref: str, source_path: str,
    ) -> dict | None:
        """``owner_id``'s link to this agent file, if they made one."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM public_file_shares"
            " WHERE owner_id = ? AND agent_ref = ? AND source_path = ?"
            " ORDER BY created_at DESC LIMIT 1",
            (owner_id, agent_ref, source_path),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def list_public_file_shares(self, owner_id: str | None = None) -> list[dict]:
        """One owner's links (newest first), or every link when ``owner_id`` is None."""
        assert self._db is not None
        q = "SELECT * FROM public_file_shares"
        params: tuple = ()
        if owner_id is not None:
            q += " WHERE owner_id = ?"
            params = (owner_id,)
        async with self._db.execute(q + " ORDER BY created_at DESC", params) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def delete_public_file_share(self, token: str) -> bool:
        assert self._db is not None
        async with self._db.execute(
            "DELETE FROM public_file_shares WHERE token = ?", (token,),
        ) as cur:
            await self._db.commit()
            return (cur.rowcount or 0) > 0

    async def set_public_link_origin(self, owner_id: str, origin: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO public_link_origins (owner_id, origin, seen_at) VALUES (?, ?, ?)"
            " ON CONFLICT(owner_id) DO UPDATE SET origin = excluded.origin,"
            " seen_at = excluded.seen_at",
            (owner_id, origin, _utcnow()),
        )
        await self._db.commit()

    async def get_public_link_origin(self, owner_id: str | None = None) -> str:
        """``owner_id``'s browser origin, or (None) the most recently seen one."""
        assert self._db is not None
        if owner_id is None:
            q, params = "SELECT origin FROM public_link_origins ORDER BY seen_at DESC LIMIT 1", ()
        else:
            q, params = "SELECT origin FROM public_link_origins WHERE owner_id = ?", (owner_id,)
        async with self._db.execute(q, params) as cur:
            row = await cur.fetchone()
            return str(row[0]) if row else ""

    async def note_public_file_share_view(self, token: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE public_file_shares SET view_count = view_count + 1, last_viewed_at = ?"
            " WHERE token = ?",
            (_utcnow(), token),
        )
        await self._db.commit()

    # ── Resource shares (cross-user access grants) ───────────────────

    async def create_share(
        self, resource_type: str, resource_id: str, owner_id: str,
        grantee_id: str, permission: str = "view",
    ) -> dict | None:
        """Grant (or re-grant, updating the permission) access to a resource."""
        assert self._db is not None
        now = _utcnow()
        sid = _uuid()
        await self._db.execute(
            "INSERT INTO resource_shares"
            " (id, resource_type, resource_id, owner_id, grantee_id, permission, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(resource_type, resource_id, owner_id, grantee_id)"
            " DO UPDATE SET permission = excluded.permission",
            (sid, resource_type, resource_id, owner_id, grantee_id, permission, now),
        )
        if resource_type == "agent":
            await self._db.execute(
                "INSERT OR IGNORE INTO agent_share_history (agent_ref, owner_id, first_shared_at)"
                " VALUES (?, ?, ?)",
                (resource_id, owner_id, now),
            )
        await self._db.commit()
        return await self.get_share(resource_type, resource_id, owner_id, grantee_id)

    async def get_share(
        self, resource_type: str, resource_id: str, owner_id: str, grantee_id: str,
    ) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM resource_shares"
            " WHERE resource_type = ? AND resource_id = ? AND owner_id = ? AND grantee_id = ?",
            (resource_type, resource_id, owner_id, grantee_id),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_share_for_grantee(
        self, resource_type: str, resource_id: str, grantee_id: str,
        owner_id: str | None = None,
    ) -> dict | None:
        """Access-check helper: is `resource_id` shared TO `grantee_id`?

        Returns the share row (with owner_id + permission) or None. Pass
        `owner_id` to disambiguate when a resource_id (e.g. a bare project
        name) can collide across owners.
        """
        assert self._db is not None
        q = ("SELECT * FROM resource_shares"
             " WHERE resource_type = ? AND resource_id = ? AND grantee_id = ?")
        params: list = [resource_type, resource_id, grantee_id]
        if owner_id:
            q += " AND owner_id = ?"
            params.append(owner_id)
        async with self._db.execute(q, tuple(params)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def delete_share(
        self, resource_type: str, resource_id: str, owner_id: str, grantee_id: str,
    ) -> bool:
        assert self._db is not None
        async with self._db.execute(
            "DELETE FROM resource_shares"
            " WHERE resource_type = ? AND resource_id = ? AND owner_id = ? AND grantee_id = ?",
            (resource_type, resource_id, owner_id, grantee_id),
        ) as cur:
            await self._db.commit()
            return (cur.rowcount or 0) > 0

    async def delete_shares_for_resource(
        self, resource_type: str, resource_id: str, owner_id: str,
    ) -> int:
        """Drop every grant on one of ``owner_id``'s resources (it was removed)."""
        assert self._db is not None
        async with self._db.execute(
            "DELETE FROM resource_shares"
            " WHERE resource_type = ? AND resource_id = ? AND owner_id = ?",
            (resource_type, resource_id, owner_id),
        ) as cur:
            await self._db.commit()
            return int(cur.rowcount or 0)

    async def is_agent_member(self, agent_ref: str, owner_id: str, user_id: str) -> bool:
        """Has ``owner_id`` shared agent ``agent_ref`` with ``user_id``?"""
        assert self._db is not None
        async with self._db.execute(
            "SELECT 1 FROM resource_shares"
            " WHERE resource_type = 'agent' AND resource_id = ? AND owner_id = ?"
            " AND grantee_id = ? LIMIT 1",
            (agent_ref, owner_id, user_id),
        ) as cur:
            return await cur.fetchone() is not None

    async def agent_ever_shared(self, agent_ref: str, owner_id: str) -> bool:
        """Did ``owner_id``'s agent ever have members (current or removed)?"""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT 1 FROM agent_share_history WHERE agent_ref = ? AND owner_id = ?"
            " UNION ALL SELECT 1 FROM resource_shares WHERE resource_type = 'agent'"
            " AND resource_id = ? AND owner_id = ? LIMIT 1",
            (agent_ref, owner_id, agent_ref, owner_id),
        )
        return bool(rows)

    async def list_agent_members(self, agent_ref: str, owner_id: str) -> list[dict]:
        """Members of one of ``owner_id``'s shared agents."""
        return await self.list_shares_for_resource("agent", agent_ref, owner_id)

    async def list_agent_share_refs(self, user_id: str | None = None) -> list[str]:
        """Agents with members — every one, or those ``user_id`` owns or is a member of."""
        assert self._db is not None
        if user_id is None:
            rows = await self._db.execute_fetchall(
                "SELECT DISTINCT resource_id FROM resource_shares"
                " WHERE resource_type = 'agent' ORDER BY resource_id")
        else:
            rows = await self._db.execute_fetchall(
                "SELECT DISTINCT resource_id FROM resource_shares"
                " WHERE resource_type = 'agent' AND (owner_id = ? OR grantee_id = ?)"
                " ORDER BY resource_id",
                (user_id, user_id),
            )
        return [str(r[0]) for r in rows]

    # ── Context packs (shared-agent shared context, PR B) ─────────────

    async def create_context_pack(
        self, *, agent_ref: str, agent_owner: str, pack_owner: str, kind: str,
        resource_id: str = "", resource_key: str = "", alias: str = "",
        slice_json: str = "{}", max_per_agent: int = 32, max_vfs_per_owner: int = 5,
    ) -> dict | str:
        """Insert one pack; the row dict, or ``"alias_taken"`` / ``"limit"`` /
        ``"vfs_limit"``. Single statements, no explicit transaction (FD shares
        one connection, so a ROLLBACK would undo other coroutines' work): every
        check sits inside the statement that writes. ``sqlite3.IntegrityError``
        (duplicate, alias clash, a deleted user) propagates after a commit."""
        assert self._db is not None
        now = _utcnow()
        if alias:
            # Reserve the alias for this publisher (a no-op when it already is).
            await self._db.execute(
                "INSERT INTO context_pack_aliases (agent_ref, alias, pack_owner, created_at)"
                " VALUES (?, ?, ?, ?) ON CONFLICT(agent_ref, alias) DO NOTHING",
                (agent_ref, alias, pack_owner, now),
            )
            async with self._db.execute(
                "SELECT pack_owner FROM context_pack_aliases WHERE agent_ref = ? AND alias = ?",
                (agent_ref, alias),
            ) as cur:
                row = await cur.fetchone()
            if row is None or row[0] != pack_owner:
                await self._db.commit()
                return "alias_taken"
        pid = _uuid()
        try:
            async with self._db.execute(
                "INSERT INTO context_packs (id, agent_ref, agent_owner, pack_owner, kind,"
                " resource_id, resource_key, alias, slice, created_at)"
                " SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?"
                " WHERE (SELECT COUNT(*) FROM context_packs WHERE agent_ref = ?) < ?"
                " AND (? != 'vfs' OR (SELECT COUNT(*) FROM context_packs WHERE agent_ref = ?"
                " AND pack_owner = ? AND kind = 'vfs') < ?)",
                (pid, agent_ref, agent_owner, pack_owner, kind, resource_id, resource_key,
                 alias, slice_json, now,
                 agent_ref, int(max_per_agent),
                 kind, agent_ref, pack_owner, int(max_vfs_per_owner)),
            ) as cur:
                inserted = int(cur.rowcount or 0)
        except Exception:
            await self._db.commit()  # keeps the caller's own alias reservation — harmless
            raise
        await self._db.commit()
        if inserted == 0:
            if kind == "vfs":
                async with self._db.execute(
                    "SELECT COUNT(*) FROM context_packs WHERE agent_ref = ? AND pack_owner = ?"
                    " AND kind = 'vfs'", (agent_ref, pack_owner),
                ) as cur:
                    row = await cur.fetchone()
                if row and int(row[0]) >= int(max_vfs_per_owner):
                    return "vfs_limit"
            return "limit"
        return {"id": pid, "agent_ref": agent_ref, "agent_owner": agent_owner,
                "pack_owner": pack_owner, "kind": kind, "resource_id": resource_id,
                "resource_key": resource_key, "alias": alias, "slice": slice_json,
                "created_at": now}

    async def get_context_pack(self, pack_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM context_packs WHERE id = ?", (pack_id,),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def list_context_packs_for_agent(self, agent_ref: str) -> list[dict]:
        """Every pack on one agent, with its publisher's display name and email,
        oldest first."""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT p.*, u.display_name AS owner_display_name, u.email AS owner_email"
            " FROM context_packs p JOIN users u ON u.id = p.pack_owner"
            " WHERE p.agent_ref = ? ORDER BY p.created_at, p.id",
            (agent_ref,),
        )
        return [dict(r) for r in rows]

    async def list_context_packs_for_owner(self, pack_owner: str) -> list[dict]:
        """Every pack one user published, on any agent, oldest first."""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT * FROM context_packs WHERE pack_owner = ? ORDER BY created_at, id",
            (pack_owner,),
        )
        return [dict(r) for r in rows]

    async def has_context_packs(self, kind: str) -> bool:
        """Whether any agent has a pack of ``kind`` — a cheap gate the per-call
        agent routes check before identifying their caller."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT 1 FROM context_packs WHERE kind = ? LIMIT 1", (kind,),
        ) as cur:
            return await cur.fetchone() is not None

    async def count_context_packs_for_agent(self, agent_ref: str) -> int:
        assert self._db is not None
        async with self._db.execute(
            "SELECT COUNT(*) FROM context_packs WHERE agent_ref = ?", (agent_ref,),
        ) as cur:
            row = await cur.fetchone()
            return int(row[0]) if row else 0

    async def list_pack_aliases(self, agent_ref: str) -> dict[str, str]:
        """The alias reservations on one agent: alias → publisher."""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT alias, pack_owner FROM context_pack_aliases WHERE agent_ref = ?",
            (agent_ref,),
        )
        return {str(r[0]): str(r[1]) for r in rows}

    async def delete_context_pack(self, pack_id: str) -> bool:
        assert self._db is not None
        async with self._db.execute(
            "DELETE FROM context_packs WHERE id = ?", (pack_id,),
        ) as cur:
            await self._db.commit()
            return (cur.rowcount or 0) > 0

    async def delete_context_packs_for_agent(self, agent_ref: str) -> int:
        """The agent is gone or changed hands: its packs AND its alias
        reservations. Returns how many packs were deleted."""
        assert self._db is not None
        async with self._db.execute(
            "DELETE FROM context_packs WHERE agent_ref = ?", (agent_ref,),
        ) as cur:
            deleted = int(cur.rowcount or 0)
        await self._db.execute(
            "DELETE FROM context_pack_aliases WHERE agent_ref = ?", (agent_ref,))
        await self._db.commit()
        return deleted

    async def delete_context_packs_for_member(self, agent_ref: str, pack_owner: str) -> int:
        """A member lost access: their packs on the agent (their alias
        reservations stay)."""
        assert self._db is not None
        async with self._db.execute(
            "DELETE FROM context_packs WHERE agent_ref = ? AND pack_owner = ?",
            (agent_ref, pack_owner),
        ) as cur:
            await self._db.commit()
            return int(cur.rowcount or 0)

    async def delete_context_packs_for_project(self, pack_owner: str, project: str) -> list[str]:
        """``pack_owner`` deleted their VFS folder ``project``: drop their folder
        packs of it on every agent (matched case-insensitively — Unicode
        casefold, a superset of SQL NOCASE). Returns the agent refs touched."""
        assert self._db is not None
        folded = str(project or "").casefold()
        rows = await self._db.execute_fetchall(
            "SELECT id, agent_ref, resource_id FROM context_packs"
            " WHERE pack_owner = ? AND kind = 'vfs'",
            (pack_owner,),
        )
        hit = [(str(r[0]), str(r[1])) for r in rows if str(r[2]).casefold() == folded]
        for pid, _ref in hit:
            await self._db.execute("DELETE FROM context_packs WHERE id = ?", (pid,))
        await self._db.commit()
        return sorted({ref for _pid, ref in hit})

    async def list_context_pack_refs(self, user_id: str | None = None) -> list[str]:
        """Agents that have packs — every one, or those where ``user_id`` is a
        publisher or the owner recorded when a pack was created."""
        assert self._db is not None
        if user_id is None:
            rows = await self._db.execute_fetchall(
                "SELECT DISTINCT agent_ref FROM context_packs ORDER BY agent_ref")
        else:
            rows = await self._db.execute_fetchall(
                "SELECT DISTINCT agent_ref FROM context_packs"
                " WHERE pack_owner = ? OR agent_owner = ? ORDER BY agent_ref",
                (user_id, user_id),
            )
        return [str(r[0]) for r in rows]

    # ── Cost ledger (persisted run costs) ─────────────────────────────

    async def log_run_cost(
        self, owner_user_id: str, run_kind: str, run_id: str, cost: dict,
        owner_type: str = "user", owner_ref: str = "",
    ) -> None:
        """Persist one run's cost block (the pricing.summarize output)."""
        assert self._db is not None
        import json as _json
        await self._db.execute(
            "INSERT INTO cost_ledger (id, owner_user_id, owner_type, owner_ref,"
            " run_kind, run_id, usage, usd, elapsed_seconds, at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_uuid(), owner_user_id, owner_type, owner_ref, run_kind, run_id,
             _json.dumps(cost.get("tokens") or {}), cost.get("usd"),
             cost.get("elapsed_seconds"), _utcnow()),
        )
        await self._db.commit()

    async def list_run_costs(
        self, owner_user_id: str, limit: int = 200, run_kind: str | None = None,
    ) -> list[dict]:
        assert self._db is not None
        q = "SELECT * FROM cost_ledger WHERE owner_user_id = ?"
        args: list = [owner_user_id]
        if run_kind:
            q += " AND run_kind = ?"
            args.append(run_kind)
        q += " ORDER BY at DESC LIMIT ?"
        args.append(limit)
        async with self._db.execute(q, args) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def aggregate_run_costs(
        self, since: str | None = None, until: str | None = None,
    ) -> dict:
        """Team-wide cost rollup for the admin: totals by user and by run kind.

        `since`/`until` are ISO timestamps bounding `at` (inclusive/exclusive is
        loose — string compare on ISO works). Read-only over cost_ledger.
        """
        assert self._db is not None
        where = ""
        args: list = []
        if since:
            where += (" AND" if where else " WHERE") + " at >= ?"
            args.append(since)
        if until:
            where += (" AND" if where else " WHERE") + " at < ?"
            args.append(until)
        by_user = await self._db.execute_fetchall(
            "SELECT c.owner_user_id AS user_id, u.email AS email,"
            " u.display_name AS display_name, COUNT(*) AS runs,"
            " COALESCE(SUM(c.usd), 0) AS usd,"
            " SUM(CASE WHEN c.usd IS NOT NULL THEN 1 ELSE 0 END) AS priced"
            " FROM cost_ledger c LEFT JOIN users u ON u.id = c.owner_user_id"
            + where +
            " GROUP BY c.owner_user_id ORDER BY usd DESC",
            tuple(args),
        )
        by_kind = await self._db.execute_fetchall(
            "SELECT run_kind, COUNT(*) AS runs, COALESCE(SUM(usd), 0) AS usd"
            " FROM cost_ledger" + where +
            " GROUP BY run_kind ORDER BY usd DESC",
            tuple(args),
        )
        total = 0.0
        for r in by_user:
            total += float(dict(r).get("usd") or 0)
        return {
            "by_user": [dict(r) for r in by_user],
            "by_kind": [dict(r) for r in by_kind],
            "total_usd": round(total, 6),
        }

    # ── Shared Telegram bot links (telegram_shared.py) ─────────────────

    _TELEGRAM_LINK_FIELDS = frozenset({
        "agent_slug", "agent_name", "status", "link_code", "tg_user_id",
        "tg_username", "tg_first_name", "decided_at", "decided_by", "linked_at",
    })

    async def get_telegram_link(self, user_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM telegram_links WHERE user_id = ?", (user_id,),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_telegram_link_by_code(self, link_code: str) -> dict | None:
        if not link_code:
            return None
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM telegram_links WHERE link_code = ?", (link_code,),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def list_telegram_links(self) -> list[dict]:
        """Every link, with the requesting user's email and name."""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT l.*, u.email AS user_email, u.display_name AS user_name"
            " FROM telegram_links l JOIN users u ON u.id = l.user_id"
            " ORDER BY l.requested_at",
        )
        return [dict(r) for r in rows]

    async def create_telegram_link(
        self, user_id: str, agent_slug: str, agent_name: str, link_code: str,
    ) -> dict:
        """A fresh pending request (replaces whatever the user had before)."""
        assert self._db is not None
        now = _utcnow()
        await self._db.execute(
            "INSERT OR REPLACE INTO telegram_links (user_id, agent_slug, agent_name,"
            " status, link_code, requested_at, updated_at)"
            " VALUES (?, ?, ?, 'pending', ?, ?, ?)",
            (user_id, agent_slug, agent_name, link_code, now, now),
        )
        await self._db.commit()
        return await self.get_telegram_link(user_id) or {}

    async def update_telegram_link(self, user_id: str, **fields: Any) -> dict | None:
        assert self._db is not None
        bad = set(fields) - self._TELEGRAM_LINK_FIELDS
        if bad:
            raise ValueError(f"telegram_links column not writable: {sorted(bad)!r}")
        if fields:
            cols = ", ".join(f"{k} = ?" for k in fields)
            await self._db.execute(
                f"UPDATE telegram_links SET {cols}, updated_at = ? WHERE user_id = ?",
                (*fields.values(), _utcnow(), user_id),
            )
            await self._db.commit()
        return await self.get_telegram_link(user_id)

    async def delete_telegram_link(self, user_id: str) -> bool:
        assert self._db is not None
        cur = await self._db.execute("DELETE FROM telegram_links WHERE user_id = ?", (user_id,))
        await self._db.commit()
        return (cur.rowcount or 0) > 0

    # ── Shared-bot groups (telegram_agent_groups.py) ───────────────────

    async def list_tg_agent_groups(self, user_id: str | None = None) -> list[dict]:
        assert self._db is not None
        if user_id is None:
            rows = await self._db.execute_fetchall(
                "SELECT * FROM telegram_agent_groups ORDER BY added_at")
        else:
            rows = await self._db.execute_fetchall(
                "SELECT * FROM telegram_agent_groups WHERE user_id = ? ORDER BY added_at", (user_id,))
        return [dict(r) for r in rows]

    async def set_tg_agent_group(self, chat_id: str, user_id: str, title: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO telegram_agent_groups (chat_id, user_id, title, added_at) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(chat_id) DO UPDATE SET user_id = excluded.user_id, title = excluded.title",
            (chat_id, user_id, title, _utcnow()),
        )
        await self._db.commit()

    async def delete_tg_agent_group(self, chat_id: str) -> bool:
        assert self._db is not None
        cur = await self._db.execute("DELETE FROM telegram_agent_groups WHERE chat_id = ?", (chat_id,))
        await self._db.commit()
        return (cur.rowcount or 0) > 0

    async def move_tg_agent_group(self, old_chat_id: str, new_chat_id: str) -> None:
        """The group became a supergroup (a new id)."""
        assert self._db is not None
        await self._db.execute(
            "UPDATE OR REPLACE telegram_agent_groups SET chat_id = ? WHERE chat_id = ?",
            (new_chat_id, old_chat_id),
        )
        await self._db.commit()

    async def list_admin_ids(self) -> list[str]:
        assert self._db is not None
        rows = await self._db.execute_fetchall("SELECT id FROM users WHERE role = 'admin'")
        return [str(r["id"]) for r in rows]

    # ── Telegram inbox (telegram_inbox.py) ─────────────────────────────
    # `title` and `body` arrive sealed (fd_crypto); this layer never sees text.

    async def get_tg_inbox_settings(self, user_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM telegram_inbox_settings WHERE user_id = ?", (user_id,),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def set_tg_inbox_settings(
        self, user_id: str, *, enabled: bool, consented_at: str | None, retention_days: int,
    ) -> dict:
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO telegram_inbox_settings (user_id, enabled, consented_at, retention_days,"
            " updated_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET"
            " enabled = excluded.enabled, consented_at = excluded.consented_at,"
            " retention_days = excluded.retention_days, updated_at = excluded.updated_at",
            (user_id, 1 if enabled else 0, consented_at, int(retention_days), _utcnow()),
        )
        await self._db.commit()
        return await self.get_tg_inbox_settings(user_id) or {}

    async def get_tg_inbox_connection(self, connection_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM telegram_inbox_connections WHERE connection_id = ?", (connection_id,),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def list_tg_inbox_connections(self, user_id: str) -> list[dict]:
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT * FROM telegram_inbox_connections WHERE user_id = ? ORDER BY connected_at DESC",
            (user_id,),
        )
        return [dict(r) for r in rows]

    async def upsert_tg_inbox_connection(
        self, connection_id: str, user_id: str, tg_user_id: str, *, enabled: bool, rights: str,
    ) -> None:
        assert self._db is not None
        now = _utcnow()
        await self._db.execute(
            "INSERT INTO telegram_inbox_connections (connection_id, user_id, tg_user_id, enabled,"
            " rights, connected_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(connection_id) DO UPDATE SET user_id = excluded.user_id,"
            " tg_user_id = excluded.tg_user_id, enabled = excluded.enabled,"
            " rights = excluded.rights, updated_at = excluded.updated_at",
            (connection_id, user_id, tg_user_id, 1 if enabled else 0, rights, now, now),
        )
        await self._db.commit()

    async def upsert_tg_inbox_chat(
        self, user_id: str, chat_id: str, kind: str, title: str, *,
        last_at: int | None = None, active: bool | None = None,
    ) -> None:
        """Create or refresh a chat: a new title always, ``last_at`` only
        forward, ``active`` only when given."""
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO telegram_inbox_chats (user_id, chat_id, kind, title, active, last_at,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(user_id, chat_id) DO UPDATE SET"
            " kind = excluded.kind, title = CASE WHEN excluded.title != '' THEN excluded.title"
            " ELSE telegram_inbox_chats.title END,"
            " last_at = MAX(telegram_inbox_chats.last_at, excluded.last_at),"
            " active = CASE WHEN ? THEN excluded.active ELSE telegram_inbox_chats.active END",
            (user_id, chat_id, kind, title, 1 if (active is None or active) else 0,
             int(last_at or 0), _utcnow(), 1 if active is not None else 0),
        )
        await self._db.commit()

    async def get_tg_inbox_chat(self, user_id: str, chat_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM telegram_inbox_chats WHERE user_id = ? AND chat_id = ?",
            (user_id, chat_id),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def list_tg_inbox_chats(self, user_id: str) -> list[dict]:
        """The user's chats, newest activity first, with how many messages
        each holds."""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT c.*, (SELECT COUNT(*) FROM telegram_inbox_messages m"
            "  WHERE m.user_id = c.user_id AND m.chat_id = c.chat_id) AS messages"
            " FROM telegram_inbox_chats c WHERE c.user_id = ? ORDER BY c.last_at DESC",
            (user_id,),
        )
        return [dict(r) for r in rows]

    async def tg_inbox_group_subscribers(self, chat_id: str) -> list[str]:
        """Users collecting this group now (their inbox on, the group active)."""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT c.user_id FROM telegram_inbox_chats c JOIN telegram_inbox_settings s"
            " ON s.user_id = c.user_id WHERE c.chat_id = ? AND c.kind = 'group'"
            " AND c.active = 1 AND s.enabled = 1",
            (chat_id,),
        )
        return [str(r["user_id"]) for r in rows]

    async def set_tg_inbox_group_active(self, chat_id: str, active: bool, user_id: str | None = None) -> None:
        assert self._db is not None
        if user_id is None:
            await self._db.execute(
                "UPDATE telegram_inbox_chats SET active = ? WHERE chat_id = ? AND kind = 'group'",
                (1 if active else 0, chat_id),
            )
        else:
            await self._db.execute(
                "UPDATE telegram_inbox_chats SET active = ? WHERE chat_id = ? AND user_id = ?"
                " AND kind = 'group'",
                (1 if active else 0, chat_id, user_id),
            )
        await self._db.commit()

    async def move_tg_inbox_chat(self, old_chat_id: str, new_chat_id: str) -> None:
        """A group became a supergroup: a new id whose message ids start over,
        so whoever kept the old one keeps the new one; the old messages stay
        under the old (now inactive) id."""
        assert self._db is not None
        await self._db.execute(
            "INSERT OR IGNORE INTO telegram_inbox_chats (user_id, chat_id, kind, title, active,"
            " last_at, created_at) SELECT user_id, ?, kind, title, active, 0, ?"
            " FROM telegram_inbox_chats WHERE chat_id = ?",
            (new_chat_id, _utcnow(), old_chat_id),
        )
        await self._db.execute(
            "UPDATE telegram_inbox_chats SET active = 0 WHERE chat_id = ?", (old_chat_id,))
        await self._db.commit()

    async def add_tg_inbox_message(
        self, user_id: str, chat_id: str, message_id: int, *, sender_id: str, outgoing: bool,
        sent_at: int, body: str, reply_to: int | None,
    ) -> bool:
        """Store one message (a repeat of a stored one is ignored)."""
        assert self._db is not None
        cur = await self._db.execute(
            "INSERT OR IGNORE INTO telegram_inbox_messages (user_id, chat_id, message_id,"
            " sender_id, outgoing, sent_at, body, reply_to) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, chat_id, int(message_id), sender_id, 1 if outgoing else 0, int(sent_at),
             body, reply_to),
        )
        await self._db.commit()
        return (cur.rowcount or 0) > 0

    async def edit_tg_inbox_message(
        self, user_id: str, chat_id: str, message_id: int, *, body: str, edited_at: int,
    ) -> bool:
        assert self._db is not None
        cur = await self._db.execute(
            "UPDATE telegram_inbox_messages SET body = ?, edited_at = ? WHERE user_id = ?"
            " AND chat_id = ? AND message_id = ? AND deleted = 0",
            (body, int(edited_at), user_id, chat_id, int(message_id)),
        )
        await self._db.commit()
        return (cur.rowcount or 0) > 0

    async def get_tg_inbox_message(self, user_id: str, chat_id: str, message_id: int) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM telegram_inbox_messages WHERE user_id = ? AND chat_id = ?"
            " AND message_id = ?", (user_id, chat_id, int(message_id)),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def mark_tg_inbox_deleted(self, user_id: str, chat_id: str, message_id: int, *, body: str) -> None:
        """Someone deleted it in Telegram: ``body`` is what's left (no text)."""
        assert self._db is not None
        await self._db.execute(
            "UPDATE telegram_inbox_messages SET body = ?, deleted = 1 WHERE user_id = ?"
            " AND chat_id = ? AND message_id = ?",
            (body, user_id, chat_id, int(message_id)),
        )
        await self._db.commit()

    async def list_tg_inbox_messages(
        self, user_id: str, *, chat_id: str | None = None, since: int | None = None,
        until: int | None = None, limit: int = 50,
    ) -> list[dict]:
        """Newest first."""
        assert self._db is not None
        where = ["user_id = ?"]
        args: list[Any] = [user_id]
        if chat_id is not None:
            where.append("chat_id = ?")
            args.append(chat_id)
        if since is not None:
            where.append("sent_at >= ?")
            args.append(int(since))
        if until is not None:
            where.append("sent_at < ?")
            args.append(int(until))
        rows = await self._db.execute_fetchall(
            f"SELECT * FROM telegram_inbox_messages WHERE {' AND '.join(where)}"
            " ORDER BY sent_at DESC, id DESC LIMIT ?",
            (*args, max(1, int(limit))),
        )
        return [dict(r) for r in rows]

    async def tg_inbox_stats(self, user_id: str) -> dict:
        assert self._db is not None
        async with self._db.execute(
            "SELECT COUNT(*) AS messages, MIN(sent_at) AS oldest, MAX(sent_at) AS newest,"
            " COUNT(DISTINCT chat_id) AS chats FROM telegram_inbox_messages WHERE user_id = ?",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else {"messages": 0, "oldest": None, "newest": None, "chats": 0}

    async def delete_tg_inbox_data(self, user_id: str, *, chat_id: str | None = None,
                                   connections: bool = False, keep_groups: bool = False) -> int:
        """Delete a user's stored messages and chats (one chat's, or all);
        ``keep_groups`` keeps the groups they collect (emptied); with
        ``connections`` also their Chat Automation connections."""
        assert self._db is not None
        if chat_id is None:
            cur = await self._db.execute(
                "DELETE FROM telegram_inbox_messages WHERE user_id = ?", (user_id,))
            await self._db.execute(
                "DELETE FROM telegram_inbox_chats WHERE user_id = ?"
                + (" AND kind != 'group'" if keep_groups else ""), (user_id,))
        else:
            cur = await self._db.execute(
                "DELETE FROM telegram_inbox_messages WHERE user_id = ? AND chat_id = ?",
                (user_id, chat_id))
            await self._db.execute(
                "DELETE FROM telegram_inbox_chats WHERE user_id = ? AND chat_id = ?",
                (user_id, chat_id))
        if connections:
            await self._db.execute(
                "DELETE FROM telegram_inbox_connections WHERE user_id = ?", (user_id,))
        if chat_id is None:
            await self._db.execute(
                "DELETE FROM telegram_inbox_cursors WHERE user_id = ?", (user_id,))
        await self._db.commit()
        return cur.rowcount or 0

    async def tg_inbox_cursors(self, user_id: str, agent_slug: str) -> dict[str, int]:
        """``{chat_id: last_id}`` for one agent ('' = every chat)."""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT chat_id, last_id FROM telegram_inbox_cursors WHERE user_id = ? AND agent_slug = ?",
            (user_id, agent_slug),
        )
        return {str(r["chat_id"]): int(r["last_id"]) for r in rows}

    async def advance_tg_inbox_cursor(
        self, user_id: str, agent_slug: str, chat_id: str, last_id: int,
    ) -> None:
        """Move one chat's cursor forward (never back)."""
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO telegram_inbox_cursors (user_id, agent_slug, chat_id, last_id, updated_at)"
            " VALUES (?, ?, ?, ?, ?) ON CONFLICT(user_id, agent_slug, chat_id) DO UPDATE SET"
            " last_id = MAX(telegram_inbox_cursors.last_id, excluded.last_id),"
            " updated_at = excluded.updated_at",
            (user_id, agent_slug, chat_id, int(last_id), _utcnow()),
        )
        await self._db.commit()

    async def set_tg_inbox_seen_all(self, user_id: str, agent_slug: str, last_id: int) -> None:
        """The agent was told about everything up to ``last_id``: one cursor
        for every chat, the per-chat ones it covers dropped."""
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO telegram_inbox_cursors (user_id, agent_slug, chat_id, last_id, updated_at)"
            " VALUES (?, ?, '', ?, ?) ON CONFLICT(user_id, agent_slug, chat_id) DO UPDATE SET"
            " last_id = MAX(telegram_inbox_cursors.last_id, excluded.last_id),"
            " updated_at = excluded.updated_at",
            (user_id, agent_slug, int(last_id), _utcnow()),
        )
        await self._db.execute(
            "DELETE FROM telegram_inbox_cursors WHERE user_id = ? AND agent_slug = ?"
            " AND chat_id != '' AND last_id <= ?",
            (user_id, agent_slug, int(last_id)),
        )
        await self._db.commit()

    async def tg_inbox_unseen(self, user_id: str, *, after_id: int, since: int | None = None) -> list[dict]:
        """``(id, chat_id)`` of messages others sent after ``after_id`` (and
        at or after ``since``), still there — ids and chats only, no text."""
        assert self._db is not None
        sql = ("SELECT id, chat_id FROM telegram_inbox_messages WHERE user_id = ? AND id > ?"
               " AND outgoing = 0 AND deleted = 0")
        args: list[Any] = [user_id, int(after_id)]
        if since is not None:
            sql += " AND sent_at >= ?"
            args.append(int(since))
        rows = await self._db.execute_fetchall(sql, args)
        return [dict(r) for r in rows]

    async def tg_inbox_max_id(self, user_id: str) -> int:
        assert self._db is not None
        async with self._db.execute(
            "SELECT COALESCE(MAX(id), 0) AS top FROM telegram_inbox_messages WHERE user_id = ?",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
            return int(row["top"]) if row else 0

    async def purge_tg_inbox_expired(self, now: int) -> int:
        """Drop messages past their owner's retention (and everything of a user
        whose inbox is off), then private chats left with no messages."""
        assert self._db is not None
        cur = await self._db.execute(
            "DELETE FROM telegram_inbox_messages WHERE sent_at < ? - 86400 * COALESCE("
            " (SELECT s.retention_days FROM telegram_inbox_settings s"
            "  WHERE s.user_id = telegram_inbox_messages.user_id AND s.enabled = 1), 0)",
            (int(now),),
        )
        await self._db.execute(
            "DELETE FROM telegram_inbox_chats WHERE kind = 'private' AND NOT EXISTS ("
            " SELECT 1 FROM telegram_inbox_messages m WHERE m.user_id = telegram_inbox_chats.user_id"
            " AND m.chat_id = telegram_inbox_chats.chat_id)",
        )
        await self._db.commit()
        return cur.rowcount or 0

    # ── Notifications (persistent in-app bell) ─────────────────────────

    async def add_notification(
        self, user_id: str, type: str, title: str, body: str = "",
        ref_type: str = "", ref_id: str = "",
    ) -> str:
        """Insert one notification for `user_id`; returns its id."""
        assert self._db is not None
        nid = _uuid()
        await self._db.execute(
            "INSERT INTO notifications (id, user_id, type, title, body,"
            " ref_type, ref_id, read, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)",
            (nid, user_id, type, title, body, ref_type, ref_id, _utcnow()),
        )
        await self._db.commit()
        return nid

    async def list_notifications(
        self, user_id: str, limit: int = 50, unread_only: bool = False,
    ) -> list[dict]:
        assert self._db is not None
        q = "SELECT * FROM notifications WHERE user_id = ?"
        args: list = [user_id]
        if unread_only:
            q += " AND read = 0"
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        rows = await self._db.execute_fetchall(q, tuple(args))
        return [dict(r) for r in rows]

    async def list_notifications_by_ref(
        self, user_id: str, ref_type: str, ref_ids: Iterable[str],
    ) -> list[dict]:
        """This user's notifications about any of `ref_ids` (of `ref_type`),
        newest first, however old."""
        assert self._db is not None
        ids = [str(i) for i in ref_ids if i]
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = await self._db.execute_fetchall(
            f"SELECT * FROM notifications WHERE user_id = ? AND ref_type = ? AND ref_id IN ({marks})"
            " ORDER BY created_at DESC",
            (user_id, ref_type, *ids),
        )
        return [dict(r) for r in rows]

    async def count_unread_notifications(self, user_id: str) -> int:
        assert self._db is not None
        async with self._db.execute(
            "SELECT COUNT(*) FROM notifications WHERE user_id = ? AND read = 0",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
            return int(row[0]) if row else 0

    async def count_notifications_since(
        self, user_id: str, since_iso: str, types: Iterable[str] | str = (), *,
        ref_type: str = "", ref_id: str = "",
    ) -> int:
        """How many bells `user_id` got at/after `since_iso`, read or not —
        optionally only of `types` and with this `ref_type` / `ref_id`. For
        per-day bell caps and per-reason dedupe that survive a restart."""
        assert self._db is not None
        q = "SELECT COUNT(*) FROM notifications WHERE user_id = ? AND created_at >= ?"
        args: list = [user_id, since_iso or ""]
        kinds = [t for t in ([types] if isinstance(types, str) else types) if t]
        if kinds:
            q += f" AND type IN ({','.join('?' * len(kinds))})"
            args.extend(kinds)
        if ref_type:
            q += " AND ref_type = ?"
            args.append(ref_type)
        if ref_id:
            q += " AND ref_id = ?"
            args.append(ref_id)
        async with self._db.execute(q, tuple(args)) as cur:
            row = await cur.fetchone()
            return int(row[0]) if row else 0

    async def mark_notification_read(self, notif_id: str, user_id: str) -> bool:
        assert self._db is not None
        async with self._db.execute(
            "UPDATE notifications SET read = 1 WHERE id = ? AND user_id = ?",
            (notif_id, user_id),
        ) as cur:
            await self._db.commit()
            return (cur.rowcount or 0) > 0

    async def mark_all_notifications_read(self, user_id: str) -> int:
        assert self._db is not None
        async with self._db.execute(
            "UPDATE notifications SET read = 1 WHERE user_id = ? AND read = 0",
            (user_id,),
        ) as cur:
            await self._db.commit()
            return cur.rowcount or 0

    # ── Gmail sends (audit for /fd/google/gmail/send) ─────────────────

    async def add_gmail_send(
        self, owner_id: str, agent: str = "", to_addrs: str = "",
        cc_addrs: str = "", bcc_addrs: str = "", subject: str = "",
        gmail_message_id: str = "", thread_id: str = "", draft_id: str = "",
        content_hash: str = "", status: str = "sent", attachments: str = "",
        origin: str = "agent", topic_id: str = "", in_reply_to_message_id: str = "",
    ) -> str:
        """Record one email Gmail accepted (`status` 'sent') — or may have
        sent ('unknown') — for `owner_id`; returns its id. `attachments` is
        a JSON list of the attached files' names ('' = none). `origin` says
        who decided the send ('agent' | 'autoreply' | 'autoreply_approved');
        an auto-reply also records its topic and the inbound message it
        answered."""
        assert self._db is not None
        sid = _uuid()
        await self._db.execute(
            "INSERT INTO gmail_sends (id, owner_id, agent, to_addrs, cc_addrs,"
            " bcc_addrs, subject, gmail_message_id, thread_id, draft_id,"
            " content_hash, status, created_at, attachments, origin, topic_id,"
            " in_reply_to_message_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sid, owner_id, agent, to_addrs, cc_addrs, bcc_addrs, subject,
             gmail_message_id, thread_id, draft_id, content_hash, status, _utcnow(),
             attachments, origin, topic_id, in_reply_to_message_id),
        )
        await self._db.commit()
        return sid

    async def count_gmail_sends_since(
        self, owner_id: str, since_iso: str, origin: str | None = None,
    ) -> int:
        """How many emails `owner_id` sent (or may have sent) at or after
        `since_iso` (ISO UTC). `origin` narrows to one origin ('agent',
        'autoreply', …); None counts every row (the original behaviour)."""
        assert self._db is not None
        q = "SELECT COUNT(*) FROM gmail_sends WHERE owner_id = ? AND created_at >= ?"
        args: list = [owner_id, since_iso]
        if origin is not None:
            q += " AND origin = ?"
            args.append(origin)
        async with self._db.execute(q, tuple(args)) as cur:
            row = await cur.fetchone()
            return int(row[0]) if row else 0

    async def list_gmail_sends(
        self, owner_id: str, limit: int = 20, origin: str | None = None,
    ) -> list[dict]:
        """`owner_id`'s sends, newest first — only one `origin`'s when given,
        else every row."""
        assert self._db is not None
        q = "SELECT * FROM gmail_sends WHERE owner_id = ?"
        args: list = [owner_id]
        if origin is not None:
            q += " AND origin = ?"
            args.append(origin)
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        rows = await self._db.execute_fetchall(q, tuple(args))
        return [dict(r) for r in rows]

    async def find_gmail_send_by_hash(
        self, owner_id: str, content_hash: str, since_iso: str,
    ) -> dict | None:
        """The newest send by `owner_id` with this content hash at or after
        `since_iso`, else None — the duplicate check."""
        assert self._db is not None
        if not content_hash:
            return None
        async with self._db.execute(
            "SELECT * FROM gmail_sends WHERE owner_id = ? AND content_hash = ?"
            " AND created_at >= ? ORDER BY created_at DESC LIMIT 1",
            (owner_id, content_hash, since_iso),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    # ── Gmail topic auto-replies: per-owner poll state ────────────────

    async def get_autoreply_state(self, owner_id: str) -> dict | None:
        """`owner_id`'s poll state (`own_addresses` parsed to a list), or None
        when auto-replies are off for them."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM gmail_autoreply_state WHERE owner_id = ?", (owner_id,)
        ) as cur:
            row = await cur.fetchone()
            return _autoreply_state_row(row) if row else None

    async def set_autoreply_state(self, owner_id: str, **fields: Any) -> None:
        """Create `owner_id`'s state row, or merge `fields` into the existing
        one (columns not named keep their value — the classification budget
        survives a re-set). Only the enabling PUT should create a row: the
        engine writes through update_autoreply_state, which never does."""
        assert self._db is not None
        cols = _autoreply_state_fields(fields)
        cols["updated_at"] = _utcnow()
        await self._db.execute(
            f"INSERT INTO gmail_autoreply_state (owner_id, {', '.join(cols)})"
            f" VALUES (?, {', '.join('?' * len(cols))})"
            " ON CONFLICT(owner_id) DO UPDATE SET "
            + ", ".join(f"{n} = excluded.{n}" for n in cols),
            (owner_id, *cols.values()),
        )
        await self._db.commit()

    async def update_autoreply_state(
        self, owner_id: str, *, if_enabled_at: str | None = None, **fields: Any,
    ) -> bool:
        """Write `fields` into an EXISTING state row; never creates one, so a
        tick that outlives a turn-off cannot switch the engine back on. With
        `if_enabled_at` the write lands only while the row is still that
        enablement (not deleted and re-created in between). True when a row
        was updated."""
        assert self._db is not None
        cols = _autoreply_state_fields(fields)
        cols["updated_at"] = _utcnow()
        q = ("UPDATE gmail_autoreply_state SET "
             + ", ".join(f"{n} = ?" for n in cols) + " WHERE owner_id = ?")
        args: list = [*cols.values(), owner_id]
        if if_enabled_at is not None:
            q += " AND enabled_at = ?"
            args.append(if_enabled_at)
        async with self._db.execute(q, tuple(args)) as cur:
            await self._db.commit()
            return (cur.rowcount or 0) == 1

    async def delete_autoreply_state(self, owner_id: str) -> bool:
        """Drop `owner_id`'s state row (auto-replies off); True if one existed."""
        assert self._db is not None
        async with self._db.execute(
            "DELETE FROM gmail_autoreply_state WHERE owner_id = ?", (owner_id,)
        ) as cur:
            await self._db.commit()
            return (cur.rowcount or 0) > 0

    async def list_autoreply_states(self) -> list[dict]:
        """Every existing state row — the owners the engine polls."""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT * FROM gmail_autoreply_state ORDER BY enabled_at, owner_id"
        )
        return [_autoreply_state_row(r) for r in rows]

    async def incr_classify_budget(self, owner_id: str, day: str, limit: int) -> bool:
        """Take one classification from `owner_id`'s budget for UTC `day`
        ('YYYY-MM-DD'): True when it was below `limit` (now counted), False
        when spent — or when no state row exists (fail closed). One atomic
        UPDATE; a new day restarts the count at 1."""
        assert self._db is not None
        if not day or int(limit) <= 0:
            return False
        async with self._db.execute(
            "UPDATE gmail_autoreply_state SET"
            " classify_count = CASE WHEN classify_day = ? THEN classify_count + 1 ELSE 1 END,"
            " classify_day = ?, updated_at = ?"
            " WHERE owner_id = ? AND (classify_day != ? OR classify_count < ?)",
            (day, day, _utcnow(), owner_id, day, int(limit)),
        ) as cur:
            await self._db.commit()
            return (cur.rowcount or 0) == 1

    # ── Gmail topic auto-replies: the exactly-once ledger ─────────────

    async def claim_autoreply(
        self, owner_id: str, inbound_message_id: str, thread_id: str = "", *,
        state: str = "new", reason: str = "",
    ) -> str | None:
        """Claim one inbound message for `owner_id` (INSERT OR IGNORE): its new
        ledger id ('ar_<hex16>'), or None when it was already claimed — the
        exactly-once point, taken before any fetch or LLM call. `state` is
        'new', or 'skipped' with a `reason` (a burst's overflow)."""
        assert self._db is not None
        if state not in ("new", "skipped"):
            raise ValueError(f"a claim starts as 'new' or 'skipped', not {state!r}")
        if not inbound_message_id:
            return None
        rid = "ar_" + _uuid()[:16]
        now = _utcnow()
        async with self._db.execute(
            "INSERT OR IGNORE INTO gmail_autoreplies (id, owner_id, inbound_message_id,"
            " thread_id, state, reason, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (rid, owner_id, inbound_message_id, _text(thread_id), state,
             _text(reason, _AUTOREPLY_TEXT_COLUMNS["reason"]), now, now),
        ) as cur:
            await self._db.commit()
            return rid if (cur.rowcount or 0) == 1 else None

    async def cas_autoreply(
        self, row_id: str, owner_id: str, from_states: Iterable[str] | str,
        to_state: str, *, inc_attempts: bool = False, **fields: Any,
    ) -> bool:
        """Compare-and-set: move `owner_id`'s row `row_id` to `to_state` (also
        writing the whitelisted `fields`, and `attempts + 1` with
        `inc_attempts`) ONLY while it is in one of `from_states`. True when
        this call made the move; False means another writer got there first
        or the row is not this owner's."""
        assert self._db is not None
        froms = _autoreply_states(from_states)
        (to,) = _autoreply_states((to_state,))
        if not froms:
            return False
        cols = _autoreply_fields(fields)
        cols["state"] = to
        cols["updated_at"] = _utcnow()
        sets = [f"{n} = ?" for n in cols]
        if inc_attempts and "attempts" not in cols:
            sets.append("attempts = attempts + 1")
        async with self._db.execute(
            f"UPDATE gmail_autoreplies SET {', '.join(sets)}"
            f" WHERE id = ? AND owner_id = ? AND state IN ({','.join('?' * len(froms))})",
            (*cols.values(), row_id, owner_id, *froms),
        ) as cur:
            await self._db.commit()
            return (cur.rowcount or 0) == 1

    async def update_autoreply(self, row_id: str, owner_id: str, **fields: Any) -> bool:
        """Write whitelisted columns of `owner_id`'s row `row_id`, whatever its
        state (never the state itself — that is cas_autoreply's). Unknown
        names raise ValueError. True when the row was updated."""
        assert self._db is not None
        cols = _autoreply_fields(fields)
        if not cols:
            return False
        cols["updated_at"] = _utcnow()
        async with self._db.execute(
            "UPDATE gmail_autoreplies SET " + ", ".join(f"{n} = ?" for n in cols)
            + " WHERE id = ? AND owner_id = ?",
            (*cols.values(), row_id, owner_id),
        ) as cur:
            await self._db.commit()
            return (cur.rowcount or 0) == 1

    async def get_autoreply(self, owner_id: str, row_id: str) -> dict | None:
        """`owner_id`'s ledger row `row_id` (JSON columns parsed), else None."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM gmail_autoreplies WHERE id = ? AND owner_id = ?",
            (row_id, owner_id),
        ) as cur:
            row = await cur.fetchone()
            return _autoreply_row(row) if row else None

    async def next_autoreplies(
        self, owner_id: str, states: Iterable[str] | str, limit: int = 5,
    ) -> list[dict]:
        """`owner_id`'s rows in `states`, oldest claim first — the work queue."""
        assert self._db is not None
        wanted = _autoreply_states(states)
        if not wanted:
            return []
        rows = await self._db.execute_fetchall(
            "SELECT * FROM gmail_autoreplies WHERE owner_id = ?"
            f" AND state IN ({','.join('?' * len(wanted))})"
            " ORDER BY created_at ASC, rowid ASC LIMIT ?",
            (owner_id, *wanted, max(1, int(limit))),
        )
        return [_autoreply_row(r) for r in rows]

    async def count_autoreplies(
        self, owner_id: str, since_iso: str, states: Iterable[str] | str | None = None, *,
        thread_id: str = "", from_addr: str = "", topic_id: str = "",
        reasons: Iterable[str] = (), arrived_only: bool = False,
    ) -> int:
        """How many of `owner_id`'s rows were claimed OR last changed at/after
        `since_iso` ('' = ever), in `states` (None = any state). So a held
        reply the owner approved days after it arrived counts from the moment
        it went out. With `arrived_only`, claimed at/after only (arrivals, for
        the flood rule). Optional exact filters: `thread_id`, `from_addr`
        (case-folded), `topic_id`, and `reasons` (any of)."""
        assert self._db is not None
        if arrived_only:
            q = "SELECT COUNT(*) FROM gmail_autoreplies WHERE owner_id = ? AND created_at >= ?"
            args: list = [owner_id, since_iso or ""]
        else:
            q = ("SELECT COUNT(*) FROM gmail_autoreplies WHERE owner_id = ?"
                 " AND (created_at >= ? OR updated_at >= ?)")
            args = [owner_id, since_iso or "", since_iso or ""]
        wanted = _autoreply_states(states)
        if wanted:
            q += f" AND state IN ({','.join('?' * len(wanted))})"
            args.extend(wanted)
        if thread_id:
            q += " AND thread_id = ?"
            args.append(thread_id)
        if from_addr:
            q += " AND from_addr = ?"
            args.append(from_addr.strip().lower())
        if topic_id:
            q += " AND topic_id = ?"
            args.append(topic_id)
        why = [r for r in ([reasons] if isinstance(reasons, str) else reasons) if r]
        if why:
            q += f" AND reason IN ({','.join('?' * len(why))})"
            args.extend(why)
        async with self._db.execute(q, tuple(args)) as cur:
            row = await cur.fetchone()
            return int(row[0]) if row else 0

    async def list_autoreplies(
        self, owner_id: str, limit: int = 50, include_quiet: bool = False,
        quiet_reasons: Iterable[str] = AUTOREPLY_QUIET_REASONS, *,
        states: Iterable[str] | str | None = None, since_iso: str = "",
    ) -> list[dict]:
        """`owner_id`'s ledger rows, newest first (JSON columns parsed). Quiet
        skips — state 'skipped' with a reason in `quiet_reasons` — are left
        out unless `include_quiet`. Optional `states` and `since_iso` (claimed
        at/after) narrow further. `limit` is clamped to 1..500."""
        assert self._db is not None
        q = "SELECT * FROM gmail_autoreplies WHERE owner_id = ?"
        args: list = [owner_id]
        quiet = list(quiet_reasons or ())
        if not include_quiet and quiet:
            q += f" AND NOT (state = 'skipped' AND reason IN ({','.join('?' * len(quiet))}))"
            args.extend(quiet)
        wanted = _autoreply_states(states)
        if wanted:
            q += f" AND state IN ({','.join('?' * len(wanted))})"
            args.extend(wanted)
        if since_iso:
            q += " AND created_at >= ?"
            args.append(since_iso)
        q += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        args.append(max(1, min(int(limit), 500)))
        rows = await self._db.execute_fetchall(q, tuple(args))
        return [_autoreply_row(r) for r in rows]

    async def autoreply_sent_ids(self, owner_id: str, thread_id: str) -> set[str]:
        """The Gmail ids of the auto-replies `owner_id`'s ledger sent in
        `thread_id` — the human-takeover rule's "ours, not the owner's"."""
        assert self._db is not None
        if not thread_id:
            return set()
        rows = await self._db.execute_fetchall(
            "SELECT sent_message_id FROM gmail_autoreplies"
            " WHERE owner_id = ? AND thread_id = ? AND sent_message_id != ''",
            (owner_id, thread_id),
        )
        return {r["sent_message_id"] for r in rows}

    async def autoreply_match_stats(self, owner_id: str, since_iso: str) -> dict:
        """The drift breaker's sample: rows claimed at/after `since_iso` that
        were classified (`match_judge` set) and, of those, how many per
        `topic_id` (set only when the classifier accepted a topic):
        ``{"classified": n, "matched": {topic_id: n}}``."""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT topic_id, COUNT(*) AS n FROM gmail_autoreplies"
            " WHERE owner_id = ? AND created_at >= ? AND match_judge != ''"
            " GROUP BY topic_id",
            (owner_id, since_iso or ""),
        )
        classified = 0
        matched: dict[str, int] = {}
        for r in rows:
            classified += int(r["n"])
            if r["topic_id"]:
                matched[r["topic_id"]] = int(r["n"])
        return {"classified": classified, "matched": matched}

    async def claim_autoreply_runner(self, pid: int, host: str, now_iso: str,
                                     stale_before_iso: str) -> bool:
        """Claim (or keep) the right to run the auto-reply engine on this
        database: ours when no runner is recorded, it is already us, or its
        heartbeat is older than `stale_before_iso`. A held claim's heartbeat
        moves to `now_iso`. True when this process is the runner."""
        assert self._db is not None
        async with self._db.execute(
            "UPDATE gmail_autoreply_runner SET heartbeat_at = ? WHERE id = 1 AND pid = ? AND host = ?",
            (now_iso, pid, host),
        ) as cur:
            if (cur.rowcount or 0) == 1:
                await self._db.commit()
                return True
        async with self._db.execute(
            "INSERT INTO gmail_autoreply_runner (id, pid, host, claimed_at, heartbeat_at)"
            " VALUES (1, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET pid = excluded.pid, host = excluded.host,"
            " claimed_at = excluded.claimed_at, heartbeat_at = excluded.heartbeat_at"
            " WHERE gmail_autoreply_runner.heartbeat_at < ?",
            (pid, host, now_iso, now_iso, stale_before_iso),
        ) as cur:
            took = (cur.rowcount or 0) == 1
        await self._db.commit()
        return took

    async def release_autoreply_runner(self, pid: int, host: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "DELETE FROM gmail_autoreply_runner WHERE id = 1 AND pid = ? AND host = ?",
            (pid, host))
        await self._db.commit()

    async def autoreply_queue_owners(self) -> list[str]:
        """Owners with rows in `new` or `waiting_agent`."""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT DISTINCT owner_id FROM gmail_autoreplies"
            " WHERE state IN ('new', 'waiting_agent')")
        return [str(r[0]) for r in rows]

    async def interrupted_autoreplies(self) -> list[dict]:
        """Rows, every owner, that :meth:`recover_autoreplies` will fail as
        'interrupted' (a work state with its recover attempts used up) —
        read just before it runs, so the engine can release their claims."""
        assert self._db is not None
        work = ",".join("?" * len(AUTOREPLY_WORK_STATES))
        rows = await self._db.execute_fetchall(
            "SELECT id, owner_id, inbound_message_id FROM gmail_autoreplies"
            f" WHERE state IN ({work}) AND attempts >= ? AND approved_by = ''",
            (*AUTOREPLY_WORK_STATES, AUTOREPLY_RECOVER_ATTEMPTS),
        )
        return [dict(r) for r in rows]

    async def recover_autoreplies(self) -> list[dict]:
        """Boot recovery, every owner, once before the loop's first tick:

        - `sending` → `unknown` (reason 'interrupted'): the send may have
          happened, so it is NEVER retried; the rows are returned so the
          engine can ring "may have auto-replied — check Sent".
        - `approved` by the owner (`approved_by` set) → back to `held` (and
          `approved_by` cleared): the owner clicks Send again.
        - other `matching` / `writing` / `checking` / `approved` → `new` with
          `attempts + 1` while attempts < AUTOREPLY_RECOVER_ATTEMPTS, else
          `failed` (reason 'interrupted'); their unchecked reply_text is cleared.
        - `new` and `waiting_agent` are left alone.
        """
        assert self._db is not None
        now = _utcnow()
        sending = await self._db.execute_fetchall(
            "SELECT id FROM gmail_autoreplies WHERE state = 'sending'"
        )
        moved: list[str] = []
        for r in sending:
            async with self._db.execute(
                "UPDATE gmail_autoreplies SET state = 'unknown', reason = 'interrupted',"
                " updated_at = ? WHERE id = ? AND state = 'sending'",
                (now, r["id"]),
            ) as cur:
                if (cur.rowcount or 0) == 1:
                    moved.append(r["id"])
        await self._db.execute(
            "UPDATE gmail_autoreplies SET state = 'held', approved_by = '', updated_at = ?"
            " WHERE state = 'approved' AND approved_by != ''",
            (now,),
        )
        work = ",".join("?" * len(AUTOREPLY_WORK_STATES))
        await self._db.execute(
            "UPDATE gmail_autoreplies SET state = 'new', attempts = attempts + 1,"
            f" reply_text = '', updated_at = ? WHERE state IN ({work}) AND attempts < ?",
            (now, *AUTOREPLY_WORK_STATES, AUTOREPLY_RECOVER_ATTEMPTS),
        )
        await self._db.execute(
            "UPDATE gmail_autoreplies SET state = 'failed', reason = 'interrupted',"
            " detail = 'interrupted while ' || state, reply_text = '', updated_at = ?"
            f" WHERE state IN ({work})",
            (now, *AUTOREPLY_WORK_STATES),
        )
        await self._db.commit()
        if not moved:
            return []
        rows = await self._db.execute_fetchall(
            f"SELECT * FROM gmail_autoreplies WHERE id IN ({','.join('?' * len(moved))})"
            " ORDER BY created_at",
            tuple(moved),
        )
        return [_autoreply_row(r) for r in rows]

    async def purge_autoreplies(self, now: datetime | str | None = None) -> dict[str, int]:
        """Retention, every owner (by claim time): held cards older than
        AUTOREPLY_HELD_DAYS → `dismissed` (reason 'expired'); reply_text
        cleared after AUTOREPLY_TEXT_DAYS; skipped / failed rows deleted after
        AUTOREPLY_QUIET_DAYS; every row deleted after AUTOREPLY_KEEP_DAYS.
        Clearing text leaves updated_at alone, so an old send never re-enters
        a cap window. `now` (datetime or ISO; naive = UTC) defaults to the
        current time. Returns ``{"expired", "cleared", "deleted"}`` counts."""
        assert self._db is not None
        if isinstance(now, str):
            now = datetime.fromisoformat(now)
        now = now or datetime.now(timezone.utc)
        now = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)

        def before(days: int) -> str:
            return (now - timedelta(days=days)).isoformat()

        async with self._db.execute(
            "UPDATE gmail_autoreplies SET state = 'dismissed', reason = 'expired',"
            " updated_at = ? WHERE state = 'held' AND created_at < ?",
            (now.isoformat(), before(AUTOREPLY_HELD_DAYS)),
        ) as cur:
            expired = int(cur.rowcount or 0)
        async with self._db.execute(
            "UPDATE gmail_autoreplies SET reply_text = ''"
            " WHERE reply_text != '' AND created_at < ?",
            (before(AUTOREPLY_TEXT_DAYS),),
        ) as cur:
            cleared = int(cur.rowcount or 0)
        async with self._db.execute(
            "DELETE FROM gmail_autoreplies WHERE state IN ('skipped', 'failed')"
            " AND created_at < ?",
            (before(AUTOREPLY_QUIET_DAYS),),
        ) as cur:
            deleted = int(cur.rowcount or 0)
        async with self._db.execute(
            "DELETE FROM gmail_autoreplies WHERE created_at < ?",
            (before(AUTOREPLY_KEEP_DAYS),),
        ) as cur:
            deleted += int(cur.rowcount or 0)
        await self._db.commit()
        return {"expired": expired, "cleared": cleared, "deleted": deleted}

    async def list_shares_for_resource(
        self, resource_type: str, resource_id: str, owner_id: str,
    ) -> list[dict]:
        """Grants ON a resource I own — for rendering 'shared with X, Y'."""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT s.*, u.email AS grantee_email, u.display_name AS grantee_name"
            " FROM resource_shares s JOIN users u ON u.id = s.grantee_id"
            " WHERE s.resource_type = ? AND s.resource_id = ? AND s.owner_id = ?"
            " ORDER BY s.created_at",
            (resource_type, resource_id, owner_id),
        )
        return [dict(r) for r in rows]

    async def list_shares_for_owner(self, owner_id: str, resource_type: str) -> list[dict]:
        """Every grant ``owner_id`` made on resources of one type."""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT resource_id, grantee_id, created_at FROM resource_shares"
            " WHERE owner_id = ? AND resource_type = ? ORDER BY created_at",
            (owner_id, resource_type),
        )
        return [dict(r) for r in rows]

    async def list_shares_for_grantee(
        self, grantee_id: str, resource_type: str | None = None,
    ) -> list[dict]:
        """Resources shared TO me, with owner info — for badging shared lists."""
        assert self._db is not None
        q = ("SELECT s.*, u.email AS owner_email, u.display_name AS owner_name"
             " FROM resource_shares s JOIN users u ON u.id = s.owner_id"
             " WHERE s.grantee_id = ?")
        params: list = [grantee_id]
        if resource_type:
            q += " AND s.resource_type = ?"
            params.append(resource_type)
        q += " ORDER BY s.created_at"
        rows = await self._db.execute_fetchall(q, tuple(params))
        return [dict(r) for r in rows]

    async def list_shared_archetypes(self, grantee_id: str) -> list[dict]:
        """User-archetype rows shared TO `grantee_id`, tagged with owner info."""
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT a.*, s.owner_id AS shared_owner, s.permission AS shared_permission,"
            " u.email AS shared_owner_email, u.display_name AS shared_owner_name"
            " FROM resource_shares s"
            " JOIN user_archetypes a"
            "   ON a.user_id = s.owner_id AND a.archetype_id = s.resource_id"
            " JOIN users u ON u.id = s.owner_id"
            " WHERE s.resource_type = 'archetype' AND s.grantee_id = ?"
            " ORDER BY a.updated_at DESC",
            (grantee_id,),
        )
        return [dict(r) for r in rows]
