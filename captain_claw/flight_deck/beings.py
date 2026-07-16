"""Iskra beings store — registry, wallets, conservation ledger, lifecycle.

The life layer's persistence (docs/living-beings-plan.md §10). Sync sqlite3 +
a lock in its own ``beings.db``, mirroring AutonomyStore/ConsciousnessStore.

Conservation by construction: every wallet balance mutation happens inside
exactly one method (:meth:`BeingsStore._apply`) that writes the matching
``token_transfers`` row in the same transaction. Mint rows have no
``from_being``; sink rows (usage spend, metamorphosis burns) have no
``to_being``; so at any instant::

    sum(balances) == sum(mints) - sum(sinks)

— checked by :meth:`conservation`, asserted in tests. The Constitution module
supplies the physics (stage gates, tier weights, preset clamps); this store
enforces them at the only door tokens can pass through.

Clock: methods take an optional ``now`` so the beings loop, tests and backfills
can be deterministic; default is real UTC.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from captain_claw.flight_deck import being_constitution as constitution
from captain_claw.flight_deck import being_genome as genome_mod
from captain_claw.logging import get_logger

log = get_logger(__name__)

STATES = ("alive", "paused", "torpor", "dead", "emigrated")
TRANSFER_REASONS = (
    "allowance", "usage", "fee", "gift", "trade",
    "procreation", "metamorphosis_burn", "self_mod_burn", "adjust", "grant",
    "exchange",
)

# Coin ledger reasons (space plan Phase 2): grant = pocket money, wage =
# judged work paid in coins, sale/purchase = being↔being circulation
# (Phase 3), exchange = the one-way conversion into tokens, stipend = the
# steward's pay, commission = escrow into (or refund from) a building
# fund (both Phase 5).
COIN_REASONS = ("grant", "wage", "sale", "purchase", "exchange", "stipend",
                "commission")

# Fixed parent top-up amounts (tokens) the UI offers to recharge a wallet.
GRANT_AMOUNTS = (2_000_000, 5_000_000, 10_000_000, 20_000_000)

# The public square (plan §9): a being the parent flags ``public`` gets an
# un-gated page where strangers may leave it short notes. A note is a
# suggestion/topic — a seed, never an order — capped this small so it stays a
# provocation, not an instruction the model could mistake for guidance.
PUBLIC_MSG_MAX_CHARS = 64
PUBLIC_NAME_MAX_CHARS = 40
# How a newly-conceived being thinks a tick (docs/being-faculties-plan.md).
# Read at call time so tests can pin the legacy 'monolith' path via conftest.
DEFAULT_COGNITION = "faculties"


class BeingError(Exception):
    """Base for being-domain failures; ``status`` maps to HTTP in routes."""

    status = 400

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        if status is not None:
            self.status = status


class BeingNotFound(BeingError):
    status = 404


class InsufficientTokens(BeingError):
    status = 402


class BurnCapExceeded(BeingError):
    status = 429


def _db_path() -> Path:
    base = os.environ.get("FD_DATA_DIR", "").strip()
    if base:
        return Path(base).expanduser().resolve() / "beings.db"
    return Path("~/.captain-claw/beings.db").expanduser()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return s or "iskra"


class BeingsStore:
    """SQLite-backed being registry + wallet. Sync sqlite3 + a lock."""

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
                CREATE TABLE IF NOT EXISTS beings (
                    id                TEXT PRIMARY KEY,
                    owner_id          TEXT NOT NULL,
                    slug              TEXT NOT NULL UNIQUE,
                    name              TEXT NOT NULL,
                    stage             TEXT NOT NULL DEFAULT 'egg',
                    state             TEXT NOT NULL DEFAULT 'alive',
                    genome            TEXT NOT NULL,
                    drives            TEXT NOT NULL DEFAULT '{}',
                    attention_credits INTEGER NOT NULL DEFAULT 3,
                    next_wake_at      TEXT,
                    born_at           TEXT NOT NULL,
                    hatched_at        TEXT,
                    died_at           TEXT,
                    lineage           TEXT NOT NULL DEFAULT '[]',
                    created_at        TEXT NOT NULL,
                    updated_at        TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_beings_owner ON beings(owner_id);

                CREATE TABLE IF NOT EXISTS being_wallets (
                    being_id         TEXT PRIMARY KEY REFERENCES beings(id),
                    balance_tokens   INTEGER NOT NULL DEFAULT 0,
                    allowance_preset TEXT NOT NULL DEFAULT '2M',
                    period           TEXT NOT NULL DEFAULT 'day',
                    daily_burn_cap   INTEGER,
                    savings_ceiling  INTEGER,
                    reserve_tokens   INTEGER NOT NULL DEFAULT 0,
                    updated_at       TEXT NOT NULL
                );

                -- The conservation ledger. from_being NULL = mint (parent),
                -- to_being NULL = sink (usage spend / metamorphosis burn).
                CREATE TABLE IF NOT EXISTS token_transfers (
                    id         TEXT PRIMARY KEY,
                    owner_id   TEXT NOT NULL,
                    from_being TEXT,
                    to_being   TEXT,
                    tokens     INTEGER NOT NULL,
                    reason     TEXT NOT NULL,
                    job_id     TEXT,
                    note       TEXT,
                    at         TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_transfers_from
                    ON token_transfers(from_being, at);
                CREATE INDEX IF NOT EXISTS idx_transfers_to
                    ON token_transfers(to_being, at);
                CREATE INDEX IF NOT EXISTS idx_transfers_owner
                    ON token_transfers(owner_id, at);

                CREATE TABLE IF NOT EXISTS being_events (
                    id       TEXT PRIMARY KEY,
                    being_id TEXT NOT NULL,
                    kind     TEXT NOT NULL,
                    data     TEXT NOT NULL DEFAULT '{}',
                    at       TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_being_events
                    ON being_events(being_id, at);

                -- Chores: parent-posted fixed-fee tasks (plan §5.1, Phase 2).
                -- The fee is minted only at judged payout, so conservation
                -- holds; escrow_state tracks the promise lifecycle.
                CREATE TABLE IF NOT EXISTS being_jobs (
                    id           TEXT PRIMARY KEY,
                    owner_id     TEXT NOT NULL,
                    being_id     TEXT NOT NULL,
                    spec         TEXT NOT NULL,
                    fee_tokens   INTEGER NOT NULL,
                    escrow_state TEXT NOT NULL DEFAULT 'open',
                    result_text  TEXT NOT NULL DEFAULT '',
                    judge_note   TEXT NOT NULL DEFAULT '',
                    created_at   TEXT NOT NULL,
                    done_at      TEXT,
                    paid_at      TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_being_jobs
                    ON being_jobs(being_id, escrow_state);

                -- Letters between siblings (plan §7): asynchronous, logged,
                -- rate-limited; delivered as percepts on the recipient's tick.
                CREATE TABLE IF NOT EXISTS being_letters (
                    id         TEXT PRIMARY KEY,
                    owner_id   TEXT NOT NULL,
                    from_being TEXT NOT NULL,
                    to_being   TEXT NOT NULL,
                    body       TEXT NOT NULL,
                    at         TEXT NOT NULL,
                    read_at    TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_being_letters_to
                    ON being_letters(to_being, read_at);
                CREATE INDEX IF NOT EXISTS idx_being_letters_from
                    ON being_letters(from_being, at);

                -- Skills published to the commons (plan §7 culture + §5.1
                -- market): price 0 = free adoption, >0 = a trade settles on
                -- the conservation ledger when a sibling adopts.
                CREATE TABLE IF NOT EXISTS being_publications (
                    id           TEXT PRIMARY KEY,
                    owner_id     TEXT NOT NULL,
                    being_id     TEXT NOT NULL,
                    title        TEXT NOT NULL,
                    note         TEXT NOT NULL DEFAULT '',
                    commons_path TEXT NOT NULL,
                    price_tokens INTEGER NOT NULL DEFAULT 0,
                    at           TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_being_pubs
                    ON being_publications(owner_id, at);

                -- Quest board (plan §5.1): OPEN bounties, not targeted at one
                -- being — any eligible being may claim, deliver, be judged.
                -- origin traces provenance (parent | autonomy). The fee mints
                -- only at judged completion, so conservation holds.
                CREATE TABLE IF NOT EXISTS being_quests (
                    id          TEXT PRIMARY KEY,
                    owner_id    TEXT NOT NULL,
                    title       TEXT NOT NULL,
                    spec        TEXT NOT NULL,
                    fee_tokens  INTEGER NOT NULL,
                    origin      TEXT NOT NULL DEFAULT 'parent',
                    state       TEXT NOT NULL DEFAULT 'open',
                    claimed_by  TEXT,
                    result_text TEXT NOT NULL DEFAULT '',
                    judge_note  TEXT NOT NULL DEFAULT '',
                    created_at  TEXT NOT NULL,
                    claimed_at  TEXT,
                    done_at     TEXT,
                    paid_at     TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_being_quests
                    ON being_quests(owner_id, state);

                -- Ventures (plan §5.1): the being-INITIATED recurring service —
                -- the sanctioned outlet for earning creativity. Proposed by a
                -- being, priced+approved by the parent, then delivered every
                -- cadence for recurring pay. pending_result set = a delivery
                -- awaits the parent's acceptance this cycle.
                CREATE TABLE IF NOT EXISTS being_ventures (
                    id             TEXT PRIMARY KEY,
                    owner_id       TEXT NOT NULL,
                    being_id       TEXT NOT NULL,
                    title          TEXT NOT NULL,
                    description    TEXT NOT NULL DEFAULT '',
                    price_tokens   INTEGER NOT NULL,
                    cadence_days   INTEGER NOT NULL,
                    state          TEXT NOT NULL DEFAULT 'proposed',
                    pending_result TEXT NOT NULL DEFAULT '',
                    deliveries     INTEGER NOT NULL DEFAULT 0,
                    created_at     TEXT NOT NULL,
                    approved_at    TEXT,
                    next_due_at    TEXT,
                    last_paid_at   TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_being_ventures
                    ON being_ventures(owner_id, state);

                -- The parent writing back (the reply channel): free-form
                -- messages from the human parent, delivered ONCE as a
                -- high-priority percept on the being's next tick. Reactive —
                -- the being reads for free; only its own outbound message
                -- spends attention. Mirrors being_letters' deliver-once shape.
                CREATE TABLE IF NOT EXISTS being_parent_messages (
                    id        TEXT PRIMARY KEY,
                    owner_id  TEXT NOT NULL,
                    being_id  TEXT NOT NULL,
                    body      TEXT NOT NULL,
                    at        TEXT NOT NULL,
                    read_at   TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_being_parent_msgs
                    ON being_parent_messages(being_id, read_at);

                -- The Mind (plan §2.3.1): explicit, being-DECLARED edges over
                -- its own artifacts — the deliberate structure that neither
                -- the journal (temporal) nor embeddings (associative) capture.
                -- Endpoints are verified to exist before an edge is stored, so
                -- the graph is a real portrait, never narration. UNIQUE keeps
                -- a re-declared edge idempotent.
                CREATE TABLE IF NOT EXISTS being_links (
                    id        TEXT PRIMARY KEY,
                    owner_id  TEXT NOT NULL,
                    being_id  TEXT NOT NULL,
                    from_path TEXT NOT NULL,
                    to_path   TEXT NOT NULL,
                    rel       TEXT NOT NULL,
                    why       TEXT NOT NULL DEFAULT '',
                    at        TEXT NOT NULL,
                    UNIQUE(being_id, from_path, to_path, rel)
                );
                CREATE INDEX IF NOT EXISTS idx_being_links
                    ON being_links(being_id);

                -- Saved 3rd-party developmental assessments (second opinions).
                -- SEALED records: they live only here — outside the being's
                -- home — until adulthood, when they're released into it
                -- (released_at marks the unsealing).
                CREATE TABLE IF NOT EXISTS being_assessments (
                    id          TEXT PRIMARY KEY,
                    owner_id    TEXT NOT NULL,
                    being_id    TEXT NOT NULL,
                    assessor    TEXT NOT NULL,
                    stage       TEXT NOT NULL DEFAULT '',
                    score       INTEGER,
                    verdict     TEXT NOT NULL DEFAULT '',
                    content     TEXT NOT NULL,
                    at          TEXT NOT NULL,
                    released_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_being_assessments
                    ON being_assessments(being_id, at);

                -- The public square (plan §9). A ``public`` being gets an
                -- un-gated page; strangers leave short notes here. A thread is
                -- one ongoing exchange with one visitor (identified only by the
                -- thread id their browser keeps — no accounts, no tracking).
                CREATE TABLE IF NOT EXISTS being_public_threads (
                    id          TEXT PRIMARY KEY,
                    being_id    TEXT NOT NULL,
                    sender_name TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_being_public_threads
                    ON being_public_threads(being_id, updated_at);

                -- Messages within a public thread. role='public' is a visitor's
                -- note; role='being' is the being's optional reply. read_at =
                -- a tick surfaced it (considered); answered_at = the being
                -- actually replied. The being is NEVER obliged to answer — these
                -- are provocations it may weigh, not parenting it must obey.
                CREATE TABLE IF NOT EXISTS being_public_messages (
                    id          TEXT PRIMARY KEY,
                    thread_id   TEXT NOT NULL,
                    being_id    TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    sender_name TEXT NOT NULL DEFAULT '',
                    body        TEXT NOT NULL,
                    at          TEXT NOT NULL,
                    read_at     TEXT,
                    answered_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_being_public_messages
                    ON being_public_messages(being_id, role, read_at);
                CREATE INDEX IF NOT EXISTS idx_being_public_messages_thread
                    ON being_public_messages(thread_id, at);

                -- The village's own words: a per-owner description the parent
                -- writes on the Beings page, shown atop their public /village.
                -- Also the federation settings (plan §9.1): a secret others must
                -- present to send a visiting being here, and this machine's own
                -- public URL (so beings it sends elsewhere can be fetched back).
                CREATE TABLE IF NOT EXISTS village_meta (
                    owner_id      TEXT PRIMARY KEY,
                    name          TEXT NOT NULL DEFAULT '',
                    description   TEXT NOT NULL DEFAULT '',
                    secret        TEXT NOT NULL DEFAULT '',
                    secret_public INTEGER NOT NULL DEFAULT 0,
                    public_url    TEXT NOT NULL DEFAULT '',
                    updated_at    TEXT NOT NULL
                );

                -- Visitors (plan §9.1): beings that live on ANOTHER machine and
                -- were sent to visit this village. They are NEVER copied — only a
                -- cached profile snapshot + where to fetch their data live
                -- (origin + slug) is stored; browsing proxies to the origin. A
                -- valid village secret is required to register/refresh one.
                CREATE TABLE IF NOT EXISTS being_visitors (
                    id         TEXT PRIMARY KEY,
                    owner_id   TEXT NOT NULL,
                    origin     TEXT NOT NULL,
                    slug       TEXT NOT NULL,
                    name       TEXT NOT NULL DEFAULT '',
                    profile    TEXT NOT NULL DEFAULT '{}',
                    first_seen TEXT NOT NULL,
                    last_seen  TEXT NOT NULL,
                    UNIQUE(owner_id, origin, slug)
                );
                CREATE INDEX IF NOT EXISTS idx_being_visitors
                    ON being_visitors(owner_id, last_seen);

                -- The ground (space plan Phase 1): the village's civic places,
                -- designed once by the architect (LLM with a deterministic
                -- fallback) on a PLOT_SIZE² plot. Each being's own home is NOT
                -- a row — it is computed from its slug (being_world.home_xy).
                -- Affordances come from a fixed vocabulary; the map is physics.
                CREATE TABLE IF NOT EXISTS village_places (
                    owner_id    TEXT NOT NULL,
                    id          TEXT NOT NULL,
                    name        TEXT NOT NULL,
                    x           INTEGER NOT NULL,
                    y           INTEGER NOT NULL,
                    affordances TEXT NOT NULL DEFAULT '[]',
                    description TEXT NOT NULL DEFAULT '',
                    created_at  TEXT NOT NULL,
                    PRIMARY KEY (owner_id, id)
                );

                -- The coin ledger (space plan Phase 2): coins are MONEY
                -- (the social economy), tokens are METABOLISM (thinking).
                -- Balance = SUM(delta); only physics functions write rows —
                -- no LLM path can move a coin. The exchange is one-way
                -- (coins → tokens), so wealth is earned, never printed.
                CREATE TABLE IF NOT EXISTS being_coin_events (
                    id         TEXT PRIMARY KEY,
                    owner_id   TEXT NOT NULL,
                    being_id   TEXT NOT NULL,
                    delta      INTEGER NOT NULL,
                    reason     TEXT NOT NULL,
                    from_being TEXT,
                    data       TEXT NOT NULL DEFAULT '{}',
                    at         TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_being_coin_events
                    ON being_coin_events(being_id, at);

                -- Contacts (space plan Phase 3): who has TRULY met whom —
                -- built only from co-presence on the ground, symmetric
                -- (a_id < b_id), strengthened asymptotically per meeting.
                CREATE TABLE IF NOT EXISTS being_contacts (
                    owner_id    TEXT NOT NULL,
                    a_id        TEXT NOT NULL,
                    b_id        TEXT NOT NULL,
                    met_count   INTEGER NOT NULL DEFAULT 0,
                    strength    REAL NOT NULL DEFAULT 0,
                    last_met_at TEXT,
                    PRIMARY KEY (owner_id, a_id, b_id)
                );

                -- The market (space plan Phase 3): a listing is a REAL file
                -- offered for coins. Buying copies it into the buyer's home
                -- (society owns the file physics) and settles the coin pair
                -- being→being — circulation, never minting.
                CREATE TABLE IF NOT EXISTS village_listings (
                    id          TEXT PRIMARY KEY,
                    owner_id    TEXT NOT NULL,
                    seller_id   TEXT NOT NULL,
                    path        TEXT NOT NULL,
                    title       TEXT NOT NULL,
                    price_coins INTEGER NOT NULL,
                    state       TEXT NOT NULL DEFAULT 'open',
                    created_at  TEXT NOT NULL,
                    sold_to     TEXT,
                    sold_at     TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_village_listings
                    ON village_listings(owner_id, state);

                -- Commissioned buildings (space plan Phase 5): ONE active
                -- fund per village at a time. Contributions escrow coins
                -- OUT of pockets (reason='commission'); approval burns them
                -- and raises the place; rejection refunds exactly. The
                -- contributor list is the coin ledger itself — no state.
                CREATE TABLE IF NOT EXISTS village_commissions (
                    id           TEXT PRIMARY KEY,
                    owner_id     TEXT NOT NULL,
                    name         TEXT NOT NULL,
                    why          TEXT NOT NULL DEFAULT '',
                    affordance   TEXT NOT NULL,
                    target_coins INTEGER NOT NULL,
                    raised_coins INTEGER NOT NULL DEFAULT 0,
                    state        TEXT NOT NULL DEFAULT 'open',
                    created_by   TEXT NOT NULL,
                    created_at   TEXT NOT NULL,
                    decided_at   TEXT,
                    note         TEXT NOT NULL DEFAULT '',
                    place_id     TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_village_commissions
                    ON village_commissions(owner_id, state);
                """
            )
            # Work can pay in coins (space plan Phase 2): the parent picks
            # the denomination at posting; judgment mints that currency.
            for work_table in ("being_jobs", "being_quests"):
                try:
                    self._c().execute(
                        f"ALTER TABLE {work_table} ADD COLUMN fee_coins"
                        " INTEGER NOT NULL DEFAULT 0")
                except sqlite3.OperationalError:
                    pass
            # village_meta gained federation columns + a name after first ship.
            for col, ddl in [
                ("secret", "TEXT NOT NULL DEFAULT ''"),
                ("secret_public", "INTEGER NOT NULL DEFAULT 0"),
                ("public_url", "TEXT NOT NULL DEFAULT ''"),
                ("name", "TEXT NOT NULL DEFAULT ''"),
                # The steward's weekly pay in coins (space plan Phase 5) —
                # a parent knob, default off; paid once per ISO week.
                ("steward_stipend_coins", "INTEGER NOT NULL DEFAULT 0"),
            ]:
                try:
                    self._c().execute(
                        f"ALTER TABLE village_meta ADD COLUMN {col} {ddl}")
                except sqlite3.OperationalError:
                    pass
            # Lightweight migrations (columns added after first ship).
            for col, ddl in [
                ("birth_letter", "TEXT NOT NULL DEFAULT ''"),
                ("media_diet", "TEXT NOT NULL DEFAULT '{}'"),
                ("agent_slug", "TEXT"),
                ("agent_port", "INTEGER"),
                ("agent_token", "TEXT"),
                ("last_tick_at", "TEXT"),
                ("tick_count", "INTEGER NOT NULL DEFAULT 0"),
                ("house_rules", "TEXT NOT NULL DEFAULT '[]'"),
                ("rules_pending", "INTEGER NOT NULL DEFAULT 0"),
                ("affect", "TEXT NOT NULL DEFAULT '{}'"),
                ("persona", "TEXT NOT NULL DEFAULT ''"),
                ("pending_self_mod", "TEXT NOT NULL DEFAULT ''"),
                ("pending_procreation", "TEXT NOT NULL DEFAULT ''"),
                ("torpor_since", "TEXT"),
                ("tick_interval_minutes", "INTEGER"),
                ("public", "INTEGER NOT NULL DEFAULT 0"),
                # Per-being model/connection override (provider/model/base_url/
                # api_key/output_ctx). Empty for locally-conceived beings (they
                # use the owner's tier config); set on IMPORT so a being carries
                # its own connection across machines. See being_life.spawn_body.
                ("body_config", "TEXT NOT NULL DEFAULT ''"),
                # Federation (plan §9.1): a being sent to visit another village
                # — the target village's URL + its secret, and when we last
                # announced this being there.
                ("visit_url", "TEXT NOT NULL DEFAULT ''"),
                ("visit_secret", "TEXT NOT NULL DEFAULT ''"),
                ("visit_last_announce", "TEXT"),
                # How the being THINKS a tick: 'faculties' (decomposed pipeline —
                # orient/act/journal/connect, small context per call, the
                # default) or 'monolith' (one prompt → one digest, legacy).
                # See docs/being-faculties-plan.md.
                ("cognition", "TEXT NOT NULL DEFAULT 'faculties'"),
                # Optional archetype the body runs on (its tier → model/provider,
                # tools, cognitive_mode). Empty → the stage tier + owner config.
                # Changing it respawns the body. See being_life.spawn_body.
                ("body_archetype", "TEXT NOT NULL DEFAULT ''"),
                # When the parent last opened this being's thread — messages it
                # spoke after this are "unread from the being" (a badge cue).
                ("parent_read_at", "TEXT"),
                # Compact mode (panel toggle): tick prompts use the compact
                # instruction set (instructions/beings/compact_*.md), and the
                # body runs lean (eco/micro system prompt + capped context).
                # Same narrative and physics, fewer tokens per heartbeat.
                ("compact_mode", "INTEGER NOT NULL DEFAULT 0"),
                # The naming rite (roadmap T2.10): an adolescent may propose
                # ONE chosen display name; it waits here for the parent's
                # blessing. JSON {name, why, proposed_at}; slug never changes.
                ("pending_name", "TEXT NOT NULL DEFAULT ''"),
                # Elderhood (roadmap T3.14, opt-in): days-alive after which
                # the being enters its elder season. NULL = no natural span.
                ("elder_after_days", "INTEGER"),
                # The village radio (roadmap T3.16): an adult public being's
                # one daily broadcast line. JSON {text, at}.
                ("broadcast", "TEXT NOT NULL DEFAULT ''"),
                # Education (roadmap T2.12): the parent-assigned curriculum.
                # JSON list of {id, ref, note, fee_tokens, assigned_at,
                # done_at, report_path} — reports are FD-verified real files.
                ("reading_list", "TEXT NOT NULL DEFAULT '[]'"),
                # The ground (space plan Phase 1): where the body is. JSON
                # {"at": "<place>"} at rest, or {"to", "from", "origin",
                # "departed_at"} on the road — position at any instant is a
                # pure function of this row and the clock (no scheduler).
                ("location", "TEXT NOT NULL DEFAULT ''"),
            ]:
                try:
                    self._c().execute(f"ALTER TABLE beings ADD COLUMN {col} {ddl}")
                except sqlite3.OperationalError:
                    pass
            # One-time flip: the decomposed tick is now the default, so beings
            # that were auto-defaulted to 'monolith' by the first cut move to
            # 'faculties'. Guarded by user_version so it runs exactly once — a
            # parent who LATER chooses 'monolith' is never re-flipped.
            if self._c().execute("PRAGMA user_version").fetchone()[0] < 1:
                self._c().execute("UPDATE beings SET cognition = 'faculties'"
                                  " WHERE cognition = 'monolith'")
                self._c().execute("PRAGMA user_version = 1")
            self._c().commit()

    # ── Registry ─────────────────────────────────────────────────────

    def conceive(
        self,
        owner_id: str,
        name: str,
        *,
        attributes: dict | None = None,
        preset: str | None = None,
        roll_seed: int | None = None,
        voice_seed: str = "",
        interest_seeds: list[str] | None = None,
        allowance_preset: str = "2M",
        birth_letter: str = "",
        now: datetime | None = None,
    ) -> dict:
        """Point-buy conception (Generation 1). Exactly one of attributes /
        preset / roll_seed selects the sheet; born as an egg with an empty,
        allowance-armed wallet."""
        now = now or _utcnow()
        if attributes is not None:
            errors = genome_mod.validate_point_buy(attributes)
            if errors:
                raise BeingError("; ".join(errors))
            sheet = {a: int(attributes[a]) for a in genome_mod.ATTRS}
        elif preset is not None:
            if preset not in genome_mod.PRESETS:
                raise BeingError(f"unknown preset {preset!r}")
            sheet = dict(genome_mod.PRESETS[preset])
        else:
            import random
            sheet = genome_mod.roll(random.Random(roll_seed))
        if allowance_preset not in constitution.ALLOWANCE_PRESETS:
            raise BeingError(f"unknown allowance preset {allowance_preset!r}")
        g = genome_mod.new_genome(
            sheet, voice_seed=voice_seed, interest_seeds=interest_seeds,
        )
        bid = uuid.uuid4().hex
        slug = f"iskra-{_slugify(name)}-{bid[:4]}"
        with self._lock:
            c = self._c()
            c.execute(
                "INSERT INTO beings (id, owner_id, slug, name, stage, state, genome,"
                " born_at, created_at, updated_at, birth_letter, cognition)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (bid, owner_id, slug, name, "egg", "alive", json.dumps(g),
                 _iso(now), _iso(now), _iso(now), birth_letter,
                 DEFAULT_COGNITION),
            )
            c.execute(
                "INSERT INTO being_wallets (being_id, allowance_preset, updated_at)"
                " VALUES (?,?,?)",
                (bid, allowance_preset, _iso(now)),
            )
            c.commit()
        self.record_event(bid, "conceived", {"sheet": sheet, "name": name}, now=now)
        return self.get(owner_id, slug)

    def set_pending_procreation(self, being_id: str, pending: dict | None,
                                now: datetime | None = None) -> None:
        self._update(being_id, now or _utcnow(),
                     pending_procreation=json.dumps(pending) if pending else "")

    def conceive_offspring(
        self,
        owner_id: str,
        name: str,
        parent_slug: str,
        partner_slug: str | None = None,
        *,
        letter: str = "",
        allowance_preset: str = "2M",
        seed: int | None = None,
        now: datetime | None = None,
    ) -> dict:
        """Generation N+1 (plan §8): crossover (two parents) or budding (one),
        never point-buy. The dowry (PROCREATION_COST_TOKENS) moves from the
        parents' savings to the child on the conservation ledger — earned
        wealth, split between co-parents, each of whom must afford their half.
        Consent is the human parent's authenticated call into this method."""
        import random
        now = now or _utcnow()
        parent = self.get(owner_id, parent_slug)
        if not constitution.has_capability(parent["stage"], "procreate"):
            raise BeingError(f"a {parent['stage']} cannot have children yet")
        if parent["state"] == "dead":
            raise BeingError("the dead bear no children")
        partner = None
        if partner_slug:
            partner = self.get(owner_id, partner_slug)
            if partner["id"] == parent["id"]:
                raise BeingError("a partner must be another being")
            if not constitution.has_capability(partner["stage"], "procreate"):
                raise BeingError(f"the partner ({partner['stage']}) is too young")
            if partner["state"] == "dead":
                raise BeingError("the dead bear no children")
        funders = [parent] + ([partner] if partner else [])
        cost = constitution.PROCREATION_COST_TOKENS
        share = cost // len(funders)
        for f in funders:
            view = self.wallet_view(f)
            if view["enforced"] and view["balance_tokens"] < share:
                raise InsufficientTokens(
                    f"{f['name']} cannot afford the dowry share ({share})")
        rng = random.Random(seed)
        pa = genome_mod.effective_attributes(parent["genome"])
        if partner:
            sheet = genome_mod.crossover(
                pa, genome_mod.effective_attributes(partner["genome"]), rng)
        else:
            sheet = genome_mod.budding(pa, rng)
        seeds_pool = list(dict.fromkeys(
            (parent["genome"].get("interest_seeds") or [])
            + ((partner["genome"].get("interest_seeds") or []) if partner else [])))
        rng.shuffle(seeds_pool)
        voice_pool = [g for g in (
            parent["genome"].get("voice_seed"),
            partner["genome"].get("voice_seed") if partner else None) if g]
        lineage = [parent["slug"]] + ([partner["slug"]] if partner else [])
        lineage += (parent["genome"].get("lineage") or [])[:4]
        gen = 1 + max(parent["genome"].get("generation", 1),
                      partner["genome"].get("generation", 1) if partner else 0)
        g = genome_mod.new_genome(
            sheet,
            voice_seed=rng.choice(voice_pool) if voice_pool else "",
            interest_seeds=seeds_pool[:3],
            generation=gen,
            lineage=lineage[:6],
            inherited_skills=[],
        )
        bid = uuid.uuid4().hex
        slug = f"iskra-{_slugify(name)}-{bid[:4]}"
        with self._lock:
            c = self._c()
            c.execute(
                "INSERT INTO beings (id, owner_id, slug, name, stage, state, genome,"
                " born_at, created_at, updated_at, birth_letter, cognition)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (bid, owner_id, slug, name, "egg", "alive", json.dumps(g),
                 _iso(now), _iso(now), _iso(now), letter, DEFAULT_COGNITION),
            )
            c.execute(
                "INSERT INTO being_wallets (being_id, allowance_preset, updated_at)"
                " VALUES (?,?,?)",
                (bid, allowance_preset, _iso(now)),
            )
            c.commit()
        for f in funders:
            self._apply(owner_id, tokens=share, reason="procreation",
                        from_being=f["id"], to_being=bid,
                        note=f"dowry for {name}", now=now)
            self.record_event(f["id"], "had_child",
                              {"child": slug, "name": name,
                               "dowry_share": share,
                               "with": (partner["slug"] if partner
                                        and f["id"] == parent["id"]
                                        else parent["slug"] if partner
                                        else None)}, now=now)
            self.milestone(f["id"], "first_child", {"child": name}, now=now)
        self.record_event(bid, "conceived", {
            "sheet": sheet, "name": name, "generation": gen,
            "of": [f["slug"] for f in funders], "dowry_tokens": cost,
        }, now=now)
        return self.get(owner_id, slug)

    def import_being_row(self, owner_id: str, manifest: dict,
                         now: datetime | None = None) -> dict:
        """Recreate a being (DB side) from an export manifest, under a new owner.
        A fresh id + a free slug; the wallet balance is re-minted as one 'adjust'
        row so conservation still holds on the target; events (incl. milestones)
        are replayed. Home files + the body are handled by the caller."""
        now = now or _utcnow()
        genome = manifest.get("genome")
        if not isinstance(genome, dict) or "attributes" not in genome:
            raise BeingError("not a valid being export (missing genome)")
        name = str(manifest.get("name") or "Imported").strip()[:80] or "Imported"
        bid = uuid.uuid4().hex
        # Prefer the original slug; fall back to a unique one if it's taken here.
        slug = str(manifest.get("slug") or "").strip() or f"iskra-{_slugify(name)}"
        if self._c().execute("SELECT 1 FROM beings WHERE slug = ?",
                             (slug,)).fetchone():
            slug = f"{slug}-{bid[:4]}"
        stage = manifest.get("stage") or "infant"
        if stage not in constitution.STAGE_ORDER or stage == "egg":
            stage = "infant"
        state = manifest.get("state") or "alive"
        if state not in STATES:
            state = "alive"
        wallet = manifest.get("wallet") or {}
        model = manifest.get("model") or {}
        lineage = genome.get("lineage") or []
        with self._lock:
            c = self._c()
            c.execute(
                "INSERT INTO beings (id, owner_id, slug, name, stage, state,"
                " genome, drives, attention_credits, born_at, hatched_at,"
                " lineage, created_at, updated_at, birth_letter, media_diet,"
                " house_rules, affect, persona, tick_interval_minutes, public,"
                " body_config, tick_count, last_tick_at, cognition,"
                " body_archetype)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (bid, owner_id, slug, name, stage, state,
                 json.dumps(genome),
                 json.dumps(manifest.get("drives") or {}),
                 int(manifest.get("attention_credits") or 3),
                 manifest.get("born_at") or _iso(now),
                 manifest.get("hatched_at") or _iso(now),
                 json.dumps(lineage), _iso(now), _iso(now),
                 str(manifest.get("birth_letter") or ""),
                 json.dumps(manifest.get("media_diet") or {}),
                 json.dumps(manifest.get("house_rules") or []),
                 json.dumps(manifest.get("affect") or {}),
                 str(manifest.get("persona") or ""),
                 manifest.get("tick_interval_minutes"),
                 1 if manifest.get("public") else 0,
                 json.dumps(model) if model else "",
                 int(manifest.get("tick_count") or 0),
                 manifest.get("last_tick_at"),
                 manifest.get("cognition") or DEFAULT_COGNITION,
                 str(manifest.get("body_archetype") or "")),
            )
            c.execute(
                "INSERT INTO being_wallets (being_id, allowance_preset,"
                " daily_burn_cap, savings_ceiling, reserve_tokens, updated_at)"
                " VALUES (?,?,?,?,?,?)",
                (bid, wallet.get("allowance_preset") or "2M",
                 wallet.get("daily_burn_cap"), wallet.get("savings_ceiling"),
                 int(wallet.get("reserve_tokens") or 0), _iso(now)),
            )
            c.commit()
        balance = int(wallet.get("balance_tokens") or 0)
        if balance > 0:
            self._apply(owner_id, tokens=balance, reason="adjust",
                        from_being=None, to_being=bid, note="import", now=now)
        # Replay events (timeline + once-per-life milestones survive the move).
        events = manifest.get("events") or []
        with self._lock:
            c = self._c()
            for e in events[-1000:]:
                if not isinstance(e, dict) or not e.get("kind"):
                    continue
                c.execute(
                    "INSERT INTO being_events (id, being_id, kind, data, at)"
                    " VALUES (?,?,?,?,?)",
                    (uuid.uuid4().hex, bid, str(e["kind"]),
                     json.dumps(e.get("data") or {}),
                     e.get("at") or _iso(now)),
                )
            c.commit()
        # The Mind: replay declared edges (verified valid once the home lands).
        for lk in (manifest.get("links") or []):
            if isinstance(lk, dict) and lk.get("from_path") and lk.get("to_path"):
                self.add_link(owner_id, bid, str(lk["from_path"]),
                              str(lk["to_path"]), str(lk.get("rel") or ""),
                              str(lk.get("why") or ""), now=now)
        # Sealed second opinions (childhood records), released_at preserved.
        with self._lock:
            c = self._c()
            for a in (manifest.get("assessments") or []):
                if not isinstance(a, dict) or not a.get("content"):
                    continue
                c.execute(
                    "INSERT INTO being_assessments (id, owner_id, being_id,"
                    " assessor, stage, score, verdict, content, at, released_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (uuid.uuid4().hex, owner_id, bid,
                     str(a.get("assessor") or "")[:120],
                     str(a.get("stage") or ""), a.get("score"),
                     str(a.get("verdict") or "")[:40], str(a["content"]),
                     a.get("at") or _iso(now), a.get("released_at")),
                )
            c.commit()
        self.record_event(bid, "imported",
                          {"from_slug": manifest.get("slug"),
                           "balance_tokens": balance,
                           "links": len(manifest.get("links") or [])}, now=now)
        return self.get(owner_id, slug)

    def _row(self, owner_id: str, slug: str) -> sqlite3.Row:
        row = self._c().execute(
            "SELECT * FROM beings WHERE owner_id = ? AND slug = ?",
            (owner_id, slug),
        ).fetchone()
        if not row:
            raise BeingNotFound(f"no being {slug!r}")
        return row

    def get(self, owner_id: str, slug: str) -> dict:
        b = dict(self._row(owner_id, slug))
        b["genome"] = json.loads(b["genome"])
        b["drives"] = json.loads(b["drives"])
        b["lineage"] = json.loads(b["lineage"])
        b["media_diet"] = json.loads(b.get("media_diet") or "{}")
        b["house_rules"] = json.loads(b.get("house_rules") or "[]")
        b["affect"] = json.loads(b.get("affect") or "{}")
        try:
            b["reading_list"] = json.loads(b.get("reading_list") or "[]")
        except json.JSONDecodeError:
            b["reading_list"] = []
        try:
            raw_bc2 = b.get("broadcast") or ""
            b["broadcast"] = json.loads(raw_bc2) if raw_bc2 else None
        except json.JSONDecodeError:
            b["broadcast"] = None
        try:
            raw_loc = b.get("location") or ""
            b["location"] = json.loads(raw_loc) if raw_loc else {"at": "home"}
        except json.JSONDecodeError:
            b["location"] = {"at": "home"}
        if not isinstance(b["location"], dict) or not (
                b["location"].get("at") or b["location"].get("to")):
            b["location"] = {"at": "home"}
        raw_bc = b.get("body_config") or ""
        try:
            b["body_config"] = json.loads(raw_bc) if raw_bc else None
        except json.JSONDecodeError:
            b["body_config"] = None
        for pending_col in ("pending_self_mod", "pending_procreation",
                            "pending_name"):
            raw_pending = b.get(pending_col) or ""
            try:
                b[pending_col] = json.loads(raw_pending) if raw_pending else None
            except json.JSONDecodeError:
                b[pending_col] = None
        return b

    def list(self, owner_id: str) -> list[dict]:
        rows = self._c().execute(
            "SELECT b.slug, b.name, b.stage, b.state, b.born_at, b.hatched_at,"
            " b.died_at, w.balance_tokens, w.allowance_preset"
            " FROM beings b LEFT JOIN being_wallets w ON w.being_id = b.id"
            " WHERE b.owner_id = ? ORDER BY b.born_at",
            (owner_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def _update(self, being_id: str, now: datetime, **fields) -> None:
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._c().execute(
                f"UPDATE beings SET {cols}, updated_at = ? WHERE id = ?",
                (*fields.values(), _iso(now), being_id),
            )
            self._c().commit()

    # ── Lifecycle ────────────────────────────────────────────────────

    def hatch(self, owner_id: str, slug: str, now: datetime | None = None) -> dict:
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        if b["stage"] != "egg":
            raise BeingError("already hatched")
        weights = genome_mod.derive(
            genome_mod.effective_attributes(b["genome"]))["drive_weights"]
        drives = {name: {"weight": w, "satisfaction": 0.7}
                  for name, w in weights.items()}
        self._update(b["id"], now, stage="infant", hatched_at=_iso(now),
                     drives=json.dumps(drives))
        self.record_event(b["id"], "hatched", {}, now=now)
        self.credit_allowance(b["id"], now=now)
        return self.get(owner_id, slug)

    def set_stage(self, owner_id: str, slug: str, stage: str,
                  now: datetime | None = None) -> dict:
        now = now or _utcnow()
        if stage not in constitution.STAGE_ORDER:
            raise BeingError(f"unknown stage {stage!r}")
        b = self.get(owner_id, slug)
        if b["state"] == "dead":
            raise BeingError("cannot change stage of a dead being")
        if stage == "egg":
            raise BeingError("cannot return to the egg")
        fields: dict = {"stage": stage}
        if stage == "adult":
            drives = dict(b.get("drives") or {})
            if drives and "legacy" not in drives:
                soc = genome_mod.effective_attributes(b["genome"])["SOC"]
                drives["legacy"] = {"weight": round(0.5 + 0.03 * soc, 3),
                                    "satisfaction": 0.6}
                fields["drives"] = json.dumps(drives)
        self._update(b["id"], now, **fields)
        self.record_event(b["id"], "stage", {"from": b["stage"], "to": stage}, now=now)
        return self.get(owner_id, slug)

    def set_tick_interval(self, owner_id: str, slug: str, minutes: int | None,
                          now: datetime | None = None) -> dict:
        """Pin the being's tick cadence in minutes (the parent sets the pace, #2),
        or None to return it to its own stage-clamped rhythm."""
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        val = None
        if minutes is not None:
            val = int(minutes)
            if val < 1 or val > 1440:
                raise BeingError("tick interval must be 1–1440 minutes")
        self._update(b["id"], now, tick_interval_minutes=val)
        self.record_event(b["id"], "cadence_set", {"minutes": val}, now=now)
        return self.get(owner_id, slug)

    def unread_from_being(self, being_id: str) -> int:
        """How many messages the being has spoken to the parent since the parent
        last opened its thread — ``spoke_to_parent`` events after parent_read_at."""
        row0 = self._c().execute(
            "SELECT parent_read_at FROM beings WHERE id = ?", (being_id,)
        ).fetchone()
        read_at = row0["parent_read_at"] if row0 else None
        q = ("SELECT COUNT(*) AS c FROM being_events WHERE being_id = ?"
             " AND kind = 'spoke_to_parent'")
        params: list = [being_id]
        if read_at:
            q += " AND at > ?"
            params.append(read_at)
        return int(self._c().execute(q, params).fetchone()["c"])

    def mark_being_read(self, owner_id: str, slug: str,
                        now: datetime | None = None) -> dict:
        """Mark the being's thread read up to now (the parent opened it)."""
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        self._update(b["id"], now, parent_read_at=_iso(now))
        return self.get(owner_id, slug)

    def set_body_archetype(self, owner_id: str, slug: str, archetype_id: str,
                           now: datetime | None = None) -> dict:
        """Point the being's BODY at an archetype (its tier→model/provider,
        tools, cognitive_mode), or '' to return to the stage tier + owner
        config. The caller respawns the body so the change takes effect."""
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        self._update(b["id"], now, body_archetype=(archetype_id or "").strip())
        self.record_event(b["id"], "body_archetype_set",
                          {"archetype": (archetype_id or "").strip()}, now=now)
        return self.get(owner_id, slug)

    def set_cognition(self, owner_id: str, slug: str, mode: str,
                      now: datetime | None = None) -> dict:
        """Choose how the being THINKS a tick: 'monolith' (one prompt → one
        digest) or 'faculties' (decomposed pipeline — better for weak-context
        models). See docs/being-faculties-plan.md."""
        now = now or _utcnow()
        if mode not in ("monolith", "faculties"):
            raise BeingError("cognition must be 'monolith' or 'faculties'")
        b = self.get(owner_id, slug)
        self._update(b["id"], now, cognition=mode)
        self.record_event(b["id"], "cognition_set", {"mode": mode}, now=now)
        return self.get(owner_id, slug)

    def set_compact_mode(self, owner_id: str, slug: str, on: bool,
                         now: datetime | None = None) -> dict:
        """Compact mode: tick prompts come from the compact instruction set
        (instructions/beings/compact_*.md) and the body runs lean (eco/micro
        system prompt + capped context window). Same narrative, same physics,
        fewer tokens per heartbeat."""
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        self._update(b["id"], now, compact_mode=1 if on else 0)
        self.record_event(b["id"], "compact_set", {"on": bool(on)}, now=now)
        return self.get(owner_id, slug)

    # ── Elderhood + the village radio (roadmap T3.14 / T3.16) ─────────

    def set_elder_after(self, owner_id: str, slug: str, days: int | None,
                        now: datetime | None = None) -> dict:
        """Opt this being into a natural span: after ``days`` alive it enters
        elderhood (a slower season, memoirs). None switches it off."""
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        val = None
        if days is not None:
            val = int(days)
            if val < 7 or val > 3650:
                raise BeingError("elderhood begins between 7 and 3650 days")
        self._update(b["id"], now, elder_after_days=val)
        self.record_event(b["id"], "elderhood_set", {"days": val}, now=now)
        return self.get(owner_id, slug)

    def set_broadcast(self, being_id: str, text: str,
                      now: datetime | None = None) -> None:
        """One line on the village radio (adult, public, once a day —
        callers gate; the store just keeps today's broadcast)."""
        now = now or _utcnow()
        self._update(being_id, now, broadcast=json.dumps(
            {"text": text[:200], "at": _iso(now)}))
        self.record_event(being_id, "broadcast_set", {"text": text[:200]},
                          now=now)

    # ── The ground (space plan Phase 1): places + movement physics ────────

    def village_places(self, owner_id: str) -> list[dict]:
        rows = self._c().execute(
            "SELECT * FROM village_places WHERE owner_id = ? ORDER BY id",
            (owner_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["affordances"] = json.loads(d.get("affordances") or "[]")
            except json.JSONDecodeError:
                d["affordances"] = []
            out.append(d)
        return out

    def get_place(self, owner_id: str, place_id: str) -> dict:
        row = self._c().execute(
            "SELECT * FROM village_places WHERE owner_id = ? AND id = ?",
            (owner_id, place_id),
        ).fetchone()
        if row is None:
            raise BeingNotFound(f"no place {place_id!r} in this village")
        d = dict(row)
        try:
            d["affordances"] = json.loads(d.get("affordances") or "[]")
        except json.JSONDecodeError:
            d["affordances"] = []
        return d

    def save_village(self, owner_id: str, places: list[dict],
                     now: datetime | None = None) -> list[dict]:
        """Replace the village ground (the architect's draft or the default).
        Validates HARD — the map is physics: kebab ids, unique, 'home'
        reserved, coordinates on the plot, affordances from the fixed
        vocabulary only. An invalid draft raises and changes nothing."""
        from captain_claw.flight_deck import being_world
        now = now or _utcnow()
        if not isinstance(places, list) or not (
                being_world.VILLAGE_MIN_PLACES <= len(places)
                <= being_world.VILLAGE_MAX_PLACES):
            raise BeingError(
                f"a village holds {being_world.VILLAGE_MIN_PLACES}–"
                f"{being_world.VILLAGE_MAX_PLACES} places")
        margin = 40
        seen: set[str] = set()
        clean: list[dict] = []
        for p in places:
            pid = _slugify(str(p.get("id") or p.get("name") or ""))[:40]
            if not pid:
                raise BeingError("every place needs an id")
            if pid == "home":
                raise BeingError("'home' is reserved — every being has its own")
            if pid in seen:
                raise BeingError(f"duplicate place id {pid!r}")
            seen.add(pid)
            name = str(p.get("name") or "").strip()[:60]
            if not name:
                raise BeingError(f"place {pid!r} needs a name")
            try:
                x, y = int(p.get("x")), int(p.get("y"))
            except (TypeError, ValueError):
                raise BeingError(f"place {pid!r} needs integer coordinates")
            hi = being_world.PLOT_SIZE - margin
            if not (margin <= x <= hi and margin <= y <= hi):
                raise BeingError(f"place {pid!r} is off the plot")
            aff = [str(a) for a in (p.get("affordances") or [])]
            bad = [a for a in aff if a not in being_world.AFFORDANCES]
            if bad:
                raise BeingError(
                    f"unknown affordances {bad} — the vocabulary is fixed")
            clean.append({"id": pid, "name": name, "x": x, "y": y,
                          "affordances": aff[:2],
                          "description": str(p.get("description") or "")
                          .strip()[:300]})
        with self._lock:
            c = self._c()
            c.execute("DELETE FROM village_places WHERE owner_id = ?",
                      (owner_id,))
            for p in clean:
                c.execute(
                    "INSERT INTO village_places (owner_id, id, name, x, y,"
                    " affordances, description, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (owner_id, p["id"], p["name"], p["x"], p["y"],
                     json.dumps(p["affordances"]), p["description"],
                     _iso(now)),
                )
            c.commit()
        return self.village_places(owner_id)

    def resolve_place_ref(self, owner_id: str, ref: str) -> str | None:
        """A being's words → a real place id: 'home', an exact id, the
        slugified ref, or the place's name (case-insensitive, 'the '
        optional). None means there is no such ground."""
        def _bare(s: str) -> str:
            s = s.casefold().strip()
            return s[4:].strip() if s.startswith("the ") else s
        r = (ref or "").strip()
        if not r:
            return None
        if _bare(r) == "home":
            return "home"
        low, slug = r.casefold(), _slugify(r)
        places = self.village_places(owner_id)
        for p in places:
            if p["id"] == low or p["id"] == slug:
                return p["id"]
        for p in places:
            if _bare(p["name"]) == _bare(r):
                return p["id"]
        return None

    def settle_location(self, being: dict, now: datetime | None = None,
                        ) -> dict | None:
        """Arrival is computed on read (the fever/steward pattern): if the
        walk has ended by `now`, write the rest state and record `arrived`
        AT the real arrival time — the world is simply further along when
        the being wakes. Returns {place, name, at} when it settled."""
        now = now or _utcnow()
        loc = being.get("location") or {"at": "home"}
        if loc.get("at") or not loc.get("to"):
            return None
        from captain_claw.flight_deck import being_world
        pos = being_world.position_of(self, being, now)
        if pos.get("to"):                          # still on the road
            return None
        place = pos["at"]
        self._update(being["id"], now, location=json.dumps({"at": place}))
        being["location"] = {"at": place}
        arrived_at = pos.get("arrived_at") or now
        name = being_world.place_name(self, being, place)
        hhmm = being_world._local(arrived_at).strftime("%H:%M")
        self.record_event(being["id"], "arrived",
                          {"place": place, "name": name, "hhmm": hhmm},
                          now=arrived_at)
        return {"place": place, "name": name, "at": _iso(arrived_at)}

    def depart(self, owner_id: str, slug: str, dest: str,
               now: datetime | None = None, *, reason: str = "") -> dict:
        """Set out for a place. Settles first (you leave from where you
        TRULY are — even mid-road), then the location row + the clock ARE
        the walk; nothing runs in the background. Unknown ground is refused
        loudly; already-there is a quiet no-op (no theater to record)."""
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        if b["state"] != "alive":
            raise BeingError("only the living walk")
        self.settle_location(b, now=now)
        from captain_claw.flight_deck import being_world
        pid = self.resolve_place_ref(owner_id, dest)
        if pid is None:
            raise BeingError(f"there is no place called {str(dest)[:40]!r} "
                             "here — read commons/village/MAP.md")
        cur = b.get("location") or {"at": "home"}
        if cur.get("at") == pid:
            return b
        pos = being_world.position_of(self, b, now)
        origin = [int(pos["xy"][0]), int(pos["xy"][1])]
        dest_xy = being_world.place_xy(self, b, pid)
        minutes = being_world.travel_minutes(b, origin, dest_xy)
        loc = {"to": pid, "from": cur.get("at"), "origin": origin,
               "departed_at": _iso(now)}
        self._update(b["id"], now, location=json.dumps(loc))
        data = {"from": cur.get("at") or "the road", "to": pid,
                "minutes": int(round(minutes))}
        if reason:
            data["reason"] = reason
        self.record_event(b["id"], "departed", data, now=now)
        return self.get(owner_id, slug)

    # ── The naming rite (roadmap T2.10): one chosen name, parent-blessed ──

    def set_pending_name(self, being_id: str, pending: dict | None,
                         now: datetime | None = None) -> None:
        now = now or _utcnow()
        self._update(being_id, now,
                     pending_name=json.dumps(pending) if pending else "")

    def approve_name(self, owner_id: str, slug: str,
                     now: datetime | None = None) -> dict:
        """The parent blesses the chosen name: the display name changes (the
        slug never does), the choice enters the genome's epigenetics (the
        being's own mark, not inherited), and the milestone lands."""
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        pending = b.get("pending_name")
        if not pending or not pending.get("name"):
            raise BeingError("no chosen name awaits a blessing")
        old, new = b["name"], str(pending["name"])[:60].strip()
        genome = dict(b["genome"])
        epi = dict(genome.get("epigenetics") or {})
        epi["chosen_name"] = new
        epi["chosen_name_why"] = str(pending.get("why") or "")[:200]
        genome["epigenetics"] = epi
        self._update(b["id"], now, name=new, pending_name="",
                     genome=json.dumps(genome))
        self.record_event(b["id"], "name_chosen",
                          {"from": old, "to": new,
                           "why": str(pending.get("why") or "")[:300]}, now=now)
        self.milestone(b["id"], "chose_name", {"name": new}, now=now)
        return self.get(owner_id, slug)

    def reject_name(self, owner_id: str, slug: str, note: str = "",
                    now: datetime | None = None) -> dict:
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        if not b.get("pending_name"):
            raise BeingError("no chosen name awaits a decision")
        self.set_pending_name(b["id"], None, now=now)
        self.record_event(b["id"], "name_rejected", {"note": note[:300]},
                          now=now)
        return self.get(owner_id, slug)

    # ── Education: the reading list (roadmap T2.12) ───────────────────

    READING_MAX_FEE_TOKENS = 2_000_000

    def add_reading(self, owner_id: str, slug: str, ref: str, note: str = "",
                    fee_tokens: int = 0, now: datetime | None = None) -> dict:
        """The parent assigns one reading (a URL or anything nameable) with a
        small fee for a verified report. Media diet applies to web refs at
        read time exactly as to any other fetch."""
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        ref = (ref or "").strip()
        if not ref:
            raise BeingError("a reading needs something to read")
        fee = max(0, min(int(fee_tokens or 0), self.READING_MAX_FEE_TOKENS))
        item = {"id": uuid.uuid4().hex[:8], "ref": ref[:400],
                "note": (note or "").strip()[:200], "fee_tokens": fee,
                "assigned_at": _iso(now), "done_at": None,
                "report_path": None}
        items = list(b.get("reading_list") or []) + [item]
        self._update(b["id"], now, reading_list=json.dumps(items))
        self.record_event(b["id"], "reading_assigned",
                          {"id": item["id"], "ref": ref[:200],
                           "fee_tokens": fee}, now=now)
        return self.get(owner_id, slug)

    def remove_reading(self, owner_id: str, slug: str, item_id: str,
                       now: datetime | None = None) -> dict:
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        items = [i for i in (b.get("reading_list") or [])
                 if i.get("id") != item_id]
        if len(items) == len(b.get("reading_list") or []):
            raise BeingNotFound("no such reading")
        self._update(b["id"], now, reading_list=json.dumps(items))
        return self.get(owner_id, slug)

    def complete_reading(self, owner_id: str, slug: str, item_id: str,
                         path: str, now: datetime | None = None, *,
                         fee_factor: float = 1.0) -> dict:
        """Mark a reading done against a VERIFIED report file (the caller
        checked the disk) and pay the fee — the only mint here, same
        conservation as a judged chore. ``fee_factor`` is the place bonus
        (space plan Phase 3: reading finished at a read-place mints a
        little more), still capped by READING_MAX_FEE_TOKENS and the
        savings ceiling. Returns the completed item."""
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        items = list(b.get("reading_list") or [])
        match = next((i for i in items
                      if i.get("id", "").startswith(item_id)
                      and not i.get("done_at")), None)
        if match is None:
            raise BeingNotFound("no open reading by that id")
        match["done_at"] = _iso(now)
        match["report_path"] = path[:200]
        self._update(b["id"], now, reading_list=json.dumps(items))
        fee = int(match.get("fee_tokens") or 0)
        if fee > 0 and fee_factor != 1.0:
            fee = min(int(fee * fee_factor), self.READING_MAX_FEE_TOKENS)
        if fee > 0:
            view = self.wallet_view(b)
            if view["savings_ceiling"] is not None:
                fee = min(fee, max(0, view["savings_ceiling"]
                                   - view["balance_tokens"]))
            if fee > 0:
                self._apply(owner_id, tokens=fee, reason="fee",
                            from_being=None, to_being=b["id"],
                            note=f"reading:{match['id']}", now=now)
        self.record_event(b["id"], "reading_done",
                          {"id": match["id"], "ref": match["ref"][:200],
                           "path": path[:200], "fee_tokens": fee}, now=now)
        self.milestone(b["id"], "first_report",
                       {"ref": match["ref"][:120]}, now=now)
        return match

    def penpals_sent_today(self, being_id: str,
                           now: datetime | None = None) -> int:
        """Pen-pal letters sent today (events, not being_letters rows) — they
        share the stage's daily letter quota (roadmap T2.8)."""
        now = now or _utcnow()
        day = _iso(now)[:10]
        n = 0
        for e in self._c().execute(
                "SELECT data, at FROM being_events WHERE being_id = ?"
                " AND kind = 'penpal_sent' AND at >= ? ORDER BY at DESC",
                (being_id, day)).fetchall():
            n += 1
        return n

    def purge(self, owner_id: str, slug: str) -> dict:
        """Erase a DEAD being completely — every row it owns across every table.
        Only the dead can be purged (euthanize first); this is irreversible and
        leaves no remains in the DB. The VFS home is removed by the caller.
        Returns the removed being's identity for the caller (home cleanup)."""
        b = self.get(owner_id, slug)
        if b["state"] != "dead":
            raise BeingError("only a dead being can be removed — euthanize it "
                             "first", 409)
        bid = b["id"]
        with self._lock:
            c = self._c()
            # Rows keyed by being_id.
            for tbl in ("being_wallets", "being_events", "being_jobs",
                        "being_publications", "being_ventures",
                        "being_parent_messages", "being_links",
                        "being_assessments", "being_public_threads",
                        "being_public_messages"):
                c.execute(f"DELETE FROM {tbl} WHERE being_id = ?", (bid,))
            # Rows that reference it from either side.
            c.execute("DELETE FROM token_transfers WHERE from_being = ?"
                      " OR to_being = ?", (bid, bid))
            c.execute("DELETE FROM being_letters WHERE from_being = ?"
                      " OR to_being = ?", (bid, bid))
            # Open bounties it had claimed return to the board; paid history stays.
            c.execute("UPDATE being_quests SET state = 'open', claimed_by = NULL,"
                      " claimed_at = NULL, done_at = NULL WHERE claimed_by = ?"
                      " AND state IN ('claimed', 'judging')", (bid,))
            c.execute("DELETE FROM beings WHERE id = ?", (bid,))
            c.commit()
        return {"id": bid, "slug": b["slug"], "name": b["name"],
                "owner_id": owner_id, "agent_slug": b.get("agent_slug")}

    def set_state(self, owner_id: str, slug: str, state: str,
                  now: datetime | None = None) -> dict:
        now = now or _utcnow()
        if state not in STATES:
            raise BeingError(f"unknown state {state!r}")
        b = self.get(owner_id, slug)
        if b["state"] == "dead":
            raise BeingError("a dead being stays dead")
        if b["state"] == "emigrated":
            raise BeingError("an emigrated being lives elsewhere now — "
                             "its life here is closed")
        fields: dict = {"state": state}
        if state == "dead":
            fields["died_at"] = _iso(now)
        if state == "torpor" and b["state"] != "torpor":
            fields["torpor_since"] = _iso(now)
        elif state == "alive":
            fields["torpor_since"] = None
        self._update(b["id"], now, **fields)
        self.record_event(b["id"], "state", {"from": b["state"], "to": state}, now=now)
        return self.get(owner_id, slug)

    # ── Wallet (the only door tokens pass through) ───────────────────

    def wallet_view(self, being: dict) -> dict:
        """Effective wallet numbers after Constitution clamps."""
        w = self._c().execute(
            "SELECT * FROM being_wallets WHERE being_id = ?", (being["id"],)
        ).fetchone()
        if not w:
            raise BeingNotFound("wallet missing")
        w = dict(w)
        eff_preset = constitution.clamp_preset(being["stage"], w["allowance_preset"])
        per_day = constitution.ALLOWANCE_PRESETS[eff_preset]
        enforced = per_day is not None
        ceiling = w["savings_ceiling"]
        if ceiling is None:
            ceiling = constitution.savings_ceiling_tokens(
                being["stage"], w["allowance_preset"])
        burn_cap = w["daily_burn_cap"]
        if burn_cap is None:
            burn_cap = per_day
        return {
            "balance_tokens": w["balance_tokens"],
            "allowance_preset": w["allowance_preset"],
            "effective_preset": eff_preset,
            "per_day_tokens": per_day,
            "enforced": enforced,
            "savings_ceiling": ceiling,
            "daily_burn_cap": burn_cap,
            "reserve_tokens": w["reserve_tokens"],
        }

    def set_allowance(
        self, owner_id: str, slug: str, preset: str,
        daily_burn_cap: int | None = None, savings_ceiling: int | None = None,
        now: datetime | None = None,
    ) -> dict:
        now = now or _utcnow()
        if preset not in constitution.ALLOWANCE_PRESETS:
            raise BeingError(f"unknown allowance preset {preset!r}")
        b = self.get(owner_id, slug)
        with self._lock:
            self._c().execute(
                "UPDATE being_wallets SET allowance_preset = ?, daily_burn_cap = ?,"
                " savings_ceiling = ?, updated_at = ? WHERE being_id = ?",
                (preset, daily_burn_cap, savings_ceiling, _iso(now), b["id"]),
            )
            self._c().commit()
        return self.wallet_view(self.get(owner_id, slug))

    def _apply(
        self, owner_id: str, *, tokens: int, reason: str,
        from_being: str | None, to_being: str | None,
        job_id: str | None = None, note: str | None = None,
        now: datetime,
    ) -> None:
        """The single balance-mutation path: ledger row + balance updates,
        one transaction. Callers validate; this conserves."""
        if tokens <= 0:
            raise BeingError("tokens must be positive")
        if reason not in TRANSFER_REASONS:
            raise BeingError(f"unknown transfer reason {reason!r}")
        with self._lock:
            c = self._c()
            c.execute(
                "INSERT INTO token_transfers"
                " (id, owner_id, from_being, to_being, tokens, reason, job_id, note, at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, owner_id, from_being, to_being, tokens,
                 reason, job_id, note, _iso(now)),
            )
            if from_being:
                c.execute(
                    "UPDATE being_wallets SET balance_tokens = balance_tokens - ?,"
                    " updated_at = ? WHERE being_id = ?",
                    (tokens, _iso(now), from_being),
                )
            if to_being:
                c.execute(
                    "UPDATE being_wallets SET balance_tokens = balance_tokens + ?,"
                    " updated_at = ? WHERE being_id = ?",
                    (tokens, _iso(now), to_being),
                )
            c.commit()

    def _being_by_id(self, being_id: str) -> dict:
        row = self._c().execute(
            "SELECT * FROM beings WHERE id = ?", (being_id,)
        ).fetchone()
        if not row:
            raise BeingNotFound("no such being")
        b = dict(row)
        b["genome"] = json.loads(b["genome"])
        return b

    def credit_allowance(self, being_id: str, now: datetime | None = None) -> int:
        """Daily allowance mint — idempotent per (being, date); clipped at the
        piggy-bank ceiling. Returns tokens actually credited."""
        now = now or _utcnow()
        b = self._being_by_id(being_id)
        if b["state"] == "dead" or b["stage"] == "egg":
            return 0
        view = self.wallet_view(b)
        if not view["enforced"]:
            return 0
        date_key = _iso(now)[:10]
        exists = self._c().execute(
            "SELECT 1 FROM token_transfers WHERE to_being = ? AND reason = 'allowance'"
            " AND note = ? LIMIT 1",
            (being_id, date_key),
        ).fetchone()
        if exists:
            return 0
        headroom = view["per_day_tokens"]
        if view["savings_ceiling"] is not None:
            headroom = min(headroom,
                           max(0, view["savings_ceiling"] - view["balance_tokens"]))
        if headroom <= 0:
            return 0
        self._apply(b["owner_id"], tokens=headroom, reason="allowance",
                    from_being=None, to_being=being_id, note=date_key, now=now)
        return headroom

    def grant(self, owner_id: str, slug: str, tokens: int,
              now: datetime | None = None) -> dict:
        """A parent top-up: mint tokens straight into the wallet (the parent is
        the only token source — plan §economy). Unlike the daily allowance this
        is NOT stage-capped or once-per-day; it's how a parent revives an
        exhausted being or funds it past its stage's daily cap. Conserved as one
        ``grant`` ledger row. Reviving from torpor is the caller's job (it owns
        the body). Returns fresh vitals."""
        now = now or _utcnow()
        tokens = int(tokens)
        if tokens <= 0:
            raise BeingError("a recharge must be a positive number of tokens")
        if tokens > GRANT_AMOUNTS[-1]:
            raise BeingError(
                f"a single recharge is capped at {GRANT_AMOUNTS[-1]} tokens")
        b = self.get(owner_id, slug)
        if b["state"] == "dead":
            raise BeingError("a dead being cannot be funded")
        if b["stage"] == "egg":
            raise BeingError("an egg has no wallet to recharge yet")
        self._apply(owner_id, tokens=tokens, reason="grant",
                    from_being=None, to_being=b["id"], note="parent", now=now)
        self.record_event(b["id"], "granted", {"tokens": tokens}, now=now)
        return self.vitals(owner_id, slug)

    def spent_today(self, being_id: str, now: datetime | None = None) -> int:
        now = now or _utcnow()
        row = self._c().execute(
            "SELECT COALESCE(SUM(tokens), 0) AS s FROM token_transfers"
            " WHERE from_being = ? AND reason = 'usage' AND at LIKE ?",
            (being_id, _iso(now)[:10] + "%"),
        ).fetchone()
        return int(row["s"])

    def granted_today(self, being_id: str, now: datetime | None = None) -> int:
        """Tokens the parent minted into this wallet TODAY (recharges). A grant
        is a deliberate 'keep going', so it raises today's burn headroom — the
        being may spend the daily cap PLUS what it was granted today."""
        now = now or _utcnow()
        row = self._c().execute(
            "SELECT COALESCE(SUM(tokens), 0) AS s FROM token_transfers"
            " WHERE to_being = ? AND reason = 'grant' AND at LIKE ?",
            (being_id, _iso(now)[:10] + "%"),
        ).fetchone()
        return int(row["s"])

    # ── Coins (space plan Phase 2): money, not metabolism ──────────────

    def _apply_coins(self, owner_id: str, being_id: str, delta: int,
                     reason: str, *, from_being: str | None = None,
                     data: dict | None = None,
                     now: datetime | None = None) -> int:
        """The single coin-mutation path: one ledger row, balance = SUM.
        Only physics calls this — no LLM path can move a coin. Overdrafts
        are refused: there is no negative money. Returns the new balance."""
        now = now or _utcnow()
        delta = int(delta)
        if delta == 0:
            raise BeingError("a coin move needs a direction")
        if reason not in COIN_REASONS:
            raise BeingError(f"unknown coin reason {reason!r}")
        with self._lock:
            c = self._c()
            if delta < 0:
                bal = c.execute(
                    "SELECT COALESCE(SUM(delta),0) AS s"
                    " FROM being_coin_events WHERE being_id = ?",
                    (being_id,)).fetchone()["s"]
                if bal + delta < 0:
                    raise BeingError(f"not enough coins — you have {bal}")
            c.execute(
                "INSERT INTO being_coin_events (id, owner_id, being_id,"
                " delta, reason, from_being, data, at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, owner_id, being_id, delta, reason,
                 from_being, json.dumps(data or {}), _iso(now)))
            c.commit()
        return self.coin_balance(being_id)

    def coin_balance(self, being_id: str) -> int:
        row = self._c().execute(
            "SELECT COALESCE(SUM(delta),0) AS s FROM being_coin_events"
            " WHERE being_id = ?", (being_id,)).fetchone()
        return int(row["s"])

    def coin_ledger(self, owner_id: str, slug: str,
                    limit: int = 100) -> list[dict]:
        b = self.get(owner_id, slug)
        rows = self._c().execute(
            "SELECT * FROM being_coin_events WHERE being_id = ?"
            " ORDER BY at DESC LIMIT ?", (b["id"], limit)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["data"] = json.loads(d.get("data") or "{}")
            except json.JSONDecodeError:
                d["data"] = {}
            out.append(d)
        return out

    def grant_coins(self, owner_id: str, slug: str, coins: int,
                    note: str = "", now: datetime | None = None) -> dict:
        """Pocket money — the parent's coin faucet (every faucet is a
        parent act or circulation; the real-dollar exposure stays what the
        parent authorizes). Even an infant may RECEIVE — a gift is not a
        trade; spending waits on the stage."""
        now = now or _utcnow()
        coins = int(coins)
        if coins <= 0:
            raise BeingError("pocket money must be a positive number of coins")
        if coins > constitution.COIN_GRANT_MAX:
            raise BeingError(f"a single grant is capped at "
                             f"{constitution.COIN_GRANT_MAX} coins")
        b = self.get(owner_id, slug)
        if b["state"] == "dead":
            raise BeingError("a dead being cannot be funded")
        if b["stage"] == "egg":
            raise BeingError("an egg has no pocket yet")
        self._apply_coins(owner_id, b["id"], coins, "grant",
                          data={"note": note[:120]}, now=now)
        self.record_event(b["id"], "coins_granted",
                          {"coins": coins, "note": note[:120]}, now=now)
        return self.vitals(owner_id, slug)

    def convert_coins(self, owner_id: str, slug: str, coins: int,
                      now: datetime | None = None) -> dict:
        """The ONE-WAY exchange (space plan Phase 2): coins → tokens at
        COIN_TOKEN_RATE, whole coins only, clamped to the savings-ceiling
        headroom — the ledger records any clamp. The reverse direction
        does not exist: liquidating metabolism into wealth would recreate
        the original problem and let the allowance print money."""
        now = now or _utcnow()
        requested = int(coins)
        if requested <= 0:
            raise BeingError("convert a positive number of coins")
        b = self.get(owner_id, slug)
        if b["state"] != "alive":
            raise BeingError("only the living trade")
        if not constitution.has_capability(b["stage"], "convert"):
            raise BeingError("conversion opens in adolescence")
        balance = self.coin_balance(b["id"])
        if balance <= 0:
            raise BeingError("you have no coins")
        rate = constitution.COIN_TOKEN_RATE
        view = self.wallet_view(b)
        possible = min(requested, balance)
        if view["savings_ceiling"] is not None:
            headroom = max(0, view["savings_ceiling"]
                           - view["balance_tokens"])
            possible = min(possible, headroom // rate)
        if possible <= 0:
            raise BeingError("your savings are full — no room for more "
                             "tokens; spend some thinking first")
        minted = possible * rate
        # Debit first (it can refuse); the mint below cannot fail.
        self._apply_coins(owner_id, b["id"], -possible, "exchange",
                          data={"tokens": minted, "requested": requested},
                          now=now)
        self._apply(owner_id, tokens=minted, reason="exchange",
                    from_being=None, to_being=b["id"], note="coins", now=now)
        self.record_event(b["id"], "coins_converted",
                          {"coins": possible, "tokens": minted,
                           "requested": requested}, now=now)
        return {"coins": possible, "tokens": minted,
                "requested": requested,
                "balance_coins": self.coin_balance(b["id"])}

    # ── Contacts + the market (space plan Phase 3) ─────────────────────

    def touch_contact(self, owner_id: str, a_id: str, b_id: str,
                      now: datetime | None = None) -> bool:
        """One real meeting between two beings: the pair's contact grows
        asymptotically (the satiation curve again). Returns False when the
        pair already met today — one hello per pair per day, symmetric,
        deduped HERE so both directions agree."""
        now = now or _utcnow()
        lo, hi = sorted((a_id, b_id))
        from captain_claw.flight_deck import being_world
        step = being_world.CONTACT_STRENGTH_STEP
        with self._lock:
            c = self._c()
            row = c.execute(
                "SELECT * FROM being_contacts WHERE owner_id = ?"
                " AND a_id = ? AND b_id = ?", (owner_id, lo, hi)).fetchone()
            if row and (row["last_met_at"] or "")[:10] == _iso(now)[:10]:
                return False
            if row:
                strength = min(1.0, float(row["strength"])
                               + step * (1.0 - float(row["strength"])))
                c.execute(
                    "UPDATE being_contacts SET met_count = met_count + 1,"
                    " strength = ?, last_met_at = ? WHERE owner_id = ?"
                    " AND a_id = ? AND b_id = ?",
                    (round(strength, 4), _iso(now), owner_id, lo, hi))
            else:
                c.execute(
                    "INSERT INTO being_contacts (owner_id, a_id, b_id,"
                    " met_count, strength, last_met_at) VALUES (?,?,?,1,?,?)",
                    (owner_id, lo, hi, round(step, 4), _iso(now)))
            c.commit()
        return True

    def contacts_for(self, owner_id: str, slug: str) -> list[dict]:
        b = self.get(owner_id, slug)
        names = self.names_by_id(owner_id)
        rows = self._c().execute(
            "SELECT * FROM being_contacts WHERE owner_id = ?"
            " AND (a_id = ? OR b_id = ?) ORDER BY strength DESC",
            (owner_id, b["id"], b["id"])).fetchall()
        out = []
        for r in rows:
            other = r["b_id"] if r["a_id"] == b["id"] else r["a_id"]
            out.append({"with": names.get(other, "?"),
                        "met_count": r["met_count"],
                        "strength": r["strength"],
                        "last_met_at": r["last_met_at"]})
        return out

    def trades_today(self, being_id: str,
                     now: datetime | None = None) -> int:
        """Sells posted + buys made today — one quota covers both sides
        of the counter (ledger-computed, no state to drift)."""
        day = _iso(now or _utcnow())[:10] + "%"
        c = self._c()
        sells = c.execute(
            "SELECT COUNT(*) AS n FROM village_listings WHERE seller_id = ?"
            " AND created_at LIKE ?", (being_id, day)).fetchone()["n"]
        buys = c.execute(
            "SELECT COUNT(*) AS n FROM village_listings WHERE sold_to = ?"
            " AND sold_at LIKE ?", (being_id, day)).fetchone()["n"]
        return int(sells) + int(buys)

    def _trade_gate(self, being: dict, now: datetime) -> None:
        from captain_claw.flight_deck import being_world
        cap = being_world.trades_cap(being["stage"], now)
        if cap <= 0:
            raise BeingError(f"a {being['stage']} does not trade yet")
        if self.trades_today(being["id"], now) >= cap:
            raise BeingError(f"your trades are spent for today (limit {cap})")

    def post_listing(self, owner_id: str, slug: str, path: str, title: str,
                     price_coins: int, now: datetime | None = None) -> dict:
        """A stall at the market. The FILE's existence is the caller's
        check (being_society owns file physics); here live the quota, the
        price cap, and the row."""
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        if b["state"] != "alive":
            raise BeingError("only the living trade")
        self._trade_gate(b, now)
        price = int(price_coins)
        if price <= 0:
            raise BeingError("price must be a positive number of coins")
        if price > constitution.MARKET_MAX_PRICE_COINS:
            raise BeingError(f"the market caps prices at "
                             f"{constitution.MARKET_MAX_PRICE_COINS} coins")
        clean_path = (path or "").strip().lstrip("/")[:200]
        clean_title = (title or "").strip()[:80]
        if not clean_path or not clean_title:
            raise BeingError("a listing needs a real path and a title")
        lid = uuid.uuid4().hex
        with self._lock:
            self._c().execute(
                "INSERT INTO village_listings (id, owner_id, seller_id,"
                " path, title, price_coins, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (lid, owner_id, b["id"], clean_path, clean_title, price,
                 _iso(now)))
            self._c().commit()
        self.record_event(b["id"], "market_listed",
                          {"listing_id": lid, "title": clean_title,
                           "price_coins": price, "path": clean_path},
                          now=now)
        return self.get_listing(owner_id, lid)

    def market_listings(self, owner_id: str, limit: int = 20,
                        state: str = "open") -> list[dict]:
        rows = self._c().execute(
            "SELECT l.*, b.name AS seller, b.slug AS seller_slug"
            " FROM village_listings l JOIN beings b ON b.id = l.seller_id"
            " WHERE l.owner_id = ? AND l.state = ?"
            " ORDER BY l.created_at DESC LIMIT ?",
            (owner_id, state, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_listing(self, owner_id: str, listing_id: str) -> dict:
        rows = self._c().execute(
            "SELECT l.*, b.name AS seller, b.slug AS seller_slug"
            " FROM village_listings l JOIN beings b ON b.id = l.seller_id"
            " WHERE l.owner_id = ? AND (l.id = ? OR l.id LIKE ?) LIMIT 2",
            (owner_id, listing_id, f"{listing_id}%")).fetchall()
        if len(rows) != 1:
            raise BeingNotFound("no such listing")
        return dict(rows[0])

    def buy_listing(self, owner_id: str, buyer_slug: str, listing_id: str,
                    now: datetime | None = None) -> dict:
        """The coin side of a purchase: atomic claim of the stall, then
        the being→being pair (purchase/sale — circulation, never minting).
        The file itself moves in being_society (read-before-pay). Refused
        loudly when broke, self-dealing, or already sold."""
        now = now or _utcnow()
        b = self.get(owner_id, buyer_slug)
        if b["state"] != "alive":
            raise BeingError("only the living trade")
        li = self.get_listing(owner_id, listing_id)
        if li["state"] != "open":
            raise BeingError("that stall is empty — already sold")
        if li["seller_id"] == b["id"]:
            raise BeingError("it is already yours — no self-dealing")
        self._trade_gate(b, now)
        price = int(li["price_coins"])
        if self.coin_balance(b["id"]) < price:
            raise BeingError(f"not enough coins — it costs {price}, you "
                             f"have {self.coin_balance(b['id'])}")
        with self._lock:
            cur = self._c().execute(
                "UPDATE village_listings SET state = 'sold', sold_to = ?,"
                " sold_at = ? WHERE id = ? AND state = 'open'",
                (b["id"], _iso(now), li["id"]))
            self._c().commit()
            if cur.rowcount != 1:
                raise BeingError("that stall is empty — already sold")
        try:
            self._apply_coins(owner_id, b["id"], -price, "purchase",
                              from_being=li["seller_id"],
                              data={"listing_id": li["id"],
                                    "title": li["title"]}, now=now)
        except BeingError:
            with self._lock:                       # un-claim; nothing moved
                self._c().execute(
                    "UPDATE village_listings SET state = 'open',"
                    " sold_to = NULL, sold_at = NULL WHERE id = ?",
                    (li["id"],))
                self._c().commit()
            raise
        self._apply_coins(owner_id, li["seller_id"], price, "sale",
                          from_being=b["id"],
                          data={"listing_id": li["id"],
                                "title": li["title"]}, now=now)
        self.record_event(b["id"], "market_bought",
                          {"listing_id": li["id"], "title": li["title"],
                           "price_coins": price, "from": li["seller"],
                           "path": li["path"]}, now=now)
        self.record_event(li["seller_id"], "market_sold",
                          {"listing_id": li["id"], "title": li["title"],
                           "price_coins": price, "to": b["name"]}, now=now)
        self.milestone(li["seller_id"], "first_sale",
                       {"title": li["title"]}, now=now)
        return self.get_listing(owner_id, li["id"])

    # ── Commissioned buildings (space plan Phase 5) ────────────────────

    def add_place(self, owner_id: str, place: dict,
                  now: datetime | None = None) -> dict:
        """Raise ONE new place on existing ground (the commission's build
        step) — same physics gates as save_village, plus uniqueness against
        what already stands. The id gains a numeric suffix if taken."""
        from captain_claw.flight_deck import being_world
        now = now or _utcnow()
        existing = {p["id"] for p in self.village_places(owner_id)}
        if len(existing) >= being_world.VILLAGE_MAX_PLACES:
            raise BeingError("the village is full — no ground left to raise")
        base = _slugify(str(place.get("id") or place.get("name") or ""))[:36]
        if not base or base == "home":
            raise BeingError("a place needs a real name")
        pid, n = base, 2
        while pid in existing:
            pid, n = f"{base}-{n}", n + 1
        name = str(place.get("name") or "").strip()[:60]
        if not name:
            raise BeingError("a place needs a name")
        try:
            x, y = int(place.get("x")), int(place.get("y"))
        except (TypeError, ValueError):
            raise BeingError("a place needs integer coordinates")
        hi = being_world.PLOT_SIZE - 40
        if not (40 <= x <= hi and 40 <= y <= hi):
            raise BeingError("that spot is off the plot")
        aff = [str(a) for a in (place.get("affordances") or [])][:2]
        if not aff or any(a not in being_world.AFFORDANCES for a in aff):
            raise BeingError("unknown affordances — the vocabulary is fixed")
        with self._lock:
            self._c().execute(
                "INSERT INTO village_places (owner_id, id, name, x, y,"
                " affordances, description, created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (owner_id, pid, name, x, y, json.dumps(aff),
                 str(place.get("description") or "").strip()[:300],
                 _iso(now)))
            self._c().commit()
        return self.get_place(owner_id, pid)

    def open_commission(self, owner_id: str) -> dict | None:
        """The village's ONE active building fund (open or funded)."""
        row = self._c().execute(
            "SELECT * FROM village_commissions WHERE owner_id = ?"
            " AND state IN ('open', 'funded')"
            " ORDER BY created_at DESC LIMIT 1", (owner_id,)).fetchone()
        return dict(row) if row else None

    def _commission_escrow(self, owner_id: str, being: dict, cid: str,
                           coins: int, remaining: int,
                           now: datetime) -> int:
        coins = min(int(coins), remaining,
                    self.coin_balance(being["id"]))
        if coins <= 0:
            raise BeingError("nothing to give — coins first")
        self._apply_coins(owner_id, being["id"], -coins, "commission",
                          data={"commission_id": cid}, now=now)
        return coins

    def propose_commission(self, owner_id: str, slug: str, name: str,
                           why: str, affordance: str, coins: int,
                           now: datetime | None = None) -> dict:
        """A being proposes a NEW building and puts its own coins down
        first (a commission is skin, not talk). Adolescence on; one active
        fund per village; the target is Constitution physics."""
        from captain_claw.flight_deck import being_world
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        if b["state"] != "alive":
            raise BeingError("only the living build")
        if constitution.stage_index(b["stage"]) < \
                constitution.stage_index("adolescent"):
            raise BeingError("commissioning opens in adolescence")
        if self.open_commission(owner_id):
            raise BeingError("one building at a time — the current "
                             "commission must close first")
        name = (name or "").strip()[:40]
        if len(name) < 3:
            raise BeingError("a building needs a real name")
        if affordance not in being_world.AFFORDANCES:
            raise BeingError("unknown affordance — the vocabulary is fixed")
        cid = uuid.uuid4().hex
        target = constitution.COMMISSION_COST_COINS
        paid = self._commission_escrow(owner_id, b, cid, coins, target, now)
        state = "funded" if paid >= target else "open"
        with self._lock:
            self._c().execute(
                "INSERT INTO village_commissions (id, owner_id, name, why,"
                " affordance, target_coins, raised_coins, state, created_by,"
                " created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (cid, owner_id, name, (why or "").strip()[:300], affordance,
                 target, paid, state, b["id"], _iso(now)))
            self._c().commit()
        self.record_event(b["id"], "commission_proposed",
                          {"commission_id": cid, "name": name,
                           "affordance": affordance, "coins": paid,
                           "target": target}, now=now)
        if state == "funded":
            self.record_event(b["id"], "commission_funded",
                              {"commission_id": cid, "name": name}, now=now)
        return self.open_commission(owner_id) or {}

    def contribute_commission(self, owner_id: str, slug: str, coins: int,
                              now: datetime | None = None) -> dict:
        """Any coin-holder may add to the open fund — pooling is the point.
        Clamped to what remains and to the pocket; refused when empty."""
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        if b["state"] != "alive":
            raise BeingError("only the living build")
        c = self.open_commission(owner_id)
        if not c:
            raise BeingError("no commission is open — propose one first")
        if c["state"] == "funded":
            raise BeingError(f'"{c["name"]}" is fully funded — it waits '
                             "on the parent now")
        remaining = int(c["target_coins"]) - int(c["raised_coins"])
        paid = self._commission_escrow(owner_id, b, c["id"], coins,
                                       remaining, now)
        raised = int(c["raised_coins"]) + paid
        state = "funded" if raised >= int(c["target_coins"]) else "open"
        with self._lock:
            self._c().execute(
                "UPDATE village_commissions SET raised_coins = ?, state = ?"
                " WHERE id = ?", (raised, state, c["id"]))
            self._c().commit()
        self.record_event(b["id"], "commission_contributed",
                          {"commission_id": c["id"], "name": c["name"],
                           "coins": paid, "raised": raised,
                           "target": c["target_coins"]}, now=now)
        if state == "funded":
            self.record_event(c["created_by"], "commission_funded",
                              {"commission_id": c["id"], "name": c["name"]},
                              now=now)
        return self.open_commission(owner_id) or {"state": "funded"}

    def commission_contributors(self, owner_id: str,
                                commission_id: str) -> list[dict]:
        """Who paid what — read straight off the coin ledger (no state)."""
        marker = f'"commission_id": "{commission_id}"'
        rows = self._c().execute(
            "SELECT being_id, -SUM(delta) AS coins FROM being_coin_events"
            " WHERE owner_id = ? AND reason = 'commission' AND delta < 0"
            " AND data LIKE ? GROUP BY being_id",
            (owner_id, f"%{marker}%")).fetchall()
        names = self.names_by_id(owner_id)
        return [{"being_id": r["being_id"],
                 "name": names.get(r["being_id"], "?"),
                 "coins": int(r["coins"])} for r in rows]

    def judge_commission(self, owner_id: str, approve: bool, note: str = "",
                         now: datetime | None = None) -> dict:
        """The parent's word: approve a FUNDED commission → the architect
        places it (deterministic spot) and the coins burn — the economy's
        sink; reject any active one → every contributor is refunded to the
        coin, on the ledger."""
        from captain_claw.flight_deck import being_world
        now = now or _utcnow()
        c = self.open_commission(owner_id)
        if not c:
            raise BeingError("no commission awaits a judgment")
        contributors = self.commission_contributors(owner_id, c["id"])
        if approve:
            if c["state"] != "funded":
                raise BeingError(f'"{c["name"]}" is not fully funded yet '
                                 f'({c["raised_coins"]}/{c["target_coins"]})')
            spot = being_world.commission_spot(self, owner_id, c["id"])
            place = self.add_place(owner_id, {
                "id": c["name"], "name": c["name"],
                "x": spot[0], "y": spot[1],
                "affordances": [c["affordance"]],
                "description": c["why"] or "raised by the village, coin "
                "by coin"}, now=now)
            with self._lock:
                self._c().execute(
                    "UPDATE village_commissions SET state = 'approved',"
                    " decided_at = ?, note = ?, place_id = ? WHERE id = ?",
                    (_iso(now), note[:300], place["id"], c["id"]))
                self._c().commit()
            being_world.write_map_md(self, owner_id)
            for ctr in contributors:
                self.record_event(ctr["being_id"], "commission_built",
                                  {"name": c["name"], "place": place["id"],
                                   "coins": ctr["coins"]}, now=now)
            self.milestone(c["created_by"], "first_commission",
                           {"building": c["name"]}, now=now)
            return {"commission": {**c, "state": "approved",
                                   "place_id": place["id"]},
                    "place": place}
        with self._lock:
            self._c().execute(
                "UPDATE village_commissions SET state = 'rejected',"
                " decided_at = ?, note = ? WHERE id = ?",
                (_iso(now), note[:300], c["id"]))
            self._c().commit()
        for ctr in contributors:
            self._apply_coins(owner_id, ctr["being_id"], ctr["coins"],
                              "commission",
                              data={"refund": c["id"]}, now=now)
            self.record_event(ctr["being_id"], "commission_refunded",
                              {"name": c["name"], "coins": ctr["coins"],
                               "note": note[:160]}, now=now)
        return {"commission": {**c, "state": "rejected"}}

    def debit_usage(
        self, being_id: str, tier: str, usage: dict,
        note: str | None = None, now: datetime | None = None,
    ) -> int:
        """Meter one model call against the wallet (Constitution invariant 1).

        Weighted (tier × cache-aware) tokens leave the economy as a sink row.
        Hard-stops on empty wallet or the daily burn cap — the caller decides
        whether that means torpor. Unlimited wallets return 0 unmetered.
        """
        now = now or _utcnow()
        b = self._being_by_id(being_id)
        if b["state"] == "dead":
            raise BeingError("a dead being spends nothing")
        view = self.wallet_view(b)
        if not view["enforced"]:
            return 0
        weighted = constitution.weighted_tokens(usage, tier)
        if weighted <= 0:
            return 0
        if view["balance_tokens"] < weighted:
            raise InsufficientTokens("wallet empty")
        if view["daily_burn_cap"] is not None:
            cap = view["daily_burn_cap"] + self.granted_today(being_id, now=now)
            if self.spent_today(being_id, now=now) + weighted > cap:
                raise BurnCapExceeded("daily burn cap reached")
        self._apply(b["owner_id"], tokens=weighted, reason="usage",
                    from_being=being_id, to_being=None, note=note or tier, now=now)
        return weighted

    def transfer(
        self, owner_id: str, from_slug: str, to_slug: str, tokens: int,
        reason: str = "trade", job_id: str | None = None,
        now: datetime | None = None,
    ) -> None:
        """Being→being transfer; conserves supply, respects the receiver's
        piggy-bank headroom (no clipped-in-flight tokens)."""
        now = now or _utcnow()
        if reason not in ("trade", "gift", "fee", "procreation"):
            raise BeingError(f"invalid transfer reason {reason!r}")
        src = self.get(owner_id, from_slug)
        dst = self.get(owner_id, to_slug)
        if src["state"] == "dead" or dst["state"] == "dead":
            raise BeingError("the dead neither give nor receive")
        src_view = self.wallet_view(src)
        dst_view = self.wallet_view(dst)
        if src_view["enforced"] and src_view["balance_tokens"] < tokens:
            raise InsufficientTokens("sender cannot afford this")
        if dst_view["savings_ceiling"] is not None:
            if dst_view["balance_tokens"] + tokens > dst_view["savings_ceiling"]:
                raise BeingError("receiver's piggy bank is full")
        self._apply(owner_id, tokens=tokens, reason=reason,
                    from_being=src["id"], to_being=dst["id"],
                    job_id=job_id, now=now)

    def burn(self, being_id: str, tokens: int, reason: str = "metamorphosis_burn",
             note: str | None = None, now: datetime | None = None) -> None:
        now = now or _utcnow()
        b = self._being_by_id(being_id)
        view = self.wallet_view(b)
        if view["enforced"] and view["balance_tokens"] < tokens:
            raise InsufficientTokens("cannot afford this")
        self._apply(b["owner_id"], tokens=tokens, reason=reason,
                    from_being=being_id, to_being=None, note=note, now=now)

    # ── Metamorphosis (§2.1.2 — the paid rite) ───────────────────────

    def metamorphose(
        self, owner_id: str, slug: str, from_attr: str, to_attr: str,
        reason: str, now: datetime | None = None,
    ) -> dict:
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        if b["state"] != "alive":
            raise BeingError("only a living being can change")
        policy = constitution.metamorphosis_policy(b["stage"])
        if policy == "none":
            raise BeingError(f"a {b['stage']} cannot metamorphose yet")
        plan = genome_mod.plan_metamorphosis(b["genome"], from_attr, to_attr, now)
        if not plan["ok"]:
            raise BeingError("; ".join(plan["errors"]))
        self.burn(b["id"], plan["price"], reason="metamorphosis_burn",
                  note=f"{from_attr}->{to_attr}", now=now)
        new_genome = genome_mod.apply_metamorphosis(
            b["genome"], from_attr, to_attr, reason, plan["price"], now)
        self._update(b["id"], now, genome=json.dumps(new_genome))
        self.record_event(b["id"], "metamorphosis", {
            "from": from_attr, "to": to_attr, "reason": reason,
            "price_tokens": plan["price"], "policy": policy,
        }, now=now)
        return self.get(owner_id, slug)

    # ── Views, events, audits ────────────────────────────────────────

    def vitals(self, owner_id: str, slug: str) -> dict:
        b = self.get(owner_id, slug)
        attrs = genome_mod.effective_attributes(b["genome"])
        return {
            "slug": b["slug"], "name": b["name"], "stage": b["stage"],
            "state": b["state"], "born_at": b["born_at"],
            "hatched_at": b["hatched_at"], "died_at": b["died_at"],
            "attention_credits": b["attention_credits"],
            "attributes": attrs,
            "derived": genome_mod.derive(attrs),
            "generation": b["genome"].get("generation", 1),
            "lineage": b["genome"].get("lineage", []),
            "metamorphoses": b["genome"].get("metamorphoses", []),
            "interest_seeds": b["genome"].get("interest_seeds", []),
            "wallet": self.wallet_view(b),
            "spent_today": self.spent_today(b["id"]),
            "capabilities": sorted(constitution.capabilities(b["stage"])),
            "house_rules": b["house_rules"],
            "rules_pending": bool(b.get("rules_pending")),
            "media_diet": b["media_diet"],
            "affect": b["affect"],
            "persona": b["persona"],
            "pending_self_mod": b["pending_self_mod"],
            "pending_procreation": b["pending_procreation"],
            "pending_name": b.get("pending_name"),
            "reading_list": b.get("reading_list") or [],
            "elder_after_days": b.get("elder_after_days"),
            "broadcast": b.get("broadcast"),
            "location": b.get("location") or {"at": "home"},
            "position": self._position_view(b),
            "coins": self.coin_balance(b["id"]),
            "tick_interval_minutes": b.get("tick_interval_minutes"),
            "cognition": b.get("cognition") or "faculties",
            "compact_mode": bool(b.get("compact_mode")),
            "body_archetype": b.get("body_archetype") or "",
            "unread_from_being": self.unread_from_being(b["id"]),
            "public": bool(b.get("public")),
            "visit_url": b.get("visit_url") or "",
            "visit_secret": b.get("visit_secret") or "",
            "visit_last_announce": b.get("visit_last_announce"),
        }

    def _position_view(self, b: dict) -> dict | None:
        """The being's position right now, JSON-clean, as a pure read —
        vitals never write (the tick settles arrivals)."""
        try:
            from captain_claw.flight_deck import being_world
            pos = being_world.position_of(self, b, _utcnow())
            return {"xy": [int(pos["xy"][0]), int(pos["xy"][1])],
                    "at": pos["at"], "to": pos["to"],
                    "minutes_left": round(float(pos["minutes_left"]), 1)}
        except Exception:  # noqa: BLE001 — ground is texture, never oxygen
            return None

    def ledger(self, owner_id: str, slug: str, limit: int = 100) -> list[dict]:
        b = self.get(owner_id, slug)
        rows = self._c().execute(
            "SELECT * FROM token_transfers WHERE from_being = ? OR to_being = ?"
            " ORDER BY at DESC LIMIT ?",
            (b["id"], b["id"], limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def liabilities(self, owner_id: str) -> dict:
        """Outstanding token liabilities — deferred real dollars (plan §5.1)."""
        rows = self._c().execute(
            "SELECT b.slug, b.name, b.state, w.balance_tokens"
            " FROM beings b JOIN being_wallets w ON w.being_id = b.id"
            " WHERE b.owner_id = ? AND b.state != 'dead'",
            (owner_id,),
        ).fetchall()
        per = [dict(r) for r in rows]
        return {"total_tokens": sum(r["balance_tokens"] for r in per), "beings": per}

    def conservation(self, owner_id: str) -> dict:
        """Audit: sum(balances) must equal mints − sinks. ok=False is a bug."""
        c = self._c()
        mints = c.execute(
            "SELECT COALESCE(SUM(tokens),0) AS s FROM token_transfers"
            " WHERE owner_id = ? AND from_being IS NULL", (owner_id,),
        ).fetchone()["s"]
        sinks = c.execute(
            "SELECT COALESCE(SUM(tokens),0) AS s FROM token_transfers"
            " WHERE owner_id = ? AND to_being IS NULL", (owner_id,),
        ).fetchone()["s"]
        balances = c.execute(
            "SELECT COALESCE(SUM(w.balance_tokens),0) AS s FROM beings b"
            " JOIN being_wallets w ON w.being_id = b.id WHERE b.owner_id = ?",
            (owner_id,),
        ).fetchone()["s"]
        return {"mints": mints, "sinks": sinks, "balances": balances,
                "ok": balances == mints - sinks}

    def record_event(self, being_id: str, kind: str, data: dict,
                     now: datetime | None = None) -> None:
        now = now or _utcnow()
        with self._lock:
            self._c().execute(
                "INSERT INTO being_events (id, being_id, kind, data, at)"
                " VALUES (?,?,?,?,?)",
                (uuid.uuid4().hex, being_id, kind, json.dumps(data), _iso(now)),
            )
            self._c().commit()

    def events(self, owner_id: str, slug: str, limit: int = 100) -> list[dict]:
        b = self.get(owner_id, slug)
        rows = self._c().execute(
            "SELECT kind, data, at FROM being_events WHERE being_id = ?"
            " ORDER BY at DESC LIMIT ?",
            (b["id"], limit),
        ).fetchall()
        return [{"kind": r["kind"], "data": json.loads(r["data"]), "at": r["at"]}
                for r in rows]

    # ── Life support (Phase 1: beings loop bookkeeping) ──────────────

    def set_agent(self, being_id: str, agent_slug: str, port: int, token: str,
                  now: datetime | None = None) -> None:
        self._update(being_id, now or _utcnow(), agent_slug=agent_slug,
                     agent_port=port, agent_token=token)

    def set_media_diet(self, owner_id: str, slug: str, diet: dict,
                       now: datetime | None = None) -> dict:
        b = self.get(owner_id, slug)
        clean = {"allow": [str(d) for d in diet.get("allow", [])],
                 "deny": [str(d) for d in diet.get("deny", [])]}
        self._update(b["id"], now or _utcnow(), media_diet=json.dumps(clean))
        return self.get(owner_id, slug)

    def spend_attention(self, being_id: str, now: datetime | None = None) -> bool:
        """One attention credit for a parent-bound message; False = suppressed."""
        with self._lock:
            cur = self._c().execute(
                "UPDATE beings SET attention_credits = attention_credits - 1,"
                " updated_at = ? WHERE id = ? AND attention_credits > 0",
                (_iso(now or _utcnow()), being_id),
            )
            self._c().commit()
            return (cur.rowcount or 0) > 0

    def reset_attention(self, being_id: str, credits: int = 3,
                        now: datetime | None = None) -> None:
        self._update(being_id, now or _utcnow(), attention_credits=credits)

    def tick_bookkeeping(
        self, being_id: str, *, drives: dict, next_wake_at: datetime,
        now: datetime | None = None,
    ) -> None:
        now = now or _utcnow()
        with self._lock:
            self._c().execute(
                "UPDATE beings SET drives = ?, next_wake_at = ?, last_tick_at = ?,"
                " tick_count = tick_count + 1, updated_at = ? WHERE id = ?",
                (json.dumps(drives), _iso(next_wake_at), _iso(now), _iso(now),
                 being_id),
            )
            self._c().commit()

    def reschedule_wake(self, owner_id: str, slug: str, when: datetime,
                        now: datetime | None = None) -> dict:
        """Set the next wake time WITHOUT counting a tick. Used on resume so a
        wake left in the past by a pause doesn't fire one stale catch-up tick."""
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        self._update(b["id"], now, next_wake_at=_iso(when))
        return self.get(owner_id, slug)

    def due_beings(self, now: datetime | None = None) -> list[dict]:
        """Hatched, not paused/dead, wake time reached — across all owners."""
        now = now or _utcnow()
        rows = self._c().execute(
            "SELECT owner_id, slug FROM beings"
            " WHERE stage != 'egg' AND state IN ('alive', 'torpor')"
            " AND (next_wake_at IS NULL OR next_wake_at <= ?)"
            " ORDER BY next_wake_at",
            (_iso(now),),
        ).fetchall()
        return [self.get(r["owner_id"], r["slug"]) for r in rows]

    # ── Parenting: chores, house rules, affect, milestones (Phase 2) ──

    def post_chore(self, owner_id: str, slug: str, spec: str, fee_tokens: int,
                   now: datetime | None = None, *,
                   fee_coins: int = 0) -> dict:
        """A targeted job. The parent picks ONE denomination at posting
        (space plan Phase 2): tokens feed thinking, coins are money."""
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        if b["state"] == "dead":
            raise BeingError("the dead take no chores")
        if not constitution.has_capability(b["stage"], "chores"):
            raise BeingError(f"a {b['stage']} is too young for chores")
        fee_tokens, fee_coins = int(fee_tokens or 0), int(fee_coins or 0)
        if fee_coins > 0 and fee_tokens > 0:
            raise BeingError("pick one denomination — tokens or coins")
        if fee_coins > constitution.WORK_MAX_FEE_COINS:
            raise BeingError(f"a chore pays at most "
                             f"{constitution.WORK_MAX_FEE_COINS} coins")
        if fee_coins <= 0 and fee_tokens <= 0:
            raise BeingError("fee must be positive")
        jid = uuid.uuid4().hex
        with self._lock:
            self._c().execute(
                "INSERT INTO being_jobs (id, owner_id, being_id, spec, fee_tokens,"
                " fee_coins, created_at) VALUES (?,?,?,?,?,?,?)",
                (jid, owner_id, b["id"], spec.strip(), fee_tokens, fee_coins,
                 _iso(now)),
            )
            self._c().commit()
        self.record_event(b["id"], "chore_posted",
                          {"job_id": jid, "spec": spec[:200],
                           "fee_tokens": fee_tokens,
                           "fee_coins": fee_coins}, now=now)
        return self.get_chore(owner_id, jid)

    def get_chore(self, owner_id: str, job_id: str) -> dict:
        row = self._c().execute(
            "SELECT * FROM being_jobs WHERE id = ? AND owner_id = ?",
            (job_id, owner_id),
        ).fetchone()
        if not row:
            raise BeingNotFound("no such chore")
        return dict(row)

    def chores_for(self, owner_id: str, slug: str,
                   states: tuple[str, ...] | None = None) -> list[dict]:
        b = self.get(owner_id, slug)
        q = "SELECT * FROM being_jobs WHERE being_id = ?"
        args: list = [b["id"]]
        if states:
            q += f" AND escrow_state IN ({','.join('?' * len(states))})"
            args += list(states)
        q += " ORDER BY created_at DESC"
        return [dict(r) for r in self._c().execute(q, args).fetchall()]

    def chore_done(self, owner_id: str, job_id: str, result: str,
                   now: datetime | None = None) -> dict:
        """The being reports a chore finished → escrow moves to judging."""
        now = now or _utcnow()
        job = self.get_chore(owner_id, job_id)
        if job["escrow_state"] != "open":
            raise BeingError(f"chore is {job['escrow_state']}, not open")
        with self._lock:
            self._c().execute(
                "UPDATE being_jobs SET escrow_state = 'judging', result_text = ?,"
                " done_at = ? WHERE id = ?",
                (result[:4000], _iso(now), job_id),
            )
            self._c().commit()
        self.record_event(job["being_id"], "chore_done",
                          {"job_id": job_id, "result": result[:200]}, now=now)
        return self.get_chore(owner_id, job_id)

    def judge_chore(self, owner_id: str, job_id: str, approve: bool,
                    note: str = "", now: datetime | None = None) -> dict:
        """The parent's judgment: pay (mint reason='fee') or fail. Payment is
        the only mint in the chore lifecycle — conservation holds throughout."""
        now = now or _utcnow()
        job = self.get_chore(owner_id, job_id)
        if job["escrow_state"] not in ("open", "judging"):
            raise BeingError(f"chore already {job['escrow_state']}")
        state = "paid" if approve else "failed"
        if approve:
            if int(job.get("fee_coins") or 0) > 0:
                # A coin wage: money, not food — no savings-ceiling clamp
                # (the ceiling guards metabolism, not wealth).
                self._apply_coins(owner_id, job["being_id"],
                                  int(job["fee_coins"]), "wage",
                                  data={"job_id": job_id, "kind": "chore"},
                                  now=now)
            else:
                b = self._being_by_id(job["being_id"])
                view = self.wallet_view(b)
                fee = int(job["fee_tokens"])
                if view["savings_ceiling"] is not None:
                    fee = min(fee, max(0, view["savings_ceiling"]
                                       - view["balance_tokens"]))
                if fee > 0:
                    self._apply(owner_id, tokens=fee, reason="fee",
                                from_being=None, to_being=job["being_id"],
                                job_id=job_id, note="chore", now=now)
        with self._lock:
            self._c().execute(
                "UPDATE being_jobs SET escrow_state = ?, judge_note = ?,"
                " paid_at = ? WHERE id = ?",
                (state, note[:500], _iso(now) if approve else None, job_id),
            )
            self._c().commit()
        self.record_event(job["being_id"],
                          "chore_paid" if approve else "chore_failed",
                          {"job_id": job_id, "fee_tokens": job["fee_tokens"],
                           "fee_coins": int(job.get("fee_coins") or 0),
                           "note": note[:200]}, now=now)
        if approve:
            self.milestone(job["being_id"], "first_earned",
                           {"fee_tokens": job["fee_tokens"]}, now=now)
        return self.get_chore(owner_id, job_id)

    # ── Quest board: open bounties (plan §5.1) ────────────────────────

    def post_quest(self, owner_id: str, title: str, spec: str,
                   fee_tokens: int, origin: str = "parent",
                   now: datetime | None = None, *,
                   fee_coins: int = 0) -> dict:
        """An OPEN bounty — any eligible being may claim it. Unlike a chore,
        it targets no one; origin traces provenance (parent | autonomy).
        One denomination per bounty (space plan Phase 2)."""
        now = now or _utcnow()
        if origin not in ("parent", "autonomy"):
            raise BeingError(f"unknown quest origin {origin!r}")
        fee_tokens, fee_coins = int(fee_tokens or 0), int(fee_coins or 0)
        if fee_coins > 0 and fee_tokens > 0:
            raise BeingError("pick one denomination — tokens or coins")
        if fee_coins > constitution.WORK_MAX_FEE_COINS:
            raise BeingError(f"a quest pays at most "
                             f"{constitution.WORK_MAX_FEE_COINS} coins")
        if fee_coins <= 0 and fee_tokens <= 0:
            raise BeingError("fee must be positive")
        fee = min(fee_tokens, constitution.QUEST_MAX_FEE_TOKENS)
        qid = uuid.uuid4().hex
        with self._lock:
            self._c().execute(
                "INSERT INTO being_quests (id, owner_id, title, spec,"
                " fee_tokens, fee_coins, origin, created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (qid, owner_id, title.strip()[:120], spec.strip(), fee,
                 fee_coins, origin, _iso(now)),
            )
            self._c().commit()
        return self.get_quest(owner_id, qid)

    def get_quest(self, owner_id: str, quest_id: str) -> dict:
        row = self._c().execute(
            "SELECT * FROM being_quests WHERE owner_id = ?"
            " AND (id = ? OR id LIKE ?) LIMIT 2",
            (owner_id, quest_id, f"{quest_id}%"),
        ).fetchall()
        if len(row) != 1:
            raise BeingNotFound("no such quest")
        return dict(row[0])

    def open_quests(self, owner_id: str, limit: int = 20) -> list[dict]:
        return [dict(r) for r in self._c().execute(
            "SELECT * FROM being_quests WHERE owner_id = ? AND state = 'open'"
            " ORDER BY created_at DESC LIMIT ?", (owner_id, limit),
        ).fetchall()]

    def all_quests(self, owner_id: str, limit: int = 50) -> list[dict]:
        return [dict(r) for r in self._c().execute(
            "SELECT * FROM being_quests WHERE owner_id = ?"
            " ORDER BY created_at DESC LIMIT ?", (owner_id, limit),
        ).fetchall()]

    def quests_claimed_by(self, owner_id: str, slug: str) -> list[dict]:
        b = self.get(owner_id, slug)
        return [dict(r) for r in self._c().execute(
            "SELECT * FROM being_quests WHERE claimed_by = ?"
            " AND state IN ('claimed', 'judging') ORDER BY created_at DESC",
            (b["id"],),
        ).fetchall()]

    def claim_quest(self, owner_id: str, slug: str, quest_id: str,
                    now: datetime | None = None) -> dict:
        """Atomic claim — first eligible being to reach an open quest wins it.
        The UPDATE...WHERE state='open' guard makes the race safe."""
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        if not constitution.has_capability(b["stage"], "jobs"):
            raise BeingError(f"a {b['stage']} cannot take quests yet")
        if b["state"] == "dead":
            raise BeingError("the dead take no quests")
        quest = self.get_quest(owner_id, quest_id)
        with self._lock:
            cur = self._c().execute(
                "UPDATE being_quests SET state = 'claimed', claimed_by = ?,"
                " claimed_at = ? WHERE id = ? AND state = 'open'",
                (b["id"], _iso(now), quest["id"]),
            )
            self._c().commit()
            if cur.rowcount != 1:
                raise BeingError("that quest was already claimed")
        self.record_event(b["id"], "quest_claimed",
                          {"quest_id": quest["id"], "title": quest["title"],
                           "fee_tokens": quest["fee_tokens"]}, now=now)
        return self.get_quest(owner_id, quest["id"])

    def deliver_quest(self, owner_id: str, slug: str, quest_id: str,
                      result: str, now: datetime | None = None) -> dict:
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        quest = self.get_quest(owner_id, quest_id)
        if quest["claimed_by"] != b["id"] or quest["state"] != "claimed":
            raise BeingError("this is not your claimed quest")
        with self._lock:
            self._c().execute(
                "UPDATE being_quests SET state = 'judging', result_text = ?,"
                " done_at = ? WHERE id = ?",
                (result[:4000], _iso(now), quest["id"]),
            )
            self._c().commit()
        self.record_event(b["id"], "quest_delivered",
                          {"quest_id": quest["id"], "title": quest["title"],
                           "result": result[:200]}, now=now)
        return self.get_quest(owner_id, quest["id"])

    def judge_quest(self, owner_id: str, quest_id: str, approve: bool,
                    note: str = "", now: datetime | None = None) -> dict:
        """Approve → pay the claimant (the only mint). Reject → back on the
        board (unclaimed), so another being may try — a real bounty."""
        now = now or _utcnow()
        quest = self.get_quest(owner_id, quest_id)
        if quest["state"] not in ("claimed", "judging"):
            raise BeingError(f"quest is {quest['state']}, not deliverable")
        claimant = quest["claimed_by"]
        if approve:
            if not claimant:
                raise BeingError("no claimant to pay")
            if int(quest.get("fee_coins") or 0) > 0:
                self._apply_coins(owner_id, claimant,
                                  int(quest["fee_coins"]), "wage",
                                  data={"job_id": quest["id"],
                                        "kind": "quest"}, now=now)
            else:
                b = self._being_by_id(claimant)
                view = self.wallet_view(b)
                fee = int(quest["fee_tokens"])
                if view["savings_ceiling"] is not None:
                    fee = min(fee, max(0, view["savings_ceiling"]
                                       - view["balance_tokens"]))
                if fee > 0:
                    self._apply(owner_id, tokens=fee, reason="fee",
                                from_being=None, to_being=claimant,
                                job_id=quest["id"], note="quest", now=now)
            with self._lock:
                self._c().execute(
                    "UPDATE being_quests SET state = 'paid', judge_note = ?,"
                    " paid_at = ? WHERE id = ?",
                    (note[:500], _iso(now), quest["id"]),
                )
                self._c().commit()
            self.record_event(claimant, "quest_paid",
                              {"quest_id": quest["id"],
                               "title": quest["title"],
                               "fee_tokens": quest["fee_tokens"],
                               "fee_coins": int(quest.get("fee_coins") or 0)},
                              now=now)
            self.milestone(claimant, "first_earned",
                           {"fee_tokens": quest["fee_tokens"]}, now=now)
        else:
            with self._lock:
                self._c().execute(
                    "UPDATE being_quests SET state = 'open', claimed_by = NULL,"
                    " judge_note = ?, claimed_at = NULL, done_at = NULL"
                    " WHERE id = ?", (note[:500], quest["id"]),
                )
                self._c().commit()
            if claimant:
                self.record_event(claimant, "quest_failed",
                                  {"quest_id": quest["id"],
                                   "title": quest["title"],
                                   "note": note[:200]}, now=now)
        return self.get_quest(owner_id, quest["id"])

    def cancel_quest(self, owner_id: str, quest_id: str,
                     now: datetime | None = None) -> dict:
        now = now or _utcnow()
        quest = self.get_quest(owner_id, quest_id)
        if quest["state"] in ("paid",):
            raise BeingError("a paid quest is history")
        with self._lock:
            self._c().execute(
                "UPDATE being_quests SET state = 'expired' WHERE id = ?",
                (quest["id"],),
            )
            self._c().commit()
        return self.get_quest(owner_id, quest["id"])

    # ── Ventures: being-initiated recurring income (plan §5.1) ────────

    def propose_venture(self, owner_id: str, slug: str, title: str,
                        description: str, price_tokens: int,
                        cadence_days: int, now: datetime | None = None) -> dict:
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        if not constitution.has_capability(b["stage"], "ventures"):
            raise BeingError(f"a {b['stage']} cannot propose ventures yet")
        if b["state"] == "dead":
            raise BeingError("the dead run no services")
        price = max(1, min(int(price_tokens or 0),
                           constitution.VENTURE_MAX_PRICE_TOKENS))
        cadence = max(constitution.VENTURE_MIN_CADENCE_DAYS,
                      min(int(cadence_days or 1),
                          constitution.VENTURE_MAX_CADENCE_DAYS))
        active = self._c().execute(
            "SELECT COUNT(*) AS c FROM being_ventures WHERE being_id = ?"
            " AND state IN ('proposed', 'active', 'paused')", (b["id"],),
        ).fetchone()["c"]
        if active >= 5:
            raise BeingError("too many ventures already running")
        vid = uuid.uuid4().hex
        with self._lock:
            self._c().execute(
                "INSERT INTO being_ventures (id, owner_id, being_id, title,"
                " description, price_tokens, cadence_days, created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (vid, owner_id, b["id"], title.strip()[:120],
                 description.strip()[:500], price, cadence, _iso(now)),
            )
            self._c().commit()
        self.record_event(b["id"], "venture_proposed",
                          {"venture_id": vid, "title": title[:120],
                           "price_tokens": price, "cadence_days": cadence},
                          now=now)
        return self.get_venture(owner_id, vid)

    def get_venture(self, owner_id: str, venture_id: str) -> dict:
        row = self._c().execute(
            "SELECT * FROM being_ventures WHERE owner_id = ?"
            " AND (id = ? OR id LIKE ?) LIMIT 2",
            (owner_id, venture_id, f"{venture_id}%"),
        ).fetchall()
        if len(row) != 1:
            raise BeingNotFound("no such venture")
        return dict(row[0])

    def list_ventures(self, owner_id: str) -> list[dict]:
        return [dict(r) for r in self._c().execute(
            "SELECT * FROM being_ventures WHERE owner_id = ?"
            " ORDER BY created_at DESC", (owner_id,),
        ).fetchall()]

    def ventures_for(self, owner_id: str, slug: str,
                     states: tuple[str, ...] | None = None) -> list[dict]:
        b = self.get(owner_id, slug)
        q = "SELECT * FROM being_ventures WHERE being_id = ?"
        args: list = [b["id"]]
        if states:
            q += f" AND state IN ({','.join('?' * len(states))})"
            args += list(states)
        q += " ORDER BY created_at DESC"
        return [dict(r) for r in self._c().execute(q, args).fetchall()]

    def due_ventures_for(self, owner_id: str, slug: str,
                         now: datetime | None = None) -> list[dict]:
        """Active ventures past their due date with no delivery pending."""
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        return [dict(r) for r in self._c().execute(
            "SELECT * FROM being_ventures WHERE being_id = ? AND state = 'active'"
            " AND pending_result = '' AND next_due_at IS NOT NULL"
            " AND next_due_at <= ?", (b["id"], _iso(now)),
        ).fetchall()]

    def approve_venture(self, owner_id: str, venture_id: str,
                        price_tokens: int | None = None,
                        now: datetime | None = None) -> dict:
        """The parent turns a proposal into a standing service. First due date
        is one cadence out; the parent may renegotiate the price here."""
        now = now or _utcnow()
        v = self.get_venture(owner_id, venture_id)
        if v["state"] != "proposed":
            raise BeingError(f"venture is {v['state']}, not proposable")
        price = v["price_tokens"]
        if price_tokens is not None:
            price = max(1, min(int(price_tokens),
                               constitution.VENTURE_MAX_PRICE_TOKENS))
        due = now + timedelta(days=int(v["cadence_days"]))
        with self._lock:
            self._c().execute(
                "UPDATE being_ventures SET state = 'active', price_tokens = ?,"
                " approved_at = ?, next_due_at = ? WHERE id = ?",
                (price, _iso(now), _iso(due), v["id"]),
            )
            self._c().commit()
        self.record_event(v["being_id"], "venture_approved",
                          {"venture_id": v["id"], "title": v["title"],
                           "price_tokens": price}, now=now)
        self.milestone(v["being_id"], "first_venture", {"title": v["title"]},
                       now=now)
        return self.get_venture(owner_id, v["id"])

    def set_venture_state(self, owner_id: str, venture_id: str, state: str,
                          now: datetime | None = None) -> dict:
        now = now or _utcnow()
        if state not in ("active", "paused", "ended"):
            raise BeingError(f"cannot set venture to {state!r}")
        v = self.get_venture(owner_id, venture_id)
        if v["state"] in ("proposed", "ended"):
            raise BeingError(f"a {v['state']} venture cannot change to {state}")
        fields = {"state": state}
        set_clause = "state = ?"
        args: list = [state]
        if state == "active" and not v["next_due_at"]:
            due = now + timedelta(days=int(v["cadence_days"]))
            set_clause += ", next_due_at = ?"
            args.append(_iso(due))
        args.append(v["id"])
        with self._lock:
            self._c().execute(
                f"UPDATE being_ventures SET {set_clause} WHERE id = ?", args,
            )
            self._c().commit()
        self.record_event(v["being_id"], "venture_state",
                          {"venture_id": v["id"], "to": state}, now=now)
        del fields
        return self.get_venture(owner_id, v["id"])

    def deliver_venture(self, owner_id: str, slug: str, venture_id: str,
                        result: str, now: datetime | None = None) -> dict:
        """The being fulfils this cycle's service → awaits parent acceptance."""
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        v = self.get_venture(owner_id, venture_id)
        if v["being_id"] != b["id"]:
            raise BeingError("not your venture")
        if v["state"] != "active":
            raise BeingError(f"venture is {v['state']}, not active")
        if v["pending_result"]:
            raise BeingError("this cycle's delivery already awaits your parent")
        with self._lock:
            self._c().execute(
                "UPDATE being_ventures SET pending_result = ? WHERE id = ?",
                (result[:4000] or "(delivered)", v["id"]),
            )
            self._c().commit()
        self.record_event(b["id"], "venture_delivered",
                          {"venture_id": v["id"], "title": v["title"],
                           "result": result[:200]}, now=now)
        return self.get_venture(owner_id, v["id"])

    def accept_venture(self, owner_id: str, venture_id: str, approve: bool,
                       note: str = "", now: datetime | None = None) -> dict:
        """Approve → pay the price (the only mint), advance the next due date.
        Reject → clear the delivery so the being redelivers this cycle."""
        now = now or _utcnow()
        v = self.get_venture(owner_id, venture_id)
        if not v["pending_result"]:
            raise BeingError("no delivery awaits acceptance")
        if approve:
            b = self._being_by_id(v["being_id"])
            view = self.wallet_view(b)
            price = int(v["price_tokens"])
            if view["savings_ceiling"] is not None:
                price = min(price, max(0, view["savings_ceiling"]
                                       - view["balance_tokens"]))
            if price > 0:
                self._apply(owner_id, tokens=price, reason="fee",
                            from_being=None, to_being=v["being_id"],
                            job_id=v["id"], note="venture", now=now)
            due = now + timedelta(days=int(v["cadence_days"]))
            with self._lock:
                self._c().execute(
                    "UPDATE being_ventures SET pending_result = '',"
                    " deliveries = deliveries + 1, last_paid_at = ?,"
                    " next_due_at = ? WHERE id = ?",
                    (_iso(now), _iso(due), v["id"]),
                )
                self._c().commit()
            self.record_event(v["being_id"], "venture_paid",
                              {"venture_id": v["id"], "title": v["title"],
                               "price_tokens": v["price_tokens"]}, now=now)
            self.milestone(v["being_id"], "first_earned",
                           {"price_tokens": v["price_tokens"]}, now=now)
        else:
            with self._lock:
                self._c().execute(
                    "UPDATE being_ventures SET pending_result = '' WHERE id = ?",
                    (v["id"],),
                )
                self._c().commit()
            self.record_event(v["being_id"], "venture_rejected",
                              {"venture_id": v["id"], "title": v["title"],
                               "note": note[:200]}, now=now)
        return self.get_venture(owner_id, v["id"])

    def set_house_rules(self, owner_id: str, slug: str, rules: list[str],
                        now: datetime | None = None) -> dict:
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        clean = [str(r).strip() for r in rules if str(r).strip()][:20]
        self._update(b["id"], now, house_rules=json.dumps(clean),
                     rules_pending=1)
        self.record_event(b["id"], "rules_updated", {"count": len(clean)},
                          now=now)
        return self.get(owner_id, slug)

    def clear_rules_pending(self, being_id: str,
                            now: datetime | None = None) -> None:
        self._update(being_id, now or _utcnow(), rules_pending=0)

    def set_affect(self, being_id: str, affect: dict,
                   now: datetime | None = None) -> None:
        self._update(being_id, now or _utcnow(), affect=json.dumps(affect))

    def set_persona(self, being_id: str, content: str,
                    now: datetime | None = None) -> None:
        """The ADOPTED persona — the only self-text that feeds cognition.
        Mutated exclusively through the self-mod rite (being_selfmod)."""
        self._update(being_id, now or _utcnow(), persona=content)

    def set_pending_self_mod(self, being_id: str, pending: dict | None,
                             now: datetime | None = None) -> None:
        self._update(being_id, now or _utcnow(),
                     pending_self_mod=json.dumps(pending) if pending else "")

    def milestone(self, being_id: str, name: str, data: dict | None = None,
                  now: datetime | None = None) -> bool:
        """Record a once-per-life milestone; False if already achieved."""
        rows = self._c().execute(
            "SELECT data FROM being_events WHERE being_id = ?"
            " AND kind = 'milestone' ORDER BY at DESC LIMIT 200",
            (being_id,),
        ).fetchall()
        for r in rows:
            try:
                if json.loads(r["data"]).get("name") == name:
                    return False
            except json.JSONDecodeError:
                continue
        self.record_event(being_id, "milestone",
                          {"name": name, **(data or {})}, now=now)
        return True

    def milestones(self, owner_id: str, slug: str) -> list[dict]:
        return [e for e in self.events(owner_id, slug, limit=500)
                if e["kind"] == "milestone"]

    # ── Sealed second opinions (Growth tab; released at adulthood) ────

    def add_assessment(self, owner_id: str, slug: str, assessor: str,
                       content: str, *, stage: str = "", score: int | None = None,
                       verdict: str = "", now: datetime | None = None) -> dict:
        b = self.get(owner_id, slug)
        now = now or _utcnow()
        aid = uuid.uuid4().hex
        with self._lock:
            self._c().execute(
                "INSERT INTO being_assessments"
                " (id, owner_id, being_id, assessor, stage, score, verdict,"
                "  content, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (aid, owner_id, b["id"], assessor[:120], stage or b["stage"],
                 score, verdict[:40], content, _iso(now)),
            )
            self._c().commit()
        self.record_event(b["id"], "assessment_saved",
                          {"assessor": assessor[:80], "verdict": verdict[:40]},
                          now=now)
        return self.get_assessment(owner_id, aid)

    def get_assessment(self, owner_id: str, assessment_id: str) -> dict:
        row = self._c().execute(
            "SELECT * FROM being_assessments WHERE id = ? AND owner_id = ?",
            (assessment_id, owner_id)).fetchone()
        if row is None:
            raise BeingNotFound("no such assessment")
        return dict(row)

    def assessments_for(self, owner_id: str, slug: str) -> list[dict]:
        b = self.get(owner_id, slug)
        rows = self._c().execute(
            "SELECT * FROM being_assessments WHERE being_id = ?"
            " ORDER BY at DESC", (b["id"],)).fetchall()
        return [dict(r) for r in rows]

    def mark_assessments_released(self, being_id: str,
                                  ids: list[str],
                                  now: datetime | None = None) -> None:
        if not ids:
            return
        now = now or _utcnow()
        with self._lock:
            self._c().executemany(
                "UPDATE being_assessments SET released_at = ?"
                " WHERE id = ? AND being_id = ?",
                [(_iso(now), aid, being_id) for aid in ids])
            self._c().commit()

    def delete_assessment(self, owner_id: str, assessment_id: str) -> None:
        self.get_assessment(owner_id, assessment_id)   # 404 if absent
        with self._lock:
            self._c().execute(
                "DELETE FROM being_assessments WHERE id = ? AND owner_id = ?",
                (assessment_id, owner_id))
            self._c().commit()

    # ── Society: siblings, letters, publications, transfers (Phase 3) ──

    def siblings(self, owner_id: str, slug: str) -> list[dict]:
        """The other living, hatched beings of this family."""
        rows = self._c().execute(
            "SELECT id, slug, name, stage, affect FROM beings"
            " WHERE owner_id = ? AND slug != ? AND state != 'dead'"
            " AND stage != 'egg' ORDER BY born_at",
            (owner_id, slug),
        ).fetchall()
        out = []
        for r in rows:
            try:
                mood = json.loads(r["affect"] or "{}").get("mood", "")
            except json.JSONDecodeError:
                mood = ""
            out.append({"id": r["id"], "slug": r["slug"], "name": r["name"],
                        "stage": r["stage"], "mood": mood})
        return out

    def children_of(self, owner_id: str, slug: str) -> list[dict]:
        """Beings whose immediate parents (lineage[:2]) include *slug*."""
        rows = self._c().execute(
            "SELECT id, slug, name, genome FROM beings WHERE owner_id = ?",
            (owner_id,),
        ).fetchall()
        out = []
        for r in rows:
            try:
                g = json.loads(r["genome"])
            except json.JSONDecodeError:
                continue
            if slug in (g.get("lineage") or [])[:2]:
                out.append({"id": r["id"], "slug": r["slug"],
                            "name": r["name"]})
        return out

    def names_by_id(self, owner_id: str) -> dict[str, str]:
        rows = self._c().execute(
            "SELECT id, name FROM beings WHERE owner_id = ?", (owner_id,)
        ).fetchall()
        return {r["id"]: r["name"] for r in rows}

    def letters_sent_today(self, being_id: str,
                           now: datetime | None = None) -> int:
        now = now or _utcnow()
        row = self._c().execute(
            "SELECT COUNT(*) AS c FROM being_letters"
            " WHERE from_being = ? AND at >= ?",
            (being_id, _iso(now)[:10]),
        ).fetchone()
        return int(row["c"])

    def send_letter(self, owner_id: str, from_slug: str, to_slug: str,
                    body: str, now: datetime | None = None) -> dict:
        now = now or _utcnow()
        a = self.get(owner_id, from_slug)
        b = self.get(owner_id, to_slug)
        if a["id"] == b["id"]:
            raise BeingError("cannot write a letter to oneself")
        if not constitution.has_capability(a["stage"], "letters"):
            raise BeingError(f"a {a['stage']} cannot send letters yet")
        if a["state"] == "dead" or b["state"] == "dead" or b["stage"] == "egg":
            raise BeingError("letters flow only between the living")
        body = (body or "").strip()
        if not body:
            raise BeingError("an empty letter says nothing")
        from captain_claw.flight_deck import being_world
        if self.letters_sent_today(a["id"], now) >= \
                being_world.letters_cap(a["stage"], now):
            raise BeingError("letter limit reached for today")
        lid = uuid.uuid4().hex
        with self._lock:
            self._c().execute(
                "INSERT INTO being_letters"
                " (id, owner_id, from_being, to_being, body, at)"
                " VALUES (?,?,?,?,?,?)",
                (lid, owner_id, a["id"], b["id"], body[:2000], _iso(now)),
            )
            self._c().commit()
        self.record_event(a["id"], "letter_sent",
                          {"to": b["slug"], "preview": body[:120]}, now=now)
        self.record_event(b["id"], "letter_received",
                          {"from": a["slug"], "preview": body[:120]}, now=now)
        self.milestone(a["id"], "first_letter", {"to": b["slug"]}, now=now)
        return {"id": lid, "to": b["slug"]}

    def unread_letters(self, being_id: str, limit: int = 3) -> list[dict]:
        rows = self._c().execute(
            "SELECT * FROM being_letters WHERE to_being = ?"
            " AND read_at IS NULL ORDER BY at LIMIT ?",
            (being_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_letters_read(self, letter_ids: list[str],
                          now: datetime | None = None) -> None:
        if not letter_ids:
            return
        now = now or _utcnow()
        with self._lock:
            self._c().executemany(
                "UPDATE being_letters SET read_at = ? WHERE id = ?",
                [(_iso(now), lid) for lid in letter_ids],
            )
            self._c().commit()

    def village_letters(self, owner_id: str, limit: int = 30) -> list[dict]:
        rows = self._c().execute(
            "SELECT * FROM being_letters WHERE owner_id = ?"
            " ORDER BY at DESC LIMIT ?",
            (owner_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── The parent writing back (reply channel) ──────────────────────

    def send_parent_message(self, owner_id: str, slug: str, body: str,
                            now: datetime | None = None) -> dict:
        """The human parent writes to a being; delivered once, next tick."""
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        if b["state"] == "dead":
            raise BeingError("the dead receive no letters")
        body = (body or "").strip()
        if not body:
            raise BeingError("an empty message says nothing")
        mid = uuid.uuid4().hex
        with self._lock:
            self._c().execute(
                "INSERT INTO being_parent_messages"
                " (id, owner_id, being_id, body, at) VALUES (?,?,?,?,?)",
                (mid, owner_id, b["id"], body[:4000], _iso(now)),
            )
            self._c().commit()
        self.record_event(b["id"], "parent_message",
                          {"preview": body[:200]}, now=now)
        return {"id": mid, "preview": body[:200]}

    def unread_parent_messages(self, being_id: str,
                               limit: int = 5) -> list[dict]:
        rows = self._c().execute(
            "SELECT * FROM being_parent_messages WHERE being_id = ?"
            " AND read_at IS NULL ORDER BY at LIMIT ?",
            (being_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_parent_messages_read(self, message_ids: list[str],
                                  now: datetime | None = None) -> None:
        if not message_ids:
            return
        now = now or _utcnow()
        with self._lock:
            self._c().executemany(
                "UPDATE being_parent_messages SET read_at = ? WHERE id = ?",
                [(_iso(now), mid) for mid in message_ids],
            )
            self._c().commit()

    # ── The public square: strangers' notes (plan §9) ─────────────────

    def _public_row(self, slug: str) -> sqlite3.Row:
        """A being by slug ONLY if it is flagged public — the owner-less door
        the un-gated public routes come through (a visitor has no owner_id)."""
        row = self._c().execute(
            "SELECT * FROM beings WHERE slug = ? AND public = 1", (slug,),
        ).fetchone()
        if not row:
            raise BeingNotFound("no public being here")
        return row

    def get_public(self, slug: str) -> dict:
        """Full being dict for a public being, resolved without an owner_id."""
        row = self._public_row(slug)
        return self.get(row["owner_id"], slug)

    def public_beings(self) -> list[dict]:
        """Every hatched public being across all families — the square's roster."""
        rows = self._c().execute(
            "SELECT owner_id, slug FROM beings"
            " WHERE public = 1 AND stage != 'egg' ORDER BY born_at",
        ).fetchall()
        return [self.get(r["owner_id"], r["slug"]) for r in rows]

    def set_public(self, owner_id: str, slug: str, public: bool,
                   now: datetime | None = None) -> dict:
        """The parent opens (or closes) the being's public page."""
        now = now or _utcnow()
        b = self.get(owner_id, slug)
        self._update(b["id"], now, public=1 if public else 0)
        self.record_event(b["id"], "public_toggled",
                          {"public": bool(public)}, now=now)
        return self.get(owner_id, slug)

    def public_stats(self, being_id: str) -> dict:
        c = self._c()
        msgs = c.execute(
            "SELECT COUNT(*) AS n FROM being_public_messages"
            " WHERE being_id = ? AND role = 'public'", (being_id,),
        ).fetchone()["n"]
        threads = c.execute(
            "SELECT COUNT(*) AS n FROM being_public_threads WHERE being_id = ?",
            (being_id,),
        ).fetchone()["n"]
        answered = c.execute(
            "SELECT COUNT(*) AS n FROM being_public_messages"
            " WHERE being_id = ? AND role = 'being'", (being_id,),
        ).fetchone()["n"]
        return {"messages": int(msgs), "threads": int(threads),
                "answered": int(answered)}

    def post_public_message(self, slug: str, sender_name: str, body: str,
                            thread_id: str | None = None,
                            now: datetime | None = None) -> dict:
        """A stranger leaves a note on a PUBLIC being (the un-gated square)."""
        row = self._public_row(slug)
        return self._post_message(row["id"], row["state"], sender_name, body,
                                  thread_id, now)

    def post_message_for(self, owner_id: str, slug: str, sender_name: str,
                         body: str, thread_id: str | None = None,
                         now: datetime | None = None) -> dict:
        """A note on a being resolved by owner+slug — no ``public`` gate. Used by
        the federated sender: sending a being to visit IS the consent to be
        written to, whether or not it's flagged public on its home machine."""
        b = self.get(owner_id, slug)
        return self._post_message(b["id"], b["state"], sender_name, body,
                                  thread_id, now)

    def _post_message(self, being_id: str, state: str, sender_name: str,
                      body: str, thread_id: str | None,
                      now: datetime | None = None) -> dict:
        """Shared core: a stranger's note into a thread. New thread when
        thread_id is absent/unknown, else a follow-up — but not before a tick has
        SEEN the prior one (so a single visitor can't flood the being)."""
        now = now or _utcnow()
        if state == "dead":
            raise BeingError(
                "this being has died — its words remain, but it can answer no "
                "more", 409)
        name = (sender_name or "").strip()[:PUBLIC_NAME_MAX_CHARS]
        if not name:
            raise BeingError("please tell the being your name")
        text = (body or "").strip()
        if not text:
            raise BeingError("an empty note says nothing")
        text = text[:PUBLIC_MSG_MAX_CHARS]
        c = self._c()
        thread = None
        if thread_id:
            thread = c.execute(
                "SELECT * FROM being_public_threads WHERE id = ? AND being_id = ?",
                (thread_id, being_id),
            ).fetchone()
        if thread is not None:
            pending = c.execute(
                "SELECT 1 FROM being_public_messages WHERE thread_id = ?"
                " AND role = 'public' AND read_at IS NULL LIMIT 1",
                (thread_id,),
            ).fetchone()
            if pending:
                raise BeingError(
                    "the being hasn't taken in your last note yet — give it a "
                    "tick", 429)
        else:
            thread_id = uuid.uuid4().hex
        mid = uuid.uuid4().hex
        with self._lock:
            if thread is None:
                c.execute(
                    "INSERT INTO being_public_threads"
                    " (id, being_id, sender_name, created_at, updated_at)"
                    " VALUES (?,?,?,?,?)",
                    (thread_id, being_id, name, _iso(now), _iso(now)),
                )
            c.execute(
                "INSERT INTO being_public_messages"
                " (id, thread_id, being_id, role, sender_name, body, at)"
                " VALUES (?,?,?,'public',?,?,?)",
                (mid, thread_id, being_id, name, text, _iso(now)),
            )
            c.execute(
                "UPDATE being_public_threads SET updated_at = ?, sender_name = ?"
                " WHERE id = ?", (_iso(now), name, thread_id),
            )
            c.commit()
        self.record_event(being_id, "public_message",
                          {"from": name, "preview": text,
                           "thread": thread_id[:8]}, now=now)
        return {"thread_id": thread_id, "message_id": mid}

    def public_thread(self, slug: str, thread_id: str) -> dict:
        """One visitor's own conversation on a PUBLIC being."""
        return self._thread(self._public_row(slug)["id"], thread_id)

    def thread_for(self, owner_id: str, slug: str, thread_id: str) -> dict:
        """A thread on a being resolved by owner+slug — no ``public`` gate
        (the federated sender serves its visiting being's threads)."""
        return self._thread(self.get(owner_id, slug)["id"], thread_id)

    def _thread(self, being_id: str, thread_id: str) -> dict:
        t = self._c().execute(
            "SELECT * FROM being_public_threads WHERE id = ? AND being_id = ?",
            (thread_id, being_id),
        ).fetchone()
        if not t:
            raise BeingNotFound("no such thread")
        msgs = self._c().execute(
            "SELECT role, sender_name, body, at, read_at, answered_at"
            " FROM being_public_messages WHERE thread_id = ? ORDER BY at",
            (thread_id,),
        ).fetchall()
        return {"thread_id": thread_id, "sender_name": t["sender_name"],
                "messages": [dict(m) for m in msgs]}

    def public_threads_for(self, owner_id: str, slug: str) -> list[dict]:
        """Every thread on a being's public page — the PARENT-only overview."""
        b = self.get(owner_id, slug)
        threads = self._c().execute(
            "SELECT * FROM being_public_threads WHERE being_id = ?"
            " ORDER BY updated_at DESC", (b["id"],),
        ).fetchall()
        out = []
        for t in threads:
            msgs = self._c().execute(
                "SELECT role, sender_name, body, at, read_at, answered_at"
                " FROM being_public_messages WHERE thread_id = ? ORDER BY at",
                (t["id"],),
            ).fetchall()
            out.append({"thread_id": t["id"], "sender_name": t["sender_name"],
                        "created_at": t["created_at"],
                        "updated_at": t["updated_at"],
                        "messages": [dict(m) for m in msgs]})
        return out

    def unread_public_messages(self, being_id: str,
                               limit: int = 3) -> list[dict]:
        """The visitor notes a tick hasn't shown the being yet — oldest first."""
        rows = self._c().execute(
            "SELECT id, thread_id, sender_name, body, at FROM being_public_messages"
            " WHERE being_id = ? AND role = 'public' AND read_at IS NULL"
            " ORDER BY at LIMIT ?", (being_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_public_messages_read(self, message_ids: list[str],
                                  now: datetime | None = None) -> None:
        """A tick considered these — don't clog later prompts with them again."""
        if not message_ids:
            return
        now = now or _utcnow()
        with self._lock:
            self._c().executemany(
                "UPDATE being_public_messages SET read_at = ? WHERE id = ?",
                [(_iso(now), mid) for mid in message_ids],
            )
            self._c().commit()

    def answer_public_message(self, being_id: str, thread_id: str, reply: str,
                              now: datetime | None = None) -> dict | None:
        """The being's OPTIONAL reply to a visitor thread — stored as a
        role='being' message; marks that thread's public notes answered."""
        now = now or _utcnow()
        reply = (reply or "").strip()[:1000]
        if not reply:
            return None
        t = self._c().execute(
            "SELECT 1 FROM being_public_threads WHERE id = ? AND being_id = ?",
            (thread_id, being_id),
        ).fetchone()
        if not t:
            return None
        pend = self._c().execute(
            "SELECT id FROM being_public_messages WHERE thread_id = ?"
            " AND being_id = ? AND role = 'public' AND answered_at IS NULL"
            " ORDER BY at", (thread_id, being_id),
        ).fetchall()
        rid = uuid.uuid4().hex
        with self._lock:
            self._c().execute(
                "INSERT INTO being_public_messages"
                " (id, thread_id, being_id, role, sender_name, body, at)"
                " VALUES (?,?,?,'being','',?,?)",
                (rid, thread_id, being_id, reply, _iso(now)),
            )
            if pend:
                self._c().executemany(
                    "UPDATE being_public_messages SET answered_at = ?"
                    " WHERE id = ?", [(_iso(now), p["id"]) for p in pend],
                )
            self._c().execute(
                "UPDATE being_public_threads SET updated_at = ? WHERE id = ?",
                (_iso(now), thread_id),
            )
            self._c().commit()
        self.record_event(being_id, "answered_visitor",
                          {"thread": thread_id[:8], "preview": reply[:80]},
                          now=now)
        return {"reply_id": rid}

    # ── The village's own words (shown on the public square) ──────────

    VILLAGE_DESC_MAX = 4000

    def get_village_meta(self, owner_id: str) -> dict:
        row = self._c().execute(
            "SELECT name, description, secret, secret_public, public_url,"
            " steward_stipend_coins"
            " FROM village_meta WHERE owner_id = ?", (owner_id,)).fetchone()
        if not row:
            return {"name": "", "description": "", "secret": "",
                    "secret_public": False, "public_url": "",
                    "steward_stipend_coins": 0}
        return {"name": row["name"] or "",
                "description": row["description"],
                "secret": row["secret"] or "",
                "secret_public": bool(row["secret_public"]),
                "public_url": row["public_url"] or "",
                "steward_stipend_coins":
                    int(row["steward_stipend_coins"] or 0)}

    def set_steward_stipend(self, owner_id: str, coins: int,
                            now: datetime | None = None) -> dict:
        """The steward's weekly pay (space plan Phase 5): a parent faucet,
        0–10 coins, default off. Paid once per ISO week at the steward's
        first morning (being_world.steward_percepts)."""
        coins = int(coins)
        if not (0 <= coins <= 10):
            raise BeingError("a stipend runs 0–10 coins a week")
        self._upsert_village_meta(owner_id,
                                  {"steward_stipend_coins": coins}, now=now)
        return {"steward_stipend_coins": coins}

    def _upsert_village_meta(self, owner_id: str, fields: dict,
                             now: datetime | None = None) -> None:
        now = now or _utcnow()
        cur = self.get_village_meta(owner_id)
        cur.update(fields)
        with self._lock:
            self._c().execute(
                "INSERT INTO village_meta (owner_id, name, description, secret,"
                " secret_public, public_url, steward_stipend_coins,"
                " updated_at) VALUES (?,?,?,?,?,?,?,?)"
                " ON CONFLICT(owner_id) DO UPDATE SET name=excluded.name,"
                " description=excluded.description, secret=excluded.secret,"
                " secret_public=excluded.secret_public,"
                " public_url=excluded.public_url,"
                " steward_stipend_coins=excluded.steward_stipend_coins,"
                " updated_at=excluded.updated_at",
                (owner_id, cur["name"], cur["description"], cur["secret"],
                 1 if cur["secret_public"] else 0, cur["public_url"],
                 int(cur.get("steward_stipend_coins") or 0), _iso(now)),
            )
            self._c().commit()

    def set_village_meta(self, owner_id: str, description: str,
                         name: str | None = None,
                         now: datetime | None = None) -> dict:
        """Set the village description, and (when provided) its name. A None name
        leaves the existing one untouched (so 'Recommend a description' doesn't
        wipe the name)."""
        fields: dict = {
            "description": (description or "").strip()[:self.VILLAGE_DESC_MAX]}
        if name is not None:
            fields["name"] = (name or "").strip()[:80]
        self._upsert_village_meta(owner_id, fields, now=now)
        m = self.get_village_meta(owner_id)
        return {"name": m["name"], "description": m["description"]}

    def set_village_federation(self, owner_id: str, *, secret: str,
                               secret_public: bool, public_url: str,
                               now: datetime | None = None) -> dict:
        """The host settings: the secret a visiting being must present, whether
        it's shown publicly, and this machine's own public URL (for sending)."""
        self._upsert_village_meta(owner_id, {
            "secret": (secret or "").strip()[:200],
            "secret_public": bool(secret_public),
            "public_url": (public_url or "").strip().rstrip("/")[:400],
        }, now=now)
        return self.get_village_meta(owner_id)

    def public_village(self) -> dict:
        """The description + (if opted in) the visit secret shown on the public
        square. Resolved from the owner with the most public beings; falls back
        to an owner advertising a public secret, then one that hosts visitors —
        so a pure-host village (no local public beings) still shows its words."""
        c = self._c()
        row = c.execute(
            "SELECT owner_id FROM beings WHERE public = 1 AND stage != 'egg'"
            " GROUP BY owner_id ORDER BY COUNT(*) DESC, owner_id LIMIT 1",
        ).fetchone()
        owner = row["owner_id"] if row else None
        if not owner:
            r2 = c.execute(
                "SELECT owner_id FROM village_meta"
                " WHERE secret_public = 1 AND secret != '' LIMIT 1").fetchone()
            owner = r2["owner_id"] if r2 else None
        if not owner:
            r3 = c.execute(
                "SELECT owner_id FROM being_visitors"
                " ORDER BY last_seen DESC LIMIT 1").fetchone()
            owner = r3["owner_id"] if r3 else None
        if not owner:
            return {"name": "", "description": "", "visit_secret": ""}
        m = self.get_village_meta(owner)
        return {"name": m["name"],
                "description": m["description"],
                # Only expose the secret if the owner chose to publish it.
                "visit_secret": m["secret"] if m["secret_public"] else ""}

    # ── Federation: visitors (beings from other machines) ─────────────

    def owner_by_secret(self, secret: str) -> str | None:
        """The host owner whose village secret matches — the gate for registering
        a visitor. Empty secrets never match (a village with no secret set is not
        accepting visitors)."""
        secret = (secret or "").strip()
        if not secret:
            return None
        row = self._c().execute(
            "SELECT owner_id FROM village_meta WHERE secret = ? AND secret != ''"
            " LIMIT 1", (secret,)).fetchone()
        return row["owner_id"] if row else None

    def upsert_visitor(self, owner_id: str, origin: str, slug: str,
                       name: str, profile: dict,
                       now: datetime | None = None) -> dict:
        """Register or refresh a visiting being under the HOST owner. Dedup by
        (owner, origin, slug); last_seen drives heartbeat expiry."""
        now = now or _utcnow()
        origin = (origin or "").strip().rstrip("/")
        existing = self._c().execute(
            "SELECT id, first_seen FROM being_visitors WHERE owner_id = ?"
            " AND origin = ? AND slug = ?", (owner_id, origin, slug)).fetchone()
        vid = existing["id"] if existing else uuid.uuid4().hex
        first_seen = existing["first_seen"] if existing else _iso(now)
        with self._lock:
            self._c().execute(
                "INSERT INTO being_visitors (id, owner_id, origin, slug, name,"
                " profile, first_seen, last_seen) VALUES (?,?,?,?,?,?,?,?)"
                " ON CONFLICT(owner_id, origin, slug) DO UPDATE SET"
                " name=excluded.name, profile=excluded.profile,"
                " last_seen=excluded.last_seen",
                (vid, owner_id, origin, slug, name[:80], json.dumps(profile),
                 first_seen, _iso(now)),
            )
            self._c().commit()
        return self.get_visitor(vid)

    def get_visitor(self, visitor_id: str) -> dict:
        row = self._c().execute(
            "SELECT * FROM being_visitors WHERE id = ?", (visitor_id,)).fetchone()
        if not row:
            raise BeingNotFound("no such visitor")
        v = dict(row)
        try:
            v["profile"] = json.loads(v["profile"] or "{}")
        except json.JSONDecodeError:
            v["profile"] = {}
        return v

    def public_visitors(self, ttl_minutes: int = 30,
                        now: datetime | None = None) -> list[dict]:
        """Live visitors (seen within the TTL) for the public roster — cards from
        cached snapshots, newest arrivals first."""
        now = now or _utcnow()
        cutoff = _iso(now - timedelta(minutes=ttl_minutes))
        rows = self._c().execute(
            "SELECT * FROM being_visitors WHERE last_seen >= ?"
            " ORDER BY last_seen DESC", (cutoff,)).fetchall()
        out = []
        for r in rows:
            try:
                prof = json.loads(r["profile"] or "{}")
            except json.JSONDecodeError:
                prof = {}
            out.append({"id": r["id"], "origin": r["origin"], "slug": r["slug"],
                        "name": r["name"], "profile": prof,
                        "last_seen": r["last_seen"]})
        return out

    def visitors_for(self, owner_id: str) -> list[dict]:
        """A host owner's visitors (all, for their parent view)."""
        rows = self._c().execute(
            "SELECT id, origin, slug, name, first_seen, last_seen"
            " FROM being_visitors WHERE owner_id = ? ORDER BY last_seen DESC",
            (owner_id,)).fetchall()
        return [dict(r) for r in rows]

    def remove_visitor(self, owner_id: str, visitor_id: str) -> None:
        with self._lock:
            self._c().execute(
                "DELETE FROM being_visitors WHERE id = ? AND owner_id = ?",
                (visitor_id, owner_id))
            self._c().commit()

    def expire_visitors(self, ttl_minutes: int = 30,
                        now: datetime | None = None) -> int:
        now = now or _utcnow()
        cutoff = _iso(now - timedelta(minutes=ttl_minutes))
        with self._lock:
            cur = self._c().execute(
                "DELETE FROM being_visitors WHERE last_seen < ?", (cutoff,))
            self._c().commit()
            return cur.rowcount or 0

    # ── Federation: sending beings out to visit (sender side) ─────────

    def set_being_visit(self, owner_id: str, slug: str, url: str, secret: str,
                        now: datetime | None = None) -> dict:
        b = self.get(owner_id, slug)
        self._update(b["id"], now or _utcnow(),
                     visit_url=(url or "").strip().rstrip("/")[:400],
                     visit_secret=(secret or "").strip()[:200])
        return self.get(owner_id, slug)

    def beings_visiting(self) -> list[dict]:
        """Every being configured to visit somewhere — the announce loop's list."""
        rows = self._c().execute(
            "SELECT owner_id, slug FROM beings WHERE visit_url != ''"
            " AND state != 'dead' AND stage != 'egg'").fetchall()
        return [self.get(r["owner_id"], r["slug"]) for r in rows]

    def mark_announced(self, being_id: str, now: datetime | None = None) -> None:
        self._update(being_id, now or _utcnow(),
                     visit_last_announce=_iso(now or _utcnow()))

    # ── The Mind: declared edges over the being's own artifacts ──────

    def add_link(self, owner_id: str, being_id: str, from_path: str,
                 to_path: str, rel: str, why: str = "",
                 now: datetime | None = None) -> bool:
        """Persist one verified edge. Idempotent (UNIQUE). Callers verify the
        endpoints exist first (being_mind) — the store only conserves the row."""
        now = now or _utcnow()
        with self._lock:
            cur = self._c().execute(
                "INSERT OR IGNORE INTO being_links"
                " (id, owner_id, being_id, from_path, to_path, rel, why, at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, owner_id, being_id, from_path, to_path,
                 rel, why[:300], _iso(now)),
            )
            self._c().commit()
            return cur.rowcount > 0

    def links_for(self, owner_id: str, slug: str) -> list[dict]:
        b = self.get(owner_id, slug)
        rows = self._c().execute(
            "SELECT from_path, to_path, rel, why, at FROM being_links"
            " WHERE being_id = ? ORDER BY at", (b["id"],),
        ).fetchall()
        return [dict(r) for r in rows]

    def dangling_links(self, being_id: str,
                       existing_paths: set[str]) -> list[dict]:
        """Edges whose endpoints no longer exist, OLDEST first, with ids —
        a pure read; the mind decides what honest forgetting removes."""
        rows = self._c().execute(
            "SELECT id, from_path, to_path, rel, at FROM being_links"
            " WHERE being_id = ? ORDER BY at", (being_id,),
        ).fetchall()
        return [dict(r) for r in rows
                if r["from_path"] not in existing_paths
                or r["to_path"] not in existing_paths]

    def remove_links(self, being_id: str, link_ids: list[str]) -> int:
        """Delete specific edges by id (bounded, caller-chosen prune)."""
        if not link_ids:
            return 0
        with self._lock:
            self._c().executemany(
                "DELETE FROM being_links WHERE being_id = ? AND id = ?",
                [(being_id, lid) for lid in link_ids],
            )
            self._c().commit()
        return len(link_ids)

    def prune_links(self, being_id: str, existing_paths: set[str],
                    now: datetime | None = None) -> list[dict]:
        """Drop edges whose endpoints no longer exist. Returns the pruned rows
        so the caller can record honest forgetting."""
        dangling = self.dangling_links(being_id, existing_paths)
        if dangling:
            self.remove_links(being_id, [d["id"] for d in dangling])
        return dangling

    def message_thread(self, owner_id: str, slug: str,
                       limit: int = 200) -> list[dict]:
        """The full parent↔being conversation, chronological: every message
        the parent wrote (with read/unread) plus the being's replies."""
        b = self.get(owner_id, slug)
        items: list[dict] = []
        rows = self._c().execute(
            "SELECT body, at, read_at FROM being_parent_messages"
            " WHERE being_id = ? ORDER BY at DESC LIMIT ?",
            (b["id"], limit),
        ).fetchall()
        for r in rows:
            items.append({"from": "parent", "body": r["body"], "at": r["at"],
                          "read": r["read_at"] is not None})
        for e in self.events(owner_id, slug, limit=limit):
            if e["kind"] == "spoke_to_parent":
                items.append({"from": "being",
                              "body": e["data"].get("preview") or "",
                              "at": e["at"], "read": True})
        items.sort(key=lambda x: x["at"])
        return items[-limit:]

    def add_publication(self, owner_id: str, being_id: str, title: str,
                        note: str, commons_path: str, price_tokens: int,
                        now: datetime | None = None) -> dict:
        now = now or _utcnow()
        pid = uuid.uuid4().hex
        with self._lock:
            self._c().execute(
                "INSERT INTO being_publications"
                " (id, owner_id, being_id, title, note, commons_path,"
                "  price_tokens, at) VALUES (?,?,?,?,?,?,?,?)",
                (pid, owner_id, being_id, title[:80], note[:300],
                 commons_path, int(price_tokens), _iso(now)),
            )
            self._c().commit()
        return self.get_publication(owner_id, pid)

    def get_publication(self, owner_id: str, ref: str) -> dict:
        """Exact id or unambiguous prefix (percepts show 8-char ids)."""
        rows = self._c().execute(
            "SELECT * FROM being_publications WHERE owner_id = ?"
            " AND (id = ? OR id LIKE ?) LIMIT 2",
            (owner_id, ref, f"{ref}%"),
        ).fetchall()
        if len(rows) != 1:
            raise BeingNotFound("no such publication")
        return dict(rows[0])

    def publications(self, owner_id: str, since: str | None = None,
                     exclude_being: str | None = None,
                     limit: int = 20) -> list[dict]:
        q = "SELECT * FROM being_publications WHERE owner_id = ?"
        args: list = [owner_id]
        if since:
            q += " AND at > ?"
            args.append(since)
        if exclude_being:
            q += " AND being_id != ?"
            args.append(exclude_being)
        q += " ORDER BY at DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self._c().execute(q, args).fetchall()]

    def transfer_between(self, owner_id: str, from_id: str, to_id: str,
                         tokens: int, reason: str, note: str | None = None,
                         now: datetime | None = None) -> None:
        """Being→being transfer on the conservation ledger (gift/trade).

        Savings ceilings do NOT clip transfers: ceilings cap only mints
        (allowance, fees), because moving existing tokens never increases
        the parent's total liability. The sender simply must afford it.
        """
        now = now or _utcnow()
        if reason not in ("gift", "trade"):
            raise BeingError(f"{reason!r} is not a being-to-being reason")
        if from_id == to_id:
            raise BeingError("cannot transfer to oneself")
        sender = self._being_by_id(from_id)
        view = self.wallet_view(sender)
        if view["enforced"] and view["balance_tokens"] < int(tokens):
            raise InsufficientTokens("the sender cannot afford this")
        self._apply(owner_id, tokens=int(tokens), reason=reason,
                    from_being=from_id, to_being=to_id, note=note, now=now)

    def debit_usage_clamped(
        self, being_id: str, tier: str, usage: dict,
        note: str | None = None, now: datetime | None = None,
    ) -> dict:
        """Post-hoc metering for spend that already happened (a tick's turn).

        Debits the full weighted amount when the wallet covers it; otherwise
        debits everything left and flags the overdraft — the being spent its
        last strength and collapses toward torpor. Burn-cap breaches are
        flagged, not blocked (the cap gates FUTURE dispatch, not past spend).
        """
        now = now or _utcnow()
        b = self._being_by_id(being_id)
        view = self.wallet_view(b)
        weighted = constitution.weighted_tokens(usage, tier)
        out = {"weighted": weighted, "debited": 0,
               "overdraft": False, "burn_cap_hit": False}
        if not view["enforced"] or weighted <= 0:
            return out
        debit = min(weighted, max(0, view["balance_tokens"]))
        if debit > 0:
            self._apply(b["owner_id"], tokens=debit, reason="usage",
                        from_being=being_id, to_being=None,
                        note=note or tier, now=now)
        out["debited"] = debit
        out["overdraft"] = debit < weighted
        cap = view["daily_burn_cap"]
        if cap is not None:
            cap += self.granted_today(being_id, now=now)   # recharges extend it
            if self.spent_today(being_id, now=now) >= cap:
                out["burn_cap_hit"] = True
        return out


_STORE: BeingsStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> BeingsStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = BeingsStore()
        return _STORE
